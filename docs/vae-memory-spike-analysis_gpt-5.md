# VAE Memory Spike Analysis

Updated for `main` at commit `0305fa83` (post-rebase onto upstream `origin-main@970f83e…`).

This document focuses specifically on VAE decode memory: why it can spike on Apple Silicon with MLX, what the current code already does to mitigate it, and what remains if you still hit OOM at very large resolutions.

## Where VAE Decode Happens Today

Standard pipeline:

- `Sources/ZImage/Pipeline/ZImagePipeline.swift` calls `PipelineUtilities.decodeLatents(...)` after denoising.
- `PipelineUtilities.decodeLatents(...)` casts latents to `.bfloat16` and calls `vae.decode(...)`.
  - Code: `Sources/ZImage/Pipeline/PipelineUtilities.swift`
- The VAE instance in the standard pipeline is decoder-only (`AutoencoderDecoderOnly`), created in `loadModel(...)`.
  - Code: `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`, `Sources/ZImage/Pipeline/ZImagePipeline.swift`

Control pipeline:

- Control generation uses a full `AutoencoderKL` (it needs `encode(...)` for control/inpaint context building), but it still decodes through `PipelineUtilities.decodeLatents(...)`.
  - Code: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`, `Sources/ZImage/Model/VAE/AutoencoderKL.swift`

## Why MLX VAE Decode Can Spike

The peak is typically caused by **temporary conv workspaces** during the last upsampling stages. A common lowering for conv is im2col + GEMM, where the im2col buffer size scales roughly like:

`workspaceElements ≈ (H × W) × (kH × kW × C_in)`

`workspaceBytes ≈ workspaceElements × bytesPerElement`

So at large `H×W`, a single 3×3 conv near full resolution can require multi‑GiB transient buffers. If evaluation is lazy and multiple convs overlap, peaks can be much higher.

Relevant VAE decoder code:
- `VAEDecoder` and up blocks: `Sources/ZImage/Model/VAE/AutoencoderKL.swift`

## What the Current Code Already Does (Mitigations)

These changes are specifically aimed at preventing the “float32 promotion + overlapping workspaces” failure mode:

1) **Force bf16 inputs into the VAE decode**
- `PipelineUtilities.decodeLatents(...)` casts latents to `.bfloat16` before decode.
- Why it matters: it halves the size of feature maps and most conv workspaces compared to float32.
- Code: `Sources/ZImage/Pipeline/PipelineUtilities.swift`

2) **Prevent accidental dtype promotion inside decode**
- `AutoencoderKL.decode(...)` and `AutoencoderDecoderOnly.decode(...)` cast `scalingFactor`/`shiftFactor` scalars to the same dtype as the input.
- Why it matters: mixing bf16 inputs with float32 scalars can cause promotion (and blow up workspace size).
- Code: `Sources/ZImage/Model/VAE/AutoencoderKL.swift`, `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`

3) **Ensure VAE weights are bf16 even for quantized snapshots**
- `ZImageWeightsMapper.loadVAE(dtype:)` honors dtype conversion even when loading from a quantized snapshot component.
- Why it matters: bf16 input + fp16/fp32 weights can still promote some ops.
- Code: `Sources/ZImage/Weights/ZImageWeightsMapper.swift`

4) **Reduce overlap of large temporaries**
- Eval barriers (`MLX.eval(...)`) exist in hot VAE blocks (resnet blocks and upsampler conv) to reduce lazy-eval overlap between successive convs.
- Code: `Sources/ZImage/Model/VAE/AutoencoderKL.swift`

5) **Use a less wasteful nearest-neighbor upsample**
- The VAE upsampler uses `MLXNN.Upsample(..., mode: .nearest)` (instead of broadcast+reshape).
- Code: `Sources/ZImage/Model/VAE/AutoencoderKL.swift`

6) **Decoder-only VAE in the standard pipeline**
- Standard pipeline builds only the decoder and applies only `decoder.*` weights.
- Why it matters: it reduces resident memory and load-time work (though the decode-time workspace is still the main peak driver).
- Code: `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`, `Sources/ZImage/Pipeline/ZImagePipeline.swift`

## Concrete Back-of-the-Envelope Sizing (Why Big Resolutions Hurt)

For a 3×3 conv at full resolution with `C_in = 128`:

- `workspaceElements ≈ (H×W) × (3×3×128) = (H×W) × 1152`
- bf16: `workspaceBytes ≈ (H×W) × 1152 × 2`
- float32: `workspaceBytes ≈ (H×W) × 1152 × 4`

Example at 1536×2304:

- `H×W = 3,538,944`
- `workspaceElements ≈ 4,076,863,488`
- bf16 workspace ≈ **7.6 GiB** for a single conv’s im2col buffer
- float32 workspace ≈ **15.2 GiB**

This is why “bf16 end-to-end + fewer overlaps” is necessary but may still not be sufficient for extremely large outputs.

## If You Still See a Large Spike

The remaining fixes are about bounding or avoiding the workspace:

- **Tile/stripe VAE decode (best):** decode in smaller spatial tiles to cap workspace size. This is the only approach that gives a predictable memory ceiling.
- **Offload transformer before decode (default):** the pipelines clear transformer caches and unload the transformer before VAE decode to reduce baseline resident memory (the next generation will reload it).
- **Reduce model-load spikes (optional):** VAE weight application still transposes 4D tensors into a mapped dictionary (`vaeMapping(...)` in `Sources/ZImage/Weights/WeightsMapping.swift`). A streaming apply can reduce the model-load peak further (see `docs/vae-streaming@21136131.md` for notes).

## Quick Sanity Checks for Debugging

- Confirm latents dtype entering VAE decode is `.bfloat16` (it should be, via `PipelineUtilities.decodeLatents(...)`).
- If testing large resolutions, try holding width/height constant across runs to avoid transformer cache churn.
- Use the CLI’s `--cache-limit` (calls `GPU.set(cacheLimit:)`) to keep allocator caches from ballooning in unified memory between runs.
