# Memory Footprint Analysis (MLX / Apple Silicon)

Updated for `main` at commit `0305fa83` (post-rebase onto upstream `origin-main@970f83e…`).

This document explains why Z-Image.swift can show higher peak resident memory (RSS) in Activity Monitor than a PyTorch/Diffusers/MPS baseline, and where that memory comes from in the current codebase.

## What’s Different vs. Diffusers (Still True)

- **Kernel strategy:** This project runs on `mlx-swift` (`MLX`/`MLXNN`), not PyTorch MPS. For stride=1 3×3 convs, MLX can lower to GEMM via an **im2col**-style workspace, which becomes very large at full resolution.
- **Unified memory reporting:** Activity Monitor includes unified memory + allocator pools + temporary workspaces + CPU-side copies. Diffusers “memory” reports often exclude allocator caches and some workspaces.

## Pipeline Shape Today (As of `370f2d65`)

The standard pipeline loads components through `loadModel(...)`, but **intentionally releases large modules across phases during a generation** to reduce peak RSS:

- **Model load (`loadModel`)** loads tokenizer, text encoder, transformer, and a **decoder-only VAE**.
  - Code: `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- **Generation (`generateCore`)** encodes prompts, then unloads the text encoder; runs denoising, then unloads the transformer before VAE decode.
  - Code: `Sources/ZImage/Pipeline/ZImagePipeline.swift`, `Sources/ZImage/Pipeline/PipelineUtilities.swift`

Implication: by default, the transformer/text-encoder are reloaded across generations, trading throughput for lower peak memory during VAE decode.

## Major Memory Contributors

### 1) Transformer residency (baseline)

Transformer weights are the largest persistent allocation once the model is loaded. The transformer also maintains a shape-keyed cache (see `clearCache()` in `Sources/ZImage/Model/Transformer/ZImageTransformer2D.swift`) which can contribute incremental memory across varying shapes.

### 2) VAE decode workspace (peak driver at high resolution)

The VAE decoder runs multiple 3×3 convs near full resolution. If convs lower through im2col, the workspace scales like:

`(H × W) × (kH × kW × C_in) × bytesPerElement`

At large resolutions, a single conv’s workspace can be several GiB, and overlapping/lazy evaluation can temporarily keep multiple workspaces alive.

VAE code paths:
- VAE decoder graph and upsample: `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- Decoder-only wrapper: `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`
- Decode entry point with dtype handling: `Sources/ZImage/Pipeline/PipelineUtilities.swift`

### 3) Weight-application transposes (model-load spike, not per-generation)

When applying VAE weights, `vaeMapping(...)` transposes 4D tensors (OIHW → OHWI) and constructs a mapped dictionary:
- Code: `Sources/ZImage/Weights/WeightsMapping.swift`

Because the standard pipeline applies only `decoder.*` weights to a decoder-only module, this spike is smaller than “full VAE apply”, but it can still be noticeable during `loadModel`.

## What Has Already Been Fixed vs. Older Analyses

Older docs for pre-rebase commits assumed a “drop transformer then load VAE after denoising” flow and float32 VAE math. That is not representative of the current code.

Current mitigations that materially reduce peak memory:

- **bf16 end-to-end for VAE decode input:** `PipelineUtilities.decodeLatents(...)` casts latents to `.bfloat16` before calling `vae.decode(...)`.
  - Code: `Sources/ZImage/Pipeline/PipelineUtilities.swift`
- **Avoid dtype promotion inside VAE decode:** `AutoencoderKL.decode` and `AutoencoderDecoderOnly.decode` cast `scalingFactor` and `shiftFactor` scalars to the input dtype.
  - Code: `Sources/ZImage/Model/VAE/AutoencoderKL.swift`, `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`
- **Quantized snapshot safety:** `ZImageWeightsMapper.loadVAE(dtype:)` honors dtype requests even when loading from quantized snapshots, preventing mixed-dtype convs that upcast.
  - Code: `Sources/ZImage/Weights/ZImageWeightsMapper.swift`
- **Reduce lazy-eval overlap:** eval barriers exist in hot VAE blocks (e.g., resnet blocks and upsampler conv).
  - Code: `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- **More memory-friendly upsample:** VAE upsample uses `MLXNN.Upsample(..., mode: .nearest)` to avoid large broadcast intermediates.
  - Code: `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- **Decoder-only VAE in standard pipeline:** `ZImagePipeline.loadModel(...)` builds `AutoencoderDecoderOnly` and applies only decoder weights.
  - Code: `Sources/ZImage/Pipeline/ZImagePipeline.swift`

## If You Still Hit OOM at Very Large Resolutions

These are the remaining high-leverage options:

- **Tile/stripe VAE decode:** the most robust way to cap memory; bounds the im2col workspace to a fixed tile size.
- **Offload transformer before decode (default):** `generateCore` clears transformer caches and unloads the transformer before VAE decode to lower the baseline resident memory at the peak point.
- **Reduce model-load peak further:** implement streaming VAE weight apply to avoid holding both the raw tensor dictionary and the transposed mapping at once (see `docs/vae-streaming@21136131.md` for a prior attempt/notes).
- **Tune allocator caching:** the CLI supports `--cache-limit` (calls `GPU.set(cacheLimit:)` in `Sources/ZImageCLI/main.swift`). This helps memory return faster after peaks but doesn’t change the true workspace requirement at peak decode.
