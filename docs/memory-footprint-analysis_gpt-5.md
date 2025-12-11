Here’s a focused, code‑aware diagnosis of why your Swift/MLX pipeline peaks much higher on Apple Silicon than the Diffusers/MPS baseline, and exactly where it happens in this repo.

**What’s Different vs. Diffusers**
- Runtime: Your pipeline runs on MLX (Metal + MLXNN), not PyTorch MPS. Diffusers’ VAE convs are MPS kernels (no im2col materialization); MLX Conv2d likely lowers to GEMM via im2col for stride=1, k=3 convs, which allocates very large intermediate “column” buffers at high resolution.
- Compute dtype: Weights are loaded as bf16, but the latent path and most intermediates default to float32 unless forced. In MLX, if the input is float32 and weights are bf16, compute tends to upcast to float32. That doubles the working set for feature maps and any im2col buffers compared to bf16.
- Measurement: Activity Monitor shows total process memory (unified memory + MLX allocator pools + temp buffers + CPU-side copies). Diffusers’ reported numbers are typically tensor bytes only (e.g., torch.mps.*) and exclude allocator caches.

**Where The Spike Actually Comes From**
- Offloading is correct: You do drop the transformer before VAE decode, and call `GPU.clearCache()` (Sources/ZImage/Pipeline/ZImagePipeline.swift:321). The memory drop to “a few GB” matches that.
- VAE decode rebuilds large NHWC feature maps and runs many 3×3 convs at full resolution:
  - VAEDecoder: six resnet convs per final up block + one final output conv (7 convs) at full res, plus previous up block at half-res (Sources/ZImage/Model/VAE/AutoencoderKL.swift:190–229, 303–347).
  - Upsampling is done via a broadcast+reshape (nearest) that isn’t the main culprit, but it’s not memory‑optimal either (Sources/ZImage/Model/VAE/AutoencoderKL.swift:150–167).
- im2col blow‑up at full res: For the last up block at 1536×2304 and C_in=128, each 3×3 conv’s im2col has roughly
  - rows = H×W = 1536×2304 = 3,538,944
  - cols = kH×kW×C_in = 3×3×128 = 1,152
  - elements ≈ 4,076,863,488
  - bytes ≈ 7.6 GiB (bf16) or ≈ 15.2 GiB (float32)
- Because you have several such convs back‑to‑back in the final up block (and more in the previous block at 768×1152 where each im2col is ≈ 3.8 GiB bf16/7.6 GiB f32), the peak reflects multiple im2col + outputs live together due to MLX’s lazy eval and scheduling. In float32 it’s easy to hit 80–90 GB transiently.

**Code Evidence**
- VAE up stack and convs (7 full‑res convs in the last stage):
  - VAEResnetBlock2D convs (2 per resnet): Sources/ZImage/Model/VAE/AutoencoderKL.swift:126–133
  - VAEUpBlock structure: Sources/ZImage/Model/VAE/AutoencoderKL.swift:190–229
  - Final output conv: Sources/ZImage/Model/VAE/AutoencoderKL.swift:336–347
- Nearest upsample via broadcast+reshape:
  - Sources/ZImage/Model/VAE/AutoencoderKL.swift:150–167
- You do decoder‑only weight loading (good), but still transpose and duplicate the 4D VAE weights in memory once:
  - Transpose on load (`vaeMapping`): Sources/ZImage/Weights/WeightsMapping.swift:61–69
  - Decoder‑only filter: Sources/ZImage/Pipeline/ZImagePipeline.swift:323–326
  - Note: this duplication is small (~hundreds of MB) vs. the spike, but worth smoothing.

**Why The Baseline 0→~40 GB Before VAE**
- Model residency + allocator pools: Transformer (bf16) + text encoder + MLX / Metal allocator caches together typically bring you to the mid/upper 30s of GB on M‑series. With a few transient buffers, Activity Monitor reading ~40 GB is unsurprising.
- Your pipeline frees transformer weights before VAE (do‑block scope + `transformer.clearCache()` at Sources/ZImage/Pipeline/ZImagePipeline.swift:317 + `GPU.clearCache()` at 321), hence the drop just before decode.

**Expected Peaks (From Your Code Path)**
- If VAE compute runs in float32:
  - Final up block at 1536×2304: ~15.2 GiB per 3×3 conv transient (just the im2col), × ~7 convs → order‑of‑magnitude ~100 GiB worst‑case if the scheduler can’t free between ops.
  - Previous up block at 768×1152: ~7.6 GiB per conv transient × ~7 → another ~50 GiB worst‑case if overlaps exist.
  - In practice you won’t sum all maxima, but a transient in the 80–90 GB range is very plausible.
- If VAE runs in bf16:
  - Halves the above (e.g., ~7.6 GiB → ~3.8 GiB per im2col in the final block). Peak still big, but typically ~35–55 GB instead of ~80–90 GB.

**Other Small Contributors**
- Weight mapping duplication (transpose) during VAE load: small, but present (Sources/ZImage/Weights/WeightsMapping.swift:61–69).
- Nearest upsample via broadcast+reshape before the conv (Sources/ZImage/Model/VAE/AutoencoderKL.swift:150–167); not dominant, but a bit wasteful.

**Actionable Fixes (Ordered, minimal to invasive)**
- Force bf16 (or fp16) compute for VAE decode.
  - Downcast latents right before decode so the whole decoder runs in bf16 (or fp16):
    - In `decodeLatents`, cast `latents` to `.bfloat16` first, or create `latents` as bf16 up front. Then ensure scalars used in `decode` (`scalingFactor`, `shiftFactor`) are cast to the same dtype.
  - Load VAE weights as fp16 to cut weight bytes and encourage lower‑precision compute:
    - Change `let vaeWeights = try weightsMapper.loadVAE()` to `loadVAE(dtype: .float16)` in Sources/ZImage/Pipeline/ZImagePipeline.swift:324.
- Add evaluation barriers in the decoder hot path.
  - Insert `MLX.eval(hidden)` at the end of each resnet and after major upsample transitions to force materialization and free earlier im2col buffers before the next conv. This reduces scheduler overlap of huge temporaries.
- Swap the manual nearest upsample for MLXNN.Upsample.
  - Replace `VAEUpSampler.upSampleNearest` (Sources/ZImage/Model/VAE/AutoencoderKL.swift:155–167) with `MLXNN.Upsample(mode: .nearest)` to avoid broadcast+reshape intermediates.
- Stream VAE weight apply (optional).
  - Avoid building a second transposed dictionary in `vaeMapping`; transpose and apply per weight, then drop it. This saves a few hundred MB during VAE load.
- Tiled/striped VAE decode (most robust).
  - Decode the image in vertical stripes (e.g., 256 rows at a time) with minimal overlap. This bounds im2col working sets to the tile, bringing peaks down dramatically at the cost of a little plumbing. It’s the standard fix for high‑res VAEs in custom runtimes.
- Align divisor facts (minor).
  - There’s an inconsistency: `VAEConfig.latentDivisor` returns `vaeScaleFactor * 2` (Sources/ZImage/Model/VAE/AutoencoderKL.swift:46–48) whereas `ZImageVAEConfig.latentDivisor` returns `vaeScaleFactor` (Sources/ZImage/Weights/ModelConfigs.swift:63–66). You’re effectively using divisor 8 everywhere (correct for this VAE), so consider fixing the former to avoid confusion later.

**Why Diffusers’ Peaks Look Lower**
- MPS conv kernels in PyTorch don’t do explicit im2col and recycle workspace more aggressively, so peak memory is closer to “weights + feature maps + a bit of workspace”. Your MLX path materializes massive im2col buffers at full resolution.
- Diffusers runs bf16 end‑to‑end on many recent M‑series stacks. Your code path likely runs VAE in float32 unless inputs are explicitly bf16/fp16.

**Quick Validation Hooks**
- Right before VAE decode:
  - Log the dtype of `latents` and a few VAE parameters to confirm if computation will be bf16 or fp16.
  - Add a one‑line eval barrier after each resnet in the final up block and re‑measure via Activity Monitor to see the peak drop.
- If you want, I can:
  - Make the minimal changes now (bf16 VAE decode + MLXNN.Upsample + eval barriers).
  - Add a tiled decode helper to cap VAE memory at a fixed ceiling and avoid spikes at very large resolutions."
