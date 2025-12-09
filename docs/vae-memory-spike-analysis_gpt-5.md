Here’s what the pipeline is doing today, where memory peaks, and concrete ways to reduce and smooth the spike.

**Workflow Overview**
- Text encode
  - Loads tokenizer and Qwen text encoder, applies weights, encodes prompts, then calls `GPU.clearCache()` but keeps `textEncoder` alive in scope.
  - Code: Sources/ZImage/Pipeline/ZImagePipeline.swift:164–214
- Transformer load + denoising
  - Loads transformer, applies base weights (+ optional override/LoRA), runs `steps` denoising iterations on latents.
  - Code: Sources/ZImage/Pipeline/ZImagePipeline.swift:216–294
- VAE load + decode
  - Logs “Denoising complete, loading VAE…”, calls `GPU.clearCache()`, constructs the full VAE (encoder+decoder), loads VAE weights, applies to model, decodes, writes image.
  - Code: Sources/ZImage/Pipeline/ZImagePipeline.swift:296–313

**Memory Footprint by Phase**
- Text encode
  - Persistent: text encoder weights (bf16 by default) and module buffers (until ARC releases).
  - Temporary: activations during encoding, prompt/negative embeddings.
- Denoising
  - Persistent: transformer weights (largest resident), caches in transformer blocks (see `clearCache()` API).
  - Temporary: per-step hidden states and attentions; latents are small (1×16×H/8×W/8; ~0.5 MB at 1024²).
- VAE load and decode
  - Persistent: VAE weights (both encoder and decoder are constructed).
  - Temporary spikes:
    - The entire VAE weights dictionary is loaded first: `let vaeWeights = try weightsMapper.loadVAE()` (Sources/ZImage/Pipeline/ZImagePipeline.swift:300).
    - Then `ZImageWeightsMapping.applyVAE` builds a second dictionary with transposed 4D weights (duplication): `vaeMapping(weights)` (Sources/ZImage/Weights/WeightsMapping.swift:61–75).
    - Only after that are parameters updated via `update(parameters:)`.
  - Net effect: for a moment you hold transformer weights + VAE weights + a second transposed copy of most VAE 4D weights + VAE parameters being written, plus decode activations.

**Why the Spike Happens**
- Transformer weights remain in scope while VAE loads. `GPU.clearCache()` does not free model weights.
- VAE loading duplicates weight memory:
  - First copy in `vaeWeights` (bf16 arrays).
  - Second copy from `vaeMapping()` which transposes all 4D tensors.
- The VAE module includes both encoder and decoder, but only decoder is used for image generation, so you load/update unnecessary encoder weights.
- Decoding then adds activation memory on top.

**Plan Options To Reduce and Smooth Peak**
- Release large modules before VAE load
  - Change `let transformer = ...` to `var transformer = ...` and set `transformer = nil` immediately before loading the VAE, then call `GPU.clearCache()` again. Do the same for `textEncoder` right after encoding.
  - Impact: reduces baseline resident memory right before VAE load.
  - Where: Sources/ZImage/Pipeline/ZImagePipeline.swift:296 (right before VAE load) and 212–214 (after text encoding).
- Stream VAE weight application (avoid double copies)
  - Replace the “load-all dict → map/transposes → apply” path with a streaming apply:
    - Iterate safetensor shards; for each tensor, transpose on-the-fly if needed, and immediately update the matching parameter; then discard it.
    - Avoid building `vaeMapping` dictionary; adapt `applyToModule` to accept a key-transform closure or apply directly by parameter path.
  - Impact: removes the N× duplication during VAE load (biggest spike reducer).
  - Where: Sources/ZImage/Weights/ZImageWeightsMapper.swift:62–70, Sources/ZImage/Weights/WeightsMapping.swift:61–75 and 131–159.
- Decoder-only VAE
  - Construct a “decoder-only” module or gate encoder creation behind a flag; apply weights only to decoder.* keys.
  - Impact: halves VAE weight residency and reduces parameter update work.
  - Where: Sources/ZImage/Model/VAE/AutoencoderKL.swift:462–520; Sources/ZImage/Pipeline/ZImagePipeline.swift:299–303; Sources/ZImage/Weights/WeightsMapping.swift:131–159 (filter to decoder.*).
- Lower VAE dtype to fp16
  - `loadVAE` supports `dtype` selection. Load VAE weights as `.float16` (keep transformer in bf16).
  - Impact: ~2× reduction in VAE memory; minimal quality impact for VAE decode.
  - Where: Sources/ZImage/Pipeline/ZImagePipeline.swift:299–301 (pass dtype), Sources/ZImage/Weights/ZImageWeightsMapper.swift:61–70 (already accepts dtype).
- Tile VAE decode
  - Decode latents in tiles/stripes, stitch outputs. This bounds activation memory inside the VAE at the tile size.
  - Impact: smooths peak during decode; especially useful at large resolutions.
  - Where: Sources/ZImage/Pipeline/ZImagePipeline.swift:309–320 (replace `vae.decode(latents)` with tiled decode).
- Avoid 4D transpose duplication
  - If possible, move the transpose into the parameter-update step (compute-and-assign per weight) rather than allocating a second dictionary; or teach Conv2d loader to accept source layout and transpose into parameter buffers in-place.
  - Impact: removes `vaeMapping` duplication similar to “streaming” but keeps current API shapes.
  - Where: Sources/ZImage/Weights/WeightsMapping.swift:61–75 and 145–159.
- Optional: run VAE on CPU
  - Provide a CLI flag to force VAE decode on CPU (or switch default device for the VAE module only).
  - Impact: eliminates GPU spike at the cost of slower decode; useful on small GPUs.
  - Where: Sources/ZImageCLI/main.swift (add flag), pipeline to construct VAE on CPU if set.
- Conservative GPU cache
  - Encourage `--cache-limit` usage to cap allocator caching so memory returns to the OS sooner after denoising.
  - Impact: doesn’t remove weight duplication but can reduce max footprint during transitions.
  - Where: CLI already supports it: Sources/ZImageCLI/main.swift:100–104.

**Recommended Order (Low Risk → High Impact)**
- Nil-out and clear: explicitly release `textEncoder` after encoding and `transformer` before VAE load; call `GPU.clearCache()` right after.
- Load VAE as fp16: `weightsMapper.loadVAE(dtype: .float16)`.
- Decoder-only: avoid creating/updating encoder; update only decoder params.
- Streaming VAE apply: avoid building `vaeWeights` and `vaeMapping` in memory at once.
- Tiled decode: bound VAE activation memory.
- CPU fallback: add a `--vae-device cpu` option for constrained devices.

If you want, I can implement the minimal changes first (nil-out modules + fp16 VAE) and then iterate toward streaming apply and decoder-only loading."
