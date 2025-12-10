I’m going to wire in a streaming VAE apply so we don’t materialize a full weights dict and a second transposed copy.

**What Changed**
- Added streaming VAE apply
  - New API: `ZImageWeightsMapper.applyVAEStreaming(to:dtype:logger:)` reads VAE safetensors tensor‑by‑tensor and applies each directly to the module, transposing 4D weights on the fly and avoiding large intermediate maps.
  - Implementation scans `snapshot/vae/*.safetensors` (fallback to `ZImageFiles.vaeWeights`) and updates parameters per tensor.
  - File: Sources/ZImage/Weights/ZImageWeightsMapper.swift:71

- Reused existing apply helper for single‑weight updates
  - Made `ZImageWeightsMapping.applyToModule` internal (was private) so streaming code can apply a tiny map per tensor.
  - File: Sources/ZImage/Weights/WeightsMapping.swift:146

- Switched pipelines to streaming VAE apply
  - Generation: replaces `loadVAE() + applyVAE(weights:)` with streaming apply.
  - File: Sources/ZImage/Pipeline/ZImagePipeline.swift:309
  - Control: same replacement when loading VAE for control image encode.
  - File: Sources/ZImage/Pipeline/ZImageControlPipeline.swift:241

**How It Reduces Memory**
- Avoids building:
  - a full `[String: MLXArray]` of all VAE weights
  - and a second mapped/transposed copy (`vaeMapping`) of all 4D weights
- Only a single tensor (and its transposed view) is resident during each update, plus the module’s parameter buffer.

**Behavior**
- Outputs are unchanged. Dtype remains `.bfloat16` (same as before). Transpose semantics match the previous `vaeMapping`.

**Next Options**
- batch updates per small group (e.g., 64 tensors) to cut `Module.update` calls while keeping memory low.

---

Here’s why streaming VAE apply likely didn’t change the peak much, and what to check next.

**Main Causes**
- Whole-file mapping persists. `SafeTensorsReader` maps the entire safetensors file into memory via `Data(..., options: [.mappedIfSafe])`, so you still incur O(file_size) memory when “loading VAE”, regardless of streaming vs bulk. See Sources/ZImage/Weights/SafeTensorsReader.swift:27.
- VAE constructs both encoder + decoder. `AutoencoderKL` allocates parameter tensors for both even though generation only decodes; that doubles resident params at this phase. See Sources/ZImage/Model/VAE/AutoencoderKL.swift:462.
- Parameter buffers dominate. The big allocation is initializing VAE parameter buffers on device; removing a temporary weights dictionary is small by comparison.
- First-use Metal allocations. First convs/GroupNorms trigger kernel compilation and convolution weight transforms/workspaces, adding transient spikes independent of loading strategy.
- Decode activations overshadow. If the sampling captures just after “loading VAE…”, the upsampling path’s activations at 1024×1024 can be sizeable; streaming doesn’t touch this.
- Mapped file still resident until end of reader scope. Even with streaming per-tensor, each reader holds the file mapping for its lifetime; Activity Monitor memory reflects this.

**Secondary Contributors**
- Per-tensor updates churn. Calling `applyToModule` for every single tensor causes many small tree updates and transient arrays; memory-neutral overall, but adds churn without changing peak.
- Same dtype. VAE stays bfloat16; fp16 would noticeably shrink the VAE parameter footprint and temps.

**What to Verify**
- Measure phases separately by adding logs around:
  - After VAE init (before any weights applied).
  - After weights applied (before decode).
  - During first decode call.
- Confirm whole-file mapping: watch memory bump immediately after reader init versus after per-tensor transposes. Sources/ZImage/Weights/SafeTensorsReader.swift:24–33.

**Next Actions That Will Move the Needle**
- Decoder-only VAE: instantiate only decoder and apply only `decoder.*` weights. Cuts VAE params roughly in half.
- Avoid whole-file mapping: add a non-mmap reader path that uses `FileHandle` to read only the current tensor slice, releasing it immediately (no `.mappedIfSafe`).
- VAE fp16: load VAE tensors as `.float16` during streaming apply.
- Tiled decode: bound VAE activation memory by decoding in tiles/stripes.
- Batch updates: group 32–128 tensors per `Module.update` to reduce churn without raising peak.

