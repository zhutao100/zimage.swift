Below is an implementation-oriented plan to add **first-class support for [`Tongyi-MAI/Z-Image`](https://huggingface.co/Tongyi-MAI/Z-Image)** (the new, non-Turbo variant) to the existing **`zimage.swift`** Swift/MLX port.

---

## 0) Terminology (avoid “variant” ambiguity)

Two different things are commonly called “variant”:

1. **Model family / checkpoint**: `Tongyi-MAI/Z-Image-Turbo` vs `Tongyi-MAI/Z-Image` (Base).
2. **HF weights “variant” (precision)**: `bf16`, `fp16`, etc. This changes filenames like `*.bf16.safetensors` and the corresponding `*.index.json`.

In this plan:

- **“model” / “family”** refers to Turbo vs Base.
- **“weightsVariant”** refers to HF precision variants (`bf16`, `fp16`, …).

---

## 1) Establish what “support Z-Image” means in this codebase

### What already looks compatible

From the two model snapshots you attached, `Z-Image` and `Z-Image-Turbo` share the same high-level component layout (`transformer/`, `text_encoder/`, `vae/`, `scheduler/`, `tokenizer/`). The project already:

* resolves model snapshots from a local directory or HF model id (`ModelResolution` / `PipelineSnapshot`)
* loads configs from JSON (`ZImageModelConfigs`)
* resolves shard lists dynamically via the `*.safetensors.index.json` weight maps (`ZImageFiles.resolveTransformerWeights/resolveTextEncoderWeights`)
* implements the Z-Image denoising loop and (optional) CFG path

So “adding support” is primarily:

1. **first-class model selection + presets** (so users get sane defaults when choosing `Z-Image`),
2. **robust weight resolution** (avoid mixed bf16/fp16 shard loads; deterministic index selection),
3. **parity with the official pipeline semantics** that matter for Z-Image in practice:

   - steps semantics (Turbo’s “9 steps → ~8 effective updates” behavior)
   - CFG truncation / CFG normalization
4. **tests/docs/examples** that prove it works and are runnable by contributors.

---

## 2) Add a model registry (Turbo vs Base vs future variants)

### Goal

Avoid “stringly-typed” model ids scattered across CLI/pipeline and ensure the defaults change when the user selects `Z-Image`.

### Work items

**A. Create a small model registry type**
Add a new file, e.g. `Sources/ZImage/Support/ZImageModelRegistry.swift`:

* `enum ZImageKnownModel`

  * `.zImageTurbo` → `"Tongyi-MAI/Z-Image-Turbo"`
  * `.zImage` → `"Tongyi-MAI/Z-Image"`
  * (optional) `.zImageTurbo8bit` → `"mzbac/z-image-turbo-8bit"` (existing tests use this)

* `struct ZImagePreset`

  * `recommendedSteps`
  * `recommendedGuidance`
  * `recommendedResolution` (keep current 1024 defaults for Turbo; for `Z-Image` you may prefer 1024 too, but document its recommended operating range)
  * `recommendedMaxSequenceLength` (default 512; document that 1024 is useful for long prompts)
  * `defaultNegativePrompt` (often `""`)
  * (optional) `recommendedCFGTruncation` / `recommendedCFGNormalization` defaults for “known” presets

**B. Update `ZImageRepository` / cache naming**
Right now `ZImageRepository.id` and `defaultCacheDirectory()` are Turbo-specific. Refactor to:

* keep Turbo as the default (to preserve current behavior),
* add a helper:

  * `static func defaultCacheDirectory(for modelId: String, base: URL = ...) -> URL`

    * e.g. `.../z-image-turbo` vs `.../z-image`

This prevents confusing cache layouts when users switch between `Z-Image-Turbo` and `Z-Image`.

**C. Make “variant reuse” aware of `Z-Image`**
Update `areZImageVariants(_:_:)` in:

* `ZImagePipeline.swift`
* `ZImageControlPipeline.swift`

so the base model id is included in the “same-family” check, allowing clean in-place weight swapping between `Z-Image-Turbo` and `Z-Image` (instead of forcing a full teardown/rebuild).

**D. Ensure defaults propagate through *all* entry points**

Default model id/revision and caching decisions currently flow through more than CLI + `ZImageRepository`.
Explicitly include these in the refactor checklist so Turbo-default leakage doesn’t persist:

* `Sources/ZImage/Pipeline/PipelineSnapshot.swift` (default model selection)
* `Sources/ZImage/Weights/ModelResolution.swift` (default model id/revision and cache lookup)

---

## 3) Make weight/index resolution robust to HF precision variants (bf16/fp16/…)

Today the “dynamic” weight resolvers still assume fixed filenames for index.json (e.g. `transformer/diffusion_pytorch_model.safetensors.index.json`).
This becomes fragile as soon as a repo contains multiple weight variants (bf16/fp16/etc.), because directory scans can accidentally select a mixed set.
`ModelResolution` also downloads `["*.safetensors", "*.json", "tokenizer/*"]`, which will pull **all** variants if present.

### Work items

**A. Introduce an explicit `weightsVariant`**

Add an optional `weightsVariant: String?` to the model selection/config surface area (CLI + library). This should map to HF’s “variant” concept (e.g. `"bf16"`).

**B. Make index selection variant-aware**

Update `ZImageFiles.resolveTransformerWeights` / `resolveTextEncoderWeights` (and VAE if needed) to accept `weightsVariant` and:

* Prefer `*.{weightsVariant}.safetensors.index.json` when `weightsVariant != nil`
* Fall back to the non-variant index filename if the variant index isn’t present
* When scanning directories, only accept candidates whose filenames match the chosen variant (to prevent mixing)

**C. Make downloads variant-aware**

Update `ModelResolution` to allow downloading only the selected variant:

* If `weightsVariant != nil`, use patterns like:

  * `"*.\(weightsVariant).safetensors"`
  * `"*.\(weightsVariant).safetensors.index.json"`
  * plus the required `*.json` configs + tokenizer files

* If multiple variants exist and the user didn’t specify one, keep a deterministic default (prefer non-variant) and log a warning listing discovered variants.

**D. Add guardrails**

If `weightsVariant` is set but required components don’t have matching weights (transformer/text_encoder/vae), fail with a clear error that points out the mismatch (avoid partial/mixed loads).

---

## 4) Verify and align step semantics with the reference Z-Image pipeline

Turbo’s model card guidance says `num_inference_steps=9` “results in ~8 DiT forwards”. The current Swift implementation runs exactly `request.steps` denoise iterations, and the current scheduler does not naturally terminate at `sigma=0`, so the last step is *not* a no-op.

The reference Diffusers pipeline forces the schedule to end at `sigma=0` (`scheduler.sigma_min = 0.0`) and appends a terminal sigma; this makes the last `dt` zero, so the final iteration doesn’t change latents (and can be skipped as a perf optimization without changing output).

### Work items

**A. Decide what the public “steps” semantic is**

* Keep CLI/library semantics aligned with Diffusers: `--steps` maps to `num_inference_steps`.
* Document that (for this scheduler) the last iteration can be a no-op when the schedule ends at `sigma=0`, which is why “9 steps → ~8 effective updates” can be true.

**B. Update the Swift scheduler/pipeline to match**

* Extend `FlowMatchEulerScheduler` to support an explicit terminal sigma (e.g. `sigmaMinOverride = 0.0`).
* In both `ZImagePipeline` and `ZImageControlPipeline`, construct the scheduler with terminal sigma = 0.0 (matching Diffusers pipeline behavior) unless there’s evidence Base needs different semantics.
* Optional optimization: if `dt == 0` on the last step, skip the transformer forward for that iteration. Call this out explicitly as “perf optimization; numerically equivalent because the update is a no-op.”

**C. Add a fast unit test for step semantics**

Add a unit test (no weights) that validates:

* scheduler produces `numInferenceSteps` timesteps and `numInferenceSteps + 1` sigmas
* terminal sigma is 0 and last `dt` is 0 when `sigmaMinOverride == 0`
* Turbo preset’s “9 steps → ~8 effective updates” statement is true under the chosen semantics

---

## 5) Add the missing `Z-Image` inference knobs: CFG truncation + CFG normalization

The official Z-Image pipeline supports:

* `cfg_truncation`: disables CFG after a normalized time threshold
* `cfg_normalization`: renormalizes/clamps the CFG output norm relative to the original “positive” prediction norm ([Hugging Face][1])

Even if the `Tongyi-MAI/Z-Image` model card’s minimal recommendation doesn’t require them, adding them makes your Swift port “real pipeline compatible”.

### Work items

**A. Extend request types**
Update `ZImageGenerationRequest` (and the corresponding control request if separate) to include:

* `cfgTruncation: Float?`

  * semantics: threshold in `[0, 1]` on `t_norm` (same normalization you already compute: `(1000 - t)/1000`)
  * default: `nil` (or `1.0`) meaning “never truncate”

* `cfgNormalization: Bool`

  * semantics: match Diffusers API (boolean). When enabled, apply the same renormalization logic as reference.
  * default: `false` (disabled)

* (optional, advanced) `cfgNormalizationFactor: Float?`

  * semantics: treat Diffusers’ bool as `k = 1.0`; allow advanced callers to override `k` if desired.
  * default: `nil` meaning “use `1.0` when `cfgNormalization == true`”

**B. Implement in both denoising loops**
You currently have duplicated denoising code in:

* `ZImagePipeline.generateCore(...)`
* `ZImageControlPipeline` (two denoise paths)

Implement the same logic in both:

1. compute `tNorm` (you already compute `normalizedTimestep`)
2. compute `currentGuidanceScale`:

   * start with `request.guidanceScale`
   * if `cfgTruncation != nil` and `tNorm > cfgTruncation` ⇒ set `currentGuidanceScale = 0`
3. `applyCFG = doCFG && currentGuidanceScale > 0`
4. if `applyCFG`:

   * run the 2× batch
   * compute `pred = pos + currentGuidanceScale * (pos - neg)` (this matches the official Z-Image pipeline behavior) ([Hugging Face][1])
   * if `cfgNormalization`:

     * `k = cfgNormalizationFactor ?? 1.0`
     * `ori = l2Norm(pos)` and `new = l2Norm(pred)` using a **global** vector norm over all elements (Diffusers uses `torch.linalg.vector_norm`).
     * `maxNew = ori * k`
     * if `new > maxNew`: `pred *= maxNew / (new + eps)` (`eps` to avoid division by zero) ([Hugging Face][1])

This gives you parity with the pipeline’s behavior and avoids edge-case “blow-ups” at high guidance.

**C. Reduce drift: factor out a shared helper**

To avoid subtle future divergence between `ZImagePipeline` and `ZImageControlPipeline`, implement the CFG/truncation/normalization logic via a small shared helper (e.g. `PipelineUtilities.applyCFG(...)`), keeping it minimal and testable.

---

## 6) Make presets correct for Z-Image vs Z-Image-Turbo

### What should change

`Z-Image-Turbo` is intended to run with **guidance = 0** and very few steps (the HF example uses `num_inference_steps=9` and `guidance_scale=0.0`) ([Hugging Face][2]).
`Z-Image` (base) recommends **guidance ~ 3–5** and **steps ~ 28–50** ([Hugging Face][3]).

### Work items

**A. CLI: detect whether the user explicitly set flags**
Right now CLI defaults always come from `ZImageModelMetadata` (Turbo-specific). Change `Sources/ZImageCLI/main.swift` to track whether the user explicitly provided:

* `--steps`
* `--guidance`
* `--width/--height`
* `--max-sequence-length`

Then after parsing:

* if `--model` is `Tongyi-MAI/Z-Image` (or user uses a new `--family z-image`):

  * if steps not set → default to e.g. 50
  * if guidance not set → default to e.g. 4 or 5
* if Turbo:

  * if guidance not set → default to 0

**B. Library API: provide a preset helper**
Because Swift default args can’t depend on `model`, don’t fight that. Instead add:

* `ZImageGenerationRequest.makePreset(prompt:modelId:...)`
  or
* `ZImagePreset.defaults(for modelId: String) -> ZImagePreset`

so library callers can do the right thing easily.

---

## 7) Ensure scheduler/config differences are truly “data driven”

In your attached repos, the main config delta that matters is the scheduler `shift` (Turbo vs base differ). Your code already reads `scheduler_config.json`, so you likely get this “for free”—but add explicit validation:

### Work items

* Add a small log line during scheduler construction:

  * `logger.info("Scheduler shift=\(modelConfigs.scheduler.shift) dynamic=\(modelConfigs.scheduler.useDynamicShifting)")`
  * include terminal sigma / step semantics info, e.g. `sigmaMinOverride=0` if implemented
* Add a unit test that loads configs from a local snapshot directory and asserts:

  * `Z-Image-Turbo.shift != Z-Image.shift` (using the two attached repo directories as fixtures in tests, if you want to keep it fully offline)

---

## 8) Testing strategy

### A. Fast “offline config” tests (CI-friendly)

No real weights needed:

* parse configs for both models
* confirm shard resolution finds the correct shard count from the index json
* confirm scheduler/timesteps shape and monotonicity
* confirm step semantics: terminal sigma, last `dt`, and “effective updates” behavior (see section 4)

**Fixture strategy (make this reproducible):**

Do **not** assume you can check in full HF snapshots.
Instead, add lightweight test fixtures that mimic the snapshot layout:

* `model_index.json` + component `config.json` files
* minimal `*.safetensors.index.json` files
* empty placeholder `*.safetensors` shard files (only for resolver existence checks; do not attempt to load them)

This allows fully offline tests without shipping model weights.

### B. Integration tests (skipped in CI)

Add a new integration test similar to existing ones:

* `testBasicGeneration_ZImage_BaseModel()`

  * model: `"Tongyi-MAI/Z-Image"`
  * steps: 28 (lower bound) to reduce runtime
  * guidance: 4
  * resolution: 512 (to reduce memory/time)
  * assert output exists and is a valid PNG

(Keep the current 8-bit Turbo integration tests as-is; they’re valuable for regression.)

**Gating:** keep integration tests opt-in via env var / local-only target so contributors don’t accidentally run heavy Base generations by default.

---

## 9) Docs / examples updates

### Work items

* Update CLI usage header from “Z-Image-Turbo Swift port” → “Z-Image Swift port (Turbo + Base)”
* Add examples:

  * Turbo:

    * `--model Tongyi-MAI/Z-Image-Turbo --steps 9 --guidance 0`
  * Base:

    * `--model Tongyi-MAI/Z-Image --steps 50 --guidance 4 --negative-prompt "..."` ([Hugging Face][3])
* Document new advanced knobs:

  * `--cfg-truncation`
  * `--cfg-normalization`
  * `--weights-variant bf16|fp16|...` (if implemented)
* Document step semantics (so users aren’t confused by “9 → ~8 effective updates” on Turbo).
* Add a short “Apple Silicon practicality” note for Base: recommend starting with 512px and/or quantized transformer weights to avoid OOM/very slow runs.

---

## 10) Concrete file-by-file checklist

**Core**

* `Sources/ZImage/Weights/ModelPaths.swift`

  * refactor `ZImageRepository` to support both ids + cache dirs
  * make index paths variant-aware (weightsVariant)
* `Sources/ZImage/Support/ModelMetadata.swift`

  * split Turbo-specific constants into per-model presets
* `Sources/ZImage/Support/ZImageModelRegistry.swift` (new)

  * known model ids + presets (steps/guidance/resolution/maxSequenceLength/etc.)
* `Sources/ZImage/Weights/ModelResolution.swift`

  * make filePatterns variant-aware (download only the selected weightsVariant)
  * ensure default model id/revision propagation is correct
* `Sources/ZImage/Pipeline/PipelineSnapshot.swift`

  * ensure default model id/revision flows through non-CLI entry points
* `Sources/ZImage/Pipeline/FlowMatchScheduler.swift`

  * add terminal sigma override to match Diffusers step semantics
* `Sources/ZImage/Pipeline/ZImagePipeline.swift`

  * add request fields; implement cfg truncation + normalization in denoise loop
  * include base model in `areZImageVariants`
  * align scheduler construction with section 4 (terminal sigma)
* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

  * mirror the same CFG truncation/normalization changes (keep behavior consistent)
  * align scheduler construction with section 4 (terminal sigma)
* `Sources/ZImage/Pipeline/PipelineUtilities.swift` (or similar)

  * shared CFG helper to avoid duplication drift

**CLI**

* `Sources/ZImageCLI/main.swift`

  * add `--family turbo|z-image` (optional but recommended; “model family”)
  * add `--weights-variant bf16|fp16|...` (HF precision variant)
  * or auto-detect when `--model` matches known ids
  * only apply preset defaults when the user didn’t explicitly set those flags

**Tests**

* `Tests/.../PipelineIntegrationTests.swift`

  * add base model test (skipped in CI)
* add small config-only tests to validate parsing/resolution/step semantics (with lightweight fixtures)

---

## Definition of done

1. `ZImageCLI -p "..." -m Tongyi-MAI/Z-Image` produces a valid image using sensible defaults (steps/guidance) without the user hand-tuning.
2. `ZImageCLI` still works unchanged for Turbo users.
3. Step semantics are explicit and verified (Turbo preset’s “9 steps → ~8 effective updates” behavior is true under the chosen semantics).
4. `cfg_truncation` and `cfg_normalization` behavior matches the reference pipeline logic (including the global-norm renormalization behavior) ([Hugging Face][1]).
5. Weight resolution is deterministic and does not mix precision variants when multiple are present.
6. Tests cover:

   * config parsing for both model dirs
   * at least one integration generation run for base model (manual/skip in CI)

---

If you want, I can turn this plan into a sequence of PR-sized commits (with exact diffs per file) starting with “model registry + CLI presets/default propagation”, then “weightsVariant + step semantics parity”, and finally “CFG truncation/normalization parity.”

[1]: https://huggingface.co/spaces/AiSudo/ZIT-Controlnet/raw/d2c9b66866ffcf001aeffec3c19628902f112d22/videox_fun/pipeline/pipeline_z_image.py "huggingface.co"
[2]: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo?utm_source=chatgpt.com "Tongyi-MAI/Z-Image-Turbo"
[3]: https://huggingface.co/Tongyi-MAI/Z-Image?utm_source=chatgpt.com "Tongyi-MAI/Z-Image"
