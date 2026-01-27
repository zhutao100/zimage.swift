Below is an implementation-oriented plan to add **first-class support for [`Tongyi-MAI/Z-Image`](https://huggingface.co/Tongyi-MAI/Z-Image)** (the new, non-Turbo variant) to the existing **`zimage.swift`** Swift/MLX port.

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
2. **feature parity with the official pipeline knobs** that matter for `Z-Image` (CFG truncation / CFG normalization),
3. **tests/docs/examples** that prove it works.

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
  * `defaultNegativePrompt` (often `""`)

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

---

## 3) Add the missing `Z-Image` inference knobs: CFG truncation + CFG normalization

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

* `cfgNormalization: Float?`

  * semantics: if set to `k > 0`, clamp `||pred||` to `k * ||pos||` (exactly as the official logic does) ([Hugging Face][1])
  * default: `nil` / `0` (disabled)

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
   * if `cfgNormalization = k > 0`:

     * `ori = norm(pos)`
     * `new = norm(pred)`
     * `maxNew = ori * k`
     * if `new > maxNew`: `pred *= maxNew/new` ([Hugging Face][1])

This gives you parity with the pipeline’s behavior and avoids edge-case “blow-ups” at high guidance.

---

## 4) Make presets correct for Z-Image vs Z-Image-Turbo

### What should change

`Z-Image-Turbo` is intended to run with **guidance = 0** and very few steps (the HF example uses `num_inference_steps=9` and `guidance_scale=0.0`) ([Hugging Face][2]).
`Z-Image` (base) recommends **guidance ~ 3–5** and **steps ~ 28–50** ([Hugging Face][3]).

### Work items

**A. CLI: detect whether the user explicitly set flags**
Right now CLI defaults always come from `ZImageModelMetadata` (Turbo-specific). Change `Sources/ZImageCLI/main.swift` to track whether the user explicitly provided:

* `--steps`
* `--guidance`
* `--width/--height`

Then after parsing:

* if `--model` is `Tongyi-MAI/Z-Image` (or user uses a new `--variant z-image`):

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

## 5) Ensure scheduler/config differences are truly “data driven”

In your attached repos, the main config delta that matters is the scheduler `shift` (Turbo vs base differ). Your code already reads `scheduler_config.json`, so you likely get this “for free”—but add explicit validation:

### Work items

* Add a small log line during scheduler construction:

  * `logger.info("Scheduler shift=\(modelConfigs.scheduler.shift) dynamic=\(modelConfigs.scheduler.useDynamicShifting)")`
* Add a unit test that loads configs from a local snapshot directory and asserts:

  * `Z-Image-Turbo.shift != Z-Image.shift` (using the two attached repo directories as fixtures in tests, if you want to keep it fully offline)

---

## 6) Testing strategy

### A. Fast “offline config” tests (CI-friendly)

No real weights needed:

* parse configs for both models
* confirm shard resolution finds the correct shard count from the index json
* confirm scheduler/timesteps shape and monotonicity

### B. Integration tests (skipped in CI)

Add a new integration test similar to existing ones:

* `testBasicGeneration_ZImage_BaseModel()`

  * model: `"Tongyi-MAI/Z-Image"`
  * steps: 28 (lower bound) to reduce runtime
  * guidance: 4
  * resolution: 512 (to reduce memory/time)
  * assert output exists and is a valid PNG

(Keep the current 8-bit Turbo integration tests as-is; they’re valuable for regression.)

---

## 7) Docs / examples updates

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

---

## 8) Concrete file-by-file checklist

**Core**

* `Sources/ZImage/Weights/ModelPaths.swift`

  * refactor `ZImageRepository` to support both ids + cache dirs
* `Sources/ZImage/Support/ModelMetadata.swift`

  * split Turbo-specific constants into per-model presets
* `Sources/ZImage/Pipeline/ZImagePipeline.swift`

  * add request fields; implement cfg truncation + normalization in denoise loop
  * include base model in `areZImageVariants`
* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

  * mirror the same CFG truncation/normalization changes (keep behavior consistent)

**CLI**

* `Sources/ZImageCLI/main.swift`

  * add `--variant turbo|z-image` (optional but recommended)
  * or auto-detect when `--model` matches known ids
  * only apply preset defaults when the user didn’t explicitly set those flags

**Tests**

* `Tests/.../PipelineIntegrationTests.swift`

  * add base model test (skipped in CI)
* add small config-only tests to validate parsing/resolution

---

## Definition of done

1. `ZImageCLI -p "..." -m Tongyi-MAI/Z-Image` produces a valid image using sensible defaults (steps/guidance) without the user hand-tuning.
2. `ZImageCLI` still works unchanged for Turbo users.
3. `cfg_truncation` and `cfg_normalization` behavior matches the reference pipeline logic ([Hugging Face][1]).
4. Tests cover:

   * config parsing for both model dirs
   * at least one integration generation run for base model (manual/skip in CI)

---

If you want, I can turn this plan into a sequence of PR-sized commits (with exact diffs per file) starting with “model registry + CLI presets” and then “CFG truncation/normalization parity.”

[1]: https://huggingface.co/spaces/AiSudo/ZIT-Controlnet/raw/d2c9b66866ffcf001aeffec3c19628902f112d22/videox_fun/pipeline/pipeline_z_image.py "huggingface.co"
[2]: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo?utm_source=chatgpt.com "Tongyi-MAI/Z-Image-Turbo"
[3]: https://huggingface.co/Tongyi-MAI/Z-Image?utm_source=chatgpt.com "Tongyi-MAI/Z-Image"
