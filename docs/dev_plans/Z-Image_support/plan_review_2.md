Here’s my evaluation of `docs/dev_plans/Z-Image_support/plan.md`, cross-checked against the **current Swift codebase** and the **two HF snapshot archives** you attached (Turbo + Base).

## Verdict

The plan is fundamentally sound: it correctly identifies that “supporting Z-Image” is mostly about **first-class model selection + sane defaults + a couple of pipeline knobs**, not a full architectural rewrite. It also targets the right files for refactors (`ModelMetadata`, `ModelPaths`, `areZImageVariants`, CLI defaults, tests).

That said, there are a few **material gaps / under-specified areas** that will bite you during implementation (mostly around *step semantics parity*, *exact CFG-normalization math*, and *test fixtures + resource packaging*).

---

## What the plan gets right

### 1) “Compatibility baseline” assessment is correct

From your attached Turbo/Base repositories, the **transformer / text_encoder / vae / scheduler / tokenizer layout and configs are effectively the same**, with the main meaningful delta being **scheduler shift** (Turbo vs Base). So the plan’s framing (“mostly presets + parity knobs”) is appropriate.

### 2) Defaults/presets match the official recommendations

The plan correctly calls out that:

* Turbo should run **guidance = 0** and very few steps (HF example: `num_inference_steps=9`, `guidance_scale=0.0`). ([Hugging Face][1])
* Base Z-Image recommends **guidance 3–5** and **28–50 steps**. ([Hugging Face][2])

### 3) Adding `cfg_truncation` + `cfg_normalization` is a good “pipeline compatibility” move

Those knobs are part of the reference pipeline behavior (truncate guidance after a normalized threshold; optional renorm/clamp of CFG output). The plan’s intended control flow matches the reference implementation. ([Hugging Face][3])

---

## Key gaps / risks to fix in the plan

### A) **Step semantics parity (high risk)**

Your codebase already encodes the statement “`steps = 9` gives 8 DiT forwards” in `ModelMetadata.swift`, and the Turbo model card says the same. ([Hugging Face][1])
But **your Swift scheduler/denoise loop currently appears to run one transformer forward per loop iteration**, i.e. “steps == forwards” (unless there’s an implicit skip elsewhere). This is exactly the kind of subtle mismatch that will cause “it works but doesn’t match reference output” confusion.

**Plan amendment:** add an explicit work item:

* Verify whether Swift’s current interpretation of `num_inference_steps` matches Diffusers Z-Image for Turbo *and* Base.
* If it’s off-by-one, decide whether to:

  * keep public CLI semantics aligned with HF (`--steps 9` behaves like HF), or
  * keep internal semantics “forwards == steps” and adjust presets/help text accordingly.
* Add a small regression test that asserts the number of denoise iterations / forward calls for the Turbo preset.

### B) **CFG normalization math is under-specified**

The plan says “clamp `||pred||` to `k * ||pos||`”, but the reference behavior is **not a single global norm**; it’s computed in a tensor-shaped way (norm over channel dimension, preserving spatial). If you implement the wrong reduction axes, you’ll get different aesthetics and stability characteristics.

**Plan amendment:** specify the exact intended computation:

* `ori_std = norm(pos, dim=channel, keepdim=true)`
* `new_std = norm(pred, dim=channel, keepdim=true)`
* clamp factor applied elementwise with epsilon to avoid div-by-zero. ([Hugging Face][3])

### C) Model registry refactor: missing call sites

The plan mentions `ZImageRepository`, cache naming, `areZImageVariants`, CLI defaults—but in your actual codebase, **default model id flows through more places** than those bullets capture (e.g. `PipelineSnapshot`, `ModelResolution` defaults, and both control-pipeline paths referencing `ZImageRepository.id`).

**Plan amendment:** explicitly include these files/edges in the checklist:

* `Sources/ZImage/Pipeline/PipelineSnapshot.swift` (defaultModelId/defaultRevision)
* `Sources/ZImage/Weights/ModelResolution.swift` (default model id + revision + cache dir decisions)
* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift` already noted, but ensure both of its “entry points” inherit the same preset model selection logic (you have multiple denoise paths).

### D) Test fixture packaging is glossed over

The plan suggests using the two attached model directories as fixtures for offline tests. That works locally, but for an actual repo/CI setup you’ll need to either:

* commit **minimal JSON fixtures** (configs + `*.index.json`), or
* generate fixtures as part of tests, or
* gate the offline tests behind an environment variable that points to local snapshots.

**Plan amendment:** decide and document one approach. Otherwise “offline config tests” won’t be reliably runnable by contributors.

### E) Base model practicality on Apple Silicon (memory/runtime)

Base Z-Image at 1024² with 50 steps is **much heavier** than Turbo. The plan mentions lowering to 512 for integration tests (good), but it should also add:

* CLI warnings / docs on recommended settings for Apple Silicon (resolution and step count)
* recommendation to use quantized transformer weights for Base in practice (your toolchain already supports quantization)

This matters because otherwise “support is added” but most users will just OOM or wait forever.

---

## Notes on correctness vs reference pipeline

The plan states the CFG combine formula as:

`pred = pos + scale * (pos - neg)` ([Hugging Face][3])

That matches the referenced pipeline file, but it’s *non-standard vs the common* `neg + scale*(pos-neg)` pattern. Since you’re explicitly targeting pipeline parity, this is fine—just make sure you implement **exactly this** for both Base and Turbo when CFG is active.

---

## Recommended tightened “Definition of Done” (small edits)

Add these to your existing DoD:

1. **Step semantics verified**: Turbo preset produces the same number of DiT forwards as the HF recommended invocation (`steps=9` → “8 forwards” behavior). ([Hugging Face][1])
2. **CFG normalization axes match reference** (channel-wise norm with keepdim; epsilon-safe). ([Hugging Face][3])
3. **Registry is the single source of truth** for:

   * default model id
   * default cache dir naming
   * default step/guidance presets (CLI + library)

---

## Bottom line

If you add the missing step-parity validation, tighten the CFG-normalization spec, and make the registry refactor hit *all* default-model call sites (plus make the offline tests realistically runnable), this plan is in good shape and should result in clean “Turbo + Base” support with minimal churn.

If you want, I can produce a PR-ordered patch plan (commit-by-commit, exact file diffs) that applies the above amendments while keeping current Turbo behavior as the default.

[1]: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo "Tongyi-MAI/Z-Image-Turbo · Hugging Face"
[2]: https://huggingface.co/Tongyi-MAI/Z-Image "Tongyi-MAI/Z-Image · Hugging Face"
[3]: https://huggingface.co/spaces/AiSudo/ZIT-Controlnet/raw/d2c9b66866ffcf001aeffec3c19628902f112d22/videox_fun/pipeline/pipeline_z_image.py "huggingface.co"
