Overall the plan is **directionally right** and would get you to “`Tongyi-MAI/Z-Image` works out of the box with sane defaults” with relatively small code churn. The biggest value is correctly identifying that *your current code is Turbo-defaulted everywhere* (steps/guidance/model id), so adding **model-aware presets** is the real “support” work.

## What the plan gets right (and matches upstream)

* **Different defaults for Turbo vs Base are mandatory.**
  The official `Z-Image` card explicitly calls out **CFG ✅**, **steps 28–50**, and **guidance 3–5**, plus negative prompting support. ([Hugging Face][1])
  The Turbo card’s example uses **`num_inference_steps=9`** (noting it yields 8 DiT forwards) and **`guidance_scale=0.0`**. ([Hugging Face][2])
  So your plan’s “only apply new defaults when `--model` is base and user didn’t override flags” is exactly the right direction.

* **Scheduler/config should stay data-driven, and Turbo vs Base differ in `shift`.**
  `Z-Image` scheduler config has `shift: 6.0` ([Hugging Face][3]) while Turbo has `shift: 3.0`. ([Hugging Face][4])
  Adding an assertion/log that you’re actually reading these from the snapshot is a good guardrail.

* **CFG truncation + normalization logic is correctly described.**
  The reference implementation computes `t_norm = (1000 - t) / 1000`, sets guidance to 0 when `t_norm > cfg_truncation`, and optionally clamps the guided prediction norm relative to the positive prediction norm.
  So the plan’s implementation sketch is faithful.

## Gaps / risks to fix before implementing

### 1) Weight/index selection needs to become “variant-aware” (bf16 is already real)

Right now your dynamic resolvers assume fixed index names like:

* `transformer/diffusion_pytorch_model.safetensors.index.json`
* `text_encoder/model.safetensors.index.json`

That’s fragile as soon as a repo contains multiple “variants” (bf16 / fp16 / etc.). HF already has an explicit **bf16 variant workflow** for Turbo using `variant="bf16"` (and warns about mixed bf16/non-bf16 filenames). ([Hugging Face][5])

**What to add (missing from the plan):**

* A `variant` concept in your model spec (even if optional), and deterministic selection of:

  * the right `*.safetensors.index.json` (or shard set) for that variant
  * matching component variants (transformer/text_encoder/vae) to avoid mixed loads
* In `ModelResolution.filePatterns`, allow downloading **only** the selected variant (otherwise you can accidentally pull *both* sets, or select the “wrong” shards by directory scan).

This is the single highest “future breakage” risk.

### 2) `cfg_normalization` type/semantics: upstream is **bool**, code behaves like a **factor**

The `Z-Image` model card shows `cfg_normalization=False` (boolean). ([Hugging Face][1])
But the reference code path effectively treats it like a multiplicative cap (it converts to float and clamps norms).

**Recommendation:** keep your API ergonomic but compatible:

* Implement **two layers**:

  * public: `cfgNormalization: Bool` (matches model card)
  * advanced: `cfgNormalizationFactor: Float?` (optional override; `true` maps to `1.0`)
    This avoids locking you into a possibly-moving upstream signature.

### 3) CFG truncation optimization vs strict parity

The reference code still executes the “2× batch” path even when it truncates guidance to 0 (wasted compute but behaviorally consistent).
Your plan suggests skipping CFG when `currentGuidanceScale == 0`. That’s *probably* safe, but it is a (minor) behavioral divergence.

**Call it explicitly** in code/comments: “perf optimization; should be numerically equivalent.”

### 4) Presets should include `maxSequenceLength`

Your CLI already has `--max-sequence-length`. For “first-class support”, the preset for `Z-Image` should likely recommend **1024** when users want long prompts (keep 512 as fast default if you prefer). HF guidance explicitly mentions the 1024 option in prompting guidance. ([Hugging Face][6])

### 5) Tests: avoid assuming you can check in full snapshots

The plan talks about “two attached model snapshots” and using them as fixtures. That’s rarely practical (size/licensing/CI). Prefer:

* **config-only tests** that download `*.json` + tokenizer files (no safetensors)
* integration tests gated behind an env var / local-only target

## Suggested reprioritized implementation order (more robust than the plan’s ordering)

1. **Model registry + CLI/library presets** (Base vs Turbo defaults; update `areZImageVariants` to include base)
2. **Variant-aware weight/index resolution** (bf16-proofing; deterministic shard selection) ([Hugging Face][5])
3. **CFG truncation + normalization** in both pipelines, ideally via a shared helper to avoid drift
4. **Tests + docs/examples** (show Turbo vs Base commands; document Turbo’s “guidance 0” and Base’s “guidance 3–5, steps 28–50”) ([Hugging Face][2])

## Bottom line

The plan is **solid and mostly accurate**, and the CFG math matches reference behavior.
The main thing it misses is **making weight resolution robust to repo variants** (bf16, future quantized variants, multiple index files), which is already a reality in the ecosystem. ([Hugging Face][5])

[1]: https://huggingface.co/Tongyi-MAI/Z-Image "https://huggingface.co/Tongyi-MAI/Z-Image"
[2]: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"
[3]: https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/scheduler/scheduler_config.json "https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/scheduler/scheduler_config.json"
[4]: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json"
[5]: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/102 "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/102"
[6]: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8 "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8"
