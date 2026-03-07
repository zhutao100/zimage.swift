# Numerical parity & golden fixtures

This repo sometimes needs **numerical parity checks** against an external reference implementation (typically Python/Transformers) while porting models to **MLX Swift**.

These checks are intentionally:

- **opt-in** (so `swift test` stays fast and hermetic by default)
- **deterministic** (fixed inputs + small slices)
- **diagnostic** (designed to localize drift quickly)

## Core principles (portable to any model)

### 1) Make the inputs deterministic

- Use a deterministic prompt and a deterministic synthetic image (or a tiny fixed fixture file).
- Make preprocessing deterministic: explicit resize policy, explicit color space, explicit mean/std.
- Record the exact snapshot/version you used (HF snapshot hash, config summary).

### 2) Align numerics end-to-end (device + dtype)

Golden parity is only meaningful if **the reference run and Swift run use the same numerical regime**:

- **Device** (CPU vs GPU/MPS) can change kernels and accumulation order.
- **DType** (FP16 vs BF16 vs FP32) can change intermediate rounding and saturation.
- Some reference stacks force **float32** for specific blocks (common for rotary/positional math).

Guideline: when a golden fixture is produced under a given device/dtype, the Swift test should **force both weights and inputs** to the same dtype for that parity run. Casting only one side (weights *or* inputs) is a common source of “mysterious” drift.

### 3) Localize drift early (compare intermediates before logits)

When a golden check fails, avoid jumping straight to “logits mismatch”:

1. Compare **vision embeddings** (post-vision encoder or post-merger). Small drift here will amplify downstream.
2. Compare **text embeddings** (token embedding output).
3. Compare **position IDs / RoPE inputs** used by attention.
4. Only then compare logits/top-k.

Prefer summary statistics (mean/std/l2 + a few elements) over full tensors.

### 4) Treat layout/conventions as part of the model

Ports often fail due to *convention mismatches* rather than “math bugs”:

- Tensor layout (channels-first vs channels-last, patch packing order, flatten order)
- Weight layout expectations (e.g., conv kernels)
- RoPE conventions (rotate-half variant, cos/sin expansion, any model-specific indexing)
- Decoder block ordering (norm/residual placement)

Always cross-check against the **reference implementation’s code**, not just config values.

### 5) Avoid non-contiguity pitfalls in reference tooling

In Python/PyTorch, tensors coming from preprocessors or device transfers may be non-contiguous.

- Prefer `.reshape(...)` over `.view(...)` unless you know the tensor is contiguous.
- If you must use `.view(...)`, use `.contiguous()` explicitly first.

This reduces “works on CPU, fails on GPU/MPS” issues in fixture generation scripts.

### 6) Beware in-place mutation when instrumenting the Python reference

When building “intermediate parity” fixtures, be careful with Python containers passed into modules:

- some models mutate lists/dicts of tensors in-place (e.g. reassigning feature levels),
- which can silently corrupt what you think you are recording.

### 7) Minimize version drift in preprocessing

Hugging Face processors can change behavior across versions and “fast vs slow” implementations.

Guideline:

- Treat the image processor/tokenizer configuration as part of the golden fixture.
- Prefer recording the reference library versions and any critical processor flags (e.g. `use_fast`) in the fixture metadata or the generator output.

### 8) If the model performs internal selection, record the indices

Many detectors are effectively “two-stage”: the decoder queries are created by selecting top-scoring encoder locations.
Tiny numeric drift (device kernels, dtype promotions, half precision rounding) can change the **ordering** of that selection even when the underlying features are close.

Golden probes that assume “query i is query i” become meaningless if the query identities differ.

Guideline:

- Capture the reference selection indices (e.g. encoder `topk_ind`, kept indices after masking) in the fixture.
- In Swift, either:
  - override the selection indices during the parity run, or
  - map “python query i” → “swift query j” by matching the underlying selected encoder index.

This makes golden slices compare like-for-like, instead of accidentally comparing different objects.

### 9) Use dtype-appropriate sentinel values for masking

Masking often uses “very large” numbers so that invalid entries disappear under `min/max/topk`.
In half precision, `Float.greatestFiniteMagnitude` does **not** fit and becomes `inf`, which can turn downstream ops into `NaN` (especially if you later apply `log`, `exp`, or reduce across mixed valid/invalid values).

Guideline:

- Use dtype-aware finite maxima (e.g. `Float16.greatestFiniteMagnitude` when running FP16).
- Be deliberate about dtype promotions in comparisons (some backends compare FP16 values against FP32 scalars).

### 10) Boundary-sensitive ops can amplify tiny errors

`grid_sample(align_corners=false)` and deformable attention are highly sensitive to sampling locations near the `[0, 1]` boundary.
In FP16, rounding can push an “almost-in-bounds” coordinate slightly out of bounds, triggering zero padding and disproportionately large downstream drift.

Guideline:

- Keep the sampling-location math faithful to the reference implementation (including dtype).
- When diagnosing parity, always include at least one probe that is fully in-bounds to separate “core math mismatch” from “OOB/border effects”.
- Prefer probe indices that are stable across backends; if an index is consistently border-sensitive, replace it with a nearby index that exercises the same codepaths.
