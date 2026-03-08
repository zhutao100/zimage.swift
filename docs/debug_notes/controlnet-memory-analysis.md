# ControlNet Memory Analysis

This note was originally written before the March 7, 2026 control-path remediation phases.
It was pruned on March 8, 2026 so that it keeps only the observations that are still accurate and relevant after phases 1 through 3 landed.

Use this as a narrow ControlNet-specific follow-up note.
For the validated phase-by-phase measurements and the fixes that already landed, see:

- `docs/debug_notes/control-context-memory-remediation.md`
- `docs/dev_plans/control-context-memory-remediation.md`

## Validation Basis

- Current source inspection of:
  - `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - `Sources/ZImage/Model/VAE/AutoencoderEncoder.swift`
  - `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
  - `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift`
- Current measured state from the phase 3 remediation outcome in:
  - `docs/debug_notes/control-context-memory-remediation.md`
- Local Diffusers reference source inspection:
  - `~/workspace/custom-builds/diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py`

No new model run was performed for this cleanup pass.
Any statement below is source-based unless it explicitly cites the measured phase 3 state.

## Still-Relevant Findings

### 1. The stored control-context tensor is still not the memory problem

The final control-context tensor remains small relative to the observed process footprint.

The measured high-resolution reference run in `control-context-memory-remediation.md` still ends with:

- control-context shape: `[1, 33, 1, 288, 192]`
- phase 3 peak memory footprint: `59,328,863,512` bytes

That means the remaining issue is still dominated by how control context is built and how the control denoiser runs, not by the size of the tensor that is ultimately stored.

### 2. Full-resolution control and inpaint VAE encode remain the main build-time hotspot

The current control path still builds control latents by resizing the full input image and then running a monolithic VAE encode:

- `encodeImageToLatents(...)` calls `vae.encode(normalized)`
- the inpaint branch in `buildControlContext(...)` also calls `vae.encode(maskedNormalized)`

The encoder-only VAE split reduced residency pressure, but it did not change the fact that control and inpaint inputs are still encoded at full requested resolution in one pass.

There is still no tiled or striped control-image VAE encode path in the current Swift repo.

### 3. The early transformer and ControlNet load churn is now fixed, but the denoising load boundary still dominates

The March 8, 2026 phase 1 follow-up work changed the control lifecycle to this:

1. load tokenizer only
2. load text encoder and build prompt embeddings
3. build control context without transformer or ControlNet residency
4. load transformer and optional ControlNet right before denoising

That means the old early-load complaint is no longer current.
The remaining practical issue is narrower:

- the deferred denoising load boundary still climbs back to about `29.5 GiB` resident before the first denoising step
- peak memory footprint stays around `59.3 GiB` even after the prompt-stage baseline drops sharply

So the remaining question is not whether eager loading can be removed. It can, and it was.
The remaining question is whether the denoising load boundary itself can be reduced or attributed more precisely.

### 4. Diffusers parity is still about math, not lifecycle

The local Diffusers control pipeline still follows the same high-level control-image semantics:

- preprocess control image
- run VAE encode
- shift and scale into latent space
- add the singleton frame dimension for ControlNet context

So the remaining Swift gap is not about having invented the wrong control algorithm.
It is about lifecycle policy and optional memory-management tools around that algorithm.

The two notable tools still missing on the Swift side are:

- tiled control and inpaint VAE encode
- a more aggressive deferred-loading or offload policy for ControlNet-specific modules

## What Is No Longer Current

The original version of this note also recommended several changes that are now already implemented and should not be treated as open work:

- post-build cache barrier after control-context materialization
- full-lifecycle `AutoencoderKL` residency in the control pipeline
- repeated `MLX.stacked(allC, axis: 0)` hint transport inside the control transformer blocks
- eager transformer and ControlNet loading before prompt and control-context work

Those are now part of the current repo state and the measured remediation history.

## Remaining Practical Question

After phase 3, the measured high-resolution reference run still peaks around `59.3 GiB`.

The remaining practical questions are:

- how much of the remaining `~29.5 GiB` denoising-start residency is unavoidable model residency versus lifecycle overhead
- whether tighter telemetry can isolate the residual jump cleanly enough to justify any further loader surgery
- whether tiled control and inpaint VAE encode is worth attempting once it is clear that it will not move the denoising-boundary peak by itself

The follow-up execution plan lives in `docs/dev_plans/controlnet-memory-followup.md`.
