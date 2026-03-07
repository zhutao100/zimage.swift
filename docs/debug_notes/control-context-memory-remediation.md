# Control-Context Memory Remediation

Validated on March 7, 2026 against the current Swift repo state.

This note replaces older conclusions that became partially stale after the control-path unload work, the VAE attention chunking change, and the phase-memory telemetry landed.

## Validation Basis

- Source inspection of:
  - `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
  - `Sources/ZImage/Model/VAE/AutoencoderEncoder.swift`
  - `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`
  - `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift`
  - `Sources/ZImage/Model/Transformer/ZImageControlTransformerBlock.swift`
- Source inspection of the local Diffusers reference:
  - `~/workspace/custom-builds/diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py`
- Four measured high-resolution control probes on March 7, 2026:
  - phase 0 baseline before the post-build cache-release change
  - phase 1 follow-up after the post-build cache-release change
  - phase 2 follow-up after the on-demand VAE encoder / deferred decoder split
  - phase 3 follow-up after the incremental ControlNet hint transport change

```bash
./.build/xcode/Build/Products/Release/ZImageCLI control \
  --prompt "memory validation" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --width 1536 \
  --height 2304 \
  --steps 1 \
  --guidance 0 \
  --seed 1234 \
  --log-control-memory \
  --no-progress \
  --output /tmp/zimage-control-remediation/phase0_memory.png
```

Diffusers was not executed in this validation pass. Any statement below about Diffusers behavior is source-based unless explicitly marked as measured.

## Current Verdict

The large control-path memory footprint is still not caused by the stored `controlContext` tensor itself. After phase 3 landed, the current dominant contributors are:

1. full-resolution VAE encode during `buildControlContext(...)`
2. the remaining transformer and ControlNet denoising activations after the stacked-hint transport was removed
3. final decode-time VAE residency, which is now deferred and bounded rather than overlapping the whole control lifecycle

The final control tensor shape from the measured high-resolution run was `[1, 33, 1, 288, 192]`. That is only a few MiB in bf16 or fp32. The large footprint comes from how it is built and what remains cached afterward, not from the tensor that is ultimately kept.

## What Is Already True In The Current Repo

These older failure modes are no longer current:

- `ZImageControlPipeline` unloads the transformer, ControlNet, and active LoRA state before `buildControlContext(...)`.
- prompt-embedding cache cleanup already happens before the control build begins.
- the VAE mid-block self-attention path is query-chunked by default via `VAEAttention.defaultQueryChunkSize = 1024`.
- the control pipeline now uses an encoder-only VAE for control/inpaint encode and a decoder-only VAE for final decode.
- ControlNet hint transport now keeps the current control state separate from accumulated skip hints instead of restacking them at every control block.
- the control path now materializes the stored control-context tensor, clears cache immediately, and logs `control-context.after-clear-cache` before transformer/controlnet reload.
- `--log-control-memory` already emits resident, active, cache, and peak markers around the main control-path phases.

Those mitigations are already visible in the measured baseline:

- `control-context.after-baseline-reduction`: resident `371.59 MiB`, active `162.37 MiB`, cache `0 B`
- `control-context.before-build`: resident `371.59 MiB`, active `162.37 MiB`, cache `0 B`

So the current problem is not "the transformer stayed resident during control-context build" and not "the VAE attention path is still fully unchunked."

## What Is Still True In The Current Repo

### 1. Phase 1 fixed the retained-cache reload boundary

`generateCore(...)` now does:

- `let result = try buildControlContext(...)`
- `MLX.eval(result)`
- `let materializedControlContext = result.asType(vae.dtype)`
- `MLX.eval(materializedControlContext)`
- `Memory.clearCache()`
- logs `control-context.after-clear-cache`
- then proceeds toward transformer/controlnet reload

Measured effect on the high-resolution probe:

- phase 0 `control-context.after-eval`: cache `28.07 GiB`
- phase 1 `control-context.after-eval`: cache `28.08 GiB`
- phase 1 `control-context.after-clear-cache`: cache `0 B`
- phase 0 `denoising.before-start`: cache `28.08 GiB`
- phase 1 `denoising.before-start`: cache `5.71 KiB`

Measured process impact:

- `/usr/bin/time -l` maximum resident set size stayed effectively flat:
  - phase 0: `42,832,363,520` bytes
  - phase 1: `42,830,249,984` bytes
- `/usr/bin/time -l` peak memory footprint dropped by `25,890,701,600` bytes (`24.11 GiB`):
  - phase 0: `112,574,979,616` bytes
  - phase 1: `86,684,278,016` bytes

So the post-build barrier was worth landing, and it should be treated as part of the retained control-path policy now.

### 2. Phase 2 split the control VAE lifecycle

`ZImageControlPipeline` no longer keeps a full `AutoencoderKL` resident through the control request lifecycle.

The control path now does this instead:

- load `AutoencoderEncoderOnly` on demand for control or inpaint encode
- release that encoder immediately after the typed control context is materialized
- defer `AutoencoderDecoderOnly` loading until final decode
- unload the decoder immediately after `decode.after-eval`

The change was validated with targeted unit coverage:

- `VAEComponentTests.testEncoderOnlyMatchesFullAutoencoderEncode`
- `VAEComponentTests.testDecoderOnlyMatchesFullAutoencoderDecode`

Measured effect on the high-resolution probe versus phase 1:

- `control-context.after-baseline-reduction` dropped from `370.17 MiB` to `273.34 MiB`
- `control-context.after-eval` dropped from `463.17 MiB` to `365.64 MiB`
- `control-context.after-clear-cache` dropped from `422.67 MiB` to `325.14 MiB`
- `/usr/bin/time -l` maximum resident set size dropped from `42,830,249,984` bytes to `42,659,020,800` bytes
- `/usr/bin/time -l` peak memory footprint dropped from `86,684,278,016` bytes to `86,580,223,280` bytes

The fixed-seed phase 2 output stayed bit-identical to phase 0 and phase 1.

### 3. Phase 3 removed repeated stacked ControlNet hint transport

`ZImageControlTransformerBlock` no longer rebuilds `MLX.stacked(allC, axis: 0)` at every block.

The control path now carries:

- the current control state
- an incrementally grown hint list

The legacy stacked representation is now only materialized in the compatibility wrapper used by targeted tests. The production `ZImageControlNetModel` path converts the accumulated hints to `ZImageControlBlockSamples` at the transformer boundary instead of rebuilding a stacked tensor at every control block.

Measured effect on the high-resolution probe versus phase 2:

- `/usr/bin/time -l` peak memory footprint dropped from `86,580,223,280` bytes to `59,328,863,512` bytes on the repeat run
- `/usr/bin/time -l` maximum resident set size stayed effectively flat:
  - phase 2: `42,659,020,800` bytes
  - phase 3 repeat: `42,656,612,352` bytes
- the fixed-seed phase 3 output stayed bit-identical to phase 0, phase 1, and phase 2

One marker moved in a less intuitive way:

- `control-context.after-baseline-reduction` resident bytes rose to about `1.0 GiB`, while active bytes stayed at `67.87 MiB` and cache stayed `0 B`

That appears to be resident-only accounting noise rather than renewed active MLX pressure. This is an inference from the surrounding measurements:

- `control-context.after-eval` and `control-context.after-clear-cache` stayed aligned with phase 2
- `denoising.before-start` stayed aligned with phase 2
- overall peak footprint dropped by about `25.38 GiB`

## What The Measured Probes Say

Measured high-resolution control probes, March 7, 2026:

- Phase 0 baseline:
  - `prompt-embeddings.after-clear-cache`: resident `36.08 GiB`, active `29.34 GiB`, cache `0 B`
  - `control-context.after-baseline-reduction`: resident `371.59 MiB`, active `162.37 MiB`, cache `0 B`
  - `control-context.after-eval`: resident `464.34 MiB`, active `172.72 MiB`, cache `28.07 GiB`
  - `denoising.before-start`: resident `29.63 GiB`, active `29.34 GiB`, cache `28.08 GiB`
  - `decode.after-eval`: resident `124.06 MiB`, active `192.86 MiB`, cache `38.95 GiB`, MLX peak `39.03 GiB`
  - `/usr/bin/time -l` maximum resident set size: `42,832,363,520` bytes
  - `/usr/bin/time -l` peak memory footprint: `112,574,979,616` bytes
- Phase 1 after the post-build barrier:
  - `control-context.after-baseline-reduction`: resident `370.17 MiB`, active `162.37 MiB`, cache `0 B`
  - `control-context.after-eval`: resident `463.17 MiB`, active `165.86 MiB`, cache `28.08 GiB`
  - `control-context.after-clear-cache`: resident `422.67 MiB`, active `165.86 MiB`, cache `0 B`
  - `denoising.before-start`: resident `29.60 GiB`, active `29.34 GiB`, cache `5.71 KiB`
  - `decode.after-eval`: resident `424.09 MiB`, active `192.86 MiB`, cache `38.95 GiB`, MLX peak `39.03 GiB`
  - `/usr/bin/time -l` maximum resident set size: `42,830,249,984` bytes
  - `/usr/bin/time -l` peak memory footprint: `86,684,278,016` bytes
- Phase 2 after the VAE lifecycle split:
  - `prompt-embeddings.after-clear-cache`: resident `35.58 GiB`, active `29.18 GiB`, cache `0 B`
  - `control-context.after-baseline-reduction`: resident `273.34 MiB`, active `67.87 MiB`, cache `0 B`
  - `control-context.after-eval`: resident `365.64 MiB`, active `78.22 MiB`, cache `28.07 GiB`
  - `control-context.after-clear-cache`: resident `325.14 MiB`, active `71.36 MiB`, cache `0 B`
  - `denoising.before-start`: resident `29.50 GiB`, active `29.19 GiB`, cache `65.30 MiB`
  - `decode.after-eval`: resident `420.88 MiB`, active `127.48 MiB`, cache `39.00 GiB`, MLX peak `38.88 GiB`
  - `/usr/bin/time -l` maximum resident set size: `42,659,020,800` bytes
  - `/usr/bin/time -l` peak memory footprint: `86,580,223,280` bytes
- Phase 3 after incremental hint accumulation:
  - `prompt-embeddings.after-clear-cache`: resident `33.81 GiB`, active `29.18 GiB`, cache `0 B`
  - `control-context.after-baseline-reduction`: resident `1012.41 MiB`, active `67.87 MiB`, cache `0 B`
  - `control-context.after-eval`: resident `364.31 MiB`, active `71.36 MiB`, cache `28.08 GiB`
  - `control-context.after-clear-cache`: resident `323.81 MiB`, active `71.36 MiB`, cache `0 B`
  - `denoising.before-start`: resident `29.50 GiB`, active `29.19 GiB`, cache `65.30 MiB`
  - `decode.after-eval`: resident `419.42 MiB`, active `127.48 MiB`, cache `39.00 GiB`, MLX peak `37.02 GiB`
  - `/usr/bin/time -l` maximum resident set size: `42,656,612,352` bytes
  - `/usr/bin/time -l` peak memory footprint: `59,328,863,512` bytes

Three points matter here:

1. The build-time baseline reduction is working. The repo now really does collapse resident memory before the control VAE encode.
2. The post-build barrier removed the retained-cache overlap at the denoising start boundary, and phase 2 lowered the baseline even further by deferring VAE residency.
3. Phase 3 removed a large denoising-memory multiplier without changing the image, so the remaining high-resolution pressure is now more clearly concentrated in the encode/decode workspace and the core denoiser itself.

The `/usr/bin/time -l` footprint is much higher than the MLX "peak" counter, which is expected. They are not the same metric.

## Diffusers Parity Check

The local Diffusers control pipeline still does the same high-level control-image preparation:

- preprocess control image
- `self.vae.encode(control_image)`
- shift/scale to latent space
- `unsqueeze(2)` for control-context shape

That means the Swift port is not paying the control-context cost because it invented a different algorithm. The remaining gap is about residency policy and data transport, not about basic control-image semantics.

## Ranked Remediation Order

1. No additional code phase from this note is currently required.
2. Only consider tiled/sliced encode if the remaining `~59.3 GiB` high-resolution peak footprint is still above the deployment target for a specific machine or workflow.

The measured execution plan for these changes lives in `docs/dev_plans/control-context-memory-remediation.md`.
