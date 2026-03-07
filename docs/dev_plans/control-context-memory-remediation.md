# Control-Context Memory Remediation Plan

This plan turns the validated findings in `docs/debug_notes/control-context-memory-remediation.md` into a measured implementation sequence.

Execution status: completed on March 7, 2026.

## Scope

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Model/VAE/*`
- `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift`
- `Sources/ZImage/Model/Transformer/ZImageControlTransformerBlock.swift`
- targeted unit coverage under `Tests/ZImageTests/`
- supporting docs under `docs/`

## Baseline Protocol

Every phase must be evaluated before it is wrapped and committed.

Repo verification:

```bash
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests
xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

High-resolution memory probe:

```bash
/usr/bin/time -l ./.build/xcode/Build/Products/Release/ZImageCLI control \
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
  --output /tmp/zimage-control-remediation/<phase>_memory.png
```

Fixed-seed quality probe:

```bash
/usr/bin/time -l ./.build/xcode/Build/Products/Release/ZImageCLI control \
  --prompt "a stone archway covered in moss, cinematic lighting" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --width 512 \
  --height 512 \
  --steps 4 \
  --guidance 0 \
  --seed 1234 \
  --no-progress \
  --output /tmp/zimage-control-remediation/<phase>_quality.png
```

Required metrics per phase:

1. `control-context.after-baseline-reduction`
2. `control-context.after-eval`
3. `control-context.after-clear-cache` once phase 1 lands
4. `denoising.before-start`
5. `decode.after-eval`
6. `/usr/bin/time -l` maximum resident set size
7. `/usr/bin/time -l` peak memory footprint
8. image drift versus the phase 0 baseline image and versus the immediately previous phase image

Preferred image metrics:

- mean absolute pixel error
- max absolute pixel delta
- PSNR

Acceptance bar:

- no broken image or NaN-like failure
- no unexpected regression in the high-resolution memory probe
- phase 1, 2, and 3 should preserve output semantics closely enough that the fixed-seed image drift is either zero or trivially small and explainable

## Phase 1: Post-Build Materialization Barrier

Status: completed on March 7, 2026

Objective:

- release retained MLX cache immediately after the control-context tensor is materialized in its stored dtype

Files:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Tests/ZImageTests/`
- `docs/`

Implementation notes:

- cast the built control context to the stored dtype
- force evaluation of the typed control context before clearing cache
- add a new telemetry phase after the cache clear
- keep the change narrow; do not mix VAE lifecycle refactors into this phase

Acceptance criteria:

- the control path logs `control-context.after-clear-cache`
- `denoising.before-start` shows reduced retained cache or resident pressure versus phase 0
- the fixed-seed quality probe remains effectively unchanged

## Phase 2: On-Demand Control VAE Encoder And Deferred Decoder

Status: completed on March 7, 2026

Objective:

- stop keeping a full `AutoencoderKL` resident across the entire control generation lifecycle

Files:

- `Sources/ZImage/Model/VAE/*`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Tests/ZImageTests/`
- `docs/`

Implementation notes:

- add an encoder-only VAE variant parallel to the existing decoder-only variant
- load the encoder only when control or inpaint inputs must be encoded
- release the encoder after control-context materialization
- defer decoder-only loading until final decode
- prefer reusing the existing VAE weight-mapping behavior instead of inventing a new mapping layer

Acceptance criteria:

- the control pipeline no longer needs a full `AutoencoderKL`
- high-resolution baseline reduction before control build stays low
- the post-build reload boundary improves further
- the fixed-seed quality probe remains effectively unchanged

## Phase 3: Incremental Control Hint Accumulation

Status: completed on March 7, 2026

Objective:

- reduce control-mode denoising memory by removing repeated stacked-hint transport

Files:

- `Sources/ZImage/Model/Transformer/ZImageControlTransformerBlock.swift`
- `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift`
- `Tests/ZImageTests/`
- `docs/`

Implementation notes:

- keep current control state separate from accumulated skip hints
- stop rebuilding `MLX.stacked(allC, axis: 0)` at every control block
- preserve the externally visible `ZImageControlBlockSamples` contract at the transformer boundary

Acceptance criteria:

- denoising memory improves on the high-resolution probe without changing the control outputs materially
- targeted tests cover the new hint-transport behavior where practical
- the fixed-seed quality probe remains effectively unchanged

## Execution Log

- Baseline: completed on March 7, 2026.
- Phase 0 high-resolution memory probe:
  - `prompt-embeddings.after-clear-cache`: resident `36.08 GiB`, active `29.34 GiB`, cache `0 B`
  - `control-context.after-baseline-reduction`: resident `371.59 MiB`, active `162.37 MiB`, cache `0 B`
  - `control-context.after-eval`: resident `464.34 MiB`, active `172.72 MiB`, cache `28.07 GiB`
  - `denoising.before-start`: resident `29.63 GiB`, active `29.34 GiB`, cache `28.08 GiB`
  - `decode.after-eval`: resident `124.06 MiB`, active `192.86 MiB`, cache `38.95 GiB`, MLX peak `39.03 GiB`
  - `/usr/bin/time -l` maximum resident set size: `42,832,363,520` bytes
  - `/usr/bin/time -l` peak memory footprint: `112,574,979,616` bytes
- Phase 0 fixed-seed quality probe:
  - output dimensions: `512x512`
  - output SHA-256: `2afd1fa9ba4398ad2b8b53510f44d602d5d7d5cc2631cee99d35c6d0752f8f70`
- Phase 1: completed on March 7, 2026.
  - Scope landed:
    - materialize the stored control-context tensor before reloading transformer/controlnet
    - clear MLX cache immediately after the typed control context is materialized
    - add `control-context.after-clear-cache` telemetry
  - Phase 1 high-resolution memory probe:
    - `control-context.after-baseline-reduction`: resident `370.17 MiB`, active `162.37 MiB`, cache `0 B`
    - `control-context.after-eval`: resident `463.17 MiB`, active `165.86 MiB`, cache `28.08 GiB`
    - `control-context.after-clear-cache`: resident `422.67 MiB`, active `165.86 MiB`, cache `0 B`
    - `denoising.before-start`: resident `29.60 GiB`, active `29.34 GiB`, cache `5.71 KiB`
    - `decode.after-eval`: resident `424.09 MiB`, active `192.86 MiB`, cache `38.95 GiB`, MLX peak `39.03 GiB`
    - `/usr/bin/time -l` maximum resident set size: `42,830,249,984` bytes
    - `/usr/bin/time -l` peak memory footprint: `86,684,278,016` bytes
  - Phase 1 fixed-seed quality probe:
    - output SHA-256: `2afd1fa9ba4398ad2b8b53510f44d602d5d7d5cc2631cee99d35c6d0752f8f70`
    - phase 1 vs phase 0 MAE: `0.0000`
    - phase 1 vs phase 0 max abs pixel delta: `0`
    - phase 1 vs phase 0 PSNR: `inf`
  - Assessment:
    - the retained cache at `denoising.before-start` collapsed from about `28 GiB` to effectively zero
    - peak memory footprint improved by `24.11 GiB`, while maximum RSS stayed effectively flat
    - the fixed-seed control output remained bit-identical to phase 0
- Phase 2: completed on March 7, 2026.
  - Scope landed:
    - add `AutoencoderEncoderOnly` beside the existing decoder-only VAE path
    - load VAE encoder weights only when control or inpaint inputs must be encoded
    - unload the encoder immediately after the typed control-context tensor is materialized
    - defer decoder-only loading until final decode, then unload it after `decode.after-eval`
    - add `VAEComponentTests` to confirm encoder-only and decoder-only parity with the full autoencoder
  - Phase 2 high-resolution memory probe:
    - `prompt-embeddings.after-clear-cache`: resident `35.58 GiB`, active `29.18 GiB`, cache `0 B`
    - `control-context.after-baseline-reduction`: resident `273.34 MiB`, active `67.87 MiB`, cache `0 B`
    - `control-context.after-eval`: resident `365.64 MiB`, active `78.22 MiB`, cache `28.07 GiB`
    - `control-context.after-clear-cache`: resident `325.14 MiB`, active `71.36 MiB`, cache `0 B`
    - `denoising.before-start`: resident `29.50 GiB`, active `29.19 GiB`, cache `65.30 MiB`
    - `decode.after-eval`: resident `420.88 MiB`, active `127.48 MiB`, cache `39.00 GiB`, MLX peak `38.88 GiB`
    - `/usr/bin/time -l` maximum resident set size: `42,659,020,800` bytes
    - `/usr/bin/time -l` peak memory footprint: `86,580,223,280` bytes
  - Phase 2 fixed-seed quality probe:
    - output SHA-256: `2afd1fa9ba4398ad2b8b53510f44d602d5d7d5cc2631cee99d35c6d0752f8f70`
    - phase 2 vs phase 0 MAE: `0.0000`
    - phase 2 vs phase 0 max abs pixel delta: `0`
    - phase 2 vs phase 0 PSNR: `inf`
    - phase 2 vs phase 1 MAE: `0.0000`
    - phase 2 vs phase 1 max abs pixel delta: `0`
    - phase 2 vs phase 1 PSNR: `inf`
  - Assessment:
    - the control build now starts from a materially lower resident baseline than phase 1 because the full VAE no longer stays live between prompt encoding and control-context construction
    - the post-build reload boundary stayed clean, while maximum RSS and peak memory footprint both improved modestly again
    - the fixed-seed control output remained bit-identical to phase 0 and phase 1
- Phase 3: completed on March 7, 2026.
  - Scope landed:
    - replace per-block stacked hint transport with `ZImageControlHintState`
    - keep accumulated hints and current control state separate inside the control transformer path
    - preserve the legacy stacked representation only in the compatibility wrapper used by targeted tests
    - add `ControlTransformerBlockTests` to prove the state-based path matches the legacy stacked path
  - Phase 3 high-resolution memory probe:
    - `prompt-embeddings.after-clear-cache`: resident `33.81 GiB`, active `29.18 GiB`, cache `0 B`
    - `control-context.after-baseline-reduction`: resident `1012.41 MiB`, active `67.87 MiB`, cache `0 B`
    - `control-context.after-eval`: resident `364.31 MiB`, active `71.36 MiB`, cache `28.08 GiB`
    - `control-context.after-clear-cache`: resident `323.81 MiB`, active `71.36 MiB`, cache `0 B`
    - `denoising.before-start`: resident `29.50 GiB`, active `29.19 GiB`, cache `65.30 MiB`
    - `decode.after-eval`: resident `419.42 MiB`, active `127.48 MiB`, cache `39.00 GiB`, MLX peak `37.02 GiB`
    - `/usr/bin/time -l` maximum resident set size: `42,656,612,352` bytes
    - `/usr/bin/time -l` peak memory footprint: `59,328,863,512` bytes
  - Phase 3 fixed-seed quality probe:
    - output SHA-256: `2afd1fa9ba4398ad2b8b53510f44d602d5d7d5cc2631cee99d35c6d0752f8f70`
    - phase 3 vs phase 0 MAE: `0.0000`
    - phase 3 vs phase 0 max abs pixel delta: `0`
    - phase 3 vs phase 0 PSNR: `inf`
    - phase 3 vs phase 2 MAE: `0.0000`
    - phase 3 vs phase 2 max abs pixel delta: `0`
    - phase 3 vs phase 2 PSNR: `inf`
  - Assessment:
    - the denoising-memory target for phase 3 was met: peak memory footprint dropped by `27,251,359,768` bytes (`25.38 GiB`) versus phase 2 while maximum RSS stayed effectively flat
    - the `control-context.after-baseline-reduction` resident marker rose to about `1.0 GiB`, but active bytes and cache bytes stayed low and the surrounding phases remained aligned with phase 2; this looks like resident-accounting variation rather than renewed active pressure
    - the fixed-seed control output remained bit-identical to every earlier phase
