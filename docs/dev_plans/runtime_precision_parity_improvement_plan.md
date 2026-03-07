# Runtime Precision Parity Improvement Plan

This plan turns the validated findings in `docs/context/zimage_runtime_precision_parity_report.md` into a measured implementation sequence.

Execution status: in progress on March 7, 2026.

Scope:

- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Model/Transformer/ZImageTimestepEmbedder.swift`
- `Sources/ZImage/Model/TextEncoder/TextEncoder.swift`
- targeted unit coverage under `Tests/ZImageTests/`
- manual fixed-seed parity probes against the local Diffusers checkout

Out of scope for this execution pass:

- RoPE parity changes in the runtime path

RoPE remains a documented mismatch, but it is intentionally deferred until an intermediate-tensor probe is in place. The first three phases below are lower risk, easier to verify end to end, and align with the strongest confirmed mismatches.

## Baseline Protocol

Every phase must be evaluated with the same fixed-seed probes before the phase is wrapped and committed.

Repo verification:

- `xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests`
- `xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode`

Base-image parity probe:

- model: `Tongyi-MAI/Z-Image-Turbo`
- prompt: `a brass compass on a wooden desk, dramatic sunlight, product photo`
- width/height: `512x512`
- steps: `4`
- guidance: `0`
- seed: `1234`

Control parity probe:

- base model: `Tongyi-MAI/Z-Image-Turbo`
- control weights: `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`
- control file: `Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors`
- control image: `images/canny.jpg`
- prompt: `a stone archway covered in moss, cinematic lighting`
- width/height: `512x512`
- steps: `4`
- guidance: `0`
- control scale: `0.75`
- seed: `1234`

Required metrics per phase:

1. Peak RSS for the Swift CLI run via `/usr/bin/time -l`
2. Output-drift metric between Swift and Diffusers reference images
3. Output-drift metric between the current phase and the immediately previous Swift phase

Preferred image metrics:

- mean absolute pixel error
- max absolute pixel error
- PSNR

Acceptance bar:

- no broken images or NaN-like failures
- no material regression in peak RSS on the fixed probe without an explicit parity reason
- parity metric should improve or remain effectively flat for the targeted phase

## Phase 1: Denoiser Ingress Dtype Normalization

Status: completed on March 7, 2026

Objective:

- keep scheduler latents in fp32
- explicitly cast the model-facing latent tensors to the runtime dtype of the loaded transformer and controlnet right before forward

Files:

- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Tests/ZImageTests/`

Implementation notes:

- do not globally narrow scheduler state
- normalize only the tensors handed to `transformer.forward(...)` and `controlnet.forward(...)`
- prefer a shared helper for resolving the first floating parameter dtype from a module to avoid duplicated pipeline logic

Acceptance criteria:

- base and control denoising paths use explicit model-ingress dtype normalization
- unit tests cover the dtype helper and the cast boundary behavior where feasible
- fixed-seed base/control probes are rerun and recorded before commit

## Phase 2: Timestep-MLP Ingress Dtype Normalization

Status: completed on March 7, 2026

Objective:

- align `ZImageTimestepEmbedder` with Diffusers by explicitly casting the sinusoidal timestep features to the first MLP layer runtime dtype before `mlp.0`

Files:

- `Sources/ZImage/Model/Transformer/ZImageTimestepEmbedder.swift`
- `Tests/ZImageTests/`

Implementation notes:

- keep the sinusoidal feature construction numerically stable
- only normalize the tensor at the MLP ingress boundary
- avoid introducing a broader model-wide dtype policy here

Acceptance criteria:

- timestep embedding path matches the validated Diffusers behavior more closely at the MLP boundary
- focused unit tests cover the cast target and output dtype expectations
- fixed-seed base/control probes are rerun and recorded before commit

## Phase 3: Bool Prompt-Attention Masking

Status: completed on March 7, 2026

Objective:

- replace the current additive hidden-dtype prompt mask in `TextEncoder.swift` with a boolean mask that remains compatible with causal masking in MLX fast SDPA

Files:

- `Sources/ZImage/Model/TextEncoder/TextEncoder.swift`
- `Tests/ZImageTests/`

Implementation notes:

- preserve the existing prompt-encoding semantics
- construct a combined keep/disallow boolean mask that matches the current causal-plus-padding behavior
- do not change generation-time KV-cache masking in this phase

Acceptance criteria:

- prompt encoding uses a boolean SDPA mask for the non-generation path
- unit tests cover the combined causal/padding mask behavior
- fixed-seed base/control probes are rerun and recorded before commit

## Deferred Follow-Up: RoPE Precision Parity

Status: deferred

Reason for deferral:

- the mismatch is real, but an end-to-end image delta alone is not a strong enough signal to distinguish RoPE precision effects from ordinary denoiser drift
- this should be handled with an intermediate-tensor probe around rotary application, then landed separately if the measured error warrants the added complexity

## Execution Log

This section is updated as phases land.

- Baseline: completed on March 7, 2026.
- Diffusers reference assets:
  - base probe: `diffusers_base.png`
  - control probe: `diffusers_control.png`
- Swift baseline assets:
  - base probe: `swift_base_phase0.png`
  - control probe: `swift_control_phase0.png`
- Baseline base metrics:
  - Swift peak RSS: `38.16 GiB`
  - Swift peak memory footprint: `38.98 GiB`
  - Swift vs Diffusers MAE: `28.6232`
  - Swift vs Diffusers max abs pixel delta: `255`
  - Swift vs Diffusers PSNR: `14.6503 dB`
- Baseline control metrics:
  - Swift peak RSS: `42.82 GiB`
  - Swift peak memory footprint: `51.76 GiB`
  - Swift vs Diffusers MAE: `33.3260`
  - Swift vs Diffusers max abs pixel delta: `238`
  - Swift vs Diffusers PSNR: `15.2456 dB`
- Phase 1: completed on March 7, 2026.
  - Scope landed:
    - explicit runtime-dtype casts at transformer ingress in the base pipeline
    - explicit runtime-dtype casts at controlnet and transformer ingress in the control pipeline
    - shared module-runtime-dtype helper coverage in `ZImageTests`
  - Phase 1 base metrics:
    - Swift peak RSS: `38.16 GiB` (`38161776640` bytes)
    - Swift peak memory footprint: `38.94 GiB` (`38943542040` bytes)
    - Swift vs Diffusers MAE: `28.7489`
    - Swift vs Diffusers max abs pixel delta: `255`
    - Swift vs Diffusers PSNR: `14.6354 dB`
    - Swift phase 1 vs phase 0 MAE: `2.2592`
    - Swift phase 1 vs phase 0 max abs pixel delta: `197`
    - Swift phase 1 vs phase 0 PSNR: `31.6204 dB`
  - Phase 1 control metrics:
    - Swift peak RSS: `42.83 GiB` (`42829496320` bytes)
    - Swift peak memory footprint: `51.40 GiB` (`51400154488` bytes)
    - Swift vs Diffusers MAE: `33.3900`
    - Swift vs Diffusers max abs pixel delta: `236`
    - Swift vs Diffusers PSNR: `15.2305 dB`
    - Swift phase 1 vs phase 0 MAE: `1.2675`
    - Swift phase 1 vs phase 0 max abs pixel delta: `46`
    - Swift phase 1 vs phase 0 PSNR: `42.2552 dB`
  - Assessment:
    - parity remained effectively flat on the fixed probes
    - peak memory footprint improved modestly in both probes, while peak RSS stayed effectively flat
- Phase 2: completed on March 7, 2026.
  - Scope landed:
    - explicit runtime-dtype cast for timestep frequency features at the `mlp.0` ingress in `ZImageTimestepEmbedder`
    - focused tests covering both the cast helper and the end-to-end output dtype under forced `bfloat16` weights
  - Phase 2 base metrics:
    - Swift peak RSS: `38.16 GiB` (`38164889600` bytes)
    - Swift peak memory footprint: `38.94 GiB` (`38935251640` bytes)
    - Swift vs Diffusers MAE: `28.6765`
    - Swift vs Diffusers max abs pixel delta: `254`
    - Swift vs Diffusers PSNR: `14.6461 dB`
    - Swift phase 2 vs phase 1 MAE: `1.0000`
    - Swift phase 2 vs phase 1 max abs pixel delta: `164`
    - Swift phase 2 vs phase 1 PSNR: `39.2637 dB`
  - Phase 2 control metrics:
    - Swift peak RSS: `42.82 GiB` (`42824695808` bytes)
    - Swift peak memory footprint: `51.39 GiB` (`51390979472` bytes)
    - Swift vs Diffusers MAE: `33.3201`
    - Swift vs Diffusers max abs pixel delta: `236`
    - Swift vs Diffusers PSNR: `15.2483 dB`
    - Swift phase 2 vs phase 1 MAE: `1.0363`
    - Swift phase 2 vs phase 1 max abs pixel delta: `43`
    - Swift phase 2 vs phase 1 PSNR: `43.6782 dB`
  - Assessment:
    - parity improved in both fixed probes relative to phase 1
    - peak memory footprint improved again, while peak RSS stayed effectively flat
- Phase 3: completed on March 7, 2026.
  - Scope landed:
    - prompt-encoding attention now uses a combined causal-plus-padding boolean keep mask in `TextEncoder.swift`
    - generation-time KV-cache masking remained unchanged
    - focused tests cover both the no-padding causal fallback and the combined boolean mask layout
  - Phase 3 base metrics:
    - Swift peak RSS: `38.16 GiB` (`38164676608` bytes)
    - Swift peak memory footprint: `39.08 GiB` (`39078529816` bytes)
    - Swift vs Diffusers MAE: `28.6765`
    - Swift vs Diffusers max abs pixel delta: `254`
    - Swift vs Diffusers PSNR: `14.6461 dB`
    - Swift phase 3 vs phase 2 MAE: `0.0000`
    - Swift phase 3 vs phase 2 max abs pixel delta: `0`
    - Swift phase 3 vs phase 2 PSNR: `inf`
  - Phase 3 control metrics:
    - Swift peak RSS: `42.82 GiB` (`42824761344` bytes)
    - Swift peak memory footprint: `51.39 GiB` (`51393944928` bytes)
    - Swift vs Diffusers MAE: `33.3201`
    - Swift vs Diffusers max abs pixel delta: `236`
    - Swift vs Diffusers PSNR: `15.2483 dB`
    - Swift phase 3 vs phase 2 MAE: `0.0000`
    - Swift phase 3 vs phase 2 max abs pixel delta: `0`
    - Swift phase 3 vs phase 2 PSNR: `inf`
  - Assessment:
    - prompt-path behavior remained bit-identical to phase 2 on both fixed probes
    - peak RSS stayed effectively flat, and the small peak-footprint movement fell within measurement noise for these runs
