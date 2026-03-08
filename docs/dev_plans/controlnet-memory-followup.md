# ControlNet Memory Follow-Up Plan

This plan turns the still-open findings in `docs/debug_notes/controlnet-memory-analysis.md` into a measured next sequence after the completed March 7, 2026 remediation work.

Execution status: phase 1 completed on March 8, 2026; phase 2 remains open.

## Goal

Reduce the remaining control-path peak memory pressure without changing control-image semantics or fixed-seed output quality.

Current measured comparison baseline for this follow-up pass:

- source: March 8, 2026 rerun of commit `555fec6`
- `/usr/bin/time -l` maximum resident set size: `42,644,701,184` bytes
- `/usr/bin/time -l` peak memory footprint: `59,326,700,800` bytes
- fixed-seed output SHA-256: `b5f1585314323c7e12f3a4871644346ac9d5f2470cfbf74c11935e9f2c558b98`

Historical note:

- the March 7, 2026 remediation log recorded the saved quality artifact hash `2afd1fa9...`
- current reruns of commit `555fec6` on March 8, 2026 reproduce `b5f15853...`
- this follow-up plan uses the March 8 rerun as the acceptance baseline for any new phase

## Scope

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- targeted lifecycle helpers if they reduce duplicate load and unload logic
- optional control and inpaint VAE encode work under `Sources/ZImage/Model/VAE/*`
- targeted unit coverage under `Tests/ZImageTests/`
- supporting docs under `docs/`

## Non-Goals

- changing ControlNet conditioning math
- changing scheduler behavior
- broad refactors across the base text-to-image pipeline unless needed to keep loader conventions aligned

## Validation Protocol

Every phase must be evaluated before it is wrapped.

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
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --width 1536 \
  --height 2304 \
  --steps 1 \
  --guidance 0 \
  --seed 1234 \
  --log-control-memory \
  --no-progress \
  --output /tmp/zimage-control-followup/<phase>_memory.png
```

Fixed-seed quality probe:

```bash
/usr/bin/time -l ./.build/xcode/Build/Products/Release/ZImageCLI control \
  --prompt "a stone archway covered in moss, cinematic lighting" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --width 512 \
  --height 512 \
  --steps 4 \
  --guidance 0 \
  --seed 1234 \
  --no-progress \
  --output /tmp/zimage-control-followup/<phase>_quality.png
```

Required metrics per phase:

1. `prompt-embeddings.after-clear-cache`
2. `control-context.after-baseline-reduction`
3. `control-context.after-clear-cache`
4. `denoising.before-start`
5. `decode.after-eval`
6. `/usr/bin/time -l` maximum resident set size
7. `/usr/bin/time -l` peak memory footprint
8. output SHA-256 and image drift versus the phase 3 baseline and the immediately previous phase

Preferred image metrics:

- mean absolute pixel error
- max absolute pixel delta
- PSNR

Acceptance bar:

- no broken image, NaN-like failure, or shape mismatch
- no unexpected regression in the high-resolution memory probe
- phase 1 should reduce or eliminate avoidable loader churn
- later phases should only proceed if the previous phase leaves a meaningful memory target unmet
- output semantics should stay bit-identical or differ only by a trivial and explainable amount

## Phase 1: Defer Transformer And ControlNet Loading

Status: completed on March 8, 2026

Objective:

- stop loading the transformer and optional ControlNet before prompt encoding and control-context construction

Files:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Tests/ZImageTests/`
- `docs/`

Implementation notes:

- keep tokenizer and text encoder lifecycle unchanged unless the refactor proves a tighter policy is simpler
- resolve model paths and weight metadata early, but instantiate transformer and ControlNet modules only when denoising is about to start
- remove the current unload and reload churn if the modules can simply remain absent until needed
- keep externally visible CLI and pipeline behavior unchanged

Acceptance criteria:

- the control path no longer performs an unnecessary transformer and ControlNet load before prompt embedding work
- the measured pre-control baseline and/or denoising start boundary improves versus the phase 3 reference
- fixed-seed output remains effectively unchanged

Execution result:

- the final landed variant defers both transformer and ControlNet loading until denoising
- versus the March 8 rerun baseline:
  - `prompt-embeddings.after-clear-cache`: `35.18 GiB -> 6.26 GiB`
  - `/usr/bin/time -l` maximum resident set size: `42,644,701,184 -> 38,382,632,960`
  - `/usr/bin/time -l` peak memory footprint: `59,326,700,800 -> 59,324,947,664`
- the fixed-seed output stayed bit-identical to the March 8 rerun baseline:
  - phase 1 SHA-256: `b5f1585314323c7e12f3a4871644346ac9d5f2470cfbf74c11935e9f2c558b98`
  - phase 1 vs baseline MAE: `0.0000`
  - phase 1 vs baseline max absolute pixel delta: `0`
  - phase 1 vs baseline PSNR: `inf`
- the large prompt-stage and RSS drop justified landing the phase even though peak memory footprint stayed effectively flat

## Phase 2: Consolidate Lifecycle Boundaries And Telemetry

Status: proposed

Objective:

- make remaining memory jumps attributable and keep loader policy easy to reason about

Files:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- targeted shared helpers if needed
- `docs/`

Implementation notes:

- add or tighten telemetry around module load, unload, and first-use boundaries only where phase 1 still leaves attribution gaps
- consolidate duplicated loader sequencing into a narrow helper if phase 1 introduces parallel load paths
- do not widen the refactor into generic loader abstractions unless duplication is real and current

Acceptance criteria:

- logs clearly isolate the residual lifecycle transitions that still matter
- loader sequencing is simpler than the pre-phase code, not more abstract
- memory and quality stay at least as good as phase 1

## Phase 3: Optional Tiled Control And Inpaint VAE Encode

Status: proposed, gated on phase 1 and 2 results

Objective:

- reduce the remaining monolithic control-context build spike if lifecycle cleanup is not sufficient

Files:

- `Sources/ZImage/Model/VAE/*`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Tests/ZImageTests/`
- `docs/`

Implementation notes:

- keep the latent-space math aligned with the existing control and inpaint encode path
- prefer the smallest practical tiling or striping strategy that can be validated with fixed-seed comparisons
- treat this as a structural phase only if the deferred-loading work still leaves the high-resolution probe above the practical target

Acceptance criteria:

- the control-context build path shows a material peak-memory reduction beyond phase 1 and 2
- quality drift is zero or small enough to quantify and justify
- the implementation does not duplicate a second independent encode stack unnecessarily

## Execution Log

- Baseline rerun: completed on March 8, 2026 from commit `555fec6`.
  - `prompt-embeddings.after-clear-cache`: resident `35.18 GiB`, active `29.18 GiB`, cache `0 B`
  - `control-context.after-baseline-reduction`: resident `997.77 MiB`, active `67.87 MiB`, cache `0 B`
  - `control-context.after-clear-cache`: resident `308.45 MiB`, active `71.36 MiB`, cache `0 B`
  - `denoising.before-start`: resident `29.48 GiB`, active `29.19 GiB`, cache `65.30 MiB`
  - `decode.after-eval`: resident `408.55 MiB`, active `127.48 MiB`, cache `39.00 GiB`, MLX peak `37.02 GiB`
  - `/usr/bin/time -l` maximum resident set size: `42,644,701,184` bytes
  - `/usr/bin/time -l` peak memory footprint: `59,326,700,800` bytes
  - output SHA-256: `b5f1585314323c7e12f3a4871644346ac9d5f2470cfbf74c11935e9f2c558b98`
- Phase 1: completed on March 8, 2026.
  - Scope landed:
    - defer transformer loading until prompt embeddings and control-context construction are complete
    - defer ControlNet loading until the denoising boundary
    - keep tokenizer, text encoder, VAE encoder, and final decoder lifecycles unchanged
  - Phase 1 high-resolution memory probe:
    - `prompt-embeddings.after-clear-cache`: resident `6.26 GiB`, active `2.50 MiB`, cache `0 B`
    - `control-context.after-baseline-reduction`: resident `1006.81 MiB`, active `67.87 MiB`, cache `0 B`
    - `control-context.after-clear-cache`: resident `317.66 MiB`, active `71.36 MiB`, cache `0 B`
    - `denoising.before-start`: resident `29.50 GiB`, active `29.19 GiB`, cache `65.30 MiB`
    - `decode.after-eval`: resident `417.75 MiB`, active `127.48 MiB`, cache `39.00 GiB`, MLX peak `32.67 GiB`
    - `/usr/bin/time -l` maximum resident set size: `38,382,632,960` bytes
    - `/usr/bin/time -l` peak memory footprint: `59,324,947,664` bytes
  - Phase 1 fixed-seed quality probe:
    - output SHA-256: `b5f1585314323c7e12f3a4871644346ac9d5f2470cfbf74c11935e9f2c558b98`
    - phase 1 vs baseline MAE: `0.0000`
    - phase 1 vs baseline max absolute pixel delta: `0`
    - phase 1 vs baseline PSNR: `inf`
  - Assessment:
    - the prompt-stage baseline collapsed by roughly `28.92 GiB`
    - maximum RSS improved by about `4.26 GiB`
    - peak memory footprint stayed effectively flat, so the remaining issue moved cleanly to the denoising load boundary rather than disappearing
