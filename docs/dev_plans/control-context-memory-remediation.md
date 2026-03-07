# Control-Context Memory Remediation Plan

This plan translates the control-path memory investigation into a repo-aligned implementation sequence for the Swift codebase.

Scope:

- `ZImageCLI control`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- targeted unit-test coverage under `Tests/ZImageTests/`
- manual validation against the local Diffusers reference when needed

## Current Repo State

The plan below is validated against the current code, not just the older debug notes.

Confirmed baseline:

- `buildControlContext(...)` still funnels large control/inpaint images through `vae.encode(...)`.
- Control-image and inpaint-image tensors are already created in `vae.dtype`; the old fp32-only issue is already fixed.
- `ZImageControlPipeline` does not retain a long-lived text encoder instance. Phase 2 should therefore focus on prompt-encoding scope, MLX cache state, and transformer/controlnet residency, not on “unloading a cached text encoder”.
- The control pipeline already has a reload path for `transformer == nil`, and it already knows how to reload ControlNet and reapply LoRA after that reload.
- `AutoencoderKL` already has distinct encoder and decoder submodules internally. Any later lifecycle split should be about load/unload ownership and weight application, not a wholesale VAE architecture rewrite.
- Prompt embeddings are intentionally cached across requests. Phase 2 should not throw that cache away unless a later measurement proves it is part of the problem.

## Problem Statement

At `1536x2304`, earlier control-mode runs showed a steep resident-memory increase during:

- `Loading text encoder...` (`~20 GB -> ~30 GB`)
- `Building control context...` (`~30 GB -> ~100 GB`)

The final control-context tensor is small (`[1, 33, 1, 288, 192]`), so the surge occurs while building it, not while storing it.

## Validated Findings

### 1. The hot path is `vae.encode(...)` during control-context construction

Relevant code paths:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - `buildControlContext(...)`
  - `encodeImageToLatents(...)`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
  - `VAEEncoder`
  - `VAEMidBlock`
  - `VAESelfAttention`

### 2. The VAE mid-block attention is still a prime transient-memory suspect

At `1536x2304`, the control-image latent grid is:

- `2304 / 8 = 288`
- `1536 / 8 = 192`
- latent tokens: `288 * 192 = 55,296`

The VAE encoder mid-block attention operates over that token count. That is large enough for SDPA workspace growth to be a plausible dominant peak contributor.

### 3. The final control context is not the peak driver

Logged shape:

- `[1, 33, 1, 288, 192]`
- `1,824,768` elements total

Approximate storage:

- bf16: `~3.5 MiB`
- fp32: `~7 MiB`

### 4. The remaining suspects are now narrower

Given the existing `vae.dtype` fix, the main remaining suspects are:

- VAE mid-block attention workspace
- VAE convolution/downsample activation and workspace growth
- MLX cache / lazy-evaluation overlap
- unnecessary module residency before entering the control-context build

### 5. Diffusers parity is a reference point, not an explanation

The local Diffusers control pipeline also VAE-encodes the control image, so the Swift port is following the same high-level algorithm.

Do not assume any Swift-versus-Diffusers gap is explained solely by “PyTorch flash attention”. Treat the MPS comparison as evidence that backend/runtime details differ enough to justify Swift-specific memory mitigation.

## Goals

Primary goals:

1. Reduce peak resident memory during control-context construction.
2. Keep the control path functional at `1536x2304` on high-end Apple Silicon without pathological memory spikes.
3. Preserve output parity as much as practical relative to the current Swift path and the Diffusers reference.

Non-goals:

- full VAE architecture changes
- custom Metal kernels in the first pass
- broad API churn in both pipelines when the issue is isolated to the control path

## Verification Strategy

Each landed phase should have two kinds of verification:

1. Fast repo verification:
   - `xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests`
   - `xcodebuild build -scheme ZImageCLI -destination 'platform=macOS' -derivedDataPath .build/xcode` when CLI flags/help text change
2. Manual control-path validation with cached local weights:
   - use `ZImageCLI control`
   - keep denoising short when the phase only targets control-context construction
   - preserve a `1536x2304` reference run for memory comparisons once instrumentation exists

Reference manual command shape:

```bash
.build/xcode/Build/Products/Debug/ZImageCLI control \
  --prompt "memory validation" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --width 1536 \
  --height 2304 \
  --steps 1 \
  --output /tmp/zimage-control-memory-check.png
```

## Patch Strategy

The work is split into small phases that can land independently. Phases 4 and 5 are intentionally conditional so the repo does not accumulate speculative complexity if Phases 0 through 3 are already sufficient.

---

## Phase 0: Reproducible Memory Instrumentation

### Objective

Make the control-path peak measurable by phase so later changes can be compared cleanly.

### Files

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImageCLI/main.swift`
- `Sources/ZImage/Util/` helper if needed
- `docs/CLI.md`

### Changes

Add opt-in phase-level memory logging around:

- after prompt embeddings are fully evaluated
- immediately before control-context construction starts
- immediately before each large `vae.encode(...)` call used for control/inpaint encoding
- immediately after the encoded latents or combined control context are evaluated
- immediately after explicit `Memory.clearCache()` calls that are part of the control-path workflow
- immediately before denoising starts
- immediately after final decode

Log at least:

- process resident memory (best-effort via Mach task info)
- `Memory.peakMemory`
- `Memory.activeMemory`
- `Memory.cacheMemory`

Implementation guidance:

- prefer a small reusable helper instead of repeating Mach-task code inline
- make instrumentation opt-in so normal CLI runs do not become noisy
- phase markers should be stable strings suitable for diffing across runs

Preferred surface:

- request-level runtime option in `ZImageControlGenerationRequest`
- CLI flag that turns it on for `ZImageCLI control`

### Acceptance criteria

- A single control run emits stable memory markers for the major phases.
- The markers include both resident memory and MLX memory counters.
- The code path is cheap enough to keep in the tree after this phase.

---

## Phase 1: Diagnostic VAE Mid-Block Attention Switch

### Objective

Quickly determine how much of the peak is attributable to VAE self-attention versus the surrounding convolution path.

### Files

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImageCLI/main.swift`
- `docs/CLI.md`

### Changes

Add a diagnostic path that disables VAE mid-block attention for control-image encoding only.

Implementation guidance:

- keep the default behavior unchanged
- scope the diagnostic to control-context construction only; do not change the general text-to-image VAE decode path
- prefer wiring through the same request/runtime option surface introduced in Phase 0

Expected user-visible shape:

- `ZImageCLI control --debug-disable-control-vae-attention`

### Validation

Run the `1536x2304` control reference twice:

- baseline
- with mid-block attention disabled

### Acceptance criteria

- The diagnostic path is clearly isolated and opt-in.
- Instrumentation makes the before/after delta obvious.
- The flag is documented as diagnostic-only, not as a supported quality mode.

---

## Phase 2: Lower Baseline Residency Before Control-Context Build

### Objective

Reduce avoidable resident memory before entering the most expensive VAE-encode phase.

### Files

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

### Changes

After prompt embeddings are evaluated:

1. ensure prompt-encoding temporaries fall out of scope
2. reuse the existing unload path to release transformer, ControlNet, and active LoRA state before `buildControlContext(...)`
3. explicitly clear MLX cache before large control-image VAE encode work
4. reuse the existing reload path after control-context construction, including ControlNet reload and LoRA reapplication when required

Guardrails:

- keep the tokenizer, VAE, cached prompt embeddings, and model metadata intact
- do not introduce a second transformer-loading architecture
- preserve current external CLI semantics

### Acceptance criteria

- Pre-control resident memory is lower than the Phase 0 baseline.
- Control generation still succeeds without new user flags.
- The unload/reload logic remains single-path and readable.

---

## Phase 3: Query-Chunked VAE Attention

### Objective

Cap the largest transient attention workspace in `VAESelfAttention` without changing weights.

### Files

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Tests/ZImageTests/` additions for attention equivalence or shape regression

### Changes

Replace the full-query SDPA call with a chunked-query path:

1. reshape Q/K/V exactly as today
2. split the query sequence dimension into chunks
3. run `MLXFast.scaledDotProductAttention(...)` per chunk against full K/V
4. concatenate outputs along the query dimension
5. preserve the existing projection and residual layout

Configuration guidance:

- keep the chunk size internal by default
- expose a narrow override only if it materially helps validation
- default experimental values can start around `512`, `1024`, or `2048`

### Validation

Compare chunked versus unchunked attention for:

- output shape
- numerical drift on a smaller deterministic tensor
- manual memory/run-time behavior on the `1536x2304` reference run

### Acceptance criteria

- Control-context peak memory drops materially relative to the earlier baseline.
- Output drift stays within an acceptable tolerance for the control path.
- Any runtime regression is documented.

### Gate After Phase 3

Only continue to Phase 4 if real measurements still show a pathological peak or instability at the `1536x2304` reference size.

---

## Phase 4: Conditional Tiled VAE Encode For Large Control Images

### Objective

Reduce VAE-encode activation/workspace growth for large control images if Phases 2 and 3 are not sufficient.

### Files

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Tests/ZImageTests/`
- `docs/ARCHITECTURE.md`
- `docs/CLI.md` only if a user-visible override survives

### Changes

Add an encode path that splits large images into overlapping tiles before VAE encode and stitches the latent tiles.

Recommended policy:

- automatic use above a conservative image-size threshold
- internal override for development and validation

Important details:

- overlap must be sufficient to avoid visible seams
- stitching should happen in latent space
- the simple non-tiled path should remain for smaller images

### Acceptance criteria

- Large-image control encode completes with materially lower peak memory than Phase 3 alone.
- Tile seam artifacts are absent or acceptably small.
- The implementation is isolated to the large-image control encode path.

### Gate After Phase 4

Only continue to Phase 5 if peak steady-state residency outside the encode hotspot is still a practical blocker.

---

## Phase 5: Conditional VAE Lifecycle Split

### Objective

Avoid carrying full VAE residency when only the encoder or decoder half is needed, if the measured steady-state footprint still justifies the extra complexity.

### Files

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- related weight-loading code only if strictly necessary
- `docs/ARCHITECTURE.md`

### Changes

Refactor ownership so the control pipeline can:

- keep encoder capability for control-context construction
- release encoder-only state after that phase if beneficial
- instantiate or retain decoder capability later for final decode

Constraints:

- prefer a narrow ownership/load-path refactor over a broad model rewrite
- do not duplicate weight-application logic across multiple bespoke VAE loaders
- keep the lifecycle understandable for both the control and text-to-image pipelines

### Acceptance criteria

- Steady-state resident memory is lower outside the encode hotspot.
- Final decode behavior remains correct.
- The resulting load/unload lifecycle is still easy to reason about.

---

## Phase 6: Final Runtime Policy And Documentation

### Objective

Turn the successful experiments into a stable runtime policy and document the outcome.

### Files

- `docs/CLI.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/dev_plans/ROADMAP.md`

### Changes

Document:

- how to enable memory instrumentation
- whether any diagnostic flag remains available after the investigation
- whether chunked attention is now default
- whether tiled encode is now default or conditional
- measured runtime versus memory tradeoffs for large control images
- which later phases were intentionally skipped because earlier phases were sufficient

### Acceptance criteria

- The docs match the actual runtime policy.
- A contributor can reproduce the memory investigation without needing prior chat context.

## Execution Order

1. Phase 0: instrumentation
2. Phase 1: diagnostic attention toggle
3. Phase 2: baseline residency reduction
4. Phase 3: query-chunked VAE attention
5. Phase 4: tiled encode only if Phase 3 metrics still justify it
6. Phase 5: VAE lifecycle split only if Phase 4 metrics still justify it
7. Phase 6: final docs and cleanup

## File-Level Checklist

Likely code changes:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - instrumentation hooks
  - pre-encode unload and post-encode reload around control-context construction
  - optional runtime policy for control-path diagnostics
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
  - control-only attention-disable diagnostic hook
  - query-chunked `VAESelfAttention`
  - optional tiled encode support
  - optional later ownership/lifecycle changes
- `Sources/ZImageCLI/main.swift`
  - control-only debug and instrumentation flags

Likely tests:

- helper tests for any memory-log formatting that can be tested deterministically
- shape and equivalence tests for chunked attention
- tiled-encode dimensional/stitching regression tests if Phase 4 lands

Likely docs updates:

- `docs/CLI.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/dev_plans/ROADMAP.md`

## Risks And Tradeoffs

Query-chunked attention:

- lower peak memory
- higher runtime
- possible numerical drift depending on chunk size and kernel behavior

Tiled encode:

- strongest general-purpose memory reduction
- more implementation complexity
- risk of seam artifacts if overlap/blending is wrong

VAE lifecycle split:

- lower steady-state residency
- highest refactor risk because it reaches into weight loading and ownership

## Success Criteria

Treat the remediation as successful when the `1536x2304` control reference case satisfies all of the following:

1. Control-context construction peak memory is materially below the earlier `~100 GB` observation.
2. The run completes without pathological swap or memory-compression behavior.
3. Output quality remains acceptable for the existing control examples.
4. The retained runtime policy is documented in `docs/`.

## Relationship To Existing Notes

- `docs/debug_notes/control-memory-phased-plan.md` records the earlier diagnosis.
- This plan supersedes that note for implementation because the repo already landed some of the earlier fixes and now has a different baseline.
