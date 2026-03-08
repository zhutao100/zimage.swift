# Z-Image Support Plan

Validated on 2026-03-08 against the current `zimage.swift` codebase plus the current upstream `Tongyi-MAI/Z-Image` Hugging Face checkpoint and Diffusers `pipeline_z_image.py`.

## Summary

This is a **finish first-class Base support** project, not a new-model port.

The Swift runtime already has the right major architecture for `Tongyi-MAI/Z-Image`:

- `ZImagePipeline`
- `ZImageControlPipeline`
- `ZImageTransformer2DModel`
- `Qwen3` text encoder path
- `AutoencoderKL`
- `FlowMatchEulerDiscreteScheduler`-style scheduler config loading

The remaining work is concentrated in four areas:

1. stale Base fixtures and upstream-reference docs
2. Turbo-biased CLI defaults
3. missing Base CFG parity controls
4. missing Base-targeted smoke coverage

## Validated Upstream Facts

Current upstream Base checkpoint facts that this repo should align with:

- `transformer/config.json` currently exposes `dim = 3840`, `n_layers = 30`, `n_heads = 30`, `cap_feat_dim = 2560`
- `scheduler/scheduler_config.json` currently exposes `shift = 6.0` and `use_dynamic_shifting = false`
- Base transformer weights are currently sharded across **2** files
- Base text encoder weights are currently sharded across **3** files
- The current model card recommends roughly `28-50` steps and `guidance_scale` in the `3-5` range
- The current model-card example uses `cfg_normalization=False`
- Current Diffusers Base pipeline behavior includes:
  - `cfg_normalization`
  - `cfg_truncation`
  - `scheduler.sigma_min = 0.0`
  - CFG math `pos + scale * (pos - neg)` with optional norm renormalization

## Current Repo State

What is already in place:

- `Sources/ZImage/Support/ZImageModelRegistry.swift` already recognizes both Turbo and Base and already has distinct presets
- `Sources/ZImage/Weights/ModelConfigs.swift` loads transformer, scheduler, text-encoder, and VAE behavior from snapshot configs
- `Sources/ZImage/Pipeline/FlowMatchScheduler.swift` already appends a terminal `0.0` sigma
- `Tests/ZImageTests/Scheduler/FlowMatchSchedulerTests.swift` already covers monotonicity and final-zero-sigma behavior
- Repo docs already describe Base as supported, but with caveats

What is currently wrong or incomplete:

- Base snapshot fixtures in `Tests/ZImageTests/Fixtures/Snapshots/ZImageBase/` are stale
- `Sources/ZImageCLI/main.swift` still seeds text-to-image and control defaults from Turbo-oriented `ZImageModelMetadata`
- Both pipelines implement only basic CFG and do not expose Base parity controls
- Default integration coverage is Turbo-only
- `docs/z-image.md` is partially stale relative to the current upstream Base model card and layout

## Corrections To Earlier Assumptions

The earlier draft direction was mostly right, but these points need to be tightened:

- **Do not treat scheduler rewrite as a default requirement.**
  The current scheduler already appends the terminal zero sigma and has reasonable unit coverage. Base support should only change scheduler code if parity validation proves a concrete mismatch.
- **Do not frame Base support as architecturally missing.**
  The architecture is already shared across Turbo and Base. The problem is fidelity of defaults, parity controls, fixtures, and validation.
- **Treat fixture drift as a correctness bug, not just a documentation issue.**
  Current Base fixture values are validating an outdated approximation of the upstream checkpoint.
- **Treat the CLI as the highest-priority defaulting surface.**
  The library request initializers still have Turbo-oriented scalar defaults by construction, but the immediate user-facing regression is the CLI because it already has model selection and known-model presets available.

## Scope

In scope:

- Base fixture and doc rebaseline
- model-aware CLI defaults
- Base CFG parity knobs in both pipelines
- Base smoke coverage and doc cleanup

Out of scope for this project:

- separate Base-specific model implementations
- `Z-Image-Omni-Base`, `Z-Image-Edit`, or unrelated model families
- speculative scheduler rewrites without parity evidence
- major CLI parser replacement
- broader ControlNet feature expansion unrelated to Base support

## Execution Principles

- Keep Base and Control paths behaviorally aligned when changing denoising logic
- Prefer a shared helper in `PipelineUtilities.swift` over duplicating CFG math
- Update docs in the same phase as user-visible behavior changes
- Verify each phase independently before committing
- Record only measurable runtime-impact results; do not add transient debugging scaffolding

## Recommended Implementation Order

### Phase 1: Rebaseline Base Fixtures And Upstream Reference Docs

Goal:
Make the repo's Base assumptions match the current upstream checkpoint before changing runtime behavior.

Required work:

- update Base snapshot fixture configs to current upstream values
- update Base shard-layout fixtures to current upstream shard counts
- tighten snapshot and model-path tests around the real Base layout
- sync `docs/z-image.md` with the current model card and repo layout

Why first:

- it removes false confidence from outdated tests
- later CLI and pipeline work should not build on stale fixture assumptions

### Phase 2: Make CLI Defaults Model-Aware

Goal:
`ZImageCLI --model Tongyi-MAI/Z-Image` and `ZImageCLI control --model Tongyi-MAI/Z-Image` should pick Base-appropriate defaults unless the user explicitly overrides them.

Required work:

- track whether width, height, steps, guidance, and other preset-driven fields were explicitly provided
- apply `ZImagePreset.defaults(for:)` only to unset fields
- reuse the same policy in both top-level and `control` parsing paths
- update help text and user-facing docs to describe the new behavior accurately

Important constraint:

- Turbo behavior must remain unchanged when `--model` is omitted

### Phase 3: Add Base CFG Parity Controls

Goal:
Expose and implement the most meaningful current Base inference knobs from Diffusers.

Required work:

- add request-surface fields for CFG truncation and CFG normalization
- wire equivalent CLI flags
- centralize the CFG combine-and-renormalize logic in shared pipeline utility code
- apply the same behavior in `ZImagePipeline` and `ZImageControlPipeline`
- add unit coverage for the helper and request/CLI plumbing

Important constraint:

- keep guidance disabled when the effective guidance scale for a step is zero

### Phase 4: Add Base Validation Coverage And Retire Stale Messaging

Goal:
Prove that Base loading and denoising work in at least one opt-in real-model path, then clean up docs and roadmap wording.

Required work:

- add an env-gated Base integration smoke test
- keep it out of default CI unless weights are present
- update `README.md`, `docs/CLI.md`, `docs/MODELS_AND_WEIGHTS.md`, and `docs/dev_plans/ROADMAP.md`
- make sure completed planning docs no longer describe Base support as mostly missing

## Verification Gates

Each phase must end with targeted verification before commit.

Phase 1:

- `xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/Config/SnapshotModelConfigsTests -only-testing:ZImageTests/Weights/ModelPathsResolutionTests`

Phase 2:

- focused `ZImageTests` coverage for presets and any CLI parsing helpers added
- build the CLI target

Phase 3:

- focused `ZImageTests` coverage for new CFG helper behavior and request defaults/plumbing
- build the package or CLI target that exercises the changed surfaces

Phase 4:

- run the opt-in Base smoke test only when the required snapshot is available
- rerun the targeted fast suite covering all touched areas

## Commit Strategy

Use one Conventional Commit per completed phase. Recommended subjects:

- `docs(z-image-support): harden base support plan`
- `test(z-image-support): rebaseline base fixtures`
- `feat(cli): apply model-aware z-image presets`
- `feat(pipeline): add base cfg parity controls`
- `test(z-image-support): add base smoke coverage`

## Exit Criteria

This project is complete when all of the following are true:

- Base fixtures and tests reflect the current upstream checkpoint
- Base CLI runs no longer silently inherit Turbo sampling defaults
- both pipelines expose and implement Base CFG parity controls
- there is an opt-in validation path that exercises real Base loading
- docs no longer describe current Base support in outdated or contradictory terms
