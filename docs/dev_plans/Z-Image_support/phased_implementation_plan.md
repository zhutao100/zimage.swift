# Phased Implementation Plan: Z-Image Base Support

This file converts the validated support plan into executable phases with file targets, verification, and commit boundaries.

## Phase 0: Plan Hardening

Outcome:

- current `plan.md` reflects validated upstream and repo facts
- implementation phases are explicit and independently verifiable

Files:

- `docs/dev_plans/Z-Image_support/plan.md`
- `docs/dev_plans/Z-Image_support/phased_implementation_plan.md`

Verification:

- manual validation against current source, tests, Hugging Face Base checkpoint, and Diffusers Base pipeline

Commit:

- `docs(z-image-support): harden base support plan`

## Phase 1: Rebaseline Base Fixtures

Outcome:

- Base snapshot fixtures match current upstream Base config and shard layout
- tests stop encoding stale Base assumptions
- upstream reference doc is corrected

Primary files:

- `Tests/ZImageTests/Fixtures/Snapshots/ZImageBase/transformer/config.json`
- `Tests/ZImageTests/Fixtures/Snapshots/ZImageBase/scheduler/scheduler_config.json`
- `Tests/ZImageTests/Fixtures/Snapshots/ZImageBase/transformer/diffusion_pytorch_model.safetensors.index.json`
- `Tests/ZImageTests/Fixtures/Snapshots/ZImageBase/text_encoder/model.safetensors.index.json`
- `Tests/ZImageTests/Config/SnapshotModelConfigsTests.swift`
- `Tests/ZImageTests/Weights/ModelPathsResolutionTests.swift`
- `docs/z-image.md`

Verification:

- `xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/Config/SnapshotModelConfigsTests -only-testing:ZImageTests/Weights/ModelPathsResolutionTests`

Commit:

- `test(z-image-support): rebaseline base fixtures`

## Phase 2: Apply Model-Aware CLI Defaults

Outcome:

- CLI applies Base vs Turbo presets only to fields the user did not set
- top-level and `control` subcommand follow the same defaulting rules
- help text and docs describe the new behavior

Primary files:

- `Sources/ZImageCLI/main.swift`
- `Tests/ZImageTests/Support/ZImageModelRegistryTests.swift`
- `README.md`
- `docs/CLI.md`
- `docs/MODELS_AND_WEIGHTS.md`
- `docs/ARCHITECTURE.md`

Verification:

- `xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/Support/ZImageModelRegistryTests`
- `xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -skipPackagePluginValidation ENABLE_PLUGIN_PREPAREMLSHADERS=YES CLANG_COVERAGE_MAPPING=NO`

Commit:

- `feat(cli): apply model-aware z-image presets`

## Phase 3: Add Base CFG Parity Controls

Outcome:

- request types, CLI flags, and both pipelines support CFG truncation and normalization
- CFG math lives in shared utility code instead of duplicated pipeline blocks

Primary files:

- `Sources/ZImage/Pipeline/PipelineUtilities.swift`
- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImageCLI/main.swift`
- `Tests/ZImageTests/Support/`
- `README.md`
- `docs/CLI.md`

Verification:

- `xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/Support`
- `xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -skipPackagePluginValidation ENABLE_PLUGIN_PREPAREMLSHADERS=YES CLANG_COVERAGE_MAPPING=NO`

Commit:

- `feat(pipeline): add base cfg parity controls`

## Phase 4: Add Base Validation Coverage And Final Cleanup

Outcome:

- repo has an opt-in Base integration smoke path
- roadmap and docs no longer present Base support as mostly unfinished
- final verification covers all changed areas

Primary files:

- `Tests/ZImageIntegrationTests/PipelineIntegrationTests.swift`
- `README.md`
- `docs/CLI.md`
- `docs/MODELS_AND_WEIGHTS.md`
- `docs/dev_plans/ROADMAP.md`

Verification:

- `xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests`
- opt-in Base integration run when weights are available

Commit:

- `test(z-image-support): add base smoke coverage`

## Notes

- If Phase 3 uncovers a scheduler mismatch that cannot be explained by CFG behavior, add a narrowly scoped follow-up phase instead of expanding this project opportunistically.
- If Base weights are not locally available during Phase 4, keep the smoke test env-gated and document the exact skipped verification.
