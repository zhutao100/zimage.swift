# AGENTS.md (LLM / Agent Guide)

This guide is for agents working inside `zimage.swift`. Treat the code and tests as the final source of truth, use the docs to narrow the search space, and keep docs synchronized whenever user-visible behavior changes.

## Read First By Task

- Any task: `README.md`, then `docs/README.md`
- Docs-only work: `README.md`, `docs/README.md`, then the source-of-truth doc for the area you are editing
- Triage or bugfix: the relevant failing test file, then `docs/ARCHITECTURE.md`; if the issue is about model loading, also read `docs/MODELS_AND_WEIGHTS.md`
- CLI work: `docs/CLI.md`, then `Sources/ZImageCLICommon/`, `Sources/ZImageCLI/main.swift`, and `Sources/ZImageServe/main.swift`
- Pipeline or feature work: `docs/ARCHITECTURE.md`, then the relevant files under `Sources/ZImage/Pipeline/`, `Sources/ZImage/Model/`, and `Sources/ZImage/Weights/`
- Release or packaging work: `docs/DEVELOPMENT.md`, `.github/workflows/ci.yml`, `scripts/build.sh`
- Control-memory work: `docs/DEVELOPMENT.md`, then `docs/dev_plans/controlnet-memory-followup.md`, then `Sources/ZImage/Util/ControlMemoryTelemetry.swift`
- Precision or parity work: `docs/golden_checks.md`, then `docs/context/zimage_runtime_precision_parity_report.md`
- Upstream model layout questions: `docs/z-image-turbo.md` and `docs/z-image.md`

## Repo Map

- `Package.swift`
  - package graph, products, target list, and platform support
- `Sources/ZImageCLI/`
  - `main.swift`: thin one-shot entrypoint for the shared CLI layer
- `Sources/ZImageCLICommon/`
  - shared CLI parsing, request building, usage text, and one-shot execution wiring
- `Sources/ZImageServe/`
  - `main.swift`: staging-daemon entrypoint and client-side submission flow
- `Sources/ZImageServeCore/`
  - local socket transport and serial daemon coordination
- `Sources/ZImage/`
  - `Pipeline/`: `ZImagePipeline`, `ZImageControlPipeline`, `RuntimeOptions.swift`, scheduler wiring, snapshot helpers
  - `Model/`: Qwen text encoder, diffusion transformer, VAE
  - `Weights/`: cache lookup, Hugging Face download, safetensors reading, AIO detection, tensor mapping
  - `Quantization/`: quantization manifest format and quantize commands
  - `LoRA/`: LoRA and LoKr loading/application
  - `Support/`: model metadata and known-model registry
  - `Util/`: image I/O and control-memory telemetry
- `Tests/`
  - `ZImageTests/`: default fast suite
  - `ZImageIntegrationTests/`: heavier tests that require weights
  - `ZImageE2ETests/`: CLI build-and-run tests
- `docs/`
  - core reference docs, focused follow-up notes, background context, and archive material
- `scripts/`
  - build and developer helper scripts

## Current Source Of Truth

- package layout and supported platforms: `Package.swift`
- CLI flags and help output: `Sources/ZImageCLICommon/`
- staging service protocol and queue semantics: `Sources/ZImageServeCore/ServiceModels.swift` and `Sources/ZImageServeCore/StagingService.swift`
- Text-to-image API: `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- ControlNet and inpainting API: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- Serving residency policy surface: `Sources/ZImage/Pipeline/RuntimeOptions.swift`
- Known model ids and presets: `Sources/ZImage/Support/ZImageModelRegistry.swift`
- Default model id and weight-file resolution: `Sources/ZImage/Weights/ModelPaths.swift`
- Snapshot resolution, cache lookup, and Hugging Face download behavior: `Sources/ZImage/Weights/ModelResolution.swift` and `Sources/ZImage/Weights/HuggingFaceHub.swift`
- AIO checkpoint detection and canonicalization: `Sources/ZImage/Weights/AIOCheckpoint.swift`
- Quantization manifests and application: `Sources/ZImage/Quantization/ZImageQuantization.swift` and `Sources/ZImage/Weights/WeightsMapping.swift`
- CI build and release packaging: `.github/workflows/ci.yml`
- Fast behavior checks: the matching files under `Tests/ZImageTests/`, `Tests/ZImageIntegrationTests/`, and `Tests/ZImageE2ETests/`

## Conventions And Expectations

- The repo is package-first. There is no checked-in Xcode project or workspace; `Package.swift` is authoritative.
- Keep changes small and targeted unless the code clearly needs a broader refactor.
- Check for parallel implementations before stopping:
  - `ZImagePipeline` and `ZImageControlPipeline`
  - `ZImageTransformer2DModel` and `ZImageControlTransformer2DModel`
  - CLI help text and docs
- When changing user-visible behavior, update the matching docs in the same change:
  - CLI behavior: `README.md`, `docs/CLI.md`, `Sources/ZImageCLICommon/`, `Sources/ZImageCLI/main.swift`, `Sources/ZImageServe/main.swift`, and `Sources/ZImageServeCore/ServiceModels.swift` when the staged protocol or status output changes
  - Model loading semantics: `docs/MODELS_AND_WEIGHTS.md`, `docs/ARCHITECTURE.md`, relevant files in `Sources/ZImage/Weights/`
  - Build/test/release workflow: `docs/DEVELOPMENT.md`, `.github/workflows/ci.yml`, and helper scripts
- Prefer `docs/` as the detailed explanation layer. Keep `README.md` short and link outward.
- Known `Tongyi-MAI` ids do get model-aware presets. Local paths and unknown ids still fall back to the Turbo-compatible preset unless the caller overrides the relevant fields explicitly.
- `ZImageCLI control` does not expose the control-pipeline LoRA or prompt-enhancement fields that exist in `ZImageControlGenerationRequest`.
- Treat `docs/debug_notes/` and `docs/archive/` as historical or explanatory context unless a current doc explicitly points you there.

## Build, Test, And Lint

Default release build:

```bash
./scripts/build.sh
```

Default verification:

```bash
swift test
```

CI-like release-path build:

```bash
DERIVED_DATA_PATH=./dist ./scripts/build.sh
```

Opt-in heavier suites:

```bash
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter PipelineIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter ControlNetIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter LoRAIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter PerformanceTests
ZIMAGE_RUN_E2E_TESTS=1 swift test --filter CLIEndToEndTests
ZIMAGE_RUN_E2E_TESTS=1 swift test --filter ServeEndToEndTests
```

Opt-in Base smoke test:

```bash
ZIMAGE_RUN_INTEGRATION_TESTS=1 \
ZIMAGE_RUN_BASE_SMOKE=1 \
swift test --filter PipelineIntegrationTests/testBaseModelSmokeGeneration
```

If a task needs local LoRA integration coverage against a downloaded adapter, `Tests/ZImageIntegrationTests/LoRAIntegrationTests.swift` also honors `ZIMAGE_TEST_LORA_PATH`.

## CI And Release Expectations

- Pull requests run the SwiftPM verification job.
- Pushes to `main` run verification, then build and publish the nightly release artifact.
- The nightly artifact is `zimage.macos.arm64.zip`.
- The release job builds with Xcode 16.0, enables the MLX shader-preparation plugin non-interactively, copies `default.metallib` next to `ZImageCLI`, and smoke-runs `ZImageCLI --help`.

If you change build flags, artifact names, or release semantics, update `docs/DEVELOPMENT.md`, `README.md`, and `.github/workflows/ci.yml` together.

## Focused Workflows

- Control-memory work: use `docs/DEVELOPMENT.md` for the runnable probe, `docs/dev_plans/controlnet-memory-followup.md` for the current status, and `Sources/ZImage/Util/ControlMemoryTelemetry.swift` for the logging contract. Use the historical March 2026 notes only when you need the phase-by-phase background.
- Precision or numerical-parity work: read `docs/golden_checks.md` and `docs/context/zimage_runtime_precision_parity_report.md` before changing numerics.
- Model-loading bugs: inspect `Tests/ZImageTests/Weights/*` before changing resolver logic.
- Base vs Turbo behavior: inspect `Sources/ZImage/Support/ZImageModelRegistry.swift` and `Tests/ZImageTests/Support/ZImageModelRegistryTests.swift`.

## External References

Useful external reference projects are listed in `config/external-projects.example.yaml`.

For external-project references, prefer local clones configured in `config/external-projects.local.yaml`.

## Useful Local Resources

- Inspect `.safetensors` contents:
  - `~/bin/stls.py --format toon <file.safetensors>`
  - If not present: `curl https://gist.githubusercontent.com/zhutao100/cc481d2cd248aa8769e1abb3887facc8/raw/89d644c490bcf5386cb81ebcc36c92471f578c60/stls.py > /tmp/stls.py`
- Common Hugging Face snapshot roots:
  - `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo`
  - `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image`
  - `~/.cache/huggingface/hub/models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1`
  - `~/.cache/huggingface/hub/models--alibaba-pai--Z-Image-Fun-Controlnet-Union-2.1`
  - `~/.cache/huggingface/hub/models--alibaba-pai--Z-Image-Fun-Lora-Distill`
