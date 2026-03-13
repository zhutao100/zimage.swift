# Development

This document covers the current contributor workflow: build, test, format, release, and the targeted validation paths that still matter in this repo.

## Build

Default release-path build:

```bash
./scripts/build.sh
```

SwiftPM can also build the staging executable directly:

```bash
swift build --product ZImageServe
```

Override the derived-data path or configuration when needed:

```bash
DERIVED_DATA_PATH=./dist ./scripts/build.sh
CONFIGURATION=Debug ./scripts/build.sh
```

Equivalent explicit command:

```bash
xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath ./dist -skipPackagePluginValidation ENABLE_PLUGIN_PREPAREMLSHADERS=YES CLANG_COVERAGE_MAPPING=NO
```

`scripts/build.sh` uses the same non-interactive plugin flags as CI so local scripted builds match the release path more closely.

### SwiftPM-Only Binary Builds

If you intentionally build the CLI with `swift build`, you may also need to colocate `mlx.metallib`:

```bash
swift build -c debug
./scripts/build_mlx_metallib.sh --configuration debug
```

That workflow is mainly for local experimentation. The default repo path is still the Xcode build above.

## Tests

Default verification path:

```bash
swift test
```

The MLX-backed test support prepares the SwiftPM metallib automatically on demand, and the opt-in E2E suite builds the SwiftPM `ZImageCLI` product automatically when needed.

Heavier test suites are opt-in:

- `ZImageIntegrationTests`: require real model weights
- `ZImageE2ETests`: build and execute the CLI

Enable the heavier suites explicitly:

```bash
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter PipelineIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter ControlNetIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter LoRAIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter PerformanceTests
ZIMAGE_RUN_E2E_TESTS=1 swift test --filter CLIEndToEndTests
```

`ZImageE2ETests` use the `ZImageCLI` executable built by the same SwiftPM stack as `swift test`. They do not invoke `xcodebuild` internally.
The same preparation flow now builds `ZImageServe` on demand for the staging-daemon E2E checks.

Additional integration-test knobs:

- `ZIMAGE_BASE_SMOKE_MODEL`: optional local override for the Base smoke test snapshot path
- `ZIMAGE_TEST_LORA_PATH`: optional local LoRA path override for `LoRAIntegrationTests`

### Opt-In Base Smoke Test

For a real-model Base sanity check without enabling the full integration suite by default:

```bash
ZIMAGE_RUN_INTEGRATION_TESTS=1 \
ZIMAGE_RUN_BASE_SMOKE=1 \
ZIMAGE_BASE_SMOKE_MODEL="$HOME/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021" \
swift test --filter PipelineIntegrationTests/testBaseModelSmokeGeneration
```

Notes:

- `ZIMAGE_RUN_BASE_SMOKE=1` is required; otherwise the test skips.
- `ZIMAGE_BASE_SMOKE_MODEL` is optional. When omitted, the test uses `Tongyi-MAI/Z-Image` and resolves it through the normal cache/download path.

## CI And Packaging

Current CI behavior:

- triggers:
  - pull requests: run the SwiftPM verification job
  - pushes to `main`: run verification, then build/package/release the nightly artifact
- runner: `macos-latest`
- Xcode: `16.0`
- artifact: `zimage.macos.arm64.zip`
- release target: GitHub prerelease tag `nightly`
- smoke checks:
  - `swift test`
  - `ZImageCLI --help` from the packaged release directory after `default.metallib` is copied alongside the binary

Source of truth:

- `.github/workflows/ci.yml`

If you change build flags, artifact names, or release semantics, update this doc, the workflow, and the root `README.md` together.

## Docs Expectations

When user-visible behavior changes, update the docs in the same patch:

- CLI behavior: `README.md`, `docs/CLI.md`, `Sources/ZImageCLI/main.swift`
- model loading or cache behavior: `docs/MODELS_AND_WEIGHTS.md`
- code structure and ownership: `docs/ARCHITECTURE.md`
- build/test/release workflow: this file and `.github/workflows/ci.yml`

Prefer one detailed explanation in `docs/` and link to it rather than duplicating long prose in multiple places.

## Targeted Validation

### Control-Memory Validation

When changing `ZImageControlPipeline`, ControlNet loading, or the VAE encode/decode path, use the retained high-resolution probe:

```bash
swift test
xcodebuild build -scheme ZImageCLI -configuration Debug -destination 'platform=macOS' -derivedDataPath .build/xcode
.build/xcode/Build/Products/Debug/ZImageCLI control \
  --prompt "memory validation" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --width 1536 \
  --height 2304 \
  --steps 1 \
  --log-control-memory \
  --no-progress \
  --output /tmp/zimage-control-memory-check.png
```

Watch these markers:

- `control-context.after-baseline-reduction`
- `control-context.after-eval`
- `control-context.after-clear-cache`
- `transformer.denoising-load.after-apply`
- `controlnet.denoising-load.after-apply`
- `decode.after-eval`

Current retained policy:

- keep `--log-control-memory` as the public probe
- keep transformer, ControlNet, and active LoRA state absent until denoising is about to start
- load the control-path VAE encoder on demand and unload it immediately after the typed control context is materialized
- clear MLX cache before denoiser modules are loaded
- keep incremental ControlNet hint accumulation
- keep query-chunked VAE self-attention enabled by default

Current measured status from the March 8, 2026 follow-up run:

- high-resolution `1536x2304` control-context residency after cache clear stayed around `315 MiB`
- the remaining large jump happens at the deferred denoiser load boundary, not during control-context storage
- the retained high-resolution probe still peaked around `59.3 GiB` process footprint

The current follow-up summary lives in [dev_plans/controlnet-memory-followup.md](dev_plans/controlnet-memory-followup.md).

### Numerical-Parity Work

If you are chasing Swift vs Python or Diffusers drift, read:

- [golden_checks.md](golden_checks.md)
- [context/zimage_runtime_precision_parity_report.md](context/zimage_runtime_precision_parity_report.md)

Those docs are the current background set for parity and precision work.

## Performance Notes

These models are large. First-time downloads can be tens of GB, and higher resolutions still stress unified memory. Historical investigations live under `docs/debug_notes/` and `docs/archive/`; the current operating summary lives in [dev_plans/controlnet-memory-followup.md](dev_plans/controlnet-memory-followup.md).
