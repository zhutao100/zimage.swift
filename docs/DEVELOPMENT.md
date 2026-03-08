# Development

This document covers the current contributor workflow: build, test, format, release, and the targeted validation paths that still matter in this repo.

## Build

Fast path:

```bash
./scripts/build.sh
```

Explicit release build:

```bash
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

CI uses a more explicit non-interactive form:

```bash
xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath ./dist -skipPackagePluginValidation ENABLE_PLUGIN_PREPAREMLSHADERS=YES CLANG_COVERAGE_MAPPING=NO
```

The package depends on the MLX shader-preparation plugin. In local interactive builds, allow the prompt if Xcode asks. In CI or scripted environments, use the same plugin flags as `.github/workflows/ci.yml`.

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

Heavier test suites are opt-in:

- `ZImageIntegrationTests`: require real model weights
- `ZImageE2ETests`: build and execute the CLI

Use those only when the task specifically needs them.

Enable the heavier suites explicitly:

```bash
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter PipelineIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter ControlNetIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter LoRAIntegrationTests
ZIMAGE_RUN_INTEGRATION_TESTS=1 swift test --filter PerformanceTests
ZIMAGE_RUN_E2E_TESTS=1 swift test --filter CLIEndToEndTests
```

`ZImageE2ETests` now use the `ZImageCLI` executable built by the same SwiftPM stack as `swift test`. They no longer invoke `xcodebuild` internally.

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
- If you run the test through SwiftPM, keep `mlx.metallib` available via `scripts/build_mlx_metallib.sh` first.

## CI And Packaging

Current CI behavior:

- trigger: pushes to `main`
- runner: `macos-latest`
- Xcode: `16.0`
- artifact: `zimage.macos.arm64.zip`
- release target: GitHub prerelease tag `nightly`

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
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests
xcodebuild build -scheme ZImageCLI -destination 'platform=macOS' -derivedDataPath .build/xcode
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
- `decode.after-eval`

Current retained policy:

- keep `--log-control-memory` as the public probe
- unload transformer, ControlNet, and active LoRA state before control-context build when they are not needed
- load the control-path VAE encoder on demand and unload it immediately after the typed control context is materialized
- clear MLX cache before denoiser modules are reloaded
- keep incremental ControlNet hint accumulation
- keep query-chunked VAE self-attention enabled by default

### Numerical-Parity Work

If you are chasing Swift vs Python or Diffusers drift, read:

- [golden_checks.md](golden_checks.md)
- `docs/context/`

Those docs are the current background set for parity and precision work.

## Performance Notes

These models are large. First-time downloads can be tens of GB, and higher resolutions still stress unified memory. Historical investigations live under `docs/archive/`; the current control-memory outcome is captured in `docs/dev_plans/control-context-memory-remediation.md` and `docs/dev_plans/controlnet-memory-followup.md`.
