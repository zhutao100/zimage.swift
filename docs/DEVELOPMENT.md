# Development

This doc is for contributors / maintainers working on the Swift package and CLI.

## Build

Release build (macOS):

```bash
xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

Note: `mlx-swift` may use a package plugin that prepares Metal shader libraries. In CI/non-interactive contexts, the repo’s GitHub Actions workflow uses:

- `-skipPackagePluginValidation`
- `ENABLE_PLUGIN_PREPAREMLSHADERS=YES`

See `.github/workflows/ci.yml` for the exact command line used for releases.

## Tests

Unit tests (recommended default):

```bash
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests
```

Other test suites exist but are intentionally heavier:

- Integration tests: `ZImageIntegrationTests` (require model downloads)
- E2E tests: `ZImageE2ETests` (build + run the CLI)

## Repo Conventions

- Swift Package Manager layout: `Sources/`, `Tests/`
- Keep model/weights behavior consistent with Diffusers when possible (see `~/workspace/custom-builds/diffusers`).
- Prefer updating docs in `docs/` (and linking from the root `README.md`) instead of duplicating large explanations in multiple places.

## Control-Path Memory Validation

When changing `ZImageControlPipeline` or the VAE encoder, use the same verification shape as the control-context memory remediation work:

```bash
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests
xcodebuild build -scheme ZImageCLI -destination 'platform=macOS' -derivedDataPath .build/xcode
.build/xcode/Build/Products/Debug/ZImageCLI control \
  --prompt "memory validation" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --width 1536 \
  --height 2304 \
  --steps 1 \
  --log-control-memory \
  --no-progress \
  --output /tmp/zimage-control-memory-check.png
```

Watch `control-context.after-baseline-reduction`, `control-context.after-eval`, `control-context.after-clear-cache`, and `decode.after-eval`. The retained runtime policy is:

- keep `--log-control-memory` as the supported probe
- unload transformer, ControlNet, and active LoRA state before `buildControlContext(...)`
- materialize the stored control-context tensor and clear cache before transformer/controlnet reload
- keep query-chunked VAE self-attention on by default
- skip tiled encode and VAE lifecycle splitting unless new measurements show a pathological regression again

## Performance & Memory Notes

Running these models on Apple Silicon can be memory-heavy, especially at high resolutions. Historical investigations are kept in `docs/archive/`, and the latest control-path outcome is recorded in `docs/dev_plans/control-context-memory-remediation.md`.
