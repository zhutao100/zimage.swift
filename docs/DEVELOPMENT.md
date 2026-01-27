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

## Performance & Memory Notes

Running these models on Apple Silicon can be memory-heavy, especially at high resolutions. Historical investigations are kept in `docs/archive/`.

