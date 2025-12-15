# Repository Map

## Root
- `Package.swift`: Swift Package Manager manifest.
- `README.md`: User documentation.
- `CLAUDE.md`: Developer guide and architecture overview.
- `AGENTS.md`: Specific context for AI agents.

## Sources
- `Sources/ZImageCLI/`: Command Line Interface executable.
  - `main.swift`: Entry point.
- `Sources/ZImage/`: Core library.
  - `Model/`: Neural network architecture definitions.
    - `TextEncoder/`: Qwen-based text encoder.
    - `Transformer/`: DiT transformer (ZImageTransformer2D).
    - `VAE/`: Variational Autoencoder.
  - `Pipeline/`: Generation pipelines and schedulers.
  - `Weights/`: Weight loading, mapping, and quantization support.
  - `Tokenizer/`: Tokenizer implementation.
  - `Quantization/`: Quantization utilities.
  - `Util/`: General utilities (ImageIO).
  - `Support/`: Metadata support.

## Tests
- `Tests/ZImageTests/`: Unit tests.
- `Tests/ZImageIntegrationTests/`: Integration tests (require weights).
- `Tests/ZImageE2ETests/`: End-to-end CLI tests.
