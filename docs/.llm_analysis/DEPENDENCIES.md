# Dependencies

## External (from Package.swift)
- **mlx-swift** (Apple): Machine learning framework for Apple Silicon.
  - Modules: `MLX`, `MLXFast`, `MLXNN`, `MLXOptimizers`, `MLXRandom`.
- **swift-transformers** (HuggingFace): Tokenizers and transformer utilities.
  - Modules: `Transformers`.
- **swift-log** (Apple): Logging API.
  - Modules: `Logging`.
- **swift-argument-parser** (Apple): CLI argument parsing (Implied by CLI usage, standard for Swift CLIs).

## Internal
- `ZImage` depends on `mlx-swift`, `swift-transformers`, `swift-log`.
- `ZImageCLI` depends on `ZImage`.
