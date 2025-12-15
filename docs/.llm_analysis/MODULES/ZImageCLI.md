# Module: ZImageCLI

## Purpose
Command-line interface for the ZImage application. Handles user input, argument parsing, and orchestration of the generation pipeline.

## Key Components
- **ZImageCLI**: Main struct with `run()` static method.
- **Argument Parsing**: Custom manual parsing loop (no `swift-argument-parser` dependency used).
- **Subcommands**:
  - Default (Generation): `ZImagePipeline`
  - `quantize`: `ZImageQuantizer`
  - `control`: `ZImageControlPipeline`
- **Progress Reporting**: TTY-aware `ProgressBar` or `PlainProgress`.

## Entry Point
- `run()`: Parses args -> Configures Global `GPU` cache -> Creates `ZImageGenerationRequest` -> Runs Pipeline.

## Dependencies
- `ZImage` (Core Library)
- `MLX`
- `Logging`
- `Foundation`
