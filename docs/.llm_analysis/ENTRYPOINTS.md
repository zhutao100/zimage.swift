# Entry Points

## CLI
- **Path**: `Sources/ZImageCLI/main.swift`
- **Description**: Main executable for the command-line interface. Uses `swift-argument-parser` to handle subcommands (generation, controlnet, quantization).

## Library
- **Module**: `ZImage`
- **Key Public Types** (Inferred from CLAUDE.md, to be verified):
  - `ZImagePipeline`
  - `ZImageControlPipeline`
  - `FlowMatchScheduler`
