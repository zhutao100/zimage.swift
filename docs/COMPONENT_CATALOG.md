# Component Catalog

## Subsystems

### 1. Application / CLI
- **Module**: `ZImageCLI`
- **Purpose**: Command-line interface for end-users.
- **Key Files**: `main.swift`.
- **Entry Point**: `ZImageCLI.run()`.

### 2. Pipelines
- **Module**: `ZImage` (Pipeline)
- **Purpose**: High-level orchestration of the generation process.
- **Key Classes**:
  - `ZImagePipeline`: Standard text-to-image.
  - `ZImageControlPipeline`: ControlNet-conditioned generation.
  - `ZImageGenerationRequest`: Configuration struct for generation jobs.

### 3. Core Models
- **Module**: `ZImage` (Model)
- **Purpose**: Neural network architectures.
- **Key Classes**:
  - `QwenTextEncoder`: Text encoding and prompt enhancement.
  - `ZImageTransformer2DModel`: Main Diffusion Transformer.
  - `ZImageControlTransformer2DModel`: DiT with ControlNet support.
  - `AutoencoderKL`: VAE for latent encoding/decoding.

### 4. Scheduler
- **Module**: `ZImage` (Pipeline)
- **Purpose**: Noise scheduling and sampling.
- **Key Classes**: `FlowMatchEulerScheduler`.
- **Algorithm**: Flow Matching with Euler integration.

### 5. Weight Management
- **Module**: `ZImage` (Weights)
- **Purpose**: Loading, mapping, and quantizing weights.
- **Key Classes**:
  - `ZImageWeightsMapper`: Main API for loading components.
  - `SafeTensorsReader`: Low-level file parser.
  - `LoRALoader`: Handles LoRA weight loading and fusion.
  - `HubSnapshot`: Manages Hugging Face Hub downloads.

### 6. Utilities
- **Module**: `ZImage` (Util/Quantization)
- **Purpose**: Support functions.
- **Key Classes**:
  - `QwenImageIO`: Image resizing and file I/O.
  - `ZImageQuantizer`: Offline model quantization tool.
  - `QwenTokenizer`: Text tokenization wrapper.

## External Dependencies
- **MLX Swift**: Tensor computation framework.
- **Swift Transformers**: Tokenization support.
- **Swift Argument Parser**: CLI argument parsing.
