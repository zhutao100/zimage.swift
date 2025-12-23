# Module: Weights (`Sources/ZImage/Weights`)

## Purpose
This module handles the discovery, downloading (via Hugging Face Hub), mapping, and loading of model weights into the application's memory. It bridges the gap between on-disk artifacts (SafeTensors, JSON configs) and runtime MLX arrays.

## Key Components

### `ModelResolution`
- **Responsibility**: Locates model artifacts.
- **Logic**: Checks if a string is a local path or a Hugging Face Hub ID.
- **Fallback**: Tries to find models in the local Hugging Face cache (`~/.cache/huggingface/hub`) before initiating a download.
- **Dependencies**: `Hub` (swift-transformers).

### `HubSnapshot`
- **Responsibility**: Manages the interface with the `HubApi` for downloading specific revisions and patterns.
- **Caching**: Resolves and creates cache directories for storing model snapshots.

### `SafeTensorsReader`
- **Responsibility**: Low-level parsing of the `.safetensors` binary format.
- **Mechanism**:
  - Memory-maps the file.
  - Parses the JSON header to locate tensor offsets and metadata.
  - Returns `MLXArray` instances backed by the mapped memory (zero-copy where possible).
- **Format Support**: Handles various data types (F32, F16, BF16, I64, etc.) mapped to MLX `DType`.

### `ZImageAIOCheckpoint`
- **Responsibility**: Handles detection and component extraction for "All-In-One" single-file `.safetensors` checkpoints.
- **Features**:
  - **Inspection**: Quickly reads the header to determine if a file contains Transformer + Text Encoder + VAE weights.
  - **Component Splitting**: Isolates weights for each component based on key prefixes (e.g., `model.diffusion_model.` for Transformer, `vae.` for VAE).
  - **Text Encoder Prefix Discovery**: Heuristically finds the correct prefix for text encoder weights (e.g., `text_encoders.clip_l.transformer.`).
  - **Validation**: Reports diagnostics if expected keys are missing.

### `ZImageWeightsMapper`
- **Responsibility**: High-level orchestrator for loading specific model components (Transformer, TextEncoder, VAE).
- **Features**:
  - Detects quantization (looks for `quantization.json`).
  - Aggregates sharded weights (e.g., `model-00001-of-00003.safetensors`).
  - Supports loading "override" weights (e.g., specific LoRA or fine-tuned weights).

### `ZImageWeightsMapping` & `ZImageWeightsParameters`
- **Responsibility**: Translates PyTorch/HuggingFace naming conventions to the project's internal MLX module structure.
- **Complexity**:
  - **Re-mapping**: Changes prefixes (e.g., `model.layers` -> `encoder.layers`).
  - **Structure Reconstruction**: `ZImageWeightsParameters` manually reconstructs nested dictionary structures expected by MLX's `ModuleParameters.unflattened`.
  - **Tensor Manipulation**: Handles transposing (e.g., for Conv2D weights) and splitting (though explicit q/k/v splitting logic seems delegated to the model layers themselves or handled via specific key lookups).
  - **Quantization Support**: Handles `scales` and `biases` keys for quantized layers.

### `WeightsAudit`
- **Responsibility**: Verifies that loaded weights match the expected model parameters.
- **Reporting**: Logs summary statistics (matched, missing, extra) and samples of missing/extra keys to avoid log spam. Critical for debugging sharding or naming mismatches.

### `ModelConfigs`
- **Responsibility**: Strongly-typed decoding of JSON configuration files (`config.json`, `scheduler_config.json`).
- **Structs**: `ZImageTransformerConfig`, `ZImageVAEConfig`, `ZImageTextEncoderConfig`, `ZImageSchedulerConfig`.

## Usage Pattern
1. `ModelResolution.resolve(...)` -> `URL` (snapshot path).
2. `ZImageWeightsMapper(snapshot: url)` is instantiated.
3. `mapper.loadTransformer()` / `mapper.loadVAE()` is called.
4. Raw dictionaries `[String: MLXArray]` are passed to `ZImageWeightsMapping.apply...` or `ZImageWeightsParameters` to be injected into the instantiated model.
