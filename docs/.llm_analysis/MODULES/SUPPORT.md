# Module: Utilities & Support

## Purpose
Provides essential infrastructure for image processing, weight management, and model optimization.

## Components

### Image I/O (`QwenImageIO`)
- **Role**: Conversion between `CGImage` and `MLXArray`.
- **Features**:
  - High-quality Lanczos resampling (custom Swift implementation).
  - Normalization (`[-1, 1]`) and denormalization (`[0, 1]`).
  - PNG saving.

### LoRA Loader (`LoRALoader`)
- **Role**: Loads and applies Low-Rank Adaptation weights.
- **Features**:
  - **Format Support**: Standard LoRA, LyCORIS/LoKr.
  - **Key Remapping**: Heuristics to map external weight keys (Diffusers/ComfyUI) to internal `ZImage` module paths.
  - **Fusion**: Merges LoRA deltas into base weights (Linear or QuantizedLinear) at runtime.

### Quantization (`ZImageQuantizer`)
- **Role**: Reduces model size and memory footprint.
- **Features**:
  - **Offline**: Quantizes and saves models to disk with a manifest.
  - **Runtime**: Loads quantized weights (4-bit/8-bit, Group Size 32/64/128).
  - **Modes**: Affine, MXFP4.

### LLM Generation (`QwenGeneration`)
- **Role**: Uses the Qwen Text Encoder as a generative LLM.
- **Features**:
  - **Prompt Enhancement**: Rewrites user prompts using a specialized system prompt to improve image quality.
  - **Autoregressive Loop**: Simple sampling implementation with KV-caching.

## Dependencies
- `CoreGraphics` / `ImageIO`
- `MLX`
- `MLXNN`
- `Hub`
