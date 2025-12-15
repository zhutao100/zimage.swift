# Module: Pipelines & Scheduler

## Purpose
Orchestrates the text-to-image generation process, managing model loading, memory optimization, and the diffusion loop.

## Components

### Pipelines

#### `ZImagePipeline`
- **Role**: Standard text-to-image generation.
- **Process**:
  1.  **Setup**: Downloads models, loads configs.
  2.  **Text Encoding**: Loads `QwenTextEncoder` & `Tokenizer`. Encodes prompt. **Unloads** encoder to free memory.
  3.  **Latent Initialization**: Generates random latents.
  4.  **Diffusion**: Loads `ZImageTransformer2D`. Runs `FlowMatchEulerScheduler` loop. Supports LoRA and weight overrides. **Unloads** transformer.
  5.  **Decoding**: Loads `AutoencoderKL` (Decoder). Decodes latents to image.
- **Key Features**: Aggressive memory management (sequential loading/unloading), LoRA support, Prompt Enhancement (via LLM).

#### `ZImageControlPipeline`
- **Role**: ControlNet-conditioned generation (e.g., Edge -> Image).
- **Process**: Similar to `ZImagePipeline` but:
  - Loads `ZImageControlTransformer2D` (modified DiT).
  - Loads & Encodes **Control Image** using VAE Encoder (`controlContext`).
  - Passes `controlContext` to transformer during diffusion.

### Scheduler

#### `FlowMatchEulerScheduler`
- **Algorithm**: Flow Matching with Euler integration.
- **Logic**: Updates latents by moving along the vector field predicted by the transformer: `x_{t+1} = x_t + v_t * dt`.
- **Timesteps**: Supports dynamic shifting of sigmas.

### Tokenizer

#### `QwenTokenizer`
- **Role**: Tokenizes text for the Qwen Text Encoder.
- **Features**: Supports Chat Templates (`applyChatTemplate`) and plain text. Handles special tokens (`<|im_start|>`, etc.).

## Dependencies
- `ZImage` (Models, Weights)
- `MLX`
- `Hub`
- `Tokenizers` (swift-transformers)
