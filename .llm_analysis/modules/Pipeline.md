# Module: Pipeline & Generation (`Sources/ZImage/Pipeline`)

## Purpose
Orchestrates the end-to-end image generation process. It ties together the Model components (Transformer, TextEncoder, VAE) and the Scheduler to turn a user prompt into a final image.

## Key Components

### `ZImagePipeline`
- **Responsibility**: The high-level API for generation.
- **Capabilities**:
  - **Model Management**: Handles loading/unloading of individual components to manage memory usage.
  - **Memory Optimization**: Implements phase-scoped lifetimes. Text encoder is released immediately after embedding generation. Transformer is unloaded and cache cleared before VAE decoding to prevent memory spikes.
  - **AIO Support**: Detects and loads single-file "All-In-One" checkpoints (Transformer + Text Encoder + VAE) via `ZImageAIOCheckpoint`, bypassing base model weight loading.
  - **LoRA Support**: Dynamically loads/unloads LoRA adapters into the Transformer.
  - **Transformer Overrides**: Allows swapping the base transformer weights with a fine-tuned checkpoint (safetensors) without reloading the rest of the model.
  - **Prompt Enhancement**: Invokes the `QwenTextEncoder`'s generation capability to rewrite prompts before diffusion.
  - **CFG**: Implements standard Classifier-Free Guidance.
- **Flow**:
  1.  **Prep**: Resolve model paths, load components, compile LoRA.
  2.  **Enhance**: (Optional) Expand prompt via LLM.
  3.  **Encode**: Convert Text -> Embeddings (Positive & Negative).
  4.  **Init Latents**: Create random noise `(1, 16, h, w)`.
  5.  **Denoise**: Loop `N` steps using `FlowMatchEulerScheduler`.
  6.  **Decode**: Latents -> VAE -> Image.
  7.  **Save**: VAE Output -> PNG.

### `FlowMatchEulerScheduler`
- **Responsibility**: Controls the noise schedule and update step.
- **Algorithm**: Euler method for Flow Matching.
- **Feature**: **Dynamic Shifting**. Adjusts the time/noise schedule based on the target image resolution (`mu` parameter). This ensures consistent signal-to-noise ratios across different aspect ratios and resolutions.

### `ZImageGenerationRequest`
- **Responsibility**: Configuration object for a single generation job.
- **Fields**: Prompt, Negative Prompt, Dimensions, Steps, Guidance Scale, Seed, LoRA Config, etc.

## Data Flow
`User Prompt` -> `[TextEncoder]` -> `Embeddings`
`Random Noise` -> `[Transformer (Loop)]` <- `Embeddings`
                                      <- `Timestep`
`Refined Latents` -> `[VAE]` -> `Image`

## Dependencies
- Relies on all `Sources/ZImage/Model` components.
- Uses `HubApi` for model resolution.
- Uses `QwenImageIO` for saving results.
