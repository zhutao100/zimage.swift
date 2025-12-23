# Project Architecture

## Executive Summary
**ZImage.swift** is a high-performance, native Swift implementation of the "Z-Image-Turbo" text-to-image generation model, built on top of Apple's MLX framework. It is designed specifically for Apple Silicon (M-series chips), leveraging unified memory and the Neural Engine for efficient inference.

The project implements a full Diffusion Transformer (DiT) pipeline, including a Qwen-based text encoder (which doubles as a prompt enhancer), a Variational Autoencoder (VAE) for latent space compression, and a Flow Matching scheduler. It features advanced capabilities such as ControlNet support (for guided generation), dynamic LoRA/LoKr adaptation, model quantization (4-bit/8-bit), and memory-optimized execution (phase-scoped lifetimes) to run on consumer hardware with limited RAM.

## Technology Stack
- **Language**: Swift 5.9+
- **Core Framework**: [MLX Swift](https://github.com/ml-explore/mlx-swift) (NumPy-like array library for Apple Silicon)
- **Neural Networks**: `MLXNN` (Layers), `MLXFast` (Optimized kernels like RoPE, Attention)
- **Tokenization**: `swift-transformers` (Hugging Face Tokenizers wrapper)
- **Model Loading**: `Hub` (Hugging Face Hub interaction)
- **Image Processing**: CoreGraphics, ImageIO (Native Apple SDKs)

## Component Map

### 1. Application Layer (`Sources/ZImageCLI`)
The entry point for users. It parses command-line arguments and orchestrates the high-level tasks.
- **`ZImageCLI`**: Handles `generate`, `control`, and `quantize` commands. Manages global GPU cache settings and progress reporting.

### 2. Pipeline Layer (`Sources/ZImage/Pipeline`)
The "brain" that coordinates models to produce an image.
- **`ZImagePipeline`**: Standard Text-to-Image pipeline. Handles model loading, prompt encoding, latent initialization, denoising loop, and decoding. Implements **phase-scoped lifetimes**: unloads the Text Encoder after prompt processing and the Transformer before VAE decoding to minimize peak memory.
- **`ZImageControlPipeline`**: Extends the standard pipeline to support ControlNet (conditioning via edge maps, depth maps, pose, etc.) and Inpainting.
- **`FlowMatchEulerScheduler`**: Implements the Flow Matching Euler discrete scheduler with "dynamic shifting" for resolution-dependent noise scheduling.

### 3. Model Layer (`Sources/ZImage/Model`)
Contains the neural network architectures.
- **Text Encoder (`/TextEncoder`)**: A Qwen-based Transformer.
  - Acts as an **Encoder** to produce embeddings for the DiT.
  - Acts as a **Generator** (LLM) to rewrite/enhance user prompts before generation.
- **Transformer (`/Transformer`)**: The core Diffusion Transformer (DiT).
  - **`ZImageTransformer2DModel`**: Uses a hybrid architecture with separate "refiner" streams for image and text that merge into joint blocks.
  - **`ZImageControlTransformer2DModel`**: Adds ControlNet blocks to the standard DiT.
- **VAE (`/VAE`)**: `AutoencoderKL` for encoding images to latents and decoding latents to images.

### 4. Infrastructure Layer (`Sources/ZImage/Weights`, `/Quantization`, `/LoRA`)
Handles the "heavy lifting" of model management.
- **Weights**: Downloads models from Hugging Face (`HubSnapshot`), parses `.safetensors` files (`SafeTensorsReader`), and maps PyTorch keys to MLX parameter hierarchies (`ZImageWeightsMapper`). Supports **AIO Checkpoints** (`ZImageAIOCheckpoint`) to load Transformer, Text Encoder, and VAE from a single file.
- **Quantization**: Compresses models to 4-bit/8-bit precision (`ZImageQuantizer`), enabling them to fit in RAM. Supports Group-wise quantization.
- **LoRA**: Implements Low-Rank Adaptation (`LoRAApplicator`). Supports both "Baked-in" (merge weights) and "Dynamic" (runtime wrapper layers) application, including LoKr (Kronecker product) support.

### 5. Support Layer (`Sources/ZImage/Support`, `/Tokenizer`, `/Util`)
- **`QwenTokenizer`**: Handles text tokenization (Chat templates vs Plain).
- **`QwenImageIO`**: High-performance image resizing (Lanczos) and normalization.
- **`ModelMetadata`**: Static configuration defaults for the Z-Image architecture.

## Data Flow (Text-to-Image)

1.  **Input**: User Prompt ("A cute cat")
2.  **Enhancement (Optional)**:
    - Text Encoder (LLM Mode) rewrites prompt -> "A hyper-realistic photo of a cute cat..."
3.  **Encoding**:
    - Tokenizer -> Token IDs
    - Text Encoder -> Text Embeddings (Hidden States)
4.  **Latent Initialization**:
    - Generate random Gaussian noise `(1, 16, H/8, W/8)`
5.  **Denoising Loop** (e.g., 20 steps):
    - **Input**: Latents + Timestep + Text Embeddings
    - **Transformer**: Predicts "velocity" (noise difference)
    - **Scheduler**: Updates Latents based on velocity
6.  **Decoding**:
    - VAE Decoder -> Pixel Space `(1, 3, H, W)`
7.  **Output**: Save to PNG

## Key Design Patterns
- **Dual-Mode Models**: The Text Encoder serves two distinct purposes (Encoding vs Generation) sharing the same weights.
- **Dynamic LoRA**: Instead of modifying weights permanently, LoRA adapters wrap existing layers at runtime, allowing instant switching between styles without reloading the base model.
- **Quantization-Aware**: The pipeline works seamlessly with quantized base weights, even when applying high-precision LoRA adapters on top.
- **Manifest-Driven Loading**: Quantization and Model config are driven by JSON manifests, decoupling the code from specific file naming conventions where possible.

## Build & Deployment
- **Build System**: Swift Package Manager (SPM).
- **Targets**: `ZImage` (Library) and `ZImageCLI` (Executable).
- **Optimization**: Uses `MLXFast` for optimized metal kernels (RoPE, Attention).
