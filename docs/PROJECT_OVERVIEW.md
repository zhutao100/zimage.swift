# Project Overview

## 1. Introduction
**Z-Image.swift** is a high-performance, native Swift port of the **Z-Image-Turbo** text-to-image generation model, specifically optimized for Apple Silicon (M-series chips) using the [MLX](https://github.com/ml-explore/mlx-swift) framework. It provides both a reusable library (`ZImage`) and a command-line interface (`ZImageCLI`) for generating images from text prompts.

## 2. Core Value Proposition
- **Apple Silicon Native**: Leverages the Unified Memory Architecture (UMA) and Metal Performance Shaders via MLX for efficient inference on macOS.
- **Memory Efficient**: Supports aggressive memory management (loading/unloading models per stage) and 4-bit/8-bit quantization to run on consumer hardware (e.g., 8GB/16GB Macs).
- **Feature Rich**: Supports advanced features like LoRA (Low-Rank Adaptation), ControlNet (conditional generation), and Prompt Enhancement (using the text encoder as an LLM).

## 3. Primary Use Cases
- **Text-to-Image Generation**: Generating high-quality images from natural language descriptions.
- **Conditioned Generation**: Using ControlNet (Canny, Depth, Pose) to guide image structure.
- **Style Transfer/Fine-tuning**: Applying LoRA weights to adapt the model's style or subject matter.
- **Model Optimization**: Quantizing standard `.safetensors` models to reduced precision for lower memory usage.

## 4. Key Abstractions
- **Pipeline**: The central orchestrator (e.g., `ZImagePipeline`) that manages the multi-stage generation process (Encode -> Diffuse -> Decode).
- **Transformer (DiT)**: The core diffusion model (`ZImageTransformer2D`) that predicts noise in latent space.
- **Scheduler**: The algorithm (`FlowMatchEulerScheduler`) that controls the denoising steps.
- **Weights Mapper**: A unified system (`ZImageWeightsMapper`) for loading, mapping, and quantizing model weights from the Hugging Face Hub or local files.

## 5. Runtime Model
- **Sequential Execution**: To minimize peak memory usage, the pipeline loads models sequentially:
  1.  **Text Encoder**: Loaded, encodes prompt, then unloaded.
  2.  **Transformer**: Loaded, runs diffusion loop, then unloaded.
  3.  **VAE**: Loaded, decodes latents to pixels.
- **Unified Memory**: Heavy reliance on MLX's unified memory arrays (`MLXArray`) to avoid unnecessary CPU-GPU copies.
