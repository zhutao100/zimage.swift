# Glossary

## Project-Specific Terms
- **Z-Image**: The name of this Swift project/library.
- **Z-Image-Turbo**: The specific model architecture (a fast Diffusion Transformer) this project implements.
- **Refiner**: In the Z-Image DiT architecture, specific transformer blocks (`noiseRefiner`, `contextRefiner`) that process latents or context embeddings separately before they are merged in the main layers.
- **Prompt Enhancement**: A feature where the Text Encoder (Qwen LLM) is used to rewrite/expand a user's simple prompt into a detailed visual description before generation.

## General AI/ML Terms
- **DiT (Diffusion Transformer)**: A class of diffusion models that use a Transformer backbone instead of the traditional U-Net.
- **VAE (Variational Autoencoder)**: A model used to compress images into a lower-dimensional "latent space" for efficient processing, and decode them back to pixels.
- **LoRA (Low-Rank Adaptation)**: A technique for fine-tuning large models by injecting trainable low-rank matrices into layers (usually Linear layers).
- **ControlNet**: A neural network structure to control diffusion models by adding extra conditions (edges, depth, pose).
- **Quantization**: Reducing the precision of model weights (e.g., from Float16 to Int8 or Int4) to save memory and increase speed.
- **RoPE (Rotary Positional Embeddings)**: A method for encoding position information in Transformers, allowing for better handling of sequence lengths.
- **Safetensors**: A safe, fast, and zero-copy file format for storing tensors (model weights).
- **LyCORIS/LoKr**: Advanced LoRA variants (Locally Optimized Low-Rank adaptation) supported by the loader.
