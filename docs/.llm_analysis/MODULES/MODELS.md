# Module: Core Models

## Purpose
Defines the neural network architectures used for text encoding, latent diffusion, and image encoding/decoding.

## Components

### Text Encoder (`QwenTextEncoder`)
- **Architecture**: Qwen (Transformer-based LLM).
- **Role**: Encodes text prompts into embeddings.
- **Key Features**:
  - Uses RoPE (Rotary Positional Embeddings).
  - `encodeForZImage`: Extracts second-to-last hidden state.
  - `encodeJoint`: Supports replacing tokens with vision embeddings (multi-modal support).

### Transformer (`ZImageTransformer2D`)
- **Architecture**: Diffusion Transformer (DiT).
- **Role**: Predicts noise/velocity in latent space conditioned on text and timestep.
- **Key Features**:
  - **Structure**: `tEmbedder` + `xEmbedder` -> `noiseRefiner` + `contextRefiner` -> `layers` -> `finalLayer`.
  - **Refiners**: Distinct blocks for processing noise (latents) and context (captions) before merging.
  - **Modulation**: `adaLN` (Adaptive Layer Norm) for conditioning.
  - **Caching**: `TransformerCache` pre-computes RoPE frequencies for variable resolutions/aspect ratios.

### VAE (`AutoencoderKL`)
- **Architecture**: Variational Autoencoder with KL divergence.
- **Role**: Compresses images into latents (Encode) and reconstructs images from latents (Decode).
- **Key Features**:
  - `Encoder`: Downsampling blocks + Mid-block attention.
  - `Decoder`: Mid-block attention + Upsampling blocks.
  - Uses scaling and shifting factors for latent normalization.

## Dependencies
- `MLX`
- `MLXNN`
- `MLXFast`
