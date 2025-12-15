# Module: Weights

## Purpose
Manages the loading, mapping, and caching of model weights and configurations from the Hugging Face Hub or local filesystem. Handles `.safetensors` parsing and quantization support.

## Key Components
- **ZImageWeightsMapper**: High-level API to load components (Transformer, VAE, TextEncoder) as `[String: MLXArray]`. Handles sharding and quantization detection.
- **SafeTensorsReader**: Low-level reader for the `.safetensors` format. Memory-maps files and extracts tensors as `MLXArray`.
- **ZImageModelConfigs**: Codable structs for model configurations (`transformer`, `vae`, `scheduler`, `text_encoder`).
- **ZImageFiles & ZImageRepository**: Constants and helpers for file paths and weight resolution logic (handling `model_index.json` vs legacy filenames).
- **HubSnapshot**: Abstraction over Hugging Face Hub downloads/caching.

## Data Flow
1. `HubSnapshot` ensures files exist locally.
2. `ZImageModelConfigs` loads JSON configs.
3. `ZImageWeightsMapper` uses `SafeTensorsReader` to map files to tensors.
4. Tensors are returned as dictionary maps `[String: MLXArray]` to be consumed by the Model layers.

## Dependencies
- `MLX`
- `Foundation`
- `Hub` (likely from `swift-transformers` or similar)
