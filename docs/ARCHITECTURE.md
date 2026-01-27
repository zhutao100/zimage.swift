# Architecture

This document explains **how the codebase is structured today** and where the main “source of truth” entry points live.

For a runnable quickstart, see the root `README.md`.

## Executive Summary

**ZImage.swift** is a native Swift implementation of the `Tongyi-MAI/Z-Image-Turbo` text-to-image model built on top of **MLX (mlx-swift)**. It ships as:

- A library product: `ZImage`
- A CLI executable: `ZImageCLI`

At a high level, the project implements a full DiT pipeline:

- **Tokenizer + Qwen text encoder** → prompt embeddings (and optional prompt enhancement)
- **Diffusion Transformer (DiT)** → denoising loop driven by Flow-Matching
- **VAE** → decode latents → PNG output

## Technology Stack

- Swift 5.9+ (Package.swift uses Swift tools 5.9)
- MLX Swift (`MLX`, `MLXNN`, `MLXFast`, `MLXRandom`) for tensor compute (Metal / CPU fallback)
- `swift-transformers` for tokenizers + Hugging Face Hub access (`Hub`)
- `swift-log` for logging
- CoreGraphics / ImageIO for image I/O (where available)

## Repository Map

- CLI entry point: `Sources/ZImageCLI/main.swift`
- Pipelines: `Sources/ZImage/Pipeline/*`
  - `ZImagePipeline.swift` (text-to-image + LoRA + optional prompt enhancement)
  - `ZImageControlPipeline.swift` (ControlNet + inpaint + optional prompt enhancement)
  - `FlowMatchScheduler.swift` (`FlowMatchEulerScheduler`)
- Models: `Sources/ZImage/Model/*`
  - Text encoder (Qwen): `Sources/ZImage/Model/TextEncoder/*`
  - Transformer (DiT): `Sources/ZImage/Model/Transformer/*`
  - VAE: `Sources/ZImage/Model/VAE/*`
- Weights / downloading / mapping: `Sources/ZImage/Weights/*`
- LoRA: `Sources/ZImage/LoRA/*`
- Quantization: `Sources/ZImage/Quantization/*`
- Tests: `Tests/*` (unit / integration / e2e)

## Public API (Library)

The library surface is intentionally small and “pipeline-first”:

- `ZImageGenerationRequest` + `ZImagePipeline`
  - `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `ZImageControlGenerationRequest` + `ZImageControlPipeline`
  - `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

The CLI is a thin wrapper around these pipelines:

- `Sources/ZImageCLI/main.swift`

## Model Loading & Weight Resolution

The pipeline accepts a `--model` spec that can be:

- A Hugging Face repo id: `org/repo` (optionally `org/repo:revision`)
- A local directory containing a Diffusers-style layout (e.g. `transformer/`, `text_encoder/`, `vae/`, `tokenizer/`)
- A local `.safetensors` file:
  - Treated as an **AIO checkpoint** if it contains transformer + text encoder + VAE tensors with recognized key prefixes
  - Otherwise treated as a **transformer-only override** layered on top of the base model

Key code:

- Cache / download resolution: `Sources/ZImage/Weights/ModelResolution.swift`
- Model selection logic (AIO vs override): `Sources/ZImage/Pipeline/ZImagePipeline.swift` (`resolveModelSelection`)
- AIO inspection + canonicalization: `Sources/ZImage/Weights/AIOCheckpoint.swift`

## Quantization

Quantized weights are represented as a regular model folder plus a manifest:

- Base model: `quantization.json`
- ControlNet: `controlnet_quantization.json`

Key code:

- Quantization implementation: `Sources/ZImage/Quantization/ZImageQuantization.swift`
- Applying quantization when loading weights: `Sources/ZImage/Weights/WeightsMapping.swift`

## LoRA

LoRA can be loaded from a local path or a Hugging Face repo id. The loader supports:

- Classic LoRA pairs (`lora_down`/`lora_up`, or `lora_A`/`lora_B`)
- LyCORIS LoKr (`lokr_w1`/`lokr_w2`)

Key code:

- CLI flag parsing: `Sources/ZImageCLI/main.swift` (`--lora`, `--lora-scale`)
- Loader + validation: `Sources/ZImage/LoRA/LoRAWeightLoader.swift`
- Application: `Sources/ZImage/LoRA/LoRAApplicator.swift`

## Data Flow (Text-to-Image)

1. Prompt text (and optional negative prompt)
2. Optional prompt enhancement via the Qwen text encoder (LLM mode)
3. Tokenize + encode to prompt embeddings
4. Initialize random latents
5. Denoising loop: transformer → scheduler steps (`FlowMatchEulerScheduler`)
6. Decode latents via VAE
7. Write PNG (`ImageIO`)

## Tests

- Unit tests: `Tests/ZImageTests/` (fast, no model downloads)
- Integration tests: `Tests/ZImageIntegrationTests/` (require weights, slower)
- E2E tests: `Tests/ZImageE2ETests/` (build + run the CLI)

For agent workflows / what to run by default, see `AGENTS.md`.
