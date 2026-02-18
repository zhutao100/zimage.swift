# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Z-Image.swift is a Swift port of [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) using mlx-swift for Apple Silicon. It provides a CLI tool and library for text-to-image generation with support for LoRA and ControlNet.

## Build Commands

```bash
# Build release CLI binary
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode

# Run all tests (use -enableCodeCoverage NO to avoid creating default.profraw)
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO

# Run specific test target
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests

# Run a single test class
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/FlowMatchSchedulerTests

# Run a single test method
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests/FlowMatchSchedulerTests/testTimestepsDecreasing
```

## Architecture

### Core Components

**Application Layer** (`Sources/ZImageCLI`):
- `ZImageCLI`: Entry point for generation, controlnet, and quantization commands. Handles argument parsing and global GPU settings.

**Pipeline Layer** (`Sources/ZImage/Pipeline`):
- `ZImagePipeline`: Orchestrates Text-to-Image generation. Manages dynamic model loading/unloading (phase-scoped lifetimes), LoRA application, and the denoising loop.
- `ZImageControlPipeline`: Extends generation with ControlNet conditioning (image/mask inputs) and Inpainting.
- `FlowMatchScheduler`: Implements Flow Matching Euler scheduler with "Dynamic Shifting" for resolution-dependent noise schedules.

**Model Layer** (`Sources/ZImage/Model`):
- **TextEncoder**: Qwen-based transformer. Acts as both an Encoder (for conditioning) and a Generator (for Prompt Enhancement via LLM).
- **Transformer**: The Diffusion Transformer (DiT).
  - `ZImageTransformer2DModel`: Hybrid architecture with separate "refiner" streams (Noise & Context) that merge into joint attention blocks.
  - `ZImageControlTransformer2DModel`: Adds ControlNet blocks to the base DiT architecture.
- **VAE**: `AutoencoderKL` for encoding/decoding images to/from latents. Supports `AutoencoderDecoderOnly` for inference optimization.

**Infrastructure** (`Sources/ZImage/Weights`, `/Quantization`, `/LoRA`):
- **Weights**: Handles downloading from Hugging Face (`HubSnapshot`), parsing `.safetensors`, and detecting AIO checkpoints.
- **Quantization**: `ZImageQuantizer` supports 4-bit and 8-bit group-wise quantization (Affine/MXFP4) for reduced memory footprint.
- **LoRA**: `LoRAApplicator` supports both baked-in and dynamic (runtime) adapters, including LoKr (Kronecker product) and quantization compatibility.

### Key Data Flow

1. **Enhancement (Optional)**: Text prompt → QwenTextEncoder (LLM Mode) → Enhanced Prompt.
2. **Encoding**: Enhanced Prompt → QwenTokenizer → QwenTextEncoder → Text Embeddings.
3. **Initialization**: Random Gaussian Latents generated.
4. **Denoising Loop**:
   - Latents + Timestep + Text Embeddings → `ZImageTransformer2D` (Refiners → Joint Blocks) → Noise Prediction.
   - `FlowMatchEulerScheduler` updates latents based on prediction.
5. **Decoding**: Refined Latents → `AutoencoderKL` (Decoder) → RGB Image.

### Test Structure

- `Tests/ZImageTests/` - Unit tests (scheduler, config parsing, image I/O).
- `Tests/ZImageIntegrationTests/` - Integration tests requiring model weights (pipeline, ControlNet, LoRA).
- `Tests/ZImageE2ETests/` - End-to-end CLI tests (builds and runs the actual binary).

## Requirements

- macOS 14.0+ / iOS 16+
- Apple Silicon
- Swift 6.0+
