# Architecture

This document maps the current implementation to the source files that define it. For runnable commands, start with the root `README.md` and [CLI.md](CLI.md).

## Overview

`zimage.swift` is a native Swift + MLX port of the `Tongyi-MAI/Z-Image` family. It provides:

- `ZImage`: library target
- `ZImageCLI`: macOS CLI wrapper

The package is defined in `Package.swift`. There is no checked-in Xcode project or workspace; Xcode builds operate through the Swift package.

The major runtime pieces are:

1. tokenizer and Qwen text encoder
2. diffusion transformer and Flow Match scheduler
3. VAE encode/decode path
4. weight resolution and safetensors mapping
5. optional LoRA and quantization layers

## High-Level File Map

- `Package.swift`
  - package graph, platforms, products, and target boundaries
- `Sources/ZImageCLI/main.swift`
  - CLI argument parsing, subcommands, progress reporting, and user-facing help text
- `Sources/ZImage/Pipeline/`
  - `ZImagePipeline.swift`: text-to-image pipeline
  - `ZImageControlPipeline.swift`: ControlNet and inpainting pipeline
  - `FlowMatchScheduler.swift`: scheduler implementation
  - `PipelineSnapshot.swift`: snapshot download and file-pattern helpers
  - `PipelineUtilities.swift`: shared prompt-encoding, CFG, dtype, and snapshot helpers
- `Sources/ZImage/Model/`
  - `TextEncoder/`: Qwen tokenizer-facing encoder and optional generation path for prompt enhancement
  - `Transformer/`: base and control transformer blocks
  - `VAE/`: encode/decode implementation
- `Sources/ZImage/Weights/`
  - Hugging Face resolution, local-path handling, safetensors reader, AIO detection, tensor mapping
- `Sources/ZImage/Quantization/`
  - quantization manifest format and quantize commands
- `Sources/ZImage/LoRA/`
  - LoRA/LoKr parsing, mapping, and application
- `Sources/ZImage/Support/`
  - model metadata and known-model registry
- `Sources/ZImage/Util/`
  - image I/O and control-memory telemetry

## Public Library Surface

The library is pipeline-first:

- `ZImageGenerationRequest` + `ZImagePipeline`
- `ZImageControlGenerationRequest` + `ZImageControlPipeline`

Both request types expose CFG truncation and normalization controls in addition to the base guidance scale.

Current asymmetry to know about:

- the control request type already has LoRA and prompt-enhancement fields
- `ZImageCLI control` does not expose those flags yet

## Model Selection And Snapshot Loading

The text-to-image runtime treats model specs in four forms:

- default repo id from `ZImageRepository.id`
- Hugging Face repo id, optionally with `:revision`
- local Diffusers-style directory
- local `.safetensors`
  - AIO checkpoint when it contains all expected components
  - transformer-only override otherwise

The control pipeline uses the standard snapshot resolver and expects a regular snapshot or local directory. It does not currently expose the text-to-image AIO / transformer-only override path.

The loading path is split across:

- `Sources/ZImage/Weights/ModelResolution.swift`
- `Sources/ZImage/Pipeline/PipelineSnapshot.swift`
- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Weights/AIOCheckpoint.swift`

Known model ids and per-model presets are centralized in `Sources/ZImage/Support/ZImageModelRegistry.swift`. The CLI applies those presets only to fields the user did not set. Current nuance: preset lookup covers known ids, inspectable local or cached snapshots, and common Z-Image-style aliases. Completely unrecognized models still fall back to the Turbo-compatible preset unless the caller overrides the relevant fields explicitly.

## Weight Mapping

Once a snapshot or local source is resolved, config JSONs are loaded through `ZImageModelConfigs` and tensor files are mapped into MLX modules through:

- `Sources/ZImage/Weights/ZImageWeightsMapper.swift`
- `Sources/ZImage/Weights/WeightsMapping.swift`
- `Sources/ZImage/Weights/ZImageControlWeightsMapping.swift`
- `Sources/ZImage/Weights/ModuleWeightsApplier.swift`

Weights-variant handling is part of the mapping layer, not just the downloader. Resolver and mapper code coordinate to avoid mixed `fp16`/`bf16` shard loads.

## Text-To-Image Flow

`ZImagePipeline.generate(...)` currently does this:

1. resolve model source and load configs
2. load tokenizer, text encoder, transformer, and VAE decoder as needed
3. optionally load or swap LoRA
4. optionally enhance the prompt through the Qwen generation path
5. tokenize and encode prompt and optional negative prompt
6. run the denoising loop through the diffusion transformer and scheduler
7. decode final latents through the VAE decoder
8. write PNG output or return encoded bytes

Source of truth: `Sources/ZImage/Pipeline/ZImagePipeline.swift`

## ControlNet And Inpainting Flow

`ZImageControlPipeline.generate(...)` adds:

- control-image encoding
- optional inpaint-image and mask encoding
- separate ControlNet weight loading
- control-context construction before denoising
- cached prompt embeddings across repeated runs

The CLI requires `--controlnet-weights` plus at least one of `--control-image`, `--inpaint-image`, or `--mask`.

Source of truth: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

## Current Control-Memory Policy

The control pipeline intentionally narrows module residency while building control context:

1. prompt embeddings are produced and cached
2. transformer, ControlNet, and active LoRA state stay absent until denoising is about to start
3. an encoder-only VAE is loaded for control or inpaint encoding and unloaded again after the typed control-context tensor is materialized
4. MLX cache is cleared before denoiser modules are loaded
5. a decoder-only VAE is loaded only for final decode

Supporting implementation points:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Util/ControlMemoryTelemetry.swift`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`

The supported runtime probe for that path is `ZImageCLI control --log-control-memory`.

## Quantization And LoRA

Quantization is manifest-driven:

- `quantization.json` for base-model directories
- `controlnet_quantization.json` for ControlNet directories

Relevant code:

- `Sources/ZImage/Quantization/ZImageQuantization.swift`
- `Sources/ZImage/Weights/WeightsMapping.swift`

LoRA support is split into three stages:

1. source resolution and validation in `Sources/ZImage/LoRA/LoRAWeightLoader.swift`
2. key remapping in `Sources/ZImage/LoRA/LoRAKeyMapper.swift`
3. application or removal in `Sources/ZImage/LoRA/LoRAApplicator.swift`

## Tests

- `Tests/ZImageTests/`: default fast suite
- `Tests/ZImageIntegrationTests/`: slower, weight-dependent tests gated by `ZIMAGE_RUN_INTEGRATION_TESTS=1`
- `Tests/ZImageE2ETests/`: CLI build-and-run tests gated by `ZIMAGE_RUN_E2E_TESTS=1`

If you need to understand intended behavior before touching code, inspect the matching unit tests first, especially:

- `Tests/ZImageTests/Weights/*`
- `Tests/ZImageTests/Support/*`
- `Tests/ZImageTests/Quantization/*`
- `Tests/ZImageTests/Scheduler/*`
