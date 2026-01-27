# Documentation

This folder contains the reference docs for **Z-Image.swift** (the Swift + MLX port of `Tongyi-MAI/Z-Image-Turbo`).

If you’re new here, start with the root `README.md` for a runnable quickstart.

## Start Here (Recommended Order)

1. [`docs/CLI.md`](CLI.md) — build + run the CLI, examples, flags, and subcommands.
2. [`docs/MODELS_AND_WEIGHTS.md`](MODELS_AND_WEIGHTS.md) — how model specs, caches, quantization, overrides, and AIO checkpoints work.
3. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — code-level architecture and “where to look”.
4. [`docs/z-image-turbo.md`](z-image-turbo.md) — upstream model layout notes (Diffusers/HF reference).
5. [`docs/dev_plans/ROADMAP.md`](dev_plans/ROADMAP.md) — prioritized next steps.

## “Source Of Truth” Pointers

- CLI behavior + flags: `Sources/ZImageCLI/main.swift`
- Text-to-image pipeline API: `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- ControlNet/inpaint pipeline API: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- Weight download / cache resolution: `Sources/ZImage/Weights/ModelResolution.swift`
- Weight mapping into MLX modules: `Sources/ZImage/Weights/WeightsMapping.swift`, `Sources/ZImage/Weights/ZImageWeightsMapper.swift`
- Quantization: `Sources/ZImage/Quantization/ZImageQuantization.swift`
- LoRA: `Sources/ZImage/LoRA/*`
- Unit tests: `Tests/ZImageTests/*`

## Archive

Older “point in time” investigation docs live under [`docs/archive/`](archive/README.md) (not required reading, may be stale).
