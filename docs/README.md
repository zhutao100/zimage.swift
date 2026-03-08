# Documentation

This folder contains the reference docs for **Z-Image.swift** (the Swift + MLX port of `Tongyi-MAI/Z-Image-Turbo`).

If you’re new here, start with the root `README.md` for a runnable quickstart.

## Start Here (Recommended Order)

1. [`docs/CLI.md`](CLI.md) — build + run the CLI, examples, flags, and subcommands.
2. [`docs/MODELS_AND_WEIGHTS.md`](MODELS_AND_WEIGHTS.md) — how model specs, caches, quantization, overrides, and AIO checkpoints work.
3. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — code-level architecture and “where to look”.
4. [`docs/z-image-turbo.md`](z-image-turbo.md), [`docs/z-image.md`](z-image.md)— upstream model layout notes (Diffusers/HF reference).
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

## Current context / investigations

- [`docs/debug_notes/control-context-memory-remediation.md`](debug_notes/control-context-memory-remediation.md) — validated current-state diagnosis for the control-path memory issue, including the March 7, 2026 high-resolution probe.
- [`docs/debug_notes/controlnet-memory-analysis.md`](debug_notes/controlnet-memory-analysis.md) — pruned ControlNet-specific follow-up analysis that keeps only the post-remediation findings still true in the current repo state.
- [`docs/dev_plans/control-context-memory-remediation.md`](dev_plans/control-context-memory-remediation.md) — completed remediation plan and measurement log for phases 1 through 3.
- [`docs/dev_plans/controlnet-memory-followup.md`](dev_plans/controlnet-memory-followup.md) — active follow-up plan, including the completed March 8, 2026 deferred denoiser-load phase and the remaining telemetry work.
- [`docs/context/zimage_runtime_precision_parity_report.md`](context/zimage_runtime_precision_parity_report.md) — confirmed parity, confirmed mismatches, and runtime hypotheses for Swift vs Diffusers precision handling.
- [`docs/dev_plans/runtime_precision_parity_improvement_plan.md`](dev_plans/runtime_precision_parity_improvement_plan.md) — measured execution plan for the first runtime precision parity fixes.
- [`docs/context/mlx_pytorch_bf16_inference_dtype_deep_dive.md`](context/mlx_pytorch_bf16_inference_dtype_deep_dive.md) — backend-level BF16 behavior notes for MLX and PyTorch/MPS.
- [`docs/context/precision_formats_on_apple_silicon.md`](context/precision_formats_on_apple_silicon.md) — broader Apple Silicon precision-format background.

## Archive

Older “point in time” investigation docs live under [`docs/archive/`](archive/README.md) (not required reading, may be stale).
