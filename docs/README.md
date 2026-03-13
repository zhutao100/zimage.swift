# Documentation

This folder is the current reference set for `zimage.swift`. Start with the root [README.md](../README.md) for the shortest runnable path, then use the docs below for the area you are touching.

## Core References

1. [CLI.md](CLI.md)
   Current `ZImageCLI` commands, flags, and usage patterns.
2. [MODELS_AND_WEIGHTS.md](MODELS_AND_WEIGHTS.md)
   Model ids, local-path handling, Hugging Face cache behavior, AIO checkpoints, transformer overrides, and quantization manifests.
3. [ARCHITECTURE.md](ARCHITECTURE.md)
   Runtime layout, entry points, and source-of-truth files.
4. [DEVELOPMENT.md](DEVELOPMENT.md)
   Build, test, CI, packaging, and targeted validation workflows.
5. [dev_plans/ROADMAP.md](dev_plans/ROADMAP.md)
   Short prioritized list of still-open work.

## Read By Task

- CLI work:
  - [CLI.md](CLI.md)
  - `Sources/ZImageCLICommon/`
  - `Sources/ZImageCLI/main.swift`
  - `Sources/ZImageServe/main.swift`
- Model loading, cache lookup, or safetensors behavior:
  - [MODELS_AND_WEIGHTS.md](MODELS_AND_WEIGHTS.md)
  - `Sources/ZImage/Weights/*`
  - `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- Pipeline or general code navigation:
  - [ARCHITECTURE.md](ARCHITECTURE.md)
- Build, CI, packaging, or release work:
  - [DEVELOPMENT.md](DEVELOPMENT.md)
  - `.github/workflows/ci.yml`
  - `scripts/build.sh`
- Precision or parity work:
  - [golden_checks.md](golden_checks.md)
  - [context/zimage_runtime_precision_parity_report.md](context/zimage_runtime_precision_parity_report.md)
- Upstream checkpoint structure:
  - [z-image-turbo.md](z-image-turbo.md)
  - [z-image.md](z-image.md)

## Active Follow-Up Docs

- [dev_plans/controlnet-memory-followup.md](dev_plans/controlnet-memory-followup.md)
  Current control-memory status, retained measurement recipe, and re-entry criteria.
- [dev_plans/staging-cli-service/README.md](dev_plans/staging-cli-service/README.md)
  Active implementation plan for the local staging daemon, warm serving, and batch submission workflows.

## Background And Historical Notes

These files can still be useful, but they are not the primary source of truth for day-to-day behavior:

- [debug_notes/build-workflow-analysis.md](debug_notes/build-workflow-analysis.md)
- [debug_notes/control-context-memory-remediation.md](debug_notes/control-context-memory-remediation.md)
- [debug_notes/controlnet-memory-analysis.md](debug_notes/controlnet-memory-analysis.md)
- [context/mlx_pytorch_bf16_inference_dtype_deep_dive.md](context/mlx_pytorch_bf16_inference_dtype_deep_dive.md)
- [context/precision_formats_on_apple_silicon.md](context/precision_formats_on_apple_silicon.md)
- [archive/dev_plans/runtime_precision_parity_improvement_plan.md](archive/dev_plans/runtime_precision_parity_improvement_plan.md)
  Completed RoPE numeric-staging follow-up, archived after the March 11, 2026 parity pass.

## Archive

Historical investigations and completed implementation plans live under [archive/](archive/README.md). Move finished or superseded material there instead of leaving it mixed with current operating docs.
