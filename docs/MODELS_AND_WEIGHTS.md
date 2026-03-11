# Models And Weights

This document explains how the current repo resolves model sources, chooses weight files, and handles quantization, AIO checkpoints, and transformer overrides.

## Default And Known Model IDs

- Default CLI model: `Tongyi-MAI/Z-Image-Turbo`
- Also supported in the current codebase:
  - `Tongyi-MAI/Z-Image`
  - `mzbac/z-image-turbo-8bit`
  - `mzbac/Z-Image-Turbo-8bit` as a capitalization-compatible alias

Known ids and per-model presets live in `Sources/ZImage/Support/ZImageModelRegistry.swift`.

## Accepted Model Specs

The text-to-image path accepts:

1. Hugging Face repo id: `org/repo`
2. Hugging Face repo id with revision: `org/repo:revision`
3. Local Diffusers-style model directory containing the expected configs, tokenizer files, and weights
4. Local `.safetensors`
   - AIO checkpoint if it contains transformer, text-encoder, and VAE tensors with recognized prefixes
   - transformer-only override otherwise

Resolution behavior comes from:

- `Sources/ZImage/Weights/ModelResolution.swift`
- `Sources/ZImage/Pipeline/PipelineSnapshot.swift`
- `Sources/ZImage/Pipeline/ZImagePipeline.swift`

The control pipeline currently uses the standard snapshot resolver and expects a normal local directory or Hugging Face snapshot for `request.model`. It does not expose the text-to-image AIO / transformer-only `.safetensors` path.

## Resolution Order

### Text-To-Image Resolver

When a text-to-image model spec is provided:

1. If it is an existing local path, the repo uses that path directly.
   - Local Diffusers-style directory: use it as-is.
   - Local `.safetensors`: inspect for AIO coverage first; treat it as a transformer-only override otherwise.
   - Local directory without the expected configs but with `.safetensors`: pick a preferred file from the directory, favoring filenames that contain `v2`, otherwise the largest `.safetensors`, then inspect that file as a local checkpoint.
   - Local directory without the expected configs and without any `.safetensors`: the text-to-image pipeline fails with an explicit local-path error instead of falling back to the default model.
2. Otherwise, if it looks like `org/repo` or `org/repo:revision`, the repo checks the Hugging Face cache.
3. If no matching snapshot is cached, it downloads the required files into the Hugging Face cache and then loads from that snapshot.

When no model spec is provided, the same logic is applied to the default model id and revision.

### Control Resolver

`ZImageControlPipeline` goes through `PipelineSnapshot.prepare(...)` and expects a regular snapshot layout. In practice that means:

- Hugging Face repo id, optionally with `:revision`
- local Diffusers-style model directory

If you need text-to-image-style AIO or transformer-only override behavior, that currently exists only on the text-to-image path.

## Hugging Face Cache And Environment Variables

The resolver uses the standard Hugging Face cache layout. The practical controls are:

- `HF_HUB_CACHE`: direct hub-cache override
- `HF_HOME`: uses `<HF_HOME>/hub`
- `HF_ENDPOINT`: alternate Hugging Face host

If neither cache variable is set, the default cache root is:

```text
~/.cache/huggingface/hub
```

The repo first checks the normal `models--ORG--REPO/snapshots/<commit>/` layout and then the `swift-transformers` local cache layout under `<cache>/models/ORG/REPO/`.

Source of truth:

- `Sources/ZImage/Weights/ModelResolution.swift`
- `Sources/ZImage/Weights/HuggingFaceHub.swift`

## Authentication

The repo relies on the Hugging Face client libraries' environment-based authentication. In practice, `HF_TOKEN` is the most direct user-facing way to authenticate for gated or private repos.

If a repo requires auth and loading fails, the error now points directly at `HF_TOKEN` and local-snapshot fallback. The two supported fallback paths are:

- authenticate and rerun
- download the weights locally and point `--model` or `--controlnet-weights` at the local path

## Weights Variants (`--weights-variant`)

Some repos ship multiple precision-specific weight sets such as `fp16` or `bf16`.

When `--weights-variant <name>` is set:

- resolution prefers `*.{name}.safetensors.index.json` where present
- directory scans avoid mixing shards from different variants
- download patterns are narrowed to the requested variant plus required JSON and tokenizer files
- loading fails if transformer, text encoder, or VAE weights are missing for that variant

This behavior is implemented in:

- `Sources/ZImage/Weights/ModelPaths.swift`
- `Sources/ZImage/Weights/ZImageWeightsMapper.swift`
- `Sources/ZImage/Pipeline/PipelineSnapshot.swift`

Current nuance: `weightsVariant` selects which files are loaded. It is not a global runtime compute-dtype switch.

If the requested variant is incomplete, the error names the missing components and suggests removing `--weights-variant` unless transformer, text encoder, and VAE all ship matching files.

## Quantized Models

Quantized models are regular model directories plus a manifest:

- `quantization.json` for base-model quantization
- `controlnet_quantization.json` for ControlNet quantization

At load time, the pipelines detect the manifest and wrap the mapped tensors with quantized MLX modules.

Source of truth:

- `Sources/ZImage/Quantization/ZImageQuantization.swift`
- `Sources/ZImage/Weights/WeightsMapping.swift`

## AIO Checkpoints

A local `.safetensors` can be treated as an all-in-one checkpoint if it contains the expected component prefixes:

- transformer tensors under `model.diffusion_model.*` or `diffusion_model.*`
- text-encoder tensors under a recognized `text_encoders.*.transformer.model.*` prefix
- VAE tensors under `vae.*`

If the file is detected as AIO:

- base-model snapshot loading is bypassed
- the transformer and text encoder are loaded from the single file
- the VAE decoder weights may be canonicalized from ComfyUI-style naming when needed
- if AIO VAE coverage is insufficient, the pipeline falls back to the base VAE weights

Source of truth:

- `Sources/ZImage/Weights/AIOCheckpoint.swift`
- `Sources/ZImage/Pipeline/ZImagePipeline.swift`

## Transformer Overrides

If a local `.safetensors` is not detected as AIO, it is treated as a transformer-only override:

- the base snapshot is still resolved and loaded
- override tensors are canonicalized and applied on top of the base transformer

To skip AIO auto-detection and force that behavior, use:

- CLI: `--force-transformer-override-only`

This override path is currently implemented on the text-to-image pipeline only.

The relevant code is in `Sources/ZImage/Pipeline/ZImagePipeline.swift`.

## ControlNet Weights

`ZImageCLI control --controlnet-weights` accepts:

- a local `.safetensors`
- a local directory
- a Hugging Face repo id

`--control-file` can be used to choose a specific `.safetensors` file when the source contains more than one.

Quantized ControlNet directories use `controlnet_quantization.json`.

Source of truth:

- `Sources/ZImageCLI/main.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Quantization/ZImageQuantization.swift`

## Troubleshooting

### Model Not Found

- If you passed a local path, verify the path exists, expands correctly from `~`, and points to a supported local model form.
  Supported text-to-image forms are a Diffusers-style directory or a local `.safetensors`.
- If you passed `org/repo`, verify the repo exists and your network is available.
- If the repo is gated or private, set `HF_TOKEN` and retry or use a local download.

### Wrong Results From `Tongyi-MAI/Z-Image`

For the built-in `Tongyi-MAI/Z-Image` id, the CLI applies Base-friendly defaults (`50` steps, guidance `4.0`).

The CLI now inspects local or cached snapshot metadata, nearby snapshot directories for local checkpoint files, and common Z-Image-style aliases before falling back to Turbo-compatible defaults.
Completely unrecognized models can still need explicit `--steps` and `--guidance` if you do not want the Turbo-compatible preset.

For repo-side regression checking against the real Base checkpoint, there is also an opt-in Base smoke test in `Tests/ZImageIntegrationTests/PipelineIntegrationTests.swift`; see [DEVELOPMENT.md](DEVELOPMENT.md) for the invocation.

### Control Model Path Looks Valid But Fails

The control pipeline expects a standard snapshot or local directory. If you pass a local text-to-image `.safetensors` file that works on the base path, the control path still cannot use that as a model override today.

### Offline Reuse

Once a snapshot has been downloaded into the Hugging Face cache, reruns can reuse it without downloading again.
