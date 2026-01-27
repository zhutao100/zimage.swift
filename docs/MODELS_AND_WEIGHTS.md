# Models & Weights

This doc explains how **Z-Image.swift** resolves model specs, where weights are cached, and how quantization / overrides / AIO checkpoints work.

## Model Spec Forms

The pipelines accept `model` as either:

1. Hugging Face model id: `org/repo`
   - Optional revision syntax: `org/repo:revision`
2. A local directory containing a Diffusers-style layout:
   - `transformer/`, `text_encoder/`, `vae/`, `tokenizer/`, plus JSON configs
3. A local `.safetensors` file:
   - Interpreted as an **AIO checkpoint** if it contains transformer + text encoder + VAE tensors with recognized key prefixes
   - Otherwise interpreted as a **transformer-only override**

The CLI flag is `--model/-m` (see `Sources/ZImageCLI/main.swift`).

## Hugging Face Cache

When downloading from Hugging Face, the resolver looks for an existing snapshot first, and otherwise downloads into the standard HF cache layout.

Environment variables recognized by the resolver:

- `HF_HUB_CACHE`: overrides the hub cache directory directly
- `HF_HOME`: used as `<HF_HOME>/hub`

If neither is set, the default is:

- `~/.cache/huggingface/hub`

Source of truth: `Sources/ZImage/Weights/ModelResolution.swift` (`getHuggingFaceCacheDirectory()`).

## Authentication (Gated / Private Repos)

The underlying Hugging Face client (`HubApi` from `swift-transformers`) supports auth tokens via common conventions, including:

- `HF_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`
- `HF_TOKEN_PATH` (file containing a token)
- `HF_HOME/token`
- `~/.cache/huggingface/token` or `~/.huggingface/token`

If you hit an “authorization required” error, set `HF_TOKEN` (or one of the above) and retry.

## Quantized Models

Quantized models are stored as a directory containing the usual model files plus a manifest:

- `quantization.json` (base model)
- `controlnet_quantization.json` (ControlNet)

At load time, the pipeline detects these manifests and applies quantization transforms when mapping tensors into MLX modules.

Sources of truth:

- `Sources/ZImage/Quantization/ZImageQuantization.swift`
- `Sources/ZImage/Weights/WeightsMapping.swift`

## AIO Checkpoints

An “AIO checkpoint” is a single `.safetensors` file containing all components (transformer + text encoder + VAE).

The pipeline performs a lightweight header inspection to decide whether a `.safetensors` file is AIO. Expected tensor prefixes include:

- Transformer: `model.diffusion_model.*` (or `diffusion_model.*`)
- Text encoder: `text_encoders.<name>.transformer.model.*`
- VAE: `vae.*` (and a recognizable decoder layout)

If a file is detected as AIO:

- The pipeline bypasses downloading/loading base model weights.
- The AIO VAE decoder may be canonicalized from ComfyUI naming to Diffusers naming (when needed).

Source of truth: `Sources/ZImage/Weights/AIOCheckpoint.swift` and AIO handling in `Sources/ZImage/Pipeline/ZImagePipeline.swift`.

## Transformer Overrides (`.safetensors`)

If a local `.safetensors` file is **not** detected as AIO, the pipeline treats it as a transformer-only override:

- Base model weights are loaded as usual.
- Then the override tensors are canonicalized and applied on top of the base transformer weights.

If you want to force a `.safetensors` to be treated as transformer-only (skipping AIO detection), use:

- CLI: `--force-transformer-override-only`

Source of truth: `Sources/ZImage/Pipeline/ZImagePipeline.swift` (`resolveModelSelection`, `applyTransformerOverrideIfNeeded`).

## Troubleshooting

### “Model not found”

- If you passed a local path, ensure it exists and is readable.
- If you passed `org/repo`, ensure you have a working network connection and the repo is public.
- For gated/private repos, authenticate (see above) or download weights locally and point `--model` at a local directory.

### “No internet connection”

Pre-download the snapshot (run once with network), then rerun offline. The resolver will reuse an existing snapshot if present.
