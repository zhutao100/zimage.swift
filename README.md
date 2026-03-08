# Z-Image.swift

Native Swift + MLX implementation of the `Tongyi-MAI/Z-Image` family for Apple Silicon.

The repo ships two products:

- `ZImage`: a Swift library for macOS and iOS targets
- `ZImageCLI`: a macOS command-line interface

## What It Supports

- Text-to-image generation with the Z-Image diffusion transformer and Flow Match scheduler
- ControlNet conditioning and inpainting via `ZImageCLI control`
- LoRA and LoKr adapters on the text-to-image CLI and in the library pipelines
- 4-bit and 8-bit quantization for base-model and ControlNet weights
- Hugging Face snapshots, local Diffusers-style model folders, local AIO `.safetensors`, and transformer-only overrides
- Optional prompt enhancement on the text-to-image CLI via the Qwen text encoder's generation path

The default CLI model is `Tongyi-MAI/Z-Image-Turbo`.

## Quickstart

### Prerequisites

- Apple Silicon Mac
- macOS 14.0+
- Xcode 16.x or another Swift 6 toolchain that can build the package
- Network access for the first run unless you already have the weights locally

### Build

The shortest path is the repo script:

```bash
./scripts/build.sh
```

Equivalent explicit command:

```bash
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

### Run

```bash
cd .build/xcode/Build/Products/Release
./ZImageCLI --help
./ZImageCLI -p "a studio photo of a red apple on black velvet" -o output.png
```

First run will download the default model snapshot into the Hugging Face cache.

### Minimal Examples

Turbo defaults:

```bash
./ZImageCLI -p "a neon-lit alley in the rain" -o turbo.png
```

Base model:

```bash
./ZImageCLI \
  -m Tongyi-MAI/Z-Image \
  -p "a black tiger in a bamboo forest" \
  -o base.png
```

ControlNet:

```bash
./ZImageCLI control \
  --prompt "a dancer on a stage" \
  --control-image /path/to/pose.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --output control.png
```

## Configuration

Common CLI settings:

- `--model/-m`: Hugging Face repo id, local model directory, or local `.safetensors`
- known `Tongyi-MAI` model ids apply model-aware presets:
  - Turbo: `1024x1024`, `9` steps, guidance `0.0`
  - Base: `1024x1024`, `50` steps, guidance `4.0`
- `--cfg-normalization`: clamp CFG output norm back to the positive-branch norm
- `--cfg-truncation`: turn CFG off after the normalized denoising timestep passes the given value
- `--weights-variant`: precision-specific weights selection such as `fp16` or `bf16`
- `--cache-limit`: MLX GPU cache limit in MB
- `--max-sequence-length`: prompt token limit for text encoding
- `--force-transformer-override-only`: treat a local `.safetensors` as a transformer override and skip AIO auto-detection

Environment variables:

- `HF_HUB_CACHE` or `HF_HOME`: override the Hugging Face cache location
- `HF_TOKEN`: practical choice for gated or private Hugging Face repos
- `HF_ENDPOINT`: override the Hugging Face API host

The authoritative details for model resolution, cache lookup, AIO checkpoints, quantization manifests, and ControlNet weight loading live in [docs/MODELS_AND_WEIGHTS.md](docs/MODELS_AND_WEIGHTS.md).

## Current Limitations

- Model-aware CLI defaults currently key off the known `Tongyi-MAI` ids. Local paths and unknown aliases still fall back to Turbo-compatible defaults unless you set `--steps` and `--guidance` explicitly.
- `ZImageCLI control` exposes control, inpainting, and `--log-control-memory`, but it does not currently expose the control-pipeline LoRA and prompt-enhancement hooks that exist in the library request type.
- First-time downloads are large, and high-resolution runs can still be memory-heavy on unified-memory systems.
- The CLI target is macOS-only. The package also declares an iOS library target, but there is no first-party sample app in this repo.

## Docs

- [docs/README.md](docs/README.md): docs index and task-based reading order
- [docs/CLI.md](docs/CLI.md): CLI usage, flags, and examples
- [docs/MODELS_AND_WEIGHTS.md](docs/MODELS_AND_WEIGHTS.md): model selection, caching, AIO checkpoints, overrides, quantization
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): code structure and source-of-truth files
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): build, test, lint, CI, and validation workflows
- [docs/dev_plans/ROADMAP.md](docs/dev_plans/ROADMAP.md): prioritized next steps

## Next Steps

The current roadmap is kept in [docs/dev_plans/ROADMAP.md](docs/dev_plans/ROADMAP.md).

## License

MIT License
