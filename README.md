# Z-Image.swift

> An enhanced fork of the `mzbac/zimage.swift` project.

Native Swift + MLX implementation of the `Tongyi-MAI/Z-Image` and `Tongyi-MAI/Z-Image-Turbo` model family for Apple Silicon.

The repo ships two products:

- `ZImage`: a Swift library for macOS and iOS targets
- `ZImageCLI`: a macOS command-line interface

## What It Supports

- Text-to-image generation with the Z-Image diffusion transformer and Flow Match scheduler
- ControlNet conditioning and inpainting via `ZImageCLI control`
- LoRA and LoKr adapters on the text-to-image CLI and in the library pipelines
- 4-bit and 8-bit quantization for the Turbo model and ControlNet weights
- Hugging Face snapshots, local Diffusers-style model folders, local AIO `.safetensors`, and transformer-only overrides
- Optional prompt enhancement on the text-to-image CLI via the Qwen text encoder's generation path

The default CLI model is `Tongyi-MAI/Z-Image-Turbo`.

## Examples

### Text To Image Examples

> Z-Image-Turbo

| Prompt | Output |
|--------|--------|
| A dramatic, cinematic japanese-action scene in a edo era Kyoto city. A woman named Harley Quinn from the movie "Birds of Prey" in colorful, punk-inspired comic-villain attire walks confidently while holding the arm of a serious-looking man named John Wick played by Keanu Reeves from the fantastic film John Wick 2 in a black suit, her t-shirt says "Birds of Prey", the characters are capture in a postcard held by a hand in front of a beautiful realistic city at sunset and there is cursive writing that says "Z-Image-Turbo, Now in MLX" | ![Output](examples/z-image-turbo.png) |


> Z-Image (Base)

| Prompt | Output |
|--------|--------|
| A dramatic, cinematic japanese-action scene in a edo era Kyoto city. A woman named Harley Quinn from the movie "Birds of Prey" in colorful, punk-inspired comic-villain attire walks confidently while holding the arm of a serious-looking man named John Wick played by Keanu Reeves from the fantastic film John Wick 2 in a black suit, her t-shirt says "Birds of Prey", the characters are capture in a postcard held by a hand in front of a beautiful realistic city at sunset and there is cursive writing that says "Z-Image, Now in MLX" | ![Output](examples/z-image.png) |

Note:
- generated with `--negative-prompt "卡通,油画质感,低分辨率,塑料材质,光滑"`

### ControlNet Examples (Z-Image-Turbo)

| Control Type | Prompt | Control Image | Output |
|--------------|--------|---------------|--------|
| Canny | A hyper-realistic close-up portrait of a leopard face hiding behind dense green jungle leaves, camouflaged, direct eye contact, intricate fur detail, bright yellow eyes, cinematic lighting, soft shadows, National Geographic photography, 8k, sharp focus, depth of field | ![Canny](images/canny.jpg) | ![Canny Output](examples/canny.png) |
| HED | A photorealistic film still of a man in a dark shirt sitting at a dining table in a modern kitchen at night, looking down at a bowl of soup. A glass bottle and a glass of white wine are in the foreground. Warm, low, cinematic lighting, soft shadows, shallow depth of field, contemplative atmosphere, highly detailed. | ![HED](images/hed.jpg) | ![HED Output](examples/hed.png) |
| Depth | A hyperrealistic architectural photograph of a spacious, minimalist modern hallway interior. Large floor-to-ceiling windows on the right wall fill the space with bright natural daylight. A light gray sectional sofa and a low, modern coffee table are placed in the foreground on a light wood floor. A large potted plant is visible further down the hallway. Besides the plant, the hallway extends into the darkness, suggesting further space. White walls, clean lines, serene atmosphere, highly detailed, 8k resolution, cinematic lighting | ![Depth](images/depth.jpg) | ![Depth Output](examples/depth.png) |
| Pose | 一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动... | ![Pose](images/pose.jpg) | ![Pose Output](examples/pose.png) |

Note:
- generated with `--negative-prompt "卡通,油画质感,低分辨率,塑料材质,光滑"` and `--control-scale 0.75`
- ControlNet weights: `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`
- Control file: `Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors`

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

For repo-side regression checking against the real Base checkpoint, use the opt-in smoke test documented in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

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

## Acknowledgements

- The original `mzbac/zimage.swift` repo for the initial implementation and reference point
- The `Tongyi-MAI/Z-Image` and `Tongyi-MAI/Z-Image-Turbo` teams for the models, weights, and reference outputs that made this possible
- The MLX team for the Swift bindings and MLX improvements that enabled the implementation and optimizations in this repo.

## License

MIT License
