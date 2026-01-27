# Z-Image.swift

Swift + MLX implementation of [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) for Apple Silicon.

Ships as:

- `ZImage` (library)
- `ZImageCLI` (macOS CLI)

**Try it with an easy UI:** [Lingdong Desktop App](https://lingdong.app/en)

## Features

- Text-to-image generation (Flow Matching scheduler)
- ControlNet conditioning + inpainting (`ZImageCLI control`)
- LoRA / LoKr adapters (`--lora`)
- Quantization (4-bit / 8-bit) for base model and ControlNet (`ZImageCLI quantize*`)
- Model loading from Hugging Face (cached) or local paths
- Optional “prompt enhancement” using the Qwen text encoder in LLM mode (`--enhance`)

## Requirements

- Apple Silicon
- macOS 14.0+ (CLI) / iOS 16+ (library target)
- Swift 5.9+ (CI uses Xcode 16.0)

## Quickstart (CLI)

Build a release binary:

```bash
./build.sh
```

Or equivalently:

```bash
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

Run from the build products directory:

```bash
cd .build/xcode/Build/Products/Release
./ZImageCLI -p "A beautiful mountain landscape at sunset" -o output.png
```

Show help / subcommands:

```bash
./ZImageCLI --help
./ZImageCLI control --help
```

## Configuration

Common knobs:

- `--model/-m`: Hugging Face id (`org/repo[:revision]`), local model directory, or local `.safetensors`
- `--cache-limit`: limit MLX GPU cache (MB)
- `--max-sequence-length`: prompt encoding length (default 512)

Hugging Face cache location (used when downloading weights):

- `HF_HUB_CACHE` (direct override), or
- `HF_HOME` (uses `<HF_HOME>/hub`), or
- default `~/.cache/huggingface/hub`

See `docs/MODELS_AND_WEIGHTS.md` for the full model-resolution behavior (AIO checkpoints, transformer overrides, quantization manifests, etc.).

## Examples

Assuming you’re running from `.build/xcode/Build/Products/Release` (see Quickstart):

```bash
# Basic generation
./ZImageCLI -p "a cute cat sitting on a windowsill" -o cat.png

# Portrait image with custom size
./ZImageCLI -p "portrait of a woman in renaissance style" -W 768 -H 1152 -o portrait.png

# Using a quantized model (example HF repo id)
./ZImageCLI -p "a futuristic city at night" -m mzbac/Z-Image-Turbo-8bit -o city.png

# With memory limit
./ZImageCLI -p "abstract art" --cache-limit 2048 -o art.png

# With LoRA
./ZImageCLI -p "a lion" --lora ostris/z_image_turbo_childrens_drawings -o lion.png
```

## LoRA

Apply LoRA weights for style customization:

```bash
./ZImageCLI -p "a lion" --lora ostris/z_image_turbo_childrens_drawings --lora-scale 1.0 -o lion.png
```

### LoRA Example

<table width="100%">
<tr>
<th>Prompt</th>
<th>LoRA</th>
<th>Output</th>
</tr>
<tr>
<td>a lion</td>
<td><a href="https://huggingface.co/ostris/z_image_turbo_childrens_drawings">ostris/z_image_turbo_childrens_drawings</a></td>
<td><img src="examples/lora_lion.png" height="256"></td>
</tr>
</table>

## ControlNet

Generate images with ControlNet conditioning using Canny, HED, Depth, Pose, or MLSD control images:

```bash
./ZImageCLI control \
  --prompt "A hyper-realistic close-up portrait of a leopard" \
  --control-image /path/to/canny_edges.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --control-scale 0.75 \
  --output leopard.png
```

For all options, run:

```bash
./ZImageCLI control --help
```

### ControlNet Examples

| Control Type | Prompt | Control Image | Output |
|--------------|--------|---------------|--------|
| Canny | A hyper-realistic close-up portrait of a leopard face hiding behind dense green jungle leaves, camouflaged, direct eye contact, intricate fur detail, bright yellow eyes, cinematic lighting, soft shadows, National Geographic photography, 8k, sharp focus, depth of field | ![Canny](images/canny.jpg) | ![Canny Output](examples/canny.png) |
| HED | A photorealistic film still of a man in a dark shirt sitting at a dining table in a modern kitchen at night, looking down at a bowl of soup. A glass bottle and a glass of white wine are in the foreground. Warm, low, cinematic lighting, soft shadows, shallow depth of field, contemplative atmosphere, highly detailed. | ![HED](images/hed.jpg) | ![HED Output](examples/hed.png) |
| Depth | A hyperrealistic architectural photograph of a spacious, minimalist modern hallway interior. Large floor-to-ceiling windows on the right wall fill the space with bright natural daylight. A light gray sectional sofa and a low, modern coffee table are placed in the foreground on a light wood floor. A large potted plant is visible further down the hallway. White walls, clean lines, serene atmosphere, highly detailed, 8k resolution, cinematic lighting | ![Depth](images/depth.jpg) | ![Depth Output](examples/depth.png) |
| Pose | 一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动... | ![Pose](images/pose.jpg) | ![Pose Output](examples/pose.png) |

## Example Text To Image Output

| Prompt | Output |
|--------|--------|
| A dramatic, cinematic japanese-action scene in a edo era Kyoto city. A woman named Harley Quinn from the movie "Birds of Prey" in colorful, punk-inspired comic-villain attire walks confidently while holding the arm of a serious-looking man named John Wick played by Keanu Reeves from the fantastic film John Wick 2 in a black suit, her t-shirt says "Birds of Prey", the characters are capture in a postcard held by a hand in front of a beautiful realistic city at sunset and there is cursive writing that says "ZImage, Now in MLX" | ![Output](examples/z-image.png) |

## Quantization

Quantize the model to reduce memory usage:

```bash
./ZImageCLI quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --bits 8 --group-size 32 --verbose
```

ControlNet quantization is available via `./ZImageCLI quantize-controlnet ...` (see `docs/CLI.md`).

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) - MLX bindings for Swift
- [swift-transformers](https://github.com/huggingface/swift-transformers) - tokenizers + Hugging Face Hub access
- [swift-log](https://github.com/apple/swift-log) - logging

## Limitations / Known Gaps

- The CLI target is macOS-only (the package also declares an iOS library target).
- First run may download many GB of weights; high resolutions can be memory-heavy on unified memory systems.
- Hugging Face gated/private repos require authentication (e.g. `HF_TOKEN`). If downloads fail, download locally and point `--model` at a local path.

## Docs

- [`docs/README.md`](docs/README.md) — docs index / “start here”
- [`docs/CLI.md`](docs/CLI.md) — CLI usage and subcommands
- [`docs/MODELS_AND_WEIGHTS.md`](docs/MODELS_AND_WEIGHTS.md) — model specs, caches, AIO, overrides, quantization
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — code architecture and source-of-truth pointers
- [`docs/dev_plans/ROADMAP.md`](docs/dev_plans/ROADMAP.md) — prioritized next steps

## Next Steps

See `docs/dev_plans/ROADMAP.md`.

## Documentation Changelog

- Added: `docs/README.md`, `docs/CLI.md`, `docs/MODELS_AND_WEIGHTS.md`, `docs/DEVELOPMENT.md`, `docs/dev_plans/ROADMAP.md`.
- Restructured: moved architecture doc to `docs/ARCHITECTURE.md`, moved `docs/CODE_QUALITY_REPORT.md` into `docs/archive/`.
- Removed/updated: outdated README dependency list and CLI flag details; README now links to `docs/` as the source of truth.

## License

MIT License
