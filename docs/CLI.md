# CLI Guide (`ZImageCLI`)

This project ships a macOS CLI executable target: `ZImageCLI`.

## Build

Build a release binary with Xcode:

```bash
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

Tip: if Xcode prompts about a package plugin (MLX metal shader preparation), allow it. In CI/non-interactive builds, see `docs/DEVELOPMENT.md`.

Run the binary from the build products directory:

```bash
cd .build/xcode/Build/Products/Release
./ZImageCLI --help
```

## Text-to-Image

```bash
./ZImageCLI -p "A beautiful mountain landscape at sunset" -o output.png
```

Common flags:

- `--width/-W`, `--height/-H` (defaults: `ZImageModelMetadata.recommendedWidth/Height`)
- `--steps/-s`, `--guidance/-g` (defaults: `ZImageModelMetadata.recommendedInferenceSteps/GuidanceScale`)
- `--model/-m` (defaults to `Tongyi-MAI/Z-Image-Turbo`)
- `--max-sequence-length` (default: 512)
- `--cache-limit` (GPU cache limit in MB; default: unlimited)

Run `./ZImageCLI --help` for the complete, authoritative option list (kept in `Sources/ZImageCLI/main.swift`).

## Model Specs (`--model`)

`--model/-m` supports:

- Hugging Face model id: `org/repo` (optionally `org/repo:revision`)
- Local model directory (Diffusers-style layout)
- Local `.safetensors` file:
  - AIO checkpoint (transformer + text encoder + VAE in one file), **or**
  - Transformer-only override layered on top of the base model

See `docs/MODELS_AND_WEIGHTS.md` for details and edge cases.

## LoRA

```bash
./ZImageCLI -p "a lion" --lora ostris/z_image_turbo_childrens_drawings --lora-scale 1.0 -o lion.png
```

Notes:

- `--lora` accepts a local path or a Hugging Face repo id.
- `--lora-scale` is clamped to `[0.0, 1.0]`.
- If a Hugging Face LoRA repo contains multiple `.safetensors`, the loader currently picks the first one; use a local path when you need a specific filename.

## Prompt Enhancement (Optional)

```bash
./ZImageCLI -p "cat with a hat" --enhance --enhance-max-tokens 512 -o cat.png
```

This uses the Qwen text encoder in “LLM mode” to rewrite the prompt before generation. It increases memory usage (see the CLI help text for current guidance).

## ControlNet + Inpainting

```bash
./ZImageCLI control \
  --prompt "A hyper-realistic close-up portrait of a leopard" \
  --control-image /path/to/canny_edges.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --control-scale 0.75 \
  --output leopard.png
```

For inpainting, provide `--inpaint-image` and `--mask`:

```bash
./ZImageCLI control -p "a dancer" -c pose.jpg -i photo.jpg --mask mask.png \
  --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors --cs 0.75 -s 25
```

Run `./ZImageCLI control --help` for the complete option list and control image expectations.

## Quantization

Quantize the base model:

```bash
./ZImageCLI quantize -i models/z-image-turbo -o models/z-image-turbo-q8 --bits 8 --group-size 32 --verbose
```

Quantize ControlNet weights:

```bash
./ZImageCLI quantize-controlnet -i alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors -o controlnet-2.1-q8 --verbose
```

After quantization, point `--model` (or `--controlnet-weights`) at the output directory.
