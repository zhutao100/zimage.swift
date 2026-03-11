# CLI Guide (`ZImageCLI`)

`ZImageCLI` is the macOS executable in this repo. The authoritative option list lives in `Sources/ZImageCLI/main.swift`; this document is the stable usage guide around it.

## Build And Locate The Binary

Fast path:

```bash
./scripts/build.sh
```

Explicit build command:

```bash
xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode -skipPackagePluginValidation ENABLE_PLUGIN_PREPAREMLSHADERS=YES CLANG_COVERAGE_MAPPING=NO
```

Run from the release products directory:

```bash
cd .build/xcode/Build/Products/Release
./ZImageCLI --help
./ZImageCLI control --help
```

If Xcode prompts about the MLX shader-preparation package plugin, allow it. For non-interactive or CI builds, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Command Summary

- `ZImageCLI`: text-to-image generation
- `ZImageCLI control`: ControlNet conditioning and inpainting
- `ZImageCLI quantize`: quantize base-model weights
- `ZImageCLI quantize-controlnet`: quantize ControlNet weights

## Text-To-Image

Minimal run:

```bash
./ZImageCLI -p "a mountain lake at sunrise" -o output.png
```

Useful flags:

- `--prompt/-p`: required prompt
- `--negative-prompt/--np`: optional negative prompt
- `--width/-W`, `--height/-H`: output size, must be at least `64` and divisible by `16`
- `--steps/-s`, `--guidance/-g`: denoising and CFG settings
- `--cfg-normalization`: clamp CFG output norm back to the positive-branch norm
- `--cfg-truncation`: disable CFG after the normalized timestep passes the given value
- `--seed`: deterministic sampling seed
- `--output/-o`: output path, default `z-image.png`
- `--model/-m`: model id, local Diffusers-style directory, or local `.safetensors`
- `--weights-variant`: precision-specific weight selection such as `fp16` or `bf16`
- `--force-transformer-override-only`: force a local `.safetensors` to be treated as a transformer override instead of AIO
- `--cache-limit`: MLX cache limit in MB
- `--max-sequence-length`: text-encoding token cap, default `512`
- `--lora/-l`, `--lora-scale`: text-to-image LoRA support
- `--enhance/-e`, `--enhance-max-tokens`: prompt enhancement through the Qwen text encoder
- `--no-progress`: disable progress output

Invalid CLI invocations exit non-zero. Missing required flags, missing option values, unknown flags, and invalid numeric values also print the matching command usage.

### Important Default Behavior

The CLI applies model-aware defaults for the built-in `Tongyi-MAI` model ids:

- `Tongyi-MAI/Z-Image-Turbo`: `1024x1024`, `9` steps, guidance `0.0`
- `Tongyi-MAI/Z-Image`: `1024x1024`, `50` steps, guidance `4.0`

`--steps` is the literal denoising-iteration count in this repo. The scheduler keeps one extra terminal sigma internally, so `8` steps means `8` transformer forwards and `9` sigma values.

Explicit flags still override those values field by field. Example:

```bash
./ZImageCLI \
  --model Tongyi-MAI/Z-Image \
  --prompt "a black tiger in a bamboo forest" \
  --output base.png
```

Important nuance: preset lookup is id-based. Local paths and unknown model ids keep the Turbo-compatible preset unless you set the relevant flags explicitly.

LoRA nuance: third-party adapter cards can recommend sampling settings that differ from the base-model defaults. The CLI does not auto-parse adapter README files into presets, so keep `--steps` and `--guidance` explicit when an adapter card calls out values.

## Model Specs (`--model`)

On the text-to-image command, `--model/-m` accepts:

- Hugging Face repo id: `org/repo`
- Hugging Face repo id with revision: `org/repo:revision`
- Local Diffusers-style model directory
- Local `.safetensors`
  - AIO checkpoint if the file contains all expected components
  - transformer-only override otherwise

If you point `--model` at a local directory that does not contain the expected Diffusers-style configs but does contain `.safetensors`, the text-to-image resolver picks a preferred file from that directory and inspects it as a local checkpoint. If the directory contains neither the expected configs nor any `.safetensors`, the command now fails with an explicit local-path error instead of falling back to the default model.

See [MODELS_AND_WEIGHTS.md](MODELS_AND_WEIGHTS.md) for resolver details.

## LoRA

Text-to-image LoRA usage:

```bash
./ZImageCLI \
  -p "a lion painted like a children's book illustration" \
  --lora ostris/z_image_turbo_childrens_drawings \
  --lora-scale 1.0 \
  -o lion.png
```

`--lora` accepts a local path or a Hugging Face repo id.

## Prompt Enhancement

```bash
./ZImageCLI -p "cat with a hat" --enhance --enhance-max-tokens 512 -o cat.png
```

This re-prompts through the Qwen text encoder's generation path before normal encoding. It increases memory use and is currently exposed on the text-to-image CLI only.

## ControlNet And Inpainting

Minimal ControlNet example:

```bash
./ZImageCLI control \
  --prompt "a dancer on a stage" \
  --control-image /path/to/pose.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --output control.png
```

Inpainting example:

```bash
./ZImageCLI control \
  --prompt "restore the missing area as a stained-glass window" \
  --inpaint-image /path/to/photo.png \
  --mask /path/to/mask.png \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --output inpaint.png
```

Important `control` flags:

- `--prompt/-p`: required
- `--control-image/-c`: optional control image
- `--inpaint-image/-i`: optional inpaint source
- `--mask` or `--mask-image`: optional mask for inpainting
- `--controlnet-weights/--cw`: required ControlNet source
- `--control-file/--cf`: optional file selector within a repo or directory
- `--control-scale/--cs`: control-context scale, default `0.75`
- `--width/-W`, `--height/-H`, `--steps/-s`, `--guidance/-g`
  Width and height must be at least `64` and divisible by `16`.
- `--cfg-normalization`, `--cfg-truncation`
- `--weights-variant`, `--cache-limit`, `--max-sequence-length`, `--no-progress`
- `--log-control-memory`: emit control-path memory markers

Known `Tongyi-MAI` model ids use the same model-aware defaults on the `control` subcommand. Local paths and unknown ids still need explicit sampling flags if you do not want the Turbo-compatible preset.

At least one of `--control-image`, `--inpaint-image`, or `--mask` must be present.

Current control-path nuances:

- `--model` on the control command resolves a standard model snapshot or local directory. The control pipeline does not expose the text-to-image AIO / transformer-only `.safetensors` override path.
- `--weights-variant` applies to the base model snapshot, not the ControlNet weights source.
- `ZImageCLI control` does not expose the control-pipeline LoRA or prompt-enhancement fields that exist in the library request type.

## Quantization

Base-model quantization:

```bash
./ZImageCLI quantize \
  --input models/z-image-turbo \
  --output models/z-image-turbo-q8 \
  --bits 8 \
  --group-size 32 \
  --verbose
```

ControlNet quantization:

```bash
./ZImageCLI quantize-controlnet \
  --input alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --output controlnet-2.1-q8 \
  --bits 8 \
  --group-size 32 \
  --verbose
```

After quantization, point `--model` or `--controlnet-weights` at the output directory.

## Diagnostics

- `--no-progress`: suppress progress reporting
- `ZImageCLI control --log-control-memory`: log process-resident and MLX memory markers around prompt encoding, control-context construction, denoiser loading, and decode

The repo also keeps an opt-in real-model Base smoke test for regression checking. The exact command lives in [DEVELOPMENT.md](DEVELOPMENT.md).

For the underlying control-memory policy and validation recipe, see [DEVELOPMENT.md](DEVELOPMENT.md) and [dev_plans/controlnet-memory-followup.md](dev_plans/controlnet-memory-followup.md).
