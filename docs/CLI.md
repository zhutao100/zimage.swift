# CLI Guide (`ZImageCLI`)

`ZImageCLI` is the macOS one-shot executable in this repo. The authoritative option list now lives in the shared CLI layer under `Sources/ZImageCLICommon/`; this document is the stable usage guide around that surface.

The repo also ships `ZImageServe`, a local staging daemon/client that reuses the same generation flags for ad hoc submissions. The daemon-specific behavior is summarized here where it overlaps with the CLI surface.

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
- `ZImageServe serve`: start the local staging daemon
- `ZImageServe status`: inspect the daemon, active job, queue, and resident worker
- `ZImageServe cancel <job-id>`: cancel an active or queued staged job
- `ZImageServe shutdown`: stop the daemon when it is idle
- `ZImageServe batch <jobs.json>`: submit a structured JSON batch manifest
- `ZImageServe markdown <prompts.md>`: submit fenced shell invocations from markdown
- `ZImageServe`: submit text-to-image generation to the daemon with the same flags as `ZImageCLI`
- `ZImageServe control`: submit ControlNet generation to the daemon with the same flags as `ZImageCLI control`
- `ZImageServe quantize` / `quantize-controlnet`: run the same local one-shot quantization paths as `ZImageCLI`

## Staging Daemon

Start the daemon:

```bash
./ZImageServe serve --residency-policy adaptive --warm-model mzbac/z-image-turbo-8bit
```

Use a custom socket path when needed:

```bash
./ZImageServe serve --socket /tmp/zimage-stage.sock
./ZImageServe --socket /tmp/zimage-stage.sock -p "a mountain lake at sunrise" -o lake.png
./ZImageServe --socket /tmp/zimage-stage.sock status
```

The ad hoc generation flags remain the same as `ZImageCLI`; only the executable name changes.

Useful daemon flags:

- `--residency-policy`: `one-shot`, `warm`, or `adaptive`; the default is `adaptive`
- `--warm-model`: prewarm a text-to-image worker on startup
- `--warm-controlnet-weights`, `--warm-control-file`: prewarm a control worker on startup
- `--idle-timeout`: evict the resident worker after the specified idle interval

The daemon now keeps a single resident worker profile by default. Matching staged requests reuse that worker until the profile changes, the idle timeout expires, or adaptive low-memory fallback evicts it.

Operational commands:

```bash
./ZImageServe status
./ZImageServe cancel <job-id>
./ZImageServe shutdown
```

`status` reports the socket path, residency policy, idle timeout, active job id, queued job ids, and resident worker summary. Ad hoc staged submissions now log `Accepted job <uuid>` so that `cancel` has a concrete id to target. `shutdown` is intentionally idle-only; it rejects the request while a job is active or queued.

Structured batch example:

```json
{
  "version": 1,
  "defaults": {
    "model": "mzbac/z-image-turbo-8bit",
    "width": 256,
    "height": 256
  },
  "jobs": [
    {
      "id": "lake-1",
      "kind": "text",
      "prompt": "a mountain lake at sunrise",
      "outputPath": "out/lake.png"
    }
  ]
}
```

Submit it with:

```bash
./ZImageServe batch jobs.json
```

Markdown example:

````markdown
```bash
ZImageServe --prompt "a mountain lake at sunrise" --model mzbac/z-image-turbo-8bit --output out/lake.png
```
````

Markdown ingestion is single-command only. Each accepted `bash`, `sh`, or `zsh` fence must reduce to exactly one direct `ZImageCLI` or `ZImageServe` invocation. Explicit relative or absolute executable paths are accepted, and command substitutions are evaluated when each markdown item starts. Wrappers such as `env` or `time`, shell control operators, redirects, and other shell expansion syntax remain rejected instead of executed. Batch and markdown submissions continue through the manifest in client order and report an aggregated non-zero failure if any staged job fails.

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
- `--lora/-l`, `--lora-file`, `--lora-scale`: text-to-image LoRA support
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

Important nuance: preset lookup now covers known ids, inspectable local or cached snapshots, and common Z-Image-style aliases. Completely unrecognized models still keep the Turbo-compatible preset unless you set the relevant flags explicitly.

LoRA nuance: third-party adapter cards can recommend sampling settings that differ from the base-model defaults. The CLI does not auto-parse adapter README files into presets, but it now emits a known-adapter warning for the validated `Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors` path with the recommended `--steps 8 --guidance 1.0 --lora-scale 0.8` recipe.

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

`--lora` accepts a local path or a Hugging Face repo id. The same `--lora` and `--lora-scale` flags are also available on `ZImageCLI control`.

When the LoRA source is a local directory or Hugging Face snapshot that contains multiple `.safetensors` files, `--lora-file` is now required. The loader no longer picks an arbitrary file or merges multiple LoRA files implicitly.
For the known multi-file repo id `alibaba-pai/Z-Image-Fun-Lora-Distill`, `--lora-file` is required even when the local cache only contains one previously downloaded adapter file.

Example for the validated Distill adapter path:

```bash
./ZImageCLI \
  -p "a cinematic portrait at golden hour" \
  --model Tongyi-MAI/Z-Image \
  --lora alibaba-pai/Z-Image-Fun-Lora-Distill \
  --lora-file Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors \
  --lora-scale 0.8 \
  --steps 8 \
  --guidance 1.0 \
  -o distill.png
```

## Prompt Enhancement

```bash
./ZImageCLI -p "cat with a hat" --enhance --enhance-max-tokens 512 -o cat.png
```

This re-prompts through the Qwen text encoder's generation path before normal encoding. It increases memory use, and the same `--enhance` and `--enhance-max-tokens` flags are available on both generation commands.

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

For `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1`, the initial supported file is the full Union checkpoint:

```text
Z-Image-Fun-Controlnet-Union-2.1.safetensors
```

The loader now rejects ambiguous multi-file ControlNet sources unless `--control-file` is set, and it rejects the current upstream Lite and Tile filenames for the Z-Image Fun Base family.
For the known multi-file repo id `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1`, `--control-file` is required even when the local cache only contains the full Union file.

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

ControlNet with LoRA and prompt enhancement:

```bash
./ZImageCLI control \
  --prompt "a minimalist modern hallway interior" \
  --control-image /path/to/depth.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --lora F16/z-image-turbo-flow-dpo \
  --lora-scale 1.0 \
  --enhance \
  --enhance-max-tokens 512 \
  --steps 9 \
  --guidance 1.0 \
  --output control-lora.png
```

Additional full-Union control patterns:

```bash
# HED-style control
./ZImageCLI control \
  --prompt "a noir kitchen scene lit by a single warm pendant light" \
  --control-image /path/to/hed.png \
  --controlnet-weights alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Fun-Controlnet-Union-2.1.safetensors \
  --control-scale 0.75 \
  --model Tongyi-MAI/Z-Image \
  --output hed-base.png

# Gray / tonal structure control
./ZImageCLI control \
  --prompt "a portrait with soft cinematic grayscale structure" \
  --control-image /path/to/gray.png \
  --controlnet-weights alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Fun-Controlnet-Union-2.1.safetensors \
  --control-scale 0.8 \
  --model Tongyi-MAI/Z-Image \
  --output gray-base.png

# Scribble control
./ZImageCLI control \
  --prompt "a whimsical concept sketch rendered as a polished illustration" \
  --control-image /path/to/scribble.png \
  --controlnet-weights alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Fun-Controlnet-Union-2.1.safetensors \
  --control-scale 0.9 \
  --model Tongyi-MAI/Z-Image \
  --output scribble-base.png

# Inpaint + control
./ZImageCLI control \
  --prompt "restore the missing section as a weathered stone archway" \
  --control-image /path/to/canny.png \
  --inpaint-image /path/to/source.png \
  --mask /path/to/mask.png \
  --controlnet-weights alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Fun-Controlnet-Union-2.1.safetensors \
  --control-scale 0.75 \
  --model Tongyi-MAI/Z-Image \
  --output inpaint-control-base.png
```

Important `control` flags:

- `--prompt/-p`: required
- `--control-image/-c`: optional control image
- `--inpaint-image/-i`: optional inpaint source
- `--mask` or `--mask-image`: optional mask for inpainting
- `--controlnet-weights/--cw`: required ControlNet source
- `--control-file/--cf`: file selector within a repo or directory; required when the source contains multiple `.safetensors`
- `--lora/-l`, `--lora-file`, `--lora-scale`: optional LoRA adapter and filename selector
- `--control-scale/--cs`: control-context scale, default `0.75`; upstream full-Union guidance is `0.65 ... 1.00`
- `--width/-W`, `--height/-H`, `--steps/-s`, `--guidance/-g`
  Width and height must be at least `64` and divisible by `16`.
- `--cfg-normalization`, `--cfg-truncation`
- `--lora/-l`, `--lora-scale`: optional control-path LoRA adapter, same local-path or Hugging Face semantics as text-to-image
- `--enhance/-e`, `--enhance-max-tokens`: optional prompt enhancement before text encoding; increases memory use
- `--weights-variant`, `--cache-limit`, `--max-sequence-length`, `--no-progress`
- `--log-control-memory`: emit control-path memory markers

Known `Tongyi-MAI` model ids, inspectable local or cached snapshots, and common Z-Image-style aliases use the same model-aware defaults on the `control` subcommand. Completely unrecognized models still need explicit sampling flags if you do not want the Turbo-compatible preset.

At least one of `--control-image`, `--inpaint-image`, or `--mask` must be present.

Current control-path nuances:

- `--model` on the control command resolves a standard model snapshot or local directory. The control pipeline does not expose the text-to-image AIO / transformer-only `.safetensors` override path.
- `--weights-variant` applies to the base model snapshot, not the ControlNet weights source.

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
