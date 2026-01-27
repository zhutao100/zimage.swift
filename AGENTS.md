# AGENTS.md (LLM / Agent Guide)

This file is the operational guide for agents working in this repo. Prefer concrete pointers (file paths, commands) and keep docs in sync when behavior changes.

## Read First (Pick The Minimal Set)

- **Any task**: `README.md`, then `docs/README.md`
- **Architecture / code navigation**: `docs/ARCHITECTURE.md`, then `CLAUDE.md`
- **CLI work** (flags/subcommands/help): `docs/CLI.md` and `Sources/ZImageCLI/main.swift`
- **Model loading / weights / safetensors**: `docs/MODELS_AND_WEIGHTS.md` and `Sources/ZImage/Weights/*`
- **Upstream model structure**: `docs/z-image-turbo.md`

If you’re in a hurry: `CLAUDE.md` is a concise “project overview + build/test commands” summary.

## Repo Map (Where Things Live)

- `Sources/ZImageCLI/` — CLI entry point and help text (`main.swift`)
- `Sources/ZImage/` — library code
  - `Pipeline/` — `ZImagePipeline`, `ZImageControlPipeline`, scheduler
  - `Model/` — TextEncoder, Transformer, VAE
  - `Weights/` — download/cache resolution, safetensors parsing, mapping tensors → modules
  - `Quantization/` — 4/8-bit quantization + manifests
  - `LoRA/` — LoRA/LoKr loading + application
- `Tests/`
  - `ZImageTests/` — unit tests (fast, no model weights)
  - `ZImageIntegrationTests/` — requires weights
  - `ZImageE2ETests/` — builds and runs the CLI
- `docs/` — reference docs (keep as source of truth; link from README instead of duplicating)
- `.llm_analysis/` — module-level analysis notes (helpful for orientation; keep reasonably in sync)

## Conventions & Expectations

- Before wrapping up, check whether similar changes need to be applied across multiple places/files (e.g., both pipelines, both transformers, help text + docs).
- Avoid over-engineering: prefer small, targeted refactors unless a bigger change is clearly required.
- When changing externally visible behavior (CLI flags, model resolution semantics), update:
  - `Sources/ZImageCLI/main.swift` (help text)
  - `docs/CLI.md` and/or `docs/MODELS_AND_WEIGHTS.md`
  - root `README.md` (high-level only; link to docs for details)

## Build & Test Commands (Preferred)

Build release CLI:

```bash
xcodebuild -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

Run unit tests only (default verification path):

```bash
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests
```

Constraint: **Do not run integration or e2e tests by default.** Leave `ZImageIntegrationTests` and `ZImageE2ETests` for users unless explicitly requested.

## “Source Of Truth” Pointers

- CLI flags/subcommands: `Sources/ZImageCLI/main.swift`
- Default model id: `Sources/ZImage/Weights/ModelPaths.swift` (`ZImageRepository.id`)
- HF cache and model resolution: `Sources/ZImage/Weights/ModelResolution.swift`
- AIO detection and canonicalization: `Sources/ZImage/Weights/AIOCheckpoint.swift`
- Quantization manifests + application: `Sources/ZImage/Quantization/ZImageQuantization.swift`, `Sources/ZImage/Weights/WeightsMapping.swift`

## External Reference: Diffusers

Some core implementations were written by referencing Hugging Face Diffusers:

- Local checkout: `~/workspace/custom-builds/diffusers`
- Upstream: `https://github.com/huggingface/diffusers`

Use it when validating weight naming/mapping, scheduler behavior, or architecture parity.

## Useful Tools / Resources

- Inspect `.safetensors` structure:
  - `~/bin/stls.py --format toon <file.safetensors>`
  - If missing: `curl https://gist.githubusercontent.com/zhutao100/cc481d2cd248aa8769e1abb3887facc8/raw/89d644c490bcf5386cb81ebcc36c92471f578c60/stls.py > ~/bin/stls.py`
- Default model snapshot cache (common location):
  - `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots`
  - `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots`
