# Z-Image Fun Support Plan

Status: active on March 13, 2026

## Implementation status

- Phase 1 complete:
  - `--lora-file` is exposed across text CLI, control CLI, batch manifests, and staged request payloads
  - multi-file LoRA sources now fail closed unless a specific file is selected
  - the known repo id `alibaba-pai/Z-Image-Fun-Lora-Distill` now requires explicit filename selection even when the local cache only contains one downloaded adapter file
  - the inspected Distill underscore-form keys now map onto valid Swift target paths
  - LoRA loads that resolve to zero valid target layers now fail clearly instead of silently no-oping
- Phase 2 complete:
  - ambiguous multi-file ControlNet directories now fail closed unless `--control-file` is set
  - the current Z-Image Fun Base ControlNet support target is the full Union 2.1 file only
  - the known repo id `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` now requires `--control-file` even when the local cache only contains the full Union file
  - current upstream Lite and Tile filenames for the Z-Image Fun Base family are rejected explicitly
  - selected ControlNet weights are validated against the current full-layout contract (`15` layer blocks, `2` refiner blocks, `control_in_dim = 33`)
- Phase 3 partially complete:
  - the CLI now emits a known-adapter warning for the validated Distill `8-Steps-2603` file with the documented `8 / 1.0 / 0.8` recipe
  - broader Lite/Tile generalization remains deferred

## Scope

This plan covers the work needed to add practical, first-class support for:

- `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1`
- `alibaba-pai/Z-Image-Fun-Lora-Distill`

The target is the current `Z-Image.swift` package and CLI surface, not the broader `VideoX-Fun` Python stack.

## Research summary

### `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1`

Upstream currently describes the requested full Union 2.1 file as:

- a Z-Image Base ControlNet
- multi-condition support for Canny, Depth, Pose, MLSD, Scribble, HED, and Gray
- inpainting support
- an optimal `control_context_scale` range of `0.65 ... 1.00`
- a large model with control added on `15` layer blocks plus `2` refiner blocks

The upstream repo also currently publishes:

- `Z-Image-Fun-Controlnet-Union-2.1-lite.safetensors`
- `Z-Image-Fun-Controlnet-Tile-2.1.safetensors`
- `Z-Image-Fun-Controlnet-Tile-2.1-lite.safetensors`

Those variants are not just alternate filenames for the same runtime contract. The local `VideoX-Fun` reference config for Lite uses `control_layers_places: [0, 10, 20]` instead of the full model's `15` control layer placements, so Lite and Tile should be treated as explicit follow-on work rather than part of the initial support claim.

### `alibaba-pai/Z-Image-Fun-Lora-Distill`

Upstream currently describes the Distill LoRA family as:

- a Z-Image Base distillation adapter that distills both step count and CFG
- trained from scratch rather than reusing Z-Image-Turbo weights
- compatible with other Z-Image LoRAs and Control paths
- published as a multi-file repo with `2` / `4` / `8` step variants plus ComfyUI exports
- recommending `cfg = 1.0`, `steps = 8`, and `lora_weight = 0.8` for the normal 8-step path
- recommending the simple scheduler for inference, with the `2603` release specifically improving low-sigma behavior below `0.500`

The upstream Hugging Face repo currently contains multiple normal and ComfyUI `.safetensors` files. That makes explicit filename selection a required part of any first-class support claim for repo-id-based loading.

## Current repo fit

### ControlNet fit

The requested full Union 2.1 Base ControlNet is a close fit for the current control-path architecture:

- the current Swift control model already uses `15` control layer placements plus `2` refiner placements
- the uploaded tensor summary for `Z-Image-Fun-Controlnet-Union-2.1.safetensors` matches that full layout rather than the lite layout
- the control pipeline already supports multi-file Hugging Face repos through `--control-file`
- the control pipeline already supports inpainting and arbitrary precomputed control images

That means the full requested Union 2.1 file should be approached as a validation-and-polish task, not a major architecture rewrite.

There is still one important loading guardrail missing today:

- when a local directory or Hugging Face snapshot contains multiple `.safetensors` files and `preferredFile` is not set, the current Swift loader merges every `.safetensors` file it finds into one weights map

That behavior is acceptable for single-file repos, but it is unsafe for `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` because the repo now contains full Union, Lite Union, Tile, and Lite Tile files side by side. Initial support should therefore either:

- require an explicit control filename for this repo family, or
- reject ambiguous multi-file sources instead of merging them silently

### Distill LoRA fit

The Distill LoRA is not yet a drop-in fit.

Three concrete gaps need to be closed before the support claim is solid:

1. **Hugging Face file selection**
   - the upstream repo contains many `.safetensors` files
   - the current LoRA CLI surface has `--lora`, but no `--lora-file`
   - the current Hugging Face LoRA resolver falls back to the first `.safetensors` file in the snapshot when no filename is provided

2. **underscore-form adapter key mapping**
   - the uploaded `2603` adapter uses underscore-style keys such as `_layers_0_attention_to_q` and `_feed_forward_w1`
   - the current key-mapper logic does not merely need a compatibility pass; it currently maps these names into invalid targets such as `attention.to.q` and `feed.forward.w1`
   - local validation against the cached `Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors` file produced `204` mapped LoRA layers and `0` valid targets, so this is a current blocker rather than a speculative risk

3. **missing fail-fast validation for no-op LoRA loads**
   - the current loader can return a non-empty `LoRAWeights.weights` map even when every mapped key is invalid for the transformer
   - first-class support should fail clearly when a LoRA resolves to zero valid target layers instead of silently proceeding with an effectively no-op adapter

There is also a smaller compatibility item that has now been narrowed:

4. **standard-LoRA per-layer alpha handling**
   - the uploaded adapter contains per-layer `.alpha` tensors
   - the current standard-LoRA path only uses `adapter_config.json` alpha, not tensor-local standard LoRA alpha values
   - for the cached `Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors` file, all `204` per-target alpha tensors are `128.0` and the inferred rank is also `128`, so tensor-local alpha is rank-equivalent for this specific file and is not a blocker for the initial target
   - generic per-target alpha support can remain a validation/robustness item for later variants rather than a day-one requirement for `8-Steps-2603`

## Decision

Chosen direction: **phased support with a hardened first release contract**.

1. treat full Base Union 2.1 ControlNet support as a validation, docs, and guardrail pass
2. treat Distill LoRA support as a focused compatibility project with explicit fail-fast validation
3. defer Lite/Tile generalization and adapter-aware preset UX until the requested full files are stable

## Delivery phases

### Phase 0 — lock down the support contract

- document that the initial support target is:
  - `Z-Image-Fun-Controlnet-Union-2.1.safetensors`
  - `Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors`
- do not claim support for Lite or Tile variants yet
- do not claim automatic LoRA preset inference yet
- treat repo-id-based loading as incomplete until explicit filename selection exists for Distill LoRAs
- treat multi-file ControlNet sources as invalid unless the intended `.safetensors` file is selected explicitly
- add a fail-fast rule for LoRA loads that map zero valid target layers

### Phase 1 — Distill LoRA compatibility blockers

- add explicit Hugging Face LoRA filename selection across text CLI, control CLI, batch manifests, and staged request transport
- patch underscore-form key compatibility for the inspected Distill adapter layout:
  - `attention_to_{q,k,v}` -> `attention.to_{q,k,v}`
  - `feed_forward_w{1,2,3}` -> `feed_forward.w{1,2,3}`
- add loader diagnostics and validation so the inspected `8-Steps-2603` file maps to valid transformer targets instead of producing a silent no-op
- add unit coverage for the actual `lora_unet__layers_*`, `noise_refiner_*`, and `context_refiner_*` naming family
- record in docs/tests that the inspected `8-Steps-2603` file has rank-equivalent per-target alpha values (`128.0` for rank `128`)

### Phase 2 — full Base Union 2.1 ControlNet

- validate the full Union 2.1 file against the existing control architecture on the Base model path
- require explicit file selection or reject ambiguous multi-file sources for this repo family
- add docs examples for the supported control modes and inpainting combinations, including HED, Gray, Scribble, and inpaint + control
- document the supported `control_context_scale` range and the stronger inpaint guidance
- add guardrails so unsupported Lite/Tile files do not fail silently under the full-Union assumptions

### Phase 3 — optional UX and family expansion

- optional adapter-aware defaults or warnings for known Distill repos
- optional support for Lite and Tile Base ControlNet variants through explicit config selection or variant-aware validation
- optional broader Z-Image Fun registry support after the initial files are stable

## Verification bar

The support work is not complete until all of the following are true:

- full Base Union 2.1 ControlNet can be selected explicitly and exercised from the current control pipeline
- Distill LoRA can be selected from Hugging Face by repo id plus explicit filename
- the inspected `Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors` file maps to valid target layers instead of silently skipping all `204`
- LoRA loads that resolve to zero valid target layers fail clearly
- the recommended `8`-step Distill path is documented with explicit `--steps`, `--guidance`, and `--lora-scale` guidance
- unsupported Base Lite/Tile files fail clearly or are explicitly supported, but are not left in an ambiguous partial state

## Source docs

- [design_options.md](design_options.md)
