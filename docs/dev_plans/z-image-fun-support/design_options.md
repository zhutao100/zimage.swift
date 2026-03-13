# Design Options For `Z-Image-Fun` Support

## Validated findings from the current repo and uploaded artifacts

### 1. Full Base Union 2.1 ControlNet is architecturally close to the current Swift control path

The current repo already ships a full control path with:

- a separate `ZImageControlPipeline`
- explicit ControlNet weight loading
- `--control-file` support for multi-file Hugging Face repos
- inpainting support
- a control architecture whose default full layout is:
  - `15` control layer placements: `[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]`
  - `2` control refiner placements: `[0, 1]`
  - `control_in_dim = 33`

The uploaded tensor summary for `Z-Image-Fun-Controlnet-Union-2.1.safetensors` matches that full layout:

- `295` tensors total
- `15` `control_layers.*` blocks
- `2` `control_noise_refiner.*` blocks
- `control_all_x_embedder.2-1.*`
- `BF16` tensors with the expected `3840`-dimensional control blocks

That is a strong signal that the requested **full** Base Union 2.1 file should fit the existing architecture with little or no model-structure work.

### 2. Multi-file adapter repos need explicit file selection, not best-effort defaults

The upstream Hugging Face Distill repo currently publishes multiple normal LoRA files in one repo:

- `2`-step
- `4`-step
- `8`-step
- ComfyUI exports
- older `2602` and pre-`2602` variants

The current Swift CLI only exposes `--lora`, not a paired filename selector. Programmatic support exists in `LoRAConfiguration.huggingFace(modelId, filename:)`, but the user-facing CLI, batch manifest, and staged submission shapes do not currently expose that choice.

As a result, the repo cannot currently claim clean direct support for `alibaba-pai/Z-Image-Fun-Lora-Distill` by Hugging Face repo id alone.

The same issue now exists on the ControlNet side in a different form: the full Union repo contains full, lite, and tile `.safetensors` files side by side, while the current control loader merges every `.safetensors` file in a directory if no preferred file is selected.

That means deterministic file selection is required on both adapter families:

- `--lora-file` is missing and must be added
- ambiguous multi-file ControlNet sources should be rejected or forced to specify `--control-file`

### 3. The uploaded Distill LoRA key format is a current blocker, not a likely risk

The uploaded `Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors` uses underscore-style adapter keys such as:

- `_layers_0_attention_to_q`
- `_layers_0_attention_to_k`
- `_layers_0_feed_forward_w1`
- `_noise_refiner_0_feed_forward_w2`
- `_context_refiner_1_attention_to_v`

The current mapper is tuned for dot-form keys and older underscore forms. Under the current normalization rules, keys in this family are currently normalized into invalid targets such as:

- `attention.to.q` instead of `attention.to_q`
- `feed.forward.w1` instead of `feed_forward.w1`

Local validation against the cached `8-Steps-2603` file produced `204` mapped LoRA layers and `0` valid target paths. This should be treated as a required compatibility patch plus a fail-fast validation item, not a documentation-only issue.

### 4. Standard-LoRA per-layer alpha is not a blocker for the inspected `8-Steps-2603` file

The uploaded Distill file contains standard-LoRA `.alpha` tensors for each adapted target.

The current standard-LoRA path does not consume those tensor-local alpha values; it only consumes `adapter_config.json` alpha when present.

For the cached `8-Steps-2603` file, every inspected per-target alpha value is `128.0` and the inferred LoRA rank is also `128`, so `alpha / rank == 1.0` throughout the inspected file.

That means one of the following should be done explicitly for the plan:

- document that per-target alpha has already been validated as rank-equivalent for `8-Steps-2603`
- optionally add standard-LoRA per-target alpha handling later for robustness across other adapter variants

Per-target alpha should therefore be treated as a robustness item rather than a day-one blocker for the initial `8-Steps-2603` support target.

---

## Work needed by area

### A. CLI, batch, and staged request surface

#### Required

Add a LoRA filename selector parallel to the existing ControlNet filename selector, and harden the multi-file selection rules for ControlNet.

Recommended shape:

- text path: `--lora-file`
- control path: `--lora-file`
- batch defaults + per-job fields: `loraFile`
- staged submission JSON transport: inherit the updated option structs rather than inventing a new parallel field shape
- keep `--control-file` as the ControlNet selector, but reject ambiguous multi-file sources when it is omitted for repos like `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1`

#### Why this is required

Without filename selection, users cannot cleanly and reproducibly select:

- `Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors`
- instead of the `2`-step file
- or a ComfyUI export
- or an older `2602` variant

Without stricter ControlNet selection rules, users can also accidentally combine full, lite, and tile weights from the same repo when loading from a directory or snapshot without `--control-file`.

#### Likely files

- `Sources/ZImageCLICommon/CLIModels.swift`
- `Sources/ZImageCLICommon/CLIParser.swift`
- `Sources/ZImageCLICommon/CLIUsage.swift`
- `Sources/ZImageCLICommon/BatchManifest.swift`
- `Sources/ZImageCLICommon/CLICommandRunner.swift`
- `Sources/ZImageServeCore/ServiceModels.swift`

### B. LoRA loader and mapper compatibility

#### Required

Extend the current LoRA key normalization to support the Distill adapter’s underscore family.

Minimum compatibility targets:

- `_layers_0_attention_to_q` -> `layers.0.attention.to_q`
- `_layers_0_attention_to_k` -> `layers.0.attention.to_k`
- `_layers_0_attention_to_v` -> `layers.0.attention.to_v`
- `_layers_0_feed_forward_w1` -> `layers.0.feed_forward.w1`
- `_layers_0_feed_forward_w2` -> `layers.0.feed_forward.w2`
- `_layers_0_feed_forward_w3` -> `layers.0.feed_forward.w3`
- and the corresponding `noise_refiner` / `context_refiner` targets

#### Recommended guardrail

If a LoRA file resolves to zero valid target layers after mapping, fail fast with a structured error rather than proceeding with a silent no-op adapter.

#### Likely files

- `Sources/ZImage/LoRA/LoRAKeyMapper.swift`
- `Sources/ZImage/LoRA/LoRAWeightLoader.swift`
- `Tests/ZImageTests/Weights/LoRALoaderTests.swift`
- `Tests/ZImageIntegrationTests/LoRAIntegrationTests.swift`

### C. Full Base Union 2.1 ControlNet validation and guardrails

#### Required

For the requested **full Union 2.1** file, the existing control architecture should be validated and documented.

The main work here is:

- exercise Base-model control generation using the requested full Union file
- add docs examples for HED, Gray, and Scribble style inputs in addition to the already-familiar Canny / Depth / Pose patterns
- document the upstream-recommended `control_context_scale` range for this family

#### Recommended guardrail

Do not silently claim support for Lite or Tile variants in the first pass.

Those variants have different control-layer placement assumptions and should either:

- be rejected clearly in the initial pass
- or be supported through an explicit config-selection layer

Also reject ambiguous multi-file ControlNet sources unless the intended file is selected explicitly.

#### Likely files

- `Tests/ZImageIntegrationTests/ControlNetIntegrationTests.swift`
- `README.md`
- `docs/CLI.md`
- `docs/MODELS_AND_WEIGHTS.md`

### D. Adapter-aware presets and warnings

#### Optional but useful

The current CLI already warns when LoRA is used with model defaults, because adapter cards can require different sampling.

For `alibaba-pai/Z-Image-Fun-Lora-Distill`, the upstream guidance is concrete enough that the repo could optionally add a known-adapter warning or preset hint:

- `steps = 8`
- `guidance = 1.0`
- `lora_scale ~= 0.8`
- prefer the current simple scheduler path

This should remain a warning or opt-in preset layer rather than an implicit forced override.

#### Likely files

- `Sources/ZImageCLICommon/CLICommandRunner.swift`
- optionally a new support registry under `Sources/ZImage/Support/`
- docs updates in `README.md` and `docs/CLI.md`

---

## Design options

| Option | Description | Pros | Cons | Fit |
|---|---|---|---|---|
| A. Minimal compatibility | Validate full Base Union 2.1; document local-file Distill usage only; no new CLI filename selection | Lowest code churn | Does not solve repo-id Distill support; leaves weak staged/batch ergonomics; still needs fail-fast validation to avoid a no-op adapter | Acceptable only as a temporary stopgap |
| B. Balanced first-class support | Validate full Base Union 2.1; add `--lora-file`; reject ambiguous multi-file ControlNet loads; patch underscore-key mapping; add tests and docs | Practical, reproducible, and aligned with current architecture | Moderate code changes across CLI + loader + tests | **Best option** |
| C. Generalized Z-Image-Fun family support | Option B plus dynamic handling for Union/Lite/Tile variants and richer adapter-aware presets | Most future-proof | Higher complexity, more validation burden, wider support surface than the current ask | Good follow-on after Option B |

---

## Recommended option: B

### Why Option B is the right scope

It matches the actual shape of the problem:

- the requested full Base Union 2.1 ControlNet is already close to the current control architecture
- the Distill LoRA needs real compatibility work, but the work is narrow and now precisely identified
- the repo already has the right abstractions for Hugging Face filename selection and LoRA configuration; the missing pieces are primarily CLI exposure, key-shape compatibility, and fail-closed loading behavior

Option B gives the repo a support claim that is specific, testable, and useful without prematurely expanding into Lite/Tile generalization.

---

## Proposed implementation sequence

### Phase 1 — Distill compatibility and deterministic file selection

1. add `--lora-file` across CLI, batch, and staged request paths
2. reject ambiguous multi-file ControlNet sources unless `--control-file` is specified
3. patch underscore-form key normalization
4. add a fail-fast validation error when a LoRA maps to zero valid Swift target paths
5. add unit tests covering the inspected Distill key family and the new filename-selection path

### Phase 2 — full Base Union 2.1 ControlNet

1. add an explicit support note for the requested full file only
2. add or update integration coverage for Base model + Union 2.1
3. document usage examples for:
   - HED
   - Gray
   - Scribble
   - inpaint + control
4. add a clear error or explicit non-support note for Lite/Tile files until their placement logic is implemented
5. document the inspected `8-Steps-2603` Distill recipe:
   - `steps = 8`
   - `guidance = 1.0`
   - `lora_scale = 0.8`
   - simple scheduler preference

### Phase 3 — follow-on UX

1. add known-adapter warning text or opt-in preset support
2. decide whether Lite and Tile Base ControlNet variants should be:
   - rejected permanently
   - supported through a small config registry
   - or supported through lightweight file-name heuristics plus validation

---

## Acceptance criteria

### ControlNet

- the repo can exercise `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` with the full Union 2.1 file on the Base model path
- docs make the intended file selection explicit
- ambiguous multi-file ControlNet sources fail clearly instead of merging full/lite/tile weights silently
- unsupported Lite/Tile files do not silently run under the full-Union assumptions

### Distill LoRA

- the CLI can select `alibaba-pai/Z-Image-Fun-Lora-Distill` with an explicit filename
- the loader maps the uploaded Distill key family into valid Z-Image transformer targets
- the inspected `8-Steps-2603` file does not resolve to zero valid target layers
- the adapter applies non-zero target layers instead of silently skipping them
- the docs show an explicit `8`-step `2603` example with the recommended guidance and LoRA scale

### Repo quality

- user-facing docs and CLI help stay in sync
- batch JSON and staged submission stay behavior-compatible with the existing request model
- tests cover both the new filename-selection path and the new underscore-key normalization rules
