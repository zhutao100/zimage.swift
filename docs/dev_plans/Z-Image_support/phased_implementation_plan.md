Below is a **PR-sized, phased implementation plan** that tracks the refreshed plan’s major workstreams: **model registry/presets**, **weightsVariant safety**, **step semantics parity**, **CFG truncation + normalization**, plus **tests/docs** as “definition-of-done” gates.

**Terminology (avoid “variant” ambiguity)**

- **Model family:** `Tongyi-MAI/Z-Image-Turbo` vs `Tongyi-MAI/Z-Image` (Base)
- **weightsVariant:** HF precision variants like `bf16`, `fp16`, etc. (affects filenames and `*.index.json`)

---

## Phase 0 — Baseline + test scaffolding (fast, low risk)

**Goal:** Set up CI-friendly tests and fixture strategy so subsequent phases can land safely (especially scheduler/CFG behavior changes).

**Work**

* Create **lightweight fixtures** for config/index parsing (no real weights) as described in the plan’s testing strategy.
* Add config-only/unit test target(s) for:

  * model config parsing (Turbo + Base snapshot-like layouts)
  * shard index parsing / resolver behavior (without loading tensors)
  * scheduler stepping test harness (placeholder until Phase 3)

**Deliverables**

* `Tests/...` new unit tests + fixture directory (minimal `config.json`, `scheduler_config.json`, `*.safetensors.index.json`).
* CI runs these tests by default; integration tests remain opt-in (per plan).

**Gate (must pass)**

* All existing tests + new config-only tests green.
* No behavior changes yet.

---

## Phase 1 — Model registry + cache naming + default propagation (Turbo remains default)

**Goal:** Make Turbo/Base **first-class selectable** with correct preset plumbing and without “Turbo-default leakage.”

**Work**

1. **Add model registry + presets**

* New `Sources/ZImage/Support/ZImageModelRegistry.swift` with:

  * `ZImageKnownModel` (`.zImageTurbo`, `.zImage`, optional `.zImageTurbo8bit`)
  * `ZImagePreset` (recommended steps/guidance/resolution/maxSequenceLength, etc.)
* Add a small library helper so callers consume presets consistently (avoid “registry exists but no one uses it”):

  * `ZImagePreset.defaults(for modelId: String) -> ZImagePreset` **or**
  * `ZImageGenerationRequest.makePreset(prompt:modelId:...)`

2. **Refactor cache naming**

* Update `ZImageRepository` / default cache directory to be **modelId-aware**, while keeping Turbo as default behavior.

3. **Same-family swapping**

* Update `areZImageVariants(_:_:)` in both pipelines to include the Base model id (so switching Turbo↔Base can reuse pipeline instance as intended).

4. **Default propagation audit**

* Ensure model defaults flow through *all* entry points: `PipelineSnapshot.swift` and `ModelResolution.swift` explicitly listed in the plan.

**Gate**

* Existing Turbo flows unchanged by default (Turbo remains default model id behavior).
* New “select Base model id” path resolves snapshot/caches under a distinct folder.
* Unit test: cache directory changes with modelId.
* Unit test: preset lookup returns different defaults for Turbo vs Base.

---

## Phase 2 — weightsVariant: deterministic index selection + downloads + guardrails

**Goal:** Prevent mixed precision shards and make resolution deterministic when repos contain multiple HF precision “variants.”

**Work**

1. **Public surface**

* Add optional `weightsVariant: String?` across CLI + library request/config surface.

2. **Variant-aware index selection**

* Update weight resolution functions to:

  * prefer `*.{weightsVariant}.safetensors.index.json`
  * fall back to non-variant index
  * disallow filename-mismatched candidates when scanning to prevent mixing

3. **Variant-aware downloads**

* Update `ModelResolution` download patterns so variant selection downloads only that variant when specified.

  * Be flexible about HF naming conventions: match both `*{weightsVariant}*.safetensors` and `*.{weightsVariant}.safetensors` styles, and the corresponding `*.index.json`.
  * Always include non-weight essentials (`*.json`, `tokenizer/*`).
  * When unspecified and multiple variants exist, log + choose a deterministic default.

4. **Guardrails**

* If variant requested but missing required component weights (transformer/text_encoder/vae), fail with a clear error. (VAE is currently a single-file weight, so it needs explicit handling.)

**Gate**

* New unit tests: resolver rejects mixed variant shard sets; resolver selects correct index when both exist.
* No regression when running existing Turbo integration tests (opt-in).

---

## Phase 3 — Step semantics parity: terminal sigma override + “dt==0” optional skip + unit tests

**Goal:** Align step semantics to the reference behavior (schedule ends at `sigma=0`) and make “9 steps → ~8 effective updates” verifiable.

**Work**

1. **Decide and document public semantics**

* `--steps` maps to `num_inference_steps`, and document last iteration may be a no-op when ending at `sigma=0`.

2. **Scheduler changes**

* Extend `FlowMatchEulerScheduler` with terminal sigma override (e.g. `sigmaMinOverride = 0.0`).
* Construct scheduler with terminal sigma = 0.0 in both pipelines.
  * Apply to `ZImagePipeline` and to **all** denoise paths in `ZImageControlPipeline` (there are multiple).

3. **Optional perf optimization**

* Skip transformer forward when `dt == 0` (explicitly documented as numerically equivalent).

4. **Unit tests**

* Add the step semantics unit test described in the plan (sigmas length, terminal sigma==0, last dt==0, Turbo preset statement holds).

**Gate**

* Scheduler tests green.
* Turbo integration remains stable when run (image generation still succeeds).
* If you’re risk-averse: land `sigmaMinOverride` + tests first, then the “skip dt==0” optimization as a tiny follow-up PR.

---

## Phase 4 — CFG truncation + CFG normalization (shared helper, both pipelines)

**Goal:** Add missing inference knobs for Base parity: `cfg_truncation` and `cfg_normalization`, implemented identically in `ZImagePipeline` and `ZImageControlPipeline`.

**Work**

1. **Request surface**

* Extend `ZImageGenerationRequest`:

  * `cfgTruncation: Float?`
  * `cfgNormalization: Bool`
  * optional `cfgNormalizationFactor: Float?` (advanced override)

2. **Implement in both denoising loops**

* Apply truncation by turning guidance to 0 after threshold; gate CFG computation accordingly; implement normalization as specified.

3. **Deduplicate logic**

* Add a shared helper (e.g. `PipelineUtilities.applyCFG(...)`) to avoid drift between the two pipelines.

**Tests**

* Unit tests for CFG gating behavior (pure tensor math with small shapes; no real weights required).
* If feasible: golden/approx tests for “normalization clamps norm” property.

**Gate**

* Turbo path with guidance=0 unaffected.
* New knobs functionally exercised by unit tests.
* No behavior regression when running existing integration tests (opt-in).

---

## Phase 5 — CLI presets, docs, and integration coverage (Base defaults + “definition of done”)

**Goal:** Ship the “first-class Base experience”: correct defaults when `--model Tongyi-MAI/Z-Image` is selected, plus docs/examples and an opt-in integration test.

**Work**

1. **CLI: preset-default application**

* Track whether user explicitly set `--steps/--guidance/...`, and only apply model preset defaults when flags are absent (plan requirement).

2. **Docs**

* Update the concrete docs where users look for behavior:

  * `docs/CLI.md` (flags/help/examples)
  * `docs/MODELS_AND_WEIGHTS.md` (model selection, caching, weightsVariant semantics)
* Add CLI examples for Turbo and Base; document new knobs (`--cfg-truncation`, `--cfg-normalization`, `--weights-variant`), plus step semantics note.

3. **Integration tests**

* Add Base model integration test, **skipped in CI** (env-var gated), using conservative settings (e.g., 512px, modest steps) to keep it usable locally.

**Gate = Definition of done**

* `ZImageCLI -m Tongyi-MAI/Z-Image` works with sensible defaults; Turbo unchanged; step semantics verified; CFG behaviors match reference logic; weight resolution deterministic; tests cover config + semantics.

---

## Suggested PR breakdown (practical sequencing)

1. **PR1:** Phase 0 (fixtures + unit test scaffolding)
2. **PR2:** Phase 1 (model registry + cache naming + default propagation + areZImageVariants)
3. **PR3:** Phase 2 (weightsVariant resolution + downloads + guardrails + unit tests)
4. **PR4:** Phase 3 (scheduler terminal sigma override + step semantics tests; optional dt==0 skip as PR4b)
5. **PR5:** Phase 4 (CFG truncation/normalization + shared helper + unit tests)
6. **PR6:** Phase 5 (CLI preset behavior + docs/examples + opt-in Base integration test)
