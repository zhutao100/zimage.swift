# Z-Image.swift runtime precision parity report

This report summarizes the current precision-parity status of the Swift + MLX implementation against the directly relevant Diffusers Z-Image pipelines.

Validation status: hardened on March 7, 2026 against the current `zimage.swift` tree, the local Diffusers checkout at commit `e1b5db52bda85d47a4f8f75954f77e672a8f7f1c`, and the checked-out `mlx-swift` 0.30.6 sources under `.build/checkouts/mlx-swift/`.

Hardening notes:

- Statements in the **confirmed parity** and **confirmed mismatch** sections are intended to be source-backed, not inferred from examples alone.
- Statements about MLX mixed-dtype kernel behavior remain hypotheses unless they are explicitly documented by MLX or directly measured.
- The text-mask mismatch is a confirmed Swift-vs-Diffusers difference, and MLX itself now appears to support both boolean and additive SDPA masks; this means the current Swift additive-mask path is an implementation choice, not an obvious MLX limitation.
- Line numbers in upstream Diffusers may drift across revisions; the file-level behavior was revalidated for the local commit above before this report was updated.

It intentionally separates findings into three buckets:

1. **confirmed parity** — directly supported by current source code on both sides
2. **confirmed mismatch** — directly supported code-level divergence
3. **runtime hypothesis requiring measurement** — plausible runtime effect not fully pinned down by the current source or public backend docs

## Scope

### Swift paths inspected

- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Pipeline/PipelineUtilities.swift`
- `Sources/ZImage/Pipeline/FlowMatchScheduler.swift`
- `Sources/ZImage/Weights/ZImageWeightsMapper.swift`
- `Sources/ZImage/Model/Transformer/ZImageTimestepEmbedder.swift`
- `Sources/ZImage/Model/Transformer/ZImageTransformer2D.swift`
- `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift`
- `Sources/ZImage/Model/Transformer/ZImageRopeEmbedder.swift`
- `Sources/ZImage/Model/Transformer/ZImageSelfAttention.swift`
- `Sources/ZImage/Model/Transformer/ZImageAttentionUtils.swift`
- `Sources/ZImage/Model/TextEncoder/TextEncoder.swift`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`

### Diffusers parity targets

- `src/diffusers/pipelines/z_image/pipeline_z_image.py`
- `src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py`
- `src/diffusers/pipelines/z_image/pipeline_z_image_controlnet_inpaint.py`
- `src/diffusers/models/transformers/transformer_z_image.py`
- `src/diffusers/models/controlnets/controlnet_z_image.py`

### External backend references used only for runtime-hypothesis framing

- MLX data types doc: default floating dtype is `float32`
- MLX `random.normal`: default output dtype is `float32`
- MLX fast SDPA: softmax runs in `float32` regardless of input precision
- MLX fast SDPA source/docs: array masks may be boolean or additive
- PyTorch SDPA: bool masks are supported; math backend keeps intermediates in `torch.float` for `torch.half` / `torch.bfloat16`

See the references section at the end for links.

---

## Confirmed parity

### 1) Standard non-quantized Swift weight loading is BF16-oriented

The standard Swift loaders default to `.bfloat16`:

- `loadAll(dtype: DType? = .bfloat16)`
- `loadTextEncoder(dtype: DType? = .bfloat16)`
- `loadTransformer(dtype: DType? = .bfloat16)`
- `loadVAE(dtype: DType? = .bfloat16)`
- `loadControlnetWeights(from:dtype:)` defaults to `.bfloat16`

Evidence:

- `Sources/ZImage/Weights/ZImageWeightsMapper.swift:25-30`
- `Sources/ZImage/Weights/ZImageWeightsMapper.swift:33-50`
- `Sources/ZImage/Weights/ZImageWeightsMapper.swift:68-88`

The base pipeline also forces BF16 for transformer override loading:

- `Sources/ZImage/Pipeline/ZImagePipeline.swift:310-323`

Diffusers example usage is also BF16-oriented for Z-Image/Turbo:

- `pipeline_z_image.py` example: `torch_dtype=torch.bfloat16`
- `pipeline_z_image_controlnet.py` examples: `torch_dtype=torch.bfloat16`
- `pipeline_z_image_controlnet_inpaint.py` examples: `torch_dtype=torch.bfloat16`

### 2) Latent state initialization is FP32-oriented, and scheduler state remains in latent dtype

Swift initializes base/control latents via `MLXRandom.normal(...)` without a dtype override:

- base pipeline: `Sources/ZImage/Pipeline/ZImagePipeline.swift:786-791`
- control pipeline: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:907-910`

MLX documents the default floating dtype as `float32`, and `random.normal` also defaults to `float32`, so these latent tensors are effectively FP32 unless explicitly recast later.

The scheduler then preserves the latent/sample dtype:

- `Sources/ZImage/Pipeline/FlowMatchScheduler.swift:52-55`

Diffusers likewise keeps scheduler latents in `torch.float32`:

- `pipeline_z_image.py:449-462, 561-562`
- `pipeline_z_image_controlnet.py:568-577, 693-694`
- `pipeline_z_image_controlnet_inpaint.py:590-599, 715-716`

### 3) VAE decode is explicitly normalized to VAE dtype before decode

Swift decode path:

- `PipelineUtilities.decodeLatents(...)` casts to `vae.dtype` when needed
- `AutoencoderKL.decode(...)` / `AutoencoderDecoderOnly.decode(...)` cast scalar scaling factors to the input dtype

Evidence:

- `Sources/ZImage/Pipeline/PipelineUtilities.swift:34-47`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift:440-447`
- `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift:28-35`

Diffusers follows the same pattern:

- base pipeline: `latents = latents.to(self.vae.dtype)` before decode
- control pipeline: same
- controlnet inpaint pipeline: same

### 4) Control / inpaint VAE preprocessing is VAE-dtype-oriented

Swift control-context construction uses `vae.dtype` for control/inpaint image preparation and keeps scale/shift math in the latent dtype:

- resized control image uses `dtype: vaeDType`
- resized inpaint image uses `dtype: vaeDType`
- shift and scale factors are cast to `latents.dtype`
- final control context is cast to `vae.dtype`

Evidence:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:357-371`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:433-491`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:876-887`

Diffusers control and controlnet-inpaint pipelines also prepare these images in `self.vae.dtype` before VAE encode.

### 5) Both stacks intentionally use higher precision in at least some attention internals

Swift attention code uses `MLXFast.scaledDotProductAttention(...)` in the transformer, VAE attention, and text encoder:

- `Sources/ZImage/Model/Transformer/ZImageSelfAttention.swift`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Model/TextEncoder/TextEncoder.swift`

MLX documents that fast SDPA performs the **softmax in `float32` regardless of input precision**.

Diffusers uses PyTorch SDPA / attention implementations that also explicitly document higher-precision math in the math backend for half / BF16 inputs.

This is not identical implementation, but it is parity in the important design sense that neither stack is “pure BF16 end-to-end” through attention.

---

## Confirmed mismatch

### 1) Swift does not explicitly cast denoiser latents to transformer/controlnet runtime dtype before forward

This is the clearest precision-parity gap.

#### Swift base pipeline

At each denoising step:

- `latents` are optionally duplicated for CFG
- `transformer.forward(latents: modelLatents, ...)` is called directly
- there is **no explicit** `modelLatents.asType(transformer.dtype)` boundary cast

Evidence:

- `Sources/ZImage/Pipeline/ZImagePipeline.swift:821-840`

#### Swift control pipeline

The same applies to both `controlnet.forward(...)` and `transformer.forward(...)`:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:969-990`

#### Diffusers reference

Diffusers explicitly casts latents to model dtype on every denoising step while preserving FP32 scheduler state:

- base pipeline: `pipeline_z_image.py:515-520`
- control pipeline: `pipeline_z_image_controlnet.py:634-640`
- controlnet inpaint: `pipeline_z_image_controlnet_inpaint.py:656-662`

### 2) Swift does not explicitly cast timestep features to timestep-MLP dtype before the MLP

Swift timestep embedder:

- derives the embedding dtype from `timesteps.dtype`
- computes sin/cos features in that dtype
- feeds `tFreq` directly into `mlp.0`

Evidence:

- `Sources/ZImage/Model/Transformer/ZImageTimestepEmbedder.swift:27-52`

Swift later casts `tEmb` to `image.dtype`, but **after** the timestep MLP:

- `Sources/ZImage/Model/Transformer/ZImageTransformer2D.swift:272-273`
- `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift:379-381`

Diffusers explicitly casts timestep features to the first MLP weight dtype or compute dtype before the MLP:

- `transformer_z_image.py:65-70`
- `controlnet_z_image.py:67-72`

### 3) RoPE table construction and rotary application are lower-precision / differently implemented in Swift

Swift RoPE precompute:

- uses `float32` indices / timesteps
- stores real-valued `[cos, sin]` tables
- applies rotary in real-valued arithmetic

Evidence:

- `Sources/ZImage/Model/Transformer/ZImageRopeEmbedder.swift:22-39`
- `Sources/ZImage/Model/Transformer/ZImageAttentionUtils.swift`

Diffusers RoPE path:

- precomputes frequencies using `torch.float64`
- forms `complex64` frequency tables
- applies rotary on `x_in.float()` and casts back

Evidence:

- `transformer_z_image.py:113-121, 331-334`
- `controlnet_z_image.py:116-124, 308-311`

### 4) Text-encoder attention-mask representation differs

Swift text encoder converts the token mask into an additive mask in hidden-state dtype using `-inf`:

- `paddingMask = attentionMask.asType(h.dtype)`
- `negInf = MLXArray(-Float.infinity).asType(h.dtype)`
- returns `.array(combinedMask)`

Evidence:

- `Sources/ZImage/Model/TextEncoder/TextEncoder.swift:421-445`

Diffusers prompt path converts the tokenizer attention mask to `bool`:

- `pipeline_z_image.py:233-239`
- `pipeline_z_image_controlnet.py:287-293`
- `pipeline_z_image_controlnet_inpaint.py:294-300`

PyTorch SDPA accepts boolean masks directly, so this is a real implementation-level divergence.

MLX also documents/supports boolean SDPA masks, so the current additive-mask path in `TextEncoder.swift` is not the only viable representation on the Swift side.

### 5) `weightsVariant` is not a full runtime precision selector in Swift

Swift `weightsVariant` influences which files are selected, but standard loaders still default to runtime BF16 unless the caller explicitly overrides dtype.

Evidence:

- `Sources/ZImage/Weights/ZImageWeightsMapper.swift:33-50, 68-88`

This is not directly equivalent to common Diffusers usage where `torch_dtype=...` is the primary runtime precision selector.

---

## Runtime hypothesis requiring measurement

These items are important, but they require profiling or targeted probes to move from “likely” to “confirmed runtime behavior.”

### 1) Effective compute dtype at transformer/controlnet ingress in MLX

Because Swift currently feeds likely-FP32 latents into BF16-weight modules without an explicit boundary cast, the exact runtime behavior depends on MLX mixed-dtype kernel semantics.

What is confirmed:

- latent tensors start FP32 unless recast
- transformer/controlnet ingress cast is absent
- weights are BF16 in standard non-quantized paths

What is **not** currently pinned down by source or public docs:

- whether `MLXNN.Linear` / underlying GEMM runs compute in BF16, FP32, or some mixed mode when activations are FP32 and weights are BF16
- whether the visible output dtype at that boundary stays FP32 or is narrowed earlier than expected

### 2) Effective compute dtype inside the timestep MLP

The same open question applies to `ZImageTimestepEmbedder`:

- timestep features are built in `timesteps.dtype`
- no explicit cast is applied before `mlp.0`
- exact mixed-dtype compute behavior depends on MLX layer/kernel rules

### 3) Practical impact of the RoPE precision gap

The code-level mismatch is confirmed, but the user-visible effect is not yet measured.

Requires measurement to determine:

- whether the FP32-table / real-valued Swift RoPE path causes materially different denoiser outputs from Diffusers
- whether any difference is negligible relative to ordinary denoiser stochasticity / scheduler sensitivity

### 4) Kernel selection and memory consequences of additive float masks in the Swift text encoder

The representation mismatch is confirmed, but the backend/runtime impact is still a hypothesis.

Requires measurement to determine:

- whether additive hidden-dtype masks materially change memory behavior or kernel selection versus Diffusers-style boolean masks
- whether the Swift masking form blocks any optimized path in MLX fast attention for this workload

### 5) Quantified effect of the missing explicit denoiser ingress cast on parity, memory, and throughput

The missing cast is a confirmed mismatch. The **size** of its impact is still a runtime question.

Requires measurement to determine:

- memory delta
- tokens/sec or step/sec delta
- image/output parity delta
- whether explicit casts reduce mixed-dtype graph growth or intermediate allocation pressure

---

## Suggested measurement plan

To resolve the runtime-hypothesis bucket, prioritize these probes.

### Probe A — explicit latent ingress cast A/B

Add an experiment branch that inserts:

```swift
modelLatents = modelLatents.asType(transformer.dtype)
```

before:

- `transformer.forward(...)` in `ZImagePipeline`
- `controlnet.forward(...)` and `transformer.forward(...)` in `ZImageControlPipeline`

Compare against baseline for:

- peak resident memory
- MLX/Metal kernel trace
- step latency
- output drift with fixed seed

### Probe B — explicit timestep-MLP ingress cast A/B

Add an experiment branch inside `ZImageTimestepEmbedder.callAsFunction`:

```swift
let target = mlp.0.weight.dtype
let tFreqTyped = tFreq.dtype == target ? tFreq : tFreq.asType(target)
```

and compare memory / latency / output drift.

### Probe C — RoPE parity probe

For a fixed latent sample and fixed prompt embeddings:

- dump Swift attention inputs / outputs around RoPE
- compare against the Python reference for the same tensors
- record max / mean absolute error before and after rotary application

### Probe D — text-mask representation probe

A/B compare:

- current additive hidden-dtype mask
- bool mask path if one can be threaded through the MLX attention wrapper

Measure both memory and latency during prompt encoding.

The executable phase plan for the first three fixes now lives in `docs/dev_plans/runtime_precision_parity_improvement_plan.md`.

---

## Practical interpretation

The current Swift implementation is **not globally “wrong” on precision**. The repo already matches Diffusers in several important places:

- BF16-oriented standard weight loading
- FP32-oriented latent/scheduler state
- VAE-dtype encode/decode boundaries
- higher-precision attention internals where the backend provides them

The main precision-parity problems are concentrated at **denoiser ingress and timestep ingress**, with RoPE and mask representation as secondary implementation differences.

If a single first fix is needed for tighter parity, it should be:

1. keep scheduler state FP32
2. explicitly cast denoiser inputs to transformer/controlnet dtype at the forward boundary
3. separately test whether timestep-MLP ingress should also be explicitly normalized

---

## References

### Repo source paths

- `Sources/ZImage/Weights/ZImageWeightsMapper.swift`
- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Pipeline/PipelineUtilities.swift`
- `Sources/ZImage/Pipeline/FlowMatchScheduler.swift`
- `Sources/ZImage/Model/Transformer/ZImageTimestepEmbedder.swift`
- `Sources/ZImage/Model/Transformer/ZImageTransformer2D.swift`
- `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift`
- `Sources/ZImage/Model/Transformer/ZImageRopeEmbedder.swift`
- `Sources/ZImage/Model/TextEncoder/TextEncoder.swift`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`

### Diffusers reference paths

- `src/diffusers/pipelines/z_image/pipeline_z_image.py`
- `src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py`
- `src/diffusers/pipelines/z_image/pipeline_z_image_controlnet_inpaint.py`
- `src/diffusers/models/transformers/transformer_z_image.py`
- `src/diffusers/models/controlnets/controlnet_z_image.py`

### External docs

- MLX data types: <https://ml-explore.github.io/mlx/build/html/python/data_types.html>
- MLX `random.normal`: <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.normal.html>
- MLX fast SDPA: <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html>
- MLX fast SDPA Python binding source (bool/additive mask contract): `.build/checkouts/mlx-swift/Source/Cmlx/mlx/python/src/fast.cpp`
- MLX fast SDPA tests (bool-mask equivalence probe): `.build/checkouts/mlx-swift/Source/Cmlx/mlx/python/tests/test_fast_sdpa.py`
- PyTorch SDPA: <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>
