### Key takeaways (Apple Silicon reality check)

* **FP16 is the “default fast path”** across Apple GPU stacks (Metal/MPS, PyTorch MPS, MLX, Core ML GPU/ANE). Core ML specifically runs GPU+ANE segments in **Float16** for best performance in its common execution modes. ([Apple Developer][1])
* **BF16 is real on Apple now, but it’s OS-/stack-gated.** Apple added **bfloat16 support in MPSGraph starting macOS 14 (Sonoma)**. ([Apple Developer][2]) PyTorch’s MPS BF16 support is also **macOS 14+ gated**.
* **FP8: assume “not supported” on Apple GPU stacks today** (Metal/MPS/PyTorch-MPS/MLX/Core ML). In practice you’ll hit explicit “float8 not supported on MPS” errors. ([GitHub][3])
* **INT8 speedups are mostly via Core ML (ANE) or framework-specific quantized kernels, not “generic INT8 GPU GEMM.”** Core ML tooling explicitly calls out **int8–int8 compute path on newer hardware (A17 Pro, M4)** for W8A8 on ANE. ([Apple GitHub][4]) MLX provides its own quantized matmul path (weights packed into integers + per-group scales/bias). ([ml-explore.github.io][5])

**Versions referenced (as of Feb 11, 2026):**

* PyTorch **2.10** (released **Jan 21, 2026**). ([PyTorch][6])
* MLX **0.30.6** (GitHub release dated **Feb 6, 2026**). ([ml-explore.github.io][5])
* coremltools **9.0** (released **Nov 10, 2025**). ([PyPI][7])
* macOS notes: **macOS 14 (Sonoma)** is the key pivot point for BF16 in Apple’s MPSGraph story. ([Apple Developer][2])

---

## 1) Precision format deep dive

### FP32 (IEEE 754 single)

**Definition / spec**

* **Bit layout:** 1 sign, 8 exponent (bias 127), 23 fraction.
* **Dynamic range:** ~1e−38 to ~3e38 (normal); has subnormals; NaN/Inf supported.
* **Common DL kernel behavior:** inputs FP32; accumulation FP32 (obviously). Many “mixed precision” paths still keep *some* accumulators/reductions in FP32 even when inputs are narrower.

**Strengths**

* Robust numerics: stable for softmax, layernorm, reductions, long-sequence attention, and training optimizer states.
* Lowest overflow/underflow risk.

**Weaknesses**

* 2× memory vs FP16/BF16; higher bandwidth pressure; typically slower on tensor-accelerated paths.

**Primary use cases**

* **Training:** master weights, optimizer states, many reduction-heavy ops, debugging.
* **Inference:** “reference” or when stability dominates speed; some logits/reductions.

**Practical gotchas**

* “FP32 everywhere” often fails to be fast on Apple GPU paths that are optimized for FP16 (and sometimes BF16). You can end up bottlenecked by bandwidth and kernel choice rather than compute.

---

### FP16 (IEEE 754 half)

**Definition / spec**

* **Bit layout:** 1 sign, 5 exponent (bias 15), 10 fraction.
* **Range:** max finite **65504**; min normal **~6.10e−5**; subnormals down to **~5.96e−8**; NaN/Inf supported.
* **DL kernel behavior:** very commonly **FP16 inputs with FP32 accumulation** in matmul/conv; reductions may upcast.

**Strengths**

* Big speed + bandwidth wins; best-supported “fast dtype” on Apple GPU stacks.
* Works well for inference and mixed-precision training with loss scaling.

**Weaknesses**

* Small exponent range → overflow/underflow risk (softmax exponentials, variance computations).
* Sensitive in long-context attention if you keep too much in FP16 (esp. QKᵀ logits, softmax stats).

**Primary use cases**

* **Inference:** weights/activations/KV-cache (common).
* **Training (Apple / non-Apple):** AMP with FP32 master weights and loss scaling.

**Practical gotchas**

* Silent casts: frameworks may keep “model weights FP16” but do internal FP32 for stability (good) or bounce between CPU/GPU (bad).
* FP16 **KV cache** can be a major memory driver for long context; weight quantization doesn’t help KV.

---

### BF16 (bfloat16)

**Definition / spec**

* **Bit layout:** 1 sign, 8 exponent (bias 127), 7 fraction.
* **Range:** **similar to FP32** (same exponent width); much coarser precision than FP16.
* **DL kernel behavior:** like FP16, BF16 often uses **FP32 accumulation** in GEMM/conv.

**Strengths**

* Great for training stability: reduced overflow vs FP16 due to larger exponent range.
* Often better “drop-in” than FP16 for models with wide activation distributions.

**Weaknesses**

* Only 7 fraction bits → precision is coarse; can hurt when tiny deltas matter (some normalization/low-variance regimes).
* Support on Apple is newer and more conditional than FP16.

**Primary use cases**

* **Training:** activations/gradients with FP32 master weights (common mixed recipe in CUDA world).
* **Inference:** sometimes preferred over FP16 when overflow is the failure mode (but only if the stack truly supports it efficiently).

**Practical gotchas**

* “BF16 supported” often means “some ops / some OS / some device.” On Apple specifically, BF16 enablement is tied to **macOS 14+ + stack support** (MPSGraph/MLX/PyTorch MPS). ([Apple Developer][2])

---

### FP8 (E4M3, E5M2, vendor variants)

**Definition / spec**

* Two common families:

  * **E4M3**: 4 exponent bits, 3 mantissa bits (often with vendor-specific NaN/Inf handling; e.g., “finite-only” variants).
  * **E5M2**: 5 exponent bits, 2 mantissa bits (larger range, lower precision).
* **Typical behavior in DL:** FP8 is usually used with **scaling** (per-tensor/per-channel) and **FP16/BF16/FP32 accumulation**, often in transformer GEMMs.

**Strengths**

* Large bandwidth + compute wins where hardware supports it (notably on NVIDIA tensor cores).
* Can be strong for training throughput with appropriate scaling recipes.

**Weaknesses**

* Very low precision; requires careful scaling, and some ops (softmax/norms/reductions) typically stay in higher precision.

**Primary use cases**

* **Training:** GEMM-heavy layers with scaling and higher-precision accumulators.
* **Inference:** more niche; often INT8/INT4 weight quantization is the practical path instead.

**Practical gotchas**

* FP8 “exists” in a framework doesn’t mean it works on your device backend. On Apple GPU backends (MPS/MLX), you should assume **no FP8 execution support** today. ([GitHub][3])

---

### INT8 (quantization)

**Definition / spec (DL-relevant)**

* **Representation:** integer values with associated scale (and optionally zero-point).

  * **Symmetric:** real ≈ scale × int8 (zero-point = 0). Common for weights; simpler and often faster.
  * **Asymmetric:** real ≈ scale × (int8 − zero_point). Useful for activations when distributions aren’t zero-centered.
* **Granularity:** per-tensor vs per-channel (weights often per-output-channel); activations often per-tensor or per-group.
* **Compute:** can be

  * **true int8×int8 dot-product accumulation** (fastest when hardware supports), or
  * **“fake quant” / dequantize→float compute→requantize** (bandwidth saving sometimes, but not the same win).

**Strengths**

* Big memory + bandwidth wins; can be high-performance if mapped to real int8 compute units.
* Often best ROI for inference deployment.

**Weaknesses**

* Accuracy risk if calibration/scales are poor; outliers hurt.
* Quantized operator coverage is backend-dependent and often incomplete.

**Primary use cases**

* **Inference:** weights (W8) and sometimes activations (A8) when supported (e.g., W8A8 on ANE/newer chips).
* **Training:** QAT or PTQ workflows produce the quantized model; actual training typically higher precision.

**Practical gotchas**

* Many toolchains “support INT8” as a storage format but still compute in FP16/FP32 unless you hit the right backend and patterns.

---

## 2) Apple Silicon acceleration reality check (by stack)

### A) Metal / MPS kernels (MPSGraph in practice)

**API-exposed dtypes**

* Apple states **MPSGraph adds bfloat16 support starting macOS 14 (Sonoma)**. ([Apple Developer][2])
* Apple also highlights **integer quantization (8-bit)** support in the same MPSGraph context (quantize/dequantize + fusion patterns). ([Apple Developer][2])

**What is actually accelerated**

* **FP16:** reliably GPU-accelerated across core ops (GEMM/conv/etc.)—this is the primary fast path.
* **BF16:** supported in MPSGraph on **macOS 14+**; whether a specific op uses BF16 end-to-end can vary, but the intent is GPU acceleration (Apple positions it as a first-class MPSGraph dtype). ([Apple Developer][2])
* **FP32:** GPU-accelerated for many ops, but may be slower than FP16 paths and sometimes avoided by runtimes unless you force it.
* **INT8:** “supported” mostly as **graph-level quantization** (memory/throughput improvements depend on op fusion and backend). ([Apple Developer][2])
* **FP8:** no evidence of Metal/MPSGraph FP8 execution dtype; assume not supported.

**Core ops (typical)**

* Accelerated: matmul/GEMM, conv, many elementwise ops, many reductions.
* Numerically sensitive ops (softmax/layernorm) commonly run in higher precision internally even when inputs are FP16/BF16.

**Version notes**

* BF16: macOS 14+ pivot. ([Apple Developer][2])

---

### B) PyTorch MPS backend (PyTorch 2.10)

**API-exposed dtypes**

* PyTorch 2.10 is current and released Jan 21, 2026. ([PyTorch][6])
* For BF16 on MPS, PyTorch explicitly gates support: **“MPS BFloat16 is only supported on MacOS 14 or newer.”**

**What is actually accelerated**

* **FP16 / FP32:** broadly accelerated where ops are implemented in MPS.
* **BF16:** supported on **macOS 14+**, but expect **op-by-op coverage** and possible fallbacks for unsupported kernels.
* **FP8:** not supported on MPS; users hit explicit runtime errors when float8 tensors reach the MPS backend. ([GitHub][3])
* **INT8:** PyTorch’s classic quantized operator backends are **CPU-oriented** (FBGEMM on x86, QNNPACK on ARM). ([PyTorch][8]) In other words: INT8 is “supported” in PyTorch, but not as a general MPS-accelerated inference path; for Apple-hardware INT8 you usually route through Core ML export.

**Core ops (reality)**

* GEMM/conv/elementwise are the strongest coverage.
* Attention/transformer stacks vary; newer ops may still lack MPS kernels and fall back.

**Version notes**

* BF16 requires macOS 14+ (Sonoma).

---

### C) MLX (Apple MLX stack)

**API-exposed dtypes**

* MLX supports **float32, float16, bfloat16**, and integer types; **float64 is CPU-only**. ([ml-explore.github.io][9])
* MLX 0.30.6 is a Feb 2026 release (fresh). ([ml-explore.github.io][5])

**What is actually accelerated**

* **FP16 / FP32 / BF16:** MLX is designed for Apple silicon accelerators; these are first-class dtypes (with the float64 CPU-only exception). ([ml-explore.github.io][9])
* **FP8:** MLX maintainers explicitly note no plans / not likely for FP8. ([Reddit][10])
* **INT8 (and lower-bit) in MLX is typically “weight quantization with specialized kernels,”** not “store tensors as int8 and run generic int8 ops.”

**Quantized matmul in MLX (implementation-oriented)**

* `mlx.core.quantized_matmul(x, w, scales, biases, group_size, bits)`:

  * **Weights `w` are packed into unsigned integers (uint32)**; each weight element uses `bits` bits.
  * Uses **one floating-point scale + bias per `group_size` weight elements** (groupwise affine dequantization).
  * The matmul is computed against the *quantized* packed weights directly (i.e., a dedicated kernel path). ([ml-explore.github.io][5])
* `mlx.nn.quantize(model, group_size, bits)` will quantize leaf modules that implement `to_quantized(...)` (notably Linear and Embedding). ([ml-explore.github.io][11])
* In MLX Swift, the `quantized(...)` matmul API documents that **inputs can be quantized on the fly**, and weights are used as-is if already quantized (otherwise quantized on the fly). ([The Swift Package Index][12])

**Core ops**

* GEMM/matmul and quantized matmul are central and well-supported.
* As with other stacks, some reductions/softmax/layernorm may use higher-precision internal math.

---

### D) Core ML (tools + runtime)

**API-exposed precision model**

* Core ML “typed execution” is essentially **float16 vs float32** for compute precision. ([Apple GitHub][13])
* For neural-network execution, Apple explicitly states:

  * **GPU and Neural Engine use Float16**
  * **CPU uses Float32**
  * And Core ML partitions graphs across compute units dynamically unless constrained. ([Apple Developer][1])

**What is actually accelerated**

* **FP16:** fast path on GPU/ANE. ([Apple Developer][1])
* **FP32:** CPU path (and selected GPU float32 ops exist, but “default best perf” uses float16 segments). ([Apple Developer][1])
* **INT8:** Core ML supports quantization workflows; coremltools explicitly calls out:

  * **W8A8 benefits on Neural Engine** by leveraging **faster int8–int8 compute on newer hardware (A17 Pro, M4)**. ([Apple GitHub][4])
* **BF16 / FP8:** not presented as Core ML execution precisions in the public tooling model; assume not supported as compute precisions. ([Apple GitHub][13])

**Core ops**

* Core ML is graph-compiled; operator availability and fusions matter more than per-op dtype control.
* For LLMs, you typically use ML Programs and Apple’s recommended patterns; precision is constrained to fp16/fp32 and quantized-weight modes.

---

## 3) Support matrix (Apple Silicon)

Legend: **A** = Supported + accelerated, **S** = Supported but not accelerated / fallback-limited, **N** = Not supported.

| Format   | Metal/MPS kernels                                                                          | PyTorch MPS (2.10)                                                                         | MLX (0.30.6)                                                                                             | Core ML (CPU / GPU / ANE)                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **FP32** | **A** (common, but slower than FP16 paths in many kernels)                                 | **A**                                                                                      | **A**                                                                                                    | **CPU: A (fp32)** / GPU+ANE: usually **fp16** segments ([Apple Developer][1])                                                      |
| **FP16** | **A** (primary fast path)                                                                  | **A**                                                                                      | **A**                                                                                                    | **GPU: A (fp16)** / **ANE: A (fp16)** ([Apple Developer][1])                                                                       |
| **BF16** | **A/S** (MPSGraph BF16 on macOS 14+; op-dependent) ([Apple Developer][2])                  | **A/S** (macOS 14+ gated; op-dependent)                                                    | **A** (dtype supported; float64 CPU-only) ([ml-explore.github.io][9])                                    | **N** (Core ML execution precision is fp16/fp32) ([Apple GitHub][13])                                                              |
| **FP8**  | **N**                                                                                      | **N** (explicit MPS backend errors) ([GitHub][3])                                          | **N** (no FP8 plan) ([Reddit][10])                                                                       | **N**                                                                                                                              |
| **INT8** | **S** (graph quantize/dequantize + fusion; not “generic int8 GEMM”) ([Apple Developer][2]) | **S** (PyTorch quantized ops are CPU backends; not a general MPS int8 path) ([PyTorch][8]) | **A** (quantized matmul uses packed integer weights + per-group scales/bias) ([ml-explore.github.io][5]) | **CPU: S/A** (depends) / **GPU: S/A** (e.g., per-block int4 weights noted) / **ANE: A** for W8A8 on A17 Pro/M4 ([Apple GitHub][4]) |

(Cells cite the primary sources used to classify support/acceleration; see MPSGraph BF16 + int8 quantization notes, PyTorch macOS-14 BF16 gate, MLX quantized matmul definition, and Core ML fp16/fp32 typed execution + int8-int8 path callouts. ([Apple Developer][2]))

---

## 4) Decision guidance (actionable)

### LLM inference (weights / activations / KV cache)

**Best practical options on Apple today**

* **MLX path (developer-centric, flexible):**

  * Start with **FP16 or BF16 activations** (BF16 if you see overflow/instability and you’re on macOS 14+). ([ml-explore.github.io][9])
  * Use **weight quantization** (often 4-bit or 8-bit) via MLX quantized layers; matmuls go through `quantized_matmul`-style kernels. ([ml-explore.github.io][5])
  * Keep **KV cache in FP16** (memory dominates; Apple stacks don’t give you FP8 KV-cache tricks).
* **Core ML path (deployment-centric, ANE/GPU optimized):**

  * Expect **fp16 execution** on GPU/ANE for most segments. ([Apple Developer][1])
  * If you can use **W8A8 on ANE** (especially on **A17 Pro / M4** class hardware), it can be materially faster due to int8–int8 compute. ([Apple GitHub][4])
  * Don’t plan on BF16/FP8 in Core ML compute precision.

**When to prefer BF16 over FP16 (Apple-specific)**

* Prefer **BF16** when you hit FP16 overflow/underflow (attention logits, variance/normalization instabilities) *and* your stack/backend is confirmed BF16-enabled (macOS 14+ for MPSGraph/PyTorch-MPS; MLX supports BF16 as a dtype). ([Apple Developer][2])
* Prefer **FP16** when you’re throughput-bound and stable (most inference), or when the backend is fp16-optimized (Core ML GPU/ANE). ([Apple Developer][1])

### CNN / vision inference

* **Core ML** is often the simplest fast deployment route:

  * GPU/ANE fp16 is the common best-perf mode. ([Apple Developer][1])
  * INT8/W8A8 gains are most convincing when targeting ANE and supported hardware generations (A17 Pro/M4). ([Apple GitHub][14])
* **PyTorch MPS** is good for development, but:

  * treat it as **fp16/fp32 primarily**, with BF16 only if your macOS/backend supports it.

### Training on Apple Silicon (optional, realism)

* **FP16 mixed precision** is the most broadly viable; BF16 is improving but still backend/OS-gated. ([Apple Developer][2])
* **FP8 training**: not an Apple GPU story today; don’t plan on it. ([GitHub][3])

---

## Common misconceptions to avoid (Apple-specific)

* **“BF16 is universally accelerated like on NVIDIA.”** Not on Apple: BF16 enablement is tied to **macOS 14+ and specific stacks**; Core ML’s public precision model is still fp16/fp32. ([Apple Developer][2])
* **“FP8 is available if the framework has float8 tensors.”** On Apple backends you should assume FP8 will error or fall back; MPS explicitly rejects float8 in practice. ([GitHub][3])
* **“INT8 in PyTorch means INT8 on GPU.”** PyTorch’s quantized operator backends are fundamentally CPU backends (FBGEMM/QNNPACK). ([PyTorch][8]) For Apple Silicon “real” INT8 acceleration, plan around **Core ML (ANE)** or **MLX’s quantized kernels**. ([Apple GitHub][4])

[1]: https://developer.apple.com/videos/play/wwdc2021/10038/?utm_source=chatgpt.com "Tune your Core ML models - WWDC21 - Videos"
[2]: https://developer.apple.com/la/videos/play/wwdc2023/10050/?utm_source=chatgpt.com "Optimize machine learning for Metal apps - WWDC23 - Videos"
[3]: https://github.com/Comfy-Org/ComfyUI/issues/8988?utm_source=chatgpt.com "Trying to convert Float8_e4m3fn to the MPS backend but it ..."
[4]: https://apple.github.io/coremltools/docs-guides/source/opt-overview.html?utm_source=chatgpt.com "Overview — Guide to Core ML Tools"
[5]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html?utm_source=chatgpt.com "mlx.core.quantized_matmul — MLX 0.30.4 documentation"
[6]: https://pytorch.org/blog/pytorch-2-10-release-blog/?utm_source=chatgpt.com "PyTorch 2.10 Release Blog"
[7]: https://pypi.org/project/coremltools/?utm_source=chatgpt.com "coremltools"
[8]: https://pytorch.org/blog/quantization-in-practice/?utm_source=chatgpt.com "Practical Quantization in PyTorch"
[9]: https://ml-explore.github.io/mlx/build/html/python/data_types.html?utm_source=chatgpt.com "Data Types — MLX 0.30.6 documentation"
[10]: https://www.reddit.com/r/LocalLLaMA/comments/14pz4v0/apples_metal_is_getting_bfloat16_support/?utm_source=chatgpt.com "Apple's Metal is getting bfloat16 support : r/LocalLLaMA"
[11]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html?utm_source=chatgpt.com "mlx.nn.quantize — MLX 0.30.6 documentation"
[12]: https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/quantized%28_%3Agroupsize%3Abits%3Amode%3Astream%3A%29?utm_source=chatgpt.com "quantized(_:groupSize:bits:mode:stream:) | Documentation"
[13]: https://apple.github.io/coremltools/docs-guides/source/typed-execution.html?utm_source=chatgpt.com "Typed Execution — Guide to Core ML Tools"
[14]: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html?utm_source=chatgpt.com "Performance — Guide to Core ML Tools - Apple"
