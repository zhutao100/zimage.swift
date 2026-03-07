Below is the **“no explicit dtype override”** behavior for **BF16-weight Transformer inference** on Apple Silicon, split into **MLX** vs **PyTorch (MPS backend)**.

I’m pinning references to these versions/docs (current as of **Feb 11, 2026**):

* **MLX docs:** 0.30.6 (Python) ([ml-explore.github.io][1])
* **PyTorch stable:** 2.10.0 (released **Jan 21, 2026**) ([PyPI][2])
* **macOS baseline for BF16 on MPS:** **macOS 14+ (Sonoma+)**; PyTorch MPS now requires macOS 14+ ([GitHub][3])

---

## MLX: BF16 weights, no dtype override

### A) What actually happens (summary)

**Storage dtypes (visible to you):**

* **Weights:** stay **BF16** (assuming checkpoint tensors are BF16).
* **Activations / intermediates:** typically remain **BF16** *if all inputs to each op are BF16* (no type-promotion triggers).
* **KV cache:** whatever the attention implementation uses; commonly matches activation dtype (often BF16 in this scenario).

**Compute / accumulation (what MLX kernels do):**

* **Attention softmax:** if you hit the **fast SDPA path**, softmax is **computed in FP32 regardless of input dtype**. ([ml-explore.github.io][4])
* **RMSNorm:** MLX explicitly documents that the **mean accumulation is done in 32-bit precision** (so effectively FP32 accumulation for the reduction). ([Hugging Face][5])
* **LayerNorm / RMSNorm “fast” kernels:** maintainer guidance indicates fast norm kernels “accumulate in higher precision” (practically: FP32 reductions); treat as **high-confidence but not as formally specified** as the SDPA note above. ([ml-explore.github.io][6])

**Top implicit conversion gotchas (real-world):**

1. **Mask dtype mismatch can upcast attention compute**
   If your attention mask is **float32** while Q/K/V are BF16, you can trigger type promotion / less efficient kernels (and in practice, compute can run at FP32-ish behavior). This is a known footgun called out in MLX discussions. ([The Swift Package Index][7])
2. **Fast attention changes internal precision**
   Even with BF16 Q/K/V, **softmax runs in FP32** in `mx.core.fast.scaled_dot_product_attention`. ([ml-explore.github.io][4])
3. **Default dtype for newly-created floats is float32**
   MLX’s default floating type is `float32`, so constants/intermediates you construct without care can introduce mixed-dtype graphs (and then promotion). ([ml-explore.github.io][1])

---

### B) MLX per-op dtype table (typical Transformer block)

Assume: weights BF16; inputs/activations BF16; mask is **bool** or **BF16** (not float32); and attention uses **fast SDPA** when available.

| Op                         | Input dtypes (typical) | Compute dtype            | Accum dtype                        | Output dtype | Notes                                                                                                             |
| -------------------------- | ---------------------- | ------------------------ | ---------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------- |
| Embedding lookup           | ids int32 + embed BF16 | BF16                     | N/A                                | BF16         | Output follows embedding weight dtype (BF16).                                                                     |
| RoPE                       | BF16                   | BF16                     | N/A                                | BF16         | If you introduce float32 constants, you may trigger promotion. ([ml-explore.github.io][1])                        |
| Q/K/V projections (linear) | BF16 × BF16            | **Usually BF16**         | **Likely FP32** (kernel-dependent) | BF16         | MLX docs don’t explicitly pin GEMM accumulation; validate via probe (below).                                      |
| Attention scores (QKᵀ)     | BF16                   | Usually BF16             | Likely FP32                        | BF16         | Same caveat as GEMM accumulation.                                                                                 |
| Softmax                    | BF16 scores            | **FP32**                 | FP32                               | BF16         | Explicitly documented for **fast SDPA**. ([ml-explore.github.io][4])                                              |
| Attn * V                   | BF16                   | Usually BF16             | Likely FP32                        | BF16         | Kernel-dependent; validate.                                                                                       |
| Output projection          | BF16                   | Usually BF16             | Likely FP32                        | BF16         | Kernel-dependent.                                                                                                 |
| MLP up/gate/down linears   | BF16                   | Usually BF16             | Likely FP32                        | BF16         | Kernel-dependent.                                                                                                 |
| Activation (SiLU/GELU)     | BF16                   | BF16                     | N/A                                | BF16         | If implemented via `exp`/`tanh`, internal math may be higher precision depending on kernel; validate if critical. |
| RMSNorm                    | BF16                   | BF16 (+ FP32 reductions) | **FP32 reductions**                | BF16         | Mean accumulation in 32-bit is documented. ([Hugging Face][5])                                                    |
| Residual adds              | BF16 + BF16            | BF16                     | N/A                                | BF16         | Mixed inputs (BF16 + FP32) will promote. ([ml-explore.github.io][1])                                              |

**Key point:** in MLX, **visible tensor dtypes usually stay BF16**, while **select ops** (softmax in fast SDPA, norm reductions) **use FP32 internally** for stability/perf. ([ml-explore.github.io][4])

---

### C) MLX conversion pipeline (diagram)

```
BF16 checkpoint
   │  (deserialization keeps BF16)
   ▼
BF16 model params
   │  (place/execute on GPU via MLX Metal backend)
   ▼
Lazy graph build
   │
   ├─ type promotion points:
   │    - mixed BF16 + FP32 inputs (masks/constants)
   │    - scalar defaults (float32)
   │
   └─ kernel dispatch:
        - fast SDPA: softmax internally FP32
        - norms: reductions accumulate in 32-bit
   ▼
GPU kernels run (mostly BF16 compute, some FP32 internal)
```

---

### D) Minimal reproducible probes (MLX)

**Probe 1 — show visible dtypes and force fast SDPA**

```python
import mlx.core as mx

# Sanity: default float type is float32 (don’t accidentally mix)
print("default float:", mx.float32)

# Pretend these came from a BF16 checkpoint:
Wq = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
x  = mx.random.normal((1, 128, 4096)).astype(mx.bfloat16)

q = x @ Wq
print("q dtype:", q.dtype)  # expect bfloat16

# Fast SDPA path (if you use it): softmax is FP32 internally (doc’d)
# You won't see FP32 in q/k/v dtypes, but it affects stability/perf.
```

The “softmax is FP32 internally” behavior is explicitly documented for `mx.core.fast.scaled_dot_product_attention`. ([ml-explore.github.io][4])

**Probe 2 — detect dtype-promotion footguns (mask mismatch)**

```python
import mlx.core as mx

scores = mx.random.normal((1, 8, 128, 128)).astype(mx.bfloat16)

mask_fp32 = mx.zeros((1, 1, 1, 128), dtype=mx.float32)
mask_bf16 = mask_fp32.astype(mx.bfloat16)

# If you incorporate mask_fp32 into attention math, you may upcast / lose fast paths.
# Keep masks bool or match compute dtype.
```

This “mask dtype mismatch” class of issue is called out in MLX discussions. ([The Swift Package Index][7])

**Probe 3 — capture GPU trace for kernel-level inspection**

* MLX exposes Metal capture start/stop. ([ml-explore.github.io][8])
* For better debugging labels, build MLX with `MLX_METAL_DEBUG=ON`. ([ml-explore.github.io][9])
  This won’t *automatically* print dtypes, but you can often infer them from kernel specializations / compiled variants in the trace.

---

## PyTorch MPS: BF16 weights, no dtype override

### A) What actually happens (summary)

**Hard constraints / prerequisites:**

* PyTorch MPS now targets **macOS 14+** (Sonoma+) going forward. ([GitHub][3])
* Apple highlights that **MPSGraph adds bfloat16 support starting with macOS Sonoma**. ([Apple Developer][10])

**Storage dtypes (visible to you):**

* If your checkpoint loads parameters as `torch.bfloat16`, and you do `model.to("mps")`, **the tensors remain BF16** *as tensor objects* (you can print `param.dtype`).
* There is **no default autocast** unless you enable it; your scenario explicitly disables that.

**Compute dtypes (what may happen under the hood):**

* The MPS backend can route ops through **MPSGraph** or (for some ops like matmul) optional **Metal kernels**.

  * `PYTORCH_MPS_PREFER_METAL=1` prefers Metal kernels for matmul. ([PyTorch Documentation][11])
* For unsupported ops, you can either:

  * **error**, or
  * **fallback to CPU** if `PYTORCH_ENABLE_MPS_FALLBACK=1`, which changes not just performance but often dtype/accumulation characteristics (CPU kernels may accumulate in FP32, etc.). ([PyTorch Documentation][11])

**Most important implicit “dtype changes” in practice:**

1. **Graph-lowering may insert internal casts**
   Even when tensors are BF16, backends frequently upcast sensitive subgraphs (softmax, norm stats, reductions) for numerical stability. **This is plausible on MPSGraph**, but you must confirm via probes/profiling because it’s not consistently spelled out per-op in public docs.
2. **Fallback paths silently change semantics**
   With CPU fallback enabled, the op runs on CPU with CPU dtype behavior; from the outside you still see BF16 tensors flowing, but the internal math path is different. ([PyTorch Documentation][11])

---

### B) PyTorch MPS per-op dtype table (what you can assert vs what you should probe)

Assume: PyTorch 2.10.0; macOS 14+; `device="mps"`; checkpoint weights BF16; no autocast; no explicit `.to(dtype=...)`.

| Op                         | Visible input dtypes          | Compute dtype                                    | Accum dtype     | Visible output dtype | Notes                                                                                                            |
| -------------------------- | ----------------------------- | ------------------------------------------------ | --------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Embedding                  | ids int64/int32 + weight BF16 | **Probe**                                        | **Probe**       | BF16                 | Kernel coverage can vary; verify with profiler.                                                                  |
| Q/K/V linears (GEMM)       | BF16                          | **BF16 or FP16 (backend-dependent)**             | **Probe**       | BF16                 | Matmul may use MPSGraph or Metal kernels; `PYTORCH_MPS_PREFER_METAL` affects path. ([PyTorch Documentation][11]) |
| Attention softmax          | BF16                          | **Likely FP32 internal** (common), but **probe** | **Likely FP32** | BF16                 | Don’t assume without measurement on MPS.                                                                         |
| LayerNorm / RMSNorm        | BF16                          | **Likely mixed** (FP32 stats) but **probe**      | **Likely FP32** | BF16                 | PyTorch has long-standing concerns around low-precision norm overflow; MPS behavior is backend-specific.         |
| MLP activation (SiLU/GELU) | BF16                          | **Probe**                                        | **Probe**       | BF16                 | Often implemented via exp/tanh; could upcast internally.                                                         |
| Reductions                 | BF16                          | **Probe**                                        | **Probe**       | BF16                 | Reduction accumulation precision is often higher than input dtype.                                               |
| Residual adds              | BF16 + BF16                   | BF16                                             | N/A             | BF16                 | Mixed dtype inputs promote (PyTorch type promotion).                                                             |

**Why so much “probe”?** Because **PyTorch’s public MPS docs focus on availability/coverage**, not a stable per-op contract for internal casting/accumulation, and the backend can change across releases. The correct way to answer “what actually ran” is **profiling + cast detection**, below. ([PyTorch Documentation][12])

---

### C) PyTorch MPS conversion pipeline (diagram)

```
BF16 checkpoint
   │  (torch.load -> bf16 params)
   ▼
BF16 params on CPU
   │  model.to("mps")  (dtype preserved)
   ▼
BF16 tensors on MPS
   │
   ├─ lowering to MPSGraph / Metal kernels
   │    - backend may insert internal casts (op-dependent)
   │
   ├─ optional matmul path selection:
   │    PYTORCH_MPS_PREFER_METAL=1  (matmul prefers Metal kernels)
   │
   └─ unsupported ops:
        - error OR
        - CPU fallback if PYTORCH_ENABLE_MPS_FALLBACK=1
   ▼
Execution
```

Environment controls are documented in PyTorch’s MPS environment variables notes. ([PyTorch Documentation][11])

---

### D) Minimal reproducible probes (PyTorch MPS)

**Probe 1 — catch *visible* casts and CPU fallbacks**

```python
import os
import torch

print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())

# Strongly recommended while investigating:
# 1) Disable fallback first so unsupported ops ERROR (you'll see exactly where coverage breaks)
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# Optional: toggle matmul backend preference (matmul path can change)
# os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"

# Load BF16 checkpoint normally (no dtype override)
# model = ...
# model.eval().to("mps")

# Instrument: forward hooks to log tensor dtypes at module boundaries
def log_io(name):
    def hook(mod, inp, out):
        def dtype(x):
            return getattr(x, "dtype", None)
        print(f"{name}: in={list(map(dtype, inp))} out={dtype(out)}")
    return hook

# Register hooks on key blocks (attn, mlp, norms)
# for n, m in model.named_modules():
#     if any(k in n.lower() for k in ["attn", "mlp", "norm"]):
#         m.register_forward_hook(log_io(n))

# Run one step; look for explicit aten::to/cast in profiler (next probe)
```

**Probe 2 — profiler + Instruments (MPS signposts)**
Apple explicitly calls out that PyTorch nightly builds (and now broadly MPS tooling) can emit **signposts** you can inspect in **Metal System Trace** to see op execution and **CPU fallbacks**. ([Apple Developer][10])
This is the most reliable way to answer “did it actually run on GPU, and where did it fall back?”

**Probe 3 — force fallback to reveal dtype-sensitive gaps**

* Run once with `PYTORCH_ENABLE_MPS_FALLBACK=0` (fail-fast).
* Run again with `PYTORCH_ENABLE_MPS_FALLBACK=1` (and compare outputs / perf).
  This is explicitly supported/documented. ([PyTorch Documentation][11])

---

## Practical conclusions: MLX vs PyTorch MPS for BF16-weight Transformer inference

### MLX (today)

* If your model uses **MLX fast attention**, you get a **clear, documented internal precision rule**: **softmax is FP32** even when inputs are BF16. ([ml-explore.github.io][4])
* Norms (at least **RMSNorm**) document **32-bit accumulation** for mean. ([Hugging Face][5])
* Biggest BF16 “surprises” are **type promotion** (mask/constants) and “fast path” kernel selection. ([The Swift Package Index][7])

### PyTorch MPS (today, 2.10 on macOS 14+)

* BF16 is a **supported path** on modern macOS (Sonoma+) in the MPS ecosystem. ([Apple Developer][10])
* But “what dtype actually ran” is less of a published contract; it’s **backend-dependent** (MPSGraph vs Metal kernels) and can change across releases. Use:

  * **fallback toggles** to localize unsupported ops ([PyTorch Documentation][11])
  * **Metal System Trace** to confirm GPU execution vs CPU fallback ([Apple Developer][10])
  * matmul backend knob `PYTORCH_MPS_PREFER_METAL` to see if GEMMs switch behavior ([PyTorch Documentation][11])

---

If you want, I can also turn the probe snippets into a **single “Transformer block harness”** (MLX + PyTorch) that prints a **per-op dtype log** and produces a “cast graph” summary (explicit casts + fallback points). That’s the fastest path to turning the “Probe” cells in the PyTorch table into “Observed on your machine” facts.

[1]: https://ml-explore.github.io/mlx/build/html/python/data_types.html?utm_source=chatgpt.com "Data Types — MLX 0.30.6 documentation"
[2]: https://pypi.org/project/torch/?utm_source=chatgpt.com "torch"
[3]: https://github.com/pytorch/pytorch/releases?utm_source=chatgpt.com "Releases · pytorch/pytorch"
[4]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html?utm_source=chatgpt.com "mlx.core.fast.scaled_dot_product_attention"
[5]: https://huggingface.co/datasets/koyeb/Apple-MLX-QA?utm_source=chatgpt.com "koyeb/Apple-MLX-QA · Datasets at Hugging Face"
[6]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.layer_norm.html?utm_source=chatgpt.com "mlx.core.fast.layer_norm — MLX 0.30.6 documentation"
[7]: https://swiftpackageindex.com/ml-explore/mlx-swift/0.29.1/documentation/mlxfast/rmsnorm%28_%3Aweight%3Aeps%3Astream%3A%29?utm_source=chatgpt.com "rmsNorm(_:weight:eps:stream:) | Documentation"
[8]: https://ml-explore.github.io/mlx/build/html/python/metal.html?utm_source=chatgpt.com "Metal — MLX 0.30.4 documentation"
[9]: https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html?utm_source=chatgpt.com "Metal Debugger — MLX 0.30.4 documentation"
[10]: https://developer.apple.com/la/videos/play/wwdc2023/10050/?utm_source=chatgpt.com "Optimize machine learning for Metal apps - WWDC23 - Videos"
[11]: https://docs.pytorch.org/docs/stable/mps_environment_variables.html?utm_source=chatgpt.com "MPS Environment Variables"
[12]: https://docs.pytorch.org/docs/stable/notes/mps.html?utm_source=chatgpt.com "MPS backend — PyTorch 2.10 documentation"
