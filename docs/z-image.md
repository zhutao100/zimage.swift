# `Tongyi-MAI/Z-Image`

**Top-level files and folders** (Hub file browser):
`model_index.json`, `scheduler/`, `text_encoder/`, `tokenizer/`, `transformer/`, `vae/`, `README.md`, `teaser.jpg`. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/tree/main)

> Note: This document is a **reference for the upstream model repository layout** (Hugging Face / Diffusers concepts). For how **Z-Image.swift** resolves and loads model weights (local paths, HF cache, quantization, AIO checkpoints), start with `docs/MODELS_AND_WEIGHTS.md`.

---

## `model_index.json` — the wiring diagram

```json
{
  "_class_name": "ZImagePipeline",
  "_diffusers_version": "0.37.0.dev0",
  "scheduler": [
    "diffusers",
    "FlowMatchEulerDiscreteScheduler"
  ],
  "text_encoder": [
    "transformers",
    "Qwen3Model"
  ],
  "tokenizer": [
    "transformers",
    "Qwen2Tokenizer"
  ],
  "transformer": [
    "diffusers",
    "ZImageTransformer2DModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

This instructs Diffusers to instantiate **`ZImagePipeline`** and load each component from its subfolder using the declared class. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/model_index.json)

- Pipeline: `ZImagePipeline` (Diffusers) — docs: [Diffusers]({"https://huggingface.co/docs/diffusers/en/api/pipelines/z_image"})
- Transformer: `ZImageTransformer2DModel` (Diffusers) — docs: [Diffusers]({"https://huggingface.co/docs/diffusers/main/en/api/models/z_image_transformer2d"})
- Scheduler: `FlowMatchEulerDiscreteScheduler` (Diffusers)
- Text encoder: `Qwen3Model` (Transformers)
- Tokenizer: `Qwen2Tokenizer` (Transformers)
- VAE: `AutoencoderKL` (Diffusers)

The repo declares `_diffusers_version: 0.37.0.dev0`, so you’ll generally want a recent Diffusers (often from source) to avoid “missing pipeline/model class” issues.

---

## `transformer/` — the DiT “backbone” (S3‑DiT)

- **Files:** `config.json`, `diffusion_pytorch_model-00001-of-00002.safetensors`, `diffusion_pytorch_model-00002-of-00002.safetensors`, `diffusion_pytorch_model.safetensors.index.json`
- **Published size (Hub):** ~12.3 GB (sharded) [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/tree/main/transformer)
- **Class:** `ZImageTransformer2DModel`

**Key hyperparameters** (from `transformer/config.json`): [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/transformer/config.json)

- `in_channels: 16` (latent channels)
- `dim: 3840`
- `n_layers: 30` + `n_refiner_layers: 2`
- `n_heads: 30` (`n_kv_heads: 30` → no GQA)
- `qk_norm: true`, `norm_eps: 1e-5`
- RoPE: `rope_theta: 256.0`
- Patch sizes: `all_patch_size: [2]`, `all_f_patch_size: [1]`

This is the Z‑Image series’ public “6B‑class” single‑stream DiT backbone: text features and latent/image tokens are concatenated into one stream for diffusion.

---

## `text_encoder/` — prompt encoder (Qwen3 family)

- **Files:** `config.json`, `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, `model-00003-of-00003.safetensors`, `model.safetensors.index.json`
- **Published size (Hub):** ~8.05 GB (sharded) [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/tree/main/text_encoder)
- **Architecture:** `Qwen3ForCausalLM` in config, while `model_index.json` loads `Qwen3Model`. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/text_encoder/config.json)

**Key hyperparameters** (from `text_encoder/config.json`):

- `hidden_size: 2560`, `num_hidden_layers: 36`
- `num_attention_heads: 32`, `num_key_value_heads: 8` (GQA)
- `max_position_embeddings: 40960`
- `torch_dtype: "bfloat16"`, `vocab_size: 151936`

Practically: the “text encoder” here is a full Qwen3 LM checkpoint used as a prompt feature extractor inside the diffusion pipeline.

---

## `tokenizer/` — Qwen tokenizer + chat template

- **Files:** `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/tree/main/tokenizer)
- **Tokenizer class:** `Qwen2Tokenizer` (see `model_index.json`)

Interesting detail: `tokenizer_config.json` embeds a fairly full-featured **chat template** (ChatML-like with `<|im_start|>…<|im_end|>`), including optional tool-call XML tags and `<think>` tags. Even if you only pass a single `prompt: str` to Diffusers, this hints that upstream prompt tooling can be applied consistently with the Qwen ecosystem.

Also note:
- `model_max_length: 131072` is set in the tokenizer config (tokenizer-side cap). [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/tokenizer/tokenizer_config.json)

---

## `vae/` — latent image codec (FLUX-style VAE)

- **Files:** `config.json`, `diffusion_pytorch_model.safetensors`
- **Published size (Hub):** ~168 MB [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/tree/main/vae)
- **Class:** `AutoencoderKL`

**Key hyperparameters** (from `vae/config.json`): [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/vae/config.json)

- `latent_channels: 16`
- `scaling_factor: 0.3611`
- `shift_factor: 0.1159`
- `sample_size: 1024`
- `force_upcast: true`

This is consistent with the FLUX “16‑channel latent” VAE lineage (vs SDXL’s 4‑channel latents).

---

## `scheduler/` — FlowMatch Euler (important: `shift = 6.0`)

- **File:** `scheduler_config.json` [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image/tree/main/scheduler)
- **Class:** `FlowMatchEulerDiscreteScheduler`

```json
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.37.0.dev0",
  "num_train_timesteps": 1000,
  "use_dynamic_shifting": false,
  "shift": 6.0
}
```

That `shift` value matters: many community ports and re-implementations treat the model’s scheduler config as part of the checkpoint’s “contract”.

---

## `README.md` — recommended inference settings (Z‑Image vs Turbo)

The model card’s “best practice” settings for **Z‑Image (base)** are meaningfully different from the distilled Turbo variant:

- **CFG is enabled** (uses `guidance_scale` and supports `negative_prompt`)
- **More steps**: recommend `num_inference_steps` in the ~28–50 range (28 is suggested as “fastest”)
- Example code uses `cfg_normalization=True`

Typical usage snippet (from the model card): [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image)

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
).to("cuda")

image = pipe(
    prompt="A close-up photo of a black tiger...",
    negative_prompt="worst quality, low quality, ...",
    guidance_scale=4,
    num_inference_steps=50,
    height=1024,
    width=1024,
    cfg_normalization=True,
).images[0]
```

---

## Quick compare: `Z-Image` vs `Z-Image-Turbo`

| Dimension | Z‑Image (this doc) | Z‑Image‑Turbo (previous doc) |
|---|---|---|
| Intended use | quality / fidelity | speed / distilled few-step |
| `num_inference_steps` | typically ~28–50 | typically ~8–9 |
| CFG / negative prompt | **enabled** | **disabled** (guidance≈0) |
| Scheduler `shift` | **6.0** | **3.0** |
| Total checkpoint size (Hub) | ~20.5 GB | ~24.6 GB |

---

## Practical integrator notes

1. **Diffusers version matters.** The model repo declares a dev Diffusers version; for “attribute not found” errors, install Diffusers from source (git main) first (the model card explicitly recommends installing Diffusers from source).

   ```bash
   pip install git+https://github.com/huggingface/diffusers
   pip install -U huggingface_hub
   HF_XET_HIGH_PERFORMANCE=1 hf download Tongyi-MAI/Z-Image
   ```
2. **Treat `scheduler_config.json` as part of the weights.** If you swap schedulers/samplers, prefer `FlowMatchEulerDiscreteScheduler.from_config(...)` so `shift` and timesteps match the checkpoint.
3. **VRAM planning:** the text encoder alone is ~8 GB, transformer ~12 GB; bf16 is the intended dtype (as shown in model card examples).
