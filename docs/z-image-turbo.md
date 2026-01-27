# `Tongyi-MAI/Z-Image-Turbo`

This document is a **reference for the upstream model repository layout** (Hugging Face / Diffusers concepts). For how **Z-Image.swift** resolves and loads model weights (local paths, HF cache, quantization, AIO checkpoints), start with `docs/MODELS_AND_WEIGHTS.md`.


**Top-level files and folders** (authoritative list from the Hub API):
`assets/`, `model_index.json`, `scheduler/`, `text_encoder/`, `tokenizer/`, `transformer/`, `vae/`, `README.md`, `.gitattributes`. [Hugging Face](https://huggingface.co/api/models/Tongyi-MAI/Z-Image-Turbo)


## `model_index.json` — the wiring diagram


```json
{
  "_class_name": "ZImagePipeline",
  "_diffusers_version": "0.36.0.dev0",
  "scheduler": ["diffusers","FlowMatchEulerDiscreteScheduler"],
  "text_encoder": ["transformers","Qwen3Model"],
  "tokenizer": ["transformers","Qwen2Tokenizer"],
  "transformer": ["diffusers","ZImageTransformer2DModel"],
  "vae": ["diffusers","AutoencoderKL"]
}

```


This instructs Diffusers to instantiate **ZImagePipeline** and load each component from its subfolder using the provided class. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/model_index.json)



Note: Support for `ZImagePipeline`/`ZImageTransformer2DModel` was merged upstream in Diffusers (PRs #12703 and #12715). [GitHub+1](https://github.com/huggingface/diffusers/pull/12703)



## `transformer/` — the DiT “backbone”


- **Files:** `config.json`, `diffusion_pytorch_model-00001…00003.safetensors`, `diffusion_pytorch_model.safetensors.index.json`. Sharded weights total ~24.6 GB. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/transformer)
- **Class:** `ZImageTransformer2DModel`.
**Key hyperparameters** (excerpt): `dim: 3840`, `n_layers: 30`, `n_heads: 30`, `n_kv_heads: 30`, `in_channels: 16`, `cap_feat_dim: 2560`, plus RoPE settings and S3-DiT-specific shape metadata (`axes_dims`, `axes_lens`, `all_patch_size`, etc.). [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/transformer/config.json)

**What “Single-Stream” means here:** per the model card, **text tokens, “visual semantic” tokens, and VAE latent tokens are concatenated into a single sequence** processed by every DiT block—contrasting with dual-stream designs that cross-attend between separate text and image streams. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/README.md)


## `text_encoder/` and `tokenizer/` — prompt encoding


- **Text encoder class:** **Qwen3Model**, with architecture metadata indicating `Qwen3ForCausalLM` shape (e.g., `hidden_size: 2560`, `num_hidden_layers: 36`, `num_attention_heads: 32`), stored in three shards (`model-00001…00003.safetensors`). [Hugging Face+1](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/text_encoder/config.json)
- **Tokenizer:** **Qwen2Tokenizer** with `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/tokenizer)


Practical implication: unlike SD/SDXL (CLIP encoders) or FLUX variants (often T5-style encoders), Z-Image-Turbo uses a **Qwen-family LLM** as the text encoder, yielding larger, instruction-friendly embeddings.



## `vae/` — latent codec


- **Class:** `AutoencoderKL`, ~168 MB weights. **Config** exposes `latent_channels: 16`, `scaling_factor: 0.3611`, `shift_factor: 0.1159`, and `force_upcast: true`. Those scaling/shift constants are critical to obtain correct dynamic ranges when mapping latent space ↔ pixels. [Hugging Face+1](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/vae)

## `scheduler/` — sampler parameters


- **Class:** `FlowMatchEulerDiscreteScheduler` with a minimal config (e.g., `num_train_timesteps`, `shift`, `use_dynamic_shifting`). The **few-step behavior** (≈8 forwards) comes from the **distilled model and runtime num_inference_steps**, not from the config alone. [Hugging Face+1](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/scheduler/scheduler_config.json)

## `README.md` — usage and design notes worth keeping


- **Architecture:** “Scalable **Single-Stream DiT**” (S3-DiT). [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/README.md)
- **Few-step generation:** example uses `num_inference_steps=9` which leads to *8 DiT forwards*; set `guidance_scale=0.0` for Turbo variants. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/README.md)
- **Environment:** recommends latest Diffusers (source install) and optional Flash-Attention backends. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/README.md)


# How these pieces map to Diffusers concepts


| Folder / File | Diffusers class (source) | Role in the pipeline |
| ---- | ---- | ---- |
| `model_index.json` | N/A (loader manifest) | Declares `_class_name` (`ZImagePipeline`) and maps component keys → libraries/classes. [Hugging Face+1](https://huggingface.co/docs/diffusers/v0.21.0/en/using-diffusers/loading?utm_source=chatgpt.com) |
| `transformer/…` | `ZImageTransformer2DModel` (`diffusers`) | **Noise/flow predictor** for latent tokens, operating on a **single concatenated sequence** of text + image tokens. [Hugging Face+1](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/transformer/config.json) |
| `text_encoder/…` | `Qwen3Model` (`transformers`) | Encodes prompts into embeddings fed to the DiT (and to the “visual semantic” tokenizer path described in the paper/card). [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/text_encoder/config.json) |
| `tokenizer/…` | `Qwen2Tokenizer` (`transformers`) | Produces input IDs for the Qwen3 encoder. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/tokenizer) |
| `vae/…` | `AutoencoderKL` (`diffusers`) | Compresses/decodes images ↔ latents; scaling/shift factors must be respected for numerically correct results. [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/vae/config.json) |
| `scheduler/…` | `FlowMatchEulerDiscreteScheduler` (`diffusers`) | Flow-matching sampler introduced post-SD3; determines the integration path over timesteps. [Hugging Face](https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete?utm_source=chatgpt.com) |



# Cross-model perspective: DiT layouts are converging


- **FLUX.1-dev**: `transformer/`, `vae/`, `scheduler/`, `text_encoder/` (+ sometimes `text_encoder_2`), `tokenizer/` (+ sometimes `tokenizer_2`), and `model_index.json`. Same pattern. [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)
- **PixArt family** in Diffusers uses dedicated Transformer2D model classes under `transformers/` models directory (e.g., `pixart_transformer_2d.py`), again surfaced via `transformer/` in checkpoints. [GitHub](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/pixart_transformer_2d.py?utm_source=chatgpt.com)


# Practical notes for users and evaluators


1. **Sharding & indices.** Where weights exceed a few GB, expect `*-0000X-of-0000Y.safetensors` with a `*.index.json` that maps tensor names to shard files; Diffusers resolves this automatically via `from_pretrained`. Verified for both the **transformer** and **text_encoder** in Z-Image-Turbo. [Hugging Face+1](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/transformer)
2. **Few-step sampling.** The **8-forward** behavior comes from **distillation** + **FlowMatch** and the **runtime** `num_inference_steps`; follow the README recipe (`guidance_scale=0.0`, `num_inference_steps≈9`). [Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/README.md)
3. **Upstream support.** Use a recent Diffusers (the README points to source install) because **ZImagePipeline** and **ZImageTransformer2DModel** landed in late Nov-2025 PRs. [Hugging Face+1](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/blob/main/README.md)
