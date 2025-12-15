# Open Questions

## Unknowns
- **Training Code**: The repository is inference-only. There is no code for training or fine-tuning the base model.
- **`ZImageTransformer2D` Override Logic**: The pipeline supports overriding the transformer weights with a single file. The logic for canonicalizing keys from external checkpoints (ComfyUI/Diffusers style) is present but complex (`canonicalizeTransformerOverride`). Coverage of all external formats is unverified.
- **Multi-GPU**: The current implementation explicitly targets a single Metal device (`MTLCreateSystemDefaultDevice`). Multi-GPU support (e.g., for Mac Pro or Ultra chips behaving as multiple GPUs) is not explicitly handled beyond what MLX provides automatically.

## Maintenance
- **Model URL Stability**: The code defaults to specific Hugging Face repositories (`Tongyi-MAI/Z-Image-Turbo`). If these move or change structure, default behavior breaks.
- **Prompt Enhancement System Prompt**: The system prompt for enhancement is hardcoded in Chinese (`QwenGeneration.swift`). This implies the base Qwen model is Chinese-aligned or the project targets Chinese users primarily, though it works with English inputs. Localization plans are unknown.
