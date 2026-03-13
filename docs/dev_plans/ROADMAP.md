# Roadmap

This is the short, current priority list for the repo. It is intentionally small and should only contain work that still makes sense from the current codebase state.
The previous pipeline/CLI/model-loading cleanup items were completed in the March 11, 2026 refresh, so the list below starts from the next still-open work.

## Near Term

1. **Add support for the newly released Z-Image Fun Base add-ons**
   - Prioritize `alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1` and `alibaba-pai/Z-Image-Fun-Lora-Distill`.
   - Use [z-image-fun-support/README.md](z-image-fun-support/README.md) as the active plan.
2. **Add a first-party app example**
   - The package declares an iOS library target, but the repo still has no maintained sample app.
3. **Consider batch or multi-image generation**

## Follow-On Work

4. **Re-evaluate the CLI parsing approach after the next user-facing feature pass**
   - The current manual parser is much stricter now, so replacing it is no longer blocking on basic ergonomics.
5. **Expand preset detection only if more upstream model families make the current heuristics insufficient**
   - Known ids, local snapshot metadata, cached metadata, and common Z-Image aliases now cover the current practical cases.

## Ongoing Maintenance

- Keep `README.md`, `docs/CLI.md`, and CLI help text in sync.
- Use `ZImageCLI control --log-control-memory` with the `1536x2304` reference probe when changing control-memory-sensitive paths.
- Keep `PipelinePrecisionTests`, `QwenEncoderAttentionMaskTests`, and `ZImageRoPEParityTests` aligned with any denoiser/control precision changes.
- Keep completed investigations clearly marked as historical so the active docs stay forward-looking.
