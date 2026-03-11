# Roadmap

This is the short, current priority list for the repo. It is intentionally small and should only contain work that still makes sense from the current codebase state.
The previous pipeline/CLI/model-loading cleanup items were completed in the March 11, 2026 refresh, so the list below starts from the next still-open work.

## Near Term

1. **Finish the next precision-parity pass**
   - The remaining documented runtime gap is RoPE parity and the intermediate-tensor probes needed to validate it safely.
2. **Expose more of the library-only control features in the CLI**
   - Control-path LoRA and prompt enhancement exist in the library request type but are not surfaced in `ZImageCLI control`.
3. **Add a first-party app example**
   - The package declares an iOS library target, but the repo still has no maintained sample app.
4. **Consider batch or multi-image generation**

## Follow-On Work

5. **Re-evaluate the CLI parsing approach after the next user-facing feature pass**
   - The current manual parser is much stricter now, so replacing it is no longer blocking on basic ergonomics.
6. **Expand preset detection only if more upstream model families make the current heuristics insufficient**
   - Known ids, local snapshot metadata, cached metadata, and common Z-Image aliases now cover the current practical cases.

## Ongoing Maintenance

- Keep `README.md`, `docs/CLI.md`, and CLI help text in sync.
- Use `ZImageCLI control --log-control-memory` with the `1536x2304` reference probe when changing control-memory-sensitive paths.
- Keep completed investigations clearly marked as historical so the active docs stay forward-looking.
