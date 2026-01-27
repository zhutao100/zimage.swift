# Roadmap (Next Steps)

This is a lightweight, prioritized list of *potential* next steps for the project. It’s meant to guide work, not to promise timelines.

## Highest Priority

1. **Reduce duplication across pipelines and transformer variants**
   - `ZImagePipeline` vs `ZImageControlPipeline`
   - `ZImageTransformer2DModel` vs `ZImageControlTransformer2DModel`
2. **Harden the CLI UX**
   - Improve argument validation / error messages
   - Consider migrating to `apple/swift-argument-parser` (not currently used)
3. **Make model resolution more ergonomic**
   - Clearer docs and errors for local paths vs HF ids
   - Better surfacing of auth requirements for gated/private repos (e.g. guidance for `HF_TOKEN`)

## Nice To Have

4. **Batch / multi-image generation**
5. **More control over sampler and prompt encoding**
6. **First-class iOS example app**

## Ongoing Maintenance

- Keep CLI help text and docs in sync.
- Prefer adding new docs under `docs/` and linking from `README.md` (avoid duplicated explanations).
