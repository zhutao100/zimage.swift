# Staging CLI Service Plan

Status: active implementation plan

## Decision

Chosen option: warm worker daemon

Why:

- the current pipelines still unload heavy modules after each run, so a thin daemon alone would not actually keep the service warm
- the repo already has the right library entry points, so the missing piece is shared CLI parsing plus serving-oriented residency controls
- a local Unix socket keeps the transport small and avoids pulling HTTP concerns into a CLI-first package

## Delivery phases

1. shared CLI parsing/building plus a local `ZImageServe` daemon/client for ad hoc generation requests
2. serving residency policy and warm worker reuse with idle eviction and memory-pressure fallback
3. JSON batch, markdown ingestion, and operational commands

## Verification bar

- keep `ZImageCLI` behavior-compatible
- add fast coverage for parser, JSON, and markdown ingestion
- add staged end-to-end coverage for daemon lifecycle and client submission
- run a repeated-request warm-serving check against a locally cached model profile so reuse is measured rather than inferred

## Source docs

- [requirements.md](requirements.md)
- [design_options.md](design_options.md)
