# Requirements

A local staging CLI service for ad hoc and batch image-generation requests.

## Validated scope

The current repo state adds three hard constraints to the implementation:

- `ZImageCLI` is the source-of-truth command surface for one-shot generation and must stay behavior-compatible.
- There is no shared parser layer today; `Sources/ZImageCLI/main.swift` currently owns the parsing, validation, help text, and request-building logic.
- A daemon alone is insufficient for warm serving because both generation pipelines still unload heavy modules after each run:
  - `Sources/ZImage/Pipeline/ZImagePipeline.swift` clears the text encoder after prompt encoding and unloads the transformer after denoising.
  - `Sources/ZImage/Pipeline/ZImageControlPipeline.swift` unloads transformer and ControlNet state after denoising and keeps its control-context path aggressively memory-oriented.

## Functional requirements

- Add a new local executable dedicated to staged generation requests without replacing the existing `ZImageCLI` one-shot workflow.
- The service must support ad hoc generation requests with the same argv surface as `ZImageCLI` for:
  - text-to-image generation
  - `control` generation
- Users must be able to take an existing generation command, replace the program name with the staging executable, and submit the job without rewriting flags.
- The service must accept structured batch requests from a JSON manifest.
- The service must accept markdown files that contain multiple fenced `bash`, `sh`, or `zsh` command blocks, where each accepted block resolves to a single `ZImageCLI` or `ZImageServe` invocation.
- Markdown ingestion must parse commands and reject shell features that would execute additional commands or redirect data.

## Warm-serving requirements

- The service must be able to warm model state in advance of the first submitted job.
- A repeated request with the same warm profile must avoid reloading heavy model components unless:
  - the effective profile changes
  - the residency policy requires eviction
  - memory-pressure handling forces eviction
- Default GPU execution remains serialized:
  - multiple client submissions may queue
  - at most one generation job executes on the GPU at a time

## Operational requirements

- The staging service is local-only by default and uses a Unix domain socket transport.
- The client executable must expose:
  - `serve`
  - `status`
  - `cancel`
  - `shutdown`
  - `batch`
  - `markdown`
- The service must stream progress and return non-zero failures comparably to `ZImageCLI`.
- `ZImageCLI quantize` and `ZImageCLI quantize-controlnet` remain one-shot operations; the staging service does not need to remote them through the daemon.

## Verification requirements

- The implementation must keep `ZImageCLI` help text, validation behavior, and request semantics intact.
- New parser and batch/markdown logic must have fast tests.
- The staged executable must have end-to-end coverage for:
  - daemon lifecycle
  - ad hoc submission
  - batch submission
  - markdown submission
  - status and shutdown
- Warm-serving changes must be validated with a repeated-request run against a locally cached model profile so the residency behavior is measured, not assumed.
