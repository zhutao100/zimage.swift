## Recommended design: new local daemon + shared parser + residency policy

The cleanest design is **a new local daemon-style executable** that sits beside the existing one-shot `ZImageCLI`, plus a **shared request/parser layer** so both tools speak the same request model.

I would recommend this shape:

* Keep **`ZImageCLI`** as the current one-shot executable.
* Add **`ZImageServe`** as the staging daemon/client executable.
* Extract the current CLI parsing/building logic out of `Sources/ZImageCLI/main.swift` into a shared internal target, so:

  * `ZImageCLI` remains behavior-compatible.
  * `ZImageServe` can accept the **same argv surface** for ad hoc requests.
* Add a **runtime residency policy** to the library pipelines, because a daemon alone does **not** fully solve warm serving with the current code.

The key reason is architectural: the current pipelines are stateful, but still intentionally unload heavy modules after a run.

* `ZImagePipeline.loadModel(...)` already loads tokenizer, text encoder, transformer, and VAE into memory.
* But `ZImagePipeline.generateCore(...)` later sets `textEncoder = nil` and calls `unloadTransformer()`.
* `ZImageControlPipeline.generateCore(...)` also unloads transformer/controlnet and VAE encoder/decoder as part of its memory policy.

So today, a daemon would help with process startup and some cached state, but it would **not** give you a true “fully warm, immediate kick-off” server unless you add a serving-oriented residency mode.

---

## Constraints from the current repo

The repo already gives you the right primitives:

* The canonical CLI surface is in `Sources/ZImageCLI/main.swift`.
* The canonical generation APIs are:

  * `ZImageGenerationRequest` + `ZImagePipeline`
  * `ZImageControlGenerationRequest` + `ZImageControlPipeline`
* Both pipelines already support in-memory output:

  * `generateToMemory(...)`
* The control pipeline already has some daemon-friendly behavior:

  * cached prompt embeddings
  * memory-aware loading/unloading decisions

That means the server can be added without inventing a second inference stack. The right move is to **normalize all inputs into the existing request types** and make the pipelines optionally more residency-friendly.

---

## Design options

| Option                 | Description                                                             | Pros                                                                            | Cons                                                                  | Fit                                                 |
| ---------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------- |
| A. Thin daemon wrapper | Add a local daemon, but keep current pipeline unload behavior unchanged | Fastest to ship, low-risk                                                       | Only partially warm; transformer/text-encoder reloads still happen    | Acceptable as a short first step                    |
| B. Warm worker daemon  | Add daemon + shared parser + adaptive module residency                  | Meets the stated warm-server requirement; clean CLI compatibility; future-proof | Requires moderate library refactor                                    | **Best option**                                     |
| C. Full HTTP service   | Add local/remote HTTP JSON API                                          | Broadest integration surface                                                    | More dependencies, more protocol surface, more security/path concerns | Overkill for this repo’s current CLI-first use case |

---

## Recommended architecture

### 1) New executable: `ZImageServe`

Use a single new executable with two modes:

* **daemon mode**
* **client mode**

Examples:

```bash
# Start the staging daemon
ZImageServe serve --socket ~/.cache/zimage/stage.sock --warm-model Tongyi-MAI/Z-Image-Turbo

# Ad hoc text-to-image, same syntax shape as ZImageCLI
ZImageServe -p "a mountain lake at sunrise" -o out/lake.png

# Ad hoc control request, same syntax shape as ZImageCLI control
ZImageServe control \
  -p "a dancer on a stage" \
  -c pose.png \
  --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  -o out/dancer.png

# Structured batch
ZImageServe batch jobs.json

# Markdown with fenced bash blocks
ZImageServe markdown prompts.md
```

This matches your requirement well: users can take an old `ZImageCLI ...` command, change the program name to `ZImageServe`, and submit it.

---

### 2) Shared internal target for CLI compatibility

Do **not** reimplement the parser twice.

Refactor `Sources/ZImageCLI/main.swift` into a shared internal target such as:

* `Sources/ZImageCLICommon/`

  * `CLICompatParser.swift`
  * `CLIRequestBuilder.swift`
  * `BatchManifest.swift`
  * `MarkdownCommandExtractor.swift`
  * `ShellWordsLexer.swift`

Then:

* `ZImageCLI` becomes a thin one-shot runner.
* `ZImageServe` becomes a thin client/daemon wrapper around the same parsing/building code.

This preserves help text, flag semantics, and test expectations.

---

### 3) Canonical job model

All inputs should normalize into one internal enum:

```swift
enum GenerationJob {
    case text(ZImageGenerationRequest)
    case control(ZImageControlGenerationRequest)
}
```

All submission formats should compile into that:

* raw argv
* structured JSON
* markdown fenced bash blocks

That keeps the server core format-agnostic.

---

### 4) Transport: Unix domain socket + NDJSON

For a staging/local daemon, use a **Unix domain socket**, not HTTP.

Why this is the right default:

* local-only by default
* no HTTP framework dependency required
* filesystem permissions are enough for basic access control
* easy progress streaming
* keeps path semantics simple for local files like control images and outputs

Use newline-delimited JSON messages for:

* submit
* accepted
* progress
* completed
* failed
* status
* cancel
* shutdown

The client can render the same progress bar behavior currently used by `ZImageCLI`.

---

## The important library change: residency policy

This is the core enabling change.

### Text pipeline

Add something like:

```swift
public enum ModuleResidencyPolicy: Sendable {
    case oneShot      // current behavior
    case warm
    case adaptive
}
```

Then add runtime options to the text path, analogous to the existing control runtime options.

In `ZImagePipeline`, serving mode should be able to:

* keep tokenizer loaded
* keep text encoder loaded
* keep transformer loaded
* keep VAE loaded
* optionally keep LoRA applied when the worker profile matches

Current one-shot behavior remains the default for `ZImageCLI`.

### Control pipeline

Extend `ZImageControlRuntimeOptions` to include residency policy.

In warm/adaptive serving mode, allow the pipeline to retain:

* tokenizer
* transformer
* controlnet
* VAE decoder
* optionally VAE encoder if memory budget permits
* existing cached prompt embeddings

For control, I would make the default server policy **adaptive**, not fully pinned:

* keep transformer/controlnet warm across jobs
* unload when memory pressure is detected
* unload on profile switch
* unload after idle timeout

That fits the current codebase, which is already somewhat memory-aware.

---

## Worker model

Use one **actor-backed worker** per warm profile.

A profile should be keyed by something like:

* job kind: text / control
* base model
* weights variant
* controlnet weights + selected file, if control
* max sequence length
* residency policy

Default behavior should be conservative:

* `maxResidentProfiles = 1`
* `maxConcurrentGPUJobs = 1`

This repo is large-model, unified-memory, Apple-Silicon-first code. Serial GPU execution is the correct default. Intake can be concurrent; inference should not be.

Optional later enhancement:

* small LRU of warm profiles on high-memory systems

---

## Batch input formats

### Structured JSON

Use an explicit schema, not raw command strings as the primary format.

Example:

```json
{
  "version": 1,
  "defaults": {
    "model": "Tongyi-MAI/Z-Image-Turbo"
  },
  "jobs": [
    {
      "id": "lake-1",
      "kind": "text",
      "prompt": "a mountain lake at sunrise",
      "output": "out/lake.png"
    },
    {
      "id": "dancer-1",
      "kind": "control",
      "prompt": "a dancer on a stage",
      "controlImage": "pose.png",
      "controlnetWeights": "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
      "controlFile": "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
      "output": "out/dancer.png"
    }
  ]
}
```

You can also support an optional compatibility field like `argv`, but it should be secondary.

### Markdown fenced bash blocks

Support fenced blocks labeled `bash`, `sh`, or `zsh`.

Important rule: **parse, do not execute**.

Only accept blocks that reduce to a single CLI invocation whose first token is one of:

* `ZImageCLI`
* `./ZImageCLI`
* `ZImageServe`
* `./ZImageServe`

Support:

* quoting
* escaped spaces
* backslash line continuations
* comments

Reject:

* pipes
* redirects
* `&&`
* `;`
* subshells
* command substitution

That keeps this feature safe and deterministic.

---

## Suggested product layout

In `Package.swift`, I would move toward:

* library: `ZImage`
* executable: `ZImageCLI`
* executable: `ZImageServe`
* internal target: `ZImageCLICommon`
* optional internal target: `ZImageServeCore`

That keeps the package-first design intact.

---

## Recommended implementation phases

### Phase 1 — parser extraction + thin daemon

* Extract current CLI parsing/building into shared code.
* Add `ZImageServe serve`.
* Add client submission for argv-compatible ad hoc jobs.
* Queue jobs serially.
* Keep existing pipeline unload behavior.

This gets the protocol, daemon lifecycle, and CLI compatibility in place.

### Phase 2 — true warm serving

* Add residency policy to both pipelines.
* Keep transformer/text encoder/controlnet resident between matching jobs.
* Add idle eviction and memory-pressure fallback.
* Add warm profile selection.

This is the phase that actually satisfies the “immediately kick off” requirement.

### Phase 3 — batch formats + operational polish

* Add structured JSON batch ingestion.
* Add markdown fenced-command ingestion.
* Add `status`, `cancel`, `shutdown`.
* Add job history / result manifest output if useful.

---

## Acceptance criteria

I would consider this done when all of the following are true:

* `ZImageServe` accepts the same ad hoc argv shape as `ZImageCLI`.
* `ZImageServe batch jobs.json` works.
* `ZImageServe markdown prompts.md` works.
* A second request with the same warm profile does **not** reload the heavy model components unless policy or memory pressure requires it.
* Progress and non-zero error behavior remain comparable to `ZImageCLI`.
* `ZImageCLI` remains behavior-compatible.

---

## Bottom line

The best design is **not** “turn `ZImageCLI` into a server.”
The best design is:

1. keep `ZImageCLI` as the stable one-shot tool,
2. add a new daemon/client executable `ZImageServe`,
3. refactor parsing into shared code, and
4. add a **serving residency policy** to the pipelines.

That gives you:

* clean copy-paste compatibility for old commands
* proper warm serving
* batch JSON
* markdown command ingestion
* minimal disruption to the current CLI and tests
