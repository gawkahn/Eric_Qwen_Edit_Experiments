# ADR-006: Comfyless Dual-Mode Design and JSON Bridge Contract

**Date:** 2026-04-21 (written retroactively; decision made ~2026-04-14)
**Status:** accepted

---

## Context

comfyless needs to serve two distinct callers with conflicting I/O requirements:

**Human CLI user** — wants argparse flags, progress printed to the terminal,
human-readable completion messages, and a file on disk at the end.

**LLM agent** — wants structured input (JSON), structured output (JSON), machine-
parseable errors, no terminal noise on stdout, and a versioned contract so the
caller can detect schema drift without silent failures.

Additionally, reproducibility is a first-class requirement: every generation must
be replayable from parameters alone, and a subsequent run with the same params and
seed must produce the same image.

## Decision

**Dual-mode invocation on a single entry point (`comfyless/generate.py`)**:

- **Default (human) mode**: argparse CLI flags; progress and diagnostics to stderr;
  sidecar JSON written alongside every output PNG for replay.
- **`--json` bridge mode**: JSON blob on stdin; structured JSON result on stdout;
  all human-readable output goes to stderr so stdout stays machine-parseable.
  Contract version field (currently `v1`) with explicit mismatch rejection —
  callers must declare the version they were built against.

**`--params` / `--override` replay**: load a sidecar JSON as base params, apply
`key=value` patches via `--override`, then explicit CLI flags win over both.
Merge priority (lowest to highest): `sidecar < --override < explicit CLI flags`.

Sidecar JSON written alongside every output image regardless of mode. The sidecar
is the replay spec — it is complete, standalone, and includes the resolved seed.

Stdout / stderr split is strict: in `--json` mode, nothing machine-meaningful
ever goes to stderr and nothing human-readable ever goes to stdout. This makes
it safe to pipe stdout to `jq` or a parser while watching stderr in a terminal.

## Alternatives Rejected

**Separate binaries for human/agent** — doubles maintenance burden; two codebases
sharing a `generate()` function with diverging behaviors.

**HTTP API** — adds a server dependency and latency for a tool that's primarily
invoked once per image. The Unix socket daemon (ADR-001) covers the persistent-
process use case; `--json` covers the one-shot agent use case.

**No contract versioning** — makes it impossible to detect schema drift between an
agent caller built against an old schema and an updated `generate.py`. Silent
wrong-field failures are worse than an explicit `ContractVersionMismatch` error.

**Merged stdout/stderr** — breaks pipe-to-parser patterns; forces the agent to
parse human-readable text rather than structured JSON.

## Deferred / Out of Scope

**Persistent model cache / daemon (`--serve` mode)** — covered by ADR-001. Not yet
implemented as of 2026-04-21. When implemented, normal invocations will auto-detect
the socket and delegate, with no change to the `--json` bridge contract.

**`--params <image.png>` PNG metadata replay** — read embedded comfyless metadata
from a PNG's sidecar baked in at save time. Queued in Backlog.

**`--savepath` template output paths** — template-based output with date/model/seed
variables and auto-incrementing counter. Queued (this session's pending feature).

**Batch generation mode** — multiple generations from a single invocation. Queued
in Backlog Ideas.

## Changelog

- ~2026-04-14: Initial CLI + `--json` bridge + sidecar JSON (`6211c8a`)
- ~2026-04-14: LoRA stacking + sampler/scheduler selection (`9df01b9`)
- ~2026-04-16: `--params` / `--override` replay mode (`dea801f`)
- ~2026-04-20: Component loader parity (`fcadfd0`)
- ~2026-04-21: `--attention-slicing` / `--sequential-offload` added (`3ab8a5a`)
- 2026-04-21: ADR written retroactively; decision still active

## AI-Disclosure

ADR authored by Claude Sonnet 4.6, 2026-04-21. Design by Eric Hiss; rationale
reconstructed from code and commit history. Reviewed by Grant Kahn.
