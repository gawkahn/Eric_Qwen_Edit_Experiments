# Security Review — ADR-035 slice 4 (comfyless daemon reference-image path)

**AI-Disclosure:** Reviewed by the `security-auditor` subagent (Claude, Fable) on
2026-07-21; findings triaged and remediated by Claude (Opus 4.8), Grant reviewed.
**Scope:** `comfyless/{server,generate,params_validation,ref_image}.py` + the four
touched test suites. **Surface:** §12 Red Zone — Unix-socket daemon IPC + a new
untrusted-file-read/VAE-encode primitive behind the socket. **Design authority:**
`docs/decisions/ADR-035-comfyless-reference-image-surface.md` (6a–6g / decision 2
Finding 4 / decision 3).

## Verdict

**Design contract MET — no CRITICAL, no HIGH.** All eight audited invariants hold
on the daemon wire path (containment disjointness, non-regular-file guard, wire
shape/mode/NUL, count cap, drop strictness on the wire, no trust-class wire field,
cache-key invariant, error hygiene). One MEDIUM and the material LOW/INFO items
were remediated in-slice or explicitly deferred with a TECH_DEBT entry; see the
Disposition column.

## Findings and disposition

| # | Sev | Finding | Disposition |
|---|-----|---------|-------------|
| 1 | MEDIUM | Drop-strictness fail-closed only at the daemon boundary; the in-process CLI call site was unconditionally lenient, so a scripted run's strictness depended on whether a daemon happened to be up (`generate.py` in-process `generate()` call). | **Fixed in-slice** — in-process call now passes `ref_drop_strict=not (sys.stdin is not None and sys.stdin.isatty())`; scripted/piped = strict, interactive TTY = lenient, matching the wire. |
| 2 | LOW | TOCTOU: containment realpaths at check time but `load_ref_image_capped` re-resolves at open time; a symlink swap inside a ref root (the daemon's own `--output-dir` is always one) reads a file outside every ref root. Same-UID confused-deputy only. | **Deferred (TECH_DEBT)** — same trust-class shift the ADR wills to the MCP ADR (output-dir read-back loop); tied to an agent-driven/less-trusted transport. Fix shape recorded (realpath write-back or open-then-fstat). |
| 3 | LOW | `ref_drop_strict` is a client-declared leniency field — any wire client can send `false` to opt into silent drops. Confers no path authority (decision 7's forgeable-trust bar met); today's only lenient sender is the interactive CLI, MCP strips refs. | **Deferred with ADR note** — when LLM agents gain a raw-wire/MCP ref path, hard-code strict for non-CLI transports. Recorded in the ADR-035 Changelog INFO. |
| 4 | LOW | `_cli_ref_image_roots` included `--model-base` (a weight root) → a ref under it delegated and was then refused by the daemon (`_check_ref_paths`), a wrong refusal of a legit typed path when a daemon is up (Finding 2 conflict). | **Fixed in-slice** — `--model-base` removed; the CLI delegation root set is now `--ref-root` only, a strict subset of the daemon's ref roots. Test pin flipped. |
| 5 | INFO | Breadth warning fired only on exact `/` and `$HOME`; the comment claimed "mount root" coverage it did not have (and `/` is inert under `_within`). | **Fixed in-slice** — `os.path.ismount(root_real)` added to the breadth predicate; loop extracted to the unit-tested `_resolve_ref_roots`. |
| 6 | INFO | Aggregate decoded-pixel host RAM is unbounded by the byte/count caps (8 × ~67 MP float32 tensors held before VAE encode). Same-UID DoS, largely outside the threat model. | **Accepted** — design-accepted caps; an aggregate pixel budget would close it if a less-trusted transport ever fronts the daemon. |

## Per-invariant confirmation (as reported by the auditor)

1. **Containment (6a) — MET.** `_check_ref_paths` uses only `ref_roots`; `_check_paths` only the weight union; passed as separate args, never merged. Empty ref_roots fails closed; relative refused; `_within` realpaths both sides with `rb + os.sep` (no prefix-sibling escape).
2. **Non-regular-file guard — MET.** `RefImageError` is a plain `Exception` (not `OSError`) so the `S_ISREG` reject propagates past `except OSError`; `fd = -1` before the read gives the with-block sole ownership; no leak/double-close on any path.
3. **Wire strictness / NUL / N19 — MET.** Canonical validation precedes the NUL loop, so `ref["path"]` cannot Key/TypeError; no `isinstance` added to `_validate_request`; downstream `RefImageError`/`ValueError` absorbed into a structured `InferenceError` with the reservation unlinked.
4. **Count cap (6f) — MET** at the canonical boundary (every wire request); 8-accept/9-reject pinned in both suites.
5. **Drop strictness — MET on the wire** (`req.get("ref_drop_strict", True)`); the in-process asymmetry (finding 1) fixed.
6. **No trust-class wire field — MET.** New wire fields confer no path authority; the daemon applies row-3 containment unconditionally.
7. **Cache-key invariant — MET structurally**; no ref field in `_request_cache_key`, no from_pipe/class-swap on ref presence; pinned.
8. **Error hygiene (6g) — MET.** Ingestion errors name path+reason; PIL failures surface class name only; `RefPathError` audit log redacts the prompt.

No scope creep observed. Full auditor transcript summarized here; the raw run was
a `security-auditor` (Fable) invocation over the slice-4 diff.
