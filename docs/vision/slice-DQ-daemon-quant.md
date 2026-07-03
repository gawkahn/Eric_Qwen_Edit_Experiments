# Slice DQ — Daemon quant carriage (ADR-019 follow-up)

Status: implemented 2026-07-03 (unit proof complete; live daemon smoke pending Grant)
Parent: docs/decisions/ADR-019-native-quantization-support.md
AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.

## Problem

`--quant` currently skips daemon delegation ("daemon protocol doesn't carry
quant yet; running in-process", generate.py ~1841). Every quantized iterate
run pays a full cold load per invocation batch and cannot share the daemon's
warm pipeline cache. TECH_DEBT F2.

## What must be true when done

1. A client invocation with `--quant fp8` (± `--quant-skip`/`--quant-only`)
   delegates to a running daemon exactly like an unquantized one; the wire
   request carries all three fields.
2. The daemon's pipeline cache key discriminates on the quant triple: a
   request with different quant settings against the same model NEVER reuses
   the cached pipeline (a quantized pipeline is a different artifact than a
   bf16 one; components cannot be re-quantized in place on cache hit).
3. When quant is active, ANY change to the requested LoRA set (path or
   weight) evicts the pipeline and reloads from scratch — the incremental
   LoRA diff path is never taken on a quantized pipeline. Rationale:
   direct-merge adapters merge into requantized weights (slice DMR);
   `pipe.delete_adapters` cannot undo that, so an incremental "remove"
   would silently leave the merge baked in. (Mirrors the MCP server's
   `_pipeline_cache_key` policy.)
4. A malformed `quant` mode value on the wire is rejected at validation
   time with a ValidationError response (allowed set = QUANT_MODES), not by
   an exception escaping into a LoadError, and never kills the accept loop.
5. `quant_skip`/`quant_only` wire entries remain covered by the canonical
   validator's slot-name hygiene (str, no NUL, no path separators, ≤32
   entries) — already in `params_validation.validate_machine_request`;
   this slice adds no new bypass around it.

## What must never happen

- N1: A cached bf16 pipeline served for a `--quant fp8` request, or vice
  versa (cache-key collision across quant settings).
- N2: Incremental LoRA add/remove applied to a quantized cached pipeline.
- N3: `quant_skip`/`quant_only` strings reaching any filesystem API in
  server.py (they feed only `build_quant_config` component selection).
- N4: Unquantized daemon behavior changing in any way (cache key for
  quant="none" requests keeps its existing LoRA-diff semantics).
- N5: A crafted quant field crashing the server loop.

## Proof

- Unit: cache-key discrimination (quant vs none, skip/only sets, LoRA set
  under quant), validation negatives (bogus mode, path-shaped slot names,
  oversized lists), client wire-request contents include the triple, the
  delegation-skip branch is gone.
- Live: daemon started, same `--quant fp8` command twice — second run hits
  the warm cache; then a LoRA change under quant shows "evicting" not
  "LoRA removed".

## Edit scope

- `comfyless/server.py` — validation semantic check, cache key, load calls
  (§12 surface: security-auditor delta review REQUIRED before commit)
- `comfyless/generate.py` — wire request fields, remove delegation-skip
- `test_server_robustness.py` — new tests
- `CLAUDE.md` — suite count line
- `docs/decisions/ADR-019-native-quantization-support.md` — changelog
- `TECH_DEBT.md` — resolve F2 entry

## Out of scope

- Daemon LoRA-lifecycle defects for UNQUANTIZED pipelines (merged adapters
  never unload; weight-only changes ignored) — separate TECH_DEBT entries,
  separate slice. This slice sidesteps them for quant via full-evict.
- MCP server changes (already carries quant since slice A).
