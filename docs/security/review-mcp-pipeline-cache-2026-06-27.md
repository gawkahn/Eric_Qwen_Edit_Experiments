# Security review — MCP in-process pipeline cache + LoRA-apply fix

**Date:** 2026-06-27
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored the change + this disposition; `code-reviewer` (Opus) and `security-auditor` (Opus) reviewed; Grant reviewed.
**Subject:** `comfyless/mcp_server.py` (new `_PIPELINE_CACHE` + `_evict_pipeline_cache` + `_pipeline_cache_key` + `_get_or_load_cached_pipeline`; `_handle_generate` + `_handle_generate_cascade` rewiring) and `comfyless/generate.py` (extracted `_apply_loras`).
**Trigger (§12):** change to the MCP machine-boundary generate path; must preserve the ADR-015 no-abs-path egress contract.
**Relates to:** [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md), [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md). Mirrors the cache+evict pattern already in `comfyless/server.py` (the daemon).

---

## Problem fixed

1. **VRAM leak / OOM (primary):** the MCP path loaded a fresh ~20 GB pipeline every
   `generate` call with no cache and no teardown; a long-lived server OOM'd after
   many generations / model switches. The cascade handler also built both pipelines
   and never disposed them.
2. **LoRAs silently never applied over MCP:** the handler passed a pre-loaded pipeline
   to `generate()`, whose LoRA loop is skipped when `_cached_pipeline` is set — so
   `loras` only reached metadata, never the pipeline.

## Fix

Single-slot in-process pipeline cache keyed on the full effective config
(model + transformer override + `vae_from_transformer` + ordered LoRA path/weight
set). Cache hit → reuse (fast); miss → evict prior (`del` + `gc.collect()` +
`torch.cuda.empty_cache()`) → load → **apply LoRAs** via the shared `_apply_loras`.
Cascade evicts the non-cascade cache before building and disposes both pipelines in
a `finally`.

## Verdicts

**code-reviewer (Opus): APPROVED.** Cache-key covers every load-affecting input that
varies in the MCP path (the excluded params are hard-coded constants); eviction drops
the ref before gc/empty_cache (actually frees); cascade `pil` is CPU-side and survives
disposal; `generate()`'s CLI path is behavior-preserved by the extraction.

**security-auditor (Opus): ACCEPT WITH FINDINGS.** ADR-015 no-abs-path egress holds:
LoRA-warning strings embed abs paths but live only in the internal cache dict + stderr,
never in the response (`generate()` leaves `lora_warnings` empty on the cached path;
`_resolved_params_as_names` pops it defensively; the handler omits it from `notices`).
No cross-request response leak; eviction is memory-only; no new caller-influenced path
(keys use `resolve_reference`-validated abs paths, re-checked by `_check_paths` every call).

## Findings and disposition

| # | Sev | Source | Finding | Disposition |
|---|-----|--------|---------|-------------|
| 1 | INFO | sec | Comment at `_resolved_params_as_names` said the LoRA-warning loop "never runs / list is empty" — now stale; the `pop` is load-bearing (cached loader runs `_apply_loras` on every miss). A future reader could delete it and re-open the egress. | **Fixed.** Comment rewritten to mark the pop LOAD-BEARING and cite N11. |
| 2 | LOW | code | No-lock safety silently depends on the handlers having no `await` between key-check and cache-update; moving the blocking calls to an executor would make it racy with no warning. | **Fixed.** Cache-block comment now documents the no-`await` invariant and that executor offload requires a lock. |
| 3 | LOW | code | `vae_from_transformer` in the key even when no transformer is set (a no-op) caused spurious miss + reload on toggle. | **Fixed.** Key folds it out unless a transformer is set. |
| 4 | note | code | Eviction-before-load means a failed model switch loses the warm cache. | **Accepted** — matches the daemon; cache left clean (fail-closed). No change. |

A regression test (N11, `test_mcp_server.py`) asserts `mb` / `/loras/` / `/diffusion_models/`
are absent from the response, locking the egress contract. All 1412 unit tests pass.

## Residual / accepted risk

- Single-tenant, stdio-serialized server; the module-level cache has no lock by design
  (finding 2 documents the precondition).
- Agent-facing LoRA-failure signal is deferred — surfacing it needs name-based redaction
  (the raw warnings carry abs paths). LoRAs that fail to apply are logged operator-side only.
