# Slice 2b Vision — `list_transformers` MCP tool

**Date:** 2026-05-31
**ADR:** [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) (Status: accepted). Carved out of the [slice-2 Vision](slice-2-mcp-catalog.md) "Out of scope" (line 73) + invariants 16–17. Parent: [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md).
**Status:** approved by Grant 2026-05-31. **IMPLEMENTED 2026-05-31** (commit `d5d1b68`; `code-reviewer` APPROVED + `security-auditor` CLEAN; `test_mcp_server.py` 375→394, full suite 1244/1244).
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant reviewed.

---

## Posture

> Boundary: integration (LLM-agent reads a third catalog-discovery surface). Risk factors: near security truth (same catalog / `abs_path`-containment surface as slice 2 — the keystone). Narrow: the catalog, its scan dispatch, and the `kind:"transformer"` entries already shipped and were reviewed in slice 2 (N24–N28); this slice only *advertises* a tool and *serializes* entries that already exist.

## Slice

Expose the already-built-and-dormant `kind:"transformer"` catalog entries via a third
list tool, `list_transformers`. The catalog construction, scan dispatch (slice-2
invariant 17), and entries already exist and are tested — this slice adds one `Tool(...)`
advert, one dispatch branch, and one handler (`_handle_list_transformers`), a faithful
mirror of `_handle_list_models` with the kind filter changed `"model"`→`"transformer"`.
No catalog-build changes, no `generate` changes, no `list_models`/`list_loras` changes.

## Risk level

**L3 (Red Zone).** Same catalog / `abs_path`-containment surface as slice 2. Runs
`code-reviewer` (Opus) **and** `security-auditor` (Opus), model pinned at invocation per
global §5A; review saved to `docs/security/review-slice-2b-list-transformers-2026-05-31.md`
and referenced in the commit body.

## Invariants (must always be true)

1. **`_list_tools_impl` advertises exactly FOUR tools**: `generate`, `list_models`,
   `list_loras`, `list_transformers` (updates slice-2 invariant 8's count 3→4). The
   other three Tools' schemas and descriptions are byte-unchanged.
2. **`list_transformers` response entries contain ONLY** `{name, kind:"transformer",
   source}` + `model_family` when present (manifest-declared — scan-derived transformers
   carry `model_family=None` in the catalog and the field is omitted). **No `abs_path` /
   `path` / any filesystem string under any key.** The keystone guarantee extends verbatim
   to the new handler; the response dict is constructed key-by-key from a fixed allowlist,
   never spread from the catalog entry.
3. **`list_transformers` returns ONLY `kind:"transformer"` entries**; `list_models` /
   `list_loras` continue to exclude them (slice-2 carry-forward). Transformers now appear
   in exactly one tool.
4. **Empty-object inputSchema** (`additionalProperties:false`); audit-payload reduced to
   `{}` like the other two list tools (the step-4 LOW-1 fold); one audit line per
   invocation; no `abs_path` in the audit line.
5. **`generate` untouched**; existing suites stay green. Traceback-strip / sanitized-error
   path carries forward unchanged (the handler raises nothing custom; unexpected
   exceptions route through `_call_tool_impl`'s outer `except BaseException`).

## Failure semantics

- Request-time fail-closed via slice-1's traceback-strip + sanitized-error pattern. No
  caller-supplied name is *resolved* here (the tool enumerates), so no reference-resolution
  failure mode is reachable — the uniform-error contract remains a slice-3 concern.
- Audit-line write failure does not block the response (carry-forward).

## Negative / proof cases (in `test_mcp_server.py`)

- **N20 / Inv-8 / Step3-carry** — flipped to assert FOUR tools + `list_transformers`
  empty-input schema; the other three descriptions/schemas asserted byte-unchanged.
- **N27 (extended)** — transformer present in catalog; excluded from `list_models` /
  `list_loras`; **now surfaced by `list_transformers`** with no model/lora bleed-through.
- **Tb1** — response shape: key-subset `{name, kind, source, model_family}`, no
  `abs_path`/`path`, every entry `kind:"transformer"`, scan-derived omits `model_family`.
- **Tb2** — directly inject a catalog entry carrying a real `abs_path` + manifest
  `model_family`; assert the path/filename never appears in the serialized body and
  `model_family` surfaces (proves the serializer itself is the containment boundary).
- **Tb3** — empty catalog → `[]`, count 0.
- **Tb4** — dispatch through `_call_tool_impl`; 10 KB flood arg ignored; exactly one audit
  line with `input=={}`, `status=="ok"`, `count` present.

## Out of scope (explicit)

- LoRA `target_family` inference; `vae` / `text_encoder` catalog kinds; cascade
  `scaffolding_repo` modeling.
- The slice-3 `generate` migration and the uniform-error contract (no caller-supplied name
  is resolved here).

## Red Zone ownership

- **The `abs_path`-never-crosses-MCP-boundary guarantee** in `list_transformers`: owned by
  **Grant** — signs off that the new serializer leaks no `abs_path` under any key,
  including the audit line. (Verified CLEAN by `security-auditor`, 2026-05-31.)

## Pointers

- Parent slice-2 Vision: [slice-2-mcp-catalog.md](slice-2-mcp-catalog.md) (invariants 16–17 mint the transformer entries this slice exposes).
- ADR: [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) (Changelog 2026-05-31 entry).
- Security review: [review-slice-2b-list-transformers-2026-05-31.md](../security/review-slice-2b-list-transformers-2026-05-31.md).
