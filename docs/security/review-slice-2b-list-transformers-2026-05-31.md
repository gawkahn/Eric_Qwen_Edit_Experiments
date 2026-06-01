# Security + code review — ADR-015 slice 2b (`list_transformers` MCP tool)

**Date:** 2026-05-31
**AI-Disclosure:** Reviews authored by Claude (Opus 4.8, 1M context) sub-agents
(`code-reviewer` and `security-auditor`, both Opus, model pinned at invocation per
global §5A — the Claude Code 2.1.117 frontmatter-pin bug means the pin must be passed
at the Agent call). Slice implemented by Claude (Opus 4.8); reviewed by Grant.
**Slice:** ADR-015 slice 2b — adds the read-only `list_transformers` MCP tool to
`comfyless/mcp_server.py`, surfacing the already-built-and-dormant `kind:"transformer"`
catalog entries. Mirror of the slice-2 `_handle_list_models` handler.
**Risk:** L3 (Red Zone) — same catalog / `abs_path`-containment surface as slice 2.
**Verdict:** `code-reviewer` **APPROVED**; `security-auditor` **CLEAN** (no HIGH/MEDIUM/LOW).
**Files reviewed:** `comfyless/mcp_server.py`, `test_mcp_server.py` (working-tree diff).

---

## security-auditor (Opus) — verdict CLEAN

No findings at HIGH/MEDIUM/LOW. One INFO (no action required).

Keystone verification — why `abs_path` cannot escape:

1. **abs_path containment (keystone) — HOLDS.** `_handle_list_transformers` builds each
   response entry by *explicit key assignment*, not by copying the catalog entry:
   `name` (the catalog key), `kind` as the hardcoded literal `"transformer"` (not
   `entry["kind"]`), `source` from `entry["source"]`, and `model_family` from
   `entry.get("model_family")` only when non-`None`. It never reads `entry["abs_path"]`
   or `entry["target_family"]`. The catalog entry is a closed five-key shape
   (`catalog.py`: `abs_path, kind, source, model_family, target_family`) — no other key
   could carry a filesystem string. The `--catalog` manifest *path* is a local variable
   in `_validate_startup_args`, never stored on `_StartupConfig` (`__slots__`), so it is
   unreachable from the handler. Structurally identical to the slice-2 `_handle_list_models`
   serializer.
2. **Audit-line redaction — HOLDS.** `list_transformers` added to the
   `("list_models","list_loras","list_transformers")` tuple → `audit_payload` forced to
   `{}` regardless of agent input. Test Tb4 injects a 10 KB flood blob and asserts neither
   the blob nor its key appears in stderr; exactly one audit line with `input=={}`,
   `status=="ok"`, `count` present. The only catalog-derived value reaching
   `_emit_audit_line` is the integer `result_count`.
3. **Error/traceback strip — HOLDS.** The handler raises nothing custom (pure read-only
   iteration); any unexpected exception propagates to `_call_tool_impl`'s
   `except BaseException` → `_sanitize_error` → category-only `ValueError`. No new bypass.
4. **Information-disclosure oracle — NO NEW ASYMMETRY.** Exposing transformer *names* is
   the same disclosure class slice 2 already accepted for models/loras (catalog is
   discoverable by design); identical key set, no asymmetry.
5. **Input handling — HOLDS.** Empty-object inputSchema + `validate_input=False`; handler
   takes no `arguments` parameter and cannot be driven by agent input (Tb4 confirms).

Test coverage assessment: adequate for Red Zone. Tb2 directly injects a catalog entry
carrying a real `abs_path` and asserts neither the directory string nor the filename
appears in the response — strongest leak test (bypasses the build-time allowlist, proves
the *serializer itself* is the containment boundary). Tb1 asserts key-subset; N27 proves
bidirectional kind-filtering; Tb4 the audit bound; Tb3 the empty-catalog edge.

**[INFO] Tb2's abs_path leak assertion is substring-based, not key-based**
(`test_mcp_server.py` Tb2). The substring check is correct and Tb1's key-subset check
covers the structural guarantee; the combination is sufficient. Noted only so a future
maintainer does not weaken Tb1 believing Tb2 is a complete backstop on its own. No action.

---

## code-reviewer (Opus) — APPROVED

Invariants 1–5 all HOLD (four tools / other three byte-unchanged; response allowlist
keystone; kind isolation with no bleed; empty schema + audit `{}` + one audit line;
generate untouched). `_handle_list_transformers` is a byte-faithful mirror of
`_handle_list_models` with only `"model"`→`"transformer"` swapped in the filter and the
`kind` literal. The `out` dict is constructed key-by-key from a fixed allowlist (never
spread from `entry`), so `abs_path` cannot leak by construction. `model_family`
`is not None` guard correct. No boundary violations, no scope creep, no security
regression, no dead code, no import churn.

**Non-blocking doc nit (ADDRESSED):** the handlers section-header comment named only
`list_models / list_loras`; appended `+ list_transformers (slice 2b)` to the line-1102
header (and the matching line-230 tool-surface header) before commit.

---

## Disposition

- security-auditor INFO: acknowledged, no change (Tb1+Tb2 combination is sufficient).
- code-reviewer doc nit: fixed pre-commit (section-header comments updated).
- Full suite green at the new count: `test_mcp_server.py` 375→394; total 1225→**1244/1244**.
