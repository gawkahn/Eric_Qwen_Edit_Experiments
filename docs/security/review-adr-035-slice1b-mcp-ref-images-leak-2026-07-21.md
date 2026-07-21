AI-Disclosure: Claude (Fable 5) authored this security review; Grant reviewed.

# Security Review — ADR-035 slice 1b: close MCP `extract_params` `ref_images` leak

Date: 2026-07-21
Reviewer: `security-auditor` (Fable)
Surface: `comfyless/mcp_server.py` (Red-Zone — LLM agent tool surface, ADR-011)
Verdict: **Pass for slice 1b.** No CRITICAL/HIGH. One pre-existing MEDIUM (not a
1b regression) + three INFO (forward guards / test strengthening).

## Summary

ADR-035 slice 1b adds `ref_images` to the drop-outright tuple in
`_render_extracted_params` (`comfyless/mcp_server.py:390-391`), closing the leak
slice 1 opened when `ref_images` became a recognized `SCHEMA_KIND` key:
`_handle_extract_params` normalizes sidecars with the pure `_validate_params`
(which keeps every schema key and bypasses the CLI's `_SKIP_SIDECAR_KEYS`), so a
crafted sidecar under `--output-dir` carrying `"ref_images":[{"path":"/abs/..."}]`
would previously have echoed absolute paths to the LLM agent. Threat model: an
agent-adjacent attacker (or the agent itself) plants a crafted JSON sidecar under
`--output-dir` and calls `extract_params`; the invariant is "no absolute path or
directory survives the MCP boundary."

**The fix is correct, unconditional, and complete on both extract render paths;
no other MCP surface echoes a sidecar's `ref_images`; the pin test genuinely
locks the fix.**

Reasoning walk-through:
1. **Non-cascade path** — `_handle_extract_params` runs `_validate_params`, which
   preserves the key name `ref_images` verbatim and keeps values even on type
   mismatch, so a `ref_images` of any shape (list-of-dicts, string, scalar)
   reaches `_render_extracted_params`; the drop at mcp_server.py:390-392 is
   `out.pop(k, None)` — keyed, shape-independent, unconditional, before any
   serialization, so no partial render is possible. `_validate_params` warnings
   name only key + type (never the value) and go to stderr (operator stream).
2. **Cascade path** — `_is_cascade_sidecar` routes to
   `_render_extracted_cascade_params`, which builds output solely from an
   allowlist (stage names, numeric fields coerced number-or-None, dtype fields
   value-allowlisted, prompt/negative_prompt/seed/model_family with type guards);
   `raw["ref_images"]` is never read — it cannot pass structurally in either
   detection branch.
3. **Other surfaces** — `extract_params` is the only sidecar/JSON-file reader in
   the module (only `json.load` at :2800-2801); the `generate` handler calls
   `generate()` with an explicit kwarg list containing no `ref_images`, and both
   response bodies are name-laundered / hand-built. grep for `ref_images` across
   `comfyless/` shows no other consumer.

## Findings

### [MEDIUM — PRE-EXISTING, not a 1b regression] Crafted strings survive extract_params through type-mismatched non-path schema fields
Location: `comfyless/generate.py:171-177` (`_validate_params` "Type mismatches →
KEPT") reached from `comfyless/mcp_server.py:2825`.
Risk: A crafted sidecar `{"steps": "/home/gawkahn/secret", "prompt": "p"}`
normalizes with the string kept (warn-and-keep), and `_render_extracted_params`
passes `steps` through verbatim — an absolute-path string egresses through a
numeric field, the class the cascade path closed with number-or-None coercion and
the LoRA-weight coercion. Marginal exfiltration value is low (prompt/negative_prompt
pass arbitrary sidecar text verbatim by design), but it breaks the letter of the
invariant and is asymmetric with the cascade render path. Pre-existing from slice
4; **out of this slice's edit scope — TECH_DEBT'd, not a 1b blocker.**
Remediation: In `_render_extracted_params`, coerce numeric/bool schema fields to
their expected type-or-drop (mirroring `_CASCADE_NUMERIC_FIELDS`), or have the
extract path call `_validate_params` in a strict drop-on-mismatch mode.

### [INFO] Slice-5 forward guard: MCP `generate` would accept agent-supplied `ref_images` as an unvalidated bare list
Location: `params_validation.py:80` (`"ref_images": _KIND_LIST`), `mcp_server.py`
generate call site. Today inert — `generate()` never forwards `ref_images` and no
error names it. When slice 5 wires `ref_images` into generation, the MCP surface
inherits an agent-supplied absolute-path INPUT channel (arbitrary local file read)
with no per-entry validation or containment.
Remediation: at slice 5, treat MCP `ref_images` like model/LoRA references
(per-entry shape validation + path containment, or reject until containment
exists), under its own security review.

### [INFO] Pin test is unit-level on the renderer, not end-to-end through `_handle_extract_params`
Location: `test_mcp_server.py` slice-1b pin test. It genuinely locks the fix
(removing `ref_images` from the drop tuple fails both checks), but no test feeds a
ref_images-bearing sidecar file through the full handler, so a future re-route
could bypass the pinned helper without failing this test.
Remediation: add one end-to-end case via `_write_sidecar` + `_call_tool_impl(cfg,
"extract_params", ...)`. **Addressed in this slice.**

### [INFO — PRE-EXISTING] `model_family` re-injected verbatim from the raw sidecar
Location: `mcp_server.py:394-395, 513-514`. A crafted sidecar can set
`"model_family": "/home/..."` and it echoes verbatim on both render paths — an
arbitrary-string egress channel. Non-additive (prompt passthrough already provides
one) and pre-existing; inconsistent with the cascade dtype value-allowlist.
TECH_DEBT'd with the MEDIUM above.

## Verdict

**Pass for slice 1b.** The leak the slice-1 reviewer found is closed:
mcp_server.py:390-392 drops `ref_images` unconditionally on the non-cascade path;
the cascade path structurally cannot emit it; no other MCP handler reads a sidecar
or forwards/echoes `ref_images`. Existing guarantees for the other dropped fields
and the catalog-name resolution are untouched. The pin test asserts both key
absence and no surviving `/home/` string and fails on drop-list removal. The
MEDIUM is a pre-existing slice-4 gap adjacent to (not created by) this change,
queued as its own slice; remediating it here would be scope creep.

Assumption named: exploitability today is nil because no writer records
`ref_images` in a sidecar until ADR-035 slice 5; the fix holds against a
hand-crafted sidecar regardless.
