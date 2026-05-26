# Security review — ADR-015 slice 2 step 4 (`list_models` / `list_loras` MCP tools + catalog-name sanitization)

**Date:** 2026-05-25
**Reviewer agent:** `security-auditor` (Opus)
**Reviewer model:** `claude-opus-4-7[1m]`. Invocation passed `model: "opus"` explicitly per the Claude Code 2.1.117 frontmatter-pin workaround documented in CLAUDE.md §5A.
**Slice docs:** `docs/vision/slice-2-mcp-catalog.md`, `docs/decisions/ADR-015-mcp-opaque-handles.md`
**AI-Disclosure:** Claude (Opus) authored; Grant reviewed.
**Verdict:** APPROVED — 1 LOW (folded same-commit), 2 INFO (one folded to TECH_DEBT, one observation-only).

---

## Summary

Step 4 lights up the agent-facing catalog discovery surface: two new MCP
tools (`list_models`, `list_loras`) enumerate `kind:"model"` /
`kind:"lora"` catalog entries with a strict allowlist (name, kind,
source + optional model_family or target_family) and never serialize
`abs_path` under any key. Catalog-name sanitization adds a build-time
gate that rejects C0/C1 controls, zero-width chars, bidi overrides/
isolates, and LINE/PARAGRAPH SEPARATOR before NFC normalization —
closing Step-2 INFO-2 and giving slice-3 reference resolution a clean
round-trip-key contract.

Threat model: same-uid trust at spawn; agent-supplied (untrusted)
tool-call arguments after spawn. **First slice where catalog content
reaches an MCP-agent-facing response surface** (slices 1 / 2-step-3
only built the catalog; slices 4+ resolve names from it). The diff
hardens the agent boundary without changing the operator boundary.
Overall posture: clean. No CRITICAL/HIGH findings; one LOW around
audit-line echoing of agent-supplied `arguments` for the new tools
(folded), two INFO observations on robustness/future-proofing (one
TECH_DEBT-bound, one observation-only).

## Coverage

Reviewed:
- `comfyless/catalog.py` full file (focus on `_FORBIDDEN_NAME_CHARS`
  + `_add_entry` gate)
- `comfyless/mcp_server.py` full file (focus on `_LIST_*` constants,
  `_emit_audit_line` `result_count` kwarg, `_list_tools_impl` 1→3
  growth, dispatch, `_handle_list_models`, `_handle_list_loras`)
- `test_mcp_server.py` slice-2 step-4 section + sanitization cases

Not reviewed (declared out of scope):
- `_handle_generate` / `_handle_generate_cascade` (unchanged per I14)
- Catalog-build scan/manifest internals (already audited at Step 2)
- Pre-existing slice-1 audit-line / `_sanitize_error` semantics
  (unchanged shape; only `result_count` kwarg is new)

## Findings

### [LOW] list_* tools echo unvalidated agent-supplied `arguments` into the audit line — FOLDED

Location: `comfyless/mcp_server.py` `_call_tool_impl` (dispatch +
audit emission).
Risk: `_handle_list_models(cfg)` / `_handle_list_loras(cfg)` ignore
`arguments` entirely (their signatures take only `cfg`), but
`_call_tool_impl` originally passed `arguments` to `_emit_audit_line`
on both success and error paths. The framework decorator uses
`validate_input=False` (deliberate per invariant 5 — keeps audit
emission unconditional). The list_* tools' empty-object schemas with
`additionalProperties: false` are NOT enforced by the framework. An
agent could call `list_models` with `{"x": "A" * 100_000}` and the
entire payload would be written verbatim to stderr through
`json.dumps(line, default=str)`. This is the operator's audit channel
(same channel as tracebacks), so it is not an agent-facing leak; the
harm is bounded to (a) audit-log bloat / stderr flooding, (b) trivial
log-injection into the operator's log aggregator if one slurps stderr
blindly. The handlers honor the "no inputs" schema correctly; the
audit line was the only surface that echoed the payload.

**Fix folded same-commit:** at the top of `_call_tool_impl`, an
`audit_payload` local is set to `{}` when `name in ("list_models",
"list_loras")` and to `arguments` otherwise; success and error audit
emission both use `audit_payload`. Regression test N19b in
`test_mcp_server.py` plants a 10 KB flood blob, confirms it does not
appear in stderr, and includes a carry assertion that `generate`'s
audit line STILL echoes agent arguments (operator-visible signal for
legitimate calls preserved).

### [INFO] `_FORBIDDEN_NAME_CHARS` allowlist intentionally narrow — TECH_DEBT entry filed

Location: `comfyless/catalog.py:_FORBIDDEN_NAME_CHARS`.
Risk: The class set covers C0/C1, ZWSP/ZWNJ/ZWJ + LRM/RLM (U+200B-
200F), bidi overrides (U+202A-202E), line/para separators (U+2028-
2029), and bidi isolates (U+2066-2069). Codepoints **not** in the
class that are plausible adversarial-name confusables: BOM /
ZERO WIDTH NO-BREAK SPACE (U+FEFF), SOFT HYPHEN (U+00AD),
MONGOLIAN VOWEL SEPARATOR (U+180E), INTERLINEAR ANNOTATION ANCHOR
/SEPARATOR/TERMINATOR (U+FFF9-U+FFFB). Threat model: same-uid
operator-supplied names; an adversary at the same uid can already
plant anything. The omission is aesthetic/UX (two visually-identical
names mapping to different catalog entries → slice-3 reference
round-trip becomes operator-visually-ambiguous), not an exploit
surface. NFC normalization does not collapse any of these to a
sibling form.

**Filed:** `TECH_DEBT.md` → "Catalog-name allowlist intentionally
narrow" (2026-05-25). Trigger to revisit: first slice-3 agent UX
report of "two catalog entries look identical to me but resolve
differently"; or any threat-model change to multi-tenant MCP transport.
Fix shape: extend the regex one-liner with the additional ranges
(`\ufeff\u00ad\u180e\ufff9-\ufffb`) — no behaviour change on
existing valid names.

### [INFO] `_emit_audit_line` `default=str` fallback can stringify exotic payload values — pre-existing

Location: `comfyless/mcp_server.py:_emit_audit_line`.
Risk: `json.dumps(line, default=str)` is the existing slice-1 pattern
(unchanged). For `list_*` tools this is fine — the new `count` field
is always `int` and `result_count` is typed `Optional[int]`. Not a
step-4 regression. With LOW-1 above folded (`audit_payload = {}` for
list_*), the dispatch's `arguments` dict no longer reaches audit
emission for tools that accept no inputs, so the concern is moot for
the new surface. **No action.** Observation only.

---

## Vision invariant verification

- **I8** (3 tools advertised) — verified at `_list_tools_impl` and
  tested at N20.
- **I9** (strict allowlist, no abs_path, transformer hidden) —
  verified at `_handle_list_models` / `_handle_list_loras`; tested at
  N16 / N17 / N18 / N27.
- **I10** (audit on every list_* with tool / count / status /
  elapsed; no abs_path) — verified at `_call_tool_impl` + `_emit_
  audit_line`; tested at N19 + N19b (audit-payload bound).
- **I11** (traceback strip) — relies on existing `_sanitize_error`
  (unchanged) reached via `_call_tool_impl`; tested at N22.
- **I14** (`generate` byte-identical) — `_GENERATE_INPUT_SCHEMA` and
  `_GENERATE_TOOL_DESCRIPTION` not edited; byte-equality tested at
  N20.

## Audit-dimension answers (from the brief)

1. **`abs_path` leak audit:** handlers build response dicts from a
   hardcoded key allowlist; `entry["abs_path"]` is never read;
   `json.dumps(entries, ...)` only serializes the `out` dicts;
   exception in handler → caught by `_call_tool_impl`'s
   `except BaseException`, full traceback to stderr only (operator
   channel), category-only string to agent. **CLEAN.**
2. **Regex correctness:** regex matches the documented ranges. INFO
   above for the intentionally-narrow allowlist; not an exploit
   surface at same-uid trust.
3. **`_add_entry` gate ordering:** rejection runs before NFC. Every
   codepoint in `_FORBIDDEN_NAME_CHARS` is a format/control codepoint
   with no NFC decomposition. No non-forbidden BMP codepoint NFC-
   decomposes into one of these (canonical decomposition tables
   never produce format/control output). Order is safe. **CLEAN.**
   Code-reviewer LOW-1 (NFC-stability note folded into the gate
   comment) also covers this for future maintainers.
4. **`result_count` audit extension:** `int` value, emitted as plain
   JSON number, non-sensitive (entry count not entry content).
   **CLEAN.**
5. **Handler exception surfaces:** `json.dumps` cannot reach
   `entry["abs_path"]` because the handler never copies it into the
   `out` dict; the only way `abs_path` could surface in an exception
   message is if a handler explicitly read `entry["abs_path"]` and
   put it into a raised string — no such code path exists. **CLEAN.**
6. **Dispatch flow:** `name` is the MCP framework's tool name
   (already string-typed by the JSON-RPC frame). Empty string `""`
   and case-variants like `"List_Models"` fall through the `elif`
   chain to the `else` branch → `_MCPHandlerError("UnknownTool",
   ...)`. **CLEAN.**
7. **list_* input schema:** see LOW-1 above (folded).
8. **Catalog iteration:** built at spawn via `_validate_startup_args`
   → `build_catalog`, stored on `_StartupConfig.catalog` slot. No
   request-time write path exists in production code (test N18
   directly mutates the dict to inject fixture entries — acceptable
   test-only inspection). `cfg.catalog.items()` iteration in
   handlers is read-only. **CLEAN.**
9. **Sort stability:** NFC-normalized codepoint-order is
   deterministic across calls within one server lifetime. **CLEAN.**
10. **N18 fixture-direct-injection:** production code's only catalog
    writer is `build_catalog` → `_add_entry`. The test bypass is
    acceptable as test-only inspection of the no-leak property
    because it isolates the response serializer from the build-time
    gate. No production callsite mutates `cfg.catalog` after spawn —
    confirmed by grep across `comfyless/`. **CLEAN.**

## Conclusion

Slice 2 step 4 is approvable. The single LOW finding (audit-line
echo of unvalidated `arguments` for tools that accept no inputs) is
folded same-commit with regression test N19b. The two INFO items are
either filed to TECH_DEBT (allowlist breadth) or observation-only
(pre-existing pattern).
