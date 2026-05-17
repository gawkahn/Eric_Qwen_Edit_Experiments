# Security Audit — Slice 1 Step 2 (ADR-011 MCP `generate` Handler Wiring)

**Date:** 2026-05-17
**Reviewer:** `security-auditor` subagent (Opus, model pinned at invocation per project CLAUDE.md review-bar)
**Scope:** Slice 1 step 2 of ADR-011 — substantive `_call_tool_impl` / `_handle_generate` implementation; `redact_metadata_for_png` PNG-redaction helper; `_save_with_metadata` / `generate()` `mcp_caller` propagation; `test_mcp_server.py` extension; `TECH_DEBT.md` "daemon delegation deferred" entry.
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored the audit; Grant reviewed.

---

## Verdict

**CHANGES_REQUIRED at audit time** — 2 MEDIUM (F1 + F2 below), 1 LOW (F3 TECH_DEBT), 4 INFO. The 2 MEDIUMs are short, targeted edits inside `_handle_generate`; both were folded into the slice-1-step-2 commit before merge. F3 LOW landed as a `TECH_DEBT.md` entry. F4-F7 are forward-looking observations for step-3 reviewer attention.

**Post-fold verdict: CLEAN.** All step-2 promises hold; daemon-parity gaps closed; audit-class labelling correct on every rejection path.

---

## Coverage

- `comfyless/mcp_server.py:1-790` (full module, including the F1/F2 fold-in)
- `comfyless/generate.py:466-976` (`_save_with_metadata` + `generate()` edit surface)
- `comfyless/server.py:1-360` (verbatim-reused `_within`, `_check_paths`, and the daemon's null-byte/missing-field gates the MCP handler now mirrors)
- `comfyless/params_validation.py:1-270` (`validate_machine_request` semantics; type-only)
- `nodes/eric_diffusion_utils.py:89-155` (`_is_hf_repo_id`, `resolve_hf_path` cache-miss semantics)
- `test_mcp_server.py:1-1010` (130 assertions; N5-N16, N23-N31 + F1/F2 fold-in coverage)
- Vision `slice-1-mcp-generate.md` (15 invariants + N1-N33 cases)
- Prior step-1 security review (F3, F4, F5 carry-forwards)

---

## Findings

### F1 [MEDIUM] Null-byte path bypasses MCP `ValidationError` and reaches `os.path.realpath`, producing `InternalError` audit class instead of `PathAllowlist` / `ValidationError`

**Location:** `comfyless/mcp_server.py:_handle_generate` steps 1-5 vs. `comfyless/server.py:138-149` (daemon's pre-realpath null-byte rejection).

**Risk:** The daemon's `_validate_request` includes an explicit pre-realpath null-byte rejection for `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `savepath`, and `loras[*].path`. Rationale at `server.py:53-54`: "`os.path.realpath` raises on NUL and that exception would escape `_check_paths` and kill the accept loop." The MCP handler did not import or reproduce this check. An agent supplying `model="/abs/path/with/\x00null"` flows: `validate_machine_request` OK → `resolve_hf_path` returns unchanged (not HF) → `_check_paths` calls `_within → os.path.realpath` → `ValueError("embedded null byte")` propagates out → outer `BaseException` handler labels it `InternalError` and writes the full traceback to stderr. The agent gets `"internal_error: ValueError"` (sanitized) but the audit line records `error_class="InternalError"` instead of the correct `ValidationError` class, and the stderr-side traceback exposes the path-with-NUL on a stream the Vision marks as audit-only. Medium rather than high because the boundary still fails closed (no model loaded, no file written) and the agent-facing sanitization holds. The harm is daemon-parity: audit-class mislabelling defeats per-class rate-detection a future slice's review will lean on; the traceback-to-stderr pathway on malformed input is noisier than the spec describes.

**Status (post-fold): RESOLVED.** Inserted between `validate_machine_request` and the default-model fallback (step 1.6 in code comments). Mirrors the daemon's check; raises `_MCPHandlerError("ValidationError", "validation failed: <field>: null byte not allowed")` for all six path-typed scalar fields + each `loras[].path`. Tests `F1: null byte in <field> → MCP error` + `F1: audit class is ValidationError` cover every field; 14 assertions pass.

### F2 [MEDIUM] Missing-required-field (`prompt`) reaches `payload["prompt"]` KeyError, producing `InternalError` instead of `ValidationError` — and only AFTER expensive `_load_pipeline` (30-90s wasted)

**Location:** `comfyless/mcp_server.py:574-602` (the `generate(prompt=payload["prompt"], ...)` call).

**Risk:** `validate_machine_request` does not enforce required-field presence (per its own docstring: "Unknown-key rejection is out of scope for this slice"). The daemon's `_validate_request` adds the missing-`prompt`/missing-`model` gate at `server.py:133-136`. The MCP handler covered missing `model` via the `--default-model` fallback (returning `MissingField` correctly when no default is configured), but did NOT cover missing `prompt`. The framework runs the handler with `validate_input=False` by design, so JSONSchema's `"required": ["prompt"]` is not enforced before the handler. An agent that sends `{"model": "/path"}` with no `prompt` field flows: validate succeeds → default-model handled → HF resolution OK → `_check_paths` OK → output-path resolution OK → `_load_pipeline` runs (expensive!) → `generate(prompt=payload["prompt"], ...)` raises `KeyError: 'prompt'` → outer `BaseException` catch → `error_class="InternalError"`. Two harms: audit class is wrong, and the model is loaded into VRAM before the failure, wasting 30-90s and burning VRAM on a malformed request.

**Status (post-fold): RESOLVED.** Inserted alongside the null-byte check (step 1.5 in code comments). Mirrors the daemon's missing-prompt gate; raises `_MCPHandlerError("MissingField", "validation failed: prompt: required field absent")`. Empty-string and whitespace-only prompts also rejected. Tests `F2: missing prompt → MCP error (BEFORE load)` + `F2: audit class is MissingField` + `F2: whitespace-only prompt → MCP error` pass.

### F3 [LOW] HF-cache hit vs miss is observably distinguishable from the agent — coarse-grained enumeration oracle on cached repo IDs

**Location:** `comfyless/mcp_server.py:491-534`

**Risk:** Step 4 (HF resolution) runs BEFORE step 5 (path allowlist). For an HF repo ID input, `resolve_hf_path` returns either a local cache directory or raises `LocalEntryNotFoundError → ValueError`. The handler maps the latter to `HFCacheMiss`; for a cached-but-outside-`--model-base` repo, step 5 produces `PathAllowlist`. An LLM agent probing repo IDs can distinguish the three outcomes (cached + inside base = success; cached + outside base = `PathAllowlist`; not cached = `HFCacheMiss`), and so enumerate which HF repos are present in the local cache. The repo ID itself is correctly suppressed from both error strings (colon-split + `from None`); the per-class signal remains. Same-uid threat model means the agent is already authorized to drive generation; the cache contents are not an independent secret.

**Status: TECH_DEBT.** Entry "MCP server: HF-cache-hit vs cache-miss is observably distinguishable to the agent" added to `TECH_DEBT.md`. Trigger to revisit: HTTP-transport ADR, multi-tenant agent surface, or any threat-model change. Fix shape: unify the agent-facing error class while retaining the finer-grained audit classes on stderr.

### F4 [INFO] Cascade-branch and unknown-tool-branch raise discipline — closed during fold-in

**Location:** `comfyless/mcp_server.py:397-414` (the two pre-handler branches in `_call_tool_impl`).

**Original concern:** Both branches raised `ValueError(...)` outside the `_MCPHandlerError` / `_sanitize_error` discipline. Today's messages were hand-constructed string literals (`repr()` of the tool name is json-quoted) so the framework's `str(e)` outer-convert produced nothing exfiltratable. Pattern was fragile: any future maintainer adding interpolated runtime state would route around the sanitizer.

**Status (post-fold): RESOLVED in code-reviewer's parallel review.** Both branches moved inside the `try:` block and routed through `_MCPHandlerError(...)`. F4 anti-pattern fully closed.

### F5 [INFO] `_emit_audit_line` audits the agent-supplied `arguments` rather than the post-validation `payload` — intentional and correct, recorded

**Location:** `comfyless/mcp_server.py:418-441`.

The audit-line writer is invoked with the raw `arguments` dict in every code path. The validated/resolved `payload` (with int→float casts and HF-resolved paths) is NOT what the operator sees. This is correct per invariant 5 wording: the audit records what the agent actually sent. Three distinct redaction policies for three distinct consumers — by design.

### F6 [INFO] `_MCP_PATH_TYPED_FIELDS` and `_AUDIT_DROPPED_FIELDS` structurally independent — drift hazard relies on the audit's drop-list shape

**Location:** `comfyless/mcp_server.py:57-66`.

Vision invariant 12 says: "The MCP redaction map is shared in code with the §3b audit-line field list so the two cannot drift." Step 2 satisfies this in spirit — `_save_with_metadata`'s `mcp_caller=True` branch imports `_MCP_PATH_TYPED_FIELDS` directly from `comfyless.mcp_server` (single source of truth). The remaining drift surface is between PNG-redaction and audit-line redaction (which uses `_AUDIT_DROPPED_FIELDS = frozenset({"prompt", "negative_prompt"})` — a drop-list shape, not a tuple-derived shape). If step 3 inverts the audit policy when adding cascade fields, link the two sets explicitly.

### F7 [INFO] Unbounded growth of `_audit_write_failures` under sustained stderr failure

**Location:** `comfyless/mcp_server.py:72`.

If stderr is closed/broken, every audit-emission attempt appends a `time.monotonic()` to the list. Unbounded per request. Same-uid stdio MCP threat model bounds practical impact. A bounded `collections.deque(maxlen=1024)` would close this if a future slice exposes the counter to an admin tool or watchdog.

---

## Absence check (step-2-scoped)

Items NOT in the step-2 diff that are correctly NOT in the step-2 diff:

- Cascade-side path-allowlist (step 3 — `cascade_config` branch is a rejection-and-defer in step 2)
- Cascade-side audit fields (step 3)
- Cascade-side PNG redaction (step 3 will extend `_MCP_PATH_TYPED_FIELDS`)
- Daemon delegation (deferred per Grant's step-2 scope sign-off; TECH_DEBT entry recorded)
- HTTP transport (out of slice 1; ADR-011 §6 stdio-first)
- Other 5 MCP tools (out of slice 1)
- Step-1 code-reviewer F3's FD-level stderr capture — step 2 mocks torch/diffusers entirely; deferred to step 4 live smoke

---

## Scope creep

None observed beyond the F4 cascade/unknown-tool rejection refactor (folded during code-review review) and the F1/F2 daemon-parity additions (folded during security-auditor review). The diff is bounded to `comfyless/mcp_server.py`, `comfyless/generate.py` (surgical), `test_mcp_server.py`, `test_params_schema.py` (1-line exclude), and `TECH_DEBT.md`.

---

## Step-3 reviewer carry-forward

The step-3 reviewer should verify:

1. Cascade-side path-allowlist (cascade `stage_c`/`stage_b`/`stage_a`/`scaffolding_repo`) goes through the same realpath + `_within(--model-base)` discipline.
2. `_MCP_PATH_TYPED_FIELDS` extends to cover cascade fields; `redact_metadata_for_png` continues to consume the single tuple.
3. Any new pre-try-block branches added by step 3 route through `_MCPHandlerError` (F4 anti-pattern closure preserved).
4. Cascade's `build_pipelines` and `_resolve_scaffolding` are called with `allow_hf_download=False` HARD-CODED (P3 cascade-side extension).
5. F6 (audit-drop set vs PNG-redaction tuple drift) is closed structurally if cascade audit policy inverts.

F3 LOW (HF cache enumeration oracle) does NOT need step-3 attention; tracked in TECH_DEBT for HTTP-transport or multi-tenant trigger.

---

**Approve labels (post-fold):** `APPROVED` — no `BOUNDARY VIOLATION`, no `PROMISE DRIFT`, no `SECURITY REGRESSION`, no `SCOPE CREEP`. F3 LOW captured in TECH_DEBT; F4-F7 forward-looking.
