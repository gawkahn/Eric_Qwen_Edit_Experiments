# Security Audit — Slice 1 Step 3 (ADR-011 MCP Cascade Dispatch)

**Date:** 2026-05-17
**Reviewer:** `security-auditor` subagent (Opus, model pinned at invocation per project CLAUDE.md review-bar)
**Scope:** Slice 1 step 3 of ADR-011 — Stable Cascade dispatch via `cascade_config`. Files: `comfyless/mcp_server.py` (`_handle_generate_cascade` + `_MCP_CASCADE_PATH_TYPED_FIELDS` + extended `redact_metadata_for_png`), `comfyless/cascade.py` (surgical — `_save_with_metadata` mcp_caller kwarg), `test_mcp_server.py` (+28 step-3 assertions).
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored the audit; Grant reviewed.

---

## Verdict

**CLEAN.** No findings at any severity. All 10 attack-surface concerns trace end-to-end cleanly. Step-2 carry-forwards F1-F7 preserved in cascade. ADR-011 §3b's in-process trust gap ("cascade `stage_*` and `scaffolding_repo` fields gain the same allowlist enforcement as the standard `model` field when reached via MCP") is closed structurally.

---

## Threat model

Same-uid LLM agent drives `generate` over stdio MCP, supplying `cascade_config` to invoke Stable Cascade. Trust boundaries crossed in step 3:

1. JSON-RPC frame → `_call_tool_impl`
2. raw `arguments` → `validate_machine_request`
3. `cascade_config` sub-dict → `cascade.validate_config`
4. cascade paths → `resolve_hf_path(allow_download=False)` → `_within(--model-base)`
5. cascade `build_pipelines` → diffusers load (in-process)
6. PIL save → `redact_metadata_for_png` → on-disk PNG

The agent is the attacker; secrets are everything outside `--model-base` and `--output-dir`.

---

## Coverage

- `comfyless/mcp_server.py:1-999` (full module incl. `_handle_generate_cascade` lines 669-822 and `redact_metadata_for_png` cascade branch lines 127-135)
- `comfyless/cascade.py:1-960` (`validate_config`, `_resolve_scaffolding`, `build_pipelines`, `_save_with_metadata` mcp_caller branch lines 587-608)
- `comfyless/params_validation.py:1-270` (verified `validate_machine_request` passes unknown top-level keys like `cascade_config` through unchanged — line 252-254)
- `comfyless/server.py:1-200` (daemon-parity reference: `_within`, `_check_paths`, null-byte gate at 138-149)
- `nodes/eric_diffusion_utils.py:89-150` (`_is_hf_repo_id`, `resolve_hf_path` cache-miss path)
- `comfyless/generate.py:385-470` (`_expand_savepath_template`, `_resolve_savepath` — basename-token semantics)
- `test_mcp_server.py:985-1245` (step-3 assertions: N19, N20, N21, N22 + cascade NUL + missing-prompt + cascade_config-not-a-dict + redaction unit + `_MCP_CASCADE_PATH_TYPED_FIELDS` hygiene)
- `docs/security/review-slice-1-mcp-step2-2026-05-17.md` (F1-F7 carry-forward verification)

**Not reviewed (and why):**
- Live MCP framework wrapping (`mcp.server.Server` / `stdio_server`) — out of step-3 scope; step-1 audit covered framework outer-except shape.
- `test_cascade.py` / `test_server_robustness.py` — exercise CLI-side cascade, not the MCP path that step 3 adds.
- Live GPU smoke — step 3 uses mocked `build_pipelines` + `run_one` per the N21 spy pattern; live cascade exercise is deferred per ADR-011 §6.

---

## Audit detail (10 attack-surface concerns from the brief)

### 1. Handler ordering — verified ✓

Sequence at `_handle_generate_cascade`: audit-line setup (outer try at `_call_tool_impl`) → `validate_machine_request` → required-prompt gate (line 696) → top-level NUL gate on `savepath` (line 702) → `cascade.validate_config` (line 720) → cascade NUL gate on stage_*/scaffolding_repo (line 732) → `resolve_hf_path(allow_download=False)` per cascade field (line 746) → `_within(model_base)` per resolved cascade path (line 765) → `_resolve_mcp_output_path` (line 777) → `cascade.build_pipelines(allow_hf_download=False)` (line 787) → `cascade.run_one` (line 794) → `cascade._save_with_metadata(mcp_caller=True)` (line 808) → inline response. Matches the spec verbatim.

### 2. Cascade null-byte gate fires BEFORE realpath — verified ✓

`mcp_server.py:732-737` runs the NUL gate after `cascade.validate_config` (which does not call realpath — only `setdefault` + int/float coercion + `_align_cascade_dim`) but BEFORE `resolve_hf_path` and `_within`. `resolve_hf_path` on a non-HF path with NUL returns the value unchanged (`_is_hf_repo_id` rejects paths starting with "/"); the within-check at line 765 then calls `os.path.realpath` which would raise — but the explicit NUL gate catches it first and emits `ValidationError` with the correct audit class. Test `"cascade null-byte audit class is ValidationError"` + `"InternalError not in stderr"` confirms.

### 3. Cascade missing-prompt gate fires BEFORE build_pipelines — verified ✓

`mcp_server.py:696-700` runs the prompt gate well before the `build_pipelines` call at line 787 (which loads 3 stages — more expensive than non-cascade's single load). Test `"cascade missing-prompt → MissingField (BEFORE build_pipelines)"` confirms.

### 4. `cascade.validate_config` error sanitization — verified ✓

`mcp_server.py:723-729`: handler catches `(ValueError, TypeError)` and re-raises with `f"validation failed: cascade_config: {type(e).__name__}"` — only the exception class name (e.g. `"ValueError"` / `"TypeError"`) is exposed. The raw `e.args[0]` from cascade's validate_config (which contains the offending field name and may contain caller-supplied values via raw dict shape) is NOT exposed. `from None` suppresses the cause chain so the framework's `str(e)` on the outer `ValueError` sees only the pre-sanitized safe_message.

### 5. `allow_hf_download=False` enforcement — verified ✓

Two layers + one downstream:
- (a) MCP handler explicitly calls `resolve_hf_path(field, allow_download=False)` at `mcp_server.py:746`
- (b) MCP handler explicitly passes `allow_hf_download=False` to `cascade.build_pipelines` at `mcp_server.py:788`
- (c) `cascade.py:286` forwards that False to `_resolve_scaffolding` — cascade's ONLY internal `resolve_hf_path` call site (verified by grep). `_load_unet` and `_load_stage_a` use `os.path.isdir` / `from_pretrained` / `from_single_file` on already-resolved local paths and do NOT touch `resolve_hf_path`.

N21 spy test `"every recorded allow_download is False"` confirms.

### 6. Cascade-side allowlist closure ordering — verified ✓

`resolve_hf_path` happens at lines 740-753 BEFORE `_within` at lines 755-769. An HF repo ID that translates to a local cache path outside `--model-base` correctly produces `PathAllowlist` (not `HFCacheMiss`). The default `scaffolding_repo="stabilityai/stable-cascade"` — set by `cascade.validate_config`'s setdefault — will resolve to `~/.cache/huggingface/.../snapshots/...` if cached, then correctly fail `_within(--model-base)`. This is the intended fail-closed behavior per ADR-011 §3b ("closes the existing in-process trust boundary").

### 7. Cascade audit-line correctness — verified ✓

`_emit_audit_line` writer at `mcp_server.py:426-448` receives raw `arguments` dict including the full `cascade_config` sub-dict. `_AUDIT_DROPPED_FIELDS = {"prompt", "negative_prompt"}` drops top-level prompt/negative_prompt only; cascade_config.stage_* / scaffolding_repo paths are retained verbatim. Test `"N22: cascade audit retains stage_c"` + `"N22: cascade audit drops prompt"` confirm both directions on the success path. Audit emission uniform across success and rejection paths.

### 8. Cascade PNG redaction — verified ✓

`redact_metadata_for_png` at `mcp_server.py:127-135`: creates a NEW `cc = dict(out["cascade_config"])` (does not mutate input), basenames each path field via `_basename_or_repo_id` (HF repo IDs unchanged), preserves non-path fields verbatim. `output_path` / `savepath` are dropped at top level via the existing `out.pop` calls at lines 136-137. Tests `"cascade redaction: HF repo IDs in cascade_config pass through"` + `"cascade redaction: non-path fields retained"` + `"cascade redaction: output_path dropped"` confirm.

### 9. Stub model field — verified ✓

`mcp_server.py:777`: the "stablecascade" sentinel only reaches `_resolve_mcp_output_path → _expand_savepath_template → Path(model_path).name` (basename extraction for `%model%` token). No path-flow consequence; final output_path is `_within(--output-dir)`-checked at line 869. Code-reviewer's F8 fold-in adds a comment documenting the asymmetry vs `_handle_generate`.

### 10. `from None` cause-chain suppression — verified ✓

Two new cascade raise sites: `mcp_server.py:729` (cascade.validate_config catch) and `mcp_server.py:753` (HFCacheMiss). Both correctly suppress `__cause__`.

---

## Absence check (step-3-scoped)

Items NOT in the step-3 diff that are correctly NOT in the step-3 diff:

- **Daemon delegation** — still deferred per ADR-011 / step-2 TECH_DEBT entry (correct).
- **Cascade-specific iterate batch handling** — step 3 single-shot only; iterate axis lands in a later slice (correct).
- **F3 LOW (HF-cache enumeration oracle)** — unchanged from step 2; cascade extends the same observable failure-mode discrimination (PathAllowlist vs HFCacheMiss). Tracked in TECH_DEBT for HTTP-transport trigger; no new step-3 incremental risk.
- **F6 (audit-drop set vs PNG-redaction tuple drift)** — step 3 adds `_MCP_CASCADE_PATH_TYPED_FIELDS` as a tuple consumed by both the handler's resolve/allowlist/NUL-gate loops AND the redaction map. No drift surface.
- **F7 (unbounded growth of `_audit_write_failures`)** — unchanged from step 2; same-uid stdio threat-model floor still bounds practical impact.

---

## Code-reviewer carry-forward notes

From the parallel code-reviewer (Opus) pass:

- **F8 INFO** (cascade model="stablecascade" sentinel comment) — folded; mcp_server.py:777 carries a comment documenting the asymmetry vs non-cascade `_handle_generate`.
- **F9 record-only note** (audit uses raw `arguments` not validated `payload`) — by design; matches step-2 F5. Recorded so step 4 doesn't "fix" this by switching to validated_cc. The audit must reflect what the agent actually sent.

Code-reviewer also flagged two hidden surprises worth verifying for step 4:

1. **`validate_machine_request` cascade_config passthrough** — verified in this audit (coverage section): the canonical validator at `params_validation.py:252-254` passes unknown keys (including `cascade_config`) through unchanged. No spot-check needed in step 4; behavior is structural.
2. **`metadata["cascade_config"] = resolved_cc`** in the in-frame response carries the LOCAL resolved paths (not the agent's input HF repo IDs) — intentional per invariant 12 N25 ("the in-frame blob is the agent's authoritative record"). Consistent with non-cascade behavior; no change needed.

---

## Scope creep

None. Diff bounded to `comfyless/mcp_server.py` (handler + redaction extension + tuple constant), `comfyless/cascade.py` (surgical `mcp_caller` kwarg + lazy import of `redact_metadata_for_png`), and `test_mcp_server.py`. No `params_validation.py` / `params_schema.py` / `server.py` / `nodes/*` changes — consistent with the declared step-3 edit scope.

---

**Approve labels:** `APPROVED` — no `BOUNDARY VIOLATION`, no `PROMISE DRIFT`, no `SECURITY REGRESSION`, no `SCOPE CREEP`. Cascade dispatch closes ADR-011 §3b's in-process trust gap structurally.
