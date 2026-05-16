# Security Audit — Validator Slice Step 3 (ADR-012)

**Date:** 2026-05-16
**Reviewer:** `security-auditor` subagent (Opus 4.7, 1M context, model pinned at invocation per project CLAUDE.md review-bar)
**Scope:** Step 3 of the machine-boundary validator slice — `comfyless/server.py:_validate_request` rewrite + N18 cross-site / N19 AST grep activation in `test_machine_boundary_validator.py`.
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored the audit; Grant reviewed.

---

## Verdict

**CLEAN with two MEDIUM observations and several INFO notes** (audit-as-delivered). Both MEDIUMs addressed in step-3 fold-in before commit.

The rewrite preserves the existing IPC security posture, correctly activates the bool-as-int closure for every numeric field at the daemon boundary, preserves the null-byte defense, and introduces no fail-open path.

---

## Findings

### 1. INFO — Bool-on-int closure correctly active at the daemon boundary

**Severity:** INFO
**Location:** `comfyless/server.py:110-115` (canonical-validator call) feeding `comfyless/params_validation.py:121-135` (`_KIND_INT` branch)

Pre-rewrite, `_validate_request` used `isinstance(req[field], int)` which accepts `True`/`False`. Post-rewrite, the canonical validator's `_KIND_INT` branch rejects `bool` BEFORE the `int` accept. This closes the loophole for all five canonical-int fields: `seed`, `steps`, `width`, `height`, `max_sequence_length`. The parametric test iterates `SCHEMA_KIND` and asserts rejection for both `True` and `False`, so any future canonical-int field added to the schema is self-covered.

Boolean runtime fields (`offload_vae`, `attention_slicing`, `sequential_offload`) declared `_KIND_BOOL` in `_RUNTIME_KIND` correctly accept bool only. `True`/`False` accepted; `0`/`1` rejected.

**Recommendation:** None.

---

### 2. INFO — Required-field-presence ordering is safe; no bypass path

**Severity:** INFO
**Location:** `comfyless/server.py:110-130`

There is no code path where a payload with `type="generate"` reaches a `req["model"]` or `req["prompt"]` access without going through the presence check at lines 127-130. The ordering is safe.

**Recommendation:** None.

---

### 3. INFO — Type-tag attack surface — no TOCTOU or partial-state hazard

**Severity:** INFO
**Location:** `comfyless/server.py:110-123`

The canonical-first ordering imposes no additional side-effect surface beyond what step 1 already audited as benign. The validator is pure (step-1 finding 4); MappingProxyType prevents schema mutation (step-1 finding 7).

**Recommendation:** None.

---

### 4. INFO — Null-byte defense complete; canonical type-checks make `_PATH_FIELDS` loop tight

**Severity:** INFO
**Location:** `comfyless/server.py:137-143`

By the time the null-byte loop runs, every `_PATH_FIELDS` value is guaranteed `str` (or absent) by the canonical validator's `_KIND_STR` rejection. The LoRA path access at line 138 is similarly tight: `validate_lora_entry` guarantees `loras[i]` is a dict with `path` key present and `path` value is a str.

The decision to keep null-byte rejection in server.py rather than migrating into the canonical validator was discussed in step-1's audit. The rewrite's choice is defensible — null-byte rejection is filesystem-defense-in-depth, adjacent to but not the same concern as type validity. Centralizing it would also tighten `prompt`/`negative_prompt` unnecessarily.

**Recommendation:** None on code. Slice commit body references the comment.

---

### 5. INFO — IPC error-message disclosure acceptable for threat model

**Severity:** INFO
**Location:** `comfyless/server.py:113-115`

The new error strings interpolate `err['field']` and `err['reason']` from the canonical validator's structured output. Reason strings like `"bool not accepted for int field"` disclose the canonical type rules to the caller. Per step-1 finding 5, `field` is always either a validator-controlled key in `_ALL_FIELDS` or the literal `"<root>"` or the `loras[N].{path,weight}` format. No caller-controlled string is reflected in `field`.

**Recommendation:** None.

---

### 6. MEDIUM — Validated payload (with int→float casts) is discarded

**Severity:** MEDIUM (behavioral parity gap, not a security defect; regression-resistance concern)
**Location:** `comfyless/server.py:110-115` pre-fold-in

The canonical validator returns a `result.payload` dict with int→float safe casts applied per ADR-012 §3 — `cfg_scale=4` becomes `cfg_scale=4.0`. The original rewrite checked `result.ok` and `result.error` but threw away `result.payload`. Downstream `req` held the un-cast `int` values.

**Today this is not exploitable** — downstream consumers happen to be type-agnostic. The MEDIUM is for the regression-resistance posture: the validator's published contract differs from observable downstream behavior, which is exactly the drift class ADR-012 was written to eliminate.

**Status:** ADDRESSED in step-3 fold-in — `req.update(result.payload)` added at `comfyless/server.py:117` to propagate the cast into the caller's request dict. Verified by new test assertion in `test_machine_boundary_validator.py` (F6 fold-in test).

---

### 7. INFO — Fail-closed posture preserved; no path returns None on error

**Severity:** INFO
**Location:** `comfyless/server.py:95-145`

Every error condition returns an explicit string. The only `return None` lines are at the success-path terminus and the non-generate-tag fall-through. Neither is a fail-open path.

**Recommendation:** None.

---

### 8. MEDIUM — Audit-log content on path errors drops only `prompt`, not `negative_prompt` (pre-existing, out of step-3 scope)

**Severity:** MEDIUM
**Location:** `comfyless/server.py:302-303` (in `_handle_connection`'s path-error branch)

Pre-existing finding; NOT introduced by step 3. The path-error audit log line drops `prompt` but not `negative_prompt`. ADR-011 §3b says both must drop.

**Status:** DEFERRED to a separate slice. Step 5 of this slice will record a TECH_DEBT entry. Step 3 does not touch this code path.

---

### 9. MEDIUM — `_RUNTIME_KIND` fields not covered by parametric N1-N7 test

**Severity:** MEDIUM (regression-resistance gap)
**Location:** `test_machine_boundary_validator.py:104-117` pre-fold-in

The parametric test iterated only `SCHEMA_KIND` and missed `_RUNTIME_KIND`. Today `_RUNTIME_KIND` has no numeric fields (only str / bool), so the gap is preventive — but a future addition (e.g. `gpu_id: int`) would not be auto-covered.

**Status:** ADDRESSED in step-3 fold-in — parametric loop now iterates `{**pv.SCHEMA_KIND, **pv._RUNTIME_KIND}.items()`.

---

### 10. INFO — Prior-review baseline check

**Severity:** INFO
**Location:** `docs/security/review-comfyless-server-2026-04-23.md`

Step 3 does not re-open any prior finding. It actively closes finding 9 of the 2026-04-23 review ("bool subtype in int"). No previously-resolved control was undone.

**Recommendation:** Note in slice commit body that finding 9 of the 2026-04-23 review is closed by this slice.

---

## Notes on broader slice (step 4 and step 5)

**For step 4 (iterate):**

- The unknown-key pass-through (step-1 finding 8 item 4) still applies. Step 4 should confirm iterate's per-LoRA helper does not consume unknown keys past the validator without explicit allowlisting.
- The N18 cross-site grid will need a third row for iterate's per-LoRA validator. The grid as written iterates fixtures with full request payloads; iterate's per-entry helper takes a single LoRA dict — the grid will need a LoRA-only fixture set, not the full-request set.
- N19's AST grep needs extension to iterate's lora-validation function name.

**For step 5 (TECH_DEBT closure):**

- Finding 6 above (validated payload discarded) ADDRESSED in step-3 code; no TECH_DEBT entry needed.
- Finding 8 above (negative_prompt not redacted from path-error audit line) is a separate TECH_DEBT entry; not closed by this slice.
- Finding 9 of the 2026-04-23 review (bool-on-int) is now resolved.
- The TECH_DEBT entry "loras[i]['weight'] not type-checked in `_validate_request`" (H-3, 2026-04-23, "pending closure by 2026-05-04") is now resolved by `validate_lora_entry`. Flip to `Resolved: 2026-05-15 — closed by ADR-012 validator slice`.

**For ADR-011 slice 1 (MCP `generate` tool):**

- The MCP server should import `validate_machine_request` from `comfyless.params_validation` from its first commit (per ADR-012 §1). The N18 grid will need a fourth row when slice 1 lands.
- Step 3's fold-in (`req.update(result.payload)`) sets the precedent that downstream callers see the cast-applied payload. The MCP path should follow the same pattern.

---

## Files referenced

- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/server.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/params_validation.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/params_schema.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/test_machine_boundary_validator.py`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/decisions/ADR-012-machine-boundary-validator.md`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/vision/slice-machine-boundary-validator.md`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/security/review-comfyless-server-2026-04-23.md`
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/docs/security/review-validator-slice-step1-2026-05-15.md`
