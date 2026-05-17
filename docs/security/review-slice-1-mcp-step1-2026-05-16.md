# Security Audit — Slice 1 Step 1 (ADR-011 MCP Server Skeleton)

**Date:** 2026-05-16
**Reviewer:** `security-auditor` subagent (Opus, model pinned at invocation per project CLAUDE.md review-bar)
**Scope:** Slice 1 step 1 of ADR-011 — comfyless MCP server skeleton + spawn-time startup hardening + audit-line discipline + traceback-strip helper + single-tool advertisement. Files: `comfyless/mcp_server.py` (NEW), `test_mcp_server.py` (NEW), `comfyless/generate.py` (one surgical docstring on `_run_json_mode`).
**AI-Disclosure:** Claude (Opus 4.7, 1M context) authored the audit; Grant reviewed.

---

## Verdict

**CLEAN with one LOW finding and four INFO observations. No blockers.**

The step-1 surface meets every invariant the slice-1 Vision asks of it. Startup-path-escape closures are correct (verified at `os.path.realpath` + `os.path.isdir` + `_within(p, --model-base)` for `--default-model`; N18 test exercises the realpath-escaping symlink case). Audit-line discipline is correct (stderr-only, prompt + negative_prompt dropped, write failures counted but non-blocking). Traceback-strip helper returns category-shaped messages with no path / `.py:` / `Traceback` content; full traceback redirected to stderr. Schema lockdown via `additionalProperties: False` structurally blocks `max_iterations`. No `import argparse` and no CLI-dispatch callees. The deliberate `validate_input=False` framework trade-off is correct for invariant 5 (audit-every-invocation), with the cost recorded for step 2 (handler must take ownership of validation via `validate_machine_request`).

---

## Coverage

- `comfyless/mcp_server.py:1-471` (full module)
- `test_mcp_server.py:1-477` (full suite, 50/50 passing)
- `comfyless/generate.py:1080-1090` (legacy-marker docstring; behavior unchanged)
- `comfyless/server.py:158-183` (verbatim-imported `_within` helper)
- `.venv/lib/python3.12/site-packages/mcp/server/lowlevel/server.py:490-589` (framework `call_tool` decorator: `validate_input` plumbing + outer `except Exception → _make_error_result(str(e))`)
- `.venv/lib/python3.12/site-packages/mcp/server/stdio.py` (stdio_server signature)
- ADR-011 (entire ADR including all five Changelog amendments)
- Vision slice-1-mcp-generate.md (all 15 invariants and N1-N33 negative cases)
- `docs/security/review-slice-0-mcp-dep-2026-04-30.md` (prior reviewer format reference)

---

## Findings

### F1 [LOW] TOCTOU window between `realpath` and `_within` in `_validate_startup_args`

**Location:** `comfyless/mcp_server.py:264-290` (`_validate_startup_args`)

**Risk:** `os.path.realpath(default_model)` resolves the symlink at call time and yields the post-resolution target. `_within(resolved_default, resolved_base)` then calls `realpath` AGAIN internally (`comfyless/server.py:160-161`). If an attacker with write access to a directory inside `--model-base` swaps the symlink between the two calls, the second `realpath` could see a different target than the first. The window is impractical to exploit under a same-uid stdio MCP server (the attacker already has the daemon's privileges); global §5 places auth-beyond-same-uid out of scope, and ADR-011 §7 explicitly defers it. Naming the assumption: same-uid threat model means this is not exploitable today.

**Remediation:** No change required for step 1. If the same-uid assumption ever weakens (multi-user deployment, HTTP transport per a future ADR), cache `os.path.realpath(default_model)` once and pass the cached value to a `_within` form that does NOT re-resolve. Tracked in `TECH_DEBT.md` under the Security section.

### F2 [INFO] `_audit_write_failures` module-level mutable list — concurrency comment recommended

**Location:** `comfyless/mcp_server.py:72`, appended at `:202` and `:226`

`list.append` is atomic in CPython under the GIL, and MCP-over-stdio is single-connection per server process under a single asyncio event loop. Concurrent appends from coroutines do not race today. The Vision allows for rate-style alerting in future slices; if a future slice reads the list (admin tool, watchdog), the read-side will need its own coordination. Step 1's stub neither reads nor exposes the counter; no hazard.

Recommendation for step 2+: document the single-event-loop assumption next to the declaration so a future maintainer does not introduce a thread-pool-executor write site.

### F3 [INFO] `validate_input=False` trade-off is bounded; step 2 must close

**Location:** `comfyless/mcp_server.py:371-373` (`@app.call_tool(validate_input=False)`)

The framework's default `validate_input=True` runs `jsonschema.validate(...)` before the handler and short-circuits on failure, bypassing the handler's audit emission. Slice-1 Vision invariant 5 requires every invocation to emit one audit line, so taking ownership of validation in the handler is correct. In step 1, the stub raises `NotImplementedError` unconditionally before any validation happens; a schema-invalid request reaches the stub only to produce an audit line plus a fixed safe message — no internal state exposed.

Step 2 must wire `validate_machine_request` from ADR-012 as the FIRST handler action after audit-line emission, and route any validator-internal exception through `_sanitize_error` so invariant 13 (traceback-strip) is not bypassed.

### F4 [INFO] Framework's outer `except Exception → str(e)` is NOT the traceback-strip; handler must close

**Location:** `comfyless/mcp_server.py:209-227` (`_sanitize_error`); framework path `.venv/.../mcp/server/lowlevel/server.py:583-584`

The framework catches `except Exception as e` and returns `_make_error_result(str(e))`. `str(NotImplementedError("...msg..."))` returns just the message — no traceback. Step 1's hand-constructed stub message ("generate handler is scaffolded in slice 1 step 1; the actual wiring lands in step 2. See docs/vision/slice-1-mcp-generate.md") is safe by inspection.

Step 2 must NOT reuse this "raise + let framework convert" pattern when the handler body grows. Every internal exception in step 2's `_call_tool_impl` must be caught and routed through `_sanitize_error` BEFORE it reaches the framework — otherwise invariant 13 is bypassed.

### F5 [INFO] Shared-redaction-tuple import discipline for step 2

**Location:** `comfyless/mcp_server.py:57-66` (`_MCP_PATH_TYPED_FIELDS`, `_AUDIT_DROPPED_FIELDS`)

Invariant 12 declares the redaction map MUST be shared between the audit-line writer and the PNG-redaction code "in code so they cannot drift." Step 1 lands the declarations as tuple + frozenset at module scope. The shape is suitable for step 2 import. Cascade-side fields (`stage_c`, `stage_b`, `stage_a`, `scaffolding_repo`) are correctly absent — the comment at lines 55-56 names this as a step-3 addition.

Step 2's PNG-redaction code path must `from comfyless.mcp_server import _MCP_PATH_TYPED_FIELDS` rather than declare a parallel local tuple. Step 2 reviewer to confirm.

---

## Absence check (step-1-scoped)

Items NOT in the step-1 diff that are correctly NOT in the step-1 diff per the Vision's "out of scope" list:

- Per-request `_within` validation of `model` / `loras[].path` / `output_path` (step 2)
- `validate_machine_request` invocation (step 2; ADR-012 prerequisite)
- `allow_hf_download=False` hard-coding at `resolve_hf_path` / `_load_pipeline` call sites (step 2)
- PNG-chunk redaction (step 2 / invariant 12)
- Sidecar suppression for MCP-driven calls (step 2 / invariant 11)
- `iterate` cap enforcement (the `--mcp-max-iterations` flag is parsed but unused — Vision explicitly says "declared in slice 1 for stable spawn contract; iterate handler lands in a later slice")
- Cascade `stage_*` / `scaffolding_repo` fields in `_MCP_PATH_TYPED_FIELDS` (step 3)

None of these absences is a step-1 finding.

---

## Scope creep

None observed. The diff is bounded to `comfyless/mcp_server.py` (new), `test_mcp_server.py` (new), and `comfyless/generate.py` (one docstring on `_run_json_mode`). No edits to `comfyless/server.py`, `params_schema.py`, `cascade.py`, or any node files.

---

## Step-2 reviewer carry-forward

F3, F4, F5 are all forward-looking. The step-2 reviewer (this same auditor, model pinned to Opus at invocation) should verify:

1. `validate_machine_request` is the first action after audit-line emission in `_call_tool_impl`'s `generate` branch.
2. Every internal exception in the step-2 handler body is caught and routed through `_sanitize_error` before reaching the framework's outer `except Exception`.
3. PNG-redaction code in `_save_with_metadata`'s `mcp_caller=True` branch imports `_MCP_PATH_TYPED_FIELDS` from `comfyless.mcp_server` (single source of truth).

F1 (LOW TOCTOU) is tracked in `TECH_DEBT.md` and does not need step-2 attention unless the same-uid threat-model assumption changes.

---

**Approve labels:** `APPROVED` — no `BOUNDARY VIOLATION`, no `PROMISE DRIFT`, no `SECURITY REGRESSION`, no `SCOPE CREEP`. F1 LOW + F2-F5 INFO are forward-looking flags.
