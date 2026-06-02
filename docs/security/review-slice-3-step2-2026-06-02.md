# Security Review — ADR-015 Slice 3 Step 2: `_handle_generate` catalog-name migration

**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored this security review; Grant reviews and owns the Red Zone sign-off.
**Date:** 2026-06-02
**Scope:** the Step-2 handler wiring in `comfyless/mcp_server.py` (`_handle_generate`, `_resolved_params_as_names`, `_reference_error`, `_discard_notice`, `_GENERATE_REMOVED_FIELDS` guard, the canonical-validator-minus-loras change, the load-boundary `_check_paths` re-net) plus its tests in `test_mcp_server.py`. This is the commit that ships the load-bearing uniform-error contract (HIGH-1) closing the HF-cache enumeration oracle (TECH_DEBT Security 2026-05-17). Step-1 (`resolve_reference`) was reviewed CLEAN separately (`docs/security/review-slice-3-step1-2026-06-02.md`); this review verifies the six carry-forward obligations landed in the handler.

## Summary

The threat model is the same-uid stdio trust boundary: a possibly-prompt-injected LLM agent supplies reference values (`model`, `transformer`, `loras[].name`) and a `savepath`; the operator supplies `--model-base`, `--output-dir`, `--default-model`, and `--catalog` at spawn. The security invariant this slice must preserve is twofold: (1) every reference-resolution failure returns ONE byte-identical agent frame ("reference not available") with the fine cause confined to the operator stderr audit — so the agent cannot use error variation to enumerate the model-base or HF cache; and (2) no server-side `abs_path` ever crosses the MCP boundary in a response, notice, error, or audit line. The handler routes all three reference fields through Step-1's `resolve_reference` with the correct `expected_kind`, folds every `ResolveCause` into `_UNIFORM_REFERENCE_ERROR` via a single `_reference_error` constructor, retains the request-time `_within`/`_check_paths` net at the load boundary, and renders the response with catalog names via `_resolved_params_as_names`.

I traced each of the six carry-forward obligations against the code and the keystone tests (N5 byte-identity, N6 audit-cause distinctness, N9 notice-name-not-raw, N13 audit-no-abs_path), checked the removed-field guard for both rejection completeness and field-name-not-value safety, verified the default-model bypass re-runs `_within`, confirmed the canonical-validator-minus-loras change cannot smuggle an untyped `transformer` or a malformed loras entry past the resolver/manual gate, and enumerated `generate()`'s real return dict (`generate.py:940-965`) field-by-field against `_resolved_params_as_names` to look for a path-typed key the renderer fails to strip. The cascade handler is confirmed entirely unmigrated (zero `resolve_reference` calls), consistent with the slice-3b deferral — no partial-migration inconsistency. **Verdict: CLEAN on all six carry-forward obligations and the load-bearing oracle-closure property.** One MEDIUM defense-in-depth gap (the `lora_warnings` pass-through is a latent abs_path leak that is only inert because the MCP path always passes a cached pipeline) and two INFO observations are recorded below.

## Coverage

Reviewed:
- `comfyless/mcp_server.py` — `_GENERATE_REMOVED_FIELDS`; `_resolved_params_as_names` (the MCP-response renderer); `_UNIFORM_REFERENCE_ERROR`, `_reference_error`, `_discard_notice`; `_call_tool_impl` dispatch + the `_MCPHandlerError → raise ValueError(e.safe_message)` agent-frame path; `_handle_generate` in full; `_handle_generate_cascade` (to confirm it is unmigrated); `_resolve_mcp_output_path`.
- `comfyless/catalog.py` — `resolve_reference` (the Step-1 resolver this handler consumes).
- `comfyless/server.py` — `_within`, `_check_paths`.
- `comfyless/params_validation.py` — `SCHEMA_KIND` (no `transformer` key), unknown-key pass-through, loras entry validation.
- `comfyless/generate.py` — the LoRA load loop (`lora_warnings` content) and the real `generate()` return dict.
- `test_mcp_server.py` — fixtures/mocks (`_mock_generate`, `_setup_mb_and_out`, `_call`); N1-N15/N23/N25/N31 handler tests + F1 null-byte fold-in.

Not reviewed (and why):
- Cascade dispatch internals beyond confirming non-migration — slice 3b, out of scope.
- The 8 non-MCP test suites — untouched per slice plan.
- `_save_with_metadata`/`redact_metadata_for_png` on-disk PNG sink internals beyond the boundary-leak check — slice-1 surface, already reviewed; on-disk PNG under operator `--output-dir` does not cross back to the agent.

## Findings

### Obligation 1 — Uniform frame (HIGH-1): CLEAN

Every reference-resolution failure reaches the agent as one byte-identical string. All three resolve sites (`model`, `transformer`, `loras[].name`) and the load-boundary recheck raise via the single `_reference_error(cause)` constructor, which always carries `_UNIFORM_REFERENCE_ERROR` ("reference not available") as the `safe_message`. `_call_tool_impl`'s `except _MCPHandlerError` re-raises `ValueError(e.safe_message)` — the exact byte string, never the `error_class`. The keystone test N5 asserts `e1 == e2 == e3 == e4 == _UNIFORM` across UnknownName/PathMoved/KindMismatch/MalformedReference; N6 asserts the four operator-audit causes are distinct. The fold is structurally in one place. CLEAN.

### Obligation 2 — Cause to stderr only: CLEAN

The fine `ResolveCause` rides on `_MCPHandlerError.error_class`, which `_call_tool_impl` writes only to `_emit_audit_line(error_class=e.error_class)` → stderr. The agent frame is `e.safe_message`. N13 asserts `"UnknownName" in se` (stderr) while `mb not in se`. The cause never appears in the agent-facing ValueError. CLEAN.

### Obligation 3 — `abs_path` never in response/notice/error/audit: CLEAN (current flow), with one latent gap → see MEDIUM-1

- **Response:** `_resolved_params_as_names` overwrites `model` with `model_name`, drops `transformer_path`/`vae_path`/`text_encoder_path`/`text_encoder_2_path`, and rebuilds `loras` as `[{name, weight}]`. I enumerated the real `generate()` return: the only path-typed keys are `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `loras[].path` — all six handled. N11 asserts `mb not in _txt`, no `/loras/`, no `transformer_path`/`vae_path` keys.
- **Notice:** `_discard_notice` interpolates `name` only (the resolved catalog name).
- **Error:** uniform string, no path.
- **Audit:** `audit_payload = arguments` is the agent's OWN raw input (which the agent already knows), never the server-resolved abs_path. The resolved `model_abs`/`transformer_abs`/`loras_resolved[].path` are never placed on the audit line.

The one residual: `lora_warnings` was NOT stripped by `_resolved_params_as_names` and passed through verbatim; in the real `generate()` it can embed the resolved abs_path. It was inert only because the MCP path always supplies a cached pipeline. See MEDIUM-1.

### Obligation 4 — Correct `expected_kind` per field: CLEAN

`model` → `expected_kind="model"`; `transformer` → `"transformer"`; `loras[].name` → `"lora"`. N3 proves a lora name supplied in the `model` field folds to KindMismatch → uniform error, closing the wrong-kind oracle. CLEAN.

### Obligation 5 — Discard notice only on success, keyed on resolved name (INFO-2): CLEAN

`_discard_notice` is appended only inside the `if rr.ok` success arms, gated on `rr.path_was_discarded`, and always passes `rr.name` (the resolved NFC catalog name) — never `model_in`/`transformer_val`/`lora.get("name")` (the raw agent value). On any `not rr.ok` the handler raises before reaching the notice append. N9 asserts the notice contains the resolved name AND that the supplied directory text (`/agent/hallucinated/dir`) never round-trips into the response. N8 confirms the notice fires on a path-shaped value; N7 confirms no notice for a bare name. CLEAN.

### Obligation 6 — Load-boundary TOCTOU re-validation: CLEAN

After all references resolve, step 6 re-runs `_check_paths({"model": model_abs, "transformer_path": transformer_abs, "loras": loras_resolved}, cfg.model_base)` immediately before `_load_pipeline`. `_check_paths` re-`realpath`s via `_within`, catching a symlink/mount swap between resolve and load. A failure raises `_reference_error("WithinFailure")` — uniform frame, value never echoed. This is the retained slice-1 net per Vision invariant 9. CLEAN. (Note: a window still exists between this check and `open()` inside `_load_pipeline`; that residual is inherent to filesystem TOCTOU and is the same posture slice-1 shipped — not a regression, and the `--model-base` containment is operator-controlled, so the worst case is loading an operator-planted file, not an agent escape.)

### Removed-field guard (OQ-A): CLEAN

`_GENERATE_REMOVED_FIELDS` = (`transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`). The guard rejects any present removed field with a named ValidationError before any resolution. Rejecting (vs silently ignoring) is correct: silent acceptance of a raw `vae_path` would reintroduce the caller-supplied-path input vector ADR-015 removes — and these fields are passed straight to `_load_pipeline` as `vae_path=""` etc. (hard-coded ""), so a smuggled value could not even take effect, but rejecting makes that explicit and audit-visible. **Naming the field is safe:** field names are public JSON-schema knowledge (the schema is advertised in `list_tools`), so the message leaks nothing about the filesystem — unlike a reference VALUE, which would. The guard error names only `_removed` (the field key), never any value. The removed-field-guard test covers all four. CLEAN.

### Default-model bypass (OQ-D): CLEAN

When `model` is omitted, the handler uses `cfg.default_model` — an operator-trusted path validated at spawn (both `isdir` and `_within`). The handler re-runs `_within(cfg.default_model, cfg.model_base)` at request time to catch a post-startup symlink swap, raising `DefaultModelEscape` on failure. The default path then ALSO flows through the load-boundary `_check_paths` net at step 6 (it is the `model` key there). An agent cannot abuse the omitted-model path to skip containment: the only way to reach this branch is to omit `model` entirely, and the resulting abs_path is operator-chosen and double-`_within`-checked. The agent cannot influence which path is used. CLEAN.

### Canonical-validator-minus-loras change: CLEAN

`validate_machine_request(_args_no_loras)` strips `loras` before canonical validation because the shared validator hard-requires the slice-1 `loras[].path` shape, which would reject the MCP `{name, weight}` shape. The handler then validates loras manually: list-check, per-entry dict-check, `name`/`weight` presence, weight is `int`/`float` and not `bool` (matching the canonical `_KIND_FLOAT` bool-rejection), `float(w)` cast. The `name` is passed to `resolve_reference`, which type-checks it (non-str → MalformedReference → uniform). No type-confusion or injection on the loras path: a non-dict entry, missing key, or bad weight type is caught before resolution; a malformed `name` folds to the uniform error (F1 test).

One subtlety I verified safe: the canonical `SCHEMA_KIND` has NO `transformer` key (only `transformer_path`), and `validate_machine_request` passes unknown keys through unchanged. So a non-str `transformer` is NOT type-checked by the validator. But it then flows to `resolve_reference`, which returns MalformedReference for any non-str. The framework runs `validate_input=False`, so the JSON schema's `transformer: string` typing is not enforced upstream either — the resolver is the sole gate, and it is fail-closed. No type confusion reaches the load path. CLEAN.

### Cascade non-migration: CLEAN (no partial state)

`_handle_generate_cascade` contains zero `resolve_reference`/`cfg.catalog`/`_reference_error`/`_discard_notice` references (grep-confirmed). It retains the slice-1 raw-path + HF-resolution + `_within` contract end to end. Step 2 did not partially migrate it. Consistent with the slice-3b deferral. CLEAN.

### MEDIUM-1 — `lora_warnings` is an un-stripped abs_path pass-through; inert only by the cached-pipeline accident

**Location:** `comfyless/mcp_server.py` (`_resolved_params_as_names` docstring promising `lora_warnings` "passes through verbatim", and the renderer not stripping it); `comfyless/generate.py` (the warning strings embed the resolved `lora_path` abs_path; added to `metadata`); consumed into the agent response.

**Risk:** `lora_warnings` entries are `f"LoRA skipped (0 modules applied): {lora_path}"` and `f"LoRA load failed: {lora_path}: {e}"`, where `lora_path` is the server-side abs_path under `--model-base` (and `{e}` may carry further internal paths). `_resolved_params_as_names` rewrites only the path-typed weight fields and lets every other key — explicitly including `lora_warnings` — pass through into the agent-facing `resolved_params`. That would leak a server abs_path to the agent, breaking invariant 5/10 and opening a content-keyed leak (the agent learns the exact on-disk path of any LoRA that fails to apply). **It does not fire today only because** the MCP handler always passes `_cached_pipeline=cached` to `generate()`, and the LoRA load loop that populates `lora_warnings` is guarded by `if _cached_pipeline is None:` — so in the current MCP flow `lora_warnings` is always empty. The safety is therefore an emergent property of an unrelated caching decision, not an enforced boundary contract. Any future change that (a) loads LoRAs in the cached path, (b) sets `_cached_pipeline=None` for an MCP generate, or (c) adds any other path-bearing warning string to `metadata`, silently reopens the leak.

**Assumption named:** this rests on `_cached_pipeline` remaining non-None for every MCP `generate` call. If that invariant is ever relaxed, this becomes HIGH.

**Remediation (smallest targeted):** in `_resolved_params_as_names`, drop the field explicitly: `out.pop("lora_warnings", None)` — or, if the warnings are wanted for agent UX, replace each entry's path with the corresponding catalog name before passing through. Add a negative test asserting no `--model-base` abs_path appears in `resolved_params` when a LoRA warning is present.

**RESOLVED 2026-06-02:** folded before commit — `_resolved_params_as_names` now `out.pop("lora_warnings", None)` (named comment cites this finding), and `test_mcp_server.py` adds the MEDIUM-1 regression test (mock generate returns a warning carrying a `--model-base` abs path; assert it is absent from `resolved_params` and the response). The no-leak guarantee is now an enforced contract, not an emergent property.

### INFO-1 — PNG on-disk metadata uses abs_path basename, not catalog name, for manifest entries whose basename ≠ catalog name

`redact_metadata_for_png` basenames `model`/`loras[].path` to `os.path.basename(abs_path)`. For a manifest entry whose catalog NAME differs from its abs_path basename, the on-disk PNG carries the abs_path basename rather than the catalog name — a minor inconsistency with the agent-facing `resolved_params` (which uses the catalog name). No agent-facing leak: the PNG sits under the operator's `--output-dir` and does not cross back into the response. Recording for consistency only; no action required.

### INFO-2 — No length cap on agent reference values

As noted in the Step-1 review, `resolve_reference` has no length cap on `raw_ref`; the handler adds none. Per-call cost is O(n) and the MCP transport bounds the JSON-RPC frame, so for a same-uid stdio peer this is not a finding. Revisit if HTTP transport lands (the existing CLAUDE.md Red-Zone-on-scope-change note already flags HTTP).

## Verdict

**CLEAN on all six Step-1 carry-forward obligations and the load-bearing uniform-error / oracle-closure property (HIGH-1).** The removed-field guard, default-model bypass, validator-minus-loras change, and cascade non-migration are all sound. One MEDIUM defense-in-depth gap (MEDIUM-1: `lora_warnings` un-stripped abs_path pass-through) was folded before commit with a regression test; two INFO observations need no action.
