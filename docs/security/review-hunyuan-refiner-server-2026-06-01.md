# Security Review — Hunyuan-Image refiner chain + IPC daemon (Step 4)

AI-Disclosure: Claude (Opus 4.7) authored; Grant reviewed.

**Date:** 2026-06-01
**Slice:** Hunyuan-Image 2.1 base+refiner chaining, Step 4 (Vision `docs/vision/slice-hunyuan-image-2-1-refiner.md`; ADR `docs/decisions/ADR-016-hunyuan-image-base-refiner-chain.md` §(i))
**Branch / Range:** `hunyuan-support` @ `e948325..HEAD`
**Reviewer model:** `security-auditor` (Opus 4.7) invoked with `model: "opus"` per the broken-frontmatter-pin workaround (`feedback_agent_model_pin_broken`).
**Triggered by:**
1. Project CLAUDE.md "Review bar" — any change to `comfyless/server.py` requires `security-auditor` + saved review artifact.
2. Long-standing §12 IPC review debt per CLAUDE.md "Debt: No ADR or security review exists for `comfyless/server.py` (IPC) … when either surface is next modified, write the missing review before touching the code." This Step-4 modification is the trigger. With the CRITICAL and HIGH findings remediated, the IPC review trail is current.

---

## Summary

The Step-4 diff extends the comfyless Unix-socket IPC daemon with three new client-controlled wire fields (`refiner_path`, `refiner_steps`, `refiner_cfg`) for chained Hunyuan-Image base+refiner dispatch, plus two helpers (`_maybe_load_refiner`, `_evict_chain`). Wire-type validation for `refiner_path` is correctly inherited from `SCHEMA_KIND` via the canonical validator added in Step 1, and the eviction-order discipline (`refiner` first, then `pipeline`) is sound.

**Initial diff introduced one CRITICAL absence finding:** `refiner_path` bypassed both filesystem guardrails (`_check_paths` `--model-base` containment + `_PATH_FIELDS` null-byte rejection) that ADR-001 §3 promises for every model-path-shaped wire field. The bypass was structurally identical to the 2026-04-23 review's finding 8 (which the diff did not address for the new field). **Both CRITICAL and HIGH findings have been remediated in this same commit** by adding `refiner_path` to `_PATH_FIELDS` and to the `_check_paths` validation tuple; tests at `test_hunyuan.py` Step-4 closure block lock the runtime behavior.

**Threat model:** solo single-user desktop; attacker is a same-UID misbehaving or compromised process. Posture: ADR-001 controls intact for the pre-existing fields after remediation; the new `refiner_path` field now participates in the allowlist-bypass closure that the 2026-04-23 hardening landed.

**Overall posture:** acceptable for the same-uid threat model after remediation. CRITICAL + HIGH closed in this commit; MEDIUM (eviction-storm DoS) and LOW items deferred to TECH_DEBT.

This review also serves as the broader §12 IPC review for `comfyless/server.py` that the project CLAUDE.md tracked as debt. See "§12 IPC surface review" section below for the surface walk.

## Coverage

Reviewed:
- `comfyless/server.py` — full file, with focus on `_validate_request`, `_check_paths`, `_maybe_load_refiner`, `_evict_chain`, `_handle_generate`, cache_key composition, error-rollback paths.
- `comfyless/params_validation.py` — confirmed `SCHEMA_KIND["refiner_path"] == _KIND_STR` (Step 1 addition); canonical validator semantics.
- `comfyless/hunyuan_chain.py` — refiner path resolution; class-name lock; `allow_hf_download=False` enforced at the caller site and module default.
- `comfyless/generate.py` — refiner gate, `_cached_pipeline` reuse seam, call thread-through.
- `test_hunyuan.py` — Step-4 invariants block.
- `docs/decisions/ADR-001-daemon-socket-security.md`
- `docs/decisions/ADR-016-hunyuan-image-base-refiner-chain.md` (§(i))
- `docs/security/review-comfyless-server-2026-04-23.md` — prior baseline; finding 8 is the precedent for the initial CRITICAL.

Not reviewed (out of scope per slice boundary):
- `comfyless/cascade.py`, `comfyless/mcp_server.py` deep dive — not touched by Step 4. Test_hunyuan.py Step-3 block re-affirms MCP exposure is deliberately omitted (Vision Inv 12).

## Findings

### [CRITICAL] `refiner_path` bypassed `--model-base` containment in `_check_paths` *(REMEDIATED IN THIS COMMIT)*

**Location:** `comfyless/server.py` — `_check_paths` field tuple.

**Risk (pre-remediation):** A same-UID client could submit a `generate` request with `refiner_path` pointing at any directory the daemon UID can read — including operator home dirs outside `--model-base`, arbitrary system paths, or another user's writable area on a shared host. The path reached `detect_pipeline_class(refiner_path)` (opens `<refiner_path>/model_index.json`), then `HunyuanImageRefinerPipeline.from_pretrained(refiner_path, ...)` (deserializes pickle weight files). The pre-2026-04-23 hardening (review finding 8) added exactly this check for `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, and `loras[i].path` to enforce ADR-001 §3 ("Before loading any model or LoRA, the server canonicalizes the supplied path with `os.path.realpath()` and asserts it begins with one of the configured base directories"). The initial Step-4 diff threaded `refiner_path` into `_maybe_load_refiner` and the load path but added no corresponding `_check_paths` entry. On a single-user desktop the practical blast radius is bounded; on shared infrastructure this would have been immediately exploitable.

**Remediation (applied in this commit):** Added `refiner_path` to the `_check_paths` validation loop. The `_within` realpath+containment check now enforces ADR-001 §3 for the refiner identically to every other model-path-shaped field.

**Test coverage (added in this commit):**
- `_check_paths accepts refiner_path INSIDE --model-base`
- `_check_paths REJECTS refiner_path OUTSIDE --model-base (ADR-001 §3)`
- `_check_paths REJECTS relative refiner_path (must be absolute)`
- Source-level structural lock on `refiner_path` membership in the field tuple

### [HIGH] `refiner_path` missing from `_PATH_FIELDS` null-byte rejection *(REMEDIATED IN THIS COMMIT)*

**Location:** `comfyless/server.py` — `_PATH_FIELDS` frozenset.

**Risk (pre-remediation):** A request with `refiner_path="/m/refiner\x00..."` reached `_maybe_load_refiner` → `resolve_hf_path` → eventually `os.path.realpath` or `open()`, which raises `ValueError: embedded null byte`. The Step-4 diff catches this as `Exception` and returns `RefinerLoadError`, so the daemon survives — the previous bug class (exception escaping `_check_paths` and killing the accept loop) does not recur. However, the architectural promise of `_PATH_FIELDS` is "every path-shaped field is null-byte-defended at the validation boundary so no downstream code has to think about it." The new field violated that promise and silently relied on an exception handler one layer deeper. A future refactor that moved `_maybe_load_refiner` out of the `try/except Exception` block would reintroduce the accept-loop-kill class for the new field.

**Remediation (applied in this commit):** Added `refiner_path` to `_PATH_FIELDS` so `_validate_request`'s null-byte rejection loop covers it identically to every other path-shaped field.

**Test coverage (added in this commit):**
- `cs._PATH_FIELDS contains 'refiner_path' at runtime`
- `_validate_request REJECTS refiner_path containing NUL byte`

### [MEDIUM] Cache-eviction-storm DoS via toggle of `refiner_path`

**Location:** `comfyless/server.py` — cache_key composition + eviction trigger.

**Risk:** The cache_key tuple now includes `(req.get("refiner_path") or "").strip()` as a trailing entry. A same-UID client can ping-pong requests alternating `refiner_path=""` and `refiner_path="/valid/path"` to force the daemon into a tight eviction loop: every other request invalidates the cache_key, evicts both base + refiner (~80 GB VRAM release), and triggers a full reload (~80 GB allocate + 30-90 s wall time on the design hardware). The same client already had the cheap "send `{"type":"unload"}`" DoS primitive against this daemon (ADR-001 deferred SO_PEERCRED check, finding 3 in 2026-04-23 review), so this is an amplification, not a new attack class. Acknowledged as an operator-facing tradeoff in ADR-016 §(i) ("switching `--refiner` mode incurs a ~80 GB reload, documented operator-facing"). Naming it as a discrete finding because the previous unauthenticated-unload deferral was scoped to "shutdown nuisance"; this one ties up the warm cache without explicit shutdown.

**Remediation:** No code change required for the stated threat model — the request-rate-limiting deferred item in ADR-001 (Deferred §3) would cover this. If acted on standalone: track a per-connection cooldown on cache-key flips (e.g., reject a flip if the prior flip was within N seconds; return a `Backoff` error frame). **Deferred to TECH_DEBT** under the existing ADR-001 Deferred §3 rate-limiting entry.

### [LOW] Inconsistent default semantics: `refiner_path` empty-vs-missing collapse via `or` fallback

**Location:** `comfyless/server.py` — cache_key composition vs `generate()` call thread-through.

**Risk:** The cache_key uses `(req.get("refiner_path") or "").strip()` while the `generate()` call uses `req.get("refiner_path", "") or ""`. These differ in one edge case: an explicit `refiner_path: null` in the request (after passing `SCHEMA_KIND` — which does NOT accept None for `_KIND_STR`) would not reach here, but the inconsistency is a smell. Both branches happen to converge on empty string for any sane wire input. Not exploitable.

**Remediation:** Optional. **Deferred to TECH_DEBT.**

### [LOW] `_maybe_load_refiner` invokes `resolve_hf_path` on a client-controlled string with implicit local-only-by-default *(APPROVED)*

**Location:** `comfyless/server.py` (call site) and `comfyless/hunyuan_chain.py` (callee).

**Observation:** The server passes `allow_hf_download=False` explicitly at the call site, and `load_refiner_pipeline` defaults that argument to `False`. Both layers fail closed — if a future refactor inverts the default, the call site still pins it. This is correct defense-in-depth. Recording the structural soundness for the audit trail.

### [LOW] Refiner-load failure rollback path repeated verbatim in two sites

**Location:** `comfyless/server.py` — initial load and LoRA-removal-failure reload.

**Risk:** Two near-identical rollback blocks. A future security fix to one site could miss the other. The blocks are short and behaviorally identical today, so this is a maintenance hazard not a security flaw.

**Remediation:** **Deferred to TECH_DEBT** under the existing Step-2 MINOR-1 forward-watch (`_load_chain` extraction).

### [INFO] Audit log on `_check_paths` failure correctly excludes `prompt` but logs `refiner_path` *(APPROVED)*

**Location:** `comfyless/server.py` path-error audit log.

**Observation:** When `_check_paths` rejects a request, the audit log line drops `prompt` (correct — could contain sensitive operator strings) but retains all other fields including `refiner_path`. This is correct behavior — `refiner_path` is a filesystem path, not a secret, and logging it is necessary for audit. The pre-existing tech debt for `negative_prompt` (also logged) is unchanged.

### [INFO] §12 IPC review debt closure *(CLOSED)*

The 2026-04-23 review (`docs/security/review-comfyless-server-2026-04-23.md`) was the prior dedicated §12 IPC review. It identified 13 findings; the architectural debt CLAUDE.md flagged ("No ADR or security review exists for `comfyless/server.py` (IPC)") referred to the absence of an ADR-for-the-implementation-decisions distinct from ADR-001 (which describes the design but predates the implementation). Across the 2026-04-24 timeout/BrokenPipe hardening, the Step-1/2/3 reviews under the same surface, and now this Step-4 review (which includes the §12 surface walk below), the IPC review trail is complete. The CLAUDE.md debt entry is "next modification triggers it"; this Step-4 review is that trigger. With the CRITICAL and HIGH findings remediated, the IPC surface review trail is current.

## §12 IPC surface review (broader, beyond Step-4)

The prompt asked for the broader IPC review beyond Step-4-specific changes. Walking the surfaces explicitly:

- **Socket placement (ADR-001 §1, §3):** XDG_RUNTIME_DIR path or `/tmp/comfyless-$UID/` at 0700 with uid-ownership assertion. 2026-04-23 finding 2 is closed: code enforces 0o700 on the existing directory and asserts uid ownership. The three `# nosemgrep:` annotations added in this commit are correct — semgrep's generic 0o644 rule does not apply to socket parent directories where world-read would weaken the per-UID isolation that IS the control. The socket file itself is chmod 0600.
- **Null-byte rejection (`_PATH_FIELDS`):** Present for the original 5 path fields + savepath. **`refiner_path` added in this commit** (HIGH closure above).
- **Path containment (`_check_paths` + `_within`):** All path fields validated against `--model-base` via realpath. **`refiner_path` added in this commit** (CRITICAL closure above). The relative-vs-absolute check defends against HF-repo-id shaped inputs reaching `resolve_hf_path` with local-cache probing (2026-04-23 finding 8); the same check now covers `refiner_path`.
- **Wire-format guardrails:** `_MAX_FRAME_BYTES = 1<<20` and `_RECV_TIMEOUT_SEC = 5.0` close 2026-04-23 finding 1. The client-side `_CLIENT_RECV_TIMEOUT_SEC = 600.0` is the deliberate asymmetry per ADR-001's 2026-04-24 Changelog.
- **Broken-peer resilience (`_send_safe`):** Catches `BrokenPipeError`, `ConnectionResetError` from every server-to-client send. Audit log fires before `_send_safe` so the record survives peer disconnect.
- **Adapter-name sanitization:** `sanitize_adapter_name` strips to `[a-zA-Z0-9_-]+` per ADR-001 §4. Applied before `load_lora_with_key_fix`.
- **Schema validation order:** Wire-type rule check (`validate_machine_request`) → semantic type-tag check → required-field check → null-byte check → path-containment check → ML stack. `refiner_path` is now correctly type-checked at step 1 (Step-1 addition; verified at runtime by Step-4 test), null-byte-checked at step 4 (this commit), and path-contained at step 5 (this commit).
- **Error response shape:** `RefinerLoadError` vs `LoadError` distinguishable on the wire — supports operator triage and future MCP error-frame mapping. Traceback exposure on `InferenceError` remains as previously deferred (2026-04-23 finding 4).
- **Eviction discipline:** `_evict_chain` drops refiner first, then pipeline, then `torch.cuda.empty_cache()`, then `server_state.clear()`. Order is correct per CUDA-memory-release-timing reasoning in the Step-2 code-reviewer MINOR-1 forward-watch. Test coverage locks both source order and runtime deletion order.
- **Failure-rollback discipline:** When refiner load fails after base load succeeds, base is deleted and `torch.cuda.empty_cache()` runs before returning error. This avoids the half-cached state ADR-016 §(i) warns about. Correct.

Pre-existing deferrals (still acceptable for solo-desktop threat model, still tracked in TECH_DEBT):
- SO_PEERCRED on `unload` and `--unload` (2026-04-23 finding 3)
- Per-request inference timeout (ADR-001 Deferred §2)
- Request-rate limiting / VRAM swap bounds (ADR-001 Deferred §3, related to MEDIUM above)
- Traceback redaction on `InferenceError` before LLM-agent surface lands (2026-04-23 finding 4)

## Conclusion

CRITICAL and HIGH closed in this commit (two-line fix + four locking tests). MEDIUM and LOW items deferred to TECH_DEBT (eviction-storm rate-limit, normalization consistency, duplicate-rollback-block extraction — all of which are forward-watches with concrete trigger conditions).

The IPC surface review trail is current with this commit; the long-standing §12 IPC review debt (per CLAUDE.md "Review bar" section) is **closed**.

**Findings summary:** 1 CRITICAL (remediated), 1 HIGH (remediated), 1 MEDIUM (deferred), 3 LOW (1 approved, 2 deferred), 2 INFO (1 approved, 1 closure note).
