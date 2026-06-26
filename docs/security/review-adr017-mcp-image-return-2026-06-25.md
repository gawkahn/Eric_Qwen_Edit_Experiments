AI-Disclosure: Claude (Opus 4.8, 1M context) authored; Grant reviewed.

# Security review — ADR-017 step 1: comfyless MCP optional base64 image return

> Saved by the parent session on behalf of the `security-auditor` subagent
> (Opus 4.8), which had no Write tool in its sandbox. Content is the auditor's
> verbatim returned review. Disposition of findings appended at the end.

## Summary

This change adds an optional, gated, size-bounded base64 PNG return to the comfyless MCP `generate` and `generate_cascade` tools. Three new scalar request params (`return_image` bool/default false, `max_return_px` int/default 768, `max_return_bytes` int/server ceiling 1 MiB) are registered in the canonical ADR-012 validator. When `return_image=true`, a new `_encode_return_image` re-encodes the already-written, already-§3e-redacted on-disk PNG into a pixel- and byte-bounded base64 PNG (re-encoded with no `pnginfo`, so PNG text chunks do not survive) and attaches `image_b64`/`image_mime` as plain JSON fields on the response frame. The threat model is a new data-egress surface on the Red-Zone-adjacent MCP boundary: image bytes (the caller's own generated content) now leave the server to an LLM-driven caller. The load-bearing controls are (a) the bytes appear only in the response frame, never in the operator audit line / stderr; (b) the returned copy is hard-capped in size; (c) the on-disk artifact and its §3e redaction are untouched; (d) the return path is fail-soft and never fails a successful generation.

I traced `audit_payload` (= raw request `arguments`) through `_emit_audit_line` and confirmed the response dict carrying `image_b64` is a separate object that is never passed to the audit writer, on any success or error path; the new request scalars are logged but that is intended and harmless. I traced the encode helper's pixel bound, byte bound, iterative-downscale loop, clamp logic, and the pathological raise-then-fail-soft path; confirmed the 1 MiB ceiling is a true hard cap the agent cannot raise. I confirmed `output_path` is server-resolved and containment-checked upstream (`_resolve_mcp_output_path`) and is not caller-supplied to the reader, the source PNG is the server's freshly-written redacted file (`redact_metadata_for_png` drops `output_path`/`savepath` and basenames path fields), and the re-encode strips text chunks (test N3 anchors this empirically). I checked the validator widening: registering the three fields only makes the daemon/iterate surfaces *stricter* (type-rejects previously-passthrough values) and does not enable image egress there because only the MCP handlers call `_maybe_attach_return_image`. Cascade parity is exact (identical shared helper, server-resolved contained path, fail-soft). Overall posture is sound; findings are LOW/INFO hardening only.

## Coverage

Reviewed:
- `scratchpad/adr017-step1.diff` — full diff (mcp_server.py, params_validation.py, both test files)
- `comfyless/mcp_server.py:490-534` — `_emit_audit_line` (audit writer; redaction set)
- `comfyless/mcp_server.py:826-938` — new `_encode_return_image` + `_maybe_attach_return_image` (full bodies, in-context)
- `comfyless/mcp_server.py:941-1021` — `_call_tool_impl` (audit_payload provenance; all three emit sites; outer except chain)
- `comfyless/mcp_server.py:1024-1301` — `_handle_generate` (payload/output_path/notices provenance; attach call site)
- `comfyless/mcp_server.py:1304-1502` — `_handle_generate_cascade` (cascade parity)
- `comfyless/mcp_server.py:117-153` — `redact_metadata_for_png` (§3e on-disk source content)
- `comfyless/params_validation.py:1-287` — full canonical validator (new `_MCP_TRANSPORT_KIND`, `_ALL_FIELDS`, passthrough behavior, `_check_field`)
- `docs/decisions/ADR-017-...md`, `docs/vision/slice-mcp-image-return-owui.md` — threat model + invariants 1-8/3b

Not reviewed (and why):
- `_resolve_mcp_output_path` body and `redact_metadata_for_png`'s `_basename_or_repo_id` internals — pre-existing, unchanged by this diff; output-path containment correctness is an assumption carried from prior reviews (review-slice-3-*; review-adr-015-*). Flagged as an explicit assumption below.
- `comfyless/server.py` daemon socket and `generate.py` iterate handlers — confirmed they do not call the new attach helper; their only delta is stricter validation, not re-read line-by-line.
- `test_*.py` additions — read for invariant intent (N1-N10), not audited as production code.

## Findings

[LOW] Fail-soft uses `except BaseException` where `except Exception` is sufficient
Location: `comfyless/mcp_server.py:937` (`_maybe_attach_return_image`)
Risk: The global constitution (§0 rule 2) mandates that audit-emission failure raise a `BaseException` subclass precisely so broad handlers cannot swallow it. Today no audit emission is reachable inside this `try` (it calls only `_encode_return_image`, which does PIL/base64 work; `_emit_audit_line` runs later in a different frame in `_call_tool_impl`), so this is *not* a present violation. But the broad catch is a latent footgun: if a future edit adds any audit/emission call inside `_encode_return_image` or this `try`, the mandated audit-failure `BaseException` would be silently converted to a fail-soft INFO notice, breaking the "audit must always be on" invariant. It also absorbs `KeyboardInterrupt`/`SystemExit`/`MemoryError` during encode.
Remediation: Change `except BaseException:` to `except Exception:`. All real encode failures (PIL errors, `ValueError` from the byte-budget raise, the test's `RuntimeError`) are `Exception` subclasses, so fail-soft behavior is fully preserved while interpreter signals and any future audit `BaseException` propagate as the constitution requires.

[LOW] `max_return_px` has no upper clamp — transient full-resolution first-encode
Location: `comfyless/mcp_server.py:826-853` (`_encode_return_image`, `eff_px` computation and first `_encode`)
Risk: `eff_px` is `max_px` whenever it is a positive int, with no ceiling. If a caller sets `max_return_px` ≥ the on-disk longest edge (e.g. a large value against a 50 MP generation), the pixel branch is skipped and the *first* base64 encode runs on the full-resolution PNG before the byte loop shrinks it — a transient in-memory spike (tens to ~100+ MB) bounded only by the generation pixel cap, not by `max_return_px`. The final returned payload is still correctly capped at 1 MiB, so this is an internal resource concern, not an egress/output-size issue, and it is single-tenant, serial, and self-inflicted on a desktop tool. Severity is LOW for this deployment.
Remediation: Optionally bound the first encode by clamping the effective pixel cap — or document the spike as accepted. No change required to meet the stated invariants.
Assumption: generation pixel caps (16 MP edit / 50 MP gen) bound the on-disk image size; if a future path lets `output_path` point at an arbitrarily large pre-existing file, revisit.

[INFO] Metadata-free guarantee rests on PIL not re-emitting tEXt without a `PngInfo`, plus §3e as backstop
Location: `comfyless/mcp_server.py:850-853` (`_encode`, `image.save(buf, format="PNG")` with no `pnginfo=`)
Risk: Invariant 5 (no metadata leak via `image_b64`) depends on PIL not auto-persisting source tEXt/iTXt/zTXt chunks on save when no `PngInfo` is passed — which is correct and test-anchored (N3 asserts `not _pim.text`). Non-text ancillary data carried in `img.info` (icc_profile, dpi, exif/eXIf) *can* ride along on re-save; none of these are filesystem strings or server-state leaks. Defense-in-depth holds regardless: the source PNG is written by `generate(mcp_caller=True)` → `redact_metadata_for_png`, which drops `output_path`/`savepath` and basenames all path-typed fields, so even a metadata-carry bug could only surface already-redacted basenames (already present in `resolved_params`) and the caller's own prompt (their own input — not a cross-trust leak per the ADR threat-model note).
Remediation: None required. Optionally keep an eye on PIL version bumps that could change default chunk persistence (test N3 will catch a regression).

[INFO] Audit line now records `return_image`/`max_return_px`/`max_return_bytes`
Location: `comfyless/mcp_server.py:977` (`audit_payload = arguments`) → `_emit_audit_line`
Risk: The new request scalars flow into the audit line's `input` field. These are bounded scalars (bool/int), not image bytes, and contain no path or PII; this is expected and acceptable. The image payload itself is never in `arguments`, so it never reaches the audit line — invariant 2 holds.
Remediation: None.

## Per-question findings (as requested)

1. **Audit exclusion (invariant 2):** HOLDS. `_emit_audit_line` is only ever called with `audit_payload` (= raw `arguments`) at all three sites. The `response` dict that receives `image_b64`/`image_mime` is a distinct object built in the handlers and returned as the stdout JSON-RPC frame; it is never passed to the audit writer. `_maybe_attach_return_image` swallows its own failures and appends only a static notice carrying no path/exception text/bytes.
2. **Metadata leak (invariant 5):** HOLDS with defense-in-depth. Source = the §3e-redacted on-disk PNG. Re-encode uses no `pnginfo=`, dropping all text chunks (test N3). See INFO finding re: non-text ancillary chunks (non-sensitive).
3. **DoS / resource exhaustion:** Output is hard-capped — `eff_bytes = min(max_bytes, 1 MiB)`, agent cannot raise it (test N10); `max_return_bytes` ≤ 0 / non-int → ceiling; `max_return_bytes=1` → unmeetable → raise → fail-soft, no crash. Loop bounded by `_RETURN_BYTES_MAX_ITERS=8` and `_RETURN_PX_FLOOR=64` with guaranteed forward progress. Residual: no upper clamp on `max_return_px` (LOW finding above).
4. **Fail-soft safety (invariant 8):** Correct. Attach runs after generation fully succeeds, so it cannot mask a gen error. Cannot swallow an audit-emission `BaseException` (none reachable in its `try`). `except BaseException` broader than needed — LOW finding.
5. **Cascade parity (invariant 7):** HOLDS. Identical shared helper, server-resolved contained path, same validator. Test N7.
6. **Path traversal:** No new surface. `output_path` is server-resolved via `_resolve_mcp_output_path` (containment under `--output-dir`), not the caller-supplied value. (Assumption: that resolver's containment is correct — pre-existing.)
7. **Validator surface:** No risky widening. The three fields make the daemon/iterate surfaces strictly stricter, never more permissive. Image egress confined to the MCP handlers.

## Verdict

**ACCEPT.** The load-bearing invariants (audit exclusion, byte hard-cap, metadata-free transport copy, fail-soft, cascade parity, no new path-traversal or daemon egress) all hold and are test-anchored. The two LOW findings are optional hardening, not merge blockers; the strongest recommendation is `except BaseException` → `except Exception` at `mcp_server.py:937` to structurally protect the global audit-emission invariant against future edits.

---

## Disposition (parent session, 2026-06-25)

All findings actioned before commit:

- **LOW (except BaseException → except Exception):** FIXED. `_maybe_attach_return_image` now catches `except Exception` with a comment documenting that `KeyboardInterrupt`/`SystemExit`/future audit `BaseException` must propagate (global §0 rule 2). Folded the code-reviewer's matching MEDIUM-1.
- **LOW (max_return_px upper clamp):** FIXED rather than documented. Added `_MAX_RETURN_PX_CEILING = 4096`; `eff_px = min(eff_px, _MAX_RETURN_PX_CEILING)` bounds the first-encode spike. The returned payload was already byte-capped; this caps the transient in-memory copy too.
- **INFO (metadata-free guarantee / audit scalars):** No code change; the metadata-strip control is now proven by a load-bearing test (N3b) that plants a path-bearing tEXt chunk in the source PNG and asserts the transport copy strips it — folding the code-reviewer's MEDIUM-2 (the prior N3 assertion passed trivially on a metadata-free source).

Code-reviewer (Opus) ran in parallel: no HIGH findings; same two items surfaced as MEDIUM and are fixed above. Full 9-suite run green (1412 tests, 0 failures) after the fixes.
