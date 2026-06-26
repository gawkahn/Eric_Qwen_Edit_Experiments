# Security review — ADR-017 step 2: OpenWebUI native image-generation tool

**Date:** 2026-06-26
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored the tool and this disposition; `security-auditor` (Opus) and `code-reviewer` (Opus) reviewed; Grant reviewed.
**Subject:** `comfyless/integrations/openwebui/generate_image_tool.py` + `README.md`
**ADR:** [ADR-017](../decisions/ADR-017-mcp-image-return-owui-integration.md) (Decision §3). Step 1 review: [review-adr017-mcp-image-return-2026-06-25.md](review-adr017-mcp-image-return-2026-06-25.md).
**Trigger (§12):** new data-egress surface on an LLM-driven boundary; Red-Zone-adjacent (Grant: "crossing new lines").

---

## Verdict

**security-auditor: ACCEPT WITH FINDINGS.** **code-reviewer: no HIGH.**

The primary path and the four named high-risk surfaces are sound:

- **SSRF / outbound redirection:** clean. `mcpo_base_url` / `api_key` are admin
  Valves, never tool parameters — a chat model cannot redirect the POST.
- **Identity / authz:** least-privilege. Upload runs under the caller's own
  identity (`Users.get_user_by_id(__user__["id"])`); `process=False` skips RAG
  ingestion. No new authorization decision (ADR-017 threat model).
- **Injection:** `_safe_token` is a regex allowlist; the inline URL uses an
  OWUI-minted file id. No injection into the chat message or file store.
- **Secret non-disclosure:** the `api_key` Valve lives only in the
  `Authorization` header — never in a returned or emitted string.

All findings were on the error/degraded branches and the response-consumption
boundary. Every one was addressed before commit (table below).

## Findings and disposition

| # | Sev | Source | Finding | Disposition |
|---|-----|--------|---------|-------------|
| S1 | MED | sec | On the no-preview fail-soft branch, the server `output_path` was returned to the model (violates ADR threat model "model receives no filesystem path"). | **Fixed.** Branch now returns a path-free constant; surfaces only the path-free ADR-015 `notices` reason. |
| S2 | MED | sec | Transport-error string leaked the internal mcpo URL + raw exception to the model. | **Fixed.** Generic `"could not reach the image backend"`; URL/exc logged operator-side via module logger. |
| S3 | MED | sec | No independent size bound before `b64decode` — 1 MiB cap is server-side only; unauthenticated upstream → decode-bomb / OOM risk. | **Fixed.** Added `_B64_HARD_CEILING = 4 MiB` guard on `len(image_b64)` before decode. |
| S4 | MED | sec | Stored file's content-type/extension trusted server `image_mime` (content-type confusion if upstream returns e.g. `text/html`). | **Fixed.** Pinned to `image/png` + `.png` unconditionally (ADR-017 invariant 4). |
| S5 | LOW | sec | Error branches echoed raw upstream body / exception text to the model. | **Fixed.** All branches return generic strings; detail logged operator-side. |
| S6 | LOW | sec | No null-check on the resolved user object before upload. | **Fixed.** `if user_obj is None: return error`. |
| C1 | MED | code | Success string claimed "displayed inline" even when `__event_emitter__` is None (image uploaded but never rendered). | **Fixed.** Emitter-None branch returns a degraded message including the file URL, not "shown inline". |
| C2 | MED | code | Sync `Users.get_user_by_id` blocks the event loop. | **Accepted as-is, documented.** Fast indexed lookup, matches OWUI native-tool convention; comment added. Solo-defensible (single-user instance). |
| C3 | LOW | code | JSON-parse failure reported as a transport error. | **Fixed.** `resp.json` split out of the connect try; distinct "unreadable response" message. |
| C4 | LOW | code | Duplicate `resolved_params` extraction. | **Fixed.** Computed once after frame validation, reused. |
| C5 | LOW | code | Server fail-soft `notices` discarded on the no-image branch. | **Fixed.** `notices` appended to the no-preview return string (folds with S1). |

## Residual / accepted risk

- **C2 (event-loop block on user lookup):** accepted. Indexed single-row read on a
  single-user instance; offloading to a thread adds dependency surface for no
  practical gain. Re-evaluate if this tool is ever deployed multi-tenant.
- **Upstream mcpo authentication** remains deferred (ADR-017 Deferred / Out of
  Scope). The S3 decode-ceiling and S4 mime-pin are the defense-in-depth controls
  that bound the blast radius of that deferral on this consumer.

## Coverage

Reviewed the full tool (`generate_image_tool.py`), the README, ADR-017, and the
Vision slice. The server-side `_encode_return_image` / clamp was reviewed and
accepted in step 1 (treated as an upstream control here). OWUI runtime internals
(`upload_file_handler`, `Users`, tool-arg binding) are external; their behavior
was confirmed against the running container's signature during development and
otherwise treated as documented assumptions.
