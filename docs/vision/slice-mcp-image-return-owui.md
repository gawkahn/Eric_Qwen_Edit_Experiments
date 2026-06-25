# Slice Vision — Optional base64 image return on `generate` + OpenWebUI inline render

**Date:** 2026-06-25
**ADR:** to be written before code — **ADR-017** (MCP image-bytes return + OpenWebUI native-tool integration), extending [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) §2 (generate contract) and §3e (MCP-returned artifact redaction), orthogonal to [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) (reference resolution — untouched here).
**Status:** PROPOSED 2026-06-25 — awaiting Grant approval, then ADR-017.
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant reviewed.

---

## Posture

> **Posture:** Boundary: `entrypoint` (MCP `generate`/`generate_cascade` response contract) + a new `integration` deliverable (OpenWebUI native tool). Risk factors: **external exposure** (image bytes now cross the MCP response boundary to an LLM-driven caller) + **near security truth** (new data-egress surface on the Red-Zone MCP interface; user explicitly flagged "crossing new lines"). **Risk level: L3.**

## Intent

Let `generate`/`generate_cascade` **optionally** return the generated image as base64 (gated, size-bounded) so a native OpenWebUI tool can render it inline — and so the future vision-judge loop can feed it to a vision model — **without** a standalone HTTP file server and **without** the LLM ever handling the bytes.

## Why this shape (decided 2026-06-25, spike-validated)

The naive paths are dead ends (spike-confirmed with sources): mcpo flattens MCP `ImageContent` to a bare data-URI string; OpenWebUI's tool-server path feeds tool results to the **model** as text, where a base64 data-URI is truncated/bloats context; `file://` is blocked (container fs + browser security). The working pattern (real prior art: Haervwe `comfyui_image_to_image_tool.py` / `openrouter_image_pipe.py`) is a **native OpenWebUI tool** that fetches bytes, uploads them to OWUI's internal file store (`upload_file_handler(process=False)`), and emits a `message` event with `![](/api/v1/files/{id}/content)` — **OWUI renders it; the model only gets a short confirmation.** That tool needs the bytes; it gets them as a **plain JSON `image_b64` field** in our existing response (a JSON field, NOT an MCP `ImageContent` block, which mcpo mangles). The same `{base64, mime}` shape is exactly what a vLLM/OpenAI-compatible vision judge accepts in `image_url`, so one return shape serves both.

## Invariants (must always be true)

1. **Default path byte-unchanged.** `return_image` absent or `false` → the response is identical to today (no `image_b64`/`image_mime` keys, no extra work, no extra latency). Every existing MCP test holds without modification.
2. **Image bytes never enter the audit line / stderr.** The base64 payload appears ONLY in the MCP response frame returned to the caller — never in `_emit_audit_line`, never on stderr. (Audit logs request args + status, not the response; this slice must not change that.)
3. **Returned base64 is size-bounded.** When `return_image=true`, the returned image's longest edge is ≤ `max_return_px` (default 1024); a 50 MP Qwen generation does not yield a 50 MP base64. The **full-resolution PNG on disk is NOT downscaled** — the bound applies only to the transport copy.
4. **`image_mime` is exactly `"image/png"`** (we re-encode PNG); no other type crosses the boundary.
5. **No abs_path or filesystem string leaks via the new fields.** `image_b64` is image bytes; `image_mime` is a constant. The ADR-015 name/notice contract is untouched; `resolved_params` still renders catalog names only.
6. **On-disk PNG + §3e redaction unchanged.** `return_image` reads the already-written, already-§3e-redacted PNG; it does not alter what is saved or its embedded metadata.
7. **Both paths covered.** Non-cascade `_handle_generate` and `_handle_generate_cascade` honor `return_image` identically.
8. **Graceful degrade.** If the transport encode/downscale fails for any reason, the generation still succeeds and returns its normal frame (`output_path`, `resolved_params`) — the image is on disk regardless; `image_b64` is simply omitted (optionally with an INFO notice). A return-image failure MUST NOT fail the generation.

## Failure semantics

Fail-soft on the *return-image* path only (the image already exists on disk; never fail a successful generation because the optional transport copy couldn't be made). Everything else fails closed as today. Invalid `return_image`/`max_return_px` types are rejected by the canonical validator (`ValidationError`) before generation, same as any other param.

## Out of scope

- **A standalone HTTP file server** — explicitly avoided (the whole point of this shape).
- **MCP `ImageContent` blocks** — mcpo mangles them; we use a JSON field.
- **The vision-judge loop implementation** — this slice provides the substrate (`{base64, mime}`), not the judge.
- **Streaming / progressive images, multi-image batches.**
- **`extract_params` / `iterate` / `edit`** — generate + cascade only.
- **Auth on mcpo / the OWUI tool's authz** — the OWUI tool runs with the user's own OWUI identity uploading the user's own generated image; no new authz decision.

## Negative cases required

- `return_image` absent → no `image_b64`/`image_mime`; response byte-identical to pre-slice (regression anchor).
- `return_image=false` explicit → same as absent.
- `return_image=true` → `image_b64` present, base64-decodes to a valid PNG, longest edge ≤ `max_return_px`.
- A generation larger than the cap (e.g. 1536px) with `max_return_px=1024` → returned image longest edge ≤ 1024, **and the on-disk PNG is still full-res** (assert both).
- The base64 payload string does **not** appear in the captured stderr/audit line.
- `return_image` non-bool / `max_return_px` non-int → `ValidationError` before generation.
- Cascade path with `return_image=true` → `image_b64` present and valid.
- Encode-failure path (mock the encoder to raise) → generation still returns `output_path` + `resolved_params`; no `image_b64`; no exception escapes.

## Proof hooks

- Positive: `./.venv/bin/python3 test_mcp_server.py` (new return-image cases, non-cascade + cascade) + full 9-suite green.
- Negative: the eight cases above, each asserting the stated property (esp. the audit-exclusion and the on-disk-stays-full-res cases).
- Integration (manual, OWUI tool): dolphin in OpenWebUI calls `generate_image(prompt, model)`; the image renders inline in the chat; the model's visible turn contains only a short confirmation, not base64.

## Red Zone ownership

MCP response contract + new data egress — owned by **Grant**. ADR-017 written before code. `code-reviewer` (Opus) + `security-auditor` (Opus) both required on the comfyless slice; `security-auditor` also reviews the OWUI tool integration surface (model-supplied prompt → generation → upload to the user's own file store). Reviews saved to `docs/security/`.

## Change plan

| Step | Scope | Reviews |
|---|---|---|
| **0** | **ADR-017** — record the decision (native-tool-over-tool-server, base64-JSON-field-over-ImageContent, files-API-upload, no standalone server) with the rejected alternatives the spike eliminated. Written before code (§12 order). | — (design doc) |
| **1** | **comfyless base64 return.** Add `return_image` (bool, default false) + `max_return_px` (int, default 1024) to the generate schema; a `_encode_return_image(output_path, max_px) -> (b64, mime)` helper (Pillow downscale + PNG encode); wire into both handlers' response building; tests (all negative cases). | code-reviewer + security-auditor (Opus) |
| **2** | **OWUI native tool** at `comfyless/integrations/openwebui/generate_image_tool.py` + install README. Calls mcpo `/generate` with `return_image=true`, decodes, `upload_file_handler(process=False)`, emits `![](/api/v1/files/{id}/content)`. Modeled on the Haervwe templates. Manual e2e proof. | code-reviewer + security-auditor (Opus) |
| **3** | Docs closure — ADR-017 Changelog (IMPLEMENTED), CLAUDE.md test count, Backlog, Obsidian mirrors. | — |

## Open Questions

- **OQ-1.** `max_return_px` default — proposing **1024** (well within vLLM Qwen2-VL `max_pixels` defaults and cheap to render). Confirm or pick another.
- **OQ-2.** Should `return_image` also accept an optional `format: "png"|"jpeg"` (JPEG would shrink transport for photographic output)? Proposing **PNG-only for v1** (lossless, matches on-disk), JPEG as a later refinement. Confirm.
