# ADR-017: Optional base64 image return on the MCP `generate` surface + OpenWebUI native-tool inline rendering

**Date:** 2026-06-25
**Status:** accepted (design settled with Grant 2026-06-25; spike-validated. Implementation gated by `code-reviewer` + `security-auditor`, both Opus, per slice — see Changelog.)
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant reviewed.
**Relates to:** [ADR-011](ADR-011-comfyless-mcp-server.md) (MCP as the LLM-agent calling interface; this extends §2 `generate` output and builds on §3e MCP-returned artifact redaction). Orthogonal to [ADR-015](ADR-015-mcp-catalog-reference-resolution.md) (reference resolution — untouched). Vision: `docs/vision/slice-mcp-image-return-owui.md`. Enables the long-planned LLM-image-judge loop (Backlog; [[project_llm_image_judge]]).

---

## Context

The comfyless MCP `generate` tool returns a JSON frame (`output_path`, `resolved_params`, `elapsed_seconds`) — a **path**, not the image. The operational goal that surfaced 2026-06-25: drive image generation from **OpenWebUI** (an LLM, dolphin-mistral via vLLM, decides to generate) and have the result **render inline in the chat**, replacing the copy-paste-prompt-into-comfyless workflow. A second, related goal is the LLM-image-judge loop (Backlog): a vision-capable model scoring generated output, which needs the image bytes in a model-consumable form.

Two spikes (sources in the Vision) eliminated the obvious paths:

1. **mcpo** (the MCP→OpenAPI bridge OpenWebUI consumes) **flattens** an MCP `ImageContent` block to a bare `data:…;base64,…` **string** — the image block never survives as a block.
2. **OpenWebUI's tool-server path feeds the tool result back to the *model* as text.** A base64 data-URI in that text is truncated/garbled by the model and bloats context (a 1 MP PNG ≈ 1.4–4 MB base64); multiple open OWUI issues confirm this is broken with no fix.
3. **`file://`** is blocked twice over: OWUI runs in a container (so `file://` is the container's fs, not the host) and browsers refuse `file://` resources embedded in an http(s) page.

The working pattern (real prior art: Haervwe `comfyui_image_to_image_tool.py`, `openrouter_image_pipe.py`) is a **native OpenWebUI tool** (Python plugin running *inside* OWUI) that fetches the image bytes, uploads them to OWUI's internal file store, and emits a chat `message` event containing `![](/api/v1/files/{id}/content)`. **OWUI's frontend renders it; the model only receives a short confirmation.** That tool needs the bytes — and it gets them cheaply as a JSON field from our existing response. The very same `{base64, mime}` shape is what a vLLM/OpenAI-compatible vision judge accepts as `image_url`, so one return shape serves both consumers.

This is a Red-Zone-adjacent change: it opens a **new data-egress surface** on the MCP boundary (image bytes now leave the server to an LLM-driven caller). Grant explicitly flagged it ("crossing new lines") and required a security review. Per global §12 this ADR precedes the code; `security-auditor` (Opus) reviews the implementation.

**Threat-model note.** The returned image is the caller's *own* generated content — the same caller supplied the prompt that produced it — so returning it is not a new information leak about the server or other users. The load-bearing control is that the bytes appear **only** in the response frame, never in the operator audit line / stderr, and that the returned copy is **size-bounded**. The OWUI native tool runs with the user's own OWUI identity and uploads the user's own image to the user's own file store — no new authorization decision.

## Decision

### 1. comfyless: optional, gated, size-bounded base64 return

`generate` and `generate_cascade` accept two new optional request params:

- **`return_image`** (bool, default **`false`**). When `false`/absent the response is byte-identical to today — no image fields, no extra work.
- **`max_return_px`** (int, default **`1024`**). Bounds the **longest edge** of the returned image.

When `return_image=true`, the response JSON gains:

- **`image_b64`** — base64 of a PNG re-encoded from the on-disk output, **downscaled so its longest edge ≤ `max_return_px`** (aspect preserved; never upscaled). The **full-resolution PNG on disk is unchanged** — the bound applies only to the transport copy.
- **`image_mime`** — the constant **`"image/png"`** (v1 is PNG-only; JPEG deferred).

Returned as **plain JSON fields in the existing `TextContent` response** — NOT an MCP `ImageContent` block (mcpo would flatten it). `resolved_params` and the ADR-015 catalog-name/notice contract are untouched; `image_b64`/`image_mime` carry no filesystem string.

### 2. Invariants (enforced by the slice; see Vision for the full list + negative tests)

- **Audit exclusion (load-bearing):** the base64 payload appears only in the response frame, **never** in `_emit_audit_line` / stderr.
- **Size bound:** returned longest edge ≤ `max_return_px`; on-disk PNG stays full-res.
- **§3e preserved:** the bytes are re-encoded from the already-written, already-§3e-redacted PNG; nothing about the on-disk artifact changes.
- **Fail-soft:** if the transport encode/downscale fails, the generation still returns its normal frame (the image is on disk); `image_b64` is simply omitted. A return-image failure MUST NOT fail a successful generation.
- **Both paths:** non-cascade and cascade honor the flag identically.

### 3. OpenWebUI native tool (in-repo integration deliverable)

A native OWUI **Tool** at `comfyless/integrations/openwebui/` (versioned + reviewed in this repo, installed into OWUI). It: receives `generate_image(prompt, model, …)` from the model; calls mcpo `/generate` with `return_image=true`; base64-decodes; uploads via `open_webui.routers.files.upload_file_handler(process=False)` using the injected `__request__`/`__user__`; emits a `message` event with `![](/api/v1/files/{id}/content)`; returns a short text confirmation to the model. **The model never handles the bytes.** Modeled on the Haervwe templates.

### 4. Reuse for the vision-judge loop

The `{image_b64, image_mime}` shape is directly consumable by a vLLM/OpenAI-compatible vision model as `{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}`. The judge loop (Backlog) builds on this substrate; it is out of scope here.

## Alternatives Rejected

- **A. MCP `ImageContent` block.** mcpo flattens it to a data-URI string; never renders inline through the tool-server path. Full base64 cost, zero benefit.
- **B. base64 data-URI in the model's tool-result text.** Truncated/garbled by the model; context bloat; OWUI issues confirm broken. (This is *why* the bytes must be consumed by a native tool, not the model.)
- **C. Standalone HTTP file server over `--output-dir` + return a URL.** Reliable inline render, but stands up a **new network-exposed file-serving surface** (path-traversal defense, exposure of all generated output) and grows comfyless a file-serving interface (the ADR-011 §6 HTTP-transport gates). Grant explicitly preferred not to build an HTTP server. The base64 field gives the same end result with comfyless owning no file-serving.
- **D. Volume-mount `--output-dir` into the OWUI container.** Zero comfyless change, but couples the two via a shared filesystem (compose/mount change; container reads host files) and a host path is useless to the vision-judge (which needs base64/URL). Grant preferred base64; it serves both consumers.
- **E. `file://` URL.** Blocked by container fs + browser security.
- **F. OpenWebUI native Image Generation (Settings → Images).** Frontend-rendered, but config-driven and expects an A1111/ComfyUI/OpenAI-image HTTP API — a separate, larger build — and the LLM does not tool-call it the same conversational way. Noted as a viable but different product surface, not this slice.

## Deferred / Out of Scope

- **JPEG / `format` param** — v1 is PNG-only (lossless, matches on-disk). JPEG (smaller photographic transport) is a later refinement.
- **Vision-judge loop implementation** — this ADR provides the substrate only.
- **`max_return_px` as a spawn-time default override** — v1 is per-request with a fixed default.
- **Streaming / progressive / multi-image batch returns.**
- **mcpo `--api-key` / persistent-service hardening** — separate ops concern (flagged in the mcpo bridge memory).
- **Return-image on `extract_params` / `iterate` / `edit`** — generate + cascade only.

## Changelog

- **2026-06-25 (initial draft, accepted):** Design settled with Grant after two spikes (mcpo image handling; OWUI native-tool inline-render mechanism — both with sources, in the Vision). Adopts an optional, gated (`return_image`, default false), size-bounded (`max_return_px`, default 1024) base64 PNG return as plain JSON fields on `generate`/`cascade`, consumed by an in-repo OpenWebUI native tool that uploads to OWUI's file store and emits an inline image (model never handles the bytes); the same shape feeds the future vision-judge loop. Rejected: MCP ImageContent (mcpo-flattened), base64-into-model-context (truncated/bloated), standalone HTTP file server (new exposure; Grant declined), output-dir mount (filesystem coupling; useless to the judge), `file://` (blocked), OWUI native image-gen (separate product surface). OQ-1 (`max_return_px`=1024) and OQ-2 (PNG-only v1) resolved by Grant. Implementation per the Vision change plan: step 1 comfyless base64 return (code-reviewer + security-auditor), step 2 OWUI native tool (code-reviewer + security-auditor), step 3 docs closure. Security review runs on the implemented code (not a separate design pass), per Grant — the load-bearing control is the audit-exclusion + size-bound invariants on bytes of an already-§3e-redacted PNG.
