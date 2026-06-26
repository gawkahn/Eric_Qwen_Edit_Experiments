# OpenWebUI native tool — Comfyless Image Generation

A native [OpenWebUI](https://github.com/open-webui/open-webui) **Tool** that lets a
chat model generate images with the comfyless MCP server and have them render
**inline** in the conversation.

It implements **step 2** of
[`ADR-017`](../../../docs/decisions/ADR-017-mcp-image-return-owui-integration.md)
(see also the Vision slice `docs/vision/slice-mcp-image-return-owui.md`).

## Why a native tool (not "just an MCP tool")

OpenWebUI already reaches comfyless as an MCP server through the **mcpo**
OpenAPI bridge. But that path feeds a tool's result back to the *model as text*,
and a base64 data-URI in that text is truncated/garbled and bloats context
(ADR-017, alternatives A & B). The working pattern is a **native tool** —
a Python plugin that runs *inside* OpenWebUI — which:

1. calls mcpo `/generate` with `return_image=true` (ADR-017 step 1),
2. base64-decodes the returned `image_b64` field,
3. uploads the bytes into OpenWebUI's own file store **as the calling user**
   via `upload_file_handler(..., process=False)`,
4. emits a chat `message` event containing `![](/api/v1/files/{id}/content)`, and
5. returns only a **short text confirmation** to the model.

The model never handles the image bytes. OpenWebUI's frontend renders the image.

## Prerequisites

- comfyless MCP server reachable through **mcpo** (see the
  [`mcpo bridge`](../../README.md) notes / project memory). By default the tool
  posts to `http://172.17.0.1:8090/generate` — `172.17.0.1` is the docker bridge
  gateway, i.e. the host as seen from inside the OpenWebUI container.
- An OpenWebUI build that exposes `open_webui.routers.files.upload_file_handler`
  with a `process` parameter (verified against the running container's signature
  during development — `required_open_webui_version: 0.5.0`).

## Install

OpenWebUI tools are installed by pasting their source into the admin UI — there
is no file drop-in. The copy in this repo is the **source of truth**; the UI copy
is a deployment.

1. Open OpenWebUI → **Workspace → Tools → `+` (Create new tool)**.
2. Paste the entire contents of [`generate_image_tool.py`](generate_image_tool.py).
3. Save. OpenWebUI parses the docstring frontmatter (`title`, `requirements: aiohttp`)
   and registers a `generate_image` function.
4. Open the tool's **Valves** (gear icon) and adjust if needed:
   - `mcpo_base_url` — change if mcpo is not on the default docker bridge gateway.
   - `api_key` — set if mcpo was started with `--api-key`.
   - `default_model` — a catalog model name to use when the model doesn't pick one.
   - `max_return_px` / `max_return_bytes` — transport-image bounds (server clamps
     bytes to a 1 MiB ceiling regardless).
5. Enable the tool for the model/chat (the **tools** toggle in the chat input, or
   per-model default tools in the model's settings).

## Usage

Ask the model to generate an image (e.g. "generate a picture of a red fox in
snow"). When the model has the tool enabled it calls `generate_image(prompt=…)`,
and the image appears inline. The model only sees a one-line confirmation
(`model`, dimensions, seed, elapsed) — by design.

## Updating

Edit `generate_image_tool.py` here, commit, then re-paste the new source into the
OpenWebUI tool editor and save. Bump the `version:` line in the docstring so the
deployed copy is traceable to a repo revision.

## Security notes

This tool is the data-egress consumer described in ADR-017's threat model:

- `mcpo_base_url` / `api_key` live in **admin Valves**, not tool parameters, so a
  chat model cannot redirect the outbound call (no SSRF from model input).
- The returned payload is size-bounded server-side (`max_return_bytes` ≤ 1 MiB);
  the tool **also** rejects any base64 payload over a fixed 4 MiB ceiling before
  decoding, as a memory safety net against a compromised/buggy upstream (mcpo is
  unauthenticated on the bridge — ADR-017 defers mcpo auth).
- The upload runs under the **caller's own** OpenWebUI identity
  (`Users.get_user_by_id(__user__["id"])`) and `process=False` skips the RAG
  ingestion pipeline — the image is stored, not indexed.
- The stored file is pinned to `content-type: image/png` (ADR-017 invariant 4),
  not the server-reported mime, to avoid content-type confusion on the file URL.
- The model receives no filesystem path and no raw bytes — only a file-store URL
  rendered by the frontend. All upstream error detail (bridge URL, response body,
  exception text) is logged operator-side only, never returned into LLM context.
