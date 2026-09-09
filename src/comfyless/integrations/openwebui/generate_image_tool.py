"""
title: Comfyless Image Generation
author: Grant Kahn
version: 0.2.3
required_open_webui_version: 0.5.0
requirements: aiohttp
license: MIT
description: >
  Generate an image with the comfyless MCP server (reached through the mcpo
  OpenAPI bridge) and render it inline in the chat. The tool fetches the image
  bytes as a base64 JSON field (ADR-017 return_image=true), uploads them to
  OpenWebUI's own file store under the calling user's identity, and emits an
  inline `![](...)` message. The language model never receives the raw bytes —
  only a short text confirmation — which keeps the model context clean and
  avoids the broken data-URI-in-tool-result path (ADR-017, alternative B).
"""

# ADR-017 step 2 — see docs/decisions/ADR-017-mcp-image-return-owui-integration.md
# and docs/vision/slice-mcp-image-return-owui.md.
#
# This is a *native* OpenWebUI Tool (a Python plugin that runs INSIDE the
# OpenWebUI container), NOT an MCP tool. Install it via OpenWebUI →
# Workspace → Tools → "+". See README.md in this directory.

import base64
import inspect
import io
import json
import logging
import re
from typing import Any, Awaitable, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

# These imports resolve inside the OpenWebUI runtime. They are deliberately
# at module scope so a missing/renamed symbol fails loudly at install time
# rather than silently mid-generation.
from fastapi import Request, UploadFile
from starlette.datastructures import Headers

from open_webui.models.users import Users
from open_webui.routers.files import upload_file_handler

log = logging.getLogger("comfyless.owui.generate_image")

# Absolute upper bound on the base64 payload we will decode, independent of the
# server-side clamp (1 MiB, ADR-017) and the admin Valve. A memory safety net
# against an oversized response from a compromised/buggy upstream — mcpo is
# reachable unauthenticated on the docker bridge (ADR-017 defers mcpo auth).
_B64_HARD_CEILING = 4 * 1024 * 1024  # 4 MiB

# Hard ceiling on the raw mcpo response body read into memory, before JSON
# parsing — defense-in-depth against an oversized response from a compromised/
# buggy unauthenticated upstream. Comfortably above _B64_HARD_CEILING (the b64
# field is a substring of the JSON body) plus frame overhead.
_RESPONSE_BODY_CEILING = 8 * 1024 * 1024  # 8 MiB

# Short timeout for read-only catalog list calls (they return tiny payloads);
# distinct from the long generate timeout so a hung list can't block a chat turn.
_LIST_TIMEOUT_S = 30


def _coerce_frame(data: Any) -> Optional[dict]:
    """Normalise mcpo's response into the comfyless generate frame dict.

    mcpo may hand back the MCP TextContent JSON in several shapes depending on
    version/config: the frame dict directly, a JSON string, a single content
    block, or a list of content blocks. We probe defensively and return the
    first dict that looks like our frame (has output_path / image_b64 /
    resolved_params), or None if nothing matches.
    """
    if isinstance(data, dict):
        if any(k in data for k in ("image_b64", "output_path", "resolved_params")):
            return data
        for key in ("result", "data"):
            if key in data:
                inner = _coerce_frame(data[key])
                if inner is not None:
                    return inner
        if isinstance(data.get("content"), list):
            return _coerce_frame(data["content"])
        return None
    if isinstance(data, str):
        try:
            return _coerce_frame(json.loads(data))
        except (ValueError, TypeError):
            return None
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "text" in item:
                inner = _coerce_frame(item["text"])
            else:
                inner = _coerce_frame(item)
            if inner is not None:
                return inner
        return None
    return None


def _safe_token(value: Any, fallback: str = "img") -> str:
    """Filesystem-safe short token for a generated filename component."""
    token = re.sub(r"[^A-Za-z0-9_-]", "", str(value))[:32]
    return token or fallback


def _parse_weights_csv(s: str) -> list[float]:
    """Parse a comma/semicolon-separated float list. Raises ValueError on a
    non-numeric token so the caller can surface a clear chat error."""
    parts = [p for p in s.replace(";", ",").split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def _parse_loras(s: str) -> list[dict]:
    """Parse 'name:weight,name2' into [{name, weight}, ...]. Weight is optional
    (default 1.0). Raises ValueError on an empty name or non-numeric weight."""
    out: list[dict] = []
    for entry in s.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, weight = entry.rpartition(":")
        if sep:  # had a ':' — left side is the name, right side the weight
            name = name.strip()
            w = float(weight.strip())
        else:    # no ':' — the whole token is the name, default weight
            name, w = entry, 1.0
        if not name:
            raise ValueError(f"lora entry missing a name: {entry!r}")
        out.append({"name": name, "weight": w})
    return out


class Tools:
    class Valves(BaseModel):
        mcpo_base_url: str = Field(
            default="http://172.17.0.1:8090",
            description=(
                "Base URL of the mcpo bridge fronting comfyless, as reachable "
                "from inside the OpenWebUI container. 172.17.0.1 is the default "
                "docker bridge gateway (the host)."
            ),
        )
        generate_path: str = Field(
            default="/generate",
            description="mcpo path for the comfyless generate tool.",
        )
        api_key: str = Field(
            default="",
            description="Optional mcpo bearer token (sent as Authorization: Bearer). Empty = no auth header.",
        )
        max_return_px: int = Field(
            default=768,
            description="Longest-edge cap requested for the returned transport image (ADR-017).",
        )
        max_return_bytes: int = Field(
            default=1048576,
            description="Byte cap requested for the returned base64 payload; server clamps to a 1 MiB ceiling.",
        )
        request_timeout_s: int = Field(
            default=600,
            description="HTTP timeout for the generate call; generation can take minutes.",
        )
        default_model: str = Field(
            default="",
            description="Optional catalog model name used when the caller does not specify one.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def _post_mcpo(
        self, path: str, payload: dict, timeout_s: Optional[int] = None
    ) -> tuple[Any, Optional[str]]:
        """POST to an mcpo endpoint.

        Returns (data, None) on success, or (None, error) on failure where
        `error` is a generic, model-safe message. All upstream detail (the
        internal bridge URL, the raw response body, exception text) is logged
        operator-side only — none of it belongs in LLM context.

        `timeout_s` overrides the per-call timeout (defaults to the long
        generate timeout). The raw body is read with a hard ceiling so a
        compromised/buggy unauthenticated upstream cannot OOM the container.
        """
        url = self.valves.mcpo_base_url.rstrip("/") + "/" + path.lstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.valves.api_key.strip():
            headers["Authorization"] = f"Bearer {self.valves.api_key.strip()}"
        total = timeout_s if timeout_s is not None else int(self.valves.request_timeout_s)
        timeout = aiohttp.ClientTimeout(total=total)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    status = resp.status
                    # Read the FULL body in bounded chunks. resp.content.read(n)
                    # does NOT guarantee n bytes — it returns whatever is buffered,
                    # so a single read(ceiling) TRUNCATES a multi-chunk body (e.g.
                    # the ~1 MiB generate frame). Iterate to EOF, capping the
                    # running total to stay bounded against an oversized upstream.
                    body = bytearray()
                    oversized = False
                    async for chunk in resp.content.iter_chunked(65536):
                        body.extend(chunk)
                        if len(body) > _RESPONSE_BODY_CEILING:
                            oversized = True
                            break
        except Exception as exc:  # noqa: BLE001
            log.warning("mcpo POST to %s failed: %s", url, exc)
            return None, "could not reach the image backend"
        if oversized:
            log.warning("mcpo %s response exceeded %d bytes; rejecting", path, _RESPONSE_BODY_CEILING)
            return None, "image backend returned an oversized response"
        body_text = bytes(body).decode("utf-8", errors="replace")
        if status != 200:
            log.warning("mcpo %s returned HTTP %s: %.1000s", path, status, body_text)
            return None, f"image backend returned HTTP {status}"
        try:
            return json.loads(body_text), None
        except (ValueError, TypeError) as exc:
            log.warning("mcpo %s returned an unparseable body: %s", path, exc)
            return None, "image backend returned an unreadable response"

    async def _list_catalog(self, path: str, label: str) -> str:
        """Call an mcpo list_* endpoint and format it for the model.

        Read-only: returns catalog NAMES (and family), never paths — the names
        are the exact strings to pass back as `model` (ADR-015 opaque handles).
        """
        data, err = await self._post_mcpo(path, {}, timeout_s=_LIST_TIMEOUT_S)
        if err is not None:
            return f"Error: {err}."
        if not isinstance(data, list):
            log.warning("mcpo %s returned a non-list: %.200s", path, str(data))
            return f"Error: image backend returned an unexpected {label} response."
        if not data:
            return f"No {label} available."
        lines = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "?")
            family = item.get("model_family") or item.get("target_family") or ""
            lines.append(f"- {name}" + (f"  (family: {family})" if family else ""))
        if not lines:
            # Non-empty data but nothing usable — junk, not an empty catalog.
            log.warning("mcpo %s returned %d items, none usable: %.200s", path, len(data), str(data))
            return f"Error: image backend returned an unexpected {label} response."
        lines.sort(key=str.lower)
        return f"{len(lines)} {label} available:\n" + "\n".join(lines)

    async def generate_image(
        self,
        prompt: str,
        model: str = "",
        negative_prompt: str = "",
        width: int = 0,
        height: int = 0,
        seed: int = -1,
        steps: int = 0,
        loras: str = "",
        rebalance: bool = False,
        rebalance_mult: float = 0.0,
        rebalance_weights: str = "",
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Generate an image from a text prompt and display it inline in the chat.

        :param prompt: Text description of the image to generate.
        :param model: Optional catalog model name (e.g. "qwen-image", "flux2"). Empty = server/tool default.
        :param negative_prompt: Optional content to avoid (ignored by flux-family models).
        :param width: Optional width in pixels (0 = model default).
        :param height: Optional height in pixels (0 = model default).
        :param seed: Optional seed for reproducibility (-1 = random).
        :param steps: Optional number of sampling steps (0 = model default).
        :param loras: Optional LoRA(s) by catalog name (discover via list_loras), comma-separated, each optionally "name:weight" (default weight 1.0). E.g. "MysticXXX_KREA2_v1:0.8". If a name itself contains a ':', always give an explicit ":weight" so it isn't mis-split. Empty = none.
        :param rebalance: Krea-2 only: set true to boost detail / bypass the safety filter's quality dilution. Ignored by non-Krea models.
        :param rebalance_mult: Optional rebalance strength (0 = server default 4.0; try 1.5–2.0 for a gentler effect). Only used when rebalance is true.
        :param rebalance_weights: Optional 12 comma-separated per-layer-tap gains for expert rebalance tuning (e.g. "1,1,1,1,1,1,1,2.5,5,1.1,4,1"). Empty = server default preset. Only used when rebalance is true.
        :return: A short confirmation string. The image is rendered inline; do not attempt to read its bytes.
        """
        user = __user__ or {}

        async def emit_status(description: str, done: bool = False) -> None:
            if __event_emitter__ is not None:
                await __event_emitter__(
                    {"type": "status", "data": {"description": description, "done": done}}
                )

        if not prompt or not prompt.strip():
            return "Error: prompt is empty."
        if __request__ is None or not user.get("id"):
            return (
                "Error: this tool is missing OpenWebUI request/user context and "
                "cannot upload the result. Ensure it is run as a native tool."
            )

        # Build the request — only include params the caller actually set so the
        # server's FAMILY_DEFAULTS (commit f9b3c2e) fill the rest.
        payload: dict[str, Any] = {
            "prompt": prompt,
            "return_image": True,
            "max_return_px": int(self.valves.max_return_px),
            "max_return_bytes": int(self.valves.max_return_bytes),
        }
        chosen_model = (model or "").strip() or self.valves.default_model.strip()
        if chosen_model:
            payload["model"] = chosen_model
        if negative_prompt and negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt
        if width and width > 0:
            payload["width"] = int(width)
        if height and height > 0:
            payload["height"] = int(height)
        if seed is not None and seed >= 0:
            payload["seed"] = int(seed)
        if steps and steps > 0:
            payload["steps"] = int(steps)
        if loras and loras.strip():
            try:
                parsed_loras = _parse_loras(loras)
            except ValueError as exc:
                return f"Error: could not parse loras {loras!r}: {exc}."
            if parsed_loras:
                payload["loras"] = parsed_loras
        if rebalance:
            payload["rebalance"] = True
            if rebalance_mult and rebalance_mult > 0:
                payload["rebalance_mult"] = float(rebalance_mult)
            if rebalance_weights and rebalance_weights.strip():
                try:
                    payload["rebalance_weights"] = _parse_weights_csv(rebalance_weights)
                except ValueError as exc:
                    return f"Error: could not parse rebalance_weights {rebalance_weights!r}: {exc}."

        await emit_status(f"Generating image (model={chosen_model or 'default'})…")

        data, err = await self._post_mcpo(self.valves.generate_path, payload)
        if err is not None:
            await emit_status("Generation failed.", done=True)
            return f"Error: {err}."

        frame = _coerce_frame(data)
        if frame is None:
            log.warning("comfyless generate response had no recognizable frame: %.300s", str(data))
            await emit_status("Generation failed.", done=True)
            return "Error: image backend returned an unexpected response shape."

        resolved = frame.get("resolved_params") or {}
        notices = frame.get("notices")
        image_b64 = frame.get("image_b64")

        if not image_b64:
            # Server-side fail-soft (ADR-017): generation succeeded on disk but the
            # transport image was omitted. Surface the server's own reason (the
            # path-free ADR-015 notice contract) but never the on-disk path.
            await emit_status("Generated; no inline preview returned.", done=True)
            reason = ""
            if notices:
                reason = " " + (notices if isinstance(notices, str)
                                else "; ".join(str(n) for n in notices))
            return ("Image was generated but no inline preview was returned." + reason).strip()

        # Independent size bound at this trust boundary. The 1 MiB cap is enforced
        # server-side, but mcpo is unauthenticated on the bridge, so guard against a
        # decode bomb from a compromised/buggy upstream before allocating.
        if len(image_b64) > _B64_HARD_CEILING:
            log.warning("comfyless returned oversized image_b64 (%d bytes); rejecting", len(image_b64))
            await emit_status("Generation failed.", done=True)
            return "Error: image backend returned an oversized image."

        try:
            raw = base64.b64decode(image_b64)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to decode returned image_b64: %s", exc)
            await emit_status("Generation failed.", done=True)
            return "Error: could not decode the returned image."

        # Upload into OpenWebUI's file store under the caller's own identity.
        # ADR-017 guarantees PNG (invariant 4); pin the stored content-type and
        # extension to image/png rather than trusting a server-supplied mime.
        # Users.get_user_by_id is async on this OWUI build (aiosqlite DB layer);
        # the isawaitable guard awaits it while staying portable to builds where
        # it is synchronous. Passing the un-awaited coroutine makes the handler
        # fail with "'coroutine' object has no attribute 'email'".
        try:
            user_obj = Users.get_user_by_id(user["id"])
            if inspect.isawaitable(user_obj):
                user_obj = await user_obj
            if user_obj is None:
                await emit_status("Upload failed.", done=True)
                return "Error: could not resolve the OpenWebUI user for the upload."
            filename = f"comfyless_{_safe_token(resolved.get('seed', 'img'))}.png"
            upload = UploadFile(
                file=io.BytesIO(raw),
                filename=filename,
                headers=Headers({"content-type": "image/png"}),
            )
            # upload_file_handler is a FastAPI route handler; called directly
            # (outside request DI) every parameter with a Form(...)/Query(...)
            # default is the sentinel field object, not a value. Pass them all
            # explicitly — omitting `metadata` leaves it a Form(None) object the
            # handler then calls .get() on ("'Form' object has no attribute 'get'").
            result = upload_file_handler(
                request=__request__,
                file=upload,
                metadata=None,
                process=False,
                process_in_background=False,
                user=user_obj,
            )
            if inspect.isawaitable(result):
                result = await result
            file_id = result.id
        except Exception as exc:  # noqa: BLE001
            log.warning("uploading generated image into OpenWebUI failed: %s", exc)
            await emit_status("Upload failed.", done=True)
            return "Error: the image was generated but could not be stored in OpenWebUI."

        image_url = f"/api/v1/files/{file_id}/content"
        dims = f"{resolved.get('width', '?')}x{resolved.get('height', '?')}"
        used_model = resolved.get("model") or chosen_model or "default"
        used_seed = resolved.get("seed", "?")
        elapsed = frame.get("elapsed_seconds")
        elapsed_str = f", {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
        summary = f"model={used_model}, {dims}, seed={used_seed}{elapsed_str}"

        if __event_emitter__ is not None:
            await __event_emitter__(
                {"type": "message", "data": {"content": f"![generated image]({image_url})\n"}}
            )
            await emit_status("Done.", done=True)
            return (
                f"Image generated and displayed inline to the user ({summary}). "
                f"The image is already shown; do not attempt to describe its raw bytes."
            )

        # No event emitter wired: the image is stored but cannot be auto-rendered,
        # so do not claim it was shown.
        return (
            f"Image generated and stored ({summary}) but it could not be auto-rendered "
            f"in this chat. It is available at {image_url}."
        )

    async def list_models(self) -> str:
        """List the image-generation models available in the comfyless catalog.

        Call this when the user asks what models / checkpoints are available, or
        before generate_image if unsure which model to use.

        :return: A newline-separated list of model names with their family. Pass a
                 name verbatim as the `model` argument to generate_image.
        """
        return await self._list_catalog("/list_models", "models")

    async def list_loras(self) -> str:
        """List the LoRA adapters available in the comfyless catalog.

        :return: A newline-separated list of LoRA names with their target family.
        """
        return await self._list_catalog("/list_loras", "LoRAs")

    async def list_transformers(self) -> str:
        """List the transformer / UNet checkpoints available in the comfyless catalog.

        :return: A newline-separated list of transformer names with their family.
        """
        return await self._list_catalog("/list_transformers", "transformers")
