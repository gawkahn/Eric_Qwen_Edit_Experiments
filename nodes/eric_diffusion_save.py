# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Save Node
Save generated images as PNG with gen params + ComfyUI workflow embedded in
tEXt chunks. Filename and subdirectory path are built from a %token% template.

Template tokens:
  %model%          — model stem (fine-tune name if override loaded, else base dir name)
  %model:N%        — model stem truncated to N chars
  %date:FMT%       — date/time at save time; FMT uses MM dd yyyy HH mm ss tokens
  %sampler%        — sampler name
  %node%           — short node-type tag from GEN_METADATA (e.g. "basic-gen")
  %node:N%         — node tag truncated to N chars
  Unknown tokens   — passed through literally, no exception raised

Auto-incrementing 4-digit suffix (_0001, _0002, …) is always appended.
Output dir defaults to ComfyUI's output directory; override with output_dir input.

Author: Eric Hiss (GitHub: EricRollei)
"""

import json
import os
import re
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from PIL import Image, PngImagePlugin


# ── Template engine ──────────────────────────────────────────────────────────

# Date format token map: user-visible → strftime
_DATE_TOKEN_MAP = [
    ("yyyy", "%Y"),
    ("MM",   "%m"),
    ("dd",   "%d"),
    ("HH",   "%H"),
    ("mm",   "%M"),
    ("ss",   "%S"),
]

_TOKEN_RE = re.compile(r"%([^%]+)%")


def _apply_date_format(fmt: str, now: datetime) -> str:
    result = fmt
    for user_tok, strftime_tok in _DATE_TOKEN_MAP:
        result = result.replace(user_tok, strftime_tok)
    return now.strftime(result)


def _truncate(value: str, length: str | None) -> str:
    if length is None:
        return value
    try:
        return value[:int(length)]
    except (ValueError, TypeError):
        return value


def _resolve_token(token: str, metadata: dict, now: datetime) -> str | None:
    """Resolve a single token string (without surrounding %) to a value.

    Returns None if the token is unrecognised (caller leaves the original
    %token% text in place).
    """
    # tokens with optional :param suffix
    if ":" in token:
        name, param = token.split(":", 1)
    else:
        name, param = token, None

    name_lower = name.lower()

    if name_lower == "model":
        return _truncate(metadata.get("model_name", ""), param)

    if name_lower == "date":
        fmt = param if param else "yyyy-MM-dd"
        return _apply_date_format(fmt, now)

    if name_lower == "sampler":
        return metadata.get("sampler", "")

    if name_lower == "node":
        return _truncate(metadata.get("node_type", ""), param)

    return None  # unrecognised


def expand_template(template: str, metadata: dict) -> str:
    """Expand %token% placeholders in template using metadata.

    Unknown tokens are left as-is (with percent signs). Never raises.
    """
    now = datetime.now()

    def replace(m):
        token = m.group(1)
        try:
            resolved = _resolve_token(token, metadata, now)
        except Exception:
            resolved = None
        if resolved is None:
            return m.group(0)  # leave literal %token%
        # Sanitize: strip characters unsafe in filenames/paths
        # Allow forward slash so path separators in templates work.
        safe = re.sub(r'[\\:*?"<>|]', "_", resolved)
        return safe

    return _TOKEN_RE.sub(replace, template)


# ── Counter ──────────────────────────────────────────────────────────────────

def _next_counter(directory: str, stem: str) -> int:
    """Return the next available 4-digit counter for stem in directory."""
    existing = set()
    if os.path.isdir(directory):
        for fname in os.listdir(directory):
            if fname.startswith(stem + "_") and fname.endswith(".png"):
                suffix = fname[len(stem) + 1 : -4]
                try:
                    existing.add(int(suffix))
                except ValueError:
                    pass
    n = 1
    while n in existing:
        n += 1
    return n


# ── PNG writer ───────────────────────────────────────────────────────────────

def _build_params_text(metadata: dict) -> str:
    """Serialise gen params to a JSON string for the tEXt chunk."""
    safe = {k: v for k, v in metadata.items() if k != "pipeline"}
    return json.dumps(safe, ensure_ascii=False, indent=2)


def save_png_with_metadata(
    pil_image: Image.Image,
    filepath: str,
    metadata: dict,
    workflow_json: dict | None,
) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", _build_params_text(metadata))
    if workflow_json is not None:
        info.add_text("workflow", json.dumps(workflow_json, ensure_ascii=False))
    pil_image.save(filepath, format="PNG", pnginfo=info)


# ── Node ─────────────────────────────────────────────────────────────────────

class EricDiffusionSave:
    """
    Save a generated image as PNG with embedded gen params and ComfyUI workflow.

    Filename and directory path are built from a %token% template:
      %model%       — model stem (fine-tune name when override loaded)
      %model:N%     — truncated to N chars
      %date:FMT%    — e.g. %date:MM-dd-yyyy%  →  04-18-2026
      %sampler%     — sampler name
      %node%        — short node-type tag (e.g. basic-gen)
      %node:N%      — truncated

    A 4-digit counter (_0001, _0002, …) is always appended before .png.
    Subdirectories are created as needed. Output dir defaults to ComfyUI's
    output folder.
    """

    CATEGORY = "Eric Diffusion"
    FUNCTION = "save"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "metadata": ("GEN_METADATA", {}),
                "filename_template": ("STRING", {
                    "default": "%date:MM-dd-yyyy%/%model%/%node%",
                    "multiline": False,
                    "tooltip": (
                        "Path + filename template relative to output_dir. "
                        "Tokens: %model%, %model:N%, %date:FMT%, %sampler%, "
                        "%node%, %node:N%. A 4-digit counter is appended automatically."
                    ),
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Output directory. Leave blank to use ComfyUI's default output folder.",
                }),
            },
            "hidden": {
                "prompt":      "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save(
        self,
        image: torch.Tensor,
        metadata: dict,
        filename_template: str = "%date:MM-dd-yyyy%/%model%/%node%",
        output_dir: str = "",
        prompt=None,
        extra_pnginfo=None,
    ) -> Tuple[torch.Tensor]:

        # ── Resolve base output directory ──────────────────────────────────
        base_dir = output_dir.strip() if output_dir and output_dir.strip() else None
        if base_dir is None:
            try:
                import folder_paths
                base_dir = folder_paths.get_output_directory()
            except Exception:
                base_dir = os.path.join(os.path.dirname(__file__), "..", "output")
                base_dir = os.path.abspath(base_dir)

        # ── Expand template ────────────────────────────────────────────────
        expanded = expand_template(filename_template.strip(), metadata)
        # Split into directory portion and filename stem.
        # Everything before the last "/" is treated as a subdirectory path.
        parts = expanded.rsplit("/", 1)
        if len(parts) == 2:
            subdir, stem = parts
        else:
            subdir, stem = "", parts[0]

        save_dir = os.path.join(base_dir, subdir) if subdir else base_dir
        os.makedirs(save_dir, exist_ok=True)

        # ── Filename + counter ─────────────────────────────────────────────
        stem = stem or "image"
        counter = _next_counter(save_dir, stem)
        filename = f"{stem}_{counter:04d}.png"
        filepath = os.path.join(save_dir, filename)

        # ── Convert tensor to PIL ──────────────────────────────────────────
        # image shape: [B, H, W, C], float32 0–1
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np, mode="RGB")

        # ── Workflow JSON from hidden inputs ───────────────────────────────
        workflow_json = None
        if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
            workflow_json = extra_pnginfo.get("workflow")

        # ── Save ───────────────────────────────────────────────────────────
        try:
            save_png_with_metadata(pil_image, filepath, metadata, workflow_json)
        except Exception as e:
            raise RuntimeError(
                f"[EricDiffusionSave] Failed to save image to {filepath!r}: {e}"
            ) from e

        print(f"[EricDiffusionSave] Saved → {filepath}")
        return (image,)
