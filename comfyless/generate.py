#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""
Comfyless image generation — pure Python, no ComfyUI.

Drives the existing diffusers pipeline code (Flux.1, Flux.2/Klein,
Chroma, Qwen-Image) from a CLI or as a Python function.  Produces
output images with reproducible metadata sidecar JSON.

Two modes:

  Human (default):
    python -m comfyless.generate --model <path> --prompt "a cat" \\
        --seed 42 --output test.png

  Agent bridge (--json):
    echo '{"prompt":"a cat","model":"/path",...}' | \\
        python -m comfyless.generate --json

In --json mode, structured input is read from stdin and structured
output is written to stdout.  Human-readable progress goes to stderr.
See contracts/image_gen_bridge.md for the full schema.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

# Shims MUST be installed before any nodes.* import.
import comfyless  # noqa: F401 — triggers _install_shims()

import argparse
import inspect
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from nodes.eric_diffusion_utils import (
    QUANT_MODES,
    build_quant_config,
    detect_pipeline_class,
    detect_load_variant,
    quantize_module,
    read_guidance_embeds,
    read_model_index,
    resolve_component_class,
    resolve_override_component_class,
    detect_component_format,
    load_component,
    resolve_hf_path,
    resolve_vae_tiling,
    VAE_TILING_CHOICES,
    _is_hf_repo_id,
)
from nodes.eric_diffusion_samplers import sampler_choices, swap_sampler
from nodes.eric_qwen_edit_lora import load_lora_with_key_fix

from comfyless.family_defaults import DISTILLED_FAMILIES, FAMILY_DEFAULTS

CONTRACT_VERSION = 1
SAMPLER_NAMES = sampler_choices()
SCHEDULE_NAMES = ["linear", "balanced", "karras", "beta57", "bong_tangent"]
_ALIGN = 32  # dimension alignment for all supported models


# ════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════

def _align_dim(x: int) -> int:
    """Round down to nearest multiple of _ALIGN."""
    return (x // _ALIGN) * _ALIGN


def _log(msg: str) -> None:
    """Print to stderr (safe in --json mode)."""
    print(msg, file=sys.stderr, flush=True)


# ── Params schema (single source of truth) ───────────────────────────────

# Canonical name → (type-or-tuple-of-types, default).
#
# Defaults with value None mark REQUIRED fields (model, prompt) — the
# required-field gate at _run_cli_mode() around the --model/--prompt check
# still owns that check; the schema merely labels them.
#
# Types are checked by _validate_params via isinstance(); the schema's
# expected-type element may be a tuple of types for nullable fields. Coercion
# is out of scope (a string "4" stays a string + warning).
#
# COMFYLESS_SCHEMA and _CLI_TO_CANONICAL moved to comfyless.params_schema as
# part of the ADR-012 step-2 schema collapse — the canonical-key + canonical-
# type declarations now live next to the validator that consumes them. The
# names are re-exported here so external consumers (test_params_schema.py,
# downstream importers) keep working.
from comfyless.params_schema import COMFYLESS_SCHEMA, _CLI_TO_CANONICAL  # noqa: E402,F401
from comfyless.output_format import (  # noqa: E402
    OutputFormat,
    resolve_output_format,
)


# ── Sidecar / override helpers ───────────────────────────────────────────

# Non-schema keys written by generate() into the metadata sidecar / PNG
# chunk (timestamps, elapsed, etc).  Dropped on sidecar load so stale
# metrics don't leak into the next run.  Strictly narrower than the schema
# filter — these are known-and-intentional non-params, not "unknown to us".
_SKIP_SIDECAR_KEYS = {"timestamp", "elapsed_seconds", "contract_version",
                      "lora_warnings", "nag_warnings", "model_family",
                      # rebalance is a runtime CLI flag, not a schema param;
                      # recorded for provenance but re-pass --rebalance to replay.
                      "rebalance",
                      # ADR-030: derived provenance (output is 2× gen res), not
                      # an input param — re-pass --upscale-vae to replay instead.
                      "upscale_factor",
                      # ADR-034: output format/quality are an OUTPUT concern,
                      # recorded as provenance on jpeg runs but never a replay
                      # param — re-pass --output-format/--quality to replay.
                      "output_format", "quality"}


def _type_name(t) -> str:
    """Human-readable type name for validator warnings."""
    if isinstance(t, tuple):
        return " | ".join(_type_name(x) for x in t)
    if t is type(None):
        return "None"
    return getattr(t, "__name__", str(t))


def _validate_params(p: dict, *, source: str) -> dict:
    """Clean an input params dict against COMFYLESS_SCHEMA.

    Behavior:
      - Unknown keys → DROPPED, warning logged to stderr naming key + source.
      - Type mismatches → KEPT (no silent coercion), warning logged.
      - Missing required keys → not flagged here (the CLI's required-field
        gate owns that check; it runs after the final merge).
      - Returns a new dict; does not mutate input.

    `source` is a short human-readable string (e.g. "sidecar:/path/foo.json",
    "eric-save:foo.png", "cli-merged") that appears in every warning so the
    user can trace which layer introduced the bad key.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in p.items():
        if key not in COMFYLESS_SCHEMA:
            _log(f"[comfyless] schema: dropping unknown key {key!r} from {source}")
            continue
        expected_type, _default = COMFYLESS_SCHEMA[key]
        # isinstance() accepts a tuple natively — use the raw expected_type.
        if not isinstance(value, expected_type):
            _log(
                f"[comfyless] schema: {key!r} expected {_type_name(expected_type)}, "
                f"got {type(value).__name__} from {source}"
            )
            # KEEP the value; the user can see the warning and debug.
        cleaned[key] = value
    return cleaned


def _load_sidecar(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data = {k: v for k, v in data.items() if k not in _SKIP_SIDECAR_KEYS}
    return _validate_params(data, source=f"sidecar:{path}")


# Image extensions that carry NO embedded comfyless metadata — only PNG has a
# tEXt chunk (ADR-034 §2/D6). --params replay reads the JSON sidecar written
# beside such an image, never the image bytes.
_NON_PNG_IMAGE_EXTS = (".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")


def _load_params(path: str) -> dict:
    """Load base params from a comfyless sidecar JSON or a PNG with embedded metadata."""
    low = path.lower()
    if low.endswith(".png"):
        return _load_params_from_png(path)
    if low.endswith(_NON_PNG_IMAGE_EXTS):
        # ADR-034 D6: a .jpg/.jpeg (etc.) has no embedded provenance; JSON-parsing
        # its bytes would raise a confusing decode traceback. Direct the caller to
        # the sidecar instead of falling through to _load_sidecar.
        stem = os.path.splitext(path)[0]
        raise ValueError(
            f"{path!r} is an image with no embedded comfyless metadata "
            f"(only PNG carries a tEXt chunk). --params replays from the JSON "
            f"sidecar written beside it — pass '{stem}.json' instead."
        )
    return _load_sidecar(path)


def _extract_eric_save_params(params_json: str, path: str) -> dict:
    """Extract gen params from an Eric Diffusion Save 'parameters' tEXt chunk.

    Emits canonical schema keys only — node-internal fields and unknown keys
    are dropped by _validate_params.  model_path is explicitly renamed to
    model (the only non-canonical name Eric Diffusion Save emits that we
    care about).  LoRA weights stored in the chunk are not replayed
    (format mismatch); use --lora.
    """
    try:
        data = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"parameters chunk in {path!r} is not valid JSON: {e}")

    # Rename model_path → model BEFORE validation so the renamed key survives.
    if "model_path" in data and "model" not in data:
        data = dict(data)
        data["model"] = data.pop("model_path")

    had_loras = bool(data.get("loras"))

    # Validator drops node-internal fields (node_type, model_name, sampler_s2,
    # sampler_s3, model_path, etc.) as "unknown keys".  loras is a schema key
    # but Eric Diffusion Save stores it in an unreplayable format — warn and
    # drop before validation so the user relies on --lora explicitly.
    if "loras" in data:
        data = {k: v for k, v in data.items() if k != "loras"}

    out = _validate_params(data, source=f"eric-save:{path}")

    _log(f"[comfyless] Eric Diffusion Save parameters chunk — extracted {sorted(out.keys())}")

    if had_loras:
        print(
            "WARNING: LoRAs were active when this image was saved but will NOT be "
            "replayed.\n"
            "  Use --lora path:weight to re-apply them.",
            file=sys.stderr,
        )

    return out


def _load_params_from_png(path: str) -> dict:
    """Extract comfyless or ComfyUI params from a PNG file's tEXt chunks.

    Priority:
      1. comfyless chunk — full params from a prior comfyless run.
      2. parameters chunk — Eric Diffusion Save node format.
      3. ComfyUI prompt chunk — partial extraction; warns about missing fields.
    """
    from PIL import Image as _Image
    try:
        info = _Image.open(path).info
    except Exception as e:
        raise OSError(f"Cannot open PNG {path!r}: {e}")

    raw = info.get("comfyless")
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"comfyless chunk in {path!r} is not valid JSON: {e}")
        data = {k: v for k, v in data.items() if k not in _SKIP_SIDECAR_KEYS}
        return _validate_params(data, source=f"png:{path}")

    raw = info.get("parameters")
    if raw:
        _log(f"[comfyless] No comfyless chunk in {path!r} — trying Eric Diffusion Save parameters chunk")
        return _extract_eric_save_params(raw, path)

    raw = info.get("prompt")
    if raw:
        _log(f"[comfyless] No comfyless chunk in {path!r} — trying ComfyUI prompt chunk")
        return _extract_comfyui_params(raw)

    raise ValueError(
        f"No comfyless or ComfyUI metadata found in {path!r}. "
        "Only PNGs saved by comfyless (or ComfyUI) contain embedded params."
    )


def _extract_comfyui_params(prompt_json: str) -> dict:
    """Extract generation params from a ComfyUI prompt JSON string.

    Returns a partial params dict. Absent fields must be supplied via --override.
    Model path is never extracted (ComfyUI stores filenames, not full paths).
    """
    try:
        graph = json.loads(prompt_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"ComfyUI prompt chunk is not valid JSON: {e}")

    params: dict = {}
    by_class: dict = {}
    for node_id, node in graph.items():
        by_class.setdefault(node.get("class_type", ""), []).append((node_id, node))

    def _follow(val):
        if isinstance(val, list) and len(val) == 2:
            return graph.get(str(val[0]))
        return None

    # ── Sampler node ─────────────────────────────────────────────────
    sampler_node = None
    for ct in ("KSampler", "KSamplerAdvanced"):
        if ct in by_class:
            sampler_node = by_class[ct][0][1]
            break

    if sampler_node:
        inp = sampler_node.get("inputs", {})
        if "steps" in inp:
            params["steps"] = int(inp["steps"])
        if "cfg" in inp:
            params["cfg_scale"] = float(inp["cfg"])
        seed = inp.get("seed") if inp.get("seed") is not None else inp.get("noise_seed")
        if seed is not None:
            params["seed"] = int(seed)
        if "scheduler" in inp:
            params["schedule"] = {"karras": "karras"}.get(inp["scheduler"], "linear")
        for slot, key in (("positive", "prompt"), ("negative", "negative_prompt")):
            ref = _follow(inp.get(slot))
            if ref and ref.get("class_type") == "CLIPTextEncode":
                text = ref["inputs"].get("text", "")
                if isinstance(text, str):
                    params[key] = text
                else:
                    _log(f"[comfyless] ComfyUI: {slot} text is a graph connection — skipped")
            elif ref:
                _log(f"[comfyless] ComfyUI: {slot} node is {ref.get('class_type')!r} — skipped")
    else:
        _log("[comfyless] ComfyUI: no KSampler found — steps/cfg/seed not extracted")

    # ── Dimensions ────────────────────────────────────────────────────
    latent_ct = next(
        (ct for ct in by_class if ct.startswith("Empty") and "Latent" in ct), None
    )
    if latent_ct:
        inp = by_class[latent_ct][0][1].get("inputs", {})
        for dim in ("width", "height"):
            if dim in inp:
                params[dim] = int(inp[dim])

    # ── Model name (filename only — full path must be supplied by caller) ──
    for ct in ("CheckpointLoaderSimple", "CheckpointLoader", "DiffusionModelLoader", "UNETLoader"):
        if ct in by_class:
            inp = by_class[ct][0][1].get("inputs", {})
            ckpt = inp.get("ckpt_name") or inp.get("unet_name") or inp.get("model_name")
            if ckpt:
                _log(f"[comfyless] ComfyUI: model filename is {ckpt!r} — "
                     "use --override model=<full/path> to set the model directory")
            break

    if "model" not in params:
        _log("[comfyless] ComfyUI: model path not set — use --override model=<path>")

    return _validate_params(params, source="comfyui-prompt")


def _coerce(value: str):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ── Savepath helpers ─────────────────────────────────────────────────────

def _format_date_token(fmt: str) -> str:
    """Convert a ComfyUI-style date format string to a strftime result.

    Matching is case-insensitive for all tokens except MM (month) vs mm (minutes),
    which preserves the ComfyUI convention and avoids ambiguity.
    """
    s = fmt
    # Longer tokens first to avoid partial matches (YYYY before YY, etc.)
    s = re.sub(r"(?i)YYYY", "%Y", s)
    s = re.sub(r"(?i)YY",   "%y", s)
    s = re.sub(r"MM",       "%m", s)  # month — uppercase only (ComfyUI convention)
    s = re.sub(r"(?i)DD",   "%d", s)
    s = re.sub(r"(?i)HH",   "%H", s)
    s = re.sub(r"mm",       "%M", s)  # minutes — lowercase only (ComfyUI convention)
    s = re.sub(r"(?i)SS",   "%S", s)
    return datetime.now().strftime(s)


def _expand_iterate_tokens(template: str, iterate_inputs: Dict[str, str]) -> str:
    """Expand iteration-only tokens (%input%, %input_<param>%) in a template.

    Leaves every other token untouched — intended to be called BEFORE sending a
    savepath template to the daemon, which expands the rest (%seed%, %model%, etc.)
    server-side. iterate_inputs maps axis name → source file stem; the special
    key "_primary" is %input%'s value (the first --iterate flag's stem).

    Empty dict = no-op (no iteration active; tokens resolve to "").
    """
    def _replace(m: re.Match) -> str:
        name = m.group(1).lower()
        if name == "input":
            return iterate_inputs.get("_primary", "")
        if name.startswith("input_"):
            axis = name[len("input_"):]
            return iterate_inputs.get(axis, "")
        return m.group(0)
    return re.sub(r"%([^%]+)%", _replace, template)


def _expand_savepath_template(
    template: str,
    model_path: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    transformer_path: str = "",
    iterate_inputs: Optional[Dict[str, str]] = None,
) -> str:
    """Expand %var% and %var:spec% tokens in a savepath template string.

    %model%      — transformer filename if --transformer is set, otherwise base model.
                   Matches ComfyUI behavior: shows the weights that were actually used.
    %transformer% — always the transformer filename (or base model if none).
    %base_model% — always the base model directory name.
    %input%       — stem of the first --iterate source file (empty if no iteration).
    %input_<param>% — stem of the --iterate source file for that axis (empty if that
                    axis isn't iterated or no iteration is active).
    All token names are case-insensitive.
    """
    base_model_name = Path(model_path).name
    model_name = Path(transformer_path).name if transformer_path else base_model_name
    inputs = iterate_inputs or {}

    def _replace(m: re.Match) -> str:
        token = m.group(1)
        name, _, spec = token.partition(":")
        name = name.lower()
        if name == "date":
            return _format_date_token(spec) if spec else datetime.now().strftime("%Y-%m-%d")
        if name in ("model", "transformer"):
            n = int(spec) if spec.isdigit() else None
            return model_name[:n] if n else model_name
        if name == "base_model":
            n = int(spec) if spec.isdigit() else None
            return base_model_name[:n] if n else base_model_name
        if name == "seed":
            return str(seed)
        if name == "steps":
            return str(steps)
        if name == "cfg":
            return str(cfg_scale)
        if name == "sampler":
            return sampler
        if name == "input":
            return inputs.get("_primary", "")
        if name.startswith("input_"):
            axis = name[len("input_"):]
            return inputs.get(axis, "")
        return m.group(0)  # unknown token: leave as-is

    return re.sub(r"%([^%]+)%", _replace, template)


def _resolve_savepath(
    template: str,
    model_path: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    transformer_path: str = "",
    iterate_inputs: Optional[Dict[str, str]] = None,
    extension: str = ".png",
) -> str:
    """Expand template, create parent dirs, return first available counter slot.

    ``extension`` is the resolved output-format suffix (ADR-034); it defaults
    to ``.png`` so every existing caller is byte-for-byte unchanged.
    """
    expanded = _expand_savepath_template(
        template, model_path, seed, steps, cfg_scale, sampler, transformer_path,
        iterate_inputs=iterate_inputs,
    )
    parent = Path(expanded).parent
    parent.mkdir(parents=True, exist_ok=True)
    stem = Path(expanded).name
    counter = 1
    while True:
        candidate = parent / f"{stem}{counter:04d}{extension}"
        # The sidecar is per-stem (.json), the image per-extension — require
        # BOTH free so a .jpg run cannot reuse a stem whose .json a prior .png
        # run wrote, silently clobbering its provenance (ADR-034 slice 2 /
        # security review). Non-atomic here (savepath/in-process), matching this
        # path's existing exists()-then-write guarantee; the daemon auto-number
        # branch reserves both atomically.
        sidecar = parent / f"{stem}{counter:04d}.json"
        if not candidate.exists() and not sidecar.exists():
            return str(candidate)
        counter += 1


def _save_with_metadata(
    pil_image,
    path: str,
    metadata: dict,
    *,
    mcp_caller: bool = False,
    output_format: Optional[OutputFormat] = None,
) -> None:
    """Save a PIL image with comfyless metadata (ADR-034 format-aware).

    PNG (the default, and ``output_format is None``) embeds metadata as a
    ``tEXt`` chunk keyed ``"comfyless"`` — byte-for-byte the prior behavior.
    JPEG (and any future non-tEXt format) carries no embedded chunk; its
    provenance is the JSON sidecar the caller writes alongside, so the tEXt
    channel is simply absent (ADR-034 §2, D4).

    When mcp_caller=True (slice-1 invariant 12 / N26-N28), the embedded
    metadata is passed through the MCP redaction map first: path-typed
    fields are reduced to basenames (or HF repo IDs passed through),
    output_path / savepath are dropped entirely, non-path generation
    parameters are retained verbatim. CLI / daemon callers leave
    mcp_caller=False and the on-disk PNG embeds full paths (existing
    behavior; N29 regression guard).
    """
    if mcp_caller:
        # Lazy import keeps comfyless.mcp_server off the import path for
        # callers that never touch MCP (avoids transitively requiring the
        # mcp SDK at every generate() entry).
        from comfyless.mcp_server import redact_metadata_for_png
        metadata = redact_metadata_for_png(metadata)

    if output_format is None or output_format.embeds_text_chunk:
        # PNG path — unchanged; the tEXt chunk is the embedded metadata record.
        from PIL.PngImagePlugin import PngInfo
        pnginfo = PngInfo()
        pnginfo.add_text("comfyless", json.dumps(metadata, default=str))
        pil_image.save(path, pnginfo=pnginfo)
    else:
        # JPEG (and future non-tEXt formats): no embedded chunk. JPEG has no
        # alpha channel, so flatten modes PIL would otherwise reject.
        img = pil_image
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(path, format=output_format.pil_format, quality=output_format.quality)


def _apply_overrides(params: dict, overrides: list) -> dict:
    result = dict(params)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--override {spec!r}: expected key=value format")
        key, _, raw = spec.partition("=")
        result[key.strip()] = _coerce(raw.strip())
    return _validate_params(result, source="override")


def _explicit_override_keys(overrides: Optional[list]) -> set:
    """Canonical schema keys named by --override specs.

    Used to seed the explicit_keys tracker for the family-default
    overlay (ADR-009).  Malformed specs and unknown keys are filtered
    here — they're already reported by _apply_overrides' validation
    pass, so we silently skip them rather than double-warning.
    """
    if not overrides:
        return set()
    keys: set = set()
    for spec in overrides:
        if "=" not in spec:
            continue
        key = spec.partition("=")[0].strip()
        if key in COMFYLESS_SCHEMA:
            keys.add(key)
    return keys


def _apply_family_defaults(
    p_cur: dict,
    explicit_keys: set,
    iterated_axes: set,
    *,
    idx: Optional[int] = None,
) -> None:
    """Overlay FAMILY_DEFAULTS values onto p_cur in place.

    Reads p_cur["model"] (must be already resolve_hf_path'd), detects
    the family via detect_pipeline_class, and writes family-default
    values for keys NOT in explicit_keys and NOT in iterated_axes.

    No-op when:
      - p_cur has no "model" key,
      - detect_pipeline_class fails (missing/unreadable model_index.json),
      - the family has no entry in FAMILY_DEFAULTS,
      - every key in the family entry is already explicit or iterated.

    See ADR-009 for the full precedence ladder.
    """
    model_path = p_cur.get("model")
    if not model_path:
        return
    try:
        _, _, family = detect_pipeline_class(model_path)
    except (ValueError, OSError):
        return
    fam_defaults = FAMILY_DEFAULTS.get(family, {})
    if not fam_defaults:
        return
    applied: Dict[str, Any] = {}
    for key, value in fam_defaults.items():
        if key in explicit_keys or key in iterated_axes:
            continue
        if key not in COMFYLESS_SCHEMA:
            continue
        p_cur[key] = value
        applied[key] = value
    prefix = f"[comfyless] iter {idx}: " if idx is not None else "[comfyless] "
    if applied:
        kv = ", ".join(f"{k}={v!r}" for k, v in applied.items())
        _log(f"{prefix}family={family} defaults applied: {kv}")

    # Family is resolved from the BASE model path (ADR-009 name-hint), never
    # from --transformer. A few-step distilled schedule applied to override
    # weights that turn out to be non-distilled yields an under-denoised image
    # and NO error — the failure is silent and reads as model corruption or a
    # broken loader. Warn exactly where the hint is least trustworthy: an
    # override is present, the family is distilled, and at least one schedule
    # key was INHERITED rather than chosen. If the user set both cfg and steps
    # explicitly, they have already made the call — stay quiet.
    # "transformer_path" is the canonical schema key; "--transformer" is only
    # the CLI spelling (_CLI_TO_CANONICAL). p_cur holds canonical keys.
    if (p_cur.get("transformer_path") and family in DISTILLED_FAMILIES
            and ({"cfg_scale", "steps"} & set(applied))):
        _log(f"{prefix}WARNING: --transformer override inherited '{family}' "
             f"defaults (cfg_scale={p_cur.get('cfg_scale')!r}, "
             f"steps={p_cur.get('steps')!r}) from the BASE model path — the "
             f"override itself was not inspected. This is a few-step distilled "
             f"schedule; if the override is a non-distilled checkpoint the "
             f"image will be noisy and under-denoised. Pass --cfg/--steps "
             f"explicitly, or point --model at a non-turbo base.")


# NAG family table (ADR-023 krea + ADR-024 expansion). Value = whether
# classic CFG owns the negative prompt at cfg>0 for this family:
#   True  → NAG is gated to the cfg<=0 regime (krea/zimage conventions:
#           their pipelines run real CFG at guidance>0, which already
#           consumes the negative — NAG on top is out of scope).
#   False → always NAG-eligible (flux-family guidance EMBEDS are not CFG;
#           comfyless never routes negatives to these families at all).
# Each family's NAG machinery lives in the module named in _NAG_MODULES.
_NAG_CFG_OWNS_NEGATIVE: Dict[str, bool] = {
    "krea":         True,
    "krea-turbo":   True,
    "zimage":       True,
    "zimage-turbo": True,
    "flux":         False,
    "flux2":        False,
    "flux2klein":   False,
}

_NAG_MODULES: Dict[str, str] = {
    "krea":         "pipelines.nag_krea2",
    "krea-turbo":   "pipelines.nag_krea2",
    "flux":         "pipelines.nag_flux",
    "flux2":        "pipelines.nag_flux2",
    "flux2klein":   "pipelines.nag_flux2",
    "zimage":       "pipelines.nag_zimage",
    "zimage-turbo": "pipelines.nag_zimage",
}


def _nag_gate(model_family: str, nag_scale: Optional[float],
              cfg_scale: float = 0.0) -> tuple:
    """Decide whether NAG activates (family table + CFG-interplay rule).

    Returns (active, warning). nag_scale unset/<=1 is the documented off
    state — dormant, no warning. An unsupported family, or a cfg-gated
    family with classic CFG active, stays inactive with a loud warning
    (warn-don't-block; a silent no-op is invariant N1's failure mode).
    """
    if nag_scale is None or nag_scale <= 1.0:
        return False, None
    cfg_owns = _NAG_CFG_OWNS_NEGATIVE.get(model_family)
    if cfg_owns is None:
        return False, (
            f"--nag-scale {nag_scale} ignored — NAG is implemented for "
            f"{'/'.join(sorted(_NAG_CFG_OWNS_NEGATIVE))} only "
            f"(model_family={model_family!r}). Generation proceeds "
            f"WITHOUT negative guidance."
        )
    if cfg_owns and cfg_scale > 0:
        return False, (
            f"nag_scale {nag_scale} skipped — classic CFG is active "
            f"(cfg_scale={cfg_scale} > 0) and already consumes the "
            f"negative prompt on {model_family}. Run --cfg 0 to use NAG "
            f"(the distilled-checkpoint recommendation)."
        )
    return True, None


def _build_call_kwargs(
    pipe,
    model_family: str,
    guidance_embeds: bool,
    prompt: str,
    negative_prompt: Optional[str],
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    true_cfg_scale: Optional[float],
    max_sequence_length: int,
    generator,
) -> dict:
    """Build kwargs for pipe.__call__(), routing CFG by model family.

    Mirrors the logic in nodes/eric_diffusion_generate.py but without
    ComfyUI progress bar dependencies.
    """
    base = {
        "prompt":              prompt,
        "height":              height,
        "width":               width,
        "num_inference_steps": steps,
        "generator":           generator,
    }

    if model_family == "qwen-image":
        cfg = true_cfg_scale if true_cfg_scale is not None else cfg_scale
        kwargs = {**base, "true_cfg_scale": cfg}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        return kwargs

    if model_family == "hunyuan-image":
        # Hunyuan-Image 2.1: guidance-distilled — distilled_guidance_scale is
        # the documented call kwarg, NOT guidance_scale or true_cfg_scale
        # (ADR-025 §2). negative_prompt forwarded when set; the pipeline
        # decides whether to use it (ADR-025 §5). max_sequence_length is not
        # in this pipeline's signature, so it is not passed.
        kwargs = {**base, "distilled_guidance_scale": cfg_scale}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        return kwargs

    if model_family in ("flux", "flux2", "flux2klein", "chroma"):
        kwargs = {**base, "guidance_scale": cfg_scale}
        sig = inspect.signature(pipe.__call__)
        if "max_sequence_length" in sig.parameters:
            kwargs["max_sequence_length"] = max_sequence_length
        return kwargs

    if model_family in ("krea", "krea-turbo"):
        # Krea2Pipeline: single-pass guidance_scale. Turbo runs cfg=0.0
        # (CFG disabled); Raw runs real CFG at cfg≈3.5, which can use a
        # negative prompt — unlike Flux's distilled guidance embedding.
        # Introspect so a negative prompt / max_sequence_length is only
        # forwarded when the installed Krea2Pipeline.__call__ accepts it.
        kwargs = {**base, "guidance_scale": cfg_scale}
        sig = inspect.signature(pipe.__call__)
        if negative_prompt and "negative_prompt" in sig.parameters:
            kwargs["negative_prompt"] = negative_prompt
        if "max_sequence_length" in sig.parameters:
            kwargs["max_sequence_length"] = max_sequence_length
        return kwargs

    if model_family in ("sdxl", "sd3", "sd1", "zimage", "zimage-turbo"):
        # zimage-turbo shares Z-Image's guidance_scale routing; its
        # FAMILY_DEFAULTS cfg 1.0 runs REAL CFG at scale 1 (ZImagePipeline
        # enables CFG at guidance_scale > 0 — ADR-024 correction; a prior
        # comment here wrongly claimed single-pass collapse).
        # It MUST be listed here — the introspection fallback would route
        # true_cfg_scale (which ZImagePipeline.__call__ rejects) and drop
        # CFG entirely (ADR-009 2026-07-06).
        kwargs = {**base, "guidance_scale": cfg_scale}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        return kwargs

    if model_family == "auraflow":
        kwargs = {**base, "guidance_scale": cfg_scale}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        sig = inspect.signature(pipe.__call__)
        if "max_sequence_length" in sig.parameters:
            kwargs["max_sequence_length"] = max_sequence_length
        return kwargs

    # Unknown family — introspect and pass what fits
    _log(f"[comfyless] Unknown model_family={model_family!r} — introspecting")
    sig = inspect.signature(pipe.__call__)
    accepted = set(sig.parameters.keys())
    candidates = {
        **base,
        "negative_prompt":     negative_prompt or None,
        "guidance_scale":      cfg_scale if guidance_embeds else None,
        "true_cfg_scale":      cfg_scale if not guidance_embeds else None,
        "max_sequence_length": max_sequence_length,
    }
    return {k: v for k, v in candidates.items() if k in accepted and v is not None}


# ════════════════════════════════════════════════════════════════════════
#  Krea conditioning rebalance (ports nova452/ComfyUI-Conditioning-Rebalance)
# ════════════════════════════════════════════════════════════════════════

# Krea-2's text encoder emits a stack of Qwen3-VL layer-taps; the pipeline
# exposes them as prompt_embeds of shape (batch, seq, n_layers, dim). The
# rebalance scales each tap by a per-layer gain, then the whole tensor by a
# global multiplier — boosting detail and (per the source node) bypassing the
# safety filter's quality dilution. The preset below is the node's default,
# which boosts taps 8/9/11 (0-based 7/8/10).
KREA_REBALANCE_DEFAULT_MULT = 4.0
KREA_REBALANCE_DEFAULT_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  2.5, 5.0, 1.1, 4.0, 1.0]


def _parse_rebalance_weights(s: Optional[str]) -> Optional[List[float]]:
    """Parse comma/semicolon-separated per-layer gains. None if empty.

    Raises ValueError on non-numeric input so the CLI fails loud rather than
    silently dropping a malformed preset (matches warn-don't-block: this is a
    typo in the user's own argument, not a recoverable runtime condition).
    """
    if not s or not s.strip():
        return None
    vals = [x for x in s.replace(";", ",").split(",") if x.strip() != ""]
    try:
        return [float(x) for x in vals]
    except ValueError as e:
        raise ValueError(f"--rebalance-weights must be comma-separated floats: {e}")


def _apply_krea_rebalance(
    pipe,
    call_kwargs: dict,
    multiplier: float,
    per_layer_weights: Optional[List[float]],
    max_sequence_length: int,
    exec_device,
) -> dict:
    """Replace the positive `prompt` in call_kwargs with rebalanced embeds.

    Pre-encodes the prompt to (batch, seq, n_layers, dim), scales each layer-tap
    by its gain and the whole tensor by `multiplier`, then swaps `prompt` for
    `prompt_embeds`/`prompt_embeds_mask`. A negative_prompt (Raw with CFG) is
    left as a string for the pipeline to encode normally — only the positive
    conditioning is rebalanced, matching the source node.
    """
    prompt = call_kwargs.pop("prompt")
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt, device=exec_device, max_sequence_length=max_sequence_length,
    )
    # Krea2 stacks the layer-taps as a 4-D (batch, seq, n_layers, dim) tensor.
    # Guard so a variant returning 3-D embeds fails loud rather than silently
    # scaling the sequence axis.
    if prompt_embeds.ndim != 4:
        raise ValueError(
            f"--rebalance expects layer-tap-stacked prompt_embeds "
            f"(batch, seq, n_layers, dim); got {prompt_embeds.ndim}-D "
            f"shape {tuple(prompt_embeds.shape)}"
        )
    n_layers = prompt_embeds.shape[-2]
    if per_layer_weights is not None and len(per_layer_weights) != n_layers:
        raise ValueError(
            f"--rebalance-weights expects {n_layers} values (one per "
            f"text-encoder layer-tap for this model), got {len(per_layer_weights)}"
        )
    orig_dtype = prompt_embeds.dtype
    t = prompt_embeds.float()
    if per_layer_weights is not None:
        gains = torch.tensor(per_layer_weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(1, 1, n_layers, 1)
    t = t.to(orig_dtype) * multiplier
    call_kwargs["prompt_embeds"] = t
    call_kwargs["prompt_embeds_mask"] = prompt_embeds_mask
    return call_kwargs


# ════════════════════════════════════════════════════════════════════════
#  Pipeline loader (extracted so the server can cache the result)
# ════════════════════════════════════════════════════════════════════════

def _pin_krea_attention_backend(pipe, model_family: str) -> bool:
    """Pin the Krea2 transformer's attention backend to cuDNN.

    diffusers 0.39.0's Krea2AttnProcessor passes a bool key-padding mask
    together with enable_gqa=True (48 q heads / 12 kv heads). PyTorch's
    fused SDPA kernels reject that combination (flash: no arbitrary masks;
    mem-efficient: no GQA), so auto-select silently falls back to the MATH
    backend and materializes the full S^2 attention matrix — ~91 GB of
    transients at 2560x1440 (14912 tokens), an instant OOM. cuDNN handles
    GQA + bool mask fused (measured 0.17 GB for the same shapes), so pin it
    for the Krea transformer. Upstream bug; remove when Krea2AttnProcessor
    stops disqualifying the fused kernels. Returns True when pinned.
    """
    if model_family not in ("krea", "krea-turbo"):
        return False
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return False
    try:
        transformer.set_attention_backend("_native_cudnn")
    except Exception as e:
        _log(f"[comfyless] WARNING: could not pin Krea2 attention backend "
             f"to cuDNN ({e}) — high-resolution generation may OOM in the "
             f"SDPA math-backend fallback")
        return False
    _log("[comfyless] Krea2 attention backend pinned to cuDNN (avoids SDPA "
         "math-backend fallback: bool mask + GQA disqualify flash/efficient "
         "— full S^2 materialization OOMs at high resolution)")
    return True


def _load_pipeline(
    model_path: str,
    *,
    precision: str = "bf16",
    device: str = "cuda",
    offload_vae: bool = False,
    transformer_path: str = "",
    vae_path: str = "",
    text_encoder_path: str = "",
    text_encoder_2_path: str = "",
    vae_from_transformer: bool = False,
    attention_slicing: bool = False,
    sequential_offload: bool = False,
    vae_tiling: str = "auto",
    allow_hf_download: bool = False,
    quant: str = "none",
    quant_skip: tuple = (),
    quant_only: tuple = (),
):
    """Load, place, and configure a diffusers pipeline.

    Returns (pipe, model_family, guidance_embeds).
    Called by generate() for one-shot use and by the server to populate its cache.

    quant/quant_skip/quant_only (ADR-019 slice A): fp8 quantize-on-load over
    the role-based eligible component set. Warn-and-fall-back on unsupported
    hardware or missing torchao — the load itself never fails because of quant.
    """
    model_path = resolve_hf_path(model_path, allow_download=allow_hf_download)
    _log(f"[comfyless] Loading model: {model_path}")
    pipeline_class, class_name, model_family = detect_pipeline_class(model_path)
    _log(f"[comfyless] Detected: {class_name} (family: {model_family})")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(precision, torch.bfloat16)

    transformer_path    = transformer_path.strip()
    vae_path            = vae_path.strip()
    text_encoder_path   = text_encoder_path.strip()
    text_encoder_2_path = text_encoder_2_path.strip()
    if transformer_path:
        transformer_path    = resolve_hf_path(transformer_path,    allow_download=allow_hf_download)
    if vae_path:
        vae_path            = resolve_hf_path(vae_path,            allow_download=allow_hf_download)
    if text_encoder_path:
        text_encoder_path   = resolve_hf_path(text_encoder_path,   allow_download=allow_hf_download)
    if text_encoder_2_path:
        text_encoder_2_path = resolve_hf_path(text_encoder_2_path, allow_download=allow_hf_download)

    _has_components = any([transformer_path, vae_path, text_encoder_path,
                           text_encoder_2_path, vae_from_transformer])
    model_index = read_model_index(model_path) if _has_components else {}

    # ── Quant config resolved BEFORE component overrides (slice R3 / req 44)
    # so the transformer override can load in dequant-fp8 mode when it is in
    # the quant-eligible set: a natively-fp8 single file then dequants to a
    # clean bf16 model and the in-place torchao quantize below re-quantizes
    # it into Float8Tensor — the rep the DMR LoRA merge is proven on —
    # instead of staying ScaledFp8Linear-resident (which torchao skips,
    # making --quant a silent no-op on such files). build_quant_config is
    # pure over (model_path, args); on any fallback quant_selected is empty
    # → dequant_fp8 False → fp8-resident behavior unchanged (fail-closed).
    quant_config = None
    quant_selected: dict = {}
    if quant != "none":
        if sequential_offload:
            _log("[comfyless] WARNING: quant + sequential_offload is "
                 "untested together — proceeding with both")
        quant_config, quant_selected, _ = build_quant_config(
            model_path, quant, skip=tuple(quant_skip), only=tuple(quant_only),
            device=device, log_prefix="[comfyless]",
        )

    comp_kwargs: dict = {}
    if transformer_path:
        _log(f"[comfyless] Transformer override: {transformer_path!r}")
        cls_, cname = resolve_component_class(model_index, "transformer")
        transformer_slot = "transformer"
        if cls_ is None and model_index.get("unet"):
            cls_, cname = resolve_component_class(model_index, "unet")
            transformer_slot = "unet"
        if cls_ is None:
            raise ValueError(
                f"Transformer/UNet class '{cname}' not found in installed diffusers."
            )
        comp_kwargs[transformer_slot] = load_component(
            cls_, transformer_path, dtype,
            base_path=model_path, subfolder_hint=transformer_slot,
            pipeline_class=pipeline_class,
            # In the quant set → native-fp8 single files dequant to bf16 so
            # the in-place torchao quantize below gets plain nn.Linear.
            dequant_fp8=(transformer_slot in quant_selected),
        )
        _log(f"[comfyless] Custom {transformer_slot} loaded ({cname})")

    if vae_from_transformer and transformer_path and not vae_path:
        cls_, cname = resolve_component_class(model_index, "vae")
        if cls_ is None:
            raise ValueError(f"VAE class '{cname}' not found in diffusers.")
        comp_kwargs["vae"] = load_component(
            cls_, transformer_path, dtype,
            base_path=model_path, subfolder_hint="vae",
            pipeline_class=pipeline_class,
        )
        _log(f"[comfyless] VAE extracted from transformer checkpoint ({cname})")

    if vae_path:
        _log(f"[comfyless] VAE override: {vae_path!r}")
        base_cls, base_name = resolve_component_class(model_index, "vae")
        # Prefer the override's OWN class when it declares one — lets a
        # latent-compatible but differently-classed VAE (e.g. AutoencoderKLWan
        # onto a Qwen-latent model like Krea-2) load instead of failing the
        # base class's key-match guard at 0%.
        cls_, cname = resolve_override_component_class(
            vae_path, "vae", base_cls, base_name)
        if cls_ is None:
            raise ValueError(f"VAE class '{cname}' not found in diffusers.")
        if base_name and cname != base_name:
            _log(f"[comfyless] VAE override declares its own class {cname!r} "
                 f"(base model VAE is {base_name!r}) — instantiating {cname!r}; "
                 f"latent-space compatibility is the caller's responsibility")
        comp_kwargs["vae"] = load_component(
            cls_, vae_path, dtype,
            base_path=model_path, subfolder_hint="vae",
            pipeline_class=pipeline_class,
        )
        _log(f"[comfyless] Custom VAE loaded ({cname})")

    if text_encoder_path:
        _log(f"[comfyless] Text encoder (slot 1) override: {text_encoder_path!r}")
        cls_, cname = resolve_component_class(model_index, "text_encoder")
        if cls_ is None:
            raise ValueError(f"Text encoder class '{cname}' not found.")
        comp_kwargs["text_encoder"] = load_component(
            cls_, text_encoder_path, dtype,
            base_path=model_path, subfolder_hint="text_encoder",
            pipeline_class=pipeline_class,
        )
        _log(f"[comfyless] Custom text encoder (slot 1) loaded ({cname})")

    if text_encoder_2_path:
        _log(f"[comfyless] Text encoder (slot 2) override: {text_encoder_2_path!r}")
        cls_, cname = resolve_component_class(model_index, "text_encoder_2")
        if cls_ is None:
            raise ValueError(f"Text encoder 2 class '{cname}' not found or pipeline "
                             f"has no second text encoder.")
        comp_kwargs["text_encoder_2"] = load_component(
            cls_, text_encoder_2_path, dtype,
            base_path=model_path, subfolder_hint="text_encoder_2",
            pipeline_class=pipeline_class,
        )
        _log(f"[comfyless] Custom text encoder (slot 2) loaded ({cname})")

    load_kwargs: dict = dict(torch_dtype=dtype, local_files_only=True, **comp_kwargs)
    variant = detect_load_variant(model_path)
    if variant:
        load_kwargs["variant"] = variant
        _log(f"[comfyless] Detected weight variant: {variant}")

    # ── Quantize-on-load (ADR-019 slice A) ────────────────────────────────
    # Standard components quantize during from_pretrained (shard-by-shard,
    # low peak memory). Override components (comp_kwargs) were instantiated
    # above and bypass quantization_config — they get in-place quantize_
    # after load if their slot is in the eligible set. (quant_config /
    # quant_selected computed BEFORE the overrides — slice R3 hoist.)
    if quant_config is not None:
        # Slots passed in as pre-built modules skip from_pretrained's
        # loader, so drop them from the mapping (quantized below instead).
        for slot in comp_kwargs:
            quant_config.quant_mapping.pop(slot, None)
        if quant_config.quant_mapping:
            load_kwargs["quantization_config"] = quant_config

    pipe = pipeline_class.from_pretrained(model_path, **load_kwargs)

    for slot, component in comp_kwargs.items():
        if slot in quant_selected and hasattr(component, "parameters"):
            if quantize_module(component, quant, family=model_family,
                               log_prefix="[comfyless]"):
                _log(f"[comfyless] quant: override component {slot!r} "
                     f"quantized in place ({quant})")

    if sequential_offload:
        _log("[comfyless] Enabling sequential CPU offload")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)
        if offload_vae and hasattr(pipe, "vae"):
            pipe.vae = pipe.vae.to("cpu")
            _log("[comfyless] VAE offloaded to CPU")

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        if resolve_vae_tiling(model_family, vae_tiling):
            pipe.vae.enable_tiling()
            _log(f"[comfyless] VAE tiling enabled (vae_tiling={vae_tiling})")
        else:
            # Defensive disable in case a future diffusers default flips
            # use_tiling=True at construct time.
            if hasattr(pipe.vae, "disable_tiling"):
                pipe.vae.disable_tiling()
            _log(f"[comfyless] VAE tiling disabled (vae_tiling={vae_tiling})")

    if attention_slicing:
        # enable_attention_slicing only drives components that implement
        # set_attention_slice — i.e. UNet models (sd1/sdxl).  Modern DiT
        # transformers (Flux/Flux2/Qwen-Image/Chroma/Krea2) route attention
        # through dispatch_attention_fn (SDPA/flash), which never materializes
        # the N^2 score matrix — there is nothing to slice, and the pipeline
        # call is a silent no-op.  Detect that and tell the truth instead of
        # logging "enabled" when nothing happened.
        denoiser = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
        if denoiser is not None and hasattr(denoiser, "set_attention_slice"):
            try:
                pipe.enable_attention_slicing(slice_size="auto")
                _log("[comfyless] Attention slicing enabled")
            except Exception as e:
                _log(f"[comfyless] Attention slicing not available: {e}")
        else:
            _log("[comfyless] WARNING: --attention-slicing has NO EFFECT on this "
                 "model — its denoiser does not support attention slicing (modern "
                 "DiT transformers use flash/SDPA attention, which has no N^2 score "
                 "matrix to slice). Ignoring. For OOM relief use --quant (weights "
                 "are the driver, not attention) or --offload-vae.")

    _pin_krea_attention_backend(pipe, model_family)

    guidance_embeds = read_guidance_embeds(pipe)
    _log(f"[comfyless] Ready — family={model_family}, guidance_embeds={guidance_embeds}")
    return pipe, model_family, guidance_embeds


def _apply_loras(pipe, loras: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Apply LoRA specs ({path, weight}) onto an already-loaded pipeline.

    Returns a per-LoRA OUTCOME list — one dict per input spec:
        {"path": str, "adapter_name": str, "applied": bool,
         "reason": str | None}
    `applied` is False when the adapter loaded 0 modules (silent no-op) or
    raised; `reason` is a short operator-facing cause (may embed the path,
    so it MUST NOT cross the MCP boundary — callers on the agent surface
    map path→catalog-name instead; see ADR-015 2026-07-06).

    Shared by generate()'s one-shot path and the MCP server's cached-pipeline
    loader so both apply LoRAs identically — the MCP path previously passed a
    pre-loaded pipeline to generate(), which skips LoRA loading, so LoRAs were
    silently never applied over MCP.
    """
    outcomes: List[Dict[str, Any]] = []
    loras = loras or []
    for i, lora_spec in enumerate(loras):
        lora_path = lora_spec["path"]
        lora_weight = float(lora_spec.get("weight", 1.0))
        adapter_name = Path(lora_path).stem.replace(" ", "_").replace(".", "_")
        _log(f"[comfyless] LoRA {i+1}/{len(loras)}: "
             f"{Path(lora_path).name} (weight={lora_weight})")
        try:
            success = load_lora_with_key_fix(
                pipe, lora_path, adapter_name,
                log_prefix="[comfyless-LoRA]",
                weight=lora_weight,
            )
            if not success:
                _log(f"[comfyless] WARNING: LoRA skipped (0 modules "
                     f"applied): {lora_path}")
            outcomes.append({
                "path": lora_path, "adapter_name": adapter_name,
                "applied": bool(success),
                "reason": None if success
                else "0 modules applied (adapter not active)",
            })
        except Exception as e:
            _log(f"[comfyless] WARNING: LoRA load failed: {lora_path}: {e}")
            outcomes.append({
                "path": lora_path, "adapter_name": adapter_name,
                "applied": False, "reason": f"load error: {e}",
            })

    # Apply the user weights to PEFT-backed adapters (2026-07-17 fix: the
    # tier-1 fast path — pipe.load_lora_weights + return — never applied
    # `weight`, so every fast-path LoRA silently ran at full trained
    # strength; the mystic/mcnl noise investigation surfaced it).
    apply_adapter_weights(pipe, [
        (o["adapter_name"], float((loras[i] or {}).get("weight", 1.0)))
        for i, o in enumerate(outcomes) if o["applied"]
    ])
    return outcomes


def apply_adapter_weights(pipe, pairs) -> Optional[str]:
    """Scale PEFT-backed adapters with ONE cumulative set_adapters call.

    `pairs` is [(adapter_name, weight)] for every adapter meant to be
    ACTIVE — the full set, not a delta: diffusers' set_adapters REPLACES
    the active-adapter set, so per-adapter singleton calls would
    deactivate every earlier adapter in a multi-LoRA run. Direct-merge
    adapters (weight baked at merge time) are EXCLUDED here; Kohya
    "<name>_te" text-encoder halves ride at the parent's weight (a
    replaced active set that omits them would silently deactivate them).
    Shared by _apply_loras (CLI/MCP) and the daemon LoRA diff (slice DLW).

    Returns a warning string when scaling failed (adapters stay loaded at
    full strength — warn-don't-block), else None.
    """
    from nodes.eric_qwen_edit_lora import is_direct_merge_adapter
    peft_pairs = [(n, w) for n, w in pairs
                  if not is_direct_merge_adapter(pipe, n)]
    if not peft_pairs:
        return None
    names = [n for n, _ in peft_pairs]
    weights = [w for _, w in peft_pairs]
    discovery_warn = None
    try:
        _listed = pipe.get_list_adapters()
        _known = {n for _comp in _listed.values() for n in _comp}
    except Exception as e:  # noqa: BLE001
        # Never silent (security review DLW F10): with discovery down we
        # cannot see Kohya "_te" halves, and the replacing set_adapters
        # call below would deactivate any that exist.
        _known = set()
        discovery_warn = (f"adapter discovery failed ({e}) — Kohya "
                          f"text-encoder LoRA halves, if any, may be "
                          f"inactive")
        _log(f"[comfyless] WARNING: {discovery_warn}")
    for _n, _w in peft_pairs:
        if f"{_n}_te" in _known:
            names.append(f"{_n}_te")
            weights.append(_w)
    try:
        pipe.set_adapters(names, adapter_weights=weights)
        _log(f"[comfyless] LoRA weights applied: {dict(zip(names, weights))}")
        return discovery_warn
    except Exception as e:  # noqa: BLE001 — scaling failure must not kill
        # generation; the adapters ARE loaded.
        msg = (f"set_adapters failed — LoRA(s) {names} remain at their "
               f"previously applied scale (full trained strength if never "
               f"scaled): {e}")
        _log(f"[comfyless] WARNING: {msg}")
        return msg if not discovery_warn else f"{discovery_warn}; {msg}"


def lora_failure_warnings(outcomes: List[Dict[str, Any]]) -> List[str]:
    """Operator-facing warning strings (one per FAILED LoRA) for the on-disk
    metadata sidecar / CLI. These embed the LoRA PATH — operator-facing only;
    they are dropped from the agent surface (ADR-015 MEDIUM-1 / 2026-07-06)."""
    return [
        f"LoRA not applied ({o['reason']}): {o['path']}"
        for o in outcomes if not o.get("applied")
    ]


# Exit code returned when a LoRA silently did not apply but the image WAS
# still written (ADR-015 2026-07-06). Distinct from 1/2 (hard errors) so
# scripts can special-case it; the --iterate sweep treats it as NON-FATAL.
_LORA_SOFT_FAIL_RC = 3


def _report_lora_outcome(metadata: Dict[str, Any]) -> int:
    """CLI: print a prominent banner + return `_LORA_SOFT_FAIL_RC` (3) when
    any requested LoRA did not apply, else 0. The image was still written
    (WITHOUT the failed adapter), so exit-3 is the catchable 'not what you
    asked for' signal — distinct from 1/2 hard errors (ADR-015 2026-07-06).
    Reads the operator-facing `lora_warnings` strings that both the in-process
    path and the daemon wire-result carry in metadata."""
    warnings = metadata.get("lora_warnings") or []
    if not warnings:
        return 0
    bar = "=" * 64
    print(f"\n{bar}", file=sys.stderr)
    print(f"⚠️  {len(warnings)} LoRA(s) DID NOT APPLY — image was generated "
          f"WITHOUT them:", file=sys.stderr)
    for w in warnings:
        print(f"   • {w}", file=sys.stderr)
    print(bar, file=sys.stderr)
    return _LORA_SOFT_FAIL_RC


def _iterate_combo_disposition(rc: int) -> str:
    """Classify a per-combo `_run_one` return code for the --iterate sweep:
      'ok'    — rc 0.
      'soft'  — `_LORA_SOFT_FAIL_RC` (3): the combo's image WAS written but a
                LoRA didn't apply. Keep sweeping (a bad LoRA in one combo must
                not kill working combos); report at the end. (ADR-015
                2026-07-06 — exit-3 is distinct from hard errors, and the
                fan-out is the one place that distinction must be honored.)
      'fatal' — anything else: a real error; abort the sweep.
    """
    if rc == 0:
        return "ok"
    if rc == _LORA_SOFT_FAIL_RC:
        return "soft"
    return "fatal"


# ════════════════════════════════════════════════════════════════════════
#  Core generate function
# ════════════════════════════════════════════════════════════════════════

def _sigma_schedule_gate(pipe, schedule: str, model_family: str, steps: int):
    """Decide whether a custom sigma `schedule` can be applied to this pipe call
    (ADR-028). Returns (sigmas_list_or_None, warning_or_None).

    A non-linear schedule is honored when the pipeline's scheduler is a flow-match
    scheduler whose `set_timesteps` accepts a `sigmas=` kwarg. `--sampler` is
    ORTHOGONAL: it sets the integration rule (Euler vs Adams-Bashforth multistep)
    while `--schedule` sets the sigma spacing, and the multistep schedulers accept
    external sigmas verbatim — so the two compose (ADR-028 amendment 2026-07-13).
    Any other case with a non-linear schedule returns (None, "<reason>") so the
    caller warns-and-ignores. `linear` / unset return (None, None) silently."""
    if not schedule or schedule == "linear":
        return None, None
    from nodes.eric_diffusion_scheduler import is_flow_match
    # This gate runs PRE-swap: `pipe.scheduler` is the model's default scheduler,
    # not the multistep one swap_sampler installs around the call. That's a valid
    # proxy — every registry sampler is a FlowMatchEuler subclass whose
    # set_timesteps also accepts sigmas (pinned in test_samplers), so a default
    # that passes here means the swapped scheduler will too. (Do NOT move this
    # inside the swap context: is_flow_match is name-prefix based and the
    # FlowMultistep* subclasses don't start with "FlowMatch", so it would flip.)
    sched = getattr(pipe, "scheduler", None)
    sched_name = type(sched).__name__ if sched is not None else "None"
    # is_flow_match gates on CORRECT interpretation (classic schedulers would
    # misread flow sigmas); the set_timesteps signature gates on ACCEPTANCE — the
    # root-cause property (subsumes the FlowMatchHeun special case, future-proof
    # against a new flow-match scheduler that lacks sigmas support). Both needed.
    if not is_flow_match(sched):
        return None, f"{model_family} uses a non-flow-match scheduler ({sched_name})"
    try:
        _st_params = inspect.signature(sched.set_timesteps).parameters
    except (TypeError, ValueError):
        _st_params = {}
    if "sigmas" not in _st_params:
        return None, f"{sched_name}.set_timesteps does not accept custom sigmas"
    # Signature check is on the STOCK pipe.__call__; on the NAG path the actual
    # callable is an unbound NAG*Pipeline.__call__ — every current NAG mirror also
    # takes+forwards sigmas (test_nag pins this), so the two stay in lockstep.
    if "sigmas" not in inspect.signature(pipe.__call__).parameters:
        return None, f"{model_family}'s pipeline does not accept custom sigmas"
    from nodes.eric_qwen_image_multistage import build_sigma_schedule
    # comfyless is full-denoise txt2img: denoise=1.0 → keep == steps, so the
    # schedule only reshapes spacing across the full sigma range.
    return build_sigma_schedule(steps, 1.0, schedule=schedule), None


def _apply_sigma_schedule(call_kwargs: dict, pipe, schedule: str,
                          model_family: str, steps: int) -> Optional[str]:
    """Wire the schedule gate into the pipe call: inject `call_kwargs["sigmas"]`
    when the schedule is honored, else return the warning string (ADR-028). This is
    the testable seam over the one-line injection — the wiring gap this slice
    closed. A returned warning is BOTH printed to stderr (in-process visibility)
    AND carried in the sidecar `schedule_warnings` so a daemon/MCP client sees it
    across the wire (invariant N1; stderr alone doesn't cross that boundary)."""
    sigmas, warn = _sigma_schedule_gate(pipe, schedule, model_family, steps)
    if sigmas is not None:
        call_kwargs["sigmas"] = sigmas
    return warn


# ── ADR-030: 2× upscale-VAE decode ────────────────────────────────────
# spacepxl Wan2.1-VAE-upscale2x is a 12-channel decoder-only finetune;
# pixel_shuffle(2) turns its output into a 2×-resolution image, giving
# near-2048 output from a ~1024 (≈¼-cost) gen. Valid only on the families
# that emit Qwen-layout packed latents in the shared Wan/Qwen latent space
# (identical latents_mean/std); other families (flux/flux2) would decode
# garbage. NB: the actual Wan video pipeline is intentionally NOT here — it
# does not emit Qwen-style packed [B, seq, C*4] latents the decode helper
# expects, and its family string doesn't resolve through _FAMILY_PATTERNS.
_UPSCALE_COMPATIBLE_FAMILIES = frozenset(
    {"krea", "krea-turbo", "qwen-image"})
_UPSCALE_VAE_DEFAULT_SUBFOLDER = "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1"


def _load_upscale_vae(path: str, subfolder: str, precision: str,
                      allow_download: bool):
    """Load the AutoencoderKLWan 2× upscale VAE (kept on CPU until decode).

    The caller-supplied ``path`` is resolved through ``resolve_hf_path``
    (same trust boundary as ``--vae``) before any load. Returns the eval
    VAE on CPU; ``decode_latents_with_upscale_vae_safe`` moves it to the
    GPU per-decode and offloads it back.
    """
    import os
    import torch
    from diffusers import AutoencoderKLWan
    from nodes.eric_diffusion_utils import resolve_hf_path

    resolved = resolve_hf_path(path.strip(), allow_download=allow_download)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}.get(precision, torch.bfloat16)
    kwargs = {"torch_dtype": dtype}
    # Subfolder resolution: explicit wins; else if the path is a repo root
    # (no config.json at the top) fall back to spacepxl's standard diffusers
    # subdir, so `--upscale-vae <repo>` works with no extra flag. If the path
    # already points at the model dir (config.json present), use no subfolder.
    sub = (subfolder or "").strip()
    if not sub and not os.path.exists(os.path.join(resolved, "config.json")):
        sub = _UPSCALE_VAE_DEFAULT_SUBFOLDER
    if sub:
        # Security (ADR-030 review, HIGH): `sub` is joined onto the already
        # root-validated `resolved` and handed to from_pretrained, which reads
        # config + weights (torch.load a .bin = pickle) from there. An absolute
        # or `..`-traversing value escapes `resolved`, reopening the arbitrary-
        # directory-load hole that _check_paths closes for the path itself.
        # Confine to `resolved` via realpath (also catches symlink escapes).
        root = os.path.realpath(resolved)
        joined = os.path.realpath(os.path.join(resolved, sub))
        if os.path.isabs(sub) or not (
                joined == root or joined.startswith(root + os.sep)):
            raise ValueError(
                "upscale_vae_subfolder must be a relative subpath within the "
                f"upscale VAE directory; refusing {subfolder!r}")
        kwargs["subfolder"] = sub
    vae = AutoencoderKLWan.from_pretrained(
        resolved, local_files_only=not allow_download, **kwargs)
    vae.eval()
    return vae


def _decode_upscale_2x(packed_latents, pipe, upscale_vae, height, width,
                       device):
    """Decode packed pipeline latents to a 2× PIL image via the upscale VAE.

    Reuses the proven node helper (device pinning, transformer
    offload/restore, auto-tiling). ``packed_latents`` is what the pipe
    returns under ``output_type='latent'``.
    """
    import numpy as np
    from PIL import Image
    from nodes.eric_qwen_upscale_vae import decode_latents_with_upscale_vae_safe

    vsf = int(getattr(pipe, "vae_scale_factor", 8) or 8)
    img = decode_latents_with_upscale_vae_safe(
        packed_latents, upscale_vae, pipe, int(height), int(width),
        vae_scale_factor=vsf, device=device, log_prefix="[comfyless]",
    )  # [B, 2H, 2W, 3] float32 in [0, 1] on CPU
    arr = (img[0].clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def generate(
    model_path: str,
    prompt: str,
    output_path: str,
    *,
    negative_prompt: str = "",
    seed: int = -1,
    steps: int = 28,
    cfg_scale: float = 3.5,
    true_cfg_scale: Optional[float] = None,
    width: int = 1024,
    height: int = 1024,
    max_sequence_length: int = 512,
    sampler: str = "default",
    schedule: str = "linear",
    loras: Optional[List[Dict[str, Any]]] = None,
    precision: str = "bf16",
    device: str = "cuda",
    offload_vae: bool = False,
    transformer_path: str = "",
    vae_path: str = "",
    text_encoder_path: str = "",
    text_encoder_2_path: str = "",
    vae_from_transformer: bool = False,
    attention_slicing: bool = False,
    sequential_offload: bool = False,
    vae_tiling: str = "auto",
    # Hunyuan-Image refiner chain (ADR-016). refiner_path activates the
    # base+refiner chain (only for the hunyuan-image family); refiner_steps
    # / refiner_cfg tune the refiner stage. Sidecar-replayable via the
    # family-defaults overlay; all no-ops when refiner_path is empty.
    refiner_path: str = "",
    refiner_steps: int = 4,
    refiner_cfg: float = 3.5,
    allow_hf_download: bool = False,
    rebalance: bool = False,
    rebalance_mult: float = KREA_REBALANCE_DEFAULT_MULT,
    rebalance_weights: Optional[List[float]] = None,
    quant: str = "none",
    quant_skip: tuple = (),
    quant_only: tuple = (),
    nag_scale: float = 0.0,
    nag_tau: float = 2.5,
    nag_alpha: float = 0.25,
    nag_end: float = 1.0,
    upscale_vae_path: str = "",
    upscale_vae_subfolder: str = "",
    _cached_pipeline: Optional[Dict[str, Any]] = None,
    mcp_caller: bool = False,
    interactive_pause: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
    output_format: Optional[OutputFormat] = None,
) -> Dict[str, Any]:
    """Generate a single image and save it.

    Args:
        loras: List of {"path": str, "weight": float} dicts.  Applied
            in order.  LoRA load failures are non-fatal (warned, skipped).
        sampler: One of SAMPLER_NAMES ("default", "multistep2", "multistep3").
        schedule: Sigma-spacing schedule (one of SCHEDULE_NAMES: linear/balanced/
            karras/beta57/bong_tangent, ADR-028). Applied when the pipeline's
            scheduler is a flow-match scheduler whose set_timesteps accepts sigmas;
            warn-and-ignored otherwise. Orthogonal to (composes with) `sampler`.
        interactive_pause: arm the ^C pause/resume hook (slice PAUSE) around
            the pipeline call. The daemon passes False — its foreground-
            terminal shape (main thread + TTY stdin) defeats sigint_pause's
            implicit guards, and blocking on input() there wedges every
            client (2026-07-17).

    Returns a metadata dict suitable for the sidecar JSON / bridge output.
    Raises on fatal errors (model not found, inference failure).
    """
    # ── Validate inputs ───────────────────────────────────────────────
    model_path = resolve_hf_path(model_path, allow_download=allow_hf_download)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir = os.path.dirname(output_path) or "."
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # ── Align dimensions ──────────────────────────────────────────────
    aligned_w = _align_dim(width)
    aligned_h = _align_dim(height)
    if aligned_w != width or aligned_h != height:
        _log(f"[comfyless] Dimensions aligned to {_ALIGN}px: "
             f"{width}x{height} -> {aligned_w}x{aligned_h}")
        width, height = aligned_w, aligned_h

    # ── Resolve seed ──────────────────────────────────────────────────
    if seed < 0:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        _log(f"[comfyless] Random seed: {seed}")

    # ── Load pipeline (or reuse server-cached pipeline) ──────────────
    if _cached_pipeline is not None:
        pipe           = _cached_pipeline["pipeline"]
        model_family   = _cached_pipeline["model_family"]
        guidance_embeds = _cached_pipeline["guidance_embeds"]
        _log(f"[comfyless] Reusing cached pipeline (family: {model_family})")
    else:
        pipe, model_family, guidance_embeds = _load_pipeline(
            model_path, precision=precision, device=device, offload_vae=offload_vae,
            transformer_path=transformer_path, vae_path=vae_path,
            text_encoder_path=text_encoder_path, text_encoder_2_path=text_encoder_2_path,
            vae_from_transformer=vae_from_transformer, attention_slicing=attention_slicing,
            sequential_offload=sequential_offload, vae_tiling=vae_tiling,
            allow_hf_download=allow_hf_download,
            quant=quant, quant_skip=quant_skip, quant_only=quant_only,
        )

    # ── ADR-030: 2× upscale-VAE gate + load ───────────────────────────
    # When --upscale-vae is set, generation runs at the requested (gen)
    # resolution and the FINAL decode goes through the Wan 2× upscale VAE,
    # producing a 2× PNG. Gate to Qwen/Wan-latent families (shared latent
    # space); other families would decode garbage. The daemon pre-loads
    # and hands the VAE in via the cached dict; otherwise load inline.
    upscale_active = bool(upscale_vae_path) or (
        _cached_pipeline is not None
        and _cached_pipeline.get("upscale_vae") is not None)
    upscale_vae = None
    if upscale_active:
        if model_family not in _UPSCALE_COMPATIBLE_FAMILIES:
            raise ValueError(
                f"--upscale-vae is only supported for Qwen/Wan-latent "
                f"families {sorted(_UPSCALE_COMPATIBLE_FAMILIES)}; --model "
                f"resolved to family {model_family!r}. The upscale VAE shares "
                f"the Qwen/Wan latent space and would decode garbage on other "
                f"families."
            )
        if (_cached_pipeline is not None
                and _cached_pipeline.get("upscale_vae") is not None):
            upscale_vae = _cached_pipeline["upscale_vae"]
            _log("[comfyless] Reusing cached upscale VAE")
        else:
            upscale_vae = _load_upscale_vae(
                upscale_vae_path, upscale_vae_subfolder,
                precision, allow_hf_download)
            _log(f"[comfyless] Upscale VAE loaded ({upscale_vae_path!r})")

    # ── Hunyuan-Image refiner gate + load (ADR-016) ───────────────────
    # The chain activates when family is hunyuan-image AND refiner_path
    # is non-empty. Three other cases are handled here too:
    #   - refiner_path set on a non-hunyuan family → clean error
    #     (Vision Inv 10 negative; failure-semantics §5)
    #   - hunyuan-image + refiner_path unset → loud stderr warning +
    #     base-only run (Vision Inv 2; failure-semantics §1)
    #   - any other family with refiner_path unset → no-op
    # Empty string is the unset state per ADR-016 §(c) — sidecar replay
    # of an empty string lands here as falsy.
    refiner_pipe = None
    if refiner_path:
        if model_family != "hunyuan-image":
            raise ValueError(
                f"--refiner is only supported for the hunyuan-image family; "
                f"--model resolved to family {model_family!r}. Drop --refiner "
                f"or point --model at a HunyuanImage-2.1-Diffusers checkpoint."
            )
        from comfyless import hunyuan_chain
        # Daemon path may pre-load both base and refiner (server cache);
        # accept a pre-loaded refiner from the cache when the server
        # provides one, otherwise load fresh.
        if _cached_pipeline is not None and _cached_pipeline.get("refiner_pipeline") is not None:
            refiner_pipe = _cached_pipeline["refiner_pipeline"]
            _log("[comfyless] Reusing cached refiner pipeline")
        else:
            refiner_pipe = hunyuan_chain.load_refiner_pipeline(
                refiner_path, base_pipe=pipe,
                precision=precision, device=device,
                vae_tiling=vae_tiling,
                allow_hf_download=allow_hf_download,
                quant=quant, quant_skip=quant_skip, quant_only=quant_only,
            )
    elif model_family == "hunyuan-image":
        # Warn-don't-block per `feedback_warn_dont_block` + Vision Inv 2.
        # The exact warning text is locked at runtime by test_hunyuan.py
        # Inv 2; changing it requires a paired test edit.
        print(
            "WARNING: hunyuan-image quality requires a refiner; pass "
            "--refiner <path>; download with huggingface-cli download "
            "hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers",
            file=sys.stderr,
        )

    # ── Load LoRAs ────────────────────────────────────────────────────
    lora_outcomes: List[Dict[str, Any]] = []
    loras = loras or []
    # When a cached pipeline is provided the caller has already applied LoRAs
    # via _apply_loras (the MCP cached loader does this); skip re-applying but
    # keep the list so it appears correctly in metadata.
    if _cached_pipeline is None:
        lora_outcomes = _apply_loras(pipe, loras)

    # ── Build generator ───────────────────────────────────────────────
    exec_device = getattr(pipe, "_execution_device", None) or device
    generator = torch.Generator(device=exec_device).manual_seed(seed)

    # ── Build call kwargs (CFG routing) ───────────────────────────────
    neg = negative_prompt.strip() or None
    call_kwargs = _build_call_kwargs(
        pipe, model_family, guidance_embeds,
        prompt, neg, height, width, steps, cfg_scale,
        true_cfg_scale, max_sequence_length, generator,
    )

    # ── NAG negative guidance (ADR-023; krea family only) ────────────
    # Every skip/oddity lands in nag_warnings AND stderr: generate() may run
    # inside the daemon or MCP server, where this stderr is a server log the
    # caller never sees — the metadata list is what crosses that boundary
    # (invariant N1; the lora_warnings precedent). Recorded in the sidecar,
    # printed client-side from the wire metadata, surfaced as MCP notices.
    nag_warnings: List[str] = []
    # The gate owns the whole family table AND the CFG-interplay rule
    # (ADR-024): cfg-gated families (krea/zimage — their pipelines run real
    # CFG at cfg>0) skip loudly; flux-family guidance embeds are not CFG,
    # so those are always eligible. The per-pipeline mirrors carry the same
    # guards for standalone users — gating here makes every skip visible
    # across the daemon/MCP boundary.
    nag_active, nag_warning = _nag_gate(model_family, nag_scale, cfg_scale)
    if nag_warning:
        nag_warnings.append(nag_warning)
    if nag_active:
        try:
            import importlib
            nag_pipe_call = importlib.import_module(
                _NAG_MODULES[model_family]).nag_pipe_call
        except Exception as e:
            nag_warnings.append(
                f"nag_scale {nag_scale} ignored — NAG module for "
                f"{model_family} unavailable ({e}). Generation proceeds "
                f"WITHOUT negative guidance."
            )
            nag_active = False
    if nag_active:
        if not neg:
            nag_warnings.append(
                "NAG active with an EMPTY negative prompt — guidance runs "
                "against the empty prompt, which is rarely what you want. "
                "Pass --negative-prompt."
            )
        # Range sanity (warn-don't-block): a negative tau zeroes the guided
        # term via min(ratio, tau)/ratio; alpha outside [0,1] extrapolates
        # the blend; nag_end outside [0,1] is a window no-op or over-run.
        if nag_tau <= 0:
            nag_warnings.append(
                f"nag_tau {nag_tau} <= 0 zeroes the guided term — NAG will "
                f"suppress the positive signal, not the negative. Use > 0 "
                f"(default 2.5).")
        if not (0.0 <= nag_alpha <= 1.0):
            nag_warnings.append(
                f"nag_alpha {nag_alpha} outside [0, 1] extrapolates the "
                f"blend (default 0.25).")
        if not (0.0 <= nag_end <= 1.0):
            nag_warnings.append(
                f"nag_end {nag_end} outside [0, 1] — it is a fraction of "
                f"steps (default 1.0).")
        call_kwargs.update({
            "nag_scale": nag_scale,
            "nag_tau":   nag_tau,
            "nag_alpha": nag_alpha,
            "nag_end":   nag_end,
        })
        # The family CFG-routing branches only forward negative_prompt when
        # the stock pipeline consumes it (flux-family: never); NAG consumes
        # it regardless, so re-attach unconditionally.
        if neg:
            call_kwargs["negative_prompt"] = neg
        _log(f"[comfyless] NAG active: scale={nag_scale}, tau={nag_tau}, "
             f"alpha={nag_alpha}, end={nag_end}")
    for _w in nag_warnings:
        print(f"[comfyless] WARNING: NAG — {_w}", file=sys.stderr)

    # ── Krea conditioning rebalance (optional) ────────────────────────
    if rebalance and model_family in ("krea", "krea-turbo"):
        weights = rebalance_weights if rebalance_weights is not None \
            else KREA_REBALANCE_DEFAULT_WEIGHTS
        _log(f"[comfyless] Krea rebalance: mult={rebalance_mult}, weights={weights}")
        call_kwargs = _apply_krea_rebalance(
            pipe, call_kwargs, rebalance_mult, weights,
            max_sequence_length, exec_device,
        )
    elif rebalance:
        print(f"[comfyless] WARNING: --rebalance ignored — only applies to "
              f"krea/krea-turbo (model_family={model_family!r})", file=sys.stderr)

    _log(f"[comfyless] Generating: {width}x{height}, "
         f"steps={steps}, cfg={cfg_scale}, seed={seed}, sampler={sampler}")

    # ── VAE: move back to GPU for decode ──────────────────────────────
    if offload_vae and hasattr(pipe, "vae"):
        _denoiser = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if _denoiser is not None:
            pipe.vae = pipe.vae.to(next(_denoiser.parameters()).device)

    # ── Sampler guard: flow-match samplers require FlowMatch schedulers ──
    # SDXL/SD1 use DDPM-style schedulers that lack init_noise_sigma.
    # Config-driven runs may specify a sampler chosen for a different
    # model family — fall back to "default" rather than crash silently.
    effective_sampler = sampler
    if model_family in ("sdxl", "sd1") and sampler != "default":
        print(
            f"[comfyless] WARNING: sampler={sampler!r} requires a flow-match "
            f"scheduler but {model_family} uses a DDPM-style scheduler "
            f"(init_noise_sigma). Falling back to default (Euler). "
            f"Set sampler=default in your config for {model_family} runs."
        )
        effective_sampler = "default"

    # ── Sigma schedule (ADR-028): reshape flow-match sigma spacing ────
    # `schedule` was long recorded-but-ignored; apply it here via the node
    # path's build_sigma_schedule, gated to flow-match schedulers that accept
    # sigmas. Orthogonal to --sampler (spacing vs integration order — they
    # compose). Warn-and-ignore everywhere else — the warning is carried in
    # metadata (schedule_warnings) so the daemon/MCP client sees it.
    _sched_warn = _apply_sigma_schedule(
        call_kwargs, pipe, schedule, model_family, steps)
    schedule_warnings = [_sched_warn] if _sched_warn else []
    if "sigmas" in call_kwargs:
        _log(f"[comfyless] sigma schedule: {schedule} (flow-match, {steps} steps)")
    elif _sched_warn is not None:
        print(f"[comfyless] WARNING: --schedule {schedule!r} ignored — "
              f"{_sched_warn}; using the pipeline default", file=sys.stderr)

    # ── Inference (with optional sampler swap) ────────────────────────
    # When the Hunyuan-Image refiner chain is active, run_chain handles both
    # pipeline calls under a single swap_sampler context. The swap is per-pipe
    # (base only), so the refiner's scheduler is untouched — pinned per
    # ADR-016 §(g) / Vision Inv 8. hunyuan-image is not a NAG family, so the
    # refiner and NAG paths are mutually exclusive.
    t0 = time.monotonic()
    if refiner_pipe is not None:
        from comfyless import hunyuan_chain
        with swap_sampler(pipe, effective_sampler, log_prefix="[comfyless]"):
            final_pil = hunyuan_chain.run_chain(
                pipe, refiner_pipe, call_kwargs,
                prompt=prompt, negative_prompt=neg,
                refiner_steps=refiner_steps, refiner_cfg=refiner_cfg,
                generator=generator,
            )
    else:
        if upscale_active:
            # ADR-030: generate at the requested (gen) resolution and emit
            # packed latents; the Wan 2× upscale VAE decodes them to a 2×
            # image below (output PNG is 2× width × 2× height).
            call_kwargs["output_type"] = "latent"
        # Slice PAUSE: first ^C pauses at the next step boundary, second
        # aborts (docs/vision/slice-pause-sigint.md). No-op off the
        # interactive CLI (non-TTY/thread) and on pipelines without
        # callback_on_step_end; the daemon opts out explicitly via
        # interactive_pause=False. The stock pipe.__call__ signature stands
        # proxy for the NAG wrappers (all four accept the callback).
        from comfyless.pause import sigint_pause
        with swap_sampler(pipe, effective_sampler, log_prefix="[comfyless]"):
            if nag_active:
                # Unbound Krea2NAGPipeline.__call__ on the (possibly cached)
                # stock pipeline: NAG processors are installed per-call and
                # restored in a finally, so the cached object's class and
                # shape never change (cache keys stay NAG-free by design).
                with sigint_pause(pipe.__call__, call_kwargs,
                                  enabled=interactive_pause):
                    result = nag_pipe_call(pipe, **call_kwargs)
            else:
                with sigint_pause(pipe.__call__, call_kwargs,
                                  enabled=interactive_pause):
                    result = pipe(**call_kwargs)
        if upscale_active:
            final_pil = _decode_upscale_2x(
                result.images, pipe, upscale_vae, height, width, device)
        else:
            final_pil = result.images[0]
    elapsed = time.monotonic() - t0
    _log(f"[comfyless] Generated in {elapsed:.1f}s")

    # ── Build metadata (before save so it can be embedded in the PNG) ──
    metadata: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model_path,
        "model_family": model_family,
        "transformer_path":    transformer_path,
        "vae_path":            vae_path,
        "text_encoder_path":   text_encoder_path,
        "text_encoder_2_path": text_encoder_2_path,
        "vae_from_transformer": vae_from_transformer,
        "loras": [{"path": l["path"], "weight": float(l.get("weight", 1.0))}
                  for l in loras],
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "true_cfg_scale": true_cfg_scale,
        "width": width,
        "height": height,
        "sampler": sampler,
        "schedule": schedule,
        # Loud-across-the-wire: a --schedule that was warn-and-ignored (wrong
        # family/scheduler) records WHY here, so a daemon/MCP client sees
        # it — the daemon's stderr never reaches the client (invariant N1; mirrors
        # nag_warnings / lora_warnings). Empty on a clean apply or plain linear.
        "schedule_warnings": schedule_warnings,
        # Quantize-on-load triple — sidecar-replayable (2026-07-08): quant
        # affects output correctness for some transformer/LoRA combos, so a
        # --params replay must reproduce it.
        "quant": quant or "none",
        "quant_skip": list(quant_skip or ()),
        "quant_only": list(quant_only or ()),
        # NAG quadruple (ADR-023) — sidecar-replayable like quant; NAG
        # changes output content, so a --params replay must reproduce it.
        "nag_scale": nag_scale,
        "nag_tau": nag_tau,
        "nag_alpha": nag_alpha,
        "nag_end": nag_end,
        # ADR-030: 2× upscale-VAE decode. width/height above are the GEN
        # (pre-upscale) resolution; when active the saved PNG is 2× each.
        # Canonical schema keys → sidecar-replayable: re-pass --upscale-vae /
        # --upscale-vae-subfolder to reproduce. upscale_factor is derived
        # provenance (not an input) → excluded from replay via _SKIP_SIDECAR_KEYS.
        "upscale_vae_path": upscale_vae_path,
        "upscale_vae_subfolder": upscale_vae_subfolder,
        "upscale_factor": 2 if upscale_active else 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "contract_version": CONTRACT_VERSION,
    }
    # Caller-supplied provenance (e.g. inline prompt-enhancement: original
    # prompt + backend/recipe, ADR-026 §7). Non-schema keys; recorded in the
    # sidecar but not part of COMFYLESS_SCHEMA.
    if extra_metadata:
        metadata.update(extra_metadata)
    lora_warnings = lora_failure_warnings(lora_outcomes)
    if lora_warnings:
        metadata["lora_warnings"] = lora_warnings
    # NAG skip/oddity warnings ride the metadata across the daemon/MCP
    # boundary (invariant N1) — path-free by construction (family names,
    # scales, exception text from a local import only).
    if nag_warnings:
        metadata["nag_warnings"] = nag_warnings
    if rebalance and model_family in ("krea", "krea-turbo"):
        metadata["rebalance"] = {
            "mult": rebalance_mult,
            "weights": rebalance_weights if rebalance_weights is not None
                       else KREA_REBALANCE_DEFAULT_WEIGHTS,
        }
    if refiner_pipe is not None:
        # Two-stage metadata extension per ADR-016 §(h). The four keys are
        # absent (not present-and-empty) on base-only runs — Vision Inv 4.
        # Sidecar replay of a pre-refiner image carries no `pipeline` key →
        # the base-only branch reactivates correctly.
        metadata["pipeline"]      = "base+refiner"
        metadata["refiner_path"]  = refiner_path
        metadata["refiner_steps"] = refiner_steps
        metadata["refiner_cfg"]   = refiner_cfg

    # Output-format provenance (ADR-034): recorded on non-png runs only, so png
    # sidecars are unchanged. These are non-schema keys (in _SKIP_SIDECAR_KEYS),
    # so --params replay filters them — they never become generation inputs.
    # quality is the 0.0-1.0 fraction (the --quality knob), unrecoverable from
    # the output file; format is inferable from the file but recorded for a
    # complete record.
    if output_format is not None and output_format.name != "png":
        metadata["output_format"] = output_format.name
        metadata["quality"] = output_format.quality_fraction

    # ── Save image with embedded metadata (PNG tEXt only) ─────────────
    pil_image = final_pil
    _save_with_metadata(pil_image, output_path, metadata, mcp_caller=mcp_caller,
                        output_format=output_format)
    _log(f"[comfyless] Saved: {output_path}")

    # ── Clean up VAE ──────────────────────────────────────────────────
    if offload_vae and hasattr(pipe, "vae"):
        pipe.vae = pipe.vae.to("cpu")

    return metadata


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="comfyless.generate",
        description="Generate images without ComfyUI.",
    )
    p.add_argument("--json", action="store_true",
                   help="Agent bridge mode: JSON stdin/stdout")
    p.add_argument("--params", type=str, default=None,
                   metavar="FILE",
                   help="Load base params from a comfyless sidecar JSON or a PNG "
                        "with embedded comfyless/ComfyUI metadata. "
                        "Use --override key=value to patch individual fields.")
    p.add_argument("--override", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Override a param from --params (repeatable). "
                        "E.g. --override model=/path/sdxl --override cfg_scale=8")
    p.add_argument("--model", nargs="+", default=None,
                   help="Path to diffusers model directory. "
                        "Stable Cascade special form: "
                        "'--model stablecascade <config.json> [config2.json] ...' "
                        "— see docs/comfyless-stable-cascade.md.")
    p.add_argument("--prompt", type=str, default=None,
                   help="Generation prompt")
    p.add_argument("--negative-prompt", type=str, default=None,
                   help="Negative prompt (qwen-image CFG; models with "
                        "classic CFG at cfg>0; krea-turbo via --nag-scale)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (-1 for random)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cfg", type=float, default=None, help="CFG scale")
    p.add_argument("--true-cfg", type=float, default=None,
                   help="True CFG scale override (qwen-image)")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--lora", action="append", default=[],
                   metavar="PATH:WEIGHT",
                   help="LoRA to apply (repeatable).  Format: path or path:weight")
    p.add_argument("--sampler", choices=SAMPLER_NAMES, default=None,
                   help="Sampler algorithm")
    p.add_argument("--schedule", choices=SCHEDULE_NAMES, default=None,
                   help="Sigma-spacing schedule (ADR-028): linear (uniform), "
                        "balanced (Karras ρ=3), karras (Karras ρ=7, steps toward "
                        "fine detail), beta57 (RES4LYF beta α=0.5 β=0.7), "
                        "bong_tangent (RES4LYF two-stage arctan). Applied when the "
                        "model uses a flow-match scheduler (Flux/Qwen/Chroma/Krea/"
                        "Z-Image/Flux.2); warn-and-ignored for classic schedulers "
                        "(SDXL/SD1). Composes with --sampler (spacing vs "
                        "integration order).")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="Max sequence length for text encoder")
    p.add_argument("--transformer", type=str, default=None, metavar="PATH",
                   help="Custom transformer/UNet weights (dir, subdir, or .safetensors)")
    p.add_argument("--vae", type=str, default=None, metavar="PATH",
                   help="Custom VAE weights")
    p.add_argument("--upscale-vae", type=str, default=None, metavar="PATH",
                   help="Wan 2× upscale VAE (spacepxl/Wan2.1-VAE-upscale2x, "
                        "ADR-030). When set, --width/--height are the GENERATION "
                        "resolution and the saved PNG is 2× each dimension — a "
                        "clean 1024 gen decoded to a 2048 image at ~¼ the 2048 "
                        "gen cost. Qwen/Wan-latent families only "
                        "(krea/qwen-image/wan); errors on others.")
    p.add_argument("--upscale-vae-subfolder", type=str, default=None,
                   metavar="SUBDIR",
                   help="Subfolder within --upscale-vae holding the diffusers "
                        "config+weights. Default: spacepxl's standard subdir when "
                        "the path is a repo root; auto-skipped when the path is "
                        "the model dir itself.")
    p.add_argument("--te1", type=str, default=None, metavar="PATH",
                   help="Custom text encoder slot 1 (CLIP-L for Flux; Qwen2.5-VL for Qwen)")
    p.add_argument("--te2", type=str, default=None, metavar="PATH",
                   help="Custom text encoder slot 2 (T5-XXL for Flux/Chroma)")
    p.add_argument("--refiner", type=str, default=None, metavar="PATH",
                   help="Hunyuan-Image 2.1 refiner pipeline path (opt-in two-stage "
                        "chained generation). When set on a hunyuan-image --model, "
                        "the base output is passed through "
                        "HunyuanImageRefinerPipeline for the documented quality "
                        "pass (Tencent README, ADR-016). Unset on hunyuan-image: "
                        "loud stderr warning + base-only run. Unset on other "
                        "families: no-op. Set on a non-hunyuan family: clean "
                        "error. Download with: huggingface-cli download "
                        "hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers")
    p.add_argument("--vae-from-transformer", action="store_true", default=None,
                   help="Extract VAE from the --transformer AIO checkpoint")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    # Quant triple defaults are None SENTINELS (not "none"/[]) so the
    # schema merge can tell "flag not given" from "user said none": a
    # sidecar's quant survives --params replay unless the CLI explicitly
    # overrides it (sidecar-replayable since 2026-07-08; resolved to the
    # schema defaults after the merge).
    p.add_argument("--quant", choices=list(QUANT_MODES), default=None,
                   help="Quantize-on-load (ADR-019): fp8 halves VRAM on the "
                        "transformer + large text encoders (VAE/CLIP never "
                        "quantized). Needs compute capability >= 8.9; falls "
                        "back to bf16 with a warning otherwise. Recorded in "
                        "the sidecar and replayed by --params.")
    p.add_argument("--quant-skip", action="append", default=None, metavar="COMPONENT",
                   help="Exclude a component slot (e.g. text_encoder) from "
                        "quantization. Repeatable. For isolating quality "
                        "regressions to one component.")
    p.add_argument("--quant-only", action="append", default=None, metavar="COMPONENT",
                   help="Quantize exactly these component slots, overriding "
                        "the default eligible set. Repeatable. VAE is refused "
                        "even here.")
    # NAG quadruple defaults are None SENTINELS (quant precedent): a
    # sidecar's NAG params survive --params replay unless the CLI
    # explicitly overrides them; resolved to schema defaults post-merge.
    p.add_argument("--nag-scale", type=float, default=None,
                   help="Normalized Attention Guidance scale (ADR-023/024). "
                        ">1 activates NAG on krea/flux/flux2/flux2klein/"
                        "zimage families, making --negative-prompt work "
                        "where CFG is dead (guidance-distilled and cfg-0 "
                        "checkpoints). krea/zimage need --cfg 0 (at cfg>0 "
                        "classic CFG owns the negative). Try 4-5. Costs "
                        "~2x wall time on the NAG'd steps. Recorded in "
                        "the sidecar and replayed by --params.")
    p.add_argument("--nag-tau", type=float, default=None,
                   help="[--nag-scale] Norm-growth clip tau (default 2.5).")
    p.add_argument("--nag-alpha", type=float, default=None,
                   help="[--nag-scale] Blend alpha (default 0.25).")
    p.add_argument("--nag-end", type=float, default=None,
                   help="[--nag-scale] Fraction of steps NAG applies to "
                        "(default 1.0 = full window; 0.5-0.75 trades "
                        "guidance strength for speed).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--offload-vae", action="store_true")
    p.add_argument("--attention-slicing", action="store_true",
                   help="Trade speed for lower peak VRAM")
    p.add_argument("--sequential-offload", action="store_true",
                   help="Extreme VRAM savings via sequential CPU offload — very slow")
    p.add_argument("--vae-tiling", choices=list(VAE_TILING_CHOICES), default="auto",
                   help="VAE tiling policy at decode time. 'auto' (default) is "
                        "family-aware: off for Hunyuan-Image (32× VAE, tiling adds "
                        "seam artifacts without memory benefit), on for every other "
                        "family (preserves prior behavior). 'on'/'off' force the "
                        "choice regardless of family.")
    # ── Krea conditioning rebalance (ports ComfyUI-Conditioning-Rebalance) ──
    p.add_argument("--rebalance", action="store_true",
                   help="Krea only: rebalance Qwen3-VL conditioning layer-taps to "
                        "boost detail / bypass the safety filter's quality dilution. "
                        "Requires the in-process path (use --output, not --savepath).")
    p.add_argument("--rebalance-mult", type=float, default=KREA_REBALANCE_DEFAULT_MULT,
                   help="[--rebalance] Global conditioning multiplier (default 4.0).")
    p.add_argument("--rebalance-weights", type=str, default=None, metavar="W1,...",
                   help="[--rebalance] Comma-separated per-layer-tap gains (12 for "
                        "Krea). Default preset: 1,1,1,1,1,1,1,2.5,5,1.1,4,1.")
    p.add_argument("--allow-hf-download", action="store_true", default=False,
                   help="Allow downloading models from HuggingFace if not in local cache. "
                        "By default only the local cache is used (no network access)")
    p.add_argument("--enhance-prompt", type=str, default=None, metavar="BACKEND",
                   help="Enhance the prompt through an LLM before generating (ADR-026). "
                        "BACKEND is a name from the enhancer registry (see "
                        "enhancers.example.toml): e.g. 'hunyuan' (local Tencent reprompt) "
                        "or an openai-endpoint name. Enhanced once per unique prompt; the "
                        "enhanced text is what gets generated + recorded in the sidecar. "
                        "For offline batch enhancement of a prompt-list JSON use "
                        "'python -m comfyless.enhance'.")
    p.add_argument("--enhance-recipe", type=str, default=None, metavar="RECIPE",
                   help="Recipe name for openai-endpoint enhancement (generic / "
                        "preserve-subject / vary-setting / <family>-generic). Ignored by "
                        "the hunyuan backend. Default: the 'generic' recipe.")
    p.add_argument("--enhance-config", type=str, default=None, metavar="PATH",
                   help="Enhancer registry TOML path (default: $COMFYLESS_ENHANCERS → "
                        "./enhancers.toml → ~/.config/comfyless/enhancers.toml).")
    p.add_argument("--output", "-o", type=str, default="/tmp/comfyless.png",
                   help="Output image path (exact; overwrites). "
                        "Ignored when a server is running — use --savepath instead.")
    p.add_argument("--savepath", type=str, default=None,
                   metavar="TEMPLATE",
                   help="Output path template with %%date:MM-dd-YY%%, %%model:12%%, "
                        "%%seed%%, %%steps%%, %%cfg%%, %%sampler%%, %%input%%. "
                        "Auto-creates dirs; always writes comfyless0001.png, 0002, ...")
    # ── Output format (ADR-034) ──
    p.add_argument("--output-format", choices=["png", "jpeg", "jpg"], default=None,
                   help="Output image format (jpg is an alias for jpeg). Default: "
                        "png, or inferred from the --output extension "
                        "(.jpg/.jpeg -> jpeg). An explicit value that contradicts "
                        "the extension is an error, not a rewrite. JPEG runs "
                        "in-process (daemon support: ADR-034 slice 2).")
    p.add_argument("--quality", type=float, default=None, metavar="0.0-1.0",
                   help="JPEG quality as a 0.0-1.0 fraction (default 0.7 -> PIL 70). "
                        "Higher is better; the useful ceiling is 1.0 -> 95. "
                        "Ignored (with a notice) for png output.")
    # ── Iteration mode (see docs/decisions/ADR-008-comfyless-iterate.md) ──
    p.add_argument("--iterate", nargs=2, action="append", default=[],
                   metavar=("PARAM", "FILE"),
                   help="Iterate PARAM (e.g. prompt, seed, cfg_scale, lora) through a "
                        "JSON list FILE. Repeatable for Cartesian product. See ADR-008.")
    p.add_argument("--max-iterations", type=int, default=500,
                   metavar="N",
                   help="Hard cap on total generations per --iterate invocation "
                        "(default 500). Exceeds this and the run fails fast.")
    # --limit (flat first-N of the Cartesian) and --limit-per (N per group axis)
    # are two different truncation models — mutually exclusive.
    _limit_group = p.add_mutually_exclusive_group()
    _limit_group.add_argument("--limit", type=_positive_int, default=None,
                   metavar="N",
                   help="After --iterate Cartesian expansion, take only the first N "
                        "combinations. Ceiling, not requirement: if Cartesian total < N, "
                        "run them all (no error). Distinct from --max-iterations: --limit "
                        "is silent truncation by design. See also --limit-per.")
    _limit_group.add_argument("--limit-per", nargs=2, default=None,
                   metavar=("AXIS", "N"),
                   help="Per-group cap: run N generations for EACH value of the "
                        "--iterate axis AXIS, cycling the OTHER axes. E.g. "
                        "'--iterate transformer_path t.json --iterate prompt p.json "
                        "--limit-per transformer_path 25' runs 25 prompts against every "
                        "transformer (leaving the prompt list intact). AXIS must be an "
                        "active --iterate axis; mutually exclusive with --limit; "
                        "--max-iterations still applies to the total.")
    p.add_argument("--batch", type=_positive_int, default=1,
                   metavar="N",
                   help="Repeat each planned generation N times. Alone (no --iterate), "
                        "runs the base config N times — pair with --seed -1 for fresh "
                        "random seeds per repeat. Paired with --iterate, runs N shots "
                        "at each combination. Default 1.")
    p.add_argument("--yes", "-y", action="store_true", default=False,
                   help="Skip the interactive iteration-count confirmation prompt. "
                        "Useful for scripted/cron use. --max-iterations still applies.")
    # ── Server mode ──────────────────────────────────────────────────────
    p.add_argument("--serve", action="store_true",
                   help="Start the persistent model server (keeps pipeline in VRAM)")
    p.add_argument("--unload", action="store_true",
                   help="Shut down the running model server cleanly")
    p.add_argument("--output-dir", type=str, default=None, metavar="DIR",
                   help="[--serve] Directory where the server saves generated images")
    p.add_argument("--model-base", type=str, default=None, metavar="DIR",
                   help="[--serve] Root that all model and LoRA paths must be within")
    return p.parse_args()


def _parse_lora_arg(spec: str) -> Dict[str, Any]:
    """Parse 'path:weight' or 'path' into {"path": ..., "weight": ...}."""
    if ":" in spec:
        # Split on LAST colon (paths may contain colons on Windows, unlikely here)
        idx = spec.rfind(":")
        try:
            weight = float(spec[idx + 1:])
            return {"path": spec[:idx], "weight": weight}
        except ValueError:
            pass
    return {"path": spec, "weight": 1.0}


def _run_json_mode() -> int:
    """Agent bridge: read JSON from stdin, write JSON to stdout.

    LEGACY: this mode is the pre-MCP LLM-agent transport per ADR-011 §5.
    New LLM-agent integration goes through MCP: `python -m
    comfyless.mcp_server --output-dir ... --model-base ...`. This mode is
    preserved at zero further investment for any non-LLM scripted caller
    still using it. MCP supersedes for any new integration work.

    See docs/decisions/ADR-011-comfyless-mcp-server.md §5.
    """
    try:
        raw = sys.stdin.read()
        req = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        json.dump({
            "status": "error",
            "error": f"Invalid JSON input: {e}",
            "error_type": "InvalidParams",
            "contract_version": CONTRACT_VERSION,
        }, sys.stdout, indent=2)
        return 1

    # Validate contract version
    req_version = req.get("contract_version")
    if req_version != CONTRACT_VERSION:
        json.dump({
            "status": "error",
            "error": f"Contract version mismatch: got {req_version}, "
                     f"expected {CONTRACT_VERSION}",
            "error_type": "ContractVersionMismatch",
            "contract_version": CONTRACT_VERSION,
        }, sys.stdout, indent=2)
        return 1

    # Extract params
    params = req.get("params", {})
    output_dir = req.get("output_dir", ".")
    output_stem = req.get("output_stem", "output")
    output_path = os.path.join(output_dir, f"{output_stem}.png")

    try:
        metadata = generate(
            model_path=req["model"],
            prompt=req["prompt"],
            output_path=output_path,
            negative_prompt=req.get("negative_prompt", ""),
            seed=params.get("seed", -1),
            steps=params.get("steps", 28),
            cfg_scale=params.get("cfg_scale", 3.5),
            true_cfg_scale=params.get("true_cfg_scale"),
            width=params.get("width", 1024),
            height=params.get("height", 1024),
            sampler=params.get("sampler", "default"),
            schedule=params.get("schedule", "linear"),
            loras=req.get("loras", []),
            max_sequence_length=params.get("max_sequence_length", 512),
            precision=params.get("precision", "bf16"),
            device=params.get("device", "cuda"),
            offload_vae=params.get("offload_vae", False),
            attention_slicing=params.get("attention_slicing", False),
            sequential_offload=params.get("sequential_offload", False),
            vae_tiling=params.get("vae_tiling", "auto"),
            # Hunyuan-Image refiner chain (ADR-016).
            refiner_path=params.get("refiner_path", ""),
            refiner_steps=params.get("refiner_steps", 4),
            refiner_cfg=params.get("refiner_cfg", 3.5),
            transformer_path=params.get("transformer_path", ""),
            vae_path=params.get("vae_path", ""),
            upscale_vae_path=params.get("upscale_vae_path", ""),
            upscale_vae_subfolder=params.get("upscale_vae_subfolder", ""),
            text_encoder_path=params.get("text_encoder_path", ""),
            text_encoder_2_path=params.get("text_encoder_2_path", ""),
            vae_from_transformer=params.get("vae_from_transformer", False),
            # quant became schema-legal on 2026-07-08; forward rather than
            # silently ignore (it affects output correctness).
            quant=params.get("quant") or "none",
            quant_skip=tuple(params.get("quant_skip") or ()),
            quant_only=tuple(params.get("quant_only") or ()),
            nag_scale=params.get("nag_scale", 0.0),
            nag_tau=params.get("nag_tau", 2.5),
            nag_alpha=params.get("nag_alpha", 0.25),
            nag_end=params.get("nag_end", 1.0),
        )

        sidecar_path = os.path.join(output_dir, f"{output_stem}.json")
        with open(sidecar_path, "w") as f:
            json.dump(metadata, f, indent=2)

        json.dump({
            "status": "ok",
            "output_paths": {
                "image": os.path.abspath(output_path),
                "metadata": os.path.abspath(sidecar_path),
            },
            "metadata": metadata,
            "contract_version": CONTRACT_VERSION,
        }, sys.stdout, indent=2)
        return 0

    except FileNotFoundError as e:
        json.dump({
            "status": "error",
            "error": str(e),
            "error_type": "ModelNotFound",
            "contract_version": CONTRACT_VERSION,
        }, sys.stdout, indent=2)
        return 1
    except Exception as e:
        json.dump({
            "status": "error",
            "error": str(e),
            "error_type": "InferenceError",
            "contract_version": CONTRACT_VERSION,
        }, sys.stdout, indent=2)
        return 1


# ════════════════════════════════════════════════════════════════════════
#  Server mode / socket delegation
# ════════════════════════════════════════════════════════════════════════

def _run_serve_mode(args: argparse.Namespace) -> int:
    """Start the persistent model server and block until --unload is received."""
    # Daemon logs are line-oriented (journald under systemd): tqdm-style
    # progress bars (transformers' per-tensor "Loading weights", HF Hub
    # downloads, diffusers shard loading) emit ANSI cursor escapes that
    # journald renders as one "[74B blob data]" line per refresh — hundreds
    # per model load. Foreground CLI keeps its bars; the daemon disables
    # them at startup.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for _modname in ("transformers", "diffusers"):
        try:
            _mod = __import__(f"{_modname}.utils.logging",
                              fromlist=["disable_progress_bar"])
            _mod.disable_progress_bar()
        except Exception:  # noqa: BLE001 — logging cosmetics, never fatal
            pass
    from .server import run_server
    if not args.model_base:
        print("Error: --model-base is required with --serve", file=sys.stderr)
        return 1
    if not args.output_dir:
        print("Error: --output-dir is required with --serve", file=sys.stderr)
        return 1
    try:
        run_server(
            output_dir=args.output_dir,
            model_base=args.model_base,
            device=args.device,
            precision=args.precision,
        )
        return 0
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _send_server_command(req: dict, device: str = "cuda") -> Optional[dict]:
    """Connect to the running server for `device`, send one request, return the response.

    Return contract (drives _delegate_to_server's fall-through decision):
      dict — the server's response, OR a synthetic {"status": "error",
             "error_type": "ClientRecvError"} when the daemon ACCEPTED the
             request but no parseable response arrived within
             _CLIENT_RECV_TIMEOUT_SEC (busy/wedged daemon, reset mid-read).
             A live daemon still holds its GPU's VRAM, so this must surface
             as an error — never as "no daemon", which would trigger an
             in-process generation against an occupied GPU (2026-07-17;
             previously the timeout's ValueError escaped uncaught).
      None — daemon absent or unreachable (no socket, connect/send failed,
             or clean EOF: the daemon process died before responding).
             Caller may fall through to in-process generation.
    Local socket-object creation failure (FD exhaustion) raises.

    The socket is device-keyed (ADR-020): one daemon per GPU, so the caller's
    device selects which daemon to reach.
    Local import keeps server.py off the critical import path.
    """
    import socket as _socket
    from .server import socket_path, _send, _recv, _CLIENT_RECV_TIMEOUT_SEC
    sock_p = socket_path(device)
    if not sock_p.exists():
        return None
    conn = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        try:
            conn.connect(str(sock_p))
            _send(conn, req)
        except OSError:
            return None
        try:
            # Pass the client-side timeout: the server's 5s default is a DoS
            # guard on its request-read path, not a ceiling on how long
            # generation takes. _recv returns None on clean EOF (daemon died).
            resp = _recv(conn, timeout=_CLIENT_RECV_TIMEOUT_SEC)
            if resp is not None and not isinstance(resp, dict):
                # Valid JSON but not an object (security review SHOULD 1,
                # review-pause-daemon-guard-2026-07-17): callers do
                # resp.get(...) — a scalar/list would crash them. A literal
                # `null` response remains indistinguishable from clean EOF
                # at this layer (acceptable under the same-UID model).
                raise ValueError(
                    f"non-object response: {type(resp).__name__}")
            return resp
        except (ValueError, OSError) as e:
            # _recv raises ValueError on deadline expiry / oversized frame /
            # garbage JSON; OSError (e.g. ECONNRESET) means the connection
            # broke mid-read. Either way the daemon took the request —
            # report, don't fall through.
            return {
                "status": "error",
                "error_type": "ClientRecvError",
                "error": (
                    f"daemon on {device!r} accepted the request but sent no "
                    f"valid response within {_CLIENT_RECV_TIMEOUT_SEC:.0f}s "
                    f"({e}); it may be busy or wedged — not falling back to "
                    f"in-process generation"
                ),
            }
    finally:
        conn.close()


def _send_unload(device: str = "cuda") -> int:
    """Send an unload command to the running server for `device`.

    Device-scoped (ADR-020): '--unload --device cuda:1' stops only the cuda:1
    daemon; bare '--unload' (default 'cuda' -> 'cuda:0') stops the cuda:0 daemon.
    Stopping every daemon means unloading each device.
    """
    resp = _send_server_command({"type": "unload"}, device)
    if resp is None:
        print("No server found (socket missing or connection refused).", file=sys.stderr)
        return 1
    if resp.get("status") == "ok":
        print("Server unloaded.")
        return 0
    print(f"Server error: {resp.get('error', 'unknown')}", file=sys.stderr)
    return 1


def _build_server_request(
    args: argparse.Namespace,
    p: dict,
    loras: list,
    *,
    savepath_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the daemon wire request for one generation (pure; no I/O).

    Resolves all path fields to absolute before sending. The server runs
    in its own CWD and calls os.path.realpath() for validation — relative
    paths sent from the client would resolve differently there.
    """
    def _abspath(v: str) -> str:
        return os.path.abspath(v) if v else v

    req: Dict[str, Any] = {
        "type":                "generate",
        "model":               _abspath(p["model"]),
        "prompt":              p["prompt"],
        "negative_prompt":     p.get("negative_prompt", ""),
        "seed":                p.get("seed", -1),
        "steps":               p.get("steps", 28),
        "cfg_scale":           p.get("cfg_scale", 3.5),
        "true_cfg_scale":      p.get("true_cfg_scale"),
        "width":               p.get("width", 1024),
        "height":              p.get("height", 1024),
        "sampler":             p.get("sampler", "default"),
        "schedule":            p.get("schedule", "linear"),
        "loras":               [{"path": _abspath(l["path"]), "weight": l.get("weight", 1.0)}
                                for l in loras],
        "max_sequence_length": p.get("max_sequence_length", 512),
        "precision":           args.precision,
        "device":              args.device,
        "offload_vae":         args.offload_vae,
        "attention_slicing":   args.attention_slicing,
        "sequential_offload":  args.sequential_offload,
        "vae_tiling":          args.vae_tiling,
        # Hunyuan-Image refiner chain (ADR-016). refiner_path is a path →
        # _abspath so the daemon's _check_paths sees it absolute. The daemon
        # fully consumes these: server.py's _check_paths validates
        # refiner_path against --model-base, _request_cache_key carries it as
        # the trailing entry, and _maybe_load_refiner loads the chain
        # server-side. Threading them client-side keeps in-process and daemon
        # requests in lockstep so a sidecar replay does not silently drop them
        # at the wire boundary.
        "refiner_path":        _abspath(p.get("refiner_path", "")),
        "refiner_steps":       p.get("refiner_steps", 4),
        "refiner_cfg":         p.get("refiner_cfg", 3.5),
        "rebalance":           args.rebalance,
        "rebalance_mult":      args.rebalance_mult,
        "transformer_path":    _abspath(p.get("transformer_path", "")),
        "vae_path":            _abspath(p.get("vae_path", "")),
        # ADR-030: absolutize the upscale VAE path at the wire boundary
        # (same as other model paths); subfolder is an in-repo name, not a path.
        "upscale_vae_path":     _abspath(p.get("upscale_vae_path", "")),
        "upscale_vae_subfolder": p.get("upscale_vae_subfolder", ""),
        "text_encoder_path":   _abspath(p.get("text_encoder_path", "")),
        "text_encoder_2_path": _abspath(p.get("text_encoder_2_path", "")),
        "vae_from_transformer": p.get("vae_from_transformer", False),
        # Quantize-on-load triple (ADR-019 slice DQ). Slot names, not paths —
        # the canonical validator enforces that server-side per entry.
        # Sourced from the merged params (sidecar-replayable, 2026-07-08),
        # not argparse — a --params sidecar's quant reaches the daemon.
        "quant":               p.get("quant") or "none",
        "quant_skip":          list(p.get("quant_skip") or []),
        "quant_only":          list(p.get("quant_only") or []),
        # NAG quadruple (ADR-023). Sidecar-replayable schema params;
        # deliberately NOT in the daemon's pipeline cache key — NAG
        # processors are installed per-call and restored, so pipeline
        # shape is unchanged (see server._request_cache_key).
        "nag_scale":           p.get("nag_scale", 0.0),
        "nag_tau":             p.get("nag_tau", 2.5),
        "nag_alpha":           p.get("nag_alpha", 0.25),
        "nag_end":             p.get("nag_end", 1.0),
    }
    # ADR-034 output format. Raw CLI values (name + 0.0-1.0 fraction); the
    # daemon resolves the OutputFormat and owns the on-disk extension. Omitted
    # when None (like rebalance_weights) so the canonical str/float validator
    # never sees a null. Type-checked via _RUNTIME_KIND, value-checked in
    # server._validate_request, never in the pipeline cache key.
    if args.output_format is not None:
        req["output_format"] = args.output_format
    if args.quality is not None:
        req["quality"] = args.quality
    # Pre-expanded template (iteration tokens resolved client-side) takes
    # precedence over args.savepath when provided.
    wire_savepath = savepath_override if savepath_override is not None else args.savepath
    if wire_savepath:
        req["savepath"] = wire_savepath
    return req


def _delegate_to_server(
    args: argparse.Namespace,
    p: dict,
    loras: list,
    *,
    iterate_batch_id: Optional[str] = None,
    savepath_override: Optional[str] = None,
) -> Optional[int]:
    """Try to send this generation request to the running server.

    Returns an int exit code when the server handled it (success or error).
    Returns None when the server is unreachable — caller falls through to
    in-process generation.

    Delegation is skipped when --output is set explicitly: the server owns
    path resolution and cannot write to an arbitrary caller-supplied path.
    Use --savepath for naming control when a server is running.
    """
    from .server import socket_path
    if not socket_path(args.device).exists():
        return None

    req = _build_server_request(args, p, loras,
                                savepath_override=savepath_override)

    # Per-layer weights: omit when unset so the daemon's _KIND_LIST validator
    # never sees a null (it defaults to the node preset server-side).
    _rb_weights = _parse_rebalance_weights(args.rebalance_weights)
    if _rb_weights is not None:
        req["rebalance_weights"] = _rb_weights

    resp = _send_server_command(req, args.device)
    if resp is None:
        _log("[comfyless] Server socket found but connection failed — running in-process")
        return None

    if resp.get("status") == "ok":
        metadata    = resp.get("metadata", {})
        output_path = resp.get("output_path", "")
        # Iteration stamps the batch id client-side so downstream grouping works
        # without requiring a server change. PNG tEXt embedding is deferred —
        # the sidecar carries the id; the PNG does not.
        if iterate_batch_id:
            metadata["iterate_batch_id"] = iterate_batch_id
        _log(f"[comfyless] Saved: {output_path}")
        if output_path:
            stem = os.path.splitext(output_path)[0]
            sidecar_path = f"{stem}.json"
            with open(sidecar_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[comfyless] Metadata: {sidecar_path}")
        print(f"\nDone. seed={metadata.get('seed', '?')}, "
              f"time={metadata.get('elapsed_seconds', '?')}s")
        # NAG skips/oddities happened in the DAEMON's process — its stderr
        # is a log the user never watches. Surface them client-side from the
        # wire metadata (invariant N1; mirrors lora_warnings below).
        for _w in metadata.get("nag_warnings") or []:
            print(f"[comfyless] WARNING: NAG — {_w}", file=sys.stderr)
        # A --schedule that the daemon warn-and-ignored (wrong family/scheduler)
        # is likewise surfaced client-side from the wire metadata — the daemon's
        # stderr never reaches here (ADR-028; invariant N1).
        for _w in metadata.get("schedule_warnings") or []:
            print(f"[comfyless] WARNING: --schedule ignored — {_w}", file=sys.stderr)
        # Daemon already carries lora_warnings in the wire metadata — surface
        # them loudly client-side (ADR-015 2026-07-06).
        return _report_lora_outcome(metadata)

    err = resp.get("error", "unknown error")
    print(f"Error (server): {err}", file=sys.stderr)
    return 1


# ════════════════════════════════════════════════════════════════════════
#  Iteration mode (--iterate) — see docs/decisions/ADR-008-comfyless-iterate.md
# ════════════════════════════════════════════════════════════════════════

# Element-shape per axis. Determines how _validate_iterate_value checks each
# entry in an --iterate <param> <file> JSON list.
_ITERATE_SHAPES: Dict[str, Any] = {
    "prompt":              str,
    "negative_prompt":     str,
    "sampler":             str,
    "model":               str,
    "transformer_path":    str,
    "vae_path":            str,
    "text_encoder_path":   str,
    "text_encoder_2_path": str,
    "seed":                int,
    "steps":               int,
    "width":               int,
    "height":              int,
    "cfg_scale":           "number",   # int OR float; bool rejected
    "lora":                "lora_stack",
}

_ITERATE_CONFIRM_THRESHOLD = 5   # prompt for confirmation at or above this count


def _positive_int(s: str) -> int:
    """argparse type for flags that require a positive integer (--limit, --batch)."""
    try:
        n = int(s)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected positive integer, got {s!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected positive integer (>= 1), got {n}")
    return n


def _validate_iterate_value(value: Any, expected: Any) -> bool:
    """True if `value` matches the expected iteration-element shape.

    Scalar axes only (str / int / number). The `lora` axis is NOT handled
    here — it is a human-authored replay surface with lenient normalization,
    routed through `_normalize_iterate_lora_element` from `_plan_iterations`
    (ADR-012 amendment 2026-07-10). Keep this function isinstance-clean for
    the machine-boundary N19 AST scan that still covers scalar axes.
    """
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    return False


def _normalize_iterate_lora_dict(entry: Any, where: str) -> Dict[str, Any]:
    """Normalize one LoRA dict for the `--iterate lora` file (human surface).

    `path` is required and must be a non-empty string. `weight` is OPTIONAL:
    absent → 1.0 (matches `_parse_lora_arg` and `_apply_loras`' `.get("weight",
    1.0)` — the iterate file must not be the ecosystem's lone outlier). A
    PRESENT weight must be a real number (int cast to float), never bool / str
    / None — a garbled weight is an authoring mistake worth surfacing, not
    silently coercing. Unknown keys pass through (kohya `rank`/`alpha` etc.).

    Unlike the canonical machine-boundary `validate_lora_entry`, this is
    deliberately lenient about a missing weight. See ADR-012 amendment
    2026-07-10 for why the iterate file is a human-replay surface, not a wire.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: expected a path string or a "
                         f"{{path, weight}} dict, got {type(entry).__name__}")
    if "path" not in entry:
        raise ValueError(f"{where}: LoRA entry missing required 'path'")
    path = entry["path"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{where}: 'path' must be a non-empty string, "
                         f"got {path!r}")
    out = dict(entry)
    if "weight" not in entry or entry["weight"] is None:
        out["weight"] = 1.0
    else:
        w = entry["weight"]
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ValueError(f"{where}: 'weight' must be a number, "
                             f"got {type(w).__name__}={w!r}")
        out["weight"] = float(w)
    return out


def _normalize_iterate_lora_element(value: Any, index: int) -> List[Dict[str, Any]]:
    """Normalize one `--iterate lora` list element into a canonical LoRA stack.

    Accepts three ergonomic shapes (ADR-012 amendment 2026-07-10):
      - str  "path" or "path:weight"  → single-LoRA stack (via _parse_lora_arg)
      - dict {path, weight?}          → single-LoRA stack (weight defaults 1.0)
      - list of the above             → an explicit reusable stack ([] = no LoRA)

    Returns the canonical `[{path: str, weight: float}, …]` shape stored in the
    iteration plan, so everything downstream of `_plan_iterations` is unchanged.
    Raises ValueError with an element-scoped message on malformed input.
    """
    where = f"element [{index}]"
    if isinstance(value, str):
        return [_parse_iterate_lora_string(value, where)]
    if isinstance(value, dict):
        return [_normalize_iterate_lora_dict(value, where)]
    if isinstance(value, list):
        stack: List[Dict[str, Any]] = []
        for j, item in enumerate(value):
            item_where = f"{where} stack entry [{j}]"
            if isinstance(item, str):
                stack.append(_parse_iterate_lora_string(item, item_where))
            else:
                stack.append(_normalize_iterate_lora_dict(item, item_where))
        return stack
    raise ValueError(f"{where}: expected a path string, a {{path, weight}} "
                     f"dict, or a list of those, got {type(value).__name__}")


def _parse_iterate_lora_string(spec: str, where: str) -> Dict[str, Any]:
    """Parse a `"path"` / `"path:weight"` iterate string into a LoRA dict.

    Wraps `_parse_lora_arg` (the CLI `--lora` grammar) and rejects a path that
    is empty after the optional `:weight` split — e.g. `":0.8"` — so the string
    form fails as loudly as the dict form does on a blank path, rather than
    deferring to a load-time no-op (reviewer LOW, 2026-07-10).
    """
    if not spec.strip():
        raise ValueError(f"{where}: empty LoRA path string")
    parsed = _parse_lora_arg(spec)
    if not parsed["path"].strip():
        raise ValueError(f"{where}: 'path' must be a non-empty string, "
                         f"got {spec!r}")
    return parsed


def _plan_iterations(args: argparse.Namespace) -> Optional[dict]:
    """Return an iteration plan dict, or None when no --iterate / --batch is active.

    Plan keys:
      axes:               list of (param_name, file_stem, values_list) tuples, in CLI order.
                          Empty when only --batch is active (pure repetition, no axes).
      cartesian:          Cartesian product size across all axes (1 when axes is empty).
      effective_combos:   cartesian after --limit truncation (= cartesian if no limit).
      batch:              repetitions per combination (= args.batch, default 1).
      total:              effective_combos * batch — the total number of generations.
      input_tokens:       dict mapping axis_name → file_stem, plus '_primary' → first axis.
                          Empty when no --iterate axes are present.

    Plan is None when there's nothing to plan (no --iterate, --batch == 1).
    Raises ValueError with a user-facing message on any validation failure.
    """
    batch = getattr(args, "batch", 1) or 1
    limit = getattr(args, "limit", None)
    limit_per = getattr(args, "limit_per", None)

    # --limit-per names an --iterate axis to group by; it is meaningless without
    # one. Check BEFORE the early return so `--limit-per X N` alone doesn't silently
    # no-op — the axis-membership promise must always fire (code review: drift).
    if limit_per is not None and not args.iterate:
        raise ValueError(
            f"--limit-per requires at least one --iterate axis "
            f"(got --limit-per {limit_per[0]!r} with no --iterate)"
        )

    if not args.iterate and batch == 1:
        return None

    axes = []
    for param, filepath in args.iterate:
        if param not in _ITERATE_SHAPES:
            raise ValueError(
                f"--iterate parameter {param!r} not supported. "
                f"Allowed: {sorted(_ITERATE_SHAPES.keys())}"
            )
        try:
            with open(filepath) as f:
                values = json.load(f)
        except OSError as e:
            raise ValueError(f"--iterate {param} {filepath!r}: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"--iterate {param} {filepath!r}: invalid JSON ({e})") from e
        if not isinstance(values, list):
            raise ValueError(
                f"--iterate {param} {filepath!r}: top-level must be a JSON list, "
                f"got {type(values).__name__}"
            )
        if not values:
            raise ValueError(f"--iterate {param} {filepath!r}: empty list")
        expected = _ITERATE_SHAPES[param]
        if expected == "lora_stack":
            # Human-authored replay surface: normalize each element into a
            # canonical [{path, weight}, …] stack (weight defaults to 1.0,
            # "path:weight" strings accepted). ADR-012 amendment 2026-07-10.
            normalized: List[Any] = []
            for i, v in enumerate(values):
                try:
                    normalized.append(_normalize_iterate_lora_element(v, i))
                except ValueError as e:
                    raise ValueError(
                        f"--iterate {param} {filepath!r} {e}"
                    ) from e
            values = normalized
        else:
            for i, v in enumerate(values):
                if not _validate_iterate_value(v, expected):
                    shape_name = expected if isinstance(expected, str) else expected.__name__
                    raise ValueError(
                        f"--iterate {param} {filepath!r} element [{i}]: "
                        f"expected {shape_name}, got {type(v).__name__}={v!r}"
                    )
        axes.append((param, Path(filepath).stem, values))

    # An axis given twice (argparse `append` permits it) has no coherent meaning —
    # the per-run patch dict can hold only one value per param, and it silently
    # corrupts the plan/exec count invariant under --limit-per. Reject it (code
    # review: drift). This also removes the pre-existing flat-mode ambiguity.
    axis_names = [a[0] for a in axes]
    dupes = sorted({n for n in axis_names if axis_names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"--iterate axis(es) {dupes} given more than once; "
            f"each axis may be iterated at most once"
        )

    cartesian = 1
    for _, _, values in axes:
        cartesian *= len(values)

    # Two truncation models (argparse enforces they're mutually exclusive):
    #   --limit N       → first N of the flattened Cartesian (silent ceiling).
    #   --limit-per A N → N per value of axis A, cycling the OTHER axes. The named
    #                     axis is the OUTER group; N caps the inner (other-axes)
    #                     Cartesian per group value (ceiling — clamped when smaller).
    limit_per_axis: Optional[str] = None
    limit_per_n: Optional[int] = None
    if limit_per is not None:
        axis_name, n_str = limit_per
        axis_names = [a[0] for a in axes]
        if axis_name not in axis_names:
            raise ValueError(
                f"--limit-per axis {axis_name!r} is not an active --iterate axis. "
                f"Active axes: {axis_names or '(none)'}"
            )
        try:
            limit_per_n = int(n_str)
        except (TypeError, ValueError):
            raise ValueError(
                f"--limit-per {axis_name}: N must be an integer, got {n_str!r}"
            )
        if limit_per_n < 1:
            raise ValueError(f"--limit-per {axis_name}: N must be >= 1, got {limit_per_n}")
        limit_per_axis = axis_name
        inner = 1
        group_len = 0
        for name, _, values in axes:
            if name == axis_name:
                group_len = len(values)
            else:
                inner *= len(values)
        effective = group_len * min(limit_per_n, inner)
    else:
        effective = cartesian if limit is None else min(cartesian, limit)
    total = effective * batch

    if total > args.max_iterations:
        raise ValueError(
            f"{total} iterations exceeds --max-iterations={args.max_iterations}. "
            f"Raise --max-iterations, lower --batch, lower --limit / --limit-per, "
            f"or narrow the iteration files."
        )

    input_tokens: Dict[str, str] = {name: stem for name, stem, _ in axes}
    if axes:
        input_tokens["_primary"] = axes[0][1]

    return {
        "axes": axes,
        "cartesian": cartesian,
        "effective_combos": effective,
        "batch": batch,
        "total": total,
        "input_tokens": input_tokens,
        "limit_per_axis": limit_per_axis,
        "limit_per_n": limit_per_n,
    }


def _iteration_combos(plan: dict) -> Any:
    """Yield param-patch dicts in execution order.

    For each of `effective_combos` Cartesian combinations (truncated by --limit),
    yield the combination `batch` times consecutively. Pure-batch plans (empty
    axes, batch > 1) yield empty dicts `batch` times — the caller treats an
    empty patch as "use base config unchanged."

    Backward-compat: if a plan dict was built without the new `effective_combos`
    / `batch` keys (e.g. a hand-built test mock), fall back to full Cartesian
    with batch=1.
    """
    import itertools
    axes = plan["axes"]
    batch = plan.get("batch", 1)
    if "effective_combos" in plan:
        effective = plan["effective_combos"]
    else:
        # Backward-compat for older test mocks (test_iterate.py:215, 232) that
        # hand-build plans with only {axes, total, input_tokens}. Derive full
        # Cartesian from axes; combined with batch=1 default this preserves the
        # pre-`--limit` / pre-`--batch` behaviour for those mocks.
        effective = 1
        for axis in axes:
            effective *= len(axis[2])

    if not axes:
        # Pure-batch mode: yield the empty patch `batch` times.
        for _ in range(batch):
            yield {}
        return

    # --limit-per: the named axis is the OUTER group; for each of its values, run
    # the first N combinations of the OTHER axes' Cartesian (itertools.product of
    # an empty axis list yields one empty tuple, so a lone group axis runs once per
    # value). This is what makes "N prompts per transformer" land — the group axis
    # varies slowest, the cycled axes fastest.
    limit_per_axis = plan.get("limit_per_axis")
    if limit_per_axis is not None:
        per_n = plan["limit_per_n"]
        group_name, _, group_values = next(a for a in axes if a[0] == limit_per_axis)
        others = [a for a in axes if a[0] != limit_per_axis]
        other_names = [a[0] for a in others]
        other_lists = [a[2] for a in others]
        for gval in group_values:
            count = 0
            for inner in itertools.product(*other_lists):
                if count >= per_n:
                    break
                patch = {group_name: gval}
                patch.update(dict(zip(other_names, inner)))
                for _ in range(batch):
                    yield dict(patch)
                count += 1
        return

    names = [a[0] for a in axes]
    value_lists = [a[2] for a in axes]
    count = 0
    for combo in itertools.product(*value_lists):
        if count >= effective:
            break
        for _ in range(batch):
            yield dict(zip(names, combo))
        count += 1


def _iteration_replaces_loras(plan: Optional[dict], base_loras: list) -> bool:
    """True when a lora-axis iteration will replace a non-empty base lora list."""
    if plan is None or not base_loras:
        return False
    return any(axis[0] == "lora" for axis in plan["axes"])


def _confirm_iteration(total: int, auto_yes: bool) -> bool:
    """Interactive [y/N] prompt over the confirm threshold. --yes skips."""
    if auto_yes or total < _ITERATE_CONFIRM_THRESHOLD:
        return True
    print(f"Iteration inputs will result in {total} generations. Proceed? [y/N] ",
          file=sys.stderr, end="", flush=True)
    try:
        ans = input()
    except EOFError:
        ans = ""
    return ans.strip().lower() in ("y", "yes")


def _cli_value_for(args: argparse.Namespace, canonical_key: str) -> Any:
    """Return the argparse value for a canonical schema key, or None if unset.

    Walks _CLI_TO_CANONICAL in reverse to find the argparse attribute name
    that maps to the canonical key, then falls back to the canonical name
    itself (for --model, --prompt, --seed, --steps, --width, --height,
    --sampler, --schedule which are already canonical).
    """
    for cli_name, canon in _CLI_TO_CANONICAL.items():
        if canon == canonical_key:
            return getattr(args, cli_name, None)
    return getattr(args, canonical_key, None)


def _run_cli_mode(args: argparse.Namespace) -> int:
    """Human CLI mode: argparse flags, human-readable output.

    When --params is given, sidecar JSON provides base params.
    --override key=value patches apply next, then any explicit CLI
    flags (non-None) win over the sidecar.
    """
    # ── Build effective params ────────────────────────────────────────
    # explicit_keys tracks every canonical key that came from a deliberate
    # source (sidecar / --override / non-None CLI flag).  The family-default
    # overlay (ADR-009) writes ONLY into keys NOT in this set, which is how
    # we distinguish "user said 3.5" from "schema seeded 3.5" once both
    # are resident in p as cfg_scale=3.5.
    explicit_keys: set = set()

    if args.params:
        try:
            p = _load_params(args.params)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Error loading --params {args.params!r}: {e}", file=sys.stderr)
            return 1
        explicit_keys |= set(p.keys())
        explicit_keys |= _explicit_override_keys(args.override)
        try:
            p = _apply_overrides(p, args.override)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        # No sidecar: seed with schema defaults (skipping required fields
        # which will be sourced from --model / --prompt below).  Note:
        # --override only takes effect with --params (per its --help
        # text); we don't apply overrides here.
        p = {
            # list defaults are copied so runs never share/mutate the
            # schema's default objects (quant_skip/quant_only are lists).
            k: (list(default) if isinstance(default, list) else default)
            for k, (_type, default) in COMFYLESS_SCHEMA.items()
            if default is not None and k != "loras"
        }

    # Explicit CLI flags (sentinel = None means "not set") win over whatever
    # the sidecar / overrides / defaults put in the dict.  Driven off the
    # schema so every canonical key is considered exactly once.
    for canonical_key in COMFYLESS_SCHEMA:
        if canonical_key == "loras":
            continue  # loras is sourced from --lora via _parse_lora_arg below
        cli_val = _cli_value_for(args, canonical_key)
        if cli_val is not None:
            p[canonical_key] = cli_val
            explicit_keys.add(canonical_key)

    # --lora is the explicit channel for the loras stack; mirror sidecar
    # symmetry so a future loras-bearing FAMILY_DEFAULTS entry would not
    # clobber it.  (No family currently sets loras; this is preventative.)
    if getattr(args, "lora", None):
        explicit_keys.add("loras")

    # Final validation pass: catches anything an override injected that the
    # schema doesn't know about, or a CLI-merged value whose type is wrong.
    p = _validate_params(p, source="cli-merged")

    # quant is choices-gated at argparse but a sidecar/--override can inject
    # any string — fail loudly HERE rather than after model resolution
    # (mirrors the daemon's slice-DQ semantic check).
    _q = p.get("quant") or "none"
    if _q not in QUANT_MODES:
        print(f"Error: unknown quant mode {_q!r} (from --params/--override). "
              f"Expected: {' | '.join(QUANT_MODES)}", file=sys.stderr)
        return 1

    # ── Plan iterations (no-op when --iterate is absent) ──────────────────
    # Run BEFORE the required-field check so that --iterate prompt/--iterate model
    # can satisfy those requirements without a separate CLI flag.
    try:
        plan = _plan_iterations(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    iterated_axes = {axis[0] for axis in plan["axes"]} if plan else set()

    if not p.get("model") and "model" not in iterated_axes:
        print("Error: --model is required (or provide via --params / --override model=... "
              "/ --iterate model <file>)", file=sys.stderr)
        return 1
    if not p.get("prompt") and "prompt" not in iterated_axes:
        print("Error: --prompt is required (or provide via --params / --override prompt=... "
              "/ --iterate prompt <file>)", file=sys.stderr)
        return 1

    # After all overrides are applied, warn if the BASE model path looks like a
    # local path but doesn't exist. HF repo IDs (owner/repo, no leading /) are
    # skipped — they'll be resolved in the next step. Skipped entirely when
    # --iterate model is active: each iteration's model is checked per-run
    # inside _run_one's resolve_hf_path call.
    _model_val = p.get("model") or ""
    if "model" not in iterated_axes:
        _is_local = _model_val.startswith("/") or _model_val.startswith("./") or (
            len(_model_val) > 1 and _model_val[1] == ":"
        )
        if _is_local and not os.path.exists(_model_val):
            print(
                f"WARNING: model path does not exist on this host:\n"
                f"  {_model_val}\n"
                f"  If this came from a container-saved image, use --model <host-path> "
                f"or --override model=<host-path>.",
                file=sys.stderr,
            )

        # Symmetric to the local-path warning: if the model came from a PNG
        # --params sidecar and is an HF repo ID under --allow-hf-download,
        # surface what the PNG will pull BEFORE resolution — a malicious
        # sidecar could otherwise trigger an arbitrary HF fetch with only the
        # resolve_hf_path one-liner as notice. Only fires when --params was
        # used; direct --model entries are the user's own choice.
        if args.params and args.allow_hf_download and _is_hf_repo_id(_model_val):
            print(
                f"WARNING: --params supplied an HF repo ID under --allow-hf-download:\n"
                f"  {_model_val}\n"
                f"  This will be fetched from HuggingFace on cache miss. "
                f"Verify the repo is what you expect before proceeding.",
                file=sys.stderr,
            )

    base_loras = [_parse_lora_arg(s) for s in args.lora] if args.lora else p.get("loras", [])

    if plan is not None:
        if _iteration_replaces_loras(plan, base_loras):
            print("WARNING: --iterate lora replaces --lora / sidecar loras for every iteration.",
                  file=sys.stderr)
        # Footgun guard: --batch >1 with a fixed seed AND no --iterate seed axis means
        # every repeat will produce an identical image. Warn but proceed — see
        # `feedback_warn_dont_block` user memory.
        if (plan.get("batch", 1) > 1
                and p.get("seed", -1) != -1
                and "seed" not in iterated_axes):
            print(
                f"WARNING: --batch {plan['batch']} with --seed {p.get('seed')} "
                f"(fixed) will produce {plan['batch']} identical images per "
                f"combination. Use --seed -1 for fresh random seeds per repeat.",
                file=sys.stderr,
            )
        if not _confirm_iteration(plan["total"], args.yes):
            print("Aborted.", file=sys.stderr)
            return 1

    iterate_batch_id = str(uuid.uuid4()) if plan is not None else None
    iterate_inputs = plan["input_tokens"] if plan is not None else None

    # Inline prompt-enhancement state (ADR-026). Memoized per unique input
    # prompt across the whole run so a fixed prompt over a lora/transformer
    # sweep is enhanced ONCE (clean A/B; no wasted LLM calls), while a
    # --iterate prompt axis enhances each distinct prompt. Backends loaded once.
    _enhance_memo: Dict[str, str] = {}
    _enhance_backends: List[Optional[dict]] = [None]

    def _run_one(p_cur: dict, loras_cur: list,
                 idx: Optional[int] = None, total: Optional[int] = None) -> int:
        """Run a single generation from the effective params. Returns exit code."""
        if idx is not None:
            _log(f"[comfyless] iter {idx}/{total}")

        # Per-iteration HF resolution: --iterate model can swap the base model
        # or components between runs, so resolve each time rather than once
        # up front.
        for _key in ("model", "transformer_path", "vae_path",
                     "upscale_vae_path",
                     "text_encoder_path", "text_encoder_2_path"):
            if p_cur.get(_key):
                try:
                    p_cur[_key] = resolve_hf_path(
                        p_cur[_key], allow_download=args.allow_hf_download,
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"Error resolving {_key}: {e}", file=sys.stderr)
                    return 1

        # Family-default overlay (ADR-009): writes family-specific cfg/steps/etc.
        # for keys not in explicit_keys and not in iterated_axes.  Runs AFTER
        # HF resolution (model_index.json must be readable) and BEFORE the
        # daemon delegation / generate() call so values flow through unchanged.
        _apply_family_defaults(p_cur, explicit_keys, iterated_axes, idx=idx)

        _enh_provenance = None
        # ── Inline prompt enhancement (ADR-026) ───────────────────────────
        # Enhance the effective prompt just before dispatch, so it flows
        # unchanged into BOTH the daemon-delegate and in-process paths and is
        # recorded as `prompt` in the sidecar (a --params replay then reuses
        # the enhanced text and never re-calls the LLM). Memoized per unique
        # input prompt. Fail loud — never silently generate on the un-enhanced
        # prompt once the operator asked for enhancement.
        if getattr(args, "enhance_prompt", None):
            _raw = p_cur.get("prompt", "") or ""
            if _raw and _raw not in _enhance_memo:
                try:
                    from comfyless.enhance import (load_backends as _lb,
                                                   enhance as _enh)
                    if _enhance_backends[0] is None:
                        _enhance_backends[0] = _lb(getattr(args, "enhance_config", None))
                    _enhance_memo[_raw] = _enh(
                        _raw, args.enhance_prompt,
                        backends=_enhance_backends[0],
                        recipe_name=getattr(args, "enhance_recipe", None),
                        family=None, n=1,
                        # co-locate a local (hunyuan) reprompt model on the run's
                        # GPU; openai-endpoint ignores device (it's HTTP)
                        device=args.device,
                    )[0]
                    _log(f"[comfyless] prompt enhanced via {args.enhance_prompt!r} "
                         f"(recipe={getattr(args, 'enhance_recipe', None) or 'default'})")
                except Exception as e:
                    print(f"Error: prompt enhancement via "
                          f"{args.enhance_prompt!r} failed: {e}", file=sys.stderr)
                    return 1
            if _raw:
                p_cur["prompt"] = _enhance_memo[_raw]
                # Provenance for the sidecar (ADR-026 §7): the original prompt +
                # which backend/recipe produced the enhancement. Recorded on the
                # in-process path via generate(extra_metadata=...). (Daemon-path
                # provenance is a documented follow-up — A10; replay
                # determinism holds on both paths regardless.)
                _enh_provenance = {
                    "original_prompt": _raw,
                    "enhance_backend": args.enhance_prompt,
                    "enhance_recipe": getattr(args, "enhance_recipe", None) or "default",
                }

        using_default_output = args.output == "/tmp/comfyless.png"

        # Resolve output format (ADR-034 D2/D3). Extension inference applies
        # only to a caller-authored --output path — not the default sentinel
        # or a savepath template, whose extensions are not user intent.
        infer_path = None if (args.savepath or using_default_output) else args.output
        try:
            out_fmt = resolve_output_format(args.output_format, args.quality, infer_path)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        if out_fmt.name == "png" and args.quality is not None:
            _log("[comfyless] --quality is ignored for png output.")

        # Delegate to daemon when --savepath or default --output; skip on explicit --output.
        # quant delegates too since slice DQ: the wire request carries the
        # quant triple and the daemon keys its pipeline cache on it (see
        # docs/security/review-slice-DQ-daemon-quant-2026-07-03.md).
        # ADR-034 slice 2: output_format/quality ride the wire request and the
        # daemon owns the extension, so jpeg delegates like any other request.
        if args.savepath or using_default_output:
            # Pre-expand iteration tokens client-side so the daemon receives a
            # template it can finish resolving (%seed%, %model%, etc.) without
            # needing to know about iteration at all.
            wire_savepath = None
            if args.savepath and iterate_inputs is not None:
                wire_savepath = _expand_iterate_tokens(args.savepath, iterate_inputs)
            delegate_rc = _delegate_to_server(
                args, p_cur, loras_cur,
                iterate_batch_id=iterate_batch_id,
                savepath_override=wire_savepath,
            )
            if delegate_rc is not None:
                return delegate_rc

        # In-process path.
        if args.savepath:
            seed_for_path = p_cur.get("seed", -1)
            if seed_for_path < 0:
                seed_for_path = torch.randint(0, 2**32 - 1, (1,)).item()
                _log(f"[comfyless] Random seed: {seed_for_path}")
                p_cur["seed"] = seed_for_path
            output_path = _resolve_savepath(
                args.savepath,
                p_cur["model"],
                seed_for_path,
                p_cur.get("steps", COMFYLESS_SCHEMA["steps"][1]),
                p_cur.get("cfg_scale", COMFYLESS_SCHEMA["cfg_scale"][1]),
                p_cur.get("sampler", COMFYLESS_SCHEMA["sampler"][1]),
                transformer_path=p_cur.get("transformer_path", ""),
                iterate_inputs=iterate_inputs,
                extension=out_fmt.extension,
            )
            _log(f"[comfyless] Output: {output_path}")
        else:
            output_path = args.output
            # The default sentinel is /tmp/comfyless.png; when jpeg is selected
            # without an explicit --output, follow the resolved extension.
            if using_default_output and out_fmt.extension != ".png":
                output_path = os.path.splitext(args.output)[0] + out_fmt.extension

        try:
            metadata = generate(
                model_path=p_cur["model"],
                prompt=p_cur["prompt"],
                extra_metadata=_enh_provenance,
                output_path=output_path,
                negative_prompt=p_cur.get("negative_prompt", ""),
                seed=p_cur.get("seed", -1),
                steps=p_cur.get("steps", 28),
                cfg_scale=p_cur.get("cfg_scale", 3.5),
                true_cfg_scale=p_cur.get("true_cfg_scale"),
                width=p_cur.get("width", 1024),
                height=p_cur.get("height", 1024),
                sampler=p_cur.get("sampler", "default"),
                schedule=p_cur.get("schedule", "linear"),
                loras=loras_cur,
                max_sequence_length=p_cur.get("max_sequence_length", 512),
                precision=args.precision,
                device=args.device,
                offload_vae=args.offload_vae,
                attention_slicing=args.attention_slicing,
                sequential_offload=args.sequential_offload,
                vae_tiling=args.vae_tiling,
                # Hunyuan-Image refiner chain (ADR-016). refiner_path is a
                # canonical schema key resolved via _CLI_TO_CANONICAL
                # (--refiner → refiner_path) into p_cur; steps/cfg ride the
                # family-defaults overlay.
                refiner_path=p_cur.get("refiner_path", ""),
                refiner_steps=p_cur.get("refiner_steps", 4),
                refiner_cfg=p_cur.get("refiner_cfg", 3.5),
                allow_hf_download=args.allow_hf_download,
                rebalance=args.rebalance,
                rebalance_mult=args.rebalance_mult,
                rebalance_weights=_parse_rebalance_weights(args.rebalance_weights),
                transformer_path=p_cur.get("transformer_path", ""),
                vae_path=p_cur.get("vae_path", ""),
                upscale_vae_path=p_cur.get("upscale_vae_path", ""),
                upscale_vae_subfolder=p_cur.get("upscale_vae_subfolder", ""),
                text_encoder_path=p_cur.get("text_encoder_path", ""),
                text_encoder_2_path=p_cur.get("text_encoder_2_path", ""),
                vae_from_transformer=p_cur.get("vae_from_transformer", False),
                quant=p_cur.get("quant") or "none",
                quant_skip=tuple(p_cur.get("quant_skip") or ()),
                quant_only=tuple(p_cur.get("quant_only") or ()),
                nag_scale=p_cur.get("nag_scale", 0.0),
                nag_tau=p_cur.get("nag_tau", 2.5),
                nag_alpha=p_cur.get("nag_alpha", 0.25),
                nag_end=p_cur.get("nag_end", 1.0),
                output_format=out_fmt,
            )
            if iterate_batch_id:
                metadata["iterate_batch_id"] = iterate_batch_id
            stem = os.path.splitext(output_path)[0]
            sidecar_path = f"{stem}.json"
            with open(sidecar_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[comfyless] Metadata: {sidecar_path}")
            print(f"\nDone. seed={metadata['seed']}, "
                  f"time={metadata['elapsed_seconds']}s")
            return _report_lora_outcome(metadata)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1

    # ── Single-gen path (no iteration) ────────────────────────────────────
    if plan is None:
        return _run_one(p, base_loras)

    # ── Iteration fan-out ─────────────────────────────────────────────────
    t_start = time.monotonic()
    succeeded = 0
    lora_soft_fail = False
    for idx, patch in enumerate(_iteration_combos(plan), start=1):
        p_iter = dict(p)
        loras_iter = list(base_loras)
        for axis_name, axis_value in patch.items():
            if axis_name == "lora":
                loras_iter = list(axis_value)
            else:
                p_iter[axis_name] = axis_value
        rc = _run_one(p_iter, loras_iter, idx=idx, total=plan["total"])
        disposition = _iterate_combo_disposition(rc)
        if disposition == "fatal":
            print(f"[comfyless] iteration failed at {idx}/{plan['total']} — stopping. "
                  f"batch_id={iterate_batch_id}", file=sys.stderr)
            return rc
        # 'soft' (rc 3): image written but a LoRA didn't apply — _run_one
        # already printed the loud banner; keep sweeping, flag for the summary.
        if disposition == "soft":
            lora_soft_fail = True
        succeeded += 1

    elapsed = time.monotonic() - t_start
    print(f"[comfyless] iterate: {succeeded}/{plan['total']} completed in {elapsed:.1f}s "
          f"(batch_id={iterate_batch_id})", file=sys.stderr)
    if lora_soft_fail:
        print(f"[comfyless] iterate: one or more combos were generated WITHOUT a "
              f"requested LoRA (see the ⚠️  warnings above)", file=sys.stderr)
    return _LORA_SOFT_FAIL_RC if lora_soft_fail else 0


def _split_model_arg(args: argparse.Namespace) -> List[str]:
    """Normalize argparse's nargs='+' --model from list back to a string for the
    rest of the codebase. Returns the list of *extra* positional values (used by
    the Cascade dispatch for config paths after the `stablecascade` sentinel).

    After this returns, `args.model` is either None or a single string — every
    downstream code path that reads args.model continues to work unchanged.
    """
    if args.model is None:
        return []
    if not isinstance(args.model, list):
        # Defensive: argparse should always hand us a list, but tolerate a string
        # if some test or future caller bypasses parsing.
        return []
    extras = args.model[1:]
    args.model = args.model[0]
    return extras


def main() -> int:
    args = _parse_args()
    cascade_extras = _split_model_arg(args)

    # ── Stable Cascade dispatch fork ──────────────────────────────────────
    # Sentinel `--model stablecascade <config.json> [config2.json] ...` activates
    # the JSON-config family. Has its own dispatch entirely separate from the
    # standard --model path. See ADR-010 + docs/comfyless-stable-cascade.md.
    from comfyless.cascade import CASCADE_SENTINEL, dispatch as _cascade_dispatch
    if args.model == CASCADE_SENTINEL:
        # ADR-034 slice 4: cascade dispatch resolves --output-format / --quality
        # itself (numbering + _save_with_metadata are format-aware); no stopgap.
        return _cascade_dispatch(args, cascade_extras)
    if cascade_extras:
        # User passed multiple values to --model but the first wasn't `stablecascade`.
        # Most likely a path with spaces/typo; refuse rather than silently drop.
        print(
            f"Error: --model received {1 + len(cascade_extras)} values but the first "
            f"({args.model!r}) is not the cascade sentinel. Quote paths with spaces, "
            f"or use '--model stablecascade <config.json> ...' for Stable Cascade.",
            file=sys.stderr,
        )
        return 2

    if args.json:
        # Iteration semantics (--iterate, --batch, --limit) are not yet
        # expressible in the JSON bridge contract (see ADR-008 §"Interaction
        # with --json mode"). Reject rather than silently ignore — adding
        # iteration to the JSON schema is separate design work gated by the
        # LLM-agent-bridge slice.
        if (args.iterate or args.batch != 1 or args.limit is not None
                or args.limit_per is not None):
            json.dump({
                "status": "error",
                "error": "--iterate / --batch / --limit / --limit-per are not "
                         "supported in --json mode; iteration semantics will be "
                         "added to the JSON bridge contract in a future release",
                "error_type": "IterationNotSupported",
                "contract_version": CONTRACT_VERSION,
            }, sys.stdout, indent=2)
            return 1
        # Output format is not wired on the JSON bridge yet (ADR-034: the bridge
        # is a future slice). Reject loudly rather than emit PNG while the caller
        # asked for jpeg — same contract as the iteration rejection above.
        if args.output_format is not None or args.quality is not None:
            json.dump({
                "status": "error",
                "error": "--output-format / --quality are not supported in "
                         "--json mode yet; output-format handling for the JSON "
                         "bridge lands in a future ADR-034 slice",
                "error_type": "OutputFormatNotSupported",
                "contract_version": CONTRACT_VERSION,
            }, sys.stdout, indent=2)
            return 1
        return _run_json_mode()
    if args.serve:
        return _run_serve_mode(args)
    if args.unload:
        return _send_unload(args.device)
    return _run_cli_mode(args)


if __name__ == "__main__":
    sys.exit(main())
