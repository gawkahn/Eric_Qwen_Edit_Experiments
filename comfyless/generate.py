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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from nodes.eric_diffusion_utils import (
    detect_pipeline_class,
    detect_load_variant,
    read_guidance_embeds,
    read_model_index,
    resolve_component_class,
    detect_component_format,
    load_component,
    resolve_hf_path,
)
from nodes.eric_diffusion_samplers import sampler_choices, swap_sampler
from nodes.eric_qwen_edit_lora import load_lora_with_key_fix

CONTRACT_VERSION = 1
SAMPLER_NAMES = sampler_choices()
SCHEDULE_NAMES = ["linear", "balanced", "karras"]
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


# ── Sidecar / override helpers ───────────────────────────────────────────

_SKIP_SIDECAR_KEYS = {"timestamp", "elapsed_seconds", "contract_version",
                      "lora_warnings", "model_family"}

# Keys written by Eric Diffusion Save that are not comfyless CLI params.
# model_path is handled separately (renamed → model).
_ERIC_SAVE_DROP = _SKIP_SIDECAR_KEYS | {
    "node_type", "model_name", "sampler_s2", "sampler_s3", "loras", "model_path",
}

_CLI_DEFAULTS = {
    "negative_prompt": "",
    "seed": -1,
    "steps": 28,
    "cfg": 3.5,
    "true_cfg": None,
    "width": 1024,
    "height": 1024,
    "sampler": "default",
    "schedule": "linear",
    "max_seq_len": 512,
    "transformer_path": "",
    "vae_path": "",
    "text_encoder_path": "",
    "text_encoder_2_path": "",
    "vae_from_transformer": False,
}


def _load_sidecar(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k not in _SKIP_SIDECAR_KEYS}


def _load_params(path: str) -> dict:
    """Load base params from a comfyless sidecar JSON or a PNG with embedded metadata."""
    if path.lower().endswith(".png"):
        return _load_params_from_png(path)
    return _load_sidecar(path)


def _extract_eric_save_params(params_json: str, path: str) -> dict:
    """Extract gen params from an Eric Diffusion Save 'parameters' tEXt chunk.

    Renames model_path → model; drops node-internal fields not used by comfyless.
    LoRA weights stored in the chunk are not replayed (format mismatch); use --lora.
    """
    try:
        data = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"parameters chunk in {path!r} is not valid JSON: {e}")

    out = {k: v for k, v in data.items() if k not in _ERIC_SAVE_DROP}
    if "model_path" in data:
        out["model"] = data["model_path"]

    _log(f"[comfyless] Eric Diffusion Save parameters chunk — extracted {sorted(out.keys())}")

    if data.get("loras"):
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
        return {k: v for k, v in data.items() if k not in _SKIP_SIDECAR_KEYS}

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

    return params


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


def _expand_savepath_template(
    template: str,
    model_path: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    sampler: str,
    transformer_path: str = "",
) -> str:
    """Expand %var% and %var:spec% tokens in a savepath template string.

    %model%      — transformer filename if --transformer is set, otherwise base model.
                   Matches ComfyUI behavior: shows the weights that were actually used.
    %transformer% — always the transformer filename (or base model if none).
    %base_model% — always the base model directory name.
    All token names are case-insensitive.
    """
    base_model_name = Path(model_path).name
    model_name = Path(transformer_path).name if transformer_path else base_model_name

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
) -> str:
    """Expand template, create parent dirs, return first available counter slot."""
    expanded = _expand_savepath_template(
        template, model_path, seed, steps, cfg_scale, sampler, transformer_path,
    )
    parent = Path(expanded).parent
    parent.mkdir(parents=True, exist_ok=True)
    stem = Path(expanded).name
    counter = 1
    while True:
        candidate = parent / f"{stem}{counter:04d}.png"
        if not candidate.exists():
            return str(candidate)
        counter += 1


def _save_with_metadata(pil_image, path: str, metadata: dict) -> None:
    """Save a PIL image as PNG with comfyless metadata embedded as a tEXt chunk."""
    from PIL.PngImagePlugin import PngInfo
    pnginfo = PngInfo()
    pnginfo.add_text("comfyless", json.dumps(metadata, default=str))
    pil_image.save(path, pnginfo=pnginfo)


def _apply_overrides(params: dict, overrides: list) -> dict:
    result = dict(params)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--override {spec!r}: expected key=value format")
        key, _, raw = spec.partition("=")
        result[key.strip()] = _coerce(raw.strip())
    return result


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

    if model_family in ("flux", "flux2", "flux2klein", "chroma"):
        kwargs = {**base, "guidance_scale": cfg_scale}
        sig = inspect.signature(pipe.__call__)
        if "max_sequence_length" in sig.parameters:
            kwargs["max_sequence_length"] = max_sequence_length
        return kwargs

    if model_family in ("sdxl", "sd3", "sd1", "zimage"):
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
#  Pipeline loader (extracted so the server can cache the result)
# ════════════════════════════════════════════════════════════════════════

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
    allow_hf_download: bool = False,
):
    """Load, place, and configure a diffusers pipeline.

    Returns (pipe, model_family, guidance_embeds).
    Called by generate() for one-shot use and by the server to populate its cache.
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
        cls_, cname = resolve_component_class(model_index, "vae")
        if cls_ is None:
            raise ValueError(f"VAE class '{cname}' not found in diffusers.")
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

    pipe = pipeline_class.from_pretrained(model_path, **load_kwargs)

    if sequential_offload:
        _log("[comfyless] Enabling sequential CPU offload")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)
        if offload_vae and hasattr(pipe, "vae"):
            pipe.vae = pipe.vae.to("cpu")
            _log("[comfyless] VAE offloaded to CPU")

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    if attention_slicing:
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            _log("[comfyless] Attention slicing enabled")
        except Exception as e:
            _log(f"[comfyless] Attention slicing not available: {e}")

    guidance_embeds = read_guidance_embeds(pipe)
    _log(f"[comfyless] Ready — family={model_family}, guidance_embeds={guidance_embeds}")
    return pipe, model_family, guidance_embeds


# ════════════════════════════════════════════════════════════════════════
#  Core generate function
# ════════════════════════════════════════════════════════════════════════

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
    allow_hf_download: bool = False,
    _cached_pipeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate a single image and save it.

    Args:
        loras: List of {"path": str, "weight": float} dicts.  Applied
            in order.  LoRA load failures are non-fatal (warned, skipped).
        sampler: One of SAMPLER_NAMES ("default", "multistep2", "multistep3").
        schedule: Sigma schedule — reserved for future manual-loop use.

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
            sequential_offload=sequential_offload, allow_hf_download=allow_hf_download,
        )

    # ── Load LoRAs ────────────────────────────────────────────────────
    lora_warnings: List[str] = []
    loras = loras or []
    # When a cached pipeline is provided the server has already managed LoRAs;
    # skip loading but keep the list so it appears correctly in metadata.
    if _cached_pipeline is None:
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
                    msg = f"LoRA skipped (0 modules applied): {lora_path}"
                    _log(f"[comfyless] WARNING: {msg}")
                    lora_warnings.append(msg)
            except Exception as e:
                msg = f"LoRA load failed: {lora_path}: {e}"
                _log(f"[comfyless] WARNING: {msg}")
                lora_warnings.append(msg)

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

    # ── Inference (with optional sampler swap) ────────────────────────
    t0 = time.monotonic()
    with swap_sampler(pipe, effective_sampler, log_prefix="[comfyless]"):
        result = pipe(**call_kwargs)
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "contract_version": CONTRACT_VERSION,
    }
    if lora_warnings:
        metadata["lora_warnings"] = lora_warnings

    # ── Save PNG with embedded metadata ──────────────────────────────
    pil_image = result.images[0]
    _save_with_metadata(pil_image, output_path, metadata)
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
    p.add_argument("--model", type=str, default=None,
                   help="Path to diffusers model directory")
    p.add_argument("--prompt", type=str, default=None,
                   help="Generation prompt")
    p.add_argument("--negative-prompt", type=str, default=None,
                   help="Negative prompt (qwen-image only)")
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
                   help="Sigma schedule (reserved for future manual-loop use)")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="Max sequence length for text encoder")
    p.add_argument("--transformer", type=str, default=None, metavar="PATH",
                   help="Custom transformer/UNet weights (dir, subdir, or .safetensors)")
    p.add_argument("--vae", type=str, default=None, metavar="PATH",
                   help="Custom VAE weights")
    p.add_argument("--te1", type=str, default=None, metavar="PATH",
                   help="Custom text encoder slot 1 (CLIP-L for Flux; Qwen2.5-VL for Qwen)")
    p.add_argument("--te2", type=str, default=None, metavar="PATH",
                   help="Custom text encoder slot 2 (T5-XXL for Flux/Chroma)")
    p.add_argument("--vae-from-transformer", action="store_true", default=None,
                   help="Extract VAE from the --transformer AIO checkpoint")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--offload-vae", action="store_true")
    p.add_argument("--attention-slicing", action="store_true",
                   help="Trade speed for lower peak VRAM")
    p.add_argument("--sequential-offload", action="store_true",
                   help="Extreme VRAM savings via sequential CPU offload — very slow")
    p.add_argument("--allow-hf-download", action="store_true", default=False,
                   help="Allow downloading models from HuggingFace if not in local cache. "
                        "By default only the local cache is used (no network access)")
    p.add_argument("--output", "-o", type=str, default="/tmp/comfyless.png",
                   help="Output image path (exact; overwrites). "
                        "Ignored when a server is running — use --savepath instead.")
    p.add_argument("--savepath", type=str, default=None,
                   metavar="TEMPLATE",
                   help="Output path template with %%date:MM-dd-YY%%, %%model:12%%, "
                        "%%seed%%, %%steps%%, %%cfg%%, %%sampler%%. "
                        "Auto-creates dirs; always writes comfyless0001.png, 0002, ...")
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
    """Agent bridge: read JSON from stdin, write JSON to stdout."""
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
            transformer_path=params.get("transformer_path", ""),
            vae_path=params.get("vae_path", ""),
            text_encoder_path=params.get("text_encoder_path", ""),
            text_encoder_2_path=params.get("text_encoder_2_path", ""),
            vae_from_transformer=params.get("vae_from_transformer", False),
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


def _send_server_command(req: dict) -> Optional[dict]:
    """Connect to the running server, send one request, return the response.

    Returns None if the socket doesn't exist or the connection is refused.
    Local import keeps server.py off the critical import path.
    """
    import socket as _socket
    from .server import socket_path, _send, _recv
    sock_p = socket_path()
    if not sock_p.exists():
        return None
    conn = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        conn.connect(str(sock_p))
        _send(conn, req)
        return _recv(conn)
    except (OSError, ConnectionRefusedError):
        return None
    finally:
        conn.close()


def _send_unload() -> int:
    """Send an unload command to the running server."""
    resp = _send_server_command({"type": "unload"})
    if resp is None:
        print("No server found (socket missing or connection refused).", file=sys.stderr)
        return 1
    if resp.get("status") == "ok":
        print("Server unloaded.")
        return 0
    print(f"Server error: {resp.get('error', 'unknown')}", file=sys.stderr)
    return 1


def _delegate_to_server(
    args: argparse.Namespace,
    p: dict,
    loras: list,
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
    if not socket_path().exists():
        return None

    # Resolve all path fields to absolute before sending. The server runs
    # in its own CWD and calls os.path.realpath() for validation — relative
    # paths sent from the client would resolve differently there.
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
        "transformer_path":    _abspath(p.get("transformer_path", "")),
        "vae_path":            _abspath(p.get("vae_path", "")),
        "text_encoder_path":   _abspath(p.get("text_encoder_path", "")),
        "text_encoder_2_path": _abspath(p.get("text_encoder_2_path", "")),
        "vae_from_transformer": p.get("vae_from_transformer", False),
    }
    if args.savepath:
        req["savepath"] = args.savepath

    resp = _send_server_command(req)
    if resp is None:
        _log("[comfyless] Server socket found but connection failed — running in-process")
        return None

    if resp.get("status") == "ok":
        metadata    = resp.get("metadata", {})
        output_path = resp.get("output_path", "")
        _log(f"[comfyless] Saved: {output_path}")
        if output_path:
            stem = os.path.splitext(output_path)[0]
            sidecar_path = f"{stem}.json"
            with open(sidecar_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[comfyless] Metadata: {sidecar_path}")
        print(f"\nDone. seed={metadata.get('seed', '?')}, "
              f"time={metadata.get('elapsed_seconds', '?')}s")
        return 0

    err = resp.get("error", "unknown error")
    print(f"Error (server): {err}", file=sys.stderr)
    return 1


def _run_cli_mode(args: argparse.Namespace) -> int:
    """Human CLI mode: argparse flags, human-readable output.

    When --params is given, sidecar JSON provides base params.
    --override key=value patches apply next, then any explicit CLI
    flags (non-None) win over the sidecar.
    """
    # ── Build effective params ────────────────────────────────────────
    if args.params:
        try:
            p = _load_params(args.params)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"Error loading --params {args.params!r}: {e}", file=sys.stderr)
            return 1
        try:
            p = _apply_overrides(p, args.override)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        # Explicit CLI flags (sentinel = None means "not set") win over sidecar
        for sidecar_key, cli_val in [
            ("model",              args.model),
            ("prompt",             args.prompt),
            ("negative_prompt",    args.negative_prompt),
            ("seed",               args.seed),
            ("steps",              args.steps),
            ("cfg_scale",          args.cfg),
            ("true_cfg_scale",     args.true_cfg),
            ("width",              args.width),
            ("height",             args.height),
            ("sampler",            args.sampler),
            ("schedule",           args.schedule),
            ("max_sequence_length", args.max_seq_len),
            ("transformer_path",    args.transformer),
            ("vae_path",            args.vae),
            ("text_encoder_path",   args.te1),
            ("text_encoder_2_path", args.te2),
            ("vae_from_transformer", args.vae_from_transformer),
        ]:
            if cli_val is not None:
                p[sidecar_key] = cli_val
    else:
        d = _CLI_DEFAULTS
        p = {
            "model":               args.model,
            "prompt":              args.prompt,
            "negative_prompt":     args.negative_prompt   if args.negative_prompt is not None else d["negative_prompt"],
            "seed":                args.seed              if args.seed              is not None else d["seed"],
            "steps":               args.steps             if args.steps             is not None else d["steps"],
            "cfg_scale":           args.cfg               if args.cfg               is not None else d["cfg"],
            "true_cfg_scale":      args.true_cfg          if args.true_cfg          is not None else d["true_cfg"],
            "width":               args.width             if args.width             is not None else d["width"],
            "height":              args.height            if args.height            is not None else d["height"],
            "sampler":             args.sampler           if args.sampler           is not None else d["sampler"],
            "schedule":            args.schedule          if args.schedule          is not None else d["schedule"],
            "max_sequence_length": args.max_seq_len       if args.max_seq_len       is not None else d["max_seq_len"],
            "transformer_path":    args.transformer       if args.transformer       is not None else d["transformer_path"],
            "vae_path":            args.vae               if args.vae               is not None else d["vae_path"],
            "text_encoder_path":   args.te1               if args.te1               is not None else d["text_encoder_path"],
            "text_encoder_2_path": args.te2               if args.te2               is not None else d["text_encoder_2_path"],
            "vae_from_transformer": args.vae_from_transformer if args.vae_from_transformer is not None else d["vae_from_transformer"],
        }

    if not p.get("model"):
        print("Error: --model is required (or provide via --params / --override model=...)",
              file=sys.stderr)
        return 1
    if not p.get("prompt"):
        print("Error: --prompt is required (or provide via --params / --override prompt=...)",
              file=sys.stderr)
        return 1

    # After all overrides are applied, warn if the model path looks like a
    # local path but doesn't exist. HF repo IDs (owner/repo, no leading /)
    # are skipped — they'll be resolved in the next step.
    _model_val = p.get("model", "")
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

    loras = [_parse_lora_arg(s) for s in args.lora] if args.lora else p.get("loras", [])

    # ── Resolve HF repo IDs to local paths before server delegation ───
    # The server's path validation expects real filesystem paths; HF IDs
    # must be resolved client-side so the server never sees a repo ID.
    allow_hf_download = args.allow_hf_download
    for _key in ("model", "transformer_path", "vae_path", "text_encoder_path", "text_encoder_2_path"):
        if p.get(_key):
            try:
                p[_key] = resolve_hf_path(p[_key], allow_download=allow_hf_download)
            except (ValueError, RuntimeError) as e:
                print(f"Error resolving {_key}: {e}", file=sys.stderr)
                return 1

    # ── Delegate to server if one is running ──────────────────────────
    # Only delegate when using default --output or explicit --savepath.
    # Explicit --output paths can't be honoured by the server (it owns
    # path resolution), so we fall through to in-process in that case.
    using_default_output = args.output == "/tmp/comfyless.png"
    if args.savepath or using_default_output:
        result = _delegate_to_server(args, p, loras)
        if result is not None:
            return result

    # Resolve output path — savepath template takes precedence over --output
    if args.savepath:
        seed_for_path = p.get("seed", -1)
        if seed_for_path < 0:
            seed_for_path = torch.randint(0, 2**32 - 1, (1,)).item()
            _log(f"[comfyless] Random seed: {seed_for_path}")
            p["seed"] = seed_for_path
        output_path = _resolve_savepath(
            args.savepath,
            p["model"],
            seed_for_path,
            p.get("steps", _CLI_DEFAULTS["steps"]),
            p.get("cfg_scale", _CLI_DEFAULTS["cfg"]),
            p.get("sampler", _CLI_DEFAULTS["sampler"]),
            transformer_path=p.get("transformer_path", ""),
        )
        _log(f"[comfyless] Output: {output_path}")
    else:
        output_path = args.output

    try:
        metadata = generate(
            model_path=p["model"],
            prompt=p["prompt"],
            output_path=output_path,
            negative_prompt=p.get("negative_prompt", ""),
            seed=p.get("seed", -1),
            steps=p.get("steps", 28),
            cfg_scale=p.get("cfg_scale", 3.5),
            true_cfg_scale=p.get("true_cfg_scale"),
            width=p.get("width", 1024),
            height=p.get("height", 1024),
            sampler=p.get("sampler", "default"),
            schedule=p.get("schedule", "linear"),
            loras=loras,
            max_sequence_length=p.get("max_sequence_length", 512),
            precision=args.precision,
            device=args.device,
            offload_vae=args.offload_vae,
            attention_slicing=args.attention_slicing,
            sequential_offload=args.sequential_offload,
            allow_hf_download=allow_hf_download,
            transformer_path=p.get("transformer_path", ""),
            vae_path=p.get("vae_path", ""),
            text_encoder_path=p.get("text_encoder_path", ""),
            text_encoder_2_path=p.get("text_encoder_2_path", ""),
            vae_from_transformer=p.get("vae_from_transformer", False),
        )

        # Write sidecar alongside the image
        stem = os.path.splitext(output_path)[0]
        sidecar_path = f"{stem}.json"
        with open(sidecar_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[comfyless] Metadata: {sidecar_path}")

        print(f"\nDone. seed={metadata['seed']}, "
              f"time={metadata['elapsed_seconds']}s")
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


def main() -> int:
    args = _parse_args()
    if args.json:
        return _run_json_mode()
    if args.serve:
        return _run_serve_mode(args)
    if args.unload:
        return _send_unload()
    return _run_cli_mode(args)


if __name__ == "__main__":
    sys.exit(main())
