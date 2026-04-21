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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from nodes.eric_diffusion_utils import (
    detect_pipeline_class,
    read_guidance_embeds,
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

    if model_family in ("sdxl", "sd3", "sd1"):
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

    # ── Detect model family and load pipeline ─────────────────────────
    _log(f"[comfyless] Loading model: {model_path}")
    pipeline_class, class_name, model_family = detect_pipeline_class(model_path)
    _log(f"[comfyless] Detected: {class_name} (family: {model_family})")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(precision, torch.bfloat16)

    pipe = pipeline_class.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)

    # VAE tiling — required for >2MP decode
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    # VAE offload
    if offload_vae and hasattr(pipe, "vae"):
        pipe.vae = pipe.vae.to("cpu")
        _log("[comfyless] VAE offloaded to CPU")

    guidance_embeds = read_guidance_embeds(pipe)
    _log(f"[comfyless] Ready — family={model_family}, "
         f"guidance_embeds={guidance_embeds}")

    # ── Load LoRAs ────────────────────────────────────────────────────
    lora_warnings: List[str] = []
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

    # ── Inference (with optional sampler swap) ────────────────────────
    t0 = time.monotonic()
    with swap_sampler(pipe, sampler, log_prefix="[comfyless]"):
        result = pipe(**call_kwargs)
    elapsed = time.monotonic() - t0
    _log(f"[comfyless] Generated in {elapsed:.1f}s")

    # ── Save PNG ──────────────────────────────────────────────────────
    pil_image = result.images[0]
    pil_image.save(output_path)
    _log(f"[comfyless] Saved: {output_path}")

    # ── Clean up VAE ──────────────────────────────────────────────────
    if offload_vae and hasattr(pipe, "vae"):
        pipe.vae = pipe.vae.to("cpu")

    # ── Build metadata ────────────────────────────────────────────────
    metadata: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model_path,
        "model_family": model_family,
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
    p.add_argument("--model", type=str, help="Path to diffusers model directory")
    p.add_argument("--prompt", type=str, help="Generation prompt")
    p.add_argument("--negative-prompt", type=str, default="",
                   help="Negative prompt (qwen-image only)")
    p.add_argument("--seed", type=int, default=-1,
                   help="Random seed (-1 for random)")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=3.5, help="CFG scale")
    p.add_argument("--true-cfg", type=float, default=None,
                   help="True CFG scale override (qwen-image)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--lora", action="append", default=[],
                   metavar="PATH:WEIGHT",
                   help="LoRA to apply (repeatable).  Format: path or path:weight")
    p.add_argument("--sampler", choices=SAMPLER_NAMES, default="default",
                   help="Sampler algorithm")
    p.add_argument("--schedule", choices=SCHEDULE_NAMES, default="linear",
                   help="Sigma schedule (reserved for future manual-loop use)")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Max sequence length for text encoder")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--offload-vae", action="store_true")
    p.add_argument("--output", "-o", type=str, default="output.png",
                   help="Output image path")
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


def _run_cli_mode(args: argparse.Namespace) -> int:
    """Human CLI mode: argparse flags, human-readable output."""
    if not args.model:
        print("Error: --model is required", file=sys.stderr)
        return 1
    if not args.prompt:
        print("Error: --prompt is required", file=sys.stderr)
        return 1

    loras = [_parse_lora_arg(s) for s in args.lora] if args.lora else []

    try:
        metadata = generate(
            model_path=args.model,
            prompt=args.prompt,
            output_path=args.output,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            steps=args.steps,
            cfg_scale=args.cfg,
            true_cfg_scale=args.true_cfg,
            width=args.width,
            height=args.height,
            sampler=args.sampler,
            schedule=args.schedule,
            loras=loras,
            max_sequence_length=args.max_seq_len,
            precision=args.precision,
            device=args.device,
            offload_vae=args.offload_vae,
        )

        # Write sidecar alongside the image
        stem = os.path.splitext(args.output)[0]
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
    return _run_cli_mode(args)


if __name__ == "__main__":
    sys.exit(main())
