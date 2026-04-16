#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion LoRA Test Harness — comfyless, host-venv driven.

Three modes, one binary:

  --inspect <lora>                     Read LoRA header, classify adapter
                                       format, classify architecture, dump
                                       sample keys and basic stats.  No
                                       model load required (very fast).

  --check <lora> --base-model <dir>    Build a {param: shape} dict from
                                       the model directory's safetensors
                                       headers, run check_lora() against
                                       it.  Reports key-match% and
                                       dim-match%.  Fast; no full load.

  --generate <lora> --base-model <dir> Full pipeline load via the project's
                                       EricDiffusionLoader, attempt the
                                       LoRA via load_lora_with_key_fix
                                       (the path that catches conversion-
                                       worthy adapters), call pipe(...) at
                                       low resolution, save PNG.  Heavy.

Configuration lives in tests/lora_test_config.toml (gitignored — copy
lora_test_config.example.toml to start).  CLI args override config values.

Path translation: any LoRA / model path passed on the CLI is run through
the [paths.container_to_host] substring map before resolution, so paths
copied from ComfyUI workflows or LoRA metadata can be used as-is.

Usage examples
--------------
    PY=/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python
    cd /home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments

    # Format & architecture introspection — no model needed
    $PY -m tests.lora_test_harness --inspect "Flux.2 Klein 9B/klein_snofs_v1_1.safetensors"

    # Compatibility check vs. a base model
    $PY -m tests.lora_test_harness --check "Flux.2 Klein 9B/klein_snofs_v1_1.safetensors" \
        --base-model "FLUX.2-Klein-9B/transformer"

    # End-to-end generation (slow)
    $PY -m tests.lora_test_harness --generate \
        "Flux.2 Klein 9B/klein_snofs_v1_1.safetensors" \
        --base-model FLUX.2-Klein-9B \
        --prompt "test image" --out test.png

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import textwrap
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Project root on sys.path so `nodes.*` imports work ──────────────────
HARNESS_FILE = Path(__file__).resolve()
PROJECT_ROOT = HARNESS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── ComfyUI compatibility shim ──────────────────────────────────────────
# The nodes package imports `folder_paths` and `comfy.utils` (ComfyUI
# runtime modules) at the top of several files, including via
# nodes/__init__.py.  We never CALL any of those functions — every code
# path in this harness passes absolute paths directly and skips ComfyUI-
# specific UI/progress hooks — so stub modules are sufficient.  Must be
# installed BEFORE any `from nodes.*` import.
def _install_comfy_shims() -> None:
    import types

    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.get_folder_paths = lambda *_a, **_k: []
        fp.get_full_path = lambda *_a, **_k: None
        sys.modules["folder_paths"] = fp

    if "comfy" not in sys.modules:
        sys.modules["comfy"] = types.ModuleType("comfy")
    if "comfy.utils" not in sys.modules:
        cu = types.ModuleType("comfy.utils")

        class _NoopProgressBar:
            def __init__(self, *_a, **_k): pass
            def update(self, *_a, **_k): pass
            def update_absolute(self, *_a, **_k): pass

        cu.ProgressBar = _NoopProgressBar
        sys.modules["comfy.utils"] = cu
        sys.modules["comfy"].utils = cu


_install_comfy_shims()


# ════════════════════════════════════════════════════════════════════════
#  Config loading
# ════════════════════════════════════════════════════════════════════════

CONFIG_FILE = HARNESS_FILE.parent / "lora_test_config.toml"
EXAMPLE_FILE = HARNESS_FILE.parent / "lora_test_config.example.toml"


def load_config() -> dict:
    """Load TOML config; return {} if absent (CLI must supply absolute paths)."""
    if not CONFIG_FILE.exists():
        print(
            f"[harness] No {CONFIG_FILE.name} found.  Copy "
            f"{EXAMPLE_FILE.name} → {CONFIG_FILE.name} and edit "
            f"the paths.  Falling back to absolute-path-only mode."
        )
        return {}
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def translate_path(p: str, cfg: dict) -> str:
    """Apply [paths.container_to_host] substring substitutions in order."""
    table = cfg.get("paths", {}).get("container_to_host", {})
    for needle, replacement in table.items():
        if needle in p:
            return p.replace(needle, replacement)
    return p


def resolve_lora_path(p: str, cfg: dict) -> Path:
    """Translate, then resolve against [paths] loras_root if not absolute."""
    p = translate_path(p, cfg)
    path = Path(p)
    if path.is_absolute():
        return path
    root = cfg.get("paths", {}).get("loras_root")
    if root:
        return Path(root) / p
    return path  # let downstream code raise


def resolve_model_path(p: str, cfg: dict) -> Path:
    """Translate, then resolve against [paths] models_root if not absolute."""
    p = translate_path(p, cfg)
    path = Path(p)
    if path.is_absolute():
        return path
    root = cfg.get("paths", {}).get("models_root")
    if root:
        return Path(root) / p
    return path


# ════════════════════════════════════════════════════════════════════════
#  Lightweight safetensors header reader (duplicated from the LoRA
#  checker so --inspect works without importing torch)
# ════════════════════════════════════════════════════════════════════════

def read_safetensors_header(path: Path) -> Tuple[Dict, Dict]:
    """Return (tensor_index, metadata) — fast, no weight load."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    metadata = h.pop("__metadata__", {}) or {}
    return h, metadata


# ════════════════════════════════════════════════════════════════════════
#  --inspect
# ════════════════════════════════════════════════════════════════════════

def cmd_inspect(args, cfg: dict) -> int:
    lora_path = resolve_lora_path(args.lora, cfg)
    if not lora_path.is_file():
        print(f"[inspect] FILE NOT FOUND: {lora_path}", file=sys.stderr)
        return 1

    size_mb = lora_path.stat().st_size / (1024 * 1024)
    print(f"\n[inspect] {lora_path}")
    print(f"          size: {size_mb:.1f} MB")

    header, metadata = read_safetensors_header(lora_path)
    print(f"          tensor count: {len(header)}")

    if metadata:
        print("\n[inspect] Embedded metadata:")
        for k, v in sorted(metadata.items()):
            v_str = str(v)
            if len(v_str) > 200:
                v_str = v_str[:200] + "…"
            print(f"          {k}: {v_str}")

    # Adapter format detection — use the same suffix table the loader uses
    from nodes.eric_diffusion_lora_check import (
        _strip_adapter_suffix, check_lora,
    )

    layers: Dict[str, Dict[str, tuple]] = {}
    alphas: Dict[str, float] = {}
    suffixes_seen: Dict[str, int] = {}
    for k, v in header.items():
        base, sfx = _strip_adapter_suffix(k)
        if sfx is None:
            continue
        suffixes_seen[sfx] = suffixes_seen.get(sfx, 0) + 1
        shape = tuple(v["shape"])
        layers.setdefault(base, {})[sfx] = shape
        if sfx == ".alpha":
            # Alphas are usually scalars stored as a 1-element tensor;
            # we can't read the value without loading, but we can note presence.
            alphas[base] = float("nan")

    print(f"\n[inspect] Adapter suffix histogram ({len(layers)} unique base modules):")
    for sfx, count in sorted(suffixes_seen.items(), key=lambda kv: -kv[1]):
        print(f"          {sfx:>22} : {count}")

    # Heuristic adapter format
    if any(s in suffixes_seen for s in (".lokr_w1", ".lokr_w2")):
        adapter_kind = "LoKR"
    elif any(s in suffixes_seen for s in (".hada_w1_a", ".hada_w2_a")):
        adapter_kind = "LoHa"
    elif any(s in suffixes_seen for s in
             (".lora_A.weight", ".lora_B.weight", ".lora_down.weight",
              ".lora_up.weight")):
        adapter_kind = "Standard LoRA"
    else:
        adapter_kind = "UNKNOWN"
    print(f"\n[inspect] Adapter format guess: {adapter_kind}")

    # Architecture detection — check_lora() with an empty param dict will
    # populate arch_hint solely from the LoRA's own key patterns, which
    # is exactly what we want for inspect.
    empty_pd: Dict = {"_norm_lookup": {}}
    result = check_lora(lora_path, param_dict=empty_pd, log_prefix="[inspect]")
    if result.arch_hint:
        print("\n[inspect] Architecture hint:")
        for line in textwrap.wrap(result.arch_hint, width=72):
            print(f"          {line}")

    # Sample keys
    sample_n = min(5, len(layers))
    if sample_n:
        print(f"\n[inspect] Sample base module paths (first {sample_n}):")
        for k in list(layers)[:sample_n]:
            print(f"          {k}")

    return 0


# ════════════════════════════════════════════════════════════════════════
#  --check
# ════════════════════════════════════════════════════════════════════════

def cmd_check(args, cfg: dict) -> int:
    lora_path = resolve_lora_path(args.lora, cfg)
    if not lora_path.is_file():
        print(f"[check] LoRA not found: {lora_path}", file=sys.stderr)
        return 1

    base_model = resolve_model_path(args.base_model, cfg)
    # check_lora needs the directory containing transformer safetensors.
    # Prefer transformer/ subdir when present — that's what
    # diffusers from_pretrained actually loads.  Some model dirs ship
    # an additional original-format single-file checkpoint at top level
    # (e.g. flux-2-klein-9b.safetensors); checking against that would
    # give a misleading 100% match for an original-format LoRA, when
    # the live model in memory is the diffusers-reorganized version.
    if (base_model / "transformer").is_dir() and any(
        (base_model / "transformer").glob("*.safetensors")
    ):
        transformer_dir = base_model / "transformer"
    elif any(base_model.glob("*.safetensors")):
        transformer_dir = base_model
    else:
        print(
            f"[check] No safetensors found at {base_model} or "
            f"{base_model}/transformer — pass the directory that "
            f"contains the transformer shards.",
            file=sys.stderr,
        )
        return 1

    from nodes.eric_diffusion_lora_check import build_param_dict_from_dir, check_lora

    print(f"\n[check] Indexing transformer params from {transformer_dir} …")
    param_dict = build_param_dict_from_dir(str(transformer_dir))

    print(f"\n[check] Checking {lora_path.name}")
    result = check_lora(lora_path, param_dict=param_dict, log_prefix="[check]")
    for line in result.log_lines(prefix="[check]"):
        print(line)

    return 0 if result.verdict in ("OK", "NORM_TARGETING") else 2


# ════════════════════════════════════════════════════════════════════════
#  --generate
# ════════════════════════════════════════════════════════════════════════

def cmd_generate(args, cfg: dict) -> int:
    lora_path = resolve_lora_path(args.lora, cfg)
    if not lora_path.is_file():
        print(f"[generate] LoRA not found: {lora_path}", file=sys.stderr)
        return 1

    base_model = resolve_model_path(args.base_model, cfg)
    if not (base_model / "model_index.json").is_file():
        print(
            f"[generate] {base_model} is not a diffusers model dir "
            f"(no model_index.json).",
            file=sys.stderr,
        )
        return 1

    # folder_paths shim is installed at module load — no per-call setup needed
    gen_cfg = cfg.get("generate", {})
    prompt          = args.prompt          or gen_cfg.get("prompt", "test")
    negative_prompt = args.negative_prompt or gen_cfg.get("negative_prompt", "")
    steps           = args.steps           or gen_cfg.get("steps", 4)
    guidance_scale  = args.guidance_scale  or gen_cfg.get("guidance_scale", 1.0)
    true_cfg_scale  = args.true_cfg_scale  or gen_cfg.get("true_cfg_scale", 1.0)
    height          = args.height          or gen_cfg.get("height", 512)
    width           = args.width           or gen_cfg.get("width", 512)
    seed            = args.seed            if args.seed is not None else gen_cfg.get("seed", 42)
    weight          = args.weight          if args.weight is not None else gen_cfg.get("weight", 1.0)

    import torch
    import inspect as _inspect
    from nodes.eric_diffusion_loader import EricDiffusionLoader
    from nodes.eric_qwen_edit_lora import (
        load_lora_with_key_fix, _make_adapter_name,
    )

    # ── Load pipeline ─────────────────────────────────────────────────
    print(f"\n[generate] Loading pipeline from {base_model} …")
    loader = EricDiffusionLoader()
    (pipeline_dict,) = loader.load_pipeline(
        model_path=str(base_model),
        precision="bf16",
        device="cuda" if torch.cuda.is_available() else "cpu",
        keep_in_vram=False,
        offload_vae=False,
        attention_slicing=False,
        sequential_offload=False,
    )
    pipe = pipeline_dict["pipeline"]
    family = pipeline_dict["model_family"]
    print(f"[generate] family={family}  guidance_embeds="
          f"{pipeline_dict['guidance_embeds']}")

    # ── Apply LoRA ────────────────────────────────────────────────────
    adapter_name = _make_adapter_name(lora_path.name)
    print(f"\n[generate] Applying LoRA: {lora_path.name} (weight={weight})")
    success = load_lora_with_key_fix(
        pipe, str(lora_path), adapter_name,
        log_prefix="[generate-LoRA]",
        weight=weight,
        min_compatibility=0.0,  # never auto-skip in the test harness
    )
    if not success:
        print(
            "[generate] LoRA was NOT loaded.  Continuing with base "
            "model so we can compare outputs by toggling the adapter "
            "in/out of subsequent runs."
        )

    # ── Build kwargs for stock pipe(...) ──────────────────────────────
    sig = _inspect.signature(pipe.__call__)
    accepted = set(sig.parameters.keys())
    call_kwargs = {
        "prompt":             prompt,
        "num_inference_steps": steps,
        "height":             height,
        "width":              width,
        "generator":          torch.Generator(device="cpu").manual_seed(int(seed)),
    }
    # Family-aware CFG routing — same logic as the real generate node
    if "true_cfg_scale" in accepted:
        call_kwargs["true_cfg_scale"] = true_cfg_scale
    if "guidance_scale" in accepted:
        call_kwargs["guidance_scale"] = guidance_scale
    if "negative_prompt" in accepted and negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt
    # Drop anything the pipeline doesn't accept (best-effort introspection)
    call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}

    printable = {k: (v if k != "generator" else "<generator>")
                 for k, v in call_kwargs.items()}
    print(f"\n[generate] pipe(**{printable})")
    out = pipe(**call_kwargs)
    image = out.images[0]

    # ── Save ──────────────────────────────────────────────────────────
    out_path = args.out
    if out_path is None:
        out_dir = Path(cfg.get("paths", {}).get("output_dir", "tests/output"))
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = lora_path.stem.replace(".", "_")
        out_path = out_dir / f"{stem}_seed{seed}.png"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(out_path)
    print(f"\n[generate] saved → {out_path}")
    return 0


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Comfyless test harness for Eric_Qwen_Edit_Experiments LoRAs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect",  metavar="LORA",
                      help="Read LoRA header, classify format, dump samples")
    mode.add_argument("--check",    metavar="LORA",
                      help="Check LoRA against --base-model")
    mode.add_argument("--generate", metavar="LORA",
                      help="Full pipeline + LoRA + pipe(...); writes PNG")

    p.add_argument("--base-model", metavar="DIR",
                   help="Diffusers model dir (for --check / --generate)")
    p.add_argument("--prompt",          help="(generate) override prompt")
    p.add_argument("--negative-prompt", help="(generate) override negative prompt")
    p.add_argument("--steps",          type=int,   help="(generate) inference steps")
    p.add_argument("--guidance-scale", type=float, help="(generate) guidance_scale")
    p.add_argument("--true-cfg-scale", type=float, help="(generate) true_cfg_scale")
    p.add_argument("--height",         type=int,   help="(generate) image height")
    p.add_argument("--width",          type=int,   help="(generate) image width")
    p.add_argument("--seed",           type=int,   help="(generate) RNG seed")
    p.add_argument("--weight",         type=float, help="(generate) LoRA weight")
    p.add_argument("--out",            help="(generate) output PNG path")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config()

    if args.inspect:
        args.lora = args.inspect
        return cmd_inspect(args, cfg)

    if args.check:
        if not args.base_model:
            print("--check requires --base-model", file=sys.stderr)
            return 2
        args.lora = args.check
        return cmd_check(args, cfg)

    if args.generate:
        if not args.base_model:
            print("--generate requires --base-model", file=sys.stderr)
            return 2
        args.lora = args.generate
        return cmd_generate(args, cfg)

    return 2


if __name__ == "__main__":
    sys.exit(main())
