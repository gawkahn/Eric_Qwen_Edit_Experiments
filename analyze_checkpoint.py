#!/usr/bin/env python3
"""Analyze checkpoint .safetensors files to determine base model compatibility.

Reads only headers (no weights loaded into RAM). Reports:
- Key format (original vs diffusers)
- Architecture family (Flux, Chroma, Chroma-HD, etc.)
- Layer counts (double_blocks, single_blocks, guidance layers)
- Base model recommendation

Usage:
    python3 analyze_checkpoint.py /path/to/model.safetensors [...]
    python3 analyze_checkpoint.py /path/to/directory/
"""

import sys
import os
from safetensors import safe_open


def analyze(path: str) -> dict:
    """Analyze a single safetensors file."""
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())

    info = {
        "file": os.path.basename(path),
        "total_keys": len(keys),
    }

    # ── Key format detection ────────────────────────────────────────────
    orig_keys = [k for k in keys if k.startswith("model.diffusion_model.")]
    bare_keys = [k for k in keys if not k.startswith("model.")]

    if orig_keys:
        info["key_format"] = "original (model.diffusion_model.*)"
        # Strip prefix for analysis
        stripped = [k.replace("model.diffusion_model.", "") for k in keys
                    if k.startswith("model.diffusion_model.")]
    elif any(k.startswith("transformer_blocks.") for k in keys):
        info["key_format"] = "diffusers"
        stripped = keys
    else:
        # Could be bare original format (no model.diffusion_model. prefix)
        if any(k.startswith("double_blocks.") for k in keys):
            info["key_format"] = "original (bare, no prefix)"
            stripped = keys
        else:
            info["key_format"] = "unknown"
            info["sample_keys"] = keys[:10]
            return info

    # ── Architecture analysis ───────────────────────────────────────────
    # Count layer types (works for original-format keys)
    double_blocks = set()
    single_blocks = set()
    guidance_layers = set()

    for k in stripped:
        if k.startswith("double_blocks."):
            try:
                double_blocks.add(int(k.split(".")[1]))
            except (ValueError, IndexError):
                pass
        elif k.startswith("single_blocks."):
            try:
                single_blocks.add(int(k.split(".")[1]))
            except (ValueError, IndexError):
                pass
        elif k.startswith("distilled_guidance_layer.layers."):
            try:
                guidance_layers.add(int(k.split(".")[2]))
            except (ValueError, IndexError):
                pass

    # Also check diffusers-format layer names
    for k in stripped:
        if k.startswith("transformer_blocks."):
            try:
                double_blocks.add(int(k.split(".")[1]))
            except (ValueError, IndexError):
                pass
        elif k.startswith("single_transformer_blocks."):
            try:
                single_blocks.add(int(k.split(".")[1]))
            except (ValueError, IndexError):
                pass

    info["double_blocks"] = len(double_blocks)
    info["single_blocks"] = len(single_blocks)

    # ── Distinguishing features ─────────────────────────────────────────
    has_guidance_embeds = any("guidance" in k and "in_layer" in k for k in stripped)
    has_distilled_guidance = any("distilled_guidance_layer" in k for k in stripped)
    has_pooled_proj = any("pooled_projection" in k.lower() for k in stripped)
    info["has_guidance_embeds"] = has_guidance_embeds
    info["has_distilled_guidance"] = has_distilled_guidance
    info["guidance_layers"] = len(guidance_layers)

    # ── Determine family ────────────────────────────────────────────────
    if has_distilled_guidance:
        info["architecture"] = "Chroma (original, with distilled guidance)"
        info["compatible_base"] = "lodestones/Chroma"
        info["incompatible_with"] = "Chroma1-HD (no distilled_guidance_layer)"
    elif info["double_blocks"] > 0 and has_guidance_embeds:
        info["architecture"] = "Flux-like (with guidance embeds)"
        info["compatible_base"] = "black-forest-labs/FLUX.1-dev"
    elif info["double_blocks"] > 0 and not has_guidance_embeds and not has_distilled_guidance:
        # Could be Chroma-HD or Flux-schnell (no guidance)
        # Check for Chroma-HD specific markers
        has_norm_key_norm = any("norm.key_norm" in k for k in stripped)
        if has_norm_key_norm and not has_pooled_proj:
            info["architecture"] = "Chroma-HD (no guidance embeds, no distilled guidance)"
            info["compatible_base"] = "Chroma1-HD / lodestones/Chroma1-HD"
        elif info["double_blocks"] == 19 and info["single_blocks"] == 38:
            info["architecture"] = "Flux-schnell-like (no guidance embeds)"
            info["compatible_base"] = "black-forest-labs/FLUX.1-schnell"
        else:
            info["architecture"] = "Flux-variant (no guidance embeds)"
            info["compatible_base"] = "unknown — check layer counts"
    elif any(k.startswith("transformer_blocks.") for k in stripped):
        info["architecture"] = "diffusers-native (already converted)"
        if has_distilled_guidance:
            info["compatible_base"] = "lodestones/Chroma"
        else:
            info["compatible_base"] = "check config.json in same directory"
    else:
        info["architecture"] = "unknown"
        info["sample_keys"] = stripped[:10]

    return info


def main():
    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for root, dirs, files in os.walk(arg):
                dirs.sort()
                for f in sorted(files):
                    if f.endswith(".safetensors") and not f.startswith("."):
                        paths.append(os.path.join(root, f))
        elif os.path.isfile(arg) and arg.endswith(".safetensors"):
            paths.append(arg)
        else:
            print(f"Skipping: {arg}")

    if not paths:
        print("Usage: python3 analyze_checkpoint.py <file.safetensors> [...]")
        print("       python3 analyze_checkpoint.py <directory/>")
        sys.exit(1)

    for path in paths:
        print(f"\n{'=' * 70}")
        info = analyze(path)
        for k, v in info.items():
            print(f"  {k:30s}: {v}")
    print()


if __name__ == "__main__":
    main()
