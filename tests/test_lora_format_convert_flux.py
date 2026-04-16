#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""
Unit tests for the Klein-9B / Flux.2 LoRA conversion plan (slice 3).

Run from project root:

    /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python \
        -m tests.test_lora_format_convert_flux

Uses the same shim/sys-path setup as the slice 2 tests.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Set


# ── sys.path + ComfyUI shims (mirror lora_test_harness) ────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


import torch  # noqa: E402

from nodes.eric_lora_format_convert import (  # noqa: E402
    apply_rename_rules,
    get_plan,
)


# ── Synthetic Klein LoRA key inventory ─────────────────────────────────
# Mirrors the actual klein_snofs_v1_1.safetensors structure exactly:
#  - 8 double_blocks (0..7) × 8 module/block (img+txt × {qkv, proj, mlp.0, mlp.2})
#  - 24 single_blocks (0..23) × 2 modules (linear1, linear2)
# = 64 + 48 = 112 unique base modules.
#
# Each gets a `.lokr_w1`, `.lokr_w2`, and `.alpha` suffix.  Tensor data
# is irrelevant for the rename test — we use zero tensors.

def _build_klein_lora_keys() -> Set[str]:
    keys: Set[str] = set()

    for n in range(8):
        for stem in (
            f"diffusion_model.double_blocks.{n}.img_attn.qkv",
            f"diffusion_model.double_blocks.{n}.img_attn.proj",
            f"diffusion_model.double_blocks.{n}.img_mlp.0",
            f"diffusion_model.double_blocks.{n}.img_mlp.2",
            f"diffusion_model.double_blocks.{n}.txt_attn.qkv",
            f"diffusion_model.double_blocks.{n}.txt_attn.proj",
            f"diffusion_model.double_blocks.{n}.txt_mlp.0",
            f"diffusion_model.double_blocks.{n}.txt_mlp.2",
        ):
            for sfx in (".lokr_w1", ".lokr_w2", ".alpha"):
                keys.add(stem + sfx)

    for n in range(24):
        for stem in (
            f"diffusion_model.single_blocks.{n}.linear1",
            f"diffusion_model.single_blocks.{n}.linear2",
        ):
            for sfx in (".lokr_w1", ".lokr_w2", ".alpha"):
                keys.add(stem + sfx)

    return keys


def _build_klein_lora_state_dict() -> dict:
    """Convert the synthetic key set into a state dict with shape-correct tensors.

    Shapes mirror what we observed on the real LoRA so that any future
    shape-aware rename rules wouldn't be silently masked by zero-rank
    placeholders.  The actual values don't matter for these tests.
    """
    sd = {}
    for k in _build_klein_lora_keys():
        if k.endswith(".alpha"):
            sd[k] = torch.tensor(4.0)
        elif k.endswith(".lokr_w1"):
            sd[k] = torch.zeros(4, 4)
        elif k.endswith(".lokr_w2"):
            # Choose a representative shape — content irrelevant
            sd[k] = torch.zeros(1024, 1024)
    return sd


# ════════════════════════════════════════════════════════════════════════
#  Plan registration
# ════════════════════════════════════════════════════════════════════════

def test_klein_plan_is_registered():
    plan = get_plan("bfl_klein", "diffusers_klein")
    assert plan is not None, "Klein plan should auto-register on framework import"
    assert plan.source_family == "bfl_klein"
    assert plan.target_family == "diffusers_klein"
    assert len(plan.rename_rules) > 0
    assert len(plan.qkv_splits) == 2  # img stream + txt stream
    # Slice 4 plan-matching uses model_signature to distinguish Klein/Flux2
    # from Flux.1 diffusers (which lacks the fused to_qkv_mlp_proj layer).
    assert plan.model_signature == "to_qkv_mlp_proj"


def test_klein_plan_qkv_splits_use_distinct_targets():
    plan = get_plan("bfl_klein", "diffusers_klein")
    img_split = next(s for s in plan.qkv_splits if "IMG" in s.pattern)
    txt_split = next(s for s in plan.qkv_splits if "TXT" in s.pattern)
    # img-stream → standard split QKV
    assert img_split.targets == (".attn.to_q", ".attn.to_k", ".attn.to_v")
    # txt-stream → context add_*_proj
    assert txt_split.targets == (
        ".attn.add_q_proj", ".attn.add_k_proj", ".attn.add_v_proj",
    )


# ════════════════════════════════════════════════════════════════════════
#  Rename — double_blocks
# ════════════════════════════════════════════════════════════════════════

def test_double_block_img_attn_qkv_renames_to_placeholder():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {"diffusion_model.double_blocks.3.img_attn.qkv.lokr_w2": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    # Placeholder waits for slice 4 to expand into to_q/to_k/to_v
    assert "transformer_blocks.3.attn.__QKV_IMG__.lokr_w2" in out


def test_double_block_txt_attn_qkv_renames_to_placeholder():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {"diffusion_model.double_blocks.5.txt_attn.qkv.lokr_w2": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.5.attn.__QKV_TXT__.lokr_w2" in out


def test_double_block_img_attn_proj_renames():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {"diffusion_model.double_blocks.0.img_attn.proj.lokr_w2": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.attn.to_out.0.lokr_w2" in out


def test_double_block_txt_attn_proj_renames():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {"diffusion_model.double_blocks.0.txt_attn.proj.lokr_w2": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.attn.to_add_out.lokr_w2" in out


def test_double_block_img_mlp_renames():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {
        "diffusion_model.double_blocks.0.img_mlp.0.lokr_w2": torch.zeros(1, 1),
        "diffusion_model.double_blocks.0.img_mlp.2.lokr_w2": torch.zeros(1, 1),
    }
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.ff.linear_in.lokr_w2"  in out
    assert "transformer_blocks.0.ff.linear_out.lokr_w2" in out


def test_double_block_txt_mlp_renames():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {
        "diffusion_model.double_blocks.0.txt_mlp.0.lokr_w2": torch.zeros(1, 1),
        "diffusion_model.double_blocks.0.txt_mlp.2.lokr_w2": torch.zeros(1, 1),
    }
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.ff_context.linear_in.lokr_w2"  in out
    assert "transformer_blocks.0.ff_context.linear_out.lokr_w2" in out


def test_double_block_norm_renames_both_spellings():
    """ai-toolkit emits .q_norm/.k_norm; legacy tools emit .query_norm/.key_norm."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {
        "diffusion_model.double_blocks.0.img_attn.norm.q_norm.scale":
            torch.zeros(1),
        "diffusion_model.double_blocks.0.img_attn.norm.query_norm.scale":
            torch.zeros(1),
        "diffusion_model.double_blocks.0.txt_attn.norm.k_norm.scale":
            torch.zeros(1),
        "diffusion_model.double_blocks.0.txt_attn.norm.key_norm.scale":
            torch.zeros(1),
    }
    out = apply_rename_rules(sd, plan)
    # Both spellings collapse onto the same diffusers names
    out_keys = list(out.keys())
    # img → norm_q, txt → norm_added_k (both spellings)
    assert any("transformer_blocks.0.attn.norm_q.scale" == k for k in out_keys), out_keys
    assert any("transformer_blocks.0.attn.norm_added_k.scale" == k for k in out_keys), out_keys


# ════════════════════════════════════════════════════════════════════════
#  Rename — single_blocks (1:1, no QKV split)
# ════════════════════════════════════════════════════════════════════════

def test_single_block_linear1_renames_to_fused():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {"diffusion_model.single_blocks.7.linear1.lokr_w2": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "single_transformer_blocks.7.attn.to_qkv_mlp_proj.lokr_w2" in out


def test_single_block_linear2_renames_to_fused_out():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {"diffusion_model.single_blocks.20.linear2.lokr_w2": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "single_transformer_blocks.20.attn.to_out.lokr_w2" in out


def test_single_block_norm_renames():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = {
        "diffusion_model.single_blocks.0.norm.q_norm.scale":     torch.zeros(1),
        "diffusion_model.single_blocks.0.norm.key_norm.scale":   torch.zeros(1),
    }
    out = apply_rename_rules(sd, plan)
    assert "single_transformer_blocks.0.attn.norm_q.scale" in out
    assert "single_transformer_blocks.0.attn.norm_k.scale" in out


# ════════════════════════════════════════════════════════════════════════
#  End-to-end: synthetic full klein_snofs LoRA → expected output shape
# ════════════════════════════════════════════════════════════════════════

def test_full_klein_lora_shape_after_rename():
    """All 112 modules × 3 suffixes = 336 keys in.  Same key count out
    (rename only, no QKV expansion yet — that's slice 4)."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _build_klein_lora_state_dict()
    assert len(sd) == 112 * 3  # sanity: we built what we think we built

    out = apply_rename_rules(sd, plan)
    assert len(out) == len(sd)


def test_full_klein_lora_no_diffusion_model_prefix_remains():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _build_klein_lora_state_dict()
    out = apply_rename_rules(sd, plan)
    leftover = [k for k in out if k.startswith("diffusion_model.")]
    assert not leftover, f"prefix-strip failed for: {leftover[:3]}"


def test_full_klein_lora_no_double_or_single_blocks_remain():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _build_klein_lora_state_dict()
    out = apply_rename_rules(sd, plan)
    bad = [k for k in out if (
        "double_blocks." in k and "transformer_blocks." not in k
    ) or (
        "single_blocks." in k and "single_transformer_blocks." not in k
    )]
    assert not bad, f"block-prefix rename missed: {bad[:3]}"


def test_full_klein_lora_keys_match_diffusers_namespace():
    """Every renamed base key (with suffix stripped) must start with
    one of the known diffusers Klein namespaces.  Catches stray rename
    misses that would slip past the negative tests above."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _build_klein_lora_state_dict()
    out = apply_rename_rules(sd, plan)

    valid_prefixes = (
        "transformer_blocks.",
        "single_transformer_blocks.",
    )
    bad = []
    for k in out:
        # strip trailing adapter suffix
        for sfx in (".lokr_w1", ".lokr_w2", ".lokr_t2", ".alpha",
                    ".lora_A.weight", ".lora_B.weight"):
            if k.endswith(sfx):
                base = k[: -len(sfx)]
                break
        else:
            base = k
        if not base.startswith(valid_prefixes):
            bad.append(k)
    assert not bad, f"Renamed keys outside diffusers namespace: {bad[:5]}"


def test_full_klein_lora_qkv_modules_use_placeholders():
    """The 8 img + 8 txt QKV modules should land at placeholder paths —
    slice 4 expands them via QKVSplitSpec into 3 separate keys each."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _build_klein_lora_state_dict()
    out = apply_rename_rules(sd, plan)

    img_qkv = [k for k in out if "__QKV_IMG__" in k]
    txt_qkv = [k for k in out if "__QKV_TXT__" in k]
    # 8 blocks × 3 suffixes (lokr_w1, lokr_w2, alpha) = 24 keys per stream
    assert len(img_qkv) == 8 * 3, f"expected 24 img-qkv placeholders, got {len(img_qkv)}"
    assert len(txt_qkv) == 8 * 3, f"expected 24 txt-qkv placeholders, got {len(txt_qkv)}"


# ════════════════════════════════════════════════════════════════════════
#  Test runner
# ════════════════════════════════════════════════════════════════════════

def _run_all() -> int:
    funcs = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for n, f in funcs:
        try:
            f()
            print(f"  PASS   {n}")
        except AssertionError as e:
            failed.append(n)
            print(f"  FAIL   {n}: {e}")
        except Exception as e:
            failed.append(n)
            print(f"  ERROR  {n}: {type(e).__name__}: {e}")
    print(f"\n{len(funcs) - len(failed)}/{len(funcs)} tests passed.")
    if failed:
        print(f"Failures: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run_all())
