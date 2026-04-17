#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""
Unit tests for the Chroma LoRA conversion plan (slice 5).

Run from project root:

    /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python \
        -m tests.test_lora_format_convert_chroma
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
    decode_kohya_to_bfl,
    get_plan,
)
from nodes.eric_lora_format_convert_apply import (  # noqa: E402
    convert_state_dict,
    find_matching_plan,
)


# ── Synthetic Chroma LoRA key inventory ───────────────────────────────
# Chroma: 19 double_blocks × 8 modules + 38 single_blocks × 2 modules
# = 152 + 76 = 228 unique base modules (matches real Chubby_Body_Type)

def _build_chroma_lora_keys() -> Set[str]:
    keys: Set[str] = set()

    for n in range(19):
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
            for sfx in (".lora_down.weight", ".lora_up.weight", ".alpha"):
                keys.add(stem + sfx)

    for n in range(38):
        for stem in (
            f"diffusion_model.single_blocks.{n}.linear1",
            f"diffusion_model.single_blocks.{n}.linear2",
        ):
            for sfx in (".lora_down.weight", ".lora_up.weight", ".alpha"):
                keys.add(stem + sfx)

    return keys


def _build_chroma_lora_state_dict() -> dict:
    """Synthetic state dict with shape-correct tensors for standard LoRA."""
    sd = {}
    for k in _build_chroma_lora_keys():
        if k.endswith(".alpha"):
            sd[k] = torch.tensor(32.0)
        elif k.endswith(".lora_down.weight"):
            sd[k] = torch.zeros(32, 64)   # rank=32, in=64 (synthetic)
        elif k.endswith(".lora_up.weight"):
            sd[k] = torch.zeros(64, 32)   # out=64, rank=32 (synthetic)
    return sd


# ════════════════════════════════════════════════════════════════════════
#  Plan registration
# ════════════════════════════════════════════════════════════════════════

def test_chroma_plan_is_registered():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    assert plan is not None, "Chroma plan should auto-register on framework import"
    assert plan.source_family == "bfl_chroma"
    assert plan.target_family == "diffusers_chroma"
    assert len(plan.rename_rules) > 0
    assert len(plan.qkv_splits) == 3  # img + txt + linear1
    assert plan.model_signature == "proj_mlp"


def test_chroma_plan_qkv_splits_targets():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    img_split = next(s for s in plan.qkv_splits if "IMG" in s.pattern)
    txt_split = next(s for s in plan.qkv_splits if "TXT" in s.pattern)
    lin_split = next(s for s in plan.qkv_splits if "LINEAR1" in s.pattern)

    assert img_split.targets == (".attn.to_q", ".attn.to_k", ".attn.to_v")
    assert img_split.target_dims is None  # equal split

    assert txt_split.targets == (
        ".attn.add_q_proj", ".attn.add_k_proj", ".attn.add_v_proj",
    )
    assert txt_split.target_dims is None  # equal split

    assert lin_split.targets == (
        ".attn.to_q", ".attn.to_k", ".attn.to_v", ".proj_mlp",
    )
    assert lin_split.target_dims == (3072, 3072, 3072, 12288)


# ════════════════════════════════════════════════════════════════════════
#  Rename — double_blocks
# ════════════════════════════════════════════════════════════════════════

def test_double_block_img_attn_qkv_renames_to_placeholder():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {"diffusion_model.double_blocks.3.img_attn.qkv.lora_down.weight": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.3.attn.__QKV_IMG__.lora_down.weight" in out


def test_double_block_txt_attn_qkv_renames_to_placeholder():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {"diffusion_model.double_blocks.5.txt_attn.qkv.lora_down.weight": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.5.attn.__QKV_TXT__.lora_down.weight" in out


def test_double_block_img_attn_proj_renames():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {"diffusion_model.double_blocks.0.img_attn.proj.lora_down.weight": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.attn.to_out.0.lora_down.weight" in out


def test_double_block_txt_attn_proj_renames():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {"diffusion_model.double_blocks.0.txt_attn.proj.lora_down.weight": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.attn.to_add_out.lora_down.weight" in out


def test_double_block_img_mlp_renames_flux1_style():
    """Chroma uses ff.net.0.proj / ff.net.2 (Flux.1), NOT ff.linear_in/out (Klein)."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {
        "diffusion_model.double_blocks.0.img_mlp.0.lora_down.weight": torch.zeros(1, 1),
        "diffusion_model.double_blocks.0.img_mlp.2.lora_down.weight": torch.zeros(1, 1),
    }
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.ff.net.0.proj.lora_down.weight" in out
    assert "transformer_blocks.0.ff.net.2.lora_down.weight" in out


def test_double_block_txt_mlp_renames_flux1_style():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {
        "diffusion_model.double_blocks.0.txt_mlp.0.lora_down.weight": torch.zeros(1, 1),
        "diffusion_model.double_blocks.0.txt_mlp.2.lora_down.weight": torch.zeros(1, 1),
    }
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.ff_context.net.0.proj.lora_down.weight" in out
    assert "transformer_blocks.0.ff_context.net.2.lora_down.weight" in out


def test_double_block_norm_renames():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {
        "diffusion_model.double_blocks.0.img_attn.norm.q_norm.scale": torch.zeros(1),
        "diffusion_model.double_blocks.0.txt_attn.norm.k_norm.scale": torch.zeros(1),
    }
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.attn.norm_q.scale" in out
    assert "transformer_blocks.0.attn.norm_added_k.scale" in out


# ════════════════════════════════════════════════════════════════════════
#  Rename — single_blocks
# ════════════════════════════════════════════════════════════════════════

def test_single_block_linear1_renames_to_placeholder():
    """linear1 should become a placeholder for 4-way split, not a final name."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {"diffusion_model.single_blocks.7.linear1.lora_down.weight": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "single_transformer_blocks.7.__LINEAR1__.lora_down.weight" in out


def test_single_block_linear2_renames_to_proj_out():
    """linear2 → proj_out (1:1 rename, no split)."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {"diffusion_model.single_blocks.20.linear2.lora_down.weight": torch.zeros(1, 1)}
    out = apply_rename_rules(sd, plan)
    assert "single_transformer_blocks.20.proj_out.lora_down.weight" in out


def test_single_block_norm_renames():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = {
        "diffusion_model.single_blocks.0.norm.q_norm.scale": torch.zeros(1),
        "diffusion_model.single_blocks.0.norm.key_norm.scale": torch.zeros(1),
    }
    out = apply_rename_rules(sd, plan)
    assert "single_transformer_blocks.0.attn.norm_q.scale" in out
    assert "single_transformer_blocks.0.attn.norm_k.scale" in out


# ════════════════════════════════════════════════════════════════════════
#  End-to-end rename: full synthetic Chroma LoRA
# ════════════════════════════════════════════════════════════════════════

def test_full_chroma_lora_shape_after_rename():
    """228 modules × 3 suffixes = 684 keys in.  Same count out (rename only)."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = _build_chroma_lora_state_dict()
    assert len(sd) == 228 * 3

    out = apply_rename_rules(sd, plan)
    assert len(out) == len(sd)


def test_full_chroma_lora_no_bfl_prefixes_remain():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = _build_chroma_lora_state_dict()
    out = apply_rename_rules(sd, plan)
    leftover = [k for k in out if k.startswith("diffusion_model.")]
    assert not leftover, f"prefix-strip failed for: {leftover[:3]}"


def test_full_chroma_lora_keys_match_diffusers_namespace():
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = _build_chroma_lora_state_dict()
    out = apply_rename_rules(sd, plan)

    valid_prefixes = ("transformer_blocks.", "single_transformer_blocks.")
    bad = []
    for k in out:
        for sfx in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if k.endswith(sfx):
                base = k[: -len(sfx)]
                break
        else:
            base = k
        if not base.startswith(valid_prefixes):
            bad.append(k)
    assert not bad, f"Renamed keys outside diffusers namespace: {bad[:5]}"


def test_full_chroma_lora_placeholders_present():
    """QKV and LINEAR1 placeholders should exist after rename."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    sd = _build_chroma_lora_state_dict()
    out = apply_rename_rules(sd, plan)

    img_qkv = [k for k in out if "__QKV_IMG__" in k]
    txt_qkv = [k for k in out if "__QKV_TXT__" in k]
    lin1    = [k for k in out if "__LINEAR1__" in k]
    # 19 blocks × 3 suffixes = 57 keys per QKV placeholder
    assert len(img_qkv) == 19 * 3, f"expected 57 img-qkv, got {len(img_qkv)}"
    assert len(txt_qkv) == 19 * 3, f"expected 57 txt-qkv, got {len(txt_qkv)}"
    # 38 blocks × 3 suffixes = 114 linear1 placeholders
    assert len(lin1) == 38 * 3, f"expected 114 linear1 placeholders, got {len(lin1)}"


# ════════════════════════════════════════════════════════════════════════
#  convert_state_dict — 4-way single-block split
# ════════════════════════════════════════════════════════════════════════

def test_convert_chroma_linear1_4way_split():
    """Single-block linear1 must split lora_B into Q(3072)+K(3072)+V(3072)+MLP(12288)."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    rank, in_dim = 32, 128  # synthetic
    # fused output dim must match target_dims sum = 21504
    fused_out = 3072 + 3072 + 3072 + 12288  # = 21504
    A = torch.randn(rank, in_dim)
    B = torch.randn(fused_out, rank)
    sd = {
        "diffusion_model.single_blocks.0.linear1.lora_down.weight": A,
        "diffusion_model.single_blocks.0.linear1.lora_up.weight": B,
    }
    out = convert_state_dict(sd, plan)

    # Should produce 4 target modules
    expected = [
        ("single_transformer_blocks.0.attn.to_q", 3072),
        ("single_transformer_blocks.0.attn.to_k", 3072),
        ("single_transformer_blocks.0.attn.to_v", 3072),
        ("single_transformer_blocks.0.proj_mlp",  12288),
    ]
    for base, dim in expected:
        a_key = f"{base}.lora_A.weight"
        b_key = f"{base}.lora_B.weight"
        assert a_key in out, f"missing {a_key}"
        assert b_key in out, f"missing {b_key}"
        assert out[a_key].shape == (rank, in_dim), f"bad A shape for {base}"
        assert out[b_key].shape[0] == dim, f"bad B dim for {base}: {out[b_key].shape[0]} != {dim}"

    # No placeholder leakage
    assert not any("__LINEAR1__" in k for k in out)


def test_convert_chroma_linear1_split_reconstructs_original():
    """Concatenating the 4 split B slices should recover the original B."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    rank, in_dim = 32, 128
    fused_out = 21504
    A = torch.randn(rank, in_dim)
    B = torch.randn(fused_out, rank)
    # No alpha → no baking → B slices should be exact
    sd = {
        "diffusion_model.single_blocks.5.linear1.lora_down.weight": A,
        "diffusion_model.single_blocks.5.linear1.lora_up.weight": B,
    }
    out = convert_state_dict(sd, plan)

    B_cat = torch.cat([
        out["single_transformer_blocks.5.attn.to_q.lora_B.weight"],
        out["single_transformer_blocks.5.attn.to_k.lora_B.weight"],
        out["single_transformer_blocks.5.attn.to_v.lora_B.weight"],
        out["single_transformer_blocks.5.proj_mlp.lora_B.weight"],
    ], dim=0)
    assert torch.allclose(B_cat, B), \
        f"max err {(B_cat - B).abs().max()}"


def test_convert_chroma_double_block_qkv_split_3way():
    """Double-block QKV (3-way equal) still works with the generalized code."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    rank, in_dim, head_dim = 32, 128, 512
    A = torch.randn(rank, in_dim)
    B = torch.randn(3 * head_dim, rank)
    sd = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": A,
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight": B,
    }
    out = convert_state_dict(sd, plan)

    for tgt in ("to_q", "to_k", "to_v"):
        base = f"transformer_blocks.0.attn.{tgt}"
        assert f"{base}.lora_A.weight" in out
        assert f"{base}.lora_B.weight" in out
        assert out[f"{base}.lora_B.weight"].shape[0] == head_dim
    assert not any("__QKV_IMG__" in k for k in out)


def test_convert_chroma_linear2_pass_through():
    """linear2 → proj_out: 1:1 rename, no split."""
    plan = get_plan("bfl_chroma", "diffusers_chroma")
    rank, in_dim, out_dim = 32, 128, 64
    A = torch.randn(rank, in_dim)
    B = torch.randn(out_dim, rank)
    sd = {
        "diffusion_model.single_blocks.10.linear2.lora_down.weight": A,
        "diffusion_model.single_blocks.10.linear2.lora_up.weight": B,
    }
    out = convert_state_dict(sd, plan)
    base = "single_transformer_blocks.10.proj_out"
    assert f"{base}.lora_A.weight" in out
    assert f"{base}.lora_B.weight" in out


# ════════════════════════════════════════════════════════════════════════
#  find_matching_plan — Chroma model detection
# ════════════════════════════════════════════════════════════════════════

def test_find_plan_chroma_lora_with_chroma_model():
    """A bfl_original LoRA + a model with `proj_mlp` → Chroma plan."""
    lora_sd = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": torch.zeros(1, 1),
    }
    model_params = [
        "transformer_blocks.0.attn.to_q.weight",
        "single_transformer_blocks.0.proj_mlp.weight",
        "single_transformer_blocks.0.proj_out.weight",
    ]
    plan = find_matching_plan(lora_sd, model_params)
    assert plan is not None
    assert plan.target_family == "diffusers_chroma"


def test_find_plan_chroma_lora_with_klein_model_returns_klein():
    """Same bfl_original LoRA but model has `to_qkv_mlp_proj` → Klein plan (not Chroma)."""
    lora_sd = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": torch.zeros(1, 1),
    }
    model_params = [
        "transformer_blocks.0.attn.to_q.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
    ]
    plan = find_matching_plan(lora_sd, model_params)
    assert plan is not None
    assert plan.target_family == "diffusers_klein"


# ════════════════════════════════════════════════════════════════════════
#  Kohya decode — lora_unet_* → BFL dot format
# ════════════════════════════════════════════════════════════════════════

def test_kohya_decode_double_block():
    sd = {"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": torch.zeros(1)}
    out = decode_kohya_to_bfl(sd)
    assert "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight" in out


def test_kohya_decode_single_block():
    sd = {"lora_unet_single_blocks_5_linear1.lora_down.weight": torch.zeros(1)}
    out = decode_kohya_to_bfl(sd)
    assert "diffusion_model.single_blocks.5.linear1.lora_down.weight" in out


def test_kohya_decode_alpha():
    sd = {"lora_unet_double_blocks_0_img_mlp_0.alpha": torch.tensor(32.0)}
    out = decode_kohya_to_bfl(sd)
    assert "diffusion_model.double_blocks.0.img_mlp.0.alpha" in out


def test_kohya_decode_non_unet_passthrough():
    sd = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": torch.zeros(1),
        "some_other_key": torch.zeros(1),
    }
    out = decode_kohya_to_bfl(sd)
    # Non-lora_unet keys pass through unchanged
    assert "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight" in out
    assert "some_other_key" in out


def test_kohya_decode_norm_compound():
    """q_norm and k_norm internal underscores must be preserved."""
    sd = {"lora_unet_double_blocks_0_img_attn_norm_q_norm.scale": torch.zeros(1)}
    out = decode_kohya_to_bfl(sd)
    assert "diffusion_model.double_blocks.0.img_attn.norm.q_norm.scale" in out


def test_kohya_decode_txt_mlp():
    sd = {"lora_unet_double_blocks_3_txt_mlp_2.lora_up.weight": torch.zeros(1)}
    out = decode_kohya_to_bfl(sd)
    assert "diffusion_model.double_blocks.3.txt_mlp.2.lora_up.weight" in out


def test_kohya_decoded_detected_as_bfl_original():
    """After Kohya decode, detect_lora_format should recognise bfl_original."""
    from nodes.eric_lora_format_convert import detect_lora_format
    sd = {
        "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": torch.zeros(1),
        "lora_unet_single_blocks_0_linear1.lora_down.weight": torch.zeros(1),
    }
    decoded = decode_kohya_to_bfl(sd)
    family = detect_lora_format(decoded.keys())
    assert family == "bfl_original"


# ════════════════════════════════════════════════════════════════════════
#  Full pipeline: Kohya decode → find_matching_plan → convert
# ════════════════════════════════════════════════════════════════════════

def test_kohya_chroma_lora_end_to_end():
    """A Kohya-format Chroma LoRA should decode, match, and convert."""
    rank, in_dim = 8, 64
    fused_out = 21504  # 3*3072 + 12288
    A = torch.randn(rank, in_dim)
    B = torch.randn(fused_out, rank)
    sd = {
        "lora_unet_single_blocks_0_linear1.lora_down.weight": A,
        "lora_unet_single_blocks_0_linear1.lora_up.weight": B,
    }
    model_params = [
        "transformer_blocks.0.attn.to_q.weight",
        "single_transformer_blocks.0.proj_mlp.weight",
    ]

    decoded = decode_kohya_to_bfl(sd)
    plan = find_matching_plan(decoded, model_params)
    assert plan is not None
    assert plan.target_family == "diffusers_chroma"

    converted = convert_state_dict(decoded, plan)
    assert "single_transformer_blocks.0.attn.to_q.lora_A.weight" in converted
    assert "single_transformer_blocks.0.proj_mlp.lora_B.weight" in converted
    assert converted["single_transformer_blocks.0.proj_mlp.lora_B.weight"].shape[0] == 12288


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
