#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""
Unit tests for nodes/eric_lora_format_convert.py (slice 2 framework).

Run from project root:

    /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python \
        -m tests.test_lora_format_convert

No ComfyUI required — uses the same folder_paths / comfy.utils shims
as the LoRA test harness.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


# ── Project root on sys.path + ComfyUI shims (mirror lora_test_harness) ─

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
    CONVERSION_PLANS,
    ConversionPlan,
    RenameRule,
    apply_rename_rules,
    detect_lora_format,
    detect_model_format,
    get_plan,
    register_plan,
    split_fused_qkv_lora,
    split_fused_qkv_via_svd,
    split_state_key,
)


# ════════════════════════════════════════════════════════════════════════
#  Format detection
# ════════════════════════════════════════════════════════════════════════

def test_detect_format_bfl_double_blocks():
    keys = ["diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight"]
    assert detect_lora_format(keys) == "bfl_original"


def test_detect_format_bfl_single_blocks():
    keys = ["diffusion_model.single_blocks.7.linear1.lora_down.weight"]
    assert detect_lora_format(keys) == "bfl_original"


def test_detect_format_diffusers_dit():
    keys = ["transformer_blocks.0.attn.to_q.lora_A.weight"]
    assert detect_lora_format(keys) == "diffusers_dit"


def test_detect_format_diffusers_single_transformer():
    keys = ["single_transformer_blocks.3.attn.to_v.lora_A.weight"]
    assert detect_lora_format(keys) == "diffusers_dit"


def test_detect_format_sd_unet():
    keys = ["unet.up_blocks.0.attentions.0.to_q.lora_A.weight",
            "unet.down_blocks.1.resnets.0.conv1.lora_A.weight"]
    assert detect_lora_format(keys) == "sd_unet"


def test_detect_format_unknown():
    assert detect_lora_format(["some.weird.key.lora_A.weight"]) == "unknown"
    assert detect_lora_format([]) == "unknown"


def test_detect_model_format_alias():
    # detect_model_format should behave identically to detect_lora_format
    names = ["transformer_blocks.0.attn.to_q.weight"]
    assert detect_model_format(names) == "diffusers_dit"


# ════════════════════════════════════════════════════════════════════════
#  Key splitting
# ════════════════════════════════════════════════════════════════════════

def test_split_state_key_lora():
    base, sfx = split_state_key("transformer_blocks.0.attn.to_q.lora_A.weight")
    assert base == "transformer_blocks.0.attn.to_q"
    assert sfx == ".lora_A.weight"


def test_split_state_key_lokr():
    base, sfx = split_state_key("double_blocks.0.img_attn.qkv.lokr_w1")
    assert base == "double_blocks.0.img_attn.qkv"
    assert sfx == ".lokr_w1"


def test_split_state_key_alpha():
    base, sfx = split_state_key("double_blocks.0.img_attn.qkv.alpha")
    assert base == "double_blocks.0.img_attn.qkv"
    assert sfx == ".alpha"


def test_split_state_key_unknown_suffix():
    base, sfx = split_state_key("some.weird.key")
    assert base == "some.weird.key"
    assert sfx == ""


# ════════════════════════════════════════════════════════════════════════
#  Rename rule application
# ════════════════════════════════════════════════════════════════════════

def test_rename_rule_substring():
    rule = RenameRule("double_blocks.", "transformer_blocks.")
    assert rule.apply("diffusion_model.double_blocks.0.img_attn.qkv") \
           == "diffusion_model.transformer_blocks.0.img_attn.qkv"


def test_rename_rule_regex():
    rule = RenameRule(r"single_blocks\.(\d+)\.",
                      r"single_transformer_blocks.\1.",
                      regex=True)
    assert rule.apply("single_blocks.42.linear1") \
           == "single_transformer_blocks.42.linear1"


def test_apply_rename_rules_chain():
    # Two rules applied in order, both fire.
    sd = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight":
            torch.zeros(4, 8),
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight":
            torch.zeros(24, 4),
    }
    plan = ConversionPlan(
        source_family="bfl_original",
        target_family="diffusers_flux2",
        rename_rules=[
            RenameRule("diffusion_model.", ""),
            RenameRule("double_blocks.",  "transformer_blocks."),
        ],
    )
    out = apply_rename_rules(sd, plan)
    assert "transformer_blocks.0.img_attn.qkv.lora_A.weight" in out
    assert "transformer_blocks.0.img_attn.qkv.lora_B.weight" in out
    # Tensors carried through unchanged
    assert out["transformer_blocks.0.img_attn.qkv.lora_A.weight"].shape \
           == (4, 8)
    assert out["transformer_blocks.0.img_attn.qkv.lora_B.weight"].shape \
           == (24, 4)


def test_apply_rename_rules_preserves_unmatched_keys():
    # Keys that match no rule pass through with their original name.
    sd = {"unrelated.path.lora_A.weight": torch.zeros(2, 3)}
    plan = ConversionPlan(
        source_family="bfl_original",
        target_family="diffusers_flux2",
        rename_rules=[RenameRule("does_not_exist", "x")],
    )
    out = apply_rename_rules(sd, plan)
    assert "unrelated.path.lora_A.weight" in out


# ════════════════════════════════════════════════════════════════════════
#  QKV split — exact (standard LoRA)
# ════════════════════════════════════════════════════════════════════════

def test_split_fused_qkv_lora_shapes_and_split_points():
    rank, in_dim, out_dim = 4, 16, 8
    A = torch.randn(rank, in_dim)
    B = torch.randn(3 * out_dim, rank)
    parts = split_fused_qkv_lora(A, B)

    assert set(parts) == {"q", "k", "v"}
    for name in ("q", "k", "v"):
        assert parts[name]["lora_A"].shape == (rank, in_dim)
        assert parts[name]["lora_B"].shape == (out_dim, rank)
        # lora_A is shared (same values across q/k/v)
        assert torch.allclose(parts[name]["lora_A"], A)

    # lora_B comes from the right slice
    assert torch.allclose(parts["q"]["lora_B"], B[0:out_dim])
    assert torch.allclose(parts["k"]["lora_B"], B[out_dim:2 * out_dim])
    assert torch.allclose(parts["v"]["lora_B"], B[2 * out_dim:3 * out_dim])


def test_split_fused_qkv_lora_reconstruction_is_exact():
    """Concatenating the three split deltas must equal the original delta."""
    rank, in_dim, out_dim = 4, 16, 8
    A = torch.randn(rank, in_dim)
    B = torch.randn(3 * out_dim, rank)
    parts = split_fused_qkv_lora(A, B)

    full_delta = B @ A  # (3*out, in)
    reconstructed = torch.cat(
        [parts[k]["lora_B"] @ parts[k]["lora_A"] for k in ("q", "k", "v")],
        dim=0,
    )
    assert torch.allclose(full_delta, reconstructed)


def test_split_fused_qkv_lora_passes_alpha():
    A = torch.randn(2, 4)
    B = torch.randn(6, 2)
    alpha = torch.tensor(8.0)
    parts = split_fused_qkv_lora(A, B, alpha=alpha)
    for name in ("q", "k", "v"):
        assert "alpha" in parts[name]
        assert torch.equal(parts[name]["alpha"], alpha)


def test_split_fused_qkv_lora_rejects_non_divisible():
    A = torch.randn(2, 4)
    B = torch.randn(7, 2)  # 7 not divisible by 3
    try:
        split_fused_qkv_lora(A, B)
    except ValueError as e:
        assert "divisible by 3" in str(e)
    else:
        raise AssertionError("expected ValueError for non-divisible out_dim")


# ════════════════════════════════════════════════════════════════════════
#  QKV split — SVD (LoKR/LoHa path)
# ════════════════════════════════════════════════════════════════════════

def test_split_fused_qkv_via_svd_recovers_low_rank():
    """If the merged delta IS rank-r, SVD with target_rank=r reconstructs exactly."""
    in_dim, out_dim, true_rank = 16, 8, 4
    A_true = torch.randn(true_rank, in_dim, dtype=torch.float64)
    B_true = torch.randn(3 * out_dim, true_rank, dtype=torch.float64)
    merged = (B_true @ A_true).to(torch.float32)

    parts = split_fused_qkv_via_svd(merged, target_rank=true_rank)

    for name, lo in (("q", 0), ("k", out_dim), ("v", 2 * out_dim)):
        # alpha=rank → effective scale at runtime is 1.0
        assert int(parts[name]["alpha"].item()) == true_rank
        # Reconstruction = B @ A; should match the original block
        recon = parts[name]["lora_B"] @ parts[name]["lora_A"]
        original = merged[lo:lo + out_dim]
        max_err = (recon - original).abs().max().item()
        assert max_err < 5e-4, \
            f"{name}: max err {max_err} exceeds tolerance for rank-{true_rank}"


def test_split_fused_qkv_via_svd_truncates_to_target_rank():
    in_dim, out_dim, true_rank, target_rank = 16, 8, 6, 2
    A_true = torch.randn(true_rank, in_dim)
    B_true = torch.randn(3 * out_dim, true_rank)
    merged = B_true @ A_true

    parts = split_fused_qkv_via_svd(merged, target_rank=target_rank)
    for name in ("q", "k", "v"):
        # The truncated factors should have target_rank as the inner dim
        assert parts[name]["lora_A"].shape == (target_rank, in_dim)
        assert parts[name]["lora_B"].shape == (out_dim, target_rank)


# ════════════════════════════════════════════════════════════════════════
#  Plan registry
# ════════════════════════════════════════════════════════════════════════

def test_registry_starts_empty_or_unmodified():
    # Slice 2 ships with no plans registered; slices 3/5 add them.
    # We don't assert "exactly empty" because slice 3/5 may have landed
    # already by the time these tests run — instead, just verify that
    # the get_plan API behaves on an unregistered key.
    assert get_plan("does_not_exist", "also_no") is None


def test_register_and_retrieve_plan():
    plan = ConversionPlan(
        source_family="__pytest_src__",
        target_family="__pytest_dst__",
        rename_rules=[RenameRule("a", "b")],
    )
    register_plan(plan)
    try:
        assert get_plan("__pytest_src__", "__pytest_dst__") is plan
    finally:
        # Clean up so we don't pollute the registry for sibling tests
        CONVERSION_PLANS.pop(("__pytest_src__", "__pytest_dst__"), None)


# ════════════════════════════════════════════════════════════════════════
#  Test runner
# ════════════════════════════════════════════════════════════════════════

def _run_all() -> int:
    funcs = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed: List[str] = []
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
