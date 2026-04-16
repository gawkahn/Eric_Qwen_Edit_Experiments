#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""
Unit tests for the LoRA conversion apply step (slice 4).

Run from project root:

    /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python \
        -m tests.test_lora_format_convert_apply
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# ── sys.path + ComfyUI shims ────────────────────────────────────────────

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

from nodes.eric_lora_format_convert import get_plan  # noqa: E402
from nodes.eric_lora_format_convert_apply import (  # noqa: E402
    convert_state_dict,
    find_matching_plan,
    reconstruct_lokr_delta,
    svd_compress_to_lora,
)


# ════════════════════════════════════════════════════════════════════════
#  reconstruct_lokr_delta
# ════════════════════════════════════════════════════════════════════════

def test_lokr_reconstruct_kron_with_alpha_scaling():
    # alpha=4, w1.shape=(4,4) → r=4, scale = 4/4 = 1.0
    w1 = torch.ones(4, 4)
    w2 = torch.eye(8)
    alpha = torch.tensor(4.0)
    delta = reconstruct_lokr_delta(w1, w2, alpha)
    expected = torch.kron(w1.float(), w2.float())  # scale = 1.0
    assert torch.allclose(delta, expected)


def test_lokr_reconstruct_no_alpha_defaults_to_unscaled():
    w1 = torch.randn(4, 4)
    w2 = torch.randn(8, 8)
    delta = reconstruct_lokr_delta(w1, w2, alpha=None)
    expected = torch.kron(w1.float(), w2.float())
    assert torch.allclose(delta, expected)


def test_lokr_reconstruct_alpha_lt_rank_scales_down():
    # r = min(4,4) = 4;  alpha=2 → scale = 2/4 = 0.5
    w1 = torch.ones(4, 4)
    w2 = torch.eye(2)
    alpha = torch.tensor(2.0)
    delta = reconstruct_lokr_delta(w1, w2, alpha)
    expected = torch.kron(w1.float(), w2.float()) * 0.5
    assert torch.allclose(delta, expected)


# ════════════════════════════════════════════════════════════════════════
#  svd_compress_to_lora
# ════════════════════════════════════════════════════════════════════════

def test_svd_compress_recovers_low_rank_input():
    # If the input is rank-r, SVD with target_rank=r reconstructs exactly.
    in_dim, out_dim, true_rank = 16, 8, 4
    A_true = torch.randn(true_rank, in_dim, dtype=torch.float64)
    B_true = torch.randn(out_dim,  true_rank, dtype=torch.float64)
    delta = (B_true @ A_true).to(torch.float32)

    A, B, alpha = svd_compress_to_lora(delta, target_rank=true_rank)
    assert A.shape == (true_rank, in_dim)
    assert B.shape == (out_dim, true_rank)
    assert int(alpha.item()) == true_rank
    recon = B @ A
    assert (recon - delta).abs().max().item() < 5e-4


def test_svd_compress_truncates_when_target_rank_smaller():
    in_dim, out_dim, target = 16, 8, 2
    delta = torch.randn(out_dim, in_dim)
    A, B, alpha = svd_compress_to_lora(delta, target_rank=target)
    assert A.shape == (target, in_dim)
    assert B.shape == (out_dim, target)
    assert int(alpha.item()) == target


# ════════════════════════════════════════════════════════════════════════
#  find_matching_plan
# ════════════════════════════════════════════════════════════════════════

def test_find_plan_klein_lora_with_klein_model_returns_klein_plan():
    """A bfl_original LoRA + a model with `to_qkv_mlp_proj` parameter → Klein plan."""
    lora_sd = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2": torch.zeros(1, 1),
    }
    model_params = [
        "transformer_blocks.0.attn.to_q.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
    ]
    plan = find_matching_plan(lora_sd, model_params)
    assert plan is not None
    assert plan.target_family == "diffusers_klein"


def test_find_plan_klein_lora_with_non_klein_model_returns_none():
    """A bfl_original LoRA + a model lacking `to_qkv_mlp_proj` → no plan."""
    lora_sd = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2": torch.zeros(1, 1),
    }
    # Standard diffusers Flux.1 layout — no to_qkv_mlp_proj
    model_params = [
        "transformer_blocks.0.attn.to_q.weight",
        "single_transformer_blocks.0.proj_mlp.weight",
        "single_transformer_blocks.0.proj_out.weight",
    ]
    plan = find_matching_plan(lora_sd, model_params)
    assert plan is None


def test_find_plan_unknown_lora_family_returns_none():
    """A LoRA with no recognizable family markers → no plan."""
    lora_sd = {"some.weird.path.lora_A.weight": torch.zeros(1, 1)}
    model_params = ["single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"]
    plan = find_matching_plan(lora_sd, model_params)
    assert plan is None


# ════════════════════════════════════════════════════════════════════════
#  convert_state_dict — LoKR Klein end-to-end
# ════════════════════════════════════════════════════════════════════════

def _make_synthetic_klein_lokr_module(base_path: str, w1_dim: int = 4,
                                      out_w2: int = 1024, in_w2: int = 1024,
                                      alpha: float = 4.0) -> dict:
    """Build a synthetic LoKR module: lokr_w1, lokr_w2, alpha."""
    return {
        f"{base_path}.lokr_w1": torch.randn(w1_dim, w1_dim),
        f"{base_path}.lokr_w2": torch.randn(out_w2, in_w2),
        f"{base_path}.alpha":   torch.tensor(alpha),
    }


def test_convert_klein_lokr_qkv_emits_three_lora_outputs():
    """A single fused-QKV LoKR module should produce 3 split outputs after convert."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    # Klein img_attn.qkv: w2 shape (3*4096/4, 4096/4) = (3072, 1024) given w1=(4,4)
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.double_blocks.0.img_attn.qkv",
        w1_dim=4, out_w2=3072, in_w2=1024,
    )
    out = convert_state_dict(sd, plan, target_rank=8)

    expected_bases = (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
    )
    for base in expected_bases:
        assert f"{base}.lora_A.weight" in out, f"missing {base}.lora_A.weight"
        assert f"{base}.lora_B.weight" in out, f"missing {base}.lora_B.weight"
        assert f"{base}.alpha"        in out, f"missing {base}.alpha"
    # No placeholder leakage
    assert not any("__QKV_IMG__" in k for k in out)


def test_convert_klein_lokr_qkv_txt_emits_add_proj_outputs():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.double_blocks.2.txt_attn.qkv",
        w1_dim=4, out_w2=3072, in_w2=1024,
    )
    out = convert_state_dict(sd, plan, target_rank=8)

    for tgt in ("add_q_proj", "add_k_proj", "add_v_proj"):
        base = f"transformer_blocks.2.attn.{tgt}"
        assert f"{base}.lora_A.weight" in out
        assert f"{base}.lora_B.weight" in out
    assert not any("__QKV_TXT__" in k for k in out)


def test_convert_klein_lokr_non_qkv_passes_through_via_svd():
    """A non-QKV LoKR module (e.g. img_attn.proj) emits a single SVD-compressed LoRA."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.double_blocks.0.img_attn.proj",
        w1_dim=4, out_w2=1024, in_w2=1024,
    )
    out = convert_state_dict(sd, plan, target_rank=8)

    base = "transformer_blocks.0.attn.to_out.0"
    assert f"{base}.lora_A.weight" in out
    assert f"{base}.lora_B.weight" in out
    assert f"{base}.alpha"        in out
    # Single output, no spurious extra keys
    assert sum(1 for k in out if k.startswith(base + ".lora")) == 2


def test_convert_klein_single_block_lokr_renames_and_compresses():
    """single_blocks.X.linear1 → single_transformer_blocks.X.attn.to_qkv_mlp_proj."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.single_blocks.5.linear1",
        w1_dim=4, out_w2=9216, in_w2=1024,
    )
    out = convert_state_dict(sd, plan, target_rank=8)
    base = "single_transformer_blocks.5.attn.to_qkv_mlp_proj"
    assert f"{base}.lora_A.weight" in out
    assert f"{base}.lora_B.weight" in out


def test_convert_standard_lora_qkv_split_is_exact():
    """Standard LoRA (not LoKR) on a QKV module: split is exact (no SVD)."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    rank, in_dim, head_dim = 8, 1024, 1024  # synthetic
    A = torch.randn(rank, in_dim)
    B = torch.randn(3 * head_dim, rank)
    sd = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": A,
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": B,
    }
    out = convert_state_dict(sd, plan, target_rank=999)  # target_rank ignored here

    # Exact split: lora_A is shared (same values); lora_B is sliced
    A_q = out["transformer_blocks.0.attn.to_q.lora_A.weight"]
    A_k = out["transformer_blocks.0.attn.to_k.lora_A.weight"]
    A_v = out["transformer_blocks.0.attn.to_v.lora_A.weight"]
    assert torch.allclose(A_q, A) and torch.allclose(A_k, A) and torch.allclose(A_v, A)

    B_q = out["transformer_blocks.0.attn.to_q.lora_B.weight"]
    B_k = out["transformer_blocks.0.attn.to_k.lora_B.weight"]
    B_v = out["transformer_blocks.0.attn.to_v.lora_B.weight"]
    assert torch.allclose(B_q, B[0:head_dim])
    assert torch.allclose(B_k, B[head_dim:2*head_dim])
    assert torch.allclose(B_v, B[2*head_dim:3*head_dim])


def test_convert_full_klein_lora_module_count():
    """Converting a synthetic full Klein LoRA produces the expected key count.

    Math:
      - 8 double_blocks × 6 non-QKV modules (proj×2, mlp.0×2, mlp.2×2) = 48 modules → 48 single LoRA outputs
      - 8 double_blocks × 2 QKV modules (img, txt)                     = 16 modules → 16 × 3 = 48 outputs
      - 24 single_blocks × 2 modules (linear1, linear2)                = 48 modules → 48 single outputs
      Total module-level outputs: 48 + 48 + 48 = 144
      Each output emits 3 keys (.lora_A.weight, .lora_B.weight, .alpha)
      → Total state-dict keys = 144 × 3 = 432
    """
    plan = get_plan("bfl_klein", "diffusers_klein")

    sd = {}
    for n in range(8):
        for stem in (
            f"diffusion_model.double_blocks.{n}.img_attn.proj",
            f"diffusion_model.double_blocks.{n}.txt_attn.proj",
            f"diffusion_model.double_blocks.{n}.img_mlp.0",
            f"diffusion_model.double_blocks.{n}.img_mlp.2",
            f"diffusion_model.double_blocks.{n}.txt_mlp.0",
            f"diffusion_model.double_blocks.{n}.txt_mlp.2",
        ):
            sd.update(_make_synthetic_klein_lokr_module(
                stem, w1_dim=4, out_w2=512, in_w2=512,
            ))
        for stem in (
            f"diffusion_model.double_blocks.{n}.img_attn.qkv",
            f"diffusion_model.double_blocks.{n}.txt_attn.qkv",
        ):
            sd.update(_make_synthetic_klein_lokr_module(
                stem, w1_dim=4, out_w2=1536, in_w2=512,  # 1536 = 3*512
            ))
    for n in range(24):
        for stem in (
            f"diffusion_model.single_blocks.{n}.linear1",
            f"diffusion_model.single_blocks.{n}.linear2",
        ):
            sd.update(_make_synthetic_klein_lokr_module(
                stem, w1_dim=4, out_w2=512, in_w2=512,
            ))

    out = convert_state_dict(sd, plan, target_rank=4)

    # Module-level output counts
    output_modules = set()
    for k in out:
        for sfx in (".lora_A.weight", ".lora_B.weight", ".alpha"):
            if k.endswith(sfx):
                output_modules.add(k[: -len(sfx)])
                break
    assert len(output_modules) == 144, (
        f"expected 144 unique output modules, got {len(output_modules)}"
    )
    # 3 keys per module
    assert len(out) == 144 * 3, f"expected 432 keys, got {len(out)}"


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
