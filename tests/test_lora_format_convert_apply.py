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
    _apply_converted_lora_as_delta,
    convert_state_dict,
    find_matching_plan,
    reconstruct_lokr_delta,
    svd_compress_to_lora,
)


# ════════════════════════════════════════════════════════════════════════
#  reconstruct_lokr_delta
# ════════════════════════════════════════════════════════════════════════

def test_lokr_reconstruct_directly_stored_ignores_alpha():
    """When w1 and w2 are stored DIRECTLY (no _a/_b decomposition), the
    stored alpha must be ignored entirely (scale=1.0).  This matches
    ComfyUI's LoKR loader convention and is critical for ai-toolkit
    LoRAs whose stored alpha is a ~1e10 sentinel rather than a
    meaningful scale factor (e.g. klein_snofs)."""
    w1 = torch.ones(4, 4)
    w2 = torch.eye(8)
    # Even with a wildly bogus alpha, directly-stored w1+w2 → scale=1.0
    alpha = torch.tensor(9.999e9)
    delta = reconstruct_lokr_delta(w1, w2, alpha)
    expected = torch.kron(w1.float(), w2.float())  # scale = 1.0
    assert torch.allclose(delta, expected)


def test_lokr_reconstruct_no_alpha_defaults_to_unscaled():
    w1 = torch.randn(4, 4)
    w2 = torch.randn(8, 8)
    delta = reconstruct_lokr_delta(w1, w2, alpha=None)
    expected = torch.kron(w1.float(), w2.float())
    assert torch.allclose(delta, expected)


def test_lokr_reconstruct_decomposed_w1_applies_alpha():
    """When w1 is decomposed (lokr_w1_a/lokr_w1_b present in source),
    alpha IS applied as alpha/w1_b_dim — matches ComfyUI's path that
    sets dim = w1_b.shape[0]."""
    w1 = torch.ones(4, 4)
    w2 = torch.eye(2)
    alpha = torch.tensor(2.0)
    delta = reconstruct_lokr_delta(
        w1, w2, alpha,
        w1_is_decomposed=True, w1_b_dim=4,  # alpha/dim = 2/4 = 0.5
    )
    expected = torch.kron(w1.float(), w2.float()) * 0.5
    assert torch.allclose(delta, expected)


def test_lokr_reconstruct_decomposed_w2_applies_alpha():
    """Same as above but the decomposition is on w2."""
    w1 = torch.ones(4, 4)
    w2 = torch.eye(2)
    alpha = torch.tensor(8.0)
    delta = reconstruct_lokr_delta(
        w1, w2, alpha,
        w2_is_decomposed=True, w2_b_dim=4,  # alpha/dim = 8/4 = 2.0
    )
    expected = torch.kron(w1.float(), w2.float()) * 2.0
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


def test_convert_klein_lokr_qkv_emits_three_diff_outputs():
    """A single fused-QKV LoKR module should produce 3 split .diff outputs.

    LoKRs go through the lossless .diff path (full merged delta direct
    merge) instead of SVD-truncated lora_A/lora_B.  See convert_state_dict
    docstring for the rationale (rank-64 SVD throws away ~98% of singular
    components on Klein's biggest matrices).
    """
    plan = get_plan("bfl_klein", "diffusers_klein")
    # Klein img_attn.qkv: w2 shape (3*4096/4, 4096/4) = (3072, 1024), w1=(4,4)
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
        assert f"{base}.diff" in out, f"missing {base}.diff"
        # Diff shape: each Q/K/V slice is (out_w2 * w1 / 3, in_w2 * w1)
        # = (3072*4/3, 1024*4) = (4096, 4096)
        assert out[f"{base}.diff"].shape == (4096, 4096)
    # No placeholder leakage
    assert not any("__QKV_IMG__" in k for k in out)
    # No lora_A/lora_B/alpha for LoKR modules — only .diff
    assert not any(k.endswith(".lora_A.weight") for k in out)
    assert not any(k.endswith(".lora_B.weight") for k in out)
    assert not any(k.endswith(".alpha") for k in out)


def test_convert_klein_lokr_qkv_diff_reconstructs_original():
    """The 3 emitted .diff slices should concatenate to the original delta.

    Klein LoKRs always have w1 and w2 stored directly (no _a/_b
    decomposition) — so the stored alpha is intentionally IGNORED at
    reconstruction time and the expected delta is just kron(w1, w2)."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.double_blocks.0.img_attn.qkv",
        w1_dim=4, out_w2=3072, in_w2=1024,
    )
    # Direct-stored LoKR → alpha ignored → scale=1.0
    w1 = sd["diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1"]
    w2 = sd["diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2"]
    expected_delta = torch.kron(w1.float(), w2.float())

    out = convert_state_dict(sd, plan, target_rank=8)
    reconstructed = torch.cat([
        out["transformer_blocks.0.attn.to_q.diff"],
        out["transformer_blocks.0.attn.to_k.diff"],
        out["transformer_blocks.0.attn.to_v.diff"],
    ], dim=0)
    assert torch.allclose(reconstructed, expected_delta), \
        f"max err {(reconstructed - expected_delta).abs().max()}"


def test_convert_klein_lokr_with_bogus_alpha_still_produces_sane_delta():
    """The pathological ai-toolkit case: stored alpha = ~1e10 must be
    silently ignored so the resulting delta has reasonable magnitude.
    Exact reproduction of klein_snofs / Realism_Engine / breast_slider."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    # Make a synthetic LoKR with sane w1 / w2 magnitudes but bogus alpha
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.double_blocks.0.img_attn.proj",
        w1_dim=4, out_w2=1024, in_w2=1024,
        alpha=9.999e9,  # ai-toolkit's bogus sentinel
    )
    out = convert_state_dict(sd, plan, target_rank=8)
    delta = out["transformer_blocks.0.attn.to_out.0.diff"]
    # Without the bogus alpha applied, max magnitude should be O(1) for
    # randn-initialized factors; with it applied, would be O(1e9).
    assert delta.abs().max().item() < 100.0, (
        f"delta max {delta.abs().max().item()} suggests bogus alpha was "
        f"applied — should have been ignored for direct-stored LoKR"
    )


def test_convert_klein_lokr_qkv_txt_emits_add_proj_diff_outputs():
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.double_blocks.2.txt_attn.qkv",
        w1_dim=4, out_w2=3072, in_w2=1024,
    )
    out = convert_state_dict(sd, plan, target_rank=8)

    for tgt in ("add_q_proj", "add_k_proj", "add_v_proj"):
        base = f"transformer_blocks.2.attn.{tgt}"
        assert f"{base}.diff" in out, f"missing {base}.diff"
    assert not any("__QKV_TXT__" in k for k in out)


def test_convert_klein_lokr_non_qkv_emits_single_diff():
    """A non-QKV LoKR module (e.g. img_attn.proj) emits one .diff (no split)."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.double_blocks.0.img_attn.proj",
        w1_dim=4, out_w2=1024, in_w2=1024,
    )
    out = convert_state_dict(sd, plan, target_rank=8)

    base = "transformer_blocks.0.attn.to_out.0"
    assert f"{base}.diff" in out
    # Single key for the module
    assert sum(1 for k in out if k.startswith(base + ".")) == 1
    # Diff shape = (out_w2 * w1_dim, in_w2 * w1_dim) = (4096, 4096)
    assert out[f"{base}.diff"].shape == (4096, 4096)


def test_convert_klein_single_block_lokr_renames_to_diff():
    """single_blocks.X.linear1 → single_transformer_blocks.X.attn.to_qkv_mlp_proj.diff."""
    plan = get_plan("bfl_klein", "diffusers_klein")
    sd = _make_synthetic_klein_lokr_module(
        "diffusion_model.single_blocks.5.linear1",
        w1_dim=4, out_w2=9216, in_w2=1024,
    )
    out = convert_state_dict(sd, plan, target_rank=8)
    base = "single_transformer_blocks.5.attn.to_qkv_mlp_proj"
    assert f"{base}.diff" in out
    assert out[f"{base}.diff"].shape == (36864, 4096)  # = (9216*4, 1024*4)


def test_convert_standard_lora_qkv_split_is_exact():
    """Standard LoRA (not LoKR) on a QKV module: split is exact (no SVD).

    No alpha provided ⇒ no baking; A and B should pass through unchanged.
    """
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


def test_convert_standard_lora_pass_through_bakes_alpha():
    """A non-QKV standard LoRA with alpha != rank should bake the alpha
    scaling factor into the emitted factors (so PEFT default scale=1.0
    yields the original delta).

    Original effective delta = (B @ A) * (alpha/rank)
    Baked emission:       A' = A * sqrt(alpha/rank)
                          B' = B * sqrt(alpha/rank)
    Reconstructed:    B' @ A' = (alpha/rank) * (B @ A) ✓
    """
    plan = get_plan("bfl_klein", "diffusers_klein")
    rank, in_dim, out_dim = 8, 1024, 1024
    A = torch.randn(rank, in_dim, dtype=torch.float64)
    B = torch.randn(out_dim, rank, dtype=torch.float64)
    alpha = torch.tensor(32.0, dtype=torch.float64)  # alpha != rank
    sd = {
        "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": A,
        "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": B,
        "diffusion_model.double_blocks.0.img_attn.proj.alpha":         alpha,
    }
    out = convert_state_dict(sd, plan, target_rank=999)

    A_out = out["transformer_blocks.0.attn.to_out.0.lora_A.weight"]
    B_out = out["transformer_blocks.0.attn.to_out.0.lora_B.weight"]
    assert "transformer_blocks.0.attn.to_out.0.alpha" not in out

    # Original effective delta: (B @ A) * (alpha/rank)
    expected = (B @ A) * (alpha.item() / rank)
    # Baked reconstruction: B' @ A'
    actual = B_out @ A_out
    assert torch.allclose(actual, expected, atol=1e-5), \
        f"max err: {(actual - expected).abs().max()}"


def test_convert_full_klein_lokr_module_count():
    """Converting a synthetic full Klein LoKR LoRA produces 144 .diff outputs.

    Math:
      - 8 double_blocks × 6 non-QKV modules → 48 single .diff outputs
      - 8 double_blocks × 2 QKV modules     → 16 modules × 3 split → 48 .diff
      - 24 single_blocks × 2 modules        → 48 single .diff outputs
      Total LoKR module outputs: 48 + 48 + 48 = 144
      Each LoKR module emits exactly 1 .diff key (full merged delta)
      → Total state-dict keys = 144
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

    # Module-level output counts: every key is .diff (one per module)
    diff_keys = [k for k in out if k.endswith(".diff")]
    assert len(diff_keys) == 144, (
        f"expected 144 .diff keys, got {len(diff_keys)}"
    )
    assert len(out) == 144, f"expected 144 total keys, got {len(out)}"
    # No standard-LoRA / alpha keys for an all-LoKR source
    assert not any(k.endswith(".lora_A.weight") for k in out)
    assert not any(k.endswith(".lora_B.weight") for k in out)
    assert not any(k.endswith(".alpha") for k in out)


# ════════════════════════════════════════════════════════════════════════
#  Direct delta merge (the reliable fallback path)
# ════════════════════════════════════════════════════════════════════════

class _FakeTransformer(torch.nn.Module):
    """Minimal stand-in for a real transformer with two named parameters."""
    def __init__(self):
        super().__init__()
        # Two nested linear-style params at known paths
        self.transformer_blocks = torch.nn.Module()
        self.transformer_blocks._modules["0"] = torch.nn.Module()
        block = self.transformer_blocks._modules["0"]
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(8, 8, bias=False)
        block.attn.to_q.weight.data.zero_()  # so the delta is the only contribution


def test_direct_delta_merge_applies_correct_delta():
    """Direct merge: model.weight ← model.weight + (B @ A) * weight."""
    transformer = _FakeTransformer()
    target_key = "transformer_blocks.0.attn.to_q.weight"
    A = torch.randn(4, 8)  # rank=4, in=8
    B = torch.randn(8, 4)  # out=8, rank=4
    state_dict = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": A,
        "transformer_blocks.0.attn.to_q.lora_B.weight": B,
    }
    expected_delta = (B @ A) * 1.5

    success = _apply_converted_lora_as_delta(
        transformer, state_dict, "test_adapter",
        weight=1.5, log_prefix="[test]",
    )
    assert success
    # Original was zeros; param now equals the delta
    actual = dict(transformer.named_parameters())[target_key]
    assert torch.allclose(actual, expected_delta, atol=1e-5)


def test_direct_delta_merge_skips_unmatched_modules():
    """Direct merge silently skips state-dict entries with no matching param."""
    transformer = _FakeTransformer()
    state_dict = {
        # Real path → matched
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 8),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(8, 2),
        # Fake path → skipped
        "nonexistent.path.lora_A.weight": torch.randn(2, 8),
        "nonexistent.path.lora_B.weight": torch.randn(8, 2),
    }
    success = _apply_converted_lora_as_delta(
        transformer, state_dict, "skip_test",
        weight=1.0, log_prefix="[test]",
    )
    assert success  # at least one module applied


def test_direct_delta_merge_registers_in_peft_config():
    """The merged adapter should be discoverable via peft_config."""
    transformer = _FakeTransformer()
    state_dict = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 8),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(8, 2),
    }
    _apply_converted_lora_as_delta(
        transformer, state_dict, "registered",
        weight=1.0, log_prefix="[test]",
    )
    assert hasattr(transformer, "peft_config")
    assert "registered" in transformer.peft_config
    assert transformer.peft_config["registered"]["_type"] == "converted_lora_direct"
    assert transformer.peft_config["registered"]["_applied_modules"] == 1


def test_direct_delta_merge_handles_diff_keys():
    """`.diff` entries (LoKR direct path) bypass B@A and apply the raw delta."""
    transformer = _FakeTransformer()
    target_key = "transformer_blocks.0.attn.to_q.weight"
    raw_delta = torch.randn(8, 8)
    state_dict = {
        "transformer_blocks.0.attn.to_q.diff": raw_delta,
    }
    success = _apply_converted_lora_as_delta(
        transformer, state_dict, "diff_test",
        weight=2.0, log_prefix="[test]",
    )
    assert success
    actual = dict(transformer.named_parameters())[target_key]
    expected = raw_delta * 2.0  # base was zeros
    assert torch.allclose(actual, expected, atol=1e-5)


def test_direct_delta_merge_handles_mixed_diff_and_lora():
    """Same state dict can mix .diff (LoKR) and .lora_A/.lora_B (standard) keys."""
    # Two-target fake transformer for the mixed test
    transformer = torch.nn.Module()
    transformer.transformer_blocks = torch.nn.Module()
    transformer.transformer_blocks._modules["0"] = torch.nn.Module()
    block = transformer.transformer_blocks._modules["0"]
    block.attn = torch.nn.Module()
    block.attn.to_q = torch.nn.Linear(8, 8, bias=False)
    block.attn.to_k = torch.nn.Linear(8, 8, bias=False)
    block.attn.to_q.weight.data.zero_()
    block.attn.to_k.weight.data.zero_()

    diff_q = torch.randn(8, 8)
    A_k = torch.randn(2, 8)
    B_k = torch.randn(8, 2)
    state_dict = {
        # LoKR-style .diff for to_q
        "transformer_blocks.0.attn.to_q.diff":          diff_q,
        # Standard-LoRA for to_k
        "transformer_blocks.0.attn.to_k.lora_A.weight": A_k,
        "transformer_blocks.0.attn.to_k.lora_B.weight": B_k,
    }
    success = _apply_converted_lora_as_delta(
        transformer, state_dict, "mixed",
        weight=1.0, log_prefix="[test]",
    )
    assert success
    params = dict(transformer.named_parameters())
    assert torch.allclose(
        params["transformer_blocks.0.attn.to_q.weight"], diff_q, atol=1e-5,
    )
    assert torch.allclose(
        params["transformer_blocks.0.attn.to_k.weight"], B_k @ A_k, atol=1e-5,
    )


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
