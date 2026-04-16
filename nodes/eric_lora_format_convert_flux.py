# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Flux / Klein-9B LoRA conversion plans (slice 3).

Registers the rename rules for converting LoRAs trained against the
original BFL Flux.2 / Klein-9B layout (`double_blocks.X.img_attn.qkv`,
`single_blocks.X.linear1`, etc.) into the diffusers reorganised layout
that `Flux2Transformer2DModel` actually exposes.

What's covered here:
  - Klein-9B / Flux.2-dev (these share the diffusers structure with
    fused QKV+MLP single blocks and per-block modulation lifted to
    shared top-level modules).

What's NOT covered (yet):
  - Original Flux.1-dev / schnell.  Those use a different diffusers
    structure (separate `proj_mlp` + `proj_out` in single blocks).
    Add when needed by introducing a `bfl_flux1 → diffusers_flux1`
    plan with its own rename table.

Ground truth was confirmed by inspecting:
  - nodes/eric_lora_format_convert.py (framework)
  - klein_snofs_v1_1.safetensors (real LoKR LoRA, ai-toolkit, 2026-04)
  - FLUX.2-klein-9B/transformer/*.safetensors (model parameters)

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

from .eric_lora_format_convert import (
    ConversionPlan,
    QKVSplitSpec,
    RenameRule,
    register_plan,
)


# ════════════════════════════════════════════════════════════════════════
#  Klein-9B / Flux.2 — original BFL → diffusers Flux2 layout
# ════════════════════════════════════════════════════════════════════════
#
# Source side (BFL original, as emitted by ai-toolkit / kohya for Klein):
#   diffusion_model.double_blocks.{N}.img_attn.qkv            (fused QKV, img stream)
#   diffusion_model.double_blocks.{N}.img_attn.proj           (output proj, img stream)
#   diffusion_model.double_blocks.{N}.img_attn.norm.{q,k}_norm.scale
#   diffusion_model.double_blocks.{N}.img_mlp.0               (MLP linear_in)
#   diffusion_model.double_blocks.{N}.img_mlp.2               (MLP linear_out)
#   diffusion_model.double_blocks.{N}.txt_attn.qkv            (fused QKV, txt stream)
#   diffusion_model.double_blocks.{N}.txt_attn.proj           (output proj, txt stream)
#   diffusion_model.double_blocks.{N}.txt_attn.norm.{q,k}_norm.scale
#   diffusion_model.double_blocks.{N}.txt_mlp.0               (MLP_ctx linear_in)
#   diffusion_model.double_blocks.{N}.txt_mlp.2               (MLP_ctx linear_out)
#   diffusion_model.single_blocks.{N}.linear1                 (FUSED QKV+MLP_gate)
#   diffusion_model.single_blocks.{N}.linear2                 (FUSED attn_out + MLP_out)
#   diffusion_model.single_blocks.{N}.norm.{q,k}_norm.scale
#
# Target side (diffusers FLUX.2-Klein-9B transformer parameters):
#   transformer_blocks.{N}.attn.to_q                          (split — needs QKV split)
#   transformer_blocks.{N}.attn.to_k
#   transformer_blocks.{N}.attn.to_v
#   transformer_blocks.{N}.attn.to_out.0                      (img output proj)
#   transformer_blocks.{N}.attn.norm_q
#   transformer_blocks.{N}.attn.norm_k
#   transformer_blocks.{N}.ff.linear_in / linear_out
#   transformer_blocks.{N}.attn.add_q_proj                    (split — needs QKV split, txt stream)
#   transformer_blocks.{N}.attn.add_k_proj
#   transformer_blocks.{N}.attn.add_v_proj
#   transformer_blocks.{N}.attn.to_add_out                    (txt output proj)
#   transformer_blocks.{N}.attn.norm_added_q
#   transformer_blocks.{N}.attn.norm_added_k
#   transformer_blocks.{N}.ff_context.linear_in / linear_out
#   single_transformer_blocks.{N}.attn.to_qkv_mlp_proj        (1:1 rename — diffusers keeps it fused)
#   single_transformer_blocks.{N}.attn.to_out                 (1:1 rename — diffusers keeps it fused)
#   single_transformer_blocks.{N}.attn.norm_q / norm_k
#
# Ordering matters in the rename rules below — earlier rules see the
# original key, later rules see the partially-renamed result.  We order
# rules so that:
#   1. The shared diffusion_model. prefix is stripped first.
#   2. Block-prefix renames run BEFORE inner-module renames so the
#      inner renames don't accidentally fire across both block types.
#   3. The img_attn.qkv → unique placeholder rename happens BEFORE the
#      txt_attn.qkv → unique placeholder rename, then both placeholders
#      are split with their own QKVSplitSpecs.  Direct substring renames
#      to .attn.to_q would let the txt_attn pattern overwrite into the
#      same target as img_attn — placeholders avoid that.

_KLEIN_PLAN = ConversionPlan(
    source_family="bfl_klein",
    target_family="diffusers_klein",
    rename_rules=[
        # Strip the shared BFL component prefix.
        RenameRule("diffusion_model.", ""),

        # ── Block-level prefix renames ───────────────────────────────
        # Order: single_blocks before double — substring "single_blocks"
        # doesn't contain "double_blocks", so order is independent, but
        # being explicit is defensive against future rule additions.
        RenameRule("single_blocks.", "single_transformer_blocks."),
        RenameRule("double_blocks.",  "transformer_blocks."),

        # ── Double-block: img stream attention ───────────────────────
        # Use unique placeholders for the two QKV groups so the QKV-
        # split phase can target each independently.  The txt_attn
        # rename runs first (its key is more specific) so img_attn.qkv
        # later doesn't accidentally rename half of the txt path.
        RenameRule(".txt_attn.qkv",   ".attn.__QKV_TXT__"),
        RenameRule(".img_attn.qkv",   ".attn.__QKV_IMG__"),
        RenameRule(".img_attn.proj",  ".attn.to_out.0"),
        RenameRule(".txt_attn.proj",  ".attn.to_add_out"),
        # Norms — BFL stores the RMSNorm scale at .norm.q_norm.scale;
        # diffusers stores it at .attn.norm_q.weight (key suffix
        # remains .weight because adapter suffix lives elsewhere; here
        # we only rename the module path).
        RenameRule(".img_attn.norm.q_norm",   ".attn.norm_q"),
        RenameRule(".img_attn.norm.k_norm",   ".attn.norm_k"),
        RenameRule(".txt_attn.norm.q_norm",   ".attn.norm_added_q"),
        RenameRule(".txt_attn.norm.k_norm",   ".attn.norm_added_k"),
        # ai-toolkit also emits the legacy `.query_norm` / `.key_norm`
        # spelling — handle both.
        RenameRule(".img_attn.norm.query_norm", ".attn.norm_q"),
        RenameRule(".img_attn.norm.key_norm",   ".attn.norm_k"),
        RenameRule(".txt_attn.norm.query_norm", ".attn.norm_added_q"),
        RenameRule(".txt_attn.norm.key_norm",   ".attn.norm_added_k"),

        # ── Double-block: MLPs ───────────────────────────────────────
        RenameRule(".img_mlp.0",  ".ff.linear_in"),
        RenameRule(".img_mlp.2",  ".ff.linear_out"),
        RenameRule(".txt_mlp.0",  ".ff_context.linear_in"),
        RenameRule(".txt_mlp.2",  ".ff_context.linear_out"),

        # ── Single-block: 1:1 fused renames (no further QKV split) ──
        # diffusers Klein single_transformer_blocks keep the same fused
        # QKV+MLP layout the BFL original had — so the rename is the
        # only transformation needed for these modules.
        RenameRule(".linear1", ".attn.to_qkv_mlp_proj"),
        RenameRule(".linear2", ".attn.to_out"),
        # Single-block norms (same RMSNorm rename as double-block img).
        RenameRule(".norm.q_norm",     ".attn.norm_q"),
        RenameRule(".norm.k_norm",     ".attn.norm_k"),
        RenameRule(".norm.query_norm", ".attn.norm_q"),
        RenameRule(".norm.key_norm",   ".attn.norm_k"),
    ],
    qkv_splits=[
        # Double-block img stream: split into to_q / to_k / to_v.
        QKVSplitSpec(
            pattern=".attn.__QKV_IMG__",
            targets=(".attn.to_q", ".attn.to_k", ".attn.to_v"),
        ),
        # Double-block txt stream: split into add_{q,k,v}_proj.
        QKVSplitSpec(
            pattern=".attn.__QKV_TXT__",
            targets=(".attn.add_q_proj",
                     ".attn.add_k_proj",
                     ".attn.add_v_proj"),
        ),
    ],
    # Klein-9B and Flux.2-dev are the only Flux variants whose diffusers
    # form has the fused `to_qkv_mlp_proj` in single blocks.  Flux.1's
    # diffusers form keeps proj_mlp and proj_out separate.  This single
    # substring cleanly disambiguates Klein/Flux2 from Flux.1 at
    # plan-match time (slice 4).
    model_signature="to_qkv_mlp_proj",
    notes=(
        "Klein-9B / Flux.2-dev — shared diffusers layout.  Original "
        "Flux.1 has a different single_block structure (proj_mlp + "
        "proj_out separate) and would need a sibling plan."
    ),
)


def _register_all() -> None:
    """Idempotent registration of every plan in this module."""
    register_plan(_KLEIN_PLAN)


_register_all()
