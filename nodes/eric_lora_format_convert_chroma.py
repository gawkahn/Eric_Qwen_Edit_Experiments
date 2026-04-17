# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Chroma LoRA conversion plan (slice 5).

Registers the rename rules for converting LoRAs trained against the
original BFL Chroma layout (``double_blocks.X.img_attn.qkv``,
``single_blocks.X.linear1``, etc.) into the diffusers reorganised
layout that ``FluxTransformer2DModel`` (Flux.1-derivative) exposes
for Chroma1-base and Chroma1-HD.

Chroma vs Klein key differences relevant to conversion:
  - Double-block MLP uses ``ff.net.0.proj`` / ``ff.net.2``
    (Flux.1 style), NOT ``ff.linear_in`` / ``ff.linear_out`` (Klein).
  - Single blocks have SEPARATE Q/K/V + ``proj_mlp`` + ``proj_out``,
    NOT the fused ``to_qkv_mlp_proj`` / ``to_out`` that Klein uses.
  - Single-block ``linear1`` is a 4-way split:
    Q (3072) + K (3072) + V (3072) + MLP_gate (12288) = 21504.
  - 19 double blocks (vs Klein's 8), 38 single blocks (vs Klein's 24).

Ground truth confirmed by inspecting:
  - Chroma1-base & Chroma1-HD transformer parameters (identical)
  - Chubby_Body_Type.safetensors (real Chroma standard LoRA, rank 32)
  - nodes/eric_lora_format_convert_flux.py (Klein plan as template)

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
#  Chroma — original BFL → diffusers Flux.1-derivative layout
# ════════════════════════════════════════════════════════════════════════
#
# Source side (BFL original, as emitted by Kohya / ai-toolkit for Chroma):
#   diffusion_model.double_blocks.{N}.img_attn.qkv            (fused QKV, img)
#   diffusion_model.double_blocks.{N}.img_attn.proj           (output proj, img)
#   diffusion_model.double_blocks.{N}.img_attn.norm.{q,k}_norm.scale
#   diffusion_model.double_blocks.{N}.img_mlp.0               (MLP linear_in)
#   diffusion_model.double_blocks.{N}.img_mlp.2               (MLP linear_out)
#   diffusion_model.double_blocks.{N}.txt_attn.qkv            (fused QKV, txt)
#   diffusion_model.double_blocks.{N}.txt_attn.proj           (output proj, txt)
#   diffusion_model.double_blocks.{N}.txt_attn.norm.{q,k}_norm.scale
#   diffusion_model.double_blocks.{N}.txt_mlp.0               (MLP_ctx linear_in)
#   diffusion_model.double_blocks.{N}.txt_mlp.2               (MLP_ctx linear_out)
#   diffusion_model.single_blocks.{N}.linear1                 (FUSED Q+K+V+MLP_gate)
#   diffusion_model.single_blocks.{N}.linear2                 (proj_out, NOT fused)
#   diffusion_model.single_blocks.{N}.norm.{q,k}_norm.scale
#
# Target side (diffusers Chroma transformer parameters):
#   transformer_blocks.{N}.attn.to_q / .to_k / .to_v          (split from img QKV)
#   transformer_blocks.{N}.attn.to_out.0                       (img output proj)
#   transformer_blocks.{N}.attn.norm_q / .norm_k
#   transformer_blocks.{N}.ff.net.0.proj / .ff.net.2           (img MLP — Flux.1 style)
#   transformer_blocks.{N}.attn.add_q_proj / .add_k_proj / .add_v_proj
#   transformer_blocks.{N}.attn.to_add_out                     (txt output proj)
#   transformer_blocks.{N}.attn.norm_added_q / .norm_added_k
#   transformer_blocks.{N}.ff_context.net.0.proj / .ff_context.net.2
#   single_transformer_blocks.{N}.attn.to_q / .to_k / .to_v   (split from linear1)
#   single_transformer_blocks.{N}.proj_mlp                     (split from linear1)
#   single_transformer_blocks.{N}.proj_out                     (1:1 from linear2)
#   single_transformer_blocks.{N}.attn.norm_q / .norm_k

_CHROMA_PLAN = ConversionPlan(
    source_family="bfl_chroma",
    target_family="diffusers_chroma",
    rename_rules=[
        # Strip the shared BFL component prefix.
        RenameRule("diffusion_model.", ""),

        # ── Block-level prefix renames ───────────────────────────────
        RenameRule("single_blocks.", "single_transformer_blocks."),
        RenameRule("double_blocks.",  "transformer_blocks."),

        # ── Double-block: attention (same as Klein) ──────────────────
        RenameRule(".txt_attn.qkv",   ".attn.__QKV_TXT__"),
        RenameRule(".img_attn.qkv",   ".attn.__QKV_IMG__"),
        RenameRule(".img_attn.proj",  ".attn.to_out.0"),
        RenameRule(".txt_attn.proj",  ".attn.to_add_out"),
        # Norms
        RenameRule(".img_attn.norm.q_norm",     ".attn.norm_q"),
        RenameRule(".img_attn.norm.k_norm",     ".attn.norm_k"),
        RenameRule(".txt_attn.norm.q_norm",     ".attn.norm_added_q"),
        RenameRule(".txt_attn.norm.k_norm",     ".attn.norm_added_k"),
        RenameRule(".img_attn.norm.query_norm", ".attn.norm_q"),
        RenameRule(".img_attn.norm.key_norm",   ".attn.norm_k"),
        RenameRule(".txt_attn.norm.query_norm", ".attn.norm_added_q"),
        RenameRule(".txt_attn.norm.key_norm",   ".attn.norm_added_k"),

        # ── Double-block: MLPs (DIFFERENT from Klein) ────────────────
        # Chroma uses Flux.1's ff.net.{0,2} convention, NOT Klein's
        # ff.linear_in / ff.linear_out.
        RenameRule(".img_mlp.0",  ".ff.net.0.proj"),
        RenameRule(".img_mlp.2",  ".ff.net.2"),
        RenameRule(".txt_mlp.0",  ".ff_context.net.0.proj"),
        RenameRule(".txt_mlp.2",  ".ff_context.net.2"),

        # ── Single-block (COMPLETELY different from Klein) ───────────
        # Klein keeps single-block fused: linear1 → to_qkv_mlp_proj,
        # linear2 → to_out.  Chroma splits everything.
        #
        # linear1 → placeholder for 4-way split (Q+K+V+MLP_gate)
        RenameRule(".linear1", ".__LINEAR1__"),
        # linear2 → 1:1 rename to proj_out (no split)
        RenameRule(".linear2", ".proj_out"),

        # Single-block norms
        RenameRule(".norm.q_norm",     ".attn.norm_q"),
        RenameRule(".norm.k_norm",     ".attn.norm_k"),
        RenameRule(".norm.query_norm", ".attn.norm_q"),
        RenameRule(".norm.key_norm",   ".attn.norm_k"),
    ],
    qkv_splits=[
        # Double-block img stream: 3-way equal split (same as Klein).
        QKVSplitSpec(
            pattern=".attn.__QKV_IMG__",
            targets=(".attn.to_q", ".attn.to_k", ".attn.to_v"),
        ),
        # Double-block txt stream: 3-way equal split (same as Klein).
        QKVSplitSpec(
            pattern=".attn.__QKV_TXT__",
            targets=(".attn.add_q_proj",
                     ".attn.add_k_proj",
                     ".attn.add_v_proj"),
        ),
        # Single-block linear1: 4-way UNEQUAL split.
        # BFL fused dim = 3*3072 + 12288 = 21504.
        # [0:3072]      → attn.to_q
        # [3072:6144]   → attn.to_k
        # [6144:9216]   → attn.to_v
        # [9216:21504]  → proj_mlp
        QKVSplitSpec(
            pattern=".__LINEAR1__",
            targets=(".attn.to_q", ".attn.to_k",
                     ".attn.to_v", ".proj_mlp"),
            target_dims=(3072, 3072, 3072, 12288),
        ),
    ],
    # Chroma's diffusers form has `proj_mlp` in single blocks — this
    # substring is absent in Klein/Flux2 (which fuses it into
    # to_qkv_mlp_proj).  Cleanly distinguishes Chroma from Klein.
    model_signature="proj_mlp",
    notes=(
        "Chroma1-base and Chroma1-HD — identical diffusers layout "
        "(Flux.1-derivative).  19 double + 38 single blocks.  MLP uses "
        "ff.net.{0,2} convention; single blocks have separate Q/K/V + "
        "proj_mlp + proj_out."
    ),
)


def _register_all() -> None:
    """Idempotent registration of every plan in this module."""
    register_plan(_CHROMA_PLAN)


_register_all()
