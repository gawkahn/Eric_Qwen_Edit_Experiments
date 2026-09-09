# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Krea-2 LoRA conversion plan.

Registers the rename rules for converting LoRAs trained against the
ai-toolkit / ComfyUI-native Krea-2 layout (``diffusion_model.blocks.X.
attn.wq``, ``diffusion_model.txtfusion.layerwise_blocks.X.mlp.gate``,
etc.) into the diffusers ``Krea2Transformer2DModel`` layout that the
``Krea2Pipeline`` exposes.

Unlike the Flux/Chroma plans, Krea-2 needs **no QKV split**: the source
LoRA already stores separate ``wq`` / ``wk`` / ``wv`` projections, so
the conversion is a pure key rename (tensor data untouched).

Key correspondence (source → diffusers target; the loader prepends the
``transformer.`` prefix itself, so targets are relative to the model):

  diffusion_model.blocks.{N}.                  → transformer_blocks.{N}.
  diffusion_model.txtfusion.                   → text_fusion.
    (text_fusion has layerwise_blocks.{N} and refiner_blocks.{N}, both
     Krea2TextFusionBlock with the same attn/ff submodule layout)
  .attn.wq / .wk / .wv                         → .attn.to_q / .to_k / .to_v
  .attn.wo                                     → .attn.to_out.0
                                                 (to_out is a ModuleList
                                                 [Linear, Dropout])
  .attn.gate                                   → .attn.to_gate
                                                 (Krea2's gated attention)
  .mlp.{gate,up,down}                          → .ff.{gate,up,down}
                                                 (Krea2SwiGLU)

Standalone (non-block) modules — the source trainer stores these under
abbreviated ``nn.Sequential``-style indices; distillation LoRAs (e.g.
krea2_turbo_lora_rank_64) train them and their payload is precisely the
step-distilled behavior, so dropping them wrecks output quality:

  diffusion_model.first                        → img_in
  diffusion_model.last.linear                  → final_layer.linear
  diffusion_model.tmlp.{0,2}                   → time_embed.linear_{1,2}
  diffusion_model.tproj.1                      → time_mod_proj
  diffusion_model.txtmlp.{1,3}                 → txt_in.linear_{1,2}
  (txtfusion.projector is covered by the txtfusion prefix rule)

Ground truth confirmed by inspecting:
  - diffusers Krea2Transformer2DModel / Krea2Attention / Krea2SwiGLU /
    Krea2TextFusion parameters (transformer_krea2.py)
  - MysticXXX_KREA2_v1.safetensors (real ai-toolkit Krea-2 LoRA,
    ss_base_model_version=krea2; 512 keys, separate wq/wk/wv, no .alpha
    tensors → trainer alpha==rank, runtime scale 1.0)
  - krea2_turbo_lora_rank_64_bf16.safetensors (distill LoRA; 535 keys —
    adds the standalone modules above plus per-module .diff_b bias deltas)
  - diffusers 0.39.0 _convert_non_diffusers_krea2_lora_to_diffusers
    (lora_conversion_utils.py, PR #14074) — its standalone_map matches the
    table above one-for-one. (That upstream converter raises on .diff_b
    leftovers, so krea-native distill LoRAs still need THIS plan.)

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

from .eric_lora_format_convert import (
    ConversionPlan,
    RenameRule,
    register_plan,
)


# ════════════════════════════════════════════════════════════════════════
#  Krea-2 — ai-toolkit/ComfyUI-native → diffusers Krea2Transformer2DModel
# ════════════════════════════════════════════════════════════════════════

_KREA_PLAN = ConversionPlan(
    source_family="krea_native",
    target_family="diffusers_krea",
    rename_rules=[
        # ── Block-prefix renames (full prefixes, so the bare `blocks.`
        # substring inside txtfusion's `layerwise_blocks` / `refiner_blocks`
        # is never touched). text_fusion keeps its own sub-block names.
        RenameRule("diffusion_model.txtfusion.", "text_fusion."),
        RenameRule("diffusion_model.blocks.",    "transformer_blocks."),

        # ── Attention projections (separate Q/K/V — no split needed) ──
        RenameRule(".attn.wq",   ".attn.to_q"),
        RenameRule(".attn.wk",   ".attn.to_k"),
        RenameRule(".attn.wv",   ".attn.to_v"),
        RenameRule(".attn.wo",   ".attn.to_out.0"),
        RenameRule(".attn.gate", ".attn.to_gate"),

        # ── Feed-forward (SwiGLU): mlp.{gate,up,down} → ff.{gate,up,down} ──
        # (safe w.r.t. tmlp/txtmlp below: those lack the leading dot)
        RenameRule(".mlp.", ".ff."),

        # ── Standalone modules (trainer's nn.Sequential indices) ──
        # Trained by distillation LoRAs (turbo); mapping verified against
        # diffusers 0.39.0's krea2 standalone_map (PR #14074). Rules apply
        # to the BASE module path (suffix already split off), so these are
        # exact-anchored regexes — the whole base either is the module or
        # isn't.
        RenameRule(r"^diffusion_model\.first$",        "img_in",              regex=True),
        RenameRule(r"^diffusion_model\.last\.linear$", "final_layer.linear",  regex=True),
        RenameRule(r"^diffusion_model\.tmlp\.0$",      "time_embed.linear_1", regex=True),
        RenameRule(r"^diffusion_model\.tmlp\.2$",      "time_embed.linear_2", regex=True),
        RenameRule(r"^diffusion_model\.tproj\.1$",     "time_mod_proj",       regex=True),
        RenameRule(r"^diffusion_model\.txtmlp\.1$",    "txt_in.linear_1",     regex=True),
        RenameRule(r"^diffusion_model\.txtmlp\.3$",    "txt_in.linear_2",     regex=True),
    ],
    qkv_splits=[],  # Krea-2 stores separate wq/wk/wv — nothing to split.
    # `to_gate` (gated attention) is unique to Krea2 among the families
    # this converter handles (Flux/Chroma/Qwen have no to_gate), so it
    # cleanly identifies a loaded diffusers Krea2 transformer.
    model_signature="to_gate",
    notes=(
        "Krea-2 Raw/Turbo share Krea2Transformer2DModel. ai-toolkit "
        "native LoRA layout (diffusion_model.blocks / txtfusion, wq/wk/wv "
        "separate). Pure rename — no QKV split. Source LoRAs carry no "
        ".alpha tensors (alpha==rank), so runtime scale is 1.0."
    ),
)


def _register_all() -> None:
    """Idempotent registration of every plan in this module."""
    register_plan(_KREA_PLAN)


_register_all()
