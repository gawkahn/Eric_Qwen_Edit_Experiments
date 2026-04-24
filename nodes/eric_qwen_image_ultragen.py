# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Qwen-Image UltraGen helpers (retained after node cull on 2026-04-24).

The EricQwenImageUltraGen node class that previously lived here was removed
when Eric Diffusion Advanced Multi-Stage absorbed its functionality (see
nodes/REMOVED.md). This file is retained because its helpers are consumed
by the ControlNet subsystem which is still registered:

  - _apply_lora_stage_weights  → eric_qwen_image_ultragen_cn.py,
                                  eric_qwen_image_ultragen_inpaint_cn.py
  - QWEN_OFFICIAL_RESOLUTIONS,
    DEFAULT_NEGATIVE_PROMPT     → same consumers

Resurrect the node: git show HEAD~1:nodes/eric_qwen_image_ultragen.py

Model Credits:
  - Qwen-Image developed by Qwen Team (Alibaba)
  - https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import torch
import numpy as np
from typing import Tuple

from .eric_qwen_edit_utils import pil_to_tensor
from .eric_qwen_image_generate import (
    ASPECT_RATIOS,
    DIMENSION_ALIGNMENT,
    _align,
    compute_dimensions_from_ratio,
)
from .eric_qwen_image_multistage import (
    _unpack_latents,
    _pack_latents,
    _upscale_latents,
    _add_noise_flowmatch,
    _check_cancelled,
    _packed_seq_len,
    _compute_mu,
    _compute_actual_start_sigma,
    build_sigma_schedule,
)
from .eric_qwen_upscale_vae import (
    decode_latents_with_upscale_vae,
    decode_latents_with_upscale_vae_safe,
    upscale_between_stages,
)


def _apply_lora_stage_weights(pipe, pipeline: dict, stage: int) -> None:
    """Set LoRA adapter weights for the given UltraGen stage (1, 2, or 3).

    Reads ``pipeline["applied_loras"]`` and calls ``pipe.set_adapters()``
    with the per-stage weight for each loaded adapter.

    Direct-merge adapters (LoKR/LoHa/LoRA that fell through to weight
    merging) have their weight baked in at load time and cannot be
    adjusted per stage.  A note is printed for those.

    This is a no-op when no LoRAs are applied.
    """
    applied_loras = pipeline.get("applied_loras")
    if not applied_loras:
        return

    stage_key = f"weight_stage{stage}"  # e.g. "weight_stage2"
    names = []
    weights = []
    direct_merge_names = []

    # Check which adapters are direct-merge
    transformer = getattr(pipe, "transformer", None)
    peft_cfg = getattr(transformer, "peft_config", {}) if transformer else {}

    for adapter_name, info in applied_loras.items():
        cfg = peft_cfg.get(adapter_name, {})
        if isinstance(cfg, dict) and cfg.get("_type", "").endswith("_direct"):
            direct_merge_names.append(adapter_name)
        else:
            names.append(adapter_name)
            weights.append(info.get(stage_key, 0.0))

    if direct_merge_names:
        print(f"[UltraGen] Stage {stage}: direct-merge adapters "
              f"{direct_merge_names} have fixed weight (per-stage "
              f"adjustment not available)")

    if names:
        try:
            pipe.set_adapters(names, adapter_weights=weights)
            summary = ", ".join(f"{n}={w}" for n, w in zip(names, weights))
            print(f"[UltraGen] Stage {stage} LoRA weights: {summary}")
        except Exception as e:
            print(f"[UltraGen] WARNING: Could not set stage {stage} "
                  f"LoRA weights: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  Official Qwen-Image-2512 recommended resolutions (~1.76 MP)
#  Source: https://github.com/QwenLM/Qwen-Image Quick Start
# ═══════════════════════════════════════════════════════════════════════

QWEN_OFFICIAL_RESOLUTIONS = {
    # (w_ratio, h_ratio): (width, height)   — all ~1.5-1.76 MP
    (1, 1):   (1328, 1328),   # 1.76 MP
    (16, 9):  (1664, 928),    # 1.54 MP
    (9, 16):  (928, 1664),    # 1.54 MP
    (4, 3):   (1472, 1104),   # 1.62 MP
    (3, 4):   (1104, 1472),   # 1.62 MP
    (3, 2):   (1584, 1056),   # 1.67 MP
    (2, 3):   (1056, 1584),   # 1.67 MP
}

# ═══════════════════════════════════════════════════════════════════════
#  Official negative prompt (Chinese)
#  Source: Qwen-Image-2512 official examples
#  Translation: Low resolution, low quality, limb deformity, finger
#  deformity, over-saturation, wax figure feel, face without detail,
#  overly smooth, AI look, chaotic composition, blurred text, distortion.
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
    "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)

