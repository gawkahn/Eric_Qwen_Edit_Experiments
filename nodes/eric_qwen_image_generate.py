# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Qwen-Image generation helpers (retained after node cull on 2026-04-24).

The EricQwenImageGenerate node class that previously lived here was removed
when the unified "Eric Diffusion *" track superseded the Qwen-specific
generation nodes (see nodes/REMOVED.md). This file is retained because its
aspect-ratio / dimension helpers (ASPECT_RATIOS, _align,
compute_dimensions_from_ratio) are imported by the ControlNet subsystem
which is still registered (eric_qwen_image_ultragen.py and transitively
eric_qwen_image_ultragen_cn.py, eric_qwen_image_ultragen_inpaint_cn.py).

Resurrect the node: git show HEAD~1:nodes/eric_qwen_image_generate.py

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

# ── Resolution helpers ──────────────────────────────────────────────────

DIMENSION_ALIGNMENT = 16  # Qwen requires dimensions divisible by 16

# Aspect ratios ordered wide-landscape → square → tall-portrait
ASPECT_RATIOS = {
    "16:9  Wide Landscape":   (16, 9),
    "7:5   Landscape":        (7, 5),
    "3:2   Landscape":        (3, 2),
    "4:3   Landscape":        (4, 3),
    "5:4   Landscape":        (5, 4),
    "1:1   Square":           (1, 1),
    "4:5   Portrait":         (4, 5),
    "3:4   Portrait":         (3, 4),
    "2:3   Portrait":         (2, 3),
    "5:7   Portrait":         (5, 7),
    "9:16  Tall Portrait":    (9, 16),
}


def _align(val: int) -> int:
    """Round down to nearest multiple of DIMENSION_ALIGNMENT (min 16)."""
    return max(DIMENSION_ALIGNMENT, (val // DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT)


def compute_dimensions_from_ratio(
    w_ratio: int, h_ratio: int, target_mp: float
) -> Tuple[int, int]:
    """Compute width and height from aspect ratio and megapixel target.

    Solves for the dimensions that match the given aspect ratio while
    hitting the target megapixel count as closely as possible, then
    aligns both axes to ``DIMENSION_ALIGNMENT`` (16 px).
    """
    target_pixels = target_mp * 1_000_000
    # w/h = w_ratio/h_ratio  =>  w = h * (w_ratio/h_ratio)
    # w * h = target_pixels   =>  h^2 * (w_ratio/h_ratio) = target_pixels
    h = math.sqrt(target_pixels * h_ratio / w_ratio)
    w = h * w_ratio / h_ratio
    return _align(int(round(w))), _align(int(round(h)))

