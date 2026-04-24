# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Qwen-Image multi-stage helpers (retained after node cull on 2026-04-24).

The EricQwenImageMultiStage node class that previously lived here was removed
when Eric Diffusion Advanced Multi-Stage absorbed its functionality (see
nodes/REMOVED.md). This file is retained because its latent-handling and
sigma-schedule helpers are consumed by LIVE code in the unified track:

  - _unpack_latents                   → eric_qwen_upscale_vae.py
  - _pack_latents / _unpack_latents   → eric_diffusion_ultragen.py,
                                         eric_diffusion_multistage.py,
                                         + the ControlNet subsystem
  - build_sigma_schedule, _upscale_latents, _compute_mu, _add_noise_flowmatch,
    etc. → ControlNet subsystem

Resurrect the node: git show HEAD~1:nodes/eric_qwen_image_multistage.py

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


# ═══════════════════════════════════════════════════════════════════════
#  Latent helpers
# ═══════════════════════════════════════════════════════════════════════

def _unpack_latents(latents: torch.Tensor, height: int, width: int,
                    vae_scale_factor: int) -> torch.Tensor:
    """Unpack flow-packed latents back to spatial (B, C, 1, H_lat, W_lat)."""
    batch_size, num_patches, channels = latents.shape
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, h_lat // 2, w_lat // 2,
                           channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, 1, h_lat, w_lat)
    return latents


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack spatial latents (B, C, 1, H, W) back to flow format (B, seq, C*4)."""
    batch_size, c, _one, h, w = latents.shape
    latents = latents.squeeze(2)                        # (B, C, H, W)
    latents = latents.view(batch_size, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)        # (B, H/2, W/2, C, 2, 2)
    latents = latents.reshape(batch_size, (h // 2) * (w // 2), c * 4)
    return latents


def _upscale_latents(latents_packed: torch.Tensor,
                     src_h: int, src_w: int,
                     dst_h: int, dst_w: int,
                     vae_scale_factor: int) -> torch.Tensor:
    """Unpack -> bislerp upscale -> repack latents.

    Uses ComfyUI's bislerp (slerp-interpolation) instead of bicubic.
    Bislerp preserves vector norms and angular relationships between
    latent channels, producing sharper and more coherent upscaled
    latents compared to bicubic's independent-channel averaging.
    """
    import comfy.utils as comfy_utils
    spatial = _unpack_latents(latents_packed, src_h, src_w, vae_scale_factor)
    b, c, _one, h_lat, w_lat = spatial.shape
    dst_h_lat = 2 * (int(dst_h) // (vae_scale_factor * 2))
    dst_w_lat = 2 * (int(dst_w) // (vae_scale_factor * 2))
    spatial_4d = spatial.squeeze(2)  # (B, C, H, W)
    upscaled = comfy_utils.bislerp(spatial_4d, dst_w_lat, dst_h_lat)
    upscaled = upscaled.unsqueeze(2)  # (B, C, 1, H', W')
    return _pack_latents(upscaled)


def _add_noise_flowmatch(latents: torch.Tensor, noise: torch.Tensor,
                         sigma: float) -> torch.Tensor:
    """Apply flow-matching noising: x_noisy = (1 - sigma) * x + sigma * noise."""
    return (1.0 - sigma) * latents + sigma * noise


def _check_cancelled():
    """Raise InterruptProcessingException if user hit Cancel in ComfyUI."""
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


# ═══════════════════════════════════════════════════════════════════════
#  Scheduler-aware sigma helpers
# ═══════════════════════════════════════════════════════════════════════

def _packed_seq_len(height: int, width: int, vae_scale_factor: int) -> int:
    """Compute the packed latent sequence length for given pixel dimensions."""
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    return (h_lat // 2) * (w_lat // 2)


def _compute_mu(seq_len: int, scheduler) -> float:
    """Compute the time-shift mu for a given sequence length.

    Mirrors the ``calculate_shift`` call inside QwenImagePipeline.__call__.
    """
    base_seq = scheduler.config.get("base_image_seq_len", 256)
    max_seq = scheduler.config.get("max_image_seq_len", 4096)
    base_shift = scheduler.config.get("base_shift", 0.5)
    max_shift = scheduler.config.get("max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    b = base_shift - m * base_seq
    return m * seq_len + b


def _compute_actual_start_sigma(
    scheduler, raw_sigmas: list, mu: float
) -> float:
    """Replicate the scheduler's sigma transforms to find the actual start sigma.

    Steps applied by ``FlowMatchEulerDiscreteScheduler.set_timesteps``:
    1. Dynamic time shift (exponential or linear) based on *mu*.
    2. Terminal stretch — rescale so the last sigma equals ``shift_terminal``.

    We replicate these on the raw sigmas so we can noise the latent at
    exactly the level the scheduler will expect on its first step.
    """
    sigmas = np.array(raw_sigmas, dtype=np.float64)

    # 1. Dynamic time shift
    if scheduler.config.get("use_dynamic_shifting", False):
        shift_type = scheduler.config.get("time_shift_type", "exponential")
        if shift_type == "exponential":
            exp_mu = math.exp(mu)
            sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
        else:  # "linear"
            sigmas = mu / (mu + (1.0 / sigmas - 1.0))
    else:
        static_shift = scheduler.config.get("shift", 1.0)
        sigmas = static_shift * sigmas / (1.0 + (static_shift - 1.0) * sigmas)

    # 2. Terminal stretch
    shift_terminal = scheduler.config.get("shift_terminal", None)
    if shift_terminal:
        one_minus_z = 1.0 - sigmas
        scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
        sigmas = 1.0 - (one_minus_z / scale_factor)

    return float(sigmas[0])


def build_sigma_schedule(num_steps: int, denoise: float,
                         schedule: str = "linear",
                         power: float = 1.0) -> list:
    """Build a sigma schedule for flow-matching partial denoise.

    The schedule defines the **spacing** of sigma values within the active
    denoise range.  All schedules start from the same sigma (determined by
    ``denoise``) and end at the same ``sigma_min`` — they only differ in
    how steps are distributed within that range.

    Schedules
    ---------
    linear
        Uniform spacing.  Safe default.  Equal compute at every noise level.
    balanced
        Karras-style with ρ=3.  Moderately concentrates steps at mid-to-low
        sigma — reduces time spent on composition (high sigma) while giving
        balanced coverage to detail and fine texture.  **Recommended for S2.**
    karras
        Karras-style with ρ=7 (EDM-optimal).  Heavily concentrates steps at
        low sigma (fine detail/sharpening), with large jumps through high
        sigma.  **Recommended for S3.**

    Step budget distribution (S2 example: 30 steps, denoise=0.85):

        Schedule  │ HIGH σ (composition)  │ MID σ (detail)  │ LOW σ (texture)
        ──────────┼───────────────────────┼─────────────────┼────────────────
        linear    │ 46%                   │ 38%             │ 15%
        balanced  │ 27%                   │ 38%             │ 35%
        karras    │ 23%                   │ 35%             │ 42%

    Denoise
    -------
    When ``denoise < 1.0``, the schedule covers only the lower portion of
    the sigma range (from ``sigma_start`` down to ``sigma_min``), with
    ``keep = round(num_steps * denoise)`` steps.  ``sigma_start`` is the
    same value linear truncation would produce, ensuring all schedules have
    a consistent noise level regardless of curve shape.

    Args:
        num_steps: Total number of denoising steps.
        denoise:   Fraction of schedule to use (1.0 = full, <1 = partial).
        schedule:  ``"linear"``, ``"balanced"``, or ``"karras"``.
        power:     Reserved for future use.

    Returns:
        List of sigma values (descending), length = ``keep``.
    """
    sigma_max = 1.0
    sigma_min = 1.0 / num_steps

    if denoise >= 1.0:
        keep = num_steps
        sigma_start = sigma_max
    else:
        keep = max(1, int(round(num_steps * denoise)))
        # Compute starting sigma from the linear schedule so ALL schedule
        # types begin at the same noise level for a given denoise value.
        # This was the root cause of the "echo/ghosting" bug: cosine and
        # karras had wildly different starting sigmas after truncation.
        full_linear = np.linspace(sigma_max, sigma_min, num_steps)
        sigma_start = float(full_linear[num_steps - keep])

    # Distribute 'keep' steps from sigma_start to sigma_min
    t = np.linspace(0.0, 1.0, keep)

    if schedule == "balanced":
        # Karras-style with ρ=3 — moderate concentration at low sigma.
        # Reduces high-sigma (composition) steps while giving balanced
        # coverage to mid (detail) + low (fine texture) ranges.
        rho = 3.0
        sigmas = (
            sigma_start ** (1.0 / rho)
            + t * (sigma_min ** (1.0 / rho) - sigma_start ** (1.0 / rho))
        ) ** rho
    elif schedule == "karras":
        # EDM-optimal (Karras et al., NeurIPS 2022) with ρ=7.
        # Heavy concentration at low sigma — best for fine detail stages.
        rho = 7.0
        sigmas = (
            sigma_start ** (1.0 / rho)
            + t * (sigma_min ** (1.0 / rho) - sigma_start ** (1.0 / rho))
        ) ** rho
    else:  # "linear" (default)
        sigmas = np.linspace(sigma_start, sigma_min, keep)

    sigmas = np.clip(sigmas, sigma_min, sigma_start)
    return sigmas.tolist()

