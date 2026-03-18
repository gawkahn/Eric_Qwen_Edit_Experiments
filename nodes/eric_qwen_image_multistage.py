# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image Multi-Stage Generate Node
Progressive multi-stage text-to-image generation.

Stage 1 — Draft:  0.5 MP, 2× user CFG, 15 steps (txt2img from noise)
Stage 2 — Refine: 2 MP,   user CFG,     remaining steps (upscaled latent)
Stage 3 — Final:  target,  0.5× user CFG, remaining steps (if target > 2 MP)

Latents are upscaled 2× each dimension between stages (4× area) and
partially re-noised according to per-stage denoise strength before
re-sampling.  Total step budget = user steps input.

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
    """Unpack → bicubic upscale → repack latents."""
    spatial = _unpack_latents(latents_packed, src_h, src_w, vae_scale_factor)
    # spatial shape: (B, C, 1, H_lat, W_lat)
    b, c, _one, h_lat, w_lat = spatial.shape
    dst_h_lat = 2 * (int(dst_h) // (vae_scale_factor * 2))
    dst_w_lat = 2 * (int(dst_w) // (vae_scale_factor * 2))
    spatial_4d = spatial.squeeze(2)  # (B, C, H, W)
    upscaled = torch.nn.functional.interpolate(
        spatial_4d, size=(dst_h_lat, dst_w_lat), mode="bicubic", align_corners=False
    )
    upscaled = upscaled.unsqueeze(2)  # (B, C, 1, H', W')
    return _pack_latents(upscaled)


def _add_noise_flowmatch(latents: torch.Tensor, noise: torch.Tensor,
                         sigma: float) -> torch.Tensor:
    """Apply flow-matching noising: x_noisy = (1 - sigma) * x + sigma * noise."""
    return (1.0 - sigma) * latents + sigma * noise


# ═══════════════════════════════════════════════════════════════════════
#  Multi-Stage Generation Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageMultiStage:
    """
    Progressive multi-stage text-to-image generation.

    Generates a small draft (0.5 MP), then upscales the latent and
    refines at 2 MP, and optionally a third stage at the final target
    resolution.  This can produce higher quality results than a single
    pass at full resolution, especially for large images.

    Step budget: Stage 1 always uses 15 steps.  The remainder is split
    between Stage 2 (and Stage 3 if needed).

    CFG schedule: Stage 1 = 2× input, Stage 2 = input, Stage 3 = 0.5× input.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        ratio_names = list(ASPECT_RATIOS.keys())
        return {
            "required": {
                "pipeline": ("QWEN_IMAGE_PIPELINE", {
                    "tooltip": "From the Qwen-Image loader or component loader"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "Describe the image you want to generate"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid in the output"
                }),
                "aspect_ratio": (ratio_names, {
                    "default": "1:1   Square",
                    "tooltip": "Aspect ratio of the generated image"
                }),
                "target_mp": ("FLOAT", {
                    "default": 4.0,
                    "min": 2.0,
                    "max": 16.0,
                    "step": 0.25,
                    "tooltip": "Final output megapixels (minimum 2 MP, 3+ stages if > 2 MP)"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Total inference steps — 15 for Stage 1, remainder split across later stages"
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Base CFG — Stage 1 uses 2×, Stage 2 uses 1×, Stage 3 uses 0.5×"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = random)"
                }),
                "stage2_denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength for Stage 2 (1.0 = full re-denoise, lower = preserve more detail from previous stage)"
                }),
                "stage3_denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise strength for Stage 3 (1.0 = full re-denoise, lower = preserve more detail from previous stage)"
                }),
            }
        }

    # ─────────────────────────────────────────────────────────────────
    #  Main entry
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        pipeline: dict,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "1:1   Square",
        target_mp: float = 4.0,
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        stage2_denoise: float = 1.0,
        stage3_denoise: float = 1.0,
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)

        # ── Resolve aspect ratio ────────────────────────────────────────
        w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_ratio, (1, 1))
        target_mp = max(target_mp, 2.0)  # enforce 2 MP minimum

        # ── Plan stages ─────────────────────────────────────────────────
        #  Stage 1: 0.5 MP, 15 steps, 2× CFG
        #  Stage 2: 2 MP,   N steps,  1× CFG
        #  Stage 3: target, N steps,  0.5× CFG  (only if target > 2 MP)
        three_stages = target_mp > 2.0

        s1_steps = 15
        remaining = max(steps - s1_steps, 5)  # at least 5 for refine
        if three_stages:
            s2_steps = remaining // 2
            s3_steps = remaining - s2_steps
        else:
            s2_steps = remaining
            s3_steps = 0

        s1_w, s1_h = compute_dimensions_from_ratio(w_ratio, h_ratio, 0.5)
        s2_w, s2_h = compute_dimensions_from_ratio(w_ratio, h_ratio, 2.0)
        s3_w, s3_h = compute_dimensions_from_ratio(w_ratio, h_ratio, target_mp)

        s1_cfg = min(true_cfg_scale * 2.0, 20.0)
        s2_cfg = true_cfg_scale
        s3_cfg = max(true_cfg_scale * 0.5, 1.0)

        total_steps = s1_steps + s2_steps + s3_steps
        stage_count = 3 if three_stages else 2

        print(f"[EricQwenImage-MS] Multi-stage generation: {stage_count} stages, "
              f"{total_steps} total steps")
        print(f"  Stage 1: {s1_w}×{s1_h} ({s1_w*s1_h/1e6:.2f} MP), "
              f"{s1_steps} steps, CFG={s1_cfg:.1f}")
        print(f"  Stage 2: {s2_w}×{s2_h} ({s2_w*s2_h/1e6:.2f} MP), "
              f"{s2_steps} steps, CFG={s2_cfg:.1f}, denoise={stage2_denoise:.2f}")
        if three_stages:
            print(f"  Stage 3: {s3_w}×{s3_h} ({s3_w*s3_h/1e6:.2f} MP), "
                  f"{s3_steps} steps, CFG={s3_cfg:.1f}, denoise={stage3_denoise:.2f}")

        # ── Common setup ────────────────────────────────────────────────
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda"
        generator = torch.Generator(device=device).manual_seed(seed) if seed > 0 else None

        neg = negative_prompt.strip() if negative_prompt else None
        if neg == "":
            neg = None

        import comfy.utils
        pbar = comfy.utils.ProgressBar(total_steps)

        def make_cb():
            def on_step_end(_pipe, step_idx, _timestep, cb_kwargs):
                pbar.update(1)
                return cb_kwargs
            return on_step_end

        vae_scale_factor = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8

        # ── Move VAE to GPU if offloaded ────────────────────────────────
        if offload_vae and hasattr(pipe, "vae"):
            vae_device = next(pipe.transformer.parameters()).device
            pipe.vae = pipe.vae.to(vae_device)

        try:
            # ════════════════════════════════════════════════════════════
            #  STAGE 1 — Draft at 0.5 MP (full txt2img from noise)
            # ════════════════════════════════════════════════════════════
            print(f"[EricQwenImage-MS] ── Stage 1/{ stage_count} ──")
            s1_result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                height=s1_h,
                width=s1_w,
                num_inference_steps=s1_steps,
                true_cfg_scale=s1_cfg,
                generator=generator,
                callback_on_step_end=make_cb(),
                output_type="latent",
            )
            s1_latents = s1_result.images  # packed latents when output_type="latent"

            # ════════════════════════════════════════════════════════════
            #  STAGE 2 — Refine at 2 MP
            # ════════════════════════════════════════════════════════════
            print(f"[EricQwenImage-MS] ── Stage 2/{stage_count} ──")
            s2_latents = _upscale_latents(
                s1_latents, s1_h, s1_w, s2_h, s2_w, vae_scale_factor
            )
            s2_latents = self._apply_denoise_noise(
                s2_latents, stage2_denoise, generator, device
            )

            if three_stages:
                s2_output_type = "latent"
            else:
                s2_output_type = "pil"

            s2_result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                height=s2_h,
                width=s2_w,
                num_inference_steps=s2_steps,
                sigmas=self._build_sigmas(s2_steps, stage2_denoise),
                true_cfg_scale=s2_cfg,
                generator=generator,
                latents=s2_latents,
                callback_on_step_end=make_cb(),
                output_type=s2_output_type,
            )

            if not three_stages:
                # Done — decode to image
                pil_image = s2_result.images[0]
                tensor = pil_to_tensor(pil_image).unsqueeze(0)
                return (tensor,)

            s2_final_latents = s2_result.images  # still packed latents

            # ════════════════════════════════════════════════════════════
            #  STAGE 3 — Final at target MP
            # ════════════════════════════════════════════════════════════
            print(f"[EricQwenImage-MS] ── Stage 3/{stage_count} ──")
            s3_latents = _upscale_latents(
                s2_final_latents, s2_h, s2_w, s3_h, s3_w, vae_scale_factor
            )
            s3_latents = self._apply_denoise_noise(
                s3_latents, stage3_denoise, generator, device
            )

            s3_result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                height=s3_h,
                width=s3_w,
                num_inference_steps=s3_steps,
                sigmas=self._build_sigmas(s3_steps, stage3_denoise),
                true_cfg_scale=s3_cfg,
                generator=generator,
                latents=s3_latents,
                callback_on_step_end=make_cb(),
                output_type="pil",
            )

            pil_image = s3_result.images[0]
            tensor = pil_to_tensor(pil_image).unsqueeze(0)
            return (tensor,)

        finally:
            # Offload VAE back to CPU
            if offload_vae and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sigmas(num_steps: int, denoise: float) -> list:
        """Build a sigma schedule for flow-matching partial denoise.

        Full schedule goes from 1.0 → ~0.  When denoise < 1.0 we start
        from a lower sigma, effectively skipping the noisiest portion.
        """
        full_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        if denoise >= 1.0:
            return full_sigmas.tolist()
        # Trim from the front — keep the last (denoise * N) steps
        keep = max(1, int(round(num_steps * denoise)))
        return full_sigmas[-keep:].tolist()

    @staticmethod
    def _apply_denoise_noise(latents: torch.Tensor, denoise: float,
                             generator, device: str) -> torch.Tensor:
        """Add flow-matching noise to latents for partial denoise.

        At denoise=1.0 the latents are replaced with pure noise.
        At denoise<1.0 we blend the upscaled latents with noise at the
        starting sigma level.
        """
        if denoise >= 1.0:
            # Full re-noise — replace with random
            noise = torch.randn_like(latents)
            return noise

        # Partial: blend latents with noise at sigma = denoise
        sigma = denoise
        noise = torch.randn(latents.shape, generator=generator,
                            device=latents.device, dtype=latents.dtype)
        return _add_noise_flowmatch(latents, noise, sigma)
