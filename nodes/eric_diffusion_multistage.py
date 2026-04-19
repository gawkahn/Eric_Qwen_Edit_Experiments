# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Generic Multi-Stage Generate Node
Progressive multi-stage text-to-image generation for any GEN_PIPELINE model.

Compatible with Qwen-Image, Flux.2, and any flow-matching diffusion model using
the same 2×2 latent packing scheme and FlowMatchEulerDiscreteScheduler.

CFG routing:
  qwen-image  → true_cfg_scale (double forward pass)
  flux / flux2 → guidance_scale (guidance embedding)
  unknown      → signature introspection

Latent upscaling uses ComfyUI bislerp (slerp-based interpolation that
preserves vector norms and angular relationships in latent space).

Author: Eric Hiss (GitHub: EricRollei)
"""

import inspect
import math
import torch
import numpy as np
from typing import Tuple

from .eric_qwen_edit_utils import pil_to_tensor
from .eric_diffusion_generate import ASPECT_RATIOS, compute_dimensions
from .eric_qwen_edit_lora import _set_adapters_safe
from .eric_diffusion_samplers import sampler_choices, swap_sampler
from .eric_diffusion_manual_loop import _maybe_enable_vae_tiling
from .eric_diffusion_utils import build_model_metadata
from datetime import datetime

# Reuse the latent helpers and sigma math from the Qwen multistage node —
# they are purely mathematical and model-agnostic.
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


# ── Per-stage LoRA weight helper ────────────────────────────────────────────

def _apply_lora_stage_weights(pipe, pipeline_dict: dict, stage: int,
                               log_prefix: str = "[EricDiffusion-MS]") -> None:
    """Apply per-stage LoRA weights from the stacker annotation if present."""
    loras = pipeline_dict.get("applied_loras")
    if not loras:
        return
    key = f"weight_s{stage}"
    for adapter_name, info in loras.items():
        weight = info.get(key, info.get("weight_s1", 1.0))
        _set_adapters_safe(pipe, adapter_name, weight, log_prefix=log_prefix)
        print(f"{log_prefix}   Stage {stage} LoRA '{adapter_name}' weight={weight}")



def _apply_denoise_noise(latents: torch.Tensor, denoise: float,
                         actual_sigma: float, generator, device: str) -> torch.Tensor:
    if denoise >= 1.0:
        return torch.randn_like(latents)
    noise = torch.randn(latents.shape, generator=generator,
                        device=latents.device, dtype=latents.dtype)
    return _add_noise_flowmatch(latents, noise, actual_sigma)


def _flux2_unpack(latents_packed: torch.Tensor, height: int, width: int,
                  vae_scale: int) -> torch.Tensor:
    """Convert Flux.2 packed (B, seq, C) to spatial (B, C, h_lat, w_lat).

    Flux.2's internal packing is a simple channel-first reshape+permute:
        (B, C, h, w) → reshape(B, C, h*w) → permute(0, 2, 1) → (B, h*w, C)
    The inverse is permute(0, 2, 1) → reshape(B, C, h, w).

    This is completely different from Qwen's 2×2 spatial tile packing and
    MUST NOT be mixed with _unpack_latents / _pack_latents.
    """
    h_lat = height // (vae_scale * 2)   # height // 16 for vae_scale=8
    w_lat = width  // (vae_scale * 2)
    b, seq, c = latents_packed.shape
    return latents_packed.permute(0, 2, 1).reshape(b, c, h_lat, w_lat)


def _upscale_latents_flux2(latents_packed: torch.Tensor,
                            src_h: int, src_w: int,
                            dst_h: int, dst_w: int,
                            vae_scale: int) -> torch.Tensor:
    """Upscale Flux.2 latents: unpack → bislerp → return 4D (B, C, h_lat, w_lat).

    Returns 4D because Flux.2's prepare_latents expects (B, C, H, W) when latents
    are passed in — it will repack them internally before the transformer.
    """
    import comfy.utils as comfy_utils
    spatial   = _flux2_unpack(latents_packed, src_h, src_w, vae_scale)
    dst_h_lat = dst_h // (vae_scale * 2)
    dst_w_lat = dst_w // (vae_scale * 2)
    return comfy_utils.bislerp(spatial, dst_w_lat, dst_h_lat)  # (B, C, dst_h_lat, dst_w_lat)


# ── CFG kwargs builder ───────────────────────────────────────────────────────

def _cfg_kwargs(pipe, model_family: str, guidance_embeds: bool,
                cfg_scale: float, negative_prompt: str | None,
                max_sequence_length: int) -> dict:
    """Return the guidance-related kwargs for a pipe() call."""
    if model_family == "qwen-image":
        kw = {"true_cfg_scale": cfg_scale}
        if negative_prompt:
            kw["negative_prompt"] = negative_prompt
        return kw
    if model_family in ("flux", "flux2"):
        kw = {"guidance_scale": cfg_scale}
        sig = inspect.signature(pipe.__call__)
        if "max_sequence_length" in sig.parameters:
            kw["max_sequence_length"] = max_sequence_length
        return kw
    # Unknown: introspect
    sig = inspect.signature(pipe.__call__)
    accepted = set(sig.parameters.keys())
    candidates = {
        "guidance_scale":       cfg_scale if guidance_embeds else None,
        "true_cfg_scale":       cfg_scale if not guidance_embeds else None,
        "negative_prompt":      negative_prompt if not guidance_embeds else None,
        "max_sequence_length":  max_sequence_length,
    }
    return {k: v for k, v in candidates.items() if k in accepted and v is not None}


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-Stage Node
# ═══════════════════════════════════════════════════════════════════════════

class EricDiffusionMultiStage:
    """
    Progressive multi-stage text-to-image generation for any GEN_PIPELINE model.

    Up to 3 stages with independent steps, CFG, upscale ratio, and denoise:
      Stage 1  — draft at low MP (txt2img from noise)
      Stage 2  — upscale + refine  (set upscale_to_stage2 = 0 to stop here)
      Stage 3  — upscale + polish  (set upscale_to_stage3 = 0 to stop after S2)

    Latents are bislerp-upscaled between stages (preserves latent-space
    vector norms) and re-noised at the correct flow-matching sigma before
    re-sampling.  The sigma math and latent packing format are compatible
    with Qwen-Image and Flux.2 (both use 2×2 packing + FlowMatchEulerDiscreteScheduler
    with identical dynamic-shift parameters).
    """

    CATEGORY = "Eric Diffusion"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "GEN_METADATA")
    RETURN_NAMES = ("image", "metadata")

    @classmethod
    def INPUT_TYPES(cls):
        ratio_names = list(ASPECT_RATIOS.keys())
        return {
            "required": {
                "pipeline": ("GEN_PIPELINE", {
                    "tooltip": "From Eric Diffusion Loader or Component Loader.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Used by Qwen-Image (true CFG). "
                        "Ignored by Flux / guidance-distilled models."
                    ),
                }),
                "aspect_ratio": (ratio_names, {"default": "1:1   Square"}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                }),
                "seed_mode": (["same_all_stages", "offset_per_stage", "random_per_stage"], {
                    "default": "offset_per_stage",
                    "tooltip": (
                        "same = one seed for all stages. "
                        "offset = seed, seed+1, seed+2. "
                        "random = independent random seeds per stage."
                    ),
                }),
                "sampler": (sampler_choices(), {
                    "default": "default",
                    "tooltip": (
                        "Sampler for ALL stages.\n"
                        "• default — pipeline's Euler.\n"
                        "• multistep2 — 2nd-order Adams-Bashforth.\n"
                        "• multistep3 — 3rd-order.\n"
                        "Sigma math continues to use the original flow-match "
                        "scheduler config so stage transitions remain correct."
                    ),
                }),
                "max_sequence_length": ("INT", {
                    "default": 512, "min": 64, "max": 512, "step": 64,
                    "tooltip": "Max T5/text-encoder token length (Flux models). Ignored by others.",
                }),
                # ── Stage 1 ───────────────────────────────────────────────
                "s1_mp": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 4.0, "step": 0.1,
                    "tooltip": "Stage 1 initial resolution in megapixels.",
                }),
                "s1_steps": ("INT", {
                    "default": 15, "min": 1, "max": 200,
                    "tooltip": "Stage 1 steps (txt2img from noise).",
                }),
                "s1_cfg": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Stage 1 CFG scale.",
                }),
                # ── Stage 2 ───────────────────────────────────────────────
                "upscale_to_stage2": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Area upscale factor S1→S2 (0 = output S1, skip S2 & S3).",
                }),
                "s2_steps": ("INT", {
                    "default": 20, "min": 1, "max": 200,
                }),
                "s2_cfg": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "s2_denoise": ("FLOAT", {
                    "default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "1.0 = full re-denoise. <1 = preserve prior stage detail.",
                }),
                "s2_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "balanced",
                    "tooltip": (
                        "linear = uniform spacing. "
                        "balanced = Karras rho=3 (moderate detail focus). "
                        "karras = Karras rho=7 (heavy fine-detail focus)."
                    ),
                }),
                # ── Stage 3 ───────────────────────────────────────────────
                "upscale_to_stage3": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Area upscale factor S2→S3 (0 = output S2, skip S3).",
                }),
                "s3_steps": ("INT", {
                    "default": 15, "min": 1, "max": 200,
                }),
                "s3_cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "s3_denoise": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05,
                }),
                "s3_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "karras",
                    "tooltip": (
                        "linear = uniform spacing. "
                        "balanced = Karras rho=3 (moderate detail focus). "
                        "karras = Karras rho=7 (heavy fine-detail focus)."
                    ),
                }),
            },
        }

    def generate(
        self,
        pipeline: dict,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "1:1   Square",
        seed: int = 0,
        seed_mode: str = "offset_per_stage",
        max_sequence_length: int = 512,
        s1_mp: float = 0.5,
        s1_steps: int = 15,
        s1_cfg: float = 4.0,
        upscale_to_stage2: float = 2.0,
        s2_steps: int = 20,
        s2_cfg: float = 4.0,
        s2_denoise: float = 0.85,
        s2_sigma_schedule: str = "balanced",
        upscale_to_stage3: float = 2.0,
        s3_steps: int = 15,
        s3_cfg: float = 3.5,
        s3_denoise: float = 0.6,
        s3_sigma_schedule: str = "karras",
        sampler: str = "default",
    ) -> Tuple[torch.Tensor]:
        pipe            = pipeline["pipeline"]
        model_family    = pipeline.get("model_family", "unknown")
        guidance_embeds = pipeline.get("guidance_embeds", False)
        offload_vae     = pipeline.get("offload_vae", False)
        using_device_map = hasattr(pipe, "hf_device_map")

        # ── Aspect ratio + stage dimensions ────────────────────────────────
        w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_ratio, (1, 1))

        do_s2 = upscale_to_stage2 > 0
        do_s3 = do_s2 and upscale_to_stage3 > 0

        s1_w, s1_h = compute_dimensions(w_ratio, h_ratio, s1_mp)
        s1_mp_act  = s1_w * s1_h / 1e6

        s2_w = s2_h = s2_mp_act = 0
        if do_s2:
            s2_w, s2_h = compute_dimensions(w_ratio, h_ratio, s1_mp_act * upscale_to_stage2)
            s2_mp_act  = s2_w * s2_h / 1e6

        s3_w = s3_h = 0
        if do_s3:
            s3_w, s3_h = compute_dimensions(w_ratio, h_ratio, s2_mp_act * upscale_to_stage3)

        stage_count  = 3 if do_s3 else (2 if do_s2 else 1)
        total_steps  = s1_steps + (s2_steps if do_s2 else 0) + (s3_steps if do_s3 else 0)

        _base_meta = {
            **build_model_metadata(pipeline),
            "node_type":   "multi-gen",
            "seed":        seed,
            "seed_mode":   seed_mode,
            "sampler":     sampler,
            "sampler_s2":  sampler,
            "sampler_s3":  sampler,
            "prompt":      prompt,
            "negative_prompt": negative_prompt,
            "stage_1": {"steps": s1_steps, "cfg": s1_cfg, "mp": s1_mp},
            "stage_2": {"steps": s2_steps, "cfg": s2_cfg, "denoise": s2_denoise,
                        "sigma_schedule": s2_sigma_schedule, "upscale": upscale_to_stage2},
            "stage_3": {"steps": s3_steps, "cfg": s3_cfg, "denoise": s3_denoise,
                        "sigma_schedule": s3_sigma_schedule, "upscale": upscale_to_stage3},
        }

        print(f"[EricDiffusion-MS] {model_family} — {stage_count} stage(s), "
              f"{total_steps} total steps, seed_mode={seed_mode}, sampler={sampler}")
        print(f"  S1: {s1_w}×{s1_h} ({s1_mp_act:.2f} MP), {s1_steps} steps, cfg={s1_cfg}")
        if do_s2:
            print(f"  S2: {s2_w}×{s2_h} ({s2_mp_act:.2f} MP), {s2_steps} steps, "
                  f"cfg={s2_cfg}, denoise={s2_denoise}, sigma={s2_sigma_schedule}")
        if do_s3:
            s3_mp_act = s3_w * s3_h / 1e6
            print(f"  S3: {s3_w}×{s3_h} ({s3_mp_act:.2f} MP), {s3_steps} steps, "
                  f"cfg={s3_cfg}, denoise={s3_denoise}, sigma={s3_sigma_schedule}")

        # ── Common setup ────────────────────────────────────────────────────
        exec_device = getattr(pipe, "_execution_device", "cuda")
        neg         = negative_prompt.strip() or None

        # ── Per-stage generators (seed propagation) ────────────────────────
        def _make_generator(stage_seed: int):
            if stage_seed <= 0:
                return None
            return torch.Generator(device=exec_device).manual_seed(stage_seed)

        if seed_mode == "offset_per_stage":
            s1_gen = _make_generator(seed)
            s2_gen = _make_generator(seed + 1 if seed > 0 else 0)
            s3_gen = _make_generator(seed + 2 if seed > 0 else 0)
        elif seed_mode == "random_per_stage":
            import random
            rng = random.Random(seed)
            s1_gen = _make_generator(seed)
            s2_gen = _make_generator(rng.randint(1, 2**63))
            s3_gen = _make_generator(rng.randint(1, 2**63))
        else:  # same_all_stages
            s1_gen = _make_generator(seed)
            s2_gen = _make_generator(seed)
            s3_gen = _make_generator(seed)

        import comfy.utils
        pbar = comfy.utils.ProgressBar(total_steps)

        def make_cb():
            def cb(_pipe, step_idx, _ts, kw):
                pbar.update(1)
                _check_cancelled()
                return kw
            return cb

        vae_scale = getattr(pipe, "vae_scale_factor", 8)

        # Precompute mu for refinement stages using the ORIGINAL pipeline
        # scheduler (guaranteed flow-match for GEN_PIPELINE models).  If the
        # user swaps in a non-flow-match scheduler, the noise injection still
        # uses the flow-match mu-shifted sigma — that's the correct value for
        # the physical latent regardless of which solver runs the denoise.
        original_scheduler = pipe.scheduler
        if do_s2:
            s2_mu = _compute_mu(_packed_seq_len(s2_h, s2_w, vae_scale), original_scheduler)
        if do_s3:
            s3_mu = _compute_mu(_packed_seq_len(s3_h, s3_w, vae_scale), original_scheduler)

        # Move VAE to GPU if offloaded (skip when device_map manages placement)
        if offload_vae and not using_device_map and hasattr(pipe, "vae"):
            pipe.vae = pipe.vae.to(next(pipe.transformer.parameters()).device)

        # Sampler swap wraps ALL three stages.  Mu is precomputed from
        # original_scheduler above, so sigma math remains correct even if
        # the swap installs a custom scheduler.
        sampler_ctx = swap_sampler(pipe, sampler, log_prefix="[EricDiffusion-MS]")
        sampler_ctx.__enter__()
        try:
            # ── Stage 1 — draft (txt2img from noise) ──────────────────────
            _check_cancelled()
            print(f"[EricDiffusion-MS] -- Stage 1/{stage_count} --")

            _apply_lora_stage_weights(pipe, pipeline, 1)
            s1_cfg_kw = _cfg_kwargs(pipe, model_family, guidance_embeds, s1_cfg, neg,
                                    max_sequence_length)
            if not do_s2:
                _maybe_enable_vae_tiling(pipe.vae, s1_h, s1_w)
            s1_result = pipe(
                prompt=prompt,
                height=s1_h,
                width=s1_w,
                num_inference_steps=s1_steps,
                generator=s1_gen,
                callback_on_step_end=make_cb(),
                output_type="latent" if do_s2 else "pil",
                **s1_cfg_kw,
            )
            if not do_s2 and hasattr(pipe, "vae"):
                pipe.vae.disable_tiling()

            if not do_s2:
                pil = s1_result.images[0]
                meta = {**_base_meta, "width": s1_w, "height": s1_h, "timestamp": datetime.now().isoformat()}
                return (pil_to_tensor(pil).unsqueeze(0), meta)

            s1_latents = s1_result.images
            # Detect packing format from tensor shape — don't rely on model_family.
            # Flux-family (Flux, Flux.2, Chroma, etc.) returns 3D packed (B, seq, C).
            # Qwen-Image and other spatial models return 4D (B, C, H, W).
            flux_packed = s1_latents.ndim == 3
            print(f"[EricDiffusion-MS]   S1 latents: {s1_latents.shape} "
                  f"({'flux-packed' if flux_packed else 'spatial'})")

            # ── Stage 2 — upscale + refine ────────────────────────────────
            _check_cancelled()
            print(f"[EricDiffusion-MS] -- Stage 2/{stage_count} --")

            s2_latents = _upscale_latents(s1_latents, s1_h, s1_w, s2_h, s2_w, vae_scale)
            raw_s2     = build_sigma_schedule(s2_steps, s2_denoise,
                                              schedule=s2_sigma_schedule)
            # Use original flow-match scheduler for sigma transformation;
            # the swapped scheduler may not implement mu-shifting.
            act_s2     = _compute_actual_start_sigma(original_scheduler, raw_s2, s2_mu)
            s2_latents = _apply_denoise_noise(s2_latents, s2_denoise, act_s2, s2_gen,
                                              str(exec_device))

            _apply_lora_stage_weights(pipe, pipeline, 2)
            s2_cfg_kw = _cfg_kwargs(pipe, model_family, guidance_embeds, s2_cfg, neg,
                                    max_sequence_length)
            if not do_s3:
                _maybe_enable_vae_tiling(pipe.vae, s2_h, s2_w)
            s2_result = pipe(
                prompt=prompt,
                height=s2_h,
                width=s2_w,
                num_inference_steps=s2_steps,
                sigmas=raw_s2,
                latents=s2_latents,
                generator=s2_gen,
                callback_on_step_end=make_cb(),
                output_type="latent" if do_s3 else "pil",
                **s2_cfg_kw,
            )
            if not do_s3 and hasattr(pipe, "vae"):
                pipe.vae.disable_tiling()

            if not do_s3:
                pil = s2_result.images[0]
                meta = {**_base_meta, "width": s2_w, "height": s2_h, "timestamp": datetime.now().isoformat()}
                return (pil_to_tensor(pil).unsqueeze(0), meta)

            s2_latents_final = s2_result.images
            print(f"[EricDiffusion-MS]   S2 latents: {s2_latents_final.shape}")

            # ── Stage 3 — upscale + polish ────────────────────────────────
            _check_cancelled()
            print(f"[EricDiffusion-MS] -- Stage 3/{stage_count} --")

            s3_latents = _upscale_latents(s2_latents_final, s2_h, s2_w, s3_h, s3_w, vae_scale)
            raw_s3     = build_sigma_schedule(s3_steps, s3_denoise,
                                              schedule=s3_sigma_schedule)
            act_s3     = _compute_actual_start_sigma(original_scheduler, raw_s3, s3_mu)
            s3_latents = _apply_denoise_noise(s3_latents, s3_denoise, act_s3, s3_gen,
                                              str(exec_device))

            _apply_lora_stage_weights(pipe, pipeline, 3)
            s3_cfg_kw = _cfg_kwargs(pipe, model_family, guidance_embeds, s3_cfg, neg,
                                    max_sequence_length)
            _maybe_enable_vae_tiling(pipe.vae, s3_h, s3_w)
            s3_result = pipe(
                prompt=prompt,
                height=s3_h,
                width=s3_w,
                num_inference_steps=s3_steps,
                sigmas=raw_s3,
                latents=s3_latents,
                generator=s3_gen,
                callback_on_step_end=make_cb(),
                output_type="pil",
                **s3_cfg_kw,
            )
            if hasattr(pipe, "vae"):
                pipe.vae.disable_tiling()

            pil = s3_result.images[0]
            print(f"[EricDiffusion-MS] Output: {pil.size[0]}×{pil.size[1]}")
            meta = {**_base_meta, "width": s3_w, "height": s3_h, "timestamp": datetime.now().isoformat()}
            return (pil_to_tensor(pil).unsqueeze(0), meta)

        finally:
            sampler_ctx.__exit__(None, None, None)
            if offload_vae and not using_device_map and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()
