# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Generic UltraGen — Quality-Focused Multi-Stage Generation

Polished port of ``eric_qwen_image_ultragen.py`` to the generic GEN_PIPELINE
infrastructure.  Works with Qwen-Image, Flux, Flux2, Chroma, and any other
flow-matching diffusers pipeline that GEN_PIPELINE can load.

Differences from the original Qwen UltraGen
-------------------------------------------
- **Model-family routing** for CFG (true_cfg_scale vs guidance_scale) via the
  same ``_cfg_kwargs`` helper used by the generic Multi-Stage node.
- **No Spectrum acceleration** — Spectrum patches the Qwen transformer
  directly and doesn't generalise.  Skip rather than pretend.
- **Upscale VAE is opt-in and gracefully degrades** — if the loaded
  pipeline's VAE is compatible with the Wan2.1 upscale VAE (Qwen-Image
  latent space), inter-stage and final-decode modes work as in the original
  UltraGen.  For Flux/Chroma the upscale VAE input is ignored with a clear
  warning rather than producing garbled output.
- **No default negative prompt** — the Qwen Chinese default is highly
  model-specific.  Users add their own.

What's the same
---------------
- Three-stage progressive generation (1/2/3 stages via upscale-factor gates)
- Per-stage steps, CFG, denoise strength, sigma schedule (linear/balanced/karras)
- Seed propagation modes (same/offset/random per stage)
- Per-stage LoRA weights (consumed from the LoRA stacker annotation)
- Mu-shifted sigma noise injection at stage transitions

Author: Eric Hiss (GitHub: EricRollei)
"""

import inspect
import torch
import numpy as np
from datetime import datetime
from typing import Tuple

from .eric_qwen_edit_utils import pil_to_tensor
from .eric_diffusion_utils import build_model_metadata
from .eric_diffusion_generate import ASPECT_RATIOS, compute_dimensions
from .eric_qwen_edit_lora import _set_adapters_safe
from .eric_diffusion_samplers import sampler_choices, swap_sampler

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


# ── Per-stage LoRA weight helper (same as generic multistage) ───────────────

def _apply_lora_stage_weights(pipe, pipeline_dict: dict, stage: int,
                              log_prefix: str = "[EricDiffusion-UG]") -> None:
    """Apply per-stage LoRA weights from the stacker annotation if present."""
    loras = pipeline_dict.get("applied_loras")
    if not loras:
        return
    key = f"weight_s{stage}"
    for adapter_name, info in loras.items():
        weight = info.get(key, info.get("weight_s1", 1.0))
        _set_adapters_safe(pipe, adapter_name, weight, log_prefix=log_prefix)
        print(f"{log_prefix}   Stage {stage} LoRA '{adapter_name}' weight={weight}")


# ── Noise injection (same flow-matching math as generic multistage) ─────────

def _apply_denoise_noise(latents: torch.Tensor, denoise: float,
                         actual_sigma: float, generator, device: str) -> torch.Tensor:
    if denoise >= 1.0:
        return torch.randn_like(latents)
    noise = torch.randn(latents.shape, generator=generator,
                        device=latents.device, dtype=latents.dtype)
    return _add_noise_flowmatch(latents, noise, actual_sigma)


# ── CFG routing (model-family aware) ────────────────────────────────────────

def _cfg_kwargs(pipe, model_family: str, guidance_embeds: bool,
                cfg_scale: float, negative_prompt: str | None,
                max_sequence_length: int) -> dict:
    """Build the guidance kwargs for a pipe() call.

    Same routing as the generic Multi-Stage node:
      qwen-image  → true_cfg_scale (double pass) + negative_prompt
      flux/flux2  → guidance_scale (single pass, no negative)
      unknown     → introspect __call__ signature, route best-guess
    """
    if model_family == "qwen-image":
        kw = {"true_cfg_scale": cfg_scale}
        if negative_prompt:
            kw["negative_prompt"] = negative_prompt
        return kw
    if model_family in ("flux", "flux2", "flux2klein"):
        kw = {"guidance_scale": cfg_scale}
        sig = inspect.signature(pipe.__call__)
        if "max_sequence_length" in sig.parameters:
            kw["max_sequence_length"] = max_sequence_length
        return kw
    sig = inspect.signature(pipe.__call__)
    accepted = set(sig.parameters.keys())
    candidates = {
        "guidance_scale":      cfg_scale if guidance_embeds else None,
        "true_cfg_scale":      cfg_scale if not guidance_embeds else None,
        "negative_prompt":     negative_prompt if not guidance_embeds else None,
        "max_sequence_length": max_sequence_length,
    }
    return {k: v for k, v in candidates.items() if k in accepted and v is not None}


# ── Upscale VAE compatibility detection ─────────────────────────────────────

def _vae_supports_upscale(pipe) -> bool:
    """Return True if the pipeline VAE is compatible with the Wan2.1
    upscale VAE (i.e. Qwen-Image / Wan latent space).

    The upscale VAE expects 5D latents (with frame dim) and the Qwen-Image
    latent normalization config (latents_mean, latents_std, z_dim).
    Flux/Chroma VAEs are 4D and use scaling_factor/shift_factor instead.
    """
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return False
    cfg = getattr(vae, "config", None)
    if cfg is None:
        return False
    return all(hasattr(cfg, key) for key in
               ("z_dim", "latents_mean", "latents_std"))


# ═══════════════════════════════════════════════════════════════════════════
#  UltraGen Node
# ═══════════════════════════════════════════════════════════════════════════

class EricDiffusionUltraGen:
    """
    Quality-focused multi-stage text-to-image generation for any GEN_PIPELINE.

    Same three-stage architecture as the generic Multi-Stage node, with
    tuned defaults, optional upscale VAE integration, and a slightly more
    opinionated UI for users who want UltraGen-style results out of the box.

    Set ``upscale_to_stage2 = 0`` to output Stage 1 only (single-stage).
    Set ``upscale_to_stage3 = 0`` to stop after Stage 2 (two-stage).

    The optional ``upscale_vae`` input is only honored when the pipeline's
    VAE is compatible (Qwen-Image latent space).  For Flux/Chroma it is
    ignored with a warning — use a separate image-space upscaler downstream.
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
                    "tooltip": (
                        "Describe the image you want to generate.\n"
                        "For best results use detailed descriptions (~200 words)."
                    ),
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "What to avoid. Used by Qwen-Image (true CFG). "
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
                        "random = independent random seed per stage."
                    ),
                }),
                "sampler": (sampler_choices(), {
                    "default": "multistep2",
                    "tooltip": (
                        "Sampler for all stages.\n"
                        "• default — pipeline's Euler (baseline).\n"
                        "• multistep2 — 2nd-order Adams-Bashforth (recommended).\n"
                        "• multistep3 — 3rd-order, may be smoother at higher step counts."
                    ),
                }),
                "max_sequence_length": ("INT", {
                    "default": 1024, "min": 128, "max": 1024, "step": 64,
                    "tooltip": (
                        "Max prompt token length for the text encoder. "
                        "Default 1024 captures detailed prompts. "
                        "Drop to 512 for shorter prompts to save VRAM. "
                        "Capped to the model's actual limit at runtime."
                    ),
                }),

                # ── Stage 1 ───────────────────────────────────────────────
                "s1_mp": ("FLOAT", {
                    "default": 0.5, "min": 0.3, "max": 4.0, "step": 0.1,
                    "tooltip": (
                        "Stage 1 resolution in megapixels. "
                        "Lower = faster composition draft. 0.5 MP is a good "
                        "starting point for most flow-matching models."
                    ),
                }),
                "s1_steps": ("INT", {
                    "default": 15, "min": 1, "max": 200,
                    "tooltip": "Stage 1 inference steps (txt2img from noise).",
                }),
                "s1_cfg": ("FLOAT", {
                    "default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": (
                        "Stage 1 CFG. Higher CFG at low res helps lock in "
                        "composition. Qwen tolerates 8-10; Flux/Chroma 3.5-5."
                    ),
                }),

                # ── Stage 2 ───────────────────────────────────────────────
                "upscale_to_stage2": ("FLOAT", {
                    "default": 4.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": (
                        "Area upscale factor S1→S2. "
                        "0 = output S1, skip S2 & S3. "
                        "4.0 = 0.5MP → 2MP."
                    ),
                }),
                "s2_steps": ("INT", {
                    "default": 26, "min": 1, "max": 200,
                    "tooltip": (
                        "Stage 2 inference steps. "
                        "This is the main refinement stage — most of your "
                        "step budget should go here for quality."
                    ),
                }),
                "s2_cfg": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Stage 2 CFG.",
                }),
                "s2_denoise": ("FLOAT", {
                    "default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stage 2 denoise strength. "
                        "0.8-0.9 = aggressive refinement (recommended). "
                        "1.0 = full re-denoise (ignores S1 latents)."
                    ),
                }),
                "s2_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "balanced",
                    "tooltip": (
                        "Sigma schedule for Stage 2.\n"
                        "linear = uniform spacing.\n"
                        "balanced (Karras ρ=3) = moderate detail focus.\n"
                        "karras (ρ=7) = heavy fine-detail focus.\n"
                        "Balanced is recommended for the main refinement."
                    ),
                }),

                # ── Stage 3 ───────────────────────────────────────────────
                "upscale_to_stage3": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": (
                        "Area upscale factor S2→S3. "
                        "0 = output S2, skip S3."
                    ),
                }),
                "s3_steps": ("INT", {
                    "default": 18, "min": 1, "max": 200,
                    "tooltip": "Stage 3 inference steps (final polish).",
                }),
                "s3_cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": (
                        "Stage 3 CFG. Lower than S2 to avoid over-sharpening."
                    ),
                }),
                "s3_denoise": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stage 3 denoise strength. "
                        "0.4-0.6 = light polish, preserves S2 detail."
                    ),
                }),
                "s3_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "karras",
                    "tooltip": (
                        "Sigma schedule for Stage 3.\n"
                        "Karras (ρ=7) is recommended for fine-texture polish."
                    ),
                }),

                # ── Upscale VAE (optional, Qwen-compatible only) ───────────
                "upscale_vae": ("UPSCALE_VAE", {
                    "tooltip": (
                        "Optional Wan2.1 2× upscale VAE. "
                        "Compatible with Qwen-Image latent space; ignored "
                        "for Flux/Chroma (use a separate image upscaler)."
                    ),
                }),
                "upscale_vae_mode": (
                    ["disabled", "inter_stage", "final_decode", "both"], {
                    "default": "disabled",
                    "tooltip": (
                        "How the upscale VAE is used (requires upscale_vae).\n"
                        "• disabled — never use the upscale VAE.\n"
                        "• inter_stage — decode S2 at 2×, re-encode for S3 "
                        "(replaces bislerp upscale; requires 3 stages).\n"
                        "• final_decode — replace last-stage VAE decode "
                        "with the 2× upscale decode (free 2× resolution).\n"
                        "• both — inter-stage AND final 2× decode."
                    ),
                }),
            }
        }

    def generate(
        self,
        pipeline: dict,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "1:1   Square",
        seed: int = 0,
        seed_mode: str = "offset_per_stage",
        max_sequence_length: int = 1024,
        s1_mp: float = 0.5,
        s1_steps: int = 15,
        s1_cfg: float = 5.0,
        upscale_to_stage2: float = 4.0,
        s2_steps: int = 26,
        s2_cfg: float = 4.0,
        s2_denoise: float = 0.85,
        s2_sigma_schedule: str = "balanced",
        upscale_to_stage3: float = 2.0,
        s3_steps: int = 18,
        s3_cfg: float = 3.5,
        s3_denoise: float = 0.5,
        s3_sigma_schedule: str = "karras",
        upscale_vae=None,
        upscale_vae_mode: str = "disabled",
        sampler: str = "multistep2",
    ) -> Tuple[torch.Tensor]:
        pipe             = pipeline["pipeline"]
        model_family     = pipeline.get("model_family", "unknown")
        guidance_embeds  = pipeline.get("guidance_embeds", False)
        offload_vae      = pipeline.get("offload_vae", False)
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

        stage_count = 3 if do_s3 else (2 if do_s2 else 1)
        total_steps = s1_steps + (s2_steps if do_s2 else 0) + (s3_steps if do_s3 else 0)

        _base_meta = {
            **build_model_metadata(pipeline),
            "node_type":       "ultragen",
            "seed":            seed,
            "seed_mode":       seed_mode,
            "sampler":         sampler,
            "sampler_s2":      sampler,
            "sampler_s3":      sampler,
            "prompt":          prompt,
            "negative_prompt": negative_prompt,
            "upscale_vae_mode": upscale_vae_mode,
            "stage_1": {"steps": s1_steps, "cfg": s1_cfg, "mp": s1_mp},
            "stage_2": {"steps": s2_steps, "cfg": s2_cfg, "denoise": s2_denoise,
                        "sigma_schedule": s2_sigma_schedule, "upscale": upscale_to_stage2},
            "stage_3": {"steps": s3_steps, "cfg": s3_cfg, "denoise": s3_denoise,
                        "sigma_schedule": s3_sigma_schedule, "upscale": upscale_to_stage3},
        }

        # ── Upscale VAE compatibility check ────────────────────────────────
        use_inter_stage = False
        use_final_decode = False
        if upscale_vae is not None and upscale_vae_mode != "disabled":
            if not _vae_supports_upscale(pipe):
                print(f"[EricDiffusion-UG] WARNING: upscale_vae mode "
                      f"'{upscale_vae_mode}' requires a Qwen-compatible VAE. "
                      f"Pipeline VAE is {type(pipe.vae).__name__}, ignoring "
                      f"upscale_vae input.")
            else:
                if upscale_vae_mode in ("inter_stage", "both"):
                    if do_s3:
                        use_inter_stage = True
                    else:
                        print("[EricDiffusion-UG] WARNING: inter_stage requires "
                              "3 stages (upscale_to_stage3 > 0). Falling back "
                              "to final_decode.")
                        use_final_decode = True
                if upscale_vae_mode in ("final_decode", "both"):
                    use_final_decode = True

        # ── Print plan ─────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[EricDiffusion-UG] {model_family} — {stage_count} stage(s), "
              f"{total_steps} total steps, seed_mode={seed_mode}, sampler={sampler}")
        print(f"  S1: {s1_w}×{s1_h} ({s1_mp_act:.2f} MP), {s1_steps} steps, "
              f"cfg={s1_cfg}")
        if do_s2:
            print(f"  S2: {s2_w}×{s2_h} ({s2_mp_act:.2f} MP), {s2_steps} steps, "
                  f"cfg={s2_cfg}, denoise={s2_denoise}, sigma={s2_sigma_schedule}")
        if do_s3:
            s3_mp_act = s3_w * s3_h / 1e6
            print(f"  S3: {s3_w}×{s3_h} ({s3_mp_act:.2f} MP), {s3_steps} steps, "
                  f"cfg={s3_cfg}, denoise={s3_denoise}, sigma={s3_sigma_schedule}")
        if use_inter_stage or use_final_decode:
            modes = []
            if use_inter_stage:
                modes.append("inter_stage")
            if use_final_decode:
                modes.append("final_decode")
            print(f"  Upscale VAE: {' + '.join(modes)}")
        print(f"{'='*60}")

        # ── Common setup ───────────────────────────────────────────────────
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

        # Mu computed from the original (flow-match) scheduler — same logic
        # as the generic multistage node.
        original_scheduler = pipe.scheduler
        if do_s2:
            s2_mu = _compute_mu(_packed_seq_len(s2_h, s2_w, vae_scale), original_scheduler)
        if do_s3:
            s3_mu = _compute_mu(_packed_seq_len(s3_h, s3_w, vae_scale), original_scheduler)

        # Move VAE to GPU if offloaded
        if offload_vae and not using_device_map and hasattr(pipe, "vae"):
            pipe.vae = pipe.vae.to(next(pipe.transformer.parameters()).device)

        # Sampler swap wraps ALL stages.  See multistage notes: mu is
        # pre-computed from the original scheduler above so the sigma math
        # is unaffected by the swap.
        sampler_ctx = swap_sampler(pipe, sampler, log_prefix="[EricDiffusion-UG]")
        sampler_ctx.__enter__()
        try:
            # ── Stage 1 — draft (txt2img from noise) ──────────────────────
            _check_cancelled()
            print(f"[EricDiffusion-UG] -- Stage 1/{stage_count} --")

            _apply_lora_stage_weights(pipe, pipeline, 1)
            s1_cfg_kw = _cfg_kwargs(pipe, model_family, guidance_embeds, s1_cfg, neg,
                                    max_sequence_length)
            s1_output_type = "latent" if (do_s2 or use_final_decode) else "pil"
            s1_result = pipe(
                prompt=prompt,
                height=s1_h,
                width=s1_w,
                num_inference_steps=s1_steps,
                generator=s1_gen,
                callback_on_step_end=make_cb(),
                output_type=s1_output_type,
                **s1_cfg_kw,
            )

            if not do_s2:
                _meta = {**_base_meta, "width": s1_w, "height": s1_h, "timestamp": datetime.now().isoformat()}
                if use_final_decode:
                    (t,) = self._final_decode(s1_result.images, upscale_vae, pipe, s1_h, s1_w, vae_scale)
                    return (t, _meta)
                pil = s1_result.images[0]
                return (pil_to_tensor(pil).unsqueeze(0), _meta)

            s1_latents = s1_result.images
            print(f"[EricDiffusion-UG]   S1 latents: {s1_latents.shape}")

            # ── Stage 2 — refine ──────────────────────────────────────────
            _check_cancelled()
            print(f"[EricDiffusion-UG] -- Stage 2/{stage_count} --")

            s2_latents = _upscale_latents(s1_latents, s1_h, s1_w, s2_h, s2_w, vae_scale)
            raw_s2     = build_sigma_schedule(s2_steps, s2_denoise,
                                              schedule=s2_sigma_schedule)
            act_s2     = _compute_actual_start_sigma(original_scheduler, raw_s2, s2_mu)
            s2_latents = _apply_denoise_noise(s2_latents, s2_denoise, act_s2, s2_gen,
                                              str(exec_device))

            _apply_lora_stage_weights(pipe, pipeline, 2)
            s2_cfg_kw = _cfg_kwargs(pipe, model_family, guidance_embeds, s2_cfg, neg,
                                    max_sequence_length)
            s2_output_type = "latent" if (do_s3 or use_final_decode) else "pil"
            s2_result = pipe(
                prompt=prompt,
                height=s2_h,
                width=s2_w,
                num_inference_steps=s2_steps,
                sigmas=raw_s2,
                latents=s2_latents,
                generator=s2_gen,
                callback_on_step_end=make_cb(),
                output_type=s2_output_type,
                **s2_cfg_kw,
            )

            if not do_s3:
                _meta = {**_base_meta, "width": s2_w, "height": s2_h, "timestamp": datetime.now().isoformat()}
                if use_final_decode:
                    (t,) = self._final_decode(s2_result.images, upscale_vae, pipe, s2_h, s2_w, vae_scale)
                    return (t, _meta)
                pil = s2_result.images[0]
                return (pil_to_tensor(pil).unsqueeze(0), _meta)

            s2_latents_final = s2_result.images
            print(f"[EricDiffusion-UG]   S2 latents: {s2_latents_final.shape}")

            # ── Stage 3 — polish ──────────────────────────────────────────
            _check_cancelled()
            print(f"[EricDiffusion-UG] -- Stage 3/{stage_count} --")

            if use_inter_stage:
                # Decode S2 at 2× via upscale VAE, re-encode for S3 input
                from .eric_qwen_upscale_vae import upscale_between_stages
                print("[EricDiffusion-UG]   Inter-stage VAE upscale (S2→S3) ...")
                s3_latents, s3_h, s3_w = upscale_between_stages(
                    s2_latents_final, upscale_vae, pipe.vae,
                    s2_h, s2_w, vae_scale,
                )
                s3_seq = _packed_seq_len(s3_h, s3_w, vae_scale)
                s3_mu = _compute_mu(s3_seq, original_scheduler)
            else:
                s3_latents = _upscale_latents(s2_latents_final, s2_h, s2_w, s3_h, s3_w, vae_scale)

            raw_s3     = build_sigma_schedule(s3_steps, s3_denoise,
                                              schedule=s3_sigma_schedule)
            act_s3     = _compute_actual_start_sigma(original_scheduler, raw_s3, s3_mu)
            s3_latents = _apply_denoise_noise(s3_latents, s3_denoise, act_s3, s3_gen,
                                              str(exec_device))

            _apply_lora_stage_weights(pipe, pipeline, 3)
            s3_cfg_kw = _cfg_kwargs(pipe, model_family, guidance_embeds, s3_cfg, neg,
                                    max_sequence_length)
            s3_output_type = "latent" if use_final_decode else "pil"
            s3_result = pipe(
                prompt=prompt,
                height=s3_h,
                width=s3_w,
                num_inference_steps=s3_steps,
                sigmas=raw_s3,
                latents=s3_latents,
                generator=s3_gen,
                callback_on_step_end=make_cb(),
                output_type=s3_output_type,
                **s3_cfg_kw,
            )

            _meta = {**_base_meta, "width": s3_w, "height": s3_h, "timestamp": datetime.now().isoformat()}
            if use_final_decode:
                (t,) = self._final_decode(s3_result.images, upscale_vae, pipe, s3_h, s3_w, vae_scale)
                return (t, _meta)

            pil = s3_result.images[0]
            print(f"[EricDiffusion-UG] Output: {pil.size[0]}×{pil.size[1]}")
            return (pil_to_tensor(pil).unsqueeze(0), _meta)

        finally:
            sampler_ctx.__exit__(None, None, None)
            if offload_vae and not using_device_map and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

    @staticmethod
    def _final_decode(packed_latents, upscale_vae, pipe, h, w, vae_scale):
        """Run the upscale VAE decode and return the ComfyUI tensor.

        Uses the safe wrapper which offloads the transformer to CPU
        for the decode and guarantees restoration to the original
        device on all exit paths — without this, the next pipe() call
        would silently run inference on CPU (~100× slowdown).
        """
        from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae_safe
        tensor = decode_latents_with_upscale_vae_safe(
            packed_latents, upscale_vae, pipe,
            h, w, vae_scale,
            log_prefix="[EricDiffusion-UG]",
        )
        print(f"[EricDiffusion-UG] Output: {tensor.shape[2]}×{tensor.shape[1]} (2× upscaled)")
        return (tensor,)
