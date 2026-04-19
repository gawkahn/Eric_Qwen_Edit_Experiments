# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Advanced Multi-Stage Generate (Flux/Chroma/Flux2/Qwen)

Multi-stage progressive text-to-image generation that runs the manual
denoising loop for each stage instead of calling ``pipe()``.  This means
all the higher-order samplers (Heun, RK3, multistep variants) are
available across all stages — typically the biggest visible quality
upgrade from the manual loop work.

Relationship to the other multistage nodes
------------------------------------------
- **Eric Diffusion Multi-Stage**: ``pipe()`` based, all model families.
- **Eric Diffusion UltraGen**: ``pipe()`` based + upscale VAE option,
  all model families.
- **Eric Diffusion Advanced Multi-Stage** (this file): manual loop,
  Flux-family + Qwen-Image, with sampler choice per stage.

The architecture mirrors the standard Multi-Stage node:
  S1 — generate at low resolution from pure noise
  S2 — upscale latents (bislerp), partial re-noise, refine
  S3 — upscale latents again, partial re-noise, polish

The key difference: each stage uses ``generate_flux()`` from the manual
loop module instead of ``pipe()``, so you can pick a sampler per stage.

Per-stage settings
------------------
Each of the three stages has independent:
  - steps
  - cfg_scale
  - denoise (s2/s3 only — s1 is always full denoise from noise)
  - sigma_schedule (linear / balanced / karras)
  - sampler (flow_euler, flow_heun, flow_rk3, multistep variants)

Plus the standard inputs (aspect_ratio, seed, seed_mode, max_sequence_length).

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
from typing import Tuple

from .eric_qwen_edit_utils import pil_to_tensor
from .eric_diffusion_generate import (
    ASPECT_RATIOS,
    compute_dimensions,
    resolve_override_dimensions,
)
from .eric_qwen_edit_lora import _set_adapters_safe
from .eric_diffusion_manual_loop import (
    sampler_names,
    sampler_cost,
    SIGMA_SCHEDULE_NAMES,
    generate_flux,
    decode_flux_latents,
    upscale_flux_latents,
    generate_chroma,
    generate_flux2,
    decode_flux2_latents,
    upscale_flux2_latents,
    generate_qwen,
    decode_qwen_latents,
    encode_image_to_packed_latents,
)


# ── Upscale VAE compatibility probe ─────────────────────────────────────
#
# Ported verbatim from ``eric_diffusion_ultragen.py`` so both nodes share
# the same "does this pipeline's VAE speak Wan2.1's latent language?"
# check.  If ultragen's probe ever drifts, update both sites.

def _vae_supports_upscale(pipe) -> bool:
    """Return True if the pipeline VAE is compatible with the Wan2.1
    upscale VAE (i.e. Qwen-Image / Wan latent space).

    The upscale VAE expects 5D latents (with a trivial frame dim) and
    the Qwen-Image latent normalization config — ``latents_mean``,
    ``latents_std``, and ``z_dim``.  Flux/Chroma VAEs are 4D and use
    ``scaling_factor`` / ``shift_factor`` instead, so they fail the check.
    """
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return False
    cfg = getattr(vae, "config", None)
    if cfg is None:
        return False
    return all(hasattr(cfg, key) for key in
               ("z_dim", "latents_mean", "latents_std"))


def _apply_lora_stage_weights(pipe, pipeline_dict: dict, stage: int,
                              log_prefix: str = "[EricDiffusion-Adv-MS]") -> None:
    """Apply per-stage LoRA weights from the stacker annotation if present."""
    loras = pipeline_dict.get("applied_loras")
    if not loras:
        return
    key = f"weight_s{stage}"
    for adapter_name, info in loras.items():
        weight = info.get(key, info.get("weight_s1", 1.0))
        _set_adapters_safe(pipe, adapter_name, weight, log_prefix=log_prefix)
        print(f"{log_prefix}   Stage {stage} LoRA '{adapter_name}' weight={weight}")


class EricDiffusionAdvancedMultiStage:
    """
    Multi-stage generation with manual denoising loop and per-stage samplers.

    Three stages with independent steps/cfg/denoise/sigma/sampler:
      Stage 1 — draft at low MP from pure noise
      Stage 2 — upscale + refine
      Stage 3 — upscale + polish

    Each stage's denoiser uses the manual loop, so any of the higher-
    order samplers (heun, rk3, multistep) work — and you can mix samplers
    across stages if you want (e.g. heun for S2 refinement, multistep3
    for S3 polish to save model calls).

    Set ``upscale_to_stage2 = 0`` to output Stage 1 only.
    Set ``upscale_to_stage3 = 0`` to stop after Stage 2.

    Flux-family (Flux, Flux2, Chroma) and Qwen-Image (t2i).  For
    Qwen-Image-Edit use the dedicated edit nodes.
    """

    CATEGORY = "Eric Diffusion"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        ratio_names = list(ASPECT_RATIOS.keys())
        return {
            "required": {
                "pipeline": ("GEN_PIPELINE", {
                    "tooltip": (
                        "From Eric Diffusion Loader or Component Loader. "
                        "Supports Flux, Flux2, Chroma, and Qwen-Image (t2i). "
                        "For Qwen-Image-Edit use the dedicated edit nodes."
                    ),
                }),
                "prompt": ("STRING", {"multiline": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                }),

                # ── Global settings ──────────────────────────────────
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
                "max_sequence_length": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": (
                        "Max prompt token length.\n"
                        "• Flux dev / Chroma: stay at 512 (T5-XXL hard limit).\n"
                        "• Flux.2: Mistral3-Small supports longer prompts.\n"
                        "Note: multistage re-encodes the prompt at EACH stage, "
                        "so longer prompts cost extra VRAM/time per stage."
                    ),
                }),
                "s1_mp": ("FLOAT", {
                    "default": 0.5, "min": 0.25, "max": 4.0, "step": 0.1,
                    "tooltip": "Stage 1 initial resolution in megapixels.",
                }),
                "override_s1_width": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 16,
                    "tooltip": (
                        "Explicit Stage 1 width in pixels.  When this "
                        "AND override_s1_height are both non-zero, they "
                        "override aspect_ratio + s1_mp.  Must be a "
                        "multiple of 16."
                    ),
                }),
                "override_s1_height": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 16,
                    "tooltip": (
                        "Explicit Stage 1 height in pixels.  Must be "
                        "set non-zero together with override_s1_width."
                    ),
                }),
                "upscale_to_stage2": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Area upscale factor S1→S2 (0 = output S1).",
                }),
                "upscale_to_stage3": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": (
                        "Area upscale factor S2→S3 (0 = disable stage 3).\n"
                        "IGNORED when use_upscale_vae is ON — Wan2.1 VAE "
                        "forces fixed 2× linear."
                    ),
                }),
                "use_upscale_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "QWEN-IMAGE ONLY. Wan2.1 2× upscale VAE for "
                        "inter-stage and final decode (up to 4× total).\n"
                        "Ignored for Flux/Flux2/Chroma."
                    ),
                }),
                "upscale_vae": ("UPSCALE_VAE", {
                    "tooltip": (
                        "Optional. Output of 'Eric Qwen Upscale VAE Loader'. "
                        "Only honored when 'use_upscale_vae' is ON."
                    ),
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": (
                        "Optional. Stage 1 starts from encoded reference "
                        "latents (i2i).  s1_denoise controls blend ratio."
                    ),
                }),
                "s1_denoise": ("FLOAT", {
                    "default": 0.85, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stage 1 denoise (i2i only, ignored without "
                        "reference_image).  1.0 = full repaint, "
                        "0.85 = refine, 0.3 = polish."
                    ),
                }),

                # ── Stage 1 ──────────────────────────────────────────
                "STAGE_1": ("STRING", {
                    "default": "STAGE 1",
                    "tooltip": "Visual separator (no effect on generation)",
                }),
                "s1_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stochastic noise injection.  0.0 = deterministic.\n"
                        "S1 generates from noise so higher eta is safe."
                    ),
                }),
                "s1_steps": ("INT", {
                    "default": 15, "min": 1, "max": 200,
                }),
                "s1_cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "s1_sampler": (sampler_names(), {
                    "default": "flow_heun",
                    "tooltip": (
                        "flow_heun = 2nd order, general-purpose.\n"
                        "flow_rk3 = best quality but requires ≥15 steps."
                    ),
                }),
                "s1_sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "linear",
                }),

                # ── Stage 2 ──────────────────────────────────────────
                "STAGE_2": ("STRING", {
                    "default": "STAGE 2",
                    "tooltip": "Visual separator (no effect on generation)",
                }),
                "s2_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stochastic noise injection.  0.0 = deterministic.\n"
                        "Lower eta in refinement preserves S1 structure."
                    ),
                }),
                "s2_steps": ("INT", {
                    "default": 20, "min": 1, "max": 200,
                }),
                "s2_cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "s2_sampler": (sampler_names(), {
                    "default": "flow_heun",
                    "tooltip": (
                        "flow_heun = best general choice for refinement.\n"
                        "flow_rk3 = only if effective steps ≥ 15."
                    ),
                }),
                "s2_sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "balanced",
                }),
                "s2_denoise": ("FLOAT", {
                    "default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "0.85 = aggressive refinement.  Lower values "
                        "preserve more of S1's structure."
                    ),
                }),

                # ── Stage 3 ──────────────────────────────────────────
                "STAGE_3": ("STRING", {
                    "default": "STAGE 3",
                    "tooltip": "Visual separator (no effect on generation)",
                }),
                "s3_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stochastic noise injection.  0.0 = deterministic.\n"
                        "Even small eta at S3 can cause mottling."
                    ),
                }),
                "s3_steps": ("INT", {
                    "default": 15, "min": 1, "max": 200,
                }),
                "s3_cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "s3_sampler": (sampler_names(), {
                    "default": "flow_multistep2",
                    "tooltip": (
                        "multistep2 = low cost + 2nd-order accuracy.\n"
                        "Avoid flow_rk3 — S3 partial-denoise cuts "
                        "effective steps."
                    ),
                }),
                "s3_sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "karras",
                }),
                "s3_denoise": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
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
        override_s1_width: int = 0,
        override_s1_height: int = 0,
        upscale_to_stage2: float = 2.0,
        upscale_to_stage3: float = 2.0,
        # Stage 1
        STAGE_1: str = "",
        s1_eta: float = 0.0,
        s1_steps: int = 15,
        s1_cfg: float = 3.5,
        s1_sampler: str = "flow_heun",
        s1_sigma_schedule: str = "linear",
        # Stage 2
        STAGE_2: str = "",
        s2_eta: float = 0.0,
        s2_steps: int = 20,
        s2_cfg: float = 3.5,
        s2_sampler: str = "flow_heun",
        s2_sigma_schedule: str = "balanced",
        s2_denoise: float = 0.85,
        # Stage 3
        STAGE_3: str = "",
        s3_eta: float = 0.0,
        s3_steps: int = 15,
        s3_cfg: float = 3.5,
        s3_sampler: str = "flow_multistep2",
        s3_sigma_schedule: str = "karras",
        s3_denoise: float = 0.5,
        use_upscale_vae: bool = False,
        upscale_vae=None,
        reference_image=None,
        s1_denoise: float = 0.85,
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]
        model_family = pipeline.get("model_family", "unknown")
        offload_vae = pipeline.get("offload_vae", False)
        using_device_map = hasattr(pipe, "hf_device_map")

        # ── Family check ───────────────────────────────────────────────
        flux_families = (
            "flux", "flux2", "chroma",
            "fluxpipeline", "flux2pipeline", "chromapipeline",
        )
        qwen_families = (
            "qwen-image", "qwenimage", "qwenimagepipeline",
        )
        supported = flux_families + qwen_families
        if model_family not in supported:
            raise ValueError(
                f"Eric Diffusion Advanced Multi-Stage supports Flux-family "
                f"(Flux/Flux2/Chroma) and Qwen-Image pipelines.  Got "
                f"model_family={model_family!r}.\n"
                f"Use 'Eric Diffusion Multi-Stage' or 'UltraGen' for other "
                f"model types, or the dedicated Qwen-Image-Edit nodes for "
                f"Qwen edit."
            )

        # ── Aspect ratio + stage dimensions ────────────────────────────
        w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_ratio, (1, 1))

        do_s2 = upscale_to_stage2 > 0
        do_s3 = do_s2 and upscale_to_stage3 > 0

        # Stage 1 dimensions: default from aspect_ratio + s1_mp, or
        # from explicit override_s1_width/height.  When override is
        # active, the S1 ASPECT is also taken from the override (not
        # from aspect_ratio input) so S2/S3 derivations downstream
        # inherit the correct aspect ratio.
        default_s1_w, default_s1_h = compute_dimensions(w_ratio, h_ratio, s1_mp)
        s1_w, s1_h = resolve_override_dimensions(
            override_s1_width, override_s1_height,
            default_s1_w, default_s1_h,
            log_prefix="[EricDiffusion-Adv-MS]",
        )
        s1_mp_act = s1_w * s1_h / 1e6

        # For S2/S3 dimension derivation, use the S1's actual aspect
        # ratio (which may differ from the aspect_ratio preset if
        # override is active).  This guarantees S2 and S3 preserve S1's
        # aspect regardless of the path that picked S1's dimensions.
        effective_w_ratio = s1_w
        effective_h_ratio = s1_h

        s2_w = s2_h = s2_mp_act = 0
        if do_s2:
            s2_w, s2_h = compute_dimensions(
                effective_w_ratio, effective_h_ratio, s1_mp_act * upscale_to_stage2,
            )
            s2_mp_act = s2_w * s2_h / 1e6

        s3_w = s3_h = 0
        if do_s3:
            s3_w, s3_h = compute_dimensions(
                effective_w_ratio, effective_h_ratio, s2_mp_act * upscale_to_stage3,
            )

        stage_count = 3 if do_s3 else (2 if do_s2 else 1)

        # Total cost in transformer evaluations (steps × sampler cost × cfg multiplier)
        cfg_mult = 2 if (s1_cfg > 1.0 or s2_cfg > 1.0 or s3_cfg > 1.0) and negative_prompt else 1
        total_evals = (
            s1_steps * sampler_cost(s1_sampler) * (2 if s1_cfg > 1.0 and negative_prompt else 1)
            + (s2_steps * sampler_cost(s2_sampler) * (2 if s2_cfg > 1.0 and negative_prompt else 1) if do_s2 else 0)
            + (s3_steps * sampler_cost(s3_sampler) * (2 if s3_cfg > 1.0 and negative_prompt else 1) if do_s3 else 0)
        )

        print(f"\n{'='*60}")
        print(f"[EricDiffusion-Adv-MS] {model_family} — {stage_count} stage(s), "
              f"~{total_evals} transformer calls, seed_mode={seed_mode}")
        print(f"  S1: {s1_w}×{s1_h} ({s1_mp_act:.2f} MP), {s1_steps} steps, "
              f"sampler={s1_sampler}, cfg={s1_cfg}")
        if do_s2:
            print(f"  S2: {s2_w}×{s2_h} ({s2_mp_act:.2f} MP), {s2_steps} steps, "
                  f"sampler={s2_sampler}, cfg={s2_cfg}, denoise={s2_denoise}, "
                  f"sigma={s2_sigma_schedule}")
        if do_s3:
            s3_mp_act = s3_w * s3_h / 1e6
            print(f"  S3: {s3_w}×{s3_h} ({s3_mp_act:.2f} MP), {s3_steps} steps, "
                  f"sampler={s3_sampler}, cfg={s3_cfg}, denoise={s3_denoise}, "
                  f"sigma={s3_sigma_schedule}")
        print(f"{'='*60}")

        # ── Common setup ───────────────────────────────────────────────
        exec_device = getattr(pipe, "_execution_device", None) or next(
            pipe.transformer.parameters()
        ).device
        neg = negative_prompt.strip() or None

        # ── Per-stage generators (seed propagation) ───────────────────
        # Use CPU generators regardless of compute device — CPU RNG is
        # more consistent than CUDA Philox RNG for packed latent shapes
        # and doesn't exhibit shape-dependent per-row pattern artifacts.
        # diffusers' randn_tensor handles the CPU→GPU transfer.
        def _make_generator(stage_seed: int):
            if stage_seed <= 0:
                return None
            return torch.Generator(device="cpu").manual_seed(stage_seed)

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

        # ── Progress bar covers all stages combined ───────────────────
        import comfy.utils
        import comfy.model_management
        total_step_count = (
            s1_steps
            + (s2_steps if do_s2 else 0)
            + (s3_steps if do_s3 else 0)
        )
        pbar = comfy.utils.ProgressBar(total_step_count)
        steps_completed = [0]

        def _stage_progress_cb(step_in_stage):
            # step_in_stage is the absolute step count within the current stage
            # (1, 2, ...).  We track the offset across stages.
            target_total = steps_completed[0] + step_in_stage
            current = pbar.current if hasattr(pbar, 'current') else 0
            delta = target_total - current
            if delta > 0:
                pbar.update(delta)
            comfy.model_management.throw_exception_if_processing_interrupted()

        comfy.model_management.throw_exception_if_processing_interrupted()

        # ── VAE: move back to GPU if offloaded ────────────────────────
        if offload_vae and not using_device_map and hasattr(pipe, "vae"):
            pipe.vae = pipe.vae.to(next(pipe.transformer.parameters()).device)

        # Capture the transformer's device ONCE at the top of the run.
        # When use_upscale_vae is ON, the final-decode code blocks move
        # pipe.transformer to CPU to free VRAM for the Wan2.1 decode,
        # and the finally block below restores it to this device so
        # the NEXT run doesn't silently run Qwen/Flux inference on CPU
        # (GPU idle, CPU pegged, ~100× slower than intended).
        _transformer_device = next(pipe.transformer.parameters()).device

        # Four-way family dispatch per stage:
        #
        #   Flux.2  → generate_flux2 (needs position ids plumbed through,
        #             decode via BN denorm + unpatchify)
        #   Chroma  → generate_chroma (encode_prompt returns 6 values,
        #             transformer takes attention_mask instead of
        #             pooled_projections, but VAE/upscale same as Flux)
        #   Qwen    → generate_qwen (2-tuple encode_prompt, norm-preserving
        #             CFG, 5D VAE — uses decode_qwen_latents.  Packing math
        #             matches Flux so upscale_flux_latents is reused).
        #   Flux    → generate_flux (default)
        #
        # Chroma/Flux/Qwen share latent packing so they all use
        # upscale_flux_latents.  Only Flux.2 needs a separate upscale
        # because it packs differently.  Qwen has its own decode path
        # because its VAE is 5D and uses per-channel mean/std denorm.
        is_flux2 = model_family in ("flux2", "flux2pipeline")
        is_chroma = model_family in ("chroma", "chromapipeline")
        is_qwen = model_family in qwen_families
        vae_scale = getattr(pipe, "vae_scale_factor", 8)

        # ── Upscale VAE: resolve switch + input into one effective bool ──
        #
        # Five cases to handle softly (never crash, just warn and fall back):
        #
        #   1. switch ON  + Qwen    + VAE connected    → USE VAE path
        #   2. switch ON  + Qwen    + no VAE connected → warn, bislerp fallback
        #   3. switch ON  + non-Qwen (any VAE state)   → warn, bislerp fallback
        #   4. switch OFF + VAE connected              → soft warn, bislerp
        #   5. switch OFF + no VAE                     → bislerp, silent
        #
        # This lets users A/B by flipping one checkbox without having to
        # disconnect the VAE or switch loaders.
        effective_use_vae = False
        if use_upscale_vae and upscale_vae is not None and is_qwen:
            if _vae_supports_upscale(pipe):
                effective_use_vae = True
                # Stage-aware summary so the user can confirm at a glance
                # what's bislerp vs what's VAE in THIS run.  do_s2 / do_s3
                # were computed earlier from the upscale_to_stage* values.
                print(
                    "[EricDiffusion-Adv-MS] Upscale VAE: ENABLED "
                    "(Qwen + compatible pipeline VAE)"
                )
                if do_s2:
                    print(
                        f"[EricDiffusion-Adv-MS]   S1→S2 upscale: bislerp "
                        f"@ upscale_to_stage2={upscale_to_stage2}× area "
                        f"(honored — S1→S2 is always bislerp)"
                    )
                if do_s3:
                    print(
                        f"[EricDiffusion-Adv-MS]   S2→S3 upscale: Wan2.1 2× "
                        f"VAE (upscale_to_stage3={upscale_to_stage3}× IGNORED "
                        f"— VAE forces fixed 2× linear)"
                    )
                print(
                    "[EricDiffusion-Adv-MS]   Final decode: Wan2.1 2× "
                    "upscale VAE (2× linear bump on top of last stage size)"
                )
            else:
                print(
                    "[EricDiffusion-Adv-MS] WARNING: upscale_vae connected "
                    "and switch is on, but pipeline VAE is not compatible "
                    "with the Wan2.1 latent space (missing z_dim / "
                    "latents_mean / latents_std).  Falling back to bislerp "
                    "inter-stage upscale and standard final decode."
                )
        elif use_upscale_vae and upscale_vae is None and is_qwen:
            print(
                "[EricDiffusion-Adv-MS] WARNING: use_upscale_vae is ON but "
                "no upscale_vae input is connected.  Falling back to "
                "bislerp inter-stage upscale and standard final decode."
            )
        elif use_upscale_vae and not is_qwen:
            print(
                f"[EricDiffusion-Adv-MS] WARNING: use_upscale_vae is ON but "
                f"model_family={model_family!r} is not Qwen-Image.  The "
                f"Wan2.1 upscale VAE only works on Qwen latent space — "
                f"ignoring and falling back to bislerp / standard decode."
            )
        elif (not use_upscale_vae) and upscale_vae is not None:
            print(
                "[EricDiffusion-Adv-MS] Note: upscale_vae is connected but "
                "use_upscale_vae is OFF.  Not using VAE upscale this run — "
                "flip the switch ON to enable it."
            )

        # ── Image-to-image: encode reference for Stage 1 if provided ──
        # Resolved once here so the Stage 1 dispatch below can just add
        # initial_latents / denoise to s1_common uniformly.  When no
        # reference is connected the legacy Stage 1 from-noise path is
        # bit-for-bit unchanged (initial_latents=None, denoise=1.0).
        s1_initial_latents = None
        s1_effective_denoise = 1.0
        is_i2i = reference_image is not None
        if is_i2i:
            if is_flux2:
                raise ValueError(
                    "Flux.2 i2i is not yet supported — stock diffusers has "
                    "no Flux2Img2ImgPipeline and the encode direction "
                    "requires hand-rolled patchify + batch-norm inverse.  "
                    "Disconnect reference_image, or use a different "
                    "pipeline family (Flux/Chroma/Qwen-Image)."
                )
            print(
                f"[EricDiffusion-Adv-MS] i2i mode: reference image connected, "
                f"s1_denoise={s1_denoise}"
            )
            s1_initial_latents = encode_image_to_packed_latents(
                pipe, reference_image, s1_w, s1_h, model_family,
            )
            s1_effective_denoise = float(s1_denoise)

        try:
            # ── Stage 1: draft from noise (or reference if i2i) ───────
            comfy.model_management.throw_exception_if_processing_interrupted()
            print(f"[EricDiffusion-Adv-MS] -- Stage 1/{stage_count} --")
            _apply_lora_stage_weights(pipe, pipeline, 1)

            s1_common = dict(
                pipe=pipe, prompt=prompt, negative_prompt=neg,
                width=s1_w, height=s1_h,
                num_inference_steps=s1_steps, guidance_scale=s1_cfg,
                sampler_name=s1_sampler, sigma_schedule=s1_sigma_schedule,
                generator=s1_gen, max_sequence_length=max_sequence_length,
                progress_cb=_stage_progress_cb,
                initial_latents=s1_initial_latents,
                denoise=s1_effective_denoise,
                eta=s1_eta,
            )
            if is_flux2:
                s1_latents, s1_ids = generate_flux2(**s1_common)
            elif is_chroma:
                s1_latents = generate_chroma(**s1_common)
                s1_ids = None
            elif is_qwen:
                s1_latents = generate_qwen(**s1_common)
                s1_ids = None
            else:
                s1_latents = generate_flux(**s1_common)
                s1_ids = None
            steps_completed[0] += s1_steps

            if not do_s2:
                if is_flux2:
                    image_tensor = decode_flux2_latents(pipe, s1_latents, s1_ids)
                elif is_qwen and effective_use_vae:
                    from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae
                    print("[EricDiffusion-Adv-MS]   Final decode: Wan2.1 2× upscale VAE")
                    try:
                        pipe.transformer = pipe.transformer.to("cpu")
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    image_tensor = decode_latents_with_upscale_vae(
                        s1_latents, upscale_vae, pipe.vae,
                        s1_h, s1_w, vae_scale,
                    )
                elif is_qwen:
                    image_tensor = decode_qwen_latents(pipe, s1_latents, s1_h, s1_w)
                else:
                    image_tensor = decode_flux_latents(pipe, s1_latents, s1_h, s1_w)
                return (image_tensor,)

            print(f"[EricDiffusion-Adv-MS]   S1 latents: {tuple(s1_latents.shape)}")

            # ── Stage 2: upscale + refine ─────────────────────────────
            comfy.model_management.throw_exception_if_processing_interrupted()
            print(f"[EricDiffusion-Adv-MS] -- Stage 2/{stage_count} --")
            _apply_lora_stage_weights(pipe, pipeline, 2)

            s2_common_tail = dict(
                pipe=pipe, prompt=prompt, negative_prompt=neg,
                width=s2_w, height=s2_h,
                num_inference_steps=s2_steps, guidance_scale=s2_cfg,
                sampler_name=s2_sampler, sigma_schedule=s2_sigma_schedule,
                generator=s2_gen, max_sequence_length=max_sequence_length,
                progress_cb=_stage_progress_cb,
                denoise=s2_denoise,
                eta=s2_eta,
            )
            if is_flux2:
                s2_input_latents = upscale_flux2_latents(
                    s1_latents, s1_ids, s1_h, s1_w, s2_h, s2_w,
                    vae_scale_factor=vae_scale,
                )
                print(f"[EricDiffusion-Adv-MS]   S2 input latents: {tuple(s2_input_latents.shape)}")
                s2_latents, s2_ids = generate_flux2(
                    initial_latents=s2_input_latents, **s2_common_tail,
                )
            elif is_chroma:
                s2_input_latents = upscale_flux_latents(
                    s1_latents, s1_h, s1_w, s2_h, s2_w,
                    vae_scale_factor=vae_scale,
                )
                print(f"[EricDiffusion-Adv-MS]   S2 input latents: {tuple(s2_input_latents.shape)}")
                s2_latents = generate_chroma(
                    initial_latents=s2_input_latents, **s2_common_tail,
                )
                s2_ids = None
            elif is_qwen:
                s2_input_latents = upscale_flux_latents(
                    s1_latents, s1_h, s1_w, s2_h, s2_w,
                    vae_scale_factor=vae_scale,
                )
                print(f"[EricDiffusion-Adv-MS]   S2 input latents: {tuple(s2_input_latents.shape)}")
                s2_latents = generate_qwen(
                    initial_latents=s2_input_latents, **s2_common_tail,
                )
                s2_ids = None
            else:
                s2_input_latents = upscale_flux_latents(
                    s1_latents, s1_h, s1_w, s2_h, s2_w,
                    vae_scale_factor=vae_scale,
                )
                print(f"[EricDiffusion-Adv-MS]   S2 input latents: {tuple(s2_input_latents.shape)}")
                s2_latents = generate_flux(
                    initial_latents=s2_input_latents, **s2_common_tail,
                )
                s2_ids = None
            steps_completed[0] += s2_steps

            if not do_s3:
                if is_flux2:
                    image_tensor = decode_flux2_latents(pipe, s2_latents, s2_ids)
                elif is_qwen and effective_use_vae:
                    from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae
                    print("[EricDiffusion-Adv-MS]   Final decode: Wan2.1 2× upscale VAE")
                    try:
                        pipe.transformer = pipe.transformer.to("cpu")
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    image_tensor = decode_latents_with_upscale_vae(
                        s2_latents, upscale_vae, pipe.vae,
                        s2_h, s2_w, vae_scale,
                    )
                elif is_qwen:
                    image_tensor = decode_qwen_latents(pipe, s2_latents, s2_h, s2_w)
                else:
                    image_tensor = decode_flux_latents(pipe, s2_latents, s2_h, s2_w)
                return (image_tensor,)

            print(f"[EricDiffusion-Adv-MS]   S2 latents: {tuple(s2_latents.shape)}")

            # ── Stage 3: upscale + polish ─────────────────────────────
            comfy.model_management.throw_exception_if_processing_interrupted()
            print(f"[EricDiffusion-Adv-MS] -- Stage 3/{stage_count} --")
            _apply_lora_stage_weights(pipe, pipeline, 3)

            s3_common_tail = dict(
                pipe=pipe, prompt=prompt, negative_prompt=neg,
                width=s3_w, height=s3_h,
                num_inference_steps=s3_steps, guidance_scale=s3_cfg,
                sampler_name=s3_sampler, sigma_schedule=s3_sigma_schedule,
                generator=s3_gen, max_sequence_length=max_sequence_length,
                progress_cb=_stage_progress_cb,
                denoise=s3_denoise,
                eta=s3_eta,
            )
            if is_flux2:
                s3_input_latents = upscale_flux2_latents(
                    s2_latents, s2_ids, s2_h, s2_w, s3_h, s3_w,
                    vae_scale_factor=vae_scale,
                )
                print(f"[EricDiffusion-Adv-MS]   S3 input latents: {tuple(s3_input_latents.shape)}")
                s3_latents, s3_ids = generate_flux2(
                    initial_latents=s3_input_latents, **s3_common_tail,
                )
                steps_completed[0] += s3_steps
                image_tensor = decode_flux2_latents(pipe, s3_latents, s3_ids)
            elif is_chroma:
                s3_input_latents = upscale_flux_latents(
                    s2_latents, s2_h, s2_w, s3_h, s3_w,
                    vae_scale_factor=vae_scale,
                )
                print(f"[EricDiffusion-Adv-MS]   S3 input latents: {tuple(s3_input_latents.shape)}")
                s3_latents = generate_chroma(
                    initial_latents=s3_input_latents, **s3_common_tail,
                )
                steps_completed[0] += s3_steps
                image_tensor = decode_flux_latents(pipe, s3_latents, s3_h, s3_w)
            elif is_qwen:
                # Upscale path: Wan2.1 2× VAE decode→encode replaces bislerp.
                # The VAE upscale is fixed 2×, so s3_h/s3_w from the user's
                # upscale_to_stage3 factor get overridden with the VAE's
                # actual output size.  generate_qwen must run at THAT size,
                # not the user-configured one — otherwise the shape check
                # inside generate_qwen will reject the initial latents.
                if effective_use_vae:
                    from .eric_qwen_upscale_vae import upscale_between_stages
                    print("[EricDiffusion-Adv-MS]   S2→S3 upscale: Wan2.1 2× VAE")
                    # Emit a loud callout if the user's configured
                    # upscale_to_stage3 diverged from what the VAE forces.
                    # Default is 2.0 (matches the VAE exactly) so typical
                    # runs stay quiet; only non-default values get the NOTE.
                    if abs(upscale_to_stage3 - 2.0) > 1e-6:
                        print(
                            f"[EricDiffusion-Adv-MS]   NOTE: "
                            f"upscale_to_stage3={upscale_to_stage3}× was set "
                            f"but IGNORED — upscale VAE forces fixed 2× linear"
                        )
                    s3_input_latents, s3_eff_h, s3_eff_w = upscale_between_stages(
                        s2_latents, upscale_vae, pipe.vae,
                        s2_h, s2_w, vae_scale,
                    )
                    print(
                        f"[EricDiffusion-Adv-MS]   S3 target size: "
                        f"{s3_w}×{s3_h} → {s3_eff_w}×{s3_eff_h} "
                        f"(fixed 2× override from upscale VAE)"
                    )
                else:
                    s3_input_latents = upscale_flux_latents(
                        s2_latents, s2_h, s2_w, s3_h, s3_w,
                        vae_scale_factor=vae_scale,
                    )
                    s3_eff_h, s3_eff_w = s3_h, s3_w
                print(f"[EricDiffusion-Adv-MS]   S3 input latents: {tuple(s3_input_latents.shape)}")
                s3_latents = generate_qwen(
                    initial_latents=s3_input_latents,
                    **{**s3_common_tail, "width": s3_eff_w, "height": s3_eff_h},
                )
                steps_completed[0] += s3_steps
                if effective_use_vae:
                    from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae
                    print("[EricDiffusion-Adv-MS]   Final decode: Wan2.1 2× upscale VAE")
                    # Offload transformer before the high-res VAE decode
                    # to free VRAM, matching ultragen's _final_decode pattern.
                    try:
                        pipe.transformer = pipe.transformer.to("cpu")
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    image_tensor = decode_latents_with_upscale_vae(
                        s3_latents, upscale_vae, pipe.vae,
                        s3_eff_h, s3_eff_w, vae_scale,
                    )
                else:
                    image_tensor = decode_qwen_latents(pipe, s3_latents, s3_eff_h, s3_eff_w)
            else:
                s3_input_latents = upscale_flux_latents(
                    s2_latents, s2_h, s2_w, s3_h, s3_w,
                    vae_scale_factor=vae_scale,
                )
                print(f"[EricDiffusion-Adv-MS]   S3 input latents: {tuple(s3_input_latents.shape)}")
                s3_latents = generate_flux(
                    initial_latents=s3_input_latents, **s3_common_tail,
                )
                steps_completed[0] += s3_steps
                image_tensor = decode_flux_latents(pipe, s3_latents, s3_h, s3_w)

            return (image_tensor,)

        finally:
            # CRITICAL: restore transformer to its original GPU device.
            # When use_upscale_vae was ON, the final-decode blocks moved
            # pipe.transformer to CPU to free VRAM for the Wan2.1 decode.
            # If we don't restore it here, the next run starts with
            # transformer on CPU and silently runs inference on CPU.
            try:
                current_device = next(pipe.transformer.parameters()).device
                if current_device != _transformer_device:
                    pipe.transformer = pipe.transformer.to(_transformer_device)
                    print(
                        f"[EricDiffusion-Adv-MS] Transformer restored to "
                        f"{_transformer_device} (was {current_device} after "
                        f"upscale VAE decode)"
                    )
            except Exception as e:
                print(
                    f"[EricDiffusion-Adv-MS] WARNING: failed to restore "
                    f"transformer to original device: {e}.  Next run may "
                    f"be very slow if it picks up the CPU placement — "
                    f"restart ComfyUI if generation is stuck."
                )

            if offload_vae and not using_device_map and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()
