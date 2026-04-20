# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Advanced Generate (Flux/Chroma/Flux2)

Single-stage text-to-image generation using a **manual denoising loop**
instead of the pipeline's built-in ``__call__`` method.  The manual loop
lets us run higher-order samplers (Heun, RK3) that require multiple
model evaluations per denoising step — something the pipeline's
scheduler-based loop can't accommodate via a simple ``step()`` override.

Relationship to the other generate nodes
----------------------------------------
- **Eric Diffusion Generate**: the "safe" node that calls ``pipe()``
  directly.  Keeps working regardless of what we do here.
- **Eric Diffusion UltraGen**: the multi-stage "opinionated" node with
  upscale VAE support, also via ``pipe()``.
- **Eric Diffusion Advanced Generate** (this file): experimental node
  that bypasses ``pipe()`` entirely and runs the denoising loop by hand.
  Enables samplers that the other two can't use.

Supported model families
------------------------
Flux, Flux2, Chroma, and Qwen-Image (text-to-image).  Each family has
its own dispatch path in the manual loop module because their encode
/ transformer / decode signatures differ, but all five samplers work
uniformly across all families.

Qwen-Image-Edit (image-conditioning) is NOT supported here — that's a
separate pipeline with image-conditioned encode_prompt and extra
transformer kwargs.  Use the dedicated ``eric_qwen_edit_*`` nodes.

Samplers
--------
Currently exposes classical-but-solid 1st–3rd order methods:
  - flow_euler      : 1st-order (baseline, matches default pipe behavior)
  - flow_heun       : 2nd-order predictor-corrector (2× model calls/step)
  - flow_rk3        : 3rd-order Runge-Kutta (3× model calls/step)
  - flow_multistep2 : 2nd-order Adams-Bashforth (1× model calls/step)
  - flow_multistep3 : 3rd-order Adams-Bashforth (1× model calls/step)

RES4LYF integration (res_2s, deis_2m, bong_tangent, etc.) is planned for
a future session — RES4LYF's ``sample_rk_beta`` expects a ComfyUI
ModelPatcher wrapper rather than a plain denoiser callable, which
requires additional infrastructure.

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
from datetime import datetime
from typing import Tuple

import torch

from .eric_qwen_edit_utils import pil_to_tensor
from .eric_diffusion_utils import build_model_metadata
from .eric_diffusion_generate import (
    ASPECT_RATIOS,
    compute_dimensions,
    resolve_override_dimensions,
)
from .eric_diffusion_manual_loop import (
    sampler_names,
    sampler_cost,
    SIGMA_SCHEDULE_NAMES,
    generate_flux,
    decode_flux_latents,
    generate_chroma,
    generate_flux2,
    decode_flux2_latents,
    generate_qwen,
    decode_qwen_latents,
    encode_image_to_packed_latents,
)


class EricDiffusionAdvancedGenerate:
    """
    Advanced single-stage text-to-image generation for Flux-family and
    Qwen-Image pipelines.

    Uses a manual denoising loop so higher-order samplers can be applied.
    Cost varies by sampler: Heun doubles runtime, RK3 triples it — but
    accuracy improves correspondingly, especially at low step counts.

    Qwen-Image-Edit (image-conditioning) is NOT supported here — use the
    dedicated Qwen edit nodes for that.
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
                    "tooltip": (
                        "From Eric Diffusion Loader or Component Loader. "
                        "Supports Flux, Flux2, Chroma, and Qwen-Image (t2i). "
                        "Qwen-Image-Edit is NOT supported — use the dedicated "
                        "Qwen edit nodes for image-conditioned generation."
                    ),
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "Describe the image you want to generate.",
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Used when cfg_scale > 1.0. Doubles model cost per "
                        "step because each step needs a negative forward pass."
                    ),
                }),
                "aspect_ratio": (ratio_names, {
                    "default": "1:1   Square",
                }),
                "target_mp": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.25,
                    "max": 16.0,
                    "step": 0.25,
                    "tooltip": (
                        "Target megapixels.  Flux-family models are typically "
                        "trained around 1–2 MP; higher resolutions may degrade."
                    ),
                }),
                "override_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 16,
                    "tooltip": (
                        "Explicit output width in pixels.  When this AND "
                        "override_height are both non-zero, they override "
                        "aspect_ratio + target_mp entirely (those inputs "
                        "are ignored).  When both are 0 (default), the "
                        "node uses the aspect/MP path as before — "
                        "workflows that don't touch this input behave "
                        "exactly as they did.\n\n"
                        "Must be a multiple of 16.  Non-multiples are "
                        "floored with a log note (1920→1920, 1920→1920, "
                        "1080→1072).\n\n"
                        "Common video resolutions:\n"
                        "• 1920×1088 (1080p, 16-aligned — use this, NOT 1080)\n"
                        "• 1280×720 (720p, exact)\n"
                        "• 3840×2160 (4K, exact)"
                    ),
                }),
                "override_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 16,
                    "tooltip": (
                        "Explicit output height in pixels.  Must be set "
                        "non-zero together with override_width to "
                        "activate the explicit-dimension path."
                    ),
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": (
                        "Denoising steps.  Higher-order samplers (heun, rk3) "
                        "converge at lower step counts — try 15-20 for heun, "
                        "12-15 for rk3 to match euler at 25-30 steps."
                    ),
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Guidance scale.\n"
                        "• Flux dev / Flux.2: distilled guidance embedding — "
                        "typical 3.0-5.0, single transformer call per step.\n"
                        "• Chroma / Qwen-Image: classical CFG — typical 3.0-5.0 "
                        "for Chroma, 4.0 for Qwen. DOUBLES cost per step when >1.0 "
                        "because a negative forward pass is required.\n"
                        "Values >1.0 on classical-CFG models require a "
                        "negative_prompt to have any effect."
                    ),
                }),
                "sampler": (sampler_names(), {
                    "default": "flow_heun",
                    "tooltip": (
                        "Denoising sampler algorithm. All samplers are flow-matching.\n\n"
                        "• flow_euler     — 1st order baseline. Safe everywhere.\n"
                        "• flow_heun      — 2nd order predictor-corrector. 2× cost\n"
                        "  per step. General-purpose quality upgrade. Works at any\n"
                        "  step count.\n"
                        "• flow_rk3       — 3rd order Runge-Kutta. 3× cost per step.\n"
                        "  NEEDS ≥15 STEPS. At low step counts (<15) RK3 extrapolates\n"
                        "  intermediate states that land outside the model's training\n"
                        "  distribution, producing edge artifacts and spurious detail.\n"
                        "  Use at 15-25 steps for highest quality per compute dollar,\n"
                        "  NEVER at 10 or fewer.\n"
                        "• flow_multistep2 — 2nd order Adams-Bashforth. SAME cost as\n"
                        "  Euler (1 model call per step). Free quality upgrade. Uses\n"
                        "  previous step's velocity history instead of extrapolating\n"
                        "  model inputs, so robust at any step count.\n"
                        "• flow_multistep3 — 3rd order Adams-Bashforth. Also same cost\n"
                        "  as Euler. Slightly better than multistep2 at moderate+\n"
                        "  step counts.\n\n"
                        "Rule of thumb: multistep2 is the safest default. Heun is\n"
                        "worth the 2× cost when detail matters. RK3 only when you\n"
                        "have the step budget and need the best possible quality."
                    ),
                }),
                "sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "linear",
                    "tooltip": (
                        "Sigma schedule curve — controls where steps concentrate:\n"
                        "• linear    — uniform spacing (matches default pipe behavior)\n"
                        "• balanced  — Karras ρ=3, moderate low-sigma concentration\n"
                        "• karras    — Karras ρ=7, heavy low-sigma (fine detail)\n"
                        "• beta57    — beta(5,7), RES4LYF-inspired mid-low concentration\n"
                        "• beta75    — beta(7,5), mid-high concentration\n"
                        "• beta33    — beta(3,3), symmetric around mid\n"
                        "• beta13    — beta(1,3), heavy low-sigma concentration\n"
                        "• beta31    — beta(3,1), heavy high-sigma concentration\n"
                        "Beta schedules are a known good family — try beta57 on\n"
                        "Chroma and other flow-matching models that don't converge\n"
                        "as well with pure karras."
                    ),
                }),
                "eta": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Stochastic sampling coefficient.\n"
                        "• 0.0 — fully deterministic (default, same-seed = same output)\n"
                        "• 0.3-0.5 — moderate stochasticity, often improves detail\n"
                        "• 0.7+ — heavy noise re-injection per step (can be erratic)\n"
                        "Fresh gaussian noise is injected at every step scaled by eta,\n"
                        "letting the model 'escape attractors' in its velocity field.\n"
                        "RES4LYF workflows commonly use eta=0.5 for Chroma — try it\n"
                        "if deterministic output looks over-smoothed or lacks detail."
                    ),
                }),
                "max_sequence_length": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": (
                        "Max prompt token length for the text encoder.\n"
                        "• Flux dev / Chroma: T5-XXL trained at 512 — stay at 512.\n"
                        "• Flux.2: Mistral3-Small supports much more. 512 is the\n"
                        "  default but you can push higher for long prompts\n"
                        "  (VRAM cost scales ~quadratically with prompt length)."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = non-deterministic).",
                }),

                # ── Image-to-image (reference conditioning) ───────────
                "reference_image": ("IMAGE", {
                    "tooltip": (
                        "Optional. Connect a reference image to use as "
                        "the starting state for denoising (classical "
                        "image-to-image).  The image will be encoded "
                        "through the pipeline's VAE and resized to match "
                        "the target aspect_ratio + target_mp using "
                        "bicubic interpolation.\n\n"
                        "Works for Flux, Chroma, and Qwen-Image.  Flux.2 "
                        "i2i is not yet supported and will raise a clear "
                        "error.\n\n"
                        "Leave unconnected for pure text-to-image (current "
                        "behavior is unchanged — this input is bit-for-bit "
                        "optional)."
                    ),
                }),
                "denoise": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Denoise strength for image-to-image.  Only used "
                        "when reference_image is connected — ignored "
                        "otherwise.\n\n"
                        "• 1.0  — full re-denoise from noise (reference is "
                        "effectively thrown away, same as no reference)\n"
                        "• 0.85 — default: meaningful refinement, composition "
                        "preserved, details rewritten\n"
                        "• 0.65 — conservative: reference stays very visible\n"
                        "• 0.30 — polish: minimal change, mostly just VAE "
                        "round-trip\n"
                        "• 0.05 — near-pass-through: barely touched"
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
        target_mp: float = 2.0,
        override_width: int = 0,
        override_height: int = 0,
        steps: int = 20,
        cfg_scale: float = 3.5,
        sampler: str = "flow_heun",
        sigma_schedule: str = "linear",
        eta: float = 0.0,
        max_sequence_length: int = 512,
        seed: int = 0,
        reference_image=None,
        denoise: float = 0.85,
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]
        model_family = pipeline.get("model_family", "unknown")
        offload_vae = pipeline.get("offload_vae", False)
        using_device_map = hasattr(pipe, "hf_device_map")

        # ── Family check ───────────────────────────────────────────────
        flux_families = (
            "flux", "flux2", "flux2klein", "chroma",
            "fluxpipeline", "flux2pipeline", "flux2kleinpipeline", "chromapipeline",
        )
        qwen_families = (
            "qwen-image", "qwenimage", "qwenimagepipeline",
        )
        supported = flux_families + qwen_families
        if model_family not in supported:
            raise ValueError(
                f"Eric Diffusion Advanced Generate supports Flux-family "
                f"(Flux/Flux2/Chroma) and Qwen-Image pipelines.  Got "
                f"model_family={model_family!r}.\n"
                f"Use 'Eric Diffusion Generate' for other model types, or "
                f"the dedicated Qwen-Image-Edit nodes for Qwen edit."
            )

        # ── Dimensions ────────────────────────────────────────────────
        # Default path: compute from aspect_ratio + target_mp.  Override
        # path: user-supplied explicit pixels (floored to 16-alignment).
        w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_ratio, (1, 1))
        default_w, default_h = compute_dimensions(w_ratio, h_ratio, target_mp)
        width, height = resolve_override_dimensions(
            override_width, override_height,
            default_w, default_h,
            log_prefix="[EricDiffusion-Adv]",
        )

        # ── Image-to-image: encode reference if provided ─────────────
        # Resolved once up front so the dispatch below can just add
        # initial_latents / denoise to common_kwargs uniformly.  When
        # no reference is connected the legacy text-to-image path is
        # bit-for-bit unchanged (initial_latents=None, denoise=1.0).
        initial_latents = None
        effective_denoise = 1.0
        is_i2i = reference_image is not None
        if is_i2i:
            is_flux2_preflight = model_family in ("flux2", "flux2klein", "flux2pipeline", "flux2kleinpipeline")
            if is_flux2_preflight:
                raise ValueError(
                    "Flux.2 i2i is not yet supported — stock diffusers has "
                    "no Flux2Img2ImgPipeline and the encode direction "
                    "requires hand-rolled patchify + batch-norm inverse. "
                    "Disconnect reference_image, or use a different "
                    "pipeline family (Flux/Chroma/Qwen-Image)."
                )
            print(
                f"[EricDiffusion-Adv] i2i mode: reference image connected, "
                f"denoise={denoise}"
            )
            initial_latents = encode_image_to_packed_latents(
                pipe, reference_image, width, height, model_family,
            )
            effective_denoise = float(denoise)

        # Cost reporting
        cost_mult = sampler_cost(sampler)
        total_evals = steps * cost_mult * (2 if cfg_scale > 1.0 and negative_prompt else 1)

        print(
            f"[EricDiffusion-Adv] {model_family} — {width}×{height} "
            f"({width * height / 1e6:.2f} MP), steps={steps}, "
            f"sampler={sampler} (x{cost_mult}/step), cfg={cfg_scale}, "
            f"~{total_evals} transformer calls total"
            + (f", i2i denoise={effective_denoise}" if is_i2i else "")
        )

        # ── Generator ─────────────────────────────────────────────────
        # Use a CPU generator regardless of compute device.  CPU RNG
        # (Mersenne Twister) doesn't have the shape-dependent per-row
        # pattern artifacts that CUDA's Philox RNG can exhibit, and
        # diffusers' randn_tensor helper handles the CPU→GPU transfer
        # automatically.  This is the standard reproducibility pattern
        # used throughout the diffusers library.
        exec_device = getattr(pipe, "_execution_device", None) or next(
            pipe.transformer.parameters()
        ).device
        generator = (
            torch.Generator(device="cpu").manual_seed(seed) if seed > 0 else None
        )

        # ── Progress bar + cancel ─────────────────────────────────────
        import comfy.utils
        import comfy.model_management
        pbar = comfy.utils.ProgressBar(steps)

        last_step = [0]

        def progress_cb(step_int):
            if step_int > last_step[0]:
                pbar.update(step_int - last_step[0])
                last_step[0] = step_int
            comfy.model_management.throw_exception_if_processing_interrupted()

        comfy.model_management.throw_exception_if_processing_interrupted()

        # ── VAE: move back to GPU if offloaded ─────────────────────────
        if offload_vae and not using_device_map and hasattr(pipe, "vae"):
            vae_target = next(pipe.transformer.parameters()).device
            pipe.vae = pipe.vae.to(vae_target)

        # ── Run the manual loop ────────────────────────────────────────
        is_flux2 = model_family in ("flux2", "flux2klein", "flux2pipeline", "flux2kleinpipeline")
        is_chroma = model_family in ("chroma", "chromapipeline")
        is_qwen = model_family in qwen_families

        common_kwargs = dict(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt.strip() or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            sampler_name=sampler,
            sigma_schedule=sigma_schedule,
            generator=generator,
            max_sequence_length=max_sequence_length,
            progress_cb=progress_cb,
            eta=eta,
            initial_latents=initial_latents,
            denoise=effective_denoise,
        )

        try:
            if is_flux2:
                final_latents, final_ids = generate_flux2(**common_kwargs)
                image_tensor = decode_flux2_latents(pipe, final_latents, final_ids)
            elif is_chroma:
                final_latents = generate_chroma(**common_kwargs)
                image_tensor = decode_flux_latents(pipe, final_latents, height, width)
            elif is_qwen:
                final_latents = generate_qwen(**common_kwargs)
                image_tensor = decode_qwen_latents(pipe, final_latents, height, width)
            else:
                final_latents = generate_flux(**common_kwargs)
                image_tensor = decode_flux_latents(pipe, final_latents, height, width)

        finally:
            if offload_vae and not using_device_map and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        metadata = {
            **build_model_metadata(pipeline),
            "node_type":       "adv-gen",
            "seed":            seed,
            "steps":           steps,
            "cfg_scale":       cfg_scale,
            "sampler":         sampler,
            "sampler_s2":      "",
            "sampler_s3":      "",
            "sigma_schedule":  sigma_schedule,
            "eta":             eta,
            "denoise":         denoise,
            "prompt":          prompt,
            "negative_prompt": negative_prompt,
            "width":           width,
            "height":          height,
            "timestamp":       datetime.now().isoformat(),
        }
        return (image_tensor, metadata)
