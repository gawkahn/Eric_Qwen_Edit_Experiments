# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Advanced Edit Multi-Stage — Qwen-Image-Edit with per-
stage samplers, reference conditioning, and (slice 3) upscale VAE bridge.

Three-stage compositional editing with the full manual-loop feature set:
  Stage 1 — draft at low MP from fresh noise + references as context
  Stage 2 — upscale noise latents + partial re-denoise, refs still active
  Stage 3 — upscale + polish, refs still active, optional upscale VAE

References are encoded once before the stage loop and kept constant
across all stages (the default behavior).  The transformer attends to
them at every step via ``hidden_states`` concatenation and ``img_shapes``
positional encoding.

Output aspect ratio is derived from the LAST connected reference image
— same convention as single-stage Advanced Edit.  If you want a
different aspect, reorder the slot connections so the reference with the
desired shape is the last connected one.  Explicit pixel resolution
input is backlogged (tracked with the same request for Advanced Generate).

Relationship to other edit nodes
--------------------------------
- **Eric Qwen-Edit Multi-Image**: legacy pipe()-based multi-reference
  edit, single-stage.  Still supported, still the safe fallback.
- **Eric Diffusion Advanced Edit** (single-stage): manual-loop edit with
  per-stage sampler/sigma/eta.  Good for quick edits.
- **Eric Diffusion Advanced Edit Multi-Stage** (this file): three-stage
  manual-loop edit with upscale VAE bridge, the headline high-res
  compositional editing node.

Phase 1 Advanced Edit was the single-stage scaffolding.  Phase 2 (this
file) adds the multistage payoff: 16-31MP compositional edits through
the Wan2.1 upscale VAE chain, matching what Advanced Multi-Stage does
for Generate.

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
from typing import Tuple

from .eric_diffusion_advanced_multistage import (
    _apply_lora_stage_weights,
    _vae_supports_upscale,
)
from .eric_diffusion_generate import resolve_override_dimensions
from .eric_diffusion_manual_loop import (
    sampler_names,
    sampler_cost,
    SIGMA_SCHEDULE_NAMES,
    generate_qwen_edit,
    decode_qwen_latents,
    upscale_flux_latents,
    _calculate_qwen_edit_dimensions,
    _QWEN_EDIT_VAE_IMAGE_SIZE,
)


class EricDiffusionAdvancedEditMultistage:
    """
    Advanced three-stage Qwen-Image-Edit with per-stage samplers and
    up to 4 reference images (VL + Ref flags).

    Uses the manual denoising loop at every stage so all five samplers,
    all sigma schedules, and stochastic eta sampling work uniformly
    across edit workflows.  LoRA per-stage weights via the Eric Qwen-Edit
    LoRA Stacker annotation are applied at the start of each stage.

    Output dimensions: derived from the LAST connected reference image's
    aspect ratio at the stage's target MP area.  S1 at ``s1_mp``, S2 at
    ``s1_mp * upscale_to_stage2``, S3 at the S2 area × ``upscale_to_stage3``.
    To change aspect, reorder the reference slot connections — the last
    connected image's aspect drives output.

    Set ``upscale_to_stage2 = 0`` to output Stage 1 only.
    Set ``upscale_to_stage3 = 0`` to stop after Stage 2.

    Upscale VAE bridge support lands in Phase 2 slice 3 (not in this
    commit).  Re-encode-per-stage switch lands in slice 4.
    """

    CATEGORY = "Eric Diffusion"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE", {
                    "tooltip": (
                        "From Eric Qwen-Edit Loader or Component Loader. "
                        "Pipe a LoRA stacker through if you want LoRAs "
                        "with per-stage weights (Eric Qwen-Edit LoRA Stacker)."
                    ),
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": (
                        "Edit instruction.  Reference images are "
                        "automatically labeled 'Picture 1'..'Picture N' "
                        "in the order they're connected (image_1 = "
                        "Picture 1, etc.).  Use those labels for "
                        "compositional directives like 'the background "
                        "from Picture 1 with the subject from Picture 2'."
                    ),
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),

                # ── Reference image slots (same as Phase 1) ──────────
                "image_1": ("IMAGE", {
                    "tooltip": "Reference slot 1 → Picture 1 in VL prompt.",
                }),
                "image_1_vl": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Feed image_1 to the Qwen2.5-VL text encoder.",
                }),
                "image_1_ref": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use image_1 as a pixel latent anchor.",
                }),
                "image_2": ("IMAGE",),
                "image_2_vl": ("BOOLEAN", {"default": True}),
                "image_2_ref": ("BOOLEAN", {"default": True}),
                "image_3": ("IMAGE",),
                "image_3_vl": ("BOOLEAN", {"default": True}),
                "image_3_ref": ("BOOLEAN", {"default": True}),
                "image_4": ("IMAGE",),
                "image_4_vl": ("BOOLEAN", {"default": True}),
                "image_4_ref": ("BOOLEAN", {"default": True}),

                # ── Shared stage controls ─────────────────────────────
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = non-deterministic).",
                }),
                "seed_mode": (
                    ["offset_per_stage", "same_all_stages", "random_per_stage"],
                    {
                        "default": "offset_per_stage",
                        "tooltip": (
                            "How the seed propagates across stages.\n"
                            "• offset_per_stage — s1=seed, s2=seed+1, s3=seed+2\n"
                            "• same_all_stages — all stages share the same seed\n"
                            "• random_per_stage — each stage gets a fresh random seed"
                        ),
                    },
                ),
                "s1_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stage 1 stochastic sampling coefficient.\n"
                        "0.0 = deterministic.  For edit tasks usually 0.0.\n"
                        "Higher values add diversity but reduce reference fidelity."
                    ),
                }),
                "max_sequence_length": ("INT", {
                    "default": 1024, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Max prompt token length (Qwen Edit default 1024).",
                }),

                # ── Stage 1 ───────────────────────────────────────────
                "s1_mp": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 4.0, "step": 0.1,
                    "tooltip": (
                        "Stage 1 target area in megapixels.  Aspect ratio "
                        "is derived from the last connected reference image "
                        "unless override_s1_width/height are set."
                    ),
                }),
                "override_s1_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 16,
                    "tooltip": (
                        "Explicit Stage 1 width in pixels.  When this "
                        "AND override_s1_height are both non-zero, they "
                        "override the default 'derive from last "
                        "reference at s1_mp' behavior for Stage 1.  "
                        "Stage 2 and Stage 3 still scale from here via "
                        "upscale_to_stage2 / upscale_to_stage3 factors, "
                        "preserving the override's aspect ratio.\n\n"
                        "Must be a multiple of 16.  1080p requires "
                        "1088 not 1080.\n\n"
                        "Leave at 0 to use the default last-reference-"
                        "aspect behavior."
                    ),
                }),
                "override_s1_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 16,
                    "tooltip": (
                        "Explicit Stage 1 height in pixels.  Must be "
                        "set non-zero together with override_s1_width."
                    ),
                }),
                "s1_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "s1_cfg": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "s1_sampler": (sampler_names(), {
                    "default": "flow_heun",
                    "tooltip": (
                        "Stage 1 sampler.  flow_heun is the safe default "
                        "for edit drafting at low MP."
                    ),
                }),
                "s1_sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "linear",
                }),

                # ── Stage 2 ───────────────────────────────────────────
                "upscale_to_stage2": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": "Area upscale factor S1→S2 (0 = output S1).",
                }),
                "s2_steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "s2_cfg": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "s2_denoise": ("FLOAT", {
                    "default": 0.85, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Partial denoise strength at S2.  0.85 = meaningful "
                        "refinement while keeping S1's composition."
                    ),
                }),
                "s2_sampler": (sampler_names(), {"default": "flow_heun"}),
                "s2_sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "balanced",
                }),
                "s2_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stage 2 stochastic sampling coefficient.\n"
                        "0.0 = deterministic (recommended for refinement)."
                    ),
                }),

                # ── Stage 3 ───────────────────────────────────────────
                "upscale_to_stage3": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5,
                    "tooltip": (
                        "Area upscale factor S2→S3 (0 = output S2, stage 3 "
                        "disabled).\n"
                        "\n"
                        "In Phase 2 slice 3 (not yet): when use_upscale_vae "
                        "is ON, this value is IGNORED and the Wan2.1 VAE "
                        "forces a fixed 2× linear upscale."
                    ),
                }),
                "s3_steps": ("INT", {"default": 15, "min": 1, "max": 200}),
                "s3_cfg": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "s3_denoise": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Partial denoise strength at S3.  0.5 = polish pass, "
                        "minimal structural changes from S2."
                    ),
                }),
                "s3_sampler": (sampler_names(), {"default": "flow_multistep2"}),
                "s3_sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "karras",
                }),
                "s3_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Stage 3 stochastic sampling coefficient.\n"
                        "0.0 = deterministic (recommended for final polish)."
                    ),
                }),

                # ── Upscale VAE bridge ────────────────────────────────
                "use_upscale_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When ON with an upscale_vae connected, the "
                        "Wan2.1 2× VAE replaces bislerp for the S2→S3 "
                        "inter-stage upscale AND replaces the final VAE "
                        "decode.  In a 3-stage edit you get ~4× the "
                        "resolution of the stage settings — e.g. S1 at "
                        "0.5 MP, S2 at 1 MP (bislerp), S3 at 4 MP (VAE), "
                        "final image at 16 MP (VAE final decode).\n\n"
                        "References don't go through the upscale VAE — "
                        "they stay at their original encoding and are "
                        "attended to at every stage.\n\n"
                        "Soft-fails if the switch is on but the VAE isn't "
                        "connected — prints a warning and falls back to "
                        "bislerp inter-stage + standard decode, no crash."
                    ),
                }),
                "upscale_vae": ("UPSCALE_VAE", {
                    "tooltip": (
                        "Optional. Output of 'Eric Qwen Upscale VAE "
                        "Loader'. Only honored when 'use_upscale_vae' is "
                        "ON. Connecting this without flipping the switch "
                        "prints a soft note and does nothing."
                    ),
                }),
            },
        }

    def generate(
        self,
        pipeline: dict,
        prompt: str,
        negative_prompt: str = "",
        image_1=None, image_1_vl: bool = True, image_1_ref: bool = True,
        image_2=None, image_2_vl: bool = True, image_2_ref: bool = True,
        image_3=None, image_3_vl: bool = True, image_3_ref: bool = True,
        image_4=None, image_4_vl: bool = True, image_4_ref: bool = True,
        seed: int = 0,
        seed_mode: str = "offset_per_stage",
        s1_eta: float = 0.0,
        max_sequence_length: int = 1024,
        s1_mp: float = 0.5,
        override_s1_width: int = 0,
        override_s1_height: int = 0,
        s1_steps: int = 20,
        s1_cfg: float = 4.0,
        s1_sampler: str = "flow_heun",
        s1_sigma_schedule: str = "linear",
        upscale_to_stage2: float = 2.0,
        s2_steps: int = 20,
        s2_cfg: float = 4.0,
        s2_denoise: float = 0.85,
        s2_sampler: str = "flow_heun",
        s2_sigma_schedule: str = "balanced",
        s2_eta: float = 0.0,
        upscale_to_stage3: float = 2.0,
        s3_steps: int = 15,
        s3_cfg: float = 4.0,
        s3_denoise: float = 0.5,
        s3_sampler: str = "flow_multistep2",
        s3_sigma_schedule: str = "karras",
        s3_eta: float = 0.0,
        use_upscale_vae: bool = False,
        upscale_vae=None,
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]

        # ── Collect connected slots + auto-compact (same as single-stage) ──
        slot_configs = [
            (1, image_1, image_1_vl, image_1_ref),
            (2, image_2, image_2_vl, image_2_ref),
            (3, image_3, image_3_vl, image_3_ref),
            (4, image_4, image_4_vl, image_4_ref),
        ]
        connected = [
            (idx, img, vl, ref)
            for idx, img, vl, ref in slot_configs if img is not None
        ]
        if not connected:
            raise ValueError(
                "Eric Diffusion Advanced Edit Multi-Stage requires at least "
                "one reference image connected (image_1 through image_4).  "
                "For pure text-to-image without references use the regular "
                "Advanced Multi-Stage node."
            )

        original_slots = [c[0] for c in connected]
        remapped_slots = list(range(1, len(connected) + 1))
        if original_slots != remapped_slots:
            print(
                f"[EricDiffusion-AdvEditMS] WARNING: sparse slot "
                f"configuration — slots {original_slots} will be remapped "
                f"to Picture numbers {remapped_slots} for the VL encoder.  "
                f"If your prompt references 'Picture N', verify N matches "
                f"the remapped index, NOT the original slot number."
            )
            for orig, new in zip(original_slots, remapped_slots):
                if orig != new:
                    print(
                        f"[EricDiffusion-AdvEditMS]   image_{orig} → Picture {new}"
                    )

        reference_images = [c[1] for c in connected]
        vl_flags = [c[2] for c in connected]
        ref_flags = [c[3] for c in connected]
        for i, (vl, ref) in enumerate(zip(vl_flags, ref_flags)):
            if not vl and not ref:
                print(
                    f"[EricDiffusion-AdvEditMS] WARNING: Picture {i+1} has "
                    f"both VL and Ref flags disabled — the model won't see "
                    f"this image at all."
                )

        # ── Derive output aspect from the LAST connected reference ────
        # Matches Phase 1 single-stage convention.  Per-stage MP target
        # is set by s1_mp / upscale_to_stage2 / upscale_to_stage3; aspect
        # is shared across stages.
        _, last_h, last_w, _ = reference_images[-1].shape
        last_aspect = last_w / last_h

        def _dims_at_mp_aspect(target_mp: float, aspect: float) -> Tuple[int, int]:
            """Compute 32-aligned (w, h) preserving aspect at target MP."""
            return _calculate_qwen_edit_dimensions(
                int(target_mp * 1_000_000), aspect,
            )

        # ── Stage count + dimensions ─────────────────────────────────
        do_s2 = upscale_to_stage2 > 0
        do_s3 = do_s2 and upscale_to_stage3 > 0
        stage_count = 3 if do_s3 else (2 if do_s2 else 1)

        # Stage 1 dimensions: default from last-reference aspect + s1_mp,
        # or from explicit override_s1_width/height.  When override is
        # active, the S1 aspect ratio is taken from the override so
        # S2/S3 downstream inherit the correct aspect.
        default_s1_w, default_s1_h = _dims_at_mp_aspect(s1_mp, last_aspect)
        s1_w, s1_h = resolve_override_dimensions(
            override_s1_width, override_s1_height,
            default_s1_w, default_s1_h,
            log_prefix="[EricDiffusion-AdvEditMS]",
        )
        s1_mp_act = s1_w * s1_h / 1e6

        # For S2/S3, use S1's actual aspect ratio (may differ from
        # last_aspect if override is active).  This ensures the whole
        # stage chain shares S1's aspect.
        effective_aspect = s1_w / s1_h

        s2_w = s2_h = s2_mp_act = 0
        if do_s2:
            s2_w, s2_h = _dims_at_mp_aspect(
                s1_mp_act * upscale_to_stage2, effective_aspect,
            )
            s2_mp_act = s2_w * s2_h / 1e6

        s3_w = s3_h = 0
        if do_s3:
            s3_w, s3_h = _dims_at_mp_aspect(
                s2_mp_act * upscale_to_stage3, effective_aspect,
            )

        # Cost reporting
        n_vl = sum(vl_flags)
        n_ref = sum(ref_flags)
        cfg_mult = 2 if (
            (s1_cfg > 1.0 or s2_cfg > 1.0 or s3_cfg > 1.0) and negative_prompt
        ) else 1
        total_evals = (
            s1_steps * sampler_cost(s1_sampler) * (2 if s1_cfg > 1.0 and negative_prompt else 1)
            + (s2_steps * sampler_cost(s2_sampler) * (2 if s2_cfg > 1.0 and negative_prompt else 1) if do_s2 else 0)
            + (s3_steps * sampler_cost(s3_sampler) * (2 if s3_cfg > 1.0 and negative_prompt else 1) if do_s3 else 0)
        )

        print(f"\n{'=' * 60}")
        print(
            f"[EricDiffusion-AdvEditMS] Qwen Edit Multi-Stage — "
            f"{stage_count} stage(s), {len(connected)} ref(s) "
            f"(VL={n_vl}, Ref={n_ref}), ~{total_evals} transformer calls, "
            f"seed_mode={seed_mode}"
        )
        print(
            f"  Aspect: {last_w/last_h:.3f} (from last reference image_"
            f"{original_slots[-1]})"
        )
        print(
            f"  S1: {s1_w}×{s1_h} ({s1_mp_act:.2f} MP), {s1_steps} steps, "
            f"sampler={s1_sampler}, cfg={s1_cfg}"
        )
        if do_s2:
            print(
                f"  S2: {s2_w}×{s2_h} ({s2_mp_act:.2f} MP), {s2_steps} steps, "
                f"sampler={s2_sampler}, cfg={s2_cfg}, denoise={s2_denoise}"
            )
        if do_s3:
            print(
                f"  S3: {s3_w}×{s3_h} ({s3_w*s3_h/1e6:.2f} MP), {s3_steps} "
                f"steps, sampler={s3_sampler}, cfg={s3_cfg}, "
                f"denoise={s3_denoise}"
            )
        print(f"{'=' * 60}\n")

        # ── Resolve per-stage seeds based on seed_mode ───────────────
        import random as _random
        if seed_mode == "offset_per_stage":
            s1_seed = seed
            s2_seed = seed + 1 if seed > 0 else 0
            s3_seed = seed + 2 if seed > 0 else 0
        elif seed_mode == "same_all_stages":
            s1_seed = s2_seed = s3_seed = seed
        elif seed_mode == "random_per_stage":
            s1_seed = seed if seed > 0 else _random.randint(1, 0xffffffff)
            s2_seed = _random.randint(1, 0xffffffff)
            s3_seed = _random.randint(1, 0xffffffff)
        else:
            s1_seed = s2_seed = s3_seed = seed

        def _gen(s):
            return torch.Generator(device="cpu").manual_seed(s) if s > 0 else None
        s1_gen = _gen(s1_seed)
        s2_gen = _gen(s2_seed)
        s3_gen = _gen(s3_seed)

        # ── Progress bar across all stages ────────────────────────────
        import comfy.utils
        import comfy.model_management
        total_steps = s1_steps + (s2_steps if do_s2 else 0) + (s3_steps if do_s3 else 0)
        pbar = comfy.utils.ProgressBar(total_steps)
        steps_completed = [0]

        def _stage_progress_cb(step_in_stage: int):
            target_total = steps_completed[0] + step_in_stage
            current = pbar.current if hasattr(pbar, "current") else 0
            delta = target_total - current
            if delta > 0:
                pbar.update(delta)
            comfy.model_management.throw_exception_if_processing_interrupted()

        comfy.model_management.throw_exception_if_processing_interrupted()

        neg = negative_prompt.strip() or None
        offload_vae = pipeline.get("offload_vae", False)
        using_device_map = hasattr(pipe, "hf_device_map")
        if offload_vae and not using_device_map and hasattr(pipe, "vae"):
            pipe.vae = pipe.vae.to(next(pipe.transformer.parameters()).device)

        vae_scale = getattr(pipe, "vae_scale_factor", 8)

        # Capture the transformer's device ONCE at the top of the run
        # and pass it explicitly to every upscale VAE helper call below.
        # Prevents the helpers from defaulting to hardcoded "cuda"
        # (which aliases to cuda:0) and moving pipe.vae onto a
        # different GPU than the rest of the pipeline.  The transformer
        # stays here throughout the stage loop (it only moves to CPU
        # at the very end, right before the final upscale VAE decode,
        # and we've already captured the device by then).
        _transformer_device = next(pipe.transformer.parameters()).device

        # ── Upscale VAE: resolve switch + input into effective bool ──
        # Four validation cases (simpler than Generate's five because
        # this node is Qwen-Edit-only by input type):
        #
        #   1. switch ON  + VAE connected (compatible)   → USE VAE path
        #   2. switch ON  + VAE connected (incompatible) → warn, bislerp fallback
        #   3. switch ON  + no VAE connected             → warn, bislerp fallback
        #   4. switch OFF + VAE connected                → soft note, bislerp
        #   5. switch OFF + no VAE                       → bislerp, silent
        effective_use_vae = False
        if use_upscale_vae and upscale_vae is not None:
            if _vae_supports_upscale(pipe):
                effective_use_vae = True
                # Stage-aware summary so user can confirm what will happen
                print(
                    "[EricDiffusion-AdvEditMS] Upscale VAE: ENABLED "
                    "(Qwen Edit + compatible pipeline VAE)"
                )
                if do_s2:
                    print(
                        f"[EricDiffusion-AdvEditMS]   S1→S2 upscale: "
                        f"bislerp @ upscale_to_stage2={upscale_to_stage2}× "
                        f"area (honored — S1→S2 is always bislerp)"
                    )
                if do_s3:
                    print(
                        f"[EricDiffusion-AdvEditMS]   S2→S3 upscale: "
                        f"Wan2.1 2× VAE (upscale_to_stage3="
                        f"{upscale_to_stage3}× IGNORED — VAE forces "
                        f"fixed 2× linear)"
                    )
                print(
                    "[EricDiffusion-AdvEditMS]   Final decode: Wan2.1 "
                    "2× upscale VAE (2× linear bump on top of last "
                    "stage size)"
                )
            else:
                print(
                    "[EricDiffusion-AdvEditMS] WARNING: upscale_vae "
                    "connected and switch is on, but pipeline VAE is not "
                    "compatible with the Wan2.1 latent space (missing "
                    "z_dim / latents_mean / latents_std).  Falling back "
                    "to bislerp inter-stage upscale and standard final "
                    "decode."
                )
        elif use_upscale_vae and upscale_vae is None:
            print(
                "[EricDiffusion-AdvEditMS] WARNING: use_upscale_vae is "
                "ON but no upscale_vae input is connected.  Falling "
                "back to bislerp inter-stage upscale and standard "
                "final decode."
            )
        elif (not use_upscale_vae) and upscale_vae is not None:
            print(
                "[EricDiffusion-AdvEditMS] Note: upscale_vae is "
                "connected but use_upscale_vae is OFF.  Not using VAE "
                "upscale this run — flip the switch ON to enable it."
            )

        # ── Build common kwargs shared across stages ──────────────────
        # Reference lists are the same at every stage; each stage's
        # generate_qwen_edit call re-encodes refs internally (fast
        # baseline — Phase 2 slice 4 adds the encode-once optimization
        # and the per-stage re-encode switch).
        def _stage_common(width, height, steps, cfg, sampler, schedule, gen_obj, progress_cb, eta):
            return dict(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=neg,
                reference_images=reference_images,
                vl_flags=vl_flags,
                ref_flags=ref_flags,
                output_width=width,
                output_height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                sampler_name=sampler,
                sigma_schedule=schedule,
                generator=gen_obj,
                max_sequence_length=max_sequence_length,
                progress_cb=progress_cb,
                eta=eta,
            )

        try:
            # ── Stage 1: draft from fresh noise ───────────────────────
            comfy.model_management.throw_exception_if_processing_interrupted()
            print(f"[EricDiffusion-AdvEditMS] -- Stage 1/{stage_count} --")
            _apply_lora_stage_weights(
                pipe, pipeline, 1, log_prefix="[EricDiffusion-AdvEditMS]",
            )

            s1_kwargs = _stage_common(
                s1_w, s1_h, s1_steps, s1_cfg,
                s1_sampler, s1_sigma_schedule,
                s1_gen, _stage_progress_cb, s1_eta,
            )
            s1_latents, s1_out_h, s1_out_w = generate_qwen_edit(**s1_kwargs)
            steps_completed[0] += s1_steps

            if not do_s2:
                if effective_use_vae:
                    from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae
                    print("[EricDiffusion-AdvEditMS]   Final decode: Wan2.1 2× upscale VAE")
                    try:
                        pipe.transformer = pipe.transformer.to("cpu")
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    image_tensor = decode_latents_with_upscale_vae(
                        s1_latents, upscale_vae, pipe.vae,
                        s1_out_h, s1_out_w, vae_scale,
                        device=_transformer_device,
                    )
                else:
                    image_tensor = decode_qwen_latents(
                        pipe, s1_latents, s1_out_h, s1_out_w,
                    )
                return (image_tensor,)

            print(
                f"[EricDiffusion-AdvEditMS]   S1 noise latents: "
                f"{tuple(s1_latents.shape)}"
            )

            # ── Stage 2: upscale noise latents + partial re-denoise ───
            comfy.model_management.throw_exception_if_processing_interrupted()
            print(f"[EricDiffusion-AdvEditMS] -- Stage 2/{stage_count} --")
            _apply_lora_stage_weights(
                pipe, pipeline, 2, log_prefix="[EricDiffusion-AdvEditMS]",
            )

            # Bislerp upscale of the S1 noise latents to S2 resolution.
            # References are NOT touched — they remain at their original
            # encoding and are re-concatenated by generate_qwen_edit
            # internally at each stage.
            s2_input_latents = upscale_flux_latents(
                s1_latents, s1_out_h, s1_out_w, s2_h, s2_w,
                vae_scale_factor=vae_scale,
            )
            print(
                f"[EricDiffusion-AdvEditMS]   S2 input latents: "
                f"{tuple(s2_input_latents.shape)}"
            )

            s2_kwargs = _stage_common(
                s2_w, s2_h, s2_steps, s2_cfg,
                s2_sampler, s2_sigma_schedule,
                s2_gen, _stage_progress_cb, s2_eta,
            )
            s2_kwargs["initial_latents"] = s2_input_latents
            s2_kwargs["denoise"] = s2_denoise
            s2_latents, s2_out_h, s2_out_w = generate_qwen_edit(**s2_kwargs)
            steps_completed[0] += s2_steps

            if not do_s3:
                if effective_use_vae:
                    from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae
                    print("[EricDiffusion-AdvEditMS]   Final decode: Wan2.1 2× upscale VAE")
                    try:
                        pipe.transformer = pipe.transformer.to("cpu")
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    image_tensor = decode_latents_with_upscale_vae(
                        s2_latents, upscale_vae, pipe.vae,
                        s2_out_h, s2_out_w, vae_scale,
                        device=_transformer_device,
                    )
                else:
                    image_tensor = decode_qwen_latents(
                        pipe, s2_latents, s2_out_h, s2_out_w,
                    )
                return (image_tensor,)

            print(
                f"[EricDiffusion-AdvEditMS]   S2 noise latents: "
                f"{tuple(s2_latents.shape)}"
            )

            # ── Stage 3: upscale + polish ─────────────────────────────
            comfy.model_management.throw_exception_if_processing_interrupted()
            print(f"[EricDiffusion-AdvEditMS] -- Stage 3/{stage_count} --")
            _apply_lora_stage_weights(
                pipe, pipeline, 3, log_prefix="[EricDiffusion-AdvEditMS]",
            )

            # Inter-stage S2→S3: bislerp OR Wan2.1 2× VAE.  When VAE is
            # on, the output size is forced to 2× the S2 pixel dims by
            # the Wan2.1 decoder, overriding the user's upscale_to_stage3
            # and the s3_w/s3_h we computed earlier.  We use the VAE's
            # actual output size for both generate_qwen_edit and the
            # final decode.
            if effective_use_vae:
                from .eric_qwen_upscale_vae import upscale_between_stages
                print("[EricDiffusion-AdvEditMS]   S2→S3 upscale: Wan2.1 2× VAE")
                if abs(upscale_to_stage3 - 2.0) > 1e-6:
                    print(
                        f"[EricDiffusion-AdvEditMS]   NOTE: "
                        f"upscale_to_stage3={upscale_to_stage3}× was set "
                        f"but IGNORED — upscale VAE forces fixed 2× linear"
                    )
                s3_input_latents, s3_eff_h, s3_eff_w = upscale_between_stages(
                    s2_latents, upscale_vae, pipe.vae,
                    s2_out_h, s2_out_w, vae_scale,
                    device=_transformer_device,
                )
                print(
                    f"[EricDiffusion-AdvEditMS]   S3 target size: "
                    f"{s3_w}×{s3_h} → {s3_eff_w}×{s3_eff_h} "
                    f"(fixed 2× override from upscale VAE)"
                )
            else:
                s3_input_latents = upscale_flux_latents(
                    s2_latents, s2_out_h, s2_out_w, s3_h, s3_w,
                    vae_scale_factor=vae_scale,
                )
                s3_eff_h, s3_eff_w = s3_h, s3_w

            print(
                f"[EricDiffusion-AdvEditMS]   S3 input latents: "
                f"{tuple(s3_input_latents.shape)}"
            )

            s3_kwargs = _stage_common(
                s3_eff_w, s3_eff_h, s3_steps, s3_cfg,
                s3_sampler, s3_sigma_schedule,
                s3_gen, _stage_progress_cb, s3_eta,
            )
            s3_kwargs["initial_latents"] = s3_input_latents
            s3_kwargs["denoise"] = s3_denoise
            s3_latents, s3_out_h, s3_out_w = generate_qwen_edit(**s3_kwargs)
            steps_completed[0] += s3_steps

            if effective_use_vae:
                from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae
                print("[EricDiffusion-AdvEditMS]   Final decode: Wan2.1 2× upscale VAE")
                try:
                    pipe.transformer = pipe.transformer.to("cpu")
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                image_tensor = decode_latents_with_upscale_vae(
                    s3_latents, upscale_vae, pipe.vae,
                    s3_out_h, s3_out_w, vae_scale,
                    device=_transformer_device,
                )
            else:
                image_tensor = decode_qwen_latents(
                    pipe, s3_latents, s3_out_h, s3_out_w,
                )
            return (image_tensor,)

        finally:
            # CRITICAL: restore transformer to its original GPU device.
            # When use_upscale_vae was ON, the final-decode code block
            # moves pipe.transformer to CPU to free VRAM for the
            # high-res Wan2.1 decode.  If we don't restore it here,
            # the NEXT run starts with transformer on CPU and silently
            # runs Qwen Edit inference on CPU (GPU idle, CPU pegged,
            # ~100× slower).  Guard against any exception during the
            # move-back so it doesn't mask an earlier error.
            try:
                current_device = next(pipe.transformer.parameters()).device
                if current_device != _transformer_device:
                    pipe.transformer = pipe.transformer.to(_transformer_device)
                    print(
                        f"[EricDiffusion-AdvEditMS] Transformer restored to "
                        f"{_transformer_device} (was {current_device} after "
                        f"upscale VAE decode)"
                    )
            except Exception as e:
                print(
                    f"[EricDiffusion-AdvEditMS] WARNING: failed to restore "
                    f"transformer to original device: {e}.  Next run may "
                    f"be very slow if it picks up the CPU placement — "
                    f"restart ComfyUI if generation is stuck."
                )

            if offload_vae and not using_device_map and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()
