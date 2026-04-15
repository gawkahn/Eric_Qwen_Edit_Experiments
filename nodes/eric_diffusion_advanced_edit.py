# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Advanced Edit — Qwen-Image-Edit with per-stage samplers

Manual-loop single-stage edit node for Qwen-Image-Edit-2511.  Mirrors
the Advanced Generate UX but accepts 1-4 reference images with per-image
flags for VL (semantic) and Ref (pixel latent) conditioning paths.

Relationship to other edit nodes
--------------------------------
- **Eric Qwen-Edit Image** / **Eric Qwen-Edit Multi-Image**: stock
  ``pipe()``-based edit nodes.  These stay fully supported and remain
  the safe fallback if the manual loop ever regresses.
- **Eric Diffusion Advanced Edit** (this file): bypasses ``pipe()`` and
  runs the denoising loop manually, giving access to higher-order
  samplers (flow_heun, flow_rk3, multistep variants), eta stochastic
  sampling, and beta sigma schedules.

Currently Qwen-Image-Edit only.  Flux edit support (Flux.1-Kontext,
eventual Flux.2-Edit) will plug in later via a new EDIT_PIPELINE type.

Multi-reference convention
--------------------------
Each slot has two independent flags:

- **VL flag** — when True, the image is fed to the Qwen2.5-VL text
  encoder as part of the prompt fusion.  This is what enables
  compositional prompts like "the outfit from Picture 2".
- **Ref flag** — when True, the image is VAE-encoded and its packed
  latents concatenate into the transformer's hidden_states so the
  model can attend to reference pixels directly.

Both default to True.  Turn off VL to skip the semantic path (image
contributes pixels only, no prompt-level reference).  Turn off Ref
to skip the pixel path (image contributes prompt-level context only,
no latent anchor).  Turning both off on a slot is rejected.

Slot → Picture N mapping
------------------------
The Nth connected image becomes "Picture N" in Qwen2.5-VL's prompt
template.  When you write "Picture 3 wearing the outfit from Picture 2"
in your prompt, Qwen's VL encoder connects those token references to
the images at those list positions.

**If you leave gaps between connected slots** (e.g. image_1 + image_3
but not image_2), the node auto-compacts to contiguous positions
AND prints a warning telling you exactly which slot becomes which
Picture number.  Either fix your slot connections or adjust your
prompt to use the remapped numbers.

Author: Eric Hiss (GitHub: EricRollei)
"""

from typing import Tuple

import torch

from .eric_diffusion_advanced_multistage import _vae_supports_upscale
from .eric_diffusion_generate import resolve_override_dimensions
from .eric_diffusion_manual_loop import (
    sampler_names,
    sampler_cost,
    SIGMA_SCHEDULE_NAMES,
    generate_qwen_edit,
    decode_qwen_latents,
)


class EricDiffusionAdvancedEdit:
    """
    Advanced single-stage Qwen-Image-Edit with 1-4 reference images.

    Uses a manual denoising loop so higher-order samplers (flow_heun,
    flow_rk3, multistep variants) and stochastic eta sampling apply
    uniformly to edit workflows the same way they do to generate.

    Output dimensions are derived from the LAST connected reference
    image's aspect ratio at ~1MP, 32-aligned.  This matches stock
    Qwen-Image-Edit behavior.  Explicit width/height override is a
    backlog item (tracked alongside the granular-resolution request
    for Advanced Generate).
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
                        "This node only supports Qwen-Image-Edit in Phase 1. "
                        "Flux.1-Kontext / Flux.2-Edit are backlogged until "
                        "stock diffusers ships the relevant pipelines."
                    ),
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": (
                        "Edit instruction.  Reference images are automatically "
                        "labeled 'Picture 1', 'Picture 2', etc. based on the "
                        "order they're connected (image_1 = Picture 1, etc.). "
                        "Use those labels in your prompt for compositional "
                        "directives, e.g. 'the background from Picture 1 with "
                        "the subject from Picture 2 wearing the outfit from "
                        "Picture 3'."
                    ),
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional negative prompt.  Doubles model cost per "
                        "step because each step needs a second transformer "
                        "forward pass for the negative branch.  Qwen Edit "
                        "uses norm-preserving classical CFG — typical "
                        "cfg_scale 3.5–4.5."
                    ),
                }),

                # ── Reference image slots ──────────────────────────────
                "image_1": ("IMAGE", {
                    "tooltip": (
                        "Reference image for slot 1.  Becomes 'Picture 1' "
                        "in the VL prompt template."
                    ),
                }),
                "image_1_vl": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Feed image_1 to the Qwen2.5-VL text encoder for "
                        "semantic context.  Enable for compositional prompt "
                        "references.  Disable if you want image_1 as a pixel "
                        "anchor only (its latents concatenate into the "
                        "transformer but the VL encoder doesn't see it)."
                    ),
                }),
                "image_1_ref": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Use image_1 as a pixel latent anchor — its VAE-"
                        "encoded latents concatenate into the transformer's "
                        "hidden_states so the model attends to its pixels "
                        "directly.  Disable if you want image_1 as semantic "
                        "context only (VL processor sees it but transformer "
                        "doesn't get pixel latents)."
                    ),
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

                # ── Sampler controls ───────────────────────────────────
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": (
                        "Denoising steps.  Qwen Edit typically converges at "
                        "20–30 steps with flow_heun or multistep2.  Edit "
                        "tasks need fewer steps than text-to-image because "
                        "the reference anchors constrain the solution."
                    ),
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Classical CFG scale (norm-preserving rescale).  "
                        "Typical 3.5–4.5 for Qwen Edit.  Values >1.0 require "
                        "a negative_prompt to have effect and double the "
                        "per-step cost."
                    ),
                }),
                "sampler": (sampler_names(), {
                    "default": "flow_heun",
                    "tooltip": (
                        "Denoising sampler.  All are flow-matching:\n"
                        "• flow_euler     — 1st order baseline\n"
                        "• flow_heun      — 2nd order (2× cost/step)\n"
                        "• flow_rk3       — 3rd order (3× cost, ≥15 steps)\n"
                        "• flow_multistep2 — 2nd order same cost as Euler\n"
                        "• flow_multistep3 — 3rd order same cost as Euler\n"
                        "Default flow_heun is a good all-purpose edit sampler."
                    ),
                }),
                "sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "linear",
                    "tooltip": (
                        "Sigma schedule curve.  Qwen Edit works well on "
                        "linear for most edits.  Try balanced or karras if "
                        "you want more time on low-sigma detail refinement."
                    ),
                }),
                "eta": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Stochastic sampling coefficient.  0.0 = "
                        "deterministic.  Moderate values (0.3–0.5) can "
                        "improve detail but are less predictable on edit "
                        "tasks where you usually want the reference to "
                        "determine structure — start at 0.0 and only "
                        "experiment if results look over-smoothed."
                    ),
                }),
                "max_sequence_length": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": (
                        "Max prompt token length.  Qwen Edit defaults to "
                        "1024 (higher than base Qwen's 512) because edit "
                        "prompts are typically longer and include references "
                        "to multiple images."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = non-deterministic).",
                }),

                # ── Explicit output dimensions (override) ─────────────
                "override_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 16,
                    "tooltip": (
                        "Explicit output width in pixels.  When this "
                        "AND override_height are both non-zero, they "
                        "override the default 'derive from last "
                        "connected reference' behavior — the node "
                        "generates at exactly these dimensions "
                        "regardless of the reference aspect ratios.\n\n"
                        "Must be a multiple of 16.  Non-multiples are "
                        "floored with a log note.\n\n"
                        "Useful for video keyframe work where you "
                        "need exact pixel dimensions (1920×1088, "
                        "1280×720, etc.).  Note: 1080 is not a "
                        "multiple of 16 — use 1088 (16-aligned) or "
                        "accept the floor to 1072.\n\n"
                        "Leave at 0 to use the default last-reference-"
                        "aspect behavior."
                    ),
                }),
                "override_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "step": 16,
                    "tooltip": (
                        "Explicit output height in pixels.  Must be "
                        "set non-zero together with override_width to "
                        "activate the explicit-dimension path."
                    ),
                }),

                # ── Upscale VAE (final decode only) ───────────────────
                "use_upscale_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When ON with an upscale_vae connected, the "
                        "final VAE decode uses the Wan2.1 2× upscale "
                        "VAE instead of the standard pipeline VAE.  "
                        "Output resolution is 2× each dimension = 4× "
                        "the area of the native edit resolution (~1 MP "
                        "natively → ~4 MP final).\n\n"
                        "Unlike the multistage node, this does NOT do "
                        "inter-stage upscaling — there's only one edit "
                        "stage, so the upscale VAE just bumps the "
                        "final pixel resolution without running the "
                        "transformer at higher MP.\n\n"
                        "This is the identity-preserving high-res path: "
                        "Qwen Edit runs at its native training area "
                        "where compositional fidelity and feature "
                        "conditioning are strongest, and the VAE adds "
                        "the 2× bump entirely in pixel space.  Good "
                        "for video keyframe work, character-focused "
                        "edits, and any workflow where you care more "
                        "about getting the subject right than about "
                        "squeezing maximum MP from the output.\n\n"
                        "Soft-fails if the VAE isn't connected or the "
                        "pipeline VAE isn't Wan2.1-compatible — prints "
                        "a warning and falls back to the standard "
                        "decode, no crash."
                    ),
                }),
                "upscale_vae": ("UPSCALE_VAE", {
                    "tooltip": (
                        "Optional. Output of 'Eric Qwen Upscale VAE "
                        "Loader'. Only honored when 'use_upscale_vae' "
                        "is ON.  Connecting this without flipping the "
                        "switch prints a soft note and does nothing."
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
        steps: int = 25,
        cfg_scale: float = 4.0,
        sampler: str = "flow_heun",
        sigma_schedule: str = "linear",
        eta: float = 0.0,
        max_sequence_length: int = 1024,
        seed: int = 0,
        override_width: int = 0,
        override_height: int = 0,
        use_upscale_vae: bool = False,
        upscale_vae=None,
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]

        # ── Collect connected slots and auto-compact ──────────────────
        slot_configs = [
            (1, image_1, image_1_vl, image_1_ref),
            (2, image_2, image_2_vl, image_2_ref),
            (3, image_3, image_3_vl, image_3_ref),
            (4, image_4, image_4_vl, image_4_ref),
        ]
        connected = [(idx, img, vl, ref) for idx, img, vl, ref in slot_configs if img is not None]

        if not connected:
            raise ValueError(
                "Eric Diffusion Advanced Edit requires at least one reference "
                "image connected (image_1 through image_4).  For pure text-to-"
                "image without any reference, use Eric Diffusion Advanced "
                "Generate instead."
            )

        # Detect sparse slots and warn about remapping — the VL processor
        # assigns Picture N based on list position, so if the user
        # connected slot 1 + slot 3 without slot 2, we need to tell them
        # that slot 3 becomes "Picture 2" in the VL template.
        original_slots = [c[0] for c in connected]
        remapped_slots = list(range(1, len(connected) + 1))
        if original_slots != remapped_slots:
            print(
                f"[EricDiffusion-AdvEdit] WARNING: sparse slot "
                f"configuration — slots {original_slots} will be remapped "
                f"to Picture numbers {remapped_slots} for the VL encoder.  "
                f"If your prompt references 'Picture N', verify N matches "
                f"the remapped index, NOT the original slot number.  Either "
                f"fix your slot connections to be contiguous starting from "
                f"image_1, or adjust your prompt accordingly."
            )
            for orig, new in zip(original_slots, remapped_slots):
                if orig != new:
                    print(
                        f"[EricDiffusion-AdvEdit]   image_{orig} → Picture {new}"
                    )

        reference_images = [c[1] for c in connected]
        vl_flags = [c[2] for c in connected]
        ref_flags = [c[3] for c in connected]

        # Soft validation: warn if any slot has both flags False
        for i, (vl, ref) in enumerate(zip(vl_flags, ref_flags)):
            if not vl and not ref:
                print(
                    f"[EricDiffusion-AdvEdit] WARNING: Picture {i+1} has "
                    f"both VL and Ref flags disabled — the model won't see "
                    f"this image at all.  Either enable at least one flag "
                    f"or disconnect the slot."
                )

        # Cost reporting
        cost_mult = sampler_cost(sampler)
        cfg_mult = 2 if (cfg_scale > 1.0 and negative_prompt) else 1
        total_evals = steps * cost_mult * cfg_mult
        n_vl = sum(vl_flags)
        n_ref = sum(ref_flags)
        print(
            f"[EricDiffusion-AdvEdit] Qwen Edit — {len(connected)} ref(s) "
            f"(VL={n_vl}, Ref={n_ref}), steps={steps}, "
            f"sampler={sampler} (×{cost_mult}/step), cfg={cfg_scale}, "
            f"~{total_evals} transformer calls total"
        )

        # ── Generator (CPU-based for shape-artifact avoidance) ────────
        generator = (
            torch.Generator(device="cpu").manual_seed(seed) if seed > 0 else None
        )

        # ── Progress bar + cancel ──────────────────────────────────────
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

        # ── VAE: move back to GPU if offloaded ────────────────────────
        offload_vae = pipeline.get("offload_vae", False)
        using_device_map = hasattr(pipe, "hf_device_map")
        if offload_vae and not using_device_map and hasattr(pipe, "vae"):
            vae_target = next(pipe.transformer.parameters()).device
            pipe.vae = pipe.vae.to(vae_target)

        vae_scale = getattr(pipe, "vae_scale_factor", 8)

        # Capture the transformer's device ONCE before any run.  When
        # use_upscale_vae is ON, the decode path moves pipe.transformer
        # to CPU to free VRAM for the Wan2.1 decode, and the finally
        # block below restores it so the NEXT run doesn't silently run
        # inference on CPU.
        _transformer_device = next(pipe.transformer.parameters()).device

        # ── Upscale VAE: resolve switch + input into effective bool ──
        # Four validation cases (simpler than the Generate side because
        # this node is Qwen-Edit-only by input type):
        #
        #   1. switch ON  + VAE connected (compatible)   → USE VAE path
        #   2. switch ON  + VAE connected (incompatible) → warn, standard decode
        #   3. switch ON  + no VAE connected             → warn, standard decode
        #   4. switch OFF + VAE connected                → soft note, standard decode
        #   5. switch OFF + no VAE                       → standard decode, silent
        effective_use_vae = False
        if use_upscale_vae and upscale_vae is not None:
            if _vae_supports_upscale(pipe):
                effective_use_vae = True
                print(
                    "[EricDiffusion-AdvEdit] Upscale VAE: ENABLED "
                    "(Qwen Edit + compatible pipeline VAE)"
                )
                print(
                    "[EricDiffusion-AdvEdit]   Final decode: Wan2.1 2× "
                    "upscale VAE (output at 2× linear = 4× area vs "
                    "native edit resolution)"
                )
            else:
                print(
                    "[EricDiffusion-AdvEdit] WARNING: upscale_vae "
                    "connected and switch is on, but pipeline VAE is "
                    "not compatible with the Wan2.1 latent space "
                    "(missing z_dim / latents_mean / latents_std).  "
                    "Falling back to standard decode."
                )
        elif use_upscale_vae and upscale_vae is None:
            print(
                "[EricDiffusion-AdvEdit] WARNING: use_upscale_vae is "
                "ON but no upscale_vae input is connected.  Falling "
                "back to standard decode."
            )
        elif (not use_upscale_vae) and upscale_vae is not None:
            print(
                "[EricDiffusion-AdvEdit] Note: upscale_vae is "
                "connected but use_upscale_vae is OFF.  Not using VAE "
                "upscale this run — flip the switch ON to enable it."
            )

        # ── Resolve explicit output dimensions if overridden ──────────
        # Default: pass None for output_width/output_height so
        # generate_qwen_edit derives them from the last reference's
        # aspect ratio.  Override: floor to 16-alignment, log the
        # substitution, and pass explicit values.
        if override_width > 0 and override_height > 0:
            # We don't have "default dims" here because Advanced Edit
            # derives them from the last reference inside generate_qwen_edit.
            # Use the helper purely for alignment + logging, passing the
            # override values as both the override AND the defaults (the
            # helper will always take the override path since both
            # overrides are non-zero).  The "default path would have
            # produced" log line is misleading in this case but the
            # value still alignment-validates cleanly.
            effective_output_w, effective_output_h = resolve_override_dimensions(
                override_width, override_height,
                override_width, override_height,  # dummy — helper always takes override path
                log_prefix="[EricDiffusion-AdvEdit]",
            )
        elif override_width > 0 or override_height > 0:
            raise ValueError(
                f"Both override_width and override_height must be non-zero "
                f"to use explicit pixel dimensions, got "
                f"override_width={override_width}, override_height={override_height}.  "
                f"Set both to non-zero values for explicit dimensions, or both "
                f"to 0 to derive output size from the last reference image."
            )
        else:
            # Default: let generate_qwen_edit derive from last reference.
            effective_output_w = None
            effective_output_h = None

        # ── Run the manual edit loop ──────────────────────────────────
        try:
            final_noise_latents, out_h, out_w = generate_qwen_edit(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative_prompt.strip() or None,
                reference_images=reference_images,
                vl_flags=vl_flags,
                ref_flags=ref_flags,
                output_width=effective_output_w,
                output_height=effective_output_h,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                sampler_name=sampler,
                sigma_schedule=sigma_schedule,
                generator=generator,
                max_sequence_length=max_sequence_length,
                progress_cb=progress_cb,
                eta=eta,
            )

            # Final decode: standard Qwen decode at native resolution,
            # OR Wan2.1 2× upscale VAE for a 4× area bump in pixel
            # space (transformer never runs at the higher resolution).
            if effective_use_vae:
                from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae
                print("[EricDiffusion-AdvEdit]   Running Wan2.1 upscale VAE decode")
                # Offload transformer to CPU to free VRAM for the
                # high-res decode.  The finally block below restores it.
                try:
                    pipe.transformer = pipe.transformer.to("cpu")
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                image_tensor = decode_latents_with_upscale_vae(
                    final_noise_latents, upscale_vae, pipe.vae,
                    out_h, out_w, vae_scale,
                    device=_transformer_device,
                )
            else:
                image_tensor = decode_qwen_latents(
                    pipe, final_noise_latents, out_h, out_w,
                )
        finally:
            # CRITICAL: restore transformer to its original GPU device.
            # When use_upscale_vae was ON, the decode block moved the
            # transformer to CPU.  If we don't restore it here, the
            # next run starts with the transformer on CPU and silently
            # runs Qwen Edit inference on CPU (GPU idle, CPU pegged,
            # ~100× slower than intended).
            try:
                current_device = next(pipe.transformer.parameters()).device
                if current_device != _transformer_device:
                    pipe.transformer = pipe.transformer.to(_transformer_device)
                    print(
                        f"[EricDiffusion-AdvEdit] Transformer restored to "
                        f"{_transformer_device} (was {current_device} after "
                        f"upscale VAE decode)"
                    )
            except Exception as e:
                print(
                    f"[EricDiffusion-AdvEdit] WARNING: failed to restore "
                    f"transformer to original device: {e}.  Next run may "
                    f"be very slow if it picks up the CPU placement — "
                    f"restart ComfyUI if generation is stuck."
                )

            if offload_vae and not using_device_map and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        return (image_tensor,)
