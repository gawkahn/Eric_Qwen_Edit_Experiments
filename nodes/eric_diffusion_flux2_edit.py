# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Edit — image-conditioned generation for Flux.2-Klein and Flux.2-dev.

Both models take one or more reference images and a text prompt:
  - Flux.2-Klein (flux2klein): dedicated edit/reference model, Qwen3 text encoder,
    guidance-distilled. Prompt describes the desired change or composite.
  - Flux.2-dev (flux2): base generation model with image conditioning,
    Mistral3 text encoder.

Up to 4 reference images accepted. Sparse slot connections are auto-compacted
with a warning (image_1 + image_3 → Picture 1 + Picture 2 in the model's
reference ordering).

Author: Eric Hiss (GitHub: EricRollei)
"""

from datetime import datetime
from typing import Tuple

import torch
from PIL import Image

from .eric_diffusion_utils import build_model_metadata
from .eric_qwen_edit_utils import pil_to_tensor


_SUPPORTED_FAMILIES = ("flux2klein", "flux2")


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] float32 0-1 → PIL Image."""
    return Image.fromarray(tensor[0].mul(255).byte().cpu().numpy())


class EricDiffusionEdit:
    """
    Image-conditioned generation for Flux.2-Klein and Flux.2-dev.

    Load either model with the standard Eric Diffusion Loader or Component
    Loader — both are auto-detected as 'flux2klein' or 'flux2' family from
    their model_index.json.

    Neither model supports classical CFG. guidance_scale is the distillation-
    based conditioning strength (single forward pass per step).

    Multi-reference: connect image_1–image_4 in any combination. Sparse
    connections are compacted and a warning names the remapping. The model
    receives a list of PIL images in slot order; refer to them positionally
    in your prompt ("the car from the first image", etc.).
    """

    CATEGORY = "Eric Diffusion"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "GEN_METADATA")
    RETURN_NAMES = ("image", "metadata")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GEN_PIPELINE", {
                    "tooltip": (
                        "From Eric Diffusion Loader. "
                        "Accepts Flux.2-Klein (flux2klein) and Flux.2-dev (flux2). "
                        "Other families raise an error."
                    ),
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": (
                        "Edit instruction or composite description. "
                        "Reference connected images positionally: "
                        "'the car from the first image in the scene from the second image'."
                    ),
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "First reference image."}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Denoising steps. Flux.2-Klein/dev default: 28.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Distillation guidance scale. "
                        "Typical range 3.5–5.0. Higher = stronger prompt adherence."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = random).",
                }),
                "max_sequence_length": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Max prompt token length.",
                }),
            },
        }

    def generate(
        self,
        pipeline: dict,
        prompt: str,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        steps: int = 28,
        guidance_scale: float = 4.0,
        seed: int = 0,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]
        model_family = pipeline.get("model_family", "unknown")
        offload_vae = pipeline.get("offload_vae", False)

        if model_family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                f"EricDiffusionEdit requires a Flux.2-Klein or Flux.2-dev pipeline "
                f"(family 'flux2klein' or 'flux2'), got {model_family!r}. "
                f"For Qwen-Image-Edit use EricDiffusionAdvancedEdit."
            )

        # ── Collect connected slots and auto-compact ──────────────────────────
        slot_configs = [
            (1, image_1),
            (2, image_2),
            (3, image_3),
            (4, image_4),
        ]
        connected = [(idx, img) for idx, img in slot_configs if img is not None]

        if not connected:
            raise ValueError(
                "EricDiffusionEdit requires at least one reference image connected "
                "(image_1 through image_4). For text-to-image without a reference "
                "use EricDiffusionGenerate instead."
            )

        original_slots = [c[0] for c in connected]
        remapped_slots = list(range(1, len(connected) + 1))
        if original_slots != remapped_slots:
            print(
                f"[EricDiffusion-Edit] WARNING: sparse slot configuration — "
                f"slots {original_slots} remapped to reference positions "
                f"{remapped_slots}. Adjust your prompt if you reference images "
                f"by position."
            )
            for orig, new in zip(original_slots, remapped_slots):
                if orig != new:
                    print(f"[EricDiffusion-Edit]   image_{orig} → reference {new}")

        pil_images = [_tensor_to_pil(img) for _, img in connected]
        input_dims = [f"{img.shape[2]}x{img.shape[1]}" for _, img in connected]

        in_w, in_h = pil_images[0].size
        print(
            f"[EricDiffusion-Edit] {model_family} — "
            f"{len(pil_images)} reference(s), lead {in_w}×{in_h}, "
            f"steps={steps}, guidance={guidance_scale}, seed={seed}"
        )

        # ── Generator ─────────────────────────────────────────────────────────
        exec_device = getattr(pipe, "_execution_device", None) or "cuda"
        generator = torch.Generator(device=exec_device).manual_seed(seed) if seed > 0 else None

        # ── Progress bar + cancel ─────────────────────────────────────────────
        import comfy.utils
        import comfy.model_management
        pbar = comfy.utils.ProgressBar(steps)

        def on_step_end(_pipe, step_idx, _timestep, cb_kwargs):
            pbar.update(1)
            comfy.model_management.throw_exception_if_processing_interrupted()
            return cb_kwargs

        comfy.model_management.throw_exception_if_processing_interrupted()

        # ── VAE: restore from offload ─────────────────────────────────────────
        using_device_map = hasattr(pipe, "hf_device_map")
        if offload_vae and not using_device_map and hasattr(pipe, "vae"):
            vae_target = next(pipe.transformer.parameters()).device
            pipe.vae = pipe.vae.to(vae_target)

        # Pass a single PIL when only one image connected — some pipeline
        # versions don't accept a length-1 list; pass list for 2+.
        image_arg = pil_images[0] if len(pil_images) == 1 else pil_images

        call_kwargs = {
            "image":                image_arg,
            "prompt":               prompt,
            "num_inference_steps":  steps,
            "guidance_scale":       guidance_scale,
            "generator":            generator,
            "max_sequence_length":  max_sequence_length,
            "callback_on_step_end": on_step_end,
        }

        try:
            result = pipe(**call_kwargs)
        finally:
            if offload_vae and not using_device_map and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        pil_out = result.images[0]
        out_w, out_h = pil_out.size
        tensor_out = pil_to_tensor(pil_out).unsqueeze(0)

        metadata = {
            **build_model_metadata(pipeline),
            "node_type":              "edit",
            "seed":                   seed,
            "steps":                  steps,
            "cfg_scale":              guidance_scale,
            "sampler":                "euler",
            "sampler_s2":             "",
            "sampler_s3":             "",
            "prompt":                 prompt,
            "negative_prompt":        "",
            "input_image_dimensions": input_dims,
            "width":                  out_w,
            "height":                 out_h,
            "timestamp":              datetime.now().isoformat(),
        }

        return (tensor_out, metadata)
