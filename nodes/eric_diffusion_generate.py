# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Generic Generate Node
Text-to-image generation for any pipeline loaded by EricDiffusionLoader.

Routing per model family:
  qwen-image  → true_cfg_scale + negative_prompt (no guidance embedding)
  flux / flux2 → guidance_scale (guidance-distilled; negative prompts ignored)
  unknown      → signature-inspection fallback (passes only accepted params)

Author: Eric Hiss (GitHub: EricRollei)
"""

import inspect
import math
import torch
import numpy as np
from typing import Tuple

from .eric_qwen_edit_utils import pil_to_tensor
from .eric_diffusion_samplers import sampler_choices, swap_sampler


# ── Resolution helpers ───────────────────────────────────────────────────────

DIMENSION_ALIGNMENT = 16

ASPECT_RATIOS = {
    "16:9  Wide Landscape":  (16, 9),
    "7:5   Landscape":       (7,  5),
    "3:2   Landscape":       (3,  2),
    "4:3   Landscape":       (4,  3),
    "5:4   Landscape":       (5,  4),
    "1:1   Square":          (1,  1),
    "4:5   Portrait":        (4,  5),
    "3:4   Portrait":        (3,  4),
    "2:3   Portrait":        (2,  3),
    "5:7   Portrait":        (5,  7),
    "9:16  Tall Portrait":   (9, 16),
}


def _align(val: int, alignment: int = DIMENSION_ALIGNMENT) -> int:
    return max(alignment, (val // alignment) * alignment)


def compute_dimensions(w_ratio: int, h_ratio: int, target_mp: float) -> Tuple[int, int]:
    target_pixels = target_mp * 1_000_000
    h = math.sqrt(target_pixels * h_ratio / w_ratio)
    w = h * w_ratio / h_ratio
    return _align(int(round(w))), _align(int(round(h)))


def resolve_override_dimensions(
    override_width: int,
    override_height: int,
    default_width: int,
    default_height: int,
    log_prefix: str = "[EricDiffusion]",
) -> Tuple[int, int]:
    """Resolve explicit pixel overrides against default-derived dimensions.

    Used by the Advanced nodes to let users specify exact output pixel
    dimensions (e.g. 1920×1088 for video frames) as an override of the
    default aspect_ratio + target_mp or last-reference-aspect behavior.

    Behavior:
      - Both override values are 0 (default): return the default dims
        unchanged (bit-for-bit backward compatible — no user-visible
        change in workflows that don't touch the override inputs).
      - Both override values are non-zero: floor to 16-alignment (the
        minimum the transformer requires for packing math to work),
        log that the override is active AND what's being ignored, and
        return the aligned override dimensions.
      - Exactly one is non-zero: raise ValueError — ambiguous input.

    Args:
        override_width, override_height : User-provided explicit pixel
                                          dimensions.  0 = use defaults.
        default_width, default_height   : Dimensions derived from the
                                          node's default path (aspect
                                          ratio + target_mp, or last
                                          reference's aspect).
        log_prefix                      : Node-specific log prefix.

    Returns:
        (width, height) — either the aligned override or the defaults.

    Note on common video resolutions:
        1920×1088, 1280×720, 640×480 align cleanly.
        1920×1080 does NOT — 1080 floors to 1072 (an 8px loss).  Use
        1088 instead to avoid the surprise.  The override logic will
        log the 1080→1072 adjustment if you insist on 1080.
    """
    has_width = override_width > 0
    has_height = override_height > 0

    if not has_width and not has_height:
        # Default path — no override requested.
        return default_width, default_height

    if has_width != has_height:
        raise ValueError(
            f"Both override_width and override_height must be non-zero "
            f"to use explicit pixel dimensions, got "
            f"override_width={override_width}, override_height={override_height}.  "
            f"Set both to non-zero values for explicit dimensions, or both "
            f"to 0 to use the default aspect/MP path."
        )

    # Reject absurdly small overrides — diffusion models don't produce
    # meaningful output below ~256px per side, and the slider step of
    # 16 means a single accidental click puts the user at 16×16 which
    # silently runs to completion with garbage output.  256 is the
    # smallest "sensible" diffusion image size; if the user genuinely
    # wants something smaller, they can disable the override (set to 0)
    # and use the aspect+MP path which has its own minimum guards.
    _MIN_USEFUL_DIM = 256
    if override_width < _MIN_USEFUL_DIM or override_height < _MIN_USEFUL_DIM:
        raise ValueError(
            f"Override dimensions {override_width}×{override_height} are "
            f"too small (minimum {_MIN_USEFUL_DIM} per side).  Diffusion "
            f"models don't produce meaningful output below ~256px.  "
            f"Either:\n"
            f"  • Set override_width and override_height to 0 to use the "
            f"default aspect_ratio + target_mp path, OR\n"
            f"  • Set override values to at least {_MIN_USEFUL_DIM}×"
            f"{_MIN_USEFUL_DIM}.\n"
            f"Common video sizes: 1280×720, 1920×1088 (NOT 1080), "
            f"3840×2160."
        )

    # Floor to 16-alignment (minimum required by vae_scale_factor * 2
    # packing for all supported families).
    aligned_w = _align(int(override_width))
    aligned_h = _align(int(override_height))

    if aligned_w != override_width or aligned_h != override_height:
        print(
            f"{log_prefix} Override dimensions adjusted to 16-aligned: "
            f"{override_width}×{override_height} → {aligned_w}×{aligned_h} "
            f"(both dimensions must be multiples of 16 — 1920×1080 is "
            f"a common gotcha, use 1920×1088 or 1920×1072 instead)"
        )

    print(
        f"{log_prefix} Using explicit pixel dimensions: "
        f"{aligned_w}×{aligned_h} "
        f"(default path would have produced {default_width}×{default_height} "
        f"— default aspect/MP inputs IGNORED)"
    )

    return aligned_w, aligned_h


# ── Inference dispatch ───────────────────────────────────────────────────────

def _build_call_kwargs(
    pipe,
    model_family: str,
    guidance_embeds: bool,
    prompt: str,
    negative_prompt: str | None,
    height: int,
    width: int,
    steps: int,
    cfg_scale: float,
    max_sequence_length: int,
    generator,
    on_step_end,
) -> dict:
    """
    Build the kwargs dict for pipe().  Each known family gets an explicit,
    well-documented mapping.  Unknown families fall back to signature
    introspection so new models work without code changes.
    """
    base = {
        "prompt":                  prompt,
        "height":                  height,
        "width":                   width,
        "num_inference_steps":     steps,
        "generator":               generator,
        "callback_on_step_end":    on_step_end,
    }

    if model_family == "qwen-image":
        # Qwen-Image-2512: true CFG (2× forward passes), no guidance embedding.
        # guidance_scale is non-functional on this model (see DEV_NOTES.md).
        kwargs = {
            **base,
            "true_cfg_scale": cfg_scale,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        return kwargs

    if model_family in ("flux", "flux2"):
        # Flux: guidance-distilled — one forward pass, no negative prompts.
        # guidance_scale typical range 3.5–7.0.
        kwargs = {
            **base,
            "guidance_scale": cfg_scale,
        }
        # T5 encoder supports up to 512 tokens; expose for long prompts.
        sig = inspect.signature(pipe.__call__)
        if "max_sequence_length" in sig.parameters:
            kwargs["max_sequence_length"] = max_sequence_length
        return kwargs

    # ── Unknown family: introspect __call__ signature and pass what fits ──
    print(
        f"[EricDiffusion] Unknown model_family={model_family!r} — "
        "introspecting pipeline signature for parameter routing."
    )
    sig = inspect.signature(pipe.__call__)
    accepted = set(sig.parameters.keys())

    candidates = {
        **base,
        "negative_prompt":      negative_prompt or None,
        "guidance_scale":       cfg_scale if guidance_embeds else None,
        "true_cfg_scale":       cfg_scale if not guidance_embeds else None,
        "max_sequence_length":  max_sequence_length,
    }
    return {k: v for k, v in candidates.items() if k in accepted and v is not None}


# ═══════════════════════════════════════════════════════════════════════════
#  Generate Node
# ═══════════════════════════════════════════════════════════════════════════

class EricDiffusionGenerate:
    """
    Generic text-to-image generation node.

    Works with any pipeline loaded by EricDiffusionLoader.  CFG behaviour
    adapts to the loaded model:

    • Qwen-Image  — cfg_scale → true_cfg_scale (two transformer passes).
                    negative_prompt is used when provided.
    • Flux / Flux2 — cfg_scale → guidance_scale (guidance embedding, single
                    pass).  negative_prompt is silently ignored.
    • Unknown      — parameters are matched against the pipeline's __call__
                    signature automatically.
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
                    "tooltip": "From Eric Diffusion Loader.",
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
                        "What to avoid. Used by Qwen-Image (true CFG). "
                        "Ignored by Flux / guidance-distilled models."
                    ),
                }),
                "aspect_ratio": (ratio_names, {
                    "default": "1:1   Square",
                }),
                "target_mp": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.25,
                    "max": 64.0,
                    "step": 0.25,
                    "tooltip": "Target megapixels. Width/height computed from ratio + MP, aligned to 16 px.",
                }),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Denoising steps. Flux typical: 20–30. Qwen typical: 40–50.",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Guidance scale.\n"
                        "• Flux / guidance-distilled: guidance_scale (3.5–7.0 typical).\n"
                        "• Qwen-Image: true_cfg_scale (4.0 typical, runs 2× transformer passes)."
                    ),
                }),
                "max_sequence_length": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 512,
                    "step": 64,
                    "tooltip": (
                        "Max T5 token length for Flux models. "
                        "Ignored by models without a T5 encoder."
                    ),
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = random).",
                }),
                "sampler": (sampler_choices(), {
                    "default": "default",
                    "tooltip": (
                        "Sampler for the denoising loop.\n"
                        "• default — pipeline's built-in Euler (flow-matching).\n"
                        "• multistep2 — 2nd-order Adams-Bashforth (buffers previous "
                        "velocity). Sharper detail at equal steps, same speed.\n"
                        "• multistep3 — 3rd-order, buffers two previous velocities.\n"
                        "All samplers work with Flux, Flux2, Chroma, and Qwen-Image."
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
        target_mp: float = 4.0,
        steps: int = 28,
        cfg_scale: float = 3.5,
        max_sequence_length: int = 512,
        seed: int = 0,
        sampler: str = "default",
    ) -> Tuple[torch.Tensor]:
        pipe          = pipeline["pipeline"]
        model_family  = pipeline.get("model_family", "unknown")
        guidance_embeds = pipeline.get("guidance_embeds", False)
        offload_vae   = pipeline.get("offload_vae", False)

        # ── Dimensions ─────────────────────────────────────────────────────
        w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_ratio, (1, 1))
        width, height = compute_dimensions(w_ratio, h_ratio, target_mp)
        print(
            f"[EricDiffusion] {model_family} — {width}×{height} "
            f"({width * height / 1e6:.2f} MP), steps={steps}, cfg={cfg_scale}, "
            f"sampler={sampler}"
        )

        # ── Generator ──────────────────────────────────────────────────────
        exec_device = getattr(pipe, "_execution_device", None) or "cuda"
        generator = torch.Generator(device=exec_device).manual_seed(seed) if seed > 0 else None

        neg = negative_prompt.strip() or None

        # ── Progress bar + cancel ──────────────────────────────────────────
        import comfy.utils
        import comfy.model_management
        pbar = comfy.utils.ProgressBar(steps)

        def on_step_end(_pipe, step_idx, _timestep, cb_kwargs):
            pbar.update(1)
            comfy.model_management.throw_exception_if_processing_interrupted()
            return cb_kwargs

        comfy.model_management.throw_exception_if_processing_interrupted()

        # ── VAE: move back to GPU if offloaded ─────────────────────────────
        # Skip when device_map is in use — accelerate manages device placement.
        using_device_map = hasattr(pipe, "hf_device_map")
        if offload_vae and not using_device_map and hasattr(pipe, "vae"):
            vae_target = next(pipe.transformer.parameters()).device
            pipe.vae = pipe.vae.to(vae_target)

        # ── Build call kwargs for this model family ────────────────────────
        call_kwargs = _build_call_kwargs(
            pipe, model_family, guidance_embeds,
            prompt, neg, height, width, steps, cfg_scale,
            max_sequence_length, generator, on_step_end,
        )

        # ── Inference (with optional custom sampler) ───────────────────────
        try:
            with swap_sampler(pipe, sampler):
                result = pipe(**call_kwargs)
        finally:
            if offload_vae and not using_device_map and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        # ── Convert to ComfyUI tensor [B, H, W, C] ─────────────────────────
        pil_image = result.images[0]
        tensor = pil_to_tensor(pil_image).unsqueeze(0)
        return (tensor,)
