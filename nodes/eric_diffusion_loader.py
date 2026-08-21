# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Generic Loader
Load any diffusers text-to-image pipeline by pointing at a model directory.
Model type (Qwen-Image, Flux, Flux2, etc.) is auto-detected from model_index.json.

Returns a GEN_PIPELINE dict understood by Eric Diffusion Generate and future
generic nodes.  Existing Qwen-specific nodes are unaffected.

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import torch
from typing import Tuple

from comfyless.core.eric_diffusion_utils import (
    DTYPE_MAP,
    QUANT_MODES,
    build_quant_config,
    detect_pipeline_class,
    detect_load_variant,
    quant_cache_fragment,
    read_guidance_embeds,
    get_gen_pipeline_cache,
    clear_gen_pipeline_cache,
    resolve_hf_path,
    resolve_vae_tiling,
    VAE_TILING_CHOICES,
)
from .eric_diffusion_component_loader import (
    available_device_options,
    resolve_device_with_fallback,
)


class EricDiffusionLoader:
    """
    Generic text-to-image pipeline loader.

    Point model_path at any diffusers model directory.  The pipeline class
    (QwenImagePipeline, FluxPipeline, Flux2Pipeline, …) is read from
    model_index.json so this node works with models that don't exist yet
    without any code changes.

    Returns a GEN_PIPELINE dict containing the pipeline object, the detected
    model_family string, and capability flags used by downstream nodes.
    """

    CATEGORY = "Eric Diffusion"
    FUNCTION = "load_pipeline"
    RETURN_TYPES = ("GEN_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        cache = get_gen_pipeline_cache()
        if cache["pipeline"] is None:
            return float("nan")
        return cache.get("cache_key", "")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to any diffusers model directory. "
                        "Model type is auto-detected from model_index.json."
                    ),
                }),
            },
            "optional": {
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Model precision. bf16 recommended for RTX 40/50 series.",
                }),
                "quant": (list(QUANT_MODES), {
                    "default": "none",
                    "tooltip": (
                        "Quantize-on-load (ADR-019). fp8 halves VRAM on the "
                        "transformer + large text encoders and uses native "
                        "fp8 tensor cores (Ada/Hopper/Blackwell, compute "
                        "capability >= 8.9). VAE and CLIP encoders are never "
                        "quantized. Falls back to bf16 with a console warning "
                        "on unsupported hardware."
                    ),
                }),
                "device": (available_device_options(), {
                    "default": (
                        "balanced" if "balanced" in available_device_options()
                        else ("cuda" if "cuda" in available_device_options() else "cpu")
                    ),
                    "tooltip": (
                        "Device to load model on.  Options filtered by "
                        "visible hardware (run-time torch.cuda.device_count).\n\n"
                        "• balanced — splits across all visible GPUs via "
                        "accelerate device_map='balanced'.  Required for "
                        "large models (Flux.2) that don't fit on one GPU.  "
                        "ONLY shown when 2+ GPUs are visible.\n"
                        "• cuda / cuda:N — pin to a single GPU.\n"
                        "• cpu — CPU-only (very slow, for testing)."
                    ),
                }),
                "keep_in_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache model between runs (faster but uses VRAM).",
                }),
                "offload_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move VAE to CPU during transformer inference (saves ~1 GB).",
                }),
                "attention_slicing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Trade speed for lower peak VRAM.",
                }),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extreme VRAM savings via sequential CPU offload — very slow.",
                }),
                "vae_tiling": (list(VAE_TILING_CHOICES), {
                    "default": "auto",
                    "tooltip": (
                        "VAE tiling at decode time.\n\n"
                        "• auto — family-aware default. Off for Hunyuan-Image "
                        "(32× VAE produces small latents; tiling adds seam "
                        "artifacts without memory benefit). On for every other "
                        "family (preserves the prior always-tiled behavior on "
                        "8×/16× VAEs).\n"
                        "• on — force tiling on regardless of family. Use if "
                        "Hunyuan decode OOMs on a small-VRAM card.\n"
                        "• off — force tiling off regardless of family."
                    ),
                }),
                "allow_hf_download": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Accept HuggingFace repo IDs (e.g. 'black-forest-labs/FLUX.1-dev') "
                        "in addition to local paths. Checks the local HF cache first. "
                        "Enable to download if not cached; disable to fail-closed (no network)."
                    ),
                }),
            },
        }

    def load_pipeline(
        self,
        model_path: str,
        precision: str = "bf16",
        quant: str = "none",
        device: str = "cuda",
        keep_in_vram: bool = True,
        offload_vae: bool = False,
        attention_slicing: bool = False,
        sequential_offload: bool = False,
        vae_tiling: str = "auto",
        allow_hf_download: bool = False,
    ) -> Tuple:
        model_path = resolve_hf_path(model_path.strip(), allow_download=allow_hf_download)
        # quant_cache_fragment is "" for mode 'none' — the default path's
        # cache key stays byte-identical to pre-quant behavior (invariant 1);
        # any quant change evicts and reloads (invariant 4).
        cache_key = (
            f"{model_path}_{precision}_{device}_{offload_vae}"
            f"_{attention_slicing}_{sequential_offload}_{vae_tiling}"
            f"{quant_cache_fragment(quant)}"
        )
        cache = get_gen_pipeline_cache()

        if cache["pipeline"] is not None and cache.get("cache_key") == cache_key:
            print("[EricDiffusion] Using cached pipeline")
            return (cache["pipeline_dict"],)

        if cache["pipeline"] is not None:
            print("[EricDiffusion] Config changed — clearing cached pipeline")
            clear_gen_pipeline_cache()

        dtype = DTYPE_MAP.get(precision, torch.bfloat16)

        # ── Auto-detect pipeline class from model_index.json ─────────────
        pipeline_class, class_name, model_family = detect_pipeline_class(model_path)
        print(f"[EricDiffusion] Detected: {class_name} (family: {model_family})")

        # ── Load ─────────────────────────────────────────────────────────
        # Runtime device fallback handles the stale-workflow-JSON case
        # (e.g. balanced was saved on 2-GPU host, loading on 1-GPU host).
        # The dropdown filter already excludes unavailable options but
        # ComfyUI passes the saved value through so we re-validate here.
        device, use_device_map = resolve_device_with_fallback(
            device, log_prefix="[EricDiffusion]",
        )
        load_kwargs = dict(torch_dtype=dtype, local_files_only=True)
        if use_device_map:
            load_kwargs["device_map"] = "balanced"
        variant = detect_load_variant(model_path)
        if variant:
            load_kwargs["variant"] = variant

        # ── Quantize-on-load (ADR-019 slice A) ────────────────────────────
        # build_quant_config warns and returns None on unsupported hardware
        # or missing torchao — the load proceeds unquantized (never crash).
        quant_components: dict = {}
        if quant != "none":
            if sequential_offload:
                print("[EricDiffusion] WARNING: quant + sequential_offload "
                      "is untested together — proceeding with both")
            if use_device_map:
                print("[EricDiffusion] WARNING: quant + device_map='balanced' "
                      "is untested together — proceeding with both")
            quant_config, quant_components, _ = build_quant_config(
                model_path, quant, device=device,
                log_prefix="[EricDiffusion]",
            )
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config

        pipeline = pipeline_class.from_pretrained(model_path, **load_kwargs)

        # ── Optimizations ─────────────────────────────────────────────────
        if sequential_offload:
            print("[EricDiffusion] Enabling sequential CPU offload")
            pipeline.enable_sequential_cpu_offload()
        elif not use_device_map:
            # device_map already placed everything — don't call .to()
            pipeline = pipeline.to(device)
            if offload_vae and hasattr(pipeline, "vae"):
                print("[EricDiffusion] Moving VAE to CPU")
                pipeline.vae = pipeline.vae.to("cpu")

        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            if resolve_vae_tiling(model_family, vae_tiling):
                pipeline.vae.enable_tiling()
                print(f"[EricDiffusion] VAE tiling enabled (vae_tiling={vae_tiling})")
            else:
                # Defensive disable in case a future diffusers default flips
                # use_tiling=True at construct time.
                if hasattr(pipeline.vae, "disable_tiling"):
                    pipeline.vae.disable_tiling()
                print(f"[EricDiffusion] VAE tiling disabled (vae_tiling={vae_tiling})")

        if attention_slicing:
            try:
                pipeline.enable_attention_slicing(slice_size="auto")
                print("[EricDiffusion] Attention slicing enabled")
            except Exception as e:
                print(f"[EricDiffusion] Attention slicing not available: {e}")

        # ── Capability flags (used by downstream nodes) ───────────────────
        guidance_embeds = read_guidance_embeds(pipeline)

        pipeline_dict = {
            "pipeline":        pipeline,
            "model_path":      model_path,
            "model_family":    model_family,
            "offload_vae":     offload_vae,
            "guidance_embeds": guidance_embeds,
            # Effective quant state: mode 'none' when disabled OR when the
            # hardware/env fallback fired. Downstream (LoRA guard) keys off
            # the actual pipeline params, but this records intent + outcome.
            "quant": {
                "mode": quant if quant_components else "none",
                "components": quant_components,
            },
        }

        if keep_in_vram:
            cache["pipeline"]      = pipeline
            cache["pipeline_dict"] = pipeline_dict
            cache["model_path"]    = model_path
            cache["cache_key"]     = cache_key

        denoiser = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)
        params_b = sum(p.numel() for p in denoiser.parameters()) / 1e9 if denoiser else 0.0
        quant_note = (
            f", quant={quant}({'+'.join(sorted(quant_components))})"
            if quant_components else ""
        )
        print(
            f"[EricDiffusion] Loaded — {params_b:.2f}B denoiser params, "
            f"guidance_embeds={guidance_embeds}{quant_note}"
        )

        return (pipeline_dict,)


# ═══════════════════════════════════════════════════════════════════════════
#  Unload node
# ═══════════════════════════════════════════════════════════════════════════

class EricDiffusionUnload:
    """Free VRAM by unloading the cached GEN_PIPELINE."""

    CATEGORY = "Eric Diffusion"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pipeline": ("GEN_PIPELINE", {
                    "tooltip": "Connect pipeline to trigger unload after generation.",
                }),
                "images": ("IMAGE", {
                    "tooltip": "Connect to an image output to unload after generation.",
                }),
            },
        }

    def unload(self, pipeline=None, images=None) -> Tuple[str]:
        freed = clear_gen_pipeline_cache()
        status = "Pipeline unloaded" if freed else "No pipeline was cached"
        print(f"[EricDiffusion] {status}")
        return (status,)
