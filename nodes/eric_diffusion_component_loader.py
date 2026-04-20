# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Generic Component Loader
Load any diffusers text-to-image pipeline with per-component overrides.

The base pipeline provides config, scheduler, tokenizer, and defaults for
any component not overridden.  Component classes are resolved automatically
from the base model's model_index.json — no model-specific code required.

Supports transformer, VAE, and text_encoder overrides.  Path formats:
  - Directory with component subfolder  (e.g. /model/  →  /model/transformer/)
  - Directory with weight files directly
  - Single .safetensors / .bin file (config taken from base pipeline)

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import torch
from typing import Tuple

from .eric_diffusion_utils import (
    DTYPE_MAP,
    detect_pipeline_class,
    detect_load_variant,
    read_model_index,
    resolve_component_class,
    detect_component_format,
    load_component,
    read_guidance_embeds,
    get_gen_pipeline_cache,
    clear_gen_pipeline_cache,
)


def available_device_options() -> list:
    """Return the device dropdown options filtered by available hardware.

    Called from ``INPUT_TYPES()`` on all loader nodes so the dropdown
    reflects the actual hardware available to the ComfyUI container:

    - **0 GPUs visible**: ``["cpu"]`` only.
    - **1 GPU visible**: ``["cuda", "cuda:0", "cpu"]`` — no "balanced"
      (nothing to balance), no "cuda:1" (doesn't exist).
    - **2+ GPUs visible**: ``["balanced", "cuda", "cuda:0", "cuda:1", ..., "cpu"]``
      — balanced option exposed because there's something to balance across.

    This prevents users from accidentally selecting options that would
    fail or provide no benefit.  The list is computed at node
    registration time; if the container's GPU visibility changes
    (e.g. moving a workflow between a 2-GPU dev host and a 1-GPU
    production container), the dropdown may be stale but the runtime
    fallback in ``resolve_device_with_fallback`` handles that safely.

    Returns:
        List of device option strings suitable for an INPUT_TYPES enum.
    """
    n_gpus = 0
    try:
        n_gpus = torch.cuda.device_count()
    except Exception:
        pass

    options = []
    if n_gpus >= 2:
        options.append("balanced")
    if n_gpus >= 1:
        options.append("cuda")
        for i in range(n_gpus):
            options.append(f"cuda:{i}")
    options.append("cpu")
    return options


def resolve_device_with_fallback(
    device: str, log_prefix: str = "",
) -> Tuple[str, bool]:
    """Validate a device string against visible hardware and return a
    safe (device, use_device_map) pair.

    Handles the edge case where a workflow JSON saved on a multi-GPU
    host gets loaded on a single-GPU host — the dropdown's
    registration-time filter wouldn't have caught this because ComfyUI
    passes the stale value through.

    Fallback rules:
    - ``"balanced"`` on <2 GPUs → ``"cuda"`` (single-GPU) or ``"cpu"``
      (no GPUs), with a warning.
    - ``"cuda:N"`` where N >= device_count → ``"cuda:0"`` or ``"cpu"``,
      with a warning.
    - ``"cuda"`` on no GPUs → ``"cpu"``, with a warning.
    - Anything else → passed through unchanged.

    Args:
        device     : Requested device string from the loader input.
        log_prefix : Log line prefix for warnings (e.g. "[EricQwenEdit]").

    Returns:
        (resolved_device, use_device_map).  ``use_device_map`` is True
        only when the resolved device is "balanced" AND hardware
        supports it (2+ GPUs visible).
    """
    n_gpus = 0
    try:
        n_gpus = torch.cuda.device_count()
    except Exception:
        pass

    if device == "balanced":
        if n_gpus >= 2:
            return device, True
        elif n_gpus == 1:
            print(
                f"{log_prefix} WARNING: 'balanced' device requested but "
                f"only 1 GPU visible — falling back to 'cuda'.  balanced "
                f"mode provides no benefit with a single GPU and adds "
                f"accelerate hook overhead."
            )
            return "cuda", False
        else:
            print(
                f"{log_prefix} WARNING: 'balanced' device requested but "
                f"no GPUs visible — falling back to 'cpu' (very slow)."
            )
            return "cpu", False

    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":", 1)[1])
            if idx >= n_gpus:
                fallback = "cuda:0" if n_gpus > 0 else "cpu"
                print(
                    f"{log_prefix} WARNING: '{device}' requested but "
                    f"only {n_gpus} GPU(s) visible — falling back to "
                    f"'{fallback}'."
                )
                return fallback, False
        except (IndexError, ValueError):
            pass

    if device == "cuda" and n_gpus == 0:
        print(
            f"{log_prefix} WARNING: 'cuda' requested but no GPUs "
            f"visible — falling back to 'cpu' (very slow)."
        )
        return "cpu", False

    return device, False


def _fix_text_encoder_device(pipeline, log_prefix: str = "") -> None:
    """Register forward hooks on BOTH text encoders to handle CPU/GPU device splits.

    When device_map="balanced" dispatches the transformer to GPU but one or
    both text encoders live on CPU, the pipeline moves tokenizer outputs to
    GPU (execution_device) *before* calling the text encoder.  The pre-hook
    moves those inputs back to the encoder's own device so the forward runs
    correctly.  The post-hook then moves the encoder *outputs* back to the
    execution device so that downstream code can use them without device
    mismatches.

    Both ``text_encoder`` (slot 1 — CLIP-L for Flux/Chroma, Qwen2.5-VL for
    Qwen-Image) AND ``text_encoder_2`` (slot 2 — T5-XXL for Flux/Chroma,
    unused for Qwen) are hooked.  Previously this only handled slot 1, which
    left Flux's T5 unprotected — Flux dev + balanced mode would fail with
    a device mismatch when T5 was placed on CPU but input_ids were routed
    to cuda:N.
    """
    # Already patched?
    if getattr(pipeline, "_eric_te_device_patched", False):
        return

    def _make_pre_hook():
        """Pre-hook factory: moves every input tensor to the module's own device."""
        def _pre(module, args, kwargs):
            try:
                dev = next(module.parameters()).device
            except StopIteration:
                return args, kwargs
            new_args = tuple(
                a.to(dev) if isinstance(a, torch.Tensor) else a for a in args
            )
            new_kwargs = {
                k: v.to(dev) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            return new_args, new_kwargs
        return _pre

    def _make_post_hook():
        """Post-hook factory: moves every output tensor back to exec_device."""
        def _post(module, input, output):
            try:
                exec_dev = pipeline._execution_device
            except AttributeError:
                return output

            def _move(x):
                if isinstance(x, torch.Tensor):
                    return x.to(exec_dev)
                if isinstance(x, (tuple, list)):
                    moved = [_move(v) for v in x]
                    return type(x)(moved)
                return x

            # ModelOutput is an OrderedDict subclass — iterate dict items.
            # output[key] = val calls ModelOutput.__setitem__ which syncs __dict__ too.
            if isinstance(output, dict):
                for key in list(output.keys()):
                    if output[key] is not None:
                        output[key] = _move(output[key])
                return output

            return _move(output)
        return _post

    hooked_count = 0
    for te_name in ("text_encoder", "text_encoder_2"):
        te = getattr(pipeline, te_name, None)
        if te is None:
            continue

        # CRITICAL: Skip text encoders that accelerate already manages
        # via device_map dispatch.  Installing our own pre/post hooks on
        # top of accelerate's AlignDevicesHook causes a device routing
        # conflict — accelerate moves inputs to its own execution device,
        # we move them to the first parameter's device, and the two may
        # diverge on a split-across-GPUs encoder.  The result is a
        # RuntimeError in downstream ops like ``_extract_masked_hidden``
        # where ``hidden_states`` and ``attention_mask`` end up on
        # different devices.
        #
        # Check for ``_hf_hook`` on the top-level module — accelerate
        # attaches that attribute on every dispatched module.  When
        # present, accelerate owns the device routing and we should not
        # interfere.
        if hasattr(te, "_hf_hook"):
            print(
                f"{log_prefix} {te_name} has accelerate hooks — skipping "
                f"eric device-fix hooks (accelerate handles routing natively)"
            )
            continue

        # Also check submodules — accelerate may hook individual layers
        # even if the top-level module doesn't have the attribute.  If
        # ANY submodule has _hf_hook, accelerate is managing this encoder.
        has_any_accelerate_hook = False
        for submodule in te.modules():
            if hasattr(submodule, "_hf_hook"):
                has_any_accelerate_hook = True
                break
        if has_any_accelerate_hook:
            print(
                f"{log_prefix} {te_name} has accelerate hooks on "
                f"submodules — skipping eric device-fix hooks"
            )
            continue

        try:
            te_device = next(te.parameters()).device
        except StopIteration:
            print(f"{log_prefix} {te_name} has no parameters — skipping device hook")
            continue

        te.register_forward_pre_hook(_make_pre_hook(), with_kwargs=True)
        te.register_forward_hook(_make_post_hook())
        hooked_count += 1
        print(f"{log_prefix} {te_name} device hooks registered "
              f"(current device: {te_device})")

    if hooked_count:
        pipeline._eric_te_device_patched = True


class EricDiffusionComponentLoader:
    """
    Generic component loader — load a base pipeline then swap in custom
    transformer, VAE, and/or text_encoder weights.

    Works with any diffusers model (Flux.2, Qwen-Image, etc.).  Point
    base_pipeline_path at a complete model directory; that provides the
    architecture config, scheduler, and tokenizer.  Override any component
    by providing its path.

    Typical uses:
      - Fine-tuned transformer + base pipeline config
      - Abliterated / custom text encoder (e.g. uncensored Qwen VL)
      - Custom VAE for different decode characteristics
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
                "base_pipeline_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to a complete model directory. Provides config, scheduler, "
                        "tokenizer, and defaults for any component not overridden."
                    ),
                }),
            },
            "optional": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Custom transformer weights. Leave empty to use base. "
                        "Accepts: directory with weight files, directory with "
                        "'transformer/' subfolder, or single .safetensors file."
                    ),
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "Custom VAE. Leave empty to use base.",
                }),
                "text_encoder_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Custom text encoder slot 1. "
                        "For Flux/Chroma this is CLIP-L. "
                        "For Qwen-Image this is Qwen2.5-VL (e.g. abliterated VLM). "
                        "Leave empty to use base."
                    ),
                }),
                "text_encoder_2_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Custom text encoder slot 2. "
                        "For Flux/Chroma this is T5-XXL (for e.g. flan-t5-xxl or "
                        "improved T5 fine-tunes). "
                        "Not used by Qwen-Image. "
                        "Leave empty to use base."
                    ),
                }),
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "bf16 recommended for RTX 40/50 series.",
                }),
                "device": (available_device_options(), {
                    "default": (
                        "balanced" if "balanced" in available_device_options()
                        else ("cuda" if "cuda" in available_device_options() else "cpu")
                    ),
                    "tooltip": (
                        "Device to load model on.  Options filtered by "
                        "visible hardware.\n\n"
                        "• balanced — splits across all visible GPUs.  "
                        "Only shown when 2+ GPUs are visible.\n"
                        "• cuda / cuda:N — pin to a single GPU."
                    ),
                }),
                "keep_in_vram": ("BOOLEAN", {"default": True}),
                "offload_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move VAE to CPU during transformer inference (saves ~1 GB).",
                }),
                "attention_slicing": ("BOOLEAN", {"default": False}),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extreme VRAM savings via sequential CPU offload — very slow.",
                }),
                "vae_from_transformer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Extract VAE from the transformer_path AIO checkpoint instead of "
                        "using the base pipeline's VAE. Useful when the checkpoint bundles "
                        "a custom or stabilized VAE (e.g. Illustrious, NoobAI, AllInOne). "
                        "Ignored if vae_path is also set."
                    ),
                }),
            },
        }

    def load_pipeline(
        self,
        base_pipeline_path: str,
        transformer_path: str = "",
        vae_path: str = "",
        text_encoder_path: str = "",
        text_encoder_2_path: str = "",
        precision: str = "bf16",
        device: str = "balanced",
        keep_in_vram: bool = True,
        offload_vae: bool = False,
        attention_slicing: bool = False,
        sequential_offload: bool = False,
        vae_from_transformer: bool = False,
    ) -> Tuple:
        base_pipeline_path   = base_pipeline_path.strip()
        transformer_path     = transformer_path.strip()
        vae_path             = vae_path.strip()
        text_encoder_path    = text_encoder_path.strip()
        text_encoder_2_path  = text_encoder_2_path.strip()

        cache_key = (
            f"comp_{base_pipeline_path}_{transformer_path}_{vae_path}_"
            f"{text_encoder_path}_{text_encoder_2_path}_{precision}_{device}_{offload_vae}"
            f"_{attention_slicing}_{sequential_offload}_{vae_from_transformer}"
        )
        cache = get_gen_pipeline_cache()

        if cache["pipeline"] is not None and cache.get("cache_key") == cache_key:
            print("[EricDiffusion] Using cached component pipeline")
            return (cache["pipeline_dict"],)

        if cache["pipeline"] is not None:
            print("[EricDiffusion] Config changed — clearing cached pipeline")
            clear_gen_pipeline_cache()

        dtype = DTYPE_MAP.get(precision, torch.bfloat16)

        # ── Resolve pipeline class and component classes from model_index ──
        pipeline_class, class_name, model_family = detect_pipeline_class(base_pipeline_path)
        model_index = read_model_index(base_pipeline_path)
        print(f"[EricDiffusion] Component load: {class_name} (family: {model_family})")

        # ── Load component overrides ────────────────────────────────────────
        kwargs = {}

        if transformer_path:
            fmt = detect_component_format(transformer_path)
            print(f"[EricDiffusion] Transformer override: {transformer_path!r} (fmt: {fmt})")
            cls_, cname = resolve_component_class(model_index, "transformer")
            # SDXL/SD1 use a "unet" slot, not "transformer" — fall back transparently.
            transformer_slot = "transformer"
            if cls_ is None and model_index.get("unet"):
                cls_, cname = resolve_component_class(model_index, "unet")
                transformer_slot = "unet"
            if cls_ is None:
                raise ValueError(
                    f"Transformer/UNet class '{cname}' not found in installed diffusers. "
                    "Try: pip install -U diffusers"
                )
            kwargs[transformer_slot] = load_component(
                cls_, transformer_path, dtype,
                base_path=base_pipeline_path, subfolder_hint=transformer_slot,
                pipeline_class=pipeline_class,
            )
            print(f"[EricDiffusion] Custom {transformer_slot} loaded ({cname})")

        if vae_from_transformer and transformer_path and not vae_path:
            cls_, cname = resolve_component_class(model_index, "vae")
            if cls_ is None:
                raise ValueError(f"VAE class '{cname}' not found in installed diffusers.")
            kwargs["vae"] = load_component(
                cls_, transformer_path, dtype,
                base_path=base_pipeline_path, subfolder_hint="vae",
                pipeline_class=pipeline_class,
            )
            print(f"[EricDiffusion] VAE extracted from transformer checkpoint ({cname})")

        if vae_path:
            fmt = detect_component_format(vae_path)
            print(f"[EricDiffusion] VAE override: {vae_path!r} (fmt: {fmt})")
            cls_, cname = resolve_component_class(model_index, "vae")
            if cls_ is None:
                raise ValueError(f"VAE class '{cname}' not found in installed diffusers.")
            kwargs["vae"] = load_component(
                cls_, vae_path, dtype,
                base_path=base_pipeline_path, subfolder_hint="vae",
                pipeline_class=pipeline_class,
            )
            print(f"[EricDiffusion] Custom VAE loaded ({cname})")

        if text_encoder_path:
            fmt = detect_component_format(text_encoder_path)
            print(f"[EricDiffusion] Text encoder (slot 1) override: {text_encoder_path!r} (fmt: {fmt})")
            cls_, cname = resolve_component_class(model_index, "text_encoder")
            if cls_ is None:
                raise ValueError(f"Text encoder class '{cname}' not found in transformers.")
            kwargs["text_encoder"] = load_component(
                cls_, text_encoder_path, dtype,
                base_path=base_pipeline_path, subfolder_hint="text_encoder",
                pipeline_class=pipeline_class,
            )
            print(f"[EricDiffusion] Custom text encoder (slot 1) loaded ({cname})")

        if text_encoder_2_path:
            fmt = detect_component_format(text_encoder_2_path)
            print(f"[EricDiffusion] Text encoder (slot 2) override: {text_encoder_2_path!r} (fmt: {fmt})")
            cls_, cname = resolve_component_class(model_index, "text_encoder_2")
            if cls_ is None:
                raise ValueError(
                    f"Text encoder 2 class '{cname}' not found in transformers, or "
                    f"this pipeline family does not have a second text encoder. "
                    f"Leave text_encoder_2_path empty."
                )
            kwargs["text_encoder_2"] = load_component(
                cls_, text_encoder_2_path, dtype,
                base_path=base_pipeline_path, subfolder_hint="text_encoder_2",
                pipeline_class=pipeline_class,
            )
            print(f"[EricDiffusion] Custom text encoder (slot 2) loaded ({cname})")

        # ── Assemble pipeline ───────────────────────────────────────────────
        overrides = list(kwargs.keys()) or ["all from base"]
        print(f"[EricDiffusion] Assembling pipeline, overrides: {overrides}")

        # Runtime device fallback handles stale-workflow-JSON case.
        device, use_device_map = resolve_device_with_fallback(
            device, log_prefix="[EricDiffusion]",
        )
        load_kwargs = dict(torch_dtype=dtype, local_files_only=True, **kwargs)
        if use_device_map:
            load_kwargs["device_map"] = "balanced"
        variant = detect_load_variant(base_pipeline_path)
        if variant:
            load_kwargs["variant"] = variant

        pipeline = pipeline_class.from_pretrained(base_pipeline_path, **load_kwargs)

        # ── Optimizations ──────────────────────────────────────────────────
        if sequential_offload:
            print("[EricDiffusion] Enabling sequential CPU offload")
            pipeline.enable_sequential_cpu_offload()
        elif not use_device_map:
            pipeline = pipeline.to(device)
            if offload_vae and hasattr(pipeline, "vae"):
                print("[EricDiffusion] Moving VAE to CPU")
                pipeline.vae = pipeline.vae.to("cpu")

        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
            print("[EricDiffusion] VAE tiling enabled")

        if attention_slicing:
            try:
                pipeline.enable_attention_slicing(slice_size="auto")
                print("[EricDiffusion] Attention slicing enabled")
            except Exception as e:
                print(f"[EricDiffusion] Attention slicing not available: {e}")

        # ── Fix: pre-loaded components bypass device_map dispatch ─────────────
        # When a custom component is passed as a pre-loaded object to
        # from_pretrained(..., device_map="balanced"), diffusers skips
        # dispatching it — it stays wherever it was loaded (CPU by default).
        # Other components get dispatched to CUDA, so _execution_device
        # becomes cuda:N while our overrides sit on CPU → device mismatch.
        # Move any pre-loaded overrides to the execution device.
        # Skip any that accelerate already hooked (_hf_hook present).
        if use_device_map:
            try:
                exec_dev = pipeline._execution_device
            except AttributeError:
                exec_dev = None

            override_components = []
            if transformer_path:
                override_components.append("transformer")
            if text_encoder_path:
                override_components.append("text_encoder")
            if text_encoder_2_path:
                override_components.append("text_encoder_2")
            if vae_path:
                override_components.append("vae")

            for comp_name in override_components:
                comp = getattr(pipeline, comp_name, None)
                if comp is None:
                    continue
                if hasattr(comp, "_hf_hook"):
                    continue  # accelerate will handle it
                try:
                    comp_dev = next(comp.parameters()).device
                except StopIteration:
                    continue
                if exec_dev is not None and comp_dev.type == "cpu" and str(exec_dev) != "cpu":
                    print(f"[EricDiffusion] Moving {comp_name} {comp_dev} → {exec_dev} "
                          f"(was pre-loaded and missed device_map dispatch)")
                    setattr(pipeline, comp_name, comp.to(exec_dev))

        # ── Text encoder: keep on CPU but fix input routing ────────────────
        # Chroma (and similar pipelines) move input_ids to self._execution_device
        # before calling the text encoder, which may be on CPU.  The hook
        # re-checks the actual device on each call so it's a no-op when
        # encoder and execution device already match.
        _fix_text_encoder_device(pipeline, "[EricDiffusion]")

        guidance_embeds = read_guidance_embeds(pipeline)

        pipeline_dict = {
            "pipeline":        pipeline,
            "model_path":      base_pipeline_path,
            "transformer_override_name": os.path.splitext(os.path.basename(transformer_path))[0] if transformer_path else None,
            "model_family":    model_family,
            "offload_vae":     offload_vae,
            "guidance_embeds": guidance_embeds,
        }

        if keep_in_vram:
            cache["pipeline"]      = pipeline
            cache["pipeline_dict"] = pipeline_dict
            cache["model_path"]    = base_pipeline_path
            cache["cache_key"]     = cache_key

        denoiser = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)
        params_b = sum(p.numel() for p in denoiser.parameters()) / 1e9 if denoiser else 0.0
        print(
            f"[EricDiffusion] Component pipeline ready — {params_b:.2f}B denoiser params, "
            f"overrides: {overrides}"
        )

        return (pipeline_dict,)
