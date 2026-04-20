# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Utilities
Shared helpers for the generic multi-model diffusion loader/generate nodes.
"""

import gc
import json
import os
from datetime import datetime
import torch


# ── Dtype helpers ────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


# ── Model-type detection ─────────────────────────────────────────────────────

#: Map from lowercase pipeline class name substring → short family identifier.
#: Checked in order; first match wins.
_FAMILY_PATTERNS = [
    ("qwenimageeditplus",   "qwen-edit"),
    ("qwenimage",           "qwen-image"),
    ("flux2klein",          "flux2klein"),
    ("flux2",               "flux2"),
    ("chroma",              "chroma"),
    ("flux",                "flux"),
    ("auraflow",            "auraflow"),
    ("stablediffusion3",    "sd3"),
    ("stablediffusionxl",   "sdxl"),
    ("stablediffusion",     "sd1"),
]


def infer_model_family(class_name: str) -> str:
    """Map a diffusers pipeline class name to a short family string."""
    lower = class_name.lower().replace("_", "").replace("-", "")
    for substr, family in _FAMILY_PATTERNS:
        if substr in lower:
            return family
    return lower  # best-effort fallback for unrecognised models


def detect_pipeline_class(model_path: str):
    """Read model_index.json and return (pipeline_class, class_name, model_family).

    Raises ValueError if the directory has no model_index.json or if the
    named class is not present in the installed diffusers version.
    """
    import diffusers

    index_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(index_path):
        raise ValueError(
            f"No model_index.json found in {model_path!r}. "
            "Make sure this is a diffusers-format model directory."
        )

    with open(index_path) as f:
        index = json.load(f)

    class_name = index.get("_class_name", "")
    if not class_name:
        raise ValueError(f"model_index.json in {model_path!r} has no '_class_name' field.")

    pipeline_class = getattr(diffusers, class_name, None)
    if pipeline_class is None:
        raise ValueError(
            f"Your installed diffusers does not have class '{class_name}'. "
            f"Try upgrading diffusers: pip install -U diffusers"
        )

    family = infer_model_family(class_name)
    return pipeline_class, class_name, family


def read_guidance_embeds(pipeline) -> bool:
    """Return True if the pipeline's transformer uses guidance embeddings."""
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        return False
    cfg = getattr(transformer, "config", None)
    if cfg is None:
        return False
    return bool(getattr(cfg, "guidance_embeds", False))


# ── Component class resolution ──────────────────────────────────────────────

def resolve_component_class(model_index: dict, component: str):
    """Return (class, class_name) for a pipeline component using model_index.json.

    model_index entries look like:
        "transformer": ["diffusers", "Flux2Transformer2DModel"]
        "text_encoder": ["transformers", "Mistral3ForConditionalGeneration"]

    Returns (None, class_name) if the class isn't available in the installed
    package — caller should raise a clear error.
    """
    entry = model_index.get(component)
    if not entry:
        return None, None
    module_name, class_name = entry
    if module_name == "diffusers":
        import diffusers
        return getattr(diffusers, class_name, None), class_name
    if module_name == "transformers":
        import transformers
        return getattr(transformers, class_name, None), class_name
    return None, class_name


def read_model_index(model_path: str) -> dict:
    """Load and return model_index.json from a model directory."""
    index_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(index_path):
        raise ValueError(f"No model_index.json in {model_path!r}")
    with open(index_path) as f:
        return json.load(f)


# ── Component path / format detection ───────────────────────────────────────

def resolve_component_path(path: str) -> str:
    """Return an absolute path, falling back to ComfyUI folder_paths lookup.

    The component loader accepts either a full filesystem path or a
    ComfyUI-relative model name (the same string shown in 'Load Diffusion
    Model' dropdowns, e.g. 'chroma/Chroma-DC-2K.safetensors').  If the
    path doesn't exist as-is, try resolving it through folder_paths.
    """
    path = path.strip()
    if os.path.exists(path):
        return path
    # Try ComfyUI folder_paths as fallback
    try:
        import folder_paths
        for search_type in ("diffusion_models", "checkpoints", "unet"):
            resolved = folder_paths.get_full_path(search_type, path)
            if resolved and os.path.exists(resolved):
                print(f"[EricDiffusion] Resolved '{path}' → '{resolved}' "
                      f"(via folder_paths/{search_type})")
                return resolved
    except Exception:
        pass
    return path  # return original; caller will get a clear "not found" error


def detect_load_variant(model_path: str) -> str | None:
    """Return 'fp16' or 'fp32' if the model directory uses variant-named weight
    files (e.g. diffusion_pytorch_model.fp16.safetensors), else None.

    Scans the first component subfolder found; all subdirs in a given model
    release use the same variant naming convention.
    """
    for subdir in ("unet", "transformer", "vae", "text_encoder", "text_encoder_2"):
        subdir_path = os.path.join(model_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if ".fp16." in fname:
                return "fp16"
            if ".fp32." in fname:
                return "fp32"
    return None


def detect_component_format(path: str) -> str:
    """Detect the layout of a component override path.

    Returns one of:
      'none'        — path is empty / not provided
      'subfolder'   — directory containing a named subfolder (e.g. 'transformer/')
      'direct'      — directory containing weight files directly
      'single_file' — a single .safetensors / .bin / .pt / .pth file
      'not_found'   — path does not exist (caller should raise a clear error)
      'unknown'     — path exists but layout is unrecognised
    """
    if not path or not path.strip():
        return "none"
    path = resolve_component_path(path.strip())
    if os.path.isfile(path):
        if path.lower().endswith((".safetensors", ".bin", ".pt", ".pth")):
            return "single_file"
        return "unknown"
    if os.path.isdir(path):
        entries = os.listdir(path)
        # Full pipeline dir with a named component subfolder
        for sub in ("transformer", "vae", "text_encoder"):
            if os.path.isdir(os.path.join(path, sub)):
                return "subfolder"
        # model_index.json → user pointed at the whole pipeline dir
        if os.path.isfile(os.path.join(path, "model_index.json")):
            return "subfolder"
        # Weight shards or config.json sitting directly in the dir
        for fname in entries:
            if fname.startswith("diffusion_pytorch_model") and fname.endswith((".safetensors", ".bin")):
                return "direct"
            if fname in ("config.json", "model.safetensors"):
                return "direct"
            if fname.startswith("model-") and fname.endswith(".safetensors"):
                return "direct"   # HF sharded format (model-00001-of-00010.safetensors)
        # Single non-standard .safetensors in a directory
        # (e.g. user pointed at the containing folder instead of the file)
        sf = [f for f in entries if f.endswith((".safetensors", ".bin"))
              and not f.startswith(".")]
        if len(sf) == 1:
            return "single_file_in_dir"
        return "unknown"
    return "not_found"


def _diagnose_slot_mismatch(ckpt_keys: set, target_slot: str) -> str:
    """Detect why a checkpoint failed to match a target component slot.

    Returns a short diagnostic hint (possibly empty string) that will be
    appended to the 'key mismatch' error message.  Detects three failure
    modes: (1) wrong file in wrong slot, (2) quantized checkpoint that
    single-file loading can't handle, (3) full-pipeline checkpoint bundled
    in one file.
    """
    sample = list(ckpt_keys)[:50]

    # ── Check 1a: bitsandbytes-quantized (NF4 / INT8) ───────────────────
    # Detect bnb-specific key suffixes.  Scan ALL keys (not just sample)
    # because the quantization markers may not appear in the first 50
    # for large checkpoints.
    is_bnb_quantized = any(
        ".quant_state." in k or ".absmax" in k or ".bitsandbytes" in k
        or k.endswith(".SCB") or k.endswith(".weight_format")
        for k in ckpt_keys
    )
    if is_bnb_quantized:
        quant_type = "bitsandbytes NF4" if any("nf4" in k.lower() for k in ckpt_keys) else "quantized (bitsandbytes)"
        return (
            f"\n\nThis is a {quant_type} checkpoint — single-file loading "
            f"does NOT support quantized models. You have two options:\n"
            f"  1. Use the un-quantized version of this checkpoint, OR\n"
            f"  2. Run dequantize_nf4.py (in this repo's root) once to convert "
            f"the NF4 checkpoint to bf16, then use the bf16 output in the "
            f"Component Loader as a normal single-file transformer override."
        )

    # ── Check 1b: ComfyUI FP8 quantization ──────────────────────────────
    # ComfyUI has its own FP8 quantization format, distinct from bnb.
    # Distinctive markers:
    #   - .comfy_quant suffix (metadata/flag per quantized tensor)
    #   - .weight_scale (per-tensor weight scale factor)
    #   - .input_scale (activation scale factor)
    # These coexist with some plain .weight tensors (not every layer is
    # quantized — norm layers and small linears often stay in bf16).
    #
    # Diffusers' converters don't recognize these suffixes; they expect
    # plain .weight / .bias / .scale.  When the converter walks to
    # `img_mlp.0.weight` but finds only `img_mlp.0.weight_scale` +
    # `img_mlp.0.comfy_quant` instead, it raises KeyError — the exact
    # symptom user hit on pornmasterFlux2Klein_v2.safetensors.
    is_comfy_fp8 = any(
        k.endswith(".comfy_quant") or k.endswith(".weight_scale")
        or k.endswith(".input_scale")
        for k in ckpt_keys
    )
    if is_comfy_fp8:
        return (
            f"\n\nThis is a ComfyUI FP8-quantized checkpoint (detected "
            f".comfy_quant / .weight_scale / .input_scale markers).  "
            f"Our loader + diffusers' converters don't support ComfyUI's "
            f"FP8 format — the quantized tensors need to be dequantized "
            f"back to bf16 before the single-file loading path can "
            f"handle them.\n\n"
            f"Options:\n"
            f"  1. Use the un-quantized version of this checkpoint if "
            f"one is available (civitai uploaders often have both).\n"
            f"  2. Use ComfyUI's native UNETLoader or CheckpointLoaderSimple "
            f"with the standard KSampler workflow — ComfyUI can consume "
            f"its own FP8 format directly.  You lose access to this "
            f"node pack's Advanced Generate/Multistage/Edit features "
            f"for this model, but the checkpoint will run.\n"
            f"  3. A dequantize_comfy.py standalone tool is on the "
            f"backlog but not yet implemented — analogous to the "
            f"existing dequantize_nf4.py for bitsandbytes."
        )

    # ── Check 2: Full-pipeline CivitAI-style checkpoint ─────────────────
    is_full_pipeline = (
        any("model.diffusion_model" in k for k in sample)
        and any("first_stage_model" in k or "text_encoders" in k or "conditioner" in k
                for k in sample)
    )
    if is_full_pipeline:
        return (
            f"\n\nThis looks like a FULL pipeline checkpoint (contains transformer "
            f"+ VAE + text encoders bundled). The component loader overrides ONE "
            f"component at a time — you can't apply a whole pipeline file to the "
            f"'{target_slot}' slot. Use the main 'Eric Diffusion Load Model' "
            f"loader on the unpacked directory."
        )

    # ── Check 3: Wrong file type for the slot ───────────────────────────
    # Detect what KIND of component this file actually is, then check if
    # it matches the target slot or belongs elsewhere.
    slot_type = None          # what this file IS
    slot_field = None         # which field to put it in
    expected_for_target = None  # what the target slot expects

    # T5-XXL encoder: keys like "encoder.block.N.layer.M.SelfAttention.*"
    if any("SelfAttention" in k and "block." in k for k in sample):
        slot_type = "T5 encoder"
        slot_field = "text_encoder_2_path (Flux/Chroma T5-XXL slot)"

    # CLIP text model: "text_model.encoder.layers.N.self_attn.*"
    elif any(k.startswith("text_model.encoder") for k in sample):
        slot_type = "CLIP text encoder"
        slot_field = "text_encoder_path (Flux/Chroma CLIP-L slot)"

    # Qwen2.5-VL / LLM: "model.layers.*" or "language_model.*"
    elif any(k.startswith(("model.layers.", "language_model.")) for k in sample):
        slot_type = "LLM / vision-language model"
        slot_field = "text_encoder_path (Qwen-Image VL slot)"

    # Flux/Chroma transformer (original or diffusers format)
    elif any("double_blocks" in k or "single_blocks" in k for k in sample):
        slot_type = "Flux-family transformer (original format)"
        slot_field = "transformer_path"
    elif any(k.startswith(("transformer_blocks.", "single_transformer_blocks."))
             for k in sample):
        slot_type = "Flux-family transformer (diffusers format)"
        slot_field = "transformer_path"

    # VAE: "encoder.down_blocks.*" or "decoder.up_blocks.*"
    elif any(k.startswith(("encoder.down_blocks", "decoder.up_blocks",
                            "encoder.conv_in", "decoder.conv_out"))
             for k in sample):
        slot_type = "VAE"
        slot_field = "vae_path"

    # Map the target slot name to its expected component type
    target_expectation = {
        "transformer":    "transformer (diffusers format)",
        "vae":            "VAE",
        "text_encoder":   "text encoder (CLIP-L for Flux, Qwen-VL for Qwen-Image)",
        "text_encoder_2": "T5-XXL encoder",
    }.get(target_slot, target_slot)

    # Does the detected type MATCH the target slot?
    target_is_transformer = target_slot == "transformer"
    target_is_vae = target_slot == "vae"
    target_is_text_encoder_1 = target_slot == "text_encoder"
    target_is_text_encoder_2 = target_slot == "text_encoder_2"

    detected_transformer = slot_type and "transformer" in slot_type.lower()
    detected_vae = slot_type == "VAE"
    detected_t5 = slot_type == "T5 encoder"
    detected_clip = slot_type == "CLIP text encoder"
    detected_llm = slot_type and "LLM" in slot_type

    slot_matches = (
        (target_is_transformer and detected_transformer) or
        (target_is_vae and detected_vae) or
        (target_is_text_encoder_1 and (detected_clip or detected_llm)) or
        (target_is_text_encoder_2 and detected_t5)
    )

    if slot_type and not slot_matches:
        return (
            f"\n\nThis file looks like a {slot_type}, but the '{target_slot}' "
            f"slot expects a {target_expectation}. Try the {slot_field} field "
            f"instead, and leave '{target_slot}' empty."
        )

    if slot_type and slot_matches:
        # Right slot, but still doesn't load — either diffusers-version
        # mismatch, key naming drift, or some unusual fine-tune format.
        return (
            f"\n\nThe file appears to be a {slot_type} (correct type for the "
            f"'{target_slot}' slot), but its keys don't match what this diffusers "
            f"version expects. This can happen with fine-tunes built against a "
            f"different diffusers version, or with unusual key-naming conventions. "
            f"Try upgrading diffusers, or use a different version of this "
            f"checkpoint."
        )

    return (
        f"\n\nThe checkpoint's keys don't match the '{target_slot}' component's "
        f"expected keys, and the file type is unrecognised. Either the file is "
        f"for a different model architecture, or it uses a format our loader "
        f"doesn't recognise."
    )


def _load_single_weights(component_class, weights_path: str, dtype,
                         base_path: str, subfolder_hint: str,
                         pipeline_class=None):
    """Load a component from a single weight file (e.g. CivitAI checkpoint).

    Primary path: use component_class.from_single_file() which handles
    key-name conversion between original/CivitAI format and diffusers format.
    The base_path provides architecture config so diffusers knows which model
    class to instantiate.

    AIO fallback (new): if the per-component loader fails and the file is
    detected as a full-pipeline checkpoint, try pipeline_class.from_single_file()
    which uses different loading machinery and can handle older key structures
    that the per-component converter rejects.  The loaded pipeline's target
    component is extracted and the rest is freed.

    Last resort: direct load_state_dict with detailed diagnostics.
    """
    weights_path = resolve_component_path(weights_path)
    config_path = os.path.join(base_path, subfolder_hint)

    # ── Peek at checkpoint keys to detect SGM prefix ─────────────────────
    # Some architecture converters in diffusers (e.g. Flux.2's
    # convert_flux2_transformer_checkpoint_to_diffusers) parse keys by
    # splitting on '.' and indexing positions, assuming keys start with
    # `double_blocks.{N}...` or `single_blocks.{N}...`.  When checkpoints
    # have a `model.diffusion_model.` prefix (ComfyUI/SGM convention), the
    # position-based parsing breaks: parts[1] becomes "diffusion_model"
    # instead of the block index, and the converter raises KeyError on a
    # nonsense within_block_name.  Pre-stripping the prefix at our level
    # before handing the checkpoint to from_single_file sidesteps the
    # upstream bug without patching diffusers.
    #
    # Detection is the same logic used by the direct-load fallback below:
    # a prefix is "dominant" if it appears on >=50% of checkpoint keys.
    def _peek_dominant_prefix(path: str):
        try:
            if path.lower().endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(path, framework="pt") as f:
                    peek_keys = list(f.keys())
            else:
                peek_keys = list(
                    torch.load(path, map_location="cpu", weights_only=True).keys()
                )
        except Exception:
            return None

        if not peek_keys:
            return None

        candidate_prefixes = [
            "model.diffusion_model.",
            "diffusion_model.",
            "first_stage_model.",
            "cond_stage_model.",
        ]
        for prefix in candidate_prefixes:
            n_with_prefix = sum(1 for k in peek_keys if k.startswith(prefix))
            if n_with_prefix >= len(peek_keys) * 0.5:
                return prefix
        return None

    detected_prefix = _peek_dominant_prefix(weights_path)

    def _write_prefix_stripped_temp(src_path: str, prefix: str) -> str:
        """Load src_path, strip prefix from all matching keys, save to a
        temp .safetensors file, and return the temp path.  Caller owns
        cleanup via the returned temp directory.
        """
        from safetensors.torch import load_file as st_load
        from safetensors.torch import save_file as st_save
        state_dict = st_load(src_path) if src_path.lower().endswith(".safetensors") \
            else torch.load(src_path, map_location="cpu", weights_only=True)
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            else:
                stripped[k] = v
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="eric_diffusion_strip_")
        temp_path = os.path.join(temp_dir, "transformer_stripped.safetensors")
        st_save(stripped, temp_path)
        del state_dict, stripped
        return temp_dir, temp_path

    # ── Primary: from_single_file (handles key conversion) ──────────────
    if hasattr(component_class, "from_single_file"):
        print(f"[EricDiffusion] Loading {subfolder_hint} via from_single_file "
              f"(key conversion enabled)")
        if detected_prefix is not None:
            print(
                f"[EricDiffusion] Detected dominant SGM prefix "
                f"{detected_prefix!r} on checkpoint keys — will also try "
                f"the pre-stripped variant if standard loading fails"
            )
        has_config = os.path.isdir(config_path)

        # Attempt 1: use base model config (fastest, no network)
        if has_config:
            try:
                component = component_class.from_single_file(
                    weights_path,
                    config=base_path,
                    subfolder=subfolder_hint,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                print(f"[EricDiffusion] {subfolder_hint} loaded successfully via "
                      f"from_single_file (base config)")
                return component
            except Exception as e:
                print(f"[EricDiffusion] from_single_file with base config failed: {e}")

        # Attempt 1b: if we detected an SGM prefix and the base config
        # is available, pre-strip the prefix to a temp file and retry.
        # Handles the diffusers Flux2 converter bug where position-based
        # parsing breaks on prefixed keys.  Uses a temp directory that
        # gets cleaned up in a finally block regardless of outcome.
        if detected_prefix is not None and has_config:
            print(
                f"[EricDiffusion] Retrying from_single_file with prefix "
                f"{detected_prefix!r} pre-stripped (workaround for "
                f"upstream converter parsing bug)..."
            )
            temp_dir = None
            try:
                temp_dir, stripped_path = _write_prefix_stripped_temp(
                    weights_path, detected_prefix,
                )
                component = component_class.from_single_file(
                    stripped_path,
                    config=base_path,
                    subfolder=subfolder_hint,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                print(
                    f"[EricDiffusion] {subfolder_hint} loaded successfully via "
                    f"from_single_file (prefix pre-stripped, base config)"
                )
                return component
            except Exception as e:
                print(
                    f"[EricDiffusion] from_single_file (prefix pre-stripped) "
                    f"failed: {e}"
                )
            finally:
                if temp_dir is not None:
                    import shutil
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass

        print(
            f"[EricDiffusion] Retrying without base config (will auto-detect "
            f"or fetch config from HuggingFace)..."
        )

        # Attempt 2: let diffusers infer the config from the checkpoint keys.
        # This handles the case where the checkpoint architecture differs from
        # the base model (e.g. original Chroma checkpoint with Chroma1-HD base).
        # May fetch a small config file from HuggingFace if not cached locally.
        try:
            component = component_class.from_single_file(
                weights_path,
                torch_dtype=dtype,
            )
            print(f"[EricDiffusion] {subfolder_hint} loaded successfully via "
                  f"from_single_file (auto-detected config)")
            return component
        except Exception as e:
            print(f"[EricDiffusion] from_single_file (auto-detect) failed: {e}")

        # Attempt 2b: if we detected a prefix, try prefix-stripping + auto-detect.
        # This is the last from_single_file attempt before we fall through to
        # the AIO and direct-load paths.
        if detected_prefix is not None:
            print(
                f"[EricDiffusion] Retrying from_single_file with prefix "
                f"{detected_prefix!r} pre-stripped (auto-detect config)..."
            )
            temp_dir = None
            try:
                temp_dir, stripped_path = _write_prefix_stripped_temp(
                    weights_path, detected_prefix,
                )
                component = component_class.from_single_file(
                    stripped_path,
                    torch_dtype=dtype,
                )
                print(
                    f"[EricDiffusion] {subfolder_hint} loaded successfully via "
                    f"from_single_file (prefix pre-stripped, auto-detect)"
                )
                return component
            except Exception as e:
                print(
                    f"[EricDiffusion] from_single_file (prefix pre-stripped "
                    f"auto-detect) failed: {e}"
                )
            finally:
                if temp_dir is not None:
                    import shutil
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass

    # ── AIO fallback: pipeline-level from_single_file ──────────────────
    #
    # Per-component loading failed.  If the file is an AIO/full-pipeline
    # checkpoint AND we have a pipeline class that supports from_single_file
    # AT THE PIPELINE LEVEL, try that — it uses different loading machinery
    # (load_single_file_sub_model per component) which sometimes handles
    # older key structures better than the per-component converter.
    if pipeline_class is not None and hasattr(pipeline_class, "from_single_file"):
        # Peek at the file to see if it looks like an AIO checkpoint
        try:
            if weights_path.lower().endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(weights_path, framework="pt") as f:
                    peek_keys = set()
                    for i, k in enumerate(f.keys()):
                        peek_keys.add(k)
                        if i >= 200:
                            break
            else:
                peek_keys = set(torch.load(
                    weights_path, map_location="cpu", weights_only=True,
                ).keys())

            looks_like_aio = (
                any("model.diffusion_model" in k for k in peek_keys)
                and (any(k.startswith("vae.") or k.startswith("first_stage_model.")
                         for k in peek_keys)
                     or any("text_encoder" in k or "conditioner" in k
                            for k in peek_keys))
            )
        except Exception:
            looks_like_aio = False

        if looks_like_aio:
            print(f"[EricDiffusion] Detected AIO full-pipeline checkpoint — "
                  f"trying pipeline-level from_single_file to extract "
                  f"{subfolder_hint}")
            try:
                loaded_pipe = pipeline_class.from_single_file(
                    weights_path,
                    config=base_path,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                # Extract the target component
                component = getattr(loaded_pipe, subfolder_hint, None)
                if component is not None:
                    print(f"[EricDiffusion] {subfolder_hint} extracted from "
                          f"AIO pipeline successfully")
                    # Free the rest of the pipeline (VAE, text encoders) since
                    # we only wanted the component — they'll be replaced by
                    # the base pipeline's versions when assembled.
                    for attr in ("vae", "text_encoder", "text_encoder_2",
                                 "tokenizer", "tokenizer_2", "scheduler",
                                 "transformer", "image_processor"):
                        if attr != subfolder_hint:
                            try:
                                setattr(loaded_pipe, attr, None)
                            except Exception:
                                pass
                    del loaded_pipe
                    import gc
                    gc.collect()
                    return component
                else:
                    print(f"[EricDiffusion] AIO pipeline loaded but has no "
                          f"'{subfolder_hint}' attribute")
            except Exception as e:
                print(f"[EricDiffusion] AIO pipeline-level load failed: {e}")

    print(f"[EricDiffusion] Falling back to direct state_dict load")

    # ── Last resort: load base config + apply state dict directly ───────
    print(f"[EricDiffusion] Loading {subfolder_hint} via direct state_dict "
          f"(no key conversion — keys must already match diffusers format)")

    if os.path.isdir(config_path):
        component = component_class.from_pretrained(
            base_path, subfolder=subfolder_hint, torch_dtype=dtype,
            local_files_only=True)
    else:
        component = component_class.from_pretrained(
            base_path, torch_dtype=dtype, local_files_only=True)

    if weights_path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu",
                                weights_only=True)

    # ── Prefix stripping: handle ComfyUI/SGM format checkpoints ──
    # ComfyUI and the original SGM pipelines store checkpoints with
    # specific prefixes that identify the component:
    #   model.diffusion_model.*  → transformer / UNet weights
    #   first_stage_model.*      → VAE weights
    #   cond_stage_model.*       → text encoder weights
    #   model.*                  → some generic wrappers
    #
    # Diffusers' own components expect these prefixes stripped.  The
    # primary path (from_single_file) usually handles this via
    # architecture-specific converter functions — but when those
    # converters don't recognize a particular key pattern (e.g.
    # ComfyUI Qwen-Image fine-tunes), the load falls through here
    # with the raw prefixed keys intact.
    #
    # Strategy: detect the dominant prefix in the state dict, strip
    # it, and re-match.  If the stripped keys match the component's
    # expected keys better than the raw ones, use the stripped dict.
    # Never modify the original state_dict — work on a copy if we
    # decide to use the stripped version.
    model_keys = set(component.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    candidate_prefixes = [
        "model.diffusion_model.",
        "diffusion_model.",
        "first_stage_model.",
        "cond_stage_model.",
        "model.",
    ]
    chosen_prefix = None
    for prefix in candidate_prefixes:
        n_with_prefix = sum(1 for k in ckpt_keys if k.startswith(prefix))
        if n_with_prefix >= len(ckpt_keys) * 0.5:
            # >=50% of checkpoint keys share this prefix — it's dominant.
            # Compute stripped-key match quality; take the first
            # prefix that gives meaningfully better matches than raw.
            stripped_keys = {
                (k[len(prefix):] if k.startswith(prefix) else k)
                for k in ckpt_keys
            }
            raw_match_pct = (
                len(model_keys & ckpt_keys) / len(model_keys) * 100
                if model_keys else 0
            )
            stripped_match_pct = (
                len(model_keys & stripped_keys) / len(model_keys) * 100
                if model_keys else 0
            )
            if stripped_match_pct > raw_match_pct + 10:
                chosen_prefix = prefix
                print(
                    f"[EricDiffusion] Detected ComfyUI/SGM prefix "
                    f"{prefix!r} on {n_with_prefix}/{len(ckpt_keys)} "
                    f"checkpoint keys — stripping prefix lifts match "
                    f"from {raw_match_pct:.0f}% to "
                    f"{stripped_match_pct:.0f}%"
                )
                break

    if chosen_prefix is not None:
        stripped_dict = {}
        for k, v in state_dict.items():
            if k.startswith(chosen_prefix):
                stripped_dict[k[len(chosen_prefix):]] = v
            else:
                stripped_dict[k] = v
        state_dict = stripped_dict
        ckpt_keys = set(state_dict.keys())

    # Report key match quality so silent failures are visible
    matched = model_keys & ckpt_keys
    pct = len(matched) / len(model_keys) * 100 if model_keys else 0

    if pct < 10:
        # Detect "wrong file in wrong slot" cases where the checkpoint's keys
        # clearly belong to a different architecture, so we can give the user
        # a more actionable error.
        hint = _diagnose_slot_mismatch(ckpt_keys, subfolder_hint)
        raise ValueError(
            f"Component '{subfolder_hint}' key mismatch: only {len(matched)}/"
            f"{len(model_keys)} keys matched ({pct:.0f}%).{hint}\n"
            f"Sample checkpoint keys: {list(ckpt_keys)[:5]}\n"
            f"Sample model keys: {list(model_keys)[:5]}"
        )

    # Report match quality.  Three tiers:
    #   >= 99%  — healthy, minor drift at most (just log the OK line)
    #   90-99% — minor mismatch, probably benign (soft note)
    #   10-89% — major mismatch; load proceeds because strict=False
    #             but this is LIKELY not what the user wants, so be loud
    #
    # The <10% case was already rejected with ValueError above.  Anything
    # in 10-89% means load_state_dict(strict=False) will apply the matched
    # weights and SILENTLY retain base-model weights for every unmatched
    # parameter — which makes the final component a hybrid the user may
    # not notice unless they A/B against base.  We can't make that call
    # for the user (some legit fine-tunes are partial), but we can be
    # extremely clear about what just happened.
    missing_keys = model_keys - ckpt_keys
    unused_keys = ckpt_keys - model_keys

    if pct >= 99:
        print(
            f"[EricDiffusion] {len(matched)}/{len(model_keys)} keys matched "
            f"({pct:.1f}%) — healthy full override"
        )
    elif pct >= 90:
        print(
            f"[EricDiffusion] {len(matched)}/{len(model_keys)} keys matched "
            f"({pct:.1f}%) — minor mismatch; "
            f"{len(missing_keys)} param(s) missing from checkpoint will keep "
            f"base-model values"
        )
    else:
        # Loud multi-line warning for major mismatches.  User explicitly
        # doesn't want this to be a hard fail (some civitai files are
        # orphaned and have no better alternative), but the output from
        # this component will be a hybrid of fine-tune + base that may
        # not behave as intended.
        print(
            f"[EricDiffusion] ╔══════════════════════════════════════════════════════════╗"
        )
        print(
            f"[EricDiffusion] ║  WARNING: checkpoint only partially matches model.        ║"
        )
        print(
            f"[EricDiffusion] ║  The effects may not be what you expect.                  ║"
        )
        print(
            f"[EricDiffusion] ╚══════════════════════════════════════════════════════════╝"
        )
        print(
            f"[EricDiffusion]   Matched:  {len(matched)}/{len(model_keys)} "
            f"model params ({pct:.1f}%)"
        )
        print(
            f"[EricDiffusion]   Missing:  {len(missing_keys)} model params NOT in "
            f"checkpoint — these will silently retain the BASE MODEL's "
            f"values (not your fine-tune's)"
        )
        if unused_keys:
            print(
                f"[EricDiffusion]   Unused:   {len(unused_keys)} checkpoint "
                f"keys were not consumed by the model (may indicate a "
                f"different architecture variant or format drift)"
            )
        print(
            f"[EricDiffusion]   Sample missing params: "
            f"{list(missing_keys)[:3]}"
        )
        if unused_keys:
            print(
                f"[EricDiffusion]   Sample unused keys:    "
                f"{list(unused_keys)[:3]}"
            )
        print(
            f"[EricDiffusion]   The component will load anyway (strict=False). "
            f"If output looks off, this mismatch is likely why."
        )

    component.load_state_dict(state_dict, strict=False)
    return component


def _find_weight_files(directory: str) -> list:
    """Return a sorted list of .safetensors/.bin weight files in a directory."""
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith((".safetensors", ".bin")) and not f.startswith(".")
    )


def _try_from_single_file(component_class, weight_path: str, dtype,
                          base_path: str = None, subfolder_hint: str = None):
    """Attempt from_single_file with key conversion. Returns component or None.

    Two attempts: first with the base model config (no network), then without
    config (auto-detect, may fetch config from HuggingFace if the checkpoint
    architecture differs from the base).
    """
    if not hasattr(component_class, "from_single_file"):
        return None

    has_config = (base_path and subfolder_hint and
                  os.path.isdir(os.path.join(base_path, subfolder_hint)))

    # Attempt 1: use base config
    if has_config:
        try:
            component = component_class.from_single_file(
                weight_path, config=base_path, subfolder=subfolder_hint,
                torch_dtype=dtype, local_files_only=True)
            print(f"[EricDiffusion] Loaded via from_single_file (base config)")
            return component
        except Exception as e:
            print(f"[EricDiffusion] from_single_file with base config failed: {e}")

    # Attempt 2: auto-detect config
    try:
        component = component_class.from_single_file(
            weight_path, torch_dtype=dtype)
        print(f"[EricDiffusion] Loaded via from_single_file (auto-detected config)")
        return component
    except Exception as e:
        print(f"[EricDiffusion] from_single_file (auto-detect) failed: {e}")
        return None


def load_component(component_class, path: str, dtype, base_path: str = None,
                   subfolder_hint: str = None, pipeline_class=None):
    """Load a pipeline component from path, handling all layout formats.

    For every format, tries from_pretrained first (expects diffusers-format
    keys).  If that fails or the loaded state dict has original-format keys,
    falls back to from_single_file which handles key conversion automatically.

    Args:
        component_class : The diffusers/transformers class to instantiate.
        path            : Path provided by the user (file, directory, or
                          ComfyUI model name resolved via folder_paths).
        dtype           : torch dtype (bfloat16 etc.)
        base_path       : Base pipeline path — used when path is a single
                          weight file (we need the base config).
        subfolder_hint  : Subfolder name to try first (e.g. 'transformer').
        pipeline_class  : Optional pipeline class for AIO fallback loading.
                          If provided, full-pipeline single-file checkpoints
                          can be loaded at the pipeline level, and the
                          target component extracted from the result.
    """
    path = resolve_component_path(path.strip())
    fmt  = detect_component_format(path)

    if fmt == "not_found":
        raise ValueError(
            f"Component path not found: {path!r}\n"
            f"Provide a full filesystem path or a ComfyUI model name "
            f"(e.g. 'chroma/Chroma-DC-2K.safetensors')."
        )

    # ── Directory with a component subfolder (e.g. path/transformer/) ───
    if fmt == "subfolder":
        try:
            if subfolder_hint and os.path.isdir(os.path.join(path, subfolder_hint)):
                return component_class.from_pretrained(
                    path, subfolder=subfolder_hint, torch_dtype=dtype,
                    local_files_only=True)
            return component_class.from_pretrained(
                path, torch_dtype=dtype, local_files_only=True)
        except Exception as e:
            print(f"[EricDiffusion] from_pretrained failed for subfolder: {e}")
            # Find weight files in the subfolder and try from_single_file
            sub_dir = os.path.join(path, subfolder_hint) if subfolder_hint else path
            if os.path.isdir(sub_dir):
                for wf in _find_weight_files(sub_dir):
                    result = _try_from_single_file(
                        component_class, os.path.join(sub_dir, wf), dtype,
                        base_path, subfolder_hint)
                    if result is not None:
                        return result
            raise

    # ── Directory with weight files directly ────────────────────────────
    if fmt == "direct":
        try:
            return component_class.from_pretrained(
                path, torch_dtype=dtype, local_files_only=True)
        except Exception as e:
            print(f"[EricDiffusion] from_pretrained failed for direct: {e}")
            # Try from_single_file on each weight file in the directory
            for wf in _find_weight_files(path):
                result = _try_from_single_file(
                    component_class, os.path.join(path, wf), dtype,
                    base_path, subfolder_hint)
                if result is not None:
                    return result
            raise

    # ── Single weight file ──────────────────────────────────────────────
    if fmt == "single_file":
        if base_path is None or subfolder_hint is None:
            raise ValueError(
                "single_file format requires base_path and subfolder_hint to load config")
        return _load_single_weights(component_class, path, dtype,
                                    base_path, subfolder_hint,
                                    pipeline_class=pipeline_class)

    # ── Directory with exactly one weight file ──────────────────────────
    if fmt == "single_file_in_dir":
        entries = _find_weight_files(path)
        weight_path = os.path.join(path, entries[0])
        print(f"[EricDiffusion] Resolved directory to single weight file: {entries[0]}")
        if base_path is None or subfolder_hint is None:
            raise ValueError(
                "single_file format requires base_path and subfolder_hint to load config")
        return _load_single_weights(component_class, weight_path, dtype,
                                    base_path, subfolder_hint,
                                    pipeline_class=pipeline_class)

    raise ValueError(
        f"Cannot determine component format for: {path!r}\n"
        f"Expected a directory with weight files, a directory with a component "
        f"subfolder, or a single .safetensors / .bin file."
    )


# ── Generic pipeline cache ───────────────────────────────────────────────────

_GEN_PIPELINE_CACHE: dict = {
    "pipeline": None,
    "pipeline_dict": None,
    "model_path": None,
    "cache_key": None,
}


def get_gen_pipeline_cache() -> dict:
    return _GEN_PIPELINE_CACHE


def clear_gen_pipeline_cache() -> bool:
    """Move models to CPU, release references, and free VRAM."""
    global _GEN_PIPELINE_CACHE
    pipe = _GEN_PIPELINE_CACHE.get("pipeline")
    if pipe is None:
        return False

    # Best-effort move to CPU before releasing.  Detect accelerate
    # dispatch (balanced mode) — direct .to("cpu") fails on dispatched
    # pipelines; use reset_device_map() first to remove the hooks.
    using_device_map = (
        hasattr(pipe, "hf_device_map") and pipe.hf_device_map
    )

    if using_device_map:
        try:
            pipe.reset_device_map()
            pipe.to("cpu")
        except Exception as e:
            print(
                f"[EricDiffusion] accelerate reset_device_map "
                f"unavailable ({e}) — relying on GC to free "
                f"dispatched pipeline memory"
            )
    else:
        try:
            pipe.to("cpu")
        except Exception:
            for attr in ("transformer", "vae", "text_encoder", "text_encoder_2"):
                comp = getattr(pipe, attr, None)
                if comp is not None:
                    try:
                        comp.to("cpu")
                    except Exception:
                        pass

    _GEN_PIPELINE_CACHE["pipeline"] = None
    _GEN_PIPELINE_CACHE["pipeline_dict"] = None
    _GEN_PIPELINE_CACHE["model_path"] = None
    _GEN_PIPELINE_CACHE["cache_key"] = None
    del pipe

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("[EricDiffusion] Pipeline cache cleared, VRAM freed")
    return True


# ── Metadata helpers ─────────────────────────────────────────────────────────

def build_model_metadata(pipeline_dict: dict) -> dict:
    """Extract model identity and LoRA fields from a GEN_PIPELINE dict.

    Returns a partial metadata dict suitable for merging into any node's
    full metadata before returning GEN_METADATA.  Centralises the
    transformer_override_name → model_name derivation and LoRA extraction
    so every node uses the same logic.
    """
    model_name = (
        pipeline_dict.get("transformer_override_name")
        or os.path.basename(pipeline_dict.get("model_path", ""))
    )

    raw_loras = pipeline_dict.get("applied_loras") or {}
    loras = {}
    for name, info in raw_loras.items():
        w1 = info.get("weight_s1", 1.0)
        loras[name] = {
            "weight_s1": w1,
            "weight_s2": info.get("weight_s2", w1),
            "weight_s3": info.get("weight_s3", w1),
        }

    return {
        "model_name":   model_name,
        "model_path":   pipeline_dict.get("model_path", ""),
        "model_family": pipeline_dict.get("model_family", ""),
        "loras":        loras,
    }
