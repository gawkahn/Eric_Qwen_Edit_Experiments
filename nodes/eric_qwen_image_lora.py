# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image LoRA Node
Apply / unload LoRA weights on the Qwen-Image (text-to-image) pipeline.

Multiple Apply LoRA nodes can be chained to stack several LoRAs with
independent weights.

Model Credits:
- Qwen-Image developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
from typing import Tuple

# Re-use the same helpers from the edit LoRA node (they are model-agnostic)
from .eric_qwen_edit_lora import get_lora_list, get_lora_full_path


# ═══════════════════════════════════════════════════════════════════════
#  Apply LoRA
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageApplyLoRA:
    """
    Apply a LoRA to the Qwen-Image generation pipeline.

    This node loads LoRA weights onto the transformer.  Multiple LoRA
    nodes can be chained to apply several LoRAs with different weights.

    Workflow example:
    [Load Model] → [Apply LoRA (style)] → [Apply LoRA (subject)] → [Generate]

    LoRA weight:
    - 1.0 = full strength
    - 0.5 = half strength
    - 0.0 = no effect (disabled)
    - >1.0 = amplified (may cause artifacts)

    Per-stage weights (for UltraGen multi-stage generation):
    - weight_stage1/2/3 override the global weight for each UltraGen stage
    - Default -1 = use global weight for that stage
    - 0.0 = disable this LoRA for that stage
    - Example: detail LoRA at [0, 0.8, 1.0] — no detail in draft, full in polish
    - Example: lightning LoRA at [0.5, 1.0, 0] — skip the fast-LoRA in the short final stage
    - Non-UltraGen nodes ignore these and use the global weight

    LoRAs are loaded from ComfyUI's standard loras folder:
    ComfyUI/models/loras/
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "apply_lora"
    RETURN_TYPES = ("QWEN_IMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_IMAGE_PIPELINE",),
                "lora_name": (get_lora_list(), {
                    "tooltip": "Select LoRA from ComfyUI/models/loras/"
                }),
                "weight": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "LoRA weight strength (1.0 = full, 0.5 = half)"
                }),
            },
            "optional": {
                "lora_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Override with custom path (leave empty to use dropdown)"
                }),
                "weight_stage1": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "UltraGen Stage 1 (draft) weight. -1 = use global weight. 0 = disabled for this stage."
                }),
                "weight_stage2": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "UltraGen Stage 2 (refine) weight. -1 = use global weight. 0 = disabled for this stage."
                }),
                "weight_stage3": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "UltraGen Stage 3 (polish) weight. -1 = use global weight. 0 = disabled for this stage."
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Refresh LoRA list on each check
        return float("nan")

    def apply_lora(
        self,
        pipeline: dict,
        lora_name: str,
        weight: float = 1.0,
        lora_path_override: str = "",
        weight_stage1: float = -1.0,
        weight_stage2: float = -1.0,
        weight_stage3: float = -1.0,
    ) -> Tuple[dict]:
        """Apply LoRA to the Qwen-Image pipeline."""
        pipe = pipeline["pipeline"]

        # Determine which path to use
        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA file not found: {lora_path}")
        else:
            if lora_name == "none" or not lora_name:
                print("[EricQwenImage] LoRA: none selected, skipping")
                return (pipeline,)
            lora_path = get_lora_full_path(lora_name)
            if lora_path is None:
                raise ValueError(f"LoRA file not found: {lora_name}")

        lora_filename = os.path.basename(lora_path)
        adapter_name = os.path.splitext(lora_filename)[0]

        print(f"[EricQwenImage] Applying LoRA: {lora_filename}")
        print(f"[EricQwenImage] Path: {lora_path}")
        print(f"[EricQwenImage] Weight: {weight}")

        # Check if this adapter is already loaded
        loaded_adapters = set()
        try:
            adapter_list = pipe.get_list_adapters()
            for component_adapters in adapter_list.values():
                loaded_adapters.update(component_adapters)
        except Exception:
            pass

        try:
            if adapter_name in loaded_adapters:
                # Already loaded — just update the weight
                pipe.set_adapters([adapter_name], adapter_weights=[weight])
                print(f"[EricQwenImage] LoRA already loaded, updated weight: {adapter_name} -> {weight}")
            else:
                # Load fresh
                pipe.load_lora_weights(
                    lora_path,
                    adapter_name=adapter_name,
                )
                pipe.set_adapters([adapter_name], adapter_weights=[weight])
                print(f"[EricQwenImage] LoRA applied successfully: {adapter_name}")

        except Exception as e:
            print(f"[EricQwenImage] Error loading LoRA: {e}")
            raise

        # Track applied adapters in the pipeline dict
        if "applied_loras" not in pipeline:
            pipeline["applied_loras"] = {}
        pipeline["applied_loras"][adapter_name] = {
            "path": lora_path,
            "weight": weight,
            "filename": lora_filename,
            "weight_stage1": weight_stage1,
            "weight_stage2": weight_stage2,
            "weight_stage3": weight_stage3,
        }

        # Log per-stage weights if any are customized
        stage_custom = any(w >= 0 for w in [weight_stage1, weight_stage2, weight_stage3])
        if stage_custom:
            s1 = weight if weight_stage1 < 0 else weight_stage1
            s2 = weight if weight_stage2 < 0 else weight_stage2
            s3 = weight if weight_stage3 < 0 else weight_stage3
            print(f"[EricQwenImage] Per-stage weights: S1={s1}, S2={s2}, S3={s3}")

        return (pipeline,)


# ═══════════════════════════════════════════════════════════════════════
#  Unload LoRA
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageUnloadLoRA:
    """
    Unload all LoRAs from the Qwen-Image generation pipeline.

    Use this to reset the model to its base state before applying
    different LoRAs, or to free memory.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "unload_lora"
    RETURN_TYPES = ("QWEN_IMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_IMAGE_PIPELINE",),
            },
        }

    def unload_lora(self, pipeline: dict) -> Tuple[dict]:
        """Unload all LoRAs from the pipeline."""
        pipe = pipeline["pipeline"]

        try:
            pipe.unload_lora_weights()
            print("[EricQwenImage] All LoRAs unloaded")
        except Exception as e:
            print(f"[EricQwenImage] Note: {e}")

        # Clear tracking
        pipeline.pop("applied_loras", None)

        return (pipeline,)
