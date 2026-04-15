# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion LoRA Stacker
Apply up to 8 LoRAs in a single node for any GEN_PIPELINE model.

Replaces chaining multiple Apply LoRA nodes.  Each slot has independent
weights for multi-stage generation (stage1 / stage2 / stage3), matching
the three-stage structure of EricDiffusionMultiStage.  Single-stage
workflows use the stage1 weight; stages 2 and 3 default to the same value.

Empty slots (lora_name = "none", path blank) are silently skipped.

Uses the same three-tier fallback loader (load_lora_with_key_fix) as the
per-model Apply LoRA nodes, including automatic key-prefix normalisation and
support for LoRA, LoKR, and LoHa adapter formats.

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
from typing import Tuple

from .eric_qwen_edit_lora import (
    get_lora_list,
    get_lora_full_path,
    load_lora_with_key_fix,
    unload_adapters,
    _make_adapter_name,
    _set_adapters_safe,
)

_N_SLOTS = 8

_WEIGHT_OPTS = {
    "default": 1.0,
    "min": -2.0,
    "max": 2.0,
    "step": 0.05,
}


class EricDiffusionLoRAStacker:
    """
    Apply up to 8 LoRAs to any GEN_PIPELINE in a single node.

    Each slot has three independent weights — one per generation stage —
    so you can ramp a LoRA in/out across coarse → mid → fine passes when
    using EricDiffusionMultiStage.  Single-stage workflows just use weight_1_s1.

    The node uses the same three-tier loading strategy as the individual
    Apply LoRA nodes (diffusers fast path → PEFT injection → direct merge),
    so it handles non-standard key prefixes and LoKR/LoHa formats correctly.

    The pipeline dict is annotated with applied LoRA info per stage so that
    EricDiffusionMultiStage can pick up the right weight at each stage.
    """

    CATEGORY = "Eric Diffusion"
    FUNCTION = "apply_loras"
    RETURN_TYPES = ("GEN_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = get_lora_list()
        slots = {}
        for i in range(1, _N_SLOTS + 1):
            slots[f"lora_{i}"] = (lora_list, {
                "default": "none",
                "tooltip": f"LoRA slot {i}. Select from ComfyUI/models/loras/.",
            })
            for stage in (1, 2, 3):
                slots[f"weight_{i}_s{stage}"] = ("FLOAT", {
                    **_WEIGHT_OPTS,
                    "tooltip": (
                        f"Weight for LoRA slot {i}, stage {stage}. "
                        "Used by Multi-Stage Generate; single-stage workflows use s1."
                    ),
                })
            slots[f"path_{i}"] = ("STRING", {
                "default": "",
                "tooltip": (
                    f"Optional full path override for slot {i}. "
                    "Takes priority over the dropdown when non-empty."
                ),
            })
        slots["min_compatibility"] = ("FLOAT", {
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "tooltip": (
                "Skip LoRAs whose keys match less than this fraction of the "
                "model's layers. 0.0 = warn only, never skip. 0.5 = skip if "
                "fewer than half the LoRA keys exist in the model (catches "
                "SD1/SDXL LoRAs applied to Flux/Chroma). Compatibility is "
                "always logged regardless of this setting."
            ),
        })
        return {
            "required": {
                "pipeline": ("GEN_PIPELINE", {
                    "tooltip": "From Eric Diffusion Loader or Component Loader.",
                }),
            },
            "optional": slots,
        }

    def apply_loras(self, pipeline: dict, **kwargs) -> Tuple[dict]:
        pipe       = pipeline["pipeline"]
        log        = "[EricDiffusion-LoRA]"
        min_compat = kwargs.get("min_compatibility", 0.0)

        # ── Snapshot what's currently loaded on the pipe ───────────────────
        loaded_adapters: set = set()
        try:
            for component_adapters in pipe.get_list_adapters().values():
                loaded_adapters.update(component_adapters)
        except Exception:
            pass

        # ── Pass 1: resolve every slot to (path, adapter_name, weights) ───
        # We do this BEFORE touching the pipe so we can compute the diff
        # between the new stack and whatever was loaded from a previous run.
        stack_entries: list = []  # list of (slot_idx, path, adapter_name, w1, w2, w3)
        new_stack_names: set = set()

        for i in range(1, _N_SLOTS + 1):
            lora_name = kwargs.get(f"lora_{i}", "none")
            w1        = kwargs.get(f"weight_{i}_s1", 1.0)
            w2        = kwargs.get(f"weight_{i}_s2", w1)
            w3        = kwargs.get(f"weight_{i}_s3", w1)
            path_ovr  = (kwargs.get(f"path_{i}") or "").strip()

            if path_ovr:
                if not os.path.exists(path_ovr):
                    raise ValueError(f"{log} Slot {i}: file not found: {path_ovr}")
                lora_path = path_ovr
            elif lora_name and lora_name != "none":
                lora_path = get_lora_full_path(lora_name)
                if lora_path is None:
                    raise ValueError(f"{log} Slot {i}: LoRA not found: {lora_name}")
            else:
                continue  # empty slot

            adapter_name = _make_adapter_name(os.path.basename(lora_path))
            new_stack_names.add(adapter_name)
            stack_entries.append((i, lora_path, adapter_name, w1, w2, w3))

        # ── Unload stale adapters from previous runs ──────────────────────
        # This is the fix for the "LoRAs cached across runs" bug: any adapter
        # currently loaded on the pipe but NOT in the new stack gets unloaded.
        # Without this, swapping LoRA A for LoRA B would leave both active
        # because the stacker only ever added new adapters.
        stale = loaded_adapters - new_stack_names
        if stale:
            print(f"{log} Unloading {len(stale)} stale adapter(s) from "
                  f"previous run: {sorted(stale)}")
            unload_adapters(pipe, stale, log_prefix=log)
            loaded_adapters -= stale

        # ── Reset the pipeline dict so only current-stack entries remain ──
        # The multistage node reads pipeline["applied_loras"] and applies
        # weights per stage.  If we extend instead of reset, stale entries
        # would still have their weights set even though they're unloaded.
        pipeline["applied_loras"] = {}

        # ── Pass 2: load (or refresh) each LoRA in the new stack ──────────
        applied: list = []
        for (i, lora_path, adapter_name, w1, w2, w3) in stack_entries:
            lora_filename = os.path.basename(lora_path)
            print(
                f"{log} Slot {i}: {lora_filename} "
                f"(s1={w1}, s2={w2}, s3={w3}, adapter={adapter_name})"
            )

            try:
                if adapter_name in loaded_adapters:
                    # Already loaded from a prior run — just reset its weight.
                    _set_adapters_safe(pipe, adapter_name, w1, log_prefix=log)
                    print(f"{log}   Already loaded — weight set to s1={w1}")
                else:
                    loaded = load_lora_with_key_fix(
                        pipe, lora_path, adapter_name,
                        log_prefix=log, weight=w1,
                        min_compatibility=min_compat,
                    )
                    if not loaded:
                        continue  # skipped by compatibility check
                    _set_adapters_safe(pipe, adapter_name, w1, log_prefix=log)
                    loaded_adapters.add(adapter_name)
                    print(f"{log}   Loaded OK")

                applied.append((adapter_name, w1, w2, w3))

            except Exception as e:
                print(f"{log} Slot {i} FAILED: {e}")
                raise

        # ── Annotate pipeline dict for multi-stage per-stage weight lookup ─
        for name, w1, w2, w3 in applied:
            pipeline["applied_loras"][name] = {
                "weight_s1": w1,
                "weight_s2": w2,
                "weight_s3": w3,
            }

        print(f"{log} Done — {len(applied)} LoRA(s) active")
        return (pipeline,)


# ═══════════════════════════════════════════════════════════════════════
#  Qwen-Edit-typed sibling of the stacker
# ═══════════════════════════════════════════════════════════════════════
#
# Same underlying machinery as EricDiffusionLoRAStacker — the entire
# apply_loras method is inherited unchanged.  The only difference is
# the ComfyUI socket type on the pipeline input and output, so users
# can wire this stacker between EricQwenEditLoader and the edit nodes
# (EricQwenEditImage / EricQwenEditMultiImage / EricDiffusionAdvancedEdit)
# instead of chaining multiple one-LoRA-per-node Apply LoRA nodes.
#
# Why this works unchanged: the stacker's apply_loras logic operates on
# pipeline["pipeline"] and pipeline["applied_loras"] — both of which are
# dict-key operations that work identically on GEN_PIPELINE dicts and
# QWEN_EDIT_PIPELINE dicts.  The underlying LoRA loader
# (load_lora_with_key_fix from eric_qwen_edit_lora) operates on
# pipe.transformer directly, which is the same class hierarchy for both
# pipeline flavors.  There's nothing pipeline-type-specific in the
# stacker's logic at all — the type tag is purely a ComfyUI wire
# validation hint.


class EricQwenEditLoRAStacker(EricDiffusionLoRAStacker):
    """
    Apply up to 8 LoRAs to a QWEN_EDIT_PIPELINE in a single node.

    Identical machinery to Eric Diffusion LoRA Stacker — 8 slots, per-
    stage weights, three-tier fallback loader, stale adapter cleanup,
    LoKR/LoHa support.  The only difference is the ComfyUI socket types:
    this version accepts QWEN_EDIT_PIPELINE in and emits QWEN_EDIT_PIPELINE
    out, so you can wire it between Eric Qwen-Edit Loader and the edit
    nodes without chaining individual Apply LoRA nodes.

    Outputs the same ``pipeline["applied_loras"]`` annotation as the
    generic stacker, so future multistage edit nodes can read per-stage
    weights the same way multistage generate nodes do.
    """

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "apply_loras"
    RETURN_TYPES = ("QWEN_EDIT_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        # Reuse the parent's 8-slot construction, then retype the
        # pipeline input socket.  super().INPUT_TYPES() returns a fresh
        # dict on every call so mutating it here doesn't affect the
        # generic stacker's behavior.
        types = super().INPUT_TYPES()
        types["required"]["pipeline"] = ("QWEN_EDIT_PIPELINE", {
            "tooltip": (
                "From Eric Qwen-Edit Loader or Component Loader. "
                "Wire through this stacker then to any edit node "
                "(Eric Qwen-Edit Image, Multi-Image, or Advanced Edit)."
            ),
        })
        return types
