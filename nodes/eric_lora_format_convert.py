# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric LoRA Format Converter — framework / shared infrastructure.

Goal: convert a LoRA whose key layout matches one model-family
convention (e.g. original BFL Flux/Klein/Chroma — `double_blocks.X.
img_attn.qkv.lora_A.weight`) into the layout of a different family
(e.g. diffusers Flux/Flux2 — `transformer_blocks.X.attn.to_q.lora_A.
weight`), so a single LoRA file works against either runtime without
per-user manual remapping.

This module provides ONLY the framework:

  - format detection (BFL-original vs diffusers-DiT vs SD-UNet)
  - data classes for a rename plan (RenameRule, ConversionPlan)
  - mechanical key-rename application
  - QKV-split helpers (exact for standard LoRA, SVD-based for LoKR/LoHa)
  - an empty CONVERSION_PLANS registry

Slice 3 fills in the Flux/Flux2/Klein rename tables, slice 5 fills in
Chroma.  Slice 4 wires the registry into load_lora_with_key_fix so
in-memory conversion happens transparently when a LoRA fails to load
the standard way.

This module is import-safe outside ComfyUI: no folder_paths, no comfy.*
dependencies; only torch and the Python stdlib.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch


# ════════════════════════════════════════════════════════════════════════
#  Format detection
# ════════════════════════════════════════════════════════════════════════

# (family_name, list of substrings; ANY match → this family wins).
# Order matters — the first matching family is returned.
_FORMAT_MARKERS: List[Tuple[str, List[str]]] = [
    # Original BFL: shared between Flux.1, Flux.2, Klein, and Chroma.
    # Slices 3/5 distinguish them downstream from the model side.
    ("bfl_original",  ["double_blocks.", "single_blocks."]),
    # Diffusers DiT: Flux, Flux2, Chroma all reorganise to this layout.
    # Qwen-Image and Qwen-Image-Edit also use transformer_blocks (their
    # own variant); the model-side detector disambiguates.
    ("diffusers_dit", ["transformer_blocks.", "single_transformer_blocks."]),
    # SD-family UNet: SD1.5, SDXL, Pony, Illustrious, etc.
    ("sd_unet",       ["up_blocks.", "down_blocks.", "mid_block."]),
]


def detect_lora_format(lora_keys: Iterable[str]) -> str:
    """Return the family name whose markers appear in any LoRA key.

    Args:
        lora_keys: iterable of base module paths or full state-dict keys
            (the function searches by substring, so suffixes don't matter).

    Returns:
        One of: 'bfl_original', 'diffusers_dit', 'sd_unet', 'unknown'.
    """
    keys = list(lora_keys)
    for family, patterns in _FORMAT_MARKERS:
        if any(any(p in k for p in patterns) for k in keys):
            return family
    return "unknown"


def detect_model_format(model_param_names: Iterable[str]) -> str:
    """Same heuristic, applied to a loaded nn.Module's parameter names."""
    return detect_lora_format(model_param_names)


# ════════════════════════════════════════════════════════════════════════
#  Rename plan
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RenameRule:
    """A single substring or regex substitution applied to a module path.

    Substring rules (regex=False, default) use `str.replace` and apply
    every occurrence.  Regex rules use `re.sub` with the standard
    Python regex dialect.
    """
    pattern: str
    replacement: str
    regex: bool = False

    def apply(self, key: str) -> str:
        if self.regex:
            return re.sub(self.pattern, self.replacement, key)
        return key.replace(self.pattern, self.replacement)


@dataclass(frozen=True)
class QKVSplitSpec:
    """Identifies a fused module that must be split into N outputs.

    `pattern` is a substring matched against base module paths AFTER
    rename_rules have been applied.  When the substring matches, the
    framework replaces it with each of `targets` in turn to produce
    the N output module paths.

    `target_dims` is an optional tuple of integers specifying the
    output-dimension size of each split.  When ``None``, the fused
    dimension is divided equally among targets (all targets get the
    same size).  When provided, ``len(target_dims)`` must equal
    ``len(targets)`` and ``sum(target_dims)`` must equal the fused
    dimension of the tensor being split.

    Examples:
        # Klein double-block 3-way equal QKV split:
        QKVSplitSpec(".attn.qkv",
                     targets=(".attn.to_q", ".attn.to_k", ".attn.to_v"))

        # Chroma single-block 4-way UNEQUAL split (Q+K+V+MLP_gate):
        QKVSplitSpec(".__LINEAR1__",
                     targets=(".attn.to_q", ".attn.to_k",
                              ".attn.to_v", ".proj_mlp"),
                     target_dims=(3072, 3072, 3072, 12288))
    """
    pattern: str
    targets: Tuple[str, ...]
    target_dims: Optional[Tuple[int, ...]] = None


@dataclass
class ConversionPlan:
    """Recipe for mapping a LoRA from source_family layout → target_family.

    Slices 3 (Flux/Klein/Flux2) and 5 (Chroma) populate these lists for
    their respective families and register the resulting plan in
    CONVERSION_PLANS via the (source_family, target_family) key.
    """
    source_family: str            # e.g. 'bfl_klein'
    target_family: str            # e.g. 'diffusers_klein'
    # Ordered substitutions on base module paths.  ALL apply, in order.
    rename_rules: List[RenameRule] = field(default_factory=list)
    # Fused-QKV modules that must be split into 3 separate outputs
    # AFTER rename.  Each entry binds a pattern → 3 target fragments.
    qkv_splits: List[QKVSplitSpec] = field(default_factory=list)
    # Substring that uniquely identifies the diffusers TARGET model.
    # Slice 4's plan-matching code searches for this in the loaded
    # transformer's parameter names; non-empty → required-match,
    # empty → match-anything (use sparingly — risks false positives).
    # Example: Klein-9B has `single_transformer_blocks.X.attn.
    # to_qkv_mlp_proj.weight` (a fused QKV+MLP layer that vanilla
    # Flux.1 diffusers does not have), so model_signature='to_qkv_mlp_proj'
    # cleanly distinguishes Klein/Flux2 from Flux.1.
    model_signature: str = ""
    # Optional notes for users / debugging.
    notes: str = ""


# ════════════════════════════════════════════════════════════════════════
#  Plan registry — POPULATED BY SLICES 3 (Flux2/Klein) AND 5 (Chroma)
# ════════════════════════════════════════════════════════════════════════

CONVERSION_PLANS: Dict[Tuple[str, str], ConversionPlan] = {}


def register_plan(plan: ConversionPlan) -> None:
    """Register a plan in the global registry.  Slices 3/5 call this."""
    CONVERSION_PLANS[(plan.source_family, plan.target_family)] = plan


def get_plan(source_family: str, target_family: str) -> Optional[ConversionPlan]:
    """Return the plan for (source_family → target_family) or None."""
    return CONVERSION_PLANS.get((source_family, target_family))


# ════════════════════════════════════════════════════════════════════════
#  Mechanical rename application
# ════════════════════════════════════════════════════════════════════════

# Adapter suffixes — duplicated from eric_diffusion_lora_check to keep
# this module standalone (centralising would create a dependency cycle
# for a tiny constant tuple).  Order: longer/more-specific suffixes
# first so prefix matches don't shadow them.
_ADAPTER_SUFFIXES: Tuple[str, ...] = (
    ".lora_A.weight", ".lora_B.weight",
    ".lora_down.weight", ".lora_up.weight",
    ".lora_A", ".lora_B",
    ".lora_down", ".lora_up",
    ".lokr_w1", ".lokr_w2", ".lokr_t2",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
    ".alpha", ".diff", ".diff_b",
)


def split_state_key(key: str) -> Tuple[str, str]:
    """Return (base_module_path, suffix) for a LoRA state-dict key.

    The suffix is one of _ADAPTER_SUFFIXES; if no known suffix is
    present, returns (key, '').
    """
    for sfx in _ADAPTER_SUFFIXES:
        idx = key.find(sfx)
        if idx >= 0:
            return key[:idx], key[idx:]
    return key, ""


def apply_rename_rules(
    state_dict: Dict[str, "torch.Tensor"],
    plan: ConversionPlan,
) -> Dict[str, "torch.Tensor"]:
    """Return a new state dict with every base module path renamed.

    Tensor data is unchanged — only labels are rewritten.  QKV
    splitting (which DOES change tensor shape and key count) is a
    separate step performed by split_fused_qkv_lora() / _via_svd().
    """
    out: Dict[str, "torch.Tensor"] = {}
    for k, v in state_dict.items():
        base, sfx = split_state_key(k)
        for rule in plan.rename_rules:
            base = rule.apply(base)
        out[base + sfx] = v
    return out


# ════════════════════════════════════════════════════════════════════════
#  QKV split helpers
# ════════════════════════════════════════════════════════════════════════

def split_fused_qkv_lora(
    lora_A: "torch.Tensor",            # shape: (rank, in_dim)
    lora_B: "torch.Tensor",            # shape: (3 * out_dim, rank)
    alpha: Optional["torch.Tensor"] = None,
) -> Dict[str, Dict[str, "torch.Tensor"]]:
    """Split a fused-QKV standard LoRA into three separate Q/K/V LoRAs.

    A fused-QKV layer represents `[Q; K; V] = W @ x` where `W` has
    shape `(3 * out_dim, in_dim)`.  Its LoRA delta is
    `(lora_B @ lora_A) * (alpha / rank)` of the same shape.  Slicing
    `lora_B` along the output dim into thirds gives three independent
    LoRAs that share `lora_A` and the alpha — no SVD or rank truncation
    required.  This transformation is mathematically EXACT.

    Returns:
        {"q": {"lora_A": ..., "lora_B": ..., "alpha": ...},
         "k": {...},
         "v": {...}}

    Raises:
        ValueError if `lora_B`'s output dim is not divisible by 3.
    """
    out_total = lora_B.shape[0]
    if out_total % 3 != 0:
        raise ValueError(
            f"Fused-QKV lora_B has out_dim={out_total}, not divisible by 3"
        )
    out_dim = out_total // 3
    parts: Dict[str, Dict[str, "torch.Tensor"]] = {}
    for name, lo, hi in (("q", 0, out_dim),
                         ("k", out_dim, 2 * out_dim),
                         ("v", 2 * out_dim, 3 * out_dim)):
        sub: Dict[str, "torch.Tensor"] = {
            "lora_A": lora_A.clone(),
            "lora_B": lora_B[lo:hi].clone(),
        }
        if alpha is not None:
            sub["alpha"] = alpha.clone() if hasattr(alpha, "clone") else alpha
        parts[name] = sub
    return parts


def split_fused_qkv_via_svd(
    merged_delta: "torch.Tensor",      # shape: (3 * out_dim, in_dim)
    target_rank: int,
) -> Dict[str, Dict[str, "torch.Tensor"]]:
    """Split an already-merged fused-QKV delta into three rank-r LoRAs via SVD.

    For LoKR / LoHa adapters whose underlying form is not a clean
    `(A, B)` factorisation, the conversion path is:

      1. compute the merged delta — for LoKR that is
         `(w1 ⊗ w2) * (alpha / dim_w1)`; for LoHa similar.
      2. slice the `(3 * out_dim, in_dim)` delta into 3 `(out_dim, in_dim)` blocks
      3. SVD-truncate each block to target_rank → emit `(B, A)` pair

    SVD truncation is lossy by definition (the LoKR/LoHa's full-rank
    delta can have rank up to `min(out, in)`, but the LoRA we emit has
    rank `target_rank`).  Choose `target_rank` to match the original
    adapter's effective rank — for LoKR this is typically the smaller
    of the two factor dimensions.

    The emitted alpha equals `target_rank` so that diffusers' standard
    `(alpha / rank)` scaling factor at runtime is exactly 1.0; the
    magnitude is fully baked into the (B, A) factors via a sqrt(S) split.

    Returns the same structure as split_fused_qkv_lora().
    """
    out_total = merged_delta.shape[0]
    if out_total % 3 != 0:
        raise ValueError(
            f"Merged QKV delta has out_dim={out_total}, not divisible by 3"
        )
    out_dim = out_total // 3
    parts: Dict[str, Dict[str, "torch.Tensor"]] = {}
    for name, lo, hi in (("q", 0, out_dim),
                         ("k", out_dim, 2 * out_dim),
                         ("v", 2 * out_dim, 3 * out_dim)):
        block = merged_delta[lo:hi].float()
        full_rank = min(block.shape[-2], block.shape[-1])
        r = min(target_rank, full_rank)

        # Truncated (randomised) SVD when target_rank << full_rank.
        # Full SVD on Klein-9B's biggest fused QKV slice is minutes on
        # CPU; svd_lowrank is ~100× faster for typical target_rank=16.
        if r * 4 <= full_rank and r >= 1:
            q = min(r + 8, full_rank)
            U, S, V = torch.svd_lowrank(block, q=q)
            U  = U[:, :r]
            S  = S[:r]
            Vh = V[:, :r].t()
        else:
            U, S, Vh = torch.linalg.svd(block, full_matrices=False)
            U  = U[:, :r]
            S  = S[:r]
            Vh = Vh[:r]

        # Distribute singular values as sqrt(S) across both factors so
        # neither side blows up numerically and the (alpha/rank) scaling
        # at inference comes out to exactly 1.0 when alpha = rank.
        # .contiguous() — svd_lowrank's V.t() is a non-contiguous view
        # that safetensors won't serialise.
        S_root = S.sqrt()
        lora_B = (U * S_root).contiguous().to(merged_delta.dtype)
        lora_A = (Vh * S_root.unsqueeze(1)).contiguous().to(merged_delta.dtype)
        parts[name] = {
            "lora_A": lora_A,
            "lora_B": lora_B,
            "alpha":  torch.tensor(float(r)),
        }
    return parts


# ════════════════════════════════════════════════════════════════════════
#  Kohya underscore → BFL dot-format decoder
# ════════════════════════════════════════════════════════════════════════

# BFL-original module paths use underscores within component names
# (double_blocks, img_attn, etc.) but dots between them.  Kohya
# trainers encode such paths by replacing dots with underscores and
# prepending `lora_unet_`.  To reverse this we protect known compound
# names (whose internal underscores are NOT dots) before converting
# the remaining underscores back to dots.
#
# Sorted longest-first so overlapping patterns don't shadow each other
# (e.g. "query_norm" before "norm").
_KOHYA_COMPOUNDS: Tuple[str, ...] = (
    "single_transformer_blocks",
    "transformer_blocks",
    "double_blocks",
    "single_blocks",
    "query_norm",
    "key_norm",
    "img_attn",
    "txt_attn",
    "img_mlp",
    "txt_mlp",
    "img_mod",
    "txt_mod",
    "q_norm",
    "k_norm",
)

_SENTINEL = "\x00"


def decode_kohya_to_bfl(
    state_dict: Dict[str, "torch.Tensor"],
) -> Dict[str, "torch.Tensor"]:
    """Decode ``lora_unet_*`` Kohya keys to BFL dot-separated format.

    Converts keys like::

        lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight

    to::

        diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight

    Keys that don't start with ``lora_unet_`` pass through unchanged.
    Returns a new dict (never mutates the input).

    This is a LIGHTER decode than ``_decode_kohya_keys`` in the loader
    module, which calls diffusers' full Flux Kohya converter and
    produces diffusers-format output.  Here we only recover the BFL
    dot-separated form so that ``detect_lora_format`` / ``find_matching_plan``
    can recognise the LoRA's source family.
    """
    _PREFIX = "lora_unet_"
    if not any(k.startswith(_PREFIX) for k in state_dict):
        return state_dict

    out: Dict[str, "torch.Tensor"] = {}
    for key, val in state_dict.items():
        if not key.startswith(_PREFIX):
            out[key] = val
            continue

        remainder = key[len(_PREFIX):]

        # Adapter suffix boundary: first dot after the encoded module path.
        # Kohya preserves dots for adapter suffixes (.lora_down.weight, .alpha, etc.)
        dot_idx = remainder.find(".")
        if dot_idx < 0:
            # No suffix — unusual but pass through with prefix stripped
            out[f"diffusion_model.{remainder}"] = val
            continue

        encoded_path = remainder[:dot_idx]
        suffix = remainder[dot_idx:]  # includes leading dot

        # Protect compound names whose internal underscores are NOT dots
        decoded = encoded_path
        for compound in _KOHYA_COMPOUNDS:
            decoded = decoded.replace(compound, compound.replace("_", _SENTINEL))

        # Remaining underscores → dots
        decoded = decoded.replace("_", ".")
        decoded = decoded.replace(_SENTINEL, "_")

        out[f"diffusion_model.{decoded}{suffix}"] = val

    return out


# ════════════════════════════════════════════════════════════════════════
#  Auto-register family plans
# ════════════════════════════════════════════════════════════════════════
#
# Importing the framework module registers every family plan as a
# side effect.  This MUST sit at the bottom of the module — the family
# files import `register_plan` from here, so they need this module to
# be fully defined before they execute.
from . import eric_lora_format_convert_flux    # noqa: E402, F401  (registers Klein/Flux2)
from . import eric_lora_format_convert_chroma  # noqa: E402, F401  (registers Chroma)
