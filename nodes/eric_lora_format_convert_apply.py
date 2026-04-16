# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
LoRA format conversion — apply step (slice 4).

Bridges slice 2 (framework) + slice 3 (Klein/Flux2 plan) to the
existing standard-LoRA loader path in eric_qwen_edit_lora.py.

What this module does:

  1. find_matching_plan(state_dict, model_param_names) — heuristic
     plan lookup.  Walks CONVERSION_PLANS, returns the first plan
     whose source_family family root matches the LoRA's detected
     format AND whose model_signature substring is present in the
     loaded model's parameter names.  Returns None if nothing fits.

  2. convert_state_dict(state_dict, plan) — produces a new state dict
     in standard-LoRA format (`<path>.lora_A.weight` / `.lora_B.weight`
     / `.alpha`) at the rename-target paths.  The output is suitable
     for `pipe.load_lora_weights(state_dict)` or for our existing
     `_load_lora_adapter` 3-tier fallback.

     Per-module behaviour:

       LoKR  → reconstruct merged delta `kron(w1,w2) * (alpha/r)`,
               then either:
                 - if QKV-split: SVD-truncate each Q/K/V slice to
                   target_rank (lossy by definition);
                 - else: SVD-truncate the whole (out, in) delta to
                   target_rank.

       LoRA  → no reconstruction needed; if QKV-split, slice lora_B
               along the output dim (exact, lossless — lora_A is
               shared).  Else, pass through unchanged at the renamed
               key.

       LoHa  → not yet implemented.  Modules are skipped with a
               warning; standard-LoRA paths still work.

The integration point in `load_lora_with_key_fix` calls
find_matching_plan first; if a plan is found, it converts and routes
the result through `_load_lora_adapter`.  No conversion attempt is
made when no plan matches — the existing loader paths handle those
cases unchanged.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch

from .eric_lora_format_convert import (
    CONVERSION_PLANS,
    ConversionPlan,
    apply_rename_rules,
    detect_lora_format,
    split_fused_qkv_lora,
    split_fused_qkv_via_svd,
    split_state_key,
)


# ════════════════════════════════════════════════════════════════════════
#  Plan matching
# ════════════════════════════════════════════════════════════════════════

def _coarse_family(name: str) -> str:
    """'bfl_klein' → 'bfl';  'diffusers_dit' → 'diffusers';  'unknown' → ''."""
    if not name or name == "unknown":
        return ""
    return name.split("_", 1)[0]


def find_matching_plan(
    lora_state_dict: Dict[str, "torch.Tensor"],
    model_param_names: Iterable[str],
) -> Optional[ConversionPlan]:
    """Return the registered plan that fits the (LoRA, model) pair, or None.

    Selection heuristic:
      - Coarse family of the LoRA (via detect_lora_format) must equal
        the coarse family of plan.source_family.  E.g. a LoRA detected
        as 'bfl_original' matches any plan whose source_family begins
        with 'bfl_'.
      - If plan.model_signature is non-empty, it MUST appear as a
        substring in some model parameter name.  Empty signature means
        match-anything (use sparingly — risk of false positives).

    First-match wins.  Order is determined by Python dict iteration
    order over CONVERSION_PLANS, which is insertion order.
    """
    lora_family = detect_lora_format(lora_state_dict.keys())
    coarse_lora = _coarse_family(lora_family)
    if not coarse_lora:
        return None

    model_param_names = list(model_param_names)

    for plan in CONVERSION_PLANS.values():
        coarse_plan = _coarse_family(plan.source_family)
        if coarse_plan != coarse_lora:
            continue
        if plan.model_signature:
            if not any(plan.model_signature in n for n in model_param_names):
                continue
        return plan
    return None


# ════════════════════════════════════════════════════════════════════════
#  Per-module reconstruction helpers
# ════════════════════════════════════════════════════════════════════════

def reconstruct_lokr_delta(
    w1: "torch.Tensor",
    w2: "torch.Tensor",
    alpha: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Reconstruct the merged delta for a LoKR module.

    delta = kron(w1, w2) * (alpha / r), where r = min(w1.shape).
    When alpha is absent, scale defaults to 1.0 (LyCORIS convention —
    weights are assumed pre-scaled).

    Returns a float32 tensor.  Caller is responsible for casting back
    to the LoRA's working dtype.
    """
    r = min(w1.shape) if w1.ndim >= 2 else 1
    if alpha is not None:
        scale = (alpha.item() / r) if r > 0 else 1.0
    else:
        scale = 1.0
    return torch.kron(w1.float(), w2.float()) * scale


def svd_compress_to_lora(
    delta: "torch.Tensor",
    target_rank: int,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """SVD-truncate a (out, in) delta to standard LoRA factors.

    Returns (lora_A, lora_B, alpha) where:
        lora_A has shape (rank, in_dim)
        lora_B has shape (out_dim, rank)
        alpha == rank (so the runtime alpha/rank scale factor is 1.0;
            the magnitude is fully baked into B and A via sqrt(S) split)

    Uses `torch.svd_lowrank` (randomised truncated SVD) when target_rank
    is much smaller than min(out, in), which is the common case for
    LoKR/LoHa → LoRA conversion.  Full SVD on Klein-9B's largest fused
    matrix (36864, 4096) is on the order of minutes on CPU; the
    truncated path is ~100× faster for typical target_rank=16.

    Tensor dtype matches the input dtype (SVD itself runs in float32
    for numerical stability).
    """
    out_dim, in_dim = delta.shape[-2], delta.shape[-1]
    full_rank = min(out_dim, in_dim)
    r = min(target_rank, full_rank)

    delta_f = delta.float()

    # Choose SVD path based on rank ratio:
    #   - target_rank close to full_rank → full SVD is fine and exact
    #   - target_rank << full_rank → randomised truncated SVD is faster
    # Threshold of 4× empirically: above that, lowrank gives near-
    # identical results in O(out·in·r) instead of O(min(out,in)²·max).
    use_lowrank = r * 4 <= full_rank and r >= 1

    if use_lowrank:
        # svd_lowrank(A, q) returns (U, S, V) where A ≈ U @ diag(S) @ V.T.
        # Use a small oversampling (q = r + 8) for numerical headroom;
        # cap at full_rank.
        q = min(r + 8, full_rank)
        U, S, V = torch.svd_lowrank(delta_f, q=q)
        # Take only the top-r singular components
        U = U[:, :r]
        S = S[:r]
        # svd_lowrank returns V (not Vh).  Convert to Vh = V.T for
        # consistency with the standard SVD convention.
        Vh = V[:, :r].t()
    else:
        U, S, Vh = torch.linalg.svd(delta_f, full_matrices=False)
        U  = U[:, :r]
        S  = S[:r]
        Vh = Vh[:r]

    s_root = S.sqrt()
    # .contiguous() — SVD lowrank's V.t() produces a non-contiguous view
    # which safetensors refuses to serialise.  Cheap to make contiguous
    # at this size; downstream loaders also expect contiguous storage.
    lora_B = (U * s_root).contiguous().to(delta.dtype)
    lora_A = (Vh * s_root.unsqueeze(1)).contiguous().to(delta.dtype)
    alpha = torch.tensor(float(r))
    return lora_A, lora_B, alpha


# ════════════════════════════════════════════════════════════════════════
#  State-dict conversion
# ════════════════════════════════════════════════════════════════════════

# Standard-LoRA suffixes we emit (canonical: the ".weight" form).
_OUT_LORA_A = ".lora_A.weight"
_OUT_LORA_B = ".lora_B.weight"
_OUT_ALPHA  = ".alpha"


def _group_by_module(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, Dict[str, "torch.Tensor"]]:
    """Group {full_key: tensor} by base module path → {suffix: tensor}."""
    modules: Dict[str, Dict[str, "torch.Tensor"]] = {}
    for k, v in state_dict.items():
        base, sfx = split_state_key(k)
        modules.setdefault(base, {})[sfx] = v
    return modules


def _emit_lora_module(
    out_state_dict: Dict[str, "torch.Tensor"],
    base: str,
    lora_A: "torch.Tensor",
    lora_B: "torch.Tensor",
    alpha: Optional["torch.Tensor"] = None,
) -> None:
    """Add canonical .lora_A.weight / .lora_B.weight / .alpha entries."""
    out_state_dict[base + _OUT_LORA_A] = lora_A
    out_state_dict[base + _OUT_LORA_B] = lora_B
    if alpha is not None:
        out_state_dict[base + _OUT_ALPHA] = alpha


def convert_state_dict(
    source_state_dict: Dict[str, "torch.Tensor"],
    plan: ConversionPlan,
    *,
    target_rank: int = 16,
    log_prefix: str = "[LoRA-Convert]",
) -> Dict[str, "torch.Tensor"]:
    """Apply a ConversionPlan to a state dict.

    Returns a state dict containing ONLY standard-LoRA keys
    (.lora_A.weight / .lora_B.weight / .alpha) at the plan's
    rename-target paths.

    target_rank governs the SVD-truncation rank for LoKR/LoHa modules.
    Higher rank = better fidelity but larger output.  16 is a reasonable
    default for typical LoRA usage; bump to 32–64 if reconstruction
    error matters.  Standard LoRAs are not SVD-touched and keep their
    original rank.
    """
    # Phase 1: rename keys (preserves all suffixes)
    renamed = apply_rename_rules(source_state_dict, plan)
    modules = _group_by_module(renamed)

    out: Dict[str, "torch.Tensor"] = {}
    n_lokr_qkv = n_lokr_single = n_lora_qkv = n_lora_pass = n_skipped = 0
    skipped_samples: List[str] = []

    for base, parts in modules.items():
        # Find matching QKV-split spec, if any
        qkv_spec = next(
            (s for s in plan.qkv_splits if s.pattern in base),
            None,
        )

        # ── LoKR module ──────────────────────────────────────────────
        if ".lokr_w1" in parts and ".lokr_w2" in parts:
            delta = reconstruct_lokr_delta(
                parts[".lokr_w1"], parts[".lokr_w2"], parts.get(".alpha"),
            )
            if qkv_spec:
                splits = split_fused_qkv_via_svd(delta, target_rank)
                for name, tgt in zip(("q", "k", "v"), qkv_spec.targets):
                    out_base = base.replace(qkv_spec.pattern, tgt)
                    _emit_lora_module(
                        out, out_base,
                        splits[name]["lora_A"], splits[name]["lora_B"],
                        splits[name]["alpha"],
                    )
                n_lokr_qkv += 1
            else:
                A, B, alpha = svd_compress_to_lora(delta, target_rank)
                _emit_lora_module(out, base, A, B, alpha)
                n_lokr_single += 1
            continue

        # ── Standard LoRA module ─────────────────────────────────────
        # Accept both .weight and bare suffix conventions
        a_key = ".lora_A.weight" if ".lora_A.weight" in parts else ".lora_A"
        b_key = ".lora_B.weight" if ".lora_B.weight" in parts else ".lora_B"
        if a_key in parts and b_key in parts:
            A = parts[a_key]
            B = parts[b_key]
            alpha = parts.get(".alpha")
            if qkv_spec:
                splits = split_fused_qkv_lora(A, B, alpha=alpha)
                for name, tgt in zip(("q", "k", "v"), qkv_spec.targets):
                    out_base = base.replace(qkv_spec.pattern, tgt)
                    _emit_lora_module(
                        out, out_base,
                        splits[name]["lora_A"], splits[name]["lora_B"],
                        splits[name].get("alpha"),
                    )
                n_lora_qkv += 1
            else:
                _emit_lora_module(out, base, A, B, alpha)
                n_lora_pass += 1
            continue

        # ── lora_down/lora_up convention (older format) ──────────────
        # lora_down ↔ lora_A;  lora_up ↔ lora_B (same math).
        d_key = ".lora_down.weight" if ".lora_down.weight" in parts else ".lora_down"
        u_key = ".lora_up.weight" if ".lora_up.weight" in parts else ".lora_up"
        if d_key in parts and u_key in parts:
            A = parts[d_key]
            B = parts[u_key]
            alpha = parts.get(".alpha")
            if qkv_spec:
                splits = split_fused_qkv_lora(A, B, alpha=alpha)
                for name, tgt in zip(("q", "k", "v"), qkv_spec.targets):
                    out_base = base.replace(qkv_spec.pattern, tgt)
                    _emit_lora_module(
                        out, out_base,
                        splits[name]["lora_A"], splits[name]["lora_B"],
                        splits[name].get("alpha"),
                    )
                n_lora_qkv += 1
            else:
                _emit_lora_module(out, base, A, B, alpha)
                n_lora_pass += 1
            continue

        # ── Unsupported (LoHa, etc.) ─────────────────────────────────
        n_skipped += 1
        if len(skipped_samples) < 3:
            skipped_samples.append(base)

    print(
        f"{log_prefix} converted: "
        f"LoKR(qkv={n_lokr_qkv}, single={n_lokr_single}), "
        f"LoRA(qkv={n_lora_qkv}, pass={n_lora_pass}), "
        f"skipped={n_skipped}"
    )
    if n_skipped:
        print(
            f"{log_prefix} (skipped sample bases: {skipped_samples} — "
            "LoHa or unsupported adapter format; standard-LoRA paths "
            "still apply for these modules.)"
        )
    return out
