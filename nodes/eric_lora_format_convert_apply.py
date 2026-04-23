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
    QKVSplitSpec,
    apply_rename_rules,
    detect_lora_format,
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
    *,
    w1_is_decomposed: bool = False,
    w2_is_decomposed: bool = False,
    w1_b_dim: Optional[int] = None,
    w2_b_dim: Optional[int] = None,
) -> "torch.Tensor":
    """Reconstruct the merged delta for a LoKR module.

    Scaling convention matches ComfyUI's
    `comfy/weight_adapter/lokr.py::LoKrAdapter.calculate_weight`:

      - When EITHER w1 OR w2 is provided in DECOMPOSED form
        (lokr_w1_a/lokr_w1_b or lokr_w2_a/lokr_w2_b), the stored alpha
        IS applied as `scale = alpha / dim` where dim is the inner rank
        of the corresponding decomposition (w1_b.shape[0] or
        w2_b.shape[0]).
      - When BOTH w1 and w2 are stored directly, the stored alpha is
        IGNORED entirely and `scale = 1.0`.  This matches ComfyUI's
        behaviour and is critical for LoRAs trained by ai-toolkit
        whose stored alpha is a ~1e10 sentinel value rather than a
        meaningful scale (e.g. klein_snofs, Realism_Engine_Klein_V2).

    Why this matters: applying `scale = alpha / min(w1.shape) = 2.5e9`
    on top of base weights of magnitude ~1 produces a delta that
    overwhelms the model and yields pure noise.  ComfyUI silently
    avoids this by never reading the stored alpha for direct LoKRs.

    Caller is responsible for passing the decomposition flags when the
    source state dict had `_a`/`_b` factor keys present.  For Klein/
    Flux2 LoRAs in this codebase, w1 and w2 are always directly stored
    (no `_a`/`_b`), so the default flags produce the right behaviour.

    Returns a float32 tensor.  Caller is responsible for casting back
    to the LoRA's working dtype.
    """
    if w1_is_decomposed and w1_b_dim is not None:
        dim = w1_b_dim
    elif w2_is_decomposed and w2_b_dim is not None:
        dim = w2_b_dim
    else:
        dim = None

    if alpha is not None and dim is not None and dim > 0:
        scale = alpha.item() / dim
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
# Note: we deliberately do NOT emit ".alpha" keys.  Diffusers' fast-path
# pipe.load_lora_weights enforces a "every key must contain 'lora'"
# substring check that .alpha keys fail; emitting them was making the
# fast path silently reject our converted state dicts.  Instead, we
# bake the alpha/rank scaling into the lora_A/lora_B factors, and
# downstream PEFT consumers see a state dict where the default scale
# of 1.0 produces exactly the intended delta.
_OUT_LORA_A = ".lora_A.weight"
_OUT_LORA_B = ".lora_B.weight"


def _resolve_split_dims(
    spec: QKVSplitSpec,
    out_total: int,
    base: str,
    kind: str,
) -> Tuple[int, ...]:
    """Compute per-target output dimensions for a fused split.

    When ``spec.target_dims`` is provided, validates that the sum equals
    ``out_total``.  When absent, divides equally among targets (validates
    divisibility).  Returns a tuple of ``len(spec.targets)`` ints.
    """
    n = len(spec.targets)
    if spec.target_dims is not None:
        if len(spec.target_dims) != n:
            raise ValueError(
                f"{kind} split at {base}: target_dims has "
                f"{len(spec.target_dims)} entries but targets has {n}"
            )
        if sum(spec.target_dims) != out_total:
            raise ValueError(
                f"{kind} split at {base}: sum(target_dims)="
                f"{sum(spec.target_dims)} != out_dim={out_total}"
            )
        return spec.target_dims
    if out_total % n != 0:
        raise ValueError(
            f"{kind} split at {base}: out_dim={out_total} not "
            f"divisible by {n} targets"
        )
    d = out_total // n
    return tuple(d for _ in range(n))


def _group_by_module(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, Dict[str, "torch.Tensor"]]:
    """Group {full_key: tensor} by base module path → {suffix: tensor}."""
    modules: Dict[str, Dict[str, "torch.Tensor"]] = {}
    for k, v in state_dict.items():
        base, sfx = split_state_key(k)
        modules.setdefault(base, {})[sfx] = v
    return modules


def _bake_alpha(
    lora_A: "torch.Tensor",
    lora_B: "torch.Tensor",
    alpha: Optional["torch.Tensor"],
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Distribute the alpha/rank scaling factor evenly into A and B.

    Standard PEFT runtime scaling is (alpha/rank) — when alpha == rank
    (or no alpha is recorded), the scale is 1.0 and the raw `B @ A`
    product becomes the delta.  For LoRAs whose stored alpha differs
    from rank, we multiply both A and B by sqrt(alpha/rank) so that
    `(B' @ A') = (alpha/rank) * (B @ A)` — i.e. the trained scaling
    is baked in, and consumers can rely on a default scale of 1.0.

    Returns (A, B) unchanged when alpha is None or matches rank.
    """
    if alpha is None or lora_A.ndim < 2:
        return lora_A, lora_B
    rank = lora_A.shape[0]
    if rank == 0:
        return lora_A, lora_B
    alpha_val = float(alpha.item()) if hasattr(alpha, "item") else float(alpha)
    scale = alpha_val / rank
    if abs(scale - 1.0) < 1e-6:
        return lora_A, lora_B
    s_root = scale ** 0.5
    return (
        (lora_A * s_root).contiguous(),
        (lora_B * s_root).contiguous(),
    )


def _emit_lora_module(
    out_state_dict: Dict[str, "torch.Tensor"],
    base: str,
    lora_A: "torch.Tensor",
    lora_B: "torch.Tensor",
    alpha: Optional["torch.Tensor"] = None,
) -> None:
    """Add canonical .lora_A.weight / .lora_B.weight entries.

    Bakes alpha into the factors so we can omit the .alpha key
    entirely (see _OUT_LORA_A / _OUT_LORA_B comment for why).
    """
    A, B = _bake_alpha(lora_A, lora_B, alpha)
    out_state_dict[base + _OUT_LORA_A] = A
    out_state_dict[base + _OUT_LORA_B] = B


def convert_state_dict(
    source_state_dict: Dict[str, "torch.Tensor"],
    plan: ConversionPlan,
    *,
    target_rank: int = 64,
    log_prefix: str = "[LoRA-Convert]",
) -> Dict[str, "torch.Tensor"]:
    """Apply a ConversionPlan to a state dict.

    Output format depends on the SOURCE adapter type per module:

      Standard LoRA (lora_A / lora_B input)
        → emits canonical standard-LoRA keys at the rename-target paths:
          `<base>.lora_A.weight` and `<base>.lora_B.weight`.
          Alpha is baked into the factors via sqrt(alpha/rank) distribution
          — no separate .alpha keys are emitted (diffusers' fast-path
          loader rejects state dicts with non-"lora" keys).
          QKV-split is exact (no information loss).

      LoKR (lokr_w1 / lokr_w2 input)
        → emits a single `<base>.diff` per module containing the
          fully-reconstructed delta `kron(w1, w2) * (alpha/r)`.
          QKV-split slices the delta into 3 (out, in) blocks per
          module, each emitted as a `.diff` at the corresponding
          target path.  Lossless — no SVD truncation.  Trade-off: the
          downstream loader applies these via direct weight merge
          (no PEFT-managed runtime weight scaling), matching the
          existing _load_lokr_adapter_direct semantics.

    Why .diff instead of SVD-truncated lora_A/lora_B for LoKRs:
    Klein-9B LoKRs commonly have w1=(4,4) and large w2 (e.g.
    (9216, 1024) for single_blocks.linear1).  The merged delta has
    theoretical rank up to 4 * min(out_w2, in_w2) — typically thousands.
    SVD-truncating to rank 64 keeps ~1.5% of singular components and
    produces visibly noisy outputs on real LoRAs (e.g. klein_snofs).
    Direct .diff merge keeps the full content with no rank tradeoff;
    memory footprint is `out * in` per module instead of
    `(out + in) * rank` but stays well within VRAM for diffusion-model
    LoRAs.  Standard LoRAs continue to use the lora_A/lora_B path —
    they're already low-rank by construction.

    target_rank parameter is reserved for future SVD-based paths
    (e.g. LoHa support) and currently unused by LoKR conversion.
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
        # Emit .diff (full merged delta) — see convert_state_dict
        # docstring for the "why direct merge instead of SVD" rationale.
        if ".lokr_w1" in parts and ".lokr_w2" in parts:
            delta = reconstruct_lokr_delta(
                parts[".lokr_w1"], parts[".lokr_w2"], parts.get(".alpha"),
            )
            if qkv_spec:
                out_total = delta.shape[0]
                dims = _resolve_split_dims(qkv_spec, out_total, base, "LoKR")
                offset = 0
                for tgt, d in zip(qkv_spec.targets, dims):
                    out_base = base.replace(qkv_spec.pattern, tgt)
                    out[out_base + ".diff"] = delta[offset:offset + d].contiguous()
                    offset += d
                n_lokr_qkv += 1
            else:
                out[base + ".diff"] = delta.contiguous()
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
                dims = _resolve_split_dims(qkv_spec, B.shape[0], base, "LoRA")
                offset = 0
                for tgt, d in zip(qkv_spec.targets, dims):
                    out_base = base.replace(qkv_spec.pattern, tgt)
                    _emit_lora_module(
                        out, out_base,
                        A.clone(), B[offset:offset + d].clone(), alpha,
                    )
                    offset += d
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
                dims = _resolve_split_dims(qkv_spec, B.shape[0], base, "LoRA")
                offset = 0
                for tgt, d in zip(qkv_spec.targets, dims):
                    out_base = base.replace(qkv_spec.pattern, tgt)
                    _emit_lora_module(
                        out, out_base,
                        A.clone(), B[offset:offset + d].clone(), alpha,
                    )
                    offset += d
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


# ════════════════════════════════════════════════════════════════════════
#  Loader: apply a converted state dict to a pipeline
# ════════════════════════════════════════════════════════════════════════

def _apply_converted_lora_as_delta(
    transformer,
    state_dict: Dict[str, "torch.Tensor"],
    adapter_name: str,
    weight: float,
    log_prefix: str,
) -> bool:
    """Direct weight merge — bypass PEFT entirely.

    Handles both convert_state_dict output formats:
      `.diff`                     → delta = state_dict[".diff"] * weight
      `.lora_A.weight` + `.lora_B.weight` → delta = (B @ A) * weight

    For each module, computes delta and adds it to the corresponding
    model parameter at `<path>.weight`.  Alpha is already baked into
    standard-LoRA factors and into LoKR .diff at convert time, so the
    runtime scale factor is simply `weight`.

    Used as the reliable fallback when pipe.load_lora_weights and PEFT
    injection don't actually register the adapter, AND as the primary
    path for any state dict containing .diff entries (since diffusers'
    LoRA loader doesn't understand .diff).  Trades off per-stage
    runtime weight changes for guaranteed correctness — same compromise
    the existing _load_lokr_adapter_direct path makes.
    """
    model_sd = dict(transformer.named_parameters())
    modules: Dict[str, Dict[str, "torch.Tensor"]] = {}
    for k, v in state_dict.items():
        base, sfx = split_state_key(k)
        modules.setdefault(base, {})[sfx] = v

    backup_attr = f"_converted_lora_backup_{adapter_name}"
    backup = getattr(transformer, backup_attr, None)
    if backup is None:
        backup = {}
        setattr(transformer, backup_attr, backup)

    applied_diff = applied_lora = skipped = 0
    for base, parts in modules.items():
        target_key = base + ".weight"
        if target_key not in model_sd:
            target_key = base  # rare: parameter without .weight suffix
        if target_key not in model_sd:
            skipped += 1
            continue
        param = model_sd[target_key]

        # Compute delta — prefer .diff (LoKR direct path) over B@A
        if ".diff" in parts:
            delta = parts[".diff"].float() * float(weight)
            applied_kind = "diff"
        else:
            # Standard-LoRA path: explicit None-test (avoid `tensor or other`
            # which raises ambiguous-boolean for multi-element tensors).
            A = parts[".lora_A.weight"] if ".lora_A.weight" in parts \
                else parts.get(".lora_A")
            B = parts[".lora_B.weight"] if ".lora_B.weight" in parts \
                else parts.get(".lora_B")
            if A is None or B is None:
                skipped += 1
                continue
            delta = (B.float() @ A.float()) * float(weight)
            applied_kind = "lora"

        if delta.shape != param.shape:
            try:
                delta = delta.reshape(param.shape)
            except RuntimeError:
                skipped += 1
                continue

        # Snapshot original weights so unload_adapters can restore them
        if target_key not in backup:
            backup[target_key] = param.data.clone()

        param.data.add_(delta.to(dtype=param.dtype, device=param.device))
        if applied_kind == "diff":
            applied_diff += 1
        else:
            applied_lora += 1

    # Mark the adapter so set_adapters() can find it (matches how
    # _load_lokr_adapter_direct registers its direct-merge adapters).
    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    transformer.peft_config[adapter_name] = {
        "_type": "converted_lora_direct",
        "_applied_modules": applied_diff + applied_lora,
        "_weight": weight,
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(
        f"{log_prefix} direct delta merge: "
        f"diff={applied_diff}, lora={applied_lora}, skipped={skipped} "
        f"(weight={weight})"
    )
    return (applied_diff + applied_lora) > 0


def load_converted_lora(
    pipe,
    converted_state_dict: Dict[str, "torch.Tensor"],
    adapter_name: str,
    log_prefix: str,
    weight: float = 1.0,
) -> bool:
    """Load an already-converted LoRA state dict onto the pipeline.

    Tries two paths in order:

      1. pipe.load_lora_weights with `transformer.` prefix prepended.
         Diffusers' Flux/Flux2/Qwen pipelines filter LoRA keys by this
         prefix when matching against the transformer module — without
         it, the call returns silently with zero modules registered.
         We verify post-load via get_list_adapters() because the
         silent-no-op path doesn't raise.

      2. Direct weight merge (_apply_converted_lora_as_delta).  Bypasses
         PEFT machinery entirely; same trade-off as the existing
         _load_lokr_adapter_direct (no per-stage runtime weight change,
         but reliably registers and applies).

    Returns True if at least one path succeeded.
    """
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        print(f"{log_prefix} pipe has no .transformer attribute")
        return False

    # If the converted state dict contains any .diff entries (LoKR
    # direct path), skip the pipeline-load attempt — diffusers' LoRA
    # loader doesn't recognize .diff and would either error or silently
    # drop them.  Direct merge handles both .diff and .lora_A/.lora_B
    # in one pass.
    if any(k.endswith(".diff") for k in converted_state_dict):
        print(
            f"{log_prefix} converted state dict contains .diff entries "
            f"(LoKR) — using direct delta merge"
        )
        return _apply_converted_lora_as_delta(
            transformer, converted_state_dict,
            adapter_name, weight, log_prefix,
        )

    # ── Drop keys whose base module has no .weight parameter ─────────
    # Kohya LoRAs sometimes include keys for modules that exist in the BFL
    # source layout but have no counterpart in the diffusers target (e.g.
    # distilled_guidance_layer in Klein-9B).  These pass through
    # convert_state_dict unchanged and would produce "unexpected keys"
    # warnings from pipe.load_lora_weights.  Filter them here where we
    # have the transformer and can check directly.
    param_names = {n for n, _ in transformer.named_parameters()}
    valid_converted: Dict[str, "torch.Tensor"] = {}
    dropped_bases: set = set()
    for k, v in converted_state_dict.items():
        base, _ = split_state_key(k)
        if base + ".weight" in param_names or base + ".bias" in param_names:
            valid_converted[k] = v
        else:
            dropped_bases.add(base)
    if dropped_bases:
        sample = ", ".join(sorted(dropped_bases)[:5])
        tail = "..." if len(dropped_bases) > 5 else ""
        print(
            f"{log_prefix} skipping {len(dropped_bases)} unresolvable "
            f"module path(s): {sample}{tail}"
        )
    if not valid_converted:
        print(
            f"{log_prefix} no valid LoRA keys after filtering — "
            "LoRA may target a different model family"
        )
        return False

    # ── Attempt 1: pipeline path with `transformer.` prefix ───────────
    prefixed = {f"transformer.{k}": v for k, v in valid_converted.items()}
    try:
        pipe.load_lora_weights(prefixed, adapter_name=adapter_name)
        # Verify registration — diffusers can silently no-op when no
        # keys match the expected module structure.
        try:
            adapter_lists = pipe.get_list_adapters()
            present = any(
                adapter_name in v for v in adapter_lists.values()
            )
        except Exception:
            present = True  # can't verify; assume success
        if present:
            print(
                f"{log_prefix} converted adapter registered via "
                f"pipe.load_lora_weights (PEFT-managed)"
            )
            return True
        print(
            f"{log_prefix} pipe.load_lora_weights returned without "
            f"registering '{adapter_name}' — falling back to direct merge"
        )
    except (ValueError, RuntimeError) as e:
        print(
            f"{log_prefix} pipe.load_lora_weights raised "
            f"({str(e)[:120]}) — falling back to direct merge"
        )

    # ── Attempt 2: direct weight merge ────────────────────────────────
    return _apply_converted_lora_as_delta(
        transformer, valid_converted, adapter_name, weight, log_prefix,
    )
