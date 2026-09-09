"""LoRA / LoKR / LoHa adapter loading for comfyless.

comfyless's single LoRA entry point is :func:`load_lora_with_key_fix`; the
daemon additionally consults :func:`is_direct_merge_adapter` before calling
``set_adapters``. Everything else here is the machinery behind those two.

Written for ADR-046 (ADR-045 slice 3c) as the comfyless-owned replacement for
the node pack's ``eric_qwen_edit_lora`` module. The behavioural contract it is
held to -- key normalisation, adapter detection, PEFT injection configs, the
direct-merge numerics and registry, the orchestration order -- is written down
in that ADR, and ``test_lora_adapters.py`` proves it against the node-pack
original side by side. Where the original's behaviour is an accident of
implementation (the suffix table is scanned in TABLE order, not key order;
``set_adapters`` cannot re-weight a direct merge) it is preserved on purpose:
the pixel matrix hashes exact bytes.

Three load strategies, tried in order by the orchestrators:

1. ``pipe.load_lora_weights`` -- diffusers' own adapter management, with the
   state dict handed over bare and then ``transformer.``-prefixed, because
   diffusers silently no-ops when no key matches.
2. PEFT ``inject_adapter_in_model`` on the transformer -- still re-weightable
   through ``set_adapters``.
3. Direct weight merge through the ADR-019 dispatcher
   (``apply_merge_delta``) -- the user weight is baked in, the adapter is
   registered in ``peft_config`` with a ``<kind>_direct`` tag, and the
   pre-merge tensors are kept on the transformer for :func:`unload_adapters`.

Not here, deliberately: ComfyUI ``folder_paths`` lookups and the node-side
``_set_adapters_safe`` -- comfyless resolves paths itself and re-weights via
``generate.apply_adapter_weights``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, cast

import torch

LOG = "[LoRA]"

# Kohya-style keys encode the module path with underscores and append one of
# these markers before the adapter parameter name; the decoders split there.
_SUFFIX_MARKERS = (
    ".lora_down.", ".lora_up.", ".lora_A.", ".lora_B.",
    ".alpha", ".lokr_", ".hada_", ".diff",
)

# Kohya text-encoder key prefixes -> the pipeline attribute they address.
# Ordered so the single-TE form is tested after the numbered ones.
_TE_PREFIX_MAP = (
    ("lora_te1_", "text_encoder"),
    ("lora_te2_", "text_encoder_2"),
    ("lora_te_", "text_encoder"),
)

# Every way a text-encoder key can start; transformer-side normalisation
# drops these (they are handled by _apply_te_lora first).
_TE_KEY_PREFIXES = (
    "text_encoder.", "text_encoder_2.", "lora_te1_", "lora_te2_", "lora_te_",
)

_STATE_DICT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")

# Adapter parameter suffixes. adapter_module_path() scans this table IN
# ORDER and cuts the key at the first entry it finds anywhere in the key --
# table order, not position in the key. That is the contract (ADR-046 #3).
_PARAM_SUFFIXES = (
    ".lokr_w1", ".lokr_w2", ".lokr_t2",
    ".lora_A.weight", ".lora_A.default.weight",
    ".lora_B.weight", ".lora_B.default.weight",
    ".lora_down.weight", ".lora_up.weight",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
    ".alpha", ".diff", ".diff_b",
)

# Component prefixes that training tools bake into keys, most common first.
_KNOWN_PREFIXES = (
    "transformer.",
    "diffusion_model.",
    "model.diffusion_model.",
    "model.",
)

# Substrings that identify each adapter family, in detection priority.
_KIND_SIGNATURES = (
    ("lokr", ("lokr_w1", "lokr_w2")),
    ("loha", ("hada_w1_a", "hada_w2_a")),
    ("lora", ("lora_A", "lora_B", "lora_down", "lora_up")),
)


# ======================================================================
#  State-dict helpers (pure; no model)
# ======================================================================

def make_adapter_name(filename: str) -> str:
    """PEFT-safe adapter name for a LoRA file.

    PEFT uses adapter names as attribute names and state-dict key fragments,
    so dots (``style_v1.1``) and spaces break it. Drop one known extension,
    then map ``.`` and space to ``_``.
    """
    stem = filename
    lowered = filename.lower()
    for ext in _STATE_DICT_EXTENSIONS:
        if lowered.endswith(ext):
            stem = filename[:-len(ext)]
            break
    return stem.translate(str.maketrans({".": "_", " ": "_"}))


def load_state_dict(path: str) -> dict:
    """Read an adapter file: safetensors natively, anything else via torch."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def adapter_module_path(key: str) -> str:
    """Module path of an adapter key, i.e. the key minus its parameter suffix.

    ``transformer_blocks.0.attn.to_q.lora_A.weight`` ->
    ``transformer_blocks.0.attn.to_q``. The suffix table is scanned in table
    order; a key with no known suffix loses its last dotted component.
    """
    for suffix in _PARAM_SUFFIXES:
        cut = key.find(suffix)
        if cut >= 0:
            return key[:cut]
    head, dot, _ = key.rpartition(".")
    return head if dot else key


def detect_adapter_type(state_dict: dict) -> str:
    """``"lokr"``, ``"loha"``, ``"lora"`` or ``"unknown"`` from the key names."""
    for kind, needles in _KIND_SIGNATURES:
        if any(needle in key for key in state_dict for needle in needles):
            return kind
    return "unknown"


def rename_lora_down_up(state_dict: dict) -> dict:
    """Kohya ``lora_down``/``lora_up`` -> PEFT ``lora_A``/``lora_B``.

    Returns the input object untouched when it carries neither name, so the
    rest of the loader only ever sees one convention.
    """
    if not any("lora_down" in key or "lora_up" in key for key in state_dict):
        return state_dict
    renamed: dict = {}
    changed = 0
    for key, value in state_dict.items():
        new_key = (key.replace(".lora_down.weight", ".lora_A.weight")
                      .replace(".lora_up.weight", ".lora_B.weight"))
        changed += new_key != key
        renamed[new_key] = value
    if changed:
        print(f"{LOG} Renamed {changed} lora_down/lora_up keys to lora_A/lora_B")
    return renamed


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    n = len(prefix)
    return {(k[n:] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()}


def _discover_prefix(paths: set, model_names: set) -> Optional[str]:
    """Find a prefix under which >30% of *paths* land on *model_names*.

    Candidates come from any adapter path that ends with a model module name.
    Scanned in sorted order (the first 20 paths against every module name)
    so the answer is the same in every process -- ADR-046 #5's one recorded
    deviation from the original, whose candidate order was set-iteration
    order and therefore hash-seed dependent.
    """
    threshold = len(paths) * 0.3
    ordered_names = sorted(model_names)
    for path in sorted(paths)[:20]:
        for name in ordered_names:
            if len(path) > len(name) and path.endswith(name):
                prefix = path[:-len(name)]
                hits = sum(1 for p in paths
                           if p.startswith(prefix) and p[len(prefix):] in model_names)
                if hits > threshold:
                    return prefix
    return None


def normalize_keys(state_dict: dict, model=None) -> dict:
    """Make adapter keys relative to the transformer module.

    Text-encoder keys are dropped (``_apply_te_lora`` owns them). Without a
    *model* the only thing stripped is a leading ``transformer.``. With one,
    the adapter's module paths are compared to ``model.named_modules()``:
    already matching -> unchanged; else each known component prefix is tried
    in order; else a prefix is discovered from the names themselves; else the
    dict is returned as-is with a loud warning.
    """
    filtered = {k: v for k, v in state_dict.items()
                if not k.startswith(_TE_KEY_PREFIXES)}
    if not filtered:
        return filtered

    if model is None:
        if any(k.startswith("transformer.") for k in filtered):
            return _strip_prefix(filtered, "transformer.")
        return filtered

    model_names = {name for name, _ in model.named_modules() if name}
    paths = {adapter_module_path(k) for k in filtered}
    if paths & model_names:
        return filtered

    for prefix in _KNOWN_PREFIXES:
        n = len(prefix)
        if {p[n:] for p in paths if p.startswith(prefix)} & model_names:
            return _strip_prefix(filtered, prefix)

    prefix = _discover_prefix(paths, model_names)
    if prefix is not None:
        print(f"{LOG} Auto-detected key prefix: '{prefix}'")
        return _strip_prefix(filtered, prefix)

    print(f"{LOG} WARNING: Could not match state-dict keys to model modules "
          f"after trying known prefixes.")
    print(f"{LOG}   state-dict paths (sample): {sorted(paths)[:3]}")
    print(f"{LOG}   model module paths (sample): {sorted(model_names)[:5]}")
    return filtered


def _group_by_module(state_dict: dict) -> Dict[str, Dict[str, Any]]:
    """``{module_path: {param_name: tensor}}`` keyed by adapter_module_path."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for key, value in state_dict.items():
        path = adapter_module_path(key)
        grouped.setdefault(path, {})[key[len(path) + 1:]] = value
    return grouped


def _transformer_of(pipe) -> Any:
    return getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)


# ======================================================================
#  PEFT injection
# ======================================================================

def _lora_config(state_dict: dict):
    from peft import LoraConfig
    rank = next((v.shape[0] for k, v in state_dict.items()
                 if "lora_A" in k and v.ndim >= 2), 64)
    # Only an .alpha whose module still has its lora_A counts. An orphan
    # alpha (weights dropped by normalisation) must not become the GLOBAL
    # lora_alpha, or already-baked weights would be scaled twice.
    alpha = next((v.item() for k, v in state_dict.items()
                  if k.endswith(".alpha") and v.numel() == 1
                  and f"{k[:-len('.alpha')]}.lora_A.weight" in state_dict),
                 float(rank))
    # PEFT annotates alpha as int but accepts (and stores) a float.
    return LoraConfig(r=rank, lora_alpha=cast(int, alpha), target_modules=["_dummy"])


def _lokr_config(state_dict: dict):
    from peft import LoKrConfig
    # A huge r makes PEFT allocate full (undecomposed) w1/w2 so the stored
    # factors load as-is; alpha == r gives unit base scaling, matching the
    # LyCORIS / ComfyUI convention. The user weight is applied by set_adapters.
    full = 100000
    return LoKrConfig(r=full, alpha=cast(int, float(full)), decompose_both=False,
                      decompose_factor=-1, target_modules=["_dummy"])


def _loha_config(state_dict: dict):
    from peft import LoHaConfig
    rank = None
    for key, value in state_dict.items():
        if ".hada_w1_a" in key and value.ndim >= 2:
            rank = value.shape[1]          # (out, rank)
            break
        if ".hada_w1_b" in key and value.ndim >= 2:
            rank = value.shape[0]          # (rank, in)
            break
    if rank is None:
        rank = 8
    alpha = next((v.item() for k, v in state_dict.items()
                  if k.endswith(".alpha") and v.numel() == 1), float(rank))
    return LoHaConfig(r=rank, alpha=cast(int, alpha), target_modules=["_dummy"])


def _inject_peft(pipe, config, state_dict: dict, adapter_name: str,
                 log_prefix: str, label: str) -> None:
    """Structure from *config*, weights from *state_dict*, onto the transformer.

    Raises whatever PEFT raises (ValueError / RuntimeError on shape or key
    mismatch); the orchestrators catch those and fall through to direct merge.
    """
    from peft import inject_adapter_in_model, set_peft_model_state_dict

    transformer = _transformer_of(pipe)
    inject_adapter_in_model(config, transformer, adapter_name=adapter_name,
                            state_dict=state_dict)
    result = set_peft_model_state_dict(transformer, state_dict,
                                       adapter_name=adapter_name)
    # Lets diffusers' set_adapters / get_list_adapters see the transformer.
    transformer._hf_peft_config_loaded = True

    if result:
        unexpected = [k for k in getattr(result, "unexpected_keys", [])
                      if not k.endswith(".alpha")]      # .alpha is expected
        if unexpected:
            print(f"{log_prefix} {label} unexpected keys: {unexpected[:5]}...")
        missing = getattr(result, "missing_keys", [])
        if missing:
            print(f"{log_prefix} {label} missing keys: {missing[:5]}...")
    print(f"{log_prefix} {label} adapter loaded successfully via PEFT injection")


# ======================================================================
#  Direct weight merge -- one driver, three delta formulas
# ======================================================================

class _MergeKind(NamedTuple):
    tag: str                                    # "lora" | "lokr" | "loha"
    label: str
    build: Callable[[dict], Optional[Tuple[torch.Tensor, int]]]
    """params -> (unscaled delta, rank) or None when the module is incomplete."""


def _build_lora(params: dict):
    a = params.get("lora_A.weight")
    if a is None:
        a = params.get("lora_A.default.weight")
    b = params.get("lora_B.weight")
    if b is None:
        b = params.get("lora_B.default.weight")
    if a is None or b is None:
        return None
    return b.float() @ a.float(), a.shape[0]


def _build_lokr(params: dict):
    w1, w2 = params.get("lokr_w1"), params.get("lokr_w2")
    if w1 is None or w2 is None:
        return None
    rank = min(w1.shape) if w1.ndim >= 2 else 1
    return torch.kron(w1.float(), w2.float()), rank


def _build_loha(params: dict):
    w1a, w1b = params.get("hada_w1_a"), params.get("hada_w1_b")
    w2a, w2b = params.get("hada_w2_a"), params.get("hada_w2_b")
    if w1a is None or w1b is None or w2a is None or w2b is None:
        return None
    rank = w1b.shape[0] if w1b.ndim >= 2 else 1
    return (w1a.float() @ w1b.float()) * (w2a.float() @ w2b.float()), rank


_LORA = _MergeKind("lora", "LoRA", _build_lora)
_LOKR = _MergeKind("lokr", "LoKR", _build_lokr)
_LOHA = _MergeKind("loha", "LoHa", _build_loha)


def _resolve_target(model_sd: dict, module_path: str) -> Optional[str]:
    for candidate in (module_path + ".weight", module_path):
        if candidate in model_sd:
            return candidate
    return None


def _merge_direct(pipe, state_dict: dict, adapter_name: str,
                  log_prefix: str, weight: float, kind: _MergeKind) -> bool:
    """Add ``delta * scale`` straight into the transformer's parameters.

    ``scale`` is ``alpha / rank * weight`` when the module ships an alpha,
    else just ``weight`` (pre-scaled weights). Every write goes through the
    ADR-019 dispatcher so quantized bases dequant -> merge -> requant and
    unmergeable ones refuse BEFORE the first write (all-or-nothing). Pre-merge
    tensors land in ``transformer._<kind>_backup_<name>`` for unload; the
    adapter is registered in ``peft_config`` as ``<kind>_direct`` so adapter
    discovery finds it, with the caveat that the weight is now baked in.

    Returns True only when at least one module was merged.
    """
    from comfyless.core.eric_diffusion_fp8_ops import (
        apply_merge_delta, merge_resolution_map, record_direct_merge,
        refuse_unmergeable_base,
    )

    transformer = _transformer_of(pipe)
    # .weight-named BUFFERS included, so fp8-resident (ScaledFp8Linear)
    # targets resolve -- ADR-019 security review DMR-3/8.
    model_sd = merge_resolution_map(transformer)
    # All-or-nothing (ADR-019 req 65): refuse before touching anything.
    refuse_unmergeable_base(transformer, model_sd, log_prefix)

    backup_attr = f"_{kind.tag}_backup_{adapter_name}"
    applied = skipped = 0
    for module_path, params in _group_by_module(state_dict).items():
        built = kind.build(params)
        if built is None:
            skipped += 1
            continue
        raw_delta, rank = built

        alpha = params.get("alpha")
        if alpha is not None:
            scale = (alpha.item() / rank) * weight if rank > 0 else weight
        else:
            scale = weight
        delta = raw_delta * scale

        target = _resolve_target(model_sd, module_path)
        if target is None:
            skipped += 1
            continue
        param = model_sd[target]
        if delta.shape != param.shape:
            try:
                delta = delta.reshape(param.shape)
            except RuntimeError:
                print(f"{log_prefix} {kind.label} shape mismatch for {module_path}: "
                      f"delta {delta.shape} vs param {param.shape}, skipping")
                skipped += 1
                continue

        if not hasattr(transformer, backup_attr):
            setattr(transformer, backup_attr, {})
        apply_merge_delta(transformer, target, delta,
                          getattr(transformer, backup_attr),
                          log_prefix=log_prefix)
        applied += 1

    if applied:
        record_direct_merge(transformer, adapter_name)   # LIFO unload ledger

    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    transformer.peft_config[adapter_name] = {
        "_type": f"{kind.tag}_direct",
        "_applied_modules": applied,
        "_weight": weight,
    }
    transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} {kind.label} direct merge (weight={weight}): "
          f"applied={applied}, skipped={skipped}")
    if skipped > applied:
        print(f"{log_prefix} WARNING: Many modules skipped. "
              f"State-dict keys (sample): {list(state_dict.keys())[:5]}")
        print(f"{log_prefix}   Model params (sample): {sorted(model_sd.keys())[:5]}")
    return applied > 0


# ======================================================================
#  Per-kind orchestrators
# ======================================================================

def _load_lokr_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = LOG, weight: float = 1.0) -> bool:
    """LoKR: PEFT injection, else direct merge, else flatten to LoRA.

    diffusers' ``load_lora_weights`` knows nothing of LyCORIS, so LoKR never
    takes the pipeline path. The last resort (flatten + SVD, rank 64, lossy)
    exists for bases whose module names diffusers maps but whose Kronecker
    factorisation neither PEFT nor the direct merge can place (Z-Image,
    2026-07-06); the alternative was a silent no-op.
    """
    try:
        _inject_peft(pipe, _lokr_config(state_dict), state_dict, adapter_name,
                     log_prefix, "LoKR")
        return True
    except (ValueError, RuntimeError) as peft_err:
        print(f"{log_prefix} PEFT injection failed: {peft_err}")
        print(f"{log_prefix} Falling back to direct weight merge...")
        if _merge_direct(pipe, state_dict, adapter_name, log_prefix, weight, _LOKR):
            return True

        from comfyless.core.eric_lora_format_convert_apply import flatten_lokr_to_lora_sd
        try:
            flat = flatten_lokr_to_lora_sd(state_dict, log_prefix=log_prefix)
        except Exception as flat_err:  # noqa: BLE001 -- the rescue must never crash the load
            print(f"{log_prefix} LoKR->LoRA flatten failed: {str(flat_err)[:120]}")
            return False
        if not flat:
            return False
        # The 0-module merge above registered a stale marker; clear it so
        # the standard loader can register the name cleanly.
        transformer = _transformer_of(pipe)
        if transformer is not None and hasattr(transformer, "peft_config"):
            transformer.peft_config.pop(adapter_name, None)
        modules = sum(1 for k in flat if k.endswith(".lora_A.weight"))
        print(f"{log_prefix} Retrying as flattened standard LoRA ({modules} modules)...")
        return _load_lora_adapter(pipe, flat, adapter_name, log_prefix, weight=weight)


def _load_loha_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = LOG, weight: float = 1.0) -> bool:
    """LoHa: PEFT injection, else direct merge."""
    try:
        _inject_peft(pipe, _loha_config(state_dict), state_dict, adapter_name,
                     log_prefix, "LoHa")
        return True
    except (ValueError, RuntimeError) as peft_err:
        print(f"{log_prefix} PEFT injection failed for LoHa: {peft_err}")
        print(f"{log_prefix} Falling back to direct weight merge...")
        return _merge_direct(pipe, state_dict, adapter_name, log_prefix, weight, _LOHA)


def _load_lora_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = LOG, weight: float = 1.0) -> bool:
    """Standard LoRA: pipeline load (bare, then ``transformer.``-prefixed),
    else PEFT injection, else direct merge.

    Alpha is baked into the weights FIRST (see ``_bake_lora_alpha_scales``):
    diffusers drops orphan ``.alpha`` keys on an already-``lora_A/B`` dict,
    which is how an alpha != rank LoRA used to load 4x too strong.
    """
    state_dict = _bake_lora_alpha_scales(state_dict, log_prefix)
    state_dict = rename_lora_down_up(state_dict)

    # diffusers silently no-ops when no key matches, so every attempt is
    # verified against get_list_adapters before it counts as a load.
    attempts = (
        ("normalised keys", state_dict),
        ("transformer-prefixed keys",
         {f"transformer.{k}": v for k, v in state_dict.items()}),
    )
    for label, candidate in attempts:
        try:
            pipe.load_lora_weights(candidate, adapter_name=adapter_name)
        except (ValueError, RuntimeError) as e:
            print(f"{log_prefix} Pipeline LoRA load ({label}) failed: {str(e)[:120]}")
            continue
        try:
            registered = any(adapter_name in names
                             for names in pipe.get_list_adapters().values())
        except Exception:
            registered = True            # cannot verify; take diffusers' word
        if registered:
            print(f"{log_prefix} LoRA loaded via pipeline with {label}")
            return True
        print(f"{log_prefix} Pipeline load ({label}) returned but adapter "
              f"not registered -- trying next path")

    try:
        _inject_peft(pipe, _lora_config(state_dict), state_dict, adapter_name,
                     log_prefix, "LoRA")
        return True
    except (ValueError, RuntimeError) as e:
        print(f"{log_prefix} Direct PEFT LoRA injection failed: {str(e)[:120]}")

    print(f"{log_prefix} Falling back to direct LoRA weight merge...")
    return _merge_direct(pipe, state_dict, adapter_name, log_prefix, weight, _LORA)


# ======================================================================
#  Text-encoder keys, Kohya decode, alpha bake  (verbatim from the
#  node-pack module; Grant-authored, incident-hardened -- see ADR-046)
# ======================================================================

def _apply_te_lora(pipe, state_dict: dict, adapter_name: str,
                   log_prefix: str = "[LoRA]",
                   weight: float = 1.0) -> bool:
    """Decode and apply Kohya-format text encoder LoRA keys (lora_te1_*, lora_te2_*, lora_te_*).

    Called from the fallback path of load_lora_with_key_fix, before the
    transformer key normalisation that would otherwise silently discard TE keys.

    Decoding: uses the text encoder's named_modules() tree, exactly as
    _decode_kohya_keys uses the transformer's tree for lora_transformer_* keys.

    Application: tries pipe.load_lora_weights() with decoded+prefixed keys
    (diffusers routes 'text_encoder.*' / 'text_encoder_2.*' keys correctly).
    Falls back to a warning if that also fails — does not attempt direct merge
    on the TE (TE direct merge is complex and TEs are usually small enough that
    the pipeline path succeeds).

    Returns True if any TE keys were found (regardless of application success).
    """
    has_te_keys = any(
        k.startswith(pfx)
        for pfx, _ in _TE_PREFIX_MAP
        for k in state_dict
    )
    if not has_te_keys:
        return False

    te_dict: dict = {}  # combined {component.path: tensor} for pipe.load_lora_weights

    for prefix, component_attr in _TE_PREFIX_MAP:
        te_module = getattr(pipe, component_attr, None)
        if te_module is None:
            continue

        keys_for_prefix = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not keys_for_prefix:
            continue

        # Build underscore→dot lookup from the TE module tree
        underscore_to_dot = {
            name.replace(".", "_"): name
            for name, _ in te_module.named_modules()
            if name
        }

        decoded = 0
        skipped_keys = []
        for key, value in keys_for_prefix.items():
            remainder = key[len(prefix):]

            # Find adapter suffix boundary
            split_idx = -1
            for marker in _SUFFIX_MARKERS:
                idx = remainder.find(marker)
                if idx >= 0 and (split_idx < 0 or idx < split_idx):
                    split_idx = idx

            if split_idx < 0:
                skipped_keys.append(key)
                continue

            module_encoded = remainder[:split_idx]
            adapter_suffix = remainder[split_idx:]

            if module_encoded in underscore_to_dot:
                out_key = f"{component_attr}.{underscore_to_dot[module_encoded]}{adapter_suffix}"
                te_dict[out_key] = value
                decoded += 1
            else:
                skipped_keys.append(key)

        if decoded:
            print(f"{log_prefix} TE LoRA ({component_attr}): {decoded} keys decoded")
        if skipped_keys:
            print(f"{log_prefix} TE LoRA ({component_attr}): "
                  f"{len(skipped_keys)} keys could not be decoded (module not in TE)")

    if not te_dict:
        print(f"{log_prefix} TE LoRA keys found but none could be decoded — skipping TE application")
        return True  # keys were present even if not applied

    try:
        pipe.load_lora_weights(te_dict, adapter_name=f"{adapter_name}_te")
        print(f"{log_prefix} Text encoder LoRA applied ({len(te_dict)} params)")
    except Exception as e:
        print(f"{log_prefix} Text encoder LoRA load failed (non-fatal): {e}")

    return True


def _decode_kohya_keys(state_dict: dict, model) -> dict:
    """Convert Kohya underscore-encoded LoRA keys to dot-separated format.

    Kohya trainers encode module paths by replacing dots with underscores
    and prepending ``lora_transformer_`` or ``lora_unet_``.  This function
    recovers the original dot-separated paths so that PEFT/diffusers can
    route the weights to the correct modules.

    Two conventions are handled:

    ``lora_transformer_*``
        Diffusers-format module names with underscores.  Decoded by matching
        against the model's actual module tree (unambiguous because the tree
        tells us which underscores are dots).

    ``lora_unet_*``
        Original-format module names (``double_blocks``, ``single_blocks``).
        Decoded via diffusers' built-in Flux Kohya converter, which handles
        key mapping *and* QKV splitting.  Works for Chroma because it shares
        Flux's block structure.
    """
    if model is None:
        return state_dict

    has_lora_transformer = any(k.startswith("lora_transformer_") for k in state_dict)
    has_lora_unet = any(k.startswith("lora_unet_") for k in state_dict)

    if not has_lora_transformer and not has_lora_unet:
        return state_dict

    # ── lora_unet_* → use diffusers' Flux Kohya converter ──────────────
    # Chroma shares Flux's block architecture, so the Flux mapping applies.
    if has_lora_unet:
        try:
            from diffusers.loaders.lora_conversion_utils import (
                _convert_kohya_flux_lora_to_diffusers,
            )
            converted = _convert_kohya_flux_lora_to_diffusers(state_dict)
            if converted:
                print(f"[LoRA] Converted {len(converted)} lora_unet_ keys "
                      f"via Flux Kohya converter")
                return converted
        except Exception as e:
            print(f"[LoRA] Flux Kohya converter failed: {e}")

    if not has_lora_transformer:
        return state_dict

    # ── lora_transformer_* → decode using model module tree ─────────────
    model_modules = {name for name, _ in model.named_modules() if name}

    # Build lookup: underscore-encoded name → dot-separated name
    underscore_to_dot = {}
    for name in model_modules:
        underscore_to_dot[name.replace(".", "_")] = name

    decoded = {}
    converted = 0
    skipped_modules = set()

    for key, value in state_dict.items():
        if not key.startswith("lora_transformer_"):
            decoded[key] = value
            continue

        remainder = key[len("lora_transformer_"):]

        # Find where the adapter suffix starts
        split_idx = -1
        for marker in _SUFFIX_MARKERS:
            idx = remainder.find(marker)
            if idx >= 0 and (split_idx < 0 or idx < split_idx):
                split_idx = idx

        if split_idx < 0:
            decoded[key] = value
            continue

        module_encoded = remainder[:split_idx]
        adapter_suffix = remainder[split_idx:]

        if module_encoded in underscore_to_dot:
            new_key = underscore_to_dot[module_encoded] + adapter_suffix
            decoded[new_key] = value
            converted += 1
        else:
            # Module not in this model (e.g. distilled_guidance_layer on
            # Chroma-HD) — drop the key rather than crash later.
            skipped_modules.add(module_encoded.split("_")[0])

    if converted > 0:
        print(f"[LoRA] Decoded {converted} Kohya lora_transformer_ keys "
              f"to dot-separated format")
    if skipped_modules:
        print(f"[LoRA] Skipped keys targeting modules not in this model: "
              f"{skipped_modules}")

    return decoded


def _bake_lora_alpha_scales(state_dict: dict, log_prefix: str = "[LoRA]") -> dict:
    """Fold each LoRA module's ``alpha/rank`` scale into its weights and drop the
    ``.alpha`` keys, matching diffusers' ``_convert_non_diffusers_qwen_lora_to_diffusers``
    ``get_alpha_scales`` numerics exactly (scale split across down/up, product
    preserved).

    WHY (root cause of the "over-strength LoRA → noise" bug): diffusers' pipeline
    loader bakes alpha ONLY when it sees the kohya ``lora_down``/``lora_up``
    layout. If we hand it ``lora_A``/``lora_B`` + orphan ``.alpha`` (which is what
    ``_rename_lora_down_up`` used to produce before the pipeline load), the
    converter takes its "already in diffusers format" branch, copies the weights
    **unscaled**, and **discards** the ``.alpha`` keys — silently dropping the
    ``alpha/rank`` scale. On a LoRA with ``alpha != rank`` that applies the LoRA
    far too strong (e.g. rank 64 / alpha 16 → 4×) across every module → garbage.

    Baking here, BEFORE any rename/load, makes every downstream path (pipeline
    "already-diffusers" branch, direct PEFT injection, direct merge) apply the
    correct magnitude, because they all then see no ``.alpha`` and default to
    scale 1.0 on pre-scaled weights. Handles ``lora_down``/``lora_up`` OR
    ``lora_A``/``lora_B`` naming. A dict with no ``.alpha`` keys (already-diffusers
    LoRAs) is returned unchanged, and ``alpha == rank`` bakes to a numeric no-op —
    so neither case regresses.
    """
    alpha_keys = [k for k in state_dict if k.endswith(".alpha")]
    if not alpha_keys:
        return state_dict
    out = dict(state_dict)
    baked = 0
    for ak in alpha_keys:
        base = ak[:-len(".alpha")]  # module path minus the .alpha suffix
        down_k = next((f"{base}.{s}" for s in ("lora_down.weight", "lora_A.weight")
                       if f"{base}.{s}" in out), None)
        up_k = next((f"{base}.{s}" for s in ("lora_up.weight", "lora_B.weight")
                     if f"{base}.{s}" in out), None)
        if down_k is None or up_k is None:
            continue  # orphan alpha with no paired weights — leave it be
        down = out[down_k]
        rank = down.shape[0]
        if rank <= 0:
            continue
        # Corrupt-file guards (code review): a multi-element .alpha would raise
        # in .item(); a non-positive alpha would spin the balancing loop below
        # (diffusers itself hangs on negative alpha). Skip such keys — leave them
        # in place, unscaled — rather than crash or hang the whole load.
        if out[ak].numel() != 1:
            continue
        alpha = out[ak].item()
        if alpha <= 0:
            continue
        scale = alpha / rank
        # diffusers get_alpha_scales: balance the scale across down/up so neither
        # tensor gets an extreme multiplier; the PRODUCT stays alpha/rank exactly.
        scale_down, scale_up = scale, 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2
        out[down_k] = down * scale_down
        out[up_k] = out[up_k] * scale_up
        del out[ak]
        baked += 1
    if baked:
        print(f"{log_prefix} Baked alpha/rank scale into {baked} LoRA modules "
              f"(matches the diffusers converter; prevents the alpha-drop "
              f"over-strength that reads as noise)")
    return out


# ======================================================================
#  Unload / registry queries  (verbatim, Grant-authored)
# ======================================================================

def unload_adapters(pipe, adapter_names, log_prefix: str = "[LoRA]") -> None:
    """Unload a set of adapters from the pipeline's transformer.

    Handles both PEFT-managed and direct-merge adapters.  Direct-merge
    adapters are restored from backups stored on the transformer during
    load (``_<kind>_backup_<name>`` dicts mapping param name → original
    tensor).  PEFT-managed adapters are removed via ``delete_adapters``.

    Use this when the LoRA stacker needs to drop adapters that were
    loaded in a previous run but aren't in the current stack — otherwise
    stale adapters remain attached and their weights are still applied.
    """
    if not adapter_names:
        return
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return

    peft_cfg = getattr(transformer, "peft_config", None) or {}

    for adapter_name in list(adapter_names):
        cfg = peft_cfg.get(adapter_name, None)

        # Direct-merge: cfg is a dict with a ``_type`` key ending in "_direct".
        if isinstance(cfg, dict) and cfg.get("_type", "").endswith("_direct"):
            adapter_family = cfg.get("_type", "").replace("_direct", "")
            backup_key = f"_{adapter_family}_backup_{adapter_name}"
            backup = getattr(transformer, backup_key, None)
            # LIFO guard + ledger pop (DMR req 25) runs regardless of backup
            # presence so the ledger stays authoritative even if a backup was
            # manually removed (final code-review finding 1).
            from comfyless.core.eric_diffusion_fp8_ops import warn_non_lifo_unload
            warn_non_lifo_unload(transformer, adapter_name, log_prefix)
            if backup:
                from comfyless.core.eric_lora_format_convert_apply import resolve_restore_target
                from comfyless.core.eric_diffusion_fp8_ops import (
                    merge_resolution_map, restore_merge_backup,
                )
                # Same map as merge time: .weight buffers included so
                # quantized (ScaledFp8Linear) targets resolve.
                model_sd = merge_resolution_map(transformer)
                restored = 0
                for target_key, original_tensor in backup.items():
                    # Re-resolve through any PEFT wrapping added/removed since
                    # merge, so a stale .weight ↔ .base_layer.weight move
                    # doesn't leave the delta baked in.
                    live_key = resolve_restore_target(model_sd, target_key)
                    if live_key is None:
                        continue
                    # Kind-tagged quantized backups (slice DMR) restore by
                    # verbatim swap; plain tensors keep the legacy copy_.
                    if isinstance(original_tensor, dict):
                        if restore_merge_backup(transformer, live_key,
                                                original_tensor, log_prefix):
                            restored += 1
                        continue
                    param = model_sd.get(live_key)
                    if param is not None:
                        param.data.copy_(original_tensor.to(
                            dtype=param.dtype, device=param.device,
                        ))
                        restored += 1
                print(f"{log_prefix} Direct-merge '{adapter_name}' restored "
                      f"({restored}/{len(backup)} params)")
                try:
                    delattr(transformer, backup_key)
                except AttributeError:
                    pass
            else:
                print(f"{log_prefix} Direct-merge '{adapter_name}' has no "
                      f"backup to restore — weights may remain baked in")
            try:
                del peft_cfg[adapter_name]
            except (KeyError, TypeError):
                pass
            continue

        # PEFT-managed: use delete_adapters
        try:
            if hasattr(pipe, "delete_adapters"):
                pipe.delete_adapters(adapter_name)
            elif hasattr(transformer, "delete_adapters"):
                transformer.delete_adapters(adapter_name)
            else:
                try:
                    del peft_cfg[adapter_name]
                except (KeyError, TypeError):
                    pass
            print(f"{log_prefix} PEFT adapter '{adapter_name}' deleted")
        except Exception as e:
            print(f"{log_prefix} Failed to delete PEFT adapter "
                  f"'{adapter_name}': {e}")


def is_direct_merge_adapter(pipe, adapter_name: str) -> bool:
    """True when `adapter_name` was applied via direct weight merge.

    Direct-merge adapters bake the user weight into the model parameters at
    merge time and have no PEFT tuner layers — ``set_adapters()`` must not
    be pointed at them (it would either error or silently no-op). The
    ``_type`` sniff matches the registry entries the direct-merge path
    writes into ``transformer.peft_config``.
    """
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return False
    peft_cfg = getattr(transformer, "peft_config", {})
    cfg = peft_cfg.get(adapter_name, {})
    return isinstance(cfg, dict) and cfg.get("_type", "").endswith("_direct")


def plan_match_model_names(module) -> list:
    """Model-side name list that `find_matching_plan`'s `model_signature`
    substring is tested against (e.g. the Krea plan's `to_gate`).

    Includes BUFFERS, not just parameters: fp8-resident quantized Linears
    (`ScaledFp8Linear`) register their `.weight` as a buffer, and bias-free
    projections (e.g. Krea attn `to_gate`, the SwiGLU `ff.*`) carry no
    Parameter at all — so `named_parameters()` alone hides them and the plan
    signature goes undetected, making `find_matching_plan` return None. On
    such a base the krea_native→diffusers_krea rename never fires and every
    LoRA falls through to a 0-module merge. A strict superset for bf16 bases
    (whose weights are Parameters either way), and safe: `find_matching_plan`
    pins the source family from the LoRA keys, so extra names can only
    qualify the already-selected plan, never cross-match a wrong one.
    """
    return ([n for n, _ in module.named_parameters()]
            + [n for n, _ in module.named_buffers()])


# ======================================================================
#  Entry point
# ======================================================================

def load_lora_with_key_fix(pipe, lora_path: str, adapter_name: str,
                          log_prefix: str = LOG,
                          weight: float = 1.0,
                          min_compatibility: float = 0.0) -> bool:
    """Load a LoRA / LoKR / LoHa file onto *pipe* with format auto-detection.

    Order of business:

    1. Header-only compatibility check against the transformer (never fatal;
       skips the file when ``key_match_pct`` is below *min_compatibility*).
    2. If NOTHING matched and a registered conversion plan covers the
       (adapter family, model family) pair, convert in memory and load the
       result -- the original-layout Klein/Flux2 LoRA on a diffusers model.
    3. ``pipe.load_lora_weights(path)`` as-is. Errors that read as a key or
       format problem fall through; anything else propagates.
    4. Manual path: load the file, apply text-encoder keys, decode Kohya
       names, normalise prefixes, detect the family and hand off to its
       orchestrator.

    Returns True when the adapter is actually active on the model -- a direct
    merge that patched zero modules reports False rather than "loaded".
    """
    from comfyless.core.eric_diffusion_lora_check import check_lora
    from comfyless.core.eric_lora_format_convert import decode_kohya_to_bfl
    from comfyless.core.eric_lora_format_convert_apply import (
        convert_state_dict, find_matching_plan, load_converted_lora,
    )

    # ── Compatibility pre-check (header only — fast) ──────────────────
    transformer = getattr(pipe, "transformer", None)
    pre_check = None
    if transformer is not None:
        try:
            pre_check = check_lora(lora_path, transformer=transformer,
                                   log_prefix=log_prefix)
            for line in pre_check.log_lines(prefix=log_prefix):
                print(line)
            if min_compatibility > 0 and pre_check.key_match_pct < min_compatibility * 100:
                pre_check.skipped = True
                print(
                    f"{log_prefix} SKIP {os.path.basename(lora_path)}: "
                    f"compatibility {pre_check.key_match_pct:.0f}% < "
                    f"threshold {min_compatibility * 100:.0f}%"
                )
                return False
        except Exception as chk_err:
            print(f"{log_prefix} Compatibility check failed (non-fatal): {chk_err}")

    # ── Conversion attempt (slice 4) ─────────────────────────────────
    # When the compatibility check shows 0% module match AND a registered
    # ConversionPlan covers (LoRA family, model family), do the in-memory
    # rename + LoKR/LoHa→LoRA SVD compression up front and route the
    # result through the standard-LoRA loader.  This catches the
    # "original BFL Klein/Flux2 LoRA loaded against diffusers Klein"
    # case that previously fell through to a silent direct-merge no-op.
    if (transformer is not None and pre_check is not None
            and pre_check.matched == 0 and pre_check.total_layers > 0):
        try:
            source_sd = load_state_dict(lora_path)
            # Apply text encoder LoRA keys before conversion (they pass
            # through convert_state_dict unchanged and cause diffusers
            # to warn about unexpected keys in load_converted_lora).
            _apply_te_lora(pipe, source_sd, adapter_name, log_prefix, weight)
            _TE_STRIP = tuple(pfx for pfx, _ in _TE_PREFIX_MAP) + (
                "text_encoder.", "text_encoder_2.",
            )
            source_sd = {k: v for k, v in source_sd.items()
                         if not k.startswith(_TE_STRIP)}
            # Chroma LoRAs often come in Kohya underscore format
            # (lora_unet_*); decode to BFL dot format so
            # detect_lora_format recognises them as bfl_original.
            source_sd = decode_kohya_to_bfl(source_sd)
            # Buffer-aware model names so fp8-resident (ScaledFp8Linear) bases
            # are detectable by the plan's model_signature — see
            # plan_match_model_names.
            model_param_names = plan_match_model_names(transformer)
            plan = find_matching_plan(source_sd, model_param_names)
            if plan is not None:
                print(
                    f"{log_prefix} 0% match + registered plan available: "
                    f"converting {plan.source_family} → {plan.target_family}"
                )
                converted = convert_state_dict(
                    source_sd, plan, log_prefix=log_prefix,
                )
                if converted:
                    # load_converted_lora handles diffusers' transformer-
                    # prefix expectation and falls back to direct delta
                    # merge if PEFT silently no-ops (which it did for
                    # the Klein LoKR case in the first slice-4 trial).
                    success = load_converted_lora(
                        pipe, converted, adapter_name, log_prefix,
                        weight=weight,
                    )
                    if success:
                        print(
                            f"{log_prefix} Converted adapter loaded "
                            f"({plan.target_family} target)"
                        )
                        return True
                    print(
                        f"{log_prefix} Conversion produced a state dict "
                        f"but neither pipeline-load nor direct-merge "
                        f"applied it — falling back to standard paths"
                    )
                else:
                    print(
                        f"{log_prefix} Conversion produced an empty state "
                        f"dict — falling back to standard paths"
                    )
        except Exception as conv_err:
            # Any exception during conversion is non-fatal; we fall
            # through to the existing fast-path / fallback chain so a
            # broken plan doesn't break LoRAs that the legacy paths
            # could have handled.
            print(
                f"{log_prefix} Conversion path failed (non-fatal — "
                f"continuing with standard load): {conv_err}"
            )

    # ── Fast path: the file as-is ─────────────────────────────────────
    try:
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        return True
    except (ValueError, RuntimeError, KeyError) as e:
        if not _is_fixable_load_error(e):
            raise
        print(f"{log_prefix} Standard load failed, attempting format "
              f"detection...  ({str(e)[:120]})")

    # ── Manual path ───────────────────────────────────────────────────
    state_dict = load_state_dict(lora_path)
    transformer = getattr(pipe, "transformer", None)

    # TE keys first: normalisation below drops them, and they belong on the
    # text encoder(s), not the transformer.
    _apply_te_lora(pipe, state_dict, adapter_name, log_prefix, weight)

    state_dict = _decode_kohya_keys(state_dict, transformer)
    state_dict = normalize_keys(state_dict, model=transformer)

    adapter_type = detect_adapter_type(state_dict)
    print(f"{log_prefix} Detected adapter format: {adapter_type}")

    loader = _ORCHESTRATORS.get(adapter_type)
    if loader is None:
        raise ValueError(
            f"{log_prefix} Unrecognised adapter format.  First 5 keys: "
            f"{list(state_dict.keys())[:5]}"
        )
    success = loader(pipe, state_dict, adapter_name, log_prefix, weight=weight)

    # When all of the fallback paths bottomed out into direct merge and
    # the merge applied 0 modules (e.g. architecture mismatch), success
    # will be False.  Don't let the stacker claim "Loaded OK" / "active"
    # for an adapter that isn't actually patched anywhere.
    if not success:
        print(
            f"{log_prefix} FAILED — direct merge applied 0 modules; "
            f"this adapter is NOT active.  Most likely: original-format "
            f"LoRA targeting modules diffusers reorganized into a "
            f"different structure (see WRONG_ARCH diagnostic above)."
        )
        return False
    return True


_ORCHESTRATORS = {
    "lokr": _load_lokr_adapter,
    "loha": _load_loha_adapter,
    "lora": _load_lora_adapter,
}

# Error texts from diffusers / PEFT that mean "the keys did not map", which
# the manual path can often repair. Anything else is a real failure.
_FIXABLE_ERROR_MARKERS = (
    "No modules were targeted",
    "lora_A", "lora_B", "hada_",
    "PEFT backend", "Handling for key",
)
_FIXABLE_ERROR_MARKERS_CI = ("state_dict", "lokr", "loha", "not implemented")


def _is_fixable_load_error(err: BaseException) -> bool:
    # diffusers' _maybe_expand_lora_state_dict raises KeyError when a LoRA
    # targets an expanded-format key (fused QKV) the model does not have;
    # Kohya decoding / PEFT injection below handles exactly that.
    if isinstance(err, KeyError):
        return True
    text = str(err)
    lowered = text.lower()
    return (
        ("Target modules" in text and "not found" in text)
        or any(marker in text for marker in _FIXABLE_ERROR_MARKERS)
        or any(marker in lowered for marker in _FIXABLE_ERROR_MARKERS_CI)
    )
