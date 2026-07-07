"""ComfyUI-native Krea-2 → diffusers key converter (ADR-019 Changelog 2026-07-07).

Community single-file Krea-2 checkpoints (civitai / ComfyUI export, e.g.
RedCraft, Dark Beast) store the transformer with ComfyUI-native key names
(`model.diffusion_model.blocks.N.attn.wq`, `qknorm`, `mod.lin`, `first`,
`last`, `tmlp`, `txtfusion` …). Released diffusers 0.39.0 ships
`Krea2Transformer2DModel` but NO `from_single_file` for it, and its arch key
layout uses different names (`transformer_blocks.N.attn.to_q`, `norm_q`,
`scale_shift_table`, `img_in`, `final_layer`, `time_embed`, `text_fusion` …).

This module converts native → diffusers so the checkpoint can be built with
`from_config` + `load_state_dict`. EVERY mapping is a verified pure 1:1 RENAME
— no weight transformation (shapes are identical on both sides). Verified
complete against `Krea2Transformer2DModel.from_config` (0 missing / 0
unexpected keys) for the Krea-2-Turbo config.

**Bespoke-code note (deliberately minimal + isolated).** This whole file is
the shim. The rules are grouped Linear-vs-norm/embed so that, if a future
diffusers release ships a `Krea2.from_single_file` that covers this format,
whole groups can be deleted and the loader switched to `from_single_file`.
The loader asserts 0 missing keys after load, so the day upstream covers a
group these rules become provably dead (delete at leisure). The
ComfyUI-name half may remain permanently bespoke if diffusers only ever
targets the reference single-file layout.

SECURITY: this parses caller-supplied checkpoint KEY NAMES only (never
values, never paths) and emits a fixed set of diffusers key strings — the
agent/caller cannot inject arbitrary target keys (unmatched keys pass
through unchanged and are caught by the loader's missing-key assertion).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable

# Attention linear renames within a block (native → diffusers leaf).
_ATTN_MAP = {"wq": "to_q", "wk": "to_k", "wv": "to_v",
             "wo": "to_out.0", "gate": "to_gate"}
# Feed-forward leaf names that keep their name under the `ff.` parent.
_FF_LEAVES = {"gate", "up", "down"}

# Distinctive native keys used to recognise the format (fail-closed: all must
# be present-shaped before we claim it's a ComfyUI Krea-2 checkpoint).
_KREA2_NATIVE_MARKERS = ("blocks.0.attn.wq.weight", "blocks.0.mod.lin")


def _strip_known_prefix(key: str) -> str:
    for pfx in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(pfx):
            return key[len(pfx):]
    return key


def is_krea2_comfy_checkpoint(keys: Iterable[str]) -> bool:
    """True iff `keys` look like a ComfyUI-native Krea-2 transformer (the
    `blocks.N.attn.wq` / `blocks.N.mod.lin` signature). Prefix-tolerant."""
    stripped = {_strip_known_prefix(k) for k in keys}
    return all(m in stripped for m in _KREA2_NATIVE_MARKERS)


def _convert_block_internal(tail: str) -> str:
    """Rename the part after `blocks.{i}.` / `txtfusion.*.{i}.`."""
    m = re.match(r"attn\.(\w+)(.*)$", tail)
    if m:
        sub, rest = m.groups()
        if sub in _ATTN_MAP:
            return f"attn.{_ATTN_MAP[sub]}{rest}"
        if sub == "qknorm":
            if rest == ".qnorm.scale":
                return "attn.norm_q.weight"
            if rest == ".knorm.scale":
                return "attn.norm_k.weight"
    m = re.match(r"mlp\.(\w+)(.*)$", tail)
    if m:
        sub, rest = m.groups()
        if sub in _FF_LEAVES:
            return f"ff.{sub}{rest}"
    if tail == "mod.lin":
        return "scale_shift_table"
    if tail == "prenorm.scale":
        return "norm1.weight"
    if tail == "postnorm.scale":
        return "norm2.weight"
    return tail


def convert_krea2_comfy_key(key: str) -> str:
    """Map ONE ComfyUI-native Krea-2 key to its diffusers name. Prefix already
    stripped by the caller (`convert_krea2_comfy_state_dict`). Unmatched keys
    return unchanged (the loader's missing-key assertion surfaces any gap)."""
    # ── main transformer blocks ──
    m = re.match(r"^blocks\.(\d+)\.(.+)$", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.{_convert_block_internal(m.group(2))}"
    # ── text-fusion blocks (layerwise / refiner) ──
    m = re.match(r"^txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$", key)
    if m:
        grp, idx, tail = m.groups()
        return f"text_fusion.{grp}.{idx}.{_convert_block_internal(tail)}"
    if key.startswith("txtfusion."):            # e.g. txtfusion.projector.weight
        return "text_fusion." + key[len("txtfusion."):]
    # ── image input / final layer ──
    if key.startswith("first."):
        return "img_in." + key[len("first."):]
    if key == "last.norm.scale":
        return "final_layer.norm.weight"
    if key == "last.modulation.lin":
        return "final_layer.scale_shift_table"
    if key.startswith("last.linear."):
        return "final_layer.linear." + key[len("last.linear."):]
    # ── time embed (tmlp.0/2) + time modulation projection (tproj) ──
    m = re.match(r"^tmlp\.(\d+)\.(.+)$", key)
    if m and m.group(1) in ("0", "2"):
        return f"time_embed.linear_{'1' if m.group(1) == '0' else '2'}.{m.group(2)}"
    if key.startswith("tproj."):                # tproj.0.<leaf> → time_mod_proj.<leaf>
        return "time_mod_proj." + key.split(".", 2)[2]
    # ── text input embed (txtmlp.0=norm, .1=linear_1, .3=linear_2) ──
    m = re.match(r"^txtmlp\.(\d+)\.(.+)$", key)
    if m:
        idx, rest = m.groups()
        if idx == "0" and rest == "scale":
            return "txt_in.norm.weight"
        if idx == "1":
            return f"txt_in.linear_1.{rest}"
        if idx == "3":
            return f"txt_in.linear_2.{rest}"
    return key


def convert_krea2_comfy_state_dict(
    state_dict: Dict[str, "object"], strip_prefix: str | None = None,
) -> Dict[str, "object"]:
    """Return a NEW dict with ComfyUI-native Krea-2 keys renamed to diffusers.
    Values are passed through by reference (no copy). `strip_prefix`, when
    given, is removed before conversion (the dominant SGM prefix detected by
    the loader, e.g. ``model.diffusion_model.``).

    Self-guards against control characters in key names (rejects without
    echoing the raw key) so the module is safe even if a caller invokes it
    without the slice-C loader's upstream guard (security-auditor INFO,
    2026-07-07)."""
    out: Dict[str, object] = {}
    for k, v in state_dict.items():
        if any(ord(c) < 0x20 for c in k):
            raise ValueError(
                "control character in checkpoint key name — refusing")
        # Prefer the caller's detected dominant prefix; fall back to the
        # robust per-key strip (a TE+VAE BUNDLE dilutes the transformer prefix
        # below the loader's 50% dominance threshold, so strip_prefix is None
        # there even though the transformer keys ARE `model.diffusion_model.`-
        # prefixed).
        if strip_prefix and k.startswith(strip_prefix):
            nk = k[len(strip_prefix):]
        else:
            nk = _strip_known_prefix(k)
        out[convert_krea2_comfy_key(nk)] = v
    return out


# Only these targets ever need a numel-safe reshape (native block modulation
# `mod.lin` is a FLAT (6*dim,) scale_shift_table; diffusers stores (6, dim)).
_RESHAPE_TARGET_SUFFIX = "scale_shift_table"


def reshape_to_model_shapes(state_dict, model_shapes):
    """Reshape converted tensors whose NUMEL matches the target model param but
    whose stored shape differs — restricted to ``*scale_shift_table`` targets.

    The allowlist is a safety bound (code-review finding 2): a future
    mis-renamed weight that happens to share a param's numel does NOT get
    silently reshaped into it — it falls through and PyTorch's own
    size-mismatch check raises at load. `reshape` (not `view`) is copy-safe.
    Mutates and returns `state_dict`. `model_shapes` maps diffusers key →
    tuple shape."""
    import torch
    for k in list(state_dict):
        ms = model_shapes.get(k)
        if ms is None or not k.endswith(_RESHAPE_TARGET_SUFFIX):
            continue
        t = state_dict[k]
        ms = tuple(ms)
        if tuple(t.shape) != ms and t.numel() == int(torch.tensor(ms).prod()):
            state_dict[k] = t.reshape(ms)
    return state_dict


class Krea2ConversionError(ValueError):
    """Raised when the ComfyUI→diffusers Krea-2 conversion is incomplete
    (missing/unexpected model params) — fail-closed rather than generate from
    partially random-init weights."""


# Top-level namespaces of BUNDLED (non-transformer) components some community
# single-files ship alongside the transformer (e.g. Dark Beast = Qwen3-VL TE +
# VAE + transformer). When loading as a --transformer override we keep only the
# transformer.
_NON_TRANSFORMER_ROOTS = frozenset(
    ("text_encoders", "vae", "conditioner", "cond_stage_model",
     "first_stage_model", "text_encoder", "text_encoder_2"))
# Native Krea-2 transformer top-level namespaces (prefix already stripped).
_TRANSFORMER_ROOT_PREFIXES = (
    "blocks.", "txtfusion.", "first.", "last.", "tmlp.", "tproj.", "txtmlp.")


def is_krea2_bundle(keys: Iterable[str]) -> bool:
    """True iff the checkpoint bundles non-transformer components (TE/VAE)
    alongside the transformer."""
    return any(k.split(".", 1)[0] in _NON_TRANSFORMER_ROOTS for k in keys)


def extract_krea2_transformer_sd(state_dict):
    """If `state_dict` bundles TE/VAE, return ONLY the transformer sub-dict.
    Non-bundles are returned unchanged. Identifies transformer keys by their
    native namespace AFTER stripping any known prefix — robust to a bundle
    whose transformer prefix isn't the dominant one."""
    if not is_krea2_bundle(state_dict.keys()):
        return state_dict
    return {k: v for k, v in state_dict.items()
            if _strip_known_prefix(k).startswith(_TRANSFORMER_ROOT_PREFIXES)}


def build_krea2_transformer(component_class, native_sd, config_path, dtype,
                            strip_prefix=None, log_prefix="[Krea2]"):
    """Build a diffusers Krea-2 transformer from a ComfyUI-native state dict
    (values already dequantized/upcast to `dtype`).

    Shared by the scaled-fp8 loader (after dequant) and the general single-file
    path (after fp8→bf16 upcast / bf16 as-is). Handles: bundle extraction, key
    conversion, scale_shift_table reshape, strict load, fp32-norm restoration.
    Raises `Krea2ConversionError` on ANY missing/unexpected key. Returns the
    model on CPU (dtype, with `_keep_in_fp32_modules` restored to fp32)."""
    import torch
    native_sd = extract_krea2_transformer_sd(native_sd)
    diff_sd = convert_krea2_comfy_state_dict(native_sd, strip_prefix)
    config = component_class.load_config(config_path, local_files_only=True)
    model = component_class.from_config(config)
    mshapes = {n: tuple(p.shape) for n, p in
               list(model.named_parameters()) + list(model.named_buffers())}
    reshape_to_model_shapes(diff_sd, mshapes)
    incompat = model.load_state_dict(diff_sd, strict=False, assign=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        _miss = sorted(incompat.missing_keys)
        raise Krea2ConversionError(
            f"Krea-2 key conversion mismatch "
            f"({len(incompat.missing_keys)} missing / "
            f"{len(incompat.unexpected_keys)} unexpected model params; rename "
            f"table incomplete for this checkpoint) — e.g. missing "
            f"{_miss[0] if _miss else '-'!r}; refusing rather than generate "
            f"from random-init weights")
    model = model.to(dtype)
    # assign=True coerced _keep_in_fp32_modules norms to `dtype`; from_pretrained
    # keeps them fp32 for stability — restore that (base-model precision parity).
    keep = getattr(component_class, "_keep_in_fp32_modules", None) or []
    if keep:
        n_fp32 = 0
        for name, mod in model.named_modules():
            if name.rsplit(".", 1)[-1] in keep:
                mod.to(torch.float32)
                n_fp32 += 1
        print(f"{log_prefix} restored {n_fp32} _keep_in_fp32_modules to fp32")
    return model
