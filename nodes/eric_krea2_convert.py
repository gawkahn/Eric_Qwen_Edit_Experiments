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
        nk = k[len(strip_prefix):] if strip_prefix and k.startswith(strip_prefix) else k
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
