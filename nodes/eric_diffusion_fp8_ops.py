# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""ComfyUI scaled-fp8 single-file support (ADR-019 slice C).

Consumes community fp8 checkpoints on the transformer-override path:

  C-a  scaled, ``weight_scale`` / ``input_scale`` F32 scalars per Linear
       (e.g. flux-2-klein-base-9b-fp8.safetensors, BFL key layout)
  C-b  scaled, ``scale_weight`` / ``scale_input`` F32 scalars + global
       ``scaled_fp8`` marker (e.g. wan2.2 *_fp8_scaled)
  C-c  plain fp8 cast, no scales — needs no code here (from_single_file
       + torch_dtype upcasts it already); the classifier returns "cc" so
       callers can leave it on the standard path.

Weights stay fp8-resident where possible: after diffusers builds the bf16
model from our dequantized dict, eligible Linears are swapped for
``ScaledFp8Linear`` whose forward runs ``torch._scaled_mm`` (native fp8
tensor cores, sm89+).

SECURITY: this module is the "custom parsing of caller-supplied weight-file
content" surface added to the repo Review bar by ADR-019 slice C. Every
change here is security-auditor-gated. The load path implements the binding
requirements of docs/security/review-slice-C-fp8-single-file-2026-07-02.md:
header-only classification via safe_open (names + dtypes ONLY, never
__metadata__); control-char key rejection; scalar-F32 scale shape/dtype
asserts; finite-and-positive scale validation at load (before EITHER compute
path); per-tensor scale-coverage check; and scale↔weight binding by SOURCE
pairing + value fingerprint — scales are never re-attached by name through a
canonicalization step, so the F1 collision class is closed by construction.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn

_FP8_DTYPES = ("F8_E4M3", "F8_E5M2")
_TORCH_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)

#: Suffix conventions for the two scaled variants. A file must use exactly
#: one convention (security review F5 — mixing is a loud reject).
_CA_SUFFIXES = {"weight_scale": ".weight_scale", "input_scale": ".input_scale"}
_CB_SUFFIXES = {"weight_scale": ".scale_weight", "input_scale": ".scale_input"}

#: Keys that mark formats we deliberately do NOT parse (nvfp4 block layouts,
#: ComfyUI's newer comfy_quant metadata blobs). Presence → loud reject from
#: header inspection only, before any tensor materialization (F11).
_UNSUPPORTED_MARKERS = (".comfy_quant",)

_E4M3_MAX = 448.0


def _safe_name(k: str) -> str:
    """Sanitize an attacker-controlled tensor name for logs/errors (F7)."""
    return repr(k[:200])


class ScaledFp8FormatError(ValueError):
    """Loud, actionable rejection of a scaled-fp8 file we won't parse."""


def classify_fp8_single_file(path: str):
    """Header-only classification of a single weight file.

    Returns one of:
      ("ca"|"cb", info)  — scaled fp8, parseable by load_scaled_fp8_component
      ("cc", info)       — plain fp8 cast, standard path handles it
      (None, info)       — not an fp8 file at all; standard path
    Raises ScaledFp8FormatError for fp8-marked files we refuse to parse
    (mixed conventions, unsupported markers, control-char keys, wrong
    extension) — rejection happens from the header alone (F10/F11).

    Uses safetensors.safe_open only; classification reads tensor NAMES and
    DTYPES exclusively — never __metadata__ (F9).
    """
    if not path.lower().endswith(".safetensors"):
        # Non-safetensors single files never enter the fp8 parser. If they
        # contain fp8 the standard torch.load path will surface it; we do
        # not extend parsing to pickle-format inputs (F10).
        return None, {}

    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        dtypes = {}
        shapes = {}
        for k in keys:
            sl = f.get_slice(k)
            dtypes[k] = sl.get_dtype()
            shapes[k] = tuple(sl.get_shape())

    fp8_keys = [k for k in keys if dtypes[k] in _FP8_DTYPES]
    _any_marker = any(
        k.endswith(tuple(_UNSUPPORTED_MARKERS)
                   + tuple(_CA_SUFFIXES.values())
                   + tuple(_CB_SUFFIXES.values()))
        or k == "scaled_fp8"
        for k in keys
    )
    if not fp8_keys and not _any_marker:
        # Not an fp8 file: return untouched so the standard path stays
        # byte-identical (invariant 2) — no further inspection of any kind.
        return None, {}

    # Control-char key rejection for files ENTERING the fp8 parser (F7);
    # defense-in-depth — the loader re-checks at strip time.
    for k in keys:
        if any(ord(c) < 0x20 for c in k):
            raise ScaledFp8FormatError(
                f"weight file contains a tensor name with control "
                f"characters ({_safe_name(k)}) — refusing to parse"
            )

    for k in keys:
        if any(k.endswith(m) for m in _UNSUPPORTED_MARKERS):
            raise ScaledFp8FormatError(
                f"weight file carries {_safe_name(k)} — this quantization "
                f"layout (comfy_quant metadata / possibly nvfp4 block "
                f"format) is not supported. Supported: scaled fp8 with "
                f"per-tensor weight_scale/input_scale (or "
                f"scale_weight/scale_input) F32 scalars, or plain fp8 "
                f"casts. nvfp4 is deferred — see ADR-019 §Deferred."
            )

    # nvfp4 signature: packed U8 weights + F8 BLOCK-scale vectors + a
    # second-level `weight_scale_2` F32 scalar (observed layout:
    # weight U8 [out,in/2] / weight_scale F8_E4M3 [out,blocks] /
    # weight_scale_2 F32 []). Reject from the header, before any tensor
    # materialization (F11).
    if any(k.endswith(".weight_scale_2") for k in keys):
        raise ScaledFp8FormatError(
            "weight file carries .weight_scale_2 second-level scales — "
            "this is the nvfp4 block-quantized layout, which is deferred "
            "(ADR-019 §Deferred / TECH_DEBT.md). Use an fp8 or bf16 "
            "version of this checkpoint."
        )

    ca_hits = [k for k in keys if k.endswith(tuple(_CA_SUFFIXES.values()))]
    cb_hits = [k for k in keys
               if k.endswith(tuple(_CB_SUFFIXES.values())) or k == "scaled_fp8"]

    if ca_hits and cb_hits:
        raise ScaledFp8FormatError(
            f"weight file mixes both scale-suffix conventions "
            f"(e.g. {_safe_name(ca_hits[0])} and {_safe_name(cb_hits[0])}) "
            f"— a legitimate file uses exactly one; refusing to parse (F5)"
        )

    if not fp8_keys and not ca_hits and not cb_hits:
        return None, {}

    # Header-time scale sanity for scaled variants: every scale key must be
    # a SCALAR F32 (block/vector scales = nvfp4-family layouts → reject
    # before load, F3/F11). Load-time _validate_scale re-checks values.
    for k in (ca_hits + [h for h in cb_hits if h != "scaled_fp8"]):
        if dtypes[k] != "F32" or shapes[k] not in ((), (1,)):
            raise ScaledFp8FormatError(
                f"scale {_safe_name(k)} is {dtypes[k]} shape "
                f"{shapes[k]} — only per-tensor SCALAR F32 scales are "
                f"supported (block/per-channel scale layouts, e.g. nvfp4, "
                f"are deferred; see ADR-019)"
            )

    info = {"n_keys": len(keys), "n_fp8": len(fp8_keys),
            "dtypes": dtypes, "shapes": shapes}
    if ca_hits:
        return "ca", info
    if cb_hits:
        return "cb", info
    return "cc", info


def _validate_scale(name: str, t: torch.Tensor) -> None:
    """Scalar-F32 + finite-and-positive scale validation at load (F2/F3/F12).

    Runs ONCE, before either compute path (scaled_mm or bf16 dequant
    fallback) can consume the value.
    """
    if t.numel() != 1:
        raise ScaledFp8FormatError(
            f"scale {_safe_name(name)} has shape {tuple(t.shape)} — only "
            f"per-tensor SCALAR scales are supported (per-channel vectors "
            f"are rejected; see security review F3)"
        )
    if t.dtype != torch.float32:
        raise ScaledFp8FormatError(
            f"scale {_safe_name(name)} has dtype {t.dtype} — F32 required (F3)"
        )
    v = t.item()
    # Reject non-finite, non-positive, AND subnormal values (reviewer
    # finding: F2 lists denormals alongside 0/NaN/Inf — a subnormal scale
    # feeding _scaled_mm flushes to zero on most tensor-core paths).
    _F32_MIN_NORMAL = 1.1754943508222875e-38
    if (not (v > 0.0) or v != v or v == float("inf")
            or v < _F32_MIN_NORMAL):
        raise ScaledFp8FormatError(
            f"scale {_safe_name(name)} value {v!r} is not finite-positive-"
            f"normal — refusing to load (silent numerical corruption "
            f"guard, F2)"
        )


class ScaledFp8Linear(nn.Module):
    """Linear whose weight stays fp8; forward runs torch._scaled_mm.

    Semantics (matches ComfyUI fp8 ops): W_true = W_fp8 · weight_scale;
    x is quantized per-tensor with input_scale; _scaled_mm multiplies the
    two scales back, so the result equals x @ W_true.T + bias in bf16.

    The fp8 weight is registered as a buffer so device moves follow the
    module, but _apply is overridden to REFUSE dtype casts on it — a
    pipeline-level .to(torch.bfloat16) must not silently dequantize the
    model (it would double memory and change numerics vs what was loaded).
    """

    def __init__(self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor,
                 input_scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        if weight_fp8.dtype not in _TORCH_FP8:
            raise ScaledFp8FormatError(
                f"ScaledFp8Linear requires an fp8 weight, got {weight_fp8.dtype}"
            )
        self.out_features, self.in_features = weight_fp8.shape
        self.register_buffer("weight", weight_fp8)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        self.register_buffer("input_scale", input_scale.to(torch.float32))
        if bias is not None:
            # Pre-cast to bf16 once (reviewer finding 7 — a per-forward
            # .to() allocated a fresh tensor every step).
            self.bias = nn.Parameter(bias.to(torch.bfloat16),
                                     requires_grad=False)
        else:
            self.bias = None
        self._warned_fallback = False
        self._fallback_weight = None  # dequant cache, populated on first fallback

    def _apply(self, fn, recurse=True):
        # Detect dtype-converting fns (module.to(bf16), .half(), ...) via a
        # zero-element probe and strip the dtype change for the fp8 buffer:
        # apply device movement only. Scales/bias convert normally.
        probe = fn(torch.empty(0, dtype=self.weight.dtype,
                               device=self.weight.device))
        if probe.dtype != self.weight.dtype:
            w = self.weight
            self.weight = None  # hide from generic _apply
            try:
                super()._apply(fn, recurse)
            finally:
                self.weight = w.to(probe.device)
            return self
        return super()._apply(fn, recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        lead_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features)
        try:
            xq = (x2d.float() / self.input_scale).clamp(
                -_E4M3_MAX, _E4M3_MAX).to(self.weight.dtype)
            m = xq.shape[0]
            pad = (-m) % 16
            if pad:
                xq = torch.nn.functional.pad(xq, (0, 0, 0, pad))
            out = torch._scaled_mm(
                xq, self.weight.t(),
                scale_a=self.input_scale, scale_b=self.weight_scale,
                bias=self.bias if self.bias is not None else None,
                out_dtype=torch.bfloat16,
            )
            if pad:
                out = out[:m]
        except RuntimeError as e:
            # Narrow catch (reviewer finding 6): RuntimeError covers the
            # legitimate fallback cases (unsupported _scaled_mm layout /
            # CPU execution / cuda OOM, which subclasses RuntimeError);
            # anything else propagates rather than silently degrading.
            if not self._warned_fallback:
                print(f"[EricDiffusion-fp8] WARNING: _scaled_mm failed "
                      f"({type(e).__name__}: {e}) — falling back to "
                      f"dequantized matmul for this layer (slower, "
                      f"same numerics)")
                self._warned_fallback = True
            if (self._fallback_weight is None
                    or self._fallback_weight.device != x2d.device):
                # Cache the dequantized weight so the persistent-fallback
                # case doesn't rebuild it every denoising step.
                self._fallback_weight = (
                    self.weight.to(torch.float32) * self.weight_scale
                ).to(in_dtype)
            out = torch.nn.functional.linear(
                x2d, self._fallback_weight.to(in_dtype),
                self.bias.to(in_dtype) if self.bias is not None else None)
        return out.reshape(*lead_shape, self.out_features).to(in_dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, fp8_resident=True")


def _fingerprint(t: torch.Tensor):
    """Cheap value fingerprint for matching a dequantized source tensor to
    the parameter diffusers placed in the constructed model.

    Shape + strided 64-sample vector (exact bitwise values). Used only for
    1:1-renamed tensors, which survive from_single_file bit-identically; a
    transformed tensor (fused/split/transposed) simply fails to match and
    its layer stays bf16 — safe, logged.
    """
    flat = t.reshape(-1)
    stride = max(1, flat.numel() // 64)
    sample = flat[::stride][:64].to(torch.float32).cpu()
    return (tuple(t.shape), sample)


def load_scaled_fp8_component(component_class, weights_path: str, dtype,
                              config_path: str, variant: str,
                              strip_prefix: str | None = None,
                              log_prefix: str = "[EricDiffusion-fp8]"):
    """Load a C-a/C-b scaled-fp8 single file, fp8-resident where possible.

    Flow (Vision §3 as amended by the security review):
      1. Load state dict; pair every fp8 weight with its two scales by
         SOURCE key (no canonicalization); validate each scale (F2/F3);
         per-tensor coverage check (F6).
      2. Dequantize to a bf16 dict and let diffusers from_single_file
         build the model from it — all key conversion happens in
         diffusers, untouched by us.
      3. Swap: for each source fp8 weight, find the model Linear whose
         weight matches the dequantized value fingerprint exactly once;
         replace it with ScaledFp8Linear (fp8 buffer + source scales).
         Ambiguous or missing matches stay bf16, loudly counted.

    Hardware gate: without an sm89+ CUDA device the swap step is skipped
    entirely — the model simply stays bf16 (invariant 5; scales were
    already validated in step 1 per F12).
    """
    from safetensors.torch import load_file as st_load

    if not weights_path.lower().endswith(".safetensors"):
        # F10: the scaled-fp8 loader parses safetensors only. Pickle-format
        # files never enter (the classifier returns None for them), but if
        # a caller routes one here directly, refuse loudly.
        raise ScaledFp8FormatError(
            f"scaled-fp8 loading requires a .safetensors file, got "
            f"{_safe_name(os.path.basename(weights_path))} — re-save the "
            f"checkpoint in safetensors format"
        )

    suffixes = _CA_SUFFIXES if variant == "ca" else _CB_SUFFIXES
    w_sfx = suffixes["weight_scale"]
    i_sfx = suffixes["input_scale"]

    sd = st_load(weights_path)
    if strip_prefix:
        # Same dominant-prefix handling as the standard single-file path
        # (model.diffusion_model. etc.) — mechanical rename BEFORE pairing,
        # so scale↔weight binding operates on the stripped names throughout.
        sd = {(k[len(strip_prefix):] if k.startswith(strip_prefix) else k): v
              for k, v in sd.items()}
    # C-b global marker tensor — metadata, not a weight. Popped AFTER the
    # prefix strip so prefixed layouts (model.diffusion_model.scaled_fp8)
    # are caught too (reviewer finding 3).
    sd.pop("scaled_fp8", None)

    # ── 1. Pair + validate (source-key binding, never re-keyed) ─────────
    for k in sd:
        if any(ord(c) < 0x20 for c in k):
            raise ScaledFp8FormatError(
                f"tensor name with control characters at strip time "
                f"({_safe_name(k)}) — refusing (F7)"
            )
    scale_keys = {k for k in sd if k.endswith((w_sfx, i_sfx))}
    fp8_entries = {}   # base source key -> (fp8 weight, w_scale, i_scale)
    for k, t in sd.items():
        if k in scale_keys or t.dtype not in _TORCH_FP8:
            continue
        if not k.endswith(".weight"):
            raise ScaledFp8FormatError(
                f"fp8 tensor {_safe_name(k)} is not a .weight — unsupported layout"
            )
        base = k[: -len(".weight")]
        ws_key, is_key = base + w_sfx, base + i_sfx
        if ws_key not in sd or is_key not in sd:
            raise ScaledFp8FormatError(
                f"fp8 weight {_safe_name(k)} lacks its paired scales "
                f"({_safe_name(ws_key)} / {_safe_name(is_key)}) — partial "
                f"scale coverage is refused (security review F6)"
            )
        _validate_scale(ws_key, sd[ws_key])
        _validate_scale(is_key, sd[is_key])
        if t.dim() != 2:
            raise ScaledFp8FormatError(
                f"fp8 weight {_safe_name(k)} has {t.dim()}D shape "
                f"{tuple(t.shape)} — only 2D Linear weights are supported"
            )
        fp8_entries[base] = (t, sd[ws_key], sd[is_key])
    stray = [k for k in scale_keys
             if k[: -len(w_sfx)] not in fp8_entries
             and k[: -len(i_sfx)] not in fp8_entries]
    if stray:
        raise ScaledFp8FormatError(
            f"scale key {_safe_name(stray[0])} has no fp8 weight to bind to "
            f"— refusing (dangling scales; security review F1/F6)"
        )
    if not fp8_entries:
        raise ScaledFp8FormatError(
            "classified as scaled-fp8 but no valid fp8 weight/scale "
            "clusters found — refusing"
        )
    print(f"{log_prefix} {len(fp8_entries)} scaled-fp8 Linears in file "
          f"(variant {variant})")

    # ── 2. Dequantize → let diffusers build the model ────────────────────
    bf16_sd = {}
    for k, t in sd.items():
        if k in scale_keys:
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base in fp8_entries:
            w, ws, _ = fp8_entries[base]
            bf16_sd[k] = (w.to(torch.float32) * ws).to(dtype)
        else:
            bf16_sd[k] = t.to(dtype) if t.is_floating_point() else t
    del sd

    model = component_class.from_single_file(
        bf16_sd, config=config_path, torch_dtype=dtype, local_files_only=True,
    )

    # ── 3. Hardware gate, then fingerprint-swap to fp8 residency ────────
    if not (torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) >= (8, 9)):
        print(f"{log_prefix} WARNING: no sm89+ CUDA device — model stays "
              f"bf16 (dequantized; scales validated). No fp8 residency.")
        return model

    # Index model Linears by fingerprint of their (bf16) weights.
    by_fp = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            shape, sample = _fingerprint(mod.weight.data)
            by_fp.setdefault(shape, []).append((name, mod, sample))

    swapped = 0
    unmatched = []
    claimed: set = set()  # model slots already swapped (reviewer finding 2:
    # two sources with byte-identical dequant values must not double-bind
    # one Linear — the second silently overwriting the first's scales)
    parent_cache = dict(model.named_modules())
    for base, (w_fp8, ws, i_s) in fp8_entries.items():
        deq = (w_fp8.to(torch.float32) * ws).to(dtype)
        shape, sample = _fingerprint(deq)
        cands = [(n, m) for (n, m, s) in by_fp.get(shape, [])
                 if n not in claimed
                 and torch.equal(s, sample)
                 and torch.equal(m.weight.data, deq)]
        if len(cands) != 1:
            unmatched.append(base)
            continue
        name, mod = cands[0]
        parent_name, _, child = name.rpartition(".")
        parent = parent_cache[parent_name] if parent_name else model
        new = ScaledFp8Linear(
            w_fp8, ws, i_s,
            mod.bias.data.clone() if mod.bias is not None else None)
        setattr(parent, child, new)
        claimed.add(name)
        swapped += 1

    total = len(fp8_entries)
    print(f"{log_prefix} fp8-resident: {swapped}/{total} Linears swapped "
          f"to ScaledFp8Linear; {total - swapped} stayed bf16 "
          f"(transformed by key conversion — fused/split/transposed)")
    if unmatched and swapped == 0:
        print(f"{log_prefix} WARNING: NO layer matched — the converter "
              f"transformed every tensor; model runs fully bf16 "
              f"(loaded correctly, no fp8 speedup)")
    return model


def contains_scaled_fp8(module) -> bool:
    """isinstance walker for the LoRA guard (security review F8)."""
    if module is None:
        return False
    try:
        for m in module.modules():
            if isinstance(m, ScaledFp8Linear):
                return True
    except (AttributeError, TypeError):
        return False
    return False
