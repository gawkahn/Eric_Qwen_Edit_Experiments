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
requirements of docs/security/review-slice-C-fp8-single-file-2026-07-02.md
plus the C-d delta review (review-slice-Cd-comfy-quant-2026-07-02.md):
header-only classification via safe_open (names + dtypes + shapes, never
__metadata__) — with EXACTLY ONE bounded exception (delta req 11): the raw
bytes of `.comfy_quant` descriptor tensors may be read, gated by
header-declared dtype U8, 1-D shape, and numel <= 4096 BEFORE any tensor
materialization; the decoded JSON drives EXACTLY ONE decision — the format
allowlist pass/fail (delta req 15; `full_precision_matrix_mult` and every
other field is attacker-controlled and deliberately NOT read; reading any
new descriptor field requires a fresh security review). Also: control-char
key rejection; scalar-F32 scale shape/dtype asserts; finite-positive-normal
scale validation at load (before EITHER compute path); per-tensor
scale-coverage check; and scale↔weight binding by SOURCE pairing + value
fingerprint — scales are never re-attached by name through a
canonicalization step, so the F1 collision class is closed by construction.

C-d variants (comfy_quant descriptor next to the C-a scale layout):
  cq-a  descriptor + weight_scale + input_scale on EVERY fp8 weight —
        identical semantics to C-a (scaled _scaled_mm compute)
  cq-w  descriptor + weight_scale ONLY on EVERY fp8 weight — weight-only
        quantization; ScaledFp8Linear runs the dequant-matmul path,
        _scaled_mm is never called. Variant is inferred from input_scale
        tensor PRESENCE, never from the descriptor JSON (delta D5/D6).
"""

from __future__ import annotations

import json
import os
import torch
import torch.nn as nn

_FP8_DTYPES = ("F8_E4M3", "F8_E5M2")
_TORCH_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)

#: Suffix conventions for the two scaled variants. A file must use exactly
#: one convention (security review F5 — mixing is a loud reject).
_CA_SUFFIXES = {"weight_scale": ".weight_scale", "input_scale": ".input_scale"}
_CB_SUFFIXES = {"weight_scale": ".scale_weight", "input_scale": ".scale_input"}

#: comfy_quant descriptor handling (slice C-d, delta review reqs 11-17).
_CQ_SUFFIX = ".comfy_quant"
_CQ_MAX_NUMEL = 4096          # header-gated cap BEFORE materialization (D2)
_CQ_FORMAT_ALLOWLIST = {"float8_e4m3fn", "float8_e5m2"}  # literal equality
_CQ_KNOWN_FIELDS = {"format", "full_precision_matrix_mult"}

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
        k.endswith((_CQ_SUFFIX,)
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

    # (comfy_quant descriptors are handled below, after the sharper nvfp4
    # signature check — slice C-d replaced the former blanket reject.)

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

    cq_keys = [k for k in keys if k.endswith(_CQ_SUFFIX)]
    if cq_keys:
        return _classify_cq(path, keys, dtypes, shapes, fp8_keys,
                            cq_keys, cb_hits)

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


def _classify_cq(path: str, keys, dtypes, shapes, fp8_keys, cq_keys, cb_hits):
    """Classify a comfy_quant-descriptor file (slice C-d, delta reqs 11-17).

    Returns ("cq-a" | "cq-w", info) or raises ScaledFp8FormatError. The
    descriptor JSON drives EXACTLY ONE decision — the format allowlist
    (delta req 15). Variant (cq-a vs cq-w) is inferred from input_scale
    tensor PRESENCE, never from descriptor fields (D5/D6).
    """
    # D3 extension: cq never coexists with C-b signals.
    if cb_hits:
        raise ScaledFp8FormatError(
            f"weight file mixes comfy_quant descriptors with C-b scale "
            f"markers (e.g. {_safe_name(cb_hits[0])}) — refusing (D3)"
        )

    # D2/req 11: header-side gates BEFORE any tensor materialization.
    for k in cq_keys:
        if dtypes[k] != "U8" or len(shapes[k]) != 1 \
                or not (1 <= shapes[k][0] <= _CQ_MAX_NUMEL):
            raise ScaledFp8FormatError(
                f"descriptor {_safe_name(k)} is {dtypes[k]} shape "
                f"{shapes[k]} — descriptors must be 1-D U8 with at most "
                f"{_CQ_MAX_NUMEL} bytes; refusing without reading it (D2)"
            )

    # D1: every descriptor pairs with an fp8 weight at the same base.
    key_set = set(keys)
    cq_bases = set()
    for k in cq_keys:
        base = k[: -len(_CQ_SUFFIX)]
        wk = base + ".weight"
        if wk not in key_set or dtypes.get(wk) not in _FP8_DTYPES:
            raise ScaledFp8FormatError(
                f"descriptor {_safe_name(k)} has no fp8 weight at "
                f"{_safe_name(wk)} — dangling descriptors are refused (D1)"
            )
        cq_bases.add(base)

    # D3: EVERY fp8 weight carries a descriptor, or none do (none is the
    # plain ca/cb/cc path, which never reaches here).
    fp8_bases = {k[: -len(".weight")] for k in fp8_keys
                 if k.endswith(".weight")}
    naked = sorted(fp8_bases - cq_bases)
    if naked:
        raise ScaledFp8FormatError(
            f"file mixes descriptor-carrying and plain fp8 layers (e.g. "
            f"{_safe_name(naked[0] + '.weight')} has no descriptor while "
            f"{_safe_name(sorted(cq_bases)[0] + _CQ_SUFFIX)} exists) — "
            f"refusing (D3)"
        )

    # req 14: bounded read + strict parse. One decision: format allowlist.
    from safetensors import safe_open
    formats = set()
    unknown_fields = set()
    with safe_open(path, framework="pt") as f:
        for k in cq_keys:
            raw = bytes(f.get_tensor(k).numpy().tobytes())
            try:
                text = raw.decode("utf-8", errors="strict")
                desc = json.loads(text.strip())
            except (UnicodeDecodeError, ValueError) as e:
                raise ScaledFp8FormatError(
                    f"descriptor {_safe_name(k)} is not valid UTF-8 JSON "
                    f"({type(e).__name__}) — refusing (D4)"
                ) from None
            if not isinstance(desc, dict):
                raise ScaledFp8FormatError(
                    f"descriptor {_safe_name(k)} JSON root is "
                    f"{type(desc).__name__}, expected object — refusing (D4)"
                )
            fmt = desc.get("format")
            if not isinstance(fmt, str) or fmt not in _CQ_FORMAT_ALLOWLIST:
                raise ScaledFp8FormatError(
                    f"descriptor {_safe_name(k)} declares format "
                    f"{_safe_name(str(fmt))} — not in the supported set "
                    f"{sorted(_CQ_FORMAT_ALLOWLIST)} (fp4/nvfp4 formats are "
                    f"deferred; see ADR-019)"
                )
            formats.add(fmt)
            unknown_fields.update(set(desc) - _CQ_KNOWN_FIELDS)

    # D8: one aggregate line, not per-layer; unknown fields surfaced once.
    note = (f" (unknown descriptor fields ignored: "
            f"{sorted(_safe_name(u) for u in unknown_fields)})"
            if unknown_fields else "")
    print(f"[EricDiffusion-fp8] comfy_quant: {len(cq_keys)} descriptors, "
          f"formats {sorted(formats)}{note}")

    # D6/req 16: variant from tensor presence — all-or-none input_scale.
    w_sfx = _CA_SUFFIXES["weight_scale"]
    i_sfx = _CA_SUFFIXES["input_scale"]
    missing_ws = sorted(b for b in cq_bases if b + w_sfx not in key_set)
    if missing_ws:
        raise ScaledFp8FormatError(
            f"fp8 weight {_safe_name(missing_ws[0] + '.weight')} has a "
            f"descriptor but no {w_sfx} — refusing (F6)"
        )
    with_in = {b for b in cq_bases if b + i_sfx in key_set}
    if with_in and with_in != cq_bases:
        one_with = sorted(with_in)[0]
        one_without = sorted(cq_bases - with_in)[0]
        raise ScaledFp8FormatError(
            f"file mixes input_scale coverage ({_safe_name(one_with)} has "
            f"one, {_safe_name(one_without)} does not) — a legitimate file "
            f"is all-or-none; refusing (D6)"
        )

    # D7/req 17: header-time scalar-F32 sanity on all cq scale keys.
    for k in keys:
        if k.endswith((w_sfx, i_sfx)):
            if dtypes[k] != "F32" or shapes[k] not in ((), (1,)):
                raise ScaledFp8FormatError(
                    f"scale {_safe_name(k)} is {dtypes[k]} shape "
                    f"{shapes[k]} — only per-tensor SCALAR F32 scales are "
                    f"supported (D7/F3)"
                )

    info = {"n_keys": len(keys), "n_fp8": len(fp8_keys),
            "dtypes": dtypes, "shapes": shapes,
            "formats": sorted(formats)}
    return ("cq-a" if with_in else "cq-w"), info


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
                 input_scale: torch.Tensor | None, bias: torch.Tensor | None):
        super().__init__()
        if weight_fp8.dtype not in _TORCH_FP8:
            raise ScaledFp8FormatError(
                f"ScaledFp8Linear requires an fp8 weight, got {weight_fp8.dtype}"
            )
        self.out_features, self.in_features = weight_fp8.shape
        self.register_buffer("weight", weight_fp8)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        # input_scale=None → weight-only mode (cq-w, delta req 18): forward
        # ALWAYS takes the dequant-matmul path; _scaled_mm is never called.
        # fp8 storage is retained (VRAM win); a bf16 dequant copy is cached
        # on first forward (D9 — the compute-time cost of weight-only files).
        self.register_buffer(
            "input_scale",
            input_scale.to(torch.float32) if input_scale is not None else None)
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

    def _dequant_linear(self, x2d: torch.Tensor, in_dtype) -> torch.Tensor:
        """Shared dequant-matmul path: weight-only mode + _scaled_mm fallback."""
        if (self._fallback_weight is None
                or self._fallback_weight.device != x2d.device):
            self._fallback_weight = (
                self.weight.to(torch.float32) * self.weight_scale
            ).to(in_dtype)
        return torch.nn.functional.linear(
            x2d, self._fallback_weight.to(in_dtype),
            self.bias.to(in_dtype) if self.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        lead_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features)
        if self.input_scale is None:
            # Weight-only mode (cq-w): by design, not a fallback (req 18).
            out = self._dequant_linear(x2d, in_dtype)
            return out.reshape(*lead_shape, self.out_features).to(in_dtype)
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
            out = self._dequant_linear(x2d, in_dtype)
        return out.reshape(*lead_shape, self.out_features).to(in_dtype)

    def extra_repr(self) -> str:
        mode = "weight-only" if self.input_scale is None else "scaled"
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, fp8_resident=True, "
                f"mode={mode}")


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

    # ca, cq-a and cq-w all use the C-a suffix convention; only cb differs.
    suffixes = _CB_SUFFIXES if variant == "cb" else _CA_SUFFIXES
    w_sfx = suffixes["weight_scale"]
    i_sfx = suffixes["input_scale"]
    # cq-w (weight-only) is the ONLY variant without a per-tensor
    # input_scale requirement (delta D6/req 16 — classification already
    # guaranteed no input_scale exists anywhere in the file).
    require_input = variant in ("ca", "cb", "cq-a")

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
    cq_keys = {k for k in sd if k.endswith(_CQ_SUFFIX)}
    fp8_entries = {}   # base source key -> (fp8 weight, w_scale, i_scale|None)
    for k, t in sd.items():
        if k in scale_keys or k in cq_keys or t.dtype not in _TORCH_FP8:
            continue
        if not k.endswith(".weight"):
            raise ScaledFp8FormatError(
                f"fp8 tensor {_safe_name(k)} is not a .weight — unsupported layout"
            )
        base = k[: -len(".weight")]
        ws_key, is_key = base + w_sfx, base + i_sfx
        if ws_key not in sd or (require_input and is_key not in sd):
            raise ScaledFp8FormatError(
                f"fp8 weight {_safe_name(k)} lacks its paired scales "
                f"({_safe_name(ws_key)}"
                + (f" / {_safe_name(is_key)}" if require_input else "")
                + ") — partial scale coverage is refused (security review F6)"
            )
        if not require_input and is_key in sd:
            # Classification said weight-only; an input_scale here means the
            # file changed or the caller mislabeled the variant (D6).
            raise ScaledFp8FormatError(
                f"variant {variant} is weight-only but {_safe_name(is_key)} "
                f"exists — inconsistent with classification; refusing (D6)"
            )
        _validate_scale(ws_key, sd[ws_key])
        if require_input:
            _validate_scale(is_key, sd[is_key])
        if t.dim() != 2:
            raise ScaledFp8FormatError(
                f"fp8 weight {_safe_name(k)} has {t.dim()}D shape "
                f"{tuple(t.shape)} — only 2D Linear weights are supported"
            )
        fp8_entries[base] = (t, sd[ws_key],
                             sd[is_key] if require_input else None)
    # Per-suffix stray binding: an input-suffix key only counts as bound
    # when the variant expects input scales (so a stray input_scale in a
    # weight-only file rejects rather than floating unbound).
    stray = []
    for k in scale_keys:
        if k.endswith(w_sfx):
            bound = k[: -len(w_sfx)] in fp8_entries
        else:
            bound = require_input and k[: -len(i_sfx)] in fp8_entries
        if not bound:
            stray.append(k)
    if stray:
        raise ScaledFp8FormatError(
            f"scale key {_safe_name(sorted(stray)[0])} has no fp8 weight to "
            f"bind to — refusing (dangling scales; security review F1/F6)"
        )
    # D1 at load: every descriptor must sit on a loaded fp8 entry.
    for k in sorted(cq_keys):
        if k[: -len(_CQ_SUFFIX)] not in fp8_entries:
            raise ScaledFp8FormatError(
                f"descriptor {_safe_name(k)} has no fp8 weight to bind to "
                f"— refusing (dangling descriptor; delta D1)"
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
        if k in scale_keys or k in cq_keys:
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
