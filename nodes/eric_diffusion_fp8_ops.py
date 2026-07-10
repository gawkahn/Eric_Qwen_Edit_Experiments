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

Partial-quant (slice PQ, security review PQ-1..6 / reqs 31-38): a cq file
MAY also carry "naked" plain-fp8 layers ALONGSIDE the descriptored set —
ComfyUI partial-quant leaves peripheral projections (img_in, final_layer,
time/text embeds) unscaled while quantizing the repeated block Linears. The
D3 rule is therefore PER-LAYER, not whole-file: each fp8 base is EITHER
fully descriptored+scaled OR fully bare (no scale of any kind, no
descriptor); any partial coverage is a loud reject. Naked layers upcast to
bf16 (the reviewed `cc` cast). "Naked" is defined by scale-ABSENCE, never
descriptor-absence, so a present scale is never silently discarded. No
naked-fraction cap — airtightness comes from the fully-bare reject, not
from bounding the fraction (a mostly-naked file just loads as a bf16 model).

Slice I8 (ci-w, security review reqs 46-56): comfy_quant descriptors may
instead pair with INT8 weights — ComfyUI-core int8_tensorwise (unrotated).
Flavor derives from paired-weight DTYPES before any JSON read (D5); the
int8 descriptor then must be EXACTLY {"format": "int8_tensorwise"} — a
STRICT field allowlist, because this schema is demonstrated to carry
weight-SEMANTICS modifiers (ConvRot rotations): unknown fields reject,
never log-and-ignore (the fp8 D8 rule does not apply to this flavor, and
convrot-prefixed fields reject on fp8 descriptors too, I8-4). No naked-
int8 analog of PQ exists — raw int8 bytes have no float interpretation,
so every I8 tensor must be a fully-paired 2D .weight (both classify and
load enforce it; the stage-2 pass-through refuses int8 in all variants).
int8 scales are {BF16,F32}, either scalar or per-output-channel (rows, 1)
pinned to the bound weight's row count — "tensorwise" names comfy-kitchen's
layout class, not the scale granularity (review Amendment 2026-07-10, req
57); any other shape broadcasts along the wrong axis and rejects. fp8
scales stay scalar F32-only, via a separate validator. ci-w has NO
residency op (no Int8Linear): it always dequantizes to bf16 — with
--quant fp8, torchao re-quantizes downstream. Marker-less all-int8 files
never enter this parser (entry condition) and keep their standard-path
behavior (invariant 2).

Slice R1/R2 (security review R1R2R3, reqs 39-45): the per-layer rule extends
to NON-`.weight` fp8 tensors (norm scales, biases, modulation tables —
aggressive ComfyUI exports cast these to fp8 too): fully bare → upcast to
bf16; any bound scale/descriptor → loud reject (req 39). Residual assumption
(req 43): non-`.weight` fp8 tensors are assumed unscaled under all
conventions — a hypothetical format binding a scale to one under a name
outside the recognized suffix set would upcast at scale 1.0; airtightness
comes from the binding reject, not from proving the class. `dequant_fp8=True`
(req 40) returns the all-bf16 model after step 2, skipping only the
validation-free residency swap; in that mode `weight_scale` is applied by the
dequant and `input_scale` is validated-then-DROPPED (req 42) — an
activation-quant param correctly superseded by torchao dynamic activation
quantization when --quant re-quantizes downstream.
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

    # D1 (amended — slice I8, req 46): every descriptor pairs with a
    # QUANTIZED weight at the same base — fp8 (cq flavors) or int8 (ci-w).
    # The FLAVOR derives from the paired tensor DTYPES exclusively, computed
    # BEFORE any descriptor JSON is read (D5: attacker JSON never selects
    # the flavor; the per-flavor format allowlist then cross-checks it).
    key_set = set(keys)
    cq_bases = set()
    _paired = {}   # base -> header dtype of its .weight
    for k in cq_keys:
        base = k[: -len(_CQ_SUFFIX)]
        wk = base + ".weight"
        wdt = dtypes.get(wk)
        if wk not in key_set or wdt not in _FP8_DTYPES + ("I8",):
            raise ScaledFp8FormatError(
                f"descriptor {_safe_name(k)} has no quantized weight at "
                f"{_safe_name(wk)} (found: {wdt or 'missing'}) — dangling "
                f"descriptors are refused (D1)"
            )
        _paired[base] = wdt
        cq_bases.add(base)
    if any(d == "I8" for d in _paired.values()):
        if any(d != "I8" for d in _paired.values()):
            _bi = next(b for b, d in _paired.items() if d == "I8")
            _bf = next(b for b, d in _paired.items() if d != "I8")
            raise ScaledFp8FormatError(
                f"file mixes int8 ({_safe_name(_bi)}) and fp8 "
                f"({_safe_name(_bf)}) descriptored weights — a legitimate "
                f"file uses one quant flavor; refusing (I8 req 46)"
            )
        return _classify_ci(path, keys, dtypes, shapes, cq_keys, cq_bases,
                            key_set)
    # fp8 flavor + any I8 tensor anywhere = ambiguous/crafted now that I8
    # carries meaning in this parser; the loader re-asserts (req 49/54).
    _i8_stray = [k for k in keys if dtypes[k] == "I8"]
    if _i8_stray:
        raise ScaledFp8FormatError(
            f"fp8-flavored comfy_quant file carries int8 tensor "
            f"{_safe_name(_i8_stray[0])} — refusing (I8 req 54)"
        )

    # D3 (amended — slice PQ, security review PQ-1 / req 31): a cq file MAY
    # carry "naked" plain-fp8 layers ALONGSIDE the descriptored set — ComfyUI
    # partial-quant commonly leaves the peripheral projections (img_in,
    # final_layer, time/text embeds) unscaled while quantizing the repeated
    # block Linears. The old all-or-nothing rule becomes PER-LAYER: each fp8
    # base is EITHER fully descriptored+scaled OR fully bare. A naked base
    # carrying ANY scale or descriptor is the ambiguous/crafted case and is
    # refused — "naked" is defined by scale-ABSENCE so a PRESENT scale is
    # never silently ignored (the D6 present-value argument, applied per
    # layer). No naked-fraction cap: airtightness comes from this reject, not
    # from bounding the fraction (req 36). The cq + C-b co-presence reject at
    # the top of this function is preserved untouched (req 33).
    fp8_bases = {k[: -len(".weight")] for k in fp8_keys
                 if k.endswith(".weight")}
    naked = sorted(fp8_bases - cq_bases)
    _naked_forbidden = (_CA_SUFFIXES["weight_scale"], _CA_SUFFIXES["input_scale"],
                        _CB_SUFFIXES["weight_scale"], _CB_SUFFIXES["input_scale"],
                        _CQ_SUFFIX)
    for base in naked:
        for sfx in _naked_forbidden:
            if base + sfx in key_set:
                raise ScaledFp8FormatError(
                    f"descriptor-less fp8 layer {_safe_name(base + '.weight')} "
                    f"carries {_safe_name(base + sfx)} — a naked fp8 layer must "
                    f"be FULLY bare (no scale or descriptor of any kind); "
                    f"partial coverage is refused (D3/PQ-1)"
                )
    if naked:
        preview = sorted(_safe_name(b) for b in naked)[:4]
        print(f"[EricDiffusion-fp8] comfy_quant: {len(naked)} plain-fp8 "
              f"(naked) layer(s) upcast to bf16: {preview}"
              + ("..." if len(naked) > 4 else ""))

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
            # I8-4 (folded into slice I8 with declared scope, req 52):
            # convrot-prefixed fields declare a ROTATION baked into the
            # weights — a semantic modifier, not metadata. Log-and-ignore
            # would dequantize rotated weights UNROTATED (silent garbage),
            # so all flavors reject. Other unknown fields keep the shipped
            # D8 log-and-ignore below.
            _rot = sorted(f for f in desc
                          if isinstance(f, str) and f.startswith("convrot"))
            if _rot:
                raise ScaledFp8FormatError(
                    f"descriptor {_safe_name(k)} declares "
                    f"{_safe_name(_rot[0])} — ConvRot-rotated checkpoints "
                    f"are unsupported (the rotation must be applied at "
                    f"compute; loading unrotated is silent corruption). "
                    f"See TECH_DEBT 'INT8-ConvRot'. (I8-4)"
                )
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


#: int8-tensorwise scale dtypes (header names). BF16 is the observed
#: ComfyUI convention; F32 is a strict-superset-in-precision emitter
#: variation. fp8 flavors stay F32-only (D7 untouched). (I8 req 50 — the
#: dtype clause only; the scale SHAPE rule is req 57, see _is_ci_scale_shape.)
_CI_SCALE_DTYPES = ("BF16", "F32")

#: Smallest positive NORMAL float32. Subnormal scales flush to zero on most
#: tensor-core paths, so they are rejected alongside 0/NaN/Inf (finding F2).
_F32_MIN_NORMAL = 1.1754943508222875e-38


def _is_ci_scale_shape(shape, rows: int) -> bool:
    """True iff `shape` is a legal int8-tensorwise weight-scale shape.

    Legal: scalar (`()` / `(1,)`) or per-output-channel `(rows, 1)`, where
    `rows` is the bound weight's dim-0. `int8_tensorwise` names comfy-kitchen's
    LAYOUT class, not the scale granularity — upstream `quantize_int8_rowwise`
    emits `[..., 1]` per-row scales under the same format string, and
    `dequantize_int8_simple` is `q.float() * scale` (req 57).

    Everything else rejects, and the near-misses are the whole point: a
    `(1, in_features)` scale broadcasts COLUMN-wise and a bare `(rows,)` scale
    broadcasts along the LAST axis — both silently, with no shape error, both
    corrupting the weight. Shape is pinned exactly rather than admitting any
    `numel() > 1` vector.
    """
    shape = tuple(shape)
    return shape in ((), (1,)) or shape == (rows, 1)


def _validate_ci_scale(name: str, t: torch.Tensor, rows: int) -> None:
    """Shape + elementwise value validation for an int8-tensorwise scale.

    The ci-w counterpart to `_validate_scale`, kept SEPARATE rather than
    parameterizing that function further: `_validate_scale` stays scalar-only
    and F32-only for every fp8/cq/DMR-requant call site, so no future edit to
    the int8 path can loosen an fp8 gate (req 58; original finding F3 is an
    fp8 finding and stands).

    Dtype is checked on the RAW tensor before upcast (I8-5). The
    finite-positive-normal policy (F2) then applies to EVERY element — one
    poisoned row scale is one corrupted output row (req 59).
    """
    if t.dtype not in (torch.bfloat16, torch.float32):
        raise ScaledFp8FormatError(
            f"scale {_safe_name(name)} has dtype {t.dtype} — int8_tensorwise "
            f"scales must be bfloat16/float32 (F3/req 50)"
        )
    if not _is_ci_scale_shape(t.shape, rows):
        raise ScaledFp8FormatError(
            f"scale {_safe_name(name)} has shape {tuple(t.shape)} for a weight "
            f"with {rows} rows — int8_tensorwise scales must be SCALAR or "
            f"per-output-channel ({rows}, 1); any other shape broadcasts along "
            f"the wrong axis and silently corrupts the weight (req 57)"
        )
    v = t.to(torch.float32)
    bad = ~(torch.isfinite(v) & (v >= _F32_MIN_NORMAL))
    if bool(bad.any()):
        i = int(torch.nonzero(bad.reshape(-1), as_tuple=False)[0])
        raise ScaledFp8FormatError(
            f"scale {_safe_name(name)} element [{i}] is "
            f"{v.reshape(-1)[i].item()!r} — not finite-positive-normal; "
            f"refusing to load (silent numerical corruption guard, F2/req 59)"
        )


def _classify_ci(path: str, keys, dtypes, shapes, cq_keys, cq_bases,
                 key_set):
    """Classify a comfy_quant INT8-tensorwise file (slice I8, reqs 46-56).

    Returns ("ci-w", info) or raises ScaledFp8FormatError. Dispatched from
    _classify_cq when the descriptors pair with I8 weights (flavor from
    tensor DTYPES, decided before any JSON read — D5). The descriptor JSON
    confirms exactly one thing: `{"format": "int8_tensorwise"}`, with a
    STRICT field allowlist (req 51) — the int8 schema is demonstrated to
    carry semantic-modifier fields (convrot rotations), so unknown fields
    reject rather than log-and-ignore (D8 does NOT apply to this flavor).

    v1 rules (security review I8-1..I8-9):
      - weight-only: any `.input_scale` anywhere refuses (req 47)
      - no fp8 tensors anywhere in an int8-flavored file (req 54)
      - EVERY I8 tensor must be a 2D `.weight` with descriptor + a {BF16,F32}
        scalar-or-(rows,1) weight_scale (req 48; shape rule is req 57) — no
        naked-int8 analog of PQ (raw int8 bytes have no float
        interpretation; "upcast" is garbage)
    """
    w_sfx = _CA_SUFFIXES["weight_scale"]
    i_sfx = _CA_SUFFIXES["input_scale"]

    # req 47 — v1 is weight-only.
    _in = sorted(k for k in keys if k.endswith(i_sfx))
    if _in:
        raise ScaledFp8FormatError(
            f"int8_tensorwise file carries {_safe_name(_in[0])} — int8 "
            f"support is weight-only (v1); input_scale is refused (req 47)"
        )

    # req 54 — bidirectional flavor exclusion: no fp8 tensors here.
    _f8 = sorted(k for k in keys if dtypes[k] in _FP8_DTYPES)
    if _f8:
        raise ScaledFp8FormatError(
            f"int8-flavored comfy_quant file carries fp8 tensor "
            f"{_safe_name(_f8[0])} — refusing (I8 req 54; the PQ naked-fp8 "
            f"coexistence does not extend to int8 files without real-file "
            f"evidence)"
        )

    # req 48 — coverage matrix: every I8 tensor is a 2D .weight with
    # descriptor + weight_scale; every other cell rejects, none upcast.
    for k in keys:
        if dtypes[k] != "I8":
            continue
        if not k.endswith(".weight"):
            raise ScaledFp8FormatError(
                f"int8 tensor {_safe_name(k)} is not a .weight — raw int8 "
                f"has no float interpretation; refusing (req 48)"
            )
        base = k[: -len(".weight")]
        if base not in cq_bases:
            raise ScaledFp8FormatError(
                f"int8 weight {_safe_name(k)} has no comfy_quant "
                f"descriptor — naked int8 is refused (no valid upcast "
                f"exists; req 48)"
            )
        if base + w_sfx not in key_set:
            raise ScaledFp8FormatError(
                f"int8 weight {_safe_name(k)} lacks {_safe_name(base + w_sfx)} "
                f"— refusing (F6 analog, req 48)"
            )
        if len(shapes[k]) != 2:
            raise ScaledFp8FormatError(
                f"int8 weight {_safe_name(k)} has shape {shapes[k]} — only "
                f"2D Linear weights are supported (req 48/I8-7)"
            )

    # req 57 — header scale rule: {BF16,F32}, scalar OR per-output-channel
    # (rows, 1) bound to its weight's row count. Supersedes the original
    # scalar-only req 50 (see review Amendment 2026-07-10).
    for k in keys:
        if k.endswith(w_sfx):
            base = k[: -len(w_sfx)]
            if base not in cq_bases:
                raise ScaledFp8FormatError(
                    f"scale key {_safe_name(k)} has no descriptored int8 "
                    f"weight to bind to — refusing (dangling scale, F1/F6)"
                )
            w_shape = shapes.get(base + ".weight")
            if w_shape is None:
                raise ScaledFp8FormatError(
                    f"scale key {_safe_name(k)} has a descriptor but no "
                    f"{_safe_name(base + '.weight')} to bind to — refusing "
                    f"(dangling scale, F1/F6)"
                )
            if dtypes[k] not in _CI_SCALE_DTYPES:
                raise ScaledFp8FormatError(
                    f"scale {_safe_name(k)} is {dtypes[k]} — int8_tensorwise "
                    f"scales must be {'/'.join(_CI_SCALE_DTYPES)} (req 50)"
                )
            if not _is_ci_scale_shape(shapes[k], w_shape[0]):
                raise ScaledFp8FormatError(
                    f"scale {_safe_name(k)} has shape {shapes[k]} for a "
                    f"weight with {w_shape[0]} rows — int8_tensorwise scales "
                    f"must be SCALAR or per-output-channel ({w_shape[0]}, 1); "
                    f"any other shape broadcasts along the wrong axis and "
                    f"silently corrupts the weight (req 57)"
                )

    # req 51 — bounded descriptor read (D2 gates already ran) with a STRICT
    # field allowlist: exactly {"format"}, format exactly "int8_tensorwise".
    # This closes convrot AND every future semantic modifier by construction
    # — and enforces the req-46 cross-consistency (an I8-paired descriptor
    # declaring an fp8 format rejects here).
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        for k in cq_keys:
            raw = bytes(f.get_tensor(k).numpy().tobytes())
            try:
                desc = json.loads(raw.decode("utf-8", errors="strict").strip())
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
            if set(desc) != {"format"}:
                _extra = sorted(_safe_name(str(x))
                                for x in set(desc) - {"format"}) or ["<none>"]
                raise ScaledFp8FormatError(
                    f"int8 descriptor {_safe_name(k)} carries fields beyond "
                    f"'format' (e.g. {_extra[0]}) — the int8_tensorwise "
                    f"schema is known to encode weight-semantics modifiers "
                    f"(ConvRot rotations), so unknown fields are refused, "
                    f"not ignored (req 51/I8-3)"
                )
            if desc.get("format") != "int8_tensorwise":
                raise ScaledFp8FormatError(
                    f"descriptor {_safe_name(k)} pairs with an int8 weight "
                    f"but declares format "
                    f"{_safe_name(str(desc.get('format')))} — dtype/format "
                    f"mismatch; refusing (req 46)"
                )

    print(f"[EricDiffusion-fp8] comfy_quant: {len(cq_keys)} int8_tensorwise "
          f"descriptors (ci-w, weight-only) — dequant-to-bf16 loader")
    info = {"n_keys": len(keys), "n_fp8": 0, "n_int8": len(cq_bases),
            "dtypes": dtypes, "shapes": shapes,
            "formats": ["int8_tensorwise"]}
    return "ci-w", info


def _validate_scale(name: str, t: torch.Tensor,
                    allowed_dtypes=(torch.float32,)) -> None:
    """Scalar + finite-and-positive scale validation at load (F2/F3/F12).

    Runs ONCE, before either compute path (scaled_mm or bf16 dequant
    fallback) can consume the value. `allowed_dtypes` DEFAULTS to F32-only
    so every fp8/DMR call site keeps the shipped D7 rule without edits
    (I8 review req 50). The dtype is checked on the RAW tensor before any
    upcast — upcast-then-validate would erase the gate (I8-5).

    This is the fp8 validator: scalar-only, F32-only. int8-tensorwise scales
    may be per-output-channel and go through `_validate_ci_scale` instead —
    deliberately a separate function so the fp8 gates here cannot be loosened
    by a future edit to the int8 path (req 58).
    """
    if t.numel() != 1:
        raise ScaledFp8FormatError(
            f"scale {_safe_name(name)} has shape {tuple(t.shape)} — only "
            f"per-tensor SCALAR scales are supported (per-channel vectors "
            f"are rejected; see security review F3)"
        )
    if t.dtype not in allowed_dtypes:
        raise ScaledFp8FormatError(
            f"scale {_safe_name(name)} has dtype {t.dtype} — expected one "
            f"of {tuple(str(d) for d in allowed_dtypes)} (F3/req 50)"
        )
    v = t.to(torch.float32).item()
    # Reject non-finite, non-positive, AND subnormal values (reviewer
    # finding: F2 lists denormals alongside 0/NaN/Inf — a subnormal scale
    # feeding _scaled_mm flushes to zero on most tensor-core paths).
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
                or self._fallback_weight.device != x2d.device
                or self._fallback_weight.dtype != in_dtype):
            # Keyed on (device, dtype) — reviewer finding 3: a dtype change
            # mid-run (bf16 → fp16 caller) must recompute from the fp8
            # source, not chain-cast through the stale cache.
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
                              log_prefix: str = "[EricDiffusion-fp8]",
                              dequant_fp8: bool = False):
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
    # Slice R1 / req 39+41 (security review R1R2R3): aggressively-quantized
    # ComfyUI exports cast NON-Linear params to fp8 too — norm scales
    # (qknorm/prenorm/postnorm), modulation tables (mod.lin), biases. These
    # are never GEMM weights; a FULLY-BARE one (no scale or descriptor bound
    # under any recognized convention) upcasts to bf16 in stage 2 like any
    # naked tensor. "Bare" is defined by binding-ABSENCE, never by "it isn't
    # a .weight" — a present binding is a loud reject (the PQ-2 present-value
    # principle; silently discarding a bound scale would corrupt ~448x).
    # Skipped tensors never enter fp8_entries, so they can never reach the
    # step-3 residency swap. Enumerated in one aggregate log line below.
    _NONWEIGHT_BINDINGS = (_CA_SUFFIXES["weight_scale"],
                           _CA_SUFFIXES["input_scale"],
                           _CB_SUFFIXES["weight_scale"],
                           _CB_SUFFIXES["input_scale"],
                           _CQ_SUFFIX)
    nonweight_naked = []
    for k, t in sd.items():
        if k in scale_keys or k in cq_keys:
            continue
        # ── ci-w (int8-tensorwise) pairing — slice I8 reqs 48/49/50/54 ──
        # Loader RE-ASSERTION of the classify rules (PQ-2 two-point
        # pattern): every int8 tensor must be a fully-paired 2D .weight;
        # no fp8 tensor may appear in an int8 file. None of the fp8
        # naked/non-weight relaxations (PQ/R1) apply — raw int8 has no
        # float interpretation, so nothing int8 may fall through to the
        # stage-2 upcast.
        if variant == "ci-w":
            if t.dtype in _TORCH_FP8:
                raise ScaledFp8FormatError(
                    f"int8-flavored file carries fp8 tensor {_safe_name(k)} "
                    f"— refusing (I8 req 54)"
                )
            if t.dtype != torch.int8:
                continue
            if not k.endswith(".weight"):
                raise ScaledFp8FormatError(
                    f"int8 tensor {_safe_name(k)} is not a .weight — "
                    f"refusing (req 48)"
                )
            base = k[: -len(".weight")]
            ws_key = base + w_sfx
            if ws_key not in sd or base + _CQ_SUFFIX not in sd:
                raise ScaledFp8FormatError(
                    f"int8 weight {_safe_name(k)} lacks its descriptor/"
                    f"weight_scale pairing — naked int8 is refused "
                    f"(req 48/49)"
                )
            if t.dim() != 2:
                raise ScaledFp8FormatError(
                    f"int8 weight {_safe_name(k)} has {t.dim()}D shape "
                    f"{tuple(t.shape)} — only 2D Linear weights are "
                    f"supported (req 48)"
                )
            # req 57 — re-assert the scale's row binding against the ACTUAL
            # weight tensor, not just the header (PQ-2 two-point pattern).
            # Runs after the 2D check so `t.shape[0]` is the row count.
            _validate_ci_scale(ws_key, sd[ws_key], t.shape[0])
            fp8_entries[base] = (t, sd[ws_key], None)
            continue
        if t.dtype not in _TORCH_FP8:
            continue
        if not k.endswith(".weight"):
            for sfx in _NONWEIGHT_BINDINGS:
                if k + sfx in sd:
                    raise ScaledFp8FormatError(
                        f"non-.weight fp8 tensor {_safe_name(k)} carries "
                        f"{_safe_name(k + sfx)} — a bound scale/descriptor on "
                        f"a non-Linear fp8 tensor is refused rather than "
                        f"silently ignored (R1/req 39)"
                    )
            nonweight_naked.append(k)
            continue
        base = k[: -len(".weight")]
        ws_key, is_key = base + w_sfx, base + i_sfx
        # slice PQ / req 32 (security review PQ-2/PQ-5): a FULLY-bare fp8 base
        # — no weight_scale, no input_scale, no comfy_quant descriptor — is a
        # plain fp8-cast layer (ComfyUI partial-quant leaves peripheral
        # projections unscaled). Skip it here so stage 2 upcasts it to bf16
        # (the reviewed `cc` path). "Naked" is defined by scale-ABSENCE, NEVER
        # descriptor-absence: a base carrying a scale but no descriptor must
        # fall through to the F6/D6 checks below — never silently upcast,
        # which would DISCARD a present scale (~448x corruption). This sits
        # AFTER the `.weight` check above (req 35) so a non-.weight fp8 tensor
        # still raises rather than being skipped.
        if ws_key not in sd and is_key not in sd \
                and base + _CQ_SUFFIX not in sd:
            continue
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
    _kind = "int8-tensorwise" if variant == "ci-w" else "scaled-fp8"
    print(f"{log_prefix} {len(fp8_entries)} {_kind} Linears in file "
          f"(variant {variant})")
    if nonweight_naked:
        # req 41 — one aggregate line, never silent (PQ-4 pattern).
        _pv = sorted(_safe_name(k) for k in nonweight_naked)[:4]
        print(f"{log_prefix} {len(nonweight_naked)} non-.weight fp8 tensor(s) "
              f"(norm/bias/modulation) upcast to bf16: {_pv}"
              + ("..." if len(nonweight_naked) > 4 else ""))

    # ── 2. Dequantize → let diffusers build the model ────────────────────
    bf16_sd = {}
    for k, t in sd.items():
        if k in scale_keys or k in cq_keys:
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base in fp8_entries:
            w, ws, _ = fp8_entries[base]
            # ws upcast is explicit so BF16 int8-scales (ci-w) and F32
            # fp8-scales dequantize identically in f32.
            bf16_sd[k] = (w.to(torch.float32) * ws.to(torch.float32)).to(dtype)
        else:
            # slice I8 req 49: NO int8 tensor may take the raw pass-through
            # — load_state_dict's copy_() would silently cast integer values
            # into float params (integer-valued garbage weights, no error).
            # Active for ALL variants routed through this loader; only
            # reachable on classify/loader divergence (direct calls).
            if t.dtype == torch.int8:
                raise ScaledFp8FormatError(
                    f"unpaired int8 tensor {_safe_name(k)} reached the "
                    f"upcast stage — raw int8 has no float interpretation; "
                    f"refusing (I8 req 49)"
                )
            bf16_sd[k] = t.to(dtype) if t.is_floating_point() else t
    del sd

    # ComfyUI-native Krea-2 checkpoints (community civitai single-file) use key
    # names released diffusers 0.39.0 has no from_single_file converter for.
    # build_krea2_transformer converts + builds (shared with the general
    # single-file path so plain-fp8/bf16/bundle native-Krea files load too).
    # (ADR-019 Changelog 2026-07-07; nodes/eric_krea2_convert.py.)
    from .eric_krea2_convert import (
        is_krea2_comfy_checkpoint, build_krea2_transformer,
    )
    if is_krea2_comfy_checkpoint(bf16_sd.keys()):
        print(f"{log_prefix} ComfyUI-native Krea-2 keys — converting to "
              f"diffusers + from_config build")
        model = build_krea2_transformer(
            component_class, bf16_sd, config_path, dtype, strip_prefix, log_prefix)
        del bf16_sd
    else:
        # Classes whose only construction path is a bespoke converter (Krea2)
        # have no from_single_file. Reaching here with one means the file's
        # keys did not match that converter's signature — i.e. the checkpoint
        # is a DIFFERENT architecture than the pipeline expects. Say that,
        # rather than letting `AttributeError: no attribute 'from_single_file'`
        # surface and read as a diffusers bug.
        if not hasattr(component_class, "from_single_file"):
            raise ScaledFp8FormatError(
                f"{component_class.__name__} has no from_single_file, and this "
                f"checkpoint's keys do not match its native-format converter — "
                f"the file is very likely a different architecture than the "
                f"pipeline expects. Check that the transformer override matches "
                f"the base model's family."
            )
        model = component_class.from_single_file(
            bf16_sd, config=config_path, torch_dtype=dtype, local_files_only=True,
        )

    # Slice R2 / req 40 (security review R1R2R3): dequant-to-bf16 mode —
    # return the clean bf16 model HERE, skipping only step 3 (the residency
    # swap, which contains zero validation). Steps 1-2 above ran byte-
    # identically: every scale was validated (weight_scale applied in the
    # dequant; input_scale validated-then-dropped, correctly superseded by
    # torchao dynamic activation quant downstream). Used when --quant is
    # active so quantize_module receives plain nn.Linear modules and
    # re-quantizes into torchao Float8Tensor — the representation the DMR
    # LoRA merge is proven on (reqs 21-30).
    # slice I8 req 53: ci-w is UNCONDITIONAL dequant — there is no int8
    # residency op (no Int8Linear exists), so the model returns here
    # regardless of dequant_fp8. With --quant fp8 the downstream torchao
    # quantize re-quantizes (the proven LoRA-compatible path).
    if variant == "ci-w":
        print(f"{log_prefix} int8-tensorwise dequantized to bf16 "
              f"(no int8 residency by design; --quant fp8 re-quantizes "
              f"via torchao)")
        return model
    if dequant_fp8:
        print(f"{log_prefix} dequant-fp8 mode: all-bf16 model returned "
              f"(no fp8 residency; --quant re-quantizes via torchao)")
        return model

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


# ════════════════════════════════════════════════════════════════════════
#  Slice DMR — dequant→merge→requant (ADR-019 §4 amendment, 2026-07-03)
#  Security contract: docs/security/review-slice-DMR-quantized-merge-2026-07-03.md
#  requirements 21-30. Direct-merge adapters (LoKR/LoHa direct, converted
#  .diff, tier-3 LoRA) now apply onto quantized bases instead of raising.
# ════════════════════════════════════════════════════════════════════════

_E5M2_MAX = 57344.0

_PLAIN_MERGE_DTYPES = (torch.bfloat16, torch.float16, torch.float32,
                       torch.float64)


def _fp8_max_for(dtype) -> float:
    return _E4M3_MAX if dtype == torch.float8_e4m3fn else _E5M2_MAX


def _owner_of(root, target_key: str):
    """Resolve (owner_module, leaf_name) for a dotted target key."""
    owner_path, _, leaf = target_key.rpartition(".")
    try:
        owner = root.get_submodule(owner_path) if owner_path else root
    except AttributeError:
        return None, leaf
    return owner, leaf


def merge_resolution_map(root) -> dict:
    """name→tensor map for direct-merge target resolution.

    Parameters first, then `.weight`-named buffers override (delta review
    DMR-8: the merge order is BINDING). The `.weight`-only filter is a
    security control, not a convenience (DMR-3 / req 23): unfiltered
    buffers would let an adversarial LoRA key like `foo.weight_scale.diff`
    resolve onto a ScaledFp8Linear SCALE buffer and silently poison the
    layer's quantization via the plain merge path.
    """
    m = dict(root.named_parameters())
    m.update({k: v for k, v in root.named_buffers()
              if k.endswith(".weight")})
    return m


def apply_merge_delta(root, target_key: str, delta: torch.Tensor,
                      backup: dict, log_prefix: str = "[LoRA]") -> str:
    """Merge a weight delta into ANY supported base representation.

    The single write path for direct-merge adapters (req 24: every merge
    site routes through here; none touch param.data directly). Returns the
    kind applied: "plain" | "torchao" | "scaled_fp8". Raises RuntimeError
    (actionable, ADR-019 §4 wording) for representations it cannot merge —
    the fail-loud NARROWS, it never lapses (Vision invariant 4/7).

    Backup entries: plain targets keep the legacy raw bf16 clone (so
    existing restore code is untouched); quantized targets store
    kind-tagged dicts holding the ORIGINAL representation verbatim for
    exact restore (Vision invariant 2).
    """
    # req 21 — a non-finite delta on a quantized path would poison the
    # PERSISTED scale (NaN amax → NaN weight_scale → whole tensor gone);
    # gate before any dispatch.
    if not torch.isfinite(delta).all():
        bad = (~torch.isfinite(delta)).sum().item()
        raise RuntimeError(
            f"{log_prefix} adapter delta for {_safe_name(target_key)} "
            f"contains {bad} non-finite value(s) — refusing to merge "
            f"(scale-poisoning guard, DMR review req 21)"
        )

    owner, leaf = _owner_of(root, target_key)
    if owner is None:
        raise RuntimeError(
            f"{log_prefix} cannot resolve owner module for "
            f"{_safe_name(target_key)} — refusing to merge"
        )

    # ── req 24 precedence: positive matches, then explicit raise ────────
    # (a) ScaledFp8Linear fp8 weight buffer
    if isinstance(owner, ScaledFp8Linear) and leaf == "weight":
        return _merge_into_scaled_fp8(owner, target_key, delta, backup,
                                      log_prefix)

    t = getattr(owner, leaf, None)
    if t is None:
        raise RuntimeError(
            f"{log_prefix} no tensor at {_safe_name(target_key)} — "
            f"refusing to merge"
        )
    data = t.data if isinstance(t, nn.Parameter) else t

    # (b) torchao-quantized Parameter
    if (isinstance(t, nn.Parameter)
            and type(data).__module__.startswith("torchao")):
        return _merge_into_torchao(root, owner, leaf, t, target_key, delta,
                                   backup, log_prefix)

    # (c) orphan fp8 tensor — a rep we don't own; plain add_ would corrupt
    if isinstance(data, torch.Tensor) and data.dtype in _TORCH_FP8:
        raise RuntimeError(
            f"{log_prefix} {_safe_name(target_key)} is an fp8 tensor not "
            f"owned by a ScaledFp8Linear — unsupported quantized "
            f"representation; refusing to merge (ADR-019 §4 / DMR-4)"
        )

    # (d) plain high-precision tensor — byte-identical legacy behavior
    if isinstance(data, torch.Tensor) and data.dtype in _PLAIN_MERGE_DTYPES:
        if target_key not in backup:
            backup[target_key] = data.clone()
        data.add_(delta.to(dtype=data.dtype, device=data.device))
        return "plain"

    # (e) anything else — explicit raise, never a fallthrough
    raise RuntimeError(
        f"{log_prefix} {_safe_name(target_key)} has unsupported dtype "
        f"{getattr(data, 'dtype', type(data))} for direct merge — "
        f"refusing (ADR-019 §4 / DMR-4)"
    )


def _merge_into_scaled_fp8(owner: "ScaledFp8Linear", target_key: str,
                           delta: torch.Tensor, backup: dict,
                           log_prefix: str) -> str:
    if target_key not in backup:
        backup[target_key] = {
            "kind": "scaled_fp8",
            "weight": owner.weight.detach().clone(),
            "weight_scale": owner.weight_scale.detach().clone(),
        }
    dev = owner.weight.device
    W = (owner.weight.to(torch.float32) * owner.weight_scale
         + delta.to(device=dev, dtype=torch.float32))
    if not torch.isfinite(W).all():
        raise RuntimeError(
            f"{log_prefix} merged weights for {_safe_name(target_key)} are "
            f"non-finite (overflow in base+delta) — refusing (req 21/30)"
        )
    fp8_max = _fp8_max_for(owner.weight.dtype)
    amax = W.abs().amax()
    if amax.item() == 0.0:
        # req 22 — all-zero merged tensor: sentinel scale, zero content,
        # loud log; never divide by zero.
        print(f"{log_prefix} WARNING: merged weights for "
              f"{_safe_name(target_key)} are ALL ZERO — storing sentinel "
              f"scale 1.0 (req 22); the adapter or base is degenerate")
        new_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        wq = torch.zeros_like(owner.weight)
    else:
        new_scale = (amax / fp8_max).to(torch.float32)
        # req 30 — the requant OUTPUT scale passes the same
        # finite-positive-normal policy as load-time scales. Nothing has
        # been persisted yet, so a failure here is a clean raise.
        _validate_scale(f"{target_key} [requant]", new_scale)
        old = owner.weight_scale.item()
        ratio = new_scale.item() / old if old > 0 else float("inf")
        if ratio > 1000:
            print(f"{log_prefix} WARNING: adapter coarsens "
                  f"{_safe_name(target_key)} quantization scale {ratio:.0f}x "
                  f"— adapter likely corrupt or crafted; proceeding "
                  f"(operator-initiated; req 27)")
        elif ratio > 2:
            print(f"{log_prefix} note: {_safe_name(target_key)} requant "
                  f"scale coarsened {ratio:.1f}x by the merge (amax grew)")
        wq = (W / new_scale).clamp(-fp8_max, fp8_max).to(owner.weight.dtype)
    owner.weight = wq
    owner.weight_scale = new_scale
    owner._fallback_weight = None  # req 29 sibling: stale cache = corruption
    return "scaled_fp8"


def _merge_into_torchao(root, owner, leaf: str, p: nn.Parameter,
                        target_key: str, delta: torch.Tensor, backup: dict,
                        log_prefix: str) -> str:
    if target_key not in backup:
        # The ORIGINAL Parameter object, retained verbatim — exact restore
        # by swap (Vision invariant 2). fp8-sized, cheaper than bf16 clones.
        backup[target_key] = {"kind": "torchao_param", "param": p}
    data = p.data
    W = data.dequantize() if hasattr(data, "dequantize") \
        else data.to(torch.float32)
    merged = W.to(torch.float32) + delta.to(device=W.device,
                                            dtype=torch.float32)
    if not torch.isfinite(merged).all():
        raise RuntimeError(
            f"{log_prefix} merged weights for {_safe_name(target_key)} are "
            f"non-finite (overflow in base+delta) — refusing (req 21/30)"
        )
    if merged.abs().amax().item() == 0.0:
        # req 22 defensive posture for case 2: torchao owns its scale
        # internally, so the sentinel-scale trick isn't expressible here.
        # An all-zero merged layer is degenerate either way — raise loud.
        raise RuntimeError(
            f"{log_prefix} merged weights for {_safe_name(target_key)} are "
            f"ALL ZERO — refusing to requantize a degenerate layer (req 22)"
        )
    from torchao.quantization import quantize_
    try:
        # Source of truth: the slice-A quantize-on-load recipe, so merged
        # layers match the surrounding quantization scheme exactly.
        from .eric_diffusion_utils import _torchao_fp8_config
        _cfg = _torchao_fp8_config()
    except ImportError:
        # Spec-loaded contexts (test harnesses load this module by file
        # path, no package). MUST stay in sync with
        # eric_diffusion_utils._torchao_fp8_config.
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
        )
        _cfg = Float8DynamicActivationFloat8WeightConfig()
    out_f, in_f = merged.shape
    tmp = nn.Linear(in_f, out_f, bias=False, device=merged.device,
                    dtype=torch.bfloat16)
    tmp.weight = nn.Parameter(merged.to(torch.bfloat16),
                              requires_grad=False)
    quantize_(tmp, _cfg)
    new_p = tmp.weight
    setattr(owner, leaf, new_p)
    # req 26 — post-swap aliasing assert: the key must resolve to the NEW
    # object; a stale reachable object means tied weights or an offload
    # hook holds the old one (documented unsupported assumption).
    live = dict(root.named_parameters()).get(target_key)
    if live is not new_p:
        raise RuntimeError(
            f"{log_prefix} Parameter swap for {_safe_name(target_key)} did "
            f"not take effect — an alias (weight tying / offload hook) "
            f"still holds the old Parameter (DMR-6); model state is "
            f"inconsistent, reload the pipeline"
        )
    return "torchao"


def restore_merge_backup(root, live_key: str, entry: dict,
                         log_prefix: str = "[LoRA]") -> bool:
    """Restore a kind-tagged quantized backup — verbatim swap (exact)."""
    owner, leaf = _owner_of(root, live_key)
    if owner is None:
        print(f"{log_prefix} WARNING: cannot resolve {_safe_name(live_key)} "
              f"for quantized-backup restore — skipped")
        return False
    kind = entry.get("kind")
    if kind == "scaled_fp8":
        if not isinstance(owner, ScaledFp8Linear):
            print(f"{log_prefix} WARNING: {_safe_name(live_key)} owner is "
                  f"no longer a ScaledFp8Linear — restore skipped")
            return False
        dev = owner.weight.device
        owner.weight = entry["weight"].to(dev)
        owner.weight_scale = entry["weight_scale"].to(dev)
        # req 29 — a stale dequant cache would keep serving MERGED weights
        # after "restore"; invalidate both cache and its warn latch.
        owner._fallback_weight = None
        owner._warned_fallback = False
        return True
    if kind == "torchao_param":
        old_p = entry["param"]
        cur = getattr(owner, leaf, None)
        if cur is not None and old_p.device != cur.device:
            old_p = nn.Parameter(old_p.data.to(cur.device),
                                 requires_grad=False)
        setattr(owner, leaf, old_p)
        return True
    print(f"{log_prefix} WARNING: unknown backup kind {kind!r} for "
          f"{_safe_name(live_key)} — restore skipped")
    return False


def record_direct_merge(root, adapter_name: str) -> None:
    """Append to the transformer's merge-order ledger (LIFO guard, req 25)."""
    if not hasattr(root, "_eric_direct_merge_order"):
        root._eric_direct_merge_order = []
    root._eric_direct_merge_order.append(adapter_name)


def warn_non_lifo_unload(root, adapter_name: str,
                         log_prefix: str = "[LoRA]") -> None:
    """Warn (don't block) when direct-merge unload order isn't LIFO (req 25).

    Direct-merge backups snapshot the state at merge time; restoring adapter
    A while a later adapter B is still merged reverts B's delta on shared
    layers. That was always true of the plain-path backups — the ledger just
    makes the footgun LOUD.
    """
    order = getattr(root, "_eric_direct_merge_order", None)
    if not order or adapter_name not in order:
        return
    if order[-1] != adapter_name:
        print(f"{log_prefix} WARNING: unloading direct-merge adapter "
              f"{adapter_name!r} out of LIFO order (most recent is "
              f"{order[-1]!r}) — later adapters' deltas on shared layers "
              f"will be reverted too (req 25)")
    order.remove(adapter_name)


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
