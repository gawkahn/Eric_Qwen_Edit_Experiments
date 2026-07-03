#!/usr/bin/env python3
"""Test harness for the ComfyUI scaled-fp8 single-file loader (ADR-019 slice C).

Implements the binding negative-test list from the design-phase security
review (docs/security/review-slice-C-fp8-single-file-2026-07-02.md,
"Requirements for implementation" item 9), plus positives for the
classifier, ScaledFp8Linear numerics, and the LoRA-guard extension.

All synthetic: builds crafted .safetensors files in a tempdir (fp8 tensors,
malformed scales, hostile names). No GPU required — the _scaled_mm path
self-skips to the dequant fallback off-GPU, and the fallback's numerics are
asserted against a reference matmul. Real-collection classification spot
checks run only when the files exist (skip-as-pass otherwise).
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from safetensors.torch import save_file

_spec = importlib.util.spec_from_file_location(
    "fp8ops", Path(__file__).parent / "nodes" / "eric_diffusion_fp8_ops.py")
fp8ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp8ops)

_spec2 = importlib.util.spec_from_file_location(
    "edu", Path(__file__).parent / "nodes" / "eric_diffusion_utils.py")
edu = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(edu)

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def _fp8(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * 0.02).to(torch.float8_e4m3fn)


def _scalar(v, dtype=torch.float32):
    return torch.tensor(v, dtype=dtype)


_TMP = tempfile.mkdtemp(prefix="fp8_single_file_test_")


def _mk(name, tensors):
    p = os.path.join(_TMP, name)
    save_file(tensors, p)
    return p


def _classify(p):
    return fp8ops.classify_fp8_single_file(p)


def _expect_reject(name, path_or_fn, needle=""):
    """Assert ScaledFp8FormatError is raised, optionally matching text."""
    try:
        if callable(path_or_fn):
            path_or_fn()
        else:
            fp8ops.classify_fp8_single_file(path_or_fn)
        check(name, False, "no ScaledFp8FormatError raised")
    except fp8ops.ScaledFp8FormatError as e:
        ok = needle.lower() in str(e).lower() if needle else True
        check(name, ok, f"message lacked {needle!r}: {str(e)[:100]}")


# ──────────────────────────────────────────────────────────────────────
print("── classifier: variants (positive) ────────────────────────────")

p_ca = _mk("ca.safetensors", {
    "blocks.0.mlp.weight": _fp8((64, 32)),
    "blocks.0.mlp.weight_scale": _scalar(0.01),
    "blocks.0.mlp.input_scale": _scalar(0.5),
    "blocks.0.norm.weight": torch.randn(32, dtype=torch.bfloat16),
})
v, info = _classify(p_ca)
check("C-a classified (weight_scale/input_scale scalars)", v == "ca", f"got {v}")
check("classifier reports fp8 count", info.get("n_fp8") == 1)

p_cb = _mk("cb.safetensors", {
    "blocks.0.attn.k.weight": _fp8((64, 32)),
    "blocks.0.attn.k.scale_weight": _scalar(0.01),
    "blocks.0.attn.k.scale_input": _scalar(0.5),
    "scaled_fp8": torch.zeros(2, dtype=torch.float8_e4m3fn),
})
v, _ = _classify(p_cb)
check("C-b classified (scale_weight/scale_input + marker)", v == "cb", f"got {v}")

p_cc = _mk("cc.safetensors", {
    "blocks.0.mlp.weight": _fp8((64, 32)),
    "blocks.0.mlp.bias": torch.randn(64, dtype=torch.float16),
})
v, _ = _classify(p_cc)
check("C-c classified (plain fp8, no scales)", v == "cc", f"got {v}")

p_bf16 = _mk("bf16.safetensors", {
    "blocks.0.mlp.weight": torch.randn(64, 32, dtype=torch.bfloat16),
})
v, info = _classify(p_bf16)
check("bf16 file → None, untouched path (invariant 2)", v is None and info == {})

# Non-safetensors extension never enters the parser (F10) — returns None
# without even opening the file.
v, _ = fp8ops.classify_fp8_single_file("/nonexistent/whatever.pt")
check("non-.safetensors extension → None without file access (F10)", v is None)


# ──────────────────────────────────────────────────────────────────────
print("── classifier: rejects (security negatives) ───────────────────")

_expect_reject(
    "mixed C-a/C-b conventions rejected (F5 NEGATIVE)",
    _mk("mixed.safetensors", {
        "a.weight": _fp8((16, 16)),
        "a.weight_scale": _scalar(0.01),
        "a.input_scale": _scalar(0.5),
        "b.weight": _fp8((16, 16)),
        "b.scale_weight": _scalar(0.01),
        "b.scale_input": _scalar(0.5),
    }), "conventions")

_expect_reject(
    "comfy_quant marker rejected at header (F11 NEGATIVE)",
    _mk("cq.safetensors", {
        "a.weight": _fp8((16, 16)),
        "a.comfy_quant": torch.zeros(1, dtype=torch.uint8),
    }), "comfy_quant")

_expect_reject(
    "nvfp4 weight_scale_2 signature rejected at header (F11 NEGATIVE)",
    _mk("nvfp4.safetensors", {
        "a.weight": torch.zeros(16, 8, dtype=torch.uint8),
        "a.weight_scale": _fp8((16, 2)),
        "a.weight_scale_2": _scalar(1.0),
    }), "nvfp4")

_expect_reject(
    "per-channel scale vector rejected at header (F3 NEGATIVE)",
    _mk("chanscale.safetensors", {
        "a.weight": _fp8((16, 16)),
        "a.weight_scale": torch.full((16,), 0.01),
        "a.input_scale": _scalar(0.5),
    }), "scalar")

_expect_reject(
    "non-F32 scale dtype rejected at header (F3 NEGATIVE)",
    _mk("f16scale.safetensors", {
        "a.weight": _fp8((16, 16)),
        "a.weight_scale": _scalar(0.01, torch.float16),
        "a.input_scale": _scalar(0.5),
    }), "F32")

for i, bad in enumerate(["a\x00b.weight", "a\x1b[31m.weight", "a\nb.weight"]):
    _expect_reject(
        f"control-char tensor name #{i} rejected (F7 NEGATIVE)",
        _mk(f"ctrl{i}.safetensors", {
            bad: _fp8((16, 16)),
            bad + "_scale" if False else "x.weight_scale": _scalar(0.01),
            "x.input_scale": _scalar(0.5),
        }), "control")


# ──────────────────────────────────────────────────────────────────────
print("── comfy_quant descriptors (slice C-d, delta reqs 11-19) ──────")


def _desc(obj_or_bytes):
    """Build a U8 descriptor tensor from a JSON-able object or raw bytes."""
    raw = obj_or_bytes if isinstance(obj_or_bytes, bytes) \
        else json.dumps(obj_or_bytes).encode()
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


import json  # noqa: E402

_CQA = {
    "a.weight": _fp8((16, 16)),
    "a.weight_scale": _scalar(0.01),
    "a.input_scale": _scalar(0.5),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
}
v, info = _classify(_mk("cqa.safetensors", _CQA))
check("cq-a classified (descriptor + both scales)", v == "cq-a", f"got {v}")

_CQW = {
    "a.weight": _fp8((16, 16)),
    "a.weight_scale": _scalar(0.01),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn",
                            "full_precision_matrix_mult": True}),
}
v, _ = _classify(_mk("cqw.safetensors", _CQW))
check("cq-w classified (descriptor + weight_scale only)", v == "cq-w",
      f"got {v}")

# Two different allowlisted formats — allowed by design (delta req 19).
v, _ = _classify(_mk("cq2fmt.safetensors", {
    "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
    "b.weight": _fp8((16, 16), seed=1), "b.weight_scale": _scalar(0.01),
    "b.comfy_quant": _desc({"format": "float8_e5m2"}),
}))
check("two allowlisted formats in one file accepted", v == "cq-w", f"got {v}")

# Unknown JSON field: logs once, does not reject (D5 telemetry).
v, _ = _classify(_mk("cqxtra.safetensors", {
    "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn", "surprise": 1}),
}))
check("unknown descriptor field logged, not rejected", v == "cq-w", f"got {v}")

for label, desc in [
    ("non-string format", {"format": 123}),
    ("list JSON root", ["float8_e4m3fn"]),
    ("null JSON root", None),
    ("scalar JSON root", 7),
    ("unknown format fp4", {"format": "fp4"}),
    ("unknown format nvfp4", {"format": "nvfp4"}),
    ("unknown format e4m3fnuz", {"format": "float8_e4m3fnuz"}),
    ("trailing garbage", b'{"format": "float8_e4m3fn"} x'),
    ("invalid UTF-8", b"\xff\xfe\xfd\xfc"),
]:
    _expect_reject(
        f"descriptor {label} rejected (D4 NEGATIVE)",
        _mk(f"cqbad_{label.replace(' ', '_')}.safetensors", {
            "a.weight": _fp8((16, 16)),
            "a.weight_scale": _scalar(0.01),
            "a.comfy_quant": _desc(desc),
        }))

_expect_reject(
    "empty descriptor tensor rejected (D2/D4 NEGATIVE)",
    _mk("cqempty.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": torch.zeros(0, dtype=torch.uint8),
    }), "1-D U8")

_expect_reject(
    "oversize descriptor rejected from header, unread (D2 NEGATIVE)",
    _mk("cqhuge.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": torch.zeros(5000, dtype=torch.uint8),
    }), "without reading")

_expect_reject(
    "non-U8 descriptor dtype rejected (D2 NEGATIVE)",
    _mk("cqf32.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": torch.zeros(8, dtype=torch.float32),
    }), "U8")

_expect_reject(
    "dangling descriptor on bf16 layer rejected (D1 NEGATIVE)",
    _mk("cqdangle.safetensors", {
        "a.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "b.weight": _fp8((16, 16)), "b.weight_scale": _scalar(0.01),
        "b.comfy_quant": _desc({"format": "float8_e4m3fn"}),
    }), "dangling")

_expect_reject(
    "mixed descriptor/plain fp8 layers rejected (D3 NEGATIVE)",
    _mk("cqmixed.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "b.weight": _fp8((16, 16), seed=1), "b.weight_scale": _scalar(0.01),
    }), "mixes")

_expect_reject(
    "cq + C-b marker co-presence rejected (D3 NEGATIVE)",
    _mk("cqcb.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "scaled_fp8": torch.zeros(2, dtype=torch.float8_e4m3fn),
    }))

_expect_reject(
    "mixed input_scale coverage in cq file rejected (D6 NEGATIVE)",
    _mk("cqmixin.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.input_scale": _scalar(0.5),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "b.weight": _fp8((16, 16), seed=1), "b.weight_scale": _scalar(0.01),
        "b.comfy_quant": _desc({"format": "float8_e4m3fn"}),
    }), "all-or-none")

# Weight-only ScaledFp8Linear: numerics match the dequant reference and the
# scaled path is never entered (input_scale is None by construction).
_w8 = _fp8((32, 16), seed=5)
_ws = _scalar(0.02)
_wo = fp8ops.ScaledFp8Linear(_w8, _ws, None, None)
_x = torch.randn(3, 16, dtype=torch.bfloat16)
_yref = nn.functional.linear(
    _x, (_w8.to(torch.float32) * _ws).to(torch.bfloat16), None)
check("weight-only forward == dequant reference",
      torch.allclose(_wo(_x).float(), _yref.float(), rtol=1e-2, atol=1e-2))
check("weight-only mode reported in extra_repr",
      "weight-only" in _wo.extra_repr())

# Loader-level D6: cq-w variant with an input_scale present in the dict.
p = _mk("cqw_withinput.safetensors", {
    "a.weight": _fp8((16, 16)),
    "a.weight_scale": _scalar(0.01),
    "a.input_scale": _scalar(0.5),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
})
_expect_reject(
    "loader rejects cq-w variant when input_scale exists (D6 NEGATIVE)",
    lambda p=p: fp8ops.load_scaled_fp8_component(
        None, p, torch.bfloat16, "", "cq-w"), "inconsistent")


# ──────────────────────────────────────────────────────────────────────
print("── loader: scale validation rejects (F2/F6 negatives) ─────────")


def _load(p):
    # component_class/config are never reached for reject cases — the
    # loader raises during step-1 pairing/validation.
    return fp8ops.load_scaled_fp8_component(None, p, torch.bfloat16, "", "ca")


for i, (val, label) in enumerate([(0.0, "zero"), (-1.0, "negative"),
                                  (float("inf"), "+inf"),
                                  (float("-inf"), "-inf"),
                                  (float("nan"), "NaN")]):
    p = _mk(f"badscale{i}.safetensors", {
        "a.weight": _fp8((16, 16)),
        "a.weight_scale": _scalar(val),
        "a.input_scale": _scalar(0.5),
    })
    _expect_reject(f"scale value {label} rejected at load (F2 NEGATIVE)",
                   lambda p=p: _load(p), "finite")

# Subnormal scale (reviewer finding 1 — F2 lists denormals explicitly).
p = _mk("subnormal.safetensors", {
    "a.weight": _fp8((16, 16)),
    "a.weight_scale": _scalar(1e-40),
    "a.input_scale": _scalar(0.5),
})
_expect_reject("subnormal scale value rejected at load (F2 NEGATIVE)",
               lambda p=p: _load(p), "normal")

# Non-safetensors path routed directly to the loader (reviewer finding 4 /
# F10): loud reject, message points at re-saving.
_expect_reject(
    "loader refuses non-.safetensors path outright (F10 NEGATIVE)",
    lambda: fp8ops.load_scaled_fp8_component(
        None, "/nonexistent/model.pt", torch.bfloat16, "", "ca"),
    "safetensors")

# A scale-suffixed key carrying an fp8 tensor (name/dtype confusion —
# reviewer finding 5's adapted collision property: a "scale" can never be
# smuggled as a weight or vice versa; the F32 dtype gate rejects it).
_expect_reject(
    "fp8-dtype tensor under a scale-suffix name rejected (F1-adapted NEGATIVE)",
    _mk("fp8scale.safetensors", {
        "a.weight": _fp8((16, 16)),
        "a.weight_scale": _fp8((1,), seed=2).reshape(()),
        "a.input_scale": _scalar(0.5),
    }), "F32")

p = _mk("missingscale.safetensors", {
    "a.weight": _fp8((16, 16)),
    "a.weight_scale": _scalar(0.01),
    # input_scale missing
    "b.weight": _fp8((16, 16)),
    "b.weight_scale": _scalar(0.01),
    "b.input_scale": _scalar(0.5),
})
_expect_reject("fp8 weight missing a paired scale rejected (F6 NEGATIVE)",
               lambda: _load(p), "partial")

p = _mk("dangling.safetensors", {
    "a.weight": _fp8((16, 16)),
    "a.weight_scale": _scalar(0.01),
    "a.input_scale": _scalar(0.5),
    "ghost.weight_scale": _scalar(0.01),
})
_expect_reject("dangling scale with no fp8 weight rejected (F1/F6 NEGATIVE)",
               lambda: _load(p), "dangling")

p = _mk("fp8bias.safetensors", {
    "a.weight": _fp8((16, 16)),
    "a.weight_scale": _scalar(0.01),
    "a.input_scale": _scalar(0.5),
    "a.bias": _fp8((16,), seed=1),
})
_expect_reject("fp8 non-.weight tensor rejected (layout NEGATIVE)",
               lambda: _load(p), "not a .weight")

p = _mk("fp8_3d.safetensors", {
    "a.weight": _fp8((4, 4, 4)),
    "a.weight_scale": _scalar(0.01),
    "a.input_scale": _scalar(0.5),
})
_expect_reject("non-2D fp8 weight rejected (layout NEGATIVE)",
               lambda: _load(p), "2D")


# ──────────────────────────────────────────────────────────────────────
print("── ScaledFp8Linear numerics + dtype-cast refusal ──────────────")

w8 = _fp8((32, 16), seed=3)
ws = _scalar(0.02)
i_s = _scalar(0.8)
bias = torch.randn(32, dtype=torch.bfloat16) * 0.1
lin = fp8ops.ScaledFp8Linear(w8, ws, i_s, bias)

x = torch.randn(4, 5, 16, dtype=torch.bfloat16)
y = lin(x)
wref = (w8.to(torch.float32) * ws).to(torch.bfloat16)
yref = nn.functional.linear(x, wref, bias)
rel = ((y.float() - yref.float()).abs().mean()
       / max(yref.float().abs().mean().item(), 1e-9))
check("forward shape/dtype", y.shape == (4, 5, 32) and y.dtype == torch.bfloat16)
check("forward matches dequant reference (<5% rel err)", rel < 0.05,
      f"rel={rel:.4f}")

# .to(bfloat16) must NOT dequantize the fp8 buffer (dtype-cast refusal).
lin2 = fp8ops.ScaledFp8Linear(_fp8((8, 8)), _scalar(0.01), _scalar(0.5), None)
lin2.to(torch.bfloat16)
check("module.to(bf16) leaves weight fp8 (no silent dequant)",
      lin2.weight.dtype == torch.float8_e4m3fn,
      f"got {lin2.weight.dtype}")

try:
    fp8ops.ScaledFp8Linear(torch.randn(8, 8), _scalar(0.01), _scalar(0.5), None)
    check("ScaledFp8Linear refuses non-fp8 weight (NEGATIVE)", False, "no raise")
except fp8ops.ScaledFp8FormatError:
    check("ScaledFp8Linear refuses non-fp8 weight (NEGATIVE)", True)


# ──────────────────────────────────────────────────────────────────────
print("── LoRA guard extension (F8) ──────────────────────────────────")

host = nn.Sequential(nn.Linear(8, 8),
                     fp8ops.ScaledFp8Linear(_fp8((8, 8)), _scalar(0.01),
                                            _scalar(0.5), None))
check("contains_scaled_fp8 True on host module",
      fp8ops.contains_scaled_fp8(host) is True)
check("contains_scaled_fp8 False on plain module",
      fp8ops.contains_scaled_fp8(nn.Linear(4, 4)) is False)
check("is_quantized_module detects ScaledFp8Linear host (F8)",
      edu.is_quantized_module(host) is True)
try:
    edu.guard_direct_merge(host, "[t]", "LoKR adapter")
    check("guard_direct_merge raises on scaled-fp8 base (F8 NEGATIVE)",
          False, "did not raise")
except RuntimeError as e:
    check("guard_direct_merge raises on scaled-fp8 base (F8 NEGATIVE)",
          "--quant" in str(e) or "quantized" in str(e))


# ──────────────────────────────────────────────────────────────────────
print("── slice DMR: dequant→merge→requant (security reqs 21-30) ─────")


class _DMRHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.plain = nn.Linear(8, 8, bias=False)
        self.plain.weight.data = self.plain.weight.data.to(torch.bfloat16)
        self.q = fp8ops.ScaledFp8Linear(
            _fp8((8, 8)), _scalar(0.02), _scalar(0.5), None)
        self.qw = fp8ops.ScaledFp8Linear(
            _fp8((8, 8), seed=1), _scalar(0.02), None, None)


_h = _DMRHost()
_bk = {}
_d = torch.randn(8, 8, dtype=torch.bfloat16) * 0.01

# Invariant 1: plain path byte-identical to legacy clone+add_.
_pw = _h.plain.weight.data.clone()
check("DMR plain merge kind",
      fp8ops.apply_merge_delta(_h, "plain.weight", _d, _bk) == "plain")
check("DMR plain backup is a raw tensor clone (legacy shape)",
      isinstance(_bk["plain.weight"], torch.Tensor)
      and torch.equal(_bk["plain.weight"], _pw))
check("DMR plain merge result == legacy add_",
      torch.equal(_h.plain.weight.data, _pw + _d.to(torch.bfloat16)))

# Invariant 3: ScaledFp8Linear requant within one fp8 step of exact merge.
_qw0 = _h.q.weight.clone()
_qs0 = _h.q.weight_scale.clone()
_h.q._fallback_weight = torch.zeros(8, 8)  # simulate stale cache
_exact = _h.q.weight.to(torch.float32) * _h.q.weight_scale + _d.float()
check("DMR scaled_fp8 merge kind",
      fp8ops.apply_merge_delta(_h, "q.weight", _d, _bk) == "scaled_fp8")
_got = _h.q.weight.to(torch.float32) * _h.q.weight_scale
_rel = ((_got - _exact).abs().max() / _exact.abs().max()).item()
check("DMR requant error within one e4m3 step", _rel < 0.08, f"rel={_rel:.4f}")
check("DMR merge invalidates dequant cache (invariant 6)",
      _h.q._fallback_weight is None)
check("DMR weight-only module merges too",
      fp8ops.apply_merge_delta(_h, "qw.weight", _d, _bk) == "scaled_fp8")

# Invariant 2: exact restore (bit-equal) + cache/warn reset (req 29).
_h.q._fallback_weight = torch.ones(8, 8)
_h.q._warned_fallback = True
check("DMR restore succeeds",
      fp8ops.restore_merge_backup(_h, "q.weight", _bk["q.weight"]) is True)
check("DMR restore is bit-exact (weight + scale)",
      torch.equal(_h.q.weight, _qw0)
      and torch.equal(_h.q.weight_scale, _qs0))
check("DMR restore resets cache and warn latch (req 29 NEGATIVE)",
      _h.q._fallback_weight is None and _h.q._warned_fallback is False)

# req 21: non-finite delta rejected before any dispatch.
try:
    fp8ops.apply_merge_delta(_h, "plain.weight",
                             torch.full((8, 8), float("nan")), {})
    check("DMR non-finite delta rejected (req 21 NEGATIVE)", False, "no raise")
except RuntimeError as e:
    check("DMR non-finite delta rejected (req 21 NEGATIVE)",
          "non-finite" in str(e))

# req 22: all-zero merged tensor → sentinel scale, no div-by-zero.
_h2 = _DMRHost()
_dz = -(_h2.q.weight.to(torch.float32) * _h2.q.weight_scale)
fp8ops.apply_merge_delta(_h2, "q.weight", _dz, {})
check("DMR zero-amax sentinel scale (req 22 NEGATIVE)",
      _h2.q.weight_scale.item() == 1.0
      and _h2.q.weight.to(torch.float32).abs().max().item() == 0.0)

# req 24: orphan fp8 tensor (not a ScaledFp8Linear) must raise, never
# take the plain path.
class _Orphan(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", _fp8((4, 4)))


_o = nn.Module()
_o.orphan = _Orphan()
try:
    fp8ops.apply_merge_delta(_o, "orphan.weight", torch.randn(4, 4), {})
    check("DMR orphan fp8 raises (req 24 NEGATIVE)", False, "merged!")
except RuntimeError as e:
    check("DMR orphan fp8 raises (req 24 NEGATIVE)",
          "not owned by a ScaledFp8Linear" in str(e))

# req 23: resolution map exposes .weight buffers ONLY — scale buffers and
# descriptors are invisible so adversarial LoRA keys cannot target them.
_mm = fp8ops.merge_resolution_map(_h)
check("DMR resolution map: fp8 .weight buffer visible", "q.weight" in _mm)
check("DMR resolution map: scale buffers INVISIBLE (req 23 NEGATIVE)",
      "q.weight_scale" not in _mm and "q.input_scale" not in _mm)

# req 23 through the REAL path: a crafted .diff key aimed at a scale
# buffer must not write (resolves to None → skipped), and the scale
# must be unchanged. comfyless import installs the folder_paths shims the
# nodes package needs (established suite pattern).
import comfyless.generate  # noqa: F401,E402 — shims
from nodes.eric_lora_format_convert_apply import _apply_converted_lora_as_delta  # noqa: E402
_h3 = _DMRHost()
_s_before = _h3.q.weight_scale.clone()
_apply_converted_lora_as_delta(
    _h3, {"q.weight_scale.diff": torch.full((), 999.0)}, "evil", 1.0, "[t]")
check("DMR adversarial .weight_scale.diff does NOT write (DMR-3 NEGATIVE)",
      torch.equal(_h3.q.weight_scale, _s_before))

# req 25: LIFO ledger warns on out-of-order unload and pops correctly.
fp8ops.record_direct_merge(_h, "adapterA")
fp8ops.record_direct_merge(_h, "adapterB")
fp8ops.warn_non_lifo_unload(_h, "adapterA", "[t]")
check("DMR LIFO ledger pops the unloaded adapter (req 25)",
      _h._eric_direct_merge_order == ["adapterB"])

# req 28: PEFT-wrapped bf16 + ScaledFp8Linear coexistence resolves each
# correctly through the merged map.
from nodes.eric_lora_format_convert_apply import resolve_merge_target  # noqa: E402
class _Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(8, 8, bias=False)


_h4 = nn.Module()
_h4.a = _Wrapped()      # PEFT-wrapped plain: resolves via .base_layer.weight
_h4.b = fp8ops.ScaledFp8Linear(_fp8((8, 8)), _scalar(0.02), _scalar(0.5), None)
_m4 = fp8ops.merge_resolution_map(_h4)
check("DMR coexistence: wrapped plain resolves to base_layer.weight (req 28)",
      resolve_merge_target(_m4, "a") == "a.base_layer.weight")
check("DMR coexistence: ScaledFp8Linear resolves to its buffer (req 28)",
      resolve_merge_target(_m4, "b") == "b.weight")


# ──────────────────────────────────────────────────────────────────────
print("── real-collection spot checks (skip-as-pass if absent) ───────")

# Full Vision-§0 survey coverage (reviewer finding 9): every documented
# variant, the prefix case, a C-c control, a bf16 control, and the nvfp4
# header reject.
_CMFY = "/home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models"
_REAL = [
    (f"{_CMFY}/diffusion_models/Flux.2-Klein-9B-base/"
     f"flux-2-klein-base-9b-fp8.safetensors", "ca"),
    (f"{_CMFY}/checkpoints/ltx-2-19b-distilled-fp8.safetensors", "ca"),
    (f"{_CMFY}/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled"
     f".safetensors", "cb"),
    (f"{_CMFY}/checkpoints/Flux.1-dev/"
     f"colossusProjectFlux_v12HephaistosFP8UNET.safetensors", "cc"),
    (f"{_CMFY}/diffusion_models/Flux.2-Klein-9B-base/"
     f"flux-2-klein-base-9b.safetensors", None),
    (f"{_CMFY}/checkpoints/Flux.2-Klein-9B-base/sexy/"
     f"pornmasterFlux2Klein_v2.safetensors", "cq-a"),
    (f"{_CMFY}/checkpoints/Krea/krea2TurboUncensored_v1.safetensors", "cq-w"),
    (f"{_CMFY}/checkpoints/Qwen/absoluteRealismV01_qwenV10.safetensors",
     "cq-w"),
]
for rp, want in _REAL:
    if os.path.exists(rp):
        v, _ = _classify(rp)
        check(f"real file {os.path.basename(rp)} → {want}", v == want,
              f"got {v}")
    else:
        check(f"real file {os.path.basename(rp)} → {want} "
              f"(SKIPPED: not present)", True)

_NVFP4 = (f"{_CMFY}/diffusion_models/ZImageTurbo/base model/"
          f"ZImageTurbo-nvfp4_FP32.safetensors")
if os.path.exists(_NVFP4):
    _expect_reject("real nvfp4 file rejected at header", _NVFP4, "nvfp4")
else:
    check("real nvfp4 file rejected at header (SKIPPED: not present)", True)


# ──────────────────────────────────────────────────────────────────────
import shutil
shutil.rmtree(_TMP, ignore_errors=True)
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed")
print("─" * 50)
sys.exit(1 if failed else 0)
