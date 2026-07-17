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

# Load the modules under their REAL package names (registered in
# sys.modules) so the in-function relative imports (`from
# .eric_krea2_convert import ...` in fp8_ops, `from .eric_diffusion_fp8_ops
# import ...` in utils) resolve — required by the slice-R1/R2 POSITIVE load
# tests, which reach step 2 of the loader (the reject-only tests never did).
# A synthetic `nodes` package avoids executing the real nodes/__init__.py
# (which imports every node file and their ComfyUI deps).
import types  # noqa: E402

if "nodes" not in sys.modules:
    _pkg = types.ModuleType("nodes")
    _pkg.__path__ = [str(Path(__file__).parent / "nodes")]
    sys.modules["nodes"] = _pkg


def _load_pkg_module(alias, modname):
    spec = importlib.util.spec_from_file_location(
        f"nodes.{modname}", Path(__file__).parent / "nodes" / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"nodes.{modname}"] = mod
    spec.loader.exec_module(mod)
    return mod


krea2c = _load_pkg_module("krea2c", "eric_krea2_convert")
fp8ops = _load_pkg_module("fp8ops", "eric_diffusion_fp8_ops")
edu = _load_pkg_module("edu", "eric_diffusion_utils")

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

# ── slice PQ: partial-quant — naked plain-fp8 coexists with descriptored ──
# Security review PQ-1..6 / reqs 31-38. "Naked" = a descriptor-less fp8 base;
# legal ONLY when FULLY bare (no scale of any kind). Any scale on a naked base
# is refused — a PRESENT scale must never be silently ignored (~448x
# corruption). classify is the airtight gate (review: "the only airtight
# control"). Replaces the former all-or-nothing D3 reject.
print("── slice PQ: partial-quant naked-fp8 coexistence (reqs 31-38) ──")

# Positive: descriptored + FULLY-BARE naked → accepted (the target layout).
v, _ = _classify(_mk("pq_cqw_naked.safetensors", {
    "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
    "first.weight": _fp8((16, 16), seed=2),   # naked, fully bare
}))
check("PQ: cq-w descriptored + fully-bare naked → accepted (test 7)",
      v == "cq-w", f"got {v}")

v, _ = _classify(_mk("pq_cqa_naked.safetensors", {
    "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
    "a.input_scale": _scalar(0.5),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
    "first.weight": _fp8((16, 16), seed=2),   # naked, fully bare
}))
check("PQ: cq-a descriptored + fully-bare naked → accepted (test 6)",
      v == "cq-a", f"got {v}")

# Positive (no cap, req 36/PQ-6): mostly-naked + ONE descriptor → accepted.
v, _ = _classify(_mk("pq_mostly_naked.safetensors", {
    "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
    "b.weight": _fp8((16, 16), seed=1),        # naked
    "c.weight": _fp8((16, 16), seed=2),        # naked
    "d.weight": _fp8((16, 16), seed=3),        # naked
}))
check("PQ: mostly-naked + one descriptor → accepted, no cap (test 11)",
      v == "cq-w", f"got {v}")

# Test 10 (PQ-4): the naked-set enumeration log fires and names the count.
import io as _io, contextlib as _cl  # noqa: E402
_buf = _io.StringIO()
with _cl.redirect_stdout(_buf):
    _classify(_mk("pq_log.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "first.weight": _fp8((16, 16), seed=2),   # 1 naked layer
    }))
_logout = _buf.getvalue()
check("PQ: naked-set enumeration log fires with count (PQ-4, test 10)",
      "1 plain-fp8" in _logout and "naked" in _logout,
      f"log: {_logout[:160]!r}")

# Negative {0,1,0}: naked base carrying a weight_scale (no descriptor) — the
# old "mixed" case; a present scale must not be ignored (tests 1+2).
_expect_reject(
    "PQ: naked base carrying weight_scale rejected (PQ-1 / {0,1,0}, test 1/2)",
    _mk("pq_naked_ws.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "b.weight": _fp8((16, 16), seed=1), "b.weight_scale": _scalar(0.01),
    }), "bare")

# Negative {0,0,1}: naked base carrying an input_scale only → reject (test 3).
_expect_reject(
    "PQ: naked base carrying input_scale rejected (PQ-1 / {0,0,1}, test 3)",
    _mk("pq_naked_is.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "b.weight": _fp8((16, 16), seed=1), "b.input_scale": _scalar(0.5),
    }), "bare")

# Negative {0,1,1}: naked base carrying BOTH scales → reject (the silent
# ~448x-corruption combo; must NOT load as scaled) (test 4).
_expect_reject(
    "PQ: naked base carrying both scales rejected (PQ-1 / {0,1,1}, test 4)",
    _mk("pq_naked_both.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "b.weight": _fp8((16, 16), seed=1),
        "b.weight_scale": _scalar(0.01), "b.input_scale": _scalar(0.5),
    }), "bare")

# Loader test 9 — AMENDED by slice R1 (security review R1R2R3 req 39): a
# fully-bare non-.weight fp8 tensor now UPCASTS (positive covered in the R1
# battery below); the reject arm is a non-.weight fp8 tensor carrying a
# BOUND scale — a present binding is never silently ignored.
_expect_reject(
    "PQ/R1: loader rejects BOUND non-.weight fp8 tensor (req 39, amended test 9)",
    lambda: fp8ops.load_scaled_fp8_component(
        None, _mk("pq_nonweight.safetensors", {
            "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
            "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
            "b.bias": _fp8((16,), seed=1),          # non-.weight fp8...
            "b.bias.weight_scale": _scalar(0.01),   # ...with a bound scale
        }), torch.bfloat16, "", "cq-w"), "non-.weight")

# ── slice R1/R2: non-weight fp8 upcast + dequant-to-bf16 mode ──────────
# Security review R1R2R3 reqs 39-45 (docs/security/review-slice-R1R2R3-
# dequant-nonweight-2026-07-07.md). Positive loads run CPU-side via a
# capture stub: dequant mode returns right after from_single_file, and the
# stub records the bf16 dict the loader hands over.
print("── slice R1/R2: non-weight fp8 + dequant mode (reqs 39-45) ────")


class _CaptureComp(nn.Module):
    """from_single_file stub — records the state dict the loader built."""
    @classmethod
    def from_single_file(cls, sd, config=None, torch_dtype=None,
                         local_files_only=True):
        m = cls()
        m._received_sd = dict(sd)
        return m


import io as _io2, contextlib as _cl2  # noqa: E402

# R1 test 1 + 6 + log: cq-w set + fully-bare non-.weight fp8 (1D bias and a
# 2D modulation table) → loads; both upcast to bf16; enumeration log fires.
_buf = _io2.StringIO()
with _cl2.redirect_stdout(_buf):
    _m = fp8ops.load_scaled_fp8_component(
        _CaptureComp, _mk("r1_pos.safetensors", {
            "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
            "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
            "first.bias": _fp8((16,), seed=1),        # 1D non-.weight, bare
            "mod.lin": _fp8((6, 16), seed=2),         # 2D non-.weight, bare
        }), torch.bfloat16, "", "cq-w", dequant_fp8=True)
_r1log = _buf.getvalue()
check("R1: fully-bare non-.weight fp8 loads (test 1)",
      hasattr(_m, "_received_sd"))
check("R1: 1D non-.weight fp8 upcast to bf16 (test 1)",
      _m._received_sd["first.bias"].dtype == torch.bfloat16)
check("R1: 2D non-.weight fp8 upcast to bf16, no residency (test 6)",
      _m._received_sd["mod.lin"].dtype == torch.bfloat16
      and not any(isinstance(mm, fp8ops.ScaledFp8Linear)
                  for mm in _m.modules()))
check("R1: non-.weight enumeration log fires (req 41)",
      "2 non-.weight fp8" in _r1log, f"log: {_r1log[:200]!r}")
# Descriptored weight was DEQUANTIZED (fp8*scale), not raw-cast (test 7 half).
_expected = (_fp8((16, 16)).to(torch.float32) * 0.01).to(torch.bfloat16)
check("R1/R2: descriptored weight dequantized via weight_scale (test 7)",
      torch.equal(_m._received_sd["a.weight"], _expected))

# R1 tests 2/3/5: non-.weight fp8 carrying each binding class → reject.
for _lbl, _extra in [
    ("weight_scale (test 2)", {"g.gamma.weight_scale": _scalar(0.01)}),
    ("input_scale (test 3)", {"g.gamma.input_scale": _scalar(0.5)}),
    ("comfy_quant (test 5)",
     {"g.gamma.comfy_quant": _desc({"format": "float8_e4m3fn"})}),
]:
    _expect_reject(
        f"R1: non-.weight fp8 with bound {_lbl} rejected (req 39)",
        lambda _e=_extra: fp8ops.load_scaled_fp8_component(
            None, _mk("r1_bound.safetensors", {
                "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
                "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
                "g.gamma": _fp8((16,), seed=3), **_e,
            }), torch.bfloat16, "", "cq-w"), "non-.weight")

# R1 test 4: C-b binding on a non-.weight fp8 in a cq file → whole-file
# reject at CLASSIFY via the cb-copresence guard (safe false-reject).
_expect_reject(
    "R1: non-.weight fp8 with C-b binding → classify cb-copresence reject (test 4)",
    _mk("r1_cb.safetensors", {
        "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
        "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        "g.gamma": _fp8((16,), seed=3), "g.gamma.scale_weight": _scalar(0.01),
    }), "C-b")

# R2 test 8: dequant_fp8=True does NOT skip validation — NaN input_scale on
# a cq-a file still rejects (the finding-40 guard).
_expect_reject(
    "R2: dequant mode still validates input_scale (NaN → reject, test 8)",
    lambda: fp8ops.load_scaled_fp8_component(
        _CaptureComp, _mk("r2_nanin.safetensors", {
            "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
            "a.input_scale": _scalar(float("nan")),
            "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        }), torch.bfloat16, "", "cq-a", dequant_fp8=True))

# R2 test 9: dequant mode still validates weight_scale (zero → reject).
_expect_reject(
    "R2: dequant mode still validates weight_scale (0 → reject, test 9)",
    lambda: fp8ops.load_scaled_fp8_component(
        _CaptureComp, _mk("r2_zerows.safetensors", {
            "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.0),
            "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        }), torch.bfloat16, "", "cq-w", dequant_fp8=True))

# R2 test 10: default flag (omitted) = resident behavior — the dequant log
# must NOT fire (fail-closed default).
_buf = _io2.StringIO()
with _cl2.redirect_stdout(_buf):
    _m10 = fp8ops.load_scaled_fp8_component(
        _CaptureComp, _mk("r2_default.safetensors", {
            "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
            "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
        }), torch.bfloat16, "", "cq-w")
check("R2: default (no flag) does not enter dequant mode (test 10)",
      _m10 is not None and "dequant-fp8 mode" not in _buf.getvalue())

# R2 test 11: dequant return placement covers the KREA build branch too —
# monkeypatch build_krea2_transformer, feed a krea-marker file, assert the
# stub's return object comes back (i.e. return sits after BOTH branches).
_sentinel = nn.Module()
_orig_build = krea2c.build_krea2_transformer
krea2c.build_krea2_transformer = lambda *a, **k: _sentinel
try:
    _m11 = fp8ops.load_scaled_fp8_component(
        _CaptureComp, _mk("r2_krea.safetensors", {
            "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
            "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
            # krea-native markers (bf16) so is_krea2_comfy_checkpoint fires
            "blocks.0.attn.wq.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
            "blocks.0.mod.lin": torch.zeros(24, dtype=torch.bfloat16),
        }), torch.bfloat16, "", "cq-w", dequant_fp8=True)
finally:
    krea2c.build_krea2_transformer = _orig_build
check("R2: dequant return covers the krea build branch (test 11)",
      _m11 is _sentinel)

# ── slice I8: comfy int8-tensorwise (ci-w) — security reqs 46-56 ───────
# docs/security/review-slice-I8-int8-tensorwise-2026-07-08.md. Flavor from
# dtypes, strict descriptor-field allowlist, no naked-int8 in any cell,
# loader re-assertion, unconditional dequant (no residency).
print("── slice I8: int8-tensorwise ci-w (reqs 46-56) ────────────────")


def _i8(shape, seed=0, lo=-100, hi=100):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.int8)


_I8DESC = _desc({"format": "int8_tensorwise"})

# Test 1+2+4+5(log): target-shaped file classifies ci-w, loads via the
# stub, dequant numerics exact, no ScaledFp8Linear, unconditional dequant.
_i8w = _i8((8, 4), seed=7)
_i8file = _mk("i8_pos.safetensors", {
    "a.weight": _i8w, "a.weight_scale": _scalar(0.02, torch.bfloat16),
    "a.comfy_quant": _I8DESC,
    "norm.gamma": torch.ones(4, dtype=torch.bfloat16),  # bf16 rest
})
v, _info = _classify(_i8file)
check("I8: target-shaped file classifies ci-w (test 1)", v == "ci-w",
      f"got {v}")
check("I8: info reports int8 count", _info.get("n_int8") == 1)
for _flag in (False, True):   # unconditional dequant (req 53, test 4)
    _m = fp8ops.load_scaled_fp8_component(
        _CaptureComp, _i8file, torch.bfloat16, "", "ci-w",
        dequant_fp8=_flag)
    check(f"I8: ci-w loads + returns before residency (dequant_fp8={_flag})",
          hasattr(_m, "_received_sd")
          and not any(isinstance(mm, fp8ops.ScaledFp8Linear)
                      for mm in _m.modules()))
_exp = (_i8w.to(torch.float32)
        * torch.tensor(0.02, dtype=torch.bfloat16).to(torch.float32)
        ).to(torch.bfloat16)
check("I8: dequant numerics exact (int8 x bf16-scale -> bf16, test 2)",
      torch.equal(_m._received_sd["a.weight"], _exp))
check("I8: bf16 rest passes through as bf16",
      _m._received_sd["norm.gamma"].dtype == torch.bfloat16)

# Test 3: F32 scalar scale also accepted for int8.
v, _ = _classify(_mk("i8_f32scale.safetensors", {
    "a.weight": _i8((8, 4)), "a.weight_scale": _scalar(0.02),
    "a.comfy_quant": _I8DESC}))
check("I8: F32 scale accepted for int8 (test 3)", v == "ci-w", f"got {v}")

# Test 6: mixed descriptored I8 + F8 in one file → reject (req 46).
_expect_reject("I8: mixed int8/fp8 descriptored weights reject (test 6)",
               _mk("i8_mixed.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
                   "b.weight": _fp8((8, 4)), "b.weight_scale": _scalar(0.01),
                   "b.comfy_quant": _desc({"format": "float8_e4m3fn"}),
               }), "mixes int8")

# Test 7: I8-paired descriptor declaring an fp8 format → reject (req 46).
_expect_reject("I8: int8 weight + fp8-format descriptor rejects (test 7)",
               _mk("i8_xfmt.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
               }), "mismatch")

# Test 8: F8-paired descriptor declaring int8_tensorwise → reject (the fp8
# allowlist was NOT widened).
_expect_reject("I8: fp8 weight + int8-format descriptor rejects (test 8)",
               _mk("i8_xfmt2.safetensors", {
                   "a.weight": _fp8((8, 4)), "a.weight_scale": _scalar(0.01),
                   "a.comfy_quant": _desc({"format": "int8_tensorwise"}),
               }), "not in the supported set")

# Test 9: descriptor paired with a BF16 weight → D1 message names dtype.
_expect_reject("I8: descriptor on BF16 weight rejects naming dtype (test 9)",
               _mk("i8_bf16w.safetensors", {
                   "a.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
               }), "BF16")

# Test 10: input_scale anywhere in an int8 file → reject (req 47).
_expect_reject("I8: input_scale in int8 file rejects (test 10)",
               _mk("i8_inscale.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.input_scale": _scalar(0.5),
                   "a.comfy_quant": _I8DESC,
               }), "weight-only")

# Test 11: strict field allowlist — convrot, convrot_groupsize, and an
# arbitrary unknown field each reject (req 51).
for _fld, _val in [("convrot", True), ("convrot_groupsize", 256),
                   ("zero_point", 0)]:
    _expect_reject(
        f"I8: descriptor field {_fld!r} rejects (strict allowlist, test 11)",
        _mk("i8_fld.safetensors", {
            "a.weight": _i8((8, 4)),
            "a.weight_scale": _scalar(0.02, torch.bfloat16),
            "a.comfy_quant": _desc({"format": "int8_tensorwise",
                                    _fld: _val}),
        }))

# Test 12 (I8-4): convrot on an FP8-format descriptor now rejects too.
_expect_reject("I8-4: convrot on fp8 descriptor rejects (test 12)",
               _mk("i8_fp8rot.safetensors", {
                   "a.weight": _fp8((8, 4)), "a.weight_scale": _scalar(0.01),
                   "a.comfy_quant": _desc({"format": "float8_e4m3fn",
                                           "convrot": True}),
               }), "ConvRot")
# (fp8 non-convrot unknown fields keep log-and-ignore — pinned by the
# existing 'unknown descriptor field logged, not rejected' case above.)

# Tests 13-15: naked/partial int8 cells all reject (req 48).
_expect_reject("I8: naked int8 weight rejects (test 13)",
               _mk("i8_naked.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
                   "b.weight": _i8((8, 4), seed=1),   # fully bare int8
               }), "naked int8")
_expect_reject("I8: int8 + scale, no descriptor rejects (test 14)",
               _mk("i8_scale_nodesc.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
                   "b.weight": _i8((8, 4), seed=1),
                   "b.weight_scale": _scalar(0.02, torch.bfloat16),
               }))
_expect_reject("I8: int8 + descriptor, no scale rejects (test 15)",
               _mk("i8_desc_noscale.safetensors", {
                   "a.weight": _i8((8, 4)), "a.comfy_quant": _I8DESC,
               }))

# Test 16: non-.weight int8 (bare) rejects — no R1 relaxation for int8.
_expect_reject("I8: non-.weight int8 tensor rejects (test 16)",
               _mk("i8_nonw.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
                   "b.bias": _i8((8,), seed=1),
               }), "not a .weight")

# Test 17 (req 49): loader re-assert — an fp8-variant load whose dict
# carries an unpaired int8 tensor must reject at the pass-through guard,
# never silently cast (classify/loader divergence simulation).
_expect_reject("I8: loader pass-through guard refuses stray int8 (test 17)",
               lambda: fp8ops.load_scaled_fp8_component(
                   _CaptureComp, _mk("i8_stray.safetensors", {
                       "a.weight": _fp8((8, 4)),
                       "a.weight_scale": _scalar(0.01),
                       "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
                       "sneak.weight": _i8((8, 4)),
                   }), torch.bfloat16, "", "cq-w", dequant_fp8=True),
               "unpaired int8")

# Test 18: same fixture at CLASSIFY → fp8-flavored + I8 rejects (req 54).
_expect_reject("I8: int8 tensor in fp8-flavored file rejects (test 18)",
               _mk("i8_infp8.safetensors", {
                   "a.weight": _fp8((8, 4)), "a.weight_scale": _scalar(0.01),
                   "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
                   "sneak.weight": _i8((8, 4)),
               }), "req 54")

# Test 19: naked fp8 inside an int8-flavored file rejects (v1 polarity).
_expect_reject("I8: naked fp8 in int8 file rejects (test 19)",
               _mk("i8_nakedf8.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
                   "first.weight": _fp8((8, 4)),
               }), "req 54")

# Test 20: scale dtype F16 rejects; a bare 1-D (rows,) scale rejects because
# it would broadcast along the LAST axis, not the row axis (req 57).
_expect_reject("I8: F16 scale rejects (test 20a)",
               _mk("i8_f16s.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.float16),
                   "a.comfy_quant": _I8DESC,
               }), "BF16/F32")
_expect_reject("I8: bare 1-D (rows,) scale rejects (test 20b)",
               _mk("i8_vecs.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": torch.full((8,), 0.02,
                                                dtype=torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
               }), "per-output-channel")

# ── req 57: per-output-channel (rows, 1) scales — the greed_int8.safetensors
# shape. `int8_tensorwise` names comfy-kitchen's LAYOUT class, not the scale
# granularity: quantize_int8_rowwise emits [..., 1] per-row scales under the
# same format string, and dequantize_int8_simple is `q.float() * scale`.
_pcw = _i8((8, 4), seed=11)
_pcs = torch.rand(8, 1, generator=torch.Generator().manual_seed(3)) * 0.05 + 0.01
_pcfile = _mk("i8_perchannel.safetensors", {
    "a.weight": _pcw, "a.weight_scale": _pcs.to(torch.float32),
    "a.comfy_quant": _I8DESC,
})
v, _info = _classify(_pcfile)
check("I8: per-channel (rows,1) F32 scale classifies ci-w (req 57)",
      v == "ci-w", f"got {v}")
_m_pc = fp8ops.load_scaled_fp8_component(
    _CaptureComp, _pcfile, torch.bfloat16, "", "ci-w")
_exp_pc = (_pcw.to(torch.float32) * _pcs.to(torch.float32)).to(torch.bfloat16)
check("I8: per-channel dequant broadcasts ROW-wise (req 57)",
      torch.equal(_m_pc._received_sd["a.weight"], _exp_pc))
# Guard the fixture itself: scales must actually differ across rows, else a
# wrong-axis broadcast would still produce the expected tensor and the
# numerics check above would pass vacuously.
check("I8: per-channel fixture discriminates row vs column broadcast",
      _pcs.min().item() != _pcs.max().item())

# BF16 per-channel also accepted.
v, _ = _classify(_mk("i8_pc_bf16.safetensors", {
    "a.weight": _i8((8, 4)),
    "a.weight_scale": torch.full((8, 1), 0.02, dtype=torch.bfloat16),
    "a.comfy_quant": _I8DESC}))
check("I8: per-channel BF16 scale accepted (req 57)", v == "ci-w", f"got {v}")

# NEGATIVE: (1, in_features) broadcasts COLUMN-wise, silently, with no shape
# error from torch. This is the corruption path req 57 exists to close.
_expect_reject("I8: (1, in) column-broadcast scale rejects (req 57)",
               _mk("i8_colscale.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": torch.full((1, 4), 0.02,
                                                dtype=torch.float32),
                   "a.comfy_quant": _I8DESC,
               }), "per-output-channel")

# NEGATIVE: (r, 1) with r != rows — right rank, wrong binding.
_expect_reject("I8: (rows+1, 1) mismatched scale rejects (req 57)",
               _mk("i8_wrongrows.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": torch.full((9, 1), 0.02,
                                                dtype=torch.float32),
                   "a.comfy_quant": _I8DESC,
               }), "per-output-channel")

# NEGATIVE: one poisoned element in an otherwise-valid per-channel vector —
# the value policy is elementwise, not `.item()` (req 59).
for _lbl, _v in [("NaN", float("nan")), ("Inf", float("inf")),
                 ("zero", 0.0), ("negative", -1.0),
                 ("subnormal", 2.0 ** -133)]:
    def _poison(_v=_v):
        _s = torch.full((8, 1), 0.02, dtype=torch.float32)
        _s[5, 0] = _v
        return fp8ops.load_scaled_fp8_component(
            _CaptureComp, _mk("i8_pcval.safetensors", {
                "a.weight": _i8((8, 4)),
                "a.weight_scale": _s,
                "a.comfy_quant": _I8DESC,
            }), torch.bfloat16, "", "ci-w")
    _expect_reject(
        f"I8: per-channel scale with one {_lbl} row rejects (req 59)",
        _poison, "element [5]")

# Test 21: BF16 scale VALUE battery — rejected at load (validated raw,
# upcast for the value check).
for _lbl, _v in [("NaN", float("nan")), ("Inf", float("inf")),
                 ("zero", 0.0), ("negative", -1.0),
                 ("subnormal", 2.0 ** -133)]:
    _expect_reject(
        f"I8: BF16 scale value {_lbl} rejects at load (test 21)",
        lambda _v=_v: fp8ops.load_scaled_fp8_component(
            _CaptureComp, _mk("i8_val.safetensors", {
                "a.weight": _i8((8, 4)),
                "a.weight_scale": _scalar(_v, torch.bfloat16),
                "a.comfy_quant": _I8DESC,
            }), torch.bfloat16, "", "ci-w"))

# Test 22 regression: BF16 scale on an fp8 cq-w file STILL rejects (the
# {BF16,F32} acceptance is int8-only; fp8 keeps D7 F32-only).
_expect_reject("I8: BF16 scale on fp8 cq-w still rejects (test 22)",
               _mk("i8_fp8bf16s.safetensors", {
                   "a.weight": _fp8((8, 4)),
                   "a.weight_scale": _scalar(0.01, torch.bfloat16),
                   "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
               }), "F32")

# Test 23: non-2D descriptored int8 weight rejects.
_expect_reject("I8: non-2D int8 weight rejects (test 23)",
               _mk("i8_3d.safetensors", {
                   "a.weight": _i8((2, 4, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
               }), "2D")

# Test 24 regressions on int8-shaped inputs: nvfp4 second-level scale and
# C-b co-presence fire their existing gates.
_expect_reject("I8: weight_scale_2 on int8 file hits nvfp4 reject (test 24a)",
               _mk("i8_ws2.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.weight_scale_2": _scalar(1.0),
                   "a.comfy_quant": _I8DESC,
               }), "weight_scale_2")
_expect_reject("I8: C-b marker + int8 descriptors reject (test 24b)",
               _mk("i8_cb.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
                   "scaled_fp8": torch.zeros(2, dtype=torch.float8_e4m3fn),
               }), "C-b")
_expect_reject("I8: oversize descriptor on int8 file hits D2 (test 24c)",
               _mk("i8_bigdesc.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": torch.zeros(5000, dtype=torch.uint8),
               }), "D2")
_expect_reject("I8: control-char key on int8 file rejects (test 24d)",
               _mk("i8_ctrl.safetensors", {
                   "a.weight": _i8((8, 4)),
                   "a.weight_scale": _scalar(0.02, torch.bfloat16),
                   "a.comfy_quant": _I8DESC,
                   "b\x01d.weight": torch.zeros(2, dtype=torch.bfloat16),
               }), "control")

# Req 50(d) pin: _validate_scale's DEFAULT stays F32-only — this default is
# what protects every legacy call site (incl. the DMR requant) unedited.
_dmr_pin = False
try:
    fp8ops._validate_scale("pin", torch.tensor(0.1, dtype=torch.bfloat16))
except fp8ops.ScaledFp8FormatError:
    _dmr_pin = True
check("I8: _validate_scale default rejects BF16 (F32-only pin, req 50d)",
      _dmr_pin)

# Test 5 (routing): ci-w routes through load_component with the
# int8-tensorwise detection line.
_buf = _io2.StringIO()
with _cl2.redirect_stdout(_buf):
    _m5 = edu.load_component(_CaptureComp, _i8file, torch.bfloat16,
                             base_path=_TMP, subfolder_hint="transformer")
check("I8: load_component routes ci-w + detection log (test 5)",
      hasattr(_m5, "_received_sd")
      and "int8-tensorwise" in _buf.getvalue(),
      f"log: {_buf.getvalue()[:200]!r}")


# R3/test 12 (threading): load_component → _load_single_weights →
# load_scaled_fp8_component carries dequant_fp8 through the utils layer.
_p12 = _mk("r3_thread.safetensors", {
    "a.weight": _fp8((16, 16)), "a.weight_scale": _scalar(0.01),
    "a.comfy_quant": _desc({"format": "float8_e4m3fn"}),
})
_buf = _io2.StringIO()
with _cl2.redirect_stdout(_buf):
    _m12 = edu.load_component(_CaptureComp, _p12, torch.bfloat16,
                              base_path=_TMP, subfolder_hint="transformer",
                              dequant_fp8=True)
check("R3: dequant_fp8 threads through load_component (test 12)",
      hasattr(_m12, "_received_sd") and "dequant-fp8 mode" in _buf.getvalue(),
      f"log: {_buf.getvalue()[:200]!r}")
# generate.py's hoist (quant_selected → dequant_fp8 at the override site) is
# verified by code review (req 44); it has no unit seam without a pipeline.

_expect_reject(
    "cq + C-b marker co-presence rejected (D3 NEGATIVE, test 5)",
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

# AMENDED by slice R1 (req 39): a fully-bare non-.weight fp8 tensor now
# upcasts (R1 battery covers the positive); the reject arm requires a bound
# scale — a present binding is never silently ignored.
p = _mk("fp8bias.safetensors", {
    "a.weight": _fp8((16, 16)),
    "a.weight_scale": _scalar(0.01),
    "a.input_scale": _scalar(0.5),
    "a.bias": _fp8((16,), seed=1),
    "a.bias.weight_scale": _scalar(0.01),   # bound → reject (R1/req 39)
})
_expect_reject("fp8 non-.weight tensor with bound scale rejected (R1 NEGATIVE)",
               lambda: _load(p), "non-.weight")

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
# ── ComfyUI-native Krea-2 → diffusers key converter (ADR-019 2026-07-07) ──
print("\n── Krea-2 ComfyUI→diffusers key converter ──")
_CK = krea2c.convert_krea2_comfy_key
# detection (prefix-tolerant, fail-closed)
check("krea2-detect: native keys → True",
      krea2c.is_krea2_comfy_checkpoint(
          ["blocks.0.attn.wq.weight", "blocks.0.mod.lin", "first.weight"]))
check("krea2-detect: with model.diffusion_model. prefix → True",
      krea2c.is_krea2_comfy_checkpoint(
          ["model.diffusion_model.blocks.0.attn.wq.weight",
           "model.diffusion_model.blocks.0.mod.lin"]))
check("krea2-detect: diffusers-format keys → False",
      not krea2c.is_krea2_comfy_checkpoint(
          ["transformer_blocks.0.attn.to_q.weight", "img_in.weight"]))
check("krea2-detect: unrelated/empty → False",
      not krea2c.is_krea2_comfy_checkpoint(["some.other.model.weight"])
      and not krea2c.is_krea2_comfy_checkpoint([]))
# per-rule key mapping (main blocks)
_RULES = {
    "blocks.5.attn.wq.weight": "transformer_blocks.5.attn.to_q.weight",
    "blocks.5.attn.wk.weight": "transformer_blocks.5.attn.to_k.weight",
    "blocks.5.attn.wv.weight": "transformer_blocks.5.attn.to_v.weight",
    "blocks.5.attn.wo.weight": "transformer_blocks.5.attn.to_out.0.weight",
    "blocks.5.attn.gate.weight": "transformer_blocks.5.attn.to_gate.weight",
    "blocks.5.attn.qknorm.qnorm.scale": "transformer_blocks.5.attn.norm_q.weight",
    "blocks.5.attn.qknorm.knorm.scale": "transformer_blocks.5.attn.norm_k.weight",
    "blocks.5.mlp.gate.weight": "transformer_blocks.5.ff.gate.weight",
    "blocks.5.mlp.up.weight": "transformer_blocks.5.ff.up.weight",
    "blocks.5.mlp.down.weight": "transformer_blocks.5.ff.down.weight",
    "blocks.5.mod.lin": "transformer_blocks.5.scale_shift_table",
    "blocks.5.prenorm.scale": "transformer_blocks.5.norm1.weight",
    "blocks.5.postnorm.scale": "transformer_blocks.5.norm2.weight",
    # text-fusion blocks
    "txtfusion.layerwise_blocks.1.attn.wk.weight":
        "text_fusion.layerwise_blocks.1.attn.to_k.weight",
    "txtfusion.refiner_blocks.0.mlp.down.weight":
        "text_fusion.refiner_blocks.0.ff.down.weight",
    "txtfusion.refiner_blocks.0.attn.qknorm.qnorm.scale":
        "text_fusion.refiner_blocks.0.attn.norm_q.weight",
    "txtfusion.projector.weight": "text_fusion.projector.weight",
    # top-level embeds / final layer
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "last.linear.bias": "final_layer.linear.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.0.weight": "time_mod_proj.weight",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
}
for _src, _want in _RULES.items():
    check(f"krea2-map: {_src} → {_want}", _CK(_src) == _want,
          detail=f"got {_CK(_src)!r}")
# state-dict conversion: strips prefix, renames, values by reference
_nat = {"model.diffusion_model.blocks.0.attn.wq.weight": torch.zeros(2),
        "model.diffusion_model.first.weight": torch.ones(3)}
_out = krea2c.convert_krea2_comfy_state_dict(_nat, "model.diffusion_model.")
check("krea2-sd: prefix stripped + keys renamed",
      set(_out) == {"transformer_blocks.0.attn.to_q.weight", "img_in.weight"})
check("krea2-sd: no residual native markers",
      not any(".wq." in k or k.startswith("first.") for k in _out))
check("krea2-sd: tensor values passed by reference (no copy)",
      _out["img_in.weight"] is _nat["model.diffusion_model.first.weight"])
# security: an unrecognised key passes through UNCHANGED (no injection of an
# arbitrary target; the loader's missing-key assertion catches real gaps)
check("krea2-map: unknown key passes through unchanged",
      _CK("totally.unknown.key.weight") == "totally.unknown.key.weight")
# reshape helper: flat scale_shift_table → 2D (numel-safe); allowlisted
_rsd = {
    "transformer_blocks.0.scale_shift_table": torch.arange(12).float(),  # (12,)
    "transformer_blocks.0.attn.to_q.weight": torch.zeros(6),             # numel-match, NOT allowlisted
    "final_layer.scale_shift_table": torch.zeros(4),                     # numel MISMATCH vs (2,3)
}
_mshapes = {
    "transformer_blocks.0.scale_shift_table": (6, 2),
    "transformer_blocks.0.attn.to_q.weight": (2, 3),
    "final_layer.scale_shift_table": (2, 3),
}
krea2c.reshape_to_model_shapes(_rsd, _mshapes)
check("krea2-reshape: flat scale_shift_table reshaped to model 2D shape",
      tuple(_rsd["transformer_blocks.0.scale_shift_table"].shape) == (6, 2))
check("krea2-reshape: reshape is lossless (values preserved row-major)",
      _rsd["transformer_blocks.0.scale_shift_table"][1, 0].item() == 2.0)
check("krea2-reshape: NON-scale_shift_table key NOT reshaped (allowlist bound)",
      tuple(_rsd["transformer_blocks.0.attn.to_q.weight"].shape) == (6,))
check("krea2-reshape: numel-mismatch scale_shift_table left alone (load raises)",
      tuple(_rsd["final_layer.scale_shift_table"].shape) == (4,))
# self-guard: control character in a key name is rejected without echoing it
_ctrl_raised = False
try:
    krea2c.convert_krea2_comfy_state_dict({"blocks.0.attn.wq.weight\x01": torch.zeros(1)})
except ValueError as _e:
    _ctrl_raised = "\x01" not in str(_e)  # rejected AND raw key not echoed
check("krea2-guard: control-char key rejected without echoing the raw key",
      _ctrl_raised)
# bundle detection + prefix-robust extraction (Dark Beast = TE+VAE+transformer)
check("krea2-bundle: TE+VAE keys detected as bundle",
      krea2c.is_krea2_bundle(["text_encoders.x.weight", "vae.y",
                              "model.diffusion_model.blocks.0.attn.wq.weight"]))
check("krea2-bundle: transformer-only NOT a bundle",
      not krea2c.is_krea2_bundle(
          ["model.diffusion_model.blocks.0.attn.wq.weight", "first.weight"]))
_bundle_sd = {
    "text_encoders.te.weight": torch.zeros(1),
    "vae.enc.weight": torch.zeros(1),
    "model.diffusion_model.blocks.0.attn.wq.weight": torch.zeros(2),
    "model.diffusion_model.first.weight": torch.zeros(3),
}
_ex = krea2c.extract_krea2_transformer_sd(_bundle_sd)
check("krea2-bundle: extraction keeps ONLY transformer keys (prefix-robust)",
      set(_ex) == {"model.diffusion_model.blocks.0.attn.wq.weight",
                   "model.diffusion_model.first.weight"})
_nb = {"blocks.0.attn.wq.weight": torch.zeros(1)}
check("krea2-bundle: non-bundle returned unchanged (same object)",
      krea2c.extract_krea2_transformer_sd(_nb) is _nb)
# prefix-robust conversion: strip_prefix=None but keys ARE prefixed (bundle
# dilutes the dominant-prefix detection to None)
_pr = krea2c.convert_krea2_comfy_state_dict(
    {"model.diffusion_model.blocks.0.attn.wq.weight": torch.zeros(1)}, None)
check("krea2-sd: prefix-robust — strips model.diffusion_model. with strip_prefix=None",
      "transformer_blocks.0.attn.to_q.weight" in _pr)
# build_krea2_transformer fail-closed: incomplete conversion raises (GPU-free
# via a tiny fake component class; code-review F8).
class _FakeKrea2(nn.Module):
    _keep_in_fp32_modules = []
    def __init__(self):
        super().__init__()
        self.img_in = nn.Linear(4, 3)  # img_in.weight (3,4) + img_in.bias (3,)
    @classmethod
    def load_config(cls, p, **k):
        return {}
    @classmethod
    def from_config(cls, c):
        return cls()
# complete: first.weight+bias → img_in.weight+bias (all model params filled)
_m = krea2c.build_krea2_transformer(
    _FakeKrea2, {"first.weight": torch.zeros(3, 4), "first.bias": torch.zeros(3)},
    "ignored", torch.float32)
check("krea2-build: complete native sd builds without error (fake model)",
      hasattr(_m, "img_in"))
# incomplete: only first.weight → img_in.bias missing → fail closed
_bc_raised = False
try:
    krea2c.build_krea2_transformer(
        _FakeKrea2, {"first.weight": torch.zeros(3, 4)}, "ignored", torch.float32)
except krea2c.Krea2ConversionError:
    _bc_raised = True
check("krea2-build: incomplete conversion → Krea2ConversionError (fail-closed)",
      _bc_raised)

# ── Embedded-adapter stripping (x3n0-style checkpoints, 2026-07-08) ────
# A checkpoint packing an un-merged LoRA (lora_A/B + .diff keys under a
# bare diffusion_model. prefix) loads its BASE with a loud notice; the
# adapter key shapes can never be model params so dropping them only
# converts a guaranteed-fatal strict-gate hit into a visible warning.
_adapter_sd = {
    "first.weight": torch.zeros(3, 4), "first.bias": torch.zeros(3),
    "diffusion_model.first.lora_A.weight": torch.zeros(2, 4),
    "diffusion_model.first.lora_B.weight": torch.zeros(3, 2),
    "diffusion_model.txtfusion.projector.diff": torch.zeros(3, 4),
}
_buf = _io2.StringIO()
with _cl2.redirect_stdout(_buf):
    _ma = krea2c.build_krea2_transformer(
        _FakeKrea2, dict(_adapter_sd), "ignored", torch.float32)
check("krea2-build: embedded adapter stripped, base loads",
      hasattr(_ma, "img_in"))
check("krea2-build: embedded-adapter notice fires (3 tensors named)",
      "adapter" in _buf.getvalue() and "3 adapter tensors" in _buf.getvalue(),
      f"log: {_buf.getvalue()[:200]!r}")
# The strict gate is NOT loosened: a non-adapter unknown key still raises,
# and the message now names the unexpected key (formatting fix).
_ug_raised = ""
try:
    krea2c.build_krea2_transformer(
        _FakeKrea2, {"first.weight": torch.zeros(3, 4),
                     "first.bias": torch.zeros(3),
                     "first.bogus": torch.zeros(3)}, "ignored", torch.float32)
except krea2c.Krea2ConversionError as _e:
    _ug_raised = str(_e)
check("krea2-build: non-adapter unknown key still fail-closed, named",
      "unexpected" in _ug_raised and "bogus" in _ug_raised,
      f"msg: {_ug_raised[:160]!r}")

# ── Usability fixes 2026-07-10: arch-mismatch diagnostics ────────────────
# Driven by a real incident: greed_int8.safetensors (a Z-Image transformer)
# was loaded as a --transformer override against a Krea base pipeline. The
# mismatch surfaced as `Krea2Transformer2DModel has no attribute
# from_single_file`, naming neither the file nor the real problem.

# Fix 1: a component_class with no from_single_file, reached with keys its
# native converter did not claim, reports the ARCHITECTURE mismatch.
class _NoSingleFileComp(nn.Module):
    """Stands in for Krea2Transformer2DModel: converter-only, no
    from_single_file."""


_expect_reject(
    "arch: converter-only class + unmatched keys names the mismatch (fix 1)",
    lambda: fp8ops.load_scaled_fp8_component(
        _NoSingleFileComp, _i8file, torch.bfloat16, "", "ci-w"),
    "different architecture")

# ...and the same converter-only class still loads its OWN native format: the
# krea2 branch must short-circuit BEFORE the hasattr raise, or fix 1 would have
# broken every native-Krea load. Driving a krea2-native file through the loader
# with the converter stubbed proves the ordering — reaching the stub means the
# hasattr raise was never consulted.
_k2_sentinel = object()
_real_build = krea2c.build_krea2_transformer
try:
    krea2c.build_krea2_transformer = lambda *a, **k: _k2_sentinel  # noqa: E731
    _k2file = _mk("i8_krea2_native.safetensors", {
        "blocks.0.attn.wq.weight": _i8((8, 4)),
        "blocks.0.attn.wq.weight_scale": _scalar(0.02, torch.bfloat16),
        "blocks.0.attn.wq.comfy_quant": _I8DESC,
        "blocks.0.mod.lin": torch.zeros(4, dtype=torch.bfloat16),
    })
    _k2out = fp8ops.load_scaled_fp8_component(
        _NoSingleFileComp, _k2file, torch.bfloat16, "", "ci-w")
finally:
    krea2c.build_krea2_transformer = _real_build
check("arch: krea2 native branch short-circuits before hasattr raise (fix 1)",
      _k2out is _k2_sentinel)

# Fix 2: detect_transformer_arch + _check_transformer_arch.
_ZIMAGE_KEYS = ["model.diffusion_model.cap_embedder.0.weight",
                "model.diffusion_model.context_refiner.0.attention.qkv.weight",
                "model.diffusion_model.noise_refiner.0.attention.out.weight"]
_KREA_NATIVE_KEYS = ["blocks.0.attn.wq.weight", "blocks.0.mod.lin"]
_KREA_DIFFUSERS_KEYS = ["text_fusion.0.weight", "time_mod_proj.weight",
                        "img_in.weight"]
_UNKNOWN_KEYS = ["double_blocks.0.img_attn.qkv.weight", "final_layer.linear.weight"]

check("arch: detects zimage (prefixed, ComfyUI-native)",
      edu.detect_transformer_arch(_ZIMAGE_KEYS) == "zimage")
check("arch: detects krea2 from ComfyUI-native markers",
      edu.detect_transformer_arch(_KREA_NATIVE_KEYS) == "krea2")
check("arch: detects krea2 from diffusers-native markers",
      edu.detect_transformer_arch(_KREA_DIFFUSERS_KEYS) == "krea2")
check("arch: unknown keys -> None, never a false 'mismatch'",
      edu.detect_transformer_arch(_UNKNOWN_KEYS) is None)
check("arch: flux-shaped keys stay unknown (ambiguous by design)",
      edu.detect_transformer_arch(
          ["double_blocks.0.x", "single_blocks.0.y"]) is None)


class ZImageTransformer2DModel(nn.Module):
    pass


class Krea2Transformer2DModel(nn.Module):
    pass


class _UnmappedTransformer(nn.Module):
    pass


# POSITIVE contradiction -> loud, and the message names file, found, expected.
_blocked = ""
try:
    edu._check_transformer_arch(Krea2Transformer2DModel, _ZIMAGE_KEYS,
                                "/models/Krea/greed_int8.safetensors")
except ValueError as _e:
    _blocked = str(_e)
check("arch: zimage file + krea2 class is BLOCKED (fix 2)", bool(_blocked))
check("arch: message names the file, found arch, and expected arch",
      all(s in _blocked for s in ("greed_int8.safetensors", "'zimage'",
                                  "'krea2'", "Krea2Transformer2DModel")),
      _blocked)

# F7: the basename ships with the checkpoint and reaches a terminal / the MCP
# log — it must be _safe_name'd, so control chars never emit verbatim.
_evil = ""
try:
    edu._check_transformer_arch(Krea2Transformer2DModel, _ZIMAGE_KEYS,
                                "/m/pwn\x1b[31m\n.safetensors")
except ValueError as _e:
    _evil = str(_e)
check("arch: mismatch message sanitizes the basename (F7)",
      bool(_evil) and "\x1b" not in _evil and "\n" not in _evil,
      repr(_evil[:120]))

# NEGATIVES — the guard must never block a legitimate load.
for _lbl, _cls, _keys in [
    ("matching zimage", ZImageTransformer2DModel, _ZIMAGE_KEYS),
    ("matching krea2 (native)", Krea2Transformer2DModel, _KREA_NATIVE_KEYS),
    ("matching krea2 (diffusers)", Krea2Transformer2DModel,
     _KREA_DIFFUSERS_KEYS),
    ("unknown checkpoint + mapped class", ZImageTransformer2DModel,
     _UNKNOWN_KEYS),
    ("known checkpoint + unmapped class", _UnmappedTransformer, _ZIMAGE_KEYS),
]:
    _ok = True
    try:
        edu._check_transformer_arch(_cls, _keys, "x.safetensors")
    except ValueError:
        _ok = False
    check(f"arch: {_lbl} proceeds (permissive, fix 2)", _ok)

# The guard runs on the real loader seam, before any converter.
_zfile = _mk("arch_zimage.safetensors", {
    "model.diffusion_model.cap_embedder.0.weight": torch.zeros(2, 2),
    "model.diffusion_model.context_refiner.0.attention.qkv.weight":
        torch.zeros(6, 2),
    "model.diffusion_model.noise_refiner.0.attention.out.weight":
        torch.zeros(2, 2),
})
_seam = ""
try:
    edu.load_component(Krea2Transformer2DModel, _zfile, torch.bfloat16,
                       base_path=_TMP, subfolder_hint="transformer")
except ValueError as _e:
    _seam = str(_e)
check("arch: load_component blocks the mismatch at the door (fix 2)",
      "Architecture mismatch" in _seam, _seam[:160])


# ── family-aware requant recipe match on the LoRA-merge path (2026-07-10;
# slice NV 2026-07-16 hardened it with a class ALLOWLIST, reqs 61-63) ──
# _requant_config_matching_base reads the base tensor's act_quant_kwargs to
# requantize a merged layer with the SAME recipe (weight-only vs dynamic-
# activation). None ⇒ weight-only; set ⇒ dynamic; ABSENT ⇒ loud raise (a
# torchao API change must never silently requantize a Z-Image weight-only base
# as dynamic-activation and reintroduce the speckle bug). Slice NV: the sniff
# is now gated on isinstance(data, Float8Tensor), so the three arms use REAL
# CPU-quantized Float8Tensors (weight-only quantize needs no fp8 GEMM
# hardware); the old duck-typed fakes now exercise the ALLOWLIST refusal.
from torchao.quantization import (                       # noqa: E402
    Float8DynamicActivationFloat8WeightConfig as _DynActCfg,
    Float8Tensor as _F8T,
    Float8WeightOnlyConfig as _WOnlyCfg,
    quantize_ as _tao_quantize,
)


def _real_f8_weight_only():
    m = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
    _tao_quantize(m, _WOnlyCfg())
    return m.weight.data


_f8_none = _real_f8_weight_only()           # genuine weight-only: akw None
check("requant-match arms are real Float8Tensors (test-fixture sanity)",
      isinstance(_f8_none, _F8T) and _f8_none.act_quant_kwargs is None)
check("requant-match: base act_quant_kwargs=None -> weight-only config",
      isinstance(fp8ops._requant_config_matching_base(
          _f8_none, "layer.weight", "[t]"), _WOnlyCfg))
_f8_set = _real_f8_weight_only()
_f8_set.act_quant_kwargs = object()         # non-None ⇒ dynamic-activation
check("requant-match: base act_quant_kwargs set -> dynamic-activation config",
      isinstance(fp8ops._requant_config_matching_base(
          _f8_set, "layer.weight", "[t]"), _DynActCfg))
_f8_gone = _real_f8_weight_only()
object.__delattr__(_f8_gone, "act_quant_kwargs")   # simulate API drift
_raised = False
try:
    fp8ops._requant_config_matching_base(_f8_gone, "layer.weight", "[t]")
except RuntimeError as _e:
    _raised = "act_quant_kwargs" in str(_e)
check("requant-match: absent act_quant_kwargs RAISES (no silent mismatch)",
      _raised)


# ──────────────────────────────────────────────────────────────────────
print("── slice NV: nvfp4 merge refusal (security reqs 61-65) ────────")
# review-slice-NV-nvfp4-merge-guard-2026-07-16.md. NVFP4Tensor ALSO carries
# act_quant_kwargs (None when weight-only), so without the req-61 class
# allowlist a LoRA direct merge would silently requantize an nvfp4 base as
# fp8 — a mixed-representation model with no error.
from torchao.prototype.mx_formats import (               # noqa: E402
    NVFP4WeightOnlyConfig as _NV4WOnlyCfg,
)


def _real_nvfp4_linear():
    m = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
    _tao_quantize(m, _NV4WOnlyCfg())
    return m


_nv4 = _real_nvfp4_linear().weight.data
check("tripwire (req 63b): NVFP4Tensor is NOT a Float8Tensor subclass — a "
      "torchao bump that changes this defeats the allowlist and must fail CI",
      not isinstance(_nv4, _F8T))
check("the misclassification is real: NVFP4Tensor carries act_quant_kwargs",
      hasattr(_nv4, "act_quant_kwargs"))

# req 61: allowlist refusal names the class and the escape routes.
_msg = ""
try:
    fp8ops._requant_config_matching_base(_nv4, "blk.attn.weight", "[t]")
except RuntimeError as _e:
    _msg = str(_e)
check("req 61: NVFP4 base REFUSED by the recipe matcher (NEGATIVE)",
      "NVFP4Tensor" in _msg and "PEFT" in _msg, _msg[:160])

# req 63c: generic allowlist — ANY non-Float8Tensor refuses, not just NVFP4
# (a third torchao rep must hit the same wall; duck-typed akw is not enough).
class _FakeAkwNone:
    act_quant_kwargs = None


_raised = False
try:
    fp8ops._requant_config_matching_base(_FakeAkwNone(), "layer.weight", "[t]")
except RuntimeError:
    _raised = True
check("req 63c: duck-typed akw-carrier (non-Float8Tensor) refused (NEGATIVE)",
      _raised)

# reqs 61+62 through the dispatcher: a real nvfp4 Parameter refuses, weights
# bit-identical, backup dict UNTOUCHED (the gate fires before the backup
# record — no stale entry on the refusal path).
class _NVHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = _real_nvfp4_linear()


_nvh = _NVHost()
_nv_before = _nvh.blk.weight.data.dequantize().clone()
_nv_bk = {}
_raised = False
try:
    fp8ops.apply_merge_delta(_nvh, "blk.weight",
                             torch.randn(32, 32) * 0.01, _nv_bk)
except RuntimeError as _e:
    _raised = "NVFP4Tensor" in str(_e)
check("req 61/62: apply_merge_delta refuses nvfp4 Parameter (NEGATIVE)",
      _raised)
check("req 62: refused merge left weights bit-identical",
      torch.equal(_nvh.blk.weight.data.dequantize(), _nv_before))
check("req 62: refused merge recorded NO backup entry (no stale state)",
      _nv_bk == {})

# Positive control (req 63d): the genuine-Float8 torchao merge path still
# works end-to-end after the gate hoist (weight-only requants on CPU).
class _F8Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        _tao_quantize(self.blk, _WOnlyCfg())


_f8h = _F8Host()
_f8_bk = {}
check("req 63d: Float8 weight-only base still merges (kind 'torchao')",
      fp8ops.apply_merge_delta(_f8h, "blk.weight",
                               torch.randn(32, 32, dtype=torch.bfloat16)
                               * 0.01, _f8_bk) == "torchao")
check("req 63d: Float8 merge recorded its kind-tagged backup",
      _f8_bk.get("blk.weight", {}).get("kind") == "torchao_param")

# req 64: a torchao rep held as a BUFFER (not nn.Parameter) skips branch (b),
# reports its LOGICAL dtype (so branch (c)'s fp8-dtype sniff misses it), and
# must hit the (c2) guard — never branch (d)'s in-place add_.
class _NVBufHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = nn.Module()
        self.blk.register_buffer("weight", _real_nvfp4_linear().weight.data)


_nvb = _NVBufHost()
_raised = False
try:
    fp8ops.apply_merge_delta(_nvb, "blk.weight",
                             torch.randn(32, 32) * 0.01, {})
except RuntimeError as _e:
    _raised = "req 64" in str(_e) or "outside an nn.Parameter" in str(_e)
check("req 64: torchao tensor as buffer refused, no plain-path fallthrough "
      "(NEGATIVE)", _raised)

# req 65: the all-or-nothing entry gate — refuses BEFORE the first merge so
# a direct-merge adapter can never leave a partially-merged nvfp4 model.
_raised = False
try:
    fp8ops.refuse_unmergeable_base(_NVHost(), log_prefix="[t]")
except RuntimeError as _e:
    _raised = "BEFORE any target" in str(_e) and "NVFP4Tensor" in str(_e)
check("req 65: entry gate refuses an nvfp4-bearing base (NEGATIVE)", _raised)
_raised = False
try:
    fp8ops.refuse_unmergeable_base(_NVBufHost(), log_prefix="[t]")
except RuntimeError:
    _raised = True
check("req 65: entry gate sees nvfp4 held as a .weight buffer too", _raised)


class _OKHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.plain = nn.Linear(8, 8, bias=False)
        self.f8 = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        _tao_quantize(self.f8, _WOnlyCfg())
        self.sf8 = fp8ops.ScaledFp8Linear(
            _fp8((8, 8)), _scalar(0.02), None, None)


check("req 65: entry gate PASSES plain + Float8 + ScaledFp8Linear bases",
      fp8ops.refuse_unmergeable_base(_OKHost(), log_prefix="[t]") is None)


# ──────────────────────────────────────────────────────────────────────
# Slice NF4: bitsandbytes 4-bit single-file consumption (security review
# reqs 67-90 + delta addendum). Golden vectors are REAL bitsandbytes
# 0.49.2 output (dequantize_4bit on the projectGaia file / a quantize_4bit
# round-trip), so the decode is proven against bnb, not against itself.
print("── slice NF4: bnb 4-bit classify + dequant (reqs 67-90) ───────")

#: Canonical NF4 codebook as stored in the real projectGaia file.
_NF4_MAP = [-1.0, -0.696193, -0.525073, -0.394917, -0.284441, -0.184773,
            -0.09105, 0.0, 0.07958, 0.16093, 0.246112, 0.337915, 0.44071,
            0.562617, 0.722957, 1.0]


def _qs_blob(obj=None, raw=None):
    data = raw if raw is not None else json.dumps(obj).encode()
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def _packed(byte_list):
    return torch.tensor([[b] for b in byte_list], dtype=torch.uint8)


def _nf4_family(base, packed, absmax, shape, blocksize=64, flavor="nf4",
                qmap=None, qs=None):
    """Four-key bnb4 family fixture, measured projectGaia layout."""
    state = qs if qs is not None else {
        "quant_type": flavor, "blocksize": blocksize,
        "dtype": "bfloat16", "shape": list(shape)}
    return {
        base: packed,
        base + ".absmax": torch.tensor(absmax, dtype=torch.float32),
        base + ".quant_map": (qmap if qmap is not None
                              else torch.tensor(_NF4_MAP,
                                                dtype=torch.float32)),
        base + ".quant_state.bitsandbytes__" + flavor: _qs_blob(state),
    }


# Golden #1: first 8 packed bytes of the real projectGaia
# double_blocks.0.img_attn.proj.weight, absmax[0]=0.03125; expected values
# from bnb 0.49.2 dequantize_4bit on GPU. The byte pattern is
# nibble-ASYMMETRIC (e.g. 0x6E → hi 6, lo 14), so a low-nibble-first
# implementation decodes DIFFERENT values and fails (req 74 negative).
_G1_BYTES = [110, 90, 88, 44, 228, 163, 9, 25]
_G1_EXPECT = [-0.002838, 0.022583, -0.005768, 0.00769, -0.005768, 0.002487,
              -0.016357, 0.013794, 0.022583, -0.008911, 0.00769, -0.012329,
              -0.03125, 0.005035, -0.021729, 0.005035]
# Golden #2: bnb quantize_4bit round-trip, shape (2,5) — partial final
# block (numel 10 < blocksize 64), absmax from bnb itself.
_G2_BYTES = [108, 209, 242, 94, 128]
_G2_EXPECT = [-0.007697, 0.037257, 0.047563, -0.058856, 0.08454, -0.044389,
              -0.015621, 0.061118, 0.006728, -0.08454]

# Positive: UNET-style file classifies bnb4 (req 67) and decodes to the
# bnb golden values; family keys consumed, plain key passes (req 80).
_p_nf4 = _mk("nf4_unet.safetensors", {
    **_nf4_family("double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                  [0.03125], (2, 8)),
    "double_blocks.0.norm.weight": torch.ones(4, dtype=torch.bfloat16),
})
_v, _info = _classify(_p_nf4)
check("NF4: UNET-style file classifies bnb4 (req 67)",
      _v == "bnb4" and _info.get("n_bnb4") == 1
      and _info.get("flavors") == ["nf4"], f"got {_v} {_info.get('n_bnb4')}")
_buf = _io2.StringIO()
with _cl2.redirect_stdout(_buf):
    _m = fp8ops._load_bnb4_component(
        _CaptureComp, _p_nf4, torch.bfloat16, "")
_rsd = _m._received_sd
check("NF4: decode matches bitsandbytes golden #1 (req 74 — hi-nibble "
      "order proven against bnb 0.49.2)",
      torch.allclose(_rsd["double_blocks.0.mlp.weight"].float().flatten(),
                     torch.tensor(_G1_EXPECT), atol=2e-4),
      f"got {_rsd['double_blocks.0.mlp.weight'].float().flatten()[:4]}")
check("NF4: family keys consumed, plain key passes, all-float dict "
      "(reqs 79/80)",
      set(_rsd) == {"double_blocks.0.mlp.weight",
                    "double_blocks.0.norm.weight"}
      and all(t.is_floating_point() for t in _rsd.values()))
check("NF4: one aggregate notice names flavor + count (req 82)",
      "1 nf4 4-bit" in _buf.getvalue(), f"log: {_buf.getvalue()[:200]!r}")

# Low-nibble-first negative (req 74): swapped unpack ≠ golden.
_lo_first = torch.stack(
    (torch.tensor(_G1_BYTES, dtype=torch.uint8) & 0xF,
     (torch.tensor(_G1_BYTES, dtype=torch.uint8) >> 4) & 0xF),
    dim=1).reshape(-1).long()
_lo_vals = torch.tensor(_NF4_MAP)[_lo_first] * 0.03125
check("NF4: low-nibble-first decode differs from golden (req 74 NEGATIVE "
      "— the byte pattern is order-asymmetric)",
      not torch.allclose(_lo_vals, torch.tensor(_G1_EXPECT), atol=2e-4))

# Golden #2: partial final block (req 73).
_p_partial = _mk("nf4_partial.safetensors", _nf4_family(
    "double_blocks.0.a.weight", _packed(_G2_BYTES), [0.08454], (2, 5)))
_m2 = fp8ops._load_bnb4_component(_CaptureComp, _p_partial,
                                  torch.bfloat16, "")
check("NF4: partial-block decode matches bnb golden #2 (reqs 72/73)",
      torch.allclose(
          _m2._received_sd["double_blocks.0.a.weight"].float().flatten(),
          torch.tensor(_G2_EXPECT), atol=2e-4))

# Pad nibble inert (req 73): odd numel — only the pad nibble differs.
_pa = _mk("nf4_pad_a.safetensors", _nf4_family(
    "double_blocks.0.b.weight", _packed([0x12, 0x30]), [1.0], (1, 3)))
_pb = _mk("nf4_pad_b.safetensors", _nf4_family(
    "double_blocks.0.b.weight", _packed([0x12, 0x3F]), [1.0], (1, 3)))
check("NF4: pad nibble of odd-numel tensor is inert (req 73)",
      torch.equal(
          fp8ops._load_bnb4_component(_CaptureComp, _pa, torch.bfloat16,
                                      "")._received_sd[
              "double_blocks.0.b.weight"],
          fp8ops._load_bnb4_component(_CaptureComp, _pb, torch.bfloat16,
                                      "")._received_sd[
              "double_blocks.0.b.weight"]))

# 0xFF byte → indices (15, 15) → codebook max both nibbles (req 74).
_pff = _mk("nf4_ff.safetensors", _nf4_family(
    "double_blocks.0.c.weight", _packed([0xFF]), [2.0], (1, 2)))
check("NF4: byte 0xFF decodes to (map[15], map[15]) * absmax (req 74)",
      torch.allclose(
          fp8ops._load_bnb4_component(_CaptureComp, _pff, torch.bfloat16,
                                      "")._received_sd[
              "double_blocks.0.c.weight"].float(),
          torch.tensor([[2.0, 2.0]]), atol=1e-2))

# Zero absmax is LEGAL (req 76): block decodes to zeros.
_pz = _mk("nf4_zero.safetensors", _nf4_family(
    "double_blocks.0.d.weight", _packed([0x59]), [0.0], (1, 2)))
check("NF4: zero absmax accepted, block decodes to zeros (req 76)",
      torch.equal(
          fp8ops._load_bnb4_component(_CaptureComp, _pz, torch.bfloat16,
                                      "")._received_sd[
              "double_blocks.0.d.weight"],
          torch.zeros(1, 2, dtype=torch.bfloat16)))

# fp4 flavor loads (req 84 / verdict a) — same machinery, marker-driven.
_p_fp4 = _mk("fp4_flavor.safetensors", _nf4_family(
    "double_blocks.0.e.weight", _packed(_G2_BYTES), [0.08454], (2, 5),
    flavor="fp4"))
_vf, _infof = _classify(_p_fp4)
check("NF4: __fp4 marker classifies bnb4 with flavor fp4 (req 84)",
      _vf == "bnb4" and _infof.get("flavors") == ["fp4"])
check("NF4: fp4 flavor loads through the same decode (verdict a)",
      hasattr(fp8ops._load_bnb4_component(_CaptureComp, _p_fp4,
                                          torch.bfloat16, ""),
              "_received_sd"))

# AIO shape (delta req 68): the motivating real-file layout — NF4
# transformer under model.diffusion_model.* + fp8 T5 + .SCB int8 under
# text_encoders.* — classifies bnb4 (NEVER cc: precedence pin) and loads
# with the exempt subtrees absent from the handed-over dict (reqs 78/5).
_p_aio = _mk("nf4_aio.safetensors", {
    **_nf4_family("model.diffusion_model.double_blocks.0.mlp.weight",
                  _packed(_G1_BYTES), [0.03125], (2, 8)),
    "model.diffusion_model.double_blocks.0.norm.weight":
        torch.ones(4, dtype=torch.bfloat16),
    "text_encoders.t5xxl.block.0.k.weight": _fp8((8, 8), seed=3),
    "text_encoders.t5xxl.block.0.k.weight.SCB":
        torch.ones(8, dtype=torch.float32),
    "text_encoders.clip.int8.weight":
        torch.ones(4, 4, dtype=torch.int8),
    "vae.decoder.conv.weight": torch.ones(2, 2, dtype=torch.float32),
})
_va, _infoa = _classify(_p_aio)
check("NF4: AIO with fp8 TE classifies bnb4, never cc (delta req 68 "
      "precedence pin)", _va == "bnb4", f"got {_va}")
_ma = fp8ops._load_bnb4_component(
    _CaptureComp, _p_aio, torch.bfloat16, "",
    strip_prefix="model.diffusion_model.")
check("NF4: AIO load drops TE/VAE subtrees — no exempt key, no fp8/int8 "
      "dtype in the handed-over dict (reqs 78/79, invariant 5)",
      set(_ma._received_sd) == {"double_blocks.0.mlp.weight",
                               "double_blocks.0.norm.weight"}
      and all(t.is_floating_point()
              for t in _ma._received_sd.values()))

# Delta alias closure: first_stage_model. is BOTH an exempt root AND a
# dominant-prefix candidate — raw-key drop must remove it BEFORE the strip
# could pull it into the transformer namespace (delta reqs 78/81).
_p_alias = _mk("nf4_alias.safetensors", {
    **_nf4_family("double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                  [0.03125], (2, 8)),
    **{f"first_stage_model.evil{i}.weight": _fp8((4, 4), seed=i)
       for i in range(6)},
})
_malias = fp8ops._load_bnb4_component(
    _CaptureComp, _p_alias, torch.bfloat16, "",
    strip_prefix="first_stage_model.")
check("NF4: exempt-root-as-dominant-prefix dropped RAW, never stripped "
      "into the namespace (delta reqs 78/81 alias closure)",
      set(_malias._received_sd) == {"double_blocks.0.mlp.weight"})

# Subtree filter is bnb4-ONLY (req 78 invariant-1 negative): an fp8 ca
# fixture with a text_encoders. key loads exactly as today — key SURVIVES.
_p_ca_te = _mk("ca_with_te.safetensors", {
    "blocks.0.mlp.weight": _fp8((64, 32)),
    "blocks.0.mlp.weight_scale": _scalar(0.01),
    "blocks.0.mlp.input_scale": _scalar(0.02),
    "text_encoders.keepme.weight": torch.ones(4, dtype=torch.float32),
})
_mca = fp8ops.load_scaled_fp8_component(
    _CaptureComp, _p_ca_te, torch.bfloat16, "", "ca", dequant_fp8=True)
check("NF4: subtree filter does NOT run for fp8 variants — "
      "text_encoders. key survives a ca load (req 78 NEGATIVE)",
      "text_encoders.keepme.weight" in _mca._received_sd)

# Constant unity (delta): one shared tuple drives BOTH the req-68 scan
# exemption and the req-78 drop — drift between them is the smuggle gap.
check("NF4: single exempt-roots constant, all six roots present (delta)",
      isinstance(fp8ops._BNB4_NON_TRANSFORMER_ROOTS, tuple)
      and set(fp8ops._BNB4_NON_TRANSFORMER_ROOTS) == {
          "text_encoders.", "text_encoder.", "vae.", "first_stage_model.",
          "conditioner.", "cond_stage_model."})

# ── NF4 negatives: classify-time (reqs 67-72) ──────────────────────────
check("NF4: .absmax/.quant_map WITHOUT marker → (None, {}) — suffixes "
      "alone never fire the sniff (req 67 NEGATIVE)",
      _classify(_mk("no_marker.safetensors", {
          "a.weight": torch.ones(2, 2, dtype=torch.float32),
          "a.weight.absmax": torch.ones(1, dtype=torch.float32),
          "a.weight.quant_map": torch.ones(16, dtype=torch.float32),
      })) == (None, {}))
check("NF4: .SCB int8 file with no 4-bit marker stays unrecognized "
      "(req 68 NEGATIVE)",
      _classify(_mk("scb_only.safetensors", {
          "a.weight": torch.ones(2, 2, dtype=torch.int8),
          "a.weight.SCB": torch.ones(2, dtype=torch.float32),
      })) == (None, {}))
_expect_reject("NF4: fp8 tensor in the TRANSFORMER namespace beside bnb4 "
               "markers rejects (req 68 NEGATIVE)",
               _mk("hybrid_fp8.safetensors", {
                   **_nf4_family("double_blocks.0.mlp.weight",
                                 _packed(_G1_BYTES), [0.03125], (2, 8)),
                   "double_blocks.1.x.weight": _fp8((4, 4)),
               }), "req 68")
_expect_reject("NF4: comfy_quant marker beside bnb4 markers rejects "
               "(req 68 NEGATIVE)",
               _mk("hybrid_cq.safetensors", {
                   **_nf4_family("double_blocks.0.mlp.weight",
                                 _packed(_G1_BYTES), [0.03125], (2, 8)),
                   "double_blocks.1.y.comfy_quant":
                       _desc({"format": "float8_e4m3fn"}),
               }), "req 68")
_expect_reject("NF4: marker without packed base rejects (req 69 NEGATIVE)",
               _mk("dangling_marker.safetensors", {
                   "double_blocks.0.z.weight.quant_state.bitsandbytes__nf4":
                       _qs_blob({"shape": [2, 2], "blocksize": 64}),
               }), "req 69")
_expect_reject("NF4: absmax member missing rejects (req 69 NEGATIVE)",
               _mk("no_absmax.safetensors", {
                   k: v for k, v in _nf4_family(
                       "double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                       [0.03125], (2, 8)).items()
                   if not k.endswith(".absmax")
               }), "req 69")
_expect_reject("NF4: malformed family under an EXEMPT root still rejects "
               "at classify (req 69 / delta ruling 2 NEGATIVE)",
               _mk("bad_te_family.safetensors", {
                   **_nf4_family("double_blocks.0.mlp.weight",
                                 _packed(_G1_BYTES), [0.03125], (2, 8)),
                   **{k: v for k, v in _nf4_family(
                       "text_encoders.t5.q.weight", _packed([0x11]),
                       [1.0], (1, 2)).items()
                      if not k.endswith(".absmax")},
               }), "req 69")
_expect_reject("NF4: quant_map wrong shape rejects (req 69 NEGATIVE)",
               _mk("bad_qmap.safetensors", _nf4_family(
                   "double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                   [0.03125], (2, 8),
                   qmap=torch.ones(8, dtype=torch.float32))), "req 69")
_expect_reject("NF4: both flavors on one base rejects as ambiguous "
               "(req 69 NEGATIVE)",
               _mk("both_flavors.safetensors", {
                   **_nf4_family("double_blocks.0.mlp.weight",
                                 _packed(_G1_BYTES), [0.03125], (2, 8)),
                   "double_blocks.0.mlp.weight.quant_state."
                   "bitsandbytes__fp4":
                       _qs_blob({"shape": [2, 8], "blocksize": 64}),
               }), "ambiguous")
_expect_reject("NF4: quant-state blob over 4096 bytes rejects at header "
               "(req 69 NEGATIVE)",
               _mk("big_blob.safetensors", _nf4_family(
                   "double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                   [0.03125], (2, 8),
                   qs=None) | {
                   "double_blocks.0.mlp.weight.quant_state."
                   "bitsandbytes__nf4":
                       torch.zeros(5000, dtype=torch.uint8)},
               ), "req 69")
_expect_reject("NF4: non-JSON quant-state rejects (req 70 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("bad_json.safetensors", _nf4_family(
                       "double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                       [0.03125], (2, 8)) | {
                       "double_blocks.0.mlp.weight.quant_state."
                       "bitsandbytes__nf4":
                           _qs_blob(raw=b"\xff\xfe not json")},
                   ), torch.bfloat16, ""), "req 70")
_expect_reject("NF4: JSON list root rejects (req 70 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("list_root.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125],
                                     (2, 8)),
                       "double_blocks.0.mlp.weight.quant_state."
                       "bitsandbytes__nf4": _qs_blob(raw=b"[1, 2]"),
                   }), torch.bfloat16, ""), "req 70")
_expect_reject("NF4: nested_absmax field rejects — double-quantized "
               "unsupported (req 71 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("nested.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125], (2, 8),
                                     qs={"shape": [2, 8], "blocksize": 64,
                                         "nested_absmax": True}),
                   }), torch.bfloat16, ""), "double-quantized")
_expect_reject("NF4: non-power-of-two blocksize rejects (reqs 71/86 "
               "NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("bs63.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125], (2, 8),
                                     qs={"shape": [2, 8],
                                         "blocksize": 63}),
                   }), torch.bfloat16, ""), "req")
_expect_reject("NF4: quant_type/marker mismatch rejects (req 71 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("qt_mismatch.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125], (2, 8),
                                     qs={"quant_type": "fp4",
                                         "shape": [2, 8],
                                         "blocksize": 64}),
                   }), torch.bfloat16, ""), "mismatch")
_expect_reject("NF4: huge declared shape over tiny packed rejects BEFORE "
               "allocation (req 72 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("shape_bomb.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125], (2, 8),
                                     qs={"shape": [65536, 65536],
                                         "blocksize": 64}),
                   }), torch.bfloat16, ""), "req 72")
_expect_reject("NF4: absmax off-by-one rejects (req 72 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("absmax_short.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(list(range(64)) * 2),
                                     [0.5, 0.5, 0.5], (4, 64)),
                   }), torch.bfloat16, ""), "req 72")
_expect_reject("NF4: NaN codebook rejects (req 75 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("nan_map.safetensors", _nf4_family(
                       "double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                       [0.03125], (2, 8),
                       qmap=torch.tensor([float("nan")] + _NF4_MAP[1:],
                                         dtype=torch.float32))),
                   torch.bfloat16, ""), "req 75")
_expect_reject("NF4: codebook |v| > 1 rejects (req 75 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("amp_map.safetensors", _nf4_family(
                       "double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                       [0.03125], (2, 8),
                       qmap=torch.tensor([448.0] + _NF4_MAP[1:],
                                         dtype=torch.float32))),
                   torch.bfloat16, ""), "req 75")
_expect_reject("NF4: negative absmax rejects, naming the index (req 76 "
               "NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("neg_absmax.safetensors", _nf4_family(
                       "double_blocks.0.mlp.weight", _packed(_G1_BYTES),
                       [-0.5], (2, 8))), torch.bfloat16, ""), "req 76")
_expect_reject("NF4: post-strip key collision rejects, never last-wins "
               "(req 77 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("collide.safetensors", {
                       **_nf4_family(
                           "model.diffusion_model.double_blocks.0.mlp"
                           ".weight", _packed(_G1_BYTES), [0.03125],
                           (2, 8)),
                       "double_blocks.0.mlp.weight":
                           torch.ones(2, 8, dtype=torch.bfloat16),
                   }), torch.bfloat16, "",
                   strip_prefix="model.diffusion_model."), "req 77")
_expect_reject("NF4: stray marker-less U8 tensor rejects at the residual "
               "scan (req 79 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("stray_u8.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125],
                                     (2, 8)),
                       "double_blocks.9.evil": torch.ones(
                           4, dtype=torch.uint8),
                   }), torch.bfloat16, ""), "req 79")
_expect_reject("NF4: still-prefixed namespace (defeated prefix detection) "
               "rejects with the prefix named (req 88 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("no_strip.safetensors", _nf4_family(
                       "model.diffusion_model.double_blocks.0.mlp.weight",
                       _packed(_G1_BYTES), [0.03125], (2, 8))),
                   torch.bfloat16, "", strip_prefix=None), "req 88")
_expect_reject("NF4: direct loader call re-asserts absmax dtype without "
               "the classifier (req 81 two-point NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("f64_absmax.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125],
                                     (2, 8)),
                       "double_blocks.0.mlp.weight.absmax":
                           torch.tensor([0.03125], dtype=torch.float64),
                   }), torch.bfloat16, ""), "req 69")

# Load-stage purity gate (delta req 81 — the AUTHORITATIVE flavor scan):
# direct loader calls bypass the classifier, so the surviving-dict scan
# must reject BOTH halves on its own — fp8/int8 dtypes AND foreign
# marker keys.
_expect_reject("NF4: direct loader call rejects fp8 tensor in the "
               "surviving dict (delta req 81 purity NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("purity_fp8.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125],
                                     (2, 8)),
                       "double_blocks.1.x.weight": _fp8((4, 4)),
                   }), torch.bfloat16, ""), "req 68/81")
_expect_reject("NF4: direct loader call rejects foreign marker key in "
               "the surviving dict (delta req 81 purity NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("purity_marker.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125],
                                     (2, 8)),
                       "double_blocks.1.x.weight_scale": _scalar(0.01),
                   }), torch.bfloat16, ""), "req 68/81")

# Dropped-family notice (delta req 82): a COMPLETE NF4 family under an
# exempt root classifies, loads, is absent from the dict, and the
# aggregate not-materialized line fires; surviving notice counts ONLY the
# materialized family.
_p_dropfam = _mk("nf4_dropfam.safetensors", {
    **_nf4_family("model.diffusion_model.double_blocks.0.mlp.weight",
                  _packed(_G1_BYTES), [0.03125], (2, 8)),
    **_nf4_family("text_encoders.t5.q.weight", _packed(_G2_BYTES),
                  [0.08454], (2, 5)),
})
check("NF4: complete family under exempt root still classifies bnb4 "
      "(delta ruling 2)", _classify(_p_dropfam)[0] == "bnb4")
_buf = _io2.StringIO()
with _cl2.redirect_stdout(_buf):
    _mdf = fp8ops._load_bnb4_component(
        _CaptureComp, _p_dropfam, torch.bfloat16, "",
        strip_prefix="model.diffusion_model.")
check("NF4: dropped family absent from dict; not-materialized line + "
      "surviving-only count (delta req 82)",
      set(_mdf._received_sd) == {"double_blocks.0.mlp.weight"}
      and "NOT materialized" in _buf.getvalue()
      and "1 nf4 4-bit" in _buf.getvalue(),
      f"log: {_buf.getvalue()[:300]!r}")

# Cheap contract sub-negatives (review finding 4).
_expect_reject("NF4: deep-nested JSON (RecursionError class) rejects as "
               "format error (req 70 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("recursion.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125],
                                     (2, 8)),
                       "double_blocks.0.mlp.weight.quant_state."
                       "bitsandbytes__nf4": _qs_blob(raw=b"[" * 2000),
                   }), torch.bfloat16, ""), "req 70")
_expect_reject("NF4: scalar JSON root rejects (req 70 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("scalar_root.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125],
                                     (2, 8)),
                       "double_blocks.0.mlp.weight.quant_state."
                       "bitsandbytes__nf4": _qs_blob(raw=b"42"),
                   }), torch.bfloat16, ""), "req 70")
_expect_reject("NF4: boolean blocksize rejects — bool is not an int here "
               "(req 71 NEGATIVE)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("bool_bs.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed(_G1_BYTES), [0.03125], (2, 8),
                                     qs={"shape": [2, 8],
                                         "blocksize": True}),
                   }), torch.bfloat16, ""), "req")
_expect_reject("NF4: packed larger than declared shape rejects (req 72 "
               "NEGATIVE, other direction)",
               lambda: fp8ops._load_bnb4_component(
                   _CaptureComp, _mk("packed_big.safetensors", {
                       **_nf4_family("double_blocks.0.mlp.weight",
                                     _packed([1, 2, 3]), [1.0], (1, 2)),
                   }), torch.bfloat16, ""), "req 72")
_expect_reject("NF4: flat [N] packed layout rejects — [N, 1] pin "
               "(req 89 NEGATIVE)",
               _mk("flat_packed.safetensors", {
                   **_nf4_family("double_blocks.0.mlp.weight",
                                 torch.tensor(_G1_BYTES,
                                              dtype=torch.uint8),
                                 [0.03125], (2, 8)),
               }), "req")
_expect_reject("NF4: in-namespace int8 tensor rejects at classify "
               "(req 68 / ruling 3 NEGATIVE)",
               _mk("ns_int8.safetensors", {
                   **_nf4_family("double_blocks.0.mlp.weight",
                                 _packed(_G1_BYTES), [0.03125], (2, 8)),
                   "double_blocks.2.q.weight":
                       torch.ones(4, 4, dtype=torch.int8),
               }), "req 68")
_expect_reject("NF4: dangling absmax beside a VALID family rejects "
               "(req 69 NEGATIVE)",
               _mk("extra_dangling.safetensors", {
                   **_nf4_family("double_blocks.0.mlp.weight",
                                 _packed(_G1_BYTES), [0.03125], (2, 8)),
                   "double_blocks.3.z.weight.absmax":
                       torch.ones(1, dtype=torch.float32),
               }), "req 69")

# Regression pin (req 67b): existing variants classify identically with
# the NF4 code present — spot-check the suite's own earlier fixtures.
check("NF4: ca fixture still classifies ca (req 67b regression)",
      _classify(p_ca)[0] == "ca")


shutil.rmtree(_TMP, ignore_errors=True)
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed")
print("─" * 50)
sys.exit(1 if failed else 0)
