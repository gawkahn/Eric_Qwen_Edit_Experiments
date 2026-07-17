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


shutil.rmtree(_TMP, ignore_errors=True)
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed")
print("─" * 50)
sys.exit(1 if failed else 0)
