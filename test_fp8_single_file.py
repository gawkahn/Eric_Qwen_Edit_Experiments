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
print("── real-collection spot checks (skip-as-pass if absent) ───────")

_REAL = [
    ("/home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models/"
     "diffusion_models/Flux.2-Klein-9B-base/flux-2-klein-base-9b-fp8.safetensors",
     "ca"),
    ("/home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models/"
     "diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "cb"),
]
for rp, want in _REAL:
    if os.path.exists(rp):
        v, _ = _classify(rp)
        check(f"real file {os.path.basename(rp)} → {want}", v == want,
              f"got {v}")
    else:
        check(f"real file {os.path.basename(rp)} → {want} "
              f"(SKIPPED: not present)", True)


# ──────────────────────────────────────────────────────────────────────
import shutil
shutil.rmtree(_TMP, ignore_errors=True)
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed")
print("─" * 50)
sys.exit(1 if failed else 0)
