#!/usr/bin/env python3
"""Tests for the comfyless reference-image EDIT wiring — ADR-035 slice 3.

CPU-only, no GPU, no model weights. Covers the comfyless.generate side of the
qwen-edit routing: the MODE→(vl,ref) flag table, the PIL→ComfyUI-tensor
conversion, and `_run_qwen_edit_refs` — that it decodes each --ref-image spec
through the real ingestion helper (comfyless/ref_image.py), maps modes to the
correct per-image flags in order ("Picture N"), forwards dims/steps/guidance to
generate_qwen_edit, and turns the decoded latents back into a PIL image.

The GPU-heavy `generate_qwen_edit` / `decode_qwen_latents` are monkeypatched on
`nodes.eric_diffusion_manual_loop` (the module `_run_qwen_edit_refs` imports at
call time) so the routing is exercised without a pipeline. End-to-end image
quality is the live smoke test (Grant), not this suite.
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import comfyless  # noqa: F401 — installs the folder_paths/comfy stubs
import comfyless.generate as cg
import nodes.eric_diffusion_manual_loop as nml

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


# ── MODE → (vl, ref) flag table (ADR-035 decision 2a) ────────────────────────
check("MODE both → (vl=T, ref=T)", cg._REF_MODE_FLAGS["both"] == (True, True))
check("MODE vl   → (vl=T, ref=F)", cg._REF_MODE_FLAGS["vl"] == (True, False))
check("MODE ref  → (vl=F, ref=T)", cg._REF_MODE_FLAGS["ref"] == (False, True))
check("no 'none' mode (an ignored image is an omitted flag)",
      "none" not in cg._REF_MODE_FLAGS)

# ── PIL → ComfyUI IMAGE tensor ───────────────────────────────────────────────
_pil = Image.new("RGB", (32, 48), (10, 20, 30))  # WxH = 32x48
_t = cg._pil_to_comfy_image(_pil)
check("pil→comfy shape is (1, H, W, 3)", tuple(_t.shape) == (1, 48, 32, 3),
      detail=str(tuple(_t.shape)))
check("pil→comfy dtype float32", _t.dtype == torch.float32)
check("pil→comfy range [0,1]", float(_t.min()) >= 0.0 and float(_t.max()) <= 1.0)
check("pil→comfy value round-trips (10/255 in R)",
      abs(float(_t[0, 0, 0, 0]) - 10.0 / 255.0) < 1e-6)

# ── _run_qwen_edit_refs routing (monkeypatched GPU functions) ────────────────
_captured = {}


def _fake_generate_qwen_edit(pipe, prompt, negative_prompt, reference_images, *,
                             vl_flags, ref_flags, output_width, output_height,
                             num_inference_steps, guidance_scale, sigma_schedule,
                             generator, max_sequence_length):
    _captured.update(
        prompt=prompt, neg=negative_prompt, refs=reference_images,
        vl_flags=list(vl_flags), ref_flags=list(ref_flags),
        output_width=output_width, output_height=output_height,
        steps=num_inference_steps, guidance=guidance_scale,
        sigma=sigma_schedule, msl=max_sequence_length,
    )
    # (packed_latents, out_height, out_width)
    return torch.zeros(1), 512, 768


def _fake_decode_qwen_latents(pipe, latents, height, width):
    _captured["decode_hw"] = (height, width)
    return torch.zeros(1, height, width, 3)  # ComfyUI [1,H,W,3] in [0,1]


nml.generate_qwen_edit = _fake_generate_qwen_edit
nml.decode_qwen_latents = _fake_decode_qwen_latents

_tmp = tempfile.mkdtemp(prefix="ref_edit_test_")
_p_both = os.path.join(_tmp, "kf.png")
Image.new("RGB", (32, 48), (10, 20, 30)).save(_p_both)   # WxH 32x48
_p_vl = os.path.join(_tmp, "car.png")
Image.new("RGB", (16, 16), (40, 50, 60)).save(_p_vl)
_specs = [{"path": _p_both, "mode": "both"}, {"path": _p_vl, "mode": "vl"}]

_pil_out, _oh, _ow, _prov = cg._run_qwen_edit_refs(
    object(), "put the car in the scene", None, _specs,
    num_steps=30, guidance_scale=4.0, sigma_schedule="linear",
    generator=None, max_sequence_length=1024,
    output_width=None, output_height=None,
)

check("routing: both specs decoded into ref tensors", len(_captured["refs"]) == 2)
check("routing: ref tensor 0 is (1,H,W,3) from the real ingestion helper",
      tuple(_captured["refs"][0].shape) == (1, 48, 32, 3),
      detail=str(tuple(_captured["refs"][0].shape)))
check("routing: order preserved (Picture 1 = first spec, 32x48)",
      tuple(_captured["refs"][0].shape) == (1, 48, 32, 3)
      and tuple(_captured["refs"][1].shape) == (1, 16, 16, 3))
check("routing: MODE both → vl_flags[0]=T, ref_flags[0]=T",
      _captured["vl_flags"][0] is True and _captured["ref_flags"][0] is True)
check("routing: MODE vl → vl_flags[1]=T, ref_flags[1]=F",
      _captured["vl_flags"][1] is True and _captured["ref_flags"][1] is False)
check("routing: dims None forwarded as None (loop derives from last ref)",
      _captured["output_width"] is None and _captured["output_height"] is None)
check("routing: steps/guidance/sigma/msl forwarded",
      _captured["steps"] == 30 and _captured["guidance"] == 4.0
      and _captured["sigma"] == "linear" and _captured["msl"] == 1024)
check("routing: decode called with (out_height, out_width)",
      _captured["decode_hw"] == (512, 768))
check("routing: returns PIL sized (out_width, out_height)",
      _pil_out.size == (768, 512), detail=str(_pil_out.size))
check("routing: returns (out_height, out_width) ints",
      (_oh, _ow) == (512, 768))
check("routing: negative_prompt None passed through", _captured["neg"] is None)

# ── Provenance recorded for a truthful sidecar (F1) ──────────────────────────
import hashlib
_sha_both = hashlib.sha256(Path(_p_both).read_bytes()).hexdigest()
check("provenance: one entry per reference", len(_prov) == 2)
check("provenance: path + mode recorded in order",
      _prov[0]["path"] == _p_both and _prov[0]["mode"] == "both"
      and _prov[1]["mode"] == "vl")
check("provenance: sha256 over the exact file bytes (not discarded)",
      _prov[0]["sha256"] == _sha_both)
check("provenance: all refs marked applied", all(e["applied"] for e in _prov))

# ── Explicit dims are forwarded (not overridden to None) ─────────────────────
_captured.clear()
cg._run_qwen_edit_refs(
    object(), "p", "ugly", [{"path": _p_both, "mode": "ref"}],
    num_steps=25, guidance_scale=3.0, sigma_schedule="karras",
    generator=None, max_sequence_length=512,
    output_width=800, output_height=600,
)
check("explicit dims: output_width/height forwarded verbatim",
      _captured["output_width"] == 800 and _captured["output_height"] == 600)
check("MODE ref → vl_flags=F, ref_flags=T",
      _captured["vl_flags"] == [False] and _captured["ref_flags"] == [True])
check("negative_prompt forwarded when set", _captured["neg"] == "ugly")

# ── Delegation predicate (F4): a --ref-image run NEVER delegates ─────────────
# The regression mode of this one-token guard is a SILENT ref drop, so it gets
# an explicit negative case (both --savepath and default-output shapes).
from types import SimpleNamespace


def _args(*, savepath=None, ref_image=()):
    return SimpleNamespace(savepath=savepath, ref_image=list(ref_image))


check("delegate: savepath + no ref → True",
      cg._should_delegate_to_server(_args(savepath="out/%seed%.png"), False) is True)
check("delegate: default-output + no ref → True",
      cg._should_delegate_to_server(_args(), True) is True)
check("delegate: explicit --output + no ref → False",
      cg._should_delegate_to_server(_args(), False) is False)
check("delegate: savepath + REF → False (no silent drop)",
      cg._should_delegate_to_server(
          _args(savepath="out/%seed%.png", ref_image=["a.png:both"]), False) is False)
check("delegate: default-output + REF → False (no silent drop)",
      cg._should_delegate_to_server(_args(ref_image=["a.png"]), True) is False)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
