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

# ── Delegation seam (ADR-035 slice 4, decision 7 Finding 2) ──────────────────
# A --ref-image run delegates ONLY when every typed reference resolves inside a
# CLI-known ref_image_root (--ref-root ∪ --model-base). Outside all roots → the
# run stays in-process (row-1 user authority, never a silent drop). Without
# --ref-root a ref run is unchanged from slice 3 (in-process).
from types import SimpleNamespace


def _args(*, savepath=None, ref_image=(), ref_root=(), model_base=None):
    return SimpleNamespace(savepath=savepath, ref_image=list(ref_image),
                           ref_root=list(ref_root), model_base=model_base)


# Non-ref shapes unchanged.
check("delegate: savepath + no ref → True",
      cg._should_delegate_to_server(_args(savepath="out/%seed%.png"), False) is True)
check("delegate: default-output + no ref → True",
      cg._should_delegate_to_server(_args(), True) is True)
check("delegate: explicit --output + no ref → False",
      cg._should_delegate_to_server(_args(), False) is False)

# Ref shapes: roots-aware. Use a real dir so realpath containment is exercised.
with tempfile.TemporaryDirectory() as _rroot:
    _in_root = os.path.join(_rroot, "kf.png")
    Path(_in_root).write_bytes(b"stub")  # existence not required, but realistic
    _outside = os.path.join(tempfile.gettempdir(), "elsewhere_ref_xyz.png")

    check("delegate: REF, no --ref-root → False (slice-3 parity, in-process)",
          cg._should_delegate_to_server(
              _args(savepath="out/%seed%.png", ref_image=[f"{_in_root}:both"]),
              False) is False)
    check("delegate: REF inside --ref-root → True",
          cg._should_delegate_to_server(
              _args(savepath="out/%seed%.png", ref_image=[f"{_in_root}:both"],
                    ref_root=[_rroot]), False) is True)
    # --model-base is a WEIGHT root, NOT a daemon ref root (6a) — a ref under it
    # must NOT delegate (the daemon would refuse it), so it runs in-process.
    check("delegate: REF inside --model-base only → False (weight root ≠ ref root)",
          cg._should_delegate_to_server(
              _args(ref_image=[_in_root], model_base=_rroot), True) is False)
    check("delegate: REF outside all roots → False (row-1 in-process)",
          cg._should_delegate_to_server(
              _args(ref_image=[_outside], ref_root=[_rroot]), True) is False)
    check("delegate: one REF in-root + one out-of-root → False (ALL must pass)",
          cg._should_delegate_to_server(
              _args(savepath="out/%seed%.png",
                    ref_image=[f"{_in_root}:both", _outside],
                    ref_root=[_rroot]), False) is False)
    check("delegate: REF in-root but explicit --output → False (server owns path)",
          cg._should_delegate_to_server(
              _args(ref_image=[_in_root], ref_root=[_rroot]), False) is False)

# ── Drop strictness predicate (ADR-035 slice 4, decision 2 / Finding 4) ──────
# A reference handed to a family with no edit path: STRICT (machine/scripted) →
# hard ValueError before generation; LENIENT (interactive) → (False, loud
# warning) so it can be recorded + surfaced. qwen-edit + refs → (True, None).
_is_qe, _warn = cg._resolve_ref_family_support(
    [{"path": "kf.png", "mode": "both"}], "qwen-edit", True)
check("drop: qwen-edit + refs → (True, no warning)", _is_qe is True and _warn is None)

_is_qe, _warn = cg._resolve_ref_family_support([], "flux", True)
check("drop: no refs → (False, no warning), strict irrelevant",
      _is_qe is False and _warn is None)

_is_qe, _warn = cg._resolve_ref_family_support(
    [{"path": "kf.png", "mode": "both"}], "flux", False)
check("drop: flux + refs LENIENT → (False, loud warning)",
      _is_qe is False and _warn is not None and "not supported" in _warn
      and "without references" in _warn)

_raised = None
try:
    cg._resolve_ref_family_support(
        [{"path": "kf.png", "mode": "both"}], "flux", True)
except ValueError as e:
    _raised = str(e)
check("drop: flux + refs STRICT → ValueError naming the family + 'Refusing'",
      _raised is not None and "flux" in _raised and "Refusing" in _raised)

# ── Wire carriage of ref fields (ADR-035 slice 4) ────────────────────────────
# _build_server_request must send ref_images (abspath'd) + ref_dims_explicit,
# and — because this test process is NOT an interactive TTY — OMIT ref_drop_strict
# so the daemon inherits its strict (fail-closed) default (decision 2 / Finding 4).
import argparse as _ap

_wire_args = _ap.Namespace(
    precision="bf16", device="cuda", offload_vae=False,
    attention_slicing=False, sequential_offload=False, vae_tiling="auto",
    savepath="out/%seed%.png", quant=None, quant_skip=None, quant_only=None,
    output_format=None, quality=None, rebalance=False, rebalance_mult=4.0,
    ref_image=["kf.png:vl", "car.jpg"])
_wire = cg._build_server_request(
    _wire_args, {"model": "/m", "prompt": "p"}, [],
    ref_dims_explicit=True)
check("wire: ref_images present with 2 entries",
      len(_wire.get("ref_images", [])) == 2)
check("wire: ref_images paths absolutized",
      all(r["path"].startswith("/") for r in _wire["ref_images"]))
check("wire: ref_images modes preserved in order",
      [r["mode"] for r in _wire["ref_images"]] == ["vl", "both"])
check("wire: ref_dims_explicit forwarded", _wire.get("ref_dims_explicit") is True)
check("wire: ref_drop_strict OMITTED when not a TTY (daemon defaults strict)",
      "ref_drop_strict" not in _wire)

# No --ref-image → none of the ref fields appear (byte-identical to pre-slice).
_wire_noref = cg._build_server_request(
    _ap.Namespace(precision="bf16", device="cuda", offload_vae=False,
                  attention_slicing=False, sequential_offload=False,
                  vae_tiling="auto", savepath="out/x.png", quant=None,
                  quant_skip=None, quant_only=None, output_format=None,
                  quality=None, rebalance=False, rebalance_mult=4.0,
                  ref_image=[]),
    {"model": "/m", "prompt": "p"}, [])
check("wire: no --ref-image → no ref_images key",
      "ref_images" not in _wire_noref and "ref_dims_explicit" not in _wire_noref)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
