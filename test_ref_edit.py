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

# ── Delegation seam (ADR-035 slice 4b, decision 7 Finding 2 revised) ─────────
# A --ref-image run delegates like ANY run — the DAEMON is the authoritative gate
# for reference containment. There is no client-side ref-root gate: a ref outside
# the daemon's ref_image_roots comes back as RefPathError and _delegate_to_server
# falls back to in-process (tested below). So the predicate only depends on
# savepath / default-output / explicit --output — identical for ref and non-ref.
from types import SimpleNamespace


def _args(*, savepath=None, ref_image=()):
    return SimpleNamespace(savepath=savepath, ref_image=list(ref_image))


check("delegate: savepath + no ref → True",
      cg._should_delegate_to_server(_args(savepath="out/%seed%.png"), False) is True)
check("delegate: default-output + no ref → True",
      cg._should_delegate_to_server(_args(), True) is True)
check("delegate: explicit --output + no ref → False",
      cg._should_delegate_to_server(_args(), False) is False)
# Ref runs delegate on the SAME rule now (no ref-root gate) — the daemon decides.
check("delegate: savepath + REF → True (daemon is the gate, 4b)",
      cg._should_delegate_to_server(
          _args(savepath="out/%seed%.png", ref_image=["kf.png:both"]), False) is True)
check("delegate: default-output + REF → True (4b)",
      cg._should_delegate_to_server(_args(ref_image=["kf.png"]), True) is True)
check("delegate: explicit --output + REF → False (server owns path)",
      cg._should_delegate_to_server(_args(ref_image=["kf.png"]), False) is False)

# ── RefPathError auto-fallback (slice 4b) ────────────────────────────────────
# When the daemon refuses a ref as outside its roots, the client must FALL BACK
# to in-process (return None), not hard-fail — and ONLY for RefPathError, never
# for a model-path PathError (which stays a hard error, rc 1).
import io as _io
import contextlib as _ctxlib


class _FakeSock:
    def exists(self):
        return True


def _deleg_args():
    return SimpleNamespace(
        device="cuda", precision="bf16", offload_vae=False,
        attention_slicing=False, sequential_offload=False, vae_tiling="auto",
        savepath="out/%seed%.png", quant=None, quant_skip=None, quant_only=None,
        output_format=None, quality=None, rebalance=False, rebalance_mult=4.0,
        rebalance_weights=None, ref_image=["/somewhere/kf.png:both"])


import comfyless.server as _srvmod
_orig_socket_path = _srvmod.socket_path
_orig_send = cg._send_server_command
try:
    _srvmod.socket_path = lambda device="cuda": _FakeSock()

    cg._send_server_command = lambda req, device="cuda": {
        "status": "error", "error_type": "RefPathError",
        "error": "ref_images[0].path outside the ref-image roots: '/somewhere/kf.png'"}
    _buf = _io.StringIO()
    with _ctxlib.redirect_stderr(_buf):
        _rc = cg._delegate_to_server(_deleg_args(), {"model": "/m", "prompt": "p"}, [])
    check("fallback: RefPathError → returns None (run in-process)", _rc is None)
    check("fallback: RefPathError prints a loud reason on stderr",
          "reference-image roots" in _buf.getvalue()
          and "in-process" in _buf.getvalue())

    # A model-path PathError is NOT recoverable — hard error, rc 1, no fallback.
    cg._send_server_command = lambda req, device="cuda": {
        "status": "error", "error_type": "PathError",
        "error": "model path outside the allowed roots: '/evil'"}
    _buf2 = _io.StringIO()
    with _ctxlib.redirect_stderr(_buf2):
        _rc2 = cg._delegate_to_server(_deleg_args(), {"model": "/m", "prompt": "p"}, [])
    check("fallback: model-path PathError → rc 1 (hard error, NOT in-process)",
          _rc2 == 1)

    # An error response with NO error_type (e.g. a synthetic ClientRecvError or a
    # legacy daemon) must fail closed to rc 1 — never a fallback on an unknown error.
    cg._send_server_command = lambda req, device="cuda": {
        "status": "error", "error": "something unexpected"}
    _buf3 = _io.StringIO()
    with _ctxlib.redirect_stderr(_buf3):
        _rc3 = cg._delegate_to_server(_deleg_args(), {"model": "/m", "prompt": "p"}, [])
    check("fallback: error with no error_type → rc 1 (fail-closed default)",
          _rc3 == 1)
finally:
    _srvmod.socket_path = _orig_socket_path
    cg._send_server_command = _orig_send

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
