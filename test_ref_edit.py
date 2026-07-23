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

# ── Ref-execution-kind predicate (ADR-035 slice 4 / ADR-036 decision 1) ──────
# Returns (ref_kind, warn): "qwen-edit" (manual loop), "flux2-native" (stock
# pipeline image= kwarg), or None (no refs / drop path). Unsupported family:
# STRICT (machine/scripted) → hard ValueError before generation; LENIENT
# (interactive) → (None, loud warning) so it can be recorded + surfaced.
_kind, _warn = cg._resolve_ref_family_support(
    [{"path": "kf.png", "mode": "both"}], "qwen-edit", True)
check("kind: qwen-edit + refs → ('qwen-edit', no warning)",
      _kind == "qwen-edit" and _warn is None)

_kind, _warn = cg._resolve_ref_family_support([], "flux", True)
check("kind: no refs → (None, no warning), strict irrelevant",
      _kind is None and _warn is None)

_kind, _warn = cg._resolve_ref_family_support(
    [{"path": "kf.png", "mode": "both"}], "flux", False)
check("drop: flux + refs LENIENT → (None, loud warning)",
      _kind is None and _warn is not None and "not supported" in _warn
      and "without references" in _warn)

_raised = None
try:
    cg._resolve_ref_family_support(
        [{"path": "kf.png", "mode": "both"}], "flux", True)
except ValueError as e:
    _raised = str(e)
check("drop: flux + refs STRICT → ValueError naming the family + 'Refusing'",
      _raised is not None and "flux" in _raised and "Refusing" in _raised)

# ── flux2-native kind (ADR-036 decisions 1/2/3) ──────────────────────────────
# flux2klein AND flux2 resolve to the native image= path, in BOTH strictness
# modes (no drop — the family supports refs). MODE vl/ref is a hard error in
# BOTH modes (decision 3: a typed :vl is deliberate, never stumbled into).
for _fam in ("flux2klein", "flux2klein-base", "flux2"):
    for _strict in (True, False):
        _kind, _warn = cg._resolve_ref_family_support(
            [{"path": "kf.png", "mode": "both"}], _fam, _strict)
        check(f"flux2-native: {_fam} + both (strict={_strict}) → "
              f"('flux2-native', None)",
              _kind == "flux2-native" and _warn is None)

for _strict in (True, False):
    _raised = None
    try:
        cg._resolve_ref_family_support(
            [{"path": "kf.png", "mode": "vl"}], "flux2klein", _strict)
    except ValueError as e:
        _raised = str(e)
    check(f"flux2-native: MODE vl → hard ValueError even when strict={_strict}",
          _raised is not None and "flux2klein" in _raised
          and "vl" in _raised and "both" in _raised)

_raised = None
try:
    cg._resolve_ref_family_support(
        [{"path": "a.png", "mode": "both"}, {"path": "b.png", "mode": "ref"}],
        "flux2", True)
except ValueError as e:
    _raised = str(e)
check("flux2-native: mixed modes → ValueError names the offending mode(s)",
      _raised is not None and "ref" in _raised and "flux2" in _raised)

# ── _load_ref_pils shared ingestion (ADR-036 decision 7) ─────────────────────
_pils, _prov2 = cg._load_ref_pils(_specs)
check("load_ref_pils: one PIL per spec, real ingestion (RGB, right size)",
      len(_pils) == 2 and _pils[0].mode == "RGB"
      and _pils[0].size == (32, 48) and _pils[1].size == (16, 16))
check("load_ref_pils: provenance path/mode/sha256/applied in order",
      _prov2[0]["path"] == _p_both and _prov2[0]["mode"] == "both"
      and _prov2[0]["sha256"] == _sha_both
      and _prov2[1]["mode"] == "vl" and all(e["applied"] for e in _prov2))

# ── _apply_flux2_native_refs call-kwargs threading (ADR-036 d1/d5/d7) ────────
_both_specs = [{"path": _p_both, "mode": "both"}, {"path": _p_vl, "mode": "both"}]

_ck = {"prompt": "p", "height": 1024, "width": 1024, "guidance_scale": 4.0}
_prov3 = cg._apply_flux2_native_refs(_ck, _both_specs, False)
check("flux2 refs: image= gets the PIL list in order",
      isinstance(_ck.get("image"), list) and len(_ck["image"]) == 2
      and _ck["image"][0].size == (32, 48))
check("flux2 refs: dims NOT explicit → height/width dropped (pipeline derives)",
      "height" not in _ck and "width" not in _ck)
check("flux2 refs: other call kwargs untouched",
      _ck["prompt"] == "p" and _ck["guidance_scale"] == 4.0)
check("flux2 refs: provenance returned (same shape as qwen-edit)",
      len(_prov3) == 2 and _prov3[0]["sha256"] == _sha_both
      and all(e["applied"] for e in _prov3))

_ck2 = {"prompt": "p", "height": 768, "width": 512}
cg._apply_flux2_native_refs(_ck2, _both_specs, True)
check("flux2 refs: dims explicit → height/width forwarded verbatim",
      _ck2["height"] == 768 and _ck2["width"] == 512)

# ── NAG pre-gate on the flux2-native ref path (ADR-036 decision 6) ───────────
# nag_flux2's HF2-1 guard would skip at runtime on the daemon's stderr —
# invisible to a delegated client. _nag_gate deactivates NAG client-visibly.
_active, _nwarn = cg._nag_gate("flux2klein", 5.0, 0.0, ref_kind="flux2-native")
check("nag pre-gate: flux2-native refs → inactive + loud warning",
      _active is False and _nwarn is not None
      and "reference-image" in _nwarn and "WITHOUT negative guidance" in _nwarn)
_active, _nwarn = cg._nag_gate("flux2klein", 5.0, 0.0)
check("nag pre-gate: no refs → NAG still activates on flux2klein",
      _active is True and _nwarn is None)
_active, _nwarn = cg._nag_gate("flux2klein", None, 0.0, ref_kind="flux2-native")
check("nag pre-gate: dormant nag_scale stays silent even with refs",
      _active is False and _nwarn is None)

# ── Dims read-back for a truthful sidecar (ADR-036 decision 5) ───────────────
# When dims are pipeline-derived (not explicit), generate() must read the
# resolved size back off the output image BEFORE building metadata, so the
# sidecar records the truth. Source-text pin (the test_nag.py idiom) — the
# full generate() path needs a GPU; end-to-end dims are the live smoke.
_gen_src = Path(__file__).parent.joinpath("comfyless", "generate.py").read_text()
check("dims read-back: final_pil.size gated on flux2-native + derived dims",
      'if ref_kind == "flux2-native" and not ref_dims_explicit:' in _gen_src
      and "width, height = final_pil.size" in _gen_src)
check("dims read-back precedes metadata build (sidecar records the truth)",
      _gen_src.index("width, height = final_pil.size")
      < _gen_src.index('metadata: Dict[str, Any] = {'))

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

# ── Replay trust gate (ADR-035 slice 5, decision 7) ──────────────────────────
# File-derived ref paths (a --params sidecar / PNG chunk) are honored only
# through _gate_file_derived_refs: loud echo, outside-roots REFUSAL,
# missing-file / sha-mismatch warnings, and — load-bearing — no file I/O on a
# path that fails containment.
import contextlib as _ctx
import io as _io

import comfyless.ref_image as cri
from comfyless.ref_image import RefImageError, hash_ref_file, load_ref_image_capped

_troot = tempfile.mkdtemp()
_rroot = os.path.join(_troot, "outdir")
os.makedirs(_rroot)
_kf = os.path.join(_rroot, "kf.png")
Image.new("RGB", (16, 16), (1, 2, 3)).save(_kf)

# hash_ref_file: the gate's no-decode hash primitive.
_sha = hash_ref_file(_kf)
check("hash_ref_file matches load_ref_image_capped's sha256",
      _sha == load_ref_image_capped(_kf).sha256)
try:
    hash_ref_file(_kf, max_bytes=4)
    _hr = False
except RefImageError:
    _hr = True
check("hash_ref_file enforces the byte cap", _hr)
try:
    hash_ref_file(os.path.join(_troot, "nope.png"))
    _hr2 = False
except RefImageError:
    _hr2 = True
check("hash_ref_file raises on a missing file", _hr2)


def _gate(entries, roots, src="--params test.json"):
    """Run the gate capturing stderr → (specs|None, stderr, error|None)."""
    err = _io.StringIO()
    with _ctx.redirect_stderr(err):
        try:
            out = cg._gate_file_derived_refs(entries, roots, src)
        except ValueError as e:
            return None, err.getvalue(), str(e)
    return out, err.getvalue(), None


_specs, _err, _exc = _gate(
    [{"path": _kf, "mode": "vl", "sha256": _sha, "applied": True}], (_rroot,))
check("gate: in-roots entry passes with absolutized spec",
      _exc is None and _specs == [{"path": _kf, "mode": "vl"}], detail=str(_exc))
check("gate: echo names the path and mode", _kf in _err and "(vl)" in _err)
check("gate: matching sha → no mismatch flag", "MISMATCH" not in _err)

_specs2, _err2, _exc2 = _gate(
    [{"path": _kf, "mode": "both", "sha256": "0" * 64}], (_rroot,))
check("gate: sha mismatch warns LOUDLY but passes (warn-don't-block)",
      _exc2 is None and len(_specs2) == 1 and "SHA-256 MISMATCH" in _err2)

_gone = os.path.join(_rroot, "gone.png")
_specs3, _err3, _exc3 = _gate([{"path": _gone, "mode": "ref"}], (_rroot,))
check("gate: missing file warns (replay never relocates) but passes",
      _exc3 is None and len(_specs3) == 1 and "MISSING" in _err3
      and "never relocates" in _err3)

_out_png = os.path.join(_troot, "outside.png")
Image.new("RGB", (8, 8)).save(_out_png)
_specs4, _err4, _exc4 = _gate(
    [{"path": _out_png, "mode": "both", "sha256": _sha}], (_rroot,))
check("gate: outside-roots path REFUSED with the retype escape hatch",
      _specs4 is None and _exc4 is not None and "--ref-image" in _exc4)
check("gate: refusal echo flags the offender", "REFUSED" in _err4)

# NEGATIVE (the gate's core security property): a refused path is never read —
# the gate must not itself perform the attacker-directed read it refuses.
_read_trap = {"hit": False}
_orig_hash = cri.hash_ref_file


def _trap(path, *a, **k):
    _read_trap["hit"] = True
    return _orig_hash(path, *a, **k)


cri.hash_ref_file = _trap
try:
    _gate([{"path": _out_png, "mode": "both", "sha256": _sha}], (_rroot,))
finally:
    cri.hash_ref_file = _orig_hash
check("gate: NO file read on a refused path", _read_trap["hit"] is False)

# Malformed structure → hard error, never a default (fail-closed, Finding 4).
for _name, _bad in [
    ("non-list ref_images", {"path": _kf}),
    ("non-dict entry", ["x"]),
    ("entry without a path", [{"mode": "both"}]),
    ("NUL byte in path (6e)", [{"path": "/a\x00b.png", "mode": "both"}]),
    ("unknown mode", [{"path": _kf, "mode": "wild"}]),
    ("absent mode never defaults", [{"path": _kf}]),
    ("count over the 6f cap", [{"path": _kf, "mode": "both"}] * 9),
]:
    _s, _e, _x = _gate(_bad, (_rroot,))
    check(f"gate: {_name} → hard error", _s is None and _x is not None)

_specs5, _, _exc5 = _gate([{"path": _kf, "mode": "vl", "applied": False}], (_rroot,))
check("gate: recorded applied=False is provenance-only (still replayed)",
      _exc5 is None and len(_specs5) == 1)

# Echo escapes control chars via repr() (security review MEDIUM-1): a refused
# path carrying an ESC sequence must not reach the terminal raw.
_esc = "/x/\x1b]0;pwned\x07evil.png"
_, _err_esc, _ = _gate([{"path": _esc, "mode": "both"}], (_rroot,))
check("gate: echo escapes control chars (no raw ESC on stderr)",
      "\x1b" not in _err_esc and "\\x1b" in _err_esc)

# ── _replay_ref_roots (decision 7 Finding 1, hardened per CRITICAL-1) ─────────
# Roots come ONLY from operator sources: explicit --output dir, --ref-root, and
# CLI-TYPED weight args — NEVER the untrusted sidecar's own params.
_ra = _ap.Namespace(output=os.path.join(_rroot, "img.png"), ref_root=[_troot],
                    model="owner/repo", transformer=_kf, vae="/does/not/exist",
                    upscale_vae=None, te1=None, te2=None)
_roots = cg._replay_ref_roots(_ra)
check("roots: explicit --output dir included", os.path.realpath(_rroot) in _roots)
check("roots: --ref-root included", os.path.realpath(_troot) in _roots)
check("roots: existing CLI-typed weight dir included (deduped)",
      _roots.count(os.path.realpath(_rroot)) == 1)
check("roots: HF repo id (--model) contributes no cwd-relative root",
      all("owner" not in r for r in _roots))
check("roots: nonexistent CLI weight path contributes nothing",
      all(not r.startswith("/does") for r in _roots))

# CRITICAL-1 negative: the default --output sentinel does NOT make /tmp a root.
_ra_def = _ap.Namespace(output="/tmp/comfyless.png", ref_root=[], model=None,
                        transformer=None, vae=None, upscale_vae=None,
                        te1=None, te2=None)
check("roots: default /tmp output sentinel is NOT a root (LOW-1)",
      cg._replay_ref_roots(_ra_def) == ())

# ── The seam: _apply_replay_ref_trust (pop → gate → inject) ───────────────────
# CRITICAL-1: a sidecar's OWN weight path must NOT authorize a co-located ref.
# The gate reads roots from args only, so a sidecar naming a secret's directory
# as "model" cannot self-authorize a ref beside it.
_secret_dir = tempfile.mkdtemp()
_secret = os.path.join(_secret_dir, "secret.png")  # stand-in for any readable file
Image.new("RGB", (8, 8)).save(_secret)
_read_after_refuse = {"hit": False}
cri.hash_ref_file = lambda p, *a, **k: (_read_after_refuse.__setitem__("hit", True)
                                        or _orig_hash(p, *a, **k))
_p_attack = {"model": _secret_dir,  # sidecar tries to self-authorize its dir
             "ref_images": [{"path": _secret, "mode": "both"}]}
_args_attack = _ap.Namespace(ref_image=[], params="evil.json", output=None,
                             ref_root=[], model=None, transformer=None, vae=None,
                             upscale_vae=None, te1=None, te2=None)
_err_at = _io.StringIO()
with _ctx.redirect_stderr(_err_at):
    _rc_at = cg._apply_replay_ref_trust(_args_attack, _p_attack)
cri.hash_ref_file = _orig_hash
check("seam: sidecar weight path CANNOT authorize a co-located ref (CRITICAL-1)",
      _rc_at == 2)
check("seam: refused sidecar ref triggers NO file read", _read_after_refuse["hit"] is False)
check("seam: ref_images popped off p (never reaches generate/wire)",
      "ref_images" not in _p_attack)

# Happy seam: an in-roots sidecar ref is gated and re-injected as a typed spec.
_args_ok = _ap.Namespace(ref_image=[], params="run.json", output=os.path.join(_rroot, "o.png"),
                         ref_root=[], model=None, transformer=None, vae=None,
                         upscale_vae=None, te1=None, te2=None)
_p_ok = {"ref_images": [{"path": _kf, "mode": "vl", "sha256": _sha}]}
with _ctx.redirect_stderr(_io.StringIO()):
    _rc_ok = cg._apply_replay_ref_trust(_args_ok, _p_ok)
check("seam: in-roots sidecar ref proceeds (rc None)", _rc_ok is None)
check("seam: survivor re-injected through typed args.ref_image",
      _args_ok.ref_image == [f"{_kf}:vl"] and "ref_images" not in _p_ok)

# Typed --ref-image REPLACES file-derived (Finding 8) with a NOTE, gate never runs.
_gate_ran = {"hit": False}
_orig_gate = cg._gate_file_derived_refs
cg._gate_file_derived_refs = lambda *a, **k: _gate_ran.__setitem__("hit", True) or []
_args_typed = _ap.Namespace(ref_image=["typed.png:both"], params="run.json",
                            output=None, ref_root=[], model=None, transformer=None,
                            vae=None, upscale_vae=None, te1=None, te2=None)
_p_typed = {"ref_images": [{"path": _kf, "mode": "vl"}]}
_err_typed = _io.StringIO()
with _ctx.redirect_stderr(_err_typed):
    _rc_typed = cg._apply_replay_ref_trust(_args_typed, _p_typed)
cg._gate_file_derived_refs = _orig_gate
check("seam: typed --ref-image proceeds without running the gate",
      _rc_typed is None and _gate_ran["hit"] is False)
check("seam: typed-replaces-file-derived prints a NOTE (Finding 8)",
      "replaces the ref_images recorded" in _err_typed.getvalue())
check("seam: typed args.ref_image left untouched",
      _args_typed.ref_image == ["typed.png:both"] and "ref_images" not in _p_typed)

# No ref_images → clean no-op.
_p_none = {"prompt": "p"}
check("seam: no ref_images → None, p untouched",
      cg._apply_replay_ref_trust(
          _ap.Namespace(ref_image=[], params=None), _p_none) is None
      and _p_none == {"prompt": "p"})

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
