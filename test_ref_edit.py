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

# ── ADR-040 D3a — the one-shot entry gate (slice 3) ──────────────────────────
# The fallback tested immediately above is FATAL against a warm daemon: it runs
# in-process while the daemon still holds its pipeline's VRAM. D3a refuses at
# entry instead, consuming the SAME containment helper the refine loop uses
# (query_daemon_roots + paths_outside_roots — no second spelling on the client).
# Scope is the load-bearing part: the check fires only when the daemon would
# ACTUALLY serve the request, so `--output` runs (delegation skipped) and
# daemonless runs are untouched.
_d3a_root = tempfile.mkdtemp(prefix="d3a-roots-")
_d3a_out = os.path.join(_d3a_root, "out")          # the daemon's output dir
_d3a_sib = os.path.join(_d3a_root, "output")       # prefix SIBLING of it
_d3a_far = os.path.join(_d3a_root, "photos")       # nowhere near a root
for _d in (_d3a_out, _d3a_sib, _d3a_far):
    os.makedirs(_d, exist_ok=True)
_d3a_inside = os.path.join(_d3a_out, "kf.png")
_d3a_outside = os.path.join(_d3a_far, "kf.png")
_d3a_prefix = os.path.join(_d3a_sib, "kf.png")
for _f in (_d3a_inside, _d3a_outside, _d3a_prefix):
    open(_f, "wb").close()

_d3a_sent = []


def _d3a_ping(req, device="cuda"):
    _d3a_sent.append(req)
    return {"status": "ok", "output_dir": _d3a_out, "ref_image_roots": [_d3a_out]}


def _d3a_args(*, ref_image=(), savepath="out/%seed%.png", device="cuda"):
    return SimpleNamespace(ref_image=list(ref_image), savepath=savepath,
                           device=device)


def _d3a_run(args, using_default_output=False):
    """Drive the REAL gate, returning (rc, stderr)."""
    _buf = _io.StringIO()
    with _ctxlib.redirect_stderr(_buf):
        rc = cg.refuse_out_of_roots_refs(args, using_default_output)
    return rc, _buf.getvalue()


try:
    _srvmod.socket_path = lambda device="cuda": _FakeSock()
    cg._send_server_command = _d3a_ping

    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_outside]))
    check("D3a: out-of-roots --ref-image is REFUSED at entry (rc 2), not "
          "left to the in-process fallback", _rc == 2, f"rc={_rc!r}")
    check("D3a: the refusal names the offending path",
          repr(_d3a_outside) in _err, _err)
    check("D3a: the refusal names --ref-root as an escape",
          "--ref-root" in _err, _err)

    # The in-process escape must DISCHARGE, not merely appear. `--output` alone
    # does not skip delegation when --savepath is set (the predicate is
    # `bool(savepath) or default_output`), so on this branch the advice has to
    # name the flag that must GO — otherwise it loops back to this refusal
    # (code review 2026-07-27).
    check("D3a: with --savepath set, the escape says to REPLACE it (adding "
          "--output would not skip delegation)",
          "replace --savepath with --output" in _err, _err)
    _rc_esc, _err_esc = _d3a_run(
        _d3a_args(ref_image=[_d3a_outside], savepath=None),
        using_default_output=False)
    check("D3a: following the advised escape actually clears the refusal",
          _rc_esc is None, f"rc={_rc_esc!r} err={_err_esc!r}")

    # The commonest invocation of all: no --savepath, no --output. It delegates
    # on the default-output sentinel, so it must refuse too — and it is the one
    # branch where plain "pass --output" is the correct advice.
    _rc_dfl, _err_dfl = _d3a_run(
        _d3a_args(ref_image=[_d3a_outside], savepath=None),
        using_default_output=True)
    check("D3a: a default-output run (no --savepath, no --output) is refused",
          _rc_dfl == 2, f"rc={_rc_dfl!r}")
    check("D3a: on the default-output branch the escape is plain 'pass --output'",
          "pass --output" in _err_dfl and "replace --savepath" not in _err_dfl,
          _err_dfl)
    check("D3a: --ref-root is offered for the reference's DIRECTORY (it cannot "
          "name a single file)", repr(_d3a_far) in _err, _err)
    check("D3a: fixes are ordered narrowest-first — copy under a reported root "
          "BEFORE the broad --ref-root grant",
          _err.index("copy") < _err.index("--ref-root"), _err)
    check("D3a: the refusal says WHY refusing beats the fallback (the daemon "
          "still holds VRAM)", "OOM" in _err and "in-process" in _err, _err)

    # The gate reads the PATH, not the raw spec: a MODE suffix must not make an
    # out-of-roots reference look like an unknown path and slip through.
    _rc, _err = _d3a_run(_d3a_args(ref_image=[f"{_d3a_outside}:vl"]))
    check("D3a: a MODE-suffixed spec is checked by its path, not its spec string",
          _rc == 2 and repr(_d3a_outside) in _err, f"rc={_rc!r} err={_err!r}")

    # Containment comes from the shared helper (realpath + boundary), so a
    # prefix sibling of the root is OUTSIDE. A startswith would have passed it.
    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_prefix]))
    check("D3a: a prefix-sibling directory is outside (shared helper semantics, "
          "not startswith)", _rc == 2, f"rc={_rc!r}")

    # Only the offending reference is named; an in-roots companion is not.
    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_inside, _d3a_outside]))
    check("D3a: a mixed set is refused and names ONLY the outside reference",
          _rc == 2 and repr(_d3a_outside) in _err and repr(_d3a_inside) not in _err,
          f"rc={_rc!r} err={_err!r}")

    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_inside]))
    check("D3a: an in-roots reference proceeds (rc None, no refusal, no notice "
          "— only the line naming what the entry ping is waiting on)",
          _rc is None and "is outside" not in _err and "NOTICE" not in _err,
          f"rc={_rc!r} err={_err!r}")

    # SCOPE — explicit --output skips delegation entirely, so the daemon was
    # never going to serve this run and the gate must not refuse it.
    _d3a_sent.clear()
    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_outside], savepath=None),
                         using_default_output=False)
    check("D3a: does NOT fire when --output already skips delegation",
          _rc is None, f"rc={_rc!r} err={_err!r}")
    check("D3a: an --output run does not even ping the daemon", _d3a_sent == [])

    # A run with no references never pays for a ping.
    _d3a_sent.clear()
    _rc, _err = _d3a_run(_d3a_args())
    check("D3a: a run with no --ref-image proceeds without pinging",
          _rc is None and _d3a_sent == [])

    # A daemon that reports nothing (pre-D2, or a malformed report) is UNKNOWN,
    # never "outside" — that case behaves exactly as before this slice and keeps
    # the RefPathError fallback as its only backstop.
    cg._send_server_command = lambda req, device="cuda": {"status": "ok"}
    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_outside]))
    check("D3a: a daemon reporting no roots does not refuse (pre-D2 parity)",
          _rc is None, f"rc={_rc!r}")
    check("D3a: ...but says so — a skipped entry check is announced, not silent",
          "NOTICE" in _err and "CANNOT be validated" in _err, _err)
    cg._send_server_command = lambda req, device="cuda": {
        "status": "ok", "output_dir": "relative/dir", "ref_image_roots": [_d3a_out]}
    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_outside]))
    check("D3a: a malformed report is ignored whole, not partially trusted",
          _rc is None, f"rc={_rc!r}")

    # A responder whose output_dir is NOT in its own ref_image_roots must not
    # be quoted as a safe destination — that misattribution is the one slice 2b
    # already paid for on the refine side.
    cg._send_server_command = lambda req, device="cuda": {
        "status": "ok", "output_dir": _d3a_far, "ref_image_roots": [_d3a_out]}
    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_prefix]))
    check("D3a: the 'copy it here' destination comes from the VALIDATED root "
          "list, never an output_dir the daemon did not list as a root",
          _rc == 2 and repr(_d3a_out) in _err
          and f"({_d3a_far!r} is one)" not in _err, f"rc={_rc!r} err={_err!r}")

    # The --ref-root suggestion must be the path the daemon's realpath'ing
    # _within will actually compare against: a lexical dirname does not contain
    # a symlinked reference, so the operator would restart a 20B daemon for
    # nothing.
    cg._send_server_command = _d3a_ping
    _d3a_link_dir = os.path.join(_d3a_root, "link-to-photos")
    os.symlink(_d3a_far, _d3a_link_dir)
    _rc, _err = _d3a_run(
        _d3a_args(ref_image=[os.path.join(_d3a_link_dir, "kf.png")]))
    check("D3a: --ref-root is suggested for the RESOLVED directory, not the "
          "symlinked spelling the daemon would still refuse",
          _rc == 2 and f"--ref-root {_d3a_far!r}" in _err, f"err={_err!r}")

    # Two references in two different out-of-roots directories: grant BOTH, or
    # the operator restarts the daemon twice to discover the second.
    _d3a_far2 = os.path.join(_d3a_root, "more-photos")
    os.makedirs(_d3a_far2, exist_ok=True)
    _d3a_outside2 = os.path.join(_d3a_far2, "kf2.png")
    open(_d3a_outside2, "wb").close()
    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_outside, _d3a_outside2]))
    check("D3a: every offending directory is granted at once, not just the first",
          f"--ref-root {_d3a_far!r}" in _err and f"--ref-root {_d3a_far2!r}" in _err,
          f"err={_err!r}")

    # A spec set that reaches the gate UNVALIDATED — `_apply_replay_ref_trust`
    # rewrites args.ref_image from an untrusted sidecar after main() validated
    # the typed ones — must fail CLOSED. Swallowing it would let sidecar
    # metadata silently disable the containment check for the whole run.
    _rc, _err = _d3a_run(_d3a_args(ref_image=[f"{_d3a_outside}:bogus-mode"]))
    check("D3a: an unparseable spec refuses (fail-closed), never silently "
          "skips the containment check",
          _rc == 2 and "bogus-mode" in _err, f"rc={_rc!r} err={_err!r}")

    # The ping D3a sends is the D2a literal — `report_roots` on a ping request
    # and nowhere else. A variable-built key would be a hard daemon-side
    # ValidationError instead of a False.
    cg._send_server_command = _d3a_ping
    _d3a_sent.clear()
    _d3a_run(_d3a_args(ref_image=[_d3a_inside]))
    check("D3a: the gate's only wire traffic is one report_roots PING",
          _d3a_sent == [{"type": "ping", "report_roots": True}], f"{_d3a_sent!r}")

    # Daemonless: no socket, no ping, no refusal — the in-process run is
    # unaffected by every line of this slice.
    _srvmod.socket_path = lambda device="cuda": SimpleNamespace(
        exists=lambda: False)
    _d3a_sent.clear()
    _rc, _err = _d3a_run(_d3a_args(ref_image=[_d3a_outside]))
    check("D3a: a daemonless run is unaffected (no refusal, no ping)",
          _rc is None and _d3a_sent == [], f"rc={_rc!r}")
finally:
    _srvmod.socket_path = _orig_socket_path
    cg._send_server_command = _orig_send
    import shutil as _d3a_sh
    _d3a_sh.rmtree(_d3a_root, ignore_errors=True)

# The checks above drive the helper directly, so every one of them stays green
# if the CALL SITE is deleted — and "refused at ENTRY" is a claim about
# placement, not about the helper. Pin the wiring by source (same technique as
# test_params_schema.py's inline-expression pin): the gate is called from
# _run_cli_mode, with the same default-output literal _run_one uses, BEFORE the
# iteration confirm and before _run_one (which owns every HF resolution, model
# load and GPU touch).
import inspect as _d3a_insp
_d3a_cli = _d3a_insp.getsource(cg._run_cli_mode)
check("D3a wiring: _run_cli_mode calls the gate",
      "refuse_out_of_roots_refs(" in _d3a_cli)
check("D3a wiring: the call site passes the SAME default-output literal "
      "_run_one uses (a divergence here silently breaks the --output scope gate)",
      'refuse_out_of_roots_refs(args, args.output == "/tmp/comfyless.png")'
      in _d3a_cli)
check("D3a wiring: the gate precedes the iteration confirm",
      _d3a_cli.index("refuse_out_of_roots_refs(")
      < _d3a_cli.index("_confirm_iteration"))
check("D3a wiring: the gate precedes _run_one (every model load lives inside it)",
      _d3a_cli.index("refuse_out_of_roots_refs(")
      < _d3a_cli.index("def _run_one"))

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
# The kind list generalized when ADR-043 joined (krea2-identity derives dims
# from its first reference the same way); the GATE is what this pins — the
# read-back must stay conditional on derived dims, never unconditional.
check("dims read-back: final_pil.size gated on ref kind + derived dims",
      'ref_kind in ("flux2-native", "krea2-identity")' in _gen_src
      and "and not ref_dims_explicit):" in _gen_src
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

# ── ADR-040 D1b: the delegated-path sidecar carries run_id ───────────────────
# Behavioral coverage for generate.py's stamp (code review: half the slice had
# none). Drives the real _delegate_to_server against a stubbed daemon and reads
# the sidecar it writes off disk.
import tempfile as _rid_tf  # noqa: E402
import json as _rid_json  # noqa: E402

_rid_dir = _rid_tf.mkdtemp(prefix="rid_deleg_")
_rid_img = os.path.join(_rid_dir, "x.png")
open(_rid_img, "wb").close()

_orig_socket_path = _srvmod.socket_path
_orig_send = cg._send_server_command
try:
    _srvmod.socket_path = lambda device="cuda": _FakeSock()
    cg._send_server_command = lambda req, device="cuda": {
        "status": "ok", "output_path": _rid_img,
        "metadata": {"seed": 7, "elapsed_seconds": 1.0}}

    _buf = _io.StringIO()
    with _ctxlib.redirect_stderr(_buf):
        cg._delegate_to_server(_deleg_args(), {"model": "/m", "prompt": "p"}, [],
                               run_id="c0ffee11")
    with open(os.path.splitext(_rid_img)[0] + ".json") as _fh:
        _rid_side = _rid_json.load(_fh)
    check("delegated path stamps run_id into the sidecar",
          _rid_side.get("run_id") == "c0ffee11", f"sidecar={_rid_side!r}")
    check("delegated path leaves iterate_batch_id unset when not sweeping "
          "(distinct fields, not aliases)",
          "iterate_batch_id" not in _rid_side, f"sidecar={_rid_side!r}")

    # The correlation promise D1b actually makes: ONE run_id shared across a
    # sweep, while iterate_batch_id is separately minted. Two delegated calls
    # from one invocation share run_id and share the batch id.
    _rid_seen = []
    for _i in range(2):
        _img_i = os.path.join(_rid_dir, f"s{_i}.png")
        open(_img_i, "wb").close()
        cg._send_server_command = (lambda p: (lambda req, device="cuda": {
            "status": "ok", "output_path": p,
            "metadata": {"seed": 7, "elapsed_seconds": 1.0}}))(_img_i)
        with _ctxlib.redirect_stderr(_io.StringIO()):
            cg._delegate_to_server(_deleg_args(), {"model": "/m", "prompt": "p"},
                                   [], run_id="5weep000",
                                   iterate_batch_id="ba7c4111")
        with open(os.path.splitext(_img_i)[0] + ".json") as _fh:
            _rid_seen.append(_rid_json.load(_fh))
    check("a sweep shares ONE run_id across its sidecars",
          {s.get("run_id") for s in _rid_seen} == {"5weep000"})
    check("a sweep's iterate_batch_id is carried separately, not aliased to run_id",
          {s.get("iterate_batch_id") for s in _rid_seen} == {"ba7c4111"})
finally:
    _srvmod.socket_path = _orig_socket_path
    cg._send_server_command = _orig_send
    import shutil as _rid_sh
    _rid_sh.rmtree(_rid_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
#  ADR-043 Part B — krea2-identity routing + params
# ═══════════════════════════════════════════════════════════════════════════
print("\n── ADR-043 Part B: krea2-identity routing ─────────────────────")

_k1 = [{"path": _p_both, "mode": "both"}]
_k2 = [{"path": _p_both, "mode": "both"}, {"path": _p_vl, "mode": "both"}]
_k3 = _k2 + [{"path": _p_both, "mode": "both"}]

# Routing: both krea families resolve to the identity-edit kind WHEN OPTED IN,
# in BOTH strictness modes — with --identity the family supports references, so
# there is no drop path.
for _fam in ("krea", "krea-turbo"):
    for _strict in (True, False):
        _kind, _warn = cg._resolve_ref_family_support(
            _k1, _fam, _strict, identity=True)
        check(f"krea2-identity: {_fam} + 1 ref + --identity (strict={_strict}) "
              f"→ ('krea2-identity', None)",
              _kind == "krea2-identity" and _warn is None,
              f"got {(_kind, _warn)!r}")
_kind, _warn = cg._resolve_ref_family_support(
    _k2, "krea-turbo", True, identity=True)
check("krea2-identity: two refs are the supported two-source path",
      _kind == "krea2-identity" and _warn is None)

# A krea run with NO refs is untouched — the routing table must not make a
# text2img krea run look like an edit (epic "what must be true" item 6).
check("krea2-identity: no refs → (None, None), krea text2img unchanged",
      cg._resolve_ref_family_support([], "krea-turbo", True) == (None, None))

# ---------------------------------------------------------------------------
# ADR-043 --identity OPT-IN. The identity edit is one reading of "reference
# image on a krea checkpoint", not the only conceivable one, so it must not
# claim the surface implicitly (Grant, 2026-07-31). Without the flag the refs
# take the ordinary drop path — the pre-routing-row behaviour exactly.
# ---------------------------------------------------------------------------
for _fam in ("krea", "krea-turbo"):
    # Lenient: dropped with a loud warning that NAMES the flag. "not supported
    # for krea" would be a lie — the user is one word away from what they want.
    _kind, _warn = cg._resolve_ref_family_support(_k1, _fam, False)
    check(f"--identity opt-in: {_fam} + refs WITHOUT the flag → dropped, "
          f"not routed",
          _kind is None and _warn is not None and "--identity" in _warn,
          f"got {(_kind, _warn)!r}")
    # Strict: a machine/scripted run fails closed rather than silently
    # generating something that ignores the references (Finding 4).
    _raised = None
    try:
        cg._resolve_ref_family_support(_k1, _fam, True)
    except ValueError as e:
        _raised = str(e)
    check(f"--identity opt-in: {_fam} STRICT without the flag → ValueError "
          f"naming the flag",
          _raised is not None and "--identity" in _raised
          and "Refusing" in _raised, f"got {_raised!r}")

# The opt-in is scoped to krea: it must not gate the families that already
# routed before ADR-043 existed (the epic's "non-krea --ref-image is
# untouched" invariant). NEGATIVE case — proves the gate did not over-reach.
for _fam, _expect in (("qwen-edit", "qwen-edit"), ("flux2", "flux2-native"),
                      ("flux2klein", "flux2-native")):
    check(f"--identity opt-in does NOT gate {_fam} (routes with identity=False)",
          cg._resolve_ref_family_support(_k1, _fam, True) == (_expect, None))
    check(f"--identity is inert for {_fam} (identity=True changes nothing)",
          cg._resolve_ref_family_support(_k1, _fam, True, identity=True)
          == (_expect, None))

# The flag is ENTRY MODE, not a generation parameter (Grant, 2026-07-31): a
# sidecar consumer does something FORWARD with the image and does not care how
# it was made. So it must have no schema key, and therefore no sidecar record
# and no --params replay. Pin all three, or it drifts into the schema later.
check("--identity has NO COMFYLESS_SCHEMA key (entry mode, not a gen param)",
      "identity" not in cg.COMFYLESS_SCHEMA)
import inspect as _inspect  # noqa: E402
check("generate() takes identity as a call parameter, like ref_drop_strict",
      "identity" in _inspect.signature(cg.generate).parameters)
check("the CLI sources identity from args, NOT from the merged params",
      "identity=bool(getattr(args, \"identity\", False))" in _gen_src)

# MODE vl/ref is a hard error in BOTH strictness modes (epic D3): Krea's two
# conditioning paths are always co-active, so selecting one is meaningless.
for _bad in ("vl", "ref"):
    for _strict in (True, False):
        _raised = None
        try:
            cg._resolve_ref_family_support(
                [{"path": _p_both, "mode": _bad}], "krea", _strict,
                identity=True)
        except ValueError as e:
            _raised = str(e)
        check(f"krea2-identity: MODE {_bad} → hard ValueError (strict={_strict})",
              _raised is not None and "krea" in _raised and _bad in _raised
              and "both" in _raised, f"got {_raised!r}")

# ...and MODE gating is scoped to the kinds that lack the dual path: qwen-edit
# still accepts vl/ref (NEGATIVE case — proves the table did not over-reach).
check("_REF_KINDS_MODE_BOTH_ONLY excludes qwen-edit (dual path is real there)",
      "qwen-edit" not in cg._REF_KINDS_MODE_BOTH_ONLY
      and set(cg._REF_KINDS_MODE_BOTH_ONLY) == {"flux2-native", "krea2-identity"})
check("qwen-edit + MODE vl still routes (not caught by the both-only gate)",
      cg._resolve_ref_family_support(
          [{"path": _p_both, "mode": "vl"}], "qwen-edit", True)
      == ("qwen-edit", None))

# A third reference is a HARD error naming the maximum, never a silent drop —
# a dropped reference reads to the user as a model failure (ADR-043 c2).
_raised = None
try:
    cg._resolve_ref_family_support(_k3, "krea-turbo", True, identity=True)
except ValueError as e:
    _raised = str(e)
check("krea2-identity: 3 refs → ValueError naming the 2-source max + slots",
      _raised is not None and "2" in _raised and "scene" in _raised
      and "identity" in _raised and "3" in _raised, f"got {_raised!r}")
_raised = None
try:
    cg._resolve_ref_family_support(_k3, "krea-turbo", False, identity=True)
except ValueError as e:
    _raised = str(e)
check("krea2-identity: 3 refs is hard even under LENIENT (not a droppable extra)",
      _raised is not None, f"got {_raised!r}")
# The cap does NOT leak onto other kinds: flux2-native takes many references.
check("3 refs on flux2-native still routes (cap is krea2-identity-scoped)",
      cg._resolve_ref_family_support(_k3, "flux2", True)
      == ("flux2-native", None))

# Drift guard: generate.py's local cap must equal the pipeline's MAX_SOURCES.
# They are deliberately duplicated so the CLI's validation path stays free of
# diffusers (the QUANT_MODES precedent) — this is what keeps them honest.
import pipelines.krea2_identity_edit as _k2mod  # noqa: E402
check("_KREA2_IDENTITY_MAX_REFS mirrors pipelines MAX_SOURCES",
      cg._KREA2_IDENTITY_MAX_REFS == _k2mod.MAX_SOURCES,
      f"{cg._KREA2_IDENTITY_MAX_REFS} vs {_k2mod.MAX_SOURCES}")

# ── _apply_krea2_identity_refs call-kwargs threading ─────────────────────────
print("\n── ADR-043 Part B: call-kwargs threading ──────────────────────")

_kck = {"prompt": "give him an undercut", "height": 1024, "width": 1024,
        "guidance_scale": 0.0}
_kprov = cg._apply_krea2_identity_refs(_kck, _k2, False, 1.25, 384)
check("krea refs: image= gets the PIL list in the TYPED order (#1 scene first)",
      isinstance(_kck.get("image"), list) and len(_kck["image"]) == 2
      and _kck["image"][0].size == (32, 48) and _kck["image"][1].size == (16, 16),
      f"got {_kck.get('image')!r}")
check("krea refs: ref_boost/grounding_px land in call_kwargs, coerced",
      _kck["ref_boost"] == 1.25 and isinstance(_kck["ref_boost"], float)
      and _kck["grounding_px"] == 384 and isinstance(_kck["grounding_px"], int))
check("krea refs: dims NOT explicit → height/width dropped (pipeline derives)",
      "height" not in _kck and "width" not in _kck)
check("krea refs: other call kwargs untouched",
      _kck["prompt"] == "give him an undercut" and _kck["guidance_scale"] == 0.0)
check("krea refs: provenance is the shared shape (path/mode/sha256/applied)",
      len(_kprov) == 2 and _kprov[0]["path"] == _p_both
      and _kprov[0]["sha256"] == _sha_both and all(e["applied"] for e in _kprov))
# Order invariant, stated as its own assertion: provenance order == typed order,
# which is what makes the sidecar's replay land the same face in the same slot.
check("krea refs: provenance order matches the typed reference order",
      [e["path"] for e in _kprov] == [_p_both, _p_vl])

_kck2 = {"prompt": "p", "height": 768, "width": 512}
cg._apply_krea2_identity_refs(_kck2, _k1, True, 4.0, 768)
check("krea refs: dims explicit → height/width forwarded verbatim",
      _kck2["height"] == 768 and _kck2["width"] == 512)

# ── Range warnings: warn, never block (D5's house rule) ──────────────────────
print("\n── ADR-043 Part B: ref_boost / grounding_px warnings ───────────")

check("ref_boost 1.25 + grounding 384 are silent (the measured sweet spot)",
      cg._krea2_identity_param_warnings(1.25, 384) == [])
check("ref_boost 1.0 (processor no-op) is silent",
      cg._krea2_identity_param_warnings(1.0, 768) == [])
_w0 = cg._krea2_identity_param_warnings(0.0, 768)
check("ref_boost 0 warns (log(0) is meaningless) and does not raise",
      len(_w0) == 1 and "log(ref_boost)" in _w0[0])
_w4 = cg._krea2_identity_param_warnings(4.0, 768)
check("ref_boost 4.0 (the card default) warns about face-adjacent suppression",
      len(_w4) == 1 and "SUPPRESSED" in _w4[0] and "1.25" in _w4[0])
_wg = cg._krea2_identity_param_warnings(1.25, 2048)
check("grounding_px outside 384-1024 warns",
      len(_wg) == 1 and "2048" in _wg[0])
_wb = cg._krea2_identity_param_warnings(8.0, 128)
check("both out of band → both warnings, still no exception", len(_wb) == 2)

# ── Accepted-but-unapplied knobs (invariant N1) ──────────────────────────────
# Krea2IdentityEditPipeline.__call__ NAMES guidance_scale / negative_prompt /
# max_sequence_length but consumes them only on its no-reference passthrough
# branch, so on the edit branch they are silently inert. That is not
# hypothetical: FAMILY_DEFAULTS["krea"] (Raw) sets cfg_scale 3.5, so a user who
# types nothing still loses CFG.
print("\n── ADR-043 Part B: accepted-but-unapplied knobs ───────────────")

import comfyless.family_defaults as _kfd  # noqa: E402
check("premise: Krea-2-Raw's family default really does set cfg_scale > 0",
      _kfd.FAMILY_DEFAULTS["krea"]["cfg_scale"] > 0,
      f"got {_kfd.FAMILY_DEFAULTS['krea']!r}")
check("premise: krea-turbo's family default is cfg 0 (the silent case)",
      _kfd.FAMILY_DEFAULTS["krea-turbo"]["cfg_scale"] == 0.0)

_ik_raw = cg._krea2_identity_ignored_knob_warnings(3.5, None, 512)
check("krea-Raw's default cfg 3.5 warns that CFG is not applied",
      len(_ik_raw) == 1 and "3.5" in _ik_raw[0]
      and "WITHOUT classifier-free guidance" in _ik_raw[0], f"got {_ik_raw!r}")
check("krea-turbo's cfg 0 + no negative + default msl is SILENT (NEGATIVE)",
      cg._krea2_identity_ignored_knob_warnings(0.0, None, 512) == [])
_ik_neg = cg._krea2_identity_ignored_knob_warnings(0.0, "blurry, low quality", 512)
check("a negative prompt warns that the edit path has no negative lane",
      len(_ik_neg) == 1 and "negative-prompt" in _ik_neg[0])
check("an EMPTY negative prompt does not warn (NEGATIVE)",
      cg._krea2_identity_ignored_knob_warnings(0.0, "", 512) == [])
check("max_sequence_length at the schema default is silent (not noise)",
      cg._krea2_identity_ignored_knob_warnings(0.0, None, 512) == [])
_ik_msl = cg._krea2_identity_ignored_knob_warnings(0.0, None, 1024)
check("a non-default max_sequence_length warns it is not applied",
      len(_ik_msl) == 1 and "1024" in _ik_msl[0])
check("all three at once → three warnings, still no exception",
      len(cg._krea2_identity_ignored_knob_warnings(3.5, "x", 1024)) == 3)
check("ignored-knob notices are wired into the identity block's edit_warnings",
      "_krea2_identity_ignored_knob_warnings(" in _gen_src
      and "cfg_scale, neg, max_sequence_length)" in _gen_src)

# ── NAG pre-gate on the krea2-identity path (epic D6) ────────────────────────
print("\n── ADR-043 Part B: NAG pre-gate (D6) ──────────────────────────")
_active, _nwarn = cg._nag_gate("krea-turbo", 5.0, 0.0, ref_kind="krea2-identity")
check("nag pre-gate: krea2-identity refs → inactive + loud warning",
      _active is False and _nwarn is not None
      and "attention processor" in _nwarn
      and "WITHOUT negative guidance" in _nwarn, f"got {_nwarn!r}")
check("nag pre-gate: no refs → NAG still activates on krea-turbo at cfg 0",
      cg._nag_gate("krea-turbo", 5.0, 0.0) == (True, None))
check("nag pre-gate: dormant nag_scale stays silent even with refs (NEGATIVE)",
      cg._nag_gate("krea", None, 0.0, ref_kind="krea2-identity") == (False, None))

# ── Delegation gate: Part B is CLI-foreground (epic D7) ──────────────────────
print("\n── ADR-043 Part B: daemon delegation gate (D7) ────────────────")


def _kargs(ref_image=(), identity=False):
    return _ap.Namespace(ref_image=list(ref_image), identity=identity)


_KRB_DEF = cg.COMFYLESS_SCHEMA["ref_boost"][1]
_KGP_DEF = cg.COMFYLESS_SCHEMA["grounding_px"][1]

check("delegation gate: no refs → None even at non-default tuning "
      "(text2img cannot consume these; it must keep delegating)",
      cg._krea2_identity_forces_in_process(
          _kargs(), {"ref_boost": 1.25, "grounding_px": 384}) is None)
check("delegation gate: refs at SCHEMA DEFAULTS → None (daemon resolves same)",
      cg._krea2_identity_forces_in_process(
          _kargs(["a.png"]),
          {"ref_boost": _KRB_DEF, "grounding_px": _KGP_DEF}) is None)
check("delegation gate: refs with the keys ABSENT → None (defaults implied)",
      cg._krea2_identity_forces_in_process(
          _kargs(["a.png"]), {"seed": 1, "steps": 8}) is None)
_kreason = cg._krea2_identity_forces_in_process(
    _kargs(["a.png"]), {"ref_boost": 1.25, "grounding_px": _KGP_DEF})
check("delegation gate: refs + non-default ref_boost → reason naming the flag",
      _kreason is not None and "--ref-boost" in _kreason
      and "--grounding-px" not in _kreason and "Part C" in _kreason,
      f"got {_kreason!r}")
_kreason2 = cg._krea2_identity_forces_in_process(
    _kargs(["a.png"]), {"ref_boost": 1.25, "grounding_px": 384})
check("delegation gate: both non-default → both flags named, in key order",
      _kreason2 is not None and _kreason2.index("--ref-boost")
      < _kreason2.index("--grounding-px"), f"got {_kreason2!r}")

# --identity forces in-process on its own, at ANY tuning. It has no schema key
# (by design — entry mode), so it cannot ride the wire at all; delegating would
# put server.py on the identity path having never seen the opt-in. Wire
# carriage is Part C, where server.py gets its Red Zone review.
_kid = cg._krea2_identity_forces_in_process(
    _kargs(["a.png"], identity=True),
    {"ref_boost": _KRB_DEF, "grounding_px": _KGP_DEF})
check("delegation gate: --identity forces in-process even at DEFAULT tuning",
      _kid is not None and "--identity" in _kid and "Part C" in _kid,
      f"got {_kid!r}")
# --identity with NO refs is a plain text2img run, so it still delegates —
# forcing a model load in-process for a no-op would trade a warm daemon's VRAM
# for nothing. But the NOTICE is still owed, and generate()'s no-op warning
# cannot deliver it here: the daemon runs generate() with identity=False and
# its stderr is not the user's terminal. So the CLI must warn CLIENT-side
# before delegating. The original version of this test asserted the delegation
# and claimed "the no-op warning covers it" — it did not, and the test
# enshrined a fully silent drop (code review 2026-07-31, findings 1 + 2).
check("delegation gate: --identity WITHOUT refs still delegates "
      "(plain text2img — no reason to force a load)",
      cg._krea2_identity_forces_in_process(
          _kargs(identity=True),
          {"ref_boost": _KRB_DEF, "grounding_px": _KGP_DEF}) is None)
check("...and the CLI warns CLIENT-side before that delegation, so the "
      "notice does not depend on whether a daemon is up",
      'if _may_delegate and getattr(args, "identity", False) \\\n'
      '                and not args.ref_image:' in _gen_src
      and "_identity_noop_message(_IDENTITY_NOOP_NO_REFS)" in _gen_src)
check("the client-side and generate()-side notices share ONE builder "
      "(they cannot drift into differently-worded versions)",
      _gen_src.count("_identity_noop_message(") >= 3)  # def + 2 call sites

# THE REGRESSION THIS GATE ALMOST SHIPPED (code review 2026-07-31, finding 1).
# generate() records ref_boost/grounding_px in EVERY sidecar, and --params puts
# every sidecar key into explicit_keys — so a presence-based gate forced every
# ref-bearing replay in-process, on qwen-edit and flux2 too. That breaks the
# epic invariant "Non-krea --ref-image behaviour is untouched" AND re-arms the
# 2026-07-26 warm-daemon crash (in-process load while the daemon holds VRAM).
# A replayed sidecar carries the keys at their DEFAULTS; it must still delegate.
_replayed_flux2_sidecar = {
    "model": "/m", "prompt": "p", "seed": 7, "steps": 8, "cfg_scale": 4.0,
    "ref_boost": _KRB_DEF, "grounding_px": _KGP_DEF,   # recorded, not chosen
}
check("delegation gate: REPLAY of a non-krea ref sidecar still delegates "
      "(presence of the keys is not intent — the regression negative)",
      cg._krea2_identity_forces_in_process(
          _kargs(["kf.png"]), _replayed_flux2_sidecar) is None)
# ...and the gate is not vacuous: the same replay WITH a tuned value holds back.
check("delegation gate: a replay carrying a TUNED ref_boost does hold back",
      cg._krea2_identity_forces_in_process(
          _kargs(["kf.png"]), {**_replayed_flux2_sidecar, "ref_boost": 1.25})
      is not None)
# The gate must read the merged params, never explicit_keys — pin the wiring so
# a future refactor cannot quietly reintroduce the presence test.
check("delegation gate is called with the merged params, not explicit_keys",
      "_krea2_identity_forces_in_process(args, p_cur)" in _gen_src
      and "_krea2_identity_forces_in_process(args, explicit_keys)"
          not in _gen_src)

# ── Source pins: the generate() seams a GPU-free test cannot execute ─────────
# Same idiom as the flux2 dims read-back above (test_nag.py's pattern).
print("\n── ADR-043 Part B: generate() source pins ─────────────────────")
check("dispatch: krea2-identity runs identity_edit_pipe_call, not pipe(...)",
      'if ref_kind == "krea2-identity":' in _gen_src
      and "result = cast(Any, identity_edit_pipe_call(" in _gen_src)
check("dispatch: the identity signature drives sigint_pause (no swallowed hook)",
      "with sigint_pause(Krea2IdentityEditPipeline.__call__," in _gen_src)
check("dims read-back covers krea2-identity (sidecar records derived dims)",
      'ref_kind in ("flux2-native", "krea2-identity")' in _gen_src)
check("no-LoRA warning is gated on the identity kind and rides edit_warnings",
      'if ref_kind == "krea2-identity" and not loras:' in _gen_src
      and "edit_warnings.append(_no_lora)" in _gen_src)
# The --identity no-op warning is the load-bearing implementation of the
# warn-never-fail decision, and until this pin existed deleting the whole block
# passed every suite (code review 2026-07-31, finding 4).
check("--identity no-op warning exists and rides edit_warnings",
      'if identity and ref_kind != "krea2-identity":' in _gen_src
      and "edit_warnings.append(_id_noop)" in _gen_src)
check("--identity no-op covers BOTH arms (no refs, and wrong family)",
      "_IDENTITY_NOOP_NO_REFS if not ref_images else" in _gen_src
      and "has no identity edit" in _gen_src)
# ref_boost/grounding_px on a non-identity run: accepted, RECORDED in the
# sidecar, never applied. Silent until this warning existed, while the
# delegation gate's own message implied in-process would apply them
# (code review 2026-07-31, finding 3).
check("non-identity runs warn that ref_boost/grounding_px are NOT applied",
      'if ref_kind != "krea2-identity":' in _gen_src
      and "edit_warnings.append(_inert_msg)" in _gen_src)
check("...and only when the user actually diverged from the schema default "
      "(the default is nobody's choice — warning on it would be noise)",
      "if _v != COMFYLESS_SCHEMA[_k][1]" in _gen_src)
check("--rebalance is pre-gated OFF on the identity path (prompt would be lost)",
      'if rebalance and ref_kind == "krea2-identity":' in _gen_src
      and "edit_warnings.append(_rb_skip)" in _gen_src)
# ...and the METADATA condition mirrors the apply-site skip. Without this the
# sidecar recorded a rebalance block for a run that deliberately did not
# rebalance — untruthful provenance (code review 2026-07-31, finding 3).
check("metadata rebalance block mirrors the identity-path skip",
      'if (rebalance and model_family in ("krea", "krea-turbo")\n'
      '            and ref_kind != "krea2-identity"):' in _gen_src)
# Non-vacuous: the apply-site and metadata conditions must BOTH carry the
# exclusion, so count the occurrences rather than trusting one substring.
check("the identity-path rebalance exclusion appears at apply AND record sites",
      _gen_src.count('ref_kind == "krea2-identity"') >= 2
      # Pin the RECORD-site condition verbatim rather than counting `!=`
      # globally: --identity's no-op warning legitimately uses the same
      # comparison, so a bare count stopped measuring what it claimed to.
      and _gen_src.count(
          'if (rebalance and model_family in ("krea", "krea-turbo")\n'
          '            and ref_kind != "krea2-identity"):') == 1)
check("the identity-edit import is LAZY (a text2img run must not pay it)",
      "from pipelines.krea2_identity_edit import (" in _gen_src
      and "from pipelines.krea2_identity_edit import" not in
          _gen_src[:_gen_src.index("def generate(")])
check("ref_boost/grounding_px are recorded in the metadata dict",
      '"ref_boost": ref_boost,' in _gen_src
      and '"grounding_px": grounding_px,' in _gen_src)
# NEGATIVE: the two scalars must NOT reach the daemon wire in Part B — the
# validator would accept them and server.py would drop them silently.
import inspect as _k2insp  # noqa: E402
_wire_src = _k2insp.getsource(cg._build_server_request)
check("NEGATIVE: _build_server_request does not send ref_boost/grounding_px",
      "ref_boost" not in _wire_src and "grounding_px" not in _wire_src)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
