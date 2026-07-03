#!/usr/bin/env python3
"""IPC robustness tests for comfyless/server.py.

Covers the two failure modes seen on 2026-04-24 with --iterate against a
running --serve daemon:

  1. Client recv timeout of 5s (server's DoS-guard value) vs realistic
     generation times of 30-120s → client always timed out waiting for the
     response, regardless of whether the work succeeded.

  2. When the client closed the socket on timeout, the server's final
     _send(conn, result) raised BrokenPipeError, which killed the daemon
     because _handle_connection did not catch it.

These tests use socket.socketpair() so they run without any real generation,
GPU, or diffusers dependency.
"""

import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyless.server as srv


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


# ──────────────────────────────────────────────────────────────────────
print("── _recv timeout parameter ────────────────────────────────────")

# Server-side default still trips at ~_RECV_TIMEOUT_SEC. Use a short override
# to keep the test fast but prove the deadline is honored.
a, b = socket.socketpair()
try:
    err = None
    t0 = time.monotonic()
    try:
        srv._recv(a, timeout=0.3)
    except ValueError as e:
        err = e
    elapsed = time.monotonic() - t0
    check("_recv raises ValueError on timeout",
          err is not None and "timed out" in str(err),
          f"err={err!r}")
    check("_recv deadline honored (0.25s <= elapsed <= 1.0s)",
          0.25 <= elapsed <= 1.0,
          f"elapsed={elapsed:.3f}s")
finally:
    a.close(); b.close()


# ──────────────────────────────────────────────────────────────────────
print("\n── _recv succeeds when sender responds within the deadline ───")

# Simulate a slow server: sleep 0.5s, then send. Client reads with a 2s
# timeout — this is the pattern that used to fail because the old hardcoded
# 5s applied to the client response path too, AND would fail here with any
# generation slower than 5s.
a, b = socket.socketpair()
try:
    def slow_responder():
        time.sleep(0.5)
        srv._send(b, {"status": "ok", "message": "slow-pong"})

    thr = threading.Thread(target=slow_responder, daemon=True)
    thr.start()

    t0 = time.monotonic()
    resp = srv._recv(a, timeout=2.0)
    elapsed = time.monotonic() - t0

    thr.join(timeout=1.0)
    check("_recv receives response after sender delay",
          resp == {"status": "ok", "message": "slow-pong"},
          f"resp={resp!r}")
    check("_recv elapsed roughly matches sender delay (0.4s <= elapsed <= 1.5s)",
          0.4 <= elapsed <= 1.5,
          f"elapsed={elapsed:.3f}s")
finally:
    a.close(); b.close()


# ──────────────────────────────────────────────────────────────────────
print("\n── default timeout is the server-side DoS guard ──────────────")

# _recv() without a timeout kwarg must fall back to the 5s server-side
# constant — this is the existing DoS protection for the request-read path
# and must be preserved.
import inspect
sig = inspect.signature(srv._recv)
default_to = sig.parameters["timeout"].default
check("_recv default timeout equals _RECV_TIMEOUT_SEC",
      default_to == srv._RECV_TIMEOUT_SEC,
      f"default={default_to}, const={srv._RECV_TIMEOUT_SEC}")

check("_CLIENT_RECV_TIMEOUT_SEC is substantially larger than server-side deadline",
      srv._CLIENT_RECV_TIMEOUT_SEC >= 60.0
      and srv._CLIENT_RECV_TIMEOUT_SEC > srv._RECV_TIMEOUT_SEC * 10,
      f"client={srv._CLIENT_RECV_TIMEOUT_SEC}, server={srv._RECV_TIMEOUT_SEC}")


# ──────────────────────────────────────────────────────────────────────
print("\n── _handle_connection survives BrokenPipeError on _send ──────")

# Wire _handle_connection with a request that succeeds schema + path checks
# and reaches the final _send(conn, result). Force BrokenPipeError by closing
# the peer socket before _handle_generate returns. The handler must swallow
# the error and return True so run_server keeps accepting.
#
# We stub _handle_generate and _check_paths at the module level so no real
# generation runs. Technique: monkeypatch for the duration of the test, then
# restore.

import tempfile

orig_handle_generate = srv._handle_generate
orig_check_paths = srv._check_paths

def fake_handle_generate(req, output_dir, model_base, device, precision, server_state):
    # Pretend generation succeeded; return a response shaped like the real one.
    return {"status": "ok", "output_path": "/tmp/fake.png", "seed": 42}

def fake_check_paths(req, model_base):
    return None  # accept any path

srv._handle_generate = fake_handle_generate
srv._check_paths = fake_check_paths

try:
    a, b = socket.socketpair()
    # Write a valid generate request to the server's side of the pair.
    req_payload = {
        "type": "generate",
        "model": "/tmp/fake-model",
        "prompt": "hello",
        "width": 1024, "height": 1024,
        "steps": 1, "cfg_scale": 3.5, "seed": 42,
    }
    srv._send(a, req_payload)

    # Close the client side BEFORE the handler runs — so _recv on the server
    # side still reads the pending bytes, but when the handler tries to send
    # the response back, the peer is gone.
    a.close()

    result = None
    exc = None
    try:
        result = srv._handle_connection(
            conn=b,
            output_dir="/tmp",
            model_base="/tmp",
            device="cuda",
            precision="bf16",
            server_state={},
        )
    except BaseException as e:
        exc = e
    b.close()

    check("_handle_connection did not propagate BrokenPipeError",
          exc is None,
          f"exc={exc!r}")
    check("_handle_connection returned True (keep server running)",
          result is True,
          f"result={result!r}")
finally:
    srv._handle_generate = orig_handle_generate
    srv._check_paths = orig_check_paths


# ══════════════════════════════════════════════════════════════════════
# Daemon Krea-2 parameter pass-through (ADR-009 caller-responsibility).
#
# The daemon is family-agnostic: CFG routing happens inside generate() →
# _build_call_kwargs, and per ADR-009 the daemon does NOT inject family
# defaults (the CLI client applies them in _run_one before delegating).
# These tests pin both facts for Krea-2 so a future refactor can't silently
# (a) add family-default injection to the daemon — which would double-apply
# with the client — or (b) mangle krea params. We mock _load_pipeline (krea
# isn't loadable until diffusers ships Krea2Pipeline) and generate (capture
# the forwarded kwargs); both are imported inside _handle_generate from
# comfyless.generate, so we patch them on that module.
import comfyless.generate as _gen

_orig_load = _gen._load_pipeline
_orig_generate = _gen.generate
_captured: dict = {}


def _fake_load(model_path, **kw):
    return object(), "krea-turbo", False  # (pipe, model_family, guidance_embeds)


def _fake_generate(**kwargs):
    _captured.clear()
    _captured.update(kwargs)
    return {"model_family": "krea-turbo"}


_gen._load_pipeline = _fake_load
_gen.generate = _fake_generate
try:
    _outdir = tempfile.mkdtemp()

    # 1. Explicit Turbo params are forwarded verbatim.
    _captured.clear()
    _state: dict = {}
    _resp = srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo",
         "prompt": "a cat", "steps": 8, "cfg_scale": 0.0},
        _outdir, _outdir, "cuda", "bf16", _state,
    )
    check("daemon: krea request succeeds (status ok)",
          _resp.get("status") == "ok", f"resp={_resp!r}")
    check("daemon: model_family cached as krea-turbo",
          _state.get("model_family") == "krea-turbo")
    check("daemon: explicit steps=8 forwarded to generate()",
          _captured.get("steps") == 8, f"got {_captured.get('steps')!r}")
    check("daemon: explicit cfg_scale=0.0 forwarded to generate()",
          _captured.get("cfg_scale") == 0.0, f"got {_captured.get('cfg_scale')!r}")

    # 2. Omitted params get the schema fallback, NOT a family default — the
    #    daemon does not run the FAMILY_DEFAULTS overlay (ADR-009). If a
    #    refactor wired the overlay into the daemon, steps would become 8/52
    #    and cfg 0.0/3.5 here, and this assertion would catch it.
    _captured.clear()
    _state = {}
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat"},
        _outdir, _outdir, "cuda", "bf16", _state,
    )
    check("daemon: omitted steps -> schema fallback 28 (no family overlay)",
          _captured.get("steps") == 28, f"got {_captured.get('steps')!r}")
    check("daemon: omitted cfg_scale -> schema fallback 3.5 (no family overlay)",
          _captured.get("cfg_scale") == 3.5, f"got {_captured.get('cfg_scale')!r}")
finally:
    _gen._load_pipeline = _orig_load
    _gen.generate = _orig_generate


# ══════════════════════════════════════════════════════════════════════
# Slice DQ — daemon quant carriage (ADR-019).
# Binding requirements from docs/security/review-slice-DQ-daemon-quant-2026-07-03.md.
print("\n── slice DQ: quant request validation (review F1) ─────────────")

_base_req = {"type": "generate", "model": "/m", "prompt": "p"}

for _q in (None, "none", "fp8"):
    _r = dict(_base_req)
    if _q is not None:
        _r["quant"] = _q
    _err = srv._validate_request(_r)
    check(f"quant={_q!r} passes validation", _err is None, f"err={_err!r}")

_err = srv._validate_request(dict(_base_req, quant="int4"))
check("bogus quant mode -> ValidationError, no raise",
      _err is not None and "quant" in _err, f"err={_err!r}")
_err = srv._validate_request(dict(_base_req, quant=5))
check("non-str quant type-rejected by canonical validator (no raise)",
      _err is not None and "quant" in _err, f"err={_err!r}")
_err = srv._validate_request(dict(_base_req, quant="fp8", quant_only=["a/b"]))
check("path-shaped quant_only entry rejected", _err is not None, f"err={_err!r}")
_err = srv._validate_request(dict(_base_req, quant="fp8", quant_skip=["a\x00b"]))
check("NUL quant_skip entry rejected", _err is not None, f"err={_err!r}")
_err = srv._validate_request(
    dict(_base_req, quant="fp8", quant_only=[f"c{i}" for i in range(33)]))
check("oversized quant_only (>32) rejected", _err is not None, f"err={_err!r}")

# F1 structural: the validation path must never trigger a torch import.
# Match import STATEMENTS only — comments referencing the module are fine.
import inspect as _inspect
import re as _re
_vsrc = _inspect.getsource(srv._validate_request)
check("_validate_request imports no torch-heavy module (review F1)",
      not _re.search(r"^\s*(from|import)\s+\S*(eric_diffusion_utils|torch)",
                     _vsrc, _re.M))


print("\n── slice DQ: cache-key discrimination (review F2/F3) ──────────")

_K = srv._request_cache_key
_r0 = {"model": "/m",
       "loras": [{"path": "/l/a.safetensors", "weight": 1.0},
                 {"path": "/l/b.safetensors", "weight": 0.8}]}

_k_none = _K(dict(_r0), "bf16", "cuda")
_k_fp8  = _K(dict(_r0, quant="fp8"), "bf16", "cuda")
check("quant fp8 vs none discriminates (N1)", _k_none != _k_fp8)
check("explicit quant='none' key equals absent-quant key",
      _K(dict(_r0, quant="none"), "bf16", "cuda") == _k_none)
check("quant_skip discriminates under fp8",
      _K(dict(_r0, quant="fp8", quant_skip=["text_encoder"]), "bf16", "cuda")
      != _k_fp8)
check("quant_only discriminates under fp8",
      _K(dict(_r0, quant="fp8", quant_only=["transformer"]), "bf16", "cuda")
      != _k_fp8)
check("quant_skip order-insensitive in key",
      _K(dict(_r0, quant="fp8", quant_skip=["a", "b"]), "bf16", "cuda")
      == _K(dict(_r0, quant="fp8", quant_skip=["b", "a"]), "bf16", "cuda"))

# N4 NEGATIVE: unquantized LoRA change must NOT change the key — the
# incremental diff path stays in charge.
_r_swap = dict(_r0, loras=[{"path": "/l/c.safetensors", "weight": 1.0}])
check("unquantized LoRA change keeps key (N4 NEGATIVE)",
      _K(_r_swap, "bf16", "cuda") == _k_none)

# N2: quantized pipelines evict on ANY LoRA change.
check("quant LoRA-set change changes key (N2)",
      _K(dict(_r_swap, quant="fp8"), "bf16", "cuda") != _k_fp8)
_r_wt = dict(_r0, loras=[{"path": "/l/a.safetensors", "weight": 1.0},
                         {"path": "/l/b.safetensors", "weight": 0.9}])
check("quant weight-only change changes key (N2 / review F3)",
      _K(dict(_r_wt, quant="fp8"), "bf16", "cuda") != _k_fp8)
_r_rev = dict(_r0, loras=list(reversed(_r0["loras"])))
check("quant LoRA reorder keeps key (no spurious evict)",
      _K(dict(_r_rev, quant="fp8"), "bf16", "cuda") == _k_fp8)
check("quant int weight normalizes to float in key (review F3)",
      _K(dict(_r0, quant="fp8",
              loras=[{"path": "/l/a.safetensors", "weight": 1}]),
         "bf16", "cuda")
      == _K(dict(_r0, quant="fp8",
                 loras=[{"path": "/l/a.safetensors", "weight": 1.0}]),
            "bf16", "cuda"))
check("cache key contains no realpath call (review F4: abspaths verbatim)",
      "/l/a.safetensors" in repr(_k_fp8))


print("\n── slice DQ: daemon quant forwarding + eviction ───────────────")

_load_captured: dict = {}


def _fake_load_q(model_path, **kw):
    _load_captured.clear()
    _load_captured.update(kw)
    return object(), "krea-turbo", False


_gen._load_pipeline = _fake_load_q
_gen.generate = _fake_generate
try:
    _outdir2 = tempfile.mkdtemp()
    _state_q: dict = {}
    _req_q = {"type": "generate", "model": "/fake/M", "prompt": "p",
              "quant": "fp8", "quant_skip": ["text_encoder"], "quant_only": []}
    _resp = srv._handle_generate(_req_q, _outdir2, _outdir2,
                                 "cuda", "bf16", _state_q)
    check("daemon: quant request succeeds", _resp.get("status") == "ok",
          f"resp={_resp!r}")
    check("daemon: quant forwarded to _load_pipeline",
          _load_captured.get("quant") == "fp8",
          f"got {_load_captured.get('quant')!r}")
    check("daemon: quant_skip forwarded as tuple",
          _load_captured.get("quant_skip") == ("text_encoder",),
          f"got {_load_captured.get('quant_skip')!r}")
    check("daemon: quant forwarded to generate()",
          _captured.get("quant") == "fp8", f"got {_captured.get('quant')!r}")

    _load_captured.clear()
    srv._handle_generate(dict(_req_q, prompt="p2"), _outdir2, _outdir2,
                         "cuda", "bf16", _state_q)
    check("daemon: identical quant request hits warm cache (no reload)",
          not _load_captured, f"reloaded with {_load_captured!r}")

    _load_captured.clear()
    srv._handle_generate(
        dict(_req_q, prompt="p3",
             loras=[{"path": "/fake/l.safetensors", "weight": 0.5}]),
        _outdir2, _outdir2, "cuda", "bf16", _state_q)
    check("daemon: LoRA change under quant evicts + reloads (N2)",
          _load_captured.get("quant") == "fp8",
          "no reload happened — incremental diff taken on quant pipeline")
finally:
    _gen._load_pipeline = _orig_load
    _gen.generate = _orig_generate


print("\n── slice DQ: client wire request carries the triple ───────────")

import argparse as _ap

_args_q = _ap.Namespace(precision="bf16", device="cuda", offload_vae=False,
                        attention_slicing=False, sequential_offload=False,
                        savepath=None, quant="fp8",
                        quant_skip=["text_encoder"], quant_only=[])
_wire = _gen._build_server_request(_args_q, {"model": "/m", "prompt": "p"}, [])
check("wire request carries quant", _wire.get("quant") == "fp8",
      f"got {_wire.get('quant')!r}")
check("wire request carries quant_skip",
      _wire.get("quant_skip") == ["text_encoder"],
      f"got {_wire.get('quant_skip')!r}")
check("wire request carries quant_only", _wire.get("quant_only") == [],
      f"got {_wire.get('quant_only')!r}")
check("wire request passes server validation end-to-end",
      srv._validate_request(dict(_wire)) is None,
      f"err={srv._validate_request(dict(_wire))!r}")

_gen_src = Path(_gen.__file__).read_text()
check("delegation-skip branch removed from generate.py",
      "daemon delegation skipped" not in _gen_src)
# Positive property (review N-2): the delegation guard exists and no longer
# consults args.quant anywhere.
check("delegation guard present and quant-free",
      "if args.savepath or using_default_output:" in _gen_src
      and 'args.quant != "none" and (args.savepath' not in _gen_src)


print("\n── H-1: _socket_dir symlink rejection ─────────────────────────")

# _socket_dir honors XDG_RUNTIME_DIR first; force the /tmp branch and plant
# a symlink at the expected name for a fake uid. srv.os IS the global os
# module, so patch/restore getuid around the single call under test.
import os as _os
_xdg_saved = _os.environ.pop("XDG_RUNTIME_DIR", None)
_orig_getuid = _os.getuid
_fakeuid = 900000 + (_os.getpid() % 10000)
_link = Path(f"/tmp/comfyless-{_fakeuid}")
_target = Path(tempfile.mkdtemp())
try:
    if _link.is_symlink() or _link.exists():
        _link.unlink()
    _link.symlink_to(_target)
    _os.getuid = lambda: _fakeuid
    _h1_err = None
    try:
        srv._socket_dir()
    except RuntimeError as e:
        _h1_err = e
    check("_socket_dir refuses symlinked socket dir (H-1)",
          _h1_err is not None and "symlink" in str(_h1_err),
          f"err={_h1_err!r}")
finally:
    _os.getuid = _orig_getuid
    if _xdg_saved is not None:
        _os.environ["XDG_RUNTIME_DIR"] = _xdg_saved
    try:
        _link.unlink()
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
