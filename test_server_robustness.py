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

    # 3. Rebalance knobs are forwarded to generate(); omitted -> False/preset.
    _captured.clear()
    _state = {}
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat",
         "rebalance": True, "rebalance_mult": 2.0, "rebalance_weights": [1.0, 2.0]},
        _outdir, _outdir, "cuda", "bf16", _state,
    )
    check("daemon: rebalance=True forwarded to generate()",
          _captured.get("rebalance") is True)
    check("daemon: rebalance_mult forwarded to generate()",
          _captured.get("rebalance_mult") == 2.0)
    check("daemon: rebalance_weights forwarded to generate()",
          _captured.get("rebalance_weights") == [1.0, 2.0])

    _captured.clear()
    _state = {}
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat"},
        _outdir, _outdir, "cuda", "bf16", _state,
    )
    check("daemon: omitted rebalance -> False",
          _captured.get("rebalance") is False)
    check("daemon: omitted rebalance_mult -> node preset default",
          _captured.get("rebalance_mult") == _gen.KREA_REBALANCE_DEFAULT_MULT)
    check("daemon: omitted rebalance_weights -> None (generate applies preset)",
          _captured.get("rebalance_weights") is None)

    # 3b. ADR-034 slice 2: output_format rides the wire and the daemon owns the
    #     extension; cross-format runs in one --output-dir must NOT collide on
    #     the per-stem .json sidecar (security review MEDIUM, 2026-07-21).
    _fmtdir = tempfile.mkdtemp()
    _captured.clear(); _state = {}
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat",
         "output_format": "jpeg", "quality": 0.9},
        _fmtdir, _fmtdir, "cuda", "bf16", _state,
    )
    _op1 = _captured.get("output_path", "")
    check("daemon: jpeg run reserves comfyless0001.jpg",
          _op1.endswith("comfyless0001.jpg"), f"got {_op1!r}")
    check("daemon: output_format forwarded to generate() as OutputFormat(jpeg)",
          getattr(_captured.get("output_format"), "name", None) == "jpeg")
    check("daemon: jpeg run atomically reserves the .json stem too",
          (Path(_fmtdir) / "comfyless0001.json").exists())
    # A png run in the SAME dir must skip stem 0001 (its .json is taken) rather
    # than clobber the jpeg run's provenance.
    _captured.clear(); _state = {}
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat"},
        _fmtdir, _fmtdir, "cuda", "bf16", _state,
    )
    _op2 = _captured.get("output_path", "")
    check("daemon: next png run skips taken stem 0001, uses comfyless0002.png",
          _op2.endswith("comfyless0002.png"), f"got {_op2!r}")

    # 4. Device is PINNED to the daemon's --device; the request payload's
    #    `device` is ignored (security review Finding 2). A daemon on cuda:1
    #    must run on cuda:1 even when the caller asks for cuda:0.
    _captured.clear()
    _state = {}
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat",
         "device": "cuda:0"},
        _outdir, _outdir, "cuda:1", "bf16", _state,
    )
    check("daemon: payload device cuda:0 IGNORED; runs on pinned cuda:1",
          _captured.get("device") == "cuda:1", f"got {_captured.get('device')!r}")

    # 5. A mismatched payload device is warned (not silently redirected).
    _orig_log = srv._log
    _warns: list = []
    srv._log = lambda m: _warns.append(m)
    try:
        _state = {}
        srv._handle_generate(
            {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat",
             "device": "cuda:0"},
            _outdir, _outdir, "cuda:1", "bf16", _state,
        )
    finally:
        srv._log = _orig_log
    check("daemon: mismatched payload device logs a warning",
          any("ignored" in m and "cuda:1" in m for m in _warns), f"warns={_warns!r}")

    # 6. 'cuda' and 'cuda:0' are the same physical device -> no warning.
    _warns = []
    srv._log = lambda m: _warns.append(m)
    try:
        _state = {}
        srv._handle_generate(
            {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat",
             "device": "cuda"},
            _outdir, _outdir, "cuda:0", "bf16", _state,
        )
    finally:
        srv._log = _orig_log
    check("daemon: payload 'cuda' vs pinned 'cuda:0' -> no mismatch warning",
          not any("ignored" in m for m in _warns), f"warns={_warns!r}")

    # 7. Varying payload device does NOT evict — the pinned device keeps the
    #    cache_key stable (isolation: one daemon, one GPU, one cached pipeline).
    _state = {}
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat",
         "device": "cuda:0"},
        _outdir, _outdir, "cuda:1", "bf16", _state,
    )
    _pipe_first = _state.get("pipeline")
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat",
         "device": "cuda:5"},
        _outdir, _outdir, "cuda:1", "bf16", _state,
    )
    check("daemon: varying payload device does NOT evict (same pinned device)",
          _pipe_first is not None and _state.get("pipeline") is _pipe_first)

    # 8. A truthy NON-STRING payload device must not crash the handler — the
    #    boundary validator passes `device` through un-type-checked, so the
    #    advisory slug compare must absorb a TypeError, not let it escape to the
    #    accept loop. (Regression: security review Slice-2 pass.)
    _captured.clear()
    _state = {}
    _crashed = None
    try:
        _resp8 = srv._handle_generate(
            {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a cat",
             "device": 123},
            _outdir, _outdir, "cuda:1", "bf16", _state,
        )
    except Exception as _e:  # must NOT happen — a raise here would kill the daemon
        _crashed = _e
        _resp8 = None
    check("daemon: non-string payload device does not crash handler",
          _crashed is None, f"raised {_crashed!r}")
    check("daemon: non-string payload device still succeeds on pinned device",
          _resp8 is not None and _resp8.get("status") == "ok"
          and _captured.get("device") == "cuda:1", f"resp={_resp8!r}")

    # ── Finding 1: atomic auto-numbered output reservation ──────────────
    # _fake_generate does not write output_path, so the only file that can
    # exist is the 0-byte reservation placeholder — exactly what these probe.
    import os as _os

    # 9. The name is atomically RESERVED, not exists()-checked: a second request
    #    cannot re-pick the same counter even before the first has written its
    #    image. Under the old exists()-then-write code both calls would pick
    #    comfyless0001.png (nothing was written between them); the reservation is
    #    what makes the second advance to 0002.
    _num_dir = tempfile.mkdtemp()
    _state = {}
    _r9a = srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a"},
        _num_dir, _num_dir, "cuda", "bf16", _state,
    )
    _state = {}
    _r9b = srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "b"},
        _num_dir, _num_dir, "cuda", "bf16", _state,
    )
    check("daemon: first auto-numbered output is comfyless0001.png",
          _r9a.get("output_path", "").endswith("comfyless0001.png"), f"{_r9a!r}")
    check("daemon: reservation holds — second is comfyless0002.png (no collision)",
          _r9b.get("output_path", "").endswith("comfyless0002.png"), f"{_r9b!r}")
    check("daemon: 0001 placeholder reserved on disk after first call",
          _os.path.exists(_os.path.join(_num_dir, "comfyless0001.png")))

    # 10. A pre-existing file is skipped (O_EXCL fails -> next counter).
    _num_dir2 = tempfile.mkdtemp()
    open(_os.path.join(_num_dir2, "comfyless0001.png"), "w").close()
    _state = {}
    _r10 = srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a"},
        _num_dir2, _num_dir2, "cuda", "bf16", _state,
    )
    check("daemon: pre-existing comfyless0001.png skipped -> 0002",
          _r10.get("output_path", "").endswith("comfyless0002.png"), f"{_r10!r}")

    # 11. On generation failure the reserved placeholder is removed (no orphan,
    #     counter not burned) — the next run reuses 0001.
    _num_dir3 = tempfile.mkdtemp()
    def _raising_generate(**kw):
        raise RuntimeError("boom")
    _gen.generate = _raising_generate
    try:
        _state = {}
        _r11 = srv._handle_generate(
            {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a"},
            _num_dir3, _num_dir3, "cuda", "bf16", _state,
        )
    finally:
        _gen.generate = _fake_generate
    check("daemon: generation failure returns InferenceError",
          _r11.get("error_type") == "InferenceError", f"{_r11!r}")
    check("daemon: failed run leaves NO orphan placeholder",
          not _os.path.exists(_os.path.join(_num_dir3, "comfyless0001.png")))
    _state = {}
    _r11b = srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a"},
        _num_dir3, _num_dir3, "cuda", "bf16", _state,
    )
    check("daemon: counter not burned by failure -> next run reuses 0001",
          _r11b.get("output_path", "").endswith("comfyless0001.png"), f"{_r11b!r}")

    # 12. A non-EEXIST OSError from the reservation os.open (here: an unwritable
    #     output_dir -> EACCES) must return a structured error, NOT escape and
    #     kill the accept loop. os.path.exists() never raised; the atomic-open
    #     path must preserve the daemon-survival promise. (code-review slice 3.)
    _ro_dir = tempfile.mkdtemp()
    _os.chmod(_ro_dir, 0o500)  # r-x, no write: O_CREAT will EACCES for the owner
    _crash12 = None
    _r12 = None
    try:
        _state = {}
        try:
            _r12 = srv._handle_generate(
                {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "a"},
                _ro_dir, _ro_dir, "cuda", "bf16", _state,
            )
        except Exception as _e:  # a raise here would kill the daemon
            _crash12 = _e
    finally:
        _os.chmod(_ro_dir, 0o700)  # restore so the tempdir can be cleaned up
    check("daemon: unwritable output_dir does not crash handler (OSError caught)",
          _crash12 is None, f"raised {_crash12!r}")
    check("daemon: unwritable output_dir -> structured error response",
          _r12 is not None and _r12.get("status") == "error", f"{_r12!r}")
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
# ADR-030: upscale VAE fields — accepted as strings, NUL byte in the path
# rejected (it's a _PATH_FIELDS member), non-str type rejected.
_err = srv._validate_request(dict(_base_req, upscale_vae_path="/m/x",
                                  upscale_vae_subfolder="diffusers/x"))
check("upscale_vae_path + subfolder accepted (valid strings)",
      _err is None, f"err={_err!r}")
_err = srv._validate_request(dict(_base_req, upscale_vae_path="/m/a\x00b"))
check("NUL byte in upscale_vae_path rejected", _err is not None, f"err={_err!r}")
_err = srv._validate_request(dict(_base_req, upscale_vae_path=5))
check("non-str upscale_vae_path type-rejected", _err is not None, f"err={_err!r}")
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

# Since 2026-07-08 the quant triple is sidecar-replayable: the builder
# sources it from the MERGED PARAMS dict, not argparse. The Namespace below
# deliberately carries CONFLICTING stale quant attrs — if the builder ever
# regresses to reading args, the checks fail loudly.
_args_q = _ap.Namespace(precision="bf16", device="cuda", offload_vae=False,
                        attention_slicing=False, sequential_offload=False,
                        vae_tiling="auto",
                        savepath=None, quant=None,
                        quant_skip=None, quant_only=None,
                        # ADR-034: builder reads args.output_format / args.quality.
                        output_format=None, quality=None,
                        # krea-testing's builder also reads the rebalance
                        # fields; inert extras on main (attributes unread).
                        rebalance=False, rebalance_mult=4.0)
_wire = _gen._build_server_request(
    _args_q,
    {"model": "/m", "prompt": "p", "quant": "fp8",
     "quant_skip": ["text_encoder"], "quant_only": []},
    [])
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

# ── ADR-034 slice 2: output format rides the wire + daemon boundary ──
_args_jpg = _ap.Namespace(precision="bf16", device="cuda", offload_vae=False,
                          attention_slicing=False, sequential_offload=False,
                          vae_tiling="auto", savepath=None, quant=None,
                          quant_skip=None, quant_only=None,
                          output_format="jpeg", quality=0.9,
                          rebalance=False, rebalance_mult=4.0)
_wire_jpg = _gen._build_server_request(_args_jpg, {"model": "/m", "prompt": "p"}, [])
check("wire carries output_format", _wire_jpg.get("output_format") == "jpeg",
      f"got {_wire_jpg.get('output_format')!r}")
check("wire carries quality (the fraction)", _wire_jpg.get("quality") == 0.9,
      f"got {_wire_jpg.get('quality')!r}")
check("png wire request omits explicit format (None passes through)",
      _wire.get("output_format") is None)
# Boundary value checks (server-specific semantic validation, ADR-034).
check("daemon accepts valid output_format",
      srv._validate_request({"type": "generate", "model": "m", "prompt": "p",
                             "output_format": "jpg", "quality": 0.5}) is None)
check("daemon rejects unknown output_format",
      srv._validate_request({"type": "generate", "model": "m", "prompt": "p",
                             "output_format": "gif"}) is not None)
check("daemon rejects out-of-range quality (>1)",
      srv._validate_request({"type": "generate", "model": "m", "prompt": "p",
                             "quality": 1.5}) is not None)
check("daemon rejects zero quality",
      srv._validate_request({"type": "generate", "model": "m", "prompt": "p",
                             "quality": 0}) is not None)
check("daemon rejects non-numeric quality",
      srv._validate_request({"type": "generate", "model": "m", "prompt": "p",
                             "quality": "hi"}) is not None)

_gen_src = Path(_gen.__file__).read_text()
check("delegation-skip branch removed from generate.py",
      "daemon delegation skipped" not in _gen_src)
# Positive property (review N-2): the delegation guard exists and no longer
# consults args.quant anywhere. ADR-034 slice 2 removed the slice-1 png-only
# clause, so jpeg delegates like any other request.
check("delegation guard present and quant-free",
      "args.savepath or using_default_output" in _gen_src
      and 'args.quant != "none" and (args.savepath' not in _gen_src)
# The slice-1 jpeg-forces-in-process clause must be GONE from the delegation
# guard (jpeg now delegates). Pinned so a regression re-adding it is caught.
check("ADR-034 slice-2: delegation guard no longer png-gates",
      'and out_fmt.name == "png"' not in _gen_src)


print("\n── NAG (ADR-023): key freedom + forwarding + wire carriage ────")

# DECIDED (Vision slice NAG): the NAG quadruple stays OUT of the pipeline
# cache key. NAG changes output content but not pipeline shape — the NAG
# attention processors are installed per-call and restored in a finally
# (pipelines/nag_krea2.py), so a cached pipeline serves any NAG config.
# A key that discriminated on nag_* would evict/reload on every NAG tweak.
check("output_format/quality do NOT change the cache key (output concern)",
      _K(dict(_r0, output_format="jpeg", quality=0.9), "bf16", "cuda") == _k_none)
check("output_format does NOT change the cache key under quant either",
      _K(dict(_r0, quant="fp8", output_format="jpeg", quality=0.5), "bf16", "cuda") == _k_fp8)
check("nag params do NOT change the cache key (per-request safe)",
      _K(dict(_r0, nag_scale=5.0, nag_tau=3.0, nag_alpha=0.5, nag_end=0.75),
         "bf16", "cuda") == _k_none)
check("nag params do NOT change the cache key under quant either",
      _K(dict(_r0, quant="fp8", nag_scale=5.0), "bf16", "cuda") == _k_fp8)

# ADR-035 decision 3 pin: reference-image presence NEVER changes the pipeline
# cache key. Pipeline class is selected once at load (detect_pipeline_class); a
# reference must not swap class or trigger a from_pipe upgrade on the cached
# pipeline. Same model → same class → same key, ref or no ref — so the daemon
# serves the same cached qwen-edit pipeline across ref and non-ref requests.
check("ref_images do NOT change the cache key (decision 3 pin)",
      _K(dict(_r0, ref_images=[{"path": "/out/kf.png", "mode": "both"}]),
         "bf16", "cuda") == _k_none)
check("ref_images/ref_dims_explicit/ref_drop_strict do NOT change the key under quant",
      _K(dict(_r0, quant="fp8",
              ref_images=[{"path": "/out/kf.png", "mode": "vl"}],
              ref_dims_explicit=True, ref_drop_strict=False),
         "bf16", "cuda") == _k_fp8)

# Daemon forwards the quadruple to generate(); omitted -> dormant defaults.
_load_captured_n: dict = {}


def _fake_load_n(model_path, **kw):
    _load_captured_n.clear()
    _load_captured_n.update(kw)
    return object(), "krea-turbo", False


_gen._load_pipeline = _fake_load_n
_gen.generate = _fake_generate
try:
    _outdir_n = tempfile.mkdtemp()
    _state_n: dict = {}
    _resp_n = srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "p",
         "nag_scale": 4.0, "nag_tau": 3.0, "nag_alpha": 0.5, "nag_end": 0.75},
        _outdir_n, _outdir_n, "cuda", "bf16", _state_n)
    check("daemon: NAG request succeeds", _resp_n.get("status") == "ok",
          f"resp={_resp_n!r}")
    check("daemon: nag_scale forwarded to generate()",
          _captured.get("nag_scale") == 4.0,
          f"got {_captured.get('nag_scale')!r}")
    check("daemon: nag_tau/alpha/end forwarded to generate()",
          (_captured.get("nag_tau"), _captured.get("nag_alpha"),
           _captured.get("nag_end")) == (3.0, 0.5, 0.75))

    # Same state, different NAG config: warm cache MUST be reused (this is
    # the behavioral half of the key-freedom decision above).
    _load_captured_n.clear()
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "p2",
         "nag_scale": 2.0},
        _outdir_n, _outdir_n, "cuda", "bf16", _state_n)
    check("daemon: NAG config change hits warm cache (no reload)",
          not _load_captured_n, f"reloaded with {_load_captured_n!r}")

    # Omitted quadruple -> the dormant schema defaults reach generate().
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "p3"},
        _outdir_n, _outdir_n, "cuda", "bf16", _state_n)
    check("daemon: omitted nag quadruple -> dormant defaults",
          (_captured.get("nag_scale"), _captured.get("nag_tau"),
           _captured.get("nag_alpha"), _captured.get("nag_end"))
          == (0.0, 2.5, 0.25, 1.0),
          f"got {[_captured.get(k) for k in ('nag_scale', 'nag_tau', 'nag_alpha', 'nag_end')]!r}")
finally:
    _gen._load_pipeline = _orig_load
    _gen.generate = _orig_generate

# Wire request sources the quadruple from the MERGED PARAMS dict
# (sidecar-replayable, quant precedent) and passes the boundary validator.
_wire_n = _gen._build_server_request(
    _args_q,
    {"model": "/m", "prompt": "p", "nag_scale": 4.0, "nag_tau": 2.5,
     "nag_alpha": 0.25, "nag_end": 1.0},
    [])
check("wire request carries nag_scale", _wire_n.get("nag_scale") == 4.0,
      f"got {_wire_n.get('nag_scale')!r}")
check("wire request nag params pass server validation end-to-end",
      srv._validate_request(dict(_wire_n)) is None,
      f"err={srv._validate_request(dict(_wire_n))!r}")


print("\n── ADR-030: upscale-VAE independent cache lifecycle ───────────")

# The headline daemon design: the upscale VAE is cached INDEPENDENTLY of the
# 20B pipeline (own key, not in _request_cache_key), so switching it never
# evicts the pipeline. Stub _load_pipeline + generate + _load_upscale_vae and
# drive _handle_generate through load / warm-reuse / switch / drop.
_uv_calls = {"n": 0}
_orig_load_uv = _gen._load_upscale_vae


def _fake_load_uv(path, subfolder, precision, allow_download=False):
    _uv_calls["n"] += 1
    return ("UPSCALE_VAE", path)  # sentinel object


_gen._load_pipeline = _fake_load
_gen.generate = _fake_generate
_gen._load_upscale_vae = _fake_load_uv
try:
    _outdir_u = tempfile.mkdtemp()
    _state_u: dict = {}
    _req_u = {"type": "generate", "model": "/fake/Krea-2-Raw", "prompt": "p",
              "upscale_vae_path": "/fake/UpscaleA", "upscale_vae_subfolder": ""}
    _resp_u = srv._handle_generate(
        _req_u, _outdir_u, _outdir_u, "cuda", "bf16", _state_u)
    check("daemon: upscale request succeeds",
          _resp_u.get("status") == "ok", f"resp={_resp_u!r}")
    check("daemon: upscale VAE loaded once", _uv_calls["n"] == 1,
          f"n={_uv_calls['n']}")
    check("daemon: upscale VAE cached in server_state",
          _state_u.get("upscale_vae") == ("UPSCALE_VAE", "/fake/UpscaleA"))
    check("daemon: upscale VAE forwarded via cached dict to generate()",
          _captured.get("_cached_pipeline", {}).get("upscale_vae")
          == ("UPSCALE_VAE", "/fake/UpscaleA"))
    check("daemon: upscale_vae_path forwarded to generate()",
          _captured.get("upscale_vae_path") == "/fake/UpscaleA")
    _pipe_obj = _state_u.get("pipeline")

    # Same upscale VAE → warm (no VAE reload).
    srv._handle_generate(dict(_req_u, prompt="p2"),
                         _outdir_u, _outdir_u, "cuda", "bf16", _state_u)
    check("daemon: same upscale VAE reused (no reload)", _uv_calls["n"] == 1,
          f"n={_uv_calls['n']}")

    # Different upscale VAE → reload the VAE, but the 20B pipeline stays put.
    srv._handle_generate(
        dict(_req_u, prompt="p3", upscale_vae_path="/fake/UpscaleB"),
        _outdir_u, _outdir_u, "cuda", "bf16", _state_u)
    check("daemon: switching upscale VAE reloads only the VAE",
          _uv_calls["n"] == 2, f"n={_uv_calls['n']}")
    check("daemon: switching upscale VAE does NOT evict the pipeline",
          _state_u.get("pipeline") is _pipe_obj)
    check("daemon: new upscale VAE cached",
          _state_u.get("upscale_vae") == ("UPSCALE_VAE", "/fake/UpscaleB"))

    # Drop upscale VAE → popped; cached dict carries None; pipeline survives.
    srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Raw", "prompt": "p4"},
        _outdir_u, _outdir_u, "cuda", "bf16", _state_u)
    check("daemon: dropping upscale VAE pops it from server_state",
          "upscale_vae" not in _state_u and "upscale_vae_key" not in _state_u)
    check("daemon: cached dict upscale_vae None when unused",
          _captured.get("_cached_pipeline", {}).get("upscale_vae") is None)
    check("daemon: pipeline still cached after dropping upscale VAE",
          _state_u.get("pipeline") is _pipe_obj)
finally:
    _gen._load_pipeline = _orig_load
    _gen.generate = _orig_generate
    _gen._load_upscale_vae = _orig_load_uv

# Wire request carries the upscale fields and passes the boundary validator.
_wire_u = _gen._build_server_request(
    _args_q,
    {"model": "/m", "prompt": "p", "upscale_vae_path": "/m/UpscaleA",
     "upscale_vae_subfolder": "diffusers/x"},
    [])
check("wire request carries upscale_vae_path",
      _wire_u.get("upscale_vae_path") == "/m/UpscaleA",
      f"got {_wire_u.get('upscale_vae_path')!r}")
check("wire request upscale fields pass server validation end-to-end",
      srv._validate_request(dict(_wire_u)) is None,
      f"err={srv._validate_request(dict(_wire_u))!r}")


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
print("\n── device-keyed socket routing (ADR-020) ──────────────────────")

# _device_socket_slug is the pure normalization+whitelist core; socket_path
# wraps it with the 0700 socket dir. See docs/security/
# review-parallel-daemon-2026-07-03.md Finding 3.

def _slug(d):
    return srv._device_socket_slug(d)

def _rejects(d):
    try:
        srv._device_socket_slug(d)
        return False
    except ValueError:
        return True

def _rejects_path(d):
    try:
        srv.socket_path(d)
        return False
    except ValueError:
        return True

# ── canonicalization (invariant 3 + Finding 3.3 integer folding) ──
check("cuda -> cuda0 slug", _slug("cuda") == "cuda0", f"got {_slug('cuda')!r}")
check("cuda:0 -> cuda0 slug", _slug("cuda:0") == "cuda0")
check("cuda == cuda:0 (same physical device -> same slug)",
      _slug("cuda") == _slug("cuda:0"))
check("cuda:00 folds to cuda0 (leading-zero canon, Finding 3.3)",
      _slug("cuda:00") == "cuda0", f"got {_slug('cuda:00')!r}")
check("cuda:007 folds to cuda7 (leading-zero canon)",
      _slug("cuda:007") == "cuda7", f"got {_slug('cuda:007')!r}")
check("cpu -> cpu slug", _slug("cpu") == "cpu")

# ── distinctness (invariant 2) ──
check("cuda:0 and cuda:1 are DISTINCT slugs",
      _slug("cuda:0") != _slug("cuda:1"))
check("cuda:1 -> cuda1 slug", _slug("cuda:1") == "cuda1")
check("cpu and cuda0 are distinct", _slug("cpu") != _slug("cuda:0"))

# ── whitelist rejection (invariant 4 + Finding 3.1/3.2) ──
check("rejects trailing newline 'cuda:0\\n' (fullmatch, not $; Finding 3.2)",
      _rejects("cuda:0\n"))
check("rejects path traversal '../../etc/x'", _rejects("../../etc/x"))
check("rejects embedded slash 'cuda:0/../y'", _rejects("cuda:0/../y"))
check("rejects NUL byte 'cuda:0\\x00'", _rejects("cuda:0\x00"))
check("rejects non-numeric index 'cuda:abc'", _rejects("cuda:abc"))
check("rejects bare 'cuda:' (no index)", _rejects("cuda:"))
check("rejects empty string", _rejects(""))
check("rejects 'gpu0'", _rejects("gpu0"))
check("rejects leading space ' cuda:0'", _rejects(" cuda:0"))
check("rejects 'mps' (not whitelisted)", _rejects("mps"))
check("rejects unicode digit 'cuda:\\u0660' (re.ASCII gate, not int() fold)",
      _rejects("cuda:٠"))  # ARABIC-INDIC ZERO

# ── full socket_path shape ──
_p0 = srv.socket_path("cuda:0")
_p1 = srv.socket_path("cuda:1")
_pc = srv.socket_path("cpu")
check("socket_path cuda:0 basename is comfyless-cuda0.sock",
      _p0.name == "comfyless-cuda0.sock", f"got {_p0.name!r}")
check("socket_path cuda:1 basename is comfyless-cuda1.sock",
      _p1.name == "comfyless-cuda1.sock")
check("socket_path cpu basename is comfyless-cpu.sock",
      _pc.name == "comfyless-cpu.sock")
check("socket_path cuda == cuda:0 (same path)",
      srv.socket_path("cuda") == _p0)
check("socket_path cuda:0 != cuda:1 (distinct paths)", _p0 != _p1)
check("all device sockets share one 0700 dir",
      _p0.parent == _p1.parent == _pc.parent)
check("socket_path default arg is 'cuda' (-> cuda0)",
      srv.socket_path().name == "comfyless-cuda0.sock")
check("socket_path propagates whitelist rejection", _rejects_path("../../x"))


# ──────────────────────────────────────────────────────────────────────
print("\n== ADR-018: _check_paths / run_server multi-root union ==")
# ──────────────────────────────────────────────────────────────────────
# Uses the REAL srv._check_paths (the module-level stub above was restored
# after the connection tests).

import os as _os
import tempfile as _tempfile


def _mk18(path, content=b"w"):
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


with _tempfile.TemporaryDirectory() as _mb, \
     _tempfile.TemporaryDirectory() as _lr, \
     _tempfile.TemporaryDirectory() as _tr, \
     _tempfile.TemporaryDirectory() as _outside:
    _model = _os.path.join(_mb, "QwenImage")
    _os.makedirs(_model)
    _lora = _mk18(_os.path.join(_lr, "Flux", "char", "foo.safetensors"))
    _tf = _mk18(_os.path.join(_tr, "SDXL", "jug.safetensors"))
    _esc = _mk18(_os.path.join(_outside, "escape.safetensors"))

    # str-root back-compat (single allowlist root)
    check("ADR-018: str root back-compat — in-root model passes",
          srv._check_paths({"model": _model}, _mb) is None)
    _err = srv._check_paths({"model": _esc}, _mb)
    check("ADR-018: str root back-compat — outside root rejected",
          _err is not None and "outside the allowed roots" in _err)

    # union acceptance: each field under its own root
    _roots = (_mb, _lr, _tr)
    check("ADR-018: union accepts model under mb + lora under lora root",
          srv._check_paths(
              {"model": _model,
               "loras": [{"path": _lora, "weight": 1.0}]},
              _roots) is None)
    check("ADR-018: union accepts transformer_path under transformer root",
          srv._check_paths(
              {"model": _model, "transformer_path": _tf}, _roots) is None)

    # rejection: lora root NOT in the allowlist
    _err = srv._check_paths(
        {"model": _model, "loras": [{"path": _lora, "weight": 1.0}]},
        (_mb, _tr))
    check("ADR-018: lora path rejected when its root absent from union",
          _err is not None and "loras[0]" in _err)

    # rejection: path outside ALL roots
    _err = srv._check_paths({"model": _model, "vae_path": _esc}, _roots)
    check("ADR-018: path outside all roots rejected under union",
          _err is not None and "outside the allowed roots" in _err)

    # relative paths still rejected regardless of union
    check("ADR-018: relative model path still rejected",
          srv._check_paths({"model": "rel/path"}, _roots) is not None)

    # ADR-030: upscale_vae_path is a model path — same root-containment guard.
    _uv = _mk18(_os.path.join(_tr, "Wan2.1-VAE-upscale2x", "config.json"))
    check("ADR-030: upscale_vae_path under an allowed root accepted",
          srv._check_paths(
              {"model": _model, "upscale_vae_path": _uv}, _roots) is None)
    _err = srv._check_paths(
        {"model": _model, "upscale_vae_path": _esc}, _roots)
    check("ADR-030: upscale_vae_path outside all roots rejected",
          _err is not None and "outside the allowed roots" in _err)

    # run_server root validation fails closed BEFORE binding a socket
    with _tempfile.TemporaryDirectory() as _out18:
        _raised = None
        try:
            srv.run_server(_out18, _mb,
                           lora_paths=("/nonexistent-adr018-run-xyzzy",))
        except FileNotFoundError as e:
            _raised = str(e)
        check("ADR-018: run_server missing --lora-path → FileNotFoundError",
              _raised is not None and "--lora-path" in _raised)
        _raised = None
        try:
            srv.run_server(_out18, _mb,
                           transformer_paths=(_esc,))  # file, not dir
        except FileNotFoundError as e:
            _raised = str(e)
        check("ADR-018: run_server non-dir --transformer-path → "
              "FileNotFoundError",
              _raised is not None and "--transformer-path" in _raised)


# ──────────────────────────────────────────────────────────────────────
print("\n== ADR-035 slice 4: ref_images wire hardening ==")
# ──────────────────────────────────────────────────────────────────────
# Wire-boundary validation (shape delegated to the canonical validator via
# _validate_request), NUL defense (6e), and containment against the DISJOINT
# ref_image_roots set (6a). Uses the REAL srv functions.

_rr_base = {"type": "generate", "model": "/m", "prompt": "p"}

# NUL defense (6e) — mirrors the loras[i].path NUL test. A NUL must be caught
# here so it never reaches os.path.realpath in _check_ref_paths (accept-loop
# kill). The entry is otherwise well-formed so it clears canonical shape first.
_err = srv._validate_request(
    dict(_rr_base, ref_images=[{"path": "/out/a\x00b.png", "mode": "both"}]))
check("ref_images NUL path rejected by _validate_request (6e)",
      _err is not None and "ref_images[0].path" in _err, f"err={_err!r}")

# Shape/mode/count rejections flow through the canonical validator (surfaced as
# a ValidationError string by _validate_request, no raise).
_err = srv._validate_request(
    dict(_rr_base, ref_images=[{"path": "/out/a.png", "mode": "evil"}]))
check("ref_images bad mode rejected at wire (no KeyError)",
      _err is not None and "mode" in _err, f"err={_err!r}")
_err = srv._validate_request(
    dict(_rr_base, ref_images=[{"path": "/out/a.png", "mode": "both"} for _ in range(9)]))
check("ref_images >8 rejected at wire (count cap 6f)",
      _err is not None and "ref_images" in _err, f"err={_err!r}")
_err = srv._validate_request(
    dict(_rr_base, ref_images=[{"path": "/out/a.png", "mode": "both"}]))
check("well-formed ref_images entry passes _validate_request",
      _err is None, f"err={_err!r}")

# Containment (6a) — ref_image_roots is DISJOINT from the weight roots. A ref
# inside an output/ref root passes; a ref only inside a WEIGHT root is refused.
with _tempfile.TemporaryDirectory() as _outdir, \
     _tempfile.TemporaryDirectory() as _refextra, \
     _tempfile.TemporaryDirectory() as _weightonly:
    _kf = _mk18(_os.path.join(_outdir, "kf.png"))
    _kf2 = _mk18(_os.path.join(_refextra, "sub", "kf2.png"))
    _wf = _mk18(_os.path.join(_weightonly, "in_weights.png"))
    _ref_roots = (_outdir, _refextra)

    check("ref containment: no ref_images → None",
          srv._check_ref_paths({"model": "/m"}, _ref_roots) is None)
    check("ref containment: ref under output root accepted",
          srv._check_ref_paths(
              {"ref_images": [{"path": _kf, "mode": "both"}]}, _ref_roots) is None)
    check("ref containment: ref under a --ref-root accepted",
          srv._check_ref_paths(
              {"ref_images": [{"path": _kf2, "mode": "vl"}]}, _ref_roots) is None)
    _err = srv._check_ref_paths(
        {"ref_images": [{"path": _wf, "mode": "both"}]}, _ref_roots)
    check("ref containment: ref only in a WEIGHT root refused (roots disjoint)",
          _err is not None and "outside the ref-image roots" in _err, f"err={_err!r}")
    _err = srv._check_ref_paths(
        {"ref_images": [{"path": "rel/kf.png", "mode": "both"}]}, _ref_roots)
    check("ref containment: relative ref path rejected",
          _err is not None and "must be absolute" in _err, f"err={_err!r}")
    # Empty ref roots = fail-closed (no tree readable).
    _err = srv._check_ref_paths(
        {"ref_images": [{"path": _kf, "mode": "both"}]}, ())
    check("ref containment: empty ref_roots refuses everything (fail-closed)",
          _err is not None, f"err={_err!r}")

    # _resolve_ref_roots: output_dir is always a ref root; a valid --ref-root is
    # appended; a non-dir --ref-root fails closed; NUL rejected; a broad root
    # ('/', $HOME, a mount root) emits a loud breadth warning and PROCEEDS.
    _roots = srv._resolve_ref_roots(_outdir, (_refextra,))
    check("ref roots: output_dir is always included",
          _outdir in _roots)
    check("ref roots: a valid --ref-root is appended (realpath'd)",
          _os.path.realpath(_refextra) in _roots)
    _raised = None
    try:
        srv._resolve_ref_roots(_outdir, ("/nonexistent-adr035-ref-xyzzy",))
    except FileNotFoundError as e:
        _raised = str(e)
    check("ref roots: missing --ref-root → FileNotFoundError (fail-closed)",
          _raised is not None and "--ref-root" in _raised, f"raised={_raised!r}")
    _raised = None
    try:
        srv._resolve_ref_roots(_outdir, ("a\x00b",))
    except FileNotFoundError as e:
        _raised = str(e)
    check("ref roots: NUL in --ref-root → FileNotFoundError",
          _raised is not None and "NUL" in _raised, f"raised={_raised!r}")

    # Breadth warning: capture _log output and assert it fires for '/' (a real,
    # broad, existing dir) and does NOT for a narrow keyframe dir.
    import io as _io
    _orig_log = srv._log
    _cap = _io.StringIO()
    srv._log = lambda m: _cap.write(m + "\n")
    try:
        srv._resolve_ref_roots(_outdir, ("/",))
        _warned_broad = "extremely broad" in _cap.getvalue()
        _cap2 = _io.StringIO()
        srv._log = lambda m: _cap2.write(m + "\n")
        srv._resolve_ref_roots(_outdir, (_refextra,))
        _warned_narrow = "extremely broad" in _cap2.getvalue()
    finally:
        srv._log = _orig_log
    check("ref roots: '/' emits the breadth warning and proceeds", _warned_broad)
    check("ref roots: a narrow keyframe dir emits NO breadth warning",
          not _warned_narrow)

# ──────────────────────────────────────────────────────────────────────
# Client-side recv-failure contract (2026-07-17): a daemon that ACCEPTS a
# request but never answers must surface a synthetic ClientRecvError dict —
# never None. None means "no daemon; fall through to in-process generation",
# and doing that against a live daemon starts a second model load on a GPU
# whose VRAM the daemon still holds. (Before the fix, the recv deadline's
# ValueError escaped _send_server_command entirely and crashed the client.)
print("── client recv-failure contract (_send_server_command) ────────")

import os
import tempfile
import comfyless.generate as gen

_tmpdir = tempfile.mkdtemp(prefix="comfyless-recv-test-")
_old_xdg = os.environ.get("XDG_RUNTIME_DIR")
os.environ["XDG_RUNTIME_DIR"] = _tmpdir
_old_client_timeout = srv._CLIENT_RECV_TIMEOUT_SEC
srv._CLIENT_RECV_TIMEOUT_SEC = 0.3   # picked up at call time via the local import
try:
    _sockp = srv.socket_path("cpu")

    # (a) wedged daemon: accepts + reads the request, never responds.
    _lsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _lsock.bind(str(_sockp))
    _lsock.listen(1)
    _held = []

    def _wedged_daemon():
        c, _ = _lsock.accept()
        c.recv(65536)        # consume the request…
        _held.append(c)      # …and hold the socket open past the deadline

    _t = threading.Thread(target=_wedged_daemon, daemon=True)
    _t.start()
    _resp = gen._send_server_command({"type": "generate"}, "cpu")
    check("wedged daemon → synthetic error dict, not None",
          isinstance(_resp, dict) and _resp.get("status") == "error",
          f"got {_resp!r}")
    check("wedged daemon → error_type ClientRecvError",
          isinstance(_resp, dict)
          and _resp.get("error_type") == "ClientRecvError")
    check("wedged daemon → message names the deadline and no-fallback",
          isinstance(_resp, dict)
          and "no valid response" in _resp.get("error", "")
          and "not falling back" in _resp.get("error", ""))
    for _c in _held:
        _c.close()
    _lsock.close()
    _sockp.unlink()

    # (a2) broken daemon: valid JSON that is not an object. Callers do
    # resp.get(...), so this must surface as ClientRecvError, not crash
    # (security review SHOULD 1, review-pause-daemon-guard-2026-07-17).
    _lsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _lsock.bind(str(_sockp))
    _lsock.listen(1)

    def _scalar_daemon():
        c, _ = _lsock.accept()
        c.recv(65536)
        c.sendall(b'"hello"\n')
        c.close()

    _t = threading.Thread(target=_scalar_daemon, daemon=True)
    _t.start()
    _resp = gen._send_server_command({"type": "generate"}, "cpu")
    check("non-dict JSON response → ClientRecvError dict, not crash/None",
          isinstance(_resp, dict)
          and _resp.get("error_type") == "ClientRecvError"
          and "non-object response" in _resp.get("error", ""),
          f"got {_resp!r}")
    _lsock.close()
    _sockp.unlink()

    # (b) daemon dies before responding (clean EOF) → None: the process is
    # gone, its VRAM is freed, in-process fall-through is legitimate.
    _lsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _lsock.bind(str(_sockp))
    _lsock.listen(1)

    def _dying_daemon():
        c, _ = _lsock.accept()
        c.recv(65536)
        c.close()            # EOF without a response

    _t = threading.Thread(target=_dying_daemon, daemon=True)
    _t.start()
    _resp = gen._send_server_command({"type": "generate"}, "cpu")
    check("daemon EOF before response → None (fall-through allowed, NEGATIVE)",
          _resp is None, f"got {_resp!r}")
    _lsock.close()
    _sockp.unlink()

    # (c) no socket at all → None (the original absent-daemon path).
    _resp = gen._send_server_command({"type": "generate"}, "cpu")
    check("absent socket → None (fall-through allowed, NEGATIVE)",
          _resp is None, f"got {_resp!r}")
finally:
    srv._CLIENT_RECV_TIMEOUT_SEC = _old_client_timeout
    if _old_xdg is None:
        os.environ.pop("XDG_RUNTIME_DIR", None)
    else:
        os.environ["XDG_RUNTIME_DIR"] = _old_xdg
    import shutil
    shutil.rmtree(_tmpdir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
# Slice DLW: daemon LoRA weight application. The old add-loop keyed the
# diff on path alone (weight-only changes silently ignored) and never
# called set_adapters (every PEFT adapter served at full trained
# strength — the tier-1 gap fixed on the CLI path in 1f52672).
print("── slice DLW: daemon LoRA weights ─────────────────────────────")

import comfyless.generate as _gen2
import nodes.eric_qwen_edit_lora as _nlora


import types as _types


class _FakeLoraPipe:
    def __init__(self, peft_config=None):
        self.transformer = _types.SimpleNamespace(
            peft_config=peft_config or {})
        self.deleted = []
        self.set_adapters_calls = []

    def delete_adapters(self, names):
        self.deleted.extend(names)

    def get_list_adapters(self):
        return {}

    def set_adapters(self, names, adapter_weights=None):
        self.set_adapters_calls.append((list(names), list(adapter_weights)))


_dlw_load_calls: list = []
_dlw_apply_calls: list = []
_dlw_apply_result: list = [None]


def _dlw_fake_loader(pipe, lora_path, adapter_name, log_prefix="[t]",
                     weight=1.0):
    _dlw_load_calls.append((lora_path, adapter_name, weight))
    return True


def _dlw_fake_apply(pipe, pairs):
    _dlw_apply_calls.append(list(pairs))
    return _dlw_apply_result[0]


_orig_nlora_loader = _nlora.load_lora_with_key_fix
_orig_apply = _gen2.apply_adapter_weights
_orig_load2 = _gen2._load_pipeline
_orig_generate2 = _gen2.generate
_dlw_pipe_holder: list = []


def _dlw_fake_load(model_path, **kw):
    pipe = _FakeLoraPipe()
    _dlw_pipe_holder.append(pipe)
    return pipe, "qwen-image", False


_nlora.load_lora_with_key_fix = _dlw_fake_loader
_gen2.apply_adapter_weights = _dlw_fake_apply
_gen2._load_pipeline = _dlw_fake_load
_gen2.generate = lambda **kw: {"model_family": "qwen-image"}
try:
    _outdir2 = tempfile.mkdtemp()
    _la = os.path.join(_outdir2, "la.safetensors")
    _lb = os.path.join(_outdir2, "lb.safetensors")
    for _p in (_la, _lb):
        open(_p, "wb").close()

    # 1. Fresh load of two LoRAs → ONE cumulative apply over both, at the
    #    requested weights (NEGATIVE vs pre-slice: apply was never called).
    _state2: dict = {}
    _r = srv._handle_generate(
        {"type": "generate", "model": _outdir2, "prompt": "x",
         "loras": [{"path": _la, "weight": 0.7},
                   {"path": _lb, "weight": 0.5}]},
        _outdir2, _outdir2, "cuda", "bf16", _state2,
    )
    check("DLW: request ok", _r.get("status") == "ok", f"{_r!r}")
    check("DLW: cumulative apply called once with both adapters at the "
          "requested weights (invariant 1)",
          len(_dlw_apply_calls) == 1
          and [w for _, w in _dlw_apply_calls[0]] == [0.7, 0.5],
          f"calls {_dlw_apply_calls}")

    # 2. Weight-only change on a PEFT adapter → NO reload; record updated;
    #    apply carries the new weight (invariant 2 — the old path-only
    #    diff silently ignored this, which would fail here).
    _n_loads = len(_dlw_load_calls)
    _dlw_apply_calls.clear()
    _r2 = srv._handle_generate(
        {"type": "generate", "model": _outdir2, "prompt": "x",
         "loras": [{"path": _la, "weight": 0.3},
                   {"path": _lb, "weight": 0.5}]},
        _outdir2, _outdir2, "cuda", "bf16", _state2,
    )
    check("DLW: weight-only change reloads NOTHING",
          len(_dlw_load_calls) == _n_loads, f"loads {_dlw_load_calls}")
    check("DLW: weight-only change applies the NEW weight (invariant 2 "
          "NEGATIVE vs old behavior)",
          _dlw_apply_calls and [w for _, w in _dlw_apply_calls[0]] == [0.3, 0.5],
          f"calls {_dlw_apply_calls}")
    check("DLW: loaded_loras record updated",
          _state2["loaded_loras"][0]["weight"] == 0.3,
          f"{_state2['loaded_loras']}")

    # 3. Weight-only change on a DIRECT-MERGE adapter → loud warning in
    #    metadata, record keeps the baked weight (invariant 3).
    _pipe = _state2["pipeline"]
    _an = _state2["loaded_loras"][0]["adapter_name"]
    _pipe.transformer.peft_config = {_an: {"_type": "lora_direct",
                                           "_weight": 0.3}}
    _dlw_apply_calls.clear()
    _r3 = srv._handle_generate(
        {"type": "generate", "model": _outdir2, "prompt": "x",
         "loras": [{"path": _la, "weight": 0.9},
                   {"path": _lb, "weight": 0.5}]},
        _outdir2, _outdir2, "cuda", "bf16", _state2,
    )
    _md3 = _r3.get("metadata") or {}
    check("DLW: direct-merge weight change warns loudly (invariant 3)",
          any("weight change ignored" in w
              for w in _md3.get("lora_warnings", [])),
          f"md {_md3!r}")
    check("DLW: direct-merge record keeps the baked weight (NEGATIVE — "
          "never silently pretend)",
          _state2["loaded_loras"][0]["weight"] == 0.3,
          f"{_state2['loaded_loras']}")
    _pipe.transformer.peft_config = {}

    # 4. Removal re-pins survivors: dropping lb leaves apply covering la
    #    only (cumulative over the FULL active set).
    _dlw_apply_calls.clear()
    srv._handle_generate(
        {"type": "generate", "model": _outdir2, "prompt": "x",
         "loras": [{"path": _la, "weight": 0.3}]},
        _outdir2, _outdir2, "cuda", "bf16", _state2,
    )
    check("DLW: dropped LoRA deleted from the pipe",
          _pipe.deleted != [], f"deleted {_pipe.deleted}")
    check("DLW: apply re-pins the SURVIVING adapter only",
          _dlw_apply_calls and len(_dlw_apply_calls[0]) == 1
          and _dlw_apply_calls[0][0][1] == 0.3,
          f"calls {_dlw_apply_calls}")

    # 5. apply_adapter_weights warning surfaces in response metadata.
    _dlw_apply_result[0] = "set_adapters failed — LoRA(s) ['x'] remain"
    _r5 = srv._handle_generate(
        {"type": "generate", "model": _outdir2, "prompt": "x",
         "loras": [{"path": _la, "weight": 0.3}]},
        _outdir2, _outdir2, "cuda", "bf16", _state2,
    )
    _md5 = _r5.get("metadata") or {}
    check("DLW: scaling-failure warning surfaces in metadata "
          "(warn-don't-block)",
          any("set_adapters failed" in w
              for w in _md5.get("lora_warnings", [])), f"md {_md5!r}")
    _dlw_apply_result[0] = None

    # 6. No LoRAs → apply never called (no gratuitous set_adapters on
    #    LoRA-free pipelines — NEGATIVE).
    _dlw_apply_calls.clear()
    srv._handle_generate(
        {"type": "generate", "model": _outdir2, "prompt": "x"},
        _outdir2, _outdir2, "cuda", "bf16", {})
    check("DLW: LoRA-free request never touches set_adapters (NEGATIVE)",
          _dlw_apply_calls == [], f"calls {_dlw_apply_calls}")

    # 7. Duplicate path within ONE request loads once, last weight wins
    #    (review finding 1 — the second occurrence must take the
    #    weight-update branch, never double-load under the same adapter
    #    name, which for a direct-merge fallback would merge twice into
    #    the served pipeline).
    _dlw_load_calls.clear()
    _dlw_apply_calls.clear()
    _state7: dict = {}
    srv._handle_generate(
        {"type": "generate", "model": _outdir2, "prompt": "x",
         "loras": [{"path": _la, "weight": 0.7},
                   {"path": _la, "weight": 0.4}]},
        _outdir2, _outdir2, "cuda", "bf16", _state7,
    )
    check("DLW: duplicate path in one request loads ONCE (finding 1 "
          "NEGATIVE)", len(_dlw_load_calls) == 1,
          f"loads {_dlw_load_calls}")
    check("DLW: duplicate path — last weight wins, one record",
          len(_state7["loaded_loras"]) == 1
          and _state7["loaded_loras"][0]["weight"] == 0.4
          and _dlw_apply_calls[-1] == [(_state7["loaded_loras"][0]
                                        ["adapter_name"], 0.4)],
          f"recs {_state7['loaded_loras']} calls {_dlw_apply_calls}")

    # 8. Seam test: run ONE scenario with the REAL apply_adapter_weights
    #    (reviewer f) — pins the [(name, weight)] pair contract end-to-end
    #    so both suites can't drift apart while the integration breaks.
    _gen2.apply_adapter_weights = _orig_apply
    _state8: dict = {}
    srv._handle_generate(
        {"type": "generate", "model": _outdir2, "prompt": "x",
         "loras": [{"path": _la, "weight": 0.6}]},
        _outdir2, _outdir2, "cuda", "bf16", _state8,
    )
    _pipe8 = _state8["pipeline"]
    check("DLW: REAL helper end-to-end — daemon pipe receives the "
          "cumulative set_adapters call (seam pin)",
          _pipe8.set_adapters_calls
          and _pipe8.set_adapters_calls[-1][1] == [0.6],
          f"calls {_pipe8.set_adapters_calls}")
    _gen2.apply_adapter_weights = _dlw_fake_apply
finally:
    _nlora.load_lora_with_key_fix = _orig_nlora_loader
    _gen2.apply_adapter_weights = _orig_apply
    _gen2._load_pipeline = _orig_load2
    _gen2.generate = _orig_generate2
    import shutil as _sh
    _sh.rmtree(_outdir2, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────
print("\n── ADR-040 D2: opt-in root disclosure on ping ────────────────")

# D2 removes the false premise ADR-037 D5 rested on ("the client cannot know
# its roots") by letting `ping` report the daemon's output_dir and
# ref_image_roots — but ONLY when explicitly asked (D2a). The hazard being
# designed against: `ping` is the request a future HTTP/mcpo bridge forwards as
# a health check, and root enumeration must not be the default answer to the
# cheapest unauthenticated call.

_D2_OUT = "/tmp/d2-output-dir"
_D2_ROOTS = ("/tmp/d2-output-dir", "/tmp/d2-extra-ref-root")


def _ping_response(extra: dict) -> dict:
    """Drive one ping through the real _handle_connection, return the response."""
    a, b = socket.socketpair()
    try:
        srv._send(a, {"type": "ping", **extra})
        srv._handle_connection(
            conn=b,
            output_dir=_D2_OUT,
            model_base="/tmp",
            device="cuda",
            precision="bf16",
            server_state={},
            ref_roots=_D2_ROOTS,
        )
        resp = srv._recv(a)
        return resp if resp is not None else {}
    finally:
        a.close()
        b.close()


# 1. A plain ping discloses NO paths — the leak-by-default negative.
_plain = _ping_response({})
check("plain ping still answers pong (no existing client path changed)",
      _plain.get("status") == "ok" and _plain.get("message") == "pong",
      f"resp={_plain!r}")
check("plain ping discloses no output_dir",
      "output_dir" not in _plain, f"resp={_plain!r}")
check("plain ping discloses no ref_image_roots",
      "ref_image_roots" not in _plain, f"resp={_plain!r}")

# 2. report_roots=True gets the actual values the GATE compares against.
_opted = _ping_response({"report_roots": True})
check("ping report_roots=True reports output_dir",
      _opted.get("output_dir") == _D2_OUT, f"resp={_opted!r}")
check("ping report_roots=True reports ref_image_roots exactly as the gate holds them",
      _opted.get("ref_image_roots") == list(_D2_ROOTS), f"resp={_opted!r}")

# Report/gate parity. NOTE what is and isn't provable here: feeding the
# REPORTED value back into _check_ref_paths is a tautology (this helper hands
# back what it was given), so that check is deliberately absent — both slice-1
# reviewers flagged an earlier version of it. The real invariant is a WIRING
# one: run_server must pass ref_image_roots (the _resolve_ref_roots output,
# realpath'd) into _handle_connection's single `ref_roots` parameter, which
# then serves BOTH the report and _check_ref_paths. One parameter feeding both
# is what makes divergence unrepresentable — and divergence is what would let a
# client pass D3 entry validation and still be refused mid-run, i.e. the
# incident this ADR exists to kill, surviving behind a green check.
check("run_server reports the RESOLVED roots, not the raw --ref-root spawn tuple",
      "server_state, extra_roots, ref_image_roots" in _inspect.getsource(srv.run_server))
_hc_src = _inspect.getsource(srv._handle_connection)
check("one ref_roots parameter feeds both the report and the ref gate",
      'resp["ref_image_roots"] = list(ref_roots)' in _hc_src
      and "_check_ref_paths(req, ref_roots)" in _hc_src)

# 3. Explicit False is not an opt-in.
_off = _ping_response({"report_roots": False})
check("ping report_roots=False discloses no paths",
      "output_dir" not in _off and "ref_image_roots" not in _off,
      f"resp={_off!r}")

# 4. The flag is honored ONLY on ping (D2a) — fail-closed, not ignored, so a
#    future request type cannot inherit a disclosure flag by accident.
_err = srv._validate_request({"type": "generate", "model": "m", "prompt": "p",
                              "report_roots": True})
check("report_roots on a generate request is a ValidationError",
      _err is not None and "report_roots" in _err, f"err={_err!r}")
_err = srv._validate_request({"type": "unload", "report_roots": True})
check("report_roots on an unload request is a ValidationError",
      _err is not None and "report_roots" in _err, f"err={_err!r}")
check("report_roots on ping itself validates clean",
      srv._validate_request({"type": "ping", "report_roots": True}) is None,
      f"err={srv._validate_request({'type': 'ping', 'report_roots': True})!r}")

# 5. Type is owned by the canonical validator (ADR-012), not by an isinstance
#    predicate in server.py — a non-bool is rejected before the value check.
_err = srv._validate_request({"type": "ping", "report_roots": "yes"})
check("non-bool report_roots is rejected by the canonical validator",
      _err is not None and "report_roots" in _err, f"err={_err!r}")
_err = srv._validate_request({"type": "ping", "report_roots": 1})
check("int report_roots is rejected (bool kind, not truthy-int)",
      _err is not None and "report_roots" in _err, f"err={_err!r}")
_err = srv._validate_request({"type": "ping", "report_roots": None})
check("null report_roots is rejected (the sloppy-client value)",
      _err is not None and "report_roots" in _err, f"err={_err!r}")
_err = srv._validate_request({"type": "ping", "report_roots": ["true"]})
check("list report_roots is rejected",
      _err is not None and "report_roots" in _err, f"err={_err!r}")

# 6. The gate is `is True`, not truthiness — so it fails CLOSED even if the
#    _RUNTIME_KIND registration is ever removed. Pin the identity comparison
#    itself: without the registration, validate_machine_request passes unknown
#    keys through unchanged and a truthiness gate would disclose on any
#    non-empty string (slice-1 security review MEDIUM).
check("the disclosure gate uses `is True`, not truthiness",
      'req.get("report_roots") is True' in _hc_src)


# ══════════════════════════════════════════════════════════════════════
print("\n── ADR-044: identity-edit residue never survives in the cache ──")
# ══════════════════════════════════════════════════════════════════════
# THE invariant the whole Part C cache-key decision rests on (security review
# Finding 2 / code review 2026-08-01 advisory 4). ref_boost/grounding_px/identity
# are deliberately OUT of _request_cache_key because they select output, not
# pipeline shape — which is only true if a per-call mode provably leaves nothing
# on the cached pipeline.
#
# pipelines/krea2_identity_edit.py restores its attention processors in a
# finally, but a finally is not a proof: if the RESTORE ITSELF raises, the
# identity processors stay installed on a pipeline this daemon caches and hands
# to the next request. That failure is silent at the same resolution — stale
# processors carry frozen text_len/src_len/tgt_len, so an --iterate sweep just
# gets a wrong attention bias. So the daemon verifies rather than assumes.
import contextlib  # noqa: E402
import io  # noqa: E402

from pipelines.krea2_identity_edit import (                        # noqa: E402
    Krea2IdentityEditAttnProcessor)


class _ResidueTransformer:
    def __init__(self, procs):
        self.attn_processors = procs


class _ResiduePipe:
    def __init__(self, procs):
        self.transformer = _ResidueTransformer(procs)


def _residue_run(*, identity, procs):
    """Drive _handle_generate to its except: branch and report the cache state.

    The cache key must MATCH so the fake pipeline is the one served: a mismatch
    evicts and reloads via _fake_load, replacing it with a bare object() that
    has no .transformer — and then the residue check finds nothing and the test
    passes for the wrong reason. (It did exactly that on the first run.)
    """
    _req = {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "p"}
    if identity:
        _req["identity"] = True
    _state = {"pipeline": _ResiduePipe(procs), "model_family": "krea-turbo",
              "guidance_embeds": False, "loaded_loras": [],
              "cache_key": srv._request_cache_key(_req, "bf16", "cuda")}

    def _boom(**kw):
        raise RuntimeError("simulated generation failure")

    _saved_load, _saved_gen = _gen._load_pipeline, _gen.generate
    _gen._load_pipeline, _gen.generate = _fake_load, _boom
    try:
        _d = tempfile.mkdtemp()
        _r = srv._handle_generate(_req, _d, _d, "cuda", "bf16", _state)
    finally:
        _gen._load_pipeline, _gen.generate = _saved_load, _saved_gen
    return _r, _state


# Premise for the whole block: the fake pipeline must actually be SERVED from
# cache, not reloaded. If this fails, every check below is vacuous.
_r_premise, _s_premise = _residue_run(
    identity=False, procs={"transformer_blocks.0": object()})
check("residue premise: the cached fake pipeline is served, not reloaded",
      isinstance(_s_premise.get("pipeline"), _ResiduePipe),
      f"got {type(_s_premise.get('pipeline')).__name__}")


_id_proc = Krea2IdentityEditAttnProcessor(
    text_len=4, src_len=8, tgt_len=8, ref_boost=4.0)

# 1. identity run + residue present -> the cache entry is dropped.
_r, _s = _residue_run(identity=True, procs={"transformer_blocks.0": _id_proc})
check("residue: a FAILED identity run still returns InferenceError",
      _r.get("error_type") == "InferenceError", f"resp={_r!r}")
check("residue: identity processors surviving a failure EVICT the pipeline",
      "pipeline" not in _s, f"state keys: {sorted(_s)}")

# 2. NEGATIVE — identity run, NO residue (the restore worked, the normal case).
#    Must NOT evict: an ordinary OOM should not cost a ~30 GB reload.
_r, _s = _residue_run(identity=True, procs={"transformer_blocks.0": object()})
check("residue: a clean identity failure does NOT evict (no reload tax)",
      _s.get("pipeline") is not None)

# 3. NEGATIVE — the check is gated on the request, not run for every family.
#    Residue cannot exist here (identity never ran), and paying an
#    attn_processors walk on every failed flux/qwen run would be pure cost.
_r, _s = _residue_run(identity=False, procs={"transformer_blocks.0": _id_proc})
check("residue: a NON-identity request does not run the residue check",
      _s.get("pipeline") is not None)

# 4. The check must never mask the real error nor wedge the accept loop — a
#    pipeline with no transformer at all (or a torn-down one) must still yield
#    a clean InferenceError rather than an exception out of the handler.
_r, _s = _residue_run(identity=True, procs={})
check("residue: an empty processor map is handled, not raised on",
      _r.get("error_type") == "InferenceError")


class _RaisingTransformer:
    """A transformer whose attn_processors walk itself throws."""
    @property
    def attn_processors(self):
        raise RuntimeError("exotic wrapper: attn_processors unavailable")


class _RaisingPipe:
    def __init__(self):
        self.transformer = _RaisingTransformer()


# The fail-OPEN branch: the check swallows its own failure so it can never mask
# the real InferenceError, but it must log rather than vanish — what it fails
# open INTO is the silent-wrong-bias case it exists to prevent (security review
# 2026-08-01, MEDIUM). Exercised with an inspection that RAISES; the legs above
# only cover absent/empty.
_state_raise = {"pipeline": _RaisingPipe(), "model_family": "krea-turbo",
                "guidance_embeds": False, "loaded_loras": []}
_req_raise = {"type": "generate", "model": "/fake/Krea-2-Turbo",
              "prompt": "p", "identity": True}
_state_raise["cache_key"] = srv._request_cache_key(_req_raise, "bf16", "cuda")
_saved_load, _saved_gen = _gen._load_pipeline, _gen.generate
_gen._load_pipeline = _fake_load


def _boom3(**kw):
    raise RuntimeError("simulated generation failure")


_gen.generate = _boom3
_res_log = io.StringIO()
try:
    _d3 = tempfile.mkdtemp()
    with contextlib.redirect_stderr(_res_log):
        _r_raise = srv._handle_generate(
            _req_raise, _d3, _d3, "cuda", "bf16", _state_raise)
finally:
    _gen._load_pipeline, _gen.generate = _saved_load, _saved_gen
check("residue: an inspection that RAISES does not mask the InferenceError",
      _r_raise.get("error_type") == "InferenceError", f"resp={_r_raise!r}")
check("residue: ...and the swallowed failure is LOGGED, not silent",
      "residue check failed" in _res_log.getvalue(),
      f"stderr={_res_log.getvalue()!r}")
_state_no_pipe: dict = {}
try:
    _saved_load, _saved_gen = _gen._load_pipeline, _gen.generate
    _gen._load_pipeline = _fake_load

    def _boom2(**kw):
        raise RuntimeError("boom")
    _gen.generate = _boom2
    _d2 = tempfile.mkdtemp()
    _r_np = srv._handle_generate(
        {"type": "generate", "model": "/fake/Krea-2-Turbo", "prompt": "p",
         "identity": True}, _d2, _d2, "cuda", "bf16", _state_no_pipe)
finally:
    _gen._load_pipeline, _gen.generate = _saved_load, _saved_gen
check("residue: a state with no cached pipeline is handled, not raised on",
      _r_np.get("error_type") == "InferenceError", f"resp={_r_np!r}")


# ──────────────────────────────────────────────────────────────────────
print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
