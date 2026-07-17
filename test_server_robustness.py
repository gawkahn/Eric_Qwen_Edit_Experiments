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

_gen_src = Path(_gen.__file__).read_text()
check("delegation-skip branch removed from generate.py",
      "daemon delegation skipped" not in _gen_src)
# Positive property (review N-2): the delegation guard exists and no longer
# consults args.quant anywhere.
check("delegation guard present and quant-free",
      "if args.savepath or using_default_output:" in _gen_src
      and 'args.quant != "none" and (args.savepath' not in _gen_src)


print("\n── NAG (ADR-023): key freedom + forwarding + wire carriage ────")

# DECIDED (Vision slice NAG): the NAG quadruple stays OUT of the pipeline
# cache key. NAG changes output content but not pipeline shape — the NAG
# attention processors are installed per-call and restored in a finally
# (pipelines/nag_krea2.py), so a cached pipeline serves any NAG config.
# A key that discriminated on nag_* would evict/reload on every NAG tweak.
check("nag params do NOT change the cache key (per-request safe)",
      _K(dict(_r0, nag_scale=5.0, nag_tau=3.0, nag_alpha=0.5, nag_end=0.75),
         "bf16", "cuda") == _k_none)
check("nag params do NOT change the cache key under quant either",
      _K(dict(_r0, quant="fp8", nag_scale=5.0), "bf16", "cuda") == _k_fp8)

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
print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
