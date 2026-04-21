#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""
comfyless persistent model server.

Keeps a diffusers pipeline loaded between invocations, eliminating the 30–90s
model-load overhead on every generation run.

Start:
    python -m comfyless.generate --serve \\
        --device cuda:1 \\
        --output-dir ~/gen-output \\
        --model-base /home/.../models &

Normal comfyless.generate invocations auto-detect the socket and delegate.
Send --unload to shut the server down cleanly.

Security model: see docs/decisions/ADR-001-daemon-socket-security.md
  - Socket in $XDG_RUNTIME_DIR (0700) or /tmp/comfyless-$UID/ (0700)
  - All output paths resolved within --output-dir; client never dictates paths
  - All model/LoRA paths validated against --model-base before any load
  - Adapter names sanitized to [a-zA-Z0-9_-] before use
  - Schema validated at socket boundary before any parameter reaches ML code

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# ════════════════════════════════════════════════════════════════════════
#  Socket location
# ════════════════════════════════════════════════════════════════════════

def _socket_dir() -> Path:
    """Return the per-UID socket directory, creating it at mode 0700 if needed."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        # systemd provisions XDG_RUNTIME_DIR at 0700; don't chmod what it manages
        return Path(xdg)
    d = Path(f"/tmp/comfyless-{os.getuid()}")
    d.mkdir(mode=0o700, exist_ok=True)
    return d


def socket_path() -> Path:
    """Return the Unix socket path for this user's comfyless server."""
    return _socket_dir() / "comfyless.sock"


# ════════════════════════════════════════════════════════════════════════
#  Request schema validation
# ════════════════════════════════════════════════════════════════════════

_GENERATE_REQUIRED: Dict[str, type] = {
    "model":  str,
    "prompt": str,
}

_GENERATE_OPTIONAL: Dict[str, Any] = {
    "negative_prompt":     str,
    "seed":                int,
    "steps":               int,
    "cfg_scale":           float,
    "true_cfg_scale":      (float, type(None)),
    "width":               int,
    "height":              int,
    "sampler":             str,
    "schedule":            str,
    "max_sequence_length": int,
    "loras":               list,
    "precision":           str,
    "offload_vae":         bool,
    "attention_slicing":   bool,
    "sequential_offload":  bool,
    "transformer_path":    str,
    "vae_path":            str,
    "text_encoder_path":   str,
    "text_encoder_2_path": str,
    "vae_from_transformer": bool,
    "savepath":            str,
}


def _validate_request(req: dict) -> Optional[str]:
    """Return an error string if the request is malformed, else None."""
    req_type = req.get("type")
    if req_type not in ("generate", "unload", "ping"):
        return f"Unknown request type: {req_type!r}. Expected: generate | unload | ping"
    if req_type != "generate":
        return None

    for field, expected in _GENERATE_REQUIRED.items():
        if field not in req:
            return f"Missing required field: {field!r}"
        if not isinstance(req[field], expected):
            return (f"Field {field!r}: expected {expected.__name__}, "
                    f"got {type(req[field]).__name__}")

    for field, expected in _GENERATE_OPTIONAL.items():
        val = req.get(field)
        if val is None:
            continue
        if isinstance(expected, tuple):
            if not isinstance(val, expected):
                names = " | ".join(t.__name__ for t in expected)
                return f"Field {field!r}: expected {names}, got {type(val).__name__}"
        elif not isinstance(val, expected):
            return (f"Field {field!r}: expected {expected.__name__}, "
                    f"got {type(val).__name__}")

    for i, lora in enumerate(req.get("loras") or []):
        if not isinstance(lora, dict) or "path" not in lora:
            return f"loras[{i}]: expected {{\"path\": str, \"weight\": float}}"
        if not isinstance(lora["path"], str):
            return f"loras[{i}].path: expected str"

    return None


# ════════════════════════════════════════════════════════════════════════
#  Path enforcement
# ════════════════════════════════════════════════════════════════════════

def _within(path: str, base: str) -> bool:
    """True if path resolves to base or any descendant of base."""
    r = os.path.realpath(path)
    b = os.path.realpath(base)
    return r == b or r.startswith(b + os.sep)


def _check_paths(req: dict, model_base: str) -> Optional[str]:
    """Return an error string if any path in the request is outside model_base."""
    model = req.get("model", "")
    if not _within(model, model_base):
        return f"model path outside --model-base: {model!r}"

    for field in ("transformer_path", "vae_path", "text_encoder_path", "text_encoder_2_path"):
        p = req.get(field, "") or ""
        if p and not _within(p, model_base):
            return f"{field} outside --model-base: {p!r}"

    for i, lora in enumerate(req.get("loras") or []):
        p = lora.get("path", "")
        if p and not _within(p, model_base):
            return f"loras[{i}].path outside --model-base: {p!r}"

    return None


def sanitize_adapter_name(name: str) -> str:
    """Strip characters outside [a-zA-Z0-9_-] to prevent downstream injection."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ════════════════════════════════════════════════════════════════════════
#  Wire protocol — newline-terminated JSON over Unix socket
# ════════════════════════════════════════════════════════════════════════

def _send(conn: socket.socket, payload: dict) -> None:
    conn.sendall(json.dumps(payload).encode() + b"\n")


def _recv(conn: socket.socket) -> Optional[dict]:
    """Read one newline-terminated JSON message from the connection."""
    buf = b""
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            return None
        buf += chunk
        if b"\n" in buf:
            line, _ = buf.split(b"\n", 1)
            return json.loads(line.decode())


# ════════════════════════════════════════════════════════════════════════
#  Connection handler
# ════════════════════════════════════════════════════════════════════════

def _handle_connection(
    conn: socket.socket,
    output_dir: str,
    model_base: str,
    device: str,
    precision: str,
    server_state: dict,
) -> bool:
    """
    Process one client connection.

    Returns True to keep serving, False to stop the server.
    server_state is a mutable dict shared across calls for the model cache
    (populated in Step 3; empty dict is safe here).
    """
    try:
        req = _recv(conn)
    except (json.JSONDecodeError, ValueError) as e:
        _send(conn, {"status": "error", "error_type": "ParseError",
                     "error": f"Invalid JSON: {e}"})
        return True

    if req is None:
        return True

    # ── Schema validation ────────────────────────────────────────────────
    err = _validate_request(req)
    if err:
        _send(conn, {"status": "error", "error_type": "ValidationError", "error": err})
        return True

    req_type = req["type"]

    if req_type == "ping":
        _send(conn, {"status": "ok", "message": "pong"})
        return True

    if req_type == "unload":
        # Clean up any loaded pipeline (Step 3 will populate this)
        pipeline = server_state.get("pipeline")
        if pipeline is not None:
            _log("Unloading pipeline from VRAM")
            del pipeline
            import torch; torch.cuda.empty_cache()
            server_state.clear()
        _send(conn, {"status": "ok", "message": "unloaded"})
        return False  # signal server loop to stop

    # req_type == "generate"
    # ── Path enforcement ─────────────────────────────────────────────────
    err = _check_paths(req, model_base)
    if err:
        _send(conn, {"status": "error", "error_type": "PathError", "error": err})
        return True

    # ── Generation (wired in Step 3) ─────────────────────────────────────
    result = _handle_generate(req, output_dir, model_base, device, precision, server_state)
    _send(conn, result)
    return True


def _handle_generate(
    req: dict,
    output_dir: str,
    model_base: str,
    device: str,
    precision: str,
    server_state: dict,
) -> dict:
    """
    Execute a validated generate request.  Stub until Step 3.
    server_state carries the cached pipeline and loaded LoRA set between calls.
    """
    return {
        "status": "error",
        "error_type": "NotImplemented",
        "error": "Generation handler not yet wired — Step 3 pending",
    }


# ════════════════════════════════════════════════════════════════════════
#  Server entry point
# ════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    print(f"[comfyless-server] {msg}", file=sys.stderr, flush=True)


def run_server(
    output_dir: str,
    model_base: str,
    device: str = "cuda",
    precision: str = "bf16",
) -> None:
    """Start the comfyless model server and block until --unload is received."""
    output_dir = os.path.realpath(output_dir)
    model_base = os.path.realpath(model_base)

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"--output-dir not found: {output_dir}")
    if not os.path.isdir(model_base):
        raise FileNotFoundError(f"--model-base not found: {model_base}")

    sock_path = socket_path()
    if sock_path.exists():
        sock_path.unlink()

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(4)
    os.chmod(str(sock_path), 0o600)  # belt-and-suspenders; dir is already 0700

    _log(f"Listening on {sock_path}")
    _log(f"output-dir : {output_dir}")
    _log(f"model-base : {model_base}")
    _log(f"device     : {device} / {precision}")

    server_state: dict = {}
    keep_running = True
    try:
        while keep_running:
            conn, _ = srv.accept()
            with conn:
                keep_running = _handle_connection(
                    conn, output_dir, model_base, device, precision, server_state
                )
    finally:
        srv.close()
        if sock_path.exists():
            sock_path.unlink()
        _log("Stopped.")
