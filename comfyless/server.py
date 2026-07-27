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
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from comfyless.params_validation import QUANT_MODES, validate_machine_request
from comfyless.output_format import resolve_output_format


# Wire-protocol guardrails (see docs/security/review-comfyless-server-2026-04-23.md)
_MAX_FRAME_BYTES = 1 << 20   # 1 MiB; real requests are < 10 KiB
_RECV_TIMEOUT_SEC = 5.0       # server-side request-read deadline (DoS guard)
_CLIENT_RECV_TIMEOUT_SEC = 600.0  # client-side response-read deadline: large
                                  # enough for a real generation (30–120s is
                                  # common), small enough to eventually surface
                                  # a wedged server instead of hanging forever.

# Path-shaped request fields — rejected if they contain a NUL byte, because
# os.path.realpath raises on NUL and that exception would escape _check_paths
# and kill the accept loop.
_PATH_FIELDS = frozenset({
    "model",
    "transformer_path",
    "vae_path",
    "upscale_vae_path",
    "text_encoder_path",
    "text_encoder_2_path",
    "refiner_path",
    "savepath",
})


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
    # A pre-planted symlink at this sticky-/tmp name would redirect the socket
    # into an attacker-chosen directory; d.stat() below follows symlinks so it
    # can't catch this on its own (hardening review H-1).
    if d.is_symlink():
        raise RuntimeError(f"socket dir {d} is a symlink — refusing")
    # mkdir(exist_ok=True) does not re-apply mode on existing dirs; enforce it.
    st = d.stat()
    if st.st_uid != os.getuid():
        raise RuntimeError(f"socket dir {d} not owned by current uid")
    if stat.S_IMODE(st.st_mode) != 0o700:
        # 0o700 TIGHTENS the socket dir to owner-only (the rule's 0o644
        # suggestion is for files and would be a loosening here) — ADR-001.
        os.chmod(d, 0o700)  # nosemgrep: python.lang.security.audit.insecure-file-permissions.insecure-file-permissions
    return d


# Device strings that may be turned into a socket filename. Anchored full-match
# (fullmatch, not match+$: '$' also matches before a trailing '\n', which would
# smuggle a newline into the socket name). re.ASCII keeps \d to [0-9] so the
# regex — not the later int() fold — is the sole gate (rejects unicode digits
# outright); the (?:...) group keeps the alternation anchored if this is ever
# switched to .match()/.search(). See ADR-020 §3 and
# docs/security/review-parallel-daemon-2026-07-03.md Finding 3.
_DEVICE_RE = re.compile(r"(?:cpu|cuda(:\d+)?)", re.ASCII)


def _device_socket_slug(device: str) -> str:
    """Map a device string to a canonical socket-name slug.

    One daemon serves one GPU (ADR-020, design A); the socket name is keyed by
    device so daemons for different GPUs coexist in the same 0700 dir. The
    whitelist runs on the RAW input first (never on a pre-normalized string, or
    a crafted value could be massaged past the filter); only survivors — already
    restricted to {cpu, cuda, cuda:<digits>} — are canonicalized. The integer is
    parsed so 'cuda', 'cuda:0', 'cuda:00', 'cuda:007' all fold to the same slug
    ('cuda0'/'cuda7') and can never carry a non-[a-z0-9] byte into the filename.
    """
    if _DEVICE_RE.fullmatch(device) is None:
        raise ValueError(
            f"unsupported device for socket routing: {device!r} "
            f"(expected 'cpu', 'cuda', or 'cuda:<n>')"
        )
    if device == "cpu":
        return "cpu"
    idx = int(device.split(":", 1)[1]) if ":" in device else 0
    return f"cuda{idx}"


def socket_path(device: str = "cuda") -> Path:
    """Return the Unix socket path for this user's comfyless server on `device`.

    Device-keyed so one daemon per GPU can run concurrently (ADR-020). 'cuda'
    and 'cuda:0' name the same physical device and resolve to the same socket.
    """
    return _socket_dir() / f"comfyless-{_device_socket_slug(device)}.sock"


# ════════════════════════════════════════════════════════════════════════
#  Request schema validation
# ════════════════════════════════════════════════════════════════════════

def _validate_request(req: Any) -> Optional[str]:
    """Return an error string if the request is malformed, else None.

    Type-rule validation delegates to comfyless.params_validation per ADR-012
    (accepted 2026-05-15). This function owns three server-specific concerns
    the canonical validator does not:
      - Request-type-tag semantic check ('generate' | 'unload' | 'ping')
      - Required-field presence (canonical declares defaults, not required-ness)
      - Null-byte path defense (kept here as filesystem defense-in-depth,
        not type validity; see step-3 commit body for the rationale)

    No isinstance() predicates appear in this function's body — the N19 grep
    invariant from the slice Vision is now active for server.py.
    """
    # Canonical type validation handles non-dict payloads + every type rule.
    result = validate_machine_request(req)
    if not result.ok:
        err = result.error
        if err["field"] == "<root>":
            return f"Request must be a dict; {err['reason']}"
        return f"Field {err['field']!r}: {err['reason']}"

    # Propagate the validator's int→float safe-cast (ADR-012 §3) into the
    # caller's request dict so downstream consumers see the cast-applied
    # values rather than the un-cast wire input. Closes step-3 security-
    # auditor finding 6 (validated-payload-discarded parity gap).
    req.update(result.payload)

    # Type-tag semantic check (the canonical validator type-checks 'type' as
    # a string but does not enforce the allowed-value set).
    req_type = req.get("type")
    if req_type not in ("generate", "unload", "ping"):
        return f"Unknown request type: {req_type!r}. Expected: generate | unload | ping"

    # report_roots is honored ONLY on 'ping' (ADR-040 D2a). The canonical
    # validator has already type-checked it as bool; this is the request-type
    # VALUE check, mirroring the quant/output_format allowlist precedent below.
    # Fail-closed rather than ignore: a disclosure flag that is silently
    # accepted on other request types is one a future request type inherits by
    # accident, which is exactly the leak-by-default shape D2 exists to avoid.
    if "report_roots" in req and req_type != "ping":
        return (f"Field 'report_roots': honored only on type 'ping', "
                f"not {req_type!r}. Expected: type 'ping'")

    if req_type != "generate":
        return None

    # Required-field presence — server-specific; the canonical validator's
    # schema declares defaults but has no notion of "required."
    if "model" not in req:
        return "Missing required field: 'model'"
    if "prompt" not in req:
        return "Missing required field: 'prompt'"

    # quant-mode allowed-value check (slice DQ; mirrors the type-tag semantic
    # check above — the canonical validator type-checks 'quant' as a string
    # but does not enforce the value set). QUANT_MODES comes from
    # params_validation, NOT nodes.eric_diffusion_utils: this path runs
    # unguarded in the accept loop and must never trigger a torch import
    # (security review slice-DQ F1).
    q = req.get("quant")
    if q not in (None, "") and q not in QUANT_MODES:
        return (f"Field 'quant': unknown mode {q!r}. "
                f"Expected: {' | '.join(QUANT_MODES)}")

    # Output format/quality allowed-VALUE checks (ADR-034). The canonical
    # validator (validate_machine_request, via _RUNTIME_KIND) has already
    # type-checked output_format as str and quality as float — so these are
    # pure value checks, mirroring the quant-mode allowlist above and keeping
    # this function free of isinstance predicates (N19 invariant).
    of = req.get("output_format")
    if of not in (None, "", "png", "jpeg", "jpg"):
        return (f"Field 'output_format': unknown value {of!r}. "
                f"Expected: png | jpeg | jpg")
    ql = req.get("quality")
    if ql is not None and not (0.0 < ql <= 1.0):
        return f"Field 'quality': must be in (0.0, 1.0]; got {ql}"

    # Null-byte path defense. Kept server-specific rather than migrated into
    # the canonical validator (option discussed in the step-1 security
    # review): null-byte rejection is filesystem-defense-in-depth, adjacent
    # to but not the same concern as type validity. Centralizing it would
    # also tighten prompts/etc. unnecessarily. Future slice may revisit.
    for i, lora in enumerate(req.get("loras") or []):
        if "\x00" in lora["path"]:
            return f"loras[{i}].path: null byte not allowed"
    # ADR-035 6e: list-shaped ref_images joins the NUL defense. By this point
    # validate_machine_request has confirmed each entry is a dict with a str
    # 'path' (validate_ref_image_entry), so ref["path"] is safe with no
    # isinstance predicate (N19). A NUL here would otherwise escape
    # os.path.realpath in _check_ref_paths and kill the accept loop.
    for i, ref in enumerate(req.get("ref_images") or []):
        if "\x00" in ref["path"]:
            return f"ref_images[{i}].path: null byte not allowed"
    for field in _PATH_FIELDS:
        val = req.get(field, "")
        if val and "\x00" in val:
            return f"Field {field!r}: null byte not allowed"

    return None


# ════════════════════════════════════════════════════════════════════════
#  Path enforcement
# ════════════════════════════════════════════════════════════════════════

def _within(path: str, base: str) -> bool:
    """True if path resolves to base or any descendant of base."""
    r = os.path.realpath(path)
    b = os.path.realpath(base)
    return r == b or r.startswith(b + os.sep)


def _check_paths(req: dict, roots: Union[str, Sequence[str]]) -> Optional[str]:
    """Return an error string if any path in the request is outside every
    allowed root.

    `roots` is a single allowlist root (str) or a sequence of roots —
    {model_base} ∪ lora roots ∪ transformer roots (ADR-018 §3). A path is
    accepted iff it is `_within` ANY root. The union widens WHICH
    operator-curated trees are loadable, never who chooses them: callers
    still supply catalog names, and every root is a spawn-time operator
    argument.
    """
    if isinstance(roots, str):
        roots = (roots,)

    def _in_any(p: str) -> bool:
        return any(_within(p, b) for b in roots)

    model = req.get("model", "")
    if not model.startswith("/"):
        return f"model path must be absolute: {model!r}"
    if not _in_any(model):
        return f"model path outside the allowed roots: {model!r}"

    # refiner_path is included here per security-auditor 2026-06-01 CRITICAL
    # finding: the operator's --model-base policy (which directories
    # pickle-deserialization is allowed against) applies to the refiner (also
    # a model). The `_within` realpath+containment check enforces that
    # invariant for the refiner identically to every other model-path field.
    for field in ("transformer_path", "vae_path", "upscale_vae_path",
                  "text_encoder_path", "text_encoder_2_path", "refiner_path"):
        p = req.get(field, "") or ""
        if p:
            if not p.startswith("/"):
                return f"{field} must be absolute: {p!r}"
            if not _in_any(p):
                return f"{field} outside the allowed roots: {p!r}"

    for i, lora in enumerate(req.get("loras") or []):
        p = lora.get("path", "")
        if p:
            if not p.startswith("/"):
                return f"loras[{i}].path must be absolute: {p!r}"
            if not _in_any(p):
                return f"loras[{i}].path outside the allowed roots: {p!r}"

    return None


def _check_ref_paths(req: dict, ref_roots: Sequence[str]) -> Optional[str]:
    """Return an error string if any ref_images path is outside every ref root.

    ADR-035 6a: reference-image reads are a DIFFERENT permission from weight
    loads, so ref_images is validated against `ref_image_roots` — the daemon's
    --output-dir plus --ref-root spawn additions — NOT the {model_base} ∪ lora ∪
    transformer weight roots of _check_paths. The two allowlists share the
    `_within` mechanism but are disjoint in purpose and never merged: adding
    --output-dir to the weight roots would reopen the 2026-06-01 refiner-path
    CRITICAL, and exempting ref_images from containment entirely would let any
    same-UID wire client VAE-encode any user-readable file.

    Entries are already validated dicts with str, NUL-free paths (canonical
    validator + _validate_request), so this only decides containment. An empty
    ref_roots means no tree is readable — every ref path is refused (fail-closed).
    """
    if not req.get("ref_images"):
        return None

    def _in_any(p: str) -> bool:
        return any(_within(p, b) for b in ref_roots)

    for i, ref in enumerate(req["ref_images"]):
        p = ref.get("path", "")
        if not p.startswith("/"):
            return f"ref_images[{i}].path must be absolute: {p!r}"
        if not _in_any(p):
            return f"ref_images[{i}].path outside the ref-image roots: {p!r}"
    return None


def sanitize_adapter_name(name: str) -> str:
    """Strip characters outside [a-zA-Z0-9_-] to prevent downstream injection."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ════════════════════════════════════════════════════════════════════════
#  Wire protocol — newline-terminated JSON over Unix socket
# ════════════════════════════════════════════════════════════════════════

def _send(conn: socket.socket, payload: dict) -> None:
    conn.sendall(json.dumps(payload).encode() + b"\n")


def _send_safe(conn: socket.socket, payload: dict) -> bool:
    """_send wrapper that swallows peer-gone errors.

    A client that disconnects mid-request (crash, recv timeout, SIGKILL) must
    never kill the daemon. Used for every server-to-client send inside
    _handle_connection. Returns True if delivered, False if the peer was gone.
    """
    try:
        _send(conn, payload)
        return True
    except (BrokenPipeError, ConnectionResetError) as e:
        _log(f"Client disconnected before response delivery: {e}")
        return False


def _recv(conn: socket.socket, timeout: float = _RECV_TIMEOUT_SEC) -> Optional[dict]:
    """Read one newline-terminated JSON message from the connection.

    Default timeout is the server's DoS-guard value (5s). The client passes a
    much larger timeout when reading the server's response, since generation
    can take 30–120s on 20B-parameter models — the 5s server-side bound would
    always trip on the response path.
    """
    deadline = time.monotonic() + timeout
    buf = b""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("request timed out before newline")
        conn.settimeout(remaining)
        try:
            chunk = conn.recv(65536)
        except socket.timeout:
            raise ValueError("request timed out before newline")
        if not chunk:
            return None
        buf += chunk
        if len(buf) > _MAX_FRAME_BYTES:
            raise ValueError(f"request frame exceeds {_MAX_FRAME_BYTES} bytes")
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
    extra_roots: Tuple[str, ...] = (),
    ref_roots: Tuple[str, ...] = (),
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
        _send_safe(conn, {"status": "error", "error_type": "ParseError",
                          "error": f"Invalid JSON: {e}"})
        return True

    if req is None:
        return True

    # ── Schema validation ────────────────────────────────────────────────
    err = _validate_request(req)
    if err:
        _send_safe(conn, {"status": "error", "error_type": "ValidationError", "error": err})
        return True

    req_type = req["type"]

    if req_type == "ping":
        resp: dict = {"status": "ok", "message": "pong"}
        # ADR-040 D2: opt-in root disclosure. The refine loop needs the actual
        # values to derive its run dir under (D1) and to validate every path
        # the daemon will read BEFORE the first generation (D3), replacing the
        # false "the client cannot know its roots" premise ADR-037 D5 rested on.
        #
        # OPT-IN, never the default response: `ping` is precisely the request a
        # future HTTP/mcpo bridge forwards as a health check, and root
        # enumeration — including a broad --ref-root — must not be the answer to
        # the cheapest unauthenticated call. A blindly forwarded plain ping
        # discloses nothing.
        #
        # RESIDUAL (D2a): this daemon cannot discriminate an MCP caller from a
        # CLI caller — same UID, same socket, same schema — so the flag guards
        # against accidental forwarding, not a determined caller. The structural
        # fix is the `la mcp serve` boundary (D2b). Full reasoning lives in the
        # ADR; the long form for a maintainer hitting a failure is in
        # test_mcp_server.py's D2a block.
        #
        # `is True`, not truthiness (slice-1 security review MEDIUM): the ONLY
        # thing making this bool-only is the _RUNTIME_KIND registration in
        # params_validation.py, and validate_machine_request passes UNKNOWN keys
        # through unchanged — so de-registering or renaming that entry would
        # silently turn this gate into any-truthy-string. Identity comparison
        # fails closed regardless of registration state, and adds no isinstance
        # (ADR-012 / the N19 invariant both stay intact).
        if req.get("report_roots") is True:
            # The values the GATE actually compares against, not the spawn-time
            # strings: output_dir is realpath'd by run_server and every
            # ref_image_roots member by _resolve_ref_roots. Reporting anything
            # else would let a client pass entry validation (D3) and still be
            # refused mid-run — the incident this ADR exists to kill, surviving
            # behind a green check.
            resp["output_dir"] = output_dir
            resp["ref_image_roots"] = list(ref_roots)
            # The accepted residual above is only tolerable if it is
            # OBSERVABLE (slice-1 security review MEDIUM): every other
            # security-relevant event on this surface logs (PathError,
            # RefPathError), so a disclosure that left no trace would make the
            # residual undetectable after the fact. Count-only and path-free on
            # purpose — the roots are already in the startup banner, and this
            # line should not become a second copy of them.
            _log(f"report_roots: disclosed output_dir + {len(ref_roots)} "
                 f"ref root(s) to a ping client")
        _send_safe(conn, resp)
        return True

    if req_type == "unload":
        # Clean up any loaded pipeline. _evict_chain drops the refiner first
        # then the base (chain eviction order matters for CUDA release timing).
        if "pipeline" in server_state:
            _log("Unloading pipeline (and refiner if any) from VRAM")
            _evict_chain(server_state)
        _send_safe(conn, {"status": "ok", "message": "unloaded"})
        return False  # signal server loop to stop

    # req_type == "generate"
    # ── Path enforcement ─────────────────────────────────────────────────
    # ADR-018 §3: allowlist is the union {model_base} ∪ extra kind-typed
    # roots (all spawn-time operator arguments).
    err = _check_paths(req, (model_base, *extra_roots))
    if err:
        # Server-side audit log for path-validation rejections. Must happen
        # BEFORE _send_safe so the record exists even if the client has
        # disconnected (now a graceful no-op). Prompt kept out deliberately.
        redacted = {k: v for k, v in req.items() if k != "prompt"}
        _log(f"PathError: {err} req={redacted!r}")
        _send_safe(conn, {"status": "error", "error_type": "PathError", "error": err})
        return True

    # ── Reference-image containment (ADR-035 6a) ─────────────────────────
    # A DISJOINT allowlist from _check_paths above: ref reads are validated
    # against ref_image_roots (--output-dir ∪ --ref-root), never the weight
    # roots. Runs before decode — the daemon opens/VAE-encodes these files.
    err = _check_ref_paths(req, ref_roots)
    if err:
        redacted = {k: v for k, v in req.items() if k != "prompt"}
        _log(f"RefPathError: {err} req={redacted!r}")
        # DISTINCT error_type from the model-path PathError above: a reference
        # outside the daemon's ref_image_roots is a recoverable condition the
        # interactive client turns into an in-process fallback (ADR-035 slice 4b),
        # whereas a model-path PathError is a hard misconfiguration. Keying the
        # fallback on this type — not a substring of the message — keeps the two
        # apart so a real weight-root violation never silently retries in-process.
        _send_safe(conn, {"status": "error", "error_type": "RefPathError", "error": err})
        return True

    # ── Generation (wired in Step 3) ─────────────────────────────────────
    result = _handle_generate(req, output_dir, model_base, device, precision, server_state)
    _send_safe(conn, result)
    return True


def _request_cache_key(req: dict, precision: str, device: str) -> tuple:
    """Pipeline cache key for a validated generate request.

    Covers everything that affects pipeline shape. The quant triple is always
    present (constant ("none", (), ()) for unquantized requests). When quant
    is active, the requested LoRA (path, weight) set ALSO joins the key so a
    quantized pipeline is evicted and reloaded on ANY LoRA change instead of
    taking the incremental delete_adapters/add path — direct-merge adapters
    merge into requantized weights (ADR-019 slice DMR) and cannot be removed
    incrementally. Unquantized requests keep the existing LoRA-diff semantics
    (LoRA set deliberately NOT in the key). Paths are client abspaths, not
    realpaths — over-eviction on symlink aliases is accepted by design.
    See docs/security/review-slice-DQ-daemon-quant-2026-07-03.md (F2/F3/F4).

    NAG params (nag_scale/tau/alpha/end, ADR-023) are deliberately NOT in
    the key: they change output content but not pipeline shape — the NAG
    attention processors are installed per-call and restored in a finally
    (pipelines/nag_krea2.py), so a cached pipeline serves any NAG config.
    test_server_robustness pins this decision.
    """
    quant = str(req.get("quant") or "none")
    key = (
        req["model"],
        precision,
        device,
        req.get("transformer_path",    "") or "",
        req.get("vae_path",            "") or "",
        req.get("text_encoder_path",   "") or "",
        req.get("text_encoder_2_path", "") or "",
        bool(req.get("vae_from_transformer")),
        bool(req.get("offload_vae")),
        bool(req.get("attention_slicing")),
        bool(req.get("sequential_offload")),
        # vae_tiling in the key so a client toggling the flag mid-session
        # invalidates the cached pipeline. Non-string values are rejected at
        # the IPC boundary by _RUNTIME_KIND ("vae_tiling": _KIND_STR); empty
        # / None collapse to "auto".
        req.get("vae_tiling") or "auto",
        quant,
        tuple(sorted(req.get("quant_skip") or ())),
        tuple(sorted(req.get("quant_only") or ())),
        # refiner_path trailing entry per ADR-016 §(i). Whitespace-only
        # normalizes to "" (matches _maybe_load_refiner's .strip()), so
        # toggling refiner on/off flips the key and evicts the chain.
        (req.get("refiner_path") or "").strip(),
    )
    if quant != "none":
        key += (tuple(sorted(
            (l["path"], float(l.get("weight", 1.0)))
            for l in (req.get("loras") or [])
        )),)
    return key


def _maybe_load_refiner(
    req: dict,
    base_pipe,
    model_family: str,
    req_precision: str,
    req_device: str,
):
    """Load the Hunyuan-Image refiner pipeline when the request opts in.

    Returns the loaded refiner pipeline, or None when the request did not
    set ``refiner_path``. Raises on (a) refiner_path set on a non-hunyuan
    family (no silent fallback; the opt-in signal was explicit), or (b) any
    error in the underlying HunyuanImageRefinerPipeline construction (wrong
    _class_name, missing weights, OOM during load, etc.).

    Caller passes the BASE pipeline so the asymmetric shared-encoder
    optimization per ADR-016 §(e) can inject base_pipe.text_encoder +
    tokenizer into the refiner construction.

    Wire-field-name note: the canonical key is ``refiner_path`` (matches the
    schema, ``_delegate_to_server`` output, and the
    transformer_path/vae_path/text_encoder_path convention).
    """
    refiner_path = (req.get("refiner_path") or "").strip()
    if not refiner_path:
        return None
    if model_family != "hunyuan-image":
        raise ValueError(
            f"refiner_path is only supported for the hunyuan-image family; "
            f"--model resolved to family {model_family!r}. Drop refiner_path "
            f"or point --model at a HunyuanImage-2.1-Diffusers checkpoint."
        )
    from comfyless.hunyuan_chain import load_refiner_pipeline
    return load_refiner_pipeline(
        refiner_path, base_pipe=base_pipe,
        precision=req_precision, device=req_device,
        vae_tiling=req.get("vae_tiling") or "auto",
        allow_hf_download=False,
        # Quantize the refiner to match the base (the request's quant triple is
        # already validated + drives the base load). Values are the same
        # SCHEMA_KIND-typed fields the base uses.
        quant=str(req.get("quant") or "none"),
        quant_skip=tuple(req.get("quant_skip") or ()),
        quant_only=tuple(req.get("quant_only") or ()),
    )


def _evict_chain(server_state: dict) -> None:
    """Drop both cached pipelines and clear remaining server_state.

    Drop ``refiner_pipeline`` FIRST so any Python reference cycle or
    partial-setup state on the chain releases before the base pipeline
    eviction triggers CUDA frees. ``server_state.clear()`` at the end is a
    belt-and-suspenders reset for non-pipeline keys (``cache_key``,
    ``model_family``, ``loaded_loras``, etc.).
    """
    if server_state.get("refiner_pipeline") is not None:
        del server_state["refiner_pipeline"]
    if "pipeline" in server_state:
        del server_state["pipeline"]
    import torch
    torch.cuda.empty_cache()
    server_state.clear()


def _handle_generate(
    req: dict,
    output_dir: str,
    model_base: str,
    device: str,
    precision: str,
    server_state: dict,
) -> dict:
    """Execute a validated generate request with model caching and incremental LoRA diff.

    server_state keys (mutated here):
        pipeline, model_family, guidance_embeds, cache_key, loaded_loras
    """
    # Local imports — avoids circular dependency at module level (generate.py
    # will import server.socket_path; server imports generate.* only inside here).
    from .generate import (
        _load_pipeline, _expand_savepath_template, _resolve_savepath, generate,
        KREA_REBALANCE_DEFAULT_MULT,
    )
    from nodes.eric_qwen_edit_lora import load_lora_with_key_fix

    req_precision = req.get("precision") or precision

    # This daemon owns exactly one GPU (ADR-020, design A): the launch --device.
    # The request payload's `device` is IGNORED — honoring it would let a daemon
    # pinned to cuda:N run on another GPU that belongs to a different daemon,
    # re-introducing the cross-GPU eviction thrash ADR-020 exists to remove.
    # Closes security review Finding 2 (review-parallel-daemon-2026-07-03). A
    # correctly-routed client already sends its own device; warn (don't silently
    # redirect) only when a mis-routed/stale caller asks for a different one.
    req_device = device
    _payload_device = req.get("device")
    if _payload_device:
        try:
            _mismatch = _device_socket_slug(_payload_device) != _device_socket_slug(device)
        except (ValueError, TypeError):
            # ValueError: unparseable string. TypeError: non-string payload
            # device (e.g. 123, ["cuda:0"]) — `device` is an unknown key at the
            # boundary validator and passes through un-type-checked, so this
            # advisory compare must not let a malformed value crash the accept
            # loop (matches the daemon's "malformed request never kills me"
            # invariant). Either way: treat as a mismatch, warn, and ignore it.
            _mismatch = True
        if _mismatch:
            _log(f"[server] request device {_payload_device!r} ignored; this "
                 f"daemon is pinned to {device!r}")

    req_quant      = str(req.get("quant") or "none")
    req_quant_skip = tuple(req.get("quant_skip") or ())
    req_quant_only = tuple(req.get("quant_only") or ())

    # Cache key covers everything that affects pipeline shape; for unquantized
    # requests LoRAs are tracked separately so they can be diffed
    # incrementally, while quant requests key on the LoRA set too (full evict
    # on any change — see _request_cache_key).
    cache_key = _request_cache_key(req, req_precision, req_device)

    # ── Evict on config change ────────────────────────────────────────
    if server_state.get("cache_key") != cache_key and "pipeline" in server_state:
        _log("Model config changed — evicting cached refiner + pipeline")
        _evict_chain(server_state)

    # ── Load if not cached ────────────────────────────────────────────
    if "pipeline" not in server_state:
        try:
            pipe, model_family, guidance_embeds = _load_pipeline(
                req["model"],
                precision=req_precision,
                device=req_device,
                offload_vae=bool(req.get("offload_vae")),
                transformer_path=req.get("transformer_path",    "") or "",
                vae_path=req.get("vae_path",            "") or "",
                text_encoder_path=req.get("text_encoder_path",   "") or "",
                text_encoder_2_path=req.get("text_encoder_2_path", "") or "",
                vae_from_transformer=bool(req.get("vae_from_transformer")),
                attention_slicing=bool(req.get("attention_slicing")),
                sequential_offload=bool(req.get("sequential_offload")),
                vae_tiling=req.get("vae_tiling") or "auto",
                quant=req_quant,
                quant_skip=req_quant_skip,
                quant_only=req_quant_only,
            )
        except Exception as e:
            return {"status": "error", "error_type": "LoadError", "error": str(e)}
        # Refiner load AFTER base — pipe must be loaded for the shared
        # text_encoder injection (ADR-016 §e). Failure here means the base is
        # already loaded but the chain promise can't be honored; roll back to
        # avoid a half-cached state where cache_key includes refiner_path but
        # server_state.refiner_pipeline is None.
        try:
            refiner_pipe = _maybe_load_refiner(
                req, pipe, model_family, req_precision, req_device,
            )
        except Exception as e:
            del pipe
            import torch
            torch.cuda.empty_cache()
            err_type = "RefinerLoadError" if isinstance(e, ValueError) else "LoadError"
            return {"status": "error", "error_type": err_type, "error": str(e)}
        server_state.update({
            "pipeline":         pipe,
            "model_family":     model_family,
            "guidance_embeds":  guidance_embeds,
            "cache_key":        cache_key,
            "loaded_loras":     [],  # list of {"path", "weight", "adapter_name"}
            "refiner_pipeline": refiner_pipe,
        })

    pipe         = server_state["pipeline"]
    loaded_loras = server_state["loaded_loras"]

    # ── LoRA diff ─────────────────────────────────────────────────────
    requested_loras = req.get("loras") or []
    requested_paths = {l["path"] for l in requested_loras}

    # Remove dropped LoRAs; on failure evict and reload pipeline from scratch.
    to_remove = [l for l in loaded_loras if l["path"] not in requested_paths]
    for lora_rec in to_remove:
        try:
            pipe.delete_adapters([lora_rec["adapter_name"]])
            loaded_loras.remove(lora_rec)
            _log(f"[server] LoRA removed: {lora_rec['path']}")
        except Exception as e:
            _log(f"[server] LoRA removal failed ({e}) — evicting refiner + pipeline and reloading")
            _evict_chain(server_state)
            try:
                pipe, model_family, guidance_embeds = _load_pipeline(
                    req["model"],
                    precision=req_precision,
                    device=req_device,
                    offload_vae=bool(req.get("offload_vae")),
                    transformer_path=req.get("transformer_path",    "") or "",
                    vae_path=req.get("vae_path",            "") or "",
                    text_encoder_path=req.get("text_encoder_path",   "") or "",
                    text_encoder_2_path=req.get("text_encoder_2_path", "") or "",
                    vae_from_transformer=bool(req.get("vae_from_transformer")),
                    attention_slicing=bool(req.get("attention_slicing")),
                    sequential_offload=bool(req.get("sequential_offload")),
                    vae_tiling=req.get("vae_tiling") or "auto",
                    quant=req_quant,
                    quant_skip=req_quant_skip,
                    quant_only=req_quant_only,
                )
            except Exception as e2:
                return {"status": "error", "error_type": "LoadError", "error": str(e2)}
            # Re-load refiner if the request set refiner_path. Same rollback
            # policy as the initial-load path above.
            try:
                refiner_pipe = _maybe_load_refiner(
                    req, pipe, model_family, req_precision, req_device,
                )
            except Exception as e3:
                del pipe
                import torch
                torch.cuda.empty_cache()
                err_type = "RefinerLoadError" if isinstance(e3, ValueError) else "LoadError"
                return {"status": "error", "error_type": err_type, "error": str(e3)}
            server_state.update({
                "pipeline":         pipe,
                "model_family":     model_family,
                "guidance_embeds":  guidance_embeds,
                "cache_key":        cache_key,
                "loaded_loras":     [],
                "refiner_pipeline": refiner_pipe,
            })
            loaded_loras = server_state["loaded_loras"]
            break  # all prior LoRAs are gone; add everything fresh below

    # Add LoRAs not yet applied; update weights on already-loaded ones
    # (slice DLW — the old diff keyed on path alone, so a weight-only
    # change was silently ignored on the unquantized incremental path).
    lora_warnings: list = []
    loaded_by_path = {l["path"]: l for l in loaded_loras}
    for lora_spec in requested_loras:
        lora_path    = lora_spec["path"]
        lora_weight  = float(lora_spec.get("weight", 1.0))
        rec = loaded_by_path.get(lora_path)
        if rec is not None:
            if abs(rec["weight"] - lora_weight) <= 1e-9:
                continue
            from nodes.eric_qwen_edit_lora import is_direct_merge_adapter
            if is_direct_merge_adapter(pipe, rec["adapter_name"]):
                # Direct-merge adapters bake weight at merge time — a new
                # weight CANNOT take effect without a reload. Loud, never
                # silent (invariant 3); the old weight keeps serving.
                msg = (f"LoRA weight change ignored for direct-merge "
                       f"adapter (loaded at {rec['weight']}, requested "
                       f"{lora_weight}) — reload the daemon or change "
                       f"another parameter to force a fresh load: "
                       f"{lora_path}")
                _log(f"[server] WARNING: {msg}")
                lora_warnings.append(msg)
            else:
                rec["weight"] = lora_weight
                _log(f"[server] LoRA weight updated: {lora_path} -> "
                     f"{lora_weight}")
            continue
        adapter_name = sanitize_adapter_name(Path(lora_path).stem)
        try:
            success = load_lora_with_key_fix(
                pipe, lora_path, adapter_name,
                log_prefix="[comfyless-server]",
                weight=lora_weight,
            )
            if success:
                loaded_loras.append({"path": lora_path, "weight": lora_weight,
                                     "adapter_name": adapter_name})
                # Register immediately so a duplicate path later in THIS
                # request takes the weight-update branch (last wins) instead
                # of double-loading under the same adapter name (review
                # finding 1 — a direct-merge second load would merge twice
                # into the served pipeline).
                loaded_by_path[lora_path] = loaded_loras[-1]
                _log(f"[server] LoRA loaded: {lora_path}")
            else:
                msg = f"LoRA skipped (0 modules): {lora_path}"
                _log(f"[server] WARNING: {msg}")
                lora_warnings.append(msg)
        except Exception as e:
            msg = f"LoRA load failed: {lora_path}: {e}"
            _log(f"[server] WARNING: {msg}")
            lora_warnings.append(msg)

    # Cumulative weight application over the FULL active set (slice DLW —
    # the daemon previously never called set_adapters, so every PEFT
    # adapter served at full trained strength regardless of the requested
    # weight; the same tier-1 gap fixed on the CLI path in 1f52672).
    # Covers newly-added, weight-updated, AND surviving adapters after a
    # removal. Idempotent across requests.
    if loaded_loras:
        from .generate import apply_adapter_weights
        _w = apply_adapter_weights(
            pipe, [(l["adapter_name"], l["weight"]) for l in loaded_loras])
        if _w:
            lora_warnings.append(_w)

    # ── ADR-030: upscale VAE cache ────────────────────────────────────
    # Cached INDEPENDENTLY of the pipeline (its own key, not in
    # _request_cache_key) so switching the upscale VAE never evicts the
    # 20B pipeline. Kept on CPU between requests; the decode helper moves
    # it to GPU per-call and offloads it back. allow_download=False — the
    # daemon never fetches from the network (local_files_only posture).
    # Placed BEFORE the output reservation so a load failure returns cleanly
    # without orphaning a reserved 0-byte PNG (code-review, ADR-020 parity).
    up_path = req.get("upscale_vae_path", "") or ""
    up_sub  = req.get("upscale_vae_subfolder", "") or ""
    up_key  = (up_path, up_sub, req_precision)
    if not up_path:
        server_state.pop("upscale_vae", None)
        server_state.pop("upscale_vae_key", None)
    elif server_state.get("upscale_vae_key") != up_key:
        from .generate import _load_upscale_vae
        try:
            server_state["upscale_vae"] = _load_upscale_vae(
                up_path, up_sub, req_precision, allow_download=False)
            server_state["upscale_vae_key"] = up_key
            _log(f"[server] Upscale VAE cached ({up_path!r})")
        except Exception as e:
            return {"status": "error", "error_type": "LoadError",
                    "error": f"upscale VAE load failed: {e}"}

    # ── Resolve output path (server owns this; client template is just a hint) ──
    # _reserved holds a 0-byte placeholder path when the auto-numbered branch
    # atomically claims a name (Finding 1); it is unlinked if generation fails so
    # a failed run does not leave an orphan file that also burns a counter slot.
    _reserved: Optional[str] = None
    _reserved_sidecar: Optional[str] = None
    # Resolve output format (ADR-034). Values were value-checked in
    # _validate_request; no --output path here (server owns the name), so no
    # contradiction is possible. The extension is an enum-derived constant
    # (".png"/".jpg"), never a caller string — the reservation/containment
    # below therefore keep operating on a controlled suffix.
    try:
        out_fmt = resolve_output_format(req.get("output_format") or None,
                                        req.get("quality"), None)
    except ValueError as e:
        return {"status": "error", "error_type": "ValidationError", "error": str(e)}
    savepath = req.get("savepath")
    if savepath:
        # Strip leading slashes so template can't escape output_dir.
        # Subdirectory components are allowed (e.g. %date:YYYY-MM-dd%/image).
        safe_template = savepath.lstrip("/").lstrip("\\")
        full_template = str(Path(output_dir) / safe_template)
        # Validate template expands within output_dir before creating any dirs.
        _txp = req.get("transformer_path", "") or ""
        expanded = _expand_savepath_template(
            full_template, req["model"],
            req.get("seed", -1), req.get("steps", 28),
            req.get("cfg_scale", 3.5), req.get("sampler", "default"),
            transformer_path=_txp,
        )
        if not _within(str(Path(expanded).parent), output_dir):
            return {"status": "error", "error_type": "PathError",
                    "error": "savepath template expands outside --output-dir"}
        try:
            output_path = _resolve_savepath(
                full_template, req["model"],
                req.get("seed", -1), req.get("steps", 28),
                req.get("cfg_scale", 3.5), req.get("sampler", "default"),
                transformer_path=_txp,
                extension=out_fmt.extension,
            )
        except Exception as e:
            return {"status": "error", "error_type": "PathError", "error": str(e)}
    else:
        # Atomic reservation, not exists()-then-write: O_EXCL makes the create
        # fail if the name is taken, so two daemons sharing --output-dir (the
        # canonical parallel setup in ADR-020) can never both pick
        # comfyless0001.png and silently overwrite each other. Closes security
        # review Finding 1 (review-parallel-daemon-2026-07-03). The 0-byte
        # placeholder holds the name for the whole generation; generate()
        # overwrites it with the real image.
        #
        # ADR-034 slice 2: the image name is now per-EXTENSION but the sidecar
        # (comfyless{NNNN}.json, written client-side) is per-STEM. Reserving
        # only the image would let a .jpg run reuse stem 0001 while a prior
        # .png run's comfyless0001.json still exists, silently clobbering that
        # run's provenance (its only record, for jpeg). So the counter must be
        # free for BOTH the image and the sidecar stem — atomically claim the
        # .json too (security review MEDIUM, 2026-07-21).
        counter = 1
        while True:
            candidate = str(Path(output_dir) / f"comfyless{counter:04d}{out_fmt.extension}")
            sidecar_candidate = str(Path(output_dir) / f"comfyless{counter:04d}.json")
            try:
                _fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                counter += 1
                continue
            except OSError as e:
                # Any non-EEXIST error (disk full, EACCES, read-only fs) must
                # return a structured error, never escape and kill the accept
                # loop: os.path.exists() never raised, so the atomic-open path
                # has to preserve the daemon-survival promise. (code-review,
                # slice 3 — same class as the slice-2 TypeError regression.)
                return {"status": "error", "error_type": "IOError", "error": str(e)}
            # Sidecar stem must also be free (a sibling-extension run may hold
            # it). Release the image placeholder and advance if it is taken.
            try:
                _sfd = os.open(sidecar_candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                os.close(_fd)
                os.unlink(candidate)
                counter += 1
                continue
            except OSError as e:
                os.close(_fd)
                os.unlink(candidate)
                return {"status": "error", "error_type": "IOError", "error": str(e)}
            os.close(_fd)
            os.close(_sfd)
            output_path = candidate
            _reserved = candidate
            _reserved_sidecar = sidecar_candidate
            break

    # Belt-and-suspenders: re-verify the final resolved path.
    if not _within(output_path, output_dir):
        return {"status": "error", "error_type": "PathError",
                "error": f"Resolved output path escaped output_dir: {output_path!r}"}

    # ── Generate ──────────────────────────────────────────────────────
    # The cached dict carries both pipelines forward into generate()'s refiner
    # gate; when refiner_path is non-empty AND refiner_pipeline is non-None,
    # generate() reuses the cached refiner instead of re-loading.
    cached = {
        "pipeline":         pipe,
        "model_family":     server_state["model_family"],
        "guidance_embeds":  server_state["guidance_embeds"],
        "refiner_pipeline": server_state.get("refiner_pipeline"),
        "upscale_vae":      server_state.get("upscale_vae"),
    }
    try:
        metadata = generate(
            model_path=req["model"],
            prompt=req["prompt"],
            output_path=output_path,
            negative_prompt=req.get("negative_prompt", ""),
            seed=req.get("seed", -1),
            steps=req.get("steps", 28),
            cfg_scale=req.get("cfg_scale", 3.5),
            true_cfg_scale=req.get("true_cfg_scale"),
            width=req.get("width", 1024),
            height=req.get("height", 1024),
            sampler=req.get("sampler", "default"),
            schedule=req.get("schedule", "linear"),
            loras=requested_loras,
            max_sequence_length=req.get("max_sequence_length", 512),
            precision=req_precision,
            device=req_device,
            offload_vae=bool(req.get("offload_vae")),
            attention_slicing=bool(req.get("attention_slicing")),
            sequential_offload=bool(req.get("sequential_offload")),
            rebalance=bool(req.get("rebalance")),
            rebalance_mult=req.get("rebalance_mult", KREA_REBALANCE_DEFAULT_MULT),
            rebalance_weights=req.get("rebalance_weights"),
            transformer_path=req.get("transformer_path",    "") or "",
            vae_path=req.get("vae_path",            "") or "",
            upscale_vae_path=req.get("upscale_vae_path", "") or "",
            upscale_vae_subfolder=req.get("upscale_vae_subfolder", "") or "",
            text_encoder_path=req.get("text_encoder_path",   "") or "",
            text_encoder_2_path=req.get("text_encoder_2_path", "") or "",
            vae_from_transformer=bool(req.get("vae_from_transformer")),
            quant=req_quant,
            quant_skip=req_quant_skip,
            quant_only=req_quant_only,
            nag_scale=req.get("nag_scale", 0.0),
            nag_tau=req.get("nag_tau", 2.5),
            nag_alpha=req.get("nag_alpha", 0.25),
            nag_end=req.get("nag_end", 1.0),
            # Refiner thread-through per ADR-016 §(i). The refiner_path match
            # between request and cached refiner is enforced by cache_key — a
            # mismatch evicted + reloaded the chain above. generate() reuses
            # cached["refiner_pipeline"] rather than re-loading.
            refiner_path=req.get("refiner_path", "") or "",
            refiner_steps=req.get("refiner_steps", 4),
            refiner_cfg=req.get("refiner_cfg", 3.5),
            # ADR-034: output format is resolved server-side from the wire
            # fields; deliberately NOT in _request_cache_key (output concern,
            # not pipeline shape — same rationale as NAG).
            output_format=out_fmt,
            # ADR-035 slice 4: reference-image edit over the wire. ref_images
            # entries are already shape/mode/NUL-validated and contained to
            # ref_image_roots above; generate() decodes each through the single
            # decode site (load_ref_image_capped) IN THIS daemon process (6b).
            # ref_dims_explicit is the client's "user set both width+height"
            # signal. ref_drop_strict defaults to True on the wire (absent =
            # strict, decision 2 / Finding 4): a machine client whose family
            # cannot consume a reference gets a hard InferenceError, never a
            # silent drop; only the interactive CLI sends False.
            ref_images=req.get("ref_images") or [],
            ref_dims_explicit=bool(req.get("ref_dims_explicit")),
            ref_drop_strict=bool(req.get("ref_drop_strict", True)),
            _cached_pipeline=cached,
            # Explicit pause opt-out (slice PAUSE, 2026-07-17): the daemon
            # runs generation on its MAIN thread, usually in a foreground
            # terminal (TTY stdin), so sigint_pause's implicit guards don't
            # fire — a stray ^C here would block the daemon on input()
            # mid-generation and wedge every client.
            interactive_pause=False,
        )
    except Exception as e:
        import traceback
        # No image was written — drop the reserved 0-byte placeholders (image +
        # sidecar stem) so a failed run neither litters output_dir nor
        # permanently consumes its counter.
        for _p in (_reserved, _reserved_sidecar):
            if _p is not None:
                try:
                    os.unlink(_p)
                except OSError:
                    pass
        return {
            "status":     "error",
            "error_type": "InferenceError",
            "error":      str(e),
            "traceback":  traceback.format_exc(),
        }

    if lora_warnings:
        metadata.setdefault("lora_warnings", []).extend(lora_warnings)
    return {"status": "ok", "output_path": output_path, "metadata": metadata}


# ════════════════════════════════════════════════════════════════════════
#  Server entry point
# ════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    print(f"[comfyless-server] {msg}", file=sys.stderr, flush=True)


def _resolve_ref_roots(output_dir: str, ref_roots: Sequence[str]) -> Tuple[str, ...]:
    """Build ref_image_roots = {output_dir} ∪ validated --ref-root additions
    (ADR-035 6a). `output_dir` is assumed already realpath'd by the caller.

    Each --ref-root is NUL-checked, realpath'd, and required to be an existing
    directory (fail-closed FileNotFoundError otherwise). A broad root — `/`, a
    mount root, or the user's home dir — emits a loud breadth warning and
    proceeds (re-review Finding 6, warn-don't-block: the root is operator-typed,
    not attacker-reachable). Extracted from run_server so the warning + fail-
    closed validation are unit-testable without binding a socket."""
    roots: Tuple[str, ...] = (output_dir,)
    home = os.path.realpath(os.path.expanduser("~"))
    for p in ref_roots:
        if "\x00" in p:
            raise FileNotFoundError("--ref-root contains embedded NUL byte")
        root_real = os.path.realpath(p)
        if not os.path.isdir(root_real):
            raise FileNotFoundError(f"--ref-root not found: {p}")
        if root_real == os.sep or root_real == home or os.path.ismount(root_real):
            _log(f"WARNING: --ref-root {root_real!r} is extremely broad — every "
                 f"user-readable image under it becomes readable and "
                 f"VAE-encodable by any same-UID client for this daemon's "
                 f"lifetime. Narrow it to the keyframe/working directory.")
        roots += (root_real,)
    return roots


def run_server(
    output_dir: str,
    model_base: str,
    device: str = "cuda",
    precision: str = "bf16",
    lora_paths: Tuple[str, ...] = (),
    transformer_paths: Tuple[str, ...] = (),
    ref_roots: Tuple[str, ...] = (),
) -> None:
    """Start the comfyless model server and block until --unload is received.

    `lora_paths` / `transformer_paths` (ADR-018): additional spawn-time
    allowlist roots; request paths pass `_check_paths` when within ANY of
    {model_base} ∪ these roots. Validated fail-closed here exactly like
    model_base.

    `ref_roots` (ADR-035 6a): the --ref-root spawn additions to
    `ref_image_roots` = {output_dir} ∪ these. This is a DISJOINT allowlist from
    the weight roots above — reference images are read (and VAE-encoded), never
    deserialized as weights, so they get their own root set (never merged).
    """
    output_dir = os.path.realpath(output_dir)
    model_base = os.path.realpath(model_base)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(model_base):
        raise FileNotFoundError(f"--model-base not found: {model_base}")

    extra_roots: Tuple[str, ...] = ()
    for flag, paths in (("--lora-path", lora_paths),
                        ("--transformer-path", transformer_paths)):
        for p in paths:
            # Security-audit F-1 (2026-07-05): explicit NUL pre-check before
            # realpath, mirroring the MCP startup path — a NUL would raise a
            # bare ValueError from realpath instead of a clean error.
            if "\x00" in p:
                raise FileNotFoundError(f"{flag} contains embedded NUL byte")
            root_real = os.path.realpath(p)
            if not os.path.isdir(root_real):
                raise FileNotFoundError(f"{flag} not found: {p}")
            extra_roots += (root_real,)

    # ref_image_roots = {output_dir} ∪ --ref-root additions (ADR-035 6a). The
    # output dir is a default ref root (keyframes are prior segment outputs).
    # Validation + breadth warning live in the extracted helper (unit-tested).
    ref_image_roots = _resolve_ref_roots(output_dir, ref_roots)

    sock_path = socket_path(device)
    if sock_path.exists():
        sock_path.unlink()

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(4)
    os.chmod(str(sock_path), 0o600)  # belt-and-suspenders; dir is already 0700

    _log(f"Listening on {sock_path}")
    _log(f"output-dir : {output_dir}")
    _log(f"model-base : {model_base}")
    for r in extra_roots:
        _log(f"extra-root : {r}")
    for r in ref_image_roots:
        _log(f"ref-root   : {r}")
    _log(f"device     : {device} / {precision}")

    server_state: dict = {}
    keep_running = True
    try:
        while keep_running:
            conn, _ = srv.accept()
            with conn:
                keep_running = _handle_connection(
                    conn, output_dir, model_base, device, precision,
                    server_state, extra_roots, ref_image_roots,
                )
    finally:
        srv.close()
        if sock_path.exists():
            sock_path.unlink()
        _log("Stopped.")
