# ADR-001: Comfyless Daemon — Socket Location and Path Trust Model

**Date:** 2026-04-21
**Status:** accepted
**Context:** comfyless daemon (not yet implemented as of this writing)

---

## Context

The comfyless CLI loads 20B-parameter models on every invocation, taking 30–90s per run.
A persistent daemon that keeps the model in memory between invocations would eliminate
that overhead. The natural IPC mechanism for a local daemon is a Unix domain socket.

A security review was conducted before implementation. The review identified two CRITICAL
and two HIGH findings in the naive design (socket at `/tmp/comfyless.sock`, client
supplies output path and model path freely).

---

## Decision

### 1 — Socket location: `$XDG_RUNTIME_DIR/comfyless.sock`

`$XDG_RUNTIME_DIR` is provisioned per-UID by systemd at mode `0700`. No other user can
read, write, or enumerate files in it. This eliminates the world-writable `/tmp` socket
that would allow any local user to connect, intercept requests, or replace the socket
with their own listener.

Fallback when `$XDG_RUNTIME_DIR` is not set (non-systemd systems): create
`/tmp/comfyless-$(id -u)/` with `os.makedirs(mode=0o700, exist_ok=True)` and place the
socket there. The directory's restrictive permissions are the control — the socket file's
own permissions are insufficient on their own.

### 2 — Server owns path resolution; client never supplies absolute output paths

The daemon is launched with a configured `--output-dir` (required) and `--model-base`
(required). The client sends generation parameters. The server resolves all output paths
internally using its savepath template logic. The client receives back the resolved path
in the response — it does not dictate it.

This eliminates the arbitrary-file-write-as-server-UID primitive: a client cannot cause
the server to write to `~/.ssh/authorized_keys` by crafting an output path.

### 3 — Model and LoRA paths validated against allowlisted base directories

Before loading any model or LoRA, the server canonicalizes the supplied path with
`os.path.realpath()` and asserts it begins with one of the configured base directories
(`--model-base`). Requests with paths outside those directories are rejected with an
error response — no load is attempted.

This constrains pickle-deserialization code-execution risk to files the operator has
already deliberately placed in the model directory.

### 4 — LoRA adapter names sanitized before use

Adapter names (derived from the LoRA filename stem) are sanitized to
`[a-zA-Z0-9_-]+` before being passed to `pipe.load_lora_weights()`,
`pipe.set_adapters()`, or `pipe.delete_adapters()`. Any character outside that set is
replaced with `_`. This is a defense-in-depth measure against future diffusers internals
that might pass adapter names to subprocess calls.

### 5 — Schema validation at the socket boundary

The server validates incoming JSON against an explicit required-field + type schema
before any parameter is used. Unknown fields are ignored. Missing required fields or
wrong types return a structured error response. No request parameter reaches model-
loading code before passing validation.

### 6 — Explicit server start; no autostart

The server is started explicitly by the user:

```bash
python -m comfyless.generate --serve --device cuda:1 --output-dir ~/gen-output \
    --model-base /path/to/models &
```

Normal `comfyless.generate` invocations auto-detect the socket. If no socket is found,
they fall back to in-process model loading (current behavior). There is no implicit
daemonize — the user controls when the server starts and stops.

---

## Alternatives Rejected

**Socket at `/tmp/comfyless.sock`** — rejected: world-readable/writable on shared
machines; allows socket hijacking, request interception, and unauthenticated access
by any local user.

**Client supplies output path** — rejected: creates an arbitrary-file-write primitive
with server-UID privilege. No authentication system is sufficient to make this safe
without also bounding the write scope.

**TCP localhost socket** — rejected: unnecessary; Unix sockets are faster, simpler,
and localhost TCP is accessible to any process on the machine, not just local users
(relevant if the machine runs Docker, VMs, etc. with shared network namespaces).

---

## Deferred / Out of Scope

**`SO_PEERCRED` check on `--unload`** — any local user can send a shutdown request.
On a single-user workstation this is a minor nuisance. Implement if the tool is ever
deployed on shared infrastructure. Tracked in `TECH_DEBT.md`.

**Per-request inference timeout** — a hung generation blocks all subsequent clients.
A configurable server-side timeout (abort + error response after N seconds) is the
right fix. Deferred to implementation phase. Tracked in `TECH_DEBT.md`.

**Request rate limiting / VRAM exhaustion bounds** — a client can force rapid model
reloads. Reasonable bounds on max steps / dimensions and a minimum time between model
swaps would help. Deferred. Tracked in `TECH_DEBT.md`.

---

## Changelog

- 2026-04-21: Initial ADR; security review conducted before any code written.
  See `docs/security/review-daemon-socket-2026-04-21.md`.
- 2026-04-24: Timeout / BrokenPipe fix — `_recv` timeout parametrized into
  server-default (5s, unchanged DoS guard) vs client-override
  (`_CLIENT_RECV_TIMEOUT_SEC = 600s`) because generation legitimately takes
  30–120s and the shared 5s deadline always tripped on the client response
  path. Added `_send_safe` wrapper so a client that disconnects mid-request
  never kills the daemon. Added server-side audit log for PathError
  rejections. Quantitative DoS window from a same-uid misbehaving client
  grew from "immediate daemon death" to "up to 600s per connection × listen
  backlog of 4" — strictly less bad than daemon death, but more silent.
  Per-request inference timeout (Deferred #2 above) and rate-limiting
  (Deferred #3) remain open. See
  `docs/security/review-server-timeout-brokenpipe-2026-04-24.md`.

## AI-Disclosure

ADR authored by Claude Sonnet 4.6, 2026-04-21. Design decisions made collaboratively
with Grant Kahn during security review. Reviewed by Grant Kahn.
