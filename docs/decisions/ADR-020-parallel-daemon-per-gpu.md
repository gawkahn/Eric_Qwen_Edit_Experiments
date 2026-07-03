# ADR-020: Parallel comfyless Daemon — One per GPU via Device-Keyed Sockets

**Date:** 2026-07-03
**Status:** accepted
**Builds on:** [ADR-001](ADR-001-daemon-socket-security.md) (daemon socket location + path trust model)

---

## Context

The comfyless daemon (`comfyless/server.py`) keeps one diffusers pipeline resident to
avoid the 30–90s per-invocation model load. Its structure today:

- **One fixed socket per UID** — `socket_path()` returns `<socket-dir>/comfyless.sock`
  with no parameterization (`server.py:86-88`).
- **One `server_state` dict** holding a single pipeline + `cache_key` + `loaded_loras`
  (`server.py:595`, `319-390`).
- **A single-threaded accept loop** — `srv.accept()` → `_handle_connection` runs the
  entire generation to completion before looping back to `accept()` (`server.py:598-603`).
- **Device is part of the cache key** (`server.py:351`), so a `cuda:1` request arriving
  after a `cuda:0` request **evicts and reloads** the pipeline.

Consequence: when two generation sets are aimed at the daemon (one per GPU on a
two-Blackwell box), it does not crash — it **serializes and thrashes**, alternating the
single cached pipeline between devices. We want two independent generations, one on
GPU0 and one on GPU1, running **concurrently**.

Two candidate designs were weighed:

- **A — one daemon per GPU.** Spawn a separate daemon process per device, each pinned
  via `--device cuda:N`, and route each client to the daemon for its device. Each daemon
  keeps its own single-pipeline cache. Matches how `start-mcpo.sh` already pins one GPU
  per process.
- **B — sessions inside one daemon.** Per-device pipeline slots keyed by device,
  concurrency within one process (threads or a pool), one socket endpoint.

Reading the code decided it. Under A, the entire `_handle_generate` machinery —
`server_state`, cache key, LoRA diff, path enforcement, schema validation — stays
**byte-for-byte identical**; the only new surface is making the socket name
device-aware and routing the client. Under B, the single-threaded accept loop must
become concurrent, which pulls in thread-safety around `server_state` **and** the
**module-level caches in `generate.py` and the loaders** (which are not thread-safe),
plus two live CUDA pipelines under one GIL — real risk on a §12 IPC surface for an
isolation benefit that A already provides for free via OS process separation.

This is a §12 IPC-surface change. ADR-001 established the socket trust model; this ADR
extends it to multiple sockets without weakening any of its controls.

---

## Decision

### 1 — One daemon per GPU (design A)

Each GPU is served by its own daemon process, launched explicitly and pinned to a
device:

```bash
python -m comfyless.generate --serve --device cuda:0 --output-dir ~/gen-output \
    --model-base /path/to/models &
python -m comfyless.generate --serve --device cuda:1 --output-dir ~/gen-output \
    --model-base /path/to/models &
```

Each daemon is exactly today's daemon: one process, one `server_state`, one pipeline,
single-threaded. Two GPUs → two processes → true concurrency with **no shared mutable
state** between them. This preserves the single-writer-per-process invariant (relevant
given the mergerfs/fcntl caveat: daemon state must stay single-writer).

Design B (in-daemon threading/sessions) is **rejected**, not deferred — see below.

### 2 — Device-keyed socket names in the same `0700` directory

`socket_path(device)` becomes device-parameterized. It normalizes the device string to
a canonical form and derives a distinct socket filename **inside the same per-UID
socket directory** ADR-001 §1 established (`$XDG_RUNTIME_DIR` at `0700`, or
`/tmp/comfyless-$UID/` at `0700`):

- `cuda`   → canonical `cuda:0` → `comfyless-cuda0.sock`
- `cuda:N` → `comfyless-cudaN.sock`
- `cpu`    → `comfyless-cpu.sock`

`cuda` and `cuda:0` name the **same physical device**, so they must resolve to the
**same** socket — otherwise a client using the bare `cuda` default would miss a daemon
started as `cuda:0`. Normalization guarantees this.

Because every socket still lives in the ADR-001 `0700` directory at socket mode `0600`,
**the entire ADR-001 trust model carries over unchanged**: no other user can enumerate,
connect to, or replace any of the sockets. Adding more socket *names* in a directory
that is already access-controlled adds no new exposure.

### 3 — Device string is whitelisted before it becomes a filename

The device string originates from the local `--device` CLI argument (client and
daemon), not from the untrusted socket payload. But it is now interpolated into a
**filesystem path**, so it is validated defense-in-depth before use, in the ADR-001
spirit (§3–§4: canonicalize/allowlist anything that reaches the filesystem):

- Accept only `cpu`, `cuda`, or `cuda:<digits>` (regex `^(cpu|cuda(:\d+)?)$`).
- Reject everything else with a clear error — a device string can therefore never
  contribute `/`, `..`, NUL, or other path components to the socket path.

This closes the one genuinely new attack primitive the change introduces
(device-string → path injection) at its source.

### 4 — Client routes by `--device`; no autostart (ADR-001 §6 preserved)

The client resolves its target socket from `args.device` via the same
`socket_path(device)`:

- `_send_server_command`, `_delegate_to_server`, `_send_unload` all call
  `socket_path(args.device)` instead of the arg-less form.
- If that device's socket does not exist, behavior is **exactly as today**: the client
  falls through to in-process generation (`generate.py:1462-1463`, `1509-1510`). The
  client **never spawns** a daemon.

ADR-001 §6 ("Explicit server start; no autostart") is deliberately **preserved**.
Auto-spawn was considered and rejected for this slice (see Deferred) because it would
reverse an accepted security decision and introduce a spawn attack surface (env, cwd,
argv, socket-bind race) that warrants its own review.

### 5 — `--unload` is device-scoped

`--unload` resolves the socket from `args.device`, so it stops the daemon for that
device only:

- `--unload --device cuda:1` → stops the cuda:1 daemon.
- bare `--unload` (default `cuda` → `cuda:0`) → stops the cuda:0 daemon.

Stopping every daemon means unloading each device. A one-shot `--unload-all` is deferred
(see below). This matches the per-device mental model and needs no cross-socket
enumeration (which would itself be a small new surface).

---

## Alternatives Rejected

**B — sessions inside one daemon (per-device slots + concurrency in one process).**
Rejected. It requires making the accept loop concurrent and adding thread-safety around
`server_state` *and* the module-level caches in `generate.py`/the loaders (not
thread-safe today), running two CUDA pipelines under one GIL. It concentrates two GPUs'
worth of failure into one process (one crash takes down both), and it does all this to
re-derive the isolation that OS processes give A for free. No benefit over A justified
the added complexity on a §12 surface.

**Auto-spawn a per-GPU daemon on first request.** Rejected for this slice. Convenient,
but reverses ADR-001 §6 and adds a spawn path (detached child, inherited env/cwd, argv
construction, race between bind and the client's connect) that is its own security
review. Deferred, not adopted silently.

**Keep the single `comfyless.sock` name and special-case only non-default devices.**
Rejected. Uniform device-keyed naming (`comfyless-cuda0.sock` always, including the
default) is more predictable than "the default device is magic." The cost is updating a
handful of doc/script references, done in this slice.

**A network (TCP) endpoint to multiplex devices.** Rejected for the same reasons
ADR-001 rejected TCP: Unix sockets in a `0700` dir are the access-control boundary;
TCP localhost is reachable by any process in the network namespace.

---

## Deferred / Out of Scope

- **Auto-spawn on first request** — see Alternatives. If adopted later, it needs a new
  ADR section and a `security-auditor` pass on the spawn path. Tracked in `TECH_DEBT.md`.
- **`--unload-all`** — stop every device's daemon in one call. Minor convenience;
  requires enumerating sockets in the dir. Deferred. Tracked in `TECH_DEBT.md`.
- **A launcher/helper that starts N daemons for you** — the user starts them explicitly
  this slice. A thin wrapper script is a possible follow-up.
- **Concurrency on the *same* GPU** — out of scope; a single GPU cannot run two 20B
  generations at once. One daemon per GPU correctly serializes its own device.
- **The open ADR-001 deferrals** (`SO_PEERCRED` on unload, per-request inference
  timeout, rate limiting) — unchanged by this ADR; still tracked in `TECH_DEBT.md`.
  Note that with per-device unload, a same-uid `--unload` still stops a daemon without a
  peer-cred check, exactly as before — no better, no worse.

---

## Changelog

- 2026-07-03: Initial ADR. Design A (daemon-per-GPU) chosen over B (in-daemon sessions);
  device-keyed sockets in the ADR-001 `0700` dir; device-string whitelist; no autostart
  (ADR-001 §6 preserved); device-scoped `--unload`. Security review:
  `docs/security/review-parallel-daemon-2026-07-03.md`. Vision:
  `docs/vision/slice-parallel-daemon-per-gpu.md`.
- 2026-07-03 (slice 1): device-keyed `socket_path` + client routing landed
  (`2212062`). Security review Finding 3 baked in (`_device_socket_slug`:
  `re.fullmatch(..., re.ASCII)` on the raw device string, then integer
  canonicalization). Verified by a second security-auditor pass (verdict: ship).
- 2026-07-03 (slice 2, closes Finding 2): `_handle_generate` now pins the device
  to the daemon's launch `--device` and ignores the request payload's `device`
  (was `req_device = req.get("device") or device`, now `req_device = device`). A
  daemon can no longer be induced to run on a GPU another daemon owns. A
  mis-routed/stale caller asking for a different device is warned (not silently
  redirected), per the project's warn-don't-block preference; `cuda` vs `cuda:0`
  is treated as a match and does not warn. The slice-2 security-auditor pass
  caught a regression in the new warn block (a non-string payload `device` raised
  an uncaught `TypeError` that would escape to the accept loop and kill the
  daemon); fixed by catching `(ValueError, TypeError)` so a malformed device is
  absorbed and warned, preserving the "malformed request never kills the daemon"
  invariant (`c99303b`).
- 2026-07-03 (slice 3, closes Finding 1): auto-numbered output now uses an atomic
  `os.open(O_CREAT|O_EXCL)` reservation instead of `os.path.exists()`-then-write,
  so two daemons sharing `--output-dir` cannot both select `comfyless0001.png`
  and overwrite each other. The 0-byte placeholder holds the name through
  generation (`generate()` overwrites it); on generation failure the placeholder
  is unlinked so a failed run leaves no orphan and does not burn a counter slot.
  The savepath-template branch is unchanged (user-controlled naming; out of
  Finding 1's scope).

## AI-Disclosure

ADR authored by Claude (Opus 4.8), 2026-07-03. Design decisions (A vs B, routing model,
no-autostart) made collaboratively with Grant Kahn. Reviewed by Grant Kahn.
