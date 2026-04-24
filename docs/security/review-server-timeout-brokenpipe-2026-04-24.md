# §12 Security Review — Comfyless Server Timeout / BrokenPipe Fix

**Date:** 2026-04-24
**Reviewer:** security-auditor (Claude Opus 4.7 via subagent invocation)
**Slice:** `_recv` timeout split + `_send_safe` + PathError audit logging
**AI-Disclosure:** Claude (Opus 4.7) authored; Grant reviewed.

---

## Scope

- `comfyless/server.py` — timeout constants, `_recv(conn, timeout=...)` parametrization, new `_send_safe` helper, `_send_safe` swap at all `_handle_connection` call sites, PathError audit log.
- `comfyless/generate.py` — `_send_server_command` passes `_CLIENT_RECV_TIMEOUT_SEC` to `_recv`.
- `test_server_robustness.py` — new, 8 pure-logic IPC tests via `socket.socketpair`.

Baseline: `docs/decisions/ADR-001-daemon-socket-security.md`, `docs/security/review-comfyless-server-2026-04-23.md`.

---

## Bug being closed

On 2026-04-24, `python -m comfyless.generate --iterate prompt ...` against a running `--serve` daemon reproduced this sequence:

1. Client sent a generate request over the Unix socket.
2. Server spent ~8.6s running the generation.
3. Client's `_recv(conn)` raised `TimeoutError` after 5s — because `_RECV_TIMEOUT_SEC` was a hardcoded deadline inside `_recv`, applied to *both* the server's request-read (intended DoS guard) and the client's response-read (unintended side-effect of code reuse).
4. Client disconnected.
5. Server's final `_send(conn, result)` raised `BrokenPipeError`, which propagated out of `_handle_connection` uncaught.
6. The daemon process died, dropping the loaded 20B pipeline from VRAM.

---

## Findings

### MEDIUM-1 — PathError rejections were not audit-logged *(closed by this slice)*

**Location:** `comfyless/server.py:_check_paths` / `_handle_connection` path-check branch.

Client submissions that attempted to reference paths outside `--model-base` were rejected and the error returned to the client via `_send(...)`, but no server-side log was emitted. The 2026-04-23 baseline review did not flag this. The `_send_safe` change in this slice made the gap observable: if the client has already disconnected, the outbound error frame is silently dropped and the rejection vanishes entirely.

**Closed in this slice.** Added a `_log(f"PathError: {err} req={redacted!r}")` call immediately before `_send_safe` in the PathError branch. The redacted request dict drops `prompt` to stay PII-agnostic. The log entry survives client disconnection.

### MEDIUM-2 — 600s client-side deadline widens same-uid serial-DoS window

**Location:** `comfyless/server.py:_CLIENT_RECV_TIMEOUT_SEC = 600.0`.

A same-uid misbehaving or compromised process can now hold a connection slot for up to 600s by sending a valid request and never reading the response. With `listen(4)` and single-threaded accept, five such processes block legitimate clients for up to ~40 minutes.

This is **strictly less bad than the pre-fix state** (daemon death, full pipeline reload), but is a quantitative widening of a pre-existing category. ADR-001's "Request rate limiting / VRAM exhaustion bounds" (Deferred #3) already covers this; Changelog entry in ADR-001 records the new bound. Not actionable in this slice.

### LOW / INFO — no action

- `_send_safe` log line contains only errno text (`[Errno 32] Broken pipe` or `[Errno 104] Connection reset by peer`), no payload or peer identity. Clean.
- `_send_safe` swallowing on ping/validation/unload paths does not mask state-changing security errors — no state is allocated before those sends, and the `unload` `return False` is unconditional on send success.
- New test file is not security-relevant; one NIT is that CLAUDE.md's "Test suites" block should list it. *Closed in this slice — CLAUDE.md updated.*

---

## ADR-001 invariant verification

| ADR-001 Decision | Post-fix status |
|---|---|
| Socket at `$XDG_RUNTIME_DIR/comfyless.sock` (0700 dir, 0600 file) | Unchanged |
| Server owns output path resolution | Unchanged |
| `--model-base` allowlist via `realpath` | Unchanged |
| Adapter-name sanitization | Unchanged |
| Schema validation at socket boundary | Unchanged |
| Explicit server start; no autostart | Unchanged |

---

## Conclusion

**Merge-ready.** Net security improvement vs pre-fix state: an availability-class CRITICAL-adjacent bug (daemon crash on any client disconnect) is closed, and a pre-existing audit gap (MEDIUM-1) is closed in the same slice. The quantitative DoS window widening (MEDIUM-2) is documented and tracked under ADR-001's existing deferrals.

No Red Zone scope creep. No new auth / crypto / PII / billing surface.
