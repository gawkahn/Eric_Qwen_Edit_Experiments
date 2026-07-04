# Slice — Parallel daemon, one per GPU (device-keyed sockets)

**Date:** 2026-07-03 · **Author:** Claude (Opus 4.8) · **Executor:** main session
**Implements:** [ADR-020](../decisions/ADR-020-parallel-daemon-per-gpu.md) · builds on [ADR-001](../decisions/ADR-001-daemon-socket-security.md)

---

## 0. Orientation

- Today the comfyless daemon (`comfyless/server.py`) caches a **single** in-process
  pipeline behind a **single** fixed Unix socket (`comfyless.sock`) and a
  **single-threaded** accept loop (`server.py:598-603`). Two generation sets thrown
  at it (one per GPU) are handled gracefully but **serially** — the device is part of
  the cache key (`server.py:351`), so a `cuda:1` request after a `cuda:0` request
  **evicts and reloads**, thrashing rather than running in parallel.
- **Chosen design: A — one daemon per GPU** (each pinned via `--device cuda:N`), with
  **device-keyed socket names** so two daemons coexist in the same `0700` socket dir.
  Rejected B (in-daemon threading/sessions): it would make a §12 IPC surface
  concurrent and require thread-safety around the module-level caches in
  `generate.py`/the loaders for zero isolation benefit A doesn't already give.
- **No autostart** — ADR-001 §6 is preserved. Client resolves the device-keyed socket
  from `--device`; if absent, falls through to in-process generation exactly as today.
  The user launches each per-GPU daemon explicitly.

## 1. Vision

**Outcome when done:** the user can run two daemons — `--serve --device cuda:0` and
`--serve --device cuda:1` — simultaneously, and two `comfyless.generate` clients
(one `--device cuda:0`, one `--device cuda:1`) each delegate to their own daemon and
run **truly concurrently**, one gen on each GPU, with no eviction thrash and no shared
state between them.

**Invariants (must always hold):**
1. **Single-writer per daemon** — each daemon process owns exactly one `server_state`
   and one GPU; there is **no shared mutable state across GPUs** (each is its own OS
   process, as today). (Negative test: two daemons' socket paths differ; neither can
   bind the other's.)
2. **Device-keyed socket isolation** — `socket_path(device)` maps a device string to a
   distinct socket name within the same `0700` dir; `cuda:0` and `cuda:1` never collide.
   (Negative test: `socket_path("cuda:0") != socket_path("cuda:1")`.)
3. **`cuda` and `cuda:0` are the same physical device → the same socket** — normalized
   to one canonical name so a client saying `--device cuda` reaches the `cuda:0` daemon.
   (Negative test: `socket_path("cuda") == socket_path("cuda:0")`.)
4. **Device string is whitelisted before it becomes a filename** — only `cpu`,
   `cuda`, `cuda:<digits>` are accepted; anything else is rejected, so a device string
   can never inject path components into the socket path. (Negative test:
   `socket_path("../../etc/x")` / `"cuda:0/../y"` raises, does not produce a traversing
   path.)
5. **No autostart** — a missing device socket falls through to in-process generation;
   the client never spawns a daemon. (Negative test: with no socket present,
   `_delegate_to_server` returns `None` and in-process runs. ADR-001 §6 unchanged.)
6. **ADR-001 §1–§5 trust model carried over unchanged** — device-keyed sockets live in
   the same per-UID `0700` dir at socket mode `0600`; server still owns path
   resolution, still validates against `--model-base`, still sanitizes adapter names,
   still validates schema at the boundary. Nothing in the trust model is weakened.
7. **`--unload` is device-scoped** — `--unload --device cuda:1` stops only the cuda:1
   daemon; bare `--unload` (default `cuda` → `cuda:0`) stops the cuda:0 daemon. Stopping
   both requires unloading each. (Negative test: unloading cuda:0 leaves the cuda:1
   socket present and its daemon serving.)

**Out of scope (this slice):**
- **Auto-spawn** of a per-GPU daemon on first request — explicitly deferred; it would
  reverse ADR-001 §6 and needs its own ADR section + security review of the spawn path.
- **`--unload-all`** convenience to stop every device's daemon in one call — deferred.
- **In-daemon threading / per-session slots (design B)** — rejected, not deferred.
- **A launcher that spawns N daemons for you** — the user starts them explicitly this
  slice; a helper script is a possible thin follow-up.
- **>1 gen concurrently on the *same* GPU** — one daemon per GPU serializes its own
  device, which is correct (a single GPU can't run two 20B gens at once anyway).
- **MCP/mcpo path changes** — `start-mcpo.sh` already pins one GPU per process; it needs
  only the socket-name references updated to match, no logic change.

## 2. Change boundary / edit scope

**May change:**
- `comfyless/server.py` — `socket_path()` becomes device-parameterized + a device→name
  normalizer/whitelist; `run_server` binds the device-keyed socket derived from its
  `--device`.
- `comfyless/generate.py` — `_send_server_command`, `_send_unload`, `_delegate_to_server`
  resolve the socket from `args.device`; `_send_unload` threads `args.device`.
- `comfyless/README.md` — socket-location section (currently documents the single
  `comfyless.sock`).
- `start-mcpo.sh` — only if it references the literal socket name.
- `test_server_robustness.py` (or a small new `test_socket_device_routing.py`) — unit
  tests for the normalizer/whitelist + routing.
- Docs: `docs/vision/`, `docs/decisions/ADR-020-*`, `docs/security/review-parallel-*`.

**Must NOT change (STOP and split if required):** `_handle_generate` generation/cache/
LoRA-diff logic (stays byte-identical — that's the whole point of picking A);
`resolve_hf_path`; `_run_json_mode`; the params validator.

## 3. Design (condensed — full rationale in ADR-020)

- **`socket_path(device: str = "cuda") -> Path`**: normalize `device` → canonical form
  (`cuda` → `cuda:0`; `cuda:N` kept; `cpu` kept), whitelist-reject anything else, then
  build `<dir>/comfyless-<slug>.sock` where slug replaces `:` (`cuda0`, `cuda1`, `cpu`).
- **`run_server`** already receives `device`; pass it to `socket_path(device)` when
  binding.
- **Client** (`_send_server_command`, `_send_unload`, `_delegate_to_server`) call
  `socket_path(args.device)` instead of the arg-less form.
- Everything else — accept loop, `server_state`, cache key, LoRA diff, path enforcement,
  schema validation — is untouched. Two daemons are two independent copies of today's
  daemon on two sockets.

## 4. Proof hooks

- **Unit:** `socket_path` normalization + whitelist (invariants 2–4), device-scoped
  unload path resolution (invariant 7). At least one negative case per invariant above.
- **Live (user-run, per `feedback_dont_overstep_scope`):** start two daemons on cuda:0
  and cuda:1, fire two gens concurrently, confirm both GPUs busy in `nvidia-smi` at the
  same time and neither daemon logs an eviction. I surface the exact commands; the user
  runs the GPU test personally.

## 5. Risk

**Medium.** Touches the §12 IPC socket surface (path construction from a device string
→ path-injection is the one real new risk, covered by invariant 4 + the whitelist).
No change to the generation/trust logic. Backward-compat note: the socket name changes
from `comfyless.sock` to `comfyless-cuda0.sock` for the default device — a stale client
built before this slice would not find a new daemon, but client + server ship together
in one codebase, so this only matters across a partial upgrade (called out in ADR-020).
