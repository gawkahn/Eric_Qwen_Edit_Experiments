# Infra Review — systemd user unit `comfyless@.service` (headless daemon mode)

**Date:** 2026-07-17
**Reviewer:** Claude (Fable 5), infra-auditor pass
**Change under review:** new `systemd/comfyless@.service` template unit (repo) + installed copy at `~/.config/systemd/user/`; README "Headless daemon" section. No code changes.
**AI-Disclosure:** Review authored by Claude (Fable 5); reviewed by Grant Kahn.

> Disposition note (parent session, post-review): SHOULD 1 (start-limit) and
> SHOULD 3's cheap directives (`NoNewPrivileges=yes`,
> `RestrictAddressFamilies=AF_UNIX`, `ProtectSystem=full`) were applied to the
> unit in the same slice; `ProtectHome=read-only` deliberately skipped
> (torch/triton JIT caches write under `~/.cache`) and documented in the unit.
> SHOULD 2 (socket-liveness probe in `run_server`) is a Red Zone code change —
> recorded in `TECH_DEBT.md` for its own slice.

## Summary

The unit wraps an existing, already-reviewed local daemon (`comfyless --serve`) in a systemd **user** service, one instance per GPU. Host is a single-user AI workstation; threat model is local-shell / supply-chain (repo writers), no external attacker path. The unit adds no listener, no privilege, no new secret. Launch path verified: the only `bind()` in `comfyless/` is `AF_UNIX` (`server.py:936-939`, 0600 socket in a 0700 per-UID dir per ADR-001). Net posture change is small: it converts "code the user runs by hand" into "code that auto-starts at login and auto-restarts on failure" — a mild persistence amplifier that is largely status quo (start-mcpo.sh already does the same). Overall: acceptable with two SHOULDs.

## Coverage

Reviewed: `systemd/comfyless@.service` (full); `comfyless/server.py` (`_socket_dir`, `_device_socket_slug`, `run_server` incl. socket bind/unlink at 932-961); `comfyless/generate.py` (`_send_unload` 2259-2274, arg wiring); grep for `AF_INET`/TCP across `comfyless/`.

Not reviewed: the installed copy at `~/.config/systemd/user/` (stated identical — not diffed); user-manager `DefaultStartLimit*` overrides on this host (assumed systemd defaults: burst 5 / interval 10 s); journald persistence config; outbound HF-hub network behavior (unit sets no `HF_HUB_OFFLINE`; relied on the project's `local_files_only=True` invariant, not re-verified per call site).

## Findings

**[SHOULD 1] Restart loop never trips the default start limit** — `RestartSec=5` against the user-manager default `StartLimitBurst=5 / StartLimitIntervalSec=10s` means at most ~3 starts fit in any 10 s window — the limit mathematically never fires, so a persistent fault restarts forever. Mitigated in practice because `run_server` loads models lazily (crash-loop at idle is cheap Python startup), but a wedged GPU plus a retrying client (MCP/OWUI loop) will thrash load/crash cycles indefinitely. Remediation: `StartLimitIntervalSec=600` + `StartLimitBurst=5` in `[Unit]`. **[Applied.]**

**[SHOULD 2] Startup silently hijacks a live foreign daemon's socket; ExecStop can unload a foreign daemon** — `server.py:933-934` unlinks any existing socket with no liveness check: starting `comfyless@0` while a manually-started cuda:0 daemon runs steals the path, orphaning the manual daemon (unreachable, still holding VRAM), and the orphan's shutdown `finally` later deletes the *systemd* daemon's socket. Conversely, if a manual daemon owns the socket when `systemctl --user stop` runs, `ExecStop --unload` cleanly shuts down the foreign daemon over IPC (systemd's SIGTERM correctly stays cgroup-scoped, so it can't *kill* the wrong PID — the exposure is IPC-level only). Single-user, availability-only. Remediation is a code change outside this unit's scope: connect-probe an existing socket in `run_server` and refuse to start if it answers, instead of unlinking. **[TECH_DEBT.md; Red Zone route when picked up.]**

**[SHOULD 3] Free sandboxing is missing; most heavy directives are pointless here** — worth adding (all compatible with GPU + mergerfs): `NoNewPrivileges=yes`; `RestrictAddressFamilies=AF_UNIX` — structurally enforces the "never a TCP listener" invariant (and would loudly break any accidental hub download, which is desirable given `local_files_only`; smoke-test once); `ProtectSystem=full`; `ProtectHome=read-only` + `ReadWritePaths=` (needs cache-write enumeration first). Pointless/harmful: `PrivateDevices` (blocks `/dev/nvidia*`), `MemoryDenyWriteExecute` (breaks CUDA/triton JIT), `DynamicUser`/`ProtectHome=yes` (everything lives in `$HOME`). Honest framing: a user unit is inside the user's own privilege boundary, so this is defense-in-depth for the daemon's real attack surface (caller-supplied weight-file parsing), not a trust boundary. **[First three applied; ProtectHome skipped and documented.]**

**[ACCEPT] PYTHONPATH/WorkingDirectory into the git repo = code-exec-on-pull, and auto-start makes it persistent** — anyone who can write the repo (compromised origin, concurrent agent session) now gets code that runs at every login and after every crash, surviving logout under linger. Status quo for this tool (start-mcpo.sh, interactive use run the same HEAD as the same user); the unit only removes the "user typed the command" step. Accepted, with the observation that `Restart=on-failure` means freshly-pulled code activates without user action after any crash.

**[ACCEPT] Hardcoded absolute paths + duplicated install copy drift** — 6 `/home/gawkahn/...` literals; the `~/.config/systemd/user/` copy is a snapshot that silently diverges from the repo copy on the next edit. Smallest fixes if it ever bites: `%h` for the home prefix, and `systemctl --user link` so the repo copy is authoritative (linking slightly widens the code-exec-on-pull acceptance above). Acceptable for a solo box.

**[ACCEPT] Network exposure: none added** — verified; sole listener is `AF_UNIX` at `server.py:936`, no `AF_INET`/HTTP server anywhere in `comfyless/`. `RestrictAddressFamilies` pins this permanently.

## Verdict

**PASS with conditions.** No MUST findings. Ship after the two cheap `[Unit]`/`[Service]` additions (start-limit + `NoNewPrivileges`/`RestrictAddressFamilies=AF_UNIX`) **[done]**; queue the socket-liveness probe in `server.py` as its own slice — that file is a Red Zone path (`_red-zone-paths.sh`), so that change takes the security-auditor route, not a drive-by **[TECH_DEBT.md]**.
