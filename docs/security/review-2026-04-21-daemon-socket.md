# Security Review: Comfyless Daemon — Unix Socket IPC Design

**Date:** 2026-04-21
**Reviewer:** Claude security-auditor agent
**Scope:** Design-phase review (no code written at time of review)
**Subject:** Proposed Unix domain socket client-server for comfyless persistent model cache

---

## Proposed Design (as reviewed)

- Server process starts on demand, listens on `/tmp/comfyless.sock`
- Normal CLI invocations auto-detect socket and delegate via JSON over Unix socket
- Client sends: model path, prompt, LoRA paths, output path, generation params
- Server: loads/caches model, diffs LoRA set, writes output PNG + sidecar JSON
- `--unload` flag shuts down server

---

## Findings

### CRITICAL — Unauthenticated socket → arbitrary file write as server UID

The socket has no authentication. On a shared dev server, any local user can connect
and send a crafted request specifying an arbitrary `output_path` (e.g.,
`~/.ssh/authorized_keys`, `/etc/cron.d/`). The server writes attacker-controlled
content to that path with server-process permissions.

**Remediation:** Server owns path resolution. Client never supplies an absolute output
path. Server validates all output paths against a configured `--output-dir` using
`os.path.realpath()`. Reject anything that resolves outside it.
**Status:** Accepted → ADR-001 §2.

---

### CRITICAL — World-writable `/tmp` socket enables socket hijacking

`/tmp/comfyless.sock` at default umask is readable and writable by all local users.
An adversary can connect before legitimate clients, intercept requests (capturing model
paths, prompts, LoRA paths, output paths), or replace the socket with their own
listener to return crafted results.

**Remediation:** Place socket in `$XDG_RUNTIME_DIR` (systemd-managed, 0700 per-UID)
or `mkdir /tmp/comfyless-$(id -u)/ mode=0700`. Directory permissions are the control,
not the socket file's own permissions.
**Status:** Accepted → ADR-001 §1.

---

### HIGH — Arbitrary model path → pickle deserialization code execution

Client specifies model directory path. `torch.load()` with `weights_only=False`
(default for many diffusers loaders) executes arbitrary Python via pickle. An attacker
who controls the model path — through the unauthenticated socket or by planting a file
— achieves code execution in the server process. Safetensors-format files are not
affected by pickle; `.ckpt`/`.pt` files are.

**Remediation:** Constrain accepted model paths to configured base directories
(`--model-base`). Validate with `os.path.realpath()` before any load.
**Status:** Accepted → ADR-001 §3.

---

### HIGH — LoRA path injection + adapter name injection

Same arbitrary-load surface as model paths. Additionally, adapter names derived from
LoRA filename stems are passed to `pipe.delete_adapters(name)` without sanitization.
If diffusers internals ever pass adapter names to subprocess calls, this is a command
injection vector.

**Remediation:** Apply same base-directory allowlist to LoRA paths. Sanitize adapter
names to `[a-zA-Z0-9_-]` before use.
**Status:** Accepted → ADR-001 §3, §4.

---

### MEDIUM — No schema validation; malformed input reaches model loaders

Server deserializes JSON and accesses fields by key. Missing fields cause
`KeyError`/`None` propagation into model-loading code whose error paths are sparse.
Wrong types for fields (e.g., list where string expected for `model_path`) may trigger
unanticipated behavior deep in diffusers.

**Remediation:** Explicit required-field + type schema validation at socket boundary.
Reject with structured error before any parameter reaches model-loading code.
**Status:** Accepted → ADR-001 §5.

---

### MEDIUM — No request bounds → VRAM exhaustion / server hang

Any local user can send large-dimension, high-step requests or rapidly alternate model
paths to force repeated 20B-parameter reloads. No inference timeout means a single
hung request blocks all clients indefinitely.

**Remediation:** Server-side bounds on max dimensions / steps; configurable inference
timeout with abort + error response.
**Status:** Deferred → `TECH_DEBT.md`.

---

### LOW — `--unload` is unauthenticated

Any local user can shut down the daemon. Local DoS only — no data exfiltration.

**Remediation:** `SO_PEERCRED` UID check on the Unix socket before honoring shutdown.
**Status:** Deferred → `TECH_DEBT.md`.

---

### INFO — Prompt content: no shell injection risk

Prompts are user-controlled strings passed directly to the pipeline, never interpolated
into shell commands. No risk as long as no request field is ever passed to
`subprocess.run(shell=True, ...)` or `os.system()`.

**Status:** Monitor during implementation; no action required now.

---

## Resolution Summary

| Finding | Severity | Resolution |
|---------|----------|------------|
| Arbitrary file write via client-supplied output path | CRITICAL | Fixed in design: server owns path resolution (ADR-001 §2) |
| World-writable `/tmp` socket | CRITICAL | Fixed in design: XDG_RUNTIME_DIR (ADR-001 §1) |
| Arbitrary model path → pickle execution | HIGH | Fixed in design: model-base allowlist (ADR-001 §3) |
| LoRA path injection + adapter name injection | HIGH | Fixed in design: allowlist + name sanitization (ADR-001 §3, §4) |
| No schema validation | MEDIUM | Fixed in design: boundary validation (ADR-001 §5) |
| No request bounds / VRAM exhaustion | MEDIUM | Deferred (TECH_DEBT.md) |
| Unauthenticated --unload | LOW | Deferred (TECH_DEBT.md) |
| Prompt injection | INFO | No action |
