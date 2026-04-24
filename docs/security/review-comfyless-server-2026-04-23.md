# Security Review — comfyless/server.py (Unix socket IPC)

AI-Disclosure: Claude (Opus 4.7) authored; Grant to review.

Date: 2026-04-23
Scope: `comfyless/server.py` (534 lines), §12 trigger (IPC / Unix sockets)
Reviewer: security-auditor subagent (Opus)
Coupled surface reviewed: `nodes/eric_diffusion_utils.py::resolve_hf_path` (reached via `generate._load_pipeline` from `_handle_generate`)

## Summary

The server's design implements the controls promised in ADR-001 (per-UID 0700 socket directory, server-owned output paths, `--model-base` allowlist via `realpath`, adapter-name sanitization, strict schema validation at the boundary). For a solo-desktop single-user threat model it is acceptable as-is. The highest-severity finding is a **MEDIUM** unbounded read in `_recv` (no length cap, no timeout) that allows a same-uid process to exhaust memory or indefinitely block the single-threaded accept loop; the `/tmp/comfyless-$UID/` directory-mode bootstrap (mkdir is not idempotent w.r.t. permissions) is the second MEDIUM. No CRITICAL or HIGH findings were identified against the stated threat model. Several findings become HIGH or CRITICAL the moment `--serve` grows a network transport — see "Future scope."

## Findings

### 1. MEDIUM — `_recv` has no frame-size cap or socket timeout

**Location:** `comfyless/server.py:173-183`

**Description:** `_recv` loops `conn.recv(65536)` and appends to `buf` until it sees `b"\n"`. There is no maximum frame size and no `conn.settimeout(...)`. A same-uid process (accidental runaway tool, buggy test harness, or a compromised local process) can:
- hold the connection open silently and block the single-threaded accept loop indefinitely (the server processes one connection at a time; `listen(4)` only backlogs 4 connects) — trivial local DoS against subsequent clients, and
- stream arbitrarily many bytes with no newline and OOM the server process.

On a single-user workstation the blast radius is the server itself (no privilege boundary crossed), which keeps this MEDIUM rather than HIGH. TECH_DEBT.md already tracks "daemon inference timeout" but that is a different timeout (per-generation); this is per-read/per-connection.

**Recommendation:** add a small targeted change in `_recv`:
```python
MAX_FRAME = 1 << 20   # 1 MiB; real requests are < 10 KiB
conn.settimeout(5.0)  # header must arrive within 5s
...
buf += chunk
if len(buf) > MAX_FRAME:
    raise ValueError("request frame exceeds 1 MiB")
```
Keep the constants at module scope so they are greppable. The existing `(json.JSONDecodeError, ValueError)` handler in `_handle_connection` will already translate `ValueError` into a structured `ParseError` response. Wrap the recv in `except socket.timeout` to return an error frame without killing the server.

### 2. MEDIUM — Fallback socket dir bootstrap does not enforce 0700 when the directory already exists

**Location:** `comfyless/server.py:50-52` (`_socket_dir`)

**Description:** `Path(f"/tmp/comfyless-{os.getuid()}").mkdir(mode=0o700, exist_ok=True)` applies `mode=0o700` only on creation. If the directory already exists with laxer permissions (e.g., a stale directory from a previous buggy build, a manual `mkdir /tmp/comfyless-$UID`, or an `umask` that widened a past creation), the current run will happily place the socket inside it and rely on an unchecked mode. This is the defense-in-depth gap that ADR-001 §1 explicitly calls out: the directory's 0700 permission *is* the control — the socket's own 0600 (line 514) is only "belt-and-suspenders." On a stick-bit `/tmp` only the same uid can have created the laxer dir, so the concrete attack requires either (a) a same-uid compromised process that pre-creates the directory before the server starts, or (b) operator error. Same-uid threat model keeps this MEDIUM.

**Recommendation:** verify and enforce mode after `mkdir`:
```python
d = Path(f"/tmp/comfyless-{os.getuid()}")
d.mkdir(mode=0o700, exist_ok=True)
st = d.stat()
if stat.S_IMODE(st.st_mode) != 0o700:
    os.chmod(d, 0o700)
if st.st_uid != os.getuid():
    raise RuntimeError(f"socket dir {d} not owned by current uid")
```
No action needed for the `$XDG_RUNTIME_DIR` branch — systemd guarantees 0700 there.

### 3. MEDIUM — `unload` command terminates the server with no caller authentication

**Location:** `comfyless/server.py:227-236`

**Description:** Any process able to connect to the socket can shut down the daemon with `{"type":"unload"}`. Since the socket dir is 0700 this is restricted to the same uid; still, any misbehaving script under that uid can kill an expensive warm-cache server mid-workload. ADR-001 deferred this (§SO_PEERCRED) and TECH_DEBT.md tracks it. Calling it out here so the review record is complete, not because it changes the deferral.

**Recommendation:** no code change required right now; keep the TECH_DEBT entry open. If acted on: `conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))` to confirm the peer pid/uid before honoring `unload`.

### 4. MEDIUM — `InferenceError` response returns a full Python traceback to the client

**Location:** `comfyless/server.py:471-478`

**Description:** `_handle_generate` attaches `traceback.format_exc()` to the error response on any generation failure. Under the current same-uid threat model this leaks no privilege boundary — the client already has filesystem read access to the server's code. However, it does couple the wire protocol to internal file paths and implementation details, which hardens the interface in a way that is awkward to unwind later (once an agent or script starts parsing tracebacks, removing them is a breaking change). Flagging because the `--json` bridge (TECH_DEBT noted) is explicitly a future LLM-agent tool surface, and a traceback in an agent's context window is an instruction-injection carrier.

**Recommendation:** keep the traceback for now (it is genuinely useful for debugging) but plan to gate it behind a server-side `--debug` flag once the LLM-agent bridge lands, so the default output surface is structured error codes only. No change required this pass; note it as a debt item.

### 5. LOW — `listen(4)` backlog + single-threaded accept creates an availability bottleneck

**Location:** `comfyless/server.py:513, 524-529`

**Description:** `srv.listen(4)` combined with a synchronous `while keep_running: conn, _ = srv.accept(); with conn: _handle_connection(...)` loop means the fifth concurrent connection attempt gets `ECONNREFUSED` and any in-flight generation (minutes long) blocks everyone else. Not a security bug per se — it is the intended single-worker design — but combined with finding #1 (no read timeout) and #3 (unauthenticated `unload`) it amplifies the DoS surface.

**Recommendation:** no code change; document in the server module docstring that it is single-worker by design and clients should retry on connection failure. Already partially captured in TECH_DEBT ("daemon inference timeout").

### 6. LOW — Socket file removed without ownership check before bind

**Location:** `comfyless/server.py:508-509`

**Description:** `if sock_path.exists(): sock_path.unlink()` removes any existing file at the socket path (stale socket from a prior run, or any same-uid squatter). Because the parent directory is 0700 only the same uid can have placed a file there, so the squatter scenario requires a same-uid misbehavior. Still, unlinking without confirming ownership/type is a small gratuitous broadening — e.g., if a future code path ever placed a non-socket file there (config, lock, etc.), this would silently delete it.

**Recommendation:** defense-in-depth — `st = sock_path.lstat(); if not stat.S_ISSOCK(st.st_mode): raise RuntimeError(...)` before unlinking. Low priority.

### 7. LOW — `os.path.realpath` is evaluated at check-time, not use-time (TOCTOU)

**Location:** `comfyless/server.py:134-138` (`_within`), called from `_check_paths` and output-path verification

**Description:** `_within` resolves both paths with `realpath` at the moment of validation. Between that check and the eventual `from_pretrained` / `load_file` call, a same-uid process could swap a symlink inside `--model-base` to point outside. Practical impact on a single-user box is minimal: the same uid already has the ability to place a weights file anywhere and load it through any other node; `realpath` + the 0700 socket dir constrain this further. Flagging for completeness.

**Recommendation:** no change. If this surface becomes multi-tenant or networked, revisit by using `os.open(..., O_NOFOLLOW)` + `os.path.realpath` on `/proc/self/fd/N` or by requiring canonicalized inputs from a restricted directory set.

### 8. LOW — Server invokes `resolve_hf_path` indirectly with no explicit block for HF repo IDs

**Location:** interaction between `comfyless/server.py:246` → `generate._load_pipeline` → `nodes/eric_diffusion_utils.py::resolve_hf_path`

**Description:** When the server calls `_load_pipeline(req["model"], ...)`, `_load_pipeline` passes the model string through `resolve_hf_path`. If the caller supplied an HF repo-id-shaped string (e.g., `"Qwen/Qwen-Image-2512"`), `resolve_hf_path` would normally attempt a local-cache lookup. In the server path, however, `_check_paths` runs *before* `_load_pipeline`, and `_within("Qwen/Qwen-Image-2512", model_base)` invokes `os.path.realpath` on a non-absolute string which resolves relative to the server's cwd — almost never under `--model-base` — so the request is rejected before `resolve_hf_path` is reached. This is the correct outcome but is load-bearing on a subtle behavior (`realpath` on relative strings + the allowlist check). A future refactor that weakens `_check_paths` or lets repo-id strings through would be a CRITICAL regression: `resolve_hf_path` with `allow_hf_download=False` would still leak whether a given HF repo is present in the operator's cache. Server currently does NOT set `allow_hf_download`, so network downloads are disabled via the daemon — good.

**Recommendation:** make the intent explicit by adding an up-front check in `_check_paths` that rejects any `model`/component path that doesn't begin with `/`. One-line change:
```python
if not model.startswith("/"):
    return f"model path must be absolute: {model!r}"
```
This removes the reliance on `realpath`'s relative-path behavior and prevents any future scope change from silently opening an HF-probe side channel. Add the same check for the four `*_path` fields and every `loras[i].path`.

### 9. LOW — Schema validator uses `isinstance` on `type` objects but missing `bool` subtype check

**Location:** `comfyless/server.py:94-127`, `_GENERATE_OPTIONAL`

**Description:** In Python, `bool` is a subclass of `int`. `isinstance(True, int)` is `True`. Fields declared `int` (e.g., `seed`, `steps`, `width`, `height`) will accept JSON `true`/`false`. Likewise `isinstance(True, float)` is `False` so `cfg_scale` is fine. The downstream code passes these to `torch` / diffusers which will happily coerce `True → 1` and continue. Not exploitable; creates confusing metadata in the PNG sidecar.

**Recommendation:** narrow the int checks:
```python
if field in ("seed","steps","width","height","max_sequence_length"):
    if isinstance(req[field], bool) or not isinstance(req[field], int):
        return f"Field {field!r}: expected int"
```
Low priority; a well-behaved client won't hit it.

### 10. INFO — LoRA warnings include full filesystem paths in the response

**Location:** `comfyless/server.py:388-394, 481-482`

**Description:** `lora_warnings` entries embed the full LoRA path from the request. Since the client supplied the path in the first place, nothing new is disclosed. No PII exposure in the single-user model. Flagged for the `--json`/agent bridge scope-change: in the agent setting these strings become LLM-visible, which is fine but worth naming.

### 11. INFO — `del pipeline` in unload handler does not actually free references

**Location:** `comfyless/server.py:229-234`

**Description:** `pipeline = server_state.get("pipeline"); del pipeline` only deletes the local binding; the server_state dict still holds the reference. The subsequent `server_state.clear()` is what actually releases it. Working as intended; the `del` line is cosmetic. Not a security issue; flagging because it could mislead a future reader fixing an OOM.

### 12. INFO — `torch.load(..., weights_only=True)` and pickle exposure

**Location:** indirect via `_load_pipeline` → `pipeline_class.from_pretrained(model_path, ...)` and `load_component` (`nodes/eric_diffusion_utils.py:505, 541, 568, 805`)

**Description:** Diffusers' `from_pretrained` and `from_single_file` internally handle `.safetensors` (safe) and `.bin` (pickle) — the latter is still called with torch defaults. Because `_check_paths` constrains paths to `--model-base`, and the operator controls what lives there, the pickle-RCE exposure is bounded to files the operator deliberately placed (matches ADR-001 §3). No action required; flagging so the constraint is recorded.

### 13. INFO — Logs include full LoRA paths to stderr

**Location:** `comfyless/server.py:336, 386, 389, 393`

**Description:** `_log` writes to stderr; filenames are not secrets in this project. No PII concerns. Record-keeping.

## Coupled surface: `resolve_hf_path`

The server reaches `resolve_hf_path` only after `_check_paths` has asserted the input path is within `--model-base`. In practice this means `resolve_hf_path` is invoked with an already-local filesystem path (the `_is_hf_repo_id` check returns False on anything starting with `/`), and the HF-cache/network branches are unreachable from the daemon. This is a sound outcome but, as noted in finding #8, is load-bearing on the allowlist check. A dedicated §12 review of `resolve_hf_path` is captured separately in `docs/security/review-resolve-hf-path-2026-04-23.md`.

## Out of scope / Future scope

- **Network transport (`--serve` over TCP / HTTP).** Immediately promotes findings #1 (no read limit/timeout), #2 (dir mode), #3 (unauthenticated unload), #4 (traceback leak), and #6 (unlink-on-start) to CRITICAL: an unauthenticated network attacker can DoS, shut down, or leak filesystem details on the host. Any commit that adds a network interface is Red Zone from day one, requires a fresh ADR and `security-auditor` run *before* code, and must add authentication (token or mTLS) plus a bound on request size and timeout as preconditions of merge.
- **LLM agent `--json` bridge.** Promotes finding #4 to HIGH (traceback → agent context window → injection carrier) and makes finding #10 relevant (path strings as prompt-injection surface in model output). Do not wire the agent bridge without first gating tracebacks behind a debug flag and normalizing error strings.
- **Shared-machine deployment.** Makes finding #3 HIGH (unauthenticated unload by other users) and finding #7 meaningful (symlink-race). `SO_PEERCRED` gate becomes required, not deferred.
- **Batch generation from a caller-supplied list of prompts/outputs.** §12 trigger for large-scale file writes; warrants its own review before the feature lands.

## Conclusion

**Acceptable as-is for the current single-user desktop threat model, with recommended MEDIUM fixes queued before the next non-trivial change to `server.py`.** The ADR-001 controls are faithfully implemented; the gaps found are input-validation / defense-in-depth tightenings, not broken invariants. Three MEDIUM findings (unbounded `_recv`, non-idempotent `0700` dir mode, unauthenticated `unload`) should be addressed; the first two are small targeted diffs and can ride with the next server-touching commit. Do not wire the `--json` LLM-agent bridge or a network transport until a fresh ADR + `security-auditor` pass has been completed — those scope changes move multiple LOW/INFO findings to HIGH or CRITICAL.
