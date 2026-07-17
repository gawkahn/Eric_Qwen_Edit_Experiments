# Security Review — Daemon pause opt-out + client recv-error contract (slice PAUSE follow-up)

**Date:** 2026-07-17
**Reviewer:** Claude (Fable 5), security-auditor pass
**Change under review:** uncommitted working-tree changes to `comfyless/server.py` (`interactive_pause=False` at the daemon's `generate()` call), `comfyless/pause.py` (`enabled: bool = True` opt-out on `sigint_pause`), `comfyless/generate.py` (`_send_server_command` synthetic `ClientRecvError` on recv-path failure)
**Baseline trust model:** ADR-001 (single-user workstation; `AF_UNIX` sockets in per-UID `0700` dir; cross-UID access precluded by directory mode; same-UID adversary = footgun class, not exploit class), ADR-020 (one daemon per GPU, device-keyed sockets)
**Prior reviews checked for invariant regression:** `docs/security/review-comfyless-server-2026-04-23.md`, `docs/security/review-parallel-daemon-2026-07-03.md`
**AI-Disclosure:** Review authored by Claude (Fable 5); reviewed by Grant Kahn.

> Disposition note (parent session, post-review): SHOULD 1 was applied in the
> same slice (`_send_server_command` now rejects valid-JSON non-dict responses
> with a synthetic `ClientRecvError`; negative test added to
> `test_server_robustness.py`). SHOULD 2 (connect-side backlog-full ambiguity)
> is recorded in `TECH_DEBT.md` ("Connect-refused on a LIVE daemon…",
> 2026-07-17) for its own slice. Findings 7/8 are escalation tripwires already
> covered by the CLAUDE.md review bar.

## Scope

Reviewed:
- `comfyless/pause.py` (whole file, 146 lines) — `sigint_pause` guard ordering, `enabled=False` path, signal-handler install/restore, `call_kwargs` mutation lifecycle
- `comfyless/server.py:44-323` — wire guardrails, `_socket_dir`/`socket_path`/`_device_socket_slug` (H-1), `_validate_request`, `_check_paths`, `_send`/`_send_safe`/`_recv`
- `comfyless/server.py:780-963` — `_handle_generate` generation block incl. the new `interactive_pause=False`, atomic output reservation, `run_server` accept loop
- `comfyless/generate.py:1380-1420, 1680-1720` — `generate()` signature, `sigint_pause` arming sites
- `comfyless/generate.py:2171-2265, 2351-2420, 3040-3060, 3140-3183` — `_send_server_command`, `_send_unload`, `_delegate_to_server`, iterate fan-out disposition (`_iterate_combo_disposition`)
- `comfyless/mcp_server.py:2141-2178` — the other `generate()` caller (opt-out absence check)
- Prior review docs above (parallel-daemon findings 1-5; 2026-04-23 conclusion/out-of-scope sections)

Not reviewed:
- `test_pause.py` / `test_server_robustness.py` contents — tests out of audit scope; claims about guard coverage are taken from docstrings, not verified against test bodies
- `comfyless/cascade.py` daemon path — does not route through `sigint_pause` (no arming site there); not read in depth
- Live git diff vs HEAD — audited current file state against the change description; did not mechanically confirm that *only* the three described edits are uncommitted

## Analysis (threat model and what was checked)

The trust boundary this change touches is the same-UID Unix-socket boundary between the comfyless CLI client and the per-GPU daemon, inside a `0700` per-UID directory with the H-1 symlink refusal (`server.py:82-83`) and anchored device-slug whitelist (`server.py:102-124`) intact. Untrusted-ish data flows audited: (a) the wire *request* into the daemon — could any payload field reach `interactive_pause` or the pause machinery? No: `_handle_generate` calls `generate()` with an explicitly enumerated keyword list (`server.py:815-866`), no `**req` splat, `interactive_pause=False` is a literal, and `pause.py` reads no environment or request state — the opt-out is structurally unreachable from the wire (T2-grade). (b) The wire *response* into the client — `_recv`'s failure text `e` interpolated into the synthetic error: `ValueError` messages are fixed strings (`server.py:310,315,320`), `json.JSONDecodeError`/`UnicodeDecodeError` carry only positions/byte values, `OSError` carries errno text; no daemon-controlled bytes flow into `e`, and `device` is the client's own argv (further `!r`-escaped). The string is only printed to stderr and never re-parsed. (c) Signal semantics: with `enabled=False` the context manager yields before touching `signal.signal` or `call_kwargs` (`pause.py:77-79`), so the daemon's SIGINT disposition is byte-identical to pre-slice-PAUSE behavior — a terminal ^C raises `KeyboardInterrupt`, which (being outside `except Exception` at `server.py:867`) unwinds through the accept loop's `finally` and removes the socket cleanly. That is the intended fix for the observed wedge, and it is fail-closed: dead daemon and unlinked socket, not a hung shared service.

Error/exception paths and absences: the new `ClientRecvError` closes the worst prior fail-open (recv failure → `None` → in-process generation against a GPU whose VRAM a live daemon holds), and under `--iterate` it maps to `'fatal'` in `_iterate_combo_disposition` — the sweep stops, which is the correct availability trade (a stopped sweep is recoverable; a double-generation CUDA OOM can take down the daemon for every client). What remains open is the *edges* of the new contract: `_recv` returns whatever `json.loads` produced, so a valid-JSON non-dict response evades both the exception path and the `None` path (finding 1), and the connect-time `OSError → None` fall-through can still fire against a *live but busy* daemon via a full listen backlog (finding 2). Path confinement, atomic output reservation, and request validation are untouched by this diff; no scope creep beyond the three described edits was found in the regions read.

## Findings

### 1. SHOULD — Valid-JSON non-dict daemon response evades the new recv contract: `null` re-opens the forbidden in-process fall-through; other scalars crash the client

**Location:** `comfyless/server.py:323` (`_recv` returns `json.loads(...)` untyped) → `comfyless/generate.py` (returned as-is) → `resp is None` check, then `resp.get(...)` in callers

**Risk:** `_recv` raises `ValueError` only on *unparseable* bytes. A response line that is valid JSON but not an object slips past both new guards: `null\n` makes `_recv` return `None`, which `_send_server_command` cannot distinguish from clean EOF — the client logs "connection failed — running in-process" and falls through to in-process generation while the daemon is alive and holding VRAM, i.e. exactly the failure mode this change exists to close. Any other non-dict (`"x"`, `123`, `[1]`) returns truthy, and `resp.get("status")` raises `AttributeError` — the delegate call sits *outside* the `try` in `_run_one`, so the traceback escapes and aborts an `--iterate` sweep with an unhandled crash, the other failure mode this change exists to close. Under the ADR-001 threat model the sender must be a same-UID broken/impersonating daemon, so this is a robustness gap in the new contract rather than a privilege issue — but the contract's docstring ("None — daemon absent … dict — the server's response") is currently stronger than the code.

**Remediation (smallest change, client-side only):** in `_send_server_command`, capture `resp = _recv(...)` and before returning check `if resp is not None and not isinstance(resp, dict):` → return the synthetic `ClientRecvError` dict. Optionally note in the docstring that a literal `null` response remains indistinguishable from clean EOF at this layer (closing that fully would need `_recv` to distinguish EOF from parsed-null — a larger change; acceptable to document instead under the same-UID model). **[Applied in this slice — see disposition note above.]**

### 2. SHOULD — Connect-failure fall-through still fires against a live-but-busy daemon (full listen backlog), re-creating the occupied-GPU in-process hazard the diff closes on the recv path

**Location:** `comfyless/generate.py` (`except OSError: return None` around connect/send), "Server socket found but connection failed — running in-process"; daemon side `comfyless/server.py:938` (`srv.listen(4)`), serial accept loop

**Risk:** the daemon accepts serially and a generation holds the loop for minutes; queued clients sit in a backlog of 4. On Linux, a connect to an `AF_UNIX` stream socket with a full backlog fails `ECONNREFUSED` — indistinguishable, in the current code, from a stale socket left by a dead daemon. The fifth-and-later concurrent client (concurrent sessions are a documented working pattern for this repo) falls through to in-process generation on the GPU the busy daemon occupies — the same VRAM-collision/OOM hazard, arriving one hop earlier than the recv path this diff fixed, and requiring no attacker at all. Pre-existing behavior, not introduced by this diff; flagged because the diff's own rationale now applies asymmetrically across the two paths, and because the new contract comment documents `None` as "daemon absent or unreachable," which conflates "unreachable because busy."

**Remediation (smallest change):** on connect-`OSError` with an *existing* socket file, retry the connect a small bounded number of times (e.g. 3 × 2s) before returning `None`; a stale socket keeps refusing and still falls through, while a backlog-full daemon usually drains within the window. Alternatively raise the daemon's `listen()` backlog (e.g. 16) as a one-word server-side mitigation and record the residual ambiguity in TECH_DEBT.md. Either way, this touches beyond the declared three-edit scope of the current slice — land it as its own slice, not folded in. **[Recorded in TECH_DEBT.md 2026-07-17.]**

### 3. ACCEPT — Daemon pause opt-out: no signal/privilege behavior change beyond disabling the hook; not influenceable from the wire

**Location:** `comfyless/server.py:860-865`; `comfyless/pause.py:77-79`

`enabled=False` is checked before any side effect: no `signal.signal` call, no `call_kwargs` mutation, no stdin access. The daemon's SIGINT disposition reverts exactly to pre-slice-PAUSE semantics: ^C in the daemon terminal raises `KeyboardInterrupt`, which is deliberately not caught by `except Exception` (`server.py:867`) and unwinds through `run_server`'s `finally` — socket unlinked, clean exit. Fail-closed availability (dead + restartable beats wedged-on-`input()`). The wire request cannot reach the parameter: kwargs are explicitly enumerated with a literal `False`, no splat, and `pause.py` consults no ambient state. One cosmetic note, not a finding: `KeyboardInterrupt` mid-generation skips the `except Exception` cleanup, leaving the reserved 0-byte `comfylessNNNN.png` placeholder behind — harmless (operator-initiated full shutdown; the atomic-reservation invariant concerns concurrent-writer collisions, which are unaffected).

### 4. ACCEPT — Synthetic `ClientRecvError` message composition is injection-clean

`device` is the client's own `--device` argv value, rendered with `!r` (control characters escaped). `e` derives from `_recv`'s own exceptions: fixed-string `ValueError`s, `json.JSONDecodeError` (position info only), `UnicodeDecodeError` (codec name, hex byte, position), or `OSError` (errno text). No daemon-controlled bytes reach the string; the dict is consumed only by `resp.get("error")` → stderr print and is never written to a sidecar, re-parsed, or used to select a path.

### 5. ACCEPT — `--iterate` abort on `ClientRecvError` is the correct availability trade

`ClientRecvError` → rc 1 → `'fatal'` → sweep stops with the batch id printed. The replaced behaviors were strictly worse: an escaping `ValueError` also killed the sweep (with a raw traceback), and `None`-fall-through risked in-process generation OOMing against the daemon's VRAM — an availability failure of the *shared* service versus a clean stop of one local sweep. One bounded caveat: `_CLIENT_RECV_TIMEOUT_SEC = 600` means a legitimate generation exceeding 10 minutes (plausible at this project's 17-50 MP ceilings with multi-stage chains) is misclassified as a wedged daemon and fatally aborts the sweep while the daemon finishes and writes the image anyway (its response send is swallowed by `_send_safe` — daemon survives). Pre-existing constant, warn-don't-block territory; revisit the constant if >600s runs become routine.

### 6. ACCEPT — Clean-EOF → fall-through is not usefully abusable by a daemon impersonator

An impersonator must plant a socket named `comfyless-<slug>.sock` inside the per-UID dir: cross-UID is blocked by the `0700` mode, the H-1 symlink refusal, the ownership check, and the anchored `fullmatch` slug whitelist (Finding 3 of the 2026-07-03 review — verified still in place). A same-UID actor who can plant a socket already has arbitrary code execution as the user; inducing an in-process generation is a lesser capability. Consistent with the ADR-001 verdict; no change needed.

### 7. INFO — Pre-existing, out of this diff's scope: daemon-controlled response fields drive a client-side file write and unsanitized terminal output

**Location:** `comfyless/generate.py` (sidecar written to `f"{stem}.json"` where `stem` derives from the *response's* `output_path`; raw error text to terminal — ANSI escape sequences from a hostile daemon would render)

Both require a hostile same-UID daemon and are therefore outside the ADR-001 threat model today. Recorded because the "surfaces that become Red Zone on scope change" list applies: if daemon responses ever cross a network transport or feed an LLM agent's context, both become real findings (path write primitive; injection carrier — cf. the 2026-04-23 review's out-of-scope escalation table). No action now; the escalation trigger is already documented in CLAUDE.md's review bar.

### 8. INFO — MCP server relies on the *implicit* non-TTY guard rather than the explicit opt-out

**Location:** `comfyless/mcp_server.py:2141-2178` (no `interactive_pause=False`); guard in `pause.py`

The MCP stdio server calls `generate()` with the default `interactive_pause=True` on its main thread; safety rests on stdin not being a TTY (true under mcpo/any MCP client — stdin is the JSON-RPC pipe; a TTY-stdin MCP server is unusable as a protocol endpoint anyway). Assumption named per audit rules: if the MCP server ever gains a launch mode where stdin is a TTY while the transport moves elsewhere (e.g. an SSE/HTTP transport slice), the pause hook would arm inside a service process — add `interactive_pause=False` in that slice. A one-line explicit opt-out now would also be defensible symmetry, but it touches a Red-Zone-gated file (`mcp_server.py`) outside this slice's declared scope, so it should not ride along silently.

## Regression check against prior-review invariants

| Invariant (source) | Status |
|---|---|
| Daemon survival — no client behavior may kill the accept loop (2026-04-23 review; `test_server_robustness`) | Unchanged. The diff is client-side plus one literal kwarg; `_send_safe` / structured-error paths untouched. Operator ^C killing the daemon is pre-slice-PAUSE behavior restored, not a client-triggered path. |
| Path confinement — `_check_paths` union roots, absolute-path + realpath containment (ADR-001/ADR-018) | Untouched by diff; verified present (`server.py:221-266`). |
| Atomic output reservation — `O_CREAT\|O_EXCL` counter (parallel-daemon Finding 1 fix, ADR-020) | Untouched; verified present (`server.py:780-796`) incl. reserved-file unlink on `Exception`. See finding 3 note re `KeyboardInterrupt` leaving the placeholder. |
| Socket-name whitelist ordering / fullmatch / integer fold (parallel-daemon Finding 3) | Untouched; verified present (`server.py:102-124`). |
| H-1 symlink refusal + `0700` dir + uid check | Untouched; verified present (`server.py:77-92`). |

## Verdict

**PASS — no MUST-FIX.** The three edits do what they claim, fail closed, and close a genuine fail-open (recv failure → in-process generation against an occupied GPU) plus a genuine availability wedge (daemon blocked on `input()`). The daemon opt-out is structurally unreachable from the wire and changes no signal or privilege behavior beyond restoring pre-slice SIGINT semantics. Two SHOULDs remain, both edges of the same contract the diff introduces: (1) type-check the parsed response in `_send_server_command` so valid-JSON non-dict responses can't crash the client or (via `null`) re-open the fall-through — a two-line client-side change that can ride with this slice [applied]; (2) the connect-path `ECONNREFUSED`-with-live-socket ambiguity (backlog full vs stale socket) re-creates the occupied-GPU fall-through under concurrent clients with no attacker — pre-existing, land as its own slice [TECH_DEBT.md]. Findings 7 and 8 are escalation tripwires already consistent with the project's documented Red-Zone-on-scope-change posture.
