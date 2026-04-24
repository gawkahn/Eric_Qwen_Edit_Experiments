# Security Review — comfyless/server.py hardening slice

AI-Disclosure: Claude (Opus 4.7) authored; Grant reviewed.
Date: 2026-04-23
Scope: hardening diff against `comfyless/server.py` implementing Findings 1, 2, 8 from `docs/security/review-comfyless-server-2026-04-23.md`.
Reviewer: security-auditor subagent (Opus)

## Summary

All three MEDIUM findings from the prior review have been implemented at the locations the review recommended, and each one materially reduces its target attack surface. Finding 2 (dir-mode enforcement) is closed with only a narrow residual TOCTOU that matches the original same-uid threat model. Finding 1 (recv bound + timeout) was **partially** closed by the initial diff: the byte cap is absolute and sufficient, but `conn.settimeout(5.0)` is a per-call timeout that resets on every chunk, so a slow-drip attacker below the 65 KiB chunk size could still hold a connection open up to `_MAX_FRAME_BYTES / 1 byte × 4.9s` — bounded by the byte cap but far longer than the intended 5 s wall-clock. Finding 8 (absolute-path requirement) closes the HF-probe side-channel as intended, but incidentally surfaces a **new MEDIUM**: `os.path.realpath` on a path containing an embedded NUL byte raises `ValueError` that is not caught on the `_check_paths` call site, and the same applies to the `savepath` → `_within` call — either crashes the accept loop. Both the Finding 1 residual and the new MEDIUM (H-2) were addressed with small targeted follow-up diffs in the same slice before merge; see "Post-review resolution" at the bottom.

## Verification of prior findings

### Finding 1 — `_recv` bounds + timeout

**Status in initial diff: partially closed.**

What works:
- `_MAX_FRAME_BYTES = 1 << 20` cap is enforced inside the loop before `buf += chunk` can grow unbounded. An attacker cannot OOM the server — the hard cap terminates reads at 1 MiB.
- `ValueError` is the right error class: the existing handler at `_handle_connection` catches `(json.JSONDecodeError, ValueError)` and maps both to a `ParseError` wire response. Matches convention and requires no handler change.
- `socket.timeout → ValueError("request timed out before newline")` is caught by the same handler — no server-kill on timeout.

What did not work as intended in the initial diff:
- `conn.settimeout(5.0)` is a per-`recv` call deadline, not a wall-clock deadline for the whole frame. Every time `conn.recv(65536)` returns with any bytes (even a single byte), the loop continues and the next `recv` restarts its own 5 s clock. An attacker who sends one byte every 4.9 s could hold the connection open for up to `(_MAX_FRAME_BYTES) × 4.9 s` — roughly 59 days — while blocking every subsequent client (single-threaded accept loop). The byte cap eventually terminates this, so memory is bounded, but availability (the single accept slot) was not.

**Applied in post-review follow-up:** `time.monotonic()` wall-clock deadline at the top of `_recv` with `conn.settimeout(remaining)` recomputed per iteration. The frame now terminates within 5 s wall-clock regardless of drip cadence.

### Finding 2 — socket dir mode enforcement

**Status: closed for the stated threat model, with one narrow residual and one symlink observation.**

What works:
- `mkdir(mode=0o700, exist_ok=True)` → `stat()` → uid check → `chmod(0o700)` only if `S_IMODE` differs. Matches recommendation.
- `stat.S_IMODE(st.st_mode)` correctly strips setuid/setgid/sticky bits.
- `st.st_uid != os.getuid()` is adequate for defending against a different-uid operator who pre-created the dir. On a stick-bit `/tmp` that owner can only be the same uid or root; we correctly refuse in either non-self case.
- `XDG_RUNTIME_DIR` branch is untouched — systemd maintains 0700 on that path.

Residual risks:
1. **Narrow TOCTOU between the check and the subsequent `socket.bind()`:** if a same-uid attacker wins a race between `chmod(0o700)` and `srv.bind(...)` (widening the dir back to 0o755), the socket is placed in a widened dir. Window is microseconds and same-uid only — matches the prior review's threat model. Not closable without `openat2(RESOLVE_BENEATH)` semantics that aren't available in Python's stdlib. **Acceptable — document only.**
2. **`Path.stat()` follows symlinks.** If a same-uid attacker plants `/tmp/comfyless-$UID` as a symlink to a dir they own elsewhere (with looser permissions on the target), `mkdir(exist_ok=True)` silently succeeds, `d.stat()` returns info on the symlink target, the uid check passes (their dir), and the socket is bound inside the attacker's-choice directory. `d.lstat()` would catch this by reporting `S_ISLNK`. Same-uid scope. See New Finding H-1 below.

### Finding 8 — absolute-path requirement

**Status: closed for its stated purpose; layered `_within` defense holds against traversal; NUL/control-char side effect introduced a new DoS (new finding H-2, also addressed in this slice).**

What works:
- `startswith("/")` rejects all relative paths including `Qwen/Qwen-Image-2512`, `./foo`, `../foo`, empty strings (the `or ""` guard), and any non-absolute form. The HF-repo-id probe channel described in Finding 8 of the prior review is now explicitly blocked at the schema boundary without reliance on `realpath`'s cwd-resolving behavior.
- Traversal defense via `_within` after the `startswith` check still works: `/model-base/../etc/passwd` passes the absolute check, but `os.path.realpath` collapses `..` to `/etc/passwd`, which fails `startswith(model_base + os.sep)`. Double-layer is intact.
- Tab/newline in a path that starts with `/`: realpath handles these safely — valid POSIX filename bytes, no shell expansion, still must live under `model_base`.

Residual risk (closed in post-review follow-up):
- **Embedded NUL byte (`\x00`) in an absolute-prefixed path** — e.g. `/foo\x00bar` — passes `startswith("/")` but `os.path.realpath` raises `ValueError: embedded null byte`. The raise propagates out of `_check_paths`, escapes `_handle_connection`, and takes down the daemon. Same issue affects `savepath → _within`.

**Applied in post-review follow-up:** NUL rejection added to `_validate_request` via `_PATH_FIELDS` frozenset (`model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `savepath`) plus each `loras[i]["path"]`. A NUL in any of those fields returns a `ValidationError` at the schema boundary before `realpath` is reached.

## New findings

### H-1 — LOW/MEDIUM: `_socket_dir` follows symlink on pre-existing `/tmp/comfyless-$UID`

**Severity:** LOW (same-uid threat model); promotes to MEDIUM if a shared-machine deployment ever happens.
**Location:** `comfyless/server.py` `_socket_dir`
**Description:** `d.mkdir(mode=0o700, exist_ok=True)` does not detect that `d` is a symlink; `d.stat()` follows the symlink. A same-uid attacker who pre-creates `/tmp/comfyless-$UID → /tmp/attacker-chosen-dir` before the first daemon start places the socket in the attacker's directory (same-uid).
**Recommendation:** call `d.lstat()` first and reject if `stat.S_ISLNK(st.st_mode)`, then do the uid/mode enforcement on `d.stat()`. Two-line change.
**Status:** deferred. Logged in TECH_DEBT as part of the next server-touching commit.

### H-2 — MEDIUM: Embedded-NUL path crashes the accept loop

**Severity:** MEDIUM (single-user local DoS; pre-existing, surfaced by the Finding 8 fix).
**Location:** `comfyless/server.py` `_within` (line ~148), reached from `_check_paths` and from the savepath verification in `_handle_generate`.
**Description:** `os.path.realpath("/foo\x00bar")` raises `ValueError: embedded null byte`. Neither `_check_paths` nor the savepath `_within` call was wrapped in try/except, so the exception would escape `_handle_connection`, exit the `with conn` block, bubble out of the accept-loop iteration, and take down the daemon. A same-uid local process could one-shot any running server with a single malformed JSON request.
**Recommendation:** reject `"\x00"` in path-shaped string fields inside `_validate_request`.
**Status: closed in this slice.** `_PATH_FIELDS` frozenset added at module scope; schema validator now rejects NUL in `model`, `transformer_path`, `vae_path`, `text_encoder_path`, `text_encoder_2_path`, `savepath`, and every `loras[i]["path"]` before `_check_paths` runs.

### H-3 — LOW: `lora_weight` is not type-checked in the schema validator

**Severity:** LOW.
**Location:** `comfyless/server.py` `_validate_request` (loras loop).
**Description:** `_validate_request` validates `loras[i]["path"]` is a `str` but not `loras[i].get("weight")`. A malicious JSON value of `"weight": "not-a-number"` raises `ValueError` at `float(...)` on the consumer side — caught by an outer `except`, so no crash, but inconsistent with the rest of the schema's strict type-checking.
**Recommendation:** add `if "weight" in lora and not isinstance(lora["weight"], (int, float)): return "..."` in the `loras` loop. Optional; no exploit path.
**Status:** deferred. Logged in TECH_DEBT.

### H-4 — INFO: `savepath` explicitly not required to be absolute (by design)

**Severity:** INFO (design-consistent).
**Description:** The absolute-path requirement applies to model/component/LoRA paths (client supplies full absolute paths) but not to `savepath`, which is a template under `output_dir` and must remain relative. The code correctly strips leading `/` and `\\` and validates the expanded result via `_within(Path(expanded).parent, output_dir)`. Called out so the reviewer doesn't expect symmetry. NUL-in-savepath DoS addressed by H-2 fix above.

### H-5 — INFO: `_log` not invoked for request-level validation errors

**Severity:** INFO.
**Description:** Failed `_validate_request`, `_check_paths`, and `_recv` parse errors are sent to the client but not logged to the server's stderr. For a single-user box this is fine, but the same code path is the future `--json` agent bridge: when a model-driven client starts sending malformed requests, silent rejection is harder to diagnose than logged rejection. Flag for the scope-change gate.

### H-6 — INFO: Broad `except Exception` in LoRA removal fallback

**Severity:** INFO (pre-existing, not part of this slice).
**Location:** `_handle_generate` LoRA removal block.
**Description:** Recovery block does the right thing functionally (clean evict + reload), but `except Exception` around `pipe.delete_adapters` is broad — would also swallow wrapped interrupts and hide bugs. Not a security-critical path. Worth a one-line narrowing to `except (RuntimeError, KeyError, AttributeError)` if the file is re-opened.

## Consistency check — other input-validation sites

Surveyed for sites that should mirror the new hardening:
- **`savepath` field** — treated differently by design (see H-4). NUL rejection applies.
- **`output_dir` / `model_base`** at `run_server` entry — already `os.path.realpath`'d and `isdir` checked, never client-controlled. OK.
- **`loras[i]["weight"]`** — see H-3.
- **`sanitize_adapter_name`** — already applied. OK.
- **`_expand_savepath_template` output** — verified via `_within` both before and after resolution. Consistent with the layered-defense pattern.

No other sites need the absolute-path treatment.

## Post-review resolution

Both blockers named above were addressed in the same slice before merge:

1. **Finding 1 wall-clock deadline** — `_recv` rewritten to use `time.monotonic()` deadline with per-iteration `conn.settimeout(remaining)`. Frame terminates within 5 s wall-clock regardless of drip cadence. `import time` added to module imports.

2. **H-2 NUL rejection** — `_PATH_FIELDS` frozenset defined at module scope; NUL-byte check added to `_validate_request` before `_check_paths` runs. Smoke-tested: NUL in `model`, `loras[i].path`, and `savepath` all return `ValidationError` responses.

Deferred to next server-touching commit and logged in TECH_DEBT:
- H-1 (symlink check via `lstat`).
- H-3 (lora_weight type check).

Test suites: 368/368 pass (manual-loop 186, multistage 141, samplers 41) after all changes.

## Conclusion

**Ready to merge.** The three MEDIUMs from the original review are verifiably closed. H-2 (new MEDIUM surfaced during re-review) is closed in the same slice. H-1 and H-3 deferred to the next server-touching commit with TECH_DEBT tracking. No CRITICAL or HIGH findings; the review bar's §12 surface remains in its stated single-user desktop threat model. Network transport and `--json` bridge continue to require a fresh ADR + review when they land (unchanged from prior review's future-scope guidance).
