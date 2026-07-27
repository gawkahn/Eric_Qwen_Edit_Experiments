AI-Disclosure: Claude (Opus 5, security-auditor agent) authored; Claude (Sonnet 5) requested and applied the recommended fix; Grant reviewed.

# Security review — `comfyless/server.py` `_validate_request` pyright drawdown fix

Date: 2026-07-27
Scope: a pyright type-error fix (ADR-042 comfyless/ drawdown) touching
`_validate_request` in `comfyless/server.py` — a §12 Red Zone surface
(Unix-socket IPC daemon, ADR-001).

## Change under review

`_validate_request` had 4 `reportOptionalSubscript` errors from subscripting
`result.error` (typed `Optional[dict]`) without pyright being able to narrow
it, even though `ValidationResult`'s own docstring documents the invariant
"ok=False → error is a structured dict; payload is None." The FIRST fix
attempt added `assert err is not None` before the subscripts. This review
was requested on that first attempt, before commit.

## Method

Read `server.py:1-260` (module header, socket hardening, full
`_validate_request`), `server.py:380-509` (`_recv` framing, `_handle_connection`
exception scope, dispatch), `server.py:1200-1225` (`run_server` accept loop),
`params_validation.py` in full (`ValidationResult`, `_make_err`, `_check_field`,
`validate_lora_entry`, `validate_ref_image_entry`, `validate_machine_request`),
the sibling MCP-plane consumer of the same dataclass (`mcp_server.py:1944-1965`,
`:2232-2251`), the outermost `run_server` handler (`generate.py:3062-3075`),
repo-wide greps for `PYTHONOPTIMIZE`/`-O`/`-OO` across shell/systemd/CI/
justfile/mise.toml/Dockerfiles, and `docs/vision/slice-machine-boundary-validator.md`.

## Findings

**[MEDIUM] `assert` puts a raise-capable statement inside a validator
documented as "never raises," on a path with no accept-loop exception
handler.** `docs/vision/slice-machine-boundary-validator.md:56`: "No exception
propagation across the boundary: the validator returns a result type; it
never raises." `_validate_request` is invoked at `server.py:426` outside any
`try`; `run_server` (`server.py:1212-1219`) is `try/finally` with **no**
`except`; the outermost handler (`generate.py:3073`) catches only
`(FileNotFoundError, PermissionError)`. Any raise from this function
therefore terminates the daemon and unlinks its socket. The assert is
unreachable given today's validator (every `ok=False` path in
`params_validation.py` terminates in `_make_err`, which always populates
`error` — verified across all nine `return False, X` paths in `_check_field`
plus `validate_lora_entry` and `validate_ref_image_entry`), so this is not
exploitable today. The risk is a future validator refactor converting a
should-be-rejected request into a one-packet daemon kill, and that the
sibling MCP plane already solved this exact narrowing non-raisingly
(`err = val.error or {}` at `mcp_server.py:1952`/`:2237`) — the daemon plane
diverging from that established idiom for no gain.

**[INFO] The `-O` concern is real in principle but null in impact here.**
Under `python -O` the assert is stripped and `err["field"]` would raise
`TypeError` instead of `AssertionError`. Since neither is caught anywhere
between `_validate_request` and process exit, the observable outcome is
identical either way — daemon death, socket unlinked. No `-O`/`-OO`/
`PYTHONOPTIMIZE` exists in any launch surface in this repo (verified by
grep). Not a separate finding requiring its own fix — covered by the MEDIUM
above, since the recommended fix is also `-O`-immune.

**[INFO] Invariant verified sound; the original fix was a neutral wash on
runtime safety, not a regression.** `err["field"]` with `err=None` was not
reachable before or after the assert — a pure narrowing-gap fix with no
runtime component either way. The objection is the never-raises invariant
erosion (MEDIUM), not suppression-shaped risk.

**[INFO] Symmetric unguarded Optional on the success branch (pre-existing,
not introduced by this change).** `req.update(result.payload)` at
`server.py:171` (now shifted) consumes the same dataclass's `Optional[dict]
payload` with no narrowing, relying on the mirror half of the same
docstring invariant. Flagged for whoever next drives pyright errors down
here; not actioned in this slice (this file's payload branch reports 0
pyright errors already — pyright doesn't flag it because `req.update()`
accepts `Optional[Mapping]`-compatible input structurally without erroring
on `None` the way subscripting does).

## Verdict

Not a blocker — apply the recommended one-line substitution before commit:

```python
err = result.error or {"field": "<root>", "reason": "validator returned no error detail"}
```

This satisfies pyright identically (type is `dict`, not `Optional[dict]`),
preserves the "never raises" property, fails closed (still rejects the
request with a structured error), needs no `isinstance` (keeping the N19 AST
invariant intact), and matches the MCP plane's established idiom for the
same dataclass.

## Resolution

Applied verbatim. `comfyless/server.py`'s `_validate_request` now reads:

```python
err = result.error or {"field": "<root>", "reason": "validator returned no error detail"}
if err["field"] == "<root>":
    ...
```

`mise exec -- pyright comfyless/server.py` — 0 errors (was 4).
`./.venv/bin/python3 test_server_robustness.py` — 202/202 passed, no
regressions.

As a follow-on, the SAME "never raises across the boundary" reasoning was
applied to `comfyless/params_validation.py`'s own internal `_check_field`
narrowing in the companion commit for that file (the validator's own
contract, not just its callers') — see
`docs/decisions/ADR-042-per-root-typecheck-baselines.md` and the commit
"feat: ADR-042 comfyless/ pyright drawdown — 6 plain files".
