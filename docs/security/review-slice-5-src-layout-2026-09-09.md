# Security review — ADR-045 slice 5 (src layout + packaging)

AI-Disclosure: `security-auditor` (Claude Fable 5, model confirmed
`claude-fable-5` in the agent transcript — no Fable→Opus fallback) reviewed;
Claude Opus 5 authored the slice and this write-up; Grant reviewed.

Date: 2026-09-09
Slice: ADR-045 slice 5 — `comfyless/` → `src/comfyless/`, `[build-system]`
(hatchling), six console scripts, shim deletion.
Trigger: the change moves three §12 Red Zone paths — `comfyless/server.py`
(Unix-socket IPC daemon, ADR-001), `comfyless/mcp_server.py` (MCP/LLM tool
surface, ADR-011) and `comfyless/refine.py` (LLM-judge + seed ingestion,
ADR-027). Project review bar: any change to a `_red-zone-paths.sh` path also
runs `security-auditor`.

## Scope reviewed

Index state (not the working tree): blob-hash identity of the three Red Zone
files across the rename; the full content delta of the only two
content-changed files under `src/comfyless`; the staged `_red-zone-paths.sh`
sourced and EXECUTED against real post-move paths and eight near-misses; every
remaining `folder_paths` / `comfy.*` / `sys.path` reference in the staged tree;
all six `[project.scripts]` targets; the `uv.lock` diff screened for version or
hash changes; `systemd/comfyless@.service`, `start-mcpo.sh`, `justfile`,
`.github/workflows/ci.yml`, `.claude/typecheck-baseline`, `scripts/lora_audit.py`,
`CLAUDE.md`, `TECH_DEBT.md`.

Not reviewed, with reason: the 76 R100-renamed files (byte-identical by blob
hash — unchanged by construction); hatchling 1.32.0's own dependency metadata
(not fetched; named as an assumption in finding 2).

## Verified clean

1. **Content preservation.** `server.py`, `mcp_server.py` and `refine.py` are
   byte-identical across the rename — blob `2e1084c5…`, `3486b7b1…`,
   `6bb76520…`, equal at `HEAD:comfyless/<f>` and `:src/comfyless/<f>`. The
   only content-changed files under `src/comfyless` are `__init__.py` (the
   declared shim deletion) and `generate.py` (deletion of the one-line
   `import comfyless` shim trigger; `_run_json_mode` untouched).
2. **The Red Zone commit gate still fires.** Verified by execution, not
   inspection, on all three post-move paths, with correct rejection of
   `src/comfyless/server_py`, `src/comfyless/xserver.py` and
   `src/comfyless/generate.py`. The e2e evil-merge test now plants
   `src/comfyless/server.py`. `just policy-test` 43/43.
3. **`sys.path` insert deletion strictly REDUCES import surface** — it added a
   directory; nothing replaces it.
4. **Console scripts open no new surface.** All six targets parse argv exactly
   as their `python -m` forms did. Marginal improvement, in fact: a console
   script puts the venv `bin/` at `sys.path[0]` rather than cwd, so it does not
   carry the cwd hazard `-m` does (finding 1).
5. **`uv.lock` contains zero version or hash changes** — only
   `virtual`→`editable` and lock-regeneration marker simplification.
6. **`comfy_stub.py`** is test-side, idempotent, and excluded from the wheel by
   the `packages = ["src/comfyless"]` target.

## Findings and disposition

### MEDIUM — shim deletion drops the accidental shadowing of `folder_paths`

`src/comfyless/core/eric_diffusion_utils.py:400` lazily imports
`folder_paths` inside `try/except Exception`. Before this slice, importing
anything under `comfyless` had already put a benign stub in
`sys.modules["folder_paths"]`, so that import could only resolve to the stub.
It now resolves live from `sys.path`; under `python -m comfyless.*` cwd is on
that path, so a `folder_paths.py` in an untrusted working directory would
execute at import time, and the broad `except Exception` would swallow what it
raised.

**Disposition: DEFERRED, recorded.** TECH_DEBT 2026-09-09
("`resolve_component_path` imports `folder_paths` from live `sys.path`"),
trigger = next slice touching `comfyless/core/`, no later than slice 7. Fix is
one line (`sys.modules.get("folder_paths")`). Deferred on the auditor's own
recommendation: this slice's central claim is that every moved file is
byte-identical across the rename, and editing a moved file to fix a
non-urgent issue would forfeit exactly that proof. Reachability today is narrow
— the daemon and mcpo both run from trusted working directories, and the new
console scripts are not exposed at all.

### MEDIUM — the hatchling build-dependency debt entry understated the risk

Build isolation resolves hatchling's own dependencies as floating, unhashed
ranges (larger surface than the single exact pin suggests), and with
`source = { editable = "." }` in `uv.lock` the backend is resolved and executed
on every cold-cache `uv sync` — not only on an explicit `uv build`.

**Disposition: FIXED IN THIS SLICE** — the TECH_DEBT 2026-09-09 hatchling entry
carries an amendment naming both axes, so the slice-6 / `deps-report` trigger
evaluates the real risk.

### LOW (from `code-reviewer`) — the path rewrite had NARROWED the gate

The sweep rewrote `(^|/)comfyless/server\.py$` to `(^|/)src/comfyless/server\.py$`.
The old pattern matched both spellings by suffix; the new one matched only the
new path, so `check-range` over a commit range spanning the move would have
stopped content-checking pre-move commits, and a file resurrected at the old
path (a partial revert) would have bypassed the gate.

**Disposition: FIXED IN THIS SLICE** — patterns are now
`(^|/)(src/)?comfyless/<file>\.py$`, with the reasoning recorded in the script.
Re-verified by execution: both spellings gate. Note this restores the
pre-existing over-match on paths like `notsrc/comfyless/server.py` — a false
POSITIVE that fails closed (it demands an ADR reference for a file that is not
Red Zone), and status quo ante rather than a regression.

### INFO — stale claims left behind by the deletion

A `generate.py` docstring asserted that `comfyless.__init__` had "already
stubbed" `folder_paths`; two comments referred to shims that no longer exist;
`comfy_stub.py`'s docstring overclaimed ("never displaces a real ComfyUI on
`sys.path`" — the guard is `sys.modules` membership, so an importable but
not-yet-imported ComfyUI IS masked).

**Disposition: ALL FIXED IN THIS SLICE.**

### INFO — retained `PYTHONPATH` in the daemon unit and mcpo launcher

Repo root ahead of site-packages keeps `nodes`, `pipelines`, `comfy_stub` and
every root-level `test_*.py` importable inside the daemon and mcpo processes.
Status quo, no regression from this slice.

**Disposition: DEFERRED to slice 8** (which replaces the unit with the console
script). Flagged here so slice 8 treats the removal as the security cleanup it
is, not a cosmetic one.

## Pre-existing gap, explicitly NOT closed here

`_red-zone-paths.sh` still names `nodes/eric_diffusion_fp8_ops.py` (moved to
`comfyless/core/` in slice 1b) and still omits `lora_adapters.py`. Confirmed
unchanged and not widened by this slice — the gate never matched
`comfyless/core/` before it either. TECH_DEBT 2026-08-22, now amended with the
post-move paths. Deferred per global §4: the policy layer is its own change
boundary.
