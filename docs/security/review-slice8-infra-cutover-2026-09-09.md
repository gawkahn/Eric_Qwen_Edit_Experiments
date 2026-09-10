# Infra audit — ADR-045 slice 8 infrastructure cutover

AI-Disclosure: Claude (Opus 5) authored; Grant reviewed.
Agent: `infra-auditor` (pinned `claude-fable-5` via frontmatter, no model override
passed — per global §5A, the pin is honored as of the 2026-08-08 re-verification).
Date: 2026-09-09
Scope: the uncommitted diff of `systemd/comfyless@.service` and `start-mcpo.sh`.

## Verdict

**PASS — no MUST findings.** The diff does what it claims, preserves every
property established by `review-systemd-daemon-unit-2026-07-17.md`, and
measurably shrinks the import surface on both entry points.

## What changed

The systemd user unit and the mcpo launcher were repointed from this repo's venv
(`-m comfyless.generate` + a `PYTHONPATH` export) to the sibling repo's console
scripts (`comfyless_diffusion/.venv/bin/comfyless{,-mcp}`). `WorkingDirectory`
moved with them; the inert `Environment=PYTHONPATH=` line was deleted.

## Net posture: improves

Console-script execution puts the *script's* directory on `sys.path[0]`, not the
CWD — verified empirically against the sibling venv's 3.14.7 interpreter. The old
`-m` form put the repo root on `sys.path`, and the deleted `export PYTHONPATH`
put the node-repo root (dozens of top-level `test_*.py`, `comfy_stub.py`) on the
path of both the uvx-run mcpo process and the spawned MCP server. Both channels
are now closed.

Preserved from the 2026-07-17 review: SHOULD 1 (`StartLimitIntervalSec`/`Burst`)
retained; SHOULD 3 (`NoNewPrivileges`, `RestrictAddressFamilies=AF_UNIX`,
`ProtectSystem=full`) retained, path-agnostic, and empirically live — both
instances active under the new `ExecStart`. `ProtectSystem=full` makes only
`/usr`, `/boot`, `/etc` read-only; every path the new venv needs is outside it.

Trust chain checked to ground: every directory from `/home/gawkahn` down to
`.venv/bin` is owned `gawkahn:gawkahn`; the `gawkahn` group's sole member is
`gawkahn`, so the 775 group-write bits are inert; `/home/gawkahn/projects` is
mode 700. The shebang pins the same interpreter the old explicit form invoked,
and `entry_points.txt` maps the scripts to `comfyless.generate:main` /
`comfyless.mcp_server:main` — the identical modules the old `-m` forms ran. A
future `uv sync` that regenerated the venv without `[project.scripts]` would
fail the unit closed, capped by the retained start limit.

## Findings

**[SHOULD 1] This diff is the named trigger for the open socket-hijack debt.**
The sibling register's `run_server` socket-steal entry sets its trigger as "the
next `--serve` / systemd-unit change" — this is that change. The fix is a Red
Zone `server.py` change and correctly does NOT belong in this diff, but the
trigger must be honored explicitly rather than passed over in silence.
*Closed: re-deferred with a dated note in `comfyless_diffusion/TECH_DEBT.md`.*

**[SHOULD 2] Production ran the uncommitted diff, and a revert fails silently
open.** The old `ExecStart` form still *works* — the node venv resolves
`comfyless` editable to the same sibling `src`, and its `.venv/bin` still carries
all six `comfyless*` console scripts via the path dependency. So a concurrent
session reverting the tree would silently drop the daemon back onto the node
repo's divergent transitive tree (37/105 packages differ) with no breakage
signal. *Closed: committed promptly; `Resolved:` appended to the
lockfile-divergence entry in both registers; the launcher comment now states the
hazard explicitly instead of claiming the node repo "no longer contains
comfyless at all".*

**[INFO 1]** The change partially closes the `folder_paths` live-import MEDIUM
from `review-slice-5-src-layout-2026-09-09.md`: the daemon and mcpo reachability
halves are gone. Only "operator runs `python3 -m comfyless.*` from an untrusted
directory" remains.

**[INFO 2–4]** Sandbox directives still effective and nothing new blocked; no
trust-boundary change from executing the sibling's console script; mcpo's
pre-existing `172.17.0.1:8090` no-API-key bind is byte-identical after the diff
(flagged as unchanged, not introduced) and its `sys.path` exposure is reduced.

**[INFO 5]** Two stale comment paths (`comfyless/pause.py`,
`comfyless/integrations/openwebui/…`) and one imprecise new comment.
*Closed: all three corrected in this commit.*

## Not reviewed

`comfyless` server/MCP source behaviour (unchanged by this diff — both venvs'
editable installs point at the same `src` bytes); ADR-045 full text; mcpo under
load (it was not running at audit time, so the new spawn line is unverified
live — though it was separately started, served all six tools on
`/openapi.json`, and shut down cleanly during this slice); the OWUI-container
side of the bridge.
