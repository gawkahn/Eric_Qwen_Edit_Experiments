#!/usr/bin/env bash
# Build an isolated, HASH-VERIFIED environment for a CI tool, and print its bin dir.
#
# Why this exists rather than `uv run --with <pkg>==<ver>`: `--with` pins only
# the named package. semgrep's tree resolves 66 further packages fresh from PyPI
# on every run, unhashed — code executing on a runner that has this private,
# Red-Zone-bearing source checked out. That was the one place the repo departed
# from the `uv sync --locked` posture §11 requires everywhere else.
#
# `uv run --with-requirements <lockfile>` is NOT sufficient either, and this is
# the trap: it accepts a lock whose hashes do not match and installs anyway
# (measured 2026-09-16 — `uv run` has no --require-hashes flag and ignores
# UV_REQUIRE_HASHES). Only `uv pip install --require-hashes` actually verifies,
# so the environment must be built explicitly first.
#
# Regenerate a lock after bumping its .in file:
#   uv pip compile --universal --generate-hashes --python-version 3.14 \
#       scripts/ci-tools/<name>.in -o scripts/ci-tools/<name>.txt
set -euo pipefail

name="${1:?usage: build-tool-env.sh <tool-name>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lock="$here/${name}.txt"
[ -f "$lock" ] || { echo "build-tool-env: no lock file at $lock" >&2; exit 1; }

# NB: an `[ -n "$X" ] && assign` one-liner here is a `set -e` landmine — when
# the var is unset the test returns 1 and the whole script exits 2 with no
# output, which is exactly how this first failed.
if [ -n "${TOOL_ENV_ROOT:-}" ]; then
    env_dir="$TOOL_ENV_ROOT/$name"
else
    env_dir="$(cd "$here/../.." && pwd)/.tool-envs/$name"
fi
# --clear: uv refuses to write into an existing venv, so without it every run
# after the first fails. Rebuilding from scratch is also the behaviour we want
# — a reused env could hold a package the current lock no longer pins.
# stdout is silenced (the caller captures this script's stdout as the bin path)
# but stderr is NOT: swallowing it is how the --clear failure first presented
# as a silent exit 2.
# Interpreter comes from the repo's own .python-version, not a literal: this
# script is shared with the node pack, which is on 3.12 while this repo is on
# 3.14. A hardcoded version would build the tool env on the wrong interpreter
# there and the locks (compiled per --python-version) would not match.
repo_root="$(cd "$here/../.." && pwd)"
pyver="$(cat "$repo_root/.python-version" 2>/dev/null || echo 3.12)"
uv venv --clear --python "$pyver" "$env_dir" >/dev/null
# --require-hashes is the whole point; --no-cache so a previously-cached wheel
# cannot satisfy the install without being re-verified against the lock.
uv pip install --python "$env_dir/bin/python" --require-hashes --no-cache \
    -r "$lock" >/dev/null
# Callers must put this FIRST on PATH, not merely invoke the absolute path.
# semgrep's launcher resolves its core through PATH, so on a machine with
# another semgrep installed (this dev box has 1.163.0 in ~/.local/bin) the
# pinned 1.169.0 wrapper silently runs the OTHER core. A CI runner has no
# system semgrep so it cannot hit this, which is exactly why it would have gone
# unnoticed. Measured 2026-09-16.
echo "$env_dir/bin"
