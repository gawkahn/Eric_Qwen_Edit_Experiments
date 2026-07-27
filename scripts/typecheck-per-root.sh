#!/usr/bin/env bash
# Per-root pyright ERROR counts (ADR-042). Prints `root=count` lines matching
# .claude/typecheck-baseline's format, one line per top-level root that
# actually appears in pyright's output. Shared by the local ratchet hook
# (.claude/hooks/require-typecheck-clean.sh) and the CI typecheck job so the
# file->root grouping logic can't drift between the two enforcement layers.
#
# Full pyright output goes to stderr for debugging; stdout is ONLY the
# `root=count` lines, so callers can capture it directly.
#
# Fails CLOSED (exit 1, nothing on stdout) if pyright's own summary line is
# absent from its output — a crashed/misconfigured checker must not be read
# as "zero errors everywhere" by any caller (code-reviewer 2026-07-27).
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$repo_root" ]; then
    echo "typecheck-per-root: not in a git repo (git rev-parse --show-toplevel failed) — refusing to emit counts" >&2
    exit 1
fi
cd "$repo_root"

# Testability escape hatch: feed canned pyright output instead of invoking
# the real checker. Used by scripts/git-policy/test-git-policy.sh so this
# script's parsing/bucketing logic has coverage without a live pyright run.
if [ -n "${PYRIGHT_OUTPUT_FILE:-}" ]; then
    out=$(cat "$PYRIGHT_OUTPUT_FILE")
else
    out=$(mise exec -- pyright 2>&1) || true
fi
printf '%s\n' "$out" >&2

# Require pyright's own summary line before trusting anything parsed from its
# output — a crash, a broken [tool.pyright] config, or an ENOENT would
# otherwise produce an empty `files` match and every root would silently read
# as "0 errors", which every caller (hook AND CI) treats as clean.
printf '%s\n' "$out" | grep -qE '^[0-9]+ errors?, [0-9]+ warning' || {
    echo "typecheck-per-root: no pyright summary line in output — refusing to emit counts" >&2
    exit 1
}

# ERROR diagnostic lines only (not warnings/informations — ADR-032/CI both
# talk about "error count") look like:
#   "  /abs/repo/root/rest/of/path.py:LINE:COL - error: ...".
# File-group header lines (no leading whitespace, no ":LINE:COL - error:")
# don't match, so each match is exactly one error diagnostic.
files=$(printf '%s\n' "$out" \
    | grep -E '^[[:space:]]+/[^:]+\.py:[0-9]+:[0-9]+ - error:' \
    | sed -E 's/^[[:space:]]+//; s/:[0-9]+:[0-9]+ - error:.*$//' \
    | sed "s|^${repo_root}/||")

# Bucket by first path segment (the root) DYNAMICALLY, not a hardcoded list —
# a root gaining/losing entries under [tool.pyright] include is picked up
# automatically. CI's "no baseline entry for root X" check is what fails
# closed on an unexpected/renamed root, not a fixed list here (a fixed list
# in four places — pyproject, the baseline file, this script, a keep-in-sync
# comment — was itself a silent-drop risk, code-reviewer 2026-07-27).
printf '%s\n' "$files" | grep -v '^$' | sed -E 's|/.*||' | sort | uniq -c \
    | while read -r count root; do echo "${root}=${count}"; done
