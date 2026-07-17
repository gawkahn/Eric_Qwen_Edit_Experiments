#!/usr/bin/env bash
# Static-analysis ratchet (global CLAUDE.md §15; ADR-032) — from the
# quality-gate kit template ~/.claude/templates/hooks/require-typecheck-clean.sh.
#
# Runs pyright before a `git commit` and BLOCKS when it reports MORE
# diagnostics than the committed baseline integer (.claude/typecheck-baseline).
# The count may only ratchet DOWN. When the baseline reaches 0 this becomes a
# hard must-be-clean gate.
#
# T1 for this repo, but LOCAL-CONVENIENCE only: fails OPEN if the checker
# cannot run (not installed, ENOENT, unparseable output). The AUTHORITATIVE
# gate is the CI `typecheck` job (T3), which fails CLOSED.
#
# Override: append `# user-approved` to the bash command (rare — e.g. a
# deliberate baseline bump landing in the same commit).

set -euo pipefail

# ─── project-specific config (ADR-032) ──────────────────────────────────────
TYPECHECK_CMD="mise exec -- pyright"   # scope comes from pyproject [tool.pyright]
BASELINE_FILE="${CLAUDE_PROJECT_DIR:-.}/.claude/typecheck-baseline"
# pyright prints "N errors, M warnings, K informations".
extract_count() { grep -oE '[0-9]+ error' | head -1 | grep -oE '[0-9]+' || true; }
# ────────────────────────────────────────────────────────────────────────────

input=$(cat)
tool_name=$(printf '%s' "$input" | jq -r '.tool_name // ""')
[ "$tool_name" = "Bash" ] || exit 0

command=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')

# Only act on `git commit` (any form; the ratchet is message-independent).
printf '%s' "$command" | grep -qE 'git[[:space:]]+(-C[[:space:]]+\S+[[:space:]]+)?commit' || exit 0

# Override token (trailing-anchored, matching the repo's other hooks).
if printf '%s' "$command" | grep -qE '[[:space:]]+#[[:space:]]+user-approved[[:space:]]*$'; then
    exit 0
fi

# Run the checker. Fail OPEN on any inability to produce a numeric count —
# the local hook is convenience; CI is the real gate.
output=$(eval "$TYPECHECK_CMD" 2>&1) || true
current=$(printf '%s' "$output" | extract_count)
if ! printf '%s' "$current" | grep -qE '^[0-9]+$'; then
    echo "[typecheck-ratchet] '$TYPECHECK_CMD' produced no error count — failing OPEN. CI (T3) is the authoritative gate." >&2
    exit 0
fi

baseline=0
if [ -f "$BASELINE_FILE" ]; then
    baseline=$(tr -dc '0-9' < "$BASELINE_FILE")
    [ -n "$baseline" ] || baseline=0
fi

if [ "$current" -gt "$baseline" ]; then
    cat >&2 <<EOF
BLOCKED: type-check regressed — $current diagnostics, baseline is $baseline.

    $TYPECHECK_CMD

A commit may only lower the count (global CLAUDE.md §15 ratchet; ADR-032).
Fix the new diagnostic, or — if you deliberately raised the baseline —
update $BASELINE_FILE in this same commit.

Override (rare): append \`# user-approved\` to the bash command.
EOF
    exit 2
fi

if [ "$current" -lt "$baseline" ]; then
    echo "[typecheck-ratchet] $current < baseline $baseline — good. Lower $BASELINE_FILE to $current in this commit to tighten the ratchet." >&2
fi

exit 0
