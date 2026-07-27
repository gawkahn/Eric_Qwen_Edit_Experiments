#!/usr/bin/env bash
# Static-analysis ratchet (global CLAUDE.md §15; ADR-032 posture, ADR-042
# per-root aggregation) — descended from the quality-gate kit template
# ~/.claude/templates/hooks/require-typecheck-clean.sh.
#
# Runs pyright before a `git commit` and BLOCKS when ANY root reports MORE
# diagnostics than that root's committed baseline
# (.claude/typecheck-baseline, `root=count` lines). Each root may only
# ratchet DOWN independently.
#
# T1 for this repo, but LOCAL-CONVENIENCE only: fails OPEN if the checker
# cannot run (not installed, ENOENT, unparseable output). The AUTHORITATIVE
# gate is the CI `typecheck` job (T3), which fails CLOSED.
#
# Override: append `# user-approved` to the bash command (rare — e.g. a
# deliberate baseline bump landing in the same commit).

set -euo pipefail

# ─── project-specific config (ADR-032/ADR-042) ──────────────────────────────
PER_ROOT_SCRIPT="${CLAUDE_PROJECT_DIR:-.}/scripts/typecheck-per-root.sh"
BASELINE_FILE="${CLAUDE_PROJECT_DIR:-.}/.claude/typecheck-baseline"
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

# Run from the project root — a `git -C <repo> commit` from elsewhere must not
# typecheck the wrong tree (0 errors from an empty cwd would pass trivially;
# a different repo's count could wedge a clean commit — code-reviewer 2026-07-16).
cd "${CLAUDE_PROJECT_DIR:-.}"

# Fail OPEN on any inability to produce per-root counts — the local hook is
# convenience; CI is the real gate.
if [ ! -x "$PER_ROOT_SCRIPT" ]; then
    echo "[typecheck-ratchet] $PER_ROOT_SCRIPT missing/not executable — failing OPEN. CI (T3) is the authoritative gate." >&2
    exit 0
fi

current_lines=$("$PER_ROOT_SCRIPT" 2>/dev/null) || true
if [ -z "$current_lines" ]; then
    echo "[typecheck-ratchet] no per-root counts produced — failing OPEN. CI (T3) is the authoritative gate." >&2
    exit 0
fi

# Baseline comes from HEAD, not the working tree — otherwise a commit that
# introduces new errors AND bumps a root's count in the same commit would
# self-legalize past the ratchet (code-reviewer 2026-07-16, MEDIUM, carried
# forward to the per-root form). A root with no HEAD baseline (new root, or
# HEAD still on the pre-ADR-042 bare-integer format) is not compared — there
# is nothing to ratchet against yet.
head_baseline=$(git show HEAD:.claude/typecheck-baseline 2>/dev/null || true)

blocked=0
report=""
while IFS='=' read -r root current; do
    [ -z "$root" ] && continue
    current=$(printf '%s' "$current" | tr -dc '0-9')
    [ -z "$current" ] && continue
    # `|| true`: under `set -e`, a `grep` that finds no match (a root absent
    # from head_baseline — the whole point of the "no prior baseline" branch
    # below, and the ONLY case on the transition commit itself) makes the
    # pipeline's exit status non-zero via pipefail, which would abort this
    # script mid-loop and skip every remaining root (code-reviewer 2026-07-27).
    old=$(printf '%s\n' "$head_baseline" | grep -E "^${root}=" | tail -1 | cut -d= -f2 | tr -dc '0-9' || true)
    if [ -z "$old" ]; then
        report="${report}[typecheck-ratchet] $root: $current (no prior baseline for this root — not compared)"$'\n'
        continue
    fi
    if [ "$current" -gt "$old" ]; then
        report="${report}BLOCKED: $root regressed — $current diagnostics, baseline is $old."$'\n'
        blocked=1
    elif [ "$current" -lt "$old" ]; then
        report="${report}[typecheck-ratchet] $root: $current < baseline $old — good, lower it in $BASELINE_FILE."$'\n'
    else
        report="${report}[typecheck-ratchet] $root: $current (unchanged)"$'\n'
    fi
done <<< "$current_lines"

printf '%s' "$report" >&2

if [ "$blocked" -eq 1 ]; then
    cat >&2 <<EOF

A commit may only lower a root's count (global CLAUDE.md §15 ratchet; ADR-032/ADR-042).
Fix the new diagnostic, or — if you deliberately raised the baseline —
update $BASELINE_FILE in this same commit.

Override (rare): append \`# user-approved\` to the bash command.
EOF
    exit 2
fi

exit 0
