#!/usr/bin/env bash
# Shared repo-policy check functions — git-state versions (ADR-012).
#
# These are the portable, everyone-applies-to counterparts to the AI-session
# .claude/hooks/*.sh (which parse the model's Bash tool call). Here the input is
# REAL git state: a commit message string, a changed-file list, a staged diff.
# Both the pre-commit hooks and the CI range-checker source this file, so the
# logic can't drift between the two enforcement layers.
#
# Each pc_* function prints a BLOCKED message to stderr and returns 1 on failure,
# 0 on pass. Callers accumulate failures and exit non-zero if any failed.
#
# Red Zone paths come from the single source of truth in .claude/hooks (same list
# the harness hooks use) — no second copy to drift.

_gp_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_gp_repo_root="$(cd "$_gp_lib_dir" && git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
# Red Zone path list — single source of truth. FAIL CLOSED if absent: a silent
# missing file would make is_red_zone_path a no-op and disable BOTH Red Zone gates
# (code-reviewer 2026-07-16, SECURITY). When lifting these scripts to a repo
# without .claude/hooks/, bundle a _red-zone-paths.sh beside them and repoint this.
_gp_rz_paths="$_gp_lib_dir/_red-zone-paths.sh"
# shellcheck source=/dev/null
source "$_gp_rz_paths" 2>/dev/null \
    || { echo "FATAL: Red Zone path list not found at $_gp_rz_paths" >&2; exit 1; }
declare -F is_red_zone_path >/dev/null 2>&1 \
    || { echo "FATAL: is_red_zone_path undefined after sourcing $_gp_rz_paths" >&2; exit 1; }

# Conventional subject: prefix + <=72 chars + not a catch-all. Mirrors
# .claude/hooks/commit-msg-conventional.sh.
pc_conventional() {
    local subject="$1"
    # NOTE: no "Merge "-subject exemption here — a non-merge commit literally
    # titled "Merge …" must not skip the check (code-reviewer 2026-07-16). Real
    # merge commits are exempted by the CALLER (which can check parent count).
    if ! printf '%s' "$subject" | grep -qE '^(feat|fix|docs|test|refactor|tool|workflows|update|deps|chore):[[:space:]]'; then
        echo "BLOCKED: commit subject lacks a conventional prefix: '$subject'" >&2
        echo "  Use one of: feat: fix: docs: test: refactor: tool: workflows: update: deps: chore:" >&2
        return 1
    fi
    if [ "${#subject}" -gt 72 ]; then
        echo "BLOCKED: commit subject is ${#subject} chars (max 72): '$subject'" >&2
        return 1
    fi
    case "$subject" in
        wip|WIP|"wip"*|"WIP"*|"session work"*|"stuff"*|updates|"misc"*)
            echo "BLOCKED: commit subject is a catch-all: '$subject'" >&2
            return 1 ;;
    esac
    return 0
}

# AI-disclosure trailer present anywhere in the message body.
pc_ai_disclosure() {
    local message="$1"
    if printf '%s' "$message" | grep -qiE 'AI-disclosure:'; then
        return 0
    fi
    echo "BLOCKED: commit message lacks an 'AI-disclosure:' trailer (global CLAUDE.md §7)." >&2
    echo "  Add e.g.: AI-disclosure: Claude (<tier>) authored; Grant reviewed.  (or 'AI-disclosure: none')" >&2
    return 1
}

# No floor-style version specifiers in the given pyproject.toml content.
# Mirrors .claude/hooks/block-pyproject-floors.sh's bad_pattern.
pc_no_floors() {
    local content="$1"
    local bad='["'"'"'][A-Za-z0-9_.-]+(\[[^]]*\])?[[:space:]]*(>=?|<=?|~=|!=|\^|==[^[:space:]"'"'"']*\*|==[[:space:]]*["'"'"']?latest)'
    if printf '%s' "$content" | grep -qE "$bad"; then
        echo "BLOCKED: pyproject.toml introduces a floor-style version specifier (§11 exact pins only):" >&2
        printf '%s' "$content" | grep -E "$bad" | head -5 >&2
        return 1
    fi
    return 0
}

# No TECH_DEBT.md entry header (## ) removed in the given unified=0 staged diff.
pc_tech_debt_no_deletion() {
    local diff="$1"
    local removed
    removed=$(printf '%s' "$diff" | grep -E '^-## ' | grep -v '^---' || true)
    if [ -n "$removed" ]; then
        echo "BLOCKED: TECH_DEBT.md entry header(s) deleted (append 'Resolved:' instead, §12):" >&2
        printf '%s\n' "$removed" >&2
        return 1
    fi
    return 0
}

# Typecheck-ratchet baseline may only decrease, PER ROOT (ADR-032 posture;
# ADR-042 per-root aggregation; code-reviewer 2026-07-16 MEDIUM: a same-commit
# baseline bump must not self-legalize new type errors — still holds per
# root). $1 = old file content, $2 = new file content. Content is `root=count`
# lines (comment lines starting with '#' and blanks are ignored); a bare
# integer (the pre-ADR-042 format) has no `root=` lines and is treated as "no
# history for any root" — nothing to compare against, so nothing blocks. A
# root present in only one side (newly introduced, or dropped) is not
# compared either.
pc_baseline_no_increase() {
    local old="$1" new="$2"
    local roots root old_count new_count rc=0
    roots=$({ printf '%s\n' "$old"; printf '%s\n' "$new"; } \
        | grep -oE '^[A-Za-z0-9_./-]+=' | tr -d '=' | sort -u)
    while IFS= read -r root; do
        [ -z "$root" ] && continue
        old_count=$(printf '%s\n' "$old" | grep -E "^${root}=" | tail -1 | cut -d= -f2 | tr -dc '0-9')
        new_count=$(printf '%s\n' "$new" | grep -E "^${root}=" | tail -1 | cut -d= -f2 | tr -dc '0-9')
        [ -z "$old_count" ] && continue   # introducing this root is fine
        [ -z "$new_count" ] && continue   # root removed/garbled — not a ratchet bump
        if [ "$new_count" -gt "$old_count" ]; then
            echo "BLOCKED: .claude/typecheck-baseline[$root] raised $old_count -> $new_count (ADR-032/ADR-042: ratchet only goes down)." >&2
            echo "  Fix the new type errors, or use the documented override for a deliberate bump." >&2
            rc=1
        fi
    done <<< "$roots"
    return $rc
}

# If any changed file is Red Zone, the message must reference an existing
# ADR (kind=spec) or docs/security/review-*.md (kind=review), OR such a file
# must itself be in the changed set. Mirrors require-redzone-*.sh.
#
# ADAPTED from the kit (2026-07-16): the kit's spec pattern is docs/specs/*.md;
# this repo's spec-first artifact is the ADR (global §12 — "write the ADR →
# run security review → write code", commits reference
# docs/decisions/ADR-NNN-<slug>.md). No docs/specs/ exists here.
pc_redzone_ref() {
    local message="$1" changed_files="$2" kind="$3" repo_root="${4:-$_gp_repo_root}"
    local pattern label
    case "$kind" in
        spec)   pattern='docs/decisions/ADR-[A-Za-z0-9_.-]+\.md';   label='ADR (docs/decisions/ADR-*.md)';;
        review) pattern='docs/security/review-[A-Za-z0-9_.-]+\.md'; label='security review (docs/security/review-*.md)';;
        *) echo "pc_redzone_ref: bad kind '$kind'" >&2; return 1;;
    esac
    local has_rz=0 f
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        if is_red_zone_path "$f"; then has_rz=1; break; fi
    done <<< "$changed_files"
    [ "$has_rz" -eq 0 ] && return 0

    # A referenced file that exists in the repo passes.
    local ref
    ref=$(printf '%s' "$message" | grep -oE "$pattern" | head -1 || true)
    if [ -n "$ref" ] && [ -f "$repo_root/$ref" ]; then return 0; fi
    # A file of the right kind in the changed set passes (the artifact IS here).
    if printf '%s' "$changed_files" | grep -qE "^$pattern$"; then return 0; fi

    echo "BLOCKED: Red Zone change without a referenced $label (§Red Zone handling)." >&2
    echo "  Reference it in the commit body, or include the file in the commit." >&2
    return 1
}
