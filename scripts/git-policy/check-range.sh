#!/usr/bin/env bash
# CI mirror (ADR-012) — the authoritative, can't-be-`--no-verify`-bypassed layer.
# Validates every commit in <base>..<head> against the same repo-policy checks the
# pre-commit hooks run. Usage: check-range.sh <base-ref> <head-ref>.
#
# Runs on pull_request in CI (base = the PR's merge base, head = the PR tip).
set -uo pipefail
lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$lib_dir/_lib.sh"

base="${1:?usage: check-range.sh <base> <head>}"
head="${2:?usage: check-range.sh <base> <head>}"
repo_root="$(git rev-parse --show-toplevel)"

# Fail CLOSED if either endpoint is unresolvable — otherwise `git rev-list` yields
# an empty list and the job passes vacuously (code-reviewer 2026-07-16).
for ref in "$base" "$head"; do
    git cat-file -e "${ref}^{commit}" 2>/dev/null \
        || { echo "FATAL: not a resolvable commit: $ref" >&2; exit 1; }
done

empty_tree="$(git hash-object -t tree /dev/null)"
commits="$(git rev-list --reverse "$base..$head")"
n=$(printf '%s' "$commits" | grep -c . || true)
echo "commit-policy: checking $n commit(s) in $base..$head" >&2

rc=0
for sha in $commits; do
    subject="$(git log -1 --format=%s "$sha")"
    message="$(git log -1 --format=%B "$sha")"
    # Diff against the FIRST parent (empty tree for a root commit). This is what
    # captures a MERGE commit's net change vs the mainline — `git diff-tree`
    # emits nothing for a merge, which let an evil merge evade every content check.
    parents="$(git rev-list --parents -n1 "$sha")"
    nparents=$(( $(printf '%s' "$parents" | wc -w) - 1 ))
    if [ "$nparents" -ge 1 ]; then
        parent="$(printf '%s' "$parents" | awk '{print $2}')"
    else
        parent="$empty_tree"
    fi
    is_merge=0; [ "$nparents" -ge 2 ] && is_merge=1
    changed="$(git diff --name-only "$parent" "$sha")"

    range_rc=0
    printf '== %s%s %.60s ==\n' "$sha" "$([ "$is_merge" = 1 ] && echo ' (merge)')" "$subject" >&2

    # Message-format checks apply to authored commits, not git-generated merges.
    if [ "$is_merge" -eq 0 ]; then
        pc_conventional "$subject" || range_rc=1
        pc_ai_disclosure "$message" || range_rc=1
    fi

    # Content checks apply to EVERY commit (merges included — the fix), with the
    # overridable ones gated by an explicit `Policy-override:` trailer.
    if ! printf '%s' "$message" | grep -qiE '^Policy-override:'; then
        pc_redzone_ref "$message" "$changed" spec   "$repo_root" || range_rc=1
        pc_redzone_ref "$message" "$changed" review "$repo_root" || range_rc=1
        while IFS= read -r td; do
            [ -z "$td" ] && continue
            pc_tech_debt_no_deletion "$(git diff --unified=0 "$parent" "$sha" -- "$td")" || range_rc=1
        done < <(printf '%s\n' "$changed" | grep -E '(^|/)TECH_DEBT\.md$' || true)
        # Typecheck-ratchet baseline may only decrease (ADR-032).
        if printf '%s\n' "$changed" | grep -qx '.claude/typecheck-baseline'; then
            pc_baseline_no_increase \
                "$(git show "$parent:.claude/typecheck-baseline" 2>/dev/null || true)" \
                "$(git show "$sha:.claude/typecheck-baseline" 2>/dev/null || true)" || range_rc=1
        fi
    fi
    # Floors: check EVERY pyproject.toml (incl. nested, for monorepos).
    while IFS= read -r pp; do
        [ -z "$pp" ] && continue
        pc_no_floors "$(git diff "$parent" "$sha" -- "$pp" | grep '^+' | grep -v '^+++')" || range_rc=1
    done < <(printf '%s\n' "$changed" | grep -E '(^|/)pyproject\.toml$' || true)

    [ "$range_rc" -ne 0 ] && rc=1
done

if [ "$rc" -eq 0 ]; then
    echo "commit-range policy checks passed ($base..$head)." >&2
fi
exit $rc
