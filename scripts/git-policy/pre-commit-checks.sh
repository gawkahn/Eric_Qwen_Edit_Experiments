#!/usr/bin/env bash
# pre-commit `pre-commit`-stage hook (ADR-012). Runs the staged-content checks
# that don't need the commit message: no dep floors (any pyproject.toml, incl.
# nested), and TECH_DEBT.md append-only. Local override: `SKIP=repo-policy-staged`.
set -uo pipefail
lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$lib_dir/_lib.sh"

staged="$(git diff --cached --name-only)"
rc=0

# Floors — check ADDED lines only (mirrors the harness hook; avoids flagging the
# pre-existing build-backend `uv_build>=` range, ADR-006). Every pyproject.toml,
# including nested ones (monorepo-safe).
while IFS= read -r pp; do
    [ -z "$pp" ] && continue
    pc_no_floors "$(git diff --cached -- "$pp" | grep '^+' | grep -v '^+++')" || rc=1
done < <(printf '%s\n' "$staged" | grep -E '(^|/)pyproject\.toml$' || true)

# TECH_DEBT append-only — every TECH_DEBT.md (incl. nested).
while IFS= read -r td; do
    [ -z "$td" ] && continue
    pc_tech_debt_no_deletion "$(git diff --cached --unified=0 -- "$td")" || rc=1
done < <(printf '%s\n' "$staged" | grep -E '(^|/)TECH_DEBT\.md$' || true)

# Typecheck-ratchet baseline may only decrease (ADR-032). Staged vs HEAD.
if printf '%s\n' "$staged" | grep -qx '.claude/typecheck-baseline'; then
    pc_baseline_no_increase \
        "$(git show HEAD:.claude/typecheck-baseline 2>/dev/null || true)" \
        "$(git show :.claude/typecheck-baseline 2>/dev/null || true)" || rc=1
fi

exit $rc
