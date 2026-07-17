#!/usr/bin/env bash
# Smoke tests for the git-policy check functions (ADR-012 §4). Parallel to
# .claude/hooks/test-hooks.sh (which tests the harness/AI-command versions).
# Run: bash scripts/git-policy/test-git-policy.sh   (also the CI `hooks` job).
set -uo pipefail
lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$lib_dir/_lib.sh"
repo_root="$(git rev-parse --show-toplevel)"

pass=0; fail=0
ok()   { if "$@" >/dev/null 2>&1; then pass=$((pass+1)); else fail=$((fail+1)); echo "FAIL (expected pass): $*"; fi; }
no()   { if "$@" >/dev/null 2>&1; then fail=$((fail+1)); echo "FAIL (expected block): $*"; else pass=$((pass+1)); fi; }

# --- conventional subject ---
ok pc_conventional "feat: slice 8g — tier-3 hash enforcement"
# A bare "Merge …" subject is NOT exempt in pc_conventional itself — the merge
# exemption lives in the caller (gated on real parent count), so a non-merge
# commit titled "Merge …" is correctly blocked here.
no pc_conventional "Merge branch 'x'"
no pc_conventional "add a thing"                       # no prefix
no pc_conventional "wip"                               # catch-all
no pc_conventional "feat: $(printf 'x%.0s' {1..80})"   # >72 chars

# --- AI-disclosure ---
ok pc_ai_disclosure $'feat: x\n\nbody\n\nAI-disclosure: Claude (Opus) authored; Grant reviewed.'
ok pc_ai_disclosure $'feat: x\n\nAI-disclosure: none'
no pc_ai_disclosure $'feat: x\n\nbody with no trailer'

# --- no dep floors (given added-line content) ---
ok pc_no_floors '+    "openai==2.32.0",'
no pc_no_floors '+    "openai>=2.32.0",'
no pc_no_floors "+    'httpx~=0.28',"
no pc_no_floors '+    "foo[extra]^2.0",'

# --- TECH_DEBT append-only (given a unified=0 diff) ---
ok pc_tech_debt_no_deletion $'@@ -1 +2 @@\n+Resolved: 2026-07-16 — done'
no pc_tech_debt_no_deletion $'@@ -1 +0 @@\n-## 2026-01-01 — some entry header'

# --- typecheck-ratchet baseline (ADR-032: may only decrease) ---
ok pc_baseline_no_increase "1026" "1026"      # unchanged
ok pc_baseline_no_increase "1026" "900"       # drawdown
ok pc_baseline_no_increase ""     "1026"      # introducing the file
no pc_baseline_no_increase "1026" "1076"      # bump — blocked
ok pc_baseline_no_increase "1026\n" " 900 "   # whitespace/garble tolerated

# --- more floor forms ---
no pc_no_floors '+    "baz==latest",'
no pc_no_floors '+    "bar==2.*",'
no pc_no_floors "+    'qux != 1.0',"

# --- Red Zone spec (=ADR, per the _lib.sh adaptation) references ---
ok pc_redzone_ref "no ref needed" "README.md"                     spec   "$repo_root"  # not RZ
ok pc_redzone_ref "see docs/decisions/ADR-001-daemon-socket-security.md" "comfyless/server.py" spec "$repo_root"
no pc_redzone_ref "no reference at all" "comfyless/server.py"     spec   "$repo_root"  # RZ, no ref
ok pc_redzone_ref "docs/decisions/ADR-011-comfyless-mcp-server.md" "comfyless/mcp_server.py" spec "$repo_root"
# A reference to a NON-existent ADR must NOT satisfy the gate (guards the [ -f ]
# existence check — the slice-11 HIGH-2 defense).
no pc_redzone_ref "TODO: write docs/decisions/ADR-999-ghost.md" "comfyless/server.py" spec "$repo_root"
# Function-scoped surfaces (_run_json_mode, resolve_hf_path) are deliberately
# NOT path-gated — their whole files must stay non-RZ (see _red-zone-paths.sh).
ok pc_redzone_ref "no ref needed" "comfyless/generate.py"          spec "$repo_root"
ok pc_redzone_ref "no ref needed" "nodes/eric_diffusion_utils.py"  spec "$repo_root"
# The other two listed surfaces are RZ.
no pc_redzone_ref "no reference" "comfyless/refine.py"             spec "$repo_root"
no pc_redzone_ref "no reference" "nodes/eric_diffusion_fp8_ops.py" spec "$repo_root"

# --- Red Zone review references (the whole `review` kind was previously untested) ---
ok pc_redzone_ref "no ref needed" "README.md" review "$repo_root"
ok pc_redzone_ref "see docs/security/review-comfyless-server-2026-04-23.md" "comfyless/server.py" review "$repo_root"
no pc_redzone_ref "no reference" "comfyless/server.py" review "$repo_root"
no pc_redzone_ref "docs/security/review-ghost.md" "comfyless/server.py" review "$repo_root"

# --- end-to-end: check-range must content-check a MERGE commit (finding #1) ---
# Builds a throwaway repo (under /tmp, safe from the mergerfs fcntl-lock issue),
# creates an "evil merge" that sneaks a Red Zone edit in with no spec reference,
# and asserts check-range.sh BLOCKS it (git diff-tree would show nothing for a
# merge — the exact fail-open this guards).
e2e_evil_merge_blocked() {
    local d; d="$(mktemp -d)"
    (
        cd "$d" || exit 9
        git init -q; git config user.email t@example.com; git config user.name t
        git config commit.gpgsign false
        local init; echo base > README.md; git add README.md
        git commit -qm "feat: base" -m "AI-disclosure: none"
        init="$(git branch --show-current)"
        local base; base="$(git rev-parse HEAD)"
        git checkout -qb feat; echo foo > foo.txt; git add foo.txt
        git commit -qm "feat: add foo" -m "AI-disclosure: none"
        git checkout -q "$init"
        git merge -q --no-ff feat -m "Merge branch 'feat'"
        mkdir -p comfyless; echo "x = 1" > comfyless/server.py
        git add comfyless/server.py; git commit -q --amend --no-edit
        bash "$lib_dir/check-range.sh" "$base" "$(git rev-parse HEAD)"
    ) >/dev/null 2>&1
    local rc=$?; rm -rf "$d"; return $rc
}
if e2e_evil_merge_blocked; then
    fail=$((fail+1)); echo "FAIL: check-range did NOT block an evil merge (Red Zone edit, no spec)"
else
    pass=$((pass+1))
fi

echo "git-policy tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
