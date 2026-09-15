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

# --- typecheck-ratchet baseline (ADR-032: may only decrease; ADR-042: per root) ---
ok pc_baseline_no_increase $'comfyless=52\nnodes=520\npipelines=454' $'comfyless=52\nnodes=520\npipelines=454'   # unchanged
ok pc_baseline_no_increase $'comfyless=52\nnodes=520\npipelines=454' $'comfyless=8\nnodes=520\npipelines=454'    # drawdown on one root
ok pc_baseline_no_increase ""                                        $'comfyless=52\nnodes=520\npipelines=454'  # introducing the file
no  pc_baseline_no_increase $'comfyless=52\nnodes=520\npipelines=454' $'comfyless=60\nnodes=520\npipelines=454'  # bump on one root — blocked, even with the others unchanged
no  pc_baseline_no_increase $'comfyless=52\nnodes=520\npipelines=454' $'comfyless=8\nnodes=520\npipelines=460'   # drawdown on one root does NOT excuse a bump on another
ok pc_baseline_no_increase $'comfyless=52\nnodes=520\npipelines=454' $'comfyless=52\n nodes = 520 \npipelines=454' # NOT COMPARED at this layer (grep "^root=" misses the spaced form) — CI's "no baseline entry for root X" check is the backstop, not this function
# Legacy bare-integer format at either side (pre-ADR-042, or the transition
# commit's HEAD) has no `root=` lines — nothing to compare, so it never
# blocks. This is deliberate (see _lib.sh): a root gets ratcheted only once
# it HAS a prior per-root baseline to ratchet against.
ok pc_baseline_no_increase "1026" $'comfyless=52\nnodes=520\npipelines=454'
ok pc_baseline_no_increase "1026" "1076"
# A newly-introduced root (present in new, absent from old) is not compared.
ok pc_baseline_no_increase $'comfyless=52\nnodes=520' $'comfyless=52\nnodes=520\npipelines=454'

# --- scripts/typecheck-per-root.sh (ADR-042 mechanism, PYRIGHT_OUTPUT_FILE
# test-injection escape hatch — no live pyright run) ---
per_root_script="$repo_root/scripts/typecheck-per-root.sh"
tprd="$(mktemp -d)"

# Normal multi-root sample: 2 errors in comfyless, 1 in nodes, 1 in an
# out-of-scope path (must get its OWN bucket, not be silently dropped — the
# dynamic-bucketing fix), 1 WARNING in pipelines that must NOT count.
cat > "$tprd/normal.txt" <<EOF
$repo_root/comfyless/mcp_server.py
  $repo_root/comfyless/mcp_server.py:10:1 - error: bad thing (reportX)
  $repo_root/comfyless/mcp_server.py:20:1 - error: bad thing 2 (reportX)
$repo_root/nodes/foo.py
  $repo_root/nodes/foo.py:5:1 - error: bad thing (reportX)
$repo_root/scripts/rogue.py
  $repo_root/scripts/rogue.py:1:1 - error: unexpected out-of-scope file (reportX)
$repo_root/pipelines/bar.py
  $repo_root/pipelines/bar.py:1:1 - warning: must not count (reportX)
4 errors, 1 warnings, 0 informations
EOF
got="$(PYRIGHT_OUTPUT_FILE="$tprd/normal.txt" bash "$per_root_script" 2>/dev/null)"
want=$'comfyless=2\nnodes=1\nscripts=1'
if [ "$got" = "$want" ]; then pass=$((pass+1)); else
    fail=$((fail+1)); echo "FAIL: typecheck-per-root.sh normal sample — got [$got] want [$want]"
fi

# Crashed/misconfigured checker (no pyright summary line) must fail CLOSED:
# nothing on stdout, non-zero exit. This is what every caller's "no per-root
# counts" branch depends on (code-reviewer 2026-07-27, HIGH).
echo "Traceback (most recent call last): pyright blew up" > "$tprd/crashed.txt"
got="$(PYRIGHT_OUTPUT_FILE="$tprd/crashed.txt" bash "$per_root_script" 2>/dev/null)"
rc=$?
if [ -z "$got" ] && [ "$rc" -ne 0 ]; then pass=$((pass+1)); else
    fail=$((fail+1)); echo "FAIL: typecheck-per-root.sh did not fail closed on a crashed checker (stdout=[$got] rc=$rc)"
fi

# A root that reaches 0 errors legitimately produces no line for that root
# (nothing to bucket) — not an error, not a false "still has errors".
cat > "$tprd/clean_root.txt" <<EOF
$repo_root/nodes/foo.py
  $repo_root/nodes/foo.py:5:1 - error: bad thing (reportX)
1 errors, 0 warnings, 0 informations
EOF
got="$(PYRIGHT_OUTPUT_FILE="$tprd/clean_root.txt" bash "$per_root_script" 2>/dev/null)"
if [ "$got" = "nodes=1" ]; then pass=$((pass+1)); else
    fail=$((fail+1)); echo "FAIL: typecheck-per-root.sh clean-root sample — got [$got] want [nodes=1]"
fi

rm -rf "$tprd"

# --- more floor forms ---
no pc_no_floors '+    "baz==latest",'
no pc_no_floors '+    "bar==2.*",'
no pc_no_floors "+    'qux != 1.0',"

# --- Red Zone spec (=ADR, per the _lib.sh adaptation) references ---
# The cited docs below are FIXTURES, not subject matches: pc_redzone_ref only
# asks whether a doc of the right kind exists in the validated tree, never
# whether it is about the changed file. They are drawn from the BOTH bucket of
# docs/README.md, the set this repository keeps permanently, so the 2026-09-15
# CORE-ONLY prune (and any later one) cannot silently redden these assertions.
ok pc_redzone_ref "no ref needed" "README.md"                     spec   "$repo_root"  # not RZ
ok pc_redzone_ref "see docs/decisions/ADR-019-native-quantization-support.md" "src/comfyless/server.py" spec "$repo_root"
no pc_redzone_ref "no reference at all" "src/comfyless/server.py"     spec   "$repo_root"  # RZ, no ref
ok pc_redzone_ref "docs/decisions/ADR-002-three-tier-lora-fallback.md" "src/comfyless/mcp_server.py" spec "$repo_root"
# A reference to a NON-existent ADR must NOT satisfy the gate (guards the [ -f ]
# existence check — the slice-11 HIGH-2 defense).
no pc_redzone_ref "TODO: write docs/decisions/ADR-999-ghost.md" "src/comfyless/server.py" spec "$repo_root"
# Function-scoped surfaces (_run_json_mode, resolve_hf_path) are deliberately
# NOT path-gated — their whole files must stay non-RZ (see _red-zone-paths.sh).
ok pc_redzone_ref "no ref needed" "src/comfyless/generate.py"          spec "$repo_root"
ok pc_redzone_ref "no ref needed" "src/comfyless/core/eric_diffusion_utils.py" spec "$repo_root"  # resolve_hf_path + fp8 detection/remap: BOTH function-scoped
ok pc_redzone_ref "no ref needed" "nodes/eric_diffusion_utils.py"  spec "$repo_root"  # pre-slice-5 path, deleted; kept as a negative control
# The other two listed surfaces are RZ.
no pc_redzone_ref "no reference" "src/comfyless/refine.py"             spec "$repo_root"

# ADR-045 slice 5+ : the fp8 weight-file parser and the LoRA adapter subsystem.
# The parser is matched at all three historical spellings so a check-range over
# either move still gates; lora_adapters was never gated before 2026-09-09.
ok pc_redzone_ref "see docs/decisions/ADR-019-native-quantization-support.md" "src/comfyless/core/eric_diffusion_fp8_ops.py" spec "$repo_root"
no pc_redzone_ref "no reference" "src/comfyless/core/eric_diffusion_fp8_ops.py" spec "$repo_root"
no pc_redzone_ref "no reference" "comfyless/core/eric_diffusion_fp8_ops.py"     spec "$repo_root"
no pc_redzone_ref "no reference" "nodes/eric_diffusion_fp8_ops.py"              spec "$repo_root"
ok pc_redzone_ref "see docs/decisions/ADR-046-comfyless-owned-lora-adapters.md" "src/comfyless/core/lora_adapters.py" spec "$repo_root"
no pc_redzone_ref "no reference" "src/comfyless/core/lora_adapters.py"          spec "$repo_root"
no pc_redzone_ref "no reference" "comfyless/core/lora_adapters.py"              spec "$repo_root"  # pre-slice-5 spelling must STAY gated for check-range
ok pc_redzone_ref "no ref needed" "src/comfyless/core/lora_adapters.py.bak"     spec "$repo_root"  # $-anchor: a suffixed copy is not the surface
ok pc_redzone_ref "no ref needed" "xnodes/eric_diffusion_fp8_ops.py"            spec "$repo_root"  # anchor must not match mid-segment
no pc_redzone_ref "no reference" "nodes/eric_diffusion_fp8_ops.py" spec "$repo_root"

# --- Red Zone review references (the whole `review` kind was previously untested) ---
ok pc_redzone_ref "no ref needed" "README.md" review "$repo_root"
ok pc_redzone_ref "see docs/security/review-resolve-hf-path-2026-04-23.md" "src/comfyless/server.py" review "$repo_root"
no pc_redzone_ref "no reference" "src/comfyless/server.py" review "$repo_root"
no pc_redzone_ref "docs/security/review-ghost.md" "src/comfyless/server.py" review "$repo_root"

# --- Red Zone references resolve against a TREE, not the working tree ---
# The 5th argument is a git tree-ish: a commit SHA for check-range, empty (the
# default used above and by the commit-msg hook) for "the index". This is what
# lets a referenced doc be deleted later without retroactively failing the
# commit that cited it. See _gp_ref_exists in _lib.sh.
# Both operands are derived at RUN TIME, deliberately:
#   - the ADR is whichever one this repo currently has, not a hard-coded name,
#     so the docs prune this change unblocks cannot break the suite by deleting
#     the file a test happened to cite (code review 2026-09-13).
#   - the negative case uses the EMPTY TREE, not this repo's root commit. CI
#     checks out at depth 1, where `rev-list --max-parents=0 HEAD` resolves to
#     the pushed commit itself — in which the ADR *does* exist, so the assertion
#     inverted and the job failed while passing locally on a full clone.
gp_any_adr="$(git ls-files 'docs/decisions/ADR-*.md' | head -1)"
gp_empty_tree="$(git hash-object -t tree /dev/null)"
# Present at HEAD's tree -> passes.
ok pc_redzone_ref "see $gp_any_adr" "src/comfyless/server.py" spec "$repo_root" HEAD
# The SAME reference against a tree that contains nothing -> blocked. This is
# the discriminating case: under the old working-tree test both of these passed
# identically, because the file exists on disk now.
no pc_redzone_ref "see $gp_any_adr" "src/comfyless/server.py" spec "$repo_root" "$gp_empty_tree"
# An unresolvable tree-ish must FAIL CLOSED, not wave the commit through.
no pc_redzone_ref "see $gp_any_adr" "src/comfyless/server.py" spec "$repo_root" "0000000000000000000000000000000000000000"
# _gp_ref_exists itself: a blob passes, a tree (directory) does not. The old
# `[ -f ]` also rejected a directory; keep that property.
ok _gp_ref_exists "$repo_root" "" "$gp_any_adr"
no _gp_ref_exists "$repo_root" "" "docs/decisions"
no _gp_ref_exists "$repo_root" "" "docs/decisions/ADR-999-ghost.md"
# A committed SYMLINK types as `blob`, so the mode — not the object type — is
# what decides. The old `[ -f ]` followed symlinks and rejected dangling ones;
# _gp_ref_exists refuses all of them (security review 2026-09-13, INFO).
e2e_symlink_ref_rejected() {
    local d; d="$(mktemp -d)"
    (
        cd "$d" || exit 9
        git init -q; git config user.email t@example.com; git config user.name t
        git config commit.gpgsign false
        mkdir -p docs/decisions
        ln -s /nonexistent/target docs/decisions/ADR-001-daemon-socket-security.md
        git add docs; git commit -qm "docs: a symlink" -m "AI-disclosure: none"
        _gp_ref_exists "$d" HEAD "docs/decisions/ADR-001-daemon-socket-security.md"
    ) >/dev/null 2>&1
    local rc=$?; rm -rf "$d"; return $rc
}

# A commit that DELETES an ADR while touching a Red Zone file must not be
# credited by the "the artifact IS in this commit" branch — `--name-only` lists
# deleted paths, so before the 6th parameter this passed with no citation at all
# (security review 2026-09-13, INFO).
e2e_deleted_artifact_not_credited() {
    local d; d="$(mktemp -d)"
    (
        cd "$d" || exit 9
        git init -q; git config user.email t@example.com; git config user.name t
        git config commit.gpgsign false
        mkdir -p docs/decisions docs/security src/comfyless
        echo adr    > docs/decisions/ADR-001-daemon-socket-security.md
        echo review > docs/security/review-comfyless-server-2026-04-23.md
        echo "x = 1" > src/comfyless/server.py
        git add .; git commit -qm "feat: seed" -m "AI-disclosure: none"
        local base; base="$(git rev-parse HEAD)"
        # Delete BOTH artifacts and edit the daemon, citing nothing. Deleting
        # only one is not a discriminating test: the other `kind` blocks the
        # commit for an unrelated reason and the hole stays hidden.
        git rm -q docs/decisions/ADR-001-daemon-socket-security.md \
                  docs/security/review-comfyless-server-2026-04-23.md
        echo "x = 2" > src/comfyless/server.py; git add src/comfyless/server.py
        git commit -qm "refactor: drop both docs" -m "AI-disclosure: none"
        bash "$lib_dir/check-range.sh" "$base" "$(git rev-parse HEAD)"
    ) >/dev/null 2>&1
    local rc=$?; rm -rf "$d"; return $rc
}

# e2e: deleting a referenced doc must NOT retroactively fail the commit that
# cited it. This is the whole point of the tree-scoped resolution — the docs
# prune was blocked on exactly this (TECH_DEBT 2026-09-13: 105 referenced docs,
# 24 collisions in a 38-file sample). Under the old gate this range FAILED.
e2e_redzone_ref_survives_doc_deletion() {
    local d; d="$(mktemp -d)"
    (
        cd "$d" || exit 9
        git init -q; git config user.email t@example.com; git config user.name t
        git config commit.gpgsign false
        mkdir -p docs/decisions docs/security src/comfyless
        echo adr    > docs/decisions/ADR-001-daemon-socket-security.md
        echo review > docs/security/review-comfyless-server-2026-04-23.md
        git add docs; git commit -qm "docs: seed the artifacts" -m "AI-disclosure: none"
        local base; base="$(git rev-parse HEAD)"
        echo "x = 1" > src/comfyless/server.py; git add src/comfyless/server.py
        git commit -qm "feat: touch the daemon" -m "AI-disclosure: none" \
                   -m "see docs/decisions/ADR-001-daemon-socket-security.md" \
                   -m "see docs/security/review-comfyless-server-2026-04-23.md"
        # Now prune the docs. The Red Zone commit above must still validate.
        git rm -q docs/decisions/ADR-001-daemon-socket-security.md \
                  docs/security/review-comfyless-server-2026-04-23.md
        git commit -qm "docs: prune the duplicated docs" -m "AI-disclosure: none"
        bash "$lib_dir/check-range.sh" "$base" "$(git rev-parse HEAD)"
    ) >/dev/null 2>&1
    local rc=$?; rm -rf "$d"; return $rc
}

# e2e: the converse. A commit citing a doc that did not exist in ITS OWN tree
# must be blocked, even though the doc exists by the time check-range runs.
# §12 fixes the order (ADR first, then the code), so a forward reference is a
# policy violation; the old working-tree test could not see one at all.
e2e_redzone_forward_ref_blocked() {
    local d; d="$(mktemp -d)"
    (
        cd "$d" || exit 9
        git init -q; git config user.email t@example.com; git config user.name t
        git config commit.gpgsign false
        mkdir -p docs/decisions docs/security src/comfyless
        echo base > README.md; git add README.md
        git commit -qm "docs: base" -m "AI-disclosure: none"
        local base; base="$(git rev-parse HEAD)"
        echo "x = 1" > src/comfyless/server.py; git add src/comfyless/server.py
        git commit -qm "feat: touch the daemon" -m "AI-disclosure: none" \
                   -m "see docs/decisions/ADR-001-daemon-socket-security.md" \
                   -m "see docs/security/review-comfyless-server-2026-04-23.md"
        # The cited docs arrive only AFTER the commit that cites them.
        echo adr    > docs/decisions/ADR-001-daemon-socket-security.md
        echo review > docs/security/review-comfyless-server-2026-04-23.md
        git add docs; git commit -qm "docs: add them late" -m "AI-disclosure: none"
        bash "$lib_dir/check-range.sh" "$base" "$(git rev-parse HEAD)"
    ) >/dev/null 2>&1
    local rc=$?; rm -rf "$d"; return $rc
}

# e2e: an UNTRACKED working-tree doc must not satisfy the gate. `[ -f ]` could
# not tell the difference, so a commit could cite an ADR that was never
# committed at all. Index resolution closes that.
e2e_redzone_untracked_ref_blocked() {
    local d; d="$(mktemp -d)"
    (
        cd "$d" || exit 9
        git init -q; git config user.email t@example.com; git config user.name t
        git config commit.gpgsign false
        mkdir -p docs/decisions src/comfyless
        echo base > README.md; git add README.md
        git commit -qm "docs: base" -m "AI-disclosure: none"
        # Present on disk, never added to the index.
        echo adr > docs/decisions/ADR-001-daemon-socket-security.md
        echo "x = 1" > src/comfyless/server.py; git add src/comfyless/server.py
        pc_redzone_ref "see docs/decisions/ADR-001-daemon-socket-security.md" \
                       "src/comfyless/server.py" spec "$d" ""
    ) >/dev/null 2>&1
    local rc=$?; rm -rf "$d"; return $rc
}

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
        mkdir -p src/comfyless; echo "x = 1" > src/comfyless/server.py
        git add src/comfyless/server.py; git commit -q --amend --no-edit
        bash "$lib_dir/check-range.sh" "$base" "$(git rev-parse HEAD)"
    ) >/dev/null 2>&1
    local rc=$?; rm -rf "$d"; return $rc
}

# e2e: MOVING a Red Zone file must trip the gate on the OLD path. With git's
# default rename detection the move is reported by destination only, so the
# gated path never reaches is_red_zone_path and the move commits un-gated —
# the mechanism by which the fp8 parser's gate went dead across slice 1b
# (security review 2026-09-09, HIGH). check-range.sh passes --no-renames; this
# proves it, and fails if anyone removes the flag.
e2e_redzone_move_blocked() {
    local d; d="$(mktemp -d)"
    (
        cd "$d" || exit 9
        git init -q; git config user.email t@example.com; git config user.name t
        git config commit.gpgsign false
        mkdir -p src/comfyless/core
        # >50 identical lines so git scores it a rename with high similarity
        for _ in $(seq 1 80); do echo "x = 1"; done > src/comfyless/core/lora_adapters.py
        git add src/comfyless/core/lora_adapters.py
        git commit -qm "feat: seed" -m "AI-disclosure: none" \
                   -m "see docs/decisions/ADR-046-comfyless-owned-lora-adapters.md" \
                   -m "see docs/security/review-slice-3c-lora-adapters-2026-08-22.md"
        local base; base="$(git rev-parse HEAD)"
        git mv src/comfyless/core/lora_adapters.py src/comfyless/core/moved_elsewhere.py
        # No Red Zone reference in THIS message: the move must be refused.
        git commit -qm "refactor: move it" -m "AI-disclosure: none"
        bash "$lib_dir/check-range.sh" "$base" "$(git rev-parse HEAD)"
    ) >/dev/null 2>&1
    local rc=$?; rm -rf "$d"; return $rc
}
if e2e_evil_merge_blocked; then
    fail=$((fail+1)); echo "FAIL: check-range did NOT block an evil merge (Red Zone edit, no spec)"
else
    pass=$((pass+1))
fi

if e2e_redzone_move_blocked; then
    fail=$((fail+1)); echo "FAIL: check-range did NOT block a Red Zone file MOVE (rename hid the gated path)"
else
    pass=$((pass+1))
fi

if e2e_redzone_ref_survives_doc_deletion; then
    pass=$((pass+1))
else
    fail=$((fail+1)); echo "FAIL: deleting a referenced doc retroactively failed the commit that cited it"
fi

if e2e_redzone_forward_ref_blocked; then
    fail=$((fail+1)); echo "FAIL: check-range did NOT block a commit citing a doc absent from its own tree"
else
    pass=$((pass+1))
fi

if e2e_redzone_untracked_ref_blocked; then
    fail=$((fail+1)); echo "FAIL: an untracked working-tree doc satisfied the Red Zone gate"
else
    pass=$((pass+1))
fi

if e2e_symlink_ref_rejected; then
    fail=$((fail+1)); echo "FAIL: a committed symlink satisfied the Red Zone gate"
else
    pass=$((pass+1))
fi

if e2e_deleted_artifact_not_credited; then
    fail=$((fail+1)); echo "FAIL: a DELETED ADR was credited as the commit's own artifact"
else
    pass=$((pass+1))
fi

echo "git-policy tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
