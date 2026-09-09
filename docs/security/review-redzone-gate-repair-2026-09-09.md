# Security review — Red Zone commit-gate repair

AI-Disclosure: `security-auditor` (Claude Fable 5, transcript-confirmed
`claude-fable-5`, no fallback) reviewed; Claude Opus 5 authored; Grant reviewed.

Date: 2026-09-09
Change: repair the §12 Red Zone path gate — match the fp8 weight-file parser at
all three historical spellings, gate `core/lora_adapters.py` for the first
time, sync `list_red_zone_paths` + the CLAUDE.md review bar, extend the smoke
tests. Closes TECH_DEBT 2026-08-22, pulled forward from its "no later than
slice 6" trigger.

Why it was pulled forward: the previous slice's review observed the gap
*firing* — slice 3d edited `core/eric_diffusion_fp8_ops.py` and the gate did
not trigger, because it still named the `nodes/` path deleted in slice 1b.

## Verified by execution (not inspection)

The auditor sourced the script and ran `is_red_zone_path` against a ~40-path
matrix: no path that should gate fails to; the escaped dot holds (`.pyc`,
`.py.bak`, `lora_adaptersXpy` all reject); `(^|/)` and `$` anchors hold
(`xnodes/…`, `srccomfyless/…`, `core2/…` reject); matching is case-sensitive;
regex metacharacters in the argument are inert. Over-matches
(`notsrc/comfyless/core/…`) are the deliberate fail-closed suffix design.
`list_red_zone_paths` is consistent with the regexes in both directions. Both
call sites (`commit-msg-checks.sh`, `check-range.sh`) feed bare repo-relative
paths, confirmed by reading them. The new smoke cases discriminate — each
gated path has a paired with-ref/without-ref case on the same path, so a
regression in either direction flips a case.

## Findings and disposition

### HIGH — git rename detection blinds the gate to the NEXT move

`git diff --name-only` emits only the DESTINATION path for a rename, verified
empirically in a scratch repo. So when a gated file is next moved, the move
commit never presents the old (gated) path to `is_red_zone_path`, and the move
itself — which may carry content edits up to the rename-similarity threshold —
commits un-gated, along with every edit after it until a human notices. That is
the 2026-08-22 → 2026-09-09 gap replayed. Enumerating historical spellings, as
this change does, protects `check-range` over PAST moves but is structurally
incapable of covering the next one.

**Disposition: FIXED** in the following commit — `--no-renames` on both
`git diff --name-only` invocations (`check-range.sh:51`,
`commit-msg-checks.sh:24`), kept as a separate commit because it changes gate
BEHAVIOR rather than gate paths (global §4). Verified empirically in a scratch
repo: moving `src/comfyless/core/lora_adapters.py` reports only the destination
with rename detection on, and both paths with the flag. Pinned by a new e2e
case (`e2e_redzone_move_blocked`) that MUTATION-TESTS clean: removing the flag
from `check-range.sh` makes the suite fail with "check-range did NOT block a
Red Zone file MOVE". This is the finding that matters most
in the set: it converts the gate from "correct until the next restructure" to
"forces its own update at move time", in a repository that is mid-restructure.

### MEDIUM — the fp8 DETECTION/REMAP half of the ADR-019 surface is un-gated

The review bar defines the surface as the parser *plus* detection/remap in
`eric_diffusion_utils.py`. The exclusion comment justified excluding that file
by naming only `resolve_hf_path`, so the fp8 half was excluded silently rather
than deliberately.

**Disposition: NAMED, DEFERRED.** The comment and the smoke-test annotation now
name both function-scoped surfaces in that file, and TECH_DEBT 2026-09-09
records the gap with a preferred fix (split the detection/remap into its own
module so it becomes gateable without gating a file that changes in most
feature slices). Gating the whole file today would fire constantly.

### MEDIUM — the review-bar row for `resolve_hf_path` named a deleted file

`nodes/eric_diffusion_utils.py` has not existed since slice 5. For a
function-scoped surface — human-triggered, not mechanically gated — a stale
location IS the failure mode: a future session greps the documented path, finds
nothing, concludes the surface is gone. I synced two rows of that table and
missed the third.

**Disposition: FIXED IN THIS CHANGE** — row and comment repointed to
`src/comfyless/core/eric_diffusion_utils.py`.

### INFO — three small smoke-test gaps

The pre-slice-5 `comfyless/core/lora_adapters.py` spelling matched but was not
pinned by a test; there was no `$`-anchor negative; one case duplicated another
verbatim. **All three fixed in this change** (53 passing, up from 43).

## Note on posture

This gate is a discipline mechanism, not an adversarial boundary — the
committer is trusted and `Policy-override:` / `SKIP=` are documented escapes.
It was therefore reviewed for silent-no-op modes rather than bypass resistance,
which is the correct threat model and the one under which the HIGH finding is
serious.
