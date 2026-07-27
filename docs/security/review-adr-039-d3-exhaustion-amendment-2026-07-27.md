# Security review — ADR-039 D3 amendment: exhaustion needs consecutive failed batches

AI-Disclosure: Claude (Fable 5) authored the code and this record; reviewed by
`security-auditor` (Fable 5) and `code-reviewer` (Fable 5); Grant reviewed.

Date: 2026-07-27
Surface: `comfyless/refine.py` (§12 Red Zone path), `test_refine.py`.
Decision record: `docs/decisions/ADR-039-refine-v3-promotion-gate.md` (accepted,
D3 amended — see its Changelog).
Prior records: `review-adr-039-slice{1,2,3,4}-*.md`.

**Reviewer model pin — and a correction to how the prior four records stated
it.** Both agents were invoked with `model: "fable"` and their transcripts record
`message.model = claude-fable-5` on every turn. The earlier records in this
chain asserted "no Fable→Opus fallback" on the strength of that field alone,
without establishing whether it reports the model REQUESTED or the model that
SERVED the request — an overclaim. Grant's billing independently corroborates
that these reviews did run on Fable, so the field is trustworthy in practice;
the wording in the prior records outran the evidence available at the time.
Separately, a survey of reviewer transcripts in this project found one on
2026-07-25 21:02 that ran on `claude-opus-5`, confirming the elevation is NOT
automatic: it follows the invocation-time `model:` parameter, and the agent-file
frontmatter pin is silently ignored (as CLAUDE.md §5A already documents).

## Scope

ADR-039 D3 as accepted stopped the run the first time a seed batch's winner
failed the gate. Two live runs showed that is too eager: a 20-generation budget
ended at 6, with the planner still producing its most responsive moves — a LoRA
variant swap and a "Do not add glasses" constraint answering the judge's own
critique — that it never got to evaluate. Grant directed the amendment:
exhaustion now requires `--exhaust-after-batches` consecutive failed batches
(default 2), because a batch varies only the seed, so one loss says noise cannot
rescue the config, not that the planner is out of ideas.

The rerun with the amendment ran 14 generations / 10 iterations, executed two
batches with three planner rounds between them, and fired D4's anchor duel for
the first time.

## Verdict

**`security-auditor`: no CRITICAL/HIGH/MEDIUM.** One LOW (fixed), three INFO.
The spend invariant `gens_used <= max_iterations`, the void-changes-nothing
discipline, Finding 9 history hygiene, and the by-value pins all survive intact.

**`code-reviewer`: not approvable as-is** — three MEDIUM and two LOW, all fixed
below.

## Findings and disposition

### LOW (security) / MEDIUM (code review) — the counter survived a D4 revert — FIXED

`failed_batches` is evidence about ONE config. A D4 anchor revert changes which
config that is, but reset `sideways_streak` and `plateau_streak` only. Reachable:
a batch fails at the drifted config (count 1), the anchor duel later reverts to
the pinned config, and the next failure there — the anchor config's FIRST —
exhausts the run on the claim that a single config is spent, having gathered one
loss at each of two different configs. A judge steering the anchor duel gets its
two losses without ever failing the same config twice. **Fixed:** the revert
resets `failed_batches` alongside the streaks.

### MEDIUM (code review) — the amended behaviour itself was unpinned — FIXED

Both tests exercised only the counter reaching 1 (one at threshold 1, one
stopping after a single failure). A bug reinitializing `failed_batches` each
pass would have disabled D3 entirely and passed the whole suite. **Fixed:** a
test drives two consecutive failed batches at the DEFAULT threshold with planner
iterations in between, asserting both the terminal state and both log lines;
plus a test that a promoting run never exhausts.

### MEDIUM (code review) — ADR Changelog entry missing — FIXED

The code cited "ADR-039 D3, amended 2026-07-26" against a record that did not
exist, and the ADR's D3 body still said a failed batch stops the run. §12 order
is ADR → review → code. **Fixed:** Changelog entry appended before this commit.

### LOW (code review) — exhaustion log dropped the best-iteration reference — FIXED

The threshold-fire message no longer told the operator which iteration's best it
stopped on. Restored.

### INFO (security) — the loop docstring still asserted one-batch exhaustion — FIXED

The docstring is the in-file statement of a security-reviewed stop condition; a
future slice implemented against it would have silently reinstated the old
behaviour. Corrected.

### INFO (security) — a slice-3 rationale is superseded, the invariant is not

Slice 3's record argued repeated batches are bounded because "each re-trigger
costs `sideways_cap` further sideways promotions". With `--explore-after 1` a
failed batch's own `plateau_streak` increment re-arms `pending_batch` on the
same pass, so batches can chain back-to-back with no planner round between them.
The bound that matters — `gens_used <= max_iterations`, the batch clamp
`arms = min(seed_batch, max_iterations - gens_used)`, and the `arms < 2` skip —
is untouched. This record supersedes that rationale; at defaults the worst case
is two batches.

### INFO (security) — validated at the CLI boundary only

`refine_loop` accepts `exhaust_after_batches` unvalidated; a programmatic 0 or
negative degenerates to first-failure exhaustion, i.e. the OLD, stricter
behaviour. Fails closed. Covered by the existing TECH_DEBT entry on loop-float
validation at a future machine boundary.

### INFO (code review) — `--patience` can now end a run just after a sub-threshold failed batch

A failed batch now reaches `no_improve += 1` and the patience check, which the
pre-amendment break preempted. Correct behaviour (a batch that promoted nothing
is not progress), but it means the slice-3 comment "exhaustion stops the run
before patience can matter" now holds only at threshold 1.

## Specific negatives verified

- A VOID batch neither increments nor resets the counter — right in both
  directions: a duel that never completed is evidence about the judge, not the
  config, and it must neither push toward a terminal verdict nor erase real
  evidence. The two accountings cannot be jointly evaded: losses accumulate
  `failed_batches`, voids accumulate `consecutive_judge_errors`.
- Nothing downstream of the former break assumed termination: lineage takes
  `best_cfg` on the not-promoted path (the failed batch winner's cfg and seed are
  discarded), `pending_batch` re-arms only via the triggers, and the history
  record is the standard flags-only shape with no error text.
- `--exhaust-after-batches 1` exactly reproduces pre-amendment behaviour.
- Adversary power is unchanged upward and reduced downward: forging the terminal
  `exhausted=True` signal now costs two consecutive batch losses instead of one.

## Proof

`test_refine.py` 631 → 634 (0 failures); full battery 29/29; pyright 1026 =
baseline; `just policy-test` 36/36; gitleaks clean.
