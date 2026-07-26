# Security review — ADR-039 slice 3: sideways cap + seed batch

AI-Disclosure: Claude (Fable 5) authored the code and this record; reviewed by
`security-auditor` (Fable 5) and `code-reviewer` (Fable 5); Grant reviewed.

Date: 2026-07-26
Surface: `comfyless/refine.py` (§12 Red Zone path — refinement-loop judge),
`test_refine.py`.
Decision record: `docs/decisions/ADR-039-refine-v3-promotion-gate.md` (accepted).
Prior records: `review-adr-039-design-2026-07-25.md`,
`review-adr-039-slice1-duel-primitive-2026-07-25.md`,
`review-adr-039-slice2-promotion-gate-2026-07-25.md`.

**Reviewer model pin:** both agents invoked with `model: "fable"`; transcripts
show `claude-fable-5` throughout — **no Fable→Opus fallback**.

**Tooling note — the diff caveat is CLOSED for this slice.** The first six
reviews of this ADR could not run `git diff` and reviewed file state instead
(logged in TECH_DEBT 2026-07-25). This round the diff was written to a file and
handed to both agents, and both read all 525 lines. The security auditor's
finding below turns on a two-branch interaction that a file-state read is much
less likely to surface — the change is worth keeping.

## Scope of the change

The D3 typed plateau escape: `--sideways-cap` counts consecutive promotions
that did not strictly improve the composite and schedules a `--seed-batch` of N
arms at best's config varying only the seed; the winner is picked by
single-elimination swap-paired duels in generation order with an earliest-arm
tie-break; a winner that cannot beat best ends the run as `exhausted`. A new
`gens_used` counter is now the authoritative `--max-iterations` bound, and
`_generate_one` is extracted so the batch and the normal path share the daemon
ref-refusal latch.

## Verdict

**`security-auditor`: one HIGH (fixed), one MEDIUM (fixed), three INFO.**
**`code-reviewer`: not approved as-is — one MEDIUM (the same defect, fixed),
one MEDIUM process item, four LOW (all fixed).** No boundary violation, no
scope creep, no security regression in either review.

## Findings and disposition

### HIGH / MEDIUM (both reviewers, same defect) — a VOID gate duel was reported as EXHAUSTED — FIXED

On a batch pass, `if batch_iteration and not promoted:` could not distinguish a
*decided* loss from a *void* duel. When the bracket completed but the
champion's promotion-gate duel raised, the run terminated with
`exhausted=True, aborted=False` — a flag whose own docstring tells automation
to read it as "this config is done" — while charging only 1 toward an abort
threshold of 3. One transient endpoint failure, or F8-E pixel text in the
champion making the judge emit unparsable duel output, was enough. The batch's
entire GPU spend was laundered into a false exhaustion verdict, the log
asserted "varying the seed at fixed config cannot improve on it" — a claim the
run never established — and the void never got the chance to accumulate into a
loud abort.

This is the batch-pass analogue of the fail-open closed in slice 2, in the
opposite costume: not a wrong promotion, but a wrong TERMINAL SEMANTIC handed
to automation. D3's contract is "the batch winner cannot BEAT best"; a duel
that never completed has established nothing.

**Fixed:** `and not duel_failed` — a void gate duel on a batch pass now falls
through to the ordinary not-promoted path (charged, best kept, run continues),
identical to a void inside the bracket. **Mutation-verified:** reverting the
two-token condition fails exactly the three new assertions and nothing else.

### MEDIUM (security) — a generation failure inside an arm was swallowed as a duel void — FIXED

Any `RefineError` from `run_generation` other than `RefRefusedError` propagated
into the batch's broad `except RefineError`, was logged as "seed-batch duel
unusable … VOID", charged `failed_calls=0`, and the loop continued — despite
generation failures being FATAL on the normal path precisely because every
iteration would fail identically. Effects: a generation-plane failure
misattributed to the judge plane in the operator log and the accounting, and up
to `arms-1` further generations spent after a known-fatal condition.

**Fixed:** the bracket's duel calls are wrapped inside `_run_seed_batch` so
every duel-plane failure becomes a `DuelError` (config errors keep the
charged-0 split), and the caller now catches `DuelError` only. Arm generation
sits deliberately outside the guard, so its errors propagate with the loop's
fatal discipline intact.

### LOW (code review) — bracket champion continuity was unpinned — FIXED

Every batch test used `seed_batch=2`, so the fold loop ran at most one match and
the champion handoff — the core bracket mechanism — never executed twice. An
implementation that forgot to update `champion` (always dueling arm 0) passed
the whole suite. **Fixed:** a 3-arm test asserting the pixel pairs
`[(4,3), (5,4)]`, where the second pair's B side is the *new* champion.

### LOW (code review) — a void batch pass was invisible to history and patience — FIXED

The scoring-judge error path appends `history_error_record(i)` and increments
`no_improve`; the void-batch path did neither, leaving a hole in the history's
iteration numbering after a pass that spent several generations. **Fixed:**
parity with the sibling path. The record is the structural flags-only one — no
error text, no duel keys (Finding 9 / F8-P) — and is pinned by a continuity
assertion.

### LOW (both) — `LoopOutcome` did not expose the authoritative spend — FIXED

`iterations` means loop passes while `--max-iterations` now bounds generations;
after any batch the two diverge and `gens_used` lived only in a log line, so
automation auditing spend would undercount. **Fixed:** `generations` field,
populated at both return sites and printed on the summary line.

### LOW / INFO (both) — truncated arm count was not named as truncation — FIXED

When the remaining budget clamps `arms` below `--seed-batch`, the log now says
so explicitly, per slice 2's forward constraint that bounded coverage be logged
rather than silently truncated. The hard skip below 2 arms was already loud.

### LOW (code review) — batch ref wiring and latch inheritance were unasserted — FIXED

The slice's own claim (arms edit the current source with the static refs, and a
latched run keeps arms in-process) rested on the `_generate_one` sharing with no
test. **Fixed:** assertions on the arms' `refs_seen` paths, plus a
`refuse_daemon` batch run pinning that the latch fires exactly once and every
later generation — arms included — is forced in-process.

### INFO (code review) — the docstring overclaimed order-independence — FIXED

"Makes the bracket independent of arm ordering" was too strong: the *tie-break*
is order-symmetric, but a single-elimination ladder under a non-transitive
judge is not order-independent — as the ADR itself concedes when citing the
memo's Condorcet warning. Softened to "deterministic".

### MEDIUM, PROCESS (code review) — accepted ADR text and Red Zone code disagree — OPEN, GRANT'S CALL

D3's Supersession says the ADR-037 stagnation escape is SUBSUMED by the seed
batch. This slice KEEPS it, with the reasoning at the site. Both reviewers
examined the deviation and judged the reasoning correct; the code reviewer
strengthened it with two arguments the comment omitted:

1. **t2i has no batch at all** — `pending_batch` requires duels, which require
   edit mode, so deleting `--explore-after` would strip t2i of its only plateau
   escape. D3's subsumption was never coherent for t2i.
2. **D1 already repaired the instrument the subsumption argument attacked.**
   D3's case against per-iteration resampling was "one seed at a time judged on
   a saturated scale"; post-slice-2 an in-band resampled single is gated by a
   swap-paired duel, not the saturated scalar.

Verified in code: the two triggers are provably disjoint — `sideways_streak`
increments only on `promoted and not improved`, while `no_improve` resets on
every promotion, so the "nothing promotes" plateau (the common shape under D1's
tie rule) drives `no_improve` and never touches the streak.

The code reviewer also names the design that would be truer to D3's intent:
broaden the batch trigger to fire on `no_improve` in edit mode, retiring
`--explore-after` there while keeping it for t2i. That changes budget dynamics
and is a separate slice.

**This is Grant's ruling to make, not the reviewers' and not mine.** An
accepted ADR and Red Zone code disagree until he rules; keeping both is the
conservative floor. Recorded as an OPEN question in the ADR-039 Changelog.

## Specific negatives verified

- Bracket voids charge `failed_calls` exactly once, rebind nothing (the tuple
  unpack never executes on a raise), and leave no half-built state — `entries`
  and `champion` are locals discarded by the raise.
- Losing arms are inert: a load-plane sidecar and log lines only. No verdict,
  no score, no history record, and no path by which one becomes `best` or
  `current_source`.
- `gens_used <= max_iterations` is an invariant (top-of-loop break plus the
  `min()` clamp); repeated batches cannot recur unboundedly because each
  re-trigger costs `sideways_cap` further sideways promotions, each a generation.
- The seed lattice is strictly increasing across ALL THREE triggers including
  the re-basing case: when an arm with offset `r_k` promotes, the base becomes
  `base + r_k` and every later draw is `base + r_k + r'` with `r' > r_max`.
- The daemon ref-refusal latch fires exactly once per run; batch arms read
  `current_source` and `static_refs` through the same closure at call time, so
  their ref ordering (loop source first, then operator-pinned static refs) is
  identical to any iteration.
- The `exhausted` stop cannot pre-empt the pass gate (checked earlier) and
  cannot keep a run alive past the caps.

## Forward constraints for slice 4

1. **D4 remains load-bearing**, per slice 2's escalated INFO — near the score
   ceiling the out-of-band promotion path is unreachable, so the anchor duel is
   the only compensating control for the accepted F8-E residual.
2. The bracket widened the window on ADR-038's accepted cross-run stem-collision
   residual (the champion is re-opened from disk after up to `arms-1` duels'
   wall time). Same class, already accepted; D4's by-value anchor pin must not
   reintroduce a path-based read.
3. Grant's ruling on the `--explore-after` subsumption should land in the
   ADR Changelog before or with slice 4.

## Proof

`test_refine.py` 571 → 602 (0 failures), including a mutation check that
reverting the HIGH's fix fails exactly the three new assertions; full battery
29/29; pyright 1026 = baseline; `just policy-test` 36/36; gitleaks clean.
