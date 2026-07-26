# ADR-039 — Refine v3: pairwise promotion gate and typed plateau escape

Status:   accepted
AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.

## Context

The refine loop's promotion gate has always been "compare this candidate's
composite score to best's." Two live findings say that is the wrong
instrument:

1. **The absolute scale saturates.** A 100-iteration run produced a chain of
   exact 9.6 ties in which image quality visibly DETERIORATED (photo drifting
   toward illustration). Grant, comparing the first tied candidate against the
   last side by side, picked the first instantly — pairwise discrimination
   works precisely where absolute scoring has stopped discriminating.
2. **The winner was the worst of equals.** ADR-037's D2 amendment
   (2026-07-24) made ties promote the NEWER candidate, so the run's winner was
   the last — and most drifted — member of the tie chain.

Grant's framing, which is the crux: *"We already know the scores are a tie;
it's the specific set of elements making up the score that are going to break
the tie."* Integrity to the reference in unprompted areas is exactly such an
element, and a scalar has already flattened it.

A cited research memo (`Refine_Optimization_Research_2026-07-25.md`, vault)
mapped this onto named methods. Its findings that bear on the decision:

- The LLM-judge literature is consistent that **relative comparison is more
  reliable than absolute scoring** — judges anchor scales inconsistently and
  cluster scores, failing to separate similar outputs (Zheng et al. 2023
  MT-Bench; LLM-as-a-Judge survey 2411.15594).
- **Position bias is the dominant pairwise failure mode**, and single-call
  order randomization only unbiases in expectation, not per-decision
  (CalibraEval 2410.15393). Swap-pairing is the mitigation.
- Unbounded sideways moves on a plateau are a known local-search failure
  (AIMA §4.1); the fix is a **cap plus a typed escape**, not more of the same
  move.
- Verifier over-optimization in image generation manifests as **stylistic
  drift toward illustration** — a documented reward-hacking mode, not a quirk
  of this judge (reward-hacking survey 2604.13602; NeurIPS 2025
  inference-time reward hacking).

## Decision

### D1 — Swap-paired duel replaces the promotion gate in the noise band

When a challenger's composite lands within `--duel-band` (default 1.0) of
best's, the absolute scores carry no usable information. Decide by head-to-head
comparison instead:

- **Two judge calls**, orders (best, challenger) and (challenger, best). The
  swap is not optional — per-decision position bias is the dominant failure
  mode and order randomization does not address it.
- Promote **only on a consistent win in both orders**. Disagreement between
  orders is a tie (the PandaLM convention).
- **Ties keep the INCUMBENT.** This inverts ADR-037's D2 amendment (see
  Supersession) and is the single change that fixes "winner = worst of
  equals": drift must now *beat* best under swap-consistent judgment, not
  merely match its score.

Outside the band, the existing strict-composite rule stands unchanged — a
clear improvement or a clear decline needs no duel, and duels are not free.

**A retired invariant, named deliberately** (design review INFO): today a
challenger scoring below best can never be promoted. Under D1 one up to
`--duel-band` below best CAN be, by winning both duel orders. That is the
point — the scalar is what we distrust in the band — but "composite never
decreases across promotions" is hereby knowingly retired. The pass gate and
`--until-score` are NOT affected: they read the absolute composite and are
checked BEFORE any duel, unchanged (pinned by test).

**Duel-unavailable resolution — fail-closed, and it aborts rather than
freezing** (design review HIGH). A duel that cannot complete for ANY reason
resolves as **no promotion** (keep incumbent). It never falls back to the
composite comparison: inside the band that would silently restore precisely
the rule this ADR supersedes, so a judge that scores fine on the absolute
pass but returns malformed duel output — endpoint flakiness, truncation on a
two-image prompt, or pixel text steering the duel response specifically —
would regain pre-v3 promotion behavior per iteration, unnoticed. Equally, a
*silent* keep-incumbent would let an always-erroring duel judge freeze the
run at the first promoted candidate while burning generations to the cap.
Therefore: **each failed duel call counts as an unusable verdict in the
existing `JUDGE_ERROR_ABORT_AFTER` accounting**, so a persistently broken
duel judge aborts loudly on the same discipline as a broken scoring judge.
One order succeeding is NOT a duel result — the swap is mandatory or the
duel is void.

### D2 — Duel payload, and the budget it must read

A duel shows the judge: candidate A, candidate B, and — in a multi-ref run —
the judge-marked reference(s) from ADR-038. It does NOT show the anchor:
preservation is already scored on the absolute pass, and the duel's question
is narrower ("which of these two is better, given what we're trying to
match?").

The image budget is `judge_max_images` from the backend entry (ADR-038 D3 as
amended) — **never a repo constant**. This ADR must compute its own
arithmetic rather than inherit ADR-038's, per that ADR's forward constraint:
2 candidates + N judge-marked refs ≤ `judge_max_images`. When the budget
cannot seat the refs, the duel drops refs before it drops the swap (a
non-swapped duel is worse than a ref-less one), and says so once.

**The duel payload set is computed ONCE per duel** (design review LOW): the
two calls differ only in candidate order. Recomputing per call could give the
orders different reference sets — an evidence mismatch that would present as
"disagreement" and silently resolve as a tie. Named negative test.

A duel's rubric is its own recipe (`duel-generic.toml`), not a mode of the
scoring rubric: the question, the output shape (a winner, not scores), and
the bias mitigations differ.

**Duels carry ZERO override authority** (design review MEDIUM). This must be
unrepresentable, not merely unintended:

- The duel has its OWN code-owned output contract constant — the scoring
  contract (`_JUDGE_OUTPUT_CONTRACT`) describes `overrides` and would be
  actively wrong to append here. Same never-recipe-editable composition rule.
- The parse is a CLOSED ENUM (winner ∈ first / second / tie) with
  reject-unknown, fail-closed. It does NOT reuse `parse_verdict`: reusing it
  would hand the duel a second planner-authority channel — two extra
  override-bearing calls per banded iteration, tripling the F1 surface.
- Any `overrides` / `loras` / `critique` content in a duel response is
  DISCARDED. Duel free text never reaches `prev_critique_text`, the LoRA
  offers, history records, or any other LLM-visible context (F8-P); operator
  artifacts only.
- The label→candidate mapping under swap is code-owned and pinned by a named
  negative test, so a swapped pair can never be mis-attributed.
- The duel user text is minimal and code-owned: no history block, no offers.
  A duel is a selection between two loop-owned, already-generated images —
  nothing else belongs in it.

**Inherited constraints, stated rather than assumed** (design review INFO):
temperature 0; both candidates through `downscale_for_judge`; role labels
that interpolate nothing operator-supplied; and `max_tokens` sized for a
two-image prompt — the D2-amendment's truncation LOW now applies twice per
banded iteration, so the duel recipe header carries the same token-budget
note.

### D3 — Sideways cap and typed plateau escape

Cap consecutive non-improving promotions at `--sideways-cap` (default 3).
On hitting the cap, change the MOVE TYPE rather than iterating the planner
again: run a **seed batch** — N candidates (default 3) at fixed config,
varying only the seed — and pick the batch winner by swap-paired duels.

This attacks the plateau on the axis the planner cannot reason about (noise)
in a batch where selection is statistically meaningful, instead of one seed at
a time judged on a saturated scale. It subsumes the ADR-037 stagnation escape,
which resamples one seed per iteration and judges it absolutely.

If the batch winner still cannot beat best in a duel, the config is
exhausted: surface that to the operator and stop, rather than spending the
remaining budget on rewording.

**Bracket rules** (design review LOW — unspecified, an implementation would
default to something and "later arm wins" reintroduces drift-by-recency in
miniature): single-elimination over the arms in generation order, with a
DETERMINISTIC judge-independent tie-break — the EARLIEST-generated arm wins a
tie, which is the anti-drift direction and makes the bracket independent of
arm ordering (the memo's Condorcet warning applies to any round-robin here).
Duel errors inside the bracket follow D1's rule: void, no promotion, counted
toward the abort accounting. Batch seeds come from the SAME monotonic
`seed_resamples` lattice the escapes already use, or uniqueness across mixed
triggers breaks. **Batch generations count against `--max-iterations`** — a
free escape would let repeated cap-triggered batches multiply total GPU work
past what the sanity cap exists to bound.

### D4 — Anchor duel against the run's first best

Every `--anchor-duel-every` promotions (default 5), duel current best against
the EARLIEST promoted best, swap-paired. If the old one wins, the chain has
drifted: revert to it and mark the intervening mutations as failed in the
history block.

**The first best is pinned BY VALUE at first promotion** (design review
MEDIUM): decoded image bytes held for the run under the same capped-loader
discipline as ADR-037's D5 anchor and ADR-038's `pin_static_refs`, PLUS a
`snapshot_config` copy. The anchor duel and any revert consume only those
pinned values — never a `candidates/candidate_NN.png` re-read, never a
sidecar reconstruction (sidecars legitimately carry load paths). Re-reading
the path would reopen exactly the TOCTOU window two prior amendments closed,
and it is live rather than theoretical: ADR-038's accepted residual is that
two concurrent runs sharing an `--output-dir` cross-overwrite `candidates/`
with colliding stems, so a path-based anchor duel could compare against a
FOREIGN run's image and revert this chain to a config whose image never
existed here. A revert restores the pinned config snapshot and, in edit mode,
the pinned image as `current_source`.

**History marking on revert mutates only EXISTING boolean flags**
(`is_best` / `improved`) — no new keys and no free text, so the F8-P
judge-bound surface is unchanged.

This is the KL-anchor idea from the reward-hacking literature translated to a
judge-only check, and it is the direct structural answer to the observed
photo→illustration walk. Cost is 2 judge calls per m promotions; no
generations.

### D5 — Annealed acceptance (cheap, no schedule to tune)

Early in the budget (first third), a duel TIE may advance the challenger —
exploration is worth more than incumbency when there is budget left. Late,
a strict duel win is required. This is the salvageable part of simulated
annealing without a temperature schedule that a ≤100-evaluation budget cannot
calibrate.

**This deliberately reopens tie-promotion for the early phase, and must be
bounded** (design review MEDIUM). With a binary per-call winner a "tie" is
exactly cross-order disagreement — the signature of the per-decision position
bias this ADR's own citations call the dominant failure mode. Letting that
coin-flip advance the challenger for ~33 iterations of a 100-iteration run
restores the accepted D2-amendment MEDIUM (constant-parity judge promotes
every iteration; winner = most drifted), and in edit mode compounds F8-E
source advancement, with the drifted lineage then holding incumbency into the
strict phase. Bounds, all three: consecutive tie-advances are capped by the
tie-streak limit held in reserve since the D2-amendment review; tie-advance is
DISABLED in edit mode (where the drift evidence actually lives); and
`--anchor-duel-every` must be ≤ the tie-advance window, so D4 is a real
compensating control rather than a hope. Recorded as an accepted, bounded
residual.

### D6 — Planner edit-magnitude hint (advisory only)

Track mutation success rate; below ~1/5, hint the planner toward smaller
deltas (single-clause prompt edits, ±0.1 LoRA weight); above, allow bolder
rewrites. Implemented as a line in the planner context, NOT a controller —
the (1+1)-ES 1/5th-rule mapping is qualitative because prompt space has no
step-size metric.

### D7 — What this does NOT do

Explicitly rejected at this budget scale, with reasons, so they are not
re-litigated: Bayesian optimization (no usable kernel over free-text prompt ×
categorical LoRA set); full simulated annealing (spends scarce 10-60 s
generations on moves it will revert; schedule needs more runs than this tool
will ever do per task); formal tabu search (the iteration-history block
already provides anti-cycling); population/evolutionary methods (at a ≤100
cap, populations degenerate into best-of-N with bookkeeping — revisit with
multi-GPU parallel generation); Elo/Bradley-Terry ratings over all candidates
(underdetermined at <100 items with ~1 comparison per pair); full
dueling-bandit algorithms (they assume a fixed arm set and repeated pulls —
take the duel primitive, leave the regret analysis).

## Supersession

**ADR-037 D2 amendment (2026-07-24), tie-promotion: SUPERSEDED inside the
duel band.** That amendment made ties promote the newer candidate on the
reasoning that equal scores can hide sub-score-resolution improvements. The
reasoning was right about the *information*, wrong about the *instrument*:
the fix for "the scalar cannot see the difference" is to ask a better
question, not to guess in the newer candidate's favor. Its own security
review predicted this failure mode (the accepted MEDIUM: a constant-parity
judge promotes every iteration, and the winner becomes the most drifted), and
the 100-iteration run realized it.

Outside the band the D2-amendment rule remains in force, with ties keeping
the incumbent rather than the challenger.

**The composite rule is NOT a fallback for a failed duel** (design review
HIGH — the first draft said it was, which was fail-open straight back to the
superseded behavior). Inside the band, a duel that cannot complete means no
promotion, and the failure feeds the abort accounting (D1). The first draft
also offered "budget cannot seat two candidates" as an unavailability case;
that is structurally dead — `resolve_judge_max_images` guarantees ≥ 2 and
refs are capped at entry — so judge error is the ONLY live unavailability,
which is exactly why its direction had to be fixed.

**ADR-037 stagnation escape: SUBSUMED by D3** (a seed batch is the same
intent done in a statistically meaningful way). The no-op resample stays: it
addresses a different failure (the planner proposing nothing at all).

## Alternatives Rejected

**Keep absolute scoring, make the rubric more discriminating.** Already tried
twice (decompose-then-verify, then anchored comparison). Both helped, and the
scale still saturated at 9.6 — the ceiling is the instrument, not the
wording.

**Duel every iteration, not just in the band.** Doubles judge cost for
decisions the composite already makes correctly (a 6.8 vs a 9.6 needs no
head-to-head). The band is where the information is.

**Single duel call with randomized order.** Cheaper by half, but position
bias is per-decision; randomization unbiases the aggregate while leaving each
individual promotion a coin-flip-weighted call. The whole point is to make
individual promotions trustworthy.

**Rate all candidates with Elo/Bradley-Terry.** Underdetermined at this scale
and needs a comparison budget we do not have; the incumbent chain plus D4's
anchor duel gets the same protection with no model.

## Deferred / Out of Scope

- **Population/parallel search** — revisit when multi-GPU parallel generation
  exists; at that point the seed batch (D3) becomes the natural seam.
- **Judge-model choice.** Whether a non-abliterated judge discriminates
  better is an empirical question this ADR does not settle.
- **Duels in t2i mode.** The mechanism is family-agnostic; the first
  implementation targets edit mode where the drift evidence is.

## Slice plan

1. **Duel primitive** — `duel-generic.toml`, swap-paired call, consistent-win
   parse, budget arithmetic reading `judge_max_images`. Unit-testable with a
   fake judge; no loop changes.
2. **Gate integration** — band detection, ties-keep-incumbent, fallback when
   a duel is unavailable. Supersession of the D2-amendment rule lands here.
3. **Plateau escape** — sideways cap + seed batch + batch bracket (D3).
4. **Anchor duel** (D4) and the annealing/hint knobs (D5/D6).

Flag validation at entry, exit 2 (design review LOW, the `--w-*` precedent):
`--duel-band` finite and >= 0 — a NaN band makes every band test False and
silently reverts the whole run to the superseded rule; `--sideways-cap` and
`--anchor-duel-every` integers >= 1, with 0-disables semantics documented if
adopted.

Negative tests named up front: a duel that disagrees across orders does NOT
promote; a tie keeps the incumbent (the inverted rule, pinned explicitly);
the anchor duel reverts on an old-best win; the band boundary is exclusive at
both ends; budget refusal drops refs before dropping the swap; no duel call
ever carries more images than `judge_max_images`; a failed duel promotes
nothing AND increments the abort counter; a duel response carrying
`overrides` has them discarded; the two orders of one duel carry an
identical reference set; the bracket's tie-break is the earliest arm; pass
and `--until-score` gates read the absolute composite only.

## Changelog

- 2026-07-25 — Proposed. Grant approved the research memo's B1-B5 and set
  this immediately after ADR-038. Awaiting security review (Red Zone:
  `comfyless/refine.py`; new judge-call shape) and Grant's acceptance.
- 2026-07-25 — **Accepted** by Grant, as written (design review already folded
  into the text above). Slice 1 proceeds.
- 2026-07-25 — **Slice 1 implemented** (the duel primitive: `duel-generic.toml`,
  own output contract + closed-enum parse, swap-paired call, budget arithmetic
  reading `judge_max_images`). No loop wiring, no flags. Reviews:
  `docs/security/review-adr-039-slice1-duel-primitive-2026-07-25.md` —
  `security-auditor` no CRITICAL/HIGH, `code-reviewer` approve; one MEDIUM
  (over-long int literal escaping the RefineError taxonomy, in `parse_duel` AND
  pre-existing in `parse_verdict`) fixed, one accepted F8-E residual named.
  That review carries five forward constraints for slices 2-4, including the
  still-owed named negative tests and the fact that D4's anchor duel — the
  compensating control for the duel-specific injection residual — lands only in
  slice 4, after slices 2-3.
