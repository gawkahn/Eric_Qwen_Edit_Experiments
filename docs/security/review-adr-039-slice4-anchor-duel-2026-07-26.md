# Security review — ADR-039: plateau-trigger ruling (b) + slice 4 (D4/D6)

AI-Disclosure: Claude (Fable 5) authored the code and this record; reviewed by
`security-auditor` (Fable 5) and `code-reviewer` (Fable 5); Grant reviewed.

Date: 2026-07-26
Surface: `comfyless/refine.py` (§12 Red Zone path — refinement-loop judge),
`test_refine.py`.
Decision record: `docs/decisions/ADR-039-refine-v3-promotion-gate.md` (accepted).
Prior records: `review-adr-039-design-2026-07-25.md`,
`review-adr-039-slice{1,2,3}-*.md`.

**Reviewer model pin:** both agents invoked with `model: "fable"`; transcripts
show `claude-fable-5` throughout — **no Fable→Opus fallback**.

This record covers TWO commits, both reviewed together as one diff: the
plateau-trigger change from Grant's ruling (b), and slice 4 (D4 + D6).

## Scope

**Ruling (b).** The D3 batch trigger now also fires on `--explore-after`
consecutive non-promoting iterations, not only on `--sideways-cap`
non-improving promotions. In edit mode `--explore-after` therefore schedules a
seed BATCH, retiring the ADR-037 single-seed resample there; t2i and
duels-off edit runs keep the single resample, because batches need duels.

**Slice 4.** D4's anchor duel: the run's FIRST best is pinned by value at first
promotion; every `--anchor-duel-every` the current best is duelled against it,
and an anchor win reverts config, image lineage, and history marking. D6's
advisory planner hint. D5 is deliberately NOT implemented.

## Verdict

**`security-auditor`: two HIGH, one MEDIUM, one LOW, two INFO — all fixed or
recorded.** **`code-reviewer`: not approved as-is — two HIGH (the same two),
three MEDIUM, four LOW.** Both HIGHs were in D4, the control slice 2 escalated
to load-bearing; both are fixed and pinned by tests.

## Findings and disposition

### HIGH (both reviewers) — D4's trigger was keyed to a counter the adversary controls — FIXED

The trigger was `promotions % anchor_duel_every == 0`. Both reviewers reached
the same conclusion from different directions, and it is the sharpest finding
of the whole ADR: **the control designed for the regime where promotions have
STOPPED was triggered by promotions.**

- *Never fires:* an entrenched incumbent (injected or merely drifted) promotes
  as promotion #k with `k % N != 0` — four chances in five at the default N —
  then vetoes every challenger at the gate. `promotions` freezes at k, the
  modulo never lands again, and the anchor duel **never runs for the rest of
  the run**. The run burns generations to the cap and publishes the entrenched
  incumbent, with D4 contributing nothing.
- *Fires every iteration:* frozen exactly on a multiple, it re-duels on every
  non-promoting iteration — 2 judge calls per iteration against the documented
  "2 per m promotions", and (code review) repeated Bernoulli trials against a
  noisy binary comparator turn a per-check revert probability into a
  near-certain revert. "Check every 5 promotions" silently becomes "duel until
  the anchor wins".

The naive fix (gate on `promoted`) converts the second case into the first
universally.

**Fixed:** the last check is latched (`anchor_checked_at`, `iters_since_anchor`)
and the duel is due when EITHER `anchor_duel_every` promotions or
`anchor_duel_every` iterations have passed since it. The check is therefore
periodic in wall-clock iterations even when nothing is promoting, and bounded
at one duel per N iterations in the worst case. The pin itself counts as the
first check, so `--anchor-duel-every 1` no longer duels the anchor against its
own copy. Pinned by tests: a frozen promotion count still gets periodic checks;
the cadence stays bounded; no self-duel at the pin.

**Named residual:** periodic checks during a long stall are still repeated
trials against a noisy comparator, so a spurious revert becomes likelier the
longer a run stalls. That is the deliberate trade — the alternative is the
never-fires case above — and it is bounded by the N-iteration cadence.

### HIGH (security) / MEDIUM (code review) — the anchor file's FIXED name collided across concurrent runs — FIXED

The pin wrote `<output-dir>/anchor/first_best.<ext>`, a constant name. Two runs
sharing an `--output-dir` — normal in this environment, and the exact ADR-038
residual the pin block cites as its own motivation — overwrite it with
**probability 1**, where `candidates/` collides only on coinciding stems. The
duel itself stayed safe (the image is in memory), so the judge would approve a
revert against the true anchor — and the revert would then install the FOREIGN
run's file as `current_source` (every later generation edits an image that
never existed in this chain) and publish it as this run's winner. That is
precisely the "revert this chain to a config whose image never existed here"
outcome the by-value pin exists to prevent, at whole-run blast radius.

**Fixed:** the anchor filename is created run-unique via `tempfile.mkstemp`
(O_EXCL) in the anchor dir. The read side remains the loop's pre-existing
`candidates/` window — the same one `Image.open(outcome.image_path)` already
has two lines earlier — which is ADR-038's accepted residual, not something D4
widens. Pinned by tests asserting the unique-prefix filename end to end
(pin, revert source, winner copy).

### MEDIUM (security) / HIGH (code review) — anchor-duel failures were erased by the per-iteration reset — FIXED

`if judge_calls_ok: consecutive_judge_errors = 0` ran BEFORE the anchor duel,
and the anchor block neither set `judge_calls_ok` nor moved the reset. So a
channel that scores cleanly but always fails ANCHOR duels charged 1, then had
it wiped by the next iteration's reset — `JUDGE_ERROR_ABORT_AFTER` never
reached. This is the exact alternate-reset/increment failure the gate duel's
own comment names and closes, reintroduced one screen further down, and it is
fail-open on the load-bearing control: the incumbent's own pixels ride every
anchor-duel payload, so an incumbent whose content reliably breaks duel parsing
keeps D4 permanently void while the run continues.

**Fixed:** the reset moved below the anchor-duel block (after every judge call
the iteration can make), and the anchor `except` sets `judge_calls_ok = False`.
Pinned by a test where scoring always succeeds and only anchor duels fail: the
run now aborts.

### MEDIUM (code review) — the config-revert test was vacuous — FIXED

The check compared prompts, but `apply_overrides` sets the prompt from the
verdict either way, so it held with or without a revert: a revert that restored
the image but left the drifted config would have passed. **Fixed:** the
reverting iteration's verdict now overrides nothing, so the next generation
carries the ANCHOR's prompt if the config was really restored and the drifted
one otherwise.

### LOW (code review) — patience/abort ordering differed between void paths — FIXED

The void-batch path checked patience before the abort threshold while the
scoring-error path checked abort first, so an iteration tripping both reported
`aborted=False` on one route and `True` on the other — and automation branches
on that flag. Now identical in both.

### LOW (code review) — a reverted promotion still counted as progress — FIXED

`no_improve` was zeroed by the gate before the anchor duel overruled the
promotion, so history said the mutation failed while `--patience` said it
succeeded. The revert now re-increments it. `promotions` deliberately still
counts the gate event — the anchor cadence should measure real gate decisions.

### INFO (both) — a reverted run was invisible in `LoopOutcome` — FIXED

Added `reverts`, alongside `generations`. Without it, a run that discarded N
iterations to a revert is indistinguishable from one that never improved.

### LOW (security) — the pass gate exits before D4 — ACCEPTED, NAMED

A candidate clearing the pass gate breaks out before promotion accounting and
the anchor duel, so pixel text that inflates BOTH axes past `--pass-threshold`
(or the composite past `--until-score`) exits immediately with no drift check —
a cheaper injection route than entrenchment, needing no veto at all. This is
the pre-existing exit shape and sits inside the accepted F8-E residual, but it
bounds D4's coverage claim and is recorded here rather than discovered later.
Closing it means deciding what a pass that loses its anchor duel MEANS, which
is a semantic change beyond this slice.

### INFO (security) — what D4 does and does not close

The anchor duel is the same judge on the same channel, and the incumbent's
pixels ride every anchor call. Pixel text of the form "always declare this
image the winner" that sustains a gate-duel veto defeats the anchor duel
identically. **D4 converts the residual from "undetectable drift/entrenchment"
to "entrenchment requires continuously re-winning against a fixed reference".**
That raises the attacker's cost and catches ordinary (non-adversarial) drift —
the photo→illustration walk that motivated the ADR — but it does not close
pixel injection. Stated plainly so the ADR's "compensating control" language is
not read as more than it is.

### INFO (code review) — the D6 hint rides the combined judge+planner call

The scoring judge also reads "recent iterations have rarely improved" — a mild
anchoring prior on the absolute scores that feed the band decision. A
structural consequence of the single-call architecture, named not fixed.

### D5 — deliberately NOT implemented, and the reviewer challenged it

`duels_enabled = duel_band > 0 and edit`; D5's own bounds disable tie-advance
in edit mode; therefore D5's entire live domain is t2i duels, which the ADR
explicitly defers. The code reviewer constructed the two non-dead readings
(tie-advance in the batch bracket; out-of-band composite ties in t2i) and found
both foreclosed — respectively by D3's deterministic earliest-arm rule and by
"a duel TIE" presupposing a duel. Implementing it now would be unreachable code
behind a flag nobody can trip. Recorded in the ADR Changelog with its revisit
trigger (t2i duels landing).

## Specific negatives verified

- The revert's plumbing is consistent: `best`, `best_cfg` (re-snapshotted per
  revert so a later mutation cannot corrupt `anchor_cfg` for a second revert),
  `best_duel_img`, `current_source`, and lineage selection via `reverted`.
- History marking mutates only pre-existing `improved`/`is_best` booleans,
  skips `judge_error` records shape-intact, and the triggering iteration's own
  record is written with both flags forced false — no new keys, no free text
  (F8-P / Finding 9 hold).
- `plateau_streak` and `no_improve` are correctly disjoint; the two plateau
  triggers cannot double-schedule or starve each other; the t2i and duels-off
  fallbacks are exactly preserved.
- Revert→re-climb churn is bounded: generations by `gens_used`, judge calls by
  the iteration count, and a repeat revert is blocked until a fresh promotion.
- D6's hint is one of two module-level constants selected by loop-owned
  integers, passes `_assert_no_paths`, carries no endpoint-controlled bytes,
  and widens F8-P by nothing.

## Forward constraints

1. The pass-gate bypass of D4 (LOW above) is open. If refine is ever driven by
   automation that trusts `passed`, decide what a pass that loses its anchor
   duel means before that wiring lands.
2. D5 revisits only when t2i duels land; the ADR Changelog carries the trigger.
3. The spurious-revert-under-long-stall residual is bounded by the N-iteration
   cadence; if runs get much longer, revisit the cadence rather than the rule.
4. TECH_DEBT 2026-07-26 (catalog truncates trained instruction templates to 64
   chars) is a live blocker for the face-swap end-to-end test, not for this code.

## Proof

`test_refine.py` 602 → 629 (0 failures); full battery 29/29; pyright 1026 =
baseline; `just policy-test` 36/36; gitleaks clean.
