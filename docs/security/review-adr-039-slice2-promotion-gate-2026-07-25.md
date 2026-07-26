# Security review — ADR-039 slice 2: the banded promotion gate

AI-Disclosure: Claude (Fable 5) authored the code and this record; reviewed by
`security-auditor` (Fable 5) and `code-reviewer` (Fable 5); Grant reviewed.

Date: 2026-07-25
Surface: `comfyless/refine.py` (§12 Red Zone path — refinement-loop judge),
`test_refine.py`.
Decision record: `docs/decisions/ADR-039-refine-v3-promotion-gate.md` (accepted).
Prior records: `review-adr-039-design-2026-07-25.md`,
`review-adr-039-slice1-duel-primitive-2026-07-25.md` (its "Forward constraints
for slices 2-4" were binding on this diff).

**Reviewer model pin:** both agents invoked with `model: "fable"`; transcripts
show `claude-fable-5` on every turn — **no Fable→Opus fallback**.

**Reviewer tooling caveat (third slice running):** neither agent had a working
shell this round, so neither ran `git diff`; both reviewed working-tree state
and flagged that they could not rule out hunks elsewhere in the 3200-line file.
The parent session verified `git diff --stat` covers only the declared regions.
This has now recurred on every review of this ADR — logged as tech debt rather
than re-noted a fourth time.

## Scope of the change

Wires the slice-1 duel primitive into `refine_loop`'s promotion gate. Inside
`--duel-band` of best (exclusive at both ends, epsilon-guarded) a swap-paired
duel decides; outside it the strict-composite rule stands; **ties keep the
incumbent everywhere**, superseding ADR-037's D2 tie-promotion. A void duel
promotes nothing, never falls back to the composite, and charges
`getattr(e, "failed_calls", 0)` to `JUDGE_ERROR_ABORT_AFTER`. The incumbent's
image is pinned by value at judge resolution. Duels are edit-mode only (t2i is
ADR-039-Deferred). CLI: `--duel-band`, `--duel-recipe`.

## Verdict

**`security-auditor`: no CRITICAL, no HIGH.** One MEDIUM (missing binding
test — fixed), one LOW (vacuous flag test — fixed), three INFO (recorded
below). Assessment: "the code matches the ADR and the slice-1 forward
constraints on every point I could verify in source"; the gaps were in test
enforcement, not behavior.

**`code-reviewer`: no boundary violation, no security regression, no scope
creep.** One MEDIUM (the same missing test), four LOW (three fixed, one
recorded), two INFO (one fixed, one recorded).

## Findings and disposition

### MEDIUM (both reviewers) — the binding flags-only history test was missing — FIXED

Slice 1's record made one test binding on this slice: a duel failure must put
nothing but ordinary structural flags into the judge-bound history block
(Finding 9 / F8-P). The implementation was correct — the `DuelError` message,
which embeds the endpoint URL and up to 300 chars of endpoint-controlled body,
goes only to the operator log, and the iteration records through the ordinary
`history_record` — but **nothing failed if that changed**. A future edit
annotating the record with duel context "for debuggability" would feed
endpoint-controlled bytes back into LLM context silently. This is the same
class of vacuous-pin gap the slice-1 review closed twice.

**Fixed:** two assertions on the void-duel run's `histories_seen` — no error
text, no "duel"/"http" substring anywhere in the serialized history, and every
record's key set within the closed `history_record` set.

### LOW (security) — the `--duel-band` exit-2 tests were vacuous — FIXED

The four bad-band invocations also carried an unknown `--judge-backend`, which
exits 2 on its own, so a regression deleting the `math.isfinite` check would
have passed the suite — while a NaN band makes every band test False and
silently reverts the whole run to the superseded promotion rule. **Fixed:**
stderr is captured and the specific rejection message asserted, plus a new
check that the band check fires BEFORE judge-backend resolution (so a bad band
never costs a registry read or a live autodetect GET).

### LOW (code review) — band membership had no epsilon guard — FIXED

The sibling pass-gate comparison is epsilon-tolerant because composites are
inexact float sums (`0.6*a + 0.4*b`); the band test was a bare `<`, making
"exclusive at both ends" FP-fuzzy — a nominally-exact-boundary delta could
round a ULP low and fall inside the band. **Fixed:** `< duel_band - 1e-9`,
with the reasoning and the consequence (a band below float noise reads as no
band) recorded at the site.

### LOW (code review) — seven doc/help sites described the superseded rule — FIXED

`--patience` help, `--explore-after` help, `DEFAULT_PATIENCE` /
`DEFAULT_EXPLORE_AFTER` comments, the `refine_loop` docstring's lineage and
edit-source lines, the `current_source` comment, the stagnation-escape comment,
and its log string all still said "improvement" where the counter now means
promotion, or described tie-promotes-newer. These document the exact rule this
slice changes, so they were corrected here rather than deferred.

### INFO (code review) — the stale-pin assertion was vacuous — FIXED

`_FakeGenE` painted every candidate identically, so a stale `best_duel_img`
was indistinguishable from a correctly-updated one. **Fixed:** the fake
generator paints candidate N with blue channel N, and the duel test asserts the
exact (challenger, incumbent) pixel pairs across a promote-then-duel sequence.

### INFO (security) — near the ceiling, the F8-E veto is TOTAL until slice 4 — ACCEPTED, ESCALATED

Quantification of slice 1's accepted residual, and the most important thing in
this record. With the default band 1.0 and axis maxima of 10, once best's
composite exceeds `10 - duel_band` (≥ 9.0 — **the saturated regime this ADR
exists to address**), no challenger can reach `best + duel_band`, so the
out-of-band promotion path is unreachable and *every* promotion needs the
incumbent's duel consent. A self-identifying pixel injection that entrenches
early therefore holds an absolute veto for the rest of the run, and in edit
mode also owns the `current_source` lineage.

Consequence for sequencing: **D4's anchor duel is load-bearing, not
belt-and-braces.** The operator-facing reassurance "outside the band the scalar
still promotes" is vacuous in the regime that matters. Bounded today by
`--max-iterations` and the operator watching the run.

### INFO (code review) — sideways duel-win chains are unbounded until D3 — ACCEPTED

A chain of consistent duel wins at equal-or-lower composite resets `no_improve`
every iteration, so neither `--patience` nor the stagnation escape bounds it.
D3's sideways cap (slice 3) is the control. Bounded by `--max-iterations`.

### INFO (security) — charged-0 voids withhold the reset without incrementing — ACCEPTED

A `failed_calls=0` void (the slice-1 internal backstop) freezes promotion
without ever reaching the abort threshold, and preserves a stale counter value
across iterations, so two much-later non-consecutive scoring errors could
complete a "3 consecutive" abort. Chosen deliberately in slice 1 (our bug must
not masquerade as a flaky judge); live exposure ≈ nil (the backend is validated
at entry and the backstop is unreachable behind `select_duel_refs`). Named here
so slice 3 — where bracket duels multiply void opportunities — inherits it
consciously.

### INFO (security) — band validation is CLI-layer only — TECH_DEBT

`refine_loop` trusts its `duel_band` argument; a NaN passed programmatically
silently disables duels and restores the superseded rule with no notice. Latent
today (`main()` is the only caller and validates first); it becomes real the
day refine is reachable through MCP or the `--json` bridge — both already
flagged in CLAUDE.md as Red-Zone-on-scope-change. Logged in TECH_DEBT.md with
that exposure as the trigger.

## Specific negatives verified

- Gate matches D1: exclusivity at both ends, consistent-win-only promotion,
  ties keep the incumbent, no composite fallback on a void duel, pass and
  `--until-score` read the absolute composite before any duel.
- The relocated error-counter reset has no escape: an iteration whose scoring
  succeeds and whose duel succeeds (or never runs) resets exactly as before;
  mixed scoring/duel failures accumulate into the one counter; config errors
  charge zero without inflating it.
- `best_duel_img` is in sync on every path (`best` is assigned in exactly two
  places; the pass-gate assignment exits the loop) and is derived from the
  in-memory decoded candidate — the duel path never re-reads `candidates/`,
  which is where ADR-038's accepted cross-run-collision residual lives.
- No new authority channel: the duel outcome influences exactly one boolean.
  Nothing duel-related reaches history, `prev_critique_text`, or verdict.json.
- Bounded cost: 2 judge calls per banded iteration, no retry, second call
  skipped after the first fails, one retained judge-resolution image.
- `--duel-recipe` cannot escape `judge_recipes/` (separator rejection, forced
  `.toml`, kind gate fail-closed both directions, entry-time load).

## Forward constraints for slices 3-4

1. **D4 is load-bearing** (see the escalated INFO). Do not let slice 4 slip far
   behind slice 3; if it must, say so to the operator in the band notice.
2. Slice 3's bracket multiplies void opportunities — inherit the charged-0
   observation, and log any bounded coverage rather than truncating silently.
3. Still owed from the ADR's named list: the bracket's earliest-arm tie-break,
   the anchor-duel revert on an old-best win, and the sideways cap's own
   negative tests.
4. If refine is ever exposed via MCP or `--json`, re-validate `duel_band` (and
   the other loop floats) at that boundary.

## Proof

`test_refine.py` 544 → 571 (0 failures); full battery 29/29; pyright 1026 =
baseline; `just policy-test` 36/36; gitleaks clean.
