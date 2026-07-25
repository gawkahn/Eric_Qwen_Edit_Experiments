# Security review — ADR-037 D2 addendum (stagnation seed escape, --explore-after)

AI-Disclosure: security-auditor and code-reviewer subagents ran on Claude Fable 5
(`claude-fable-5`), both explicitly confirmed in their reports — no model
fallback occurred. Parent session (Fable 5) folded the findings; Grant owns the
decision.

Scope: uncommitted diff adding the stagnation seed escape to `refine_loop`
(`comfyless/refine.py`, Red Zone path): after `no_improve >= --explore-after`
(default 2, <=0 disables) consecutive non-improving iterations, each further
non-improving derivation resamples its seed via the counter shared with the
no-op escape (renamed `noop_resamples` → `seed_resamples`; no-op branch takes
if/elif precedence). Live motivation: 12 straight declines reprinting one
seed-tied flaw while the planner rewrote prompts against it.

## Disposition summary (parent session, folded same day)

| Finding | Severity | Disposition |
|---|---|---|
| Shared-counter uniqueness across MIXED trigger sequences untested (auditor LOW; reviewer SHOULD incl. tie-chain skip-value case) | LOW/SHOULD | **Fixed** — mixed no-op/stagnation pin (strictly increasing seeds) + tie-chain skip-value pin (123→124→126) added. |
| Help-text claim "patience <= explore-after stops first" untested | SHOULD (part) | **Fixed** — pinned (patience=2 run stops at 3 iters, zero resamples). |
| Stale unconditional attribution claims (code comment + two test comments) after the escape made "changed config keeps pinned seed" conditional | SHOULD/NIT | **Fixed** — all three conditionalized on the threshold; refine_loop docstring now names the escape. |
| Planner trajectory context blind to resamples: an escape iteration changes prompt AND seed, history carries no `seed_resampled` flag → possible planner mis-attribution | SHOULD (design) | **Accepted deferral** — ADR changelog sentence + TECH_DEBT 2026-07-24 entry (history-field addition touches F8-P, own slice; escapes fire only on stagnant runs). |
| Tie-drift acceptance record accuracy: escape extends the accepted tie-chain drift MEDIUM to the seed dimension (same cap bound) | INFO | **Recorded** in ADR changelog. |
| Judge-error iterations count toward the threshold (zero evidence of a seed-tied flaw) | INFO | Accepted — consistent with patience's pre-existing accounting; D3 abort bounds consecutive judge errors at 3 anyway. |

## Auditor verdicts (condensed)

**Q1 — no meaningful authority expansion.** The judge can trigger resamples by
scoring parity/declines, but it could already do so at will via empty
overrides (no-op branch), could already freeze/extend the loop via scores,
and resamples are bounded by `--max-iterations`. The seed VALUE is
code-computed (iteration-0 metadata pin + loop-local counter) and appears in
NO judge-visible surface — payload, history records, and stubs carry no seed,
counter, or escape state; prediction would gain nothing (a seed is not a
secret and selects only noise).

**Q2 — alias safety holds.** Both mutation sites operate on the fresh
deep-copied `cfg` from `apply_overrides`; `best_cfg` is an independent
deep-copy snapshot taken before re-binding — the decline-path escape (where
`source_cfg` IS `best_cfg`) cannot write through. No planner-visible or
persisted surface carries resample state (stderr log + operator sidecar
only).

**Q3 — flag value classes safe.** argparse `type=int`; <=0 disables
(fail-safe, degrades to pre-diff behavior); huge N never fires
(`no_improve < max_iterations`). Patience ordering verified: both patience
breaks precede `apply_overrides` and the escape, so `0 < patience <=
explore_after` stops first — as the help text states.

**Q4 — misc clean.** Log lines interpolate ints only; no scope creep (three
declared files; the counter rename is load-bearing, not cosmetic); ADR
append-only discipline respected.

Reviewer's monotonicity proof (condensed): every escape assignment is
`base + c` with `c` strictly increasing per assignment and `base` equal to
best's seed at assignment time, which is non-decreasing across promotions —
so assigned seeds strictly increase; tie chains skip values but never
collide; F1's two-key allowlist means the planner can never place a value in
`base["seed"]` before the block runs.

Test state at fold: test_refine.py 390 passed / 0 failed; battery re-run at
commit.
