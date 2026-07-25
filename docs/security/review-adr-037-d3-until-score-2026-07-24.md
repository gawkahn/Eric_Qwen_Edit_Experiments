# Security review — ADR-037 D3 amendment (--until-score float composite gate)

AI-Disclosure: security-auditor and code-reviewer subagents ran on Claude Fable 5
(`claude-fable-5`), both explicitly confirmed in their reports — no model
fallback occurred. Parent session (Fable 5) folded the findings; Grant owns the
decision.

Scope: uncommitted diff adding an optional float SCORE to `--until-score`
(argparse nargs='?'), a validated composite pass gate in `refine_loop`
(`until_composite`, epsilon-tolerant, REPLACES the both-axes gate), operator
advisory helpers (`_parse_until_score`, `_nearest_reachable_composite`), tests,
and the ADR-037 changelog entry. `comfyless/refine.py` is a Red Zone path.

## Disposition summary (parent session, folded same day)

| Finding | Severity | Disposition |
|---|---|---|
| UNREACHABLE target silent: with non-default weights capping the max composite below the target, the lattice note never fired and the run rode blind to the cap (found independently by both reviewers) | SHOULD / LOW | **Fixed** — helper returns None on unreachable; main() emits a loud UNREACHABLE warning naming the max possible composite and the cap. Pinned (weights .5/.3, target 9 → None). |
| Composite weights (`--w-*`) are unvalidated CLI floats that now control TERMINATION: NaN weights make every gate compare False (silent cap ride) and scramble the lattice scan; inf fires the gate immediately | LOW | **Fixed** — `math.isfinite` check on both weights at main() entry, exit 2. Range stays operator-domain (warn-don't-block). |
| No pin that valued mode raises the default cap (a `bool()` → `is True` tidy-up would silently keep cap 10) | NIT | **Fixed** — `_resolve_max_iterations(None, bool("9.6")) == MAX_ITERATIONS_SANITY_CAP` pinned. |
| No pin that the composite target stays out of persisted/judge-visible surfaces | INFO | **Fixed** — verdict-record key-set + no-"until" pin on the composite-run's on-disk record. |
| Vault user docs (`Comfyless_Refine.md`) lacked the new flag | INFO | **Fixed** — manual updated (status, gate semantics, flags table, edit-mode section, scan-vs-DB catalog note). |
| Security-auditor process artifact required by the Red Zone gate | SHOULD (process) | **This document.** |

## Auditor verdicts (condensed)

**Q1 — termination authority: no expansion.** Both gates consume the same two
F6-coerced integers (`_coerce_score`: finite, rounded, clamped 1-10, bool
excluded, OverflowError-guarded); the composite is a deterministic monotone map
of those ints given finite weights. The judge cannot produce non-finite or
out-of-lattice composites, cannot exploit the 1e-9 epsilon (lattice spacing ≈
min-weight ≫ 1e-9), and its advisory verdict string remains never consulted
(F8). Pathological composites require pathological OPERATOR weights — closed
by the finite check above; remaining range abuse is cap-bounded operator
self-infliction.

**Q2 — input validation: sound.** Only argparse output reaches
`_parse_until_score` (str, or the literal True/False const/default; the
identity checks short-circuit before `float()`, so bools never coerce).
nan/inf/"Infinity" rejected via `math.isfinite`; range 1.0-10.0 enforced;
empty string rejected; failure is fail-closed (RefineError → exit 2, before
any catalog/GPU work).

**Q3 — no leakage: confirmed.** `until_composite` reaches only the loop gate
and log lines — not `judge_candidate`, not `verdict_record`, not history
records, not sidecars, not candidate metadata. `pass_threshold` likewise never
enters judge context. Now pinned by test.

**Q4 — misc: clean.** Log format interpolates operator-validated floats only;
`gate` bound on both branches; argparse `nargs='?'` cannot steal a following
flag token (pinned); the 9.7-target test is a genuine gate-replacement
negative. No scope creep.

Code-reviewer clean-checks (condensed): all `args.until_score` read sites
accounted for under the three-state type; epsilon direction correct;
`verdict_passes` has exactly one call site, provably inert in valued mode;
pass-break finalization sound (an earlier candidate at/above the target would
have broken first); old bool-shaped tests remain valid; help/docstring/ADR
mutually consistent.

Test state at fold: test_refine.py 383 passed / 0 failed; battery re-run at
commit.
