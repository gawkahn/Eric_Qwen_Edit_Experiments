# Security review — ADR-037 D5 amendment (judge anchor = original seed)

AI-Disclosure: security-auditor and code-reviewer subagents ran on Claude Fable 5
(`claude-fable-5`), both explicitly confirmed in their reports — no model
fallback occurred. Parent session (Fable 5) folded the findings; Grant owns the
decision.

Scope: uncommitted diff switching the edit-mode judge's comparison SOURCE from
`current_source` (loop-owned, tie-promotion-advancing accepted candidate) to
the operator's original `edit_source`. Live motivation: a 100-iteration tie
chain drifted a photo into a 3D-anime figurine at constant 9.6 (candidate-33
evidence) because preservation was only ever checked stepwise against the
drifting parent — the D2-amendment MEDIUM materializing benignly.

## Disposition summary (parent session, folded same day)

| Finding | Severity | Disposition |
|---|---|---|
| Anchor pinned to a PATH, re-opened per iteration: mid-run swap silently retargets preservation; mid-run delete = fatal at iteration N (code-reviewer SHOULD; auditor LOW "TOCTOU by design") | SHOULD/LOW | **Fixed** — anchor loaded ONCE at loop entry (bytes pinned). Slice-B LOW-3's re-open rationale applied when the judged source changed per iteration; the anchor is run-constant. Fatal-on-absent moves to entry; F5 caps at load; memory bounded by SEED_IMAGE_MAX_PIXELS; judge_candidate still downscales per call. |
| F8-E surface widened: adversarial text in the seed gets N full-fidelity judge exposures per run instead of one | LOW | **Accepted + recorded** in ADR changelog. Bounded by F1/F2/F6/F7 structural verdict guards. Trigger written down: the soft rubric mitigation stops sufficing the day refine is exposed to agent/remote callers — harden before that wiring. |
| Fallback `_DEFAULT_EDIT_RUBRIC` got only the label swap — no cumulative-drift language, stale CANDIDATE description, false "shipped verbatim" header | INFO/NIT | **Fixed** — drift sentence + corrected CANDIDATE description ported; header comment corrected to acknowledge divergence from the .toml. |
| Docstring/comment drift: refine_loop EDIT MODE paragraph implied judge source == current source and still said "strict improvement"; current_source comment ambiguous vs edit_source | NIT | **Fixed** — docstring names the anchor/lineage split and the D2 `>=` rule; comment disambiguates the two roles explicitly. |
| Tie-chain scenario didn't re-assert the anchor | INFO | **Fixed** — 8×8-anchor assertion added to the tie-promotion edit test (the shape of the live failure). |
| Mitigation is behavioral, not structural: a constant-parity judge can still tie-promote drifted lineage to the cap | INFO | **Accepted residual** (unchanged D2 MEDIUM disposition); tie-streak cap remains the deferred reserve, decided by the anchored stress-test rerun. |

Auditor's Q3 verdict (key claim): every path by which a candidate becomes
`best`/`current_source` — normal promotion, tie promotion, pass early-exit —
requires a verdict whose judge call contained the fixed original anchor;
judge-error iterations advance nothing. No path remains where drifted lineage
advances without comparison against the original. Q1: no trust-disposition
change (both anchors live on operator-controlled disk; the new arrangement is
strictly narrower — a swap perturbed judging only, and the entry-load fold
removes even that). Q4: role labels stay path-free (Finding 4 discipline);
recipe wording brace-free; the 8×8-vs-4×4 test pin is genuinely
regression-discriminating.

Test state at fold: test_refine.py 364 passed / 0 failed; battery re-run at
commit.

Full auditor and reviewer reports retained in the session transcript; the
substantive findings and verdicts are reproduced above verbatim in condensed
form.
