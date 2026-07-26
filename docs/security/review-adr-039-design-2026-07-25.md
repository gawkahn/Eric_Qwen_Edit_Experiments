# Security design review — ADR-039 refine v3 promotion gate (2026-07-25)

AI-Disclosure: security-auditor subagent ran on Claude Fable 5
(`claude-fable-5`), explicitly confirmed — no model fallback. Parent session
(Fable 5) folded the findings into the ADR before any code, per §12
(ADR → security review → code). Grant owns acceptance.

Scope: the PROPOSED design in `docs/decisions/ADR-039-refine-v3-promotion-gate.md`
— swap-paired pairwise duels replacing the score comparison inside a noise
band, sideways cap + seed-batch escape, anchor duel, annealed acceptance,
planner hint. `comfyless/refine.py` is Red Zone. No code at review time.

**Verdict: no CRITICAL. One HIGH and four MEDIUMs, all folded as text
amendments before slice 1 — no design reversal required.**

The reviewer's framing of the core shift: promotion authority inside the band
moves from a code-computed comparison of two F6-coerced integers to a
directly parsed judge selection — a new output shape with no parse contract
yet — and a duel can promote a candidate whose composite is LOWER than
best's, which today's code never does. The direction (ties keep incumbent;
drift must win twice under swap) is fail-closed against the reward-hacking
threat the ADR targets and is the correct inversion of the accepted
D2-amendment MEDIUM.

## Findings and disposition

| # | Finding | Severity | Disposition |
|---|---|---|---|
| 1 | **Duel-unavailable fallback was fail-OPEN**: the draft said a failed duel falls back to the composite rule, which inside the band silently restores the exact behavior this ADR supersedes — so a judge that scores fine on the absolute pass but returns malformed duel output regains pre-v3 promotion per iteration, unnoticed. The naive fix (silent keep-incumbent) creates the mirror denial: an always-erroring duel judge freezes the run at the first promoted candidate while burning generations to the cap. Also noted: "budget cannot seat two candidates" is structurally DEAD (`resolve_judge_max_images` guarantees ≥ 2), so judge error is the ONLY live unavailability | HIGH | **Folded** — a failed duel resolves as NO PROMOTION, never composite-fallback; each failed duel call feeds the existing `JUDGE_ERROR_ABORT_AFTER` accounting so a broken duel judge aborts loudly instead of freezing; one order succeeding is explicitly not a duel result; the dead clause is acknowledged as dead. |
| 2 | **Duel output contract and authority unspecified** — nothing said the duel carries zero override authority, leaving room for an implementation to reuse `parse_verdict` and hand the duel a second planner channel (two extra override-bearing calls per banded iteration, tripling the F1 surface), or to feed duel free text into `prev_critique_text` → LoRA offers → history | MEDIUM | **Folded** — own code-owned contract constant (the scoring contract describes `overrides` and would be actively wrong here); closed-enum parse with reject-unknown; does NOT reuse `parse_verdict`; any override/critique content DISCARDED and never persisted into LLM-visible context; code-owned label→candidate mapping pinned by test; minimal code-owned user text (no history, no offers). |
| 3 | **D5 annealing revived tie-promotion for the first budget third**, triggered by exactly the position-bias noise the ADR cites as the dominant failure mode — ~33 iterations of a 100-iteration run back under the accepted D2-amendment MEDIUM, compounding F8-E source advancement in edit mode, with the drifted lineage then holding incumbency into the strict phase | MEDIUM | **Folded as a bounded, deliberate reopening** — three bounds: tie-streak cap (the limit held in reserve since the D2-amendment review), tie-advance DISABLED in edit mode (where the drift evidence lives), and `--anchor-duel-every` ≤ the tie-advance window so D4 is a real compensating control. |
| 4 | **D4 anchor lifecycle unspecified** — re-reading the first best from `candidates/` would reopen the TOCTOU class that ADR-037's D5 anchor amendment and ADR-038's `pin_static_refs` both closed, and it is live rather than theoretical: ADR-038's accepted residual is that concurrent runs sharing an `--output-dir` cross-overwrite `candidates/` with colliding stems, so the anchor duel could compare against a FOREIGN run's image and revert to a config whose image never existed here. Revert semantics and history-marking scope also unstated | MEDIUM | **Folded** — first best pinned BY VALUE at first promotion (capped-loader bytes + `snapshot_config`), held for the run; duel and revert consume only pinned values, never a path re-read or sidecar reconstruction; revert restores the snapshot and (edit mode) the pinned image as `current_source`; history marking mutates only existing boolean flags, no new keys or free text (F8-P unchanged). |
| 5 | Swap-pair payload symmetry not stated — per-call recomputation could give the two orders different reference sets, an evidence mismatch presenting as "disagreement" and silently resolving as a tie | LOW | **Folded** — payload set computed ONCE per duel; orders differ only in candidate order; named negative test. |
| 6 | Seed-batch bracket had no tie-break, no bracket shape, no iteration/GPU accounting, and no seed-lattice tie-in | LOW | **Folded** — single-elimination in generation order with a deterministic judge-independent tie-break (earliest arm wins — the anti-drift direction); batch generations count against `--max-iterations`; batch seeds from the shared monotonic counter; in-bracket duel errors follow the amended duel-error rule. |
| 7 | New numeric flags unvalidated — a NaN `--duel-band` makes every band test False and silently reverts the run to the superseded rule (the `--w-*` NaN precedent) | LOW | **Folded** — finite/range validation at entry, exit 2. |
| 8 | Within-band promotion of a LOWER-composite challenger retires the "composite never decreases across promotions" invariant | INFO | **Folded** — named explicitly as knowingly retired; pass/`--until-score` gates pinned as absolute-composite-only, checked pre-duel. |
| 9 | Duel call hygiene (temperature 0, downscale, label discipline, `max_tokens` for a two-image prompt ×2 calls) should be stated, not assumed | INFO | **Folded** — inherited-constraints list in D2 plus a recipe-header token-budget note. |

Assessment-question cross-map from the reviewer: Q1 → finding 2 + 9;
Q2 → finding 1; Q3 → direction correct but finding 3 undercut it early;
Q4 → finding 4; Q5 → arithmetic sound, findings 5 and 1; Q6 → finding 6;
Q7 → the Supersession's judge-error clause was the gap (finding 1).
