# Security design review — ADR-038 multi-reference refinement (2026-07-25)

AI-Disclosure: security-auditor subagent ran on Claude Fable 5
(`claude-fable-5`), explicitly confirmed — no model fallback occurred.
Parent session (Fable 5) folded the findings into the ADR before any code, per
§12 order-of-operations (ADR → security review → code). Grant owns acceptance.

Scope: the PROPOSED design in
`docs/decisions/ADR-038-refine-multi-reference-edit.md` — porting generate's
multi-reference conditioning into refine's edit loop, plus an opt-in `:judge`
qualifier putting up to two operator-pinned reference images into every judge
call alongside the D5 anchor and the candidate. `comfyless/refine.py` is a Red
Zone path. No code written at review time.

**Verdict: no CRITICAL, no HIGH. Approvable for implementation with the three
MEDIUMs folded as textual amendments** — done same day (ADR changelog
2026-07-25).

## What the review affirmed

The advancing-source / static-ref split (D1) is the correct
unrepresentability fix — no path in the current loop lets a candidate
displace a static ref or pin the loop source (only promotion writes
`current_source`; decline reverts config only; the stagnation escape touches
only the seed). `:judge` opt-in with an entry-time cap matches the
fail-closed-at-boundary precedent. Reusing one spec parser and the existing
`RefPathError` latch avoids new trust machinery. Seed-sidecar `ref_images`
stay dropped. Role labels leak nothing structurally: the only new text is a
code-owned constant formatted with an integer index.

## Findings and disposition (all folded into the ADR before code)

| # | Finding | Severity | Disposition |
|---|---|---|---|
| 1 | D5 named `load_seed_image_capped` (caps only) for a file class ADR-035 6c built `load_ref_image_capped` for (format allowlist, regular-file guard, single-read + SHA-256). Static refs are arbitrary user files; the weak loader would `Image.open` across PIL's full plugin zoo (EPS → Ghostscript; rare C decoders carry Pillow's CVE history) and accept files `generate --ref-image` already refuses | MEDIUM | **Folded** — D5 now names `load_ref_image_capped`; `load_seed_image_capped` stays reserved for the seed/anchor. |
| 2 | Pinning only the judge's copy reopened the anchor-amendment TOCTOU class BETWEEN channels: a judge-marked ref is consumed twice (pinned bytes for the judge, a re-read PATH for generation), so a mid-run swap has the judge scoring identity against bytes generation no longer uses — the loop's core invariant (scores describe the generation's inputs) breaks silently | MEDIUM | **Folded** — D5 now pins ALL static refs by value at entry and copies them into a loop-owned `refs/` dir under the run dir; the copies are what ride the wire and what the judge sees. Side effect: dissolves most of finding 5. |
| 3 | The `:judge` grammar and its wire containment were unspecified in a parser where a wrong split silently changes what the model conditions on (ADR-035 decision 1 made mode parsing strict for exactly this reason). A leaked `vl:judge` mode would fail the daemon's allowlist as a plain error — fatal, not `RefPathError`, so no latch and misattributed | MEDIUM | **Folded** — D2 specifies fixed suffix order, bare-`:judge` default mode, the two-suffix colon-filename interaction, and binding wire containment (stripped before daemon/generate; negative test pinned). |
| 4 | F8-E widening undersold as "inherited": a third/fourth operator image gets N full-fidelity judge exposures per run, and the identity ref is DISTINGUISHED — D4 tells the judge to describe and match against it, so rendered directive text there gets rubric-granted attention, flowing into the critique → LoRA offers | LOW | **Folded as accepted residual** — rubric line must name ALL labeled images uniformly (recipe + fallback constant); structural bounds (F1/F6/F7) unchanged and are what keep it LOW; agent/remote-exposure hardening trigger restated. |
| 5 | Static-ref count composes with the prepended loop source: 8 typed refs → 9-entry wire list → daemon count refusal at iteration 0 (plain error, fatal, mid-run), violating D3's own fail-closed-at-entry principle | LOW | **Folded** — effective cap is `_MAX_REF_IMAGES - 1` (7), refused at entry naming the reserved slot. |
| 6 | Latch notice says "add `--ref-root` for this run's directory", but the common identity ref lives in `~/photos/…` — wrong guidance, and it invites the over-broad `--ref-root ~` that ADR-035 Finding 6 warns about; nearly every multi-ref daemon run would latch at iteration 0, quietly losing the warm daemon | LOW | **Folded** — notice names the REFUSED PATH'S directory; largely dissolved anyway by finding 2's loop-owned copies. |
| 7 | vLLM cap drift (repo constant tracking endpoint config) and stale-daemon silent ref-drop (pre-`ref_drop_strict` daemon generates WITHOUT the identity ref while the judge scores the mismatch all run) | INFO | **Folded** — drift backstop named (`JUDGE_ERROR_ABORT_AFTER`), constant carries a `--limit-mm-per-prompt` comment, restart-daemons-on-upgrade note carried forward. |
| 8 | Advancing/static separation sound — pin it with negative tests; label template should pin integer-only interpolation | INFO | **Folded** — ADR now carries a slice plan naming all six negative tests. |
| 9 | Planner-authority deferral is correct and already structurally excluded — cite the standing control rather than relying on intent | INFO | **Folded** — deferral now cites ADR-035 decision 7 / Finding 8 (`ref_images` outside the F1 allowlist). |

Deferred-list check: per-ref weighting, t2i multi-ref, and raising the vLLM
cap are genuinely out of scope. The two items that should NOT have been
deferred were not on the list but were ABSENT — the `:judge` grammar and the
static-ref pinning discipline (findings 2 and 3) — and both are now decisions
in this ADR rather than implementation-time choices.
