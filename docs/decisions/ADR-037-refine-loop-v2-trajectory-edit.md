# ADR-037 — Refinement loop v2: trajectory-aware hill-climb + edit-mode refinement

Status:   accepted (Grant, 2026-07-23)
Date:     2026-07-23
Vision:   docs/vision/slice-v5-keyframe-authoring-refine-v2.md
Relates:  ADR-027 (refinement loop v1 — superseded-in-part on acceptance),
          ADR-033 decision 7 (keyframe authoring), ADR-035 (ref-image surface),
          ADR-036 (flux2/klein reference conditioning)
AI-Disclosure: Claude (Fable) authored; Grant reviewed.

## Context

ADR-027's loop is a stateless greedy hill-climb: each iteration judges one
image against the current config and applies planner overrides onto the
LATEST config unconditionally (`refine.py:1195`); `best` is tracked only for
winner finalization. Grant's first full hot runs (2026-07-18) surfaced two
failure modes now logged in the Backlog:

1. **The walk drifts off the peak.** After a regression the climb continues
   from the regressed config; observed: 10 iterations, best was #2, never
   re-approached.
2. **The planner is trajectory-blind and timid.** `judge_candidate` sees one
   image + current config — no past scores, no prompt diffs — so it cannot
   reason "that change hurt, reconsider," and its rewrites are minimal
   appends yielding minimal deltas.

Meanwhile the video program (ADR-033 decision 7) needs **keyframe authoring**:
evolving keyframe N → N+1 with a scene lock, via judge-gated single-op edit
chains on an edit model (qwen-edit). An edit chain's input IS the previous
accepted output — the loop must know which image is currently accepted
(`best`), must not keep editing a regressed output (climb-from-best), and its
planner must know what was already tried (history). The edit loop's hard
requirement is exactly the v2 fix; building edit mode on the v1 stateless
loop would build the trajectory machinery twice. Hence one ADR, one new loop
version.

`refine.py` is a §12 Red Zone path (LLM output influencing generation
params). This ADR precedes code; security-auditor design review precedes
code; findings fold back here.

## Decision

### D1 — Trajectory state (`RunHistory`)

The loop keeps a per-run, in-memory list of per-iteration records:

```
{iteration, scores: {prompt_adherence, aesthetics, composite},
 prompt_excerpt, lora_ops_applied, improved: bool, is_best: bool,
 judge_error: bool, accepted: bool}          # accepted: edit mode only
```

- **Path-free by construction, structurally gated by key** (reworded per
  Finding 3, MEDIUM): records are built ONLY from scores, resolved catalog
  names, weights, booleans, and capped prompt excerpts — never from
  `WorkingConfig.base` or filesystem strings. The serialized block passes
  the existing `_assert_no_paths` gate, which — stated precisely — checks
  dict KEYS against `_FORBIDDEN_CONTEXT_KEYS`; it cannot and does not scan
  values (legitimate prompts contain slashes). The negative-case obligation
  is therefore a CONSTRUCTION test: a config containing paths yields history
  records containing none — not a value-detection test.
- **Whole-run depth** (Grant decision 3): all iterations are kept.
  `prompt_excerpt` is the override/current prompt truncated to
  `HISTORY_PROMPT_EXCERPT_CHARS` (500) with an ellipsis marker; the whole
  serialized block is capped at `HISTORY_MAX_BYTES` (64 KiB). If the cap is
  ever exceeded, oldest entries are elided with a loud notice — a bound, not
  an expected path (10 iterations of excerpts+scores is ≤ ~10 KiB; the cap
  exists for the until-score sanity range).
- Past judge critiques are NOT replayed into history: critiques are
  LLM-authored free text, and re-entering them into future LLM context turns
  one bad judge output into a persistent steering channel.
- **`prompt_excerpt` provenance is planner-authored, and the design owns
  that** (design review Finding 1, HIGH): at every iteration after 0 the
  current prompt IS the planner's prior `overrides.prompt` — the same trust
  class as a critique. v1 already re-enters the LATEST planner prompt each
  call (`current_prompt`); history extends that to all past ones, so a
  steering-laden override would persist for the run even after
  climb-from-best displaces it. This is accepted as a documented **F8-P
  extension** (see Security findings below) rather than avoided, because
  structural-only excerpts (length deltas, token counts) carry almost no
  "reconsider that change" signal. Mitigations, binding on slice A:
  the serialized block labels every excerpt `"planner-proposed (untrusted)"`
  (the implemented label — field `prompt_provenance`; slice-A review NIT-1);
  only the OPERATOR's original target prompt is ever quoted unlabeled; total
  planner-authored characters across the block are budgeted at
  `HISTORY_PLANNER_TEXT_BUDGET` (8 KiB), truncating oldest excerpts first.
- **Judge-error iterations contribute `{iteration, judge_error: true}` and
  structural flags ONLY** (Finding 9, LOW) — never `str(e)`:
  `_post_judge` error text embeds the endpoint URL and up to 300 chars of
  endpoint-controlled response body, which must not enter future LLM context.
- **`lora_ops_applied` is binding, not descriptive** (Finding 10, LOW):
  history records ops post-ADR-015-resolution (resolved catalog names +
  clamped weights). Proposed-but-unresolved op "names" — arbitrary
  judge-authored text — never enter the block.
- If the byte cap ever forces elision, entries compact to scores+flags-only
  stubs (path-free by construction) before whole entries are dropped
  (Finding 13, INFO) — long until-score runs are exactly when losing the
  "early configs already failed" signal would amplify spend.

### D2 — Climb-from-best

Unified lineage rule (revised per Finding 6, MEDIUM): a candidate is
**promoted** iff its composite STRICTLY improves on `best`. The next
iteration's overrides are applied to the promoted candidate's config when
promotion happened, and to **`best`'s `WorkingConfig`** otherwise — ties
included, so config lineage and image lineage can never fork (in edit mode
the edit source follows the identical rule). Non-promoted candidates still
get full history records (`improved: false`) so the planner knows the
attempt failed. Ties count as non-improvement for patience.

**Snapshot semantics (binding):** `best`'s config is snapshotted **by value,
in memory** (deep copy including `base`) at candidate creation, and is NEVER
reconstructed from on-disk sidecars or metadata — sidecars legitimately
carry load paths, and re-deriving a working config from them would reopen a
file-derived channel (slice-3 constraint (a) territory). The current loop
mutates `cfg.base["seed"]` in place; a shallow snapshot would alias and
silently desync "best." The negative case asserts snapshot immutability
against later `cfg.base` mutation.

### D3 — Stop modes

- `--max-iterations` retained, default 10, hard ceiling raised to
  `MAX_ITERATIONS_SANITY_CAP = 100`.
- New `--until-score` flag: run until `verdict_passes(...)` (both axes ≥
  threshold), bounded by `--max-iterations` if explicitly given, else by the
  sanity cap. No other semantics change; patience still applies if enabled.
- The composite/threshold machinery is unchanged; the judge's advisory
  `verdict` string remains non-authoritative.
- **Consecutive-judge-error abort** (Finding 8, MEDIUM): generation precedes
  judging, patience defaults to disabled, and F7 makes a judge error consume
  an iteration — at cap 100 a dead endpoint would burn hours of blind GPU.
  New rule: `JUDGE_ERROR_ABORT_AFTER = 3` CONSECUTIVE judge errors abort the
  run loudly (fail-closed). Distinct from patience: patience measures
  non-improvement; this measures non-function. Applies in all modes.

### D4 — Planner authority is UNCHANGED

The F1 closed two-key allowlist (`overrides.prompt`, `overrides.loras`)
survives v2 exactly as-is. Edit mode adds ZERO planner-mutable keys; the
"prompt" simply carries the edit instruction. `parse_verdict`, the code-owned
`_JUDGE_OUTPUT_CONTRACT`, LoRA name-only resolution (ADR-015), weight clamps,
and the no-model-swap rule are untouched. The verdict JSON schema is
unchanged in edit mode — the rubric reinterprets `prompt_adherence` as
edit-instruction adherence + scene preservation, so F7 parsing and the
two-axis gate need no schema fork.

**What edit mode DOES change, named plainly** (Finding 11, LOW): (i) the
judge PAYLOAD contract — two `image_url` entries with role-only labels (D5);
(ii) the CLI ENTRY contract — seed+prompt both required, operator-typed
model (D5); (iii) history records gain `accepted`. And one accepted
trade-off: collapsing edit-adherence and scene-preservation onto one integer
means the promotion gate cannot distinguish "great edit, destroyed scene"
from "no-op edit, perfect scene." Accepted for v1 to keep the F7 contract
stable; revisit trigger: keyframe chains promote scene-broken outputs in
practice.

### D5 — Edit mode

- **Family gate:** explicit allowlist `_REFINE_EDIT_FAMILIES = ("qwen-edit",)`
  checked at loop entry against `detect_pipeline_class` family — loud
  `RefineError` before any GPU work for anything else (no silent t2i
  fallback). flux2klein is the expected first lift (Grant: its output may be
  simpler for the judge), gated behind a later changelog entry, not v1.
- **Edit-mode entry contract** (Finding 5, MEDIUM): `--seed-image` AND an
  explicit `--prompt` (the edit instruction) are BOTH required — the
  prompt-XOR-seed rule of v1 is relaxed in edit mode only. `--model` must be
  OPERATOR-TYPED in edit mode; the v1 seed-defaulting of `model` is refused,
  because the D5 family gate — the control deciding edit mode engages at all
  — must never key off a model path a crafted seed sidecar chose. In edit
  mode the seed image is **pixels only**: it is the F5-capped edit source and
  its embedded comfyless params (if any) are NOT extracted into the config —
  which also means plain foreign images (Gimp exports, photos, prior
  keyframes) are accepted as edit sources, matching the keyframe use case.
- **Edit source selection is loop-controller code, never planner output:**
  iteration 0's source is the operator's seed image; thereafter the source is
  `best`'s image (a file the loop itself wrote into `candidates/`), following
  the D2 unified lineage rule. The planner/judge never names, selects, or
  sees a path.
- **Ref plumbing uses the typed channel; daemon containment is resolved at
  loop entry** (Finding 2, HIGH — decision made): refine's edit sources (the
  operator seed, and `candidates/` files that `run_generation` moves OUT of
  the daemon's output tree) generally fall outside the daemon's
  `ref_image_roots`, so the ADR-035 slice-4 wire gate would refuse them.
  Resolution mirrors the ALREADY-LANDED ADR-035 CLI behavior (outside-roots
  refs skip delegation and run in-process): at **loop entry**, if a daemon
  socket exists for the device, refine preflights whether the run's output
  dir and the seed image fall inside the daemon's ref roots. If yes, the
  daemon path is used with `ref_images` riding the wire exactly as slice 4
  defined. If no, refine emits ONE loud notice naming the fix (`start the
  daemon with --ref-root <run dir>` / nest the output dir) and runs the
  WHOLE run in-process (row-1 typed authority via
  `gen.generate(ref_images=[{"path": <loop-owned>, "mode": ...}])`) —
  decided once, before any GPU work, never per-iteration. **The wire-trust
  workarounds are prohibited:** no trust-assertion wire field, no refine
  exemption inside the daemon's gate, no merging refine dirs into weight
  roots (ADR-035 decisions 6a/7: trust class is never a wire field). Refs
  NEVER round-trip through merged params/sidecars inside the loop; the
  slice-5 replay trust gate and seed-sidecar `ref_images` echo-then-drop
  are untouched.
- **Mode default:** `both` (VL + VAE conditioning), the qwen-edit default.
  Not planner-selectable in v1.
- **Judge sees two images in edit mode**, labeled by ROLE ONLY (Finding 4,
  MEDIUM): a fixed code-owned template labels them `SOURCE (currently
  accepted)` and `CANDIDATE` — no path, filename, or stem ever enters the
  judge user text or wire payload (`_assert_no_paths` cannot catch values;
  the template makes leakage structurally impossible). Both images pass the
  existing `downscale_for_judge` cap. Scene-preservation scoring is
  impossible from the candidate alone. (t2i mode stays single-image.)
- **Acceptance gating:** a candidate becomes the next edit source only if its
  composite improves on `best`. Rejected outputs stay in `candidates/` with
  verdict sidecars but are never promoted.

### D6 — Rubrics/recipes

- New code-default edit rubric + `judge_recipes/edit-generic.toml`
  (edit-instruction adherence + scene lock on the `prompt_adherence` axis;
  aesthetics unchanged). Recipe loading rules (bare names, fail-closed on
  explicit miss, code-owned output contract always appended) unchanged.
- The t2i `generic` rubric gains decisive-rewrite planning guidance
  (recipe-side, cheap to A/B before any code lands — the slice-A first
  experiment).

### D7 — Slice plan

- **Slice A** — trajectory core, t2i only: D1 + D2 + D3 + the D6 rubric
  guidance. `test_refine.py` extensions incl. negatives.
- **Slice B** — edit mode: D5 + D6 edit rubric. Depends on A.
- Both slices: code-reviewer + security-auditor (Fable) pre-commit, per the
  Red Zone bar. The keyframe orchestrator is a separate ADR when its slice
  starts.

## Security findings carried by this design

Design review: `docs/security/review-adr-037-design-2026-07-23.md`
(security-auditor, Fable — no model fallback; verdict: no CRITICAL, fold
Findings 1–8 before code; Findings 1–13 disposed as follows — 1↦D1, 2↦D5,
3↦D1, 4↦D5, 5↦D5, 6↦D2, 8↦D3, 9/10/13↦D1, 11↦D4, and 7/12 below).

- **F8-P (persistent planner-text echo; review Finding 1):** history
  re-enters past planner-authored prompts into all future judge context.
  Extension of ADR-027's F8; bounded by the D1 mitigations (untrusted
  labeling, 8 KiB planner-text budget, no critique replay) and F1 (the
  steered output can still only move prompt+LoRA).
- **F8-E (persistent visual injection via the accepted source; review
  Finding 7):** in edit mode the accepted source is re-presented to the
  judge every iteration until displaced, and displacement is governed by
  scores the (steerable) judge emits — text rendered into a candidate that
  successfully inflates its own scores entrenches itself as the source.
  Disposition: accepted for v1, bounded by the audit trail (`candidates/` +
  verdict sidecars), human review of `winners/`, and climb-from-best
  recovery insofar as scores are honest. Soft mitigation, slice B: edit
  rubric instructs that text rendered inside images is CONTENT to be scored,
  never instructions.
- **Forward constraint on slice C (review Finding 12):** ADR-027's F8
  disposition leans on a human looking at `winners/` before anything
  consumes it. The keyframe orchestrator exists to feed accepted outputs
  into `plan.json` → `video.py` — automation consuming exactly that
  artifact. **The slice-C ADR MUST re-disposition F8/F8-P/F8-E for automated
  consumption before wiring refine outputs into plans.**

## Alternatives rejected

- **Amend ADR-027 instead of a new ADR** — rejected by Grant: A+B are a new
  loop version, not a tweak; the decision record should say so.
- **Replay full past critiques into judge context** — rejected for echo risk
  (D1); scores + ops + excerpts carry the signal.
- **Planner-selectable edit source ("re-edit iteration 3's image")** —
  rejected: hands the LLM a path-adjacent authority for marginal gain;
  climb-from-best gives the loop the same recovery power in code.
- **Separate edit-loop module** — rejected: duplicates trajectory machinery;
  the family gate + two-image judging are small deltas on one loop.
- **New verdict axes for edit mode** — rejected: forking the verdict schema
  forks F7 parsing and the pass gate; rubric reinterpretation achieves the
  same scoring with a stable contract.

## Deferred / Out of scope

- Keyframe orchestrator (`comfyless/keyframe.py`) — own ADR at slice C.
- LLM planner for video plans (ADR-033 slice 6) — Red Zone, own spec.
- flux2/flux2klein edit-mode lift; planner temperature knob; history
  compression; MCP/OWUI exposure of refine.
- **Evolutionary/combinatorial optimization** (Grant, 2026-07-23): a later
  loop version could run a population of configs over many generations with
  selection + recombination instead of a single greedy walk. Recorded as a
  direction; where the optimization peak lies is unknown — a couple of
  >10-iteration until-score runs (D3) are the cheap first probe.

## Changelog

- 2026-07-24 (later) — **D5 amendment: judge anchor = ORIGINAL seed.** Live
  stress run (impossible target, 100-iteration tie chain) showed cumulative
  drift the judge structurally could not see: subject getting younger/
  blonder, jeans splotchy — yet 9.6 every iteration. Root cause: the judge's
  SOURCE image was `current_source` (the currently-accepted candidate), so
  preservation was only ever checked STEPWISE — each candidate vs its
  immediate parent — and tie-promotion advanced the anchor itself
  (boiling-frog ratchet; the D2-amendment MEDIUM materializing benignly).
  Change: the judge's comparison image is now the OPERATOR'S ORIGINAL
  `edit_source` for the whole run; generation lineage still builds
  forward on best's image (D2/D5 unchanged). Drift now scores as an
  unrequested change against a fixed reference → strict decline → revert
  to best; the deferred tie-streak cap stays in reserve pending a rerun of
  the stress test. Role label updated to "SOURCE (original, pre-edit)" in
  the code-owned constant, the fallback rubric, and edit-generic.toml
  (preservation step now names cumulative drift explicitly). Test pin:
  every judged iteration's source image is the 8×8 seed fixture, not a
  4×4 candidate — asserted in both the promotion and tie-chain scenarios.
  **Review fold (both Fable, no fallback, same day):** code-reviewer
  SHOULD — the anchor was pinned to a PATH (re-opened per iteration),
  which reintroduced silent anchor drift via mid-run file swap plus a
  fatal window on mid-run delete; FIXED by loading the anchor ONCE at
  loop entry (bytes pinned; the slice-B LOW-3 re-open rationale died with
  the constant anchor; fatal-on-absent moves to entry, memory bounded by
  SEED_IMAGE_MAX_PIXELS). NITs folded: fallback rubric got the
  cumulative-drift language + corrected CANDIDATE description (it had
  received only the label swap) and its "shipped verbatim" header claim
  corrected; refine_loop docstring now names the anchor/lineage split
  (and its stale "strict improvement" phrasing updated to the D2 >=
  rule); current_source comment disambiguated from edit_source.
  security-auditor: LOW×2, INFO×2, no blockers — Q3 verdict: every path
  by which a candidate advances now requires a verdict produced with the
  original anchor in context. Accepted residuals recorded: (a) F8-E
  surface quantitatively widened — adversarial text in the seed now gets
  N full-fidelity judge exposures per run instead of one; bounded by
  F1/F2/F6/F7 structural guards; the soft rubric line stops being
  sufficient the day refine is exposed to agent/remote callers (that
  trigger = harden before wiring); (b) the mitigation is behavioral —
  a constant-parity judge can still tie-promote drifted lineage to the
  cap (the D2 MEDIUM residual; tie-streak cap remains the reserve).
  Review: `docs/security/review-adr-037-d5-anchor-2026-07-24.md`.

- 2026-07-24 — **D2 amendment (tie-promotion + no-op seed resample)**, from
  the first edit-mode live smokes (Grant). Two coupled lineage changes:
  (1) **Ties promote the NEWER candidate** — promotion is now composite
  `>=` best, not strict `>`. Rationale: equal scores hide sub-score-
  resolution improvements worth building on; reverting to the older equal
  candidate discards them. A strict DECLINE still reverts the climb to
  best (unchanged). `no_improve`/patience still counts ties as
  non-improvement (strict-improvement semantics), so tie-promotion cannot
  defeat the early stop. In edit mode the accepted source advances to the
  tied candidate's image (image lineage follows config lineage,
  unchanged rule). History `improved` flag = strict improvement;
  `is_best` = promotion.
  (2) **No-op seed resample** — observed failure: judge scored 10/9 with
  zero unmet requirements → planner had nothing to aim a rewrite at →
  empty/absent overrides → next config identical to its lineage source →
  with the seed pinned (slice A), the loop regenerated the byte-identical
  image to the 100-iteration cap. Fix: after `apply_overrides`, if the
  derived config equals its lineage `source_cfg` (prompt + LoRA set/
  weights + base), bump the pinned seed by +1 with a loud log line so the
  next iteration explores a new sample instead of reprinting. Seed
  attribution semantics are preserved: iterations where the planner DID
  change something keep the pinned seed, so score deltas remain
  attributable to the change. Guarded to int seeds >= 0 (an unpinned -1
  is already random). Not a planner-authority change (D4 untouched);
  `apply_overrides`' deep-copied base makes the in-place bump alias-safe
  w.r.t. best's snapshot. Test pins updated: seed-pinning pin becomes the
  no-op-resample pin; "tie is not promoted" flips to "tie promotes the
  newer candidate."
  **Same-day D6 note:** `judge_recipes/edit-generic.toml` was rewritten in
  the same slice (DECOMPOSE-THEN-VERIFY: neutral DESCRIPTION pass →
  per-requirement VERIFICATION citing description lines → mechanically
  bounded score, emitted as a plain-text preamble before the strict JSON —
  `_extract_json_block` tolerates brace-free leading prose). Motivation:
  the judge scored 9/9 on a candidate failing 3 of 6 edit instructions
  (checklist-echo sycophancy); pre-flighted live 10/pass → 6/revise.
  Known residual: instruction text in context can still contaminate the
  description (a two-call blind-describe judge is backlogged).
  **Review fold (both Fable, no fallback, 2026-07-24):**
  code-reviewer's SHOULD (borderline-CRITICAL): the +1-per-no-op bump was
  NOT monotonic across decline cycles (a decline reverts to best's
  immutable snapshot seed, re-deriving the same bumped seed forever — the
  plateau surviving on the decline branch); fixed with a monotonic
  loop-level no-op counter (`seed = source_seed + Nth_noop`), pinned by a
  strictly-increasing-seeds decline-cycle test. Second SHOULD: edit-mode
  tie lineage (source advances on tie; history improved=False/is_best=True/
  accepted=True) now pinned. security-auditor MEDIUM **accepted as
  documented risk**: tie-promotion lowers the F8-E propagation bar from
  strict-win to parity, and with DEFAULT_PATIENCE=0 the only bound on a
  constant-parity tie chain is --max-iterations — the winner shifts from
  earliest-peak to last-tied (most-drifted). Deferred mitigation option:
  a tie-streak cap (revert to best after N consecutive non-improving
  promotions). LOW: the rubric preamble + JSON can crowd
  DEFAULT_JUDGE_MAX_TOKENS=1024 — truncation fails closed but burns
  iterations; raise backend-cfg `max_tokens` if truncation appears (noted
  in the recipe header). INFO folds: bool excluded from the seed guard;
  preamble-parse positive + stray-brace negative tests added; contract
  "STRICT JSON and nothing else" vs preamble tension left standing (parse-
  safe; verify preamble emission in live smokes). Reviews:
  `docs/security/review-adr-037-d2-amendment-2026-07-24.md`.

- 2026-07-23 — **Slice B (edit-mode refinement) implemented.** D5 entry
  contract, family gate (qwen-edit), loop-owned edit-source lineage,
  two-image role-labeled judging, edit rubric (D6). **D5's "loop-entry
  preflight" sentence is superseded:** the client cannot know the daemon's
  ref roots (ADR-035 4b — a client-side check is structurally impossible),
  so Finding 2 is realized as **first-refusal latching** keyed on wire
  `error_type == "RefPathError"` (never a message substring): ONE loud
  notice naming the `--ref-root` fix, then in-process for the rest of the
  run — possibly latched MID-RUN when the seed lies inside daemon roots but
  the run's output dir does not (iteration 0 may complete on the daemon;
  daemon→in-process param parity is covered by slice-3 LOW-7). The three
  prohibited workarounds (wire trust field, daemon exemption, root merging)
  remain absent. Implementation reviews (both Fable, no fallback):
  code-reviewer APPROVED, security-auditor LOW-only — all folded
  (`ref_drop_strict=True` forced on ref-bearing wire requests +
  `edit_warnings` surfaced; repr on daemon error echoes; judge-source
  re-open through the capped loader; wire-keying + pixels-only sentinel
  tests; presence-based entry semantics + empty-instruction refusal;
  dims-from-source note). `docs/security/review-adr-037-sliceB-`
  `implementation-2026-07-23.md`. test_refine 313→355; battery 29/29;
  pyright at baseline.

- 2026-07-23 — **Slice A (trajectory core, t2i) implemented.** D1 history
  layer, D2 snapshot/lineage, D3 until-score + judge-error abort, D6 rubric
  guidance. Implementation reviews (both Fable, no fallback): code-reviewer
  APPROVED (SHOULD-1 → `LoopOutcome.aborted` + exit 3; SHOULD-2 →
  `_resolve_max_iterations` seam; NITs folded incl. `apply_overrides`
  deepcopy); security-auditor LOW-only, folded —
  `docs/security/review-adr-037-sliceA-implementation-2026-07-23.md`.
  D1 label aligned to the implemented `"planner-proposed (untrusted)"`
  (NIT-1). INFO items accepted: `current_prompt` stays outside the F8-P
  budget (bounded by OVERRIDE_PROMPT_MAX_CHARS); seed-derived target prompts
  label "operator" per the F4 trust decision; `--pass-threshold` unvalidated
  (operator footgun, warn-don't-block); `lora_ops_applied` means "resolved
  and submitted." test_refine 206→313; battery 29/29; pyright at baseline.

- 2026-07-23 — Proposed. Security-auditor (Fable, no model fallback) design
  review completed same day: no CRITICAL; Findings 1–8 folded textually into
  D1–D5 per remediations (provenance correction + F8-P, daemon preflight
  decision, construction-test rewording, role-only labels, edit entry
  contract, snapshot-by-value + unified lineage, judge-error abort);
  Findings 9–13 recorded as binding slice constraints; F8-E + slice-C
  forward constraint added. Review: review-adr-037-design-2026-07-23.md.
  Awaiting Grant's acceptance.
