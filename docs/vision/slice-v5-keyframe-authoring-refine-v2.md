# Vision — Keyframe authoring + refine loop v2 (video program slice 5)

Status: APPROVED with modifications (Grant, 2026-07-23) — see Resolved
decisions at bottom
Date: 2026-07-23
Serves: ADR-033 decision 7 (keyframe authoring), Backlog "Refine loop v2" (Queued)
Supersedes-in-part: the standalone "Refine loop v2" Queued entry (folded in here)

> **Posture:** Boundary: domain rules (refine loop internals) + entrypoint (new
> keyframe orchestrator). Risk factors: **near security truth** (LLM output
> influencing generation params — `comfyless/refine.py` is a §12 Red Zone
> path) + broad impact (loop restructure touches every refine consumer).

**Risk level: High (L3).** Any `refine.py` change requires ADR-027 amendment +
`security-auditor` (Fable) BEFORE code, per the repo Review bar.

---

## Intent

One trajectory-aware refine loop that serves both consumers: (a) plain t2i
refinement that actually hill-climbs (restart from best after regression,
planner sees history), and (b) edit-model refinement where each iteration's
input is the previous accepted image passed as a typed reference — and on top
of it, a keyframe-authoring orchestrator that produces plan-ready keyframe
N→N+1 evolutions via same-seed prompt-variation (cheap default) or judge-gated
single-op edit chains.

## Why one program, not two

The edit loop's hard requirement IS the v2 fix: an edit chain must know which
image is currently "accepted" (that's `best`), must not keep editing a
regressed output (that's climb-from-best), and its planner must know what was
already tried (that's the history block). Building edit-mode on the v1
stateless loop would mean building the trajectory machinery twice.

## Invariants (must always be true)

1. **Planner authority stays the closed two-key allowlist** (prompt + LoRA
   ops, F1). Edit mode adds ZERO planner-mutable keys. In edit mode the
   "prompt" carries the edit instruction; that is the only semantic change to
   planner output.
2. **The planner never names, selects, or sees reference-image paths.**
   Ref selection is loop-controller code: the current `best` (or the operator
   seed on iteration 0) is the edit source. Judge/planner context stays inside
   the F3 path-free projection; the new history block passes the same
   `_assert_no_paths` structural gate.
3. **Refs enter generation ONLY through the typed channel** — the in-process
   `generate(ref_images=[{"path","mode"}])` kwarg (or `args.ref_image` on the
   daemon path), never via merged params/sidecar. Seed-sidecar `ref_images`
   remain echo-then-DROPPED exactly as ADR-035 slice 5 left them; the loop's
   refs are always files the loop itself just wrote (or the F5-capped operator
   seed image).
4. **Climb-from-best:** after a composite regression, the next iteration's
   config derives from `best`'s config (and in edit mode, `best`'s image),
   not the regressed one. A regressed edit output is never promoted to edit
   source.
5. **Model/family swap remains forbidden to the planner.** Edit mode is
   selected by the loop from the operator's `--model` family via the existing
   `_REF_FAMILY_KINDS` routing; a family with no ref support fails loudly
   under refine (no silent t2i fallback mid-chain).
6. **History is bounded and path-free:** fixed-size projection (scores,
   prompt diffs, LoRA ops, improved/regressed flags per iteration), capped in
   bytes, no filesystem strings.
7. **Keyframe outputs are plan-ready:** files land in a deterministic
   directory layout the operator can reference from `plan.json`
   (`keyframe_start`/`keyframe_end` are plain paths — `video.py` is NOT
   modified by this slice).

## Failure semantics

Fail-closed throughout: malformed judge output consumes an iteration (existing
F7); generation failure aborts the run (existing); an edit iteration whose
output is rejected by the judge leaves the accepted ref unchanged (retry from
`best`); unsupported family under edit mode is a hard error at loop entry,
before any GPU work. Partial success cannot corrupt state: `candidates/` keeps
every attempt with verdict sidecars; `winners/`/keyframe outputs only ever
receive judge-accepted (or cap-expired best) images.

## Out of scope

- **LLM planner (video slice 6)** — scene → plan.json authoring. This slice
  only sets the trajectory/edit substrate it will reuse. Red Zone on its own
  spec when it comes.
- `video.py` changes, video-segment judging, MCP/OWUI exposure of refine or
  keyframe tools, HTTP anything.
- Flux2/Klein edit-quality tuning (routing already exists in generate; refine
  edit mode keys off `_REF_FAMILY_KINDS`, but qwen-edit is the validated
  target; others ride along untuned or are gated out — decided in the ADR).
- Planner temperature knob — noted in the ADR as a cheap follow-up, not built
  here unless slice A makes it trivial.

## Negative cases required

- Planner verdict containing `overrides.ref_images` (or any path-shaped
  string in any override field) → dropped/rejected with notice; never reaches
  `generate()`.
- History block construction: a config containing paths yields history
  records containing none (CONSTRUCTION test — `_assert_no_paths` gates keys
  only and cannot scan values; reworded per design review Finding 3).
- Composite regression OR tie on iteration i → iteration i+1's config
  provably derives from `best`'s by-value snapshot, and that snapshot is
  immutable against later in-place `cfg.base` mutation (unit test, no GPU).
- Judge HTTP error whose response body contains a sentinel string → sentinel
  never appears in the serialized history block (review Finding 9).
- Unresolvable planner-proposed LoRA name (500-char steering text) → absent
  from history; only post-resolution ops recorded (review Finding 10).
- Edit-mode run with `--model` defaulted from the seed sidecar → refused at
  entry; plain foreign image (no comfyless chunk) accepted as edit source
  (review Finding 5).
- Edit-mode judge payload contains no path, filename, or stem — images
  labeled by role only via the code-owned template (review Finding 4).
- 3 consecutive judge errors → run aborts loudly before the 4th generation
  (review Finding 8).
- Edit-mode run with a non-ref family (e.g. `zimage`) → loud error at entry,
  zero generations.
- Judge-rejected edit output → next iteration's `ref_images` still points at
  the prior accepted image (unit test).
- Seed sidecar carrying `ref_images` → still echoed + dropped (regression
  test against ADR-035 slice 5 behavior).

## Proof hooks

- `./.venv/bin/python3 test_refine.py` (extended: trajectory core, edit mode,
  negatives above) + full `just tests` battery.
- Live smoke (Grant): (a) t2i refine run where iteration k regresses —
  observe restart-from-best in the log + verdict trail; (b) qwen-edit keyframe
  evolution: seed keyframe → single-op instruction → judge-gated accept →
  next ref; (c) same-seed prompt-variation keyframe pair fed to a 2-segment
  `plan.json` → `video.py` renders with continuity.

## Red Zone ownership

- Judge/planner context expansion (history block) and override-authority
  decisions: **Grant owns**; AI drafts, security-auditor reviews pre-code.
- Ref-source selection logic in the loop controller: **Grant owns** the
  design sign-off (it is what keeps LLM output away from file paths).

## Proposed slice plan (each its own commit + review cycle)

- **Slice A — trajectory core (t2i only):** climb-from-best + bounded
  path-free history block in judge context + decisive-rewrite rubric guidance
  (recipe-side half first — it's editable today and cheap to A/B) + the
  until-score stop mode (run to pass threshold with a far-out sanity cap,
  ~50–100, alongside the existing `--max-iterations`). ADR-037 +
  security-auditor before code.
- **Slice B — edit-mode refinement:** family-gated edit loop; `best` image →
  typed `ref_images` kwarg; edit-instruction prompt semantics; new
  `judge_recipes/edit-*.toml` rubric (edit fidelity + scene lock axes).
- **Slice C — keyframe orchestrator:** thin `comfyless/keyframe.py` (or
  `python -m comfyless.keyframe`) over the refine loop + enhancer recipes
  (`preserve-subject`/`vary-setting`): produce keyframe N+1 from N via
  variation or edit chain; deterministic output layout; user-supplied paths
  pass through untouched. No new LLM authority beyond slices A/B.

## Resolved decisions (Grant, 2026-07-23)

1. **ADR shape:** slices A+B get their OWN new ADR (ADR-037) — they are "a
   whole new version of the refinement loop, not a tweak." ADR-027 marked
   superseded-in-part when ADR-037 is accepted. The orchestrator (slice C) is
   also its own thing — its ADR is written when C starts, informed by A/B.
2. **Edit families:** start with qwen-edit. Grant's note: flux2klein output
   may actually be SIMPLER for the judge to work with — record as the likely
   first family lift after v1, not a maybe.
3. **History depth:** whole run, assumed to fit; if limits bite, address later
   (lift byte cap / cap run length).
4. **Stop modes (added scope):** iterations stay a parameter, AND add an
   until-score mode — keep going until the pass threshold is met, bounded
   only by a far-out sanity cap (~50–100). Where the real optimization peak
   is is unknown; runs >10 are worth a couple of experiments.
5. **Deferred direction (recorded, not built):** evolutionary/combinatorial
   optimization over many generations (population of configs, selection +
   recombination) as a possible later version of the loop.
