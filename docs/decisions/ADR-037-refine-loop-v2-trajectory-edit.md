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

- 2026-07-28 — **D5 amendment: the seed image is PINNED BY VALUE into the run
  directory, exactly as ADR-038 D5 pins `--ref-image`.** Proposed here; the
  code is its own slice.

  **The core D5 decision is unchanged** — the seed is still the pixels-only
  edit source, plain foreign images (Gimp exports, photos, prior keyframes) are
  still accepted, iteration 0 still edits the operator's seed and later
  iterations still edit `best`'s image. What changes is WHICH BYTES those names
  refer to after entry, so this is an amendment, not a new decision.

  **Why: the seed is consumed on two channels, and they can disagree.** The
  D5 anchor amendment (2026-07-24) pins the judge's comparison anchor by VALUE
  at entry — `source_img = load_seed_image_capped(edit_source)`, loaded once.
  But `current_source` starts as that same PATH and is re-opened by whoever
  generates (the daemon, every pre-promotion iteration). Swap the file mid-run
  and generation conditions on the new bytes while the judge scores identity
  against the old ones — "scores describe the generation's inputs" silently
  stops holding. That is verbatim the TOCTOU `pin_static_refs` already closes
  for static refs, whose own docstring names this two-channel argument as the
  reason to pin ALL refs rather than only judge-marked ones. The seed was the
  one reference left unpinned, and the argument does not distinguish it. It is
  also independent of any daemon: the re-read happens on the cold in-process
  path too.

  **The second thing it closes, which is why it surfaced now.** ADR-040 slice
  2b left `--seed-image` outside the daemon's roots WARNING rather than
  refusing, and slice 3 (D3a) made that a visible divergence from the sibling
  `generate --ref-image` surface, which exits 2. Both slice-2b reviewers said
  the real answer was to pin. Pinning dissolves the question instead of
  answering it: the daemon only ever sees the loop-owned copy inside the run
  dir — which is inside a reference root by construction under ADR-040 D1 — so
  an out-of-roots seed is no longer a latch, no longer an OOM risk, and no
  longer needs a warning or a `--ref-root` grant over the operator's photo
  directory (the ADR-035 Finding 6 breadth exposure refusal would have forced).
  The warn-vs-refuse divergence stops being a choice we defend and becomes
  unrepresentable.

  **Decision:**
  - At loop entry, in edit mode, the seed is copied VERBATIM (never re-encoded,
    per ADR-038 D5's own re-encode finding: a re-encoded camera JPEG can inflate
    past the cap and kill iteration 0 on the loop's own artifact) into a
    loop-owned directory under the run dir. `edit_source` and the initial
    `current_source` both name the COPY; after entry the original path is never
    read again.
  - **Every read of the original goes through the capped, regular-file-guarded
    reader, and the pinned bytes are proven to be the validated bytes.** The
    first draft said "read exactly once" and both reviewers found the same HIGH:
    `load_ref_image_capped` validates one set of bytes and `shutil.copyfile`
    then re-reads the path, so what lands in `source/` had passed nothing.
    `copyfile` applies no byte cap and rejects FIFOs but not devices, so the
    write was bounded by nothing (a symlink to `/dev/zero` fills the disk), and
    a file merely still being written — an editor save, a sync client, a camera
    dump, exactly the actors motivating this amendment — would pin an oversized
    copy that the daemon then refuses at iteration 0, on the loop's OWN
    artifact. The check-then-use window ADR-035 6d was designed not to have,
    reintroduced by the fix meant to remove one. So: re-read through
    `_read_ref_bytes_capped`, refuse unless the SHA-256 matches what was
    validated, and write with `O_EXCL` at `0600` — `copyfile`'s
    `open(dst,'wb')` follows a symlink and truncates, so a planted
    `source/seed.png` on a group-writable explicit `--output-dir` could
    redirect the write.
  - **The pin refuses when the seed lives inside the directory it replaces.**
    `--output-dir /runs/r1 --seed-image /runs/r1/source/seed.png` — re-running
    from a previous run's pinned seed — would otherwise validate, `rmtree` the
    operator's file, and then fail to copy it: run refused AND seed destroyed.
  - **`refine_loop` refuses an unpinned `edit_source`.** The invariant is in the
    signature, not a docstring: a future caller (the video orchestrator and the
    LLM planner are the named candidates) passing a path alone would otherwise
    silently reinstate both closed defects — an operator path handed to the
    daemon, and an anchor validated by a weaker loader than generation uses.
  - The copy lands in `<run dir>/source/`, NOT `refs/`. `pin_static_refs` opens
    with an unconditional `rmtree(refs/)`, so sharing that directory would make
    correctness depend on call ordering between two pinning steps — a footgun
    with no upside. Separate directory, no ordering coupling.
  - The loader becomes `load_ref_image_capped` (ADR-035 6c), matching
    `pin_static_refs`. This is NOT a narrowing of D5's "plain foreign images"
    contract in practice: the seed is fed to generation AS a `ref_images` entry,
    so it already had to pass that same loader at iteration 0 — the change moves
    the failure from mid-run to entry, which is the direction the entry-refusal
    discipline requires. Byte and pixel caps are identical either way (64 MB /
    64 MP); what is gained is the format allowlist (no `Image.open` dispatch
    across PIL's plugin zoo), the regular-file guard, and a SHA-256 for the log.
  - The decoded image from that single read serves as the judge anchor — one
    read, not two.
  - ADR-040's `--seed-image` outside-roots warning is REMOVED, because the
    condition it warned about can no longer occur.

  **Accepted, and named rather than left implicit:** in the derived case a
  verbatim copy of what is often a private photograph is now written INTO the
  daemon's output root — the tree the ADR-040 review notes is readable by any
  same-UID wire client, which `0700` does not defend against. The seed joins
  `refs/` in that accepted residual. The copies persist after the run with no
  retention policy (ADR-040 Deferred), so an edit run now leaves up to 8 refs
  plus a seed on disk; a refusal after pinning says so and names the directory.

  **Not changed:** seed MODE stays `both`; the planner still never names or
  sees a path; `build_config_from_seed` (non-edit *seed mode*, where the image
  is a params source and never a ref) keeps `load_seed_image_capped` and is out
  of scope; the D5 family gate and the operator-typed `--model` requirement are
  untouched.
- 2026-07-25 (parity slice 2) — **Shared wire-warning surfacer.** From the
  refine↔generate parity audit (matrix: vault
  `Refine_Generate_Parity_Audit_2026-07-25.md`, Grant's call to run the
  audit before the UX slices). refine surfaced ONLY `edit_warnings` from
  daemon metadata while generate surfaced nag/schedule/edit/lora — so a
  planner-added LoRA that silently failed to apply was invisible to the
  operator, the loop, and the judge, and the score moved for
  unattributable reasons. Acutely relevant since the planner started
  actually proposing LoRAs (first live proposal same day, iteration 20).
  New `generate.surface_wire_warnings(metadata, emit, *,
  include_lora=True) -> int` over a `_WIRE_WARNING_CHANNELS` table;
  generate's `_delegate_to_server` replaces three inline loops
  (`include_lora=False` — `_report_lora_outcome` renders its own banner,
  emission byte-identical); refine's `run_generation` calls it on BOTH
  the daemon and cold branches. **Scope: the OPERATOR half only** — no
  caller consumes the returned count, so loop/judge accounting is
  unchanged and the planner can still re-propose a LoRA its own prior
  iteration failed to apply (TECH_DEBT 2026-07-25, own slice: it changes
  decision-making on a Red Zone file and interacts with `--pin-lora` and
  the v3 gate). **Review folds (both Fable, no fallback):**
  security-auditor Q1 PASS — traced every consumer and confirmed the
  path-bearing `lora_warnings` strings reach the operator log ONLY
  (`log = print`); the judge payload reads none of `GenOutcome.metadata`
  and `_assert_no_paths` coverage is unaffected. Its two LOWs folded at
  the new choke point: control-character stripping (a divergent
  same-UID daemon could embed ANSI/OSC in a warning line) and a
  per-channel cap of 20 items × 500 chars with an explicit
  "N more suppressed" line (attention-DoS: ~1 MiB of warning text per
  response × 100 iterations would bury the score/PASS lines). Its INFO
  taken: the four wire-warning keys joined `_FORBIDDEN_CONTEXT_KEYS`, so
  a future slice that accidentally passed daemon metadata into a judge
  payload trips the structural backstop instead of relying on
  by-construction discipline. code-reviewer APPROVED after catching a
  self-inflicted weakening: my replacement `test_nag.py` N1 pin matched
  the surfacer's DEF line, so deleting the CLI call site would have left
  it green — re-pinned on the stderr emit literal. Cold-path duplicate
  emission (stderr at origin + loop log) judged acceptable: different
  sinks, the loop log is the operator's record.

- 2026-07-25 — **Keyword LoRA offers + plateau-reword rubric.** Live
  finding: the planner NEVER received a LoRA offer in any refine run to
  date — `search_loras` phrase-quoted the ENTIRE target prompt as one FTS
  term (0 rows on any real prompt; verified against the live DB where
  single keywords surface exactly the applicable entries, e.g.
  qwen-studio-realism). Fix: `_offer_keywords` tokenizes the prompt
  (stopword-stripped content words, capped 8), per-keyword FTS,
  round-robin rank merge, dedupe; family filter is SOFT — entries tagged
  a DIFFERENT family are dropped, NULL-family entries stay proposable
  (catalog tagging is partial, 498/789 untagged; wrong proposals fail
  loudly at load per ADR-015) — with qwen-edit/qwen-image as one compat
  group. `offer_family` derives from the operator-typed model — or, in t2i
  seed mode, the seed-sidecar model (the F4 user-authority channel, loudly
  echoed) — never from LLM output (audit-verified: no planner-writable
  channel reaches `base["model"]`). F3 unchanged:
  offers still project through `_safe_lora_view`. Companion D6 rubric
  change (both recipes): NEVER return empty overrides while short of the
  gate — reword/reorder the instruction (same requirements, different
  emphasis) and/or use the offered LoRAs by catalog name. Interaction
  note: this makes planner no-ops rare, so seed exploration shifts to the
  stagnation escape (intended — config exploration beats reprints; the
  research memo's seed-batch design will supersede this territory).
  Optimization research memo (pairwise duels, sideways caps, anchor
  duels) saved to the vault: `Refine_Optimization_Research_2026-07-25.md`
  — algorithm changes await Grant's read (his direction 2026-07-25).
  **Same-day addendum — critique-driven offers.** Live iteration-25
  evidence: prompt-derived keywords surface SUBJECT LoRAs (anatomy hits
  on "body") while the quality LoRAs the run needed (realism/portrait
  enhancers) match FLAW words that live only in the judge's critique.
  `search_loras` now takes `critique_text` — the PREVIOUS iteration's
  validated critique (F7 string values only), PREPENDED so its keywords
  own the front of a raised 10-term cap; offers chase what the judge
  just complained about. Read-only FTS keyword material, quoted per
  term at the DB layer — no authority channel. NO topical filtering by
  design (Grant: body/NSFW LoRAs often genuinely improve skin texture
  and realism; the judge decides relevance). Enhance-based prompt
  rewording evaluated and NOT wired: the judge already produces full
  in-band rewrites; a second LLM hop is the fallback only if in-band
  rewording proves too timid (deferred). F8-P budget left at 8192 —
  elision keeps score/flag stubs (anti-cycling intact); the raise
  trigger is the planner re-proposing a specific pre-horizon failed
  edit.

- 2026-07-24 (night) — **D2 amendment addendum: stagnation seed escape
  (`--explore-after N`).** Live gap in the no-op resample: it fires only
  when the planner proposes NOTHING. Observed run: iter 1 hit 8.6 (best);
  every later iteration declined, reverted to best's config, and the
  planner kept rewriting the prompt against seed-tied background
  artifacts prompting cannot remove — config differed every time, so the
  seed stayed pinned to best's and 12 straight iterations reprinted the
  same flaw. Fix: when `no_improve >= --explore-after` (default 2; 0
  disables), the derived config's seed is resampled via the SAME
  monotonic counter as the no-op escape (uniqueness preserved), on every
  further non-improving iteration, resetting on strict improvement. The
  no-op branch takes precedence (no double-bump). Counter renamed
  `noop_resamples` → `seed_resamples` (two triggers, one lattice).
  Interaction note: a positive `--patience` <= `--explore-after` stops
  the run before the escape fires (help text says so). Not a planner
  authority change — the planner still cannot touch base params; the
  escape is code-triggered off the loop's own improvement accounting.
  **Review fold (both Fable, no fallback, same day):** security-auditor
  all-INFO + one LOW (mixed no-op/stagnation counter-uniqueness pin —
  added: strictly-increasing-seeds test across interleaved triggers);
  Q1 verdict: no authority expansion — the judge could already trigger
  resamples at will via empty overrides, the seed value is code-computed
  and appears in no judge-visible surface, and resamples are bounded by
  the iteration cap. Acceptance-record accuracy note (auditor INFO): the
  escape extends the accepted tie-chain drift residual to the SEED
  dimension — a constant-parity judge now compounds prompt drift, source
  advancement (edit mode), and noise-sample drift simultaneously, under
  the same --max-iterations bound and the same acceptance rationale.
  code-reviewer (Fable): mechanism verified sound (monotonicity proof
  across mixed/tie sequences); folds — stale attribution comments
  conditionalized, tie-chain skip-value + patience-stops-first +
  mixed-trigger uniqueness pins added, docstring names the escape.
  **Accepted attribution deferral:** a stagnation-resampled iteration
  changes prompt AND seed but the planner's D1 history carries no
  `seed_resampled` flag — the planner may mis-attribute the next score
  delta to its prompt edit. Deferred (TECH_DEBT 2026-07-24): a history
  field touches the F8-P surface and warrants its own pass; escapes only
  fire on already-stagnant runs.
  Review: `docs/security/review-adr-037-stagnation-escape-2026-07-24.md`.

- 2026-07-24 (evening) — **D3 amendment: `--until-score [SCORE]` float
  composite gate.** From the impossible-target stress test: axis scores are
  integers, so `--pass-threshold` is an int by design and a "very good but
  not perfect" target (Grant wanted 9.8, then 9.6) was inexpressible — a
  threshold-10 until-score run rides to the cap by rubric construction
  (aesthetics 10 = "exceptional craft"). Semantics: bare `--until-score`
  is UNCHANGED (both axes >= --pass-threshold); `--until-score SCORE`
  (float, 1-10, finite) REPLACES the gate with weighted COMPOSITE >=
  SCORE (epsilon-tolerant compare — composites are float sums, 0.6*10+
  0.4*9 may sit a ULP under 9.6). --pass-threshold is ignored in valued
  mode (help text says so). Cap rules unchanged (valued mode raises the
  default cap to the sanity cap exactly like bare mode). Warn-don't-block
  lattice note: integer axes make the reachable composite set a lattice —
  when the target sits in a gap (9.8 at weights .6/.4 → nearest reachable
  10.0), a loud note names the composite the run effectively requires.
  The judge's advisory "pass" string stays non-authoritative (F8,
  unchanged).
  **Review fold (both Fable, no fallback, same day):** code-reviewer
  SHOULD + security-auditor LOW (independent, same finding): an
  UNREACHABLE target (non-default weights capping the max composite below
  it, e.g. weights .5/.3 with target 9) silently skipped the lattice note
  and rode to the cap — FIXED: `_nearest_reachable_composite` returns
  None on unreachable and main() emits a loud UNREACHABLE warning naming
  the max possible composite. Security LOW: composite weights are
  unvalidated CLI floats that now control TERMINATION (NaN weights = every
  compare False = silent cap ride) — FIXED: finite-check on both `--w-*`
  flags at entry, exit 2 (range stays operator-domain per
  warn-don't-block). NITs/INFO folded: cap-raise pinned against
  `bool("9.6")` coercion tidy-ups; verdict-record key-set pin proves the
  target never persists (operator-side only — auditor verified it reaches
  neither judge context nor sidecars). Auditor Q1 verdict: no expansion
  of judge termination authority — the composite is a deterministic
  monotone map of the same two F6-coerced ints. Vault Comfyless_Manual
  updated for the new flag semantics. Review:
  `docs/security/review-adr-037-d3-until-score-2026-07-24.md`.

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
