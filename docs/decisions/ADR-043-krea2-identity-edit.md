# ADR-043 — Krea-2 identity edit as a pipeline-layer subclass

**Status:** accepted

**Context:**

`conradlocke/krea2-identity-edit` is an instruction-based, identity-preserving
image editor for Krea 2, shipped as a LoRA. The driving use case is generating
images of a specific person with different haircuts, good enough to hand to a
stylist — a task the qwen-edit backend demonstrably cannot do (backlog
three-variant result 2026-07-25: `prompt_adherence` pinned at 3 across every
iteration of all three variants).

The full design, the live validation that produced it, and eleven numbered
decisions (D1–D11) live in `docs/vision/epic-krea2-identity-edit.md`. This ADR
records the architectural choice and the constraints that must not be
re-litigated; it does not duplicate the epic.

The design was validated end-to-end on 2026-07-31 **before** any code here, by
running the reference port (`huan-yin/Krea2-Identity-Edit-Diffusers`, Apache-2.0,
GitHub) offline against our own model tree from a scratch harness outside the
repo. Artifacts: `krea-identity-eval/`. That validation resolved the epic's
largest unknown, falsified one of its risks, and corrected two of its decisions
— which is the whole reason it ran first.

**Decision:**

Build a thin `pipelines/krea2_identity_edit.py` on diffusers 0.39.0's **native**
`Krea2Transformer2DModel`, in the established `pipelines/nag_krea2.py` mold: an
unbound `__call__` invoked on the stock cached pipeline instance via
`identity_edit_pipe_call(pipe, **kwargs)`, with any attention-processor swap
installed per call and restored in a `finally`. Entry is the existing ADR-035
`--ref-image` surface — no new dispatch mode, no new family string, no new
module under `comfyless/` (epic D1, D2, D7).

Three constraints are load-bearing and are the reason this ADR exists:

1. **The VL image processor is constructed at runtime from the LIVE text
   encoder's `vision_config`** — `pipe.text_encoder.config.vision_config`, never
   `<model_path>/text_encoder/config.json` and never hard-coded values. Krea-2
   ships no `preprocessor_config.json`, and `--te1` lets a user substitute the
   text encoder (Grant runs the abliterated Qwen3-VL-4B this way). Sourcing from
   the checkpoint directory would silently describe an encoder that is not
   loaded. Both encoders in use today report identical geometry, so a
   checkpoint-directory implementation would pass every obvious test by
   coincidence and break on the first override that differs. (Epic D10.)

2. **Reference order is semantic, fixed, and never reordered.** `--ref-image` #1
   → RoPE frame 1 = scene/context; #2 → frame 2 = identity. Verified by a
   swapped-order positive control, which produced the other subject's face in
   the first image's setting. A third reference is a hard error naming the
   two-source maximum, never a silent drop — a dropped reference reads to the
   user as a model failure. (Epic D8.)

3. **`ref_boost` cannot separate identity from a spatially-adjacent edit.** It
   adds `log(ref_boost)` to the attention logits where target queries attend to
   source keys, so raising it instructs the model to copy harder from the region
   the edit is trying to change. Measured: hair edits work at ~1.25 and are
   suppressed outright at the card's default of 4.0. Empirically-earned values
   therefore ride the **catalog suggestion language** for iterate loops rather
   than a single schema default (epic D11), and **no hand-painted mask UI is to
   be built** (Grant, explicit).

**Alternatives Rejected:**

- **Vendor the reference port.** It re-vendors a transformer already present in
  our pinned diffusers, and requires `trust_remote_code`, which this codebase
  avoids everywhere except one hash-pinned backend. The port is read, run,
  diffed against, and credited — not copied. *(A third original reason, that it
  pins `transformers==5.12.1` against our 5.5.3, turned out to be false: its own
  import guard declares the real floor as `>=4.57`, and the entire live
  validation ran on our existing `.venv` with zero installs. The other two
  reasons stand alone.)*
- **A dedicated `comfyless.krea_identity` dispatch mode.** The Stable Cascade
  shape (ADR-010) is what that costs: 2069 lines plus a permanently
  hand-maintained denylist of unsupported flags. The forcing constraint that
  justified it there is absent here — ADR-035 decision 2 already binds: "There
  is no edit mode. Edit is generation with reference conditioning."
- **A `--vl-processor` path parameter, or shipping a `preprocessor_config.json`
  as package data.** Both were proposed by the epic before the vision-config
  finding. Both are strictly worse than constructing from the live encoder: they
  add a second place for the geometry to drift out of sync, and the path param
  additionally invites a network fetch.
- **Deferring two-source to a later slice.** Its cost was misjudged — the
  position-id builder is already `n_src`-generic and the reference-KV precompute
  the epic feared is unrelated dead code. The one live test that looked bad ran
  outside the mode's stated boundary (frame 1 held a portrait where a scene
  belongs), so it is not evidence against the capability.

**Deferred / Out of Scope:**

- Daemon and MCP carriage (epic Part C) — Red Zone, requires `security-auditor`,
  and the two scalars must stay **out of** `_request_cache_key` (they select
  output, not weights — the NAG precedent).
- Three or more source frames. The layout generalises to `n_src`; nothing
  trained or tested supports it.
- Automating the single-image → two-input chaining (use the first pass's output
  as the frame-2 person so each pass does one job). Works manually today with no
  new mechanism; a `refine` integration is its own slice.
- The per-source-token `ref_boost` bias that would properly resolve constraint 3.
  Recorded in epic D11 as ~10 lines in a bias builder we own; blocked only on
  where a mask would come from, and a hand-painted one is rejected outright.
- Any `FAMILY_DEFAULTS` change, and reconciliation with ADR-038's
  identity-judging rubric.

**Changelog:**

- 2026-07-31 — **`--identity` opt-in added; the mode no longer claims
  `--ref-image` on krea implicitly.** Grant raised this on review of Part B:
  the trigger was `--ref-image` + krea family with no flag, and "Krea can
  almost certainly take a non-identity-focused reference image conditioning
  too." Two facts bounded the concern — diffusers 0.39.0 exports only
  `Krea2Pipeline` (whose `__call__` has **no `image` parameter**) and
  `Krea2Transformer2DModel`, and before Part B `--ref-image` on krea hit the
  drop path, so nothing that worked stopped working. But the objection is
  right forward-looking: the identity edit is one reading of "reference image
  on a krea checkpoint," its behaviour lives ENTIRELY in a LoRA nothing can
  detect (D5), and a surface claimed implicitly cannot be shared later without
  a retroactive second dispatch axis.

  **Not gated on LoRA detection**, though that is the true precondition: a
  LoRA's identity is a user-chosen catalog name with no structural marker, so a
  filename heuristic would silently mis-route a renamed file in both
  directions. That argument justifies not *blocking* (D5's warn-don't-block
  stands) — it does not justify refusing an explicit opt-in.

  Shape, per Grant's decisions: `--identity` is **entry mode, not a generation
  parameter** — no `COMFYLESS_SCHEMA` key, no sidecar record, no `--params`
  replay ("a sidecar consumer doesn't care that the image was generated with
  --identity, it's going to do something going forward with that image"). It
  sits beside `ref_drop_strict` / `ref_dims_explicit`, which are call
  parameters for the same reason. On a family with no identity edit, or with no
  `--ref-image`, it **warns and proceeds** rather than failing.

  **Consequence, accepted deliberately:** having no schema key means it cannot
  ride the daemon wire, and `server.py` would otherwise enter the identity path
  never having seen the opt-in. So an `--identity` run is forced in-process
  until Part C carries it (server.py is Red Zone). Practical cost is near zero
  — the measured-best `ref_boost` is 1.25, already non-default, so those runs
  were staying in-process on the tuning branch anyway.

  Without the flag, krea references take the ordinary drop path: hard
  `ValueError` under strict (machine/scripted), loud warn-and-drop when
  lenient. Both messages NAME `--identity` rather than claiming krea is
  unsupported, which would be a lie. A `--params` replay of an identity
  sidecar therefore does not silently re-enter the mode — it hits that same
  loud path, and the flag must be re-typed.

  `code-reviewer` (Fable, 46/46 model records, no fallback) found that the
  no-op warning was a **fully silent drop on the delegated path**:
  `--identity` with no `--ref-image` and a warm daemon delegated, the daemon ran
  `generate()` with `identity=False`, and the warning fired nowhere — so
  whether the user was told depended on whether a daemon happened to be up. The
  CLI now also warns client-side before delegating (the run itself is plain
  text2img and still delegates — only the notice was owed), both sites share one
  message builder, and the test that had asserted "the no-op warning covers it"
  was corrected: it had enshrined the silent drop. Two further gaps closed in
  the same pass — the no-op block had no test at all (deleting it passed every
  suite), and `ref_boost`/`grounding_px` on a NON-identity family were accepted,
  recorded in the sidecar, and never applied with no warning, while the
  delegation gate's own message implied running in-process would apply them.
  That last one predates this slice (Part B), but this slice's own comment
  claimed the skip existed, so it is closed here rather than deferred.

  Verified live: with the flag, output is **bit-identical** to the pre-flag run
  (entry changed, generation did not) and still 0.53/255 from the port golden;
  the sidecar carries no `identity` key; wrong-family and no-refs both warn and
  proceed; qwen-edit's reference edit still runs normally under a stray
  `--identity`; with a daemon up the run stays in-process with the loud reason
  while plain krea keeps delegating.
- 2026-07-31 — **First GPU run. Parity reached; two Part A defects found, and
  constraint 1 amended.** Both defects were invisible to 57 CPU tests because
  each lived in an interface the CPU tests stubbed:

  1. **`__call__` dispatched subclass methods through `self`.** Under
     `identity_edit_pipe_call` `self` IS a stock `Krea2Pipeline`, so
     `self._normalize_sources(...)` raised `AttributeError` through diffusers'
     `ConfigMixin.__getattr__`. Every CPU test had exercised the BOUND path,
     where `self.` resolves fine — so no behavioural test on a real subclass
     instance could have caught it. Fixed by class-qualifying all six sites;
     `test_krea2_identity.py` now AST-guards both directions (no `self.`
     dispatch of an own method, and the six ARE class-qualified).
  2. **The grounded encode owned half of a contract it should not have.**
     Stock Krea-2 text2img drives this encoder **text-only**; the identity edit
     drives the same encoder **multimodally**, and that mode requires image
     placeholders expanded to one token per merged vision patch plus an
     `mm_token_type_ids` modality mask for M-RoPE. Building only the IMAGE
     processor and tokenizing separately made this repo the owner of that
     text-side contract — which is versioned with **transformers**, not with
     our checkpoint. Two of its requirements failed in succession on the first
     live run, and a third could arrive with any bump.

  **Constraint 1 is amended, not weakened.** Its point was that the processor
  must describe the LIVE encoder, never a checkpoint directory — that still
  holds exactly. What was wrong was inferring from it that we should build only
  an image processor. `build_vl_processor` now COMPOSES HF's real
  `Qwen3VLProcessor` from (a) the image processor still built from the live
  encoder's `vision_config` and (b) the pipeline's own tokenizer. No second
  repo, no `preprocessor_config.json` (Krea-2 ships none), no
  `AutoProcessor.from_pretrained` — and the text side belongs to HF again.
  Notably this is what the validated reference harness did all along
  (`run_two.py:80` calls `self.vl_processor(text=..., images=...)`); the
  image-processor-only reading was Part A's divergence from the thing the
  2026-07-31 validation actually proved.

  Also added: `strip_vision_control_tokens` removes vision control tokens from
  the USER's instruction before templating, so an instruction containing
  `<|image_pad|>` cannot crash the run on a placeholder-count mismatch it did
  not cause. Processor memoization is now keyed on `(encoder, tokenizer)`
  identity, since a tokenizer swap moves the vision token IDs.

  **Results.** Parity against the port golden (`port_rb4_gp768_s1234.png`, same
  source / prompt / `ref_boost` 4.0 / `grounding_px` 768 / seed 1234 / 10
  steps): mean abs diff **0.53/255**, RMSE 1.93, ~96% of pixels within 2/255 —
  kernel-ordering noise, not a semantic difference. Run-to-run **bit-identical**.
  Two-source (`n_src=2`) runs and honours the frame-order invariant. All four
  Part B gates fire live (no-LoRA warn, `--rebalance` skip, NAG skip, 3-ref
  hard error), `vl`/`ref` MODE hard-errors, and `edit_warnings` rides the
  daemon wire into a delegated sidecar (invariant N1 confirmed end-to-end).
  Delegation gate confirmed both ways: default tuning delegates, non-default
  stays in-process with the loud reason. Non-krea reference paths re-verified
  unchanged live (qwen-edit and flux2-klein), as was plain krea text2img.
  **Epic risk 3 is resolved** — the cuDNN-pinned attention backend accepts our
  float additive mask; no fallback, no error.
- 2026-07-31 — **Part B landed** (comfyless routing + params). The seven items
  the epic's Part B decomposition named are in, within its declared edit scope
  (`generate.py`, `params_schema.py`, `params_validation.py`, `test_ref_edit.py`,
  `test_params_schema.py`). Four gates were added beyond that list, each closing
  a silent drop found while wiring — this is the substantive part of the record,
  because none of them were foreseen by the epic:

  1. **`--rebalance` is pre-gated off on this path.** `_apply_krea_rebalance`
     pops `prompt` and substitutes `prompt_embeds`, which the identity call
     swallows through `**kwargs` while its grounded encode receives an empty
     instruction. Running both would have discarded the prompt entirely.
  2. **`cfg_scale` / `negative_prompt` / `max_sequence_length` are named by
     `Krea2IdentityEditPipeline.__call__` but consumed only on its
     no-reference branch**, so they are inert on the edit path. This is not
     hypothetical: `FAMILY_DEFAULTS["krea"]` sets `cfg_scale: 3.5`, so on
     Krea-2-**Raw** a user who typed nothing still lost CFG and any negative
     prompt. Now a loud skip in `edit_warnings` (invariant N1). The 2026-07-31
     live validation ran on **Turbo** (cfg 0.0), which is exactly why this was
     invisible to it. The requested values stay recorded in the sidecar —
     they are replay inputs, and the warning carries the truth.
  3. **Range warnings** for the two scalars, warn-never-block per D5.
  4. **A daemon-delegation gate.** Both keys are now `SCHEMA_KIND` members, so
     the canonical validator would ACCEPT them on the wire while `server.py`
     ignores them — accepted-and-dropped. A reference run whose tuning diverges
     from the schema defaults therefore runs in-process (epic D7).

  Gate 4's first implementation keyed on **presence** in `explicit_keys` and was
  wrong; `code-reviewer` (Fable) caught it. `generate()` records both keys in
  every sidecar and `--params` treats every sidecar key as explicit, so presence
  forced **every** ref-bearing replay in-process — on qwen-edit and flux2 too.
  That broke the epic invariant *"Non-krea `--ref-image` behaviour is
  untouched"* and re-armed the 2026-07-26 warm-daemon failure (an in-process
  model load while the daemon still holds its pipeline's VRAM is a crash on a
  single-GPU box, not a degrade). The gate now tests **value divergence from the
  schema default**, with the replay case pinned as an explicit negative test.
  The same review also found the metadata block recording a `rebalance` entry
  for runs gate 1 had skipped — untruthful provenance, now mirrored.

  Verification: `just tests` 32/32 suites (test_ref_edit 196, test_params_schema
  349); pyright per-root exactly at baseline (`comfyless=13`, `nodes=520`,
  `pipelines=454`); `just policy-test` 43/43; gitleaks, semgrep, `deps-cve`
  clean. **Part B has still never run on a GPU** — the live parity smoke against
  `krea-identity-eval/` is the next step, and epic risk 3 (the cuDNN attention
  pin vs. our float bias) remains unpinned by anything but a CPU-shaped test.
- 2026-07-31 — accepted. Written at Part A, as
  `docs/vision/epic-krea2-identity-edit.md` ("Numbering decisions") committed to
  doing. Supersedes nothing. Live validation preceding this ADR is recorded in
  the epic's "Live validation — 2026-07-31" section; evaluation artifacts in
  `krea-identity-eval/`.

**AI-Disclosure:** Claude (Opus 5) authored; Grant reviewed.
