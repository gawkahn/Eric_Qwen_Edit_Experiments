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

- 2026-07-31 — accepted. Written at Part A, as
  `docs/vision/epic-krea2-identity-edit.md` ("Numbering decisions") committed to
  doing. Supersedes nothing. Live validation preceding this ADR is recorded in
  the epic's "Live validation — 2026-07-31" section; evaluation artifacts in
  `krea-identity-eval/`.

**AI-Disclosure:** Claude (Opus 5) authored; Grant reviewed.
