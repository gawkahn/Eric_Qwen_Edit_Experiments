# Vision (epic): Krea 2 Identity Edit in comfyless

**Date:** 2026-07-25 · **Revised:** 2026-07-31 (live validation) · **ADR:** none
yet (deliberate — see "Numbering decisions") · **Risk:** L2 · **Status:**
**validated live; not started**

> **Posture:** No new boundary. Entry is the existing ADR-035 `--ref-image`
> surface; ingestion reuses `comfyless/ref_image.py` unchanged, so there is no
> new decode site. Parts A+B touch **zero Red Zone paths**. Part C (daemon +
> MCP carriage) is Red Zone and carries the full §12 process.

## Intent

`conradlocke/krea2-identity-edit` is instruction-based, identity-preserving
image editing for Krea 2 — "change the suit jacket to a red leather jacket" on a
photo of a specific person, with the likeness intact. It ships as a LoRA.

**The concrete use case driving this (Grant, 2026-07-31): generate images of
himself with different haircuts, good enough to hand to a stylist as reference.**
That is the product. It demands two things at once — a real structural change to
the hair, and a face that still reads as him — and it is exactly the task the
qwen-edit backend cannot do (see "Why this is not ADR-038's job").

The model card requires a dedicated ComfyUI node pack
(`lbouaraba/comfyui-krea2edit`), which prompted the question this Vision
answers: is this a new comfyless mode (`comfyless.krea_identity`) or a new
family with special characteristics — or does it fit the surface we already
have?

**Answer: it fits.** The node pack exists because *ComfyUI* has no seam for dual
conditioning, not because the model is architecturally exotic. In comfyless it
is `--ref-image portrait.png` on a Krea-2 checkpoint with the identity LoRA
loaded, plus one new pipeline module and two tuning params.

## Live validation — 2026-07-31

The whole design was exercised end-to-end **before** any comfyless code, by
running the reference port offline against our own model tree from a scratch
harness outside the repo (`~/scratch/krea2-identity-port/`). Nothing was
installed and no repo file was touched. This section is the empirical basis for
the revisions below; it replaces guesses with measurements.

**It works, and it beats qwen-edit at the identity task.** The port's own
red-leather-jacket example reproduced faithfully: jacket replaced with correct
zipper/collar/pocket hardware, while face, glasses, hair, pose, wedding ring,
belt buckle, shirt buttons and even the wall smudges were untouched.

**It runs on our existing pins — there is no dep slice.** 13–16 s per edit,
34.2 GiB resident on cuda:0, LoRA merge 3 s, at `diffusers==0.39.0` /
`transformers==5.5.3` / `torch 2.11+cu130`.

**Measured `ref_boost` behaviour on a real subject** (Grant's own photo,
instruction: undercut with shaved sides, swept back; `grounding_px=384`):

| `ref_boost` | Edit applied | Identity |
|---|---|---|
| 1.0 | full undercut | head geometry drifts **narrower/more oval**; his jaw and forehead are broader |
| **1.25** | **full undercut** | **best of the set — the winner** |
| 1.5 | applied | source's long hair bleeds through as detached wisps near the ears |
| 4.0 / 8.0 | **essentially absent** | perfect |

Naming the geometry in the instruction ("same broad forehead, wide jawline")
recovers head shape at `rb=1.0` with no wisps — a zero-code mitigation worth
documenting for users. `grounding_px` is the weaker lever (384 retained
slightly more skin texture than 768).

**Why the drift happens, and why a bigger `ref_boost` cannot fix it.**
`ref_boost` adds `log(ref_boost)` to the attention logits where target queries
attend to source-block keys — so raising it instructs the model to copy harder
from the very region the edit is trying to change. Shaving the sides forces the
model to repaint the head silhouette, and identity (skull width) and the edit
(hair) occupy the *same* pixels. One global scalar cannot separate them. This is
the binding constraint on the whole feature, and it is why D10 exists.

## Findings that shape the approach (investigation 2026-07-25, verified 2026-07-31)

| Finding | Source | Consequence |
|---|---|---|
| It is a **LoRA on Krea 2**, not a checkpoint. r64 = 457 MB, r128 = 914 MB, full = 1.83 GB | HF repo file list | No new family string, no new catalog `kind`. `--lora` carries it. Grant has the **full 1.83 GB v1.2**. |
| Card names **Krea-2-Raw** as base; the diffusers port loads **Krea-2-Turbo** | model card / port `inference.py` | Both already on disk. Support both; Turbo is the practical default. **Validated on Turbo.** |
| Mechanism 1: VAE-encode the source and **prepend it as clean tokens at RoPE frame 1** (target = frame 0) | node pack `Krea2EditModelPatch`; port `_edit_position_ids` | Pipeline-layer concern. No transformer patch. **Verified in source.** |
| Mechanism 2: **image-grounded instruction encode** — a `<vision_start><image_pad><vision_end>` block before the instruction so Qwen3-VL grounds the edit semantics | node pack `Krea2EditGroundedEncode`; port `_grounded_encode` | Needs an image processor. **Solved — see D10.** |
| `ref_boost` (default 4.0) adds `log(ref_boost)` to attention logits on source keys | port `_ref_boost_bias` | A swapped attention processor — the `NAGKrea2AttnProcessor` mold. **Default is wrong for this use case — see D4.** |
| `grounding_px` (default 768, useful 384–1024) caps the longest side fed to the vision tower | both | Plain int param. Weak lever in practice. |
| Recommended ≤ 2 MP, 8–12 steps, CFG 1.0 | model card | Already inside `krea-turbo`'s 8 steps / cfg 0.0 — no new `FAMILY_DEFAULTS` row (D5). The port itself caps at **1 MP** (`MAX_EDIT_MEGAPIXELS`). |
| A **pure-diffusers port exists** — `huan-yin/Krea2-Identity-Edit-Diffusers` (Apache-2.0), **GitHub not HF** | GitHub | Read, run, and cited. **Not vendored** (D2). Its `requirements.txt` over-pins; its own import guard declares the real floor, `transformers>=4.57`. |

### The finding that makes this cheap

diffusers 0.39.0's **native** `Krea2Transformer2DModel.forward` already accepts
everything the in-context prepend needs: caller-built `position_ids`,
arbitrary-length packed `hidden_states`, a per-block `attention_mask` reaching
`dispatch_attention_fn`, and a swappable `Krea2AttnProcessor` — the same hook
`pipelines/nag_krea2.py` already uses.

The reference port's ~1650 lines are mostly a **vendored copy of a transformer
we already have natively**. Our version is a `Krea2Pipeline` subclass plus one
attention processor.

## Design decisions

**D1 — No new mode, no new family.** ADR-035 decision 2 binds: *"There is no
edit mode. Edit is generation with reference conditioning."* `_REF_FAMILY_KINDS`
(`comfyless/generate.py:1745`) gains `"krea": "krea2-identity"` and
`"krea-turbo": "krea2-identity"`. There is no `comfyless/krea_identity.py`.

**D2 — Own thin subclass, not the vendored port.** New
`pipelines/krea2_identity_edit.py` built on the native transformer, in the
`nag_krea2.py` mold: an unbound `__call__` invoked on the stock cached pipeline
instance so the daemon's cache key stays clean and the cached object is never
mutated. Rationale: no `trust_remote_code`; no re-vendoring a transformer
already in the pinned diffusers. *(Revised 2026-07-31: the original rationale
also cited the port's `transformers==5.12.1` pin against our 5.5.3. That
argument is dead — see Risk 2 — but the other two reasons stand on their own.)*

**D3 — MODE: `both` only.** `vl`/`ref` are qwen-edit's dual-path selectors.
Krea's two conditioning paths are always co-active, so `vl`/`ref` are a hard
error, reusing the validation slot flux2-native already occupies (ADR-036
decision 3).

**D4 — Two new sidecar params, and the card's default is wrong for our use
case.** `ref_boost` (float) and `grounding_px` (int, default 768), both in
`SCHEMA_KIND` (replayable), not `_RUNTIME_KIND`. Follows the ADR-023 NAG
checklist verbatim: `params_validation.py` `SCHEMA_KIND` + `params_schema.py`
`_FIELD_DEFAULTS` (the import-time drift guard fails loudly on a half-edit) →
argparse with a **`default=None` sentinel** so sidecar replay survives →
`generate()` kwarg → consumption behind a family gate with a `*_warnings` list →
metadata dict. Loud skip on non-krea families, exactly like `--nag-*`.

*Revised 2026-07-31.* The card's `ref_boost` default of 4.0 is right for edits
whose target is **spatially separate** from the identity (the red jacket) and
wrong for anything face-adjacent, where it suppresses the edit outright.
Measured best for hair edits: **1.25**. Ship 4.0 as the schema default to match
upstream, but the *useful* value is task-dependent in a way no single default
can capture — which D11 resolves.

**D5 — The LoRA is the user's responsibility, warned not enforced.** The
identity behaviour lives in the LoRA, so nothing in `model_index.json` can
detect it. `--ref-image` on a krea checkpoint with no LoRA loaded emits a loud
warning and proceeds (house rule: warn, don't block). Corollary: **no new
`FAMILY_DEFAULTS` row.**

**D6 — NAG is pre-gated off on the krea ref path**, mirroring ADR-036
decision 6. Both features swap the same processor, and ADR-023 hazard H1 is
precisely about processor-install ordering on Krea.

**D7 — CLI foreground first; daemon and MCP are a separate part.** References
already ride the existing `ref_images` wire field, so parts A+B need **no**
change to `server.py`, `mcp_server.py`, or `ref_image.py` — all three Red Zone
paths. If part B starts needing `server.py`, stop and re-scope.

**D8 — Single source image in v1. `[REVISED 2026-07-31 — the reason changed.]`**

The original rationale was *cost*: that two-input needs "multi-frame position ids
and, in the reference port, a per-block reference-KV precompute
(`precompute_ref_kv`), the one piece that genuinely reaches into transformer
internals." **Both halves are false.** `_edit_position_ids(..., n_src)` already
builds sources at frames 1..N — only the call site hardcodes 1 — and
`precompute_ref_kv` is an unrelated ostris multi-reference KV path that is dead
code with no call sites in the port. The real delta is three seams, none of
which touch transformer internals:

1. a two-image grounded encode (the port's own `Picture N:` convention),
2. `cat([scene_packed, person_packed], dim=1)`,
3. `n_src=2`.

`_ref_boost_bias` needs no change at all — it slices every source column as one
range.

It was built and run live on 2026-07-31 (`run_two.py`, monkeypatching those
three seams onto the port so its sampler, scheduler and CFG stayed untouched).
**Frame semantics, confirmed by a swapped-order positive control: frame 1 =
scene/context, frame 2 = identity.** Swapping produced the *other* man's face in
the subject's doorway — cleanly inverted, proving the frames land where
intended. Order is fixed and matters, as the card states.

**Two-input is a COMPOSITING mode, not attribute transfer** — "this person, in
that scene." It structurally cannot do "give him the haircut from image 1",
because the slot carrying the reference also carries identity, so hair follows
the person in slot 2.

**So two-input stays out of v1 — on value, not cost.** Grant's verdict on the
live results: *"consistently preserving neither identity nor haircut in either
ordering"* — it blended the two faces. **Honest caveat: this does not disprove
the trained capability.** The card's layout is *scene* + person (its own example
is "next to the tractor" — frame 1 is a tractor, not a face). Our test put two
competing faces into a layout trained for one, which is the pathological case.
The trained scene+person use case is **untested here**. Revisit if a
put-this-person-in-that-scene need appears; it is cheap to add when it does.

**D9 — Ingestion reuses `comfyless/ref_image.py` unchanged.** The single decode
site is preserved; `load_ref_image_capped`, never `load_seed_image_capped`
(ADR-038 D5 records why). Provenance is already family-agnostic, so sidecar
recording and the ADR-035 slice-5 replay trust gate come for free.

**D10 — The VL image processor is code-owned, built from the checkpoint's own
`vision_config`. `[NEW 2026-07-31 — closes the epic's largest unknown.]`**

Krea-2's `text_encoder` is architecturally **Qwen3-VL-4B exactly**: all 15
`vision_config` fields, all four special-token ids, and the text dims match the
local `Huihui-Qwen3-VL-4B-Instruct-abliterated`, and the tokenizer vocabs are
*identical*. Krea's own `chat_template.jinja` renders the vision block unaided.

So we construct `Qwen2VLImageProcessor` in code, reading the shape-critical
values (`patch_size`, `temporal_patch_size`, `spatial_merge_size`) from
**Krea-2's own config**, with only mean/std/rescale as code constants. Verified
byte-parity against the donor's on-disk config at 384/768/1024 px, with
`image_pad` expansion matching the token grid in every case.

This **deletes** both options the epic originally proposed — the
`--vl-processor` path param and the ship-a-config-as-package-data fallback — and
is strictly better than either: no donor-model dependency, no network, and the
values that must not drift come from the checkpoint being loaded.
`local_files_only=True` holds trivially because nothing is fetched.

**D11 — Tuning params ride the catalog, not the CLI defaults. `[NEW
2026-07-31 — Grant's call.]`**

Empirically-earned values (`ref_boost≈1.25` for hair edits, ≈4 for
spatially-separate edits) belong in the **catalog suggestion language**, to be
picked up by iterate loops exactly like a LoRA strength value — not baked into a
single schema default that must serve both. This ties directly to the
experiential-enrichment backlog item (operator experience as the
highest-trust description tier).

**Explicitly rejected: any draw-a-mask component.** Grant: *"I can never make
those work the way I want even with a full GUI to play with."* A spatial mask
**inferred from the prompt** would be acceptable; a hand-painted one is not, and
no mask UI is to be built.

Recorded for whoever picks up the masking idea: a spatially-varying `ref_boost`
is only ~10 lines in the bias builder we already own under D2. The port writes
one scalar over a slice —
`bias[:, :, rows0:, text_len:rows0] = log(boost)` — whose source columns are a
known `grid_h × grid_w` token layout, so a per-source-token boost vector
broadcasts into the same slice. That is the principled fix for the
identity/edit collision described in "Live validation". The open question is
purely **where the mask comes from**, never whether the mechanism supports it.

## Decomposition

### Part A — the pipeline module

New `pipelines/krea2_identity_edit.py` (~500 lines): a
`Krea2IdentityEditPipeline(Krea2Pipeline)` subclass, a
`Krea2IdentityEditAttnProcessor`, and an `identity_edit_pipe_call(pipe,
**kwargs)` entry mirroring `nag_krea2.nag_pipe_call`. Pieces:

- grounded prompt encode — VL chat template with the vision block, tapping the
  same 12 `text_encoder_select_layers` the checkpoint declares
- the **code-owned image processor** (D10)
- source VAE encode + pack into the transformer's token layout
- `position_ids` builder for `[text | source(frame 1) | target(frame 0)]`,
  written `n_src`-generic from the start so D8 is a call-site change later
- the `ref_boost` additive attention-logit bias
- per-call processor install with `finally`-restore, inheriting
  `_attention_backend` from the cuDNN pin (risk 3)

New `test_krea2_identity.py` (~250 lines) on a tiny synthetic transformer,
following `test_nag.py`'s shape.

### Part B — comfyless routing + params

`_REF_FAMILY_KINDS` rows (D1); a `"krea2-identity"` execution branch beside
`_run_qwen_edit_refs` and `_apply_flux2_native_refs`; MODE validation (D3); the
two params through the D4 checklist; the no-LoRA warning (D5); the NAG pre-gate
(D6). ~200 lines in `generate.py`. Tests extend `test_ref_edit.py` and
`test_params_schema.py`.

### Part C — daemon + MCP (deferred; Red Zone)

The two scalars onto the wire, **excluded from `_request_cache_key`** — they
change the output, not the loaded pipeline, which is the NAG precedent — plus
MCP tool-schema properties. `security-auditor` (Fable) required. If an ADR is
ever written for this work, it is written here.

## Invariants

- **No new dispatch mode and no new module under `comfyless/`.** Reachable only
  through `--ref-image`; no `--identity-edit` flag, no mode selector.
- **The cached pipeline object is never mutated.** The subclass `__call__` runs
  unbound on the stock instance; any processor swap is per call and restored in
  a `finally`. A krea text2img run after an identity-edit run in the same
  process matches one before it.
- **`ref_boost` / `grounding_px` never enter the daemon pipeline cache key.**
- **MODE `vl` or `ref` on a krea family is a hard error** naming the family and
  the valid mode — never a silent coercion to `both`.
- **Every reference still decodes through `comfyless/ref_image.py`.**
- **`local_files_only=True` holds through the grounded encode** — trivially, by
  D10, since the processor is constructed rather than fetched.
- **A krea run with no `--ref-image` is unchanged.**
- **Non-krea `--ref-image` behaviour is untouched.**

## Failure semantics

Fail-closed on anything structural, warn-and-proceed on user-authority choices.
A bad MODE, an unparseable `--ref-image`, or a cap violation exits nonzero with
the fault named, before the GPU is touched and before anything is written. A
krea checkpoint with references but no LoRA warns loudly and proceeds (D5). A
reference handed to a family that cannot consume it keeps today's behaviour:
loud drop under lenient, `ValueError` under strict.

## Out of scope

Two-image scene + person compositing (D8); daemon and MCP carriage (part C);
ComfyUI nodes for this path; any `FAMILY_DEFAULTS` change (D5);
Krea-2-Raw-specific tuning; reconciling with ADR-038's identity-judging rubric
(risk 5); **any hand-painted mask UI (D11, permanently)**.

## Risks

1. ~~**The Qwen3-VL image processor is missing from the checkpoint.**~~
   **RESOLVED 2026-07-31 — see D10.** Verified offline with byte-parity against
   a donor config. No param, no package data, no network.
2. ~~**`transformers==5.5.3` vs the port's `5.12.1`.**~~ **FALSIFIED
   2026-07-31.** The port's `requirements.txt` over-pins; its own import guard
   declares the real floor, `transformers>=4.57`, and `torch>=2.5` for GQA-SDPA.
   The entire live validation ran on the repo's existing `.venv` with **zero
   installs**. There is no dep slice.
3. **The cuDNN attention pin interacts with the new processor.**
   `_pin_krea_attention_backend` writes `_attention_backend` onto *existing*
   processor instances — ADR-023 hazard H1 is exactly this. The new processor
   must inherit or re-apply it. Note the port explicitly wraps its denoise in
   `sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION])` because a float bias
   makes flash decline the mask and the MATH kernel would OOM on the doubled
   token count — our processor must preserve that property. Pin the outcome
   with a test.
4. **Quant interaction.** Standing rule from the Krea LoRA regression: native
   fp8/int8 checkpoints plus LoRAs ⇒ `--quant fp8`. Untested with an added
   attention bias — a live-smoke item, not a code item.
5. **ADR-038 overlap.** ADR-038 is accepted and is *literally about face
   identity*. Krea identity edit is a competing **backend** for the same goal.
   Reconcile with whichever lands first.
6. **`ref_boost` cannot separate identity from a face-adjacent edit. `[NEW]`**
   The binding quality constraint (see "Live validation"). Mitigations available
   today are the narrow 1.0–1.25 band and geometry-naming in the instruction;
   the principled fix is the per-token bias sketched in D11. If live use finds
   the band too narrow to be usable, that fix moves from "recorded" to
   "required".

## Why this is not ADR-038's job

The backlog's three-variant result (2026-07-25) showed qwen-edit **cannot** do
identity/face swap: `prompt_adherence` pinned at 3 in every iteration of all
three variants, and the `:both` runs copied the reference's *glasses* while
leaving bone structure untouched — a backend limitation, not a loop one. The
2026-07-31 validation shows Krea-2 doing the same class of task convincingly.
That is the whole argument for this epic existing alongside ADR-038.

## Proof hooks

- `./.venv/bin/python3 test_krea2_identity.py` (new, part A) — position-id
  layout for `[text | source | target]`, `ref_boost` bias placement and the
  `ref_boost == 1.0` no-op, processor install/restore, output-slice shape, and
  **D10's processor built from a stub `vision_config`**.
- `./.venv/bin/python3 test_ref_edit.py` — krea routing, MODE `vl`/`ref`
  rejection, the no-LoRA warning, NAG pre-gate.
- `./.venv/bin/python3 test_params_schema.py` — the two params' schema
  round-trip and `--params` replay.
- `just tests` — full battery, 0 failures.
- `python -m py_compile pipelines/krea2_identity_edit.py comfyless/generate.py`
- **Parity smoke against the validated harness:** the same portrait +
  instruction at `ref_boost=1.25, grounding_px=384, seed=1234` should land in
  the same place as `~/scratch/krea2-identity-port/run_local.py`. The
  2026-07-31 artifacts in `krea-identity-eval/` are the reference set.

## Edit scope (hard)

**Part A:** new `pipelines/krea2_identity_edit.py`, new
`test_krea2_identity.py`.
**Part B:** `comfyless/generate.py`, `comfyless/params_schema.py`,
`comfyless/params_validation.py`, `test_ref_edit.py`, `test_params_schema.py`.
**Part C (separate edit scope, Red Zone):** `comfyless/server.py`,
`comfyless/mcp_server.py`, `test_server_robustness.py`, `test_mcp_server.py`.

Plus this Vision doc and its vault mirror. Nothing else — in particular,
`comfyless/ref_image.py` and `comfyless/family_defaults.py` are deliberately
untouched (D5, D9). D11's catalog carriage is a **separate** slice against the
catalog plane, not part of A or B.

## Numbering decisions

**No ADR yet, deliberately.** *(Reaffirmed 2026-07-31.)* The original reason was
that load-bearing assumptions had expiry dates — and this revision is the proof:
two of the epic's five risks were resolved or falsified within one session of
contact, and D8's rationale turned out to be wrong. The design has now survived
contact, so an ADR at Part A is defensible; write it there, or at Part C.

**Lens:** team-portable. Every seam used here — `_REF_FAMILY_KINDS` routing, the
`nag_*.py` per-call override mold, the `SCHEMA_KIND`/`_FIELD_DEFAULTS` param
checklist, `ref_image.py` ingestion — is an existing repo convention, and the
decision to build rather than vendor is the one a team would make for the same
reason.

**AI-Disclosure:** Claude (Opus 5) authored; Grant reviewed.
