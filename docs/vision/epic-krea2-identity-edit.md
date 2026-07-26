# Vision (epic): Krea 2 Identity Edit in comfyless

**Date:** 2026-07-25 · **ADR:** none yet (deliberate — see "Numbering
decisions") · **Risk:** L2 · **Status:** evaluated, NOT started

> **Posture:** No new boundary. Entry is the existing ADR-035 `--ref-image`
> surface; ingestion reuses `comfyless/ref_image.py` unchanged, so there is no
> new decode site. Parts A+B touch **zero Red Zone paths**. Part C (daemon +
> MCP carriage) is Red Zone and carries the full §12 process.

## Intent

`conradlocke/krea2-identity-edit` is instruction-based, identity-preserving
image editing for Krea 2 — "change the suit jacket to a red leather jacket" on
a photo of a specific person, with the likeness intact. It ships as a LoRA.

The model card requires a dedicated ComfyUI node pack
(`lbouaraba/comfyui-krea2edit`), which prompted the question this Vision
answers: is this a new comfyless mode (`comfyless.krea_identity`) or a new
family with special characteristics — or does it fit the surface we already
have?

**Answer: it fits.** The node pack exists because *ComfyUI* has no seam for
dual conditioning, not because the model is architecturally exotic. In
comfyless it is `--ref-image portrait.png` on a Krea-2 checkpoint with the
identity LoRA loaded, plus one new pipeline module and two tuning params.

## What must be true when done

1. `python -m comfyless.generate --model $MB/Krea-2-Turbo --lora
   krea2_identity_edit_v1_2_r64:1.0 --prompt "<instruction>" --ref-image
   portrait.png` produces an edited image with the subject's likeness intact.
2. No new dispatch mode, no new family string, no new module under
   `comfyless/`. The only comfyless-side change is routing + two params.
3. `ref_boost` and `grounding_px` are sidecar-recorded and `--params`-replayable
   like every other schema key.
4. A `--ref-image` run on a non-krea, non-edit family behaves exactly as it does
   today (loud drop under lenient, hard error under strict).
5. A text2img krea run with no `--ref-image` is byte-identical to today.
6. `local_files_only=True` holds — nothing in the grounded-encode path reaches
   the network.

## Findings that shape the approach (investigation 2026-07-25)

| Finding | Source | Consequence |
|---|---|---|
| It is a **LoRA on Krea 2**, not a checkpoint. r64 = 457 MB, r128 = 914 MB, full = 1.83 GB | HF repo file list | No new family string, no new catalog `kind`. `--lora` carries it. |
| Card names **Krea-2-Raw** as base; the diffusers port loads **Krea-2-Turbo** | model card / port `inference.py` | Both already on disk (`$MB/Krea-2-Raw`, `$MB/Krea-2-Turbo`). Support both; Turbo is the practical default. |
| Mechanism 1: VAE-encode the source and **prepend it as clean tokens at RoPE frame 1** (target = frame 0) | node pack `Krea2EditModelPatch`; diffusers port `_edit_position_ids` | Pipeline-layer concern. No transformer patch — see below. |
| Mechanism 2: **image-grounded instruction encode** — a `<vision_start><image_pad><vision_end>` block before the instruction so Qwen3-VL grounds the edit semantics on the source | node pack `Krea2EditGroundedEncode`; port `_grounded_encode` | Needs an image processor. This is the largest open cost — risk 1. |
| `ref_boost` (default 4.0, useful ~1–8) adds `log(ref_boost)` to attention logits where target queries attend to source-block keys | port `_ref_boost_bias` | A swapped attention processor — the `NAGKrea2AttnProcessor` mold exactly. |
| `grounding_px` (default 768, useful 384–1024) caps the longest side fed to the vision tower; lower favours instruction adherence, higher favours likeness | both | Plain int param. |
| Recommended ≤ 2 MP, 8–12 steps, CFG 1.0 | model card | Already inside `krea-turbo`'s 8 steps / cfg 0.0 — no new `FAMILY_DEFAULTS` row (D5). |
| A **pure-diffusers port already exists** — `huan-yin/Krea2-Identity-Edit-Diffusers` (Apache-2.0), `Krea2IdentityEditPipeline`, pinning `diffusers==0.39.0` (our exact pin) | GitHub | Read as reference and cited. **Not vendored** (D2). |

### The finding that makes this cheap

diffusers 0.39.0's **native** `Krea2Transformer2DModel.forward`
(`.venv/lib/python3.12/site-packages/diffusers/models/transformers/transformer_krea2.py:448`)
already accepts everything the in-context prepend needs:

- `position_ids`, shape `(text_seq_len + image_seq_len, 3)`, is **built by the
  caller**. Source rows carrying frame index 1 and target rows carrying 0 is
  therefore a pipeline concern, not a model concern.
- `hidden_states` is packed image tokens of **arbitrary** length; the source
  block is just extra rows, dropped after the `hidden_states[:, text_seq_len:]`
  slice at `:517`.
- The per-block `attention_mask` reaches `dispatch_attention_fn(attn_mask=...)`
  (`transformer_krea2.py:77-81`), and `Krea2AttnProcessor` (`:54`) is swappable
  — the same hook `pipelines/nag_krea2.py` already uses.

The reference port's ~1800 lines are mostly a **vendored copy of a transformer
we already have natively** (it vendors so it can load via `custom_pipeline=` +
`trust_remote_code`, which this project avoids anyway). Our version is a
`Krea2Pipeline` subclass plus one attention processor.

## Design decisions

**D1 — No new mode, no new family.** ADR-035 decision 2 binds: *"There is no
edit mode. Edit is generation with reference conditioning."* `_REF_FAMILY_KINDS`
(`comfyless/generate.py:1708`) gains `"krea": "krea2-identity"` and
`"krea-turbo": "krea2-identity"`. There is no `comfyless/krea_identity.py` — the
Stable Cascade shape (ADR-010: 2069 lines plus a permanent hand-maintained
denylist of unsupported flags) is what a dedicated dispatch mode costs here, and
the forcing constraint that justified it is absent.

**D2 — Own thin subclass, not the vendored port.** New
`pipelines/krea2_identity_edit.py` built on the native transformer, in the
`nag_krea2.py` mold: an unbound `__call__` invoked on the stock cached pipeline
instance (`nag_pipe_call`, `pipelines/nag_krea2.py:506`) so the daemon's
pipeline cache key stays clean and the cached object is never mutated.
Rationale: no `trust_remote_code` (absent codebase-wide except the hash-pinned
hunyuan-reprompt backend); no re-vendoring of a transformer already in the
pinned diffusers; the port pins `transformers==5.12.1` against our `5.5.3`. The
port is read, diffed against, and credited — not copied.

**D3 — MODE: `both` only.** `vl`/`ref` are qwen-edit's dual-path selectors.
Krea's two conditioning paths are always co-active — that is what the LoRA was
trained on — so `vl`/`ref` are a hard error, reusing the validation slot
flux2-native already occupies at `_resolve_ref_family_support:1741-1749`
(ADR-036 decision 3).

**D4 — Two new sidecar params.** `ref_boost` (float, default 4.0) and
`grounding_px` (int, default 768), both in `SCHEMA_KIND` (replayable), not
`_RUNTIME_KIND`. Follows the ADR-023 NAG checklist verbatim:
`comfyless/params_validation.py` `SCHEMA_KIND` + `comfyless/params_schema.py`
`_FIELD_DEFAULTS` (the import-time drift guard at `params_schema.py:102` fails
loudly on a half-edit) → argparse with a **`default=None` sentinel** so sidecar
replay survives → `generate()` kwarg → consumption behind a family gate with a
`*_warnings` list → metadata dict. Loud skip on non-krea families, exactly like
`--nag-*`.

**D5 — The LoRA is the user's responsibility, warned not enforced.** The
identity behaviour lives in the LoRA, so nothing in `model_index.json` can
detect it; auto-detection would be guessing. `--ref-image` on a krea checkpoint
with no LoRA loaded emits a loud warning and proceeds (house rule: warn, don't
block). Corollary: **no new `FAMILY_DEFAULTS` row** — the card's 8–12 steps /
CFG 1.0 already sits inside `krea-turbo`'s 8 / 0.0. Revisit only if live testing
disagrees.

**D6 — NAG is pre-gated off on the krea ref path**, mirroring ADR-036
decision 6 (`_nag_gate(..., ref_kind=ref_kind)`, `generate.py:2029`). Stated
here rather than left to be discovered: both features swap the same processor,
and ADR-023 hazard H1 is precisely about processor-install ordering on Krea.

**D7 — CLI foreground first; daemon and MCP are a separate part.** The ADR-035
shape (slice 3, then slice 4). References already ride the existing
`ref_images` wire field, so parts A+B need **no** change to `server.py`,
`mcp_server.py`, or `ref_image.py` — all three are Red Zone paths in
`scripts/git-policy/_red-zone-paths.sh`. Keep that boundary: if part B starts
needing `server.py`, stop and re-scope (ADR-036's own rule).

**D8 — Single source image in v1.** The scene + person two-input identity
transfer — the card's headline mode, where image 1 is the scene and image 2 the
person — is deferred. It needs multi-frame position ids and, in the reference
port, a per-block reference-KV precompute (`precompute_ref_kv`), the one piece
that genuinely reaches into transformer internals. Build it against a working
single-image baseline, not blind.

**D9 — Ingestion reuses `comfyless/ref_image.py` unchanged.** The single decode
site is preserved; `load_ref_image_capped`, never `load_seed_image_capped`
(ADR-038 D5 records why: the seed loader has caps but no format allowlist).
Provenance (`path` / `mode` / `sha256` / `applied`) is already family-agnostic,
so sidecar recording and the ADR-035 slice-5 replay trust gate come for free.

## Decomposition

### Part A — the pipeline module

New `pipelines/krea2_identity_edit.py` (~500 lines): a
`Krea2IdentityEditPipeline(Krea2Pipeline)` subclass, a
`Krea2IdentityEditAttnProcessor`, and an `identity_edit_pipe_call(pipe,
**kwargs)` entry mirroring `nag_krea2.nag_pipe_call:506`. Pieces:

- grounded prompt encode — VL chat template with the vision block, tapping the
  same 12 `text_encoder_select_layers` the checkpoint's `model_index.json`
  declares
- source VAE encode + pack into the transformer's token layout
- `position_ids` builder for `[text | source(frame 1) | target(frame 0)]`
- the `ref_boost` additive attention-logit bias
- per-call processor install with `finally`-restore, inheriting
  `_attention_backend` from the cuDNN pin (risk 3)

New `test_krea2_identity.py` (~250 lines) on a tiny synthetic transformer,
following `test_nag.py`'s shape.

### Part B — comfyless routing + params

`_REF_FAMILY_KINDS` rows (D1); a `"krea2-identity"` execution branch beside
`_run_qwen_edit_refs` (`generate.py:1657`) and `_apply_flux2_native_refs`
(`:1634`); MODE validation (D3); the two params through the D4 checklist; the
no-LoRA warning (D5); the NAG pre-gate (D6). ~200 lines in `generate.py` — the
ADR-036 shape (that commit was +185). Tests extend `test_ref_edit.py` and
`test_params_schema.py`.

### Part C — daemon + MCP (deferred; Red Zone)

The two scalars onto the wire (`_build_server_request` → `server.py`
`_handle_generate` kwargs), **excluded from `_request_cache_key`** — they change
the output, not the loaded pipeline, which is the NAG precedent at
`server.py:482-486` — plus MCP tool-schema properties. `security-auditor`
(Fable) required. If an ADR is ever written for this work, it is written here.

## Invariants

- **No new dispatch mode and no new module under `comfyless/`.** The feature is
  reachable only through `--ref-image`; there is no `--identity-edit` flag and
  no mode selector. Family routing stays derived from the checkpoint.
- **The cached pipeline object is never mutated.** The subclass `__call__` runs
  unbound on the stock instance; any processor swap is installed per call and
  restored in a `finally`. A krea text2img run after an identity-edit run in the
  same process produces the same result as one before it.
- **`ref_boost` / `grounding_px` never enter the daemon pipeline cache key.**
  They select output, not weights.
- **MODE `vl` or `ref` on a krea family is a hard error** naming the family and
  the valid mode — never a silent coercion to `both`.
- **Every reference still decodes through `comfyless/ref_image.py`.** Part A
  adds no second decode site.
- **`local_files_only=True` holds through the grounded encode.** No hub call for
  the processor, the template, or anything else, in any code path.
- **A krea run with no `--ref-image` is unchanged** — existing krea, NAG, quant,
  and rebalance tests stay green.
- **Non-krea `--ref-image` behaviour is untouched** — qwen-edit and flux2-native
  routing, warnings, and strictness are byte-identical.

## Failure semantics

Fail-closed on anything structural, warn-and-proceed on user-authority choices
— the established split. A bad MODE, an unparseable `--ref-image`, a cap
violation, or a missing/unbuildable VL processor exits nonzero with the fault
named, before the GPU is touched and before anything is written. A krea
checkpoint with references but no LoRA loaded warns loudly and proceeds (D5) —
the user may be testing the base model deliberately. A reference handed to a
family that cannot consume it keeps today's behaviour: loud drop under lenient,
`ValueError` under strict.

## Out of scope

Two-image scene + person identity transfer (D8); daemon and MCP carriage (part
C); ComfyUI nodes for this path; `--iterate` axes for the two new params beyond
what falls out of the generic mechanism; any `FAMILY_DEFAULTS` change (D5);
Krea-2-Raw-specific tuning; reconciling with ADR-038's identity-judging rubric
(risk 5).

## Risks

1. **The Qwen3-VL image processor is missing from the checkpoint.** The Krea-2
   `text_encoder/` *is* a full `Qwen3VLModel` — `vision_config`,
   `image_token_id`, `vision_start_token_id` / `vision_end_token_id`, 8.3 GB —
   so the vision tower ships with the weights. But `tokenizer/` holds only
   `tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja`; there
   is **no `preprocessor_config.json`**, so `AutoProcessor` cannot be built from
   the checkpoint directory. The reference port lazily pulls it from the hub; we
   cannot. Options: a `--vl-processor` path defaulting to the local
   `$MB/Huihui-Qwen3-VL-4B-Instruct-abliterated` (which has one), or a small
   config shipped as package data. **Decide before writing code** — this is the
   single largest unknown in the epic.
2. **`transformers==5.5.3` vs the port's `5.12.1`.** Verify that `Qwen3VLModel`
   at our pin accepts `pixel_values` + `image_grid_thw` through the
   hidden-state tap, and that the shipped `chat_template.jinja` renders the
   vision block at all. If it does not, this becomes a dep slice before it is a
   feature slice.
3. **The cuDNN attention pin interacts with the new processor.**
   `_pin_krea_attention_backend` (`generate.py:948`) writes `_attention_backend`
   onto *existing* processor instances — ADR-023 hazard H1 is exactly this. The
   new processor must inherit or re-apply it. Separately, `_native_cudnn` may
   not accept a float additive mask where the stock bool key-padding mask
   worked; if not, the bias needs a different carrier or a per-call backend
   switch. Pin the outcome with a test either way.
4. **Quant interaction.** Standing project rule from the Krea LoRA regression:
   native fp8/int8 checkpoints plus LoRAs ⇒ `--quant fp8`. Untested with an
   added attention bias — a live-smoke item, not a code item.
5. **ADR-038 overlap.** `docs/decisions/ADR-038-refine-multi-reference-edit.md`
   is accepted and is *literally about face identity* (face swaps through
   qwen-edit, mixed results by hand). It defines `:judge` references,
   static-vs-advancing refs, and identity-match as a third rubric axis. Krea
   identity edit is a competing **backend** for the same goal. Reconcile with
   whichever lands first rather than building a parallel identity story.

## Proof hooks

- `./.venv/bin/python3 test_krea2_identity.py` (new, part A) — position-id
  layout for `[text | source | target]`, `ref_boost` bias placement and the
  `ref_boost == 1.0` no-op, processor install/restore, output-slice shape.
- `./.venv/bin/python3 test_ref_edit.py` — krea routing, MODE `vl`/`ref`
  rejection, the no-LoRA warning, NAG pre-gate.
- `./.venv/bin/python3 test_params_schema.py` — the two params' schema
  round-trip and `--params` replay.
- `just tests` — full battery, 0 failures.
- `python -m py_compile pipelines/krea2_identity_edit.py comfyless/generate.py`
- **Live smoke (Grant, deferred to pickup):** download the r64 LoRA (457 MB),
  then Krea-2-Turbo + one portrait + one edit instruction, `ref_boost` swept
  1 / 4 / 8 and `grounding_px` 384 / 768, judged by eye against the same edit
  run through qwen-edit.

## Edit scope (hard)

**Part A:** new `pipelines/krea2_identity_edit.py`, new
`test_krea2_identity.py`.
**Part B:** `comfyless/generate.py`, `comfyless/params_schema.py`,
`comfyless/params_validation.py`, `test_ref_edit.py`, `test_params_schema.py`.
**Part C (separate edit scope, Red Zone):** `comfyless/server.py`,
`comfyless/mcp_server.py`, `test_server_robustness.py`, `test_mcp_server.py`.

Plus this Vision doc and its vault mirror. Nothing else — in particular,
`comfyless/ref_image.py` and `comfyless/family_defaults.py` are deliberately
untouched (D5, D9).

## Numbering decisions

**No ADR yet, deliberately.** The pickup date is unknown and several load-
bearing assumptions have expiry dates: a future tagged diffusers release could
ship a Krea-2 edit pipeline class outright (which would collapse part A to an
ADR-036-shaped one-row change), the `transformers` pin will move, and ADR-038
may settle the identity story from the refine side first. An ADR written now
would be superseded before it was implemented. Write it at part C, or at part A
if the design survives contact.

**Lens:** team-portable. Every seam used here — `_REF_FAMILY_KINDS` routing, the
`nag_*.py` per-call override mold, the `SCHEMA_KIND`/`_FIELD_DEFAULTS` param
checklist, `ref_image.py` ingestion — is an existing repo convention, and the
decision to build rather than vendor is the one a team would make for the same
reason (no `trust_remote_code`, no duplicated upstream code).

**AI-Disclosure:** Claude (Opus 5) authored; Grant reviewed.
