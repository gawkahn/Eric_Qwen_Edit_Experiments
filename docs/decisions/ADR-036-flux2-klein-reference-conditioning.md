# ADR-036 — flux2-klein reference-conditioning execution

Status:   accepted
Context:  ADR-035 reserved the family; this ADR settles its execution path.
AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.

## Context

ADR-035 built the reference-image *surface* (schema, `--ref-image PATH[:MODE]`,
ingestion caps, daemon containment, provenance) and the *execution* path for
one family: `qwen-edit` (slice 3). Its routing table (decision 2) reserves
`flux2klein` for "img2img / reference conditioning (style and subject carry)",
but today a `--ref-image` on a flux2klein model takes the drop path —
`_resolve_ref_family_support` (`comfyless/generate.py:1556`) recognizes only
`qwen-edit`, so refs on Klein warn-and-drop (interactive) or hard-fail
(strict/machine mode).

### Investigation findings (2026-07-22, diffusers 0.39.0 — the pinned install)

These answer the design questions ADR-035 left to this slice, and they shape
every decision below:

1. **The auto-detected Klein class natively accepts an image input.** Both
   installed checkpoints (`FLUX.2-klein-9B` distilled, `FLUX.2-klein-base-9B`)
   declare `_class_name: Flux2KleinPipeline`, and
   `Flux2KleinPipeline.__call__` takes
   `image: list[PIL.Image.Image] | PIL.Image.Image | None = None`. This is the
   YES branch of ADR-035 decision 3's pressure point: **no class swap, no
   `from_pipe` conversion, no cache-key discriminator.** The cached pipeline is
   already the ref-capable class; refs are just a call kwarg.
2. **The semantics are multi-reference conditioning, not denoise img2img.**
   Each reference is VAE-encoded and packed into the token stream as reference
   latents with a per-reference time offset (`_prepare_image_ids` — the
   Flux.2 kontext-style path). Output latents start from pure noise; there is
   **no `strength` parameter** on `Flux2KleinPipeline.__call__`. A denoise
   `strength` exists only on `Flux2KleinInpaintPipeline`, a different class
   that also requires a mask — out of scope here.
3. **The pipeline self-normalizes references.** Refs over 1 MP are downscaled
   to ~1 MP and cropped to the processor's alignment multiple inside
   `__call__`. When `height`/`width` are not passed, output dims default to
   the **first** reference's (post-normalization) size.
4. **`Flux2Pipeline` (full Flux.2, family `flux2`) has the identical `image=`
   signature and semantics.** The wiring below enables it for free.
5. **NAG already anticipates this path.** `pipelines/nag_flux2.py` accepts and
   forwards `image=`, and when refs are present it skips NAG loudly (guard
   HF2-1: "NAG does not support the reference-image path in v1").

## Decision

**1. flux2klein reference conditioning threads through the generic text2img
call path — no qwen-edit-style fork.**

`_run_qwen_edit_refs` exists because qwen-edit runs its own manual denoising
loop. Klein does not: the stock `pipe.__call__` consumes refs directly, so the
implementation extends `_build_call_kwargs`'s flow with `image=[PIL, ...]`
rather than adding a second bespoke loop. Everything the generic path already
provides — sampler swap, sigma schedule (ADR-028), NAG gating, ^C pause,
output-format machinery, daemon delegation — keeps working unchanged.

`_resolve_ref_family_support` is generalized from returning
`(is_qwen_edit, warn)` to returning a **ref-execution kind**:
`"qwen-edit"` | `"flux2-native"` | `None` (drop path, unchanged semantics).
Family → kind: `qwen-edit` → `"qwen-edit"`; `flux2klein` and `flux2` →
`"flux2-native"`; everything else → `None`.

**2. `flux2` rides along.** ADR-035's routing table row "Other families:
img2img iff the auto-detected pipeline class natively accepts an image input"
already sanctions this, and finding 4 confirms `Flux2Pipeline` qualifies with
identical semantics. Gating it out would be an artificial family check the
table forbids. Both families are live-validatable on installed checkpoints:
`FLUX.2-klein-9B` / `FLUX.2-klein-base-9B` (`Flux2KleinPipeline`) and
`Flux.2-dev` (`Flux2Pipeline`).

**3. MODE: only `both` is valid; `vl`/`ref` are a hard error naming the
family — in both strict and lenient modes.**

This executes ADR-035 decision 2a's binding last paragraph verbatim. Klein has
no VL/Ref dual path — a reference is one thing. The error is hard even on the
interactive/lenient path because a `:vl` suffix is deliberately typed, never
stumbled into (the same reasoning 2a used for unknown-mode suffixes); honoring
warn-don't-block here would silently change what the model sees. Validation
happens inside the generalized `_resolve_ref_family_support`, which already
receives the parsed specs and the resolved family.

**4. No `--strength` flag is added.** ADR-035 decision 4 anticipated a
family-dependent img2img strength; finding 2 shows the native Klein/Flux.2
reference path has none — closeness to the reference is controlled by the
prompt and by which refs are supplied, not by a denoise fraction. Adding the
flag now would create a recorded-but-ignored parameter (the exact defect
ADR-028 existed to fix). It arrives if/when a family that consumes it lands
(e.g. Klein inpaint). ADR-035's changelog gets an entry pointing here when
this ADR is accepted.

**5. Output dimensions: explicit `--width`+`--height` are forwarded; otherwise
both are omitted and the pipeline derives dims from the first reference.**

Mirrors the qwen-edit rule (slice 3 / F6): the existing `ref_dims_explicit`
plumbing and the both-or-neither CLI warning apply unchanged. Asymmetry
recorded, not fought: qwen-edit derives from the **last** reference (node-pack
convention), Klein from the **first** (upstream diffusers convention). The
sidecar records the resolved dims either way, so replay is deterministic.

**6. NAG + refs: comfyless gates NAG off client-visibly before the call.**

nag_flux2's HF2-1 guard skips NAG at runtime, but it surfaces via the daemon's
stderr — invisible to a delegated client (invariant N1's failure mode).
`generate()` therefore pre-empts it: when the ref kind is `"flux2-native"` and
NAG would activate, NAG is deactivated with a warning appended to
`nag_warnings` (which rides the wire metadata and the sidecar), naming the
HF2-1 limitation. The in-pipeline guard stays as defense in depth for
standalone users.

**7. Provenance: same sidecar shape as qwen-edit.** Ingestion is refactored
into a shared helper (decode via `load_ref_image_capped` — the single decode
site — returning PIL list + provenance records `{path, mode, sha256,
applied}`); `_run_qwen_edit_refs` consumes it for its tensor conversion, the
flux2-native branch passes the PILs straight to `call_kwargs["image"]`.
`metadata["ref_images"]` and the replay-exclusion rule (`_SKIP_SIDECAR_KEYS`)
are already family-agnostic.

## Security

- **No `server.py` change.** The daemon is already family-agnostic for refs:
  containment (`_check_ref_paths` vs `ref_image_roots`), NUL defense, and wire
  validation (`validate_ref_image_entry`) all ran before `generate()` is
  reached, and the cache key needs no discriminator (finding 1). No Red Zone
  path (`scripts/git-policy/_red-zone-paths.sh`) is touched.
- Ingestion caps/allowlist are the audited ADR-035 slice-2/4 code, consumed
  unchanged. New code only routes an already-validated, already-capped PIL
  into a pipeline call.
- Review bar: **code-reviewer (Fable) required; security-auditor not
  required** unless implementation unexpectedly needs a `server.py` or
  containment change — in which case stop and re-scope first.

## Alternatives Rejected

- **A Klein-specific fork mirroring `_run_qwen_edit_refs`.** Duplicates the
  NAG/schedule/sampler/pause plumbing the generic path already provides, for a
  pipeline that needs none of it replaced. The qwen-edit fork exists because
  of its manual loop, not as a pattern.
- **`AutoPipelineForImage2Image.from_pipe` conversion.** Forbidden by ADR-035
  decision 3's invariant; also unnecessary — the detected class is already
  ref-capable.
- **Adding `--strength` now, warn-ignored.** A flag no family consumes is
  recorded-but-ignored schema surface; deferred until a consumer exists.
- **flux2klein-only gating (excluding `flux2`).** Contradicts ADR-035's
  "other families" row; the exclusion would be an extra check whose only
  effect is dropping refs a capable pipeline accepts.

## Deferred / Out of Scope

- **Klein inpaint** (`Flux2KleinInpaintPipeline`: mask + true denoise
  `strength`) — separate ADR when masks enter the schema.
- **Negative prompt / real CFG for Klein.** `Flux2KleinPipeline` supports
  `negative_prompt_embeds`, but comfyless's flux-family CFG routing never
  forwards negatives; changing that is orthogonal to refs.
- **ADR-035 deferred slice 5 (replay trust)** — still open, still not this
  work.
- **NAG-with-refs support** (would need nag_flux2 v2 handling reference
  tokens) — tracked by HF2-1's wording.

## Slice plan (one slice, after acceptance)

1. Generalize `_resolve_ref_family_support` → ref-execution kind + Klein MODE
   validation (decision 3); update its two call sites and drop-path tests.
2. Extract shared ref ingestion helper (decision 7); re-point
   `_run_qwen_edit_refs`.
3. Wire the `"flux2-native"` branch in `generate()`: `call_kwargs["image"]`,
   dims omission (decision 5), NAG pre-gate (decision 6), provenance,
   `edit_warnings` parity.
4. Tests in `test_ref_edit.py` (+ `test_params_schema.py` where routing is
   pinned): kind resolution per family, MODE rejection negatives, dims
   explicit/derived, NAG pre-gate warning riding metadata, provenance shape,
   drop path unchanged for still-unsupported families.
5. Live smoke (Grant): Klein 9B distilled and `Flux.2-dev`, 1-ref and
   multi-ref, daemon and in-process paths.

## Changelog

- 2026-07-22 — Drafted after signature/source investigation of diffusers
  0.39.0 `Flux2KleinPipeline` / `Flux2Pipeline` / `Flux2KleinInpaintPipeline`
  and the comfyless ADR-035 plumbing. Proposed to Grant; no code written.
- 2026-07-22 — Accepted by Grant (flux2 ride-along confirmed — `Flux.2-dev`
  is installed, `_class_name: Flux2Pipeline`; no-`--strength` verdict stands).
