# ADR-016: Hunyuan-Image 2.1 — Base + Refiner Chained Dispatch

**Date:** 2026-05-30
**Status:** accepted

---

## Context

Slice `hunyuan-support` (ADR-025, accepted 2026-05-17, amended 2026-05-24)
shipped `HunyuanImagePipeline` with auto-detection, distilled-guidance CFG
routing, and 2K-native dimension defaults. The 2026-05-24 amendment fixed
sub-native-resolution artifacts (1K → 2K family-defaults). The follow-on
tile-VAE-skip slice (Vision approved 2026-05-25; shipped 2026-05-28 across
commits `1a504ba`..`1a7ba27`) replaced unconditional `vae.enable_tiling()`
with a family-aware resolver, addressing tile-seam contributions to the
sky-banding pattern.

**A third artifact source remains:** base alone produces visible artifacts
(sail wrinkling, hull warping, foil-textured cloth) even at the documented
2K operating point with tiling off. Tencent designed Hunyuan-Image 2.1 as
a two-stage pipeline where the refiner explicitly "further enhances image
quality and clarity, while minimizing artifacts" (Tencent README
§Architecture). Empirical evidence retracted ADR-025 §3's original
"SDXL-style optional polish, edit-pipeline home" framing for the refiner
(per ADR-025's 2026-05-24 Changelog amendment): refiner is functionally
Cascade-coupled to the base — **both stages required for clean output**,
even though the data exchanged is images (structurally edit-shape) rather
than latents (structurally Cascade-shape). The architectural home is
therefore a comfyless dispatch fork analogous to `comfyless/cascade.py`
per ADR-010, NOT the edit-pipeline surface.

This ADR documents the dispatch shape, the wire/loader/cache machinery,
and the deliberately-narrow v1 scope.

The Vision (`docs/vision/slice-hunyuan-image-2-1-refiner.md`, approved
2026-05-25 after a review pass against an earlier draft) settled six
open questions during ADR drafting; this document captures the accepted
decisions for each.

Pre-flight confirmed on 2026-05-30 against diffusers 0.37.1:

- `HunyuanImageRefinerPipeline` imports cleanly on the existing pin (no
  diffusers bump required — ADR-013 §8 trailing-note does NOT trigger).
- Refiner pipeline class accepts `text_encoder=`, `tokenizer=`,
  `torch_dtype=`, `local_files_only=`, `variant=` via the standard
  diffusers `**kwargs` loader path.
- `HunyuanImageRefinerPipeline.__call__` exposes independent `prompt`,
  `negative_prompt`, `distilled_guidance_scale`, `num_inference_steps`,
  `image`, `height`, `width`, `latents`, `sigmas`, `generator`,
  `callback_on_step_end` (etc.).
- Refiner `model_index.json` carries one text encoder
  (`Qwen2_5_VLForConditionalGeneration`), one tokenizer (`Qwen2Tokenizer`),
  one transformer (`HunyuanImageTransformer2DModel`), one VAE
  (`AutoencoderKLHunyuanImageRefiner` — distinct class from base's
  `AutoencoderKLHunyuanImage`), one scheduler
  (`FlowMatchEulerDiscreteScheduler`). NO `text_encoder_2` slot, no T5,
  no ByT5, no `guider` / `ocr_guider`.
- Refiner weights present locally at
  `hf-local/HunyuanImage-2.1-Refiner-Diffusers` (operator-side download
  step completed 2026-05-28).

## Decision

### (a) Dispatch shape — opt-in only via `--refiner <path>`

The chained base+refiner flow activates ONLY when `model_family ==
"hunyuan-image"` AND `--refiner <path>` is set on the CLI (or the
ComfyUI node's `refiner_path` input is non-empty, or the daemon's
request payload carries a non-empty `refiner` field). The slice MUST
NOT derive, stat, glob, or otherwise search the filesystem from
`--model`'s parent or any sibling/derived path to discover a refiner
automatically.

**Rationale.** Path-derivation on caller-supplied input widens the
security surface the same way `lora_audit.py` had to defend against
(TOCTOU, containment escape, symlink interaction with the base path).
The Vision flagged this explicitly during the 2026-05-25 review pass
that revised the slice. Opt-in only keeps the path attack surface
identical to the existing `--model` shape; no new resolver code, no new
discovery code.

When `--refiner` is unset on a `hunyuan-image` run, the slice emits a
**loud stderr warning** ("hunyuan-image quality requires a refiner;
pass `--refiner <path>`; download with `huggingface-cli download
hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers`") and runs
base-only with zero exit code (matches the `feedback_warn_dont_block`
memory).

### (b) Module — `comfyless/hunyuan_chain.py` (NEW)

The chained-flow machinery lives in a NEW module
`comfyless/hunyuan_chain.py`, modeled architecturally on
`comfyless/cascade.py` (per ADR-010). The module exports:

- `load_refiner_pipeline(refiner_path, base_pipe, *, dtype, device,
  vae_tiling, ...)` — loads `HunyuanImageRefinerPipeline.from_pretrained`
  with shared text_encoder + tokenizer from `base_pipe` (see §(e));
  applies the family-aware VAE tiling resolution per ADR-025's
  tile-VAE-skip amendment.
- `build_refiner_call_kwargs(refiner_pipe, prompt, negative_prompt,
  refiner_steps, refiner_cfg, ...)` — builds the refiner-side
  `__call__` kwargs with `distilled_guidance_scale = refiner_cfg`
  (see §(f) CFG parity).
- `run_chain(base_pipe, refiner_pipe, base_kwargs, refiner_kwargs)`
  — runs base → PIL image → refiner; returns the final PIL image.
  The inter-stage transport is PIL via the refiner's `image=` param
  (see §(d) input shape rationale).

The module is imported lazily inside `comfyless/generate.py` only when
the chained branch is activated, so non-Hunyuan paths pay zero import
cost.

**Why a separate module, not extending `generate.py`.** The Cascade
precedent (ADR-010) established the convention: families whose load
shape is not "one repo / one pipeline" live in their own dispatch
forks. The Hunyuan chain is the same pattern in a different shape
(two pipelines from two repos chained as a single logical call). The
separation keeps `generate.py`'s argparse + dispatch logic readable and
makes future extension (e.g. a Hunyuan refiner-LoRA surface, a
reprompt-model chain) a local edit instead of more accretion on the
shared file.

### (c) Flag semantics — single enable, no `skip`/`auto`/envvar sentinels

`--refiner <path>` is the single enable:

- **Unset** (`--refiner` not passed) → base-only + the warn-don't-block
  stderr line per §(a).
- **Set to a resolvable path** → opt-in refiner stage.
- **Set to an empty string** (CLI argparse default `None` collapses to
  empty when sidecar replays an empty value) → treated as unset.

No `--refiner skip` sentinel, no `--refiner auto`, no `COMFYLESS_REFINER`
envvar override. The flag is one-dimensional: enable OR don't. This is
the same single-signal posture the Vision settled during review pass:
having both "explicit skip" and "implicit skip if unspecified" is double
signaling, and the absence signal is sufficient.

Path resolution uses the existing `resolve_hf_path(refiner_path,
allow_download=allow_hf_download)` — same machinery `--model`,
`--transformer`, `--vae`, `--te1`, `--te2` already use. Zero new
resolver code. Operators who pre-download (the most common workflow)
get the `return path` fast path at `eric_diffusion_utils.py:168`;
operators who pass HF repo IDs with `--allow-hf-download` get the
same opt-in download flow as `--model`.

### (d) Refiner input shape — PIL roundtrip via the refiner's `image=` param

Base's VAE class is `AutoencoderKLHunyuanImage`; refiner's is
`AutoencoderKLHunyuanImageRefiner`. The two VAEs are NOT
interchangeable — latents from base's VAE would not decode cleanly
through the refiner's VAE. The refiner's `__call__` exposes `latents=`
as a kwarg, but that parameter is for the refiner's own
forward-pass latent ladder (initial noise + scheduler state), not for
accepting base-VAE-encoded inputs.

The v1 inter-stage transport is therefore a **PIL roundtrip**:
1. Base pipeline `__call__` returns `images=[PIL.Image]` (existing
   `output_type="pil"` default).
2. The PIL image is passed to refiner pipeline via its `image=` kwarg.
3. Refiner re-encodes via its own VAE, runs its own forward pass, and
   returns the final PIL image.

The cost is one extra base-VAE-decode + one extra refiner-VAE-encode
per chained call (~2-3 seconds at 2K on the design hardware,
small relative to the ~2× transformer inference cost of the chain
itself). A future slice may investigate refiner-side latent passthrough
(would require both VAE classes to produce compatible latent
distributions, which they do not today by construction).

### (e) Asymmetric shared-text-encoder optimization

Refiner declares only `text_encoder` (Qwen2.5-VL) and `tokenizer`
(Qwen2Tokenizer). The base pipeline declares both `text_encoder`
(Qwen2.5-VL) and `text_encoder_2` (a separate T5/ByT5 stack) plus a
`guider` / `ocr_guider` pair. The shared optimization is **asymmetric**:

- **Shared:** `text_encoder` (Qwen2.5-VL, ~14 GB bf16) and `tokenizer`.
  After base loads, the refiner constructs via
  `HunyuanImageRefinerPipeline.from_pretrained(refiner_path,
  text_encoder=base.text_encoder, tokenizer=base.tokenizer,
  torch_dtype=dtype, local_files_only=True)`. Memory savings vs.
  loading a fresh Qwen2.5-VL on the refiner side: ~14 GB.
- **Not shared:** the base's T5/ByT5 stack (the refiner has no slot
  for it; its prompt pathway is single-encoder), the base's `guider`
  / `ocr_guider` (refiner has no guider machinery), the transformers
  (different weights), the VAEs (different classes).

Invariant locked at runtime: `id(base.text_encoder) ==
id(refiner.text_encoder)` after construction. Locked structurally by
the loader call site passing `text_encoder=` explicitly.

### (f) CFG routing — refiner-side branch mirrors base shape

The refiner is also guidance-distilled (same as base). The refiner-side
call-kwargs branch in `comfyless/hunyuan_chain.build_refiner_call_kwargs`
mirrors the base's distilled-guidance routing from ADR-025 §2:

```python
def build_refiner_call_kwargs(refiner_pipe, prompt, negative_prompt,
                              refiner_steps, refiner_cfg, ...):
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": refiner_steps,
        "distilled_guidance_scale": refiner_cfg,
        # ... base + refiner control inputs ...
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    return kwargs
```

**Key distinction from base:** the refiner reads its CFG from
`refiner_cfg`, NOT `cfg_scale`. The base call gets `cfg_scale`; the
refiner call gets `refiner_cfg`. Default `refiner_cfg` is **3.5**
(Tencent refiner README authoritative — diffusers signature default
is 3.25, but the README wins, same lesson as the 2K-mandatory
amendment in ADR-025's 2026-05-24 Changelog). Default `refiner_steps`
is **4** (Tencent refiner README).

`negative_prompt` is shared between stages — passing the same negative
to both base and refiner. Future slice may add `--refiner-negative` if
empirical evidence shows stage-specific negatives improve output;
out of scope for v1.

### (g) Refiner-side LoRA / scheduler / sampler / sigmas — pinned, no v1 surface

**LoRAs apply to base only.** The existing `--lora` machinery loads
into the base pipeline's transformer. The refiner has a separate
transformer with separate weights; no public refiner-side LoRA exists.
The slice MUST NOT call any LoRA loader against the refiner pipeline.
Locked by negative test: refiner transformer adapter count = 0 after
a chained run with base `--lora` set.

**Scheduler / sampler / sigmas pinned per-pipeline.** The refiner uses
its on-disk scheduler config (`FlowMatchEulerDiscreteScheduler` instance
from the refiner checkpoint). The slice MUST NOT mutate the refiner's
scheduler or apply base-side `--sampler` / `--sigmas` swaps to it. v1
ships no `--refiner-sampler` / `--refiner-sigmas` flags.

A future slice may add `--refiner-lora` / `--refiner-sampler` etc. if
a concrete use case lands. Out of scope for v1.

### (h) PNG metadata schema extension

The `comfyless` tEXt chunk on the output PNG gains four new keys
when a chained run produces it:

| Key | Type | Value |
|---|---|---|
| `pipeline` | string | `"base+refiner"` (literal, locked) |
| `refiner_path` | string | basename when MCP-callers (slice-1 invariant 12 — PNG-redaction); full path from CLI / daemon (per N29 regression guard) |
| `refiner_steps` | int | the effective `refiner_steps` for the run |
| `refiner_cfg` | float | the effective `refiner_cfg` for the run |

**Backward / forward compatibility:**
- Pre-refiner sidecar replayed by a post-refiner build (`--params
  prior.png`): no `pipeline` key → base-only branch (correct — pre-refiner
  generations were base-only).
- Post-refiner sidecar replayed by a pre-refiner build: unknown keys
  silently ignored (the existing schema-validator pass-through at
  `params_validation.py:251-254` already handles this).

Base-only runs continue to carry the existing single-stage metadata
shape unchanged — the new keys are absent (not present-and-empty).
This is structurally enforced by `build_model_metadata` only adding the
keys when the chained path actually ran.

### (i) IPC daemon wire-protocol extension

`comfyless/server.py:_handle_generate` gains an optional `refiner`
field in its request payload:

- Type: `_KIND_STR` (added to `_RUNTIME_KIND` in
  `comfyless/params_validation.py` — same defense-in-depth pattern as
  Step-3 tile-VAE-skip's `vae_tiling` addition).
- Default behavior on omit: empty / `None` → treated as unset →
  base-only flow with the warn-don't-block stderr line.
- Defense-in-depth: non-string values rejected at the IPC boundary as
  `ValidationError` (closes the same MEDIUM-#1 shape the
  tile-VAE-skip security-auditor identified for `vae_tiling`).

**Cache key composition** — the server's existing cache_key tuple
gains `req.get("refiner") or ""` as a trailing entry, paralleling the
`vae_tiling` entry added in commit `18fc68f`. Cache_key composition
becomes `(model, precision, device, transformer_path, vae_path,
text_encoder_path, text_encoder_2_path, vae_from_transformer,
offload_vae, attention_slicing, sequential_offload, vae_tiling,
refiner)`.

**Server-state slot for the refiner pipeline.** `server_state` gains
a `refiner_pipeline` slot (None when base-only). On cache_key mismatch
the server evicts BOTH `pipeline` and `refiner_pipeline` and reloads
both per the new cache_key. Single-slot eviction policy (option (a)
in Vision OQ-6) — matches the existing daemon shape; switching
`--refiner` mode incurs a ~80 GB reload, documented operator-facing.

Daemon behavior for clients that omit the `refiner` field is
byte-for-byte identical to today (additive field, not breaking).

### (j) Out of Red Zone — no `trust_remote_code` posture change

This slice does NOT introduce `trust_remote_code=True` anywhere.
`HunyuanImageRefinerPipeline` is a stock diffusers-shipped pipeline
class with no custom-code dependency. The Tencent reprompt model
(`HunYuanDenseV1ForCausalLM`, requires `trust_remote_code=True`) is
explicitly deferred to a separate Reprompt slice with its own ADR and
security-auditor pass per global §5/§12 (see "Deferred / Out of Scope"
§1 below). This keeps the refiner slice clean L2 (not Red Zone) and
lets the reprompt slice carry its own security artifact trail.

ADR-025 §3's 2026-05-24 amendment (which retracted the original
"refiner as SDXL-style optional polish" framing) motivated this slice's
existence; this ADR-016 closes the architectural design decision the
amendment opened.

## Alternatives Rejected

### A. Filesystem auto-discovery of sibling refiner directory

Rejected. Original Vision draft (pre-2026-05-25 review pass) had the
slice derive `<base-dir>-Refiner-Diffusers/` from `--model` and stat
it. Grant correctly flagged at review pass: this is the kind of
caller-supplied-path derivation `security-auditor` would catch (TOCTOU,
containment escape, symlink interaction with base path resolution).
And we never ran security-auditor on the Vision, despite it being a
§12 trigger. The opt-in-only posture removes the path-derivation
security surface entirely; the slice is smaller AND cleaner.

### B. `--refiner skip` sentinel + implicit "skip if unspecified"

Rejected. Having both an explicit skip sentinel AND an implicit
"skip if unspecified" is double signaling. Single signal wins:
absence = base-only-with-warning; presence = use the path.

### C. Hardcoding `refiner_steps` / `refiner_cfg` as constants

Rejected. Initial Vision draft proposed hardcoding them ("don't make
this complex"). But the refiner pipeline IS independently configurable
(confirmed via `inspect.signature` pre-flight), so the principled call
is to expose them — matches Grant's principle "if independent, leave
addressable." Defaults sourced from Tencent **refiner README**
(cfg=3.5, steps=4) not the diffusers signature default (cfg=3.25) —
README is authoritative, same lesson as the 2K-mandatory amendment.

### D. Bundling reprompt-model integration into this slice

Rejected. The reprompt model (`HunYuanDenseV1ForCausalLM`, ~7B params,
~14 GB bf16, custom tokenizer) requires `trust_remote_code=True` —
a first-time-in-codebase security posture change. Adopting
`trust_remote_code=True` as a first-class codebase capability is its
own ADR + security review per global §5/§12, and would push this
slice from L2 to L3 with `security-auditor` on every code-touching
commit. Splitting it out keeps the refiner slice clean L2 and lets the
reprompt slice carry its own security artifact trail.

### E. Refiner-side independent sampler / sigmas / scheduler-swap surface

Rejected for v1. The refiner uses its on-disk
`FlowMatchEulerDiscreteScheduler` config; no operator-facing reason to
override it has surfaced. Adding the flags speculatively would expand
the test surface without a use case. Add only when concrete need lands.

### F. Refiner-side LoRA surface (`--refiner-lora`) in v1

Rejected for v1. No public refiner-side LoRAs exist. The base LoRA
machinery cannot meaningfully apply to refiner weights (different
transformer). v1 ships zero refiner-side LoRA surface and locks the
invariant via negative test. Add if/when a use case emerges.

### G. Edit-pipeline integration (`Eric Diffusion Edit`-style refiner hookup)

Rejected per the ADR-025 2026-05-24 amendment's empirical retraction
of the "refiner is SDXL-style optional polish, edit-pipeline home"
framing. Refiner is functionally Cascade-coupled to base; the dispatch
fork shape matches that semantically. Edit-pipeline integration would
imply the refiner is an optional polish on a separately-baked image,
which it is not.

### H. Refiner-VAE latent passthrough (skip the PIL roundtrip)

Rejected for v1. Base VAE and refiner VAE are different classes by
design; their latent distributions are not interchangeable. Pre-flight
`__call__` introspection confirmed the refiner accepts `latents=`,
but that kwarg is for the refiner's own forward-pass latent ladder
(initial noise + scheduler state), not for base-VAE outputs. A future
slice may investigate if both VAEs converge on compatible latent
distributions; out of scope here.

### I. Bump diffusers to pick up a newer Hunyuan refiner pipeline

Rejected — not needed. Pre-flight on 2026-05-30 confirmed
`HunyuanImageRefinerPipeline` imports cleanly on the existing
`diffusers==0.37.1` pin with no API surface gaps for the chained-flow
shape this slice requires. The refiner `model_index.json`'s
`_diffusers_version` field reads `0.36.0.dev0`, but that's the metadata
of the saved pipeline at upload time, not a pinned requirement on the
runtime version. ADR-013 §8 trailing-note (which layers
`security-auditor` onto every code-touching commit when ML-stack pins
move) is explicitly NOT triggered by this slice — `security-auditor`
is required for Step 4's `comfyless/server.py` touch under the
existing CLAUDE.md "Review bar" rule, not as a §8 escalation.

### J. Wire the refiner through the MCP `generate` tool schema in this slice

Rejected. The MCP tool exposes operator-facing parameters; adding
refiner-shaped knobs would require its own ADR-011 §3d-style security
review for the LLM-agent surface. Out of scope per Vision Inv 12.
The MCP path keeps the existing `_load_pipeline` call shape with
`refiner` defaulting to unset (base-only + warning). The structural
test locks the deliberate omission (paralleling tile-VAE-skip's
mcp_server.py non-thread assertion).

## Deferred / Out of Scope

1. **Tencent reprompt model integration** (`HunYuanDenseV1ForCausalLM`).
   Requires `trust_remote_code=True` — first-class capability ADR
   pending. Operator workaround: use the existing `Eric Qwen Prompt
   Rewriter` API-LLM node with a Hunyuan-flavored system prompt, or
   hand-write a longer structured prompt.
2. **Refiner-side LoRA surface** (`--refiner-lora`).
3. **Refiner-side sampler / sigmas / scheduler-swap flags**.
4. **MCP tool refiner exposure** (separate ADR-011 §3d review).
5. **ComfyUI multistage / UltraGen refiner integration**. v1 adds
   chained refiner support to the unified `Eric Diffusion Generate`
   node; multistage workflow integration is its own slice if demand
   surfaces.
6. **ControlNet refiner variants**.
7. **HunyuanImage-3.0** continues to live with Eric's
   `Comfy_HunyuanImage3` ComfyUI nodes per Backlog 2026-05-17.
8. **`--refiner-negative` stage-specific negative prompts**. v1 shares
   `negative_prompt` between stages.
9. **Latent-passthrough between base and refiner** (per §H above).

## Changelog

- 2026-05-30 — proposed (initial draft, settles Vision OQ-1 through
  OQ-6 via pre-flight `inspect.signature` + `model_index.json`
  inspection + diffusers source review). Status: `proposed`.
- 2026-06-01 — Step-2 / Step-3 / Step-4 shipped on `hunyuan-support`
  branch. Step 2 (`46a48f6`): `comfyless/hunyuan_chain.py` + `--refiner`
  CLI flag + chain dispatch. Step 3 (`e948325`): ComfyUI Generate node
  `refiner_path` input + parity. Step 4 (this commit): daemon
  thread-through (`comfyless/server.py`) + `security-auditor` pass.
  Reviewers: `code-reviewer` (Opus) approved each step; `security-auditor`
  (Opus) found 1 CRITICAL + 1 HIGH on the initial Step-4 diff, both
  remediated in this commit (`refiner_path` added to `_PATH_FIELDS`
  null-byte rejection + `_check_paths` `--model-base` containment loop).
  Security artifact: `docs/security/review-hunyuan-refiner-server-2026-06-01.md`
  (also serves as the broader §12 IPC review for `comfyless/server.py`
  per CLAUDE.md "Review bar" debt — that debt is now **closed**).
- 2026-06-01 — wire-field naming reconciliation: §(i) text says the
  daemon-side wire field is `refiner` and the cache_key trailing entry
  reads `req.get("refiner")`. The implemented wire field is
  **`refiner_path`** — matches `SCHEMA_KIND["refiner_path"]` (the
  canonical schema key shipped in Step 1) and the existing
  `transformer_path` / `vae_path` / `text_encoder_path` /
  `text_encoder_2_path` `*_path` convention for path-bearing schema
  entries. `_delegate_to_server` (Step 1) and `_handle_generate`
  (Step 4) both use `refiner_path`. Treat the §(i) "`refiner`" text as
  shorthand; the implemented + tested name is `refiner_path`. No code
  change for this Changelog entry — corrects ADR text alignment with
  implementation.
- 2026-06-02 — refiner VAE tiling bug closed (`7e2f71b`): live 2K smoke
  (seed 42, 1920×1088) surfaced `RuntimeError` in
  `AutoencoderKLHunyuanImageRefiner._dcae_downsample_rearrange` during
  tiled encode. Root cause: the 32× DCAE refiner VAE has the same
  no-tile requirement as the base VAE; `hunyuan-image-refiner` was
  intentionally deferred to this slice in `_VAE_TILING_FAMILIES_DEFAULT_OFF`.
  Fix: added `"hunyuan-image-refiner"` to the set; refiner log now
  reads `Refiner VAE tiling disabled (vae_tiling=auto)`. Smoke re-run
  PASS: base 50 steps + refiner 4 steps, 38.9 s wall time, EXIT 0,
  image saved. `code-reviewer` (Opus): APPROVED. Unit gate: 244 tests,
  0 failures.
- 2026-06-02 — **slice fully closed.** All commits on `hunyuan-support`:
  Step 1 `7ca5b67` (ADR + schema/defaults), Step 2 `46a48f6`
  (`hunyuan_chain.py` + `--refiner` CLI flag), Step 3 `e948325`
  (ComfyUI node `refiner_path` input), Step 4 `138db5f` (IPC daemon
  parity + §12 security review), Step 5 `7e2f71b` (VAE-tiling fix),
  closure (this commit). Full unit gate: 1289/1289. Live 2K
  base+refiner smoke: PASS (2026-06-02).

- 2026-06-12 — **refiner resolution-collapse bug closed.** User report:
  every refined image rendered at 1024×1024 regardless of `--width`/
  `--height`; only 2048×2048 "looked right" (base-only runs). Root cause:
  `build_refiner_call_kwargs` never forwarded `height`/`width`, so
  `HunyuanImageRefinerPipeline.__call__` defaulted both to
  `default_sample_size * vae_scale_factor`. **Correction to the 2026-06-02
  entry's "32× DCAE refiner VAE" framing:** the diffusers refiner VAE
  (`AutoencoderKLHunyuanImageRefiner`) reports `spatial_compression_ratio
  = 16`, so the refiner default is `64 × 16 = 1024` — and the pipeline
  then *resized the base output down* to 1024×1024 via
  `image_processor.preprocess(image, height, width)` (refiner
  `pipeline_hunyuanimage_refiner.py:534-535, 568`). The 2026-06-02 live
  smoke missed this because it only asserted EXIT 0 + image-saved, never
  output dimensions. Fix: `build_refiner_call_kwargs` gains keyword-only
  `height`/`width`; `run_chain` passes `base_pil.height`/`base_pil.width`
  (the base output's actual dims — always a multiple of 32, so they
  satisfy the refiner's divisible-by-`vae_scale_factor*2`=32 check with no
  resample). `comfyless/hunyuan_chain.py` only; `run_chain` is the sole
  caller. `code-reviewer` (Opus): APPROVED. Unit gate: `test_hunyuan.py`
  248 tests (244 + 4 new resolution-forwarding assertions, Inv 6 + Inv 8),
  0 failures.

- 2026-07-11 — Ported onto main during the base+refiner re-apply. Internal
  references to the CFG-routing ADR updated from ADR-014 to ADR-025 (that
  ADR was renumbered to avoid a collision with main's lora-audit ADR-014).

## AI-Disclosure

Claude (Opus 4.7) authored; Grant reviewed.
Claude (Opus 4.8) authored the 2026-06-12 refiner resolution-collapse fix; Grant reviewed.
