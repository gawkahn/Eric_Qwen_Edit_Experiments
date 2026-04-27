# ADR-010: Stable Cascade as a JSON-Config Family in Comfyless

**Date:** 2026-04-26
**Status:** accepted
**AI-Disclosure:** Claude (Opus 4.7) authored; Grant reviewed.

---

## Context

Stable Cascade (Würstchen v3, Stability AI) is the last family planned
for comfyless coverage in this round. Investigation against the SAI
repos and the user's downloaded files surfaced four facts that, taken
together, make it structurally different from every other family the
loader handles today:

1. **Three weight files, not one.** Cascade decomposes into Stage C
   (prior, text → 24×24 latents), Stage B (decoder, latents →
   intermediate features), and Stage A (Paella VQ-VAE, features →
   pixels). Inference runs C → B → A. There is no single-checkpoint
   load path that wires the full stack.
2. **No combined pipeline class in practice.** `diffusers` exposes
   `StableCascadeCombinedPipeline` but it expects every component
   inside one repo with `prior_*` prefixes. Neither
   `stabilityai/stable-cascade` nor `stabilityai/stable-cascade-prior`
   is laid out that way. The model-card example in SAI's own README
   instantiates `StableCascadePriorPipeline` and
   `StableCascadeDecoderPipeline` separately and chains their outputs.
3. **Auto-detect is unsafe.** Both SAI repos carry a `model_index.json`
   `_class_name` pointing at half the stack (`StableCascadeDecoderPipeline`
   on the main repo, `StableCascadePriorPipeline` on the prior repo).
   The current `detect_pipeline_class` would happily build a half
   pipeline and silently produce broken output. Trusting the auto-detect
   contract requires adding negative logic specifically to refuse it
   for this family.
4. **Real swap candidates exist but in a different format.** The user
   has alternative Stage B and Stage C weights at
   `~/projects/ai-lab/ai-base/models/comfyui/models/checkpoints/StableCascade/`
   (r35*, altcascade_v20). All five are ComfyUI all-in-one bundles —
   diffusion weights prefixed `model.diffusion_model.*`, plus bundled
   `vae.*` and `text_encoder.*` copies. None load via
   `StableCascadeUNet.from_single_file()` as-is.

The **first** comfyless families (Flux, Qwen, Chroma, SDXL, Pony, etc.)
all share a "one repo or one safetensors → one pipeline class" load
shape. That assumption is baked into `_FAMILY_PATTERNS`,
`COMFYLESS_SCHEMA`, the override-flag set (`--transformer`, `--vae`,
`--te1`, `--te2`), and the `--iterate` axis vocabulary. Extending each
of those for a single one-off family that violates every assumption is
disproportionate.

## Decision

Treat Stable Cascade as a **special-case family** with its own
top-level dispatch path, its own JSON config format, and no integration
into the unified schema or override-flag set. Specifically:

### 1. Top-level dispatch fork on a literal sentinel

```
comfyless --model stablecascade <config.json> [config2.json] [config3.json] ...
```

The literal token `stablecascade` is the family signal. `comfyless`
does not read `model_index.json` for this family, does not extend
`_FAMILY_PATTERNS`, does not call `detect_pipeline_class`. The fork
happens at the start of run-dispatch; everything cascade-specific lives
in a new `comfyless/cascade.py` module.

### 2. Multiple positional configs auto-iterate

Passing more than one config triggers iteration: process configs
left-to-right, dump-and-reload between configs (no diff-cache logic).
Within a single config, `--batch N` runs N iterations against the same
loaded pipelines before disposal. `--limit M` caps the total run count
across all configs (matches ADR-008 semantics for the existing
`--iterate` axis).

This subsumes iteration-over-cascade-variants without extending the
ADR-008 axis vocabulary. Each config is a complete, self-contained
representation; we never patch one config's fields into another.

### 3. JSON schema (cascade-only, not part of `COMFYLESS_SCHEMA`)

```jsonc
{
  // Stage paths — each accepts:
  //   - absolute path to a Würstchen-native single-file safetensors, OR
  //   - absolute path to a diffusers tree directory.
  // ComfyUI all-in-one bundles must be converted first via
  // convert_cascade_comfyui.py — see docs/comfyless-stable-cascade.md.
  "stage_c": "/abs/path/stage_c_bf16.safetensors",
  "stage_b": "/abs/path/stage_b_bf16.safetensors",
  "stage_a": "/abs/path/stable_cascade/vqgan",   // optional; defaults to scaffolding repo
  // Repo providing text_encoder/, tokenizer/, scheduler/.
  // Defaults to "stabilityai/stable-cascade" via resolve_hf_path.
  "scaffolding_repo": "stabilityai/stable-cascade",
  // Per-stage dtype defaults match the SAI model card.
  "prior_dtype":   "bf16",
  "decoder_dtype": "fp16",
  "vae_dtype":     "fp32",
  // Two denoising loops, two sets of params. SAI defaults.
  "prior_steps":     20,
  "prior_cfg_scale": 4.0,
  "decoder_steps":   10,
  "decoder_cfg_scale": 0.0,
  // Output resolution. Aligned to 128px (round-down + warn on misalignment).
  "width":  1024,
  "height": 1024
}
```

All fields except `stage_c` and `stage_b` are optional. Defaults are
applied at JSON-load time by `_validate_cascade_config`.

### 4. Per-stage paths accept file or directory

`StableCascadeUNet.from_single_file(path)` for safetensors files;
`StableCascadeUNet.from_pretrained(path, ...)` for diffusers tree
directories. The loader branches on `os.path.isfile` vs `os.path.isdir`;
no filename sniffing, no autodetect of "what stage is this." The JSON
field name (`stage_c` vs `stage_b` vs `stage_a`) is the only signal
about which stage a path belongs to.

### 5. Build prior pipeline from the main repo by default

`StableCascadePriorPipeline.__init__` declares `feature_extractor` and
`image_encoder` as `Optional[...]` defaulting to `None`. Both are only
used when the caller passes `image=...` to the prior for image-variation
generation, which we do not support in v1. This means the
`stabilityai/stable-cascade-prior` repo is **not required** for
text-to-image.

The default load path constructs both pipelines from the main
`stabilityai/stable-cascade` repo:

- `text_encoder`, `tokenizer`, `scheduler` from the scaffolding repo's
  tree directories.
- Stage C weights from the path in `stage_c` (default: scaffolding
  repo's `stage_c_bf16.safetensors`).
- Stage B weights from the path in `stage_b` (default: scaffolding
  repo's `stage_b_bf16.safetensors`).
- Stage A weights from the path in `stage_a` (default: scaffolding
  repo's `vqgan/` tree).
- `feature_extractor=None`, `image_encoder=None`.

The user retains the option of pointing `stage_c` at the `-prior`
repo's `prior/` or `prior_lite/` tree directory if they explicitly
want diffusers-tree format for the prior; nothing in the design
prevents it.

### 6. 128px alignment with round-down + warn

Cascade's `resolution_multiple = 42.67` (compression factor 42×8 = 128
in image space) requires width and height to be multiples of 128. The
existing `_align_dim` helper in `comfyless/generate.py` rounds down to
`_ALIGN = 32` and warns. We mirror that pattern with a local
`_align_cascade_dim` rounding to 128, same warning prefix. No hard
fail — matches the user's standing "warn, don't block on user-initiated
footguns" preference and the rest of the codebase.

### 7. Lite variants permitted, documented as discouraged

`stage_c_lite*.safetensors` and `stage_b_lite*.safetensors` will load
fine if the user points the JSON at them. We do not filter, warn, or
error on lite filenames. The user-facing reference doc
(`docs/comfyless-stable-cascade.md`) recommends full variants and
states the policy explicitly.

### 8. ComfyUI all-in-one alt files via one-time conversion

The user's r35* and altcascade_v20 alt-stage files are ComfyUI all-in-one
bundles. They cannot be loaded directly. We ship a separate utility,
`convert_cascade_comfyui.py`, that:

1. Reads the input safetensors header.
2. Detects the `model.diffusion_model.*` prefix.
3. Extracts diffusion-only keys, strips the prefix.
4. Writes a clean Würstchen-native single-file safetensors.
5. Performs a strict-load smoke test against `StableCascadeUNet`
   instantiated from the SAI prior config (for stage_c) or decoder
   config (for stage_b) and reports any missing/unexpected keys
   honestly. If the smoke test fails, the conversion artifact is
   retained but the tool exits non-zero so the user knows the file is
   suspect.

This is a one-shot operation per alt file. The loader stays unaware of
ComfyUI bundle format; the conversion is a prerequisite the user runs
once.

The conversion utility is also the first concrete instance of the
broader **parallel `-comfyless` library** project (Backlog, Queued,
2026-04-26): a sibling-file convention where `<name>-comfyless.<ext>`
is a pre-validated cleanly-loading variant, used by the loader if
present and by the LLM tool catalog as the only weights it advertises
to the agent. Not in scope for this slice; the cascade conversion tool
just happens to be the first concrete artifact of that pattern.

### 9. LoRA support out of scope

Stable Cascade LoRA exists in the diffusers ecosystem (attaches to the
prior) but the community ecosystem is shallow (~40 LoRAs vs. thousands
for SDXL/Flux). Out for v1; revisit if empirical use surfaces demand.

## Alternatives Rejected

### A. Extend `COMFYLESS_SCHEMA` and add `--prior-model`, `--prior-repo`, `--decoder-repo`, `--vae-repo` flags

Rejected. This was the first design considered. It would have meant:
six new schema keys (`prior_steps`, `prior_cfg_scale`, `decoder_steps`,
`decoder_cfg_scale`, `prior_dtype`, `decoder_dtype`); five new override
flags; an entry in `FAMILY_DEFAULTS` (ADR-009); a new `--iterate
prior_model` axis (ADR-008); cache-key extension to include both
prior and decoder paths; a `_FAMILY_PATTERNS` extension with three
mappings to handle the misleading `_class_name` values; a branch in
`_load_pipeline` and `_build_call_kwargs`. The cost is high and the
design space being served is small — the user has identified at most
two alt Stage B candidates and three alt Stage C candidates. Building
infrastructure for "any user might swap any of 50 things" when the real
catalog is "five files for one user" is the wrong tradeoff.

### B. Use `StableCascadeCombinedPipeline.from_pretrained`

Rejected. Requires every component co-located in a single repo with
`prior_*` prefixes; neither SAI repo is laid out that way, and we
don't want to maintain a side repo of our own. The combined pipeline
also forces `bf16` end-to-end (refusing fp16 for the decoder) which
contradicts SAI's own model-card recommendation of `bf16` prior +
`fp16` decoder.

### C. Require both the `-prior` and the main repo

Rejected. Verified by inspecting `StableCascadePriorPipeline.__init__`:
`feature_extractor` and `image_encoder` default to `None` and are only
needed for image-variation generation (passing `image=` to the prior).
The main repo carries `text_encoder/`, `tokenizer/`, `scheduler/` and
all needed flat-file weights. Forcing a 4 GB second download for
machinery we never use is friction without value.

### D. Support ComfyUI all-in-one bundle format natively in the loader

Rejected. Detecting `model.diffusion_model.*` prefixes, extracting,
remapping, and instantiating `StableCascadeUNet` from a synthesized
config is ~50–80 lines of loader code with non-trivial verification
risk: the user's r35MC bundle has 1550 diffusion tensors vs SAI native's
1294, and we don't yet know whether `load_state_dict(strict=False)`
silently drops or actually loads cleanly. A one-time conversion script
surfaces the same risk *once* during conversion, not *every load*, and
does not entangle the runtime loader with the bundle format. If the
parallel `-comfyless` library project (Backlog) eventually replaces
all ComfyUI-bundle files with converted siblings, the loader never
needs to learn the format.

### E. Drop Cascade entirely

Considered seriously and rejected. The user's stated motivation —
exploring Stability's training tradition's response to gibberish prompts
at the largest available scale — survives the complexity. The blob-mode
JSON design recovers most of the simplicity. With the conversion utility
landing the alt-file ergonomics, daily-use cost is low.

## Deferred / Out of Scope

- LoRA support for the prior or decoder.
- Image-variation conditioning (`image=` argument to the prior).
- Stage A weight swapping UI (the field exists but no published Stage A
  variants are known to be worth swapping).
- ControlNet variants — the SAI main repo includes a `controlnet/`
  directory; not in scope.
- Lite-variant filename detection or warning; permissive, doc-only
  policy.
- Integration with the existing `--iterate` axis vocabulary; positional
  configs handle iteration without adding a new axis.
- The broader parallel `-comfyless` library project; cascade conversion
  is its first concrete output but the LoRA/transformer sweeps and the
  loader sibling-file logic are separate slices (Backlog, Queued).

## Changelog

- **2026-04-26 (initial)**: Decision recorded before any code is
  written. Design covers dispatch fork, JSON schema, per-stage paths
  and dtypes, 128px alignment, lite policy, prior-pipeline construction
  from the main repo, ComfyUI bundle handling via separate conversion
  utility, and alternatives explicitly rejected. Implementation order:
  this ADR → `docs/comfyless-stable-cascade.md` → `convert_cascade_comfyui.py`
  → `comfyless/cascade.py` + dispatch fork → tests → `code-reviewer`
  (Opus) → commit.

- **2026-04-26 (amendment)**: Original decision said `--iterate` was
  rejected wholesale ("Cascade iterates via positional configs"). That
  reasoning held for *topology* iteration but starved the legitimate
  prompt/seed sweep use case — the user's documented 1000-prompt ×
  multi-model word-salad workflow needs *one* config and *N* prompts,
  not the inverse. Loosened to accept the `--iterate prompt` and
  `--iterate seed` axes only; other axes (`cfg_scale`, `model`,
  `transformer`, etc.) remain rejected as JSON-config concerns. Plan
  expansion order: cfg (outer) × batch × prompt × seed (innermost), so
  pipelines load once per config and are reused across the sweep.
  `--limit` and `--max-iterations` apply to the post-Cartesian total.
  Decision: §2 still holds for topology; §1's "no Cartesian-product
  machinery" is narrowed to "no cfg-axis Cartesian," prompt/seed
  Cartesian is a small bounded extension.
