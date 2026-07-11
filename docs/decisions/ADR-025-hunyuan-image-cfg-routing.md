# ADR-025: Hunyuan-Image 2.1 — CFG Routing, Family Detection, Refiner Isolation

**Date:** 2026-05-17
**Status:** accepted

---

## Context

Slice `hunyuan-support` (`docs/vision/slice-hunyuan-image-2-1.md`, approved
2026-05-16, re-baselined 2026-05-17) adds `HunyuanImagePipeline` (diffusers
0.37.1, Tencent's Hunyuan-Image 2.1) as a recognized family in both the
ComfyUI generic loader/generate path and the comfyless CLI path.

The slice is structurally a routine "new family" addition — ADR-003 already
defined the GEN_PIPELINE dict shape and the `_build_call_kwargs` routing
table that lets new diffusers families plug in without invasive changes. The
reason this slice warrants its own ADR is that **Hunyuan-Image introduces
the project's third CFG routing shape**, distinct from the two existing
ones, and that decision sits on a §12 trigger (per-family routing has
appeared in security-review scopes before; pinning the third shape
explicitly now avoids "what is this branch?" confusion in future reviewers).

Existing CFG routing shapes (ADR-003):

| Family group | Pipeline kwarg used | Negative prompt | Forward passes |
|---|---|---|---|
| `qwen-image` | `true_cfg_scale` | yes | 2× per step |
| `flux` / `flux2` / `flux2klein` / `chroma` | `guidance_scale` (guidance-embedded) | no (ignored) | 1× per step |
| `sdxl` / `sd3` / `sd1` / `auraflow` / `zimage` | `guidance_scale` (classical CFG) | yes | 2× per step |

Hunyuan-Image 2.1 is a **guidance-distilled** model (the
`HunyuanImagePipeline.__call__` docstring is explicit: "guidance distilled
models take the guidance scale directly as an input parameter during
forward pass"). The kwarg name is `distilled_guidance_scale`, NOT
`guidance_scale`. The signature also accepts `negative_prompt` — which the
docstring says is "Ignored when not using guidance" — i.e., the pipeline
itself decides whether to apply negatives. The signature default for
`distilled_guidance_scale` is `3.25`; `num_inference_steps` default is
`50`. (Note: the diffusers docstring for `distilled_guidance_scale` says
"defaults to None"; the actual `__call__` signature default is `3.25`.
Verified by `inspect.signature` at decision time — the docstring is stale
relative to the signature.)

A separate consideration: diffusers 0.37.1 also ships
`HunyuanImageRefinerPipeline` (companion refiner) and `HunyuanDiTPipeline`
(older HunyuanDiT). Both are explicitly out of scope to *implement* in this
slice (Vision §"Out of scope"), but the auto-detection pattern list
(`_FAMILY_PATTERNS` in `nodes/eric_diffusion_utils.py`) is order-sensitive
(first match wins after lowercase + strip), so the entry for the base
pipeline must be placed deliberately to avoid the refiner string
accidentally collapsing into `"hunyuan-image"` and dragging the refiner
along the base-pipeline routing path.

## Decision

### 1. New family string: `"hunyuan-image"`

Match the existing kebab-case convention (`qwen-image`, `qwen-edit`,
`flux2klein`). Reserve `"hunyuan-image-refiner"` for the companion (see
§3). The bare string `"hunyuan"` is NOT a family — leaving the namespace
open if HunyuanDiT or Hunyuan-Video families ever land in this repo.

### 2. CFG routing — third explicit branch in `_build_call_kwargs`

Add a `hunyuan-image` branch in BOTH `nodes/eric_diffusion_generate.py:_build_call_kwargs`
AND `comfyless/generate.py:_build_call_kwargs`. The two copies stay in
lockstep until the runtime-core cluster (Backlog → Queued) consolidates
them; that consolidation is its own future slice.

Shape:

```python
if model_family == "hunyuan-image":
    # Hunyuan-Image 2.1: guidance-distilled — one forward pass per step.
    # distilled_guidance_scale is the documented call kwarg (NOT guidance_scale).
    # negative_prompt is accepted by the pipeline; docstring says "Ignored
    # when not using guidance" — let the pipeline decide.
    kwargs = {**base, "distilled_guidance_scale": cfg_scale}
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    return kwargs
```

Notes:

- `cfg_scale` is the canonical `COMFYLESS_SCHEMA` key (also the ComfyUI
  generate node input). It is mapped at call-build time to the pipeline's
  `distilled_guidance_scale` argument. **No new canonical schema key.**
  This matches how Flux maps `cfg_scale → guidance_scale` and how
  Qwen-Image maps `cfg_scale → true_cfg_scale` (though Qwen also has the
  `true_cfg_scale` schema key for explicit override; Hunyuan does NOT get
  an analogous explicit-override key in this slice — see §6).
- `max_sequence_length` is NOT in the Hunyuan call signature; the branch
  does not pass it. (Verified via `inspect.signature` at decision time.)
- The branch is hit only when `model_family == "hunyuan-image"`. The
  refiner (`"hunyuan-image-refiner"`, §3) falls through to the introspection
  fallback path until a future slice gives it explicit routing.

### 3. Refiner isolation — pattern order matters

`HunyuanImageRefinerPipeline` is out of scope to implement, but `_FAMILY_PATTERNS`
ordering must isolate it so a refiner-loaded GEN_PIPELINE doesn't
accidentally route through the base Hunyuan CFG branch. Add **two**
entries in this order:

```python
_FAMILY_PATTERNS = [
    ...existing entries...
    ("hunyuanimagerefiner", "hunyuan-image-refiner"),  # MUST precede "hunyuanimage"
    ("hunyuanimage",        "hunyuan-image"),
    ...
]
```

A refiner pipeline loaded today resolves `model_family ==
"hunyuan-image-refiner"`, which has no CFG-routing branch (no FAMILY_DEFAULTS
row either) — so it falls through to the introspection fallback. That's
graceful: introspection passes any of `distilled_guidance_scale`,
`negative_prompt`, etc. that match the refiner's `__call__` signature. The
refiner isn't *supported* (no tuned defaults, no explicit branch, untested
end-to-end), but it isn't silently mis-routed either. A future slice can
add explicit refiner support without renaming the family string.

Audit of all `Hunyuan*Pipeline` classes exported by diffusers 0.37.1
(verified via `diffusers/__init__.py` against `code-reviewer` round-1
2026-05-17): the proposed two-entry insertion does not collapse
`HunyuanDiTPipeline`, `HunyuanDiTControlNetPipeline`, `HunyuanDiTPAGPipeline`,
`HunyuanVideoPipeline`, `HunyuanVideo15Pipeline`, `HunyuanVideoImageToVideoPipeline`,
`HunyuanVideo15ImageToVideoPipeline`, `HunyuanSkyreelsImageToVideoPipeline`,
or `HunyuanVideoFramepackPipeline` into either Hunyuan-Image family
slot — the `dit` / `video` / `video15` / `skyreels` / `videoframepack`
infixes break the `hunyuanimage` substring after the
`lower() + replace("_","") + replace("-","")` normalization. Future
diffusers releases that ship new `Hunyuan*` pipeline classes should
re-run this audit as a one-line check in an ADR amendment.

**Important caveat on the refiner family slot — it is defensive.** Per
the `HunyuanImageRefinerPipeline.__call__` inspection (signature confirmed
via `inspect.signature` at decision time): the refiner is structurally an
edit / image-to-image pipeline (consumes a complete image via `image:
PipelineImageInput | None`, runs ~4 default denoising steps, outputs a
refined image, uses a distinct `AutoencoderKLHunyuanImageRefiner` VAE,
distributed as a separate HF model). The family-pattern slot reserved
here exists only to prevent accidental misroute of a refiner-loaded
checkpoint through the base CFG branch via substring overlap.

**Original framing — partly retracted 2026-05-24 amendment.** This ADR
initially framed the refiner as "SDXL-style optional polish, edit-pipeline
home." Empirical evidence from the Step 5 live smoke (Grant's first
generation at 1024×1024) plus an external diagnosis of the resulting
artifacts revealed two things: (a) the 1024 default was below the model's
2K-native operating point (separately addressed by the 2026-05-24 amendment
to §4), and (b) **even at 2K the base alone produces visibly artifacted
output for which the refiner is the documented remedy** — making the
refiner functionally part of "what produces a usable Hunyuan-Image 2.1
generation," not an optional after-pass. The data exchanged is still
images (structurally edit-shape), but the *coupling* of base + refiner is
Cascade-pattern (both stages required for the product). The future-refiner
work therefore belongs as a **comfyless dispatch fork** analogous to
`comfyless/cascade.py` (per ADR-010), not on the edit-pipeline surface.
The fork can either auto-chain base + refiner or expose them as a
two-step explicit pipeline; that's its own ADR's decision. The
`hunyuan-image-refiner` family slot reserved here remains defensive (it
still prevents misroute) but no longer implies the architectural home;
the refiner Vision + ADR is queued as the immediate next slice after this
2026-05-24 amendment, per Grant's direction.

### 4. Family-defaults row in `comfyless/family_defaults.py`

Add one alphabetically-ordered entry. **Amended 2026-05-24** to include
2K-native dimension defaults — see "Original §4 entry was insufficient"
note below and the 2026-05-24 Changelog amendment:

```python
# ── hunyuan-image (Hunyuan-Image 2.1) ───────────────────────────────
"hunyuan-image": {
    "cfg_scale": 3.25,
    "steps":     50,
    "width":     2048,
    "height":    2048,
},
```

`cfg_scale`/`steps` match the `HunyuanImagePipeline.__call__` signature
defaults + the Tencent model card. `width`/`height` are mandatory 2K per
Tencent README: "HunyuanImage-2.1 only supports 2K image generation …
Generating images with 1K resolution will result in artifacts." The 32×
spatial-compression VAE is trained on 64×64 latents → 2048-decoded
images; sub-2K renders are out-of-distribution. Documented aspect
buckets: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3. This row is the first
FAMILY_DEFAULTS entry to carry `width`/`height` (other families let the
caller choose); the overlay applier already supports it because the
applier walks any key that exists in `COMFYLESS_SCHEMA`, and
`width`/`height` are canonical schema keys.

`true_cfg_scale` is intentionally NOT in this row — Hunyuan does not use
the double-pass CFG path; setting it would be meaningless. The overlay
applier silently ignores unknown keys but it would still be a source of
confusion to readers.

**Original §4 entry was insufficient.** The first version of this row
shipped with only `{"cfg_scale": 3.25, "steps": 50}` (the 2026-05-17 ADR
draft). Step 5's live smoke at the schema-default 1024×1024 produced
visibly artifacted output (sky banding, foil-textured sails, hull
geometric distortion); external diagnosis identified sub-2K rendering as
the dominant cause. The 2026-05-24 amendment adds the dimension defaults.
See the 2026-05-24 Changelog entry.

Per ADR-009's precedence ladder, this row sits **above** `COMFYLESS_SCHEMA`
defaults (cfg_scale=3.5, steps=28) and **below** explicit CLI flags /
sidecar values / `--iterate` axes. A bare
`python -m comfyless.generate --model <hunyuan-2.1>` resolves to
`cfg_scale=3.25, steps=50`. An explicit `--cfg-scale 5.0` overrides to 5.0.

### 5. `negative_prompt` semantics — forward, let the pipeline decide

The pipeline's docstring says `negative_prompt` is "Ignored when not using
guidance." That's a pipeline-internal decision; the call-build layer
should not second-guess it. When the user supplies a non-empty
`negative_prompt`, the branch forwards it; otherwise the kwarg is omitted.
This matches Qwen-Image and SDXL/SD3/AuraFlow branches and avoids the Flux
pattern (silently dropping all negatives unconditionally) that some users
have found surprising.

### 6. Deferred — no explicit `distilled_guidance_scale` canonical key

ADR-009's precedence design supports family-specific canonical keys
(`true_cfg_scale` exists alongside `cfg_scale` for Qwen). The slice does
NOT add `distilled_guidance_scale` as a third canonical key. Rationale:

- Adding a canonical key means editing `comfyless/params_schema.py`
  (`SCHEMA_KIND`, `_FIELD_DEFAULTS`, `_CLI_TO_CANONICAL`), which is a
  much broader change with 135 schema-tests to walk, AND it introduces a
  comfyless-only knob with no ComfyUI-side analog (the ComfyUI generate
  node has a single `cfg_scale` input).
- The only motivating use case would be "set true_cfg AND distilled_cfg in
  the same sidecar JSON for a multi-family `--iterate` sweep." That's
  marginal; the iterate axis already supports `cfg_scale`.
- If a future need surfaces, this is a clean follow-up slice (add the key,
  extend the routing branch to honor an explicit override before falling
  back to `cfg_scale`).

### 7. Reviewer plan unchanged from Vision

- `code-reviewer` (Opus, `model: "opus"` at invocation per global §5A and
  the broken-frontmatter workaround) after each non-trivial slice step,
  before commit.
- `security-auditor` NOT invoked for this slice. Justification: no Red
  Zone surface (CLAUDE.md "Review bar"), no `resolve_hf_path` /
  `_run_json_mode` / `comfyless/server.py` touch, no IPC change, no new
  caller-supplied path widening, no ML-stack pin movement. ADR-013 §8's
  trailing-note rule ("a future slice that DOES move a pin layers
  `security-auditor` onto each code-touching commit") does not trigger —
  `from diffusers import HunyuanImagePipeline` against the existing
  `diffusers==0.37.1` pin is consumption, not a bump.
- **STOP-condition during implementation:** if step 2 (family-pattern +
  tests) or any later step surfaces an unexpected caller-supplied
  component path — e.g. a Hunyuan-specific component slot not covered by
  the existing `transformer_path` / `vae_path` / `text_encoder_*`
  surface — STOP and re-evaluate per Vision §Reviewer-plan. A note for
  future readers: HunyuanImageRefinerPipeline's `__init__` (out of scope
  for this slice but reserved as a family slot per §3) takes a `guider:
  AdaptiveProjectedMixGuidance | None` component that the existing
  loader has never seen; if the Hunyuan refiner slice ever lands, it
  must re-evaluate whether the `read_guidance_embeds`-style
  capability-flag pattern extends or whether a new GEN_PIPELINE key is
  warranted.

### 8. Test gate

CPU-only `test_hunyuan.py` covers all five Vision invariants with positive
and negative cases (see Vision §"Proof hooks" for the full list). Full
8-suite regression (`./.venv/bin/python3`, 850+N tests) at slice closure.
Live GPU smoke deferred until Hunyuan-Image 2.1 weights are on disk.

## Alternatives Rejected

### A. Route Hunyuan through the existing `flux` branch (treat `distilled_guidance_scale` and `guidance_scale` as interchangeable)

Rejected. The kwarg names differ; passing `guidance_scale=3.25` to
`HunyuanImagePipeline.__call__` would either silently get ignored (going
through `**kwargs`) or raise a `TypeError` depending on diffusers'
signature handling. Either way, the routing would be wrong. Distinct
kwarg = distinct branch.

### B. Route via signature introspection (skip the explicit branch, rely on the unknown-family fallback in `_build_call_kwargs`)

Rejected — the introspection path doesn't merely route to the wrong
kwarg, it fails hard. Detailed chain (per `code-reviewer` round-1 audit
2026-05-17):

1. Real Hunyuan-Image 2.1 checkpoints carry `transformer.config.guidance_embeds = True`
   in their `config.json`. The pipeline itself enforces this at runtime —
   `pipeline_hunyuanimage.py:727-728` raises
   `ValueError("distilled_guidance_scale is required for guidance-distilled model.")`
   when `self.transformer.config.guidance_embeds` is truthy and the kwarg
   is unset. The bare-`__init__` default of `False` on the transformer
   model class is irrelevant for a loaded checkpoint.
2. With `guidance_embeds=True`, `read_guidance_embeds()` returns `True`,
   and the GEN_PIPELINE dict carries `guidance_embeds=True`.
3. The introspection fallback in `_build_call_kwargs` (`comfyless/generate.py`
   ~line 612; `nodes/eric_diffusion_generate.py` ~line 245) reads that
   flag and proposes `guidance_scale = cfg_scale`. Hunyuan's
   `__call__` signature does NOT accept `guidance_scale` — the
   `accepted = set(sig.parameters.keys())` filter drops the candidate,
   and `distilled_guidance_scale` is also dropped because the candidates
   dict doesn't even propose it (introspection only knows about
   `guidance_scale` and `true_cfg_scale`).
4. The pipeline is called with `distilled_guidance_scale` unset →
   `ValueError` from step 1.

Net effect: introspection produces a hard `ValueError` at first call —
not silent degradation, but a noisy crash. That's *better* than silent
degradation, but still a worse user experience than the explicit branch.
Explicit family routing is the only shape that runs.

### C. Add `distilled_guidance_scale` as a third canonical schema key now

Rejected (see §6 above). Marginal use case; expansive blast radius for a
1-line gain.

### D. Collapse refiner into the base `"hunyuan-image"` family string (share routing)

Rejected (see §3 above). The refiner isn't *tested* against the base
branch; treating it as a runnable instance of the same family would imply
a tested guarantee we don't have. Isolating the refiner string keeps the
"untested = falls back to introspection" posture honest until a slice
explicitly adopts it.

### E. Reuse `read_guidance_embeds`'s GEN_PIPELINE `guidance_embeds` flag to indicate "distilled" semantics

Rejected. `guidance_embeds` is a boolean derived from
`transformer.config.guidance_embeds` and used by ADR-003's introspection
fallback path to decide `guidance_scale` vs `true_cfg_scale`. Real
Hunyuan-Image 2.1 checkpoints carry `guidance_embeds=True` in their
config (see Rejected B for the runtime assertion), and the existing
introspection branch already reads that flag for `guidance_scale`
routing — overloading it to also mean "distilled, use
`distilled_guidance_scale`" would conflict with the existing meaning and
muddy a flag that ADR-003 made deliberately narrow. The family-string-driven
routing remains the right axis.

### F. Wait for runtime-core cluster consolidation before adding the Hunyuan branch (one routing copy, not two)

Rejected. Runtime-core consolidation (Backlog → Queued cluster) is a
cross-cutting refactor that touches both `_build_call_kwargs` copies,
the shared `cfg_scale → kwarg` mapping for every existing family, and
the introspection fallback. Bundling Hunyuan into that slice would
(a) couple a user-facing feature delivery to an unrelated structural
refactor and (b) violate SRR — the Hunyuan slice would no longer fit in
a reviewer's head. The two-copy maintenance cost is one extra branch
per family addition; it has been the standing rule since ADR-003 and is
paid down by the consolidation slice when it lands. The Vision
explicitly preserves consolidation as Queued.

## Deferred / Out of Scope

- HunyuanImage-3.0 (Tencent custom-code MoE, requires `trust_remote_code=True`,
  ~165 GB) — deferred to ai-stack-project per Backlog 2026-05-16 decision.
- `HunyuanImageRefinerPipeline` as a runnable family (family-string slot is
  reserved per §3, but no CFG branch, no FAMILY_DEFAULTS row, no tests).
- `HunyuanDiTPipeline` (older HunyuanDiT) — not on disk, not requested.
- Hunyuan edit / inpaint / ControlNet variants.
- Latent upscalers tuned for Hunyuan.
- Runtime-core cluster consolidation of the two `_build_call_kwargs`
  copies (Backlog → Queued).
- MCP tool surfacing of the new family (owned by the concurrent MCP slice 1).
- Hunyuan-Image 2.1 weight download into the local HF cache (operator step,
  not a code deliverable of this slice).
- Explicit `distilled_guidance_scale` canonical schema key (§6).

## Changelog

- 2026-05-17 — proposed (initial draft, Vision-aligned).
- 2026-05-17 — `code-reviewer` (Opus) round-1 pass: 1 MEDIUM + 3 LOW + 2 INFO findings. All actionable findings folded inline: (a) MEDIUM — Rejected B rewritten to capture the real failure mode (hard `ValueError` from pipeline runtime assertion `pipeline_hunyuanimage.py:727-728`, not silent degradation); (b) LOW — Context docstring/signature default mismatch noted; (c) LOW — Rejected F added (wait for runtime-core consolidation); (d) LOW — §3 Hunyuan class-roster audit appended; (e) INFO — §7 STOP-condition for unexpected caller-supplied component paths appended. Format-compliance INFO and §6 deferral INFO required no change. Status: `proposed` → `accepted`.
- 2026-05-17 — §3 amended with a "refiner family slot is defensive, not end-state" caveat. Triggered by Grant's prompt asking whether refiner support is Cascade-style mandatory or SDXL-style optional. Confirmed SDXL-style via `HunyuanImageRefinerPipeline.__call__` signature inspection: takes `image: PipelineImageInput | None`, ~4 default steps, distinct refiner VAE, distributed as a separate HF model. Future-refiner slice belongs on the edit-pipeline surface (`EricDiffusionEdit` etc.), not on `GEN_PIPELINE`. The reserved family slot is purely a defensive misroute-blocker. Status remains `accepted` — no decision reversal, only sharpening of the deferral's framing.
- 2026-05-24 — **Implementation complete.** Code commits (post-rebase IDs on `hunyuan-support`): `f0d2399` (Step 2 — `_FAMILY_PATTERNS` entries for `hunyuan-image` + `hunyuan-image-refiner`, 29 auto-detection/non-regression tests), `288137b` (Step 3 — `_build_call_kwargs` Hunyuan branch in both `nodes/eric_diffusion_generate.py` and `comfyless/generate.py` with `distilled_guidance_scale` routing, 23 CFG-routing tests), `df74b4f` (Step 4 — `FAMILY_DEFAULTS["hunyuan-image"] = {"cfg_scale": 3.25, "steps": 50}` row, 13 precedence-ladder + graceful-degrade tests). Plus `827c6ed` Step 2 follow-on (TECH_DEBT entry for pre-existing `torch.load` CWE-502 sites surfaced by semgrep PostToolUse hook).
- 2026-05-24 — **Amendment.** Step 5's live smoke at the schema-default 1024×1024 produced visibly artifacted output (horizontal embossed banding in sky, foil-textured sails, hull warping, smeared treelines). External quality diagnosis (Claude web) identified four contributors, ranked by impact: (1) sub-native resolution — Tencent README is explicit that "HunyuanImage-2.1 only supports 2K image generation … 1K will result in artifacts" (the 32× compression VAE was trained on 64×64 latents → 2048 images; 1K renders feed a 32×32 latent which is OOD); (2) no refiner pass; (3) possibly distilled checkpoint (verified-not — we're on the undistilled variant); (4) no prompt enhancement. **This amendment addresses #1 only.** Family-defaults row updated to `{cfg_scale: 3.25, steps: 50, width: 2048, height: 2048}`; the original `__call__`-docstring default of 1024 was misleading (consistent with the other docstring bug noted in the 2026-05-17 entry — the diffusers docstring is unreliable; the Tencent README is authoritative). §3 amended to retract the original "edit-pipeline home" framing for the refiner — empirical evidence (base alone produces unusable output even at 2K) shows refiner is functionally part of "what produces a clean Hunyuan generation," Cascade-coupling rather than SDXL-style optional polish; the refiner work belongs as a comfyless dispatch fork (`comfyless/hunyuan_chain.py` shape, separate ADR-016+) per Grant's direction, queued as the immediate next slice. Item #4 (Tencent's bundled reprompt model at upstream `tencent/HunyuanImage-2.1/reprompt/` — `HunYuanDenseV1ForCausalLM`, ~7B params, ~14 GB bf16, `trust_remote_code=True` required) is also in scope for the refiner slice given Grant's "review-and-pin remote code" posture from the abandoned Hunyuan-3 project — making `trust_remote_code` an ADR-governed first-class capability rather than a blocker. Adjacent finding: `pipeline.vae.enable_tiling()` is called unconditionally on every loaded pipeline (`nodes/eric_diffusion_loader.py:179-180`, `comfyless/generate.py:784-785`) — for Hunyuan's 32× VAE on ≥100 GB GPUs this is unnecessary AND may compound the banding (tile seams). Per-family skip queued as the third immediate-next slice. Test count delta: +2 (row-shape) + 2 (bare-run precedence) in `test_hunyuan.py`. Smoke re-run at 2048×2048 captured separately in the slice's commit body. Unit gate: 10 suites / 1074 tests pass, 0 failures (`test_hunyuan.py` contributed 65 of those; `test_params_schema.py` auto-picked-up the new FAMILY_DEFAULTS row, 135→136). `code-reviewer` (Opus, `model: "opus"` at invocation) ran on each non-trivial step (Steps 1 + 2 + 3 + 4); Step 1 returned `CHANGES REQUIRED` (4 findings folded — see 2026-05-17 entries above); Steps 2 + 3 + 4 returned `CLEAN`. **Live GPU smoke (Vision proof-hook) PASSED 2026-05-24** against `hunyuanvideo-community/HunyuanImage-2.1-Diffusers` (downloaded to `hf-local/HunyuanImage-2.1-Diffusers` after the original `tencent/HunyuanImage-2.1` was found to ship in upstream non-diffusers layout — Tencent ships at `tencent/*`, the diffusers-format repackaging lives at `hunyuanvideo-community/*-Diffusers`; this finding is logged to TECH_DEBT.md as a future-UX improvement for `detect_pipeline_class`'s error message). Smoke command: `./.venv/bin/python3 -m comfyless.generate --model <local-dir> --prompt "..." --width 1024 --height 1024 --output /tmp/hunyuan-smoke.png`. Result: 1024×1024 RGB PNG generated in 24.2s on cuda:0 (RTX PRO 6000 Blackwell, 102 GB VRAM); PNG `comfyless` tEXt chunk carries `model_family="hunyuan-image"`, `steps=50`, `cfg_scale=3.25` — all three slice invariants visible end-to-end. Log line `family=hunyuan-image defaults applied: cfg_scale=3.25, steps=50` proves invariant 3 (overlay); `Detected: HunyuanImagePipeline (family: hunyuan-image)` proves invariant 1 (auto-detection); 50 steps completed without kwarg-routing errors proves invariant 2 (distilled-guidance routing). No `security-auditor` invocation (ADR-013 §8 trailing-note did not trigger — zero ML-stack pin movement). Slice closed.
- 2026-05-28 — **Amendment + follow-on slice closure: tile-VAE-skip (slice `hunyuan-support` Step 6).** The 2026-05-24 amendment identified `pipeline.vae.enable_tiling()` being called unconditionally on every loaded pipeline as a contributing factor to the Hunyuan-Image artifact pattern (tile-seam striations compound the 32× compression VAE's small-latent untiled-decode that doesn't need tiling on ≥40 GB cards). This slice replaces the unconditional call with a family-aware resolver behind a new `--vae-tiling {auto,on,off}` CLI flag (default `auto`; off for `hunyuan-image`, on for every other family — preserves the prior 8×/16× behavior). Parallel optional `vae_tiling` dropdown input on the `Eric Diffusion Load Model` ComfyUI node. Daemon parity: `comfyless/server.py` cache_key + both `_load_pipeline` call sites + `_delegate_to_server` request dict updated; `_RUNTIME_KIND` in `comfyless/params_validation.py` gains `"vae_tiling": _KIND_STR` so non-string values reject at the IPC boundary as `ValidationError`. Commits (on `hunyuan-support`): `1a504ba` (Step 1 — resolver + 19 pure-function tests), `86f62f0` (Step 2 — loader wiring + argparse flag + 16 behavior/structural/argparse tests), `18fc68f` (Step 3 — daemon thread-through + validator addition + security review + 14 IPC/validator tests). `code-reviewer` (Opus, `model: "opus"` at invocation) on every step — Step 1 APPROVED no findings; Step 2 APPROVED no findings (1 minor regex tightening applied); Step 3 1 MEDIUM (third `_load_pipeline` site in `comfyless/mcp_server.py` — deliberate operator-tuning-knob omission per L600-606 comment, locked with explicit non-thread structural assertion). `security-auditor` (Opus) on Step 3 (server.py is §12 IPC trust-boundary surface): 2 MEDIUM + 1 LOW + 4 INFO. MEDIUM #1 (validator coverage gap) addressed in this slice; MEDIUM #2 (LoadError→ValidationError reclassification for value errors) partially addressed (type errors now reject at boundary); residual value-error reclassification and the LOW error-message echo deferred to MCP-rollout slice via three new TECH_DEBT entries. Full review at `docs/security/review-hunyuan-vae-tiling-server-2026-05-28.md`. Unit gate: 10 suites / 1128 tests pass, 0 failures (`test_hunyuan.py` 70→109 across all three Steps; `test_machine_boundary_validator.py` 118→128; `test_params_schema.py` 135→136). Live 2K A/B smoke deferred — blocked on host NVML/CUDA driver mismatch (reboot fix expected); the family-aware off-default is what addresses the 2026-05-24 amendment's tile-seam concern, validated structurally + behaviorally in the test suite. The refiner slice (separate ADR-016+ when it lands) inherits this slice's family-conditional logic for its own pipeline; the `hunyuan-image-refiner` family string is intentionally NOT in `_VAE_TILING_FAMILIES_DEFAULT_OFF` in v1 — the refiner slice will move it explicitly. Slice closed.
- 2026-07-11 — Renumbered from ADR-014 to ADR-025 during the base+refiner
  re-apply onto main. Original 014 collided with main's
  ADR-014-lora-audit-tool (parallel branch numbering). No decision content
  changed by the renumber.
- 2026-07-11 — RE-VERIFY PENDING: this ADR was written against diffusers
  0.36-dev and characterizes Hunyuan-Image 2.1 as guidance-distilled
  ("1× forward pass, negative prompt ignored"). The diffusers 0.39.0
  pipeline on main ships an enabled AdaptiveProjectedMixGuidance guider
  plus negative_prompt_embeds params, implying real 2-pass guidance that
  DOES consume negatives. Slice 2 of the re-apply (see
  docs/vision/epic-hunyuan-2-1-plus-enhancer.md) must empirically confirm
  1-pass vs 2-pass and negative-prompt efficacy against 0.39.0, and correct
  the "distilled / negatives-ignored" framing here if it no longer holds.

## AI-Disclosure

Claude (Opus 4.7) authored; Grant reviewed.
