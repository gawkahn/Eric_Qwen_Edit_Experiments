# ADR-014: Hunyuan-Image 2.1 — CFG Routing, Family Detection, Refiner Isolation

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

**Important caveat on the refiner family slot — it is defensive, not a
commitment to the end-state for refiner support.** Per the
HunyuanImageRefinerPipeline `__call__` inspection (signature confirmed
via `inspect.signature` at decision time): the refiner is structurally
an **edit / image-to-image pipeline** in the SDXL-refiner sense, not a
Cascade-style mandatory second stage. Base `HunyuanImagePipeline` produces
a complete saveable image on its own — the official upstream usage example
saves `pipe(prompt, ...).images[0]` directly to disk with no refiner pass.
The refiner is opt-in polish: it consumes a complete image (`image:
PipelineImageInput | None`), runs ~4 default denoising steps, and outputs
a refined image; it uses a distinct VAE class (`AutoencoderKLHunyuanImageRefiner`)
and is distributed as a separate HF model. When refiner support eventually
lands as its own slice, the natural architectural home is the existing
**edit-pipeline surface** (where `EricDiffusionEdit` already takes image
slots and GEN_METADATA), NOT the `GEN_PIPELINE` family system this ADR
extends — the family-pattern slot reserved here exists only to prevent
accidental misroute of a refiner-loaded checkpoint through the base CFG
branch via substring overlap. The future-refiner slice may dissolve the
`hunyuan-image-refiner` family string entirely in favor of an edit-side
analog; this is left open.

### 4. Family-defaults row in `comfyless/family_defaults.py`

Add one alphabetically-ordered entry:

```python
# ── hunyuan-image (Hunyuan-Image 2.1) ───────────────────────────────
# Distilled-guidance family; cfg_scale routes to distilled_guidance_scale
# in the call-build layer. Defaults match HunyuanImagePipeline.__call__
# signature: distilled_guidance_scale=3.25, num_inference_steps=50.
# Source: diffusers 0.37.1 HunyuanImagePipeline implementation (and the
# Tencent Hunyuan-Image 2.1 model card recommendation, both 3.25 / 50).
"hunyuan-image": {"cfg_scale": 3.25, "steps": 50},
```

`true_cfg_scale` is intentionally NOT in this row — Hunyuan does not use
the double-pass CFG path; setting it would be meaningless. The overlay
applier silently ignores unknown keys but it would still be a source of
confusion to readers.

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
- 2026-05-24 — **Implementation complete.** Code commits (post-rebase IDs on `hunyuan-support`): `f0d2399` (Step 2 — `_FAMILY_PATTERNS` entries for `hunyuan-image` + `hunyuan-image-refiner`, 29 auto-detection/non-regression tests), `288137b` (Step 3 — `_build_call_kwargs` Hunyuan branch in both `nodes/eric_diffusion_generate.py` and `comfyless/generate.py` with `distilled_guidance_scale` routing, 23 CFG-routing tests), `df74b4f` (Step 4 — `FAMILY_DEFAULTS["hunyuan-image"] = {"cfg_scale": 3.25, "steps": 50}` row, 13 precedence-ladder + graceful-degrade tests). Plus `827c6ed` Step 2 follow-on (TECH_DEBT entry for pre-existing `torch.load` CWE-502 sites surfaced by semgrep PostToolUse hook). Unit gate: 10 suites / 1074 tests pass, 0 failures (`test_hunyuan.py` contributed 65 of those; `test_params_schema.py` auto-picked-up the new FAMILY_DEFAULTS row, 135→136). `code-reviewer` (Opus, `model: "opus"` at invocation) ran on each non-trivial step (Steps 1 + 2 + 3 + 4); Step 1 returned `CHANGES REQUIRED` (4 findings folded — see 2026-05-17 entries above); Steps 2 + 3 + 4 returned `CLEAN`. **Live GPU smoke (Vision proof-hook) PASSED 2026-05-24** against `hunyuanvideo-community/HunyuanImage-2.1-Diffusers` (downloaded to `hf-local/HunyuanImage-2.1-Diffusers` after the original `tencent/HunyuanImage-2.1` was found to ship in upstream non-diffusers layout — Tencent ships at `tencent/*`, the diffusers-format repackaging lives at `hunyuanvideo-community/*-Diffusers`; this finding is logged to TECH_DEBT.md as a future-UX improvement for `detect_pipeline_class`'s error message). Smoke command: `./.venv/bin/python3 -m comfyless.generate --model <local-dir> --prompt "..." --width 1024 --height 1024 --output /tmp/hunyuan-smoke.png`. Result: 1024×1024 RGB PNG generated in 24.2s on cuda:0 (RTX PRO 6000 Blackwell, 102 GB VRAM); PNG `comfyless` tEXt chunk carries `model_family="hunyuan-image"`, `steps=50`, `cfg_scale=3.25` — all three slice invariants visible end-to-end. Log line `family=hunyuan-image defaults applied: cfg_scale=3.25, steps=50` proves invariant 3 (overlay); `Detected: HunyuanImagePipeline (family: hunyuan-image)` proves invariant 1 (auto-detection); 50 steps completed without kwarg-routing errors proves invariant 2 (distilled-guidance routing). No `security-auditor` invocation (ADR-013 §8 trailing-note did not trigger — zero ML-stack pin movement). Slice closed.

## AI-Disclosure

Claude (Opus 4.7) authored; Grant reviewed.
