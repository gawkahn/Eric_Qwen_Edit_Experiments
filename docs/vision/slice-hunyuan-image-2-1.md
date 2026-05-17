# Vision Slice — Hunyuan-Image 2.1 support

**Backlog ref:** `Image_gen/Backlog.md` → Immediate → "Hunyuan-Image 2.1 support
(ComfyUI + comfyless)" (promoted 2026-05-16).

**Branch:** `hunyuan-support` (parallel-session worktree).

**Risk level:** **L2** (loader-touching; touches a shared ordered pattern list;
no PII, auth, billing, or audit-trail surfaces; not a Red Zone change).

## Posture

> **Posture:** Boundary: domain rules (family detection + per-family parameter
> routing inside the existing loader/generate paths). Risk factors: broad
> impact (a new `_FAMILY_PATTERNS` entry sits on the shared ordered list every
> loader call walks; mis-ordering or a typo regresses an existing family);
> near security-truth surface (touches the same `detect_pipeline_class` path
> covered by the 2026-04-23 `resolve_hf_path` review, though no behavior
> change to the resolver itself).

## Four signals

- **Who.** Solo desktop user (Grant) running ComfyUI workflows or
  `python -m comfyless.generate ...`. No new actors, no new permission
  boundary; no service surface. MCP exposure of the new family is a
  concurrent-session concern (Slice 1 in `comfyless/mcp_server.py`) — out of
  scope here.
- **Data.** Reads model files under existing `model_path` (resolved by the
  existing `resolve_hf_path`). Writes: tensors in VRAM, PIL output,
  `GEN_METADATA` dict, PNG `comfyless` chunk. New: the string
  `"hunyuan-image"` flows through GEN_PIPELINE dicts and sidecar metadata.
  No secrets, no PII, no payments, no audit-trail.
- **Boundary.** In scope: `nodes/eric_diffusion_utils.py:_FAMILY_PATTERNS`,
  `nodes/eric_diffusion_generate.py:_build_call_kwargs`,
  `comfyless/generate.py:_build_call_kwargs`,
  `comfyless/family_defaults.py:FAMILY_DEFAULTS`, new `test_hunyuan.py`, new
  `docs/decisions/ADR-013-hunyuan-image-cfg-routing.md` + Obsidian mirror.
  Excludes: HunyuanImage-3.0 (deferred to ai-stack), HunyuanDiTPipeline,
  HunyuanImageRefinerPipeline, edit/inpaint/ControlNet variants,
  latent-upscale, runtime-core CFG consolidation (cluster Queued), MCP
  exposure. Concurrent-session no-touch list per the parallel-sessions
  brief: `comfyless/mcp_server.py`, `test_mcp_server.py`,
  `comfyless/generate.py:_run_json_mode` docstring + `_save_with_metadata`,
  `comfyless/cascade.py:_save_with_metadata`, `comfyless/README.md`,
  `docs/vision/slice-1-mcp-generate.md`, `scripts/`,
  `nodes/eric_qwen_*_lora.py`. Merge hotspots: `CLAUDE.md` (only touched if
  the test-suite paragraph needs to grow — coordinate first) and `uv.lock`
  (no new dep planned, no touch expected).
- **Failure.** Worst observed bad outcomes: (a) `distilled_guidance_scale`
  silently mis-routed → degraded image with no error; (b)
  `_FAMILY_PATTERNS` mis-ordering or typo silently regresses an unrelated
  family's auto-detection; (c) family-defaults overlay precedence broken →
  user gets the wrong cfg/steps without a visible signal. All three are
  prevented by the proof hooks below.

## Intent

Add **`HunyuanImagePipeline`** (Hunyuan-Image 2.1, diffusers 0.37.1) as a
first-class family `"hunyuan-image"` in both code paths — auto-detected from
`model_index.json`, routed through a new explicit `distilled_guidance_scale`
CFG branch (the project's third routing shape, distinct from Qwen's
`true_cfg_scale` and Flux/SDXL's `guidance_scale`), and seeded with
model-card-sourced defaults in the comfyless overlay.

## Invariants (must always be true)

1. **Auto-detection** — Loading a model whose `model_index.json` has
   `_class_name: "HunyuanImagePipeline"` yields
   `model_family == "hunyuan-image"` from `infer_model_family`, and
   `pipeline_class is diffusers.HunyuanImagePipeline`.
2. **CFG routing** — When `model_family == "hunyuan-image"`, the kwargs dict
   passed to `pipe(**kwargs)` contains `distilled_guidance_scale` and
   contains neither `guidance_scale` nor `true_cfg_scale`. This holds
   identically in both `nodes/eric_diffusion_generate.py:_build_call_kwargs`
   and `comfyless/generate.py:_build_call_kwargs`.
3. **Defaults overlay precedence** — The `hunyuan-image` entry in
   `FAMILY_DEFAULTS` is applied above `COMFYLESS_SCHEMA` defaults and below
   explicit CLI flags / sidecar values / `--iterate` axes, as per ADR-009's
   precedence ladder. Explicit `--cfg-scale 5.0` on the command line wins
   over the family default.
4. **Pattern-list non-regression** — The new entry in `_FAMILY_PATTERNS` does
   not change `infer_model_family`'s output for any pipeline class currently
   produced by the seven existing test suites (Qwen-Image, Qwen-Edit, Flux,
   Flux2, Flux2Klein, Chroma, AuraFlow, SD1/SD3/SDXL, ZImage, Stable
   Cascade).
5. **No new caller-surface widening** — The slice adds zero new
   `--`-prefixed CLI flags, zero new GEN_PIPELINE dict keys consumed by
   external callers, zero new HF download behaviors. The existing
   `--allow-hf-download` gate is the *only* path by which 2.1 weights are
   fetched.

## Failure semantics

- **Detection failure** (e.g. user points at a directory missing
  `model_index.json` or naming an unknown class): existing
  `detect_pipeline_class` raises `ValueError` unchanged. Fail-closed. No
  change to the resolver.
- **Mis-installed diffusers** (HunyuanImagePipeline absent from the
  installed version): the existing `getattr(diffusers, class_name, None)`
  path raises `ValueError` with the existing diffusers-upgrade hint.
  Unchanged.
- **CFG kwargs construction error** (defensive): if the routing branch is
  somehow bypassed and falls through to introspection, the introspection
  path will accept `distilled_guidance_scale` if present — but the
  regression would silently use the wrong CFG semantics. Invariant 2 + its
  negative-case test prevents this from shipping.
- **Family-defaults missing** (`hunyuan-image` not in `FAMILY_DEFAULTS`):
  comfyless silently falls back to `COMFYLESS_SCHEMA` defaults (cfg=3.5,
  steps=28) — wrong but not catastrophic. Invariant 3's test catches this.

## Out of scope (explicit exclusions)

- HunyuanImage-3.0 — Tencent custom-code MoE, requires
  `trust_remote_code=True`, deferred to ai-stack-project per the 2026-05-16
  decision.
- `HunyuanDiTPipeline` (older variant) — separate slice if ever wanted; not
  on disk, not in this Vision.
- `HunyuanImageRefinerPipeline` — separate slice if/when refiner two-stage
  flows are wanted.
- Hunyuan edit / inpaint / ControlNet variants.
- Latent upscalers tuned for Hunyuan.
- Runtime-core cluster consolidation (Queued; would dedupe the two
  `_build_call_kwargs` copies — explicitly NOT part of this slice).
- MCP tool surfacing of the new family — owned by the concurrent MCP
  slice 1.
- Hunyuan-Image-2.1 weight download into the local HF cache — performed by
  the user (or as an optional sub-step of the proof phase), not a code
  deliverable of the slice.

## Proof hooks

All `test_hunyuan.py` cases run on CPU using a synthetic `model_index.json`
fixture plus monkey-patched `diffusers.HunyuanImagePipeline` stub — no GPU,
no real model load.

**Positive cases** (one per invariant, except where one test covers
multiple):

- `python3 test_hunyuan.py` — full suite must pass with **0 failures**.
  - asserts `infer_model_family("HunyuanImagePipeline") == "hunyuan-image"`
    (invariant 1).
  - loads a fixture `model_index.json` with `_class_name:
    "HunyuanImagePipeline"` and asserts `detect_pipeline_class` returns
    `(diffusers.HunyuanImagePipeline, "HunyuanImagePipeline",
    "hunyuan-image")` (invariant 1, end-to-end).
  - calls `_build_call_kwargs(..., model_family="hunyuan-image", ...)`
    against both `nodes/eric_diffusion_generate.py` and
    `comfyless/generate.py` copies and asserts the returned dict has
    `"distilled_guidance_scale"` and does not have `"guidance_scale"` or
    `"true_cfg_scale"` (invariant 2).
  - asserts `FAMILY_DEFAULTS["hunyuan-image"]` exists, has the expected
    keys, and a precedence walk confirms explicit-CLI > family-default >
    schema-default (invariant 3).
  - re-runs the family inference for `QwenImagePipeline`,
    `QwenImageEditPlusPipeline`, `FluxPipeline`, `Flux2Pipeline`,
    `Flux2KleinPipeline`, `ChromaPipeline`, `AuraFlowPipeline`,
    `StableDiffusion3Pipeline`, `StableDiffusionXLPipeline`,
    `StableDiffusionPipeline`, `ZImagePipeline` (those that currently
    round-trip through `_FAMILY_PATTERNS`) and asserts each still maps to
    its existing family string (invariant 4 — non-regression).

**Negative cases** (at least one per invariant whose silent-fail mode
matters):

- Negative for invariant 2: a test that constructs the kwargs for a known
  non-Hunyuan family (e.g. `flux`) and asserts the dict does **not**
  contain `distilled_guidance_scale` — proves we didn't smear the new key
  across all branches.
- Negative for invariant 3: a test that monkey-patches `FAMILY_DEFAULTS` to
  remove the `hunyuan-image` entry and asserts the overlay walker doesn't
  crash — only falls back to schema default (proves the overlay layer
  remains robust to a missing entry).
- Negative for invariant 4: a test that constructs a fake `_class_name`
  like `"HunyuanImageRefinerPipeline"` and asserts `infer_model_family`
  returns `"hunyuan-image"` if we want the refiner to share routing, **or**
  a different family string if we want it isolated — the assertion answers
  that question deliberately rather than letting substring ordering decide
  silently. (This forces a deliberate naming choice during the change
  plan.)

**Regression hook** — the seven existing CPU suites must continue to pass
with 0 failures:

```bash
/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 test_manual_loop.py \
  && /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 test_multistage.py \
  && /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 test_params_schema.py \
  && /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 test_cascade.py \
  && /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 test_iterate.py \
  && /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 test_samplers.py \
  && /home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 test_server_robustness.py
```

**Live GPU smoke** (separate, outside the unit gate — performed once after
the code lands and once weights are on disk):

```bash
/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3 -m comfyless.generate \
  --model <hunyuan-2.1-local-path> \
  --prompt "a quiet alpine lake at dawn, photorealistic" \
  --aspect 1:1 --target-mp 1.0 \
  --savepath /tmp/hunyuan-smoke.png
```

Smoke pass criterion: file written, PNG has comfyless metadata chunk,
`model_family` in the chunk reads `"hunyuan-image"`.

## §12 artifacts required before code

- **`ADR-013-hunyuan-image-cfg-routing.md`** — documents the introduction
  of the third CFG routing shape (`distilled_guidance_scale`), the choice
  to keep two parallel `_build_call_kwargs` copies (extending the existing
  pattern that the runtime-core cluster will eventually consolidate), the
  family-string naming choice (`hunyuan-image` vs. `hunyuan` — naming locks
  in routing eligibility), and the deliberate exclusion of the
  refiner / DiT / 3.0 variants. Mirror to Obsidian `Decisions/`.
- **Security review:** NOT required — no Red Zone surface, no new
  caller-supplied path widening, no IPC change, no new external-input
  ingestion. The §12 trigger list (CLAUDE.md "Review bar") names IPC,
  `resolve_hf_path`, and `_run_json_mode` as the existing trippers; this
  slice touches none of them.

## Reviewer plan

- **`code-reviewer` (Opus, pinned at invocation)** — run after each
  non-trivial slice step, before commit. Non-negotiable per global §5A and
  project review-bar.
- **`security-auditor`** — not invoked for this slice (no Red Zone change,
  no §12 trigger met). If during implementation an unexpected surface
  emerges (e.g. a Hunyuan-specific caller-supplied component path enters
  the picture), STOP and re-evaluate.

## Open questions (must resolve during the Change Plan, not after)

1. **Refiner naming.** Should `HunyuanImageRefinerPipeline` (out of scope
   to *implement*) be detected as family `"hunyuan-image"` (shared routing)
   or as `"hunyuan-image-refiner"` (isolated, falls to introspection until
   a separate slice)? Recommend isolating it — a
   `("hunyuanimagerefiner", "hunyuan-image-refiner")` entry placed *before*
   `("hunyuanimage", "hunyuan-image")` in `_FAMILY_PATTERNS` (because
   first-match-wins after substring-strip). Locks in safe behavior even
   though the family has no defaults row yet (introspection fallback
   handles the call). To be confirmed in the ADR.
2. **Family-defaults values.** Round-one stub per Hunyuan-Image 2.1 model
   card recommendation: `{"cfg_scale": 3.25, "steps": 50}` (matches
   `HunyuanImagePipeline.__call__` defaults: `num_inference_steps=50,
   distilled_guidance_scale=3.25`). To be confirmed against the model card
   during ADR drafting; same "starting points, not absolute truths" caveat
   as other family rows.
3. **`max_sequence_length` plumbing.** Hunyuan-Image's call signature
   doesn't accept `max_sequence_length` — confirm the
   introspection-trimmed unknown-family path is not a fallback we'll hit.
   (Invariant 2's positive test covers this implicitly.)
4. **`negative_prompt` semantics.** `HunyuanImagePipeline.__call__`
   accepts `negative_prompt` even though it's a distilled-guidance model.
   Need to confirm whether to forward user-supplied negatives (consistent
   with the new branch including `negative_prompt` when set) or silently
   drop them (the Flux pattern). Recommend forwarding — call-site already
   gates on `if negative_prompt:`. To be confirmed in the ADR.

## Status

- Approved 2026-05-16.
- Backlog updated (Obsidian `Image_gen/Backlog.md`): 2.1 promoted to
  Immediate, 3.0 deferred to ai-stack.
- Next action: `/change-slice` → ADR-013 draft → implementation.
