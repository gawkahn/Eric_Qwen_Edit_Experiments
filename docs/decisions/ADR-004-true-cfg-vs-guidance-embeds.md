# ADR-004: True CFG vs Guidance Embedding for Qwen-Image

**Date:** 2026-04-21 (written retroactively; decision made ~2026-04-12)
**Status:** accepted

---

## Context

Qwen-Image-2512 has `transformer.config.guidance_embeds = False`. In guidance-distilled
models (Flux.1-dev, Flux.2, Chroma), `guidance_scale` is fed as a learned transformer
input embedding — the model was trained to condition on it. Qwen-Image was not trained
with guidance embedding; the parameter slot is architecturally absent. Passing
`guidance_scale` to Qwen-Image produces no CFG effect: output is identical regardless
of the value passed.

The correct mechanism is standard **classifier-free guidance (true CFG)**:
run two forward passes per denoising step — one unconditional (empty/negative prompt),
one conditional (full prompt) — then combine:

```
output = uncond + scale * (cond - uncond)
```

Official Qwen-Image-2512 recommendation: 50 steps, `true_cfg_scale = 4.0`.

This distinction is not obvious from the diffusers API surface alone; both families
accept a `guidance_scale`-shaped parameter. The bug mode — using `guidance_scale` on
Qwen — produces plausible-looking output with no actual prompt adherence control,
which is hard to diagnose.

## Decision

Route Qwen-Image through `true_cfg_scale` (two forward passes per step). All other
model families use `guidance_scale` (single-pass embedding, detected at load time via
`read_guidance_embeds()` in `eric_diffusion_utils.py`). Routing is keyed on
`model_family` in `_build_call_kwargs()` (see ADR-003).

The `guidance_embeds` flag is read from `transformer.config.guidance_embeds` at load
time and stored in the `GEN_PIPELINE` dict, but family-based routing takes precedence
over flag-based routing to avoid edge cases from misconfigured model configs.

## Alternatives Rejected

**Use `guidance_scale` for Qwen-Image** — produces flat output with no CFG effect.
Users diagnose this incorrectly as a model quality issue or wrong step count.

**Always double-pass for all families** — wastes 2× compute on distilled models that
don't need it; breaks Flux.1 which expects the embedding conditioning path and was
not trained for double-pass CFG.

**Runtime detection only (no family routing)** — `guidance_embeds=False` in config
doesn't definitively establish that true CFG is the correct substitute; family
knowledge is required to make that call safely.

## Deferred / Out of Scope

**Dynamic true CFG for non-Qwen families where `guidance_embeds=False`** — no known
use case; deferred until a concrete model requires it.

**Models claiming `guidance_embeds=True` that behave otherwise** — no known cases;
handle if reported.

## Changelog

- ~2026-04-12: Initial implementation in `pipeline_qwen_edit.py` and comfyless
- ~2026-04-12: `read_guidance_embeds()` helper added to `eric_diffusion_utils.py`
- 2026-04-21: ADR written retroactively; decision still active. Documented in
  DEV_NOTES.md and project CLAUDE.md previously; this ADR is the canonical record.

## AI-Disclosure

ADR authored by Claude Sonnet 4.6, 2026-04-21. Discovery and implementation by
Eric Hiss; rationale drawn from DEV_NOTES.md and code. Reviewed by Grant Kahn.
