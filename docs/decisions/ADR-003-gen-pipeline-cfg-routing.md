# ADR-003: GEN_PIPELINE Architecture and CFG Routing

**Date:** 2026-04-21 (written retroactively; decision made ~2026-04-12)
**Status:** accepted

---

## Context

Initial nodes were Qwen-specific. As the project expanded to support Flux.1, Flux.2,
Chroma, SDXL, SD1, and AuraFlow, writing separate loader/generate node pairs per model
family would have created significant duplication and a long maintenance tail.

Additionally, different model families use fundamentally different CFG mechanisms:

- **Guidance-distilled models** (Flux.1-dev, Flux.2, Chroma): `guidance_scale` is fed
  as a learned transformer input embedding — one forward pass per step. The model was
  trained to condition on this value; it is architecturally meaningful.
- **Non-distilled models** (Qwen-Image, SDXL, SD1): standard classifier-free guidance
  — two forward passes per step (unconditional + conditional). `guidance_scale` as an
  embedding is either absent or inert (see ADR-004 for Qwen-Image specifics).

Using the wrong mechanism produces either flat/unusable output or wasted compute.

## Decision

**Generic `GEN_PIPELINE` dict** as the inter-node contract:

```python
{
    "pipeline":        <pipeline obj>,
    "model_path":      str,
    "model_family":    str,   # "qwen-image" | "flux" | "flux2" | "flux2klein" | "chroma" | "sdxl" | ...
    "offload_vae":     bool,
    "guidance_embeds": bool,  # from transformer.config.guidance_embeds
}
```

Model family detected at load time from `model_index.json → _class_name → short slug`.
New model families in diffusers work automatically via `getattr(diffusers, class_name)`.

**CFG routing table** in `_build_call_kwargs()`, keyed on `model_family`:
- `qwen-image` → `true_cfg_scale` (double-pass CFG), negative prompt used
- `flux` / `flux2` / `flux2klein` / `chroma` → `guidance_scale` (embedding, single pass), no negative
- `sdxl` / `sd3` / `sd1` / `auraflow` → `guidance_scale` + negative prompt
- unknown → introspect `pipe.__call__` signature, pass only accepted params

## Alternatives Rejected

**Per-model node classes** — high duplication; every new model family requires a new
loader, generate, and multistage node. Scales to O(N) work per family.

**Duck-typing without family detection** — too fragile. `guidance_scale` has the same
parameter name but opposite meaning in distilled vs non-distilled families. Silent
wrong-CFG-mode bugs are worse than explicit routing.

**Single CFG path for all families** — would either break Qwen-Image output (flat,
no CFG effect) or waste 2× compute on distilled models and break their conditioning.

## Deferred / Out of Scope

**HuggingFace cache integration** — loaders currently require local filesystem paths;
should also accept HF repo IDs. Queued in Backlog.

**Edit nodes for Flux.2** — Flux.2-Klein has no latent-space access for standard
img2img/inpainting. Parked in Backlog pending diffusers changes.

**Broader family coverage** — HiDream, Stable Cascade, etc. Tracked in Backlog Ideas.
The auto-detection path means most new families work with only CFG routing additions.

## Changelog

- ~2026-04-12: Initial generic pipeline infrastructure (`b951814`)
- ~2026-04-12: SDXL/SD3/SD1 CFG routing added (`18322b1`)
- ~2026-04-12: AuraFlow routing added (`3abaec9`)
- ~2026-04-12: Flux.2-Klein family and routing (`775cfec`)
- ~2026-04-12: EricDiffusionEdit node for Flux.2 edit path (`5d98718`)
- 2026-04-21: ADR written retroactively; decision still active

## AI-Disclosure

ADR authored by Claude Sonnet 4.6, 2026-04-21. Architecture designed by Eric Hiss;
rationale reconstructed from commit history. Reviewed by Grant Kahn.
