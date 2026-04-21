# ADR-002: Three-Tier LoRA Loading Fallback

**Date:** 2026-04-21 (written retroactively; decision made ~2026-04-12)
**Status:** accepted

---

## Context

LoRA files for diffusers models arrive in multiple formats with no single reliable
loading path. Community LoRAs target different base tools: diffusers-native PEFT,
Kohya-ss, BFL's internal Flux format, ai-toolkit. They use different key naming
conventions, alpha storage conventions, and weight decomposition formats (LoRA,
LoKR, LoHa). A single-path loader fails on the majority of real-world files.

## Decision

Three-tier cascade, attempted in order:

1. **`pipeline.load_lora_weights()`** — diffusers/PEFT native path. Works for
   standard PEFT-format LoRAs and most diffusers-native checkpoints. Preserves
   adapter state (loadable/unloadable via `set_adapters` / `delete_adapters`).

2. **PEFT injection** — inject adapter layers manually into the transformer, then
   load weights. Handles cases where the pipeline loader rejects the key structure
   but the weight tensors are valid PEFT-compatible shapes.

3. **Direct merge** — load state dict, apply key conversion if needed, merge scaled
   LoRA weights directly into model parameters: `weight += alpha * down @ up`. No
   adapter registration; the merge is permanent for the pipeline's lifetime.

The cascade is ordered cheapest-to-most-invasive. Tier 3 is the nuclear option —
it trades adapter bookkeeping for maximum compatibility.

LoRA key translation runs before all three tiers: Kohya → BFL prefix mapping,
LoKR/LoHa decomposition, alpha scaling normalization, ai-toolkit bogus-alpha
detection (ai-toolkit stores ~1e10 for direct-stored w1/w2; must be treated as 1.0).

## Alternatives Rejected

**Fail fast on format mismatch** — unacceptable; the majority of community Flux/Qwen
LoRAs would be unusable without pre-conversion.

**Require pre-conversion as a separate step** — too much friction; bakes in a
dependency on an external conversion tool and breaks the "just point at the file"
workflow.

**Single diffusers path only** — rejects most real-world LoRAs; diffusers-native
format is not what civitai or ai-toolkit produces.

## Deferred / Out of Scope

**Text encoder LoRA (`lora_te1_*` keys)** — currently silently dropped during
loading; need to load onto T5 / CLIP text encoder. Tracked in TECH_DEBT.md.

**Skip unresolvable Kohya keys** — keys like `distilled_guidance_layer` that cannot
be mapped should be skipped with a warning rather than silently dropped or errored.
Tracked in TECH_DEBT.md.

## Changelog

- ~2026-04-12: Initial three-tier implementation (`1d70fef`, `fc52a6d`)
- ~2026-04-12: LoKR/LoHa format support added (`afbffe0`, `fc52a6d`)
- ~2026-04-12: Kohya → BFL key conversion added (`c99f784`, `fb97dcc`)
- ~2026-04-12: LoKR alpha convention fix for ai-toolkit bogus alpha (`599f59b`)
- ~2026-04-12: Direct-merge fallback improved, module count surfaced (`834da83`, `eee3c89`)
- 2026-04-21: ADR written retroactively; decision still active

## AI-Disclosure

ADR authored by Claude Sonnet 4.6, 2026-04-21. Architecture and implementation by
Eric Hiss; rationale reconstructed from commit history and code. Reviewed by Grant Kahn.
