#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""Per-family default param overlay for comfyless.

See `docs/decisions/ADR-009-per-family-default-params.md` for the
design rationale and precedence ladder. In short:

    schema_default < family_default < sidecar < --override
                  < explicit --flag < --iterate axis

This module owns the family-default layer.  Editing this file is the
ONLY change needed to add or adjust a family's defaults — `_run_one`
in `generate.py` consumes the dict generically.

Conventions for editing FAMILY_DEFAULTS:

* Keys must match strings produced by `infer_model_family` in
  `nodes/eric_diffusion_utils.py`: ``qwen-image``, ``qwen-edit``,
  ``flux2klein``, ``flux2``, ``chroma``, ``flux``, ``auraflow``,
  ``sd3``, ``sdxl``, ``sd1``, ``zimage``.  Unknown families pass
  through with no overlay (no error).
* Each entry is a PARTIAL dict — only the keys this family has an
  opinion on.  Keys not listed fall through to ``COMFYLESS_SCHEMA``
  defaults.  Adding a key = one new line; removing one = delete the
  line and the schema default takes over.
* Alphabetical by family for predictable diffs.
* One inline comment per family naming the source of the value
  (official model card, empirical sweep, community consensus).
* Only canonical ``COMFYLESS_SCHEMA`` keys are honored — other keys
  are silently ignored by the overlay applier.

These are STARTING POINTS, not absolute truths.  Per-prompt sweet
spots are typically better expressed via ``--params`` sidecars,
which sit above this layer in the precedence ladder.
"""

from __future__ import annotations

from typing import Any, Dict


# ════════════════════════════════════════════════════════════════════════
#  Family default values
# ════════════════════════════════════════════════════════════════════════

FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # ── auraflow ────────────────────────────────────────────────────────
    # Uses guidance_scale path; HF model card recommends cfg=3.5 / 30 steps.
    "auraflow":   {"cfg_scale": 3.5, "steps": 30},

    # ── chroma ──────────────────────────────────────────────────────────
    # Flux-derivative; community consensus runs slightly hotter than Flux.
    # Source: empirical from prior project sweeps (memory: civitai_orphaned_files).
    "chroma":     {"cfg_scale": 4.0, "steps": 30},

    # ── flux (Flux.1 dev / schnell) ─────────────────────────────────────
    # Schema default already targets Flux.1; family entry kept explicit
    # so the row exists when calibration adjusts it.
    # Source: BFL Flux.1-dev model card.
    "flux":       {"cfg_scale": 3.5, "steps": 28},

    # ── flux2 (Flux.2-dev) ──────────────────────────────────────────────
    # Source: BFL Flux.2-dev release notes.
    "flux2":      {"cfg_scale": 3.5, "steps": 28},

    # ── flux2klein (FLUX.2-klein-9B, the step-distilled flagship) ───────
    # is_distilled:true in its model_index.json keeps the plain family name
    # (BFL's own naming: the flagship is just "klein"). Step-distilled to 4
    # inference steps; guidance_scale 1.0 = CFG off (Flux2KleinPipeline runs
    # REAL CFG at cfg>1, unlike flux/flux2 guidance embeds). The prior
    # 24/3.5 row matched neither Klein card (ADR-009 changelog 2026-07-22).
    # Source: FLUX.2-klein-9B README (guidance_scale=1.0, steps=4).
    "flux2klein": {"cfg_scale": 1.0, "steps": 4},

    # ── flux2klein-base (FLUX.2-klein-base-9B, non-distilled) ───────────
    # Same Flux2KleinPipeline class, NO is_distilled marker → this family
    # (infer_model_family). Trained without step or guidance distillation;
    # real CFG wants the full schedule.
    # Source: FLUX.2-klein-base-9B README (guidance_scale=4.0, steps=50).
    "flux2klein-base": {"cfg_scale": 4.0, "steps": 50},

    # ── krea (Krea-2-Raw) ───────────────────────────────────────────────
    # Krea2Pipeline, non-distilled. Single-pass guidance_scale (flux-like).
    # Raw is a fine-tuning base ("not recommended for inference" per Krea)
    # but supported. Source: krea/Krea-2-Raw model card (52 steps, cfg 3.5).
    "krea":       {"cfg_scale": 3.5, "steps": 52},

    # ── krea-turbo (Krea-2-Turbo) ───────────────────────────────────────
    # Krea2Pipeline, distilled (is_distilled=true → this family). CFG is
    # disabled (cfg=0.0 → single forward pass). Recommended mu/timestep-
    # shift of 1.15 is not yet exposed (TECH_DEBT → CFG Routing); diffusers'
    # default dynamic shift applies. Source: krea/Krea-2-Turbo model card
    # (8 steps, cfg 0.0).
    "krea-turbo": {"cfg_scale": 0.0, "steps": 8},

    # ── qwen-edit (Qwen-Image-Edit-2511) ────────────────────────────────
    # Edit pipeline uses true_cfg path. 30 steps tracks documented sweet
    # spot in pipelines/pipeline_qwen_edit.py.
    # Source: Alibaba Qwen-Image-Edit-2511 model card.
    "qwen-edit":  {"true_cfg_scale": 4.0, "steps": 30},

    # ── qwen-image (Qwen-Image-2512) ────────────────────────────────────
    # Generation pipeline. Official recommendation per CLAUDE.md and the
    # Qwen-Image-2512 model card: 50 steps, true_cfg=4.0. cfg_scale is
    # ignored on this family but the schema default of 3.5 applies if
    # someone routes through the introspection path.
    # Source: Alibaba Qwen-Image-2512 model card.
    "qwen-image": {"true_cfg_scale": 4.0, "steps": 50},

    # ── sd1 (Stable Diffusion 1.x) ──────────────────────────────────────
    # Uses guidance_scale + DDPM-style scheduler (no sampler swap).
    # Source: SAI SD1.5 model card recommended values.
    "sd1":        {"cfg_scale": 7.5, "steps": 30},

    # ── sd3 (Stable Diffusion 3 / 3.5) ──────────────────────────────────
    # Source: SAI SD3.5-Large model card.
    "sd3":        {"cfg_scale": 4.5, "steps": 28},

    # ── sdxl (SDXL base + fine-tunes including Pony/Illustrious) ────────
    # Pony and Illustrious are SDXL fine-tunes; detect_pipeline_class
    # resolves all three to "sdxl". cfg=7 is a reasonable starting point
    # for all three, though Pony/Illustrious sometimes want higher
    # (cfg=7-8). Refine via per-prompt --params overlays rather than
    # adding a sub-family layer (see ADR-009 Alternatives Rejected).
    # Source: SAI SDXL model card; community consensus for fine-tunes.
    "sdxl":       {"cfg_scale": 7.0, "steps": 28},

    # ── zimage (Z-Image-base) ───────────────────────────────────────────
    # The full base model. flux-like guidance_scale path. 30 steps / cfg 4.0
    # confirmed to render cleanly in gen-validation (2026-07-06 Phase A).
    "zimage":       {"cfg_scale": 4.0, "steps": 30},

    # ── zimage-turbo (Z-Image-Turbo) ────────────────────────────────────
    # Step-distilled variant. Z-Image ships NO is_distilled marker (unlike
    # Krea-2), so `infer_model_family` detects Turbo by "turbo" in the model
    # path (ADR-009 2026-07-06). 8 steps / cfg 1.0. NOTE (ADR-024
    # correction): ZImagePipeline runs REAL classic CFG whenever
    # guidance_scale > 0 — cfg 1.0 is a genuine double-pass at scale 1, not
    # the single-pass collapse a prior comment claimed. Negative prompts DO
    # work at this default via CFG; NAG (--nag-scale) requires --cfg 0.
    # Base defaults (30/4.0) DESTROY this distill — empirically confirmed
    # (gen-validation 2026-07-06). Routes through the zimage
    # guidance_scale branch in `_build_call_kwargs`.
    "zimage-turbo": {"cfg_scale": 1.0, "steps": 8},

    # ── hunyuan-image (Hunyuan-Image 2.1) ───────────────────────────────
    # Guidance-distilled. cfg_scale routes to distilled_guidance_scale at
    # call-build time per ADR-025 §2; the family-defaults overlay still
    # operates on the canonical cfg_scale schema key (ADR-025 §4 — same
    # pattern as the flux family). cfg=3.25 / steps=50 match both the
    # HunyuanImagePipeline.__call__ signature defaults and the Tencent
    # Hunyuan-Image 2.1 model card.
    # **2K-native**: width=2048, height=2048 is mandatory, not optional —
    # Tencent README (Usage §): "HunyuanImage-2.1 only supports 2K image
    # generation (e.g. 2048x2048 for 1:1 images, 2560x1536 for 16:9 images,
    # etc.). Generating images with 1K resolution will result in artifacts."
    # The 32× spatial compression VAE was trained on 64×64 latents → 2048
    # decoded images; sub-2K renders are out-of-distribution. Documented
    # aspect buckets: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3. cfg_scale and
    # steps keep their schema-overlay semantics; width/height are this
    # family's first defaults-overlay entries for dimensions (other
    # families let the caller choose). ADR-025 Changelog 2026-05-24
    # amendment carries the empirical evidence + README citation.
    # Source: HunyuanImage-2.1 README (Usage §); ADR-025 §4 amendment.
    "hunyuan-image": {
        "cfg_scale":     3.25,
        "steps":         50,
        "width":         2048,
        "height":        2048,
        # Refiner-stage defaults per ADR-016 §(d): Tencent refiner README
        # is authoritative (cfg=3.5, steps=4); diffusers signature default
        # for refiner cfg is 3.25 but the README wins, same lesson as the
        # 2K-mandatory amendment. Both keys are no-ops when --refiner is
        # unset (chained dispatch path skipped → these never read).
        "refiner_steps": 4,
        "refiner_cfg":   3.5,
    },
}


#: Families whose FAMILY_DEFAULTS encode a FEW-STEP DISTILLED schedule — one
#: that produces an under-denoised, noisy image if applied to non-distilled
#: weights. Membership is NOT derivable from the values: krea-turbo disables
#: CFG (0.0) while zimage-turbo runs real CFG at 1.0; what they share is the
#: few-step budget that only a distill can close.
#:
#: flux2klein joined 2026-07-22 when its defaults were corrected to the
#: 4-step distilled schedule (a prior 24-step row — matching neither Klein
#: card — was deliberately absent here as "ordinary"). flux2klein-base is
#: the non-distilled sibling and stays out.
#:
#: Consumed by _apply_family_defaults to warn when a --transformer override
#: silently inherits one of these schedules from the base model's path.
#: Keep in sync with the table above.
DISTILLED_FAMILIES = frozenset({"krea-turbo", "zimage-turbo", "flux2klein"})
