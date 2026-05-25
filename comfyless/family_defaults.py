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

    # ── flux2klein (Flux.2 Klein) ───────────────────────────────────────
    # Distilled variant; tolerates fewer steps but keeps cfg shape.
    # Source: BFL Klein model card.
    "flux2klein": {"cfg_scale": 3.5, "steps": 24},

    # ── hunyuan-image (Hunyuan-Image 2.1) ───────────────────────────────
    # Guidance-distilled. cfg_scale routes to distilled_guidance_scale at
    # call-build time per ADR-014 §2; the family-defaults overlay still
    # operates on the canonical cfg_scale schema key (ADR-014 §4 — same
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
    # families let the caller choose). ADR-014 Changelog 2026-05-24
    # amendment carries the empirical evidence + README citation.
    # Source: HunyuanImage-2.1 README (Usage §); ADR-014 §4 amendment.
    "hunyuan-image": {
        "cfg_scale": 3.25,
        "steps":     50,
        "width":     2048,
        "height":    2048,
    },

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

    # ── zimage ──────────────────────────────────────────────────────────
    # New family; no published recommendation yet. Holding at flux-like
    # values until calibration data exists.
    # Source: placeholder pending empirical sweep.
    "zimage":     {"cfg_scale": 4.0, "steps": 30},
}
