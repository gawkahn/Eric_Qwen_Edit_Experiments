"""Comfyless sidecar schema — single source of canonical-key + type truth.

Thin adapter built on top of `comfyless.params_validation.SCHEMA_KIND`. The
validator owns the canonical type rules (which fields are which kind);
this module exposes the same information in the tuple shape that
`comfyless.generate._validate_params` and downstream readers consume, plus
the per-field defaults and the CLI argparse-attr → canonical-key map.

Three-field collapse per ADR-012 §2 (accepted 2026-05-15):
  cfg_scale       was ((int, float),                3.5)  → (float,             3.5)
  true_cfg_scale  was ((int, float, type(None)),    None) → ((float, type(None)), None)
  loras[].weight   — declared canonical-float by `validate_lora_entry`; the
                     COMFYLESS_SCHEMA entry remains `(list, [])` (the
                     per-entry shape is enforced by the validator, not by
                     COMFYLESS_SCHEMA's outer tuple).

The CLI's `_validate_params` is warn-and-keep (ADR-012 §4 + Vision invariant
7) — the collapse may grow its warning set on sidecars that previously
silently accepted `cfg_scale: 7` (int), but the acceptance set is
unchanged: every value the prior validator kept, this one still keeps.
"""
from __future__ import annotations

import types
from typing import Dict

from comfyless.params_validation import (
    SCHEMA_KIND,
    _KIND_STR,
    _KIND_INT,
    _KIND_FLOAT,
    _KIND_FLOAT_NONE,
    _KIND_BOOL,
    _KIND_LIST,
)


# Adapter: kind → the `expected_type` shape COMFYLESS_SCHEMA's first tuple
# element holds. `_validate_params` and `_extract_eric_save_params` pass
# this directly to `isinstance(value, expected_type)`, so the shape must be
# something `isinstance` accepts (a single type or a tuple of types).
_KIND_TO_EXPECTED: Dict[str, object] = {
    _KIND_STR:        str,
    _KIND_INT:        int,
    _KIND_FLOAT:      float,
    _KIND_FLOAT_NONE: (float, type(None)),
    _KIND_BOOL:       bool,
    _KIND_LIST:       list,
}


# Per-field defaults. Paired with the kind from SCHEMA_KIND to build the
# COMFYLESS_SCHEMA tuple shape. Required fields have `None` defaults — they
# must be supplied at request time.
_FIELD_DEFAULTS: Dict[str, object] = {
    "model":                None,
    "prompt":               None,
    "negative_prompt":      "",
    "seed":                 -1,
    "steps":                28,
    "cfg_scale":            3.5,
    "true_cfg_scale":       None,
    "width":                1024,
    "height":               1024,
    "sampler":              "default",
    "schedule":             "linear",
    "max_sequence_length":  512,
    "transformer_path":     "",
    "vae_path":             "",
    "text_encoder_path":    "",
    "text_encoder_2_path":  "",
    "refiner_path":         "",
    "vae_from_transformer": False,
    "loras":                [],
    # Refiner schema keys per ADR-016 §(d) defaults (Tencent refiner
    # README authoritative — diffusers signature default for cfg is 3.25,
    # README wins; same lesson as the 2K-mandatory amendment in ADR-014).
    # Both are no-ops when --refiner is unset; activated only when the
    # chained dispatch path runs.
    "refiner_steps":        4,
    "refiner_cfg":          3.5,
}


# Sanity guard: every SCHEMA_KIND entry must have a default. Mismatch =
# stale module — fail loudly at import rather than producing partial schema.
_missing = set(SCHEMA_KIND) - set(_FIELD_DEFAULTS)
_extra = set(_FIELD_DEFAULTS) - set(SCHEMA_KIND)
if _missing or _extra:
    raise RuntimeError(
        f"comfyless.params_schema drift: SCHEMA_KIND vs _FIELD_DEFAULTS — "
        f"missing defaults: {sorted(_missing)}; extra defaults: {sorted(_extra)}"
    )
del _missing, _extra


# Canonical sidecar schema. Built from SCHEMA_KIND (canonical kinds) and
# `_FIELD_DEFAULTS` (per-field defaults). `expected_type` is the isinstance
# shape; downstream readers (e.g. `_validate_params`) treat it opaquely.
#
# This dict is the long-standing public contract of `comfyless.generate`; it
# now lives here for single-source ownership but is still imported and
# re-exported from `comfyless.generate` for backward compatibility.
#
# Wrapped in MappingProxyType — same rationale as `SCHEMA_KIND`: the
# canonical schema is consumed by `_validate_params` and downstream tools,
# and structural immutability prevents an in-process caller from silently
# weakening the warn-set (step-2 code-reviewer F2).
COMFYLESS_SCHEMA = types.MappingProxyType({
    key: (_KIND_TO_EXPECTED[SCHEMA_KIND[key]], _FIELD_DEFAULTS[key])
    for key in SCHEMA_KIND
})


# CLI argparse-attr name → canonical schema key. Only names that DIFFER from
# their canonical target are listed; identical pairs fall back through
# `getattr(args, canonical_key)` in `_cli_value_for`.
_CLI_TO_CANONICAL: Dict[str, str] = {
    "cfg":         "cfg_scale",
    "true_cfg":    "true_cfg_scale",
    "max_seq_len": "max_sequence_length",
    "transformer": "transformer_path",
    "vae":         "vae_path",
    "te1":         "text_encoder_path",
    "te2":         "text_encoder_2_path",
    "lora":        "loras",
    "refiner":     "refiner_path",
}
