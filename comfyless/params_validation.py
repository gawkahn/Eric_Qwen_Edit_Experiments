"""Canonical machine-boundary input validator for comfyless.

Per ADR-012 (accepted 2026-05-15). ONE function defines input-type rules for
every machine-facing comfyless surface: the daemon socket (server.py), the
future MCP server (mcp_server.py), and iterate's per-LoRA validation in
generate.py. No type predicate (isinstance) outside this module's two
public callables.

Boundary-asymmetry rule (ADR-012 §4): machine boundaries fail closed; the
CLI's _validate_params stays warn-and-keep for human replay. This module
serves the machine side only.

Pure function discipline (ADR-012 §5): no filesystem reads, no environment
reads, no network IO, no global state mutation. Results deterministic given
input.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Optional


# Field-kind tags. Drive type-checking in _check_field.
_KIND_STR        = "str"
_KIND_INT        = "int"
_KIND_FLOAT      = "float"
_KIND_FLOAT_NONE = "float|None"
_KIND_BOOL       = "bool"
_KIND_LIST       = "list"


# Canonical-key → kind. Mirrors COMFYLESS_SCHEMA (the sidecar-shaped fields)
# plus runtime fields the daemon adds at the wire boundary. Schema collapse
# per ADR-012 §2: cfg_scale, true_cfg_scale, and LoRA weight (in
# validate_lora_entry) declared as canonical float — the prior (int, float)
# widening on these fields is intentionally removed.
#
# All three maps are wrapped in MappingProxyType to make mutation
# structurally impossible — the validator is security truth and the canonical
# schema must not be reachable for runtime modification. See ADR-012 §5
# (pure function discipline) and step-1 security review finding 7.
SCHEMA_KIND = types.MappingProxyType({
    "model":                _KIND_STR,
    "prompt":               _KIND_STR,
    "negative_prompt":      _KIND_STR,
    "sampler":              _KIND_STR,
    "schedule":             _KIND_STR,
    "transformer_path":     _KIND_STR,
    "vae_path":             _KIND_STR,
    "text_encoder_path":    _KIND_STR,
    "text_encoder_2_path":  _KIND_STR,
    "seed":                 _KIND_INT,
    "steps":                _KIND_INT,
    "width":                _KIND_INT,
    "height":               _KIND_INT,
    "max_sequence_length":  _KIND_INT,
    "cfg_scale":            _KIND_FLOAT,
    "true_cfg_scale":       _KIND_FLOAT_NONE,
    "vae_from_transformer": _KIND_BOOL,
    "loras":                _KIND_LIST,
})


# Daemon-added runtime fields. Not in COMFYLESS_SCHEMA because they're not
# sidecar-shaped (they come from server spawn flags or argparse, not from
# replayable sidecar JSON).
_RUNTIME_KIND = types.MappingProxyType({
    "type":               _KIND_STR,
    "request_id":         _KIND_STR,
    "precision":          _KIND_STR,
    "offload_vae":        _KIND_BOOL,
    "attention_slicing":  _KIND_BOOL,
    "sequential_offload": _KIND_BOOL,
    "savepath":           _KIND_STR,
})


# MCP-surface transport-control fields (ADR-017). Neither sidecar-shaped
# (not in COMFYLESS_SCHEMA — they do not round-trip through replayable
# sidecar JSON) nor daemon-runtime fields (not added from spawn flags). They
# govern only the OPTIONAL base64 image return on the MCP generate surface.
# Registered here so the canonical validator (ADR-012: the sole owner of
# machine-boundary type predicates) type-rejects a non-bool return_image /
# non-int max_return_px BEFORE the expensive generate path runs. The daemon
# socket ignores these keys at its own logic layer; type-checking them
# centrally is harmless and keeps the "one validator" rule intact.
_MCP_TRANSPORT_KIND = types.MappingProxyType({
    "return_image":     _KIND_BOOL,
    "max_return_px":    _KIND_INT,
    "max_return_bytes": _KIND_INT,
})


_ALL_FIELDS = types.MappingProxyType(
    {**SCHEMA_KIND, **_RUNTIME_KIND, **_MCP_TRANSPORT_KIND})


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validate_machine_request / validate_lora_entry.

    ok=True  → payload is the validated input with int→float safe-casts
               applied per ADR-012 §3. error is None.
    ok=False → error is a structured dict {error, field, reason} that audit
               consumers (ADR-011 §3b stderr audit line, future MCP error
               frame in mcp_server.py) format without parsing free-text.
               payload is None.
    """
    ok: bool
    payload: Optional[dict] = None
    error: Optional[dict] = None


def _make_err(error_class: str, field: str, reason: str) -> ValidationResult:
    return ValidationResult(
        ok=False,
        error={"error": error_class, "field": field, "reason": reason},
    )


def _check_field(field: str, value: Any, kind: str):
    """Type-check one field by its canonical kind. Returns (ok, payload):
      (True, cast_value)  — value passed; for KIND_FLOAT this may be float(int).
      (False, err_result) — value rejected; second element is a ValidationResult.

    All machine-boundary type predicates appear only inside this function and
    inside validate_lora_entry — see ADR-012 §1 invariant 1.
    """
    if kind == _KIND_STR:
        if isinstance(value, str):
            return True, value
        return False, _make_err(
            "invalid_type", field,
            f"expected str, got {type(value).__name__}",
        )

    if kind == _KIND_INT:
        if isinstance(value, bool):
            return False, _make_err(
                "invalid_type", field, "bool not accepted for int field",
            )
        if isinstance(value, int):
            return True, value
        if isinstance(value, float):
            return False, _make_err(
                "invalid_type", field, "float not accepted for int field",
            )
        return False, _make_err(
            "invalid_type", field,
            f"expected int, got {type(value).__name__}",
        )

    if kind in (_KIND_FLOAT, _KIND_FLOAT_NONE):
        if kind == _KIND_FLOAT_NONE and value is None:
            return True, None
        if isinstance(value, bool):
            return False, _make_err(
                "invalid_type", field, "bool not accepted for float field",
            )
        if isinstance(value, float):
            return True, value
        if isinstance(value, int):
            # ADR-012 §3 safe cast — applied AFTER bool and int are
            # verified above; lossless within float64 mantissa precision.
            return True, float(value)
        return False, _make_err(
            "invalid_type", field,
            f"expected float, got {type(value).__name__}",
        )

    if kind == _KIND_BOOL:
        if isinstance(value, bool):
            return True, value
        return False, _make_err(
            "invalid_type", field,
            f"expected bool, got {type(value).__name__}",
        )

    if kind == _KIND_LIST:
        if isinstance(value, list):
            return True, value
        return False, _make_err(
            "invalid_type", field,
            f"expected list, got {type(value).__name__}",
        )

    return False, _make_err(
        "internal_error", field, f"unknown field kind {kind!r}",
    )


def validate_lora_entry(entry: Any, index: int) -> ValidationResult:
    """Validate one entry of a 'loras' list. Both 'path' and 'weight' required
    per ADR-012 §6 invariant 5.

    Returns a new dict with 'weight' cast to float when input was int. Rejects:
    not-a-dict, missing path/weight, wrong types, bool for weight.
    """
    if not isinstance(entry, dict):
        return _make_err(
            "invalid_type",
            f"loras[{index}]",
            f"expected dict, got {type(entry).__name__}",
        )
    if "path" not in entry:
        return _make_err(
            "missing_field", f"loras[{index}].path", "required field absent",
        )
    if "weight" not in entry:
        return _make_err(
            "missing_field", f"loras[{index}].weight", "required field absent",
        )
    path = entry["path"]
    weight = entry["weight"]
    if not isinstance(path, str):
        return _make_err(
            "invalid_type",
            f"loras[{index}].path",
            f"expected str, got {type(path).__name__}",
        )
    if isinstance(weight, bool):
        return _make_err(
            "invalid_type",
            f"loras[{index}].weight",
            "bool not accepted for float field",
        )
    if isinstance(weight, float):
        weight_cast: float = weight
    elif isinstance(weight, int):
        weight_cast = float(weight)
    else:
        return _make_err(
            "invalid_type",
            f"loras[{index}].weight",
            f"expected float, got {type(weight).__name__}",
        )
    cleaned = dict(entry)
    cleaned["weight"] = weight_cast
    return ValidationResult(ok=True, payload=cleaned)


def validate_machine_request(payload: Any) -> ValidationResult:
    """Validate a machine-boundary request payload against the canonical schema.

    Pure function: deterministic; no IO, no env reads, no global mutation.

    Returns:
      ValidationResult(ok=True, payload=<validated payload>) — payload is a
        new dict; canonical-float fields cast from int → float when applicable;
        loras list (if present) has each entry's weight cast to float when
        input was int.
      ValidationResult(ok=False, error=<structured dict>) — caller emits the
        error per its own transport (IPC string, MCP error frame, ...).

    Unknown keys pass through unchanged. Unknown-key rejection is out of
    scope for this slice (Vision out-of-scope: schema versioning).
    """
    if not isinstance(payload, dict):
        return _make_err(
            "invalid_payload",
            "<root>",
            f"expected dict, got {type(payload).__name__}",
        )

    validated: dict = {}
    for key, value in payload.items():
        kind = _ALL_FIELDS.get(key)
        if kind is None:
            validated[key] = value
            continue
        ok, payload_or_err = _check_field(key, value, kind)
        if not ok:
            return payload_or_err
        validated[key] = payload_or_err

    if "loras" in validated and isinstance(validated["loras"], list):
        cleaned_loras = []
        for i, entry in enumerate(validated["loras"]):
            lora_result = validate_lora_entry(entry, i)
            if not lora_result.ok:
                return lora_result
            cleaned_loras.append(lora_result.payload)
        validated["loras"] = cleaned_loras

    return ValidationResult(ok=True, payload=validated)
