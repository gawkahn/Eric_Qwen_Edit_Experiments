#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""comfyless MCP server (stdio transport).

Adapts the comfyless generate surface to the Model Context Protocol so that
LLM-driven tool-call harnesses (Claude Desktop, local_agents, etc.) can
drive image generation through a JSONSchema-typed interface with full
path-allowlist enforcement and audit-line discipline.

Slice 1 step 1: SKELETON. The `generate` tool is advertised and its input
schema is declared, but its handler raises NotImplementedError. Step 2
wires the real handler; step 3 extends it with cascade dispatch.

Per slice-1 Vision invariant 14: this module does NOT import argparse and
does NOT call into _run_cli_mode / _apply_overrides / _load_params_file
from comfyless.generate. MCP and CLI are separate parallel surfaces.

See:
  docs/decisions/ADR-011-comfyless-mcp-server.md
  docs/vision/slice-1-mcp-generate.md
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
from typing import Any, Optional

import click
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Reuse the daemon's path-allowlist primitive (slice-1 Vision: "reuses the
# daemon's _within helper verbatim — does not re-implement"). The leading
# underscore is module-private convention in comfyless.server; importing it
# from a peer module inside the comfyless package is acceptable as a
# co-package consumer of the same security primitive. Future refactor may
# promote _within to a shared helper module; out of slice-1 scope.
from comfyless.server import _within


# ════════════════════════════════════════════════════════════════════════
# Shared redaction state — single source of truth (invariant 12)
# ════════════════════════════════════════════════════════════════════════

# Path-typed fields whose values are paths that should be reduced to
# basenames before PNG-embedding (step 2 / invariant 12) AND that should
# be retained in audit lines (step 1+ / invariant 5: audit retains path
# fields; only prompt/negative_prompt are dropped).
_MCP_PATH_TYPED_FIELDS = (
    "model",
    "transformer_path",
    "vae_path",
    "text_encoder_path",
    "text_encoder_2_path",
)

# Cascade-specific path-typed fields nested under cascade_config (step 3).
# Same redaction rules as _MCP_PATH_TYPED_FIELDS, but applied to the
# cascade_config sub-dict rather than the top-level metadata.
_MCP_CASCADE_PATH_TYPED_FIELDS = (
    "stage_c",
    "stage_b",
    "stage_a",
    "scaffolding_repo",
)

# Reference field NAMES removed from the MCP `generate` surface in slice 3
# (ADR-015 OQ-A). `transformer_path` is superseded by the catalog name
# `transformer`; the three component overrides have no catalog kind and are
# dropped (the CLI retains them). Sending any of these is a CONTRACT error,
# named explicitly: field names are public schema knowledge, so this leaks
# nothing about the filesystem (unlike a reference VALUE). Rejecting — not
# silently ignoring — is required: silently accepting a raw `vae_path` would
# reintroduce the caller-supplied-path input attack surface ADR-015 removes.
_GENERATE_REMOVED_FIELDS = (
    "transformer_path",
    "vae_path",
    "text_encoder_path",
    "text_encoder_2_path",
)

# Fields dropped entirely from audit lines per invariant 5.
_AUDIT_DROPPED_FIELDS = frozenset({"prompt", "negative_prompt"})

# Audit-line write-failure counter (invariant 5: write failure does not
# block the request; the count is exposed for the next slice's security
# review). A list of monotonic timestamps lets future slices add
# rate-style alerting without a re-architecture.
_audit_write_failures: list = []


# ════════════════════════════════════════════════════════════════════════
# PNG metadata redaction for MCP-returned images  (invariant 12)
# ════════════════════════════════════════════════════════════════════════

def _basename_or_repo_id(value: str) -> str:
    """If value is an HF repo ID, return it unchanged (per invariant 12 / N30);
    else return os.path.basename(value)."""
    if not isinstance(value, str) or not value:
        return value
    # Local-import keeps the comfyless.mcp_server module light when imported
    # from comfyless.generate's lazy-import path (avoid pulling diffusers
    # transitively at module import time).
    from nodes.eric_diffusion_utils import _is_hf_repo_id
    if _is_hf_repo_id(value):
        return value
    return os.path.basename(value)


def redact_metadata_for_png(metadata: dict) -> dict:
    """Apply invariant 12's redaction map to a generation-metadata dict.

    Called by comfyless.generate._save_with_metadata when its mcp_caller
    flag is True. Returns a NEW dict (does not mutate the input). Single
    source of truth for which fields are path-typed at the MCP boundary
    (cf. _MCP_PATH_TYPED_FIELDS); step 3 will extend this when cascade
    fields land.

    Rules:
      - Path-typed top-level fields → basename, or pass-through if the
        original value was an HF repo ID (N30).
      - loras[].path → basename per entry (or HF repo-ID pass-through).
      - output_path, savepath → DROPPED entirely (invariant 12 / N27).
      - All other fields → retained verbatim (invariant 12 / N28).
    """
    out: dict = dict(metadata)
    for field in _MCP_PATH_TYPED_FIELDS:
        if field in out and out[field]:
            out[field] = _basename_or_repo_id(out[field])
    if "loras" in out and isinstance(out["loras"], list):
        out["loras"] = [
            {**l, "path": _basename_or_repo_id(l.get("path", ""))}
            for l in out["loras"]
        ]
    # Cascade-side redaction (step 3): basename each cascade_config.stage_*
    # / scaffolding_repo. Non-path fields inside cascade_config (e.g.
    # prior_steps, decoder_steps, prior_dtype) are retained verbatim.
    if "cascade_config" in out and isinstance(out["cascade_config"], dict):
        cc = dict(out["cascade_config"])
        for field in _MCP_CASCADE_PATH_TYPED_FIELDS:
            if field in cc and cc[field]:
                cc[field] = _basename_or_repo_id(cc[field])
        out["cascade_config"] = cc
    out.pop("output_path", None)
    out.pop("savepath", None)
    return out


def _resolved_params_as_names(
    metadata: dict,
    *,
    model_name: str,
    transformer_name: Optional[str],
    lora_names: list,
) -> dict:
    """Render generate()'s metadata blob with weight references as catalog
    NAMES instead of abs_paths (ADR-015 §3 / slice-3 invariant 5).

    This is the MCP-RESPONSE renderer (the agent's authoritative record),
    distinct from `redact_metadata_for_png` (the on-disk PNG sink, which
    basenames). Returns a NEW dict. ONLY the path-typed weight fields are
    rewritten:
      - `model`            -> the resolved catalog name
      - `transformer_path` -> dropped; replaced by `transformer` = the
                              resolved name (omitted entirely when no
                              transformer was used)
      - vae_path / text_encoder_path / text_encoder_2_path -> dropped
        (removed from the MCP surface per OQ-A; they are "" here regardless)
      - loras[]            -> [{name, weight}] (the `path` key is dropped)
    Every other field (resolved seed, model_family, timing, lora_warnings,
    sampler, ...) passes through verbatim. No abs_path crosses the boundary.
    """
    out = dict(metadata)
    out["model"] = model_name
    out.pop("transformer_path", None)
    if transformer_name:
        out["transformer"] = transformer_name
    out.pop("vae_path", None)
    out.pop("text_encoder_path", None)
    out.pop("text_encoder_2_path", None)
    # `lora_warnings` strings embed the resolved abs_path (generate.py:
    # "LoRA skipped ...: <abs_path>"). Drop them from the agent-facing blob —
    # an abs_path must not cross the boundary (invariant 5). LOAD-BEARING, not
    # dead code: the MCP cached-pipeline loader (_get_or_load_cached_pipeline)
    # DOES run the warning-producing loop (_apply_loras) on every cache miss, so
    # this list can be non-empty; this pop is the enforced boundary that keeps
    # those abs_paths out of the response. Removing it re-opens the egress
    # (regression test N11). (security-auditor slice-3 step-2 MEDIUM-1,
    # 2026-06-02; reaffirmed 2026-06-27 when the cached loader began applying
    # LoRAs.) The warnings remain on the operator's PNG metadata / stderr.
    out.pop("lora_warnings", None)
    src_loras = metadata.get("loras") or []
    out["loras"] = [
        {"name": lora_names[i], "weight": src_loras[i].get("weight")}
        for i in range(len(src_loras))
    ]
    return out


def _resolved_cascade_params_as_names(
    metadata: dict,
    *,
    stage_names: dict,
) -> dict:
    """Render the cascade metadata blob with `cascade_config` stage references
    as catalog NAMES instead of abs_paths (ADR-015 §3 / slice-3b invariant 1/4).

    The MCP-RESPONSE renderer (the agent's authoritative record), distinct from
    `redact_metadata_for_png` (the on-disk PNG sink, which basenames). Returns a
    NEW dict. Only the `cascade_config` sub-dict is rewritten:
      - `stage_c` / `stage_b` / `stage_a` -> the resolved catalog name
        (`stage_a` omitted entirely when unused)
      - `scaffolding_repo`                -> DROPPED. It is the operator-default
        architecture config, not agent-affectable (slice-3b removed-field rule),
        and must not cross the boundary as a path.
    Every other `cascade_config` field (dtypes, prior/decoder steps + cfg,
    width/height, timing) and every top-level field pass through verbatim. No
    abs_path crosses the boundary.
    """
    out = dict(metadata)
    cc_out = dict(metadata.get("cascade_config") or {})
    for stage in ("stage_c", "stage_b", "stage_a"):
        if stage in stage_names:
            cc_out[stage] = stage_names[stage]
        else:
            cc_out.pop(stage, None)
    cc_out.pop("scaffolding_repo", None)
    out["cascade_config"] = cc_out
    return out


# ════════════════════════════════════════════════════════════════════════
# Tool description text (refinable per ADR-011 §2 amendment 2026-04-30)
# ════════════════════════════════════════════════════════════════════════

_GENERATE_TOOL_DESCRIPTION = """\
Generate an image from a text prompt. Covers all comfyless model families:
qwen-image, flux, flux2, chroma, krea / krea-turbo, and Stable Cascade
(via cascade_config).

Model selection guidance:
- Text rendering + photorealism: qwen-image (Qwen-Image-2512)
- Anime / manga / illustration: Illustrious, Pony, or Chroma
- Fastest at modest quality: Stable Cascade via cascade_config
- Few-step / distilled: krea-turbo (Krea-2-Turbo, ~8 steps, CFG off)
- General-purpose / latest: flux2 (Flux.2)

Per-family defaults are applied automatically for any generation parameter
you omit (e.g. steps / cfg_scale / true_cfg_scale), so omitting them yields
each model's recommended values; anything you pass explicitly wins. Krea-2
in particular: Raw wants 52 steps / cfg 3.5, Turbo wants 8 steps / cfg 0.0
— just omit steps and cfg_scale to get them.

If `model` is omitted, the server uses the model configured at spawn time
via --default-model. Omitting `model` without a configured default
returns an error.

Weight references (`model`, `transformer`, `loras[].name`) are CATALOG
NAMES, not filesystem paths — discover them via `list_models`,
`list_transformers`, and `list_loras`. A path-shaped value has its
directory component discarded and its basename resolved through the
catalog; rely on the names, not on any path. A reference that does not
resolve returns a single uniform "reference not available" error.

Output paths (`savepath`) must resolve under --output-dir. HuggingFace
downloads are not performed; weights must be local or already cached.
"""


# ════════════════════════════════════════════════════════════════════════
# Generate tool input schema (JSONSchema)
# ════════════════════════════════════════════════════════════════════════
#
# Slice-1 Vision invariant 6: only `generate` is advertised in slice 1.
# Invariant 14 (N14): max_iterations is iterate-only and must NOT appear
# in this schema; `additionalProperties: False` structurally enforces that.

_GENERATE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["prompt"],
    "additionalProperties": False,
    "properties": {
        "prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "model": {
            "type": "string",
            "description": (
                "Catalog name of a model (discover via list_models). A "
                "path-shaped value has its directory discarded and its "
                "basename resolved via the catalog. Optional if "
                "--default-model is configured at spawn."
            ),
        },
        "transformer": {
            "type": "string",
            "description": (
                "Catalog name of a single-file diffusion-transformer (DiT) "
                "weight (discover via list_transformers). A path-shaped "
                "value has its directory discarded and its basename "
                "resolved via the catalog. Optional."
            ),
        },
        "seed": {"type": "integer"},
        "steps": {"type": "integer"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "cfg_scale": {"type": "number"},
        "true_cfg_scale": {"type": ["number", "null"]},
        "max_sequence_length": {"type": "integer"},
        "sampler": {"type": "string"},
        "schedule": {"type": "string"},
        "vae_from_transformer": {"type": "boolean"},
        "loras": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "weight"],
                "additionalProperties": False,
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Catalog name of a LoRA adapter (discover via "
                            "list_loras). A path-shaped value has its "
                            "directory discarded and its basename resolved "
                            "via the catalog."
                        ),
                    },
                    "weight": {"type": "number"},
                },
            },
        },
        "savepath": {"type": "string"},
        "return_image": {
            "type": "boolean",
            "description": (
                "When true, the response includes the generated image as a "
                "base64 PNG (`image_b64` + `image_mime`) in addition to "
                "`output_path`. Default false — when false/absent the "
                "response is unchanged. The full-resolution PNG on disk is "
                "never affected; only a size-bounded transport copy is "
                "returned (see max_return_px)."
            ),
        },
        "max_return_px": {
            "type": "integer",
            "description": (
                "Longest-edge cap (pixels) for the base64 image returned "
                "when return_image=true. Default 768. Aspect is preserved "
                "and the image is never upscaled; the on-disk PNG keeps its "
                "full resolution. Ignored when return_image is false/absent."
            ),
        },
        "max_return_bytes": {
            "type": "integer",
            "description": (
                "Hard ceiling (bytes) on the base64 image payload returned "
                "when return_image=true. The server enforces a 1 MiB "
                "ceiling; a smaller value is honored, a larger value is "
                "clamped down to 1 MiB. If the pixel-bounded copy still "
                "exceeds this budget the image is downscaled further until "
                "it fits. The on-disk PNG is never shrunk. Ignored when "
                "return_image is false/absent."
            ),
        },
        "cascade_config": {
            "type": "object",
            "description": (
                "Stable Cascade dispatch config (inline JSON object). "
                "Required when targeting cascade. Slice 3 wires the "
                "dispatch handler; slice 1 ships the schema slot."
            ),
        },
        "quant": {
            "type": "string",
            "enum": ["none", "fp8"],
            "description": (
                "Quantize-on-load (ADR-019). fp8 halves VRAM on the "
                "transformer + large text encoders using native fp8 tensor "
                "cores; VAE and CLIP encoders are never quantized. Falls "
                "back to unquantized with a warning on unsupported "
                "hardware. Non-cascade models only (ignored for cascade). "
                "Default none."
            ),
        },
        "quant_skip": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 32,
            "description": (
                "Component slot names (e.g. 'text_encoder') to exclude "
                "from quantization. For isolating quality regressions. "
                "Slot names only — never paths."
            ),
        },
        "quant_only": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 32,
            "description": (
                "Quantize exactly these component slots, overriding the "
                "default eligible set. 'vae' is refused even here. Slot "
                "names only — never paths."
            ),
        },
    },
}


# ════════════════════════════════════════════════════════════════════════
# list_models / list_loras tool surfaces (slice 2 step 4) + list_transformers (slice 2b)
# ════════════════════════════════════════════════════════════════════════
#
# These are the agent's discovery channel: read-only enumeration of the
# spawn-time catalog. Inputs are empty objects (`additionalProperties:
# false`) — the agent calls without arguments and the server returns the
# kind-filtered slice of the catalog. Vision invariants 8, 9 govern the
# response shape: name, kind, source — plus model_family (list_models
# only, when known from scan-time class detection) and target_family
# (list_loras only, manifest-declared entries only). NO abs_path / path /
# any filesystem string under any other key.

_LIST_MODELS_TOOL_DESCRIPTION = """\
Enumerate the diffusers-pipeline models the server knows about (the
`kind:"model"` entries in the spawn-time catalog).

Use this to discover what `model` values `generate` will accept once
slice-3 migrates `generate` to catalog-resolved name references. In
slice 2 the catalog is built but `generate` still consumes raw paths,
so calling `list_models` does NOT yet change how you call `generate`;
it lets you preview the named surface that will become required input.

Returns a JSON array of `{name, kind, source[, model_family]}` objects:
- `name`: agent-facing identifier (Unicode-NFC catalog key)
- `kind`: always `"model"` for this tool
- `source`: `"scan"` (auto-detected under --model-base) or `"manifest"`
  (declared in the operator's --catalog file)
- `model_family`: present when known from scan-time class detection
  (e.g. `"qwen-image"`, `"flux2"`, `"chroma"`); absent for entries the
  server could not classify

No inputs. No path-typed fields ever appear in the response.
"""

_LIST_MODELS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}

_LIST_LORAS_TOOL_DESCRIPTION = """\
Enumerate the LoRA adapters the server knows about (the `kind:"lora"`
entries in the spawn-time catalog).

Use this to discover what LoRA names can be cited once slice-3 migrates
`generate.loras[].path` to catalog-resolved name references. In slice 2
the catalog is built but `generate` still consumes raw `.path` strings;
calling `list_loras` previews the named surface that will become
canonical input.

Returns a JSON array of `{name, kind, source[, target_family]}` objects:
- `name`: agent-facing identifier (Unicode-NFC catalog key)
- `kind`: always `"lora"` for this tool
- `source`: `"scan"` (file found under --model-base/loras) or
  `"manifest"` (declared in the operator's --catalog file)
- `target_family`: present ONLY for manifest entries that explicitly
  declared a target diffusion family (e.g. `"qwen-image"`). Scan-
  derived LoRAs omit this field — there is no inference from
  filesystem layout or weight introspection in this slice.

No inputs. No path-typed fields ever appear in the response.
"""

_LIST_LORAS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}

_LIST_TRANSFORMERS_TOOL_DESCRIPTION = """\
Enumerate the single-file diffusion-transformer (DiT) weights the server
knows about (the `kind:"transformer"` entries in the spawn-time catalog).

These are the standalone `.safetensors` transformer checkpoints (scanned
from `--model-base/checkpoints` and `--model-base/diffusion_models`, or
declared in the operator's --catalog file) that `generate` will accept as
`transformer_path` values once slice-3 migrates `generate` to catalog-
resolved name references. In slice 2/2b the catalog is built but `generate`
still consumes raw paths, so calling `list_transformers` does NOT yet change
how you call `generate`; it previews the named surface that will become
required input.

Returns a JSON array of `{name, kind, source[, model_family]}` objects:
- `name`: agent-facing identifier (Unicode-NFC catalog key)
- `kind`: always `"transformer"` for this tool
- `source`: `"scan"` (auto-detected under --model-base/checkpoints or
  /diffusion_models) or `"manifest"` (declared in the --catalog file)
- `model_family`: present ONLY for manifest entries that explicitly
  declared one (e.g. `"flux2"`). Scan-derived transformers omit this
  field — a single-file DiT weight carries no model_index.json, so there
  is no scan-time family classification.

No inputs. No path-typed fields ever appear in the response.
"""

_LIST_TRANSFORMERS_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}


# ════════════════════════════════════════════════════════════════════════
# Audit-line writer  (invariants 4, 5)
# ════════════════════════════════════════════════════════════════════════

def _emit_audit_line(
    tool: str,
    payload: dict,
    *,
    status: str,
    error_class: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    result_count: Optional[int] = None,
) -> None:
    """Write one structured audit line to stderr.

    Invariant 5 contract:
      - stderr only; stdout is reserved for JSON-RPC frames
      - one line per invocation, success or rejection alike
      - `prompt` and `negative_prompt` are dropped
      - audit-line write failure does NOT block the request
        (failure increments _audit_write_failures for next slice's review)

    The payload is the raw argument dict passed to the tool (or an empty
    dict if the call failed before argument resolution). Path-typed
    fields are retained verbatim per invariant 5.

    `result_count` is the slice-2 step-4 invariant-19 hook: `list_*`
    handlers pass the number of entries returned to the agent so the
    audit line carries tool + count + status + elapsed without exposing
    the catalog's `abs_path` values. Omit (default `None`) for tools
    where it does not apply (e.g. `generate`).
    """
    redacted = {k: v for k, v in payload.items() if k not in _AUDIT_DROPPED_FIELDS}
    line: dict[str, Any] = {
        "tool": tool,
        "status": status,
        "input": redacted,
    }
    if error_class is not None:
        line["error_class"] = error_class
    if elapsed_seconds is not None:
        line["elapsed_seconds"] = round(elapsed_seconds, 3)
    if result_count is not None:
        line["count"] = result_count
    try:
        sys.stderr.write(json.dumps(line, default=str) + "\n")
        sys.stderr.flush()
    except Exception:
        _audit_write_failures.append(time.monotonic())


# ════════════════════════════════════════════════════════════════════════
# Traceback-strip helper  (invariant 13)
# ════════════════════════════════════════════════════════════════════════

def _sanitize_error(exc: BaseException, label: str) -> str:
    """Convert an internal exception to a sanitized MCP-facing message.

    Invariant 13 contract: the returned string MUST NOT contain
      - "Traceback (most recent call last)"
      - `.py:NNN` patterns
      - absolute paths starting with /home/, /root/, or --model-base
      - internal module names from the traceback

    The full traceback is written to stderr (audit stream) so operators
    retain visibility while the agent gets only a category-level message.
    """
    try:
        sys.stderr.write(f"[comfyless-mcp] internal exception ({label}):\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
    except Exception:
        _audit_write_failures.append(time.monotonic())
    return f"{label}: {type(exc).__name__}"


# ════════════════════════════════════════════════════════════════════════
# Startup validation  (invariants 1, 10)
# ════════════════════════════════════════════════════════════════════════

class _StartupConfig:
    """Resolved + validated spawn-time configuration.

    `catalog` is a `comfyless.catalog.CatalogDict` (typed as `dict` here
    to keep this module's top-level import surface free of catalog.py —
    `build_catalog` is lazy-imported inside `_validate_startup_args`).
    """

    __slots__ = (
        "output_dir",
        "model_base",
        "default_model",
        "mcp_max_iterations",
        "catalog",
    )

    def __init__(
        self,
        output_dir: str,
        model_base: str,
        default_model: Optional[str],
        mcp_max_iterations: int,
        catalog: dict,
    ) -> None:
        self.output_dir = output_dir
        self.model_base = model_base
        self.default_model = default_model
        self.mcp_max_iterations = mcp_max_iterations
        self.catalog = catalog


def _validate_startup_args(
    output_dir: str,
    model_base: str,
    default_model: Optional[str],
    mcp_max_iterations: int,
    catalog: Optional[str] = None,
) -> _StartupConfig:
    """Resolve + validate spawn-time CLI args. Raises click.BadParameter on bad input.

    Invariant 1: fail-closed on missing/non-existent/non-directory for the
    two required roots and (when set) --default-model.
    Invariant 10: --default-model must realpath-resolve under --model-base.

    Slice-2 Step 3: when `catalog` is supplied, the operator-manifest path
    feeds `comfyless.catalog.build_catalog(model_base, catalog_path)`. Any
    `CatalogBuildError` (manifest missing, malformed JSON, schema fail,
    name collision, scan/manifest collision, symlink escape) is wrapped
    into a `click.BadParameter(param_hint="--catalog")` with the catalog
    layer's operator-facing message passed through verbatim. Default
    `None` means "no manifest; scan-only catalog" — existing test call
    sites that don't pass `catalog` get the scan-only behaviour without
    edits.
    """
    resolved_out = os.path.realpath(output_dir)
    if not os.path.isdir(resolved_out):
        raise click.BadParameter(
            f"does not resolve to a directory: {output_dir!r}",
            param_hint="--output-dir",
        )

    resolved_base = os.path.realpath(model_base)
    if not os.path.isdir(resolved_base):
        raise click.BadParameter(
            f"does not resolve to a directory: {model_base!r}",
            param_hint="--model-base",
        )

    resolved_default: Optional[str] = None
    if default_model is not None:
        resolved_default = os.path.realpath(default_model)
        if not os.path.isdir(resolved_default):
            raise click.BadParameter(
                f"does not resolve to a directory: {default_model!r}",
                param_hint="--default-model",
            )
        if not _within(resolved_default, resolved_base):
            raise click.BadParameter(
                "escapes --model-base after realpath",
                param_hint="--default-model",
            )

    # --catalog: explicit NUL-byte pre-check before any os.* / open() so
    # direct in-process callers (tests, future internal use) get a clean
    # click.BadParameter instead of the raw `ValueError('embedded null
    # byte')` that click.Path would raise at parse time. Mirrors the
    # _handle_generate NUL-handling pattern.
    if catalog is not None and "\x00" in catalog:
        raise click.BadParameter(
            "contains embedded NUL byte",
            param_hint="--catalog",
        )

    # Lazy import: keeps mcp_server's module-top surface unchanged for
    # any future import-time tests, and matches the lazy-import idiom
    # already used elsewhere in startup validation.
    from comfyless.catalog import build_catalog, CatalogBuildError

    try:
        built_catalog = build_catalog(resolved_base, catalog)
    except CatalogBuildError as e:
        # `from None` suppresses the CatalogBuildError chain so click's
        # pretty-printed error stays clean. The catalog layer's message
        # is operator-facing (names the offending entry / repo ID) and
        # passes through verbatim — this is stderr, not the agent-facing
        # uniform-error contract.
        raise click.BadParameter(str(e), param_hint="--catalog") from None

    return _StartupConfig(
        output_dir=resolved_out,
        model_base=resolved_base,
        default_model=resolved_default,
        mcp_max_iterations=mcp_max_iterations,
        catalog=built_catalog,
    )


# ════════════════════════════════════════════════════════════════════════
# Tool handlers (impls callable directly by tests; framework decoration in
# _build_server)
# ════════════════════════════════════════════════════════════════════════

async def _list_tools_impl(cfg: _StartupConfig) -> list[Tool]:
    """Slice-2b invariant 1 (updates slice-2 invariant 8's count 3→4):
    advertise exactly FOUR tools — `generate` (slice 1, schema and
    description unchanged per slice-2 Vision invariant 14), `list_models`
    (slice 2 step 4, `kind:"model"` discovery), `list_loras` (slice 2
    step 4, `kind:"lora"` discovery), and `list_transformers` (slice 2b,
    `kind:"transformer"` discovery).

    `kind:"transformer"` entries were built and held dormant by slice 2
    (slice-2 Vision invariant 9 / N27); slice 2b surfaces them through
    `list_transformers` ONLY — they continue to be excluded from
    `list_models` and `list_loras`.
    """
    return [
        Tool(
            name="generate",
            description=_GENERATE_TOOL_DESCRIPTION,
            inputSchema=_GENERATE_INPUT_SCHEMA,
        ),
        Tool(
            name="list_models",
            description=_LIST_MODELS_TOOL_DESCRIPTION,
            inputSchema=_LIST_MODELS_INPUT_SCHEMA,
        ),
        Tool(
            name="list_loras",
            description=_LIST_LORAS_TOOL_DESCRIPTION,
            inputSchema=_LIST_LORAS_INPUT_SCHEMA,
        ),
        Tool(
            name="list_transformers",
            description=_LIST_TRANSFORMERS_TOOL_DESCRIPTION,
            inputSchema=_LIST_TRANSFORMERS_INPUT_SCHEMA,
        ),
    ]


class _MCPHandlerError(Exception):
    """Known-shape handler error carrying both an audit error_class string
    and an already-sanitized message for the agent.

    Step-1 reviewer carry-forward F1/F4: every exception path inside
    _call_tool_impl must reach the framework's outer except with a SAFE
    str(e); _MCPHandlerError exists so known-shape errors carry their own
    sanitized message instead of going through _sanitize_error.
    """

    def __init__(self, error_class: str, safe_message: str) -> None:
        self.error_class = error_class
        self.safe_message = safe_message
        super().__init__(safe_message)


# ════════════════════════════════════════════════════════════════════════
# Uniform reference-resolution error + path-discard notice (ADR-015 §2/§3)
# ════════════════════════════════════════════════════════════════════════

# The load-bearing uniform agent-facing reference-resolution error (ADR-015
# §2 step 2 / HIGH-1). EVERY reference-resolution failure — whatever its
# fine-grained cause — returns this BYTE-IDENTICAL message to the agent; the
# fine cause rides only on the stderr audit line (via
# _MCPHandlerError.error_class -> _emit_audit_line). Keeping the message in a
# single constant is what makes the byte-equality property (keystone test N5)
# auditable and closes the HF-cache enumeration oracle (TECH_DEBT 2026-05-17).
_UNIFORM_REFERENCE_ERROR = "reference not available"


def _reference_error(cause: str) -> _MCPHandlerError:
    """Build the uniform agent-facing reference error. `cause` is a
    `comfyless.catalog.ResolveCause` (UnknownName / KindMismatch /
    MalformedReference / PathMoved / WithinFailure) and lands ONLY on the
    audit line; the agent sees `_UNIFORM_REFERENCE_ERROR` regardless."""
    return _MCPHandlerError(cause, _UNIFORM_REFERENCE_ERROR)


# Path-discard INFO notice (ADR-015 §2 step 2 Hit notice / INFO-2). Interpolates
# the RESOLVED CATALOG NAME only — NEVER the agent-supplied raw reference value,
# which may carry attacker-chosen directory text that must not round-trip into
# the agent transcript.
_REFERENCE_PATH_DISCARD_NOTICE = (
    "reference '{name}' resolved via catalog; supplied path discarded — "
    "do not rely on paths for later actions."
)


def _discard_notice(name: str) -> dict:
    """INFO notice for a reference resolved from a path-shaped value. `name`
    is the resolved catalog name (NEVER the agent-supplied raw value)."""
    return {
        "level": "INFO",
        "message": _REFERENCE_PATH_DISCARD_NOTICE.format(name=name),
    }


# ════════════════════════════════════════════════════════════════════════
# Optional base64 image return  (ADR-017)
# ════════════════════════════════════════════════════════════════════════

# Default longest-edge bound for the returned transport copy (ADR-017; default
# lowered 1024 → 768 in the 2026-06-25 byte-bound refinement so the iterative
# byte-downscale below almost never fires). Also the fallback when
# max_return_px is absent or a nonsensical (<= 0) value — the bound is
# load-bearing, so an invalid value clamps to the default, never unbounded.
_DEFAULT_MAX_RETURN_PX = 768

# Hard ceiling (bytes) on the returned base64 payload length. Matches the
# local_agents MCP-proxy 1 MiB result cap (docs/specs/mcp-proxy-v1.md) and the
# scope-A "each tool bounds its own output" convention (mcp-server-v1.md). A
# request MAY ask for a SMALLER cap; a LARGER value is clamped DOWN to this
# ceiling — the agent can never raise it, so it is a true hard cap. Default
# (absent / nonsensical) = this ceiling.
_RETURN_BYTES_CEILING = 1024 * 1024

# Floor for the iterative byte-downscale loop: never shrink the transport copy
# below this longest edge. A PNG at this size is a few KB — far under any
# realistic byte cap — so the floor is a defensive backstop, not a normal exit.
_RETURN_PX_FLOOR = 64

# Upper clamp on the effective pixel cap (security review 2026-06-25 LOW-2).
# `max_return_px` has no natural ceiling, so a caller asking for a huge value
# against a large (e.g. 50 MP) on-disk image would make the FIRST base64
# encode run at full resolution before the byte loop shrinks it — a transient
# in-memory spike. Clamping the effective pixel cap bounds that first encode;
# the returned payload is already byte-capped regardless. 4096 is far above
# any sane transport copy (the byte cap shrinks it much further in practice).
_MAX_RETURN_PX_CEILING = 4096

# Bounded iteration count for the byte-downscale loop (defensive; each step
# shrinks dimensions multiplicatively so convergence is fast — typically 0–1
# steps at the 768px default).
_RETURN_BYTES_MAX_ITERS = 8

# Fail-soft INFO notice (ADR-017 invariant 8). Emitted when return_image was
# requested but the optional transport copy could not be produced (encode
# failure, or the byte budget could not be met even at the px floor); the
# generation itself succeeded and `output_path` is valid. Carries NO path,
# NO exception text — just a category signal so the caller knows image_b64 is
# absent despite the request.
_RETURN_IMAGE_FAILED_NOTICE = {
    "level": "INFO",
    "message": (
        "return_image was requested but the transport copy could not be "
        "encoded; the image is on disk (see output_path). image_b64 omitted."
    ),
}


def _encode_return_image(
    output_path: str, max_px: Any, max_bytes: Any,
) -> tuple[str, str]:
    """Re-encode the on-disk PNG into a size-bounded base64 PNG for transport.

    ADR-017 invariants:
      - Reads the already-written, already-§3e-redacted PNG at `output_path`.
        The on-disk file is NEVER modified — both bounds below apply only to
        the in-memory transport copy.
      - PIXEL bound: downscales so the longest edge ≤ the effective pixel cap
        (aspect preserved, NEVER upscaled). `max_px` <= 0 / non-int clamps to
        _DEFAULT_MAX_RETURN_PX.
      - BYTE bound: after the pixel downscale, iteratively shrinks the copy
        until len(base64) ≤ the effective byte cap = min(max_bytes,
        _RETURN_BYTES_CEILING) (a non-int / <= 0 max_bytes uses the ceiling).
        The agent cannot raise the cap above the ceiling — true hard cap.
      - Re-encodes to PNG WITHOUT carrying the source file's text chunks (no
        `pnginfo=`) — the transport copy is metadata-free, so no filesystem
        string or embedded-metadata value can ride out via the image bytes.
      - mime is the constant "image/png".

    Returns (base64_ascii_str, "image/png"). Raises on any failure (including
    the pathological case where even the px-floor copy exceeds the byte cap);
    the caller MUST treat a raise as fail-soft (omit image_b64, never fail the
    generation).
    """
    import base64
    import io

    from PIL import Image

    eff_px = (
        max_px if (isinstance(max_px, int) and not isinstance(max_px, bool)
                   and max_px > 0)
        else _DEFAULT_MAX_RETURN_PX
    )
    eff_px = min(eff_px, _MAX_RETURN_PX_CEILING)  # bound first-encode spike
    eff_bytes = (
        min(max_bytes, _RETURN_BYTES_CEILING)
        if (isinstance(max_bytes, int) and not isinstance(max_bytes, bool)
            and max_bytes > 0)
        else _RETURN_BYTES_CEILING
    )

    def _encode(image) -> str:
        buf = io.BytesIO()
        # No pnginfo= → transport copy carries no text chunks (see docstring).
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    with Image.open(output_path) as src:
        src.load()
        w, h = src.size
        longest = max(w, h)
        if longest > eff_px:  # pixel bound (never upscale)
            scale = eff_px / longest
            img = src.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.LANCZOS)
        else:
            img = src.copy()  # detach from the file; strips metadata on re-save

    b64 = _encode(img)
    # Byte bound: shrink the transport copy until the payload fits or floor.
    iters = 0
    while len(b64) > eff_bytes and iters < _RETURN_BYTES_MAX_ITERS:
        cur_longest = max(img.size)
        if cur_longest <= _RETURN_PX_FLOOR:
            break
        # base64 length scales ~linearly with pixel COUNT, so scale each
        # dimension by sqrt(target/current); the 0.9 factor undershoots to
        # converge in a few steps even when PNG compression is non-linear.
        ratio = (eff_bytes / len(b64)) ** 0.5 * 0.9
        new_longest = max(_RETURN_PX_FLOOR, int(cur_longest * ratio))
        if new_longest >= cur_longest:
            new_longest = cur_longest - 1  # guarantee forward progress
        s = new_longest / cur_longest
        img = img.resize(
            (max(1, round(img.size[0] * s)), max(1, round(img.size[1] * s))),
            Image.LANCZOS)
        b64 = _encode(img)
        iters += 1

    if len(b64) > eff_bytes:
        # Could not meet the byte budget (pathological). Raise → caller
        # fail-soft omits image_b64 rather than returning an over-budget
        # payload that would breach a downstream transport cap.
        raise ValueError("return image exceeds byte budget after downscale")
    return b64, "image/png"


def _maybe_attach_return_image(
    response: dict,
    notices: list,
    payload: dict,
    output_path: str,
) -> None:
    """Attach `image_b64`/`image_mime` to `response` when the caller set
    return_image=true. Fail-soft (ADR-017 invariant 8): any failure leaves
    the response frame intact (image fields omitted) and appends an INFO
    notice — a return-image failure MUST NOT fail a successful generation.
    Shared by both the non-cascade and cascade handlers (invariant 7)."""
    if not bool(payload.get("return_image", False)):
        return
    try:
        img_b64, img_mime = _encode_return_image(
            output_path,
            payload.get("max_return_px", _DEFAULT_MAX_RETURN_PX),
            payload.get("max_return_bytes", _RETURN_BYTES_CEILING),
        )
        response["image_b64"] = img_b64
        response["image_mime"] = img_mime
    except Exception:  # noqa: BLE001 — invariant 8: fail-soft on any encode
        # failure (PIL error, byte-budget-unmeetable ValueError, etc.). NOT
        # BaseException: KeyboardInterrupt/SystemExit and any future audit-
        # emission BaseException (global §0 rule 2) must propagate, never be
        # absorbed into a fail-soft notice.
        notices.append(dict(_RETURN_IMAGE_FAILED_NOTICE))


async def _call_tool_impl(
    cfg: _StartupConfig,
    name: str,
    arguments: dict,
) -> list[TextContent]:
    """Dispatch one MCP tools/call invocation.

    Slice 1 step 2 wires the `generate` handler for non-cascade families.
    Cascade requests (cascade_config present in arguments) are rejected
    with a NotYetWired error; step 3 fills that branch.

    Every code path emits exactly one audit line (invariant 5). All
    internal exceptions are caught and routed through either a
    pre-sanitized _MCPHandlerError message OR _sanitize_error, so the
    framework's outer `except Exception → str(e)` (which does NOT strip
    tracebacks) never sees raw exception text (step-1 reviewer F1/F4).
    """
    t0 = time.monotonic()
    # `result_count` only applies to list_* tools (invariant 19); stays
    # None for generate and for any error path. _emit_audit_line omits
    # the field from the audit JSON when None.
    result_count: Optional[int] = None

    # `list_models` / `list_loras` accept no inputs by schema (empty
    # `properties` + `additionalProperties: false`). The framework
    # decorator uses `validate_input=False` so the handler must defend
    # itself; the list_* handlers ignore `arguments` entirely. To
    # prevent an agent from flooding the operator's audit stream by
    # passing arbitrarily large `arguments` to a tool that ignores
    # them, the audit payload for these two tools is reduced to `{}` —
    # the audit line still records the call (one line per invocation
    # per invariant 5) without echoing the unbounded blob. (security-
    # auditor slice-2 step-4 LOW-1, folded 2026-05-25.)
    if name in ("list_models", "list_loras", "list_transformers"):
        audit_payload: dict = {}
    else:
        audit_payload = arguments

    try:
        if name == "generate":
            if "cascade_config" in arguments:
                result = await _handle_generate_cascade(cfg, arguments)
            else:
                result = await _handle_generate(cfg, arguments)
        elif name == "list_models":
            result, result_count = await _handle_list_models(cfg)
        elif name == "list_loras":
            result, result_count = await _handle_list_loras(cfg)
        elif name == "list_transformers":
            result, result_count = await _handle_list_transformers(cfg)
        else:
            raise _MCPHandlerError(
                "UnknownTool",
                f"Unknown tool: {name!r}",
            )

        _emit_audit_line(
            name, audit_payload, status="ok",
            elapsed_seconds=time.monotonic() - t0,
            result_count=result_count,
        )
        return result
    except _MCPHandlerError as e:
        _emit_audit_line(
            name, audit_payload, status="error",
            error_class=e.error_class,
            elapsed_seconds=time.monotonic() - t0,
        )
        # safe_message is pre-sanitized; ValueError(str) is the framework's
        # outer-except shape.
        raise ValueError(e.safe_message)
    except BaseException as e:  # noqa: BLE001 — defense in depth
        # Unexpected internal exceptions: full traceback to stderr (audit
        # stream); category-shaped sanitized string to the agent.
        safe = _sanitize_error(e, "internal_error")
        _emit_audit_line(
            name, audit_payload, status="error",
            error_class="InternalError",
            elapsed_seconds=time.monotonic() - t0,
        )
        raise ValueError(safe)


# ════════════════════════════════════════════════════════════════════════
# In-process single-slot pipeline cache (parity with the server.py daemon's
# cache + eviction). The MCP path loads pipelines in-process; without this it
# reloaded the full pipeline every call AND never freed the previous model, so
# a long-lived server OOM'd after several generations / model switches. This
# keeps ONE fully-configured pipeline (model + transformer override + LoRAs)
# resident, keyed on the whole effective config, and evicts the prior one
# (del + gc + empty_cache) on any change. Requests are serialized (stdio MCP),
# so no locking is needed — BUT that safety also depends on the generate
# handlers having NO `await` point between the cache key-check and the
# cache-update (the blocking _load_pipeline/generate/pipe() calls run inline).
# If those are ever moved to run_in_executor / made concurrent, this cache
# becomes racy and MUST get a lock.
# ════════════════════════════════════════════════════════════════════════
_PIPELINE_CACHE: dict = {
    "key": None,
    "pipeline": None,
    "model_family": None,
    "guidance_embeds": None,
    "lora_warnings": [],
}


def _evict_pipeline_cache() -> None:
    """Drop the cached pipeline and free its GPU memory, if any."""
    if _PIPELINE_CACHE.get("pipeline") is None:
        return
    _PIPELINE_CACHE["key"] = None
    _PIPELINE_CACHE["pipeline"] = None
    _PIPELINE_CACHE["model_family"] = None
    _PIPELINE_CACHE["guidance_embeds"] = None
    _PIPELINE_CACHE["lora_warnings"] = []
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _pipeline_cache_key(
    model_abs: str,
    transformer_abs: str,
    vae_from_transformer: bool,
    loras_resolved: list,
    quant: str = "none",
    quant_skip: tuple = (),
    quant_only: tuple = (),
) -> tuple:
    """Hashable key over the full effective config that varies in the MCP path.

    precision/device/offload and the vae/text-encoder component overrides are
    hard-coded constants in `_handle_generate`, so they cannot vary and are not
    part of the key. The LoRA set IS part of the key (path + weight, in order),
    so any LoRA change is a cache miss → evict + reload + re-apply. The quant
    triple is likewise keyed (ADR-019 invariant 4): an fp8→none switch must
    evict, never reuse the quantized pipeline.
    """
    lora_key = tuple(
        (str(spec.get("path", "")), float(spec.get("weight", 1.0)))
        for spec in (loras_resolved or [])
    )
    # vae_from_transformer is a no-op without a transformer override
    # (_load_pipeline only extracts a VAE when transformer_path is set), so
    # fold it out of the key when no transformer is set — avoids a spurious
    # miss + full reload when the flag toggles with no transformer.
    vft = bool(vae_from_transformer and transformer_abs)
    quant_key = (str(quant or "none"),
                 tuple(sorted(quant_skip or ())),
                 tuple(sorted(quant_only or ())))
    return (model_abs, transformer_abs, vft, lora_key, quant_key)


def _get_or_load_cached_pipeline(
    model_abs: str,
    transformer_abs: str,
    vae_from_transformer: bool,
    loras_resolved: list,
    quant: str = "none",
    quant_skip: tuple = (),
    quant_only: tuple = (),
) -> dict:
    """Return the cached pipeline dict for this config, loading on a miss.

    On a miss the prior pipeline is evicted (freeing its VRAM) before the new
    one loads, then LoRAs are applied (the MCP path previously skipped LoRA
    application entirely). Returns the shared _PIPELINE_CACHE dict, which has
    the {pipeline, model_family, guidance_embeds} keys generate() consumes.
    """
    from comfyless.generate import _load_pipeline, _apply_loras, _log
    key = _pipeline_cache_key(model_abs, transformer_abs, vae_from_transformer,
                              loras_resolved, quant, quant_skip, quant_only)
    if _PIPELINE_CACHE.get("pipeline") is not None and _PIPELINE_CACHE.get("key") == key:
        _log("[comfyless-mcp] Reusing cached pipeline (config unchanged)")
        return _PIPELINE_CACHE
    _evict_pipeline_cache()
    pipe, model_family, guidance_embeds = _load_pipeline(
        model_abs,
        precision="bf16",
        device="cuda",
        offload_vae=False,
        transformer_path=transformer_abs,
        vae_path="",
        text_encoder_path="",
        text_encoder_2_path="",
        vae_from_transformer=vae_from_transformer,
        attention_slicing=False,
        sequential_offload=False,
        allow_hf_download=False,
        quant=quant,
        quant_skip=tuple(quant_skip or ()),
        quant_only=tuple(quant_only or ()),
    )
    lora_warnings = _apply_loras(pipe, loras_resolved)
    _PIPELINE_CACHE.update({
        "key": key,
        "pipeline": pipe,
        "model_family": model_family,
        "guidance_embeds": guidance_embeds,
        "lora_warnings": lora_warnings,
    })
    return _PIPELINE_CACHE


async def _handle_generate(
    cfg: _StartupConfig,
    arguments: dict,
) -> list[TextContent]:
    """The actual generate-tool body. Raises _MCPHandlerError on every known
    failure category; unknown exceptions propagate up to _call_tool_impl's
    BaseException catch.
    """
    # 0 — Removed-field guard (slice 3 / OQ-A). Reject the slice-1 raw-path
    # field names outright. `transformer_path` is now the catalog name
    # `transformer`; vae/text_encoder overrides are dropped (no catalog kind).
    # Silently accepting a raw `vae_path` would reintroduce the caller-
    # supplied-path input attack surface ADR-015 removes. Field NAMES are
    # public schema knowledge, so naming them is not an enumeration oracle.
    for _removed in _GENERATE_REMOVED_FIELDS:
        if _removed in arguments:
            raise _MCPHandlerError(
                "ValidationError",
                f"validation failed: {_removed}: field not supported on the "
                f"MCP surface; reference weights by catalog name (see "
                f"list_models / list_transformers)",
            )

    # 1 — Canonical type validation (ADR-012), MINUS `loras`. The MCP loras
    # entry shape is {name, weight}; the SHARED canonical validator hard-
    # requires the slice-1 {path, weight} shape, so loras are validated and
    # resolved by name in step 5. Top-level `model` is still str-checked here;
    # `transformer` passes through (the resolver type-checks it). FIRST
    # substantive action after audit setup, per security-auditor F3.
    from comfyless.params_validation import validate_machine_request
    _args_no_loras = {k: v for k, v in arguments.items() if k != "loras"}
    val = validate_machine_request(_args_no_loras)
    if not val.ok:
        err = val.error or {}
        raise _MCPHandlerError(
            "ValidationError",
            f"validation failed: {err.get('field')}: {err.get('reason')}",
        )
    payload: dict = dict(val.payload or {})

    # 1.5 — Required prompt (server-specific; canonical validator is type-only
    # per ADR-012). Fail-fast BEFORE the expensive resolve+load path.
    if not (payload.get("prompt") or "").strip():
        raise _MCPHandlerError(
            "MissingField",
            "validation failed: prompt: required field absent",
        )

    # 1.6 — Null-byte gate on the non-reference write-dest field `savepath`
    # only (ValidationError, distinct from reference resolution). Reference
    # fields (model / transformer / loras[].name) get null-byte handling
    # from the resolver -> the uniform "reference not available" error
    # (MalformedReference), so they are NOT gated here.
    if "\x00" in (payload.get("savepath") or ""):
        raise _MCPHandlerError(
            "ValidationError",
            "validation failed: savepath: null byte not allowed",
        )

    # 2 — Notices accumulator (ADR-015 §3). Path-discard INFO entries are
    # appended as references resolve from path-shaped values (invariant 6/7).
    notices: list = []

    from comfyless.catalog import resolve_reference

    # 3 — Resolve the model reference. Agent-supplied -> catalog resolver
    # (uniform error on ANY failure; fine cause to audit only). Omitted ->
    # --default-model, an OPERATOR-trusted path that BYPASSES the agent-facing
    # resolver (OQ-D) but still passes the request-time _within net.
    model_in = (payload.get("model") or "").strip()
    if model_in:
        rr = resolve_reference(
            cfg.catalog, model_in, cfg.model_base, expected_kind="model")
        if not rr.ok:
            raise _reference_error(rr.cause)
        model_abs = rr.abs_path
        model_name = rr.name
        if rr.path_was_discarded:
            notices.append(_discard_notice(rr.name))
    else:
        if cfg.default_model is None:
            raise _MCPHandlerError(
                "MissingField",
                "validation failed: model: required field absent and "
                "--default-model not configured at spawn",
            )
        # Operator-trusted default; re-check containment at request time
        # (catches a post-startup symlink swap — slice-1 step-3 carry-forward).
        if not _within(cfg.default_model, cfg.model_base):
            raise _MCPHandlerError(
                "DefaultModelEscape",
                "validation failed: --default-model no longer resolves "
                "under --model-base",
            )
        model_abs = cfg.default_model
        model_name = os.path.basename(cfg.default_model)

    # 4 — Resolve the optional transformer reference (kind:"transformer").
    # Truthy covers a non-empty name; a non-str value flows to the resolver
    # and returns MalformedReference -> uniform error.
    transformer_val = payload.get("transformer")
    transformer_abs = ""
    transformer_name: Optional[str] = None
    if transformer_val:
        rr = resolve_reference(
            cfg.catalog, transformer_val, cfg.model_base,
            expected_kind="transformer")
        if not rr.ok:
            raise _reference_error(rr.cause)
        transformer_abs = rr.abs_path
        transformer_name = rr.name
        if rr.path_was_discarded:
            notices.append(_discard_notice(rr.name))

    # 5 — Validate + resolve LoRA references (kind:"lora"). The MCP entry shape
    # is {name, weight}; validate minimally here (the shared canonical lora
    # validator requires the slice-1 `path` key and is not used), then resolve
    # each name to an abs_path for the load call.
    loras_in = arguments.get("loras")
    loras_resolved: list = []   # canonical {path, weight} for the load call
    lora_names: list = []
    if loras_in is not None:
        if not isinstance(loras_in, list):
            raise _MCPHandlerError(
                "ValidationError", "validation failed: loras: expected list")
        for i, lora in enumerate(loras_in):
            if not isinstance(lora, dict):
                raise _MCPHandlerError(
                    "ValidationError",
                    f"validation failed: loras[{i}]: expected object")
            if "name" not in lora:
                raise _MCPHandlerError(
                    "MissingField",
                    f"validation failed: loras[{i}].name: required field absent")
            if "weight" not in lora:
                raise _MCPHandlerError(
                    "MissingField",
                    f"validation failed: loras[{i}].weight: required field absent")
            w = lora["weight"]
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise _MCPHandlerError(
                    "ValidationError",
                    f"validation failed: loras[{i}].weight: expected number")
            rr = resolve_reference(
                cfg.catalog, lora.get("name"), cfg.model_base,
                expected_kind="lora")
            if not rr.ok:
                raise _reference_error(rr.cause)
            loras_resolved.append({"path": rr.abs_path, "weight": float(w)})
            lora_names.append(rr.name)
            if rr.path_was_discarded:
                notices.append(_discard_notice(rr.name))

    # 6 — Defense-in-depth: re-validate every RESOLVED abs_path under
    # --model-base at the load boundary (auditor carry-forward #6 / invariant
    # 9). The resolver already _within-checked agent refs and step 3 re-checked
    # the default; this is the final net immediately before load. A failure
    # here is a containment escape on an already-resolved path -> the uniform
    # reference error (the value is never echoed).
    from comfyless.server import _check_paths
    if _check_paths(
        {"model": model_abs, "transformer_path": transformer_abs,
         "loras": loras_resolved},
        cfg.model_base,
    ):
        raise _reference_error("WithinFailure")

    # 7 — Output-path resolution + containment under --output-dir. The
    # {model}/{transformer} savepath template tokens use the resolved NAMES
    # (no abs_path leaks into generated filenames). Unchanged containment.
    try:
        output_path = _resolve_mcp_output_path(
            cfg,
            {**payload, "model": model_name,
             "transformer_path": (transformer_name or "")},
        )
    except _MCPHandlerError:
        raise
    except Exception as e:
        raise _MCPHandlerError(
            "OutputPath",
            f"validation failed: output_path resolution rejected "
            f"({type(e).__name__})",
        ) from None

    # 7.5 — Per-family default overlay (ADR-009 caller-responsibility; the MCP
    # is a caller, like the CLI's _run_one). `payload` holds exactly the keys
    # the agent supplied (validate_machine_request is type-only, no default
    # injection), so those are the "explicit" keys; FAMILY_DEFAULTS fills the
    # rest for this model's family. This is the CLI/MCP parity fix — without
    # it an agent omitting steps/cfg for Krea-2-Turbo would get 28/3.5 instead
    # of 8/0.0. `_apply_family_defaults` keys on gen_params["model"] and
    # re-detects the family; on a diffusers without the model's class (e.g.
    # Krea2Pipeline today) it no-ops — which is exactly when generation is
    # unavailable anyway. The CLI applies the SAME helper, so both surfaces
    # agree by construction.
    from comfyless.generate import _apply_family_defaults, COMFYLESS_SCHEMA
    _canon_keys = set(COMFYLESS_SCHEMA)
    gen_params: dict = {k: payload[k] for k in payload if k in _canon_keys}
    _explicit_keys = set(gen_params)
    gen_params["model"] = model_abs
    _apply_family_defaults(gen_params, _explicit_keys, set())

    # 8 — Load + generate (HARD-CODED allow_hf_download=False; in-process).
    # Component overrides vae/text_encoder are removed from the MCP surface
    # (OQ-A) -> always "" here. Operator-tuning knobs (precision/offload/...)
    # are spawn-time concerns, not agent-facing — hard-coded defaults.
    from comfyless.generate import generate
    # quant fields were type-validated in step 1 (validate_machine_request:
    # str mode + list-of-bare-slot-names, path shapes rejected). They carry no
    # filesystem strings and feed only build_quant_config's eligibility policy.
    cached = _get_or_load_cached_pipeline(
        model_abs,
        transformer_abs,
        bool(payload.get("vae_from_transformer")),
        loras_resolved or [],
        quant=str(payload.get("quant") or "none"),
        quant_skip=tuple(payload.get("quant_skip") or ()),
        quant_only=tuple(payload.get("quant_only") or ()),
    )
    # NOTE: cached["lora_warnings"] are logged operator-side by _apply_loras but
    # are NOT surfaced in `notices` — they embed the absolute LoRA path, which
    # must never cross the MCP boundary (ADR-015 no-abs-path contract). An
    # agent-facing LoRA-failure signal would need name-based redaction (future).
    metadata = generate(
        model_path=model_abs,
        prompt=payload["prompt"],
        output_path=output_path,
        negative_prompt=gen_params.get("negative_prompt", ""),
        seed=gen_params.get("seed", -1),
        steps=gen_params.get("steps", 28),
        cfg_scale=gen_params.get("cfg_scale", 3.5),
        true_cfg_scale=gen_params.get("true_cfg_scale"),
        width=gen_params.get("width", 1024),
        height=gen_params.get("height", 1024),
        max_sequence_length=gen_params.get("max_sequence_length", 512),
        sampler=gen_params.get("sampler", "default"),
        schedule=gen_params.get("schedule", "linear"),
        loras=loras_resolved or [],
        precision="bf16",
        device="cuda",
        offload_vae=False,
        attention_slicing=False,
        sequential_offload=False,
        transformer_path=transformer_abs,
        vae_path="",
        text_encoder_path="",
        text_encoder_2_path="",
        vae_from_transformer=bool(payload.get("vae_from_transformer")),
        allow_hf_download=False,
        _cached_pipeline=cached,
        mcp_caller=True,  # signals _save_with_metadata to apply MCP redaction
    )

    # 9 — Inline response (invariant 11: no sidecar on disk). resolved_params
    # renders weight references as catalog NAMES (invariant 5); the
    # path-discard notices ride alongside (invariant 6/7). The on-disk PNG
    # metadata is separately basename-redacted by generate(mcp_caller=True).
    resolved_params = _resolved_params_as_names(
        metadata,
        model_name=model_name,
        transformer_name=transformer_name,
        lora_names=lora_names,
    )
    response = {
        "output_path": output_path,
        "resolved_params": resolved_params,
        "elapsed_seconds": metadata.get("elapsed_seconds"),
    }
    # ADR-017: optional, gated, size-bounded base64 return. Reads the
    # already-written on-disk PNG; fail-soft (never fails a successful gen).
    _maybe_attach_return_image(response, notices, payload, output_path)
    if notices:
        response["notices"] = notices
    return [TextContent(type="text", text=json.dumps(response, default=str))]


async def _handle_generate_cascade(
    cfg: _StartupConfig,
    arguments: dict,
) -> list[TextContent]:
    """Stable Cascade dispatch via cascade_config.

    Slice 3b: cascade stage references are CATALOG NAMES, mirroring the
    non-cascade `_handle_generate`. Same Red Zone discipline: canonical
    validation → cascade-config schema validation → scaffolding_repo
    removed-field rejection → catalog resolution of stage_c/stage_b/stage_a
    (kind {model, transformer}; every failure folds into the uniform
    "reference not available" error, fine cause to the stderr audit only)
    → load-boundary _within net → output-path containment → in-process
    cascade.build_pipelines + cascade.run_one (allow_hf_download=False at
    every call site) → cascade._save_with_metadata with mcp_caller=True.
    The response renders cascade_config.stage_* as catalog names and drops
    scaffolding_repo; path-discard INFO notices ride alongside.
    """
    # 1 — Canonical type validation (top-level types only; cascade_config
    # passes through as a generic object).
    from comfyless.params_validation import validate_machine_request
    val = validate_machine_request(arguments)
    if not val.ok:
        err = val.error or {}
        raise _MCPHandlerError(
            "ValidationError",
            f"validation failed: {err.get('field')}: {err.get('reason')}",
        )
    payload: dict = dict(val.payload or {})

    # 1.5 — Required-prompt gate (server-specific; matches non-cascade path
    # and the daemon's missing-prompt check at server.py:133-136).
    if not (payload.get("prompt") or "").strip():
        raise _MCPHandlerError(
            "MissingField",
            "validation failed: prompt: required field absent",
        )

    # 1.6 — Null-byte gate on the non-reference write-dest field `savepath`
    # only (mirrors the non-cascade handler). Cascade stage reference fields get
    # their null-byte / malformed handling from resolve_reference in step 2.5
    # (-> the uniform "reference not available" error), so they are NOT gated
    # here; only `savepath` (a write destination, distinct from reference
    # resolution) needs the explicit ValidationError gate.
    for _nb_field in ("savepath",):
        if "\x00" in (payload.get(_nb_field) or ""):
            raise _MCPHandlerError(
                "ValidationError",
                f"validation failed: {_nb_field}: null byte not allowed",
            )

    # 2 — Cascade-config schema validation (cascade-side; ADR-010).
    raw_cc = payload.get("cascade_config")
    if not isinstance(raw_cc, dict):
        raise _MCPHandlerError(
            "ValidationError",
            "validation failed: cascade_config: expected object",
        )

    # 2.0 — Removed-field guard (slice 3b). scaffolding_repo is the cascade
    # architecture-config provider, not an aesthetic weight choice; it is NOT
    # agent-selectable on the MCP surface. Reject if supplied (checked on the
    # RAW agent input, before validate_config's setdefault masks it). Naming
    # the field leaks nothing — field names are public schema knowledge, unlike
    # a reference VALUE — and rejecting (not silently ignoring) keeps the
    # caller-supplied-path input surface closed. The server falls back to
    # cascade.validate_config's operator-trusted default (§OQ-2). Mirrors the
    # non-cascade _GENERATE_REMOVED_FIELDS rule.
    if "scaffolding_repo" in raw_cc:
        raise _MCPHandlerError(
            "ValidationError",
            "validation failed: cascade_config.scaffolding_repo: field not "
            "supported on the MCP surface; the server uses its default "
            "scaffolding repo (cascade architecture config is not "
            "agent-selectable)",
        )

    from comfyless.cascade import validate_config as _cascade_validate_config
    try:
        cfg_cc = _cascade_validate_config(raw_cc, source="mcp_request")
    except (ValueError, TypeError) as e:
        # cascade.validate_config names the offending field; keep the
        # category but suppress the value (which is agent input).
        raise _MCPHandlerError(
            "ValidationError",
            f"validation failed: cascade_config: {type(e).__name__}",
        ) from None

    # 2.5 — Resolve cascade stage references via the catalog (slice 3b). The
    # stage fields (stage_c/stage_b required, stage_a optional) are agent-
    # supplied catalog NAMES resolved against kind {model, transformer} — a
    # stage weight catalogs as transformer when single-file and model when a
    # diffusers tree. The resolver handles basename-strip, NFC-normalize, null-
    # byte / malformed gating, request-time existence + _within fail-closed; ANY
    # failure cause folds into the single uniform "reference not available"
    # error (the fine cause to the stderr audit only). This extends the HIGH-1
    # uniform-error contract to the cascade path, closing the slice-1 cascade
    # oracle (its distinct cache-miss / path-allowlist agent errors that named
    # the cause). The old per-field null-byte gate + HF resolution + allowlist
    # are all subsumed by resolve_reference. scaffolding_repo is NOT resolved
    # here — it is the
    # operator-trusted default (validate_config setdefault), resolved internally
    # by cascade.build_pipelines, and never agent-affectable.
    from comfyless.catalog import resolve_reference
    notices: list = []
    # quant is a non-cascade knob (ADR-019 slice A): the cascade path has its
    # own stage loaders that don't take quantization_config. Ignore loudly —
    # never silently — so an agent setting quant on cascade learns why the
    # VRAM didn't change.
    if arguments.get("quant") and arguments.get("quant") != "none":
        notices.append(
            "INFO: quant is not supported for Stable Cascade dispatch — "
            "ignored (generation proceeds unquantized)"
        )
    resolved_cc = dict(cfg_cc)
    stage_names: dict = {}
    for stage in ("stage_c", "stage_b", "stage_a"):
        raw_v = cfg_cc.get(stage)
        if stage == "stage_a" and not raw_v:
            continue  # stage_a is optional (validate_config requires only c + b)
        rr = resolve_reference(
            cfg.catalog, raw_v, cfg.model_base,
            expected_kind=("model", "transformer"))
        if not rr.ok:
            raise _reference_error(rr.cause)
        resolved_cc[stage] = rr.abs_path
        stage_names[stage] = rr.name
        if rr.path_was_discarded:
            notices.append(_discard_notice(rr.name))

    # 3 — Load-boundary _within net (defense-in-depth) on every resolved stage
    # abs_path immediately before load. The resolver already _within-checked at
    # request time; this is the final net (mirrors non-cascade step 6). A
    # failure here is a containment escape on an already-resolved path -> the
    # uniform reference error (the value is never echoed).
    for stage in stage_names:
        if not _within(resolved_cc[stage], cfg.model_base):
            raise _reference_error("WithinFailure")

    # 5 — Output-path resolution. Cascade dispatch ignores top-level model/
    # seed for savepath token expansion; mirror the daemon's cascade savepath
    # resolver in spirit. Use the model-token = "stablecascade" sentinel for
    # template expansion (cascade.py's existing pattern).
    #
    # Note: cascade ignores top-level `model`; --default-model does NOT apply
    # to cascade dispatch (topology is in cascade_config). The "stablecascade"
    # sentinel is non-path text the savepath resolver inserts into the
    # {model} template token. This asymmetry vs `_handle_generate` is by
    # design: cascade's identity is its config, not a single weight directory.
    output_path = _resolve_mcp_output_path(
        cfg,
        {**payload, "model": payload.get("model") or "stablecascade"},
    )

    # 6 — Build pipelines + run (HARD-CODED allow_hf_download=False per
    # invariant 4; cascade-specific extension of P3).
    from comfyless.cascade import (
        build_pipelines as _cascade_build_pipelines,
        run_one as _cascade_run_one,
        _save_with_metadata as _cascade_save_with_metadata,
        dispose_pipelines as _cascade_dispose_pipelines,
    )
    # Free any resident non-cascade pipeline first so cascade's two pipelines
    # don't share VRAM with a cached model from a prior generate call.
    _evict_pipeline_cache()
    prior_pipe, decoder_pipe = _cascade_build_pipelines(
        resolved_cc, device="cuda", allow_hf_download=False,
    )
    try:
        pil, runtime = _cascade_run_one(
            prior_pipe, decoder_pipe, resolved_cc,
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt", ""),
            seed=int(payload.get("seed") or 0),
            device="cuda",
        )
    finally:
        # Cascade does not cache (two large pipelines); dispose both every call
        # so the server doesn't leak VRAM across cascade generations. `pil` is a
        # CPU image already materialized by run_one, so it survives disposal.
        _cascade_dispose_pipelines(prior_pipe, decoder_pipe)

    # 7 — Build metadata and save PNG (with MCP redaction map applied).
    import datetime as _dt
    metadata = {
        "prompt": payload["prompt"],
        "negative_prompt": payload.get("negative_prompt", ""),
        "model": "stablecascade",
        "model_family": "stablecascade",
        "cascade_config": resolved_cc,
        "seed": int(payload.get("seed") or 0),
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "prior_seconds": runtime.get("prior_seconds"),
        "decoder_seconds": runtime.get("decoder_seconds"),
        "elapsed_seconds": (runtime.get("prior_seconds", 0) or 0)
                           + (runtime.get("decoder_seconds", 0) or 0),
    }
    _cascade_save_with_metadata(pil, output_path, metadata, mcp_caller=True)

    # 8 — Inline response (invariant 1: no abs_path crosses the boundary).
    # resolved_params renders cascade_config.stage_* as catalog NAMES and drops
    # scaffolding_repo; the path-discard notices ride alongside. The on-disk PNG
    # metadata is separately basename-redacted by _save_with_metadata
    # (mcp_caller=True) — a distinct sink that is unchanged.
    resolved_params = _resolved_cascade_params_as_names(
        metadata, stage_names=stage_names)
    response = {
        "output_path": output_path,
        "resolved_params": resolved_params,
        "elapsed_seconds": metadata["elapsed_seconds"],
    }
    # ADR-017: optional, gated, size-bounded base64 return (cascade path —
    # invariant 7: identical behavior to the non-cascade handler). Reads the
    # already-written on-disk PNG; fail-soft (never fails a successful gen).
    _maybe_attach_return_image(response, notices, payload, output_path)
    if notices:
        response["notices"] = notices
    return [TextContent(type="text", text=json.dumps(response, default=str))]


def _resolve_mcp_output_path(
    cfg: _StartupConfig,
    payload: dict,
) -> str:
    """Resolve the MCP-side output path under --output-dir.

    If `savepath` is supplied, run it through the existing template
    expansion machinery and `_within(--output-dir)`-check the result.
    Otherwise auto-generate a non-colliding `comfyless####.png` under
    --output-dir. Raises _MCPHandlerError on containment failure.
    """
    from comfyless.generate import _expand_savepath_template, _resolve_savepath
    savepath = payload.get("savepath")
    if savepath:
        safe_template = savepath.lstrip("/").lstrip("\\")
        full_template = str(os.path.join(cfg.output_dir, safe_template))
        _txp = (payload.get("transformer_path") or "")
        expanded = _expand_savepath_template(
            full_template, payload["model"],
            payload.get("seed", -1), payload.get("steps", 28),
            payload.get("cfg_scale", 3.5), payload.get("sampler", "default"),
            transformer_path=_txp,
        )
        if not _within(os.path.dirname(expanded) or cfg.output_dir, cfg.output_dir):
            raise _MCPHandlerError(
                "OutputPath",
                "validation failed: savepath template expands outside --output-dir",
            )
        output_path = _resolve_savepath(
            full_template, payload["model"],
            payload.get("seed", -1), payload.get("steps", 28),
            payload.get("cfg_scale", 3.5), payload.get("sampler", "default"),
            transformer_path=_txp,
        )
    else:
        counter = 1
        while True:
            candidate = str(os.path.join(cfg.output_dir, f"comfyless{counter:04d}.png"))
            if not os.path.exists(candidate):
                output_path = candidate
                break
            counter += 1

    # Belt-and-suspenders re-check of the final path.
    if not _within(output_path, cfg.output_dir):
        raise _MCPHandlerError(
            "OutputPath",
            "validation failed: resolved output_path escaped --output-dir",
        )
    return output_path


# ════════════════════════════════════════════════════════════════════════
# list_models / list_loras handlers (slice 2 step 4) + list_transformers (slice 2b)
# ════════════════════════════════════════════════════════════════════════
#
# Vision invariant 9 contract for the response:
#   - `list_models` returns only `kind:"model"` entries
#   - `list_loras` returns only `kind:"lora"` entries
#   - `kind:"transformer"` entries sit in the catalog but are NOT exposed
#     by either tool (N27 / dormant-until-slice-2b)
#   - Response entry keys are a strict allowlist:
#       list_models:  {name, kind, source} + model_family (if known)
#       list_loras:   {name, kind, source} + target_family (manifest-only)
#   - NO abs_path / path / any other filesystem-string under any key
#
# Each returns `(list[TextContent], int)` where the int is the number of
# entries the agent received. `_call_tool_impl` reads the count and
# passes it to `_emit_audit_line(result_count=...)` (invariant 19 — the
# audit line carries count + tool + status + elapsed without exposing
# any catalog `abs_path`).
#
# Both handlers are pure read-only iteration over `cfg.catalog`; the
# catalog is built once at spawn and is frozen for the server's
# lifetime (Vision invariant 1).

async def _handle_list_models(
    cfg: _StartupConfig,
) -> tuple[list[TextContent], int]:
    """Enumerate `kind:"model"` catalog entries for the MCP agent."""
    entries: list[dict[str, Any]] = []
    for name, entry in cfg.catalog.items():
        if entry["kind"] != "model":
            continue
        # Strict-allowlist serialization: only the keys named in
        # Vision invariant 9 are written into the response.
        out: dict[str, Any] = {
            "name": name,
            "kind": "model",
            "source": entry["source"],
        }
        model_family = entry.get("model_family")
        if model_family is not None:
            out["model_family"] = model_family
        entries.append(out)
    # Deterministic ordering for stable agent UX + reproducible audits.
    entries.sort(key=lambda e: e["name"])
    body = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return [TextContent(type="text", text=body)], len(entries)


async def _handle_list_loras(
    cfg: _StartupConfig,
) -> tuple[list[TextContent], int]:
    """Enumerate `kind:"lora"` catalog entries for the MCP agent."""
    entries: list[dict[str, Any]] = []
    for name, entry in cfg.catalog.items():
        if entry["kind"] != "lora":
            continue
        out: dict[str, Any] = {
            "name": name,
            "kind": "lora",
            "source": entry["source"],
        }
        # `target_family` is manifest-only (Vision invariant 9 / N17 /
        # OQ2 resolution); scan-derived LoRA entries omit the field
        # entirely — no inference from filesystem layout or weight
        # introspection in this slice.
        target_family = entry.get("target_family")
        if target_family is not None:
            out["target_family"] = target_family
        entries.append(out)
    entries.sort(key=lambda e: e["name"])
    body = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return [TextContent(type="text", text=body)], len(entries)


async def _handle_list_transformers(
    cfg: _StartupConfig,
) -> tuple[list[TextContent], int]:
    """Enumerate `kind:"transformer"` catalog entries for the MCP agent.

    Mirror of `_handle_list_models` over the transformer kind (slice 2b
    invariant 2): strict-allowlist serialization of `{name, kind, source}`
    plus `model_family` when present. `model_family` is manifest-declared
    only for transformers — scan-derived single-file DiT weights carry no
    model_index.json, so the field is normally absent. NO `abs_path` /
    `path` / any filesystem string ever enters the response (the slice-2
    keystone guarantee, extended verbatim to this handler).
    """
    entries: list[dict[str, Any]] = []
    for name, entry in cfg.catalog.items():
        if entry["kind"] != "transformer":
            continue
        out: dict[str, Any] = {
            "name": name,
            "kind": "transformer",
            "source": entry["source"],
        }
        model_family = entry.get("model_family")
        if model_family is not None:
            out["model_family"] = model_family
        entries.append(out)
    entries.sort(key=lambda e: e["name"])
    body = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return [TextContent(type="text", text=body)], len(entries)


# ════════════════════════════════════════════════════════════════════════
# Server builder + async runner
# ════════════════════════════════════════════════════════════════════════

def _build_server(cfg: _StartupConfig) -> Server:
    """Construct the MCP server and register handlers under the framework's
    decorators. Tests call _list_tools_impl / _call_tool_impl directly to
    bypass the framework wrapping; integration-style tests use this
    builder + the stdio loop.
    """
    app = Server(name="comfyless", version="0.1")

    @app.list_tools()
    async def _wrapped_list_tools() -> list[Tool]:
        return await _list_tools_impl(cfg)

    # validate_input=False: per invariant 5, EVERY invocation emits an
    # audit line. The framework's default validate_input=True short-
    # circuits before our handler runs on schema-invalid input, which
    # would bypass our audit emission. We take ownership of input
    # validation inside the handler (step 2 plugs in
    # validate_machine_request from the canonical validator).
    @app.call_tool(validate_input=False)
    async def _wrapped_call_tool(name: str, arguments: dict) -> list[TextContent]:
        return await _call_tool_impl(cfg, name, arguments)

    return app


async def _run_async(cfg: _StartupConfig) -> None:
    """Bring up the stdio MCP server and block until EOF / shutdown."""
    app = _build_server(cfg)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


# ════════════════════════════════════════════════════════════════════════
# CLI entry point (click; no argparse — invariant 14)
# ════════════════════════════════════════════════════════════════════════

@click.command(
    name="comfyless-mcp-server",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=False),
    help=(
        "The only directory the server may write into. All outputs land "
        "under this root after os.path.realpath. NO env-var fallback "
        "(invariant 1)."
    ),
)
@click.option(
    "--model-base",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=False),
    help=(
        "Allowlist root for every loadable-weight reference (model, "
        "transformer_path, vae_path, text_encoder_path, "
        "text_encoder_2_path, loras[].path, cascade stage_* / "
        "scaffolding_repo). NO env-var fallback (invariant 1)."
    ),
)
@click.option(
    "--default-model",
    required=False,
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=False),
    help=(
        "Optional model path used when the agent omits `model` in a "
        "generate call. Validated at startup AND re-validated at every "
        "request that uses it. Must resolve under --model-base "
        "(invariants 8, 10)."
    ),
)
@click.option(
    "--catalog",
    required=False,
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=False),
    help=(
        "Optional operator manifest (JSON) augmenting the model_base scan "
        "with named entries (HF repo IDs / explicit local paths). Built "
        "once at spawn via comfyless.catalog.build_catalog; startup fails "
        "closed on missing / malformed / schema-invalid / collision / "
        "symlink-escape (Vision invariants 1, 7 / N1-N7)."
    ),
)
@click.option(
    "--mcp-max-iterations",
    required=False,
    default=100,
    type=click.IntRange(min=1),
    show_default=True,
    help=(
        "Absolute hard ceiling on iterate totals, applied INDEPENDENTLY "
        "of any agent-supplied max_iterations. Spawn-time only. Declared "
        "in slice 1 for stable spawn contract; the iterate handler lands "
        "in a later slice."
    ),
)
def main(
    output_dir: str,
    model_base: str,
    default_model: Optional[str],
    catalog: Optional[str],
    mcp_max_iterations: int,
) -> None:
    """Run the comfyless MCP server over stdio.

    Spawn pattern (e.g. claude_desktop_config.json):

        "comfyless": {
            "command": "python",
            "args": ["-m", "comfyless.mcp_server",
                     "--output-dir", "/abs/path/outputs",
                     "--model-base", "/abs/path/models"]
        }
    """
    cfg = _validate_startup_args(
        output_dir=output_dir,
        model_base=model_base,
        default_model=default_model,
        mcp_max_iterations=mcp_max_iterations,
        catalog=catalog,
    )
    asyncio.run(_run_async(cfg))


if __name__ == "__main__":
    main()
