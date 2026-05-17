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
# fields; only prompt/negative_prompt are dropped). Cascade-side path
# fields (stage_c/stage_b/stage_a/scaffolding_repo) are added in step 3.
_MCP_PATH_TYPED_FIELDS = (
    "model",
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
    out.pop("output_path", None)
    out.pop("savepath", None)
    return out


# ════════════════════════════════════════════════════════════════════════
# Tool description text (refinable per ADR-011 §2 amendment 2026-04-30)
# ════════════════════════════════════════════════════════════════════════

_GENERATE_TOOL_DESCRIPTION = """\
Generate an image from a text prompt. Covers all comfyless model families:
qwen-image, flux, flux2, chroma, and Stable Cascade (via cascade_config).

Model selection guidance:
- Text rendering + photorealism: qwen-image (Qwen-Image-2512)
- Anime / manga / illustration: Illustrious, Pony, or Chroma
- Fastest at modest quality: Stable Cascade via cascade_config
- General-purpose / latest: flux2 (Flux.2)

If `model` is omitted, the server uses the path configured at spawn time
via --default-model. Omitting `model` without a configured default
returns an error.

All path-typed fields (model, transformer_path, vae_path,
text_encoder_path, text_encoder_2_path, loras[].path, and cascade stage
paths) must resolve under --model-base. Output paths must resolve under
--output-dir. HuggingFace downloads are not performed; models must be
local or already cached.
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
                "Absolute path to a model directory under --model-base, "
                "OR an HF repo ID already present in the local cache. "
                "Optional if --default-model is configured at spawn."
            ),
        },
        "transformer_path": {"type": "string"},
        "vae_path": {"type": "string"},
        "text_encoder_path": {"type": "string"},
        "text_encoder_2_path": {"type": "string"},
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
                "required": ["path", "weight"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "weight": {"type": "number"},
                },
            },
        },
        "savepath": {"type": "string"},
        "cascade_config": {
            "type": "object",
            "description": (
                "Stable Cascade dispatch config (inline JSON object). "
                "Required when targeting cascade. Slice 3 wires the "
                "dispatch handler; slice 1 ships the schema slot."
            ),
        },
    },
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
    """Resolved + validated spawn-time configuration."""

    __slots__ = ("output_dir", "model_base", "default_model", "mcp_max_iterations")

    def __init__(
        self,
        output_dir: str,
        model_base: str,
        default_model: Optional[str],
        mcp_max_iterations: int,
    ) -> None:
        self.output_dir = output_dir
        self.model_base = model_base
        self.default_model = default_model
        self.mcp_max_iterations = mcp_max_iterations


def _validate_startup_args(
    output_dir: str,
    model_base: str,
    default_model: Optional[str],
    mcp_max_iterations: int,
) -> _StartupConfig:
    """Resolve + validate spawn-time CLI args. Raises click.BadParameter on bad input.

    Invariant 1: fail-closed on missing/non-existent/non-directory for the
    two required roots and (when set) --default-model.
    Invariant 10: --default-model must realpath-resolve under --model-base.
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

    return _StartupConfig(
        output_dir=resolved_out,
        model_base=resolved_base,
        default_model=resolved_default,
        mcp_max_iterations=mcp_max_iterations,
    )


# ════════════════════════════════════════════════════════════════════════
# Tool handlers (impls callable directly by tests; framework decoration in
# _build_server)
# ════════════════════════════════════════════════════════════════════════

async def _list_tools_impl(cfg: _StartupConfig) -> list[Tool]:
    """Invariant 6: advertise exactly ONE tool in slice 1: `generate`."""
    return [Tool(
        name="generate",
        description=_GENERATE_TOOL_DESCRIPTION,
        inputSchema=_GENERATE_INPUT_SCHEMA,
    )]


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

    try:
        if name != "generate":
            raise _MCPHandlerError(
                "UnknownTool",
                f"Unknown tool: {name!r}",
            )

        if "cascade_config" in arguments:
            raise _MCPHandlerError(
                "CascadeNotYetWired",
                "Cascade dispatch via cascade_config lands in slice 1 "
                "step 3; see docs/vision/slice-1-mcp-generate.md",
            )

        result = await _handle_generate(cfg, arguments)
        _emit_audit_line(
            name, arguments, status="ok",
            elapsed_seconds=time.monotonic() - t0,
        )
        return result
    except _MCPHandlerError as e:
        _emit_audit_line(
            name, arguments, status="error",
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
            name, arguments, status="error",
            error_class="InternalError",
            elapsed_seconds=time.monotonic() - t0,
        )
        raise ValueError(safe)


async def _handle_generate(
    cfg: _StartupConfig,
    arguments: dict,
) -> list[TextContent]:
    """The actual generate-tool body. Raises _MCPHandlerError on every known
    failure category; unknown exceptions propagate up to _call_tool_impl's
    BaseException catch.
    """
    # 1 — Canonical type validation (ADR-012). FIRST handler action after
    # audit-line setup, per security-auditor F3.
    from comfyless.params_validation import validate_machine_request
    val = validate_machine_request(arguments)
    if not val.ok:
        err = val.error or {}
        raise _MCPHandlerError(
            "ValidationError",
            f"validation failed: {err.get('field')}: {err.get('reason')}",
        )
    payload: dict = dict(val.payload or {})

    # 1.5 — Required-field presence (server-specific; canonical validator
    # is type-only per ADR-012 design). Mirrors the daemon's missing-prompt
    # gate at server.py:133-136. Required BEFORE expensive _load_pipeline
    # so we fail-fast on malformed input — security-auditor step-2 F2.
    if not (payload.get("prompt") or "").strip():
        raise _MCPHandlerError(
            "MissingField",
            "validation failed: prompt: required field absent",
        )

    # 1.6 — Null-byte path defense (daemon parity; server.py:138-149).
    # os.path.realpath raises on NUL; without this check the NUL would
    # escape `_check_paths` (step 5 below) and fall into the outer
    # BaseException handler, producing audit class "InternalError" instead
    # of the correct "ValidationError" and emitting a stderr-side traceback
    # on malformed input — security-auditor step-2 F1.
    for _nb_field in (
        "model", "transformer_path", "vae_path",
        "text_encoder_path", "text_encoder_2_path", "savepath",
    ):
        if "\x00" in (payload.get(_nb_field) or ""):
            raise _MCPHandlerError(
                "ValidationError",
                f"validation failed: {_nb_field}: null byte not allowed",
            )
    for _nb_i, _nb_lora in enumerate(payload.get("loras") or []):
        if "\x00" in (_nb_lora.get("path") or ""):
            raise _MCPHandlerError(
                "ValidationError",
                f"validation failed: loras[{_nb_i}].path: null byte not allowed",
            )

    # 2 — --default-model fallback (invariants 8, 9; N15, N16).
    model_input = (payload.get("model") or "").strip()
    if not model_input:
        if cfg.default_model is None:
            raise _MCPHandlerError(
                "MissingField",
                "validation failed: model: required field absent and "
                "--default-model not configured at spawn",
            )
        model_input = cfg.default_model
    payload["model"] = model_input

    # 3 — Defense-in-depth re-validation of --default-model at request time
    # (invariant 8). Startup already validated; the within-check fires here
    # whenever the active model EQUALS the configured default — which covers
    # BOTH the omitted-model fallback path (model_input was just assigned)
    # AND the agent-passed-default-path-explicitly case (string equality with
    # the realpath'd cfg.default_model). Catches a hypothetical post-startup
    # symlink swap that would have escaped the model-base. Note: this is in
    # addition to the per-request _check_paths step 5 below, which validates
    # ANY model path (default or not) against --model-base.
    if cfg.default_model is not None and model_input == cfg.default_model:
        if not _within(cfg.default_model, cfg.model_base):
            raise _MCPHandlerError(
                "DefaultModelEscape",
                "validation failed: --default-model no longer resolves "
                "under --model-base",
            )

    # 4 — Resolve HF repo IDs to local paths (HARD-CODED allow_download=
    # False per invariant 4). The agent-supplied INPUT is kept separately
    # so PNG-redaction can pass HF repo IDs through unchanged (N30).
    from nodes.eric_diffusion_utils import resolve_hf_path
    try:
        resolved: dict = {}
        for field in (
            "model", "transformer_path", "vae_path",
            "text_encoder_path", "text_encoder_2_path",
        ):
            v = (payload.get(field) or "").strip()
            if v:
                resolved[field] = resolve_hf_path(v, allow_download=False)
        loras_resolved: list = []
        for i, lora in enumerate(payload.get("loras") or []):
            lpath = (lora.get("path") or "").strip()
            loras_resolved.append({
                **lora,
                "path": resolve_hf_path(lpath, allow_download=False),
            })
    except ValueError:
        # ValueError surfaces when allow_download=False and the repo is
        # not in the local cache (HFCacheMiss path; N10). DO NOT echo the
        # repo ID back — that's an enumeration oracle.
        raise _MCPHandlerError(
            "HFCacheMiss",
            "validation failed: HF repo not in local cache (set up via "
            "`huggingface-cli download <repo>` first; MCP server does "
            "not perform downloads)",
        ) from None

    # 5 — Path allowlist against --model-base (invariant 2; N5-N9).
    # Reuse the daemon's _check_paths helper verbatim.
    from comfyless.server import _check_paths
    resolved_payload = {**payload, **resolved}
    if loras_resolved:
        resolved_payload["loras"] = loras_resolved
    err_msg = _check_paths(resolved_payload, cfg.model_base)
    if err_msg:
        # _check_paths returns "field path outside --model-base: '/x/y'"
        # — split on the first colon so the rejected VALUE is not echoed
        # back (avoid enumeration oracle on the model_base tree).
        safe_head = err_msg.split(":", 1)[0] if ":" in err_msg else err_msg
        raise _MCPHandlerError(
            "PathAllowlist",
            f"validation failed: {safe_head}",
        )

    # 6 — Output-path resolution + containment under --output-dir
    # (invariant 3; N8).
    try:
        output_path = _resolve_mcp_output_path(cfg, payload)
    except _MCPHandlerError:
        raise
    except Exception as e:
        # savepath template expansion / collision logic can raise;
        # treat as a path-validation failure and DO NOT echo the value.
        raise _MCPHandlerError(
            "OutputPath",
            f"validation failed: output_path resolution rejected "
            f"({type(e).__name__})",
        ) from None

    # 7 — Load + generate (HARD-CODED allow_hf_download=False per
    # invariant 4; in-process — no daemon delegation in slice 1, see
    # TECH_DEBT entry "MCP server: daemon delegation deferred").
    #
    # Operator-tuning knobs (precision / offload_vae / attention_slicing /
    # sequential_offload) are deliberately NOT exposed on the MCP schema
    # (_GENERATE_INPUT_SCHEMA additionalProperties:False blocks them). These
    # are server-side perf concerns the operator picks at spawn time, not
    # something the LLM agent should be tuning per-call. Hard-coded defaults
    # match the CLI's defaults; a future slice may add operator-side spawn
    # flags (e.g. --precision, --offload-vae) if the demand surfaces.
    from comfyless.generate import _load_pipeline, generate
    pipe, model_family, guidance_embeds = _load_pipeline(
        resolved["model"],
        precision="bf16",
        device="cuda",
        offload_vae=False,
        transformer_path=resolved.get("transformer_path", "") or "",
        vae_path=resolved.get("vae_path", "") or "",
        text_encoder_path=resolved.get("text_encoder_path", "") or "",
        text_encoder_2_path=resolved.get("text_encoder_2_path", "") or "",
        vae_from_transformer=bool(payload.get("vae_from_transformer")),
        attention_slicing=False,
        sequential_offload=False,
        allow_hf_download=False,
    )
    cached = {
        "pipeline": pipe,
        "model_family": model_family,
        "guidance_embeds": guidance_embeds,
    }
    metadata = generate(
        model_path=resolved["model"],
        prompt=payload["prompt"],
        output_path=output_path,
        negative_prompt=payload.get("negative_prompt", ""),
        seed=payload.get("seed", -1),
        steps=payload.get("steps", 28),
        cfg_scale=payload.get("cfg_scale", 3.5),
        true_cfg_scale=payload.get("true_cfg_scale"),
        width=payload.get("width", 1024),
        height=payload.get("height", 1024),
        max_sequence_length=payload.get("max_sequence_length", 512),
        sampler=payload.get("sampler", "default"),
        schedule=payload.get("schedule", "linear"),
        loras=loras_resolved or [],
        precision="bf16",
        device="cuda",
        offload_vae=False,
        attention_slicing=False,
        sequential_offload=False,
        transformer_path=resolved.get("transformer_path", "") or "",
        vae_path=resolved.get("vae_path", "") or "",
        text_encoder_path=resolved.get("text_encoder_path", "") or "",
        text_encoder_2_path=resolved.get("text_encoder_2_path", "") or "",
        vae_from_transformer=bool(payload.get("vae_from_transformer")),
        allow_hf_download=False,
        _cached_pipeline=cached,
        mcp_caller=True,  # signals _save_with_metadata to apply MCP redaction
    )

    # 8 — Build inline response (invariant 11: no sidecar on disk; the
    # resolved-params blob is returned in-frame instead). The IN-FRAME
    # blob carries the FULL paths (the agent's authoritative record);
    # only the on-disk PNG metadata is redacted to basenames.
    response = {
        "output_path": output_path,
        "resolved_params": metadata,
        "elapsed_seconds": metadata.get("elapsed_seconds"),
    }
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
    )
    asyncio.run(_run_async(cfg))


if __name__ == "__main__":
    main()
