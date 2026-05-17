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


async def _call_tool_impl(
    cfg: _StartupConfig,
    name: str,
    arguments: dict,
) -> list[TextContent]:
    """Slice-1-step-1 stub: emit audit line + raise NotImplementedError.

    Step 2 replaces the body with: canonical validate_machine_request →
    path-allowlist check → default-model resolution → in-process
    _load_pipeline + generate (with allow_hf_download=False hard-coded)
    → inline resolved-params response.
    """
    t0 = time.monotonic()
    if name != "generate":
        _emit_audit_line(
            name, arguments, status="error",
            error_class="UnknownTool",
            elapsed_seconds=time.monotonic() - t0,
        )
        raise ValueError(f"Unknown tool: {name!r}")

    # Invariant 5: emit audit line on every rejection. The step-1 stub
    # rejects every call uniformly.
    _emit_audit_line(
        name, arguments, status="error",
        error_class="NotImplementedYet",
        elapsed_seconds=time.monotonic() - t0,
    )
    raise NotImplementedError(
        "generate handler is scaffolded in slice 1 step 1; the actual "
        "wiring lands in step 2. See "
        "docs/vision/slice-1-mcp-generate.md"
    )


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
