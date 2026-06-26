#!/usr/bin/env python3
"""Test suite for comfyless/mcp_server.py — slice 1 step 1 surface.

Coverage (slice-1 Vision negative cases):
  - N1-N4   startup fail-closed for --output-dir / --model-base
  - N17, N18 startup fail-closed for --default-model
            (incl. symlink escape under realpath)
  - N13     stdout reserved for JSON-RPC; audit lines on stderr
  - N14     `generate` input schema does NOT accept `max_iterations`
            (and no other iterate-only fields)
  - N32     AST grep: no `import argparse` in mcp_server.py; no calls
            to _run_cli_mode / _apply_overrides / _load_params_file
            from comfyless.generate
  - N33     regex grep: comfyless.generate._run_json_mode source carries
            a legacy-marker referencing ADR-011 §5

Plus step-1 positive-path / structural coverage:
  - tools/list advertises ["generate", "list_models", "list_loras"]
    (slice-2 step-4 updates slice-1 invariant 6's count from 1 → 3)
  - audit line on tool rejection drops prompt + negative_prompt
                                                   (invariant 5)
  - audit line written to stderr (not stdout)      (invariant 5)
  - startup config retains realpath-resolved roots (invariants 1, 10)
  - traceback strip helper does not leak `Traceback (most recent call
    last)` / `.py:` / absolute `/home/` paths in its return string
                                                   (invariant 13 unit)

Negative cases for the generate-tool body (N5-N12, N15-N16, N19-N22,
N23-N31) land in step 2 + step 3 — they exercise behavior this step's
stub deliberately does not implement.

Run via: ./.venv/bin/python3 test_mcp_server.py
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import io
import json
import os
import re
import sys
import tempfile
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import click  # noqa: E402
import comfyless.catalog as cat_mod  # noqa: E402
import comfyless.mcp_server as mcps  # noqa: E402
import comfyless.generate as gen_mod  # noqa: E402
from click.testing import CliRunner  # noqa: E402
from mcp.types import Tool  # noqa: E402
from nodes.eric_diffusion_utils import infer_model_family  # noqa: E402


passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def _run(coro):
    return asyncio.run(coro)


# ════════════════════════════════════════════════════════════════════════
print("\n== N1-N4: --output-dir / --model-base fail-closed at startup ==")
# ════════════════════════════════════════════════════════════════════════

runner = CliRunner()

# N1: missing --output-dir → click rejects required option
with tempfile.TemporaryDirectory() as tmp_base:
    result = runner.invoke(mcps.main, ["--model-base", tmp_base])
    check("N1: missing --output-dir → non-zero exit",
          result.exit_code != 0,
          detail=f"exit={result.exit_code} stderr={result.stderr!r}")
    check("N1: missing --output-dir error names the flag",
          "--output-dir" in result.output or "--output-dir" in (result.stderr or ""))

# N2: --output-dir points at nonexistent path → click.Path or our realpath rejects
with tempfile.TemporaryDirectory() as tmp_base:
    fake_out = "/nonexistent/comfyless-test-out-xyzzy"
    result = runner.invoke(mcps.main, [
        "--output-dir", fake_out,
        "--model-base", tmp_base,
    ])
    check("N2: nonexistent --output-dir → non-zero exit",
          result.exit_code != 0,
          detail=f"exit={result.exit_code}")

# N3: --output-dir points at a regular file (not a directory)
with tempfile.NamedTemporaryFile() as f, tempfile.TemporaryDirectory() as tmp_base:
    result = runner.invoke(mcps.main, [
        "--output-dir", f.name,
        "--model-base", tmp_base,
    ])
    check("N3: --output-dir is a file, not dir → non-zero exit",
          result.exit_code != 0,
          detail=f"exit={result.exit_code}")

# N4 (output-dir half): missing --model-base
with tempfile.TemporaryDirectory() as tmp_out:
    result = runner.invoke(mcps.main, ["--output-dir", tmp_out])
    check("N4a: missing --model-base → non-zero exit",
          result.exit_code != 0)

# N4 (nonexistent): bad --model-base path
with tempfile.TemporaryDirectory() as tmp_out:
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", "/nonexistent/comfyless-test-base-xyzzy",
    ])
    check("N4b: nonexistent --model-base → non-zero exit",
          result.exit_code != 0)

# N4 (file not dir): --model-base is a file
with tempfile.NamedTemporaryFile() as f, tempfile.TemporaryDirectory() as tmp_out:
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", f.name,
    ])
    check("N4c: --model-base is a file, not dir → non-zero exit",
          result.exit_code != 0)


# ════════════════════════════════════════════════════════════════════════
print("\n== N17-N18: --default-model fail-closed at startup ==")
# ════════════════════════════════════════════════════════════════════════

# N17: --default-model resolves outside --model-base
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base, \
     tempfile.TemporaryDirectory() as outside:
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", tmp_base,
        "--default-model", outside,
    ])
    check("N17: --default-model outside --model-base → non-zero exit",
          result.exit_code != 0,
          detail=f"exit={result.exit_code}")

# N17b: nonexistent --default-model path
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", tmp_base,
        "--default-model", "/nonexistent/default-model-xyzzy",
    ])
    check("N17b: nonexistent --default-model → non-zero exit",
          result.exit_code != 0)

# N18: --default-model is a symlink under --model-base pointing OUTSIDE
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base, \
     tempfile.TemporaryDirectory() as outside:
    # Create a subdir at the symlink target so realpath -> isdir succeeds
    real_outside = os.path.join(outside, "real-model")
    os.makedirs(real_outside)
    symlink_path = os.path.join(tmp_base, "default-symlink")
    os.symlink(real_outside, symlink_path)
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", tmp_base,
        "--default-model", symlink_path,
    ])
    check("N18: --default-model symlink escapes --model-base after realpath → non-zero exit",
          result.exit_code != 0,
          detail=f"exit={result.exit_code}; output={result.output!r}")

# Positive: --default-model under --model-base passes startup (we still
# can't run the full server in a unit test, so we only assert that the
# validator does NOT raise click.BadParameter for a valid case).
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    valid_default = os.path.join(tmp_base, "model-x")
    os.makedirs(valid_default)
    try:
        cfg = mcps._validate_startup_args(
            output_dir=tmp_out,
            model_base=tmp_base,
            default_model=valid_default,
            mcp_max_iterations=100,
        )
        check("Positive: --default-model under --model-base passes startup",
              cfg.default_model == os.path.realpath(valid_default))
    except Exception as e:
        check("Positive: --default-model under --model-base passes startup",
              False, detail=str(e))


# ════════════════════════════════════════════════════════════════════════
print("\n== N14: generate schema rejects iterate-only fields ==")
# ════════════════════════════════════════════════════════════════════════

schema = mcps._GENERATE_INPUT_SCHEMA
check("schema: additionalProperties is False (structural enforcement)",
      schema.get("additionalProperties") is False)
check("schema: `max_iterations` NOT in properties (N14)",
      "max_iterations" not in schema["properties"])
check("schema: `axes` NOT in properties (iterate-only)",
      "axes" not in schema["properties"])
check("schema: `limit` NOT in properties (iterate-only)",
      "limit" not in schema["properties"])
check("schema: `batch` NOT in properties (iterate-only)",
      "batch" not in schema["properties"])
check("schema: `image_path` NOT in properties (edit-only)",
      "image_path" not in schema["properties"])

# Positive: required fields present
check("schema: `prompt` is required",
      "prompt" in schema["required"])
check("schema: `model` IS in properties (optional in step 1; required-presence is request-time)",
      "model" in schema["properties"])
check("schema: `cascade_config` IS in properties (slot reserved for step 3)",
      "cascade_config" in schema["properties"])


# ════════════════════════════════════════════════════════════════════════
print("\n== Invariant 8 (was slice-1 inv 6): tools/list returns 3 tools ==")
# ════════════════════════════════════════════════════════════════════════
# Slice 2 step 4 updated the slice-1 invariant 6 count from 1 → 3.
# Slice 2b invariant 1 updates it again 3 → 4: `generate` (slice 1 —
# schema + description unchanged per slice-2 Vision invariant 14),
# `list_models`, `list_loras` (slice 2), `list_transformers` (slice 2b).

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    tools = _run(mcps._list_tools_impl(cfg))
    check("Slice-2b invariant 1: list_tools returns 4 elements",
          isinstance(tools, list) and len(tools) == 4,
          detail=f"len={len(tools)}")
    check("Slice-2b invariant 1: every element is a Tool",
          all(isinstance(t, Tool) for t in tools))
    _tool_names = sorted(t.name for t in tools)
    check("Slice-2b invariant 1: tool names are exactly "
          "{generate, list_models, list_loras, list_transformers}",
          _tool_names == ["generate", "list_loras", "list_models",
                          "list_transformers"],
          detail=f"names={_tool_names}")
    _tools_by_name = {t.name: t for t in tools}
    check("Invariant 8: 'generate' tool description still names qwen-image "
          "(slice-1 description preserved per invariant 14)",
          _tools_by_name["generate"].description
          and "qwen-image" in _tools_by_name["generate"].description.lower())
    check("Invariant 8: 'list_models' tool description names model_family",
          _tools_by_name["list_models"].description
          and "model_family" in _tools_by_name["list_models"].description)
    check("Invariant 8: 'list_loras' tool description names target_family",
          _tools_by_name["list_loras"].description
          and "target_family" in _tools_by_name["list_loras"].description)
    check("Slice-2b invariant 1: 'list_transformers' tool description names "
          "transformer + empty inputSchema",
          _tools_by_name["list_transformers"].description
          and "transformer" in _tools_by_name["list_transformers"].description
          and _tools_by_name["list_transformers"].inputSchema.get("properties") == {}
          and _tools_by_name["list_transformers"].inputSchema.get(
              "additionalProperties") is False)


# ════════════════════════════════════════════════════════════════════════
print("\n== Invariant 5: audit-line discipline (stderr; prompt redacted) ==")
# ════════════════════════════════════════════════════════════════════════

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )

    # Capture stderr/stdout while the stub handler runs.
    captured_err = io.StringIO()
    captured_out = io.StringIO()
    with unittest.mock.patch.object(sys, "stderr", captured_err), \
         unittest.mock.patch.object(sys, "stdout", captured_out):
        try:
            _run(mcps._call_tool_impl(cfg, "generate", {
                "prompt": "SECRET prompt text DO NOT LEAK",
                "negative_prompt": "SECRET negative",
                "model": "/some/path",
                "seed": 42,
            }))
        except ValueError:
            pass  # step 2: PathAllowlist rejection (model outside --model-base)

    err_text = captured_err.getvalue()
    out_text = captured_out.getvalue()

    # N13: stderr carries the audit line; stdout does NOT.
    check("N13: audit line lands on stderr (rejection path)",
          err_text.strip() != "",
          detail=f"stderr={err_text!r}")
    check("N13: stdout was NOT used for audit (rejection path)",
          out_text == "",
          detail=f"stdout leaked: {out_text!r}")

    # Parse the audit line. It's the LAST line on stderr per current impl;
    # find the first line that looks like our structured JSON record.
    import json as _json
    audit_record = None
    for line in err_text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = _json.loads(line)
            if obj.get("tool") and "status" in obj:
                audit_record = obj
                break
        except _json.JSONDecodeError:
            continue
    check("audit line parsed as JSON with tool/status fields",
          audit_record is not None,
          detail=f"stderr lines: {err_text!r}")

    if audit_record is not None:
        # invariant 5: prompt + negative_prompt dropped
        check("audit line drops `prompt` field",
              "prompt" not in audit_record.get("input", {}),
              detail=f"input keys: {list(audit_record.get('input', {}).keys())}")
        check("audit line drops `negative_prompt` field",
              "negative_prompt" not in audit_record.get("input", {}))
        # path-typed fields retained
        check("audit line retains `model` field (paths kept)",
              audit_record.get("input", {}).get("model") == "/some/path")
        check("audit line retains `seed` field (non-path params kept)",
              audit_record.get("input", {}).get("seed") == 42)
        # error_class names the reference-resolution cause (slice 3): a
        # path-shaped model whose basename ("path") is not a catalog entry
        # -> UnknownName. The agent saw the uniform "reference not available";
        # the operator audit retains the fine cause.
        check("audit line error_class == 'UnknownName' (slice-3 reference)",
              audit_record.get("error_class") == "UnknownName",
              detail=f"actual: {audit_record.get('error_class')!r}")
        check("audit line elapsed_seconds is a number",
              isinstance(audit_record.get("elapsed_seconds"), (int, float)))

# Unknown-tool audit-line emission
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    captured_err = io.StringIO()
    with unittest.mock.patch.object(sys, "stderr", captured_err):
        try:
            _run(mcps._call_tool_impl(cfg, "definitely_not_a_tool", {
                "prompt": "SECRET should not appear here either",
            }))
        except ValueError:
            pass
    err_text = captured_err.getvalue()
    check("unknown-tool call still emits audit line on stderr",
          "definitely_not_a_tool" in err_text and "UnknownTool" in err_text)
    check("unknown-tool audit line still drops `prompt`",
          "SECRET should not appear" not in err_text)


# ════════════════════════════════════════════════════════════════════════
print("\n== Invariant 13 unit: traceback strip ==")
# ════════════════════════════════════════════════════════════════════════

# Construct an exception and feed it through the sanitizer; assert the
# returned string contains none of the forbidden patterns.
try:
    raise RuntimeError("simulated /home/gawkahn/secret.py:42 internal error")
except RuntimeError as e:
    captured_err = io.StringIO()
    with unittest.mock.patch.object(sys, "stderr", captured_err):
        sanitized = mcps._sanitize_error(e, "loader_failure")

    check("sanitize: returned string has no 'Traceback (most recent call last)'",
          "Traceback (most recent call last)" not in sanitized)
    check("sanitize: returned string has no '.py:' line refs",
          not re.search(r"\.py:\d", sanitized))
    check("sanitize: returned string has no absolute /home/ paths",
          "/home/" not in sanitized)
    check("sanitize: returned string is brief and category-shaped",
          sanitized.startswith("loader_failure:") and "RuntimeError" in sanitized,
          detail=f"sanitized={sanitized!r}")
    # The full traceback DID land on stderr (audit stream)
    err_text = captured_err.getvalue()
    check("sanitize: full traceback redirected to stderr (audit stream)",
          "Traceback" in err_text and "simulated" in err_text)


# ════════════════════════════════════════════════════════════════════════
print("\n== N32: AST grep for argparse + CLI-dispatch absence ==")
# ════════════════════════════════════════════════════════════════════════

_mcp_src = inspect.getsource(mcps)
_mcp_tree = ast.parse(_mcp_src)

# N32a: no `import argparse` / `from argparse import ...`
argparse_imports = []
for node in ast.walk(_mcp_tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "argparse":
                argparse_imports.append(node)
    elif isinstance(node, ast.ImportFrom):
        if node.module == "argparse":
            argparse_imports.append(node)
check("N32a: no `import argparse` in mcp_server.py",
      argparse_imports == [],
      detail=f"found: {[ast.dump(n) for n in argparse_imports]}")

# N32b: no calls to _run_cli_mode / _apply_overrides / _load_params_file
forbidden_callees = {"_run_cli_mode", "_apply_overrides", "_load_params_file"}
forbidden_call_hits = []
for node in ast.walk(_mcp_tree):
    if isinstance(node, ast.Call):
        func = node.func
        # `_run_cli_mode(...)`
        if isinstance(func, ast.Name) and func.id in forbidden_callees:
            forbidden_call_hits.append((func.id, getattr(node, "lineno", "?")))
        # `gen_mod._run_cli_mode(...)` or similar attribute access
        elif isinstance(func, ast.Attribute) and func.attr in forbidden_callees:
            forbidden_call_hits.append((func.attr, getattr(node, "lineno", "?")))
check("N32b: no calls to _run_cli_mode / _apply_overrides / _load_params_file",
      forbidden_call_hits == [],
      detail=f"found: {forbidden_call_hits}")


# ════════════════════════════════════════════════════════════════════════
print("\n== N33: comfyless.generate._run_json_mode legacy marker ==")
# ════════════════════════════════════════════════════════════════════════

_json_mode_src = inspect.getsource(gen_mod._run_json_mode)
# Marker must reference ADR-011 §5 OR the word "legacy" OR "MCP supersedes"
# within the first ~20 source lines.
_marker_window = "\n".join(_json_mode_src.splitlines()[:20])
markers_found = []
if "ADR-011" in _marker_window and "§5" in _marker_window:
    markers_found.append("ADR-011 §5")
if "LEGACY" in _marker_window or "legacy" in _marker_window:
    markers_found.append("legacy")
if "MCP supersedes" in _marker_window:
    markers_found.append("MCP supersedes")
check("N33: _run_json_mode source carries a legacy marker (≥1 of "
      "'ADR-011 §5' / 'legacy' / 'MCP supersedes')",
      len(markers_found) >= 1,
      detail=f"markers found in first 20 lines: {markers_found}")


# ════════════════════════════════════════════════════════════════════════
print("\n== Invariant 1/10: startup config retains realpath-resolved roots ==")
# ════════════════════════════════════════════════════════════════════════

with tempfile.TemporaryDirectory() as tmp_out_raw, \
     tempfile.TemporaryDirectory() as tmp_base_raw:
    # Create a symlink pointing at a real dir so realpath has work to do
    real_out = os.path.join(tmp_out_raw, "real-out")
    os.makedirs(real_out)
    out_symlink = os.path.join(tmp_out_raw, "out-link")
    os.symlink(real_out, out_symlink)

    cfg = mcps._validate_startup_args(
        output_dir=out_symlink, model_base=tmp_base_raw,
        default_model=None, mcp_max_iterations=200,
    )
    check("startup: output_dir realpath-resolved (symlink followed)",
          cfg.output_dir == os.path.realpath(out_symlink))
    check("startup: model_base realpath-resolved",
          cfg.model_base == os.path.realpath(tmp_base_raw))
    check("startup: mcp_max_iterations retained",
          cfg.mcp_max_iterations == 200)
    check("startup: default_model is None when unset",
          cfg.default_model is None)


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 2: redact_metadata_for_png (invariant 12 unit) ==")
# ════════════════════════════════════════════════════════════════════════

# N26 + N28: top-level path-typed fields → basenames; non-path fields verbatim
md = {
    "model": "/abs/path/to/qwen-image-dir",
    "transformer_path": "/abs/path/to/transformer-dir",
    "vae_path": "/abs/path/to/vae",
    "text_encoder_path": "/abs/path/to/te1",
    "text_encoder_2_path": "/abs/path/to/te2",
    "prompt": "test prompt",
    "negative_prompt": "no",
    "seed": 42,
    "steps": 28,
    "cfg_scale": 3.5,
    "true_cfg_scale": None,
    "sampler": "default",
    "scheduler": "linear",
    "width": 1024,
    "height": 1024,
    "model_family": "qwen-image",
    "loras": [
        {"path": "/abs/path/to/loraA.safetensors", "weight": 0.8},
        {"path": "/abs/path/to/loraB.safetensors", "weight": 0.5},
    ],
    "output_path": "/abs/path/to/output_0001.png",
    "savepath": "{model}_{seed}.png",
}
red = mcps.redact_metadata_for_png(md)

check("N26: model basename-redacted", red["model"] == "qwen-image-dir")
check("N26: transformer_path basename-redacted",
      red["transformer_path"] == "transformer-dir")
check("N26: vae_path basename-redacted", red["vae_path"] == "vae")
check("N26: text_encoder_path basename-redacted", red["text_encoder_path"] == "te1")
check("N26: text_encoder_2_path basename-redacted",
      red["text_encoder_2_path"] == "te2")
check("N26: loras[0].path basename-redacted",
      red["loras"][0]["path"] == "loraA.safetensors")
check("N26: loras[1].path basename-redacted",
      red["loras"][1]["path"] == "loraB.safetensors")
check("N26: loras[*].weight RETAINED (non-path field)",
      red["loras"][0]["weight"] == 0.8 and red["loras"][1]["weight"] == 0.5)

# N27: output_path + savepath dropped entirely
check("N27: output_path DROPPED from redacted dict",
      "output_path" not in red)
check("N27: savepath DROPPED from redacted dict",
      "savepath" not in red)

# N28: non-path fields retained verbatim
for key in ("prompt", "negative_prompt", "seed", "steps", "cfg_scale",
            "true_cfg_scale", "sampler", "scheduler", "width", "height",
            "model_family"):
    check(f"N28: `{key}` retained verbatim",
          red.get(key) == md.get(key),
          detail=f"input={md.get(key)!r} → output={red.get(key)!r}")

# Purity check: input not mutated
check("redact_metadata_for_png does NOT mutate input",
      md["model"] == "/abs/path/to/qwen-image-dir" and "output_path" in md)

# N30: HF repo IDs pass through unchanged
md_hf = {
    "model": "Qwen/Qwen-Image",
    "transformer_path": "diffusers/transformer-x",
    "loras": [
        {"path": "ostris/lora-a", "weight": 0.5},
        {"path": "/abs/path/local-lora.safetensors", "weight": 0.7},
    ],
}
red_hf = mcps.redact_metadata_for_png(md_hf)
check("N30: HF repo-ID `model` passes through unchanged",
      red_hf["model"] == "Qwen/Qwen-Image")
check("N30: HF repo-ID `transformer_path` passes through unchanged",
      red_hf["transformer_path"] == "diffusers/transformer-x")
check("N30: HF repo-ID loras[0].path passes through unchanged",
      red_hf["loras"][0]["path"] == "ostris/lora-a")
check("N30: local-path lora ALSO basenamed (mixed input)",
      red_hf["loras"][1]["path"] == "local-lora.safetensors")


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 2: _save_with_metadata mcp_caller branch (PNG tEXt write) ==")
# ════════════════════════════════════════════════════════════════════════

from PIL import Image  # noqa: E402

def _read_comfyless_chunk(png_path: str) -> dict:
    from PIL.PngImagePlugin import PngImageFile
    img = PngImageFile(png_path)
    raw = img.text.get("comfyless") if hasattr(img, "text") else None
    if raw is None:
        info = img.info
        raw = info.get("comfyless")
    if raw is None:
        return {}
    import json as _json
    return _json.loads(raw)

# Build a 16×16 fixture image; save with mcp_caller=True; assert the
# embedded chunk reflects the redaction map.
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    mcp_png_path = f.name
try:
    img = Image.new("RGB", (16, 16), "white")
    full_meta = {
        "model": "/m/qwen/qwen-image",
        "transformer_path": "/m/components/transformer",
        "vae_path": "",
        "text_encoder_path": "",
        "text_encoder_2_path": "",
        "loras": [{"path": "/m/loras/style.safetensors", "weight": 0.8}],
        "prompt": "redaction smoke",
        "negative_prompt": "",
        "seed": 1234,
        "steps": 20,
        "cfg_scale": 3.0,
        "true_cfg_scale": None,
        "sampler": "default",
        "schedule": "linear",
        "width": 16,
        "height": 16,
        "model_family": "qwen-image",
        "output_path": mcp_png_path,
        "savepath": None,
        "elapsed_seconds": 0.1,
    }
    gen_mod._save_with_metadata(img, mcp_png_path, full_meta, mcp_caller=True)
    embedded = _read_comfyless_chunk(mcp_png_path)
    check("PNG (mcp_caller=True) embeds basename for `model`",
          embedded.get("model") == "qwen-image")
    check("PNG (mcp_caller=True) embeds basename for `transformer_path`",
          embedded.get("transformer_path") == "transformer")
    check("PNG (mcp_caller=True) embeds basename for loras[0].path",
          embedded.get("loras", [{}])[0].get("path") == "style.safetensors")
    check("PNG (mcp_caller=True) DROPS output_path",
          "output_path" not in embedded)
    check("PNG (mcp_caller=True) DROPS savepath",
          "savepath" not in embedded)
    check("PNG (mcp_caller=True) retains prompt verbatim",
          embedded.get("prompt") == "redaction smoke")
    check("PNG (mcp_caller=True) retains seed verbatim",
          embedded.get("seed") == 1234)
finally:
    try: os.unlink(mcp_png_path)
    except OSError: pass

# N29: CLI-driven generate (mcp_caller=False / default) RETAINS full paths.
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
    cli_png_path = f.name
try:
    img = Image.new("RGB", (16, 16), "white")
    cli_meta = {
        "model": "/m/qwen/qwen-image",
        "transformer_path": "/m/components/transformer",
        "loras": [{"path": "/m/loras/style.safetensors", "weight": 0.8}],
        "prompt": "cli path",
        "output_path": cli_png_path,
    }
    # No mcp_caller kwarg — default False.
    gen_mod._save_with_metadata(img, cli_png_path, cli_meta)
    embedded = _read_comfyless_chunk(cli_png_path)
    check("N29: CLI PNG retains full `model` path",
          embedded.get("model") == "/m/qwen/qwen-image")
    check("N29: CLI PNG retains full loras[0].path",
          embedded.get("loras", [{}])[0].get("path") == "/m/loras/style.safetensors")
    check("N29: CLI PNG retains `output_path`",
          embedded.get("output_path") == cli_png_path)
finally:
    try: os.unlink(cli_png_path)
    except OSError: pass


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 2: path-allowlist (N5-N9) ==")
# ════════════════════════════════════════════════════════════════════════
#
# These tests exercise _call_tool_impl with mocked _load_pipeline and
# generate (so no torch / model load); the assertion is about whether the
# pre-load path-validation correctly rejects.

import json as _json  # noqa: E402

class _FakePipe:
    """Minimal mock for a diffusers pipeline (never called in these tests)."""
    pass


def _mock_load_pipeline(*a, **kw):
    return (_FakePipe(), "qwen-image", False)


def _mock_generate(*, model_path, prompt, output_path, **kw):
    # Write a fake PNG so callers that check the file see something. Echo the
    # path-typed fields the real generate() returns (model/transformer_path/
    # loras) so the slice-3 names-rendering (_resolved_params_as_names) has the
    # same keys to remap to catalog names.
    Image.new("RGB", (8, 8), "white").save(output_path)
    return {
        "prompt": prompt, "negative_prompt": kw.get("negative_prompt", ""),
        "model": model_path, "seed": kw.get("seed", 42),
        "steps": kw.get("steps", 28), "cfg_scale": kw.get("cfg_scale", 3.5),
        "transformer_path": kw.get("transformer_path", ""),
        "loras": list(kw.get("loras") or []),
        "elapsed_seconds": 0.01,
    }


def _setup_mb_and_out():
    """Return (model_base, output_dir, model_dir_inside_base, cfg).

    Slice 3: the model dir gets a `model_index.json` so the spawn-time catalog
    mints a `kind:"model"` entry named "qwen-image"; a lora "test-lora" and a
    transformer "test-dit" are also seeded. Handler tests reference these by
    catalog NAME (or by a path-shaped value whose basename is a catalog name).
    All fixtures must exist BEFORE _validate_startup_args, which builds the
    catalog at that call.
    """
    mb = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    inside_model = os.path.join(mb, "qwen-image")
    os.makedirs(inside_model)
    with open(os.path.join(inside_model, "model_index.json"), "w",
              encoding="utf-8") as _f:
        json.dump({"_class_name": "QwenImagePipeline"}, _f)
    _lora_dir = os.path.join(mb, "loras")
    os.makedirs(_lora_dir, exist_ok=True)
    with open(os.path.join(_lora_dir, "test-lora.safetensors"), "wb") as _f:
        _f.write(b"fake-lora")
    _dit_dir = os.path.join(mb, "diffusion_models")
    os.makedirs(_dit_dir, exist_ok=True)
    with open(os.path.join(_dit_dir, "test-dit.safetensors"), "wb") as _f:
        _f.write(b"fake-dit")
    cfg = mcps._validate_startup_args(
        output_dir=out, model_base=mb,
        default_model=None, mcp_max_iterations=100,
    )
    return mb, out, inside_model, cfg


def _call(cfg, args, *, expect_error_class=None):
    """Invoke _call_tool_impl with mocked load+generate; capture stderr."""
    captured_err = io.StringIO()
    raised = None
    result = None
    with unittest.mock.patch.object(sys, "stderr", captured_err), \
         unittest.mock.patch.object(gen_mod, "_load_pipeline", _mock_load_pipeline), \
         unittest.mock.patch.object(gen_mod, "generate", _mock_generate):
        try:
            result = _run(mcps._call_tool_impl(cfg, "generate", args))
        except ValueError as e:
            raised = str(e)
        except BaseException as e:
            raised = f"UNEXPECTED-{type(e).__name__}: {e}"
    return result, raised, captured_err.getvalue()


# eu is still needed by the cascade tests below (they spy on
# eu.resolve_hf_path). Cascade is NOT migrated in slice 3 (OQ-C -> slice 3b),
# so its raw-path + HF-resolution contract is unchanged and still covered by
# the "Step 3: cascade dispatch" section further down.
import nodes.eric_diffusion_utils as eu  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
print("\n== Slice 3 Step 2: generate catalog-name migration (N1-N17) ==")
# ════════════════════════════════════════════════════════════════════════
#
# Replaces the slice-1 path-allowlist (old N5-N9) and HF-cache (old N10-N11)
# handler tests: the names contract removes the raw-path input vector those
# probed (the agent supplies catalog names; a path's basename is looked up).
# What remains is the load-bearing uniform-error contract (ADR-015 §2 step 2 /
# HIGH-1) + the resolved_params-as-names output (§3) + the path-discard notice
# (§2 INFO-2). _setup_mb_and_out seeds catalog names: model "qwen-image",
# lora "test-lora", transformer "test-dit".


def _audit_error_class(stderr_text):
    """Return the error_class from the first status=error audit line, or None."""
    for _l in stderr_text.splitlines():
        _l = _l.strip()
        if not _l.startswith("{"):
            continue
        try:
            _o = _json.loads(_l)
        except _json.JSONDecodeError:
            continue
        if _o.get("status") == "error":
            return _o.get("error_class")
    return None


_UNIFORM = "reference not available"

# --- N1: unknown model name -> uniform error; audit cause UnknownName ---
mb, out, _inside, cfg = _setup_mb_and_out()
r, e1, se1 = _call(cfg, {"prompt": "p", "model": "nonexistent-model"})
check("N1: unknown model name -> uniform 'reference not available'",
      r is None and e1 == _UNIFORM, detail=f"err={e1!r}")
check("N1: audit cause = UnknownName", _audit_error_class(se1) == "UnknownName",
      detail=f"cause={_audit_error_class(se1)!r}")

# --- N2: catalog hit whose abs_path vanished post-spawn -> uniform; PathMoved.
# (Also the request-time HF-cache-eviction case: the stored local path vanishes.)
mb, out, _inside, cfg = _setup_mb_and_out()
os.remove(os.path.join(mb, "loras", "test-lora.safetensors"))
r, e2, se2 = _call(cfg, {"prompt": "p", "model": "qwen-image",
                         "loras": [{"name": "test-lora", "weight": 0.5}]})
check("N2: deleted/moved catalog path -> uniform error (no fallback, no load)",
      r is None and e2 == _UNIFORM, detail=f"err={e2!r}")
check("N2: audit cause = PathMoved", _audit_error_class(se2) == "PathMoved",
      detail=f"cause={_audit_error_class(se2)!r}")

# --- N3: kind mismatch (lora name supplied as model) -> uniform; KindMismatch.
mb, out, _inside, cfg = _setup_mb_and_out()
r, e3, se3 = _call(cfg, {"prompt": "p", "model": "test-lora"})
check("N3: lora name supplied as model -> uniform error",
      r is None and e3 == _UNIFORM, detail=f"err={e3!r}")
check("N3: audit cause = KindMismatch", _audit_error_class(se3) == "KindMismatch",
      detail=f"cause={_audit_error_class(se3)!r}")

# --- N4: malformed (null byte in model) -> uniform; MalformedReference ---
mb, out, _inside, cfg = _setup_mb_and_out()
r, e4, se4 = _call(cfg, {"prompt": "p", "model": "qwen-image\x00x"})
check("N4: null byte in model -> uniform error",
      r is None and e4 == _UNIFORM, detail=f"err={e4!r}")
check("N4: audit cause = MalformedReference",
      _audit_error_class(se4) == "MalformedReference",
      detail=f"cause={_audit_error_class(se4)!r}")

# --- N5 (KEYSTONE): N1-N4 agent frames are BYTE-IDENTICAL (HIGH-1) ---
check("N5 (keystone): all reference-failure agent frames byte-identical",
      e1 == e2 == e3 == e4 == _UNIFORM,
      detail=f"{e1!r} {e2!r} {e3!r} {e4!r}")

# --- N6: the fine causes ARE distinct on the operator audit (oracle stays
# operator-side, not agent-side). ---
check("N6: operator audit causes are distinct across N1-N4",
      len({_audit_error_class(se1), _audit_error_class(se2),
           _audit_error_class(se3), _audit_error_class(se4)}) == 4,
      detail=f"{_audit_error_class(se1)},{_audit_error_class(se2)},"
             f"{_audit_error_class(se3)},{_audit_error_class(se4)}")

# --- N7: bare valid model name -> success; resolved_params.model is the name;
# no path-discard notice (nothing discarded). ---
mb, out, _inside, cfg = _setup_mb_and_out()
r, e, se = _call(cfg, {"prompt": "p", "model": "qwen-image"})
check("N7: bare model name -> success", r is not None and e is None,
      detail=f"err={e!r}")
_obj = _json.loads(r[0].text) if r else {}
check("N7: resolved_params.model is the catalog name",
      _obj.get("resolved_params", {}).get("model") == "qwen-image",
      detail=f"model={_obj.get('resolved_params', {}).get('model')!r}")
check("N7: no path-discard notice for a bare name",
      all("discarded" not in n.get("message", "")
          for n in _obj.get("notices", [])))

# --- N8: path-shaped model -> success via basename; discard notice present. ---
mb, out, _inside, cfg = _setup_mb_and_out()
r, e, se = _call(cfg, {"prompt": "p",
                       "model": "/agent/hallucinated/dir/qwen-image"})
check("N8: path-shaped model resolves via basename -> success",
      r is not None and e is None, detail=f"err={e!r}")
_obj = _json.loads(r[0].text) if r else {}
check("N8: resolved_params.model is the catalog name (not the supplied dir)",
      _obj.get("resolved_params", {}).get("model") == "qwen-image")
check("N8: path-discard INFO notice present",
      any(n.get("level") == "INFO" and "discarded" in n.get("message", "")
          for n in _obj.get("notices", [])))

# --- N9 (INFO-2): notice carries the resolved NAME, never the supplied dir. ---
check("N9: notice text contains the resolved catalog name",
      any("qwen-image" in n.get("message", "")
          for n in _obj.get("notices", [])))
check("N9: supplied directory text never round-trips into the response",
      r is not None and "/agent/hallucinated/dir" not in r[0].text)

# --- N10: loras[].name resolves; resolved_params.loras = name+weight; the old
# loras[].path key is rejected (contract break). ---
mb, out, _inside, cfg = _setup_mb_and_out()
r, e, se = _call(cfg, {"prompt": "p", "model": "qwen-image",
                       "loras": [{"name": "test-lora", "weight": 0.8}]})
check("N10: loras[].name resolves -> success", r is not None and e is None,
      detail=f"err={e!r}")
_obj = _json.loads(r[0].text) if r else {}
check("N10: resolved_params.loras renders name+weight (no path)",
      _obj.get("resolved_params", {}).get("loras") == [{"name": "test-lora",
                                                        "weight": 0.8}],
      detail=f"loras={_obj.get('resolved_params', {}).get('loras')!r}")
mb, out, _inside, cfg = _setup_mb_and_out()
r, e, se = _call(cfg, {"prompt": "p", "model": "qwen-image",
                       "loras": [{"path": "test-lora", "weight": 0.5}]})
check("N10: old loras[].path key rejected (name required)",
      r is None and e is not None and "loras[0].name" in e, detail=f"err={e!r}")

# --- N10b: inline loras shape validation (code-reviewer gap 1). Each malformed
# shape returns its NAMED ValidationError/MissingField (these are contract
# errors, not reference-resolution failures, so naming the field is correct). ---
mb, out, _inside, cfg = _setup_mb_and_out()
for _bad, _needle in (
    ({"loras": "notalist"}, "loras: expected list"),
    ({"loras": ["notadict"]}, "loras[0]: expected object"),
    ({"loras": [{"weight": 1.0}]}, "loras[0].name: required field absent"),
    ({"loras": [{"name": "test-lora"}]}, "loras[0].weight: required field absent"),
    ({"loras": [{"name": "test-lora", "weight": True}]}, "loras[0].weight: expected number"),
    ({"loras": [{"name": "test-lora", "weight": "x"}]}, "loras[0].weight: expected number"),
):
    _a = {"prompt": "p", "model": "qwen-image", **_bad}
    r, e, se = _call(cfg, _a)
    check(f"N10b: malformed loras -> named error ({_needle})",
          r is None and e is not None and _needle in e, detail=f"err={e!r}")

# --- N10c: non-str transformer value -> uniform error (resolver is the sole
# type gate for transformer; code-reviewer Finding 3). ---
mb, out, _inside, cfg = _setup_mb_and_out()
r, e, se = _call(cfg, {"prompt": "p", "model": "qwen-image", "transformer": 123})
check("N10c: non-str transformer -> uniform error",
      r is None and e == _UNIFORM, detail=f"err={e!r}")

# --- N11: success response carries NO --model-base abs path anywhere; the
# transformer renders as a name and the dropped component keys are absent. ---
mb, out, _inside, cfg = _setup_mb_and_out()
r, e, se = _call(cfg, {"prompt": "p", "model": "qwen-image",
                       "transformer": "test-dit",
                       "loras": [{"name": "test-lora", "weight": 1.0}]})
_txt = r[0].text if r else ""
check("N11: no --model-base abs path in success response",
      r is not None and mb not in _txt and "/loras/" not in _txt
      and "/diffusion_models/" not in _txt, detail=f"resp={_txt[:160]!r}")
_obj = _json.loads(_txt) if r else {}
check("N11: resolved_params.transformer renders the catalog name",
      _obj.get("resolved_params", {}).get("transformer") == "test-dit")
check("N11: resolved_params has no transformer_path/vae_path keys",
      "transformer_path" not in _obj.get("resolved_params", {})
      and "vae_path" not in _obj.get("resolved_params", {}))

# --- N12: a uniform-error frame leaks neither an abs path nor the fine cause. ---
mb, out, _inside, cfg = _setup_mb_and_out()
r, e, se = _call(cfg, {"prompt": "p", "model": "/etc/passwd"})  # basename unknown
check("N12: uniform error leaks no path / no fine cause to the agent",
      e == _UNIFORM and "UnknownName" not in (e or "") and "/etc" not in (e or ""),
      detail=f"err={e!r}")

# --- N13: the audit line for a reference failure carries the fine cause but
# NO catalog abs_path. ---
check("N13: audit carries fine cause, not the model-base abs path",
      "UnknownName" in se and mb not in se)

# --- removed-field guard (OQ-A): the slice-1 raw-path keys are rejected. ---
mb, out, _inside, cfg = _setup_mb_and_out()
for _rf in ("transformer_path", "vae_path", "text_encoder_path",
            "text_encoder_2_path"):
    r, e, se = _call(cfg, {"prompt": "p", "model": "qwen-image", _rf: "x"})
    check(f"removed-field guard: `{_rf}` rejected as unsupported",
          r is None and e is not None and "field not supported" in e,
          detail=f"err={e!r}")

# --- N15: generate input schema migrated (rename + drop). ---
_props = mcps._GENERATE_INPUT_SCHEMA["properties"]
check("N15: schema has `transformer` (renamed), not `transformer_path`",
      "transformer" in _props and "transformer_path" not in _props)
check("N15: schema drops vae_path/text_encoder_path/text_encoder_2_path",
      not any(k in _props for k in ("vae_path", "text_encoder_path",
                                    "text_encoder_2_path")))
check("N15: loras item requires `name` not `path`",
      _props["loras"]["items"]["required"] == ["name", "weight"]
      and "name" in _props["loras"]["items"]["properties"]
      and "path" not in _props["loras"]["items"]["properties"])

# --- MEDIUM-1: lora_warnings (which embed the resolved abs_path) must NOT
# cross into resolved_params. Mock generate() to return a warning carrying a
# --model-base abs path; assert _resolved_params_as_names strips it. ---
mb, out, _inside, cfg = _setup_mb_and_out()


def _mock_generate_with_warning(*, model_path, prompt, output_path, **kw):
    Image.new("RGB", (8, 8), "white").save(output_path)
    return {
        "prompt": prompt, "model": model_path, "seed": 1,
        "loras": list(kw.get("loras") or []),
        "lora_warnings": [f"LoRA skipped (0 modules applied): {model_path}"],
        "elapsed_seconds": 0.01,
    }


with unittest.mock.patch.object(sys, "stderr", io.StringIO()), \
     unittest.mock.patch.object(gen_mod, "_load_pipeline", _mock_load_pipeline), \
     unittest.mock.patch.object(gen_mod, "generate", _mock_generate_with_warning):
    _rr = _run(mcps._call_tool_impl(cfg, "generate",
                                    {"prompt": "p", "model": "qwen-image"}))
_robj = _json.loads(_rr[0].text)
check("MEDIUM-1: lora_warnings stripped from resolved_params (no abs_path leak)",
      "lora_warnings" not in _robj.get("resolved_params", {}))
check("MEDIUM-1: no --model-base abs path anywhere in the response",
      mb not in _rr[0].text, detail=f"resp={_rr[0].text[:160]!r}")

# --- N14 (slice 3b): cascade IS now migrated to catalog names (OQ-C resolved).
# The handler must route stage references through resolve_reference and fold
# failures into the uniform error — the inverse of the slice-3 assertion.
import inspect as _inspect  # noqa: E402
_cascade_src = _inspect.getsource(mcps._handle_generate_cascade)
check("N14: _handle_generate_cascade NOW calls resolve_reference (3b migration)",
      "resolve_reference" in _cascade_src)
check("N14: _handle_generate_cascade uses the uniform-error + notice machinery",
      "_reference_error" in _cascade_src and "_discard_notice" in _cascade_src)
check("N14: cascade handler no longer emits the slice-1 PathAllowlist agent error",
      "PathAllowlist" not in _cascade_src and "HFCacheMiss" not in _cascade_src)


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 2: audit-line success path (N12) + traceback strip (N31) ==")
# ════════════════════════════════════════════════════════════════════════

mb, out, _inside, cfg = _setup_mb_and_out()
# N12: success path drops prompt + negative_prompt from audit
captured_err = io.StringIO()
captured_out = io.StringIO()
with unittest.mock.patch.object(sys, "stderr", captured_err), \
     unittest.mock.patch.object(sys, "stdout", captured_out), \
     unittest.mock.patch.object(gen_mod, "_load_pipeline", _mock_load_pipeline), \
     unittest.mock.patch.object(gen_mod, "generate", _mock_generate):
    try:
        result = _run(mcps._call_tool_impl(cfg, "generate", {
            "prompt": "SECRET prompt text",
            "negative_prompt": "SECRET negative",
            "model": _inside,
            "seed": 42,
        }))
    except Exception:
        result = None
err_text = captured_err.getvalue()
out_text = captured_out.getvalue()
check("N12 / N13: stderr carries audit line; stdout empty (success path)",
      err_text.strip() != "" and out_text == "")
check("N12: 'SECRET prompt text' NOT in stderr (prompt redacted)",
      "SECRET prompt text" not in err_text,
      detail=f"stderr={err_text[:200]!r}")
check("N12: 'SECRET negative' NOT in stderr (negative_prompt redacted)",
      "SECRET negative" not in err_text)
check("N12: audit line has status=ok",
      '"status": "ok"' in err_text or '"status":"ok"' in err_text)

# N25: inline resolved-params blob in MCP response frame
if result is not None and isinstance(result, list) and len(result) >= 1:
    response_text = result[0].text
    response_obj = _json.loads(response_text)
    check("N25: response is a TextContent with JSON body",
          isinstance(response_obj, dict) and "output_path" in response_obj)
    check("N25: response contains `resolved_params` (names blob)",
          "resolved_params" in response_obj)
    check("N25: resolved_params.model is the catalog name (not a path)",
          response_obj.get("resolved_params", {}).get("model") == "qwen-image")
    check("N25: response contains `elapsed_seconds`",
          "elapsed_seconds" in response_obj)
else:
    check("N25: response is a TextContent with JSON body", False,
          detail=f"result={result!r}")

# N23: no sidecar JSON written for MCP-driven generate
# After the success-path call above, --output-dir should contain only PNGs.
out_listing = os.listdir(cfg.output_dir)
jsons_in_out = [f for f in out_listing if f.endswith(".json")]
check("N23: no .json sidecars in --output-dir after MCP generate",
      jsons_in_out == [],
      detail=f"jsons found: {jsons_in_out}")

# N31: traceback strip — force _load_pipeline to raise a path-revealing
# exception; check the returned MCP error has no traceback / file paths.
mb, out, _inside, cfg = _setup_mb_and_out()
captured_err = io.StringIO()
def _exploding_load(*a, **kw):
    raise RuntimeError(
        "loader failure at /home/gawkahn/secret/internal.py:42 "
        "in module _internal_helper"
    )
raised_msg = None
with unittest.mock.patch.object(sys, "stderr", captured_err), \
     unittest.mock.patch.object(gen_mod, "_load_pipeline", _exploding_load), \
     unittest.mock.patch.object(gen_mod, "generate", _mock_generate):
    try:
        _run(mcps._call_tool_impl(cfg, "generate", {
            "prompt": "p", "model": _inside,
        }))
    except ValueError as e:
        raised_msg = str(e)

check("N31: MCP error message has no 'Traceback (most recent call last)'",
      raised_msg is not None and "Traceback (most recent call last)" not in raised_msg,
      detail=f"raised={raised_msg!r}")
check("N31: MCP error message has no '.py:' line refs",
      raised_msg is not None and not re.search(r"\.py:\d", raised_msg))
check("N31: MCP error message has no absolute /home/ paths",
      raised_msg is not None and "/home/" not in raised_msg)
check("N31: MCP error message has no '/secret/' internal-name hint",
      raised_msg is not None and "/secret/" not in raised_msg)
check("N31: stderr DID receive the full traceback (operator visibility)",
      "Traceback" in captured_err.getvalue() and
      "loader failure" in captured_err.getvalue())


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 2: default-model fallback (N15, N16) ==")
# ════════════════════════════════════════════════════════════════════════

# N15: model omitted, --default-model configured → uses default; success
mb, out, _inside, cfg = _setup_mb_and_out()
# Build a cfg that DOES have a default
default_dir = os.path.join(mb, "default-model-x")
os.makedirs(default_dir)
cfg_with_default = mcps._validate_startup_args(
    output_dir=cfg.output_dir, model_base=cfg.model_base,
    default_model=default_dir, mcp_max_iterations=100,
)
result, err, stderr = _call(cfg_with_default, {"prompt": "p"})
check("N15: model omitted + --default-model configured → success",
      result is not None and err is None,
      detail=f"err={err!r}")

# N16: model omitted, no --default-model → MCP error
mb, out, _inside, cfg_no_default = _setup_mb_and_out()
result, err, stderr = _call(cfg_no_default, {"prompt": "p"})
check("N16: model omitted + no --default-model → MCP error",
      result is None and err is not None and "validation failed: model" in err,
      detail=f"err={err!r}")


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 2 audit fold-in: null-byte + missing-prompt (F1, F2) ==")
# ════════════════════════════════════════════════════════════════════════

# Slice 3: a null byte in a REFERENCE field is a malformed reference -> the
# uniform "reference not available" frame (audit cause MalformedReference),
# NOT a named ValidationError (which would be an oracle on the field). The
# non-reference write-dest field `savepath` keeps its named ValidationError.
for nb_field in ("model", "transformer"):
    mb, out, _inside, cfg = _setup_mb_and_out()
    args = {"prompt": "p", "model": "qwen-image"}
    args[nb_field] = "qwen-image\x00null"
    result, err, stderr = _call(cfg, args)
    check(f"F1: null byte in reference `{nb_field}` -> uniform error",
          result is None and err == "reference not available",
          detail=f"err={err!r}")
    check(f"F1: null-byte reference `{nb_field}` audit cause MalformedReference",
          _audit_error_class(stderr) == "MalformedReference"
          and "InternalError" not in stderr)

# Null byte in loras[i].name -> uniform error (MalformedReference).
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, {
    "prompt": "p", "model": "qwen-image",
    "loras": [{"name": "test-lora\x00null", "weight": 0.5}],
})
check("F1: null byte in loras[0].name -> uniform error",
      result is None and err == "reference not available", detail=f"err={err!r}")
check("F1: loras null-byte audit cause MalformedReference",
      _audit_error_class(stderr) == "MalformedReference")

# Null byte in the non-reference field `savepath` -> named ValidationError.
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, {
    "prompt": "p", "model": "qwen-image", "savepath": "out\x00null.png",
})
check("F1: null byte in `savepath` -> named ValidationError (not a reference)",
      result is None and err is not None and "null byte not allowed" in err,
      detail=f"err={err!r}")
check("F1: savepath null-byte audit class is ValidationError",
      "ValidationError" in stderr and "InternalError" not in stderr)

# Security-auditor F2: missing prompt must be rejected BEFORE _load_pipeline
# wastes 30-90s. Audit class must be MissingField, NOT InternalError.
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, {"model": _inside})  # NO prompt
check("F2: missing `prompt` → MCP error (BEFORE load)",
      result is None and err is not None
      and "prompt: required field absent" in err)
check("F2: missing-prompt audit class is MissingField (not InternalError)",
      "MissingField" in stderr and "InternalError" not in stderr)

# Empty-string prompt also rejected
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, {"prompt": "   ", "model": _inside})
check("F2: whitespace-only `prompt` → MCP error",
      result is None and err is not None
      and "prompt: required field absent" in err)


# ════════════════════════════════════════════════════════════════════════
print("\n== Slice 3b: cascade dispatch — catalog-name migration ==")
# ════════════════════════════════════════════════════════════════════════
#
# cascade dispatch is exercised with mocked build_pipelines + run_one to
# isolate the reference-resolution surface from torch/diffusers load overhead.
# Slice 3b: stage_c/stage_b/stage_a are catalog NAMES resolved against kind
# {model, transformer}; scaffolding_repo is rejected if supplied; all failures
# fold into the uniform "reference not available" error (cause to audit only).

import comfyless.cascade as cas_mod  # noqa: E402

def _mock_cascade_build_pipelines(cfg_cc, device, allow_hf_download):
    return (_FakePipe(), _FakePipe())  # prior, decoder

def _mock_cascade_run_one(prior, decoder, cfg_cc, *, prompt, negative_prompt,
                          seed, device):
    img = Image.new("RGB", (16, 16), "white")
    return img, {"prior_seconds": 0.05, "decoder_seconds": 0.02}


def _call_cascade(cfg, args):
    """Invoke _call_tool_impl with cascade mocks; capture stderr.

    Slice 3b: stage references resolve against the spawn-time catalog
    (in cfg), so no HF-resolution mock is needed — the build-time HF
    resolution the slice-1 handler did per-request is gone.
    """
    captured_err = io.StringIO()
    raised = None
    result = None
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(unittest.mock.patch.object(sys, "stderr", captured_err))
        stack.enter_context(unittest.mock.patch.object(
            cas_mod, "build_pipelines", _mock_cascade_build_pipelines))
        stack.enter_context(unittest.mock.patch.object(
            cas_mod, "run_one", _mock_cascade_run_one))
        try:
            result = _run(mcps._call_tool_impl(cfg, "generate", args))
        except ValueError as e:
            raised = str(e)
    return result, raised, captured_err.getvalue()


def _good_cascade_args(stage_c="test-dit", stage_b="test-dit", stage_a=None):
    """Build a cascade_config that uses catalog NAMES (slice 3b).

    Defaults use `test-dit` (a kind:"transformer" catalog entry from
    _setup_mb_and_out) for both required stages — the common single-file
    cascade-UNet case. `scaffolding_repo` is NOT supplied: it is removed from
    the agent surface; the server uses cascade.validate_config's default.
    """
    cc: dict = {"stage_c": stage_c, "stage_b": stage_b}
    if stage_a is not None:
        cc["stage_a"] = stage_a
    return {"prompt": "a cat", "cascade_config": cc}


# Cascade happy-path: catalog NAMES resolve; response renders names, no paths.
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(cfg, _good_cascade_args())
check("cascade happy-path: success returns TextContent response",
      result is not None and err is None,
      detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    rp_cc = resp.get("resolved_params", {}).get("cascade_config", {})
    check("cascade happy-path: resolved_params carries cascade_config dict",
          isinstance(rp_cc, dict))
    check("cascade happy-path: response renders stage_c as the catalog NAME",
          rp_cc.get("stage_c") == "test-dit", detail=repr(rp_cc))
    check("cascade happy-path: response renders stage_b as the catalog NAME",
          rp_cc.get("stage_b") == "test-dit", detail=repr(rp_cc))
    # Invariant 1/4: scaffolding_repo never crosses the boundary.
    check("cascade happy-path: scaffolding_repo ABSENT from response",
          "scaffolding_repo" not in rp_cc, detail=repr(rp_cc))
    # Invariant 1: no path-shaped value under ANY cascade_config key.
    check("cascade happy-path: no '/'-bearing value in response cascade_config",
          all("/" not in str(v) for v in rp_cc.values()), detail=repr(rp_cc))
    check("cascade happy-path: response carries elapsed_seconds",
          "elapsed_seconds" in resp and resp["elapsed_seconds"] >= 0)
    # PNG sink (invariant 7, unchanged): stage_* basenamed, output_path dropped.
    if os.path.exists(resp["output_path"]):
        png_meta = _read_comfyless_chunk(resp["output_path"])
        cc_embedded = png_meta.get("cascade_config", {})
        check("cascade PNG embeds basename for cascade_config.stage_c",
              cc_embedded.get("stage_c") == "test-dit.safetensors",
              detail=repr(cc_embedded))
        check("cascade PNG drops output_path", "output_path" not in png_meta)
        check("cascade PNG retains `prompt` verbatim",
              png_meta.get("prompt") == "a cat")
        # Audit retains the stage NAME (operator-side; names are not paths) and
        # drops the prompt. No abs_path is in play on the agent surface.
        check("cascade audit retains stage_c name; drops prompt",
              "test-dit" in stderr and "a cat" not in stderr)

# Both kinds resolve: stage_c=transformer (test-dit), stage_b=model (qwen-image).
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(
    cfg, _good_cascade_args(stage_c="test-dit", stage_b="qwen-image"))
check("cascade: transformer-kind + model-kind stages both resolve",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    rp_cc = _json.loads(result[0].text)["resolved_params"]["cascade_config"]
    check("cascade: model-kind stage_b renders its catalog name",
          rp_cc.get("stage_b") == "qwen-image", detail=repr(rp_cc))

# stage_a present (optional third stage) resolves + renders its name.
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(
    cfg, _good_cascade_args(stage_a="qwen-image"))
check("cascade: optional stage_a resolves + renders name",
      result is not None and err is None
      and _json.loads(result[0].text)["resolved_params"]["cascade_config"]
          .get("stage_a") == "qwen-image",
      detail=f"err={err!r}")

# Path-shaped stage value → basename-strip → resolves + path-discard notice.
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(
    cfg, _good_cascade_args(stage_c="/some/agent/dir/test-dit"))
check("cascade: path-shaped stage_c resolves via basename",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    check("cascade: path-shaped stage_c → resolved to catalog name in response",
          resp["resolved_params"]["cascade_config"].get("stage_c") == "test-dit")
    _notices = resp.get("notices", [])
    check("cascade: path-shaped stage_c emits a path-discard INFO notice",
          any(n.get("level") == "INFO" and "test-dit" in n.get("message", "")
              for n in _notices), detail=repr(_notices))
    check("cascade: path-discard notice does NOT echo the supplied directory",
          all("/some/agent/dir" not in n.get("message", "") for n in _notices),
          detail=repr(_notices))

# --- Negative: unknown stage name → uniform error; audit cause UnknownName ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(
    cfg, _good_cascade_args(stage_c="does-not-exist"))
check("cascade: unknown stage name → uniform 'reference not available'",
      result is None and err == mcps._UNIFORM_REFERENCE_ERROR, detail=f"err={err!r}")
check("cascade: unknown stage name audited as UnknownName",
      "UnknownName" in stderr)

# --- Negative: lora-kind name as a stage → uniform error; KindMismatch ------
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(
    cfg, _good_cascade_args(stage_c="test-lora"))
check("cascade: lora-kind name as stage_c → uniform error (kind-set excludes lora)",
      result is None and err == mcps._UNIFORM_REFERENCE_ERROR, detail=f"err={err!r}")
check("cascade: lora-as-stage audited as KindMismatch",
      "KindMismatch" in stderr)

# --- Negative: path-shaped value that misses → uniform error; no echo -------
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(
    cfg, _good_cascade_args(stage_c="/etc/passwd"))
check("cascade: path-shaped miss (/etc/passwd) → uniform error",
      result is None and err == mcps._UNIFORM_REFERENCE_ERROR, detail=f"err={err!r}")
check("cascade: path-shaped miss does NOT echo the supplied path to the agent",
      err is not None and "/etc/passwd" not in err and "passwd" not in err)

# --- Negative: NUL byte in a stage name → uniform error; MalformedReference -
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(
    cfg, _good_cascade_args(stage_c="test-dit\x00null"))
check("cascade: NUL in stage_c → uniform error (no exception)",
      result is None and err == mcps._UNIFORM_REFERENCE_ERROR, detail=f"err={err!r}")
check("cascade: NUL stage_c audited as MalformedReference",
      "MalformedReference" in stderr)

# --- Negative: catalog hit whose abs_path moved post-spawn → PathMoved -------
# Build the catalog (in cfg), then delete the resolved stage fixture so the
# resolver's request-time os.path.exists fails closed — proves no stale-path
# load and the uniform frame (Vision slice-3b negative case 9).
mb, out, _inside, cfg = _setup_mb_and_out()
os.remove(os.path.join(mb, "diffusion_models", "test-dit.safetensors"))
result, err, stderr = _call_cascade(cfg, _good_cascade_args(stage_c="test-dit"))
check("cascade: stage abs_path moved post-spawn → uniform 'reference not available'",
      result is None and err == mcps._UNIFORM_REFERENCE_ERROR, detail=f"err={err!r}")
check("cascade: moved stage path audited as PathMoved (no stale-path load)",
      "PathMoved" in stderr)

# --- Negative: scaffolding_repo supplied → removed-field ValidationError -----
mb, out, _inside, cfg = _setup_mb_and_out()
bad_args = _good_cascade_args()
bad_args["cascade_config"]["scaffolding_repo"] = "anything"
result, err, stderr = _call_cascade(cfg, bad_args)
check("cascade: scaffolding_repo supplied → ValidationError naming the field",
      result is None and err is not None
      and "scaffolding_repo" in err and "not supported" in err, detail=f"err={err!r}")
check("cascade: scaffolding_repo rejection audited as ValidationError",
      "ValidationError" in stderr)

# --- Byte-equality: cascade uniform error == the non-cascade uniform error ---
check("cascade: uniform error message is the shared _UNIFORM_REFERENCE_ERROR",
      mcps._UNIFORM_REFERENCE_ERROR == "reference not available")

# --- allow_hf_download=False reaches build_pipelines (invariant 6) -----------
mb, out, _inside, cfg = _setup_mb_and_out()
_recorded_hf = []

def _spy_build_pipelines(cfg_cc, device, allow_hf_download):
    _recorded_hf.append(allow_hf_download)
    return (_FakePipe(), _FakePipe())

with unittest.mock.patch.object(sys, "stderr", io.StringIO()), \
     unittest.mock.patch.object(cas_mod, "build_pipelines", _spy_build_pipelines), \
     unittest.mock.patch.object(cas_mod, "run_one", _mock_cascade_run_one):
    try:
        _run(mcps._call_tool_impl(cfg, "generate", _good_cascade_args()))
    except ValueError:
        pass
check("cascade: build_pipelines invoked with allow_hf_download=False",
      _recorded_hf == [False], detail=f"recorded={_recorded_hf}")

# Cascade-side missing-prompt (unchanged behavior)
mb, out, _inside, cfg = _setup_mb_and_out()
bad_args = _good_cascade_args()
del bad_args["prompt"]
result, err, stderr = _call_cascade(cfg, bad_args)
check("cascade missing-prompt → MissingField (BEFORE resolution)",
      result is None and err is not None
      and "prompt: required field absent" in err)
check("cascade missing-prompt audit class is MissingField",
      "MissingField" in stderr)

# Cascade-side: cascade_config missing required fields (stage_c, stage_b)
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(cfg, {
    "prompt": "p",
    "cascade_config": {},  # missing stage_c, stage_b
})
check("cascade_config missing required fields → ValidationError",
      result is None and err is not None
      and "cascade_config" in err)

# cascade_config: not a dict
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_cascade(cfg, {
    "prompt": "p",
    "cascade_config": "not a dict",
})
check("cascade_config not a dict → ValidationError",
      result is None and err is not None
      and "cascade_config" in err)


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 3: redact_metadata_for_png cascade extension (unit) ==")
# ════════════════════════════════════════════════════════════════════════

cc_md = {
    "model": "stablecascade",
    "prompt": "x",
    "seed": 1,
    "cascade_config": {
        "stage_c": "/abs/path/to/stage-c",
        "stage_b": "/abs/path/to/stage-b",
        "stage_a": "/abs/path/to/stage-a",
        "scaffolding_repo": "/abs/path/to/scaffolding",
        "prior_steps": 20,
        "decoder_steps": 10,
        "prior_cfg_scale": 4.0,
        "width": 1024,
        "height": 1024,
    },
    "output_path": "/abs/output.png",
}
red_cc = mcps.redact_metadata_for_png(cc_md)
cc_red = red_cc.get("cascade_config", {})
check("cascade redaction: stage_c basenamed",
      cc_red.get("stage_c") == "stage-c")
check("cascade redaction: stage_b basenamed",
      cc_red.get("stage_b") == "stage-b")
check("cascade redaction: stage_a basenamed",
      cc_red.get("stage_a") == "stage-a")
check("cascade redaction: scaffolding_repo basenamed",
      cc_red.get("scaffolding_repo") == "scaffolding")
check("cascade redaction: non-path cascade_config fields retained verbatim",
      cc_red.get("prior_steps") == 20
      and cc_red.get("decoder_steps") == 10
      and cc_red.get("prior_cfg_scale") == 4.0
      and cc_red.get("width") == 1024)
check("cascade redaction: output_path still dropped at top level",
      "output_path" not in red_cc)
check("cascade redaction: model field still retained (top-level)",
      red_cc.get("model") == "stablecascade")

# HF repo IDs inside cascade_config pass through unchanged
cc_hf = {
    "model": "stablecascade",
    "cascade_config": {
        "stage_c": "stabilityai/stable-cascade-prior",
        "scaffolding_repo": "stabilityai/stable-cascade",
        "prior_steps": 20,
    },
}
red_hf = mcps.redact_metadata_for_png(cc_hf)
check("cascade redaction: HF repo IDs in cascade_config pass through",
      red_hf["cascade_config"]["stage_c"] == "stabilityai/stable-cascade-prior"
      and red_hf["cascade_config"]["scaffolding_repo"] == "stabilityai/stable-cascade")


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 3: _MCP_CASCADE_PATH_TYPED_FIELDS hygiene ==")
# ════════════════════════════════════════════════════════════════════════

check("_MCP_CASCADE_PATH_TYPED_FIELDS is a tuple",
      isinstance(mcps._MCP_CASCADE_PATH_TYPED_FIELDS, tuple))
check("_MCP_CASCADE_PATH_TYPED_FIELDS contains the 4 cascade path fields",
      set(mcps._MCP_CASCADE_PATH_TYPED_FIELDS) ==
      {"stage_c", "stage_b", "stage_a", "scaffolding_repo"})


# ════════════════════════════════════════════════════════════════════════
print("\n== Krea-2 slice: MCP family-default overlay (ADR-009 parity) ==")
# ════════════════════════════════════════════════════════════════════════
#
# The MCP generate handler must apply FAMILY_DEFAULTS to canonical keys the
# agent omits, matching the CLI's _run_one (ADR-009 caller-responsibility).
# Without this, an agent omitting steps/cfg for Krea-2-Turbo would get the
# schema fallbacks (28/3.5) instead of the family defaults (8/0.0).
#
# Proven here with the qwen-image fixture (FAMILY_DEFAULTS: steps=50,
# true_cfg_scale=4.0; QwenImagePipeline IS importable, so the overlay's
# detect_pipeline_class succeeds). The krea / krea-turbo overlay rides the
# IDENTICAL code path; on a diffusers without Krea2Pipeline it no-ops by
# design (detect_pipeline_class raises -> _apply_family_defaults catches),
# which is exactly when krea generation is unavailable anyway. So this test
# guards the wiring that makes krea defaults flow the moment the dep lands.

_fam_capture: dict = {}


def _capturing_generate(*, model_path, prompt, output_path, **kw):
    _fam_capture.clear()
    _fam_capture.update(kw)
    Image.new("RGB", (8, 8), "white").save(output_path)
    return {
        "prompt": prompt, "negative_prompt": kw.get("negative_prompt", ""),
        "model": model_path, "seed": kw.get("seed", 42),
        "steps": kw.get("steps", 28), "cfg_scale": kw.get("cfg_scale", 3.5),
        "transformer_path": "", "loras": [], "elapsed_seconds": 0.01,
    }


def _call_capturing(cfg, args):
    captured_err = io.StringIO()
    with unittest.mock.patch.object(sys, "stderr", captured_err), \
         unittest.mock.patch.object(gen_mod, "_load_pipeline", _mock_load_pipeline), \
         unittest.mock.patch.object(gen_mod, "generate", _capturing_generate):
        _run(mcps._call_tool_impl(cfg, "generate", args))


# Omitted keys get the family default.
mb, out, _inside, cfg = _setup_mb_and_out()
_call_capturing(cfg, {"prompt": "p", "model": "qwen-image"})
check("MCP overlay: omitted steps -> family default 50 (qwen-image)",
      _fam_capture.get("steps") == 50,
      detail=f"got {_fam_capture.get('steps')!r}")
check("MCP overlay: omitted true_cfg_scale -> family default 4.0",
      _fam_capture.get("true_cfg_scale") == 4.0,
      detail=f"got {_fam_capture.get('true_cfg_scale')!r}")

# Explicit agent value wins over the family default (negative case).
mb, out, _inside, cfg = _setup_mb_and_out()
_call_capturing(cfg, {"prompt": "p", "model": "qwen-image", "steps": 7})
check("MCP overlay: explicit steps=7 preserved (not overwritten by family)",
      _fam_capture.get("steps") == 7,
      detail=f"got {_fam_capture.get('steps')!r}")

# cfg_scale=0.0 must survive to generate() and not be coerced to the 3.5
# fallback — this is the Krea-2-Turbo footgun (CFG disabled at 0.0). The key
# is present in the payload, so gen_params.get('cfg_scale', 3.5) returns 0.0.
mb, out, _inside, cfg = _setup_mb_and_out()
_call_capturing(cfg, {"prompt": "p", "model": "qwen-image", "cfg_scale": 0.0})
check("MCP overlay: explicit cfg_scale=0.0 reaches generate() (not 3.5 fallback)",
      _fam_capture.get("cfg_scale") == 0.0,
      detail=f"got {_fam_capture.get('cfg_scale')!r}")


print("\n== Slice 2 Step 1: scan_model_family (catalog scan-time helper) ==")
# ════════════════════════════════════════════════════════════════════════
#
# Characterization tests for the new comfyless.catalog.scan_model_family
# helper. The helper is the scan-time companion to
# nodes.eric_diffusion_utils.detect_pipeline_class — it returns the
# model_family string for a diffusers-pipeline directory without requiring
# the pipeline class to be importable in the running diffusers version.
# Tests prove (a) it returns the SAME family strings as the existing
# infer_model_family mapping for every supported family (characterization),
# (b) it returns None on every non-usable input (permissive failure mode),
# (c) it works for class names diffusers doesn't ship — proving scan-time
# independence from the operator's diffusers installation.

def _write_model_index(parent_dir: str, _class_name) -> str:
    """Write a minimal model_index.json into a fresh child dir of parent_dir.

    Returns the child dir's absolute path. Use with tempfile.TemporaryDirectory
    as the parent. _class_name may be a string (placed under _class_name),
    None (the key is omitted entirely), or any other value (placed verbatim
    — used by the not-a-string negative case).
    """
    child = os.path.join(parent_dir, "model_dir")
    os.makedirs(child, exist_ok=True)
    idx: dict = {}
    if _class_name is not _CLASS_NAME_OMIT:
        idx["_class_name"] = _class_name
    with open(os.path.join(child, "model_index.json"), "w") as f:
        json.dump(idx, f)
    return child


_CLASS_NAME_OMIT = object()  # sentinel: omit _class_name entirely


# Characterization: scan_model_family agrees with the existing
# infer_model_family mapping for every supported pipeline class family
# (the _FAMILY_PATTERNS list in nodes/eric_diffusion_utils.py).
_CHARACTERIZATION_CASES = [
    ("QwenImagePipeline",            "qwen-image"),
    ("QwenImageEditPlusPipeline",    "qwen-edit"),
    ("FluxPipeline",                 "flux"),
    ("Flux2Pipeline",                "flux2"),
    ("Flux2KleinPipeline",           "flux2klein"),
    ("ChromaPipeline",               "chroma"),
    ("AuraFlowPipeline",             "auraflow"),
    ("StableDiffusion3Pipeline",     "sd3"),
    ("StableDiffusionXLPipeline",    "sdxl"),
    ("StableDiffusionPipeline",      "sd1"),
    ("ZImagePipeline",               "zimage"),
    ("Krea2Pipeline",                "krea"),  # non-distilled (no is_distilled)
]

for _cls_name, _expected in _CHARACTERIZATION_CASES:
    with tempfile.TemporaryDirectory() as _td:
        _model_dir = _write_model_index(_td, _cls_name)
        _got = cat_mod.scan_model_family(_model_dir)
        check(
            f"scan_model_family({_cls_name!r}) -> {_expected!r}",
            _got == _expected,
            detail=f"got {_got!r}",
        )
        # Cross-check vs the single-source-of-truth helper.
        check(
            f"scan_model_family({_cls_name!r}) matches infer_model_family",
            _got == infer_model_family(_cls_name),
        )

# Krea-2: one pipeline class (Krea2Pipeline) splits into two families via
# the model's own is_distilled flag — proves the scan classifies krea vs
# krea-turbo at scan time WITHOUT diffusers shipping Krea2Pipeline (it lives
# only on diffusers main; the scan never imports the class).
def _write_krea_index(parent_dir: str, is_distilled) -> str:
    child = os.path.join(parent_dir, "krea_model")
    os.makedirs(child, exist_ok=True)
    idx: dict = {"_class_name": "Krea2Pipeline"}
    if is_distilled is not None:
        idx["is_distilled"] = is_distilled
    with open(os.path.join(child, "model_index.json"), "w") as f:
        json.dump(idx, f)
    return child

with tempfile.TemporaryDirectory() as _td:
    _d = _write_krea_index(_td, True)
    check(
        "scan_model_family Krea2Pipeline + is_distilled=true -> 'krea-turbo'",
        cat_mod.scan_model_family(_d) == "krea-turbo",
        detail=f"got {cat_mod.scan_model_family(_d)!r}",
    )
with tempfile.TemporaryDirectory() as _td:
    _d = _write_krea_index(_td, False)
    check(
        "scan_model_family Krea2Pipeline + is_distilled=false -> 'krea'",
        cat_mod.scan_model_family(_d) == "krea",
        detail=f"got {cat_mod.scan_model_family(_d)!r}",
    )

# Negative: model_dir does not exist
check(
    "scan_model_family on nonexistent dir returns None",
    cat_mod.scan_model_family(
        "/tmp/nonexistent-comfyless-catalog-fixture-xyzzy"
    ) is None,
)

# Negative: dir exists but lacks model_index.json
with tempfile.TemporaryDirectory() as _td:
    check(
        "scan_model_family on dir without model_index.json returns None",
        cat_mod.scan_model_family(_td) is None,
    )

# Negative: model_index.json is a directory, not a regular file
with tempfile.TemporaryDirectory() as _td:
    os.makedirs(os.path.join(_td, "model_index.json"))  # dir, not file
    check(
        "scan_model_family when model_index.json is a directory returns None",
        cat_mod.scan_model_family(_td) is None,
    )

# Negative: malformed JSON
with tempfile.TemporaryDirectory() as _td:
    with open(os.path.join(_td, "model_index.json"), "w") as f:
        f.write("{not-valid-json")
    check(
        "scan_model_family on malformed JSON returns None",
        cat_mod.scan_model_family(_td) is None,
    )

# Negative: top-level is a list, not an object
with tempfile.TemporaryDirectory() as _td:
    with open(os.path.join(_td, "model_index.json"), "w") as f:
        json.dump(["not", "an", "object"], f)
    check(
        "scan_model_family on non-object model_index.json returns None",
        cat_mod.scan_model_family(_td) is None,
    )

# Negative: _class_name key omitted entirely
with tempfile.TemporaryDirectory() as _td:
    _model_dir = _write_model_index(_td, _CLASS_NAME_OMIT)
    check(
        "scan_model_family with no _class_name field returns None",
        cat_mod.scan_model_family(_model_dir) is None,
    )

# Negative: _class_name is empty string
with tempfile.TemporaryDirectory() as _td:
    _model_dir = _write_model_index(_td, "")
    check(
        "scan_model_family with empty _class_name returns None",
        cat_mod.scan_model_family(_model_dir) is None,
    )

# Negative: _class_name is null (not a string)
with tempfile.TemporaryDirectory() as _td:
    _model_dir = _write_model_index(_td, None)
    check(
        "scan_model_family with null _class_name returns None",
        cat_mod.scan_model_family(_model_dir) is None,
    )

# Characterization: scan-time independence from diffusers installation.
# A class name diffusers doesn't ship MUST still classify (best-effort
# infer_model_family fallback). detect_pipeline_class would RAISE here;
# scan_model_family must NOT.
with tempfile.TemporaryDirectory() as _td:
    _model_dir = _write_model_index(_td, "FuturePipelineXYZ")
    _got = cat_mod.scan_model_family(_model_dir)
    check(
        "scan_model_family on unknown-to-diffusers class returns "
        "infer_model_family fallback (proves scan-time independence)",
        _got is not None and _got == infer_model_family("FuturePipelineXYZ"),
        detail=f"got {_got!r}",
    )

# Security-auditor MEDIUM-1 (folded 2026-05-23): bloated model_index.json
# exceeding _MAX_INDEX_BYTES (1 MiB) is rejected as None before json.loads
# is called — caps spawn-time DoS when Step 2 fans the helper over many dirs.
with tempfile.TemporaryDirectory() as _td:
    _model_dir = os.path.join(_td, "model_dir")
    os.makedirs(_model_dir)
    _bloat_path = os.path.join(_model_dir, "model_index.json")
    # Write 1 MiB + 1 byte of well-formed-prefix JSON padding so json.loads
    # itself would succeed on a smaller version; the rejection must be on
    # size alone, not parse failure.
    _padding = " " * (cat_mod._MAX_INDEX_BYTES)  # exceeds cap by JSON wrapper
    with open(_bloat_path, "w", encoding="utf-8") as f:
        f.write('{"_class_name":"QwenImagePipeline","_pad":"' + _padding + '"}')
    check(
        "scan_model_family rejects model_index.json larger than "
        "_MAX_INDEX_BYTES with None (MEDIUM-1: spawn-time DoS cap)",
        cat_mod.scan_model_family(_model_dir) is None,
    )

# Sanity: a model_index.json just UNDER the cap still parses correctly
# (proves the cap is the actual limit, not a too-tight reading).
with tempfile.TemporaryDirectory() as _td:
    _model_dir = os.path.join(_td, "model_dir")
    os.makedirs(_model_dir)
    # Build content that's well under the cap (50 KB of pad + valid JSON).
    _small_pad = " " * 50_000
    _under_path = os.path.join(_model_dir, "model_index.json")
    with open(_under_path, "w", encoding="utf-8") as f:
        f.write('{"_class_name":"QwenImagePipeline","_pad":"' + _small_pad + '"}')
    check(
        "scan_model_family accepts model_index.json well under "
        "_MAX_INDEX_BYTES (positive complement to MEDIUM-1 test)",
        cat_mod.scan_model_family(_model_dir) == "qwen-image",
    )

# Security-auditor MEDIUM-2 (folded 2026-05-23): explicit UTF-8 decoding,
# independent of host locale. A model_index.json containing non-ASCII UTF-8
# bytes parses correctly (and would not regress under `LANG=C` etc., which
# this test cannot simulate but the explicit encoding=utf-8 argument
# guarantees by construction).
with tempfile.TemporaryDirectory() as _td:
    _model_dir = os.path.join(_td, "model_dir")
    os.makedirs(_model_dir)
    # _class_name with non-ASCII char in a side field; the class_name
    # itself stays ASCII so infer_model_family classifies as expected.
    # The non-ASCII bytes exercise the UTF-8 decode path.
    _utf8_idx = {"_class_name": "QwenImagePipeline", "_note": "café-é-emoji-🚀"}
    with open(os.path.join(_model_dir, "model_index.json"),
              "w", encoding="utf-8") as f:
        json.dump(_utf8_idx, f, ensure_ascii=False)
    check(
        "scan_model_family decodes UTF-8 non-ASCII content correctly "
        "(MEDIUM-2: locale-independent encoding)",
        cat_mod.scan_model_family(_model_dir) == "qwen-image",
    )

# Module-import hygiene (HIGH-1 folded 2026-05-23): importing
# `comfyless.catalog` must NOT trigger `import torch`. The torch import
# is gated behind the lazy `from nodes.eric_diffusion_utils import
# infer_model_family` inside scan_model_family's body.
#
# Subprocess check — by the time this test file runs, torch is already
# imported (test imports cat_mod which in this fold no longer triggers it,
# but earlier imports in the file pull torch in via other paths). A
# clean-state assertion requires a fresh interpreter.
import subprocess as _subprocess  # noqa: E402
_proc = _subprocess.run(
    [sys.executable, "-c",
     "import sys; import comfyless.catalog; "
     "print('torch_imported=' + str('torch' in sys.modules))"],
    capture_output=True, text=True, cwd=str(Path(__file__).parent),
)
check(
    "importing comfyless.catalog does NOT trigger torch import "
    "(HIGH-1: lazy-import contract)",
    _proc.returncode == 0 and "torch_imported=False" in _proc.stdout,
    detail=f"stdout={_proc.stdout!r} stderr={_proc.stderr!r}",
)


# ════════════════════════════════════════════════════════════════════════
print("\n== Slice 2 Step 2: build_catalog + scan + manifest ==")
# ════════════════════════════════════════════════════════════════════════
#
# Step 2 lands the catalog data structure, name normalization, scan
# walker, manifest parser, and build_catalog() orchestrator. Tests
# exercise build_catalog() directly against fixture --model-base trees
# and fixture manifests; the MCP-server spawn-time wiring lands in
# Step 3 and is tested through the CLI there.
#
# Coverage maps to Vision negative cases N1-N15 + N24-N26 + N28.
# (N16-N23 + N27 wait for Step 4's list_* tools.)

import os as _os  # already imported as os; alias avoids any shadow inside loops


def _make_loras_fixture(model_base: str, names: list) -> None:
    """Create `<model_base>/loras/<name>.safetensors` regular files."""
    lora_dir = os.path.join(model_base, "loras")
    os.makedirs(lora_dir, exist_ok=True)
    for n in names:
        with open(os.path.join(lora_dir, f"{n}.safetensors"), "wb") as f:
            f.write(b"fake-lora-bytes")


def _make_transformer_fixture(model_base: str, subdir: str,
                              names: list) -> None:
    """Create `<model_base>/<subdir>/<name>.safetensors` regular files."""
    tdir = os.path.join(model_base, subdir)
    os.makedirs(tdir, exist_ok=True)
    for n in names:
        with open(os.path.join(tdir, f"{n}.safetensors"), "wb") as f:
            f.write(b"fake-transformer-bytes")


def _make_model_fixture(model_base: str, name: str,
                        class_name: str = "QwenImagePipeline") -> str:
    """Create `<model_base>/<name>/model_index.json` (regular file)."""
    mdir = os.path.join(model_base, name)
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "model_index.json"), "w",
              encoding="utf-8") as f:
        json.dump({"_class_name": class_name}, f)
    return mdir


def _write_manifest(model_base: str, entries: dict) -> str:
    """Write a JSON manifest under `<model_base>/.catalog.json` and return
    the path."""
    path = os.path.join(model_base, ".catalog.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    return path


def _assert_raises(label: str, fn, exc_type, *, message_contains=None):
    """Call fn() expecting `exc_type` (CatalogBuildError typically).

    Optional `message_contains` asserts the exception message contains
    the given substring(s) — pass a string or list. Records one check()
    per assertion (raised-correct-type + message-contains-each).
    """
    raised = None
    try:
        fn()
    except exc_type as e:
        raised = e
    except BaseException as e:  # noqa: BLE001
        check(f"{label} raises {exc_type.__name__}",
              False,
              detail=f"got {type(e).__name__}: {e}")
        return
    check(f"{label} raises {exc_type.__name__}",
          raised is not None,
          detail="no exception raised")
    if raised is None or message_contains is None:
        return
    msg = str(raised)
    needles = [message_contains] if isinstance(message_contains, str) \
        else message_contains
    for needle in needles:
        check(f"{label} message contains {needle!r}",
              needle in msg,
              detail=f"got {msg!r}")


# ── normalize_name + _add_entry basics ─────────────────────────────────

check("normalize_name produces NFC form",
      cat_mod.normalize_name("café") == "café")
check("normalize_name is idempotent on ASCII",
      cat_mod.normalize_name("foo") == "foo")

# _add_entry collision: same name + same abs_path = harmless alias
_cat: cat_mod.CatalogDict = {}
_e1 = {"abs_path": "/x/y", "kind": "model", "source": "scan",
        "model_family": "qwen-image", "target_family": None}
_e2 = {"abs_path": "/x/y", "kind": "model", "source": "manifest",
        "model_family": "qwen-image", "target_family": None}
cat_mod._add_entry(_cat, _e1, "foo")
cat_mod._add_entry(_cat, _e2, "foo")  # same abs_path -> harmless
check("_add_entry: same name + same abs_path is harmless alias",
      "foo" in _cat and _cat["foo"]["source"] == "scan")

# _add_entry collision: same name + different abs_path = fail closed
_cat = {}
_e3 = {"abs_path": "/x/y", "kind": "model", "source": "scan",
        "model_family": None, "target_family": None}
_e4 = {"abs_path": "/x/z", "kind": "model", "source": "manifest",
        "model_family": None, "target_family": None}
cat_mod._add_entry(_cat, _e3, "foo")
_assert_raises(
    "_add_entry: same name + different abs_path",
    lambda: cat_mod._add_entry(_cat, _e4, "foo"),
    cat_mod.CatalogBuildError,
    message_contains=["'foo'", "two distinct paths"],
)

# _add_entry collision: case-insensitive collision rejected
_cat = {}
cat_mod._add_entry(_cat, _e3, "Foo")
_assert_raises(
    "_add_entry: case-insensitive collision rejected (Foo vs foo)",
    lambda: cat_mod._add_entry(_cat, _e4, "foo"),
    cat_mod.CatalogBuildError,
    message_contains=["case-insensitively"],
)


# ── N1, N2, N3: --catalog file-level failures ──────────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    # N1: --catalog points at nonexistent path
    _assert_raises(
        "N1: --catalog nonexistent path fails",
        lambda: cat_mod.build_catalog(_mb, "/tmp/nonexistent-cat-xyz.json"),
        cat_mod.CatalogBuildError,
        message_contains="regular file",
    )
    # N2: --catalog points at a directory
    _dir_path = os.path.join(_mb, "a-directory-not-a-file")
    os.makedirs(_dir_path)
    _assert_raises(
        "N2: --catalog pointing at a directory fails",
        lambda: cat_mod.build_catalog(_mb, _dir_path),
        cat_mod.CatalogBuildError,
        message_contains="regular file",
    )
    # N3: --catalog malformed JSON
    _bad_path = os.path.join(_mb, "bad.json")
    with open(_bad_path, "w") as f:
        f.write("{not-valid-json")
    _assert_raises(
        "N3: --catalog malformed JSON fails",
        lambda: cat_mod.build_catalog(_mb, _bad_path),
        cat_mod.CatalogBuildError,
        message_contains="valid UTF-8 JSON",
    )
    # N3-bonus: --catalog top-level is a list, not object
    _list_path = os.path.join(_mb, "list.json")
    with open(_list_path, "w", encoding="utf-8") as f:
        json.dump(["not", "object"], f)
    _assert_raises(
        "N3-bonus: --catalog top-level non-object fails",
        lambda: cat_mod.build_catalog(_mb, _list_path),
        cat_mod.CatalogBuildError,
        message_contains="top-level must be a JSON object",
    )

# Manifest size cap (MEDIUM-1-equivalent for the manifest file)
with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _big_path = os.path.join(_mb, "big.json")
    _padding = "x" * (cat_mod._MAX_MANIFEST_BYTES)
    with open(_big_path, "w", encoding="utf-8") as f:
        f.write('{"_pad":"' + _padding + '"}')
    _assert_raises(
        "manifest exceeding _MAX_MANIFEST_BYTES rejected at startup",
        lambda: cat_mod.build_catalog(_mb, _big_path),
        cat_mod.CatalogBuildError,
        message_contains="exceeds",
    )


# ── N4: manifest entry shape validation ────────────────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)

    # N4a: entry value not an object
    _path = _write_manifest(_mb, {"bad": "not-an-object"})
    _assert_raises(
        "N4a: manifest entry value not an object",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["'bad'", "JSON object"],
    )

    # N4b: entry missing 'target'
    _path = _write_manifest(_mb, {"bad": {"kind": "model"}})
    _assert_raises(
        "N4b: manifest entry missing 'target'",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["'bad'", "target"],
    )

    # N4c: entry missing 'kind'
    _path = _write_manifest(_mb, {"bad": {"target": "/abs/path"}})
    _assert_raises(
        "N4c: manifest entry missing 'kind'",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["'bad'", "kind"],
    )

    # N4d: entry 'kind' not in _KINDS
    _path = _write_manifest(
        _mb, {"bad": {"target": "/abs", "kind": "vae"}})
    _assert_raises(
        "N4d: manifest entry kind not in {model,lora,transformer}",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["invalid 'kind'", "'vae'"],
    )

    # N4e: entry has unknown extra keys
    _path = _write_manifest(_mb, {
        "bad": {"target": "/abs", "kind": "model", "evil": "field"}
    })
    _assert_raises(
        "N4e: manifest entry with unknown extra keys",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["unknown keys", "evil"],
    )

    # N4f: target_family on kind:"model" → rejected
    _path = _write_manifest(_mb, {
        "bad": {"target": "/abs", "kind": "model",
                "target_family": "qwen-image"}
    })
    _assert_raises(
        "N4f: target_family only allowed on kind:'lora'",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["target_family", "kind:'lora'"],
    )

    # N4g: empty-string entry name
    _path = _write_manifest(_mb, {
        "": {"target": "/abs/path", "kind": "model"}
    })
    _assert_raises(
        "N4g: empty-string entry name",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains="empty-string entry name",
    )

    # N4h: NUL byte in manifest target raises CatalogBuildError (NOT
    # bare ValueError from os.path.realpath). security-auditor MEDIUM-1
    # folded 2026-05-24: the pre-check matches the project pattern in
    # server.py and mcp_server.py and preserves the CatalogBuildError-
    # only contract that Step 3 will wrap into click.BadParameter.
    _path = _write_manifest(_mb, {
        "nul": {"target": "/abs/path\x00/etc/passwd", "kind": "model"}
    })
    _assert_raises(
        "N4h: NUL byte in manifest target -> CatalogBuildError (not ValueError)",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["'nul'", "null byte"],
    )


# ── N5, N6: manifest target escapes --model-base ────────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    # N5: target is an absolute path outside --model-base
    _path = _write_manifest(_mb, {
        "escape": {"target": "/etc/passwd-fake.safetensors",
                   "kind": "lora"}
    })
    _assert_raises(
        "N5: manifest target outside --model-base after realpath",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["'escape'", "outside"],
    )

with tempfile.TemporaryDirectory() as _outside_td:
    # N6: manifest target is a symlink resolving outside --model-base
    _outside_real = os.path.realpath(_outside_td)
    _outside_target = os.path.join(_outside_real, "real.safetensors")
    with open(_outside_target, "wb") as f:
        f.write(b"outside-content")
    with tempfile.TemporaryDirectory() as _td:
        _mb = os.path.realpath(_td)
        _link = os.path.join(_mb, "evil.safetensors")
        os.symlink(_outside_target, _link)
        _path = _write_manifest(_mb, {
            "evil": {"target": _link, "kind": "lora"}
        })
        _assert_raises(
            "N6: manifest symlink target resolves outside --model-base",
            lambda: cat_mod.build_catalog(_mb, _path),
            cat_mod.CatalogBuildError,
            message_contains=["'evil'", "outside"],
        )


# ── N7: manifest HF repo ID not in local cache ─────────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _path = _write_manifest(_mb, {
        "missing": {"target": "FakeOrg/NonExistentRepo-xyzzy-12345",
                    "kind": "model"}
    })
    # The repo ID doesn't exist in the cache; resolve_hf_path raises
    # ValueError → CatalogBuildError naming both the entry and the repo.
    _assert_raises(
        "N7: manifest HF repo ID not in local cache fails",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["'missing'", "FakeOrg/NonExistentRepo-xyzzy-12345",
                          "local HF cache"],
    )


# ── N8: scan-internal collision (two paths same normalized name) ───────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    # Two loras/ subdirs with the same filename stem → scan collision
    os.makedirs(os.path.join(_mb, "a", "loras"))
    os.makedirs(os.path.join(_mb, "b", "loras"))
    with open(os.path.join(_mb, "a", "loras", "dupname.safetensors"),
              "wb") as f:
        f.write(b"a")
    with open(os.path.join(_mb, "b", "loras", "dupname.safetensors"),
              "wb") as f:
        f.write(b"b")
    _assert_raises(
        "N8: scan-internal name collision (two paths -> same name)",
        lambda: cat_mod.build_catalog(_mb, None),
        cat_mod.CatalogBuildError,
        message_contains=["'dupname'", "two distinct paths"],
    )


# ── N9: manifest shadows scan at different realpath ────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _make_loras_fixture(_mb, ["scanlora"])
    # Manifest declares "scanlora" pointing at a different file
    _other_path = os.path.join(_mb, "other.safetensors")
    with open(_other_path, "wb") as f:
        f.write(b"other")
    _path = _write_manifest(_mb, {
        "scanlora": {"target": _other_path, "kind": "lora"}
    })
    _assert_raises(
        "N9: manifest shadows scan name at different realpath",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["'scanlora'", "two distinct paths"],
    )


# ── N10: case-insensitive name collision ───────────────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _make_loras_fixture(_mb, ["FooBar"])
    _other = os.path.join(_mb, "other.safetensors")
    with open(_other, "wb") as f:
        f.write(b"other")
    _path = _write_manifest(_mb, {
        "foobar": {"target": _other, "kind": "lora"}
    })
    _assert_raises(
        "N10: case-insensitive name collision (FooBar vs foobar)",
        lambda: cat_mod.build_catalog(_mb, _path),
        cat_mod.CatalogBuildError,
        message_contains=["case-insensitively"],
    )


# ── N11, N28: symlinks do NOT mint independent catalog entries ─────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _make_loras_fixture(_mb, ["real_lora"])
    # Add a symlink alongside the real file → should be skipped
    _real_path = os.path.join(_mb, "loras", "real_lora.safetensors")
    _link_path = os.path.join(_mb, "loras", "link_lora.safetensors")
    os.symlink(_real_path, _link_path)
    _catalog = cat_mod.build_catalog(_mb, None)
    check(
        "N11: symlink .safetensors under loras/ does NOT mint an entry",
        "link_lora" not in _catalog,
    )
    check(
        "N11 cross-check: real_lora IS in catalog (one entry, not two)",
        "real_lora" in _catalog and len(_catalog) == 1,
    )

with tempfile.TemporaryDirectory() as _outside_td:
    # N28: symlink at checkpoints/link.safetensors → target also outside
    # conventional dirs → both symlink and target are skipped (zero
    # catalog entries from this fixture).
    _outside_real = os.path.realpath(_outside_td)
    _outside_target = os.path.join(_outside_real, "elsewhere.safetensors")
    with open(_outside_target, "wb") as f:
        f.write(b"outside")
    with tempfile.TemporaryDirectory() as _td:
        _mb = os.path.realpath(_td)
        os.makedirs(os.path.join(_mb, "checkpoints"))
        _link = os.path.join(_mb, "checkpoints", "link.safetensors")
        os.symlink(_outside_target, _link)
        _catalog = cat_mod.build_catalog(_mb, None)
        check(
            "N28: symlink checkpoints/link.safetensors (target outside) "
            "mints zero entries",
            _catalog == {},
        )


# ── N12: manifest alias to same realpath is harmless ───────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _make_loras_fixture(_mb, ["mylora"])
    _real = os.path.realpath(os.path.join(_mb, "loras",
                                           "mylora.safetensors"))
    _path = _write_manifest(_mb, {
        "mylora": {"target": _real, "kind": "lora"}
    })
    # Should NOT raise; existing scan entry retained.
    _catalog = cat_mod.build_catalog(_mb, _path)
    check(
        "N12: manifest alias to same realpath is harmless (no error)",
        "mylora" in _catalog and _catalog["mylora"]["abs_path"] == _real,
    )
    check(
        "N12: scan-derived source retained after harmless alias",
        _catalog["mylora"]["source"] == "scan",
    )


# ── N13, N14, N15: spawn-succeeds-cleanly cases ────────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    # N13: empty --model-base + no manifest → empty catalog
    _catalog = cat_mod.build_catalog(_mb, None)
    check("N13: empty model-base + no manifest -> empty catalog",
          _catalog == {})

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _make_loras_fixture(_mb, ["lora1", "lora2"])
    # N14: spawn without --catalog (None) → catalog is just the scan
    _catalog = cat_mod.build_catalog(_mb, None)
    check("N14: no --catalog flag -> catalog is just the scan",
          set(_catalog.keys()) == {"lora1", "lora2"})
    check("N14: all scan entries marked source='scan'",
          all(e["source"] == "scan" for e in _catalog.values()))

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _make_loras_fixture(_mb, ["lora1"])
    _path = _write_manifest(_mb, {})  # empty manifest object
    # N15: manifest with no entries → catalog is just the scan
    _catalog = cat_mod.build_catalog(_mb, _path)
    check("N15: empty manifest -> catalog is just the scan",
          set(_catalog.keys()) == {"lora1"})


# ── N24: transformer-kind scan classification ──────────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _make_transformer_fixture(_mb, "checkpoints", ["foo"])
    _make_transformer_fixture(_mb, "diffusion_models", ["bar"])
    _catalog = cat_mod.build_catalog(_mb, None)
    check("N24: .safetensors in checkpoints/ -> kind:'transformer'",
          _catalog.get("foo", {}).get("kind") == "transformer")
    check("N24: .safetensors in diffusion_models/ -> kind:'transformer'",
          _catalog.get("bar", {}).get("kind") == "transformer")
    check("N24: transformer entries marked source='scan'",
          _catalog["foo"]["source"] == "scan"
          and _catalog["bar"]["source"] == "scan")
    check("N24: transformer entries have no model_family (scan-derived)",
          _catalog["foo"]["model_family"] is None
          and _catalog["bar"]["model_family"] is None)


# ── N25: .safetensors outside conventional dirs is SKIPPED ─────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    # File directly under model-base root (not in any conventional dir)
    with open(os.path.join(_mb, "orphan_root.safetensors"), "wb") as f:
        f.write(b"orphan")
    # File in an unconventional subdir name (not loras/checkpoints/diffusion_models)
    os.makedirs(os.path.join(_mb, "random_dir"))
    with open(os.path.join(_mb, "random_dir",
                            "orphan_sub.safetensors"), "wb") as f:
        f.write(b"orphan2")
    _catalog = cat_mod.build_catalog(_mb, None)
    check("N25: .safetensors at model-base root is SKIPPED",
          "orphan_root" not in _catalog)
    check("N25: .safetensors in unconventional subdir is SKIPPED",
          "orphan_sub" not in _catalog)
    check("N25: scan with only unconventional files -> empty catalog",
          _catalog == {})


# ── N26: manifest declares transformer outside conventional dirs ───────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    # Drop a transformer .safetensors at a random location
    os.makedirs(os.path.join(_mb, "random"))
    _target = os.path.join(_mb, "random", "my_transformer.safetensors")
    with open(_target, "wb") as f:
        f.write(b"transformer")
    _path = _write_manifest(_mb, {
        "my_transformer": {"target": _target, "kind": "transformer",
                           "model_family": "qwen-image"}
    })
    _catalog = cat_mod.build_catalog(_mb, _path)
    check("N26: manifest mints kind:'transformer' for unconventional path",
          _catalog.get("my_transformer", {}).get("kind") == "transformer")
    check("N26: manifest entry's source is 'manifest'",
          _catalog["my_transformer"]["source"] == "manifest")
    check("N26: manifest-declared model_family is preserved",
          _catalog["my_transformer"]["model_family"] == "qwen-image")


# ── model + manifest integration (positive smoke) ──────────────────────

with tempfile.TemporaryDirectory() as _td:
    _mb = os.path.realpath(_td)
    _make_model_fixture(_mb, "qwen-image", "QwenImagePipeline")
    _make_loras_fixture(_mb, ["anime_lora"])
    _make_transformer_fixture(_mb, "checkpoints", ["dit_v1"])
    # Manifest adds a friendly target_family on the scanned lora,
    # using same-realpath alias.
    _real_lora = os.path.realpath(
        os.path.join(_mb, "loras", "anime_lora.safetensors"))
    _path = _write_manifest(_mb, {
        "anime_lora": {"target": _real_lora, "kind": "lora",
                       "target_family": "qwen-image"},
    })
    _catalog = cat_mod.build_catalog(_mb, _path)
    check("integration: catalog has model+lora+transformer entries",
          set(_catalog.keys()) == {"qwen-image", "anime_lora", "dit_v1"})
    check("integration: model entry has model_family from scan",
          _catalog["qwen-image"]["model_family"] == "qwen-image")
    check("integration: model entry source='scan'",
          _catalog["qwen-image"]["source"] == "scan")
    check("integration: lora kind correct",
          _catalog["anime_lora"]["kind"] == "lora")
    check("integration: transformer kind correct",
          _catalog["dit_v1"]["kind"] == "transformer")


# ── Module-import contract carries forward from Step 1 ─────────────────

# Verify build_catalog can be CALLED without ever importing torch (the
# only torch-pulling import is the lazy resolve_hf_path inside the
# manifest HF-source branch, which only fires when a manifest entry
# names an HF repo ID). build_catalog with scan-only or local-path-
# only manifest entries must stay torch-free in a fresh interpreter.

_proc = _subprocess.run(
    [sys.executable, "-c",
     "import sys; import os; import tempfile; "
     "import comfyless.catalog as c; "
     "td = tempfile.mkdtemp(); "
     "os.makedirs(os.path.join(td, 'loras')); "
     "open(os.path.join(td, 'loras', 'x.safetensors'), 'wb').close(); "
     "c.build_catalog(td, None); "
     "print('torch_imported=' + str('torch' in sys.modules))"],
    capture_output=True, text=True, cwd=str(Path(__file__).parent),
)
check(
    "build_catalog(scan-only) does NOT trigger torch import "
    "(Step 1 HIGH-1 contract carries forward)",
    _proc.returncode == 0 and "torch_imported=False" in _proc.stdout,
    detail=f"stdout={_proc.stdout!r} stderr={_proc.stderr!r}",
)


# ════════════════════════════════════════════════════════════════════════
print("\n== Slice 2 Step 3: --catalog flag + spawn-time wire-up ==")
# ════════════════════════════════════════════════════════════════════════
#
# Invariants exercised here (see docs/vision/slice-2-mcp-catalog.md):
#   - I1  catalog built once at server spawn, held on _StartupConfig
#   - I7  startup fails closed on missing / malformed / schema-invalid /
#         collision / symlink-escape (--catalog channel)
#   - I13 --catalog declared as click option on main()
#   - I14 _list_tools_impl tool count is unchanged by catalog content
#         (post-Step 4: returns the static 3 tools regardless of catalog)
#   - N1–N7 the catalog-layer fail-closed cases (most are unit-tested in
#         test_mcp_server's catalog section already; CLI-surface parity
#         lives here so wiring regressions are caught)

# --- Module-level checks (cheap; no fixtures) ---

check("Step3: _StartupConfig.__slots__ includes 'catalog' (I1)",
      "catalog" in mcps._StartupConfig.__slots__)

# Inspect main()'s click option metadata via the click.Command params list.
_main_params = {p.name: p for p in mcps.main.params}
check("Step3: --catalog click option declared on main() (I13)",
      "catalog" in _main_params)
if "catalog" in _main_params:
    _catalog_opt = _main_params["catalog"]
    check("Step3: --catalog has 'catalog' option name (not positional)",
          isinstance(_catalog_opt, click.Option))
    check("Step3: --catalog is not required (operator-optional)",
          _catalog_opt.required is False)
    check("Step3: --catalog default is None",
          _catalog_opt.default is None)
    check("Step3: --catalog type is click.Path",
          isinstance(_catalog_opt.type, click.Path))
    if isinstance(_catalog_opt.type, click.Path):
        check("Step3: --catalog click.Path(file_okay=True, dir_okay=False)",
              _catalog_opt.type.file_okay is True
              and _catalog_opt.type.dir_okay is False)


# --- Failure-via-CliRunner cases (I7 / N1–N7 — operator stderr surface) ---

# F1: --catalog nonexistent file → startup fails closed
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    nonexistent_catalog = os.path.join(tmp_base, "no-such-manifest.json")
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", tmp_base,
        "--catalog", nonexistent_catalog,
    ])
    check("Step3 F1: nonexistent --catalog file → non-zero exit (I7)",
          result.exit_code != 0,
          detail=f"exit={result.exit_code} stderr={(result.stderr or '')!r}")

# F2: --catalog malformed JSON → startup fails closed
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    bad_manifest = os.path.join(tmp_base, "bad.json")
    with open(bad_manifest, "w", encoding="utf-8") as f:
        f.write("{this is not valid json")
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", tmp_base,
        "--catalog", bad_manifest,
    ])
    check("Step3 F2: malformed --catalog JSON → non-zero exit (I7)",
          result.exit_code != 0,
          detail=f"exit={result.exit_code}")

# F3: --catalog points at a directory (not a file) → click rejects
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    a_dir = os.path.join(tmp_base, "i-am-a-directory")
    os.makedirs(a_dir)
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", tmp_base,
        "--catalog", a_dir,
    ])
    check("Step3 F3: --catalog points at a directory → non-zero exit",
          result.exit_code != 0,
          detail=f"exit={result.exit_code}")

# F4: manifest entry's target realpath-escapes --model-base → fail closed
# (I7 / N6). The escape mechanism here is a non-symlink absolute path
# pointing outside --model-base; the symlink variant of the same failure
# is covered at the catalog-unit level (Step-2 tests).
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base, \
     tempfile.TemporaryDirectory() as tmp_outside:
    # Real target lives OUTSIDE model_base; manifest declares a target
    # path that realpath() will resolve to the outside location.
    outside_file = os.path.join(tmp_outside, "evil.safetensors")
    open(outside_file, "wb").close()
    manifest_path = os.path.join(tmp_base, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "escape-attempt": {
                "target": outside_file,
                "kind": "lora",
            },
        }, f)
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", tmp_base,
        "--catalog", manifest_path,
    ])
    check("Step3 F4: manifest entry resolves outside --model-base → "
          "non-zero exit (I7)",
          result.exit_code != 0,
          detail=f"exit={result.exit_code}")
    check("Step3 F4: stderr names --catalog (operator-facing hint)",
          "--catalog" in (result.output or "") + (result.stderr or ""),
          detail=f"output={result.output!r} stderr={result.stderr!r}")

# F5: --catalog string contains a NUL byte → fail closed at click parse
# (click.Path.convert calls os.stat which rejects NULs)
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    result = runner.invoke(mcps.main, [
        "--output-dir", tmp_out,
        "--model-base", tmp_base,
        "--catalog", "manifest\x00.json",
    ])
    check("Step3 F5: --catalog with embedded NUL → non-zero exit",
          result.exit_code != 0,
          detail=f"exit={result.exit_code}")

# F5b: direct in-process call → explicit NUL pre-check yields a clean
# click.BadParameter (defense-in-depth for non-CLI callers).
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    raised = None
    try:
        mcps._validate_startup_args(
            output_dir=tmp_out, model_base=tmp_base,
            default_model=None, mcp_max_iterations=100,
            catalog="manifest\x00.json",
        )
    except click.BadParameter as e:
        raised = e
    except BaseException as e:
        raised = f"UNEXPECTED-{type(e).__name__}: {e}"
    check("Step3 F5b: _validate_startup_args raises click.BadParameter "
          "on NUL in catalog",
          isinstance(raised, click.BadParameter),
          detail=f"raised={raised!r}")


# --- Direct _validate_startup_args success cases (I1) ---

# S1: spawn without --catalog → cfg.catalog populated from scan only
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    # Plant one scan-recognized entry: loras/scan-lora.safetensors
    loras_dir = os.path.join(tmp_base, "loras")
    os.makedirs(loras_dir)
    scan_lora_path = os.path.join(loras_dir, "scan-lora.safetensors")
    open(scan_lora_path, "wb").close()

    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    check("Step3 S1: cfg.catalog is a dict (scan-only path)",
          isinstance(cfg.catalog, dict))
    check("Step3 S1: scan-only cfg.catalog contains scan-derived entry",
          "scan-lora" in cfg.catalog,
          detail=f"keys={list(cfg.catalog)!r}")
    if "scan-lora" in cfg.catalog:
        _entry = cfg.catalog["scan-lora"]
        check("Step3 S1: scan-derived entry has source='scan'",
              _entry.get("source") == "scan")
        check("Step3 S1: scan-derived entry has kind='lora'",
              _entry.get("kind") == "lora")

# S2: spawn with --catalog → cfg.catalog has BOTH scan and manifest entries
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    # Scan side: one lora in loras/
    loras_dir = os.path.join(tmp_base, "loras")
    os.makedirs(loras_dir)
    open(os.path.join(loras_dir, "scan-lora.safetensors"), "wb").close()

    # Manifest side: one lora pointing at an under-base path
    manifest_lora_path = os.path.join(tmp_base, "manifest-lora.safetensors")
    open(manifest_lora_path, "wb").close()
    manifest_path = os.path.join(tmp_base, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "from-manifest": {
                "target": manifest_lora_path,
                "kind": "lora",
            },
        }, f)

    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
        catalog=manifest_path,
    )
    check("Step3 S2: cfg.catalog contains scan entry under --catalog",
          "scan-lora" in cfg.catalog,
          detail=f"keys={list(cfg.catalog)!r}")
    check("Step3 S2: cfg.catalog contains manifest-declared entry",
          "from-manifest" in cfg.catalog,
          detail=f"keys={list(cfg.catalog)!r}")
    if "from-manifest" in cfg.catalog:
        check("Step3 S2: manifest entry has source='manifest'",
              cfg.catalog["from-manifest"].get("source") == "manifest")

    # --- Step-3 wiring carry: tool surface count is the same regardless
    # of catalog content (Step 3 wires the catalog onto cfg; Step 4 grew
    # the surface to 3 tools statically; slice 2b adds a 4th — catalog
    # content never changes the tool LIST, only the responses inside the
    # list_* handlers).
    tools = _run(mcps._list_tools_impl(cfg))
    check("Step3 carry: _list_tools_impl with populated catalog still "
          "returns the static four tools (slice-2b)",
          isinstance(tools, list) and len(tools) == 4,
          detail=f"len={len(tools)}")
    _names = sorted(t.name for t in tools)
    check("Step3 carry: tool names unchanged by catalog content",
          _names == ["generate", "list_loras", "list_models",
                     "list_transformers"],
          detail=f"names={_names}")


# ════════════════════════════════════════════════════════════════════════
print("\n== Slice 2 Step 4: list_models / list_loras + name sanitization ==")
# ════════════════════════════════════════════════════════════════════════
#
# Invariants exercised here (see docs/vision/slice-2-mcp-catalog.md):
#   - I8  _list_tools_impl advertises 3 tools (covered by the Inv-8 section
#         near the top of this file; not duplicated here)
#   - I9  list_*  response shape: strict-allowlist keys; no abs_path
#   - I10 audit line on every list_* call (count + status + elapsed)
#   - I11 traceback strip on list_* internal exceptions
#   - I14 generate's schema + description byte-identical to slice 1
#   - N16-N20, N22-N23, N27 + Step-2 INFO-2 fold (name-char sanitization)


# --- Step-4 fixture builder: multi-kind catalog used by several cases ---

def _build_step4_catalog(mb: str) -> tuple[str, dict]:
    """Plant a multi-kind catalog under `mb`. Returns (manifest_path, info)
    where `info` documents the expected catalog membership.

    Plants:
      scan side:
        - fixture-model/model_index.json         -> kind:"model"
        - loras/scan-lora.safetensors            -> kind:"lora"
        - checkpoints/scan-transformer.safetensors -> kind:"transformer"
      manifest side:
        - "Manifest-LoRA"  -> manifest-lora.safetensors with target_family
        - "Manifest-Model" -> manifest-model/ (no model_index.json)
    """
    # Scan: model
    model_dir = os.path.join(mb, "fixture-model")
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "model_index.json"), "w",
              encoding="utf-8") as f:
        json.dump({"_class_name": "QwenImagePipeline"}, f)

    # Scan: lora
    loras_dir = os.path.join(mb, "loras")
    os.makedirs(loras_dir)
    open(os.path.join(loras_dir, "scan-lora.safetensors"), "wb").close()

    # Scan: transformer (single-file safetensors under checkpoints/)
    checkpoints_dir = os.path.join(mb, "checkpoints")
    os.makedirs(checkpoints_dir)
    open(os.path.join(checkpoints_dir, "scan-transformer.safetensors"),
         "wb").close()

    # Manifest entries — both under mb but NOT in a scan-dispatched dir
    manifest_lora_path = os.path.join(mb, "manifest-lora.safetensors")
    open(manifest_lora_path, "wb").close()
    manifest_model_dir = os.path.join(mb, "manifest-model")
    os.makedirs(manifest_model_dir)  # no model_index.json -> scan skips

    manifest_path = os.path.join(mb, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "Manifest-LoRA": {
                "target": manifest_lora_path,
                "kind": "lora",
                "target_family": "qwen-image",
            },
            "Manifest-Model": {
                "target": manifest_model_dir,
                "kind": "model",
                "model_family": "flux2",
            },
        }, f)
    return manifest_path, {
        "expected_models": {"fixture-model", "Manifest-Model"},
        "expected_loras": {"scan-lora", "Manifest-LoRA"},
        "expected_transformers": {"scan-transformer"},
    }


# --- N16: list_models response shape ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    manifest, info = _build_step4_catalog(tmp_base)
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
        catalog=manifest,
    )
    result, count = _run(mcps._handle_list_models(cfg))
    check("N16: _handle_list_models returns (list[TextContent], int)",
          isinstance(result, list) and len(result) == 1
          and result[0].type == "text" and isinstance(count, int))
    entries = json.loads(result[0].text)
    check("N16: list_models returns a JSON array",
          isinstance(entries, list))
    check("N16: list_models entry count matches expected_models",
          {e["name"] for e in entries} == info["expected_models"],
          detail=f"got={sorted(e['name'] for e in entries)}")
    check("N16: count from handler matches len(entries)",
          count == len(entries), detail=f"count={count} len={len(entries)}")
    _allowed_keys = {"name", "kind", "source", "model_family"}
    _bad_entries = [e for e in entries if not set(e.keys()).issubset(_allowed_keys)]
    check("N16: every entry's keys subset of {name, kind, source, model_family}",
          not _bad_entries,
          detail=f"bad={_bad_entries!r}")
    check("N16: NO entry contains 'abs_path' or 'path' key",
          not any("abs_path" in e or "path" in e for e in entries))
    check("N16: every entry has kind='model'",
          all(e["kind"] == "model" for e in entries))
    # The Manifest-Model fixture declared model_family='flux2' explicitly
    _manifest_model = next((e for e in entries if e["name"] == "Manifest-Model"), None)
    if _manifest_model is not None:
        check("N16: manifest-declared model_family surfaces",
              _manifest_model.get("model_family") == "flux2",
              detail=f"got={_manifest_model!r}")


# --- N17: list_loras response shape + target_family manifest-only ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    manifest, info = _build_step4_catalog(tmp_base)
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
        catalog=manifest,
    )
    result, count = _run(mcps._handle_list_loras(cfg))
    entries = json.loads(result[0].text)
    check("N17: list_loras entry names match expected_loras",
          {e["name"] for e in entries} == info["expected_loras"],
          detail=f"got={sorted(e['name'] for e in entries)}")
    _allowed_keys = {"name", "kind", "source", "target_family"}
    _bad = [e for e in entries if not set(e.keys()).issubset(_allowed_keys)]
    check("N17: every entry's keys subset of {name, kind, source, target_family}",
          not _bad, detail=f"bad={_bad!r}")
    check("N17: NO entry contains 'abs_path' or 'path'",
          not any("abs_path" in e or "path" in e for e in entries))
    check("N17: every entry has kind='lora'",
          all(e["kind"] == "lora" for e in entries))
    _scan_lora = next((e for e in entries if e["name"] == "scan-lora"), None)
    _manifest_lora = next((e for e in entries if e["name"] == "Manifest-LoRA"), None)
    check("N17: scan-derived LoRA omits target_family entirely (no inference)",
          _scan_lora is not None and "target_family" not in _scan_lora,
          detail=f"got={_scan_lora!r}")
    check("N17: manifest-declared target_family surfaces",
          _manifest_lora is not None
          and _manifest_lora.get("target_family") == "qwen-image",
          detail=f"got={_manifest_lora!r}")


# --- N18: HF + local entries both surface only names ---
#
# We don't have a guaranteed-local HF cache hit in CI; emulate by
# constructing the catalog DIRECTLY (bypass build_catalog) with one
# entry whose abs_path is an HF-cache-shaped string and another with a
# local filesystem-style abs_path. Asserts that the response serializer
# never echoes either path under any key.

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    # Inject directly — bypasses normalize_name / _add_entry validation
    # for the sake of the no-abs_path-leak property.
    cfg.catalog["hf-repo-style"] = {
        "abs_path": "/var/cache/hf/Qwen/Qwen-Image-2512/snapshots/abcd1234",
        "kind": "model", "source": "manifest",
        "model_family": "qwen-image", "target_family": None,
    }
    cfg.catalog["local-style"] = {
        "abs_path": "/mnt/nvme-8tb/comfyui/checkpoints/local-model",
        "kind": "model", "source": "scan",
        "model_family": None, "target_family": None,
    }
    result, _ = _run(mcps._handle_list_models(cfg))
    body = result[0].text
    check("N18: HF-style abs_path does NOT appear in list_models text",
          "/var/cache/hf/Qwen" not in body and "snapshots/abcd1234" not in body,
          detail=f"body={body!r}")
    check("N18: local-style abs_path does NOT appear in list_models text",
          "/mnt/nvme-8tb" not in body
          and "comfyui/checkpoints/local-model" not in body,
          detail=f"body={body!r}")
    entries = json.loads(body)
    _by_name = {e["name"]: e for e in entries}
    check("N18: hf-repo-style entry surfaces its name only (no path)",
          "hf-repo-style" in _by_name
          and _by_name["hf-repo-style"].get("model_family") == "qwen-image"
          and "abs_path" not in _by_name["hf-repo-style"])
    check("N18: local-style entry surfaces its name only (no path)",
          "local-style" in _by_name
          and "abs_path" not in _by_name["local-style"])


# --- N19: list_models audit line carries count + tool + status + elapsed
# but does NOT carry any catalog abs_path or repo ID ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    manifest, _ = _build_step4_catalog(tmp_base)
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
        catalog=manifest,
    )
    captured_err = io.StringIO()
    with unittest.mock.patch.object(sys, "stderr", captured_err):
        _run(mcps._call_tool_impl(cfg, "list_models", {}))
    audit_text = captured_err.getvalue().strip()
    # Should be exactly one JSON line
    _audit_lines = [ln for ln in audit_text.splitlines() if ln.startswith("{")]
    check("N19: exactly one audit line emitted for list_models",
          len(_audit_lines) == 1,
          detail=f"lines={_audit_lines!r}")
    if len(_audit_lines) == 1:
        line = json.loads(_audit_lines[0])
        check("N19: audit line tool=list_models",
              line.get("tool") == "list_models")
        check("N19: audit line status=ok",
              line.get("status") == "ok")
        check("N19: audit line carries 'count' field",
              "count" in line and isinstance(line["count"], int))
        check("N19: audit line carries 'elapsed_seconds' field",
              "elapsed_seconds" in line)
        check("N19: audit line does NOT contain any catalog abs_path",
              tmp_base not in _audit_lines[0],
              detail=f"line={_audit_lines[0]!r}")
        # The manifest target is an HF-repo-shaped string in some Step-4
        # cases. For this fixture it's a local path; assert no path
        # fragment leaks.
        check("N19: audit line does NOT contain 'manifest-lora.safetensors'",
              "manifest-lora.safetensors" not in _audit_lines[0])


# --- N19b: list_* audit-payload bound (security-auditor LOW-1 fold) ---
# list_models / list_loras accept no inputs by schema. The framework
# decorator uses validate_input=False, so an agent CAN send arbitrary
# `arguments` to a list_* call. The handlers ignore them by signature
# (they take only `cfg`). To prevent the audit stream from being
# flooded with arbitrarily large agent payloads to a tool that ignores
# them, _call_tool_impl reduces the audit-line `input` to {} for these
# two tools — the call is still audited (one line, invariant 5), the
# blob is not echoed.

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    flood_blob = "A" * 10_000
    flood_args = {"unexpected_key": flood_blob, "nested": {"deep": flood_blob}}
    captured_err = io.StringIO()
    with unittest.mock.patch.object(sys, "stderr", captured_err):
        # Both list_* tools — handlers ignore args, audit emits {}
        _run(mcps._call_tool_impl(cfg, "list_models", flood_args))
        _run(mcps._call_tool_impl(cfg, "list_loras", flood_args))
    _stderr = captured_err.getvalue()
    check("N19b: list_models audit line does NOT echo the flood blob",
          flood_blob not in _stderr,
          detail=f"first 200 chars of stderr: {_stderr[:200]!r}")
    check("N19b: list_models audit line does NOT echo 'unexpected_key'",
          "unexpected_key" not in _stderr)
    _audit_lines = [json.loads(ln) for ln in _stderr.splitlines()
                    if ln.startswith("{")]
    check("N19b: both list_* audit lines emitted",
          len(_audit_lines) == 2,
          detail=f"got={len(_audit_lines)} lines")
    if len(_audit_lines) == 2:
        check("N19b: every list_* audit line has input={} (no payload echo)",
              all(line.get("input") == {} for line in _audit_lines),
              detail=f"inputs={[line.get('input') for line in _audit_lines]!r}")
        check("N19b: every list_* audit line still carries status=ok + count",
              all(line.get("status") == "ok" and "count" in line
                  for line in _audit_lines))

# Sanity carry: `generate`-tool audit payload still echoes agent args
# (that surface accepts inputs by schema; bounding it would lose
# operator-visible signal for legitimate calls). This guards against a
# regression where the audit_payload bound accidentally widens to
# generate.
with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    captured_err = io.StringIO()
    raised = None
    with unittest.mock.patch.object(sys, "stderr", captured_err):
        try:
            _run(mcps._call_tool_impl(cfg, "generate",
                                     {"prompt": "p", "model": "/etc/passwd"}))
        except ValueError as e:
            raised = str(e)
    _stderr = captured_err.getvalue()
    check("N19b carry: 'generate' audit line DOES echo agent arguments "
          "(only list_* are bounded)",
          "/etc/passwd" in _stderr,
          detail=f"stderr[:300]={_stderr[:300]!r}")


# --- N20: tools/list returns 3 tools; generate's schema + description
# byte-identical to slice 1 (compared against the unchanged constant) ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    tools = _run(mcps._list_tools_impl(cfg))
    by_name = {t.name: t for t in tools}
    check("N20: tools/list advertises exactly 4 tools (slice-2b)",
          set(by_name) == {"generate", "list_models", "list_loras",
                           "list_transformers"})
    check("N20: 'generate' inputSchema is byte-identical to _GENERATE_INPUT_SCHEMA "
          "(slice-1 invariant 14)",
          by_name["generate"].inputSchema == mcps._GENERATE_INPUT_SCHEMA)
    check("N20: 'generate' description is byte-identical to _GENERATE_TOOL_DESCRIPTION",
          by_name["generate"].description == mcps._GENERATE_TOOL_DESCRIPTION)
    check("N20: 'list_models' inputSchema accepts no inputs "
          "(empty properties + additionalProperties=False)",
          by_name["list_models"].inputSchema.get("properties") == {}
          and by_name["list_models"].inputSchema.get("additionalProperties") is False)
    check("N20: 'list_loras' inputSchema accepts no inputs",
          by_name["list_loras"].inputSchema.get("properties") == {}
          and by_name["list_loras"].inputSchema.get("additionalProperties") is False)
    check("N20: 'list_transformers' inputSchema accepts no inputs (slice-2b)",
          by_name["list_transformers"].inputSchema.get("properties") == {}
          and by_name["list_transformers"].inputSchema.get(
              "additionalProperties") is False)
    check("N20: 'list_loras'/'list_models' descriptions are byte-identical to "
          "their slice-2 constants (slice 2b does not touch them)",
          by_name["list_models"].description == mcps._LIST_MODELS_TOOL_DESCRIPTION
          and by_name["list_loras"].description == mcps._LIST_LORAS_TOOL_DESCRIPTION)


# --- N22: traceback strip on list_* internal exceptions ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )

    def _boom_with_secret(*_a, **_kw):
        raise RuntimeError(
            "/home/gawkahn/private/should-not-leak.txt: synthetic failure"
        )

    captured_err = io.StringIO()
    agent_facing_msg = None
    with unittest.mock.patch.object(sys, "stderr", captured_err), \
         unittest.mock.patch.object(mcps, "_handle_list_models",
                                    _boom_with_secret):
        try:
            _run(mcps._call_tool_impl(cfg, "list_models", {}))
        except ValueError as e:
            agent_facing_msg = str(e)
    check("N22: list_models internal exception surfaces as ValueError "
          "(framework-level sanitized shape)",
          agent_facing_msg is not None,
          detail=f"msg={agent_facing_msg!r}")
    if agent_facing_msg is not None:
        check("N22: agent-facing message does NOT contain 'Traceback'",
              "Traceback" not in agent_facing_msg)
        check("N22: agent-facing message does NOT contain '.py:' patterns",
              not re.search(r"\.py:\d+", agent_facing_msg))
        check("N22: agent-facing message does NOT contain absolute /home/ path",
              "/home/" not in agent_facing_msg)
        check("N22: agent-facing message does NOT leak the secret-path string",
              "should-not-leak" not in agent_facing_msg)
        check("N22: agent-facing message has 'internal_error: RuntimeError' shape",
              "internal_error" in agent_facing_msg
              and "RuntimeError" in agent_facing_msg)
    # Stderr SHOULD have the full traceback (operator visibility)
    stderr_text = captured_err.getvalue()
    check("N22: stderr captures the full traceback for operator audit",
          "Traceback" in stderr_text and "should-not-leak.txt" in stderr_text)
    # Audit line should be on stderr too, with error status
    _audit_lines = [ln for ln in stderr_text.splitlines()
                    if ln.startswith('{"tool":')]
    check("N22: audit line emitted with status=error + error_class=InternalError",
          any(json.loads(ln).get("status") == "error"
              and json.loads(ln).get("error_class") == "InternalError"
              for ln in _audit_lines))


# --- N23: static-source check — no argparse in mcp_server.py or catalog.py ---

_mcps_src = inspect.getsource(mcps)
_cat_src = inspect.getsource(cat_mod)
# Use a line-anchored regex (not a bare substring): docstrings and
# comments legitimately mention "import argparse" in prose ("this module
# does NOT import argparse"); the violation is a real import statement.
_argparse_import_re = re.compile(
    r"^\s*(?:import argparse\b|from argparse\b)", re.MULTILINE
)
check("N23: comfyless/mcp_server.py contains NO 'import argparse' statement",
      not _argparse_import_re.search(_mcps_src))
check("N23: comfyless/catalog.py contains NO 'import argparse' statement",
      not _argparse_import_re.search(_cat_src))
# AST-level sanity: parse the modules and verify there is no Import
# node for argparse anywhere.
for label, src in (("mcp_server", _mcps_src), ("catalog", _cat_src)):
    _tree = ast.parse(src)
    _argparse_imports = [
        n for n in ast.walk(_tree)
        if (isinstance(n, ast.Import)
            and any(a.name == "argparse" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and n.module == "argparse")
    ]
    check(f"N23: AST of {label} has no argparse imports",
          not _argparse_imports)


# --- N27: kind:"transformer" entries present in catalog, exposed ONLY
# through list_transformers (slice 2b) and excluded from list_models /
# list_loras (slice 2 carry-forward) ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    manifest, info = _build_step4_catalog(tmp_base)
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
        catalog=manifest,
    )
    # Catalog contains the transformer entry
    check("N27: catalog contains kind:'transformer' entries",
          any(e["kind"] == "transformer" for e in cfg.catalog.values()))
    _models, _ = _run(mcps._handle_list_models(cfg))
    _loras, _ = _run(mcps._handle_list_loras(cfg))
    _models_entries = json.loads(_models[0].text)
    _loras_entries = json.loads(_loras[0].text)
    check("N27: list_models response has NO transformer entries",
          not any(e["kind"] == "transformer" for e in _models_entries))
    check("N27: list_loras response has NO transformer entries",
          not any(e["kind"] == "transformer" for e in _loras_entries))
    check("N27: scan-transformer NOT in list_models response",
          "scan-transformer" not in {e["name"] for e in _models_entries})
    check("N27: scan-transformer NOT in list_loras response",
          "scan-transformer" not in {e["name"] for e in _loras_entries})
    # Slice-2b: the transformer IS surfaced by list_transformers, and
    # ONLY transformers are (no model/lora bleed-through).
    _tf, _tf_count = _run(mcps._handle_list_transformers(cfg))
    _tf_entries = json.loads(_tf[0].text)
    check("N27(2b): scan-transformer IS in list_transformers response",
          "scan-transformer" in {e["name"] for e in _tf_entries})
    check("N27(2b): list_transformers returns ONLY kind:'transformer' entries",
          _tf_entries
          and all(e["kind"] == "transformer" for e in _tf_entries))
    check("N27(2b): no model/lora names bleed into list_transformers",
          info["expected_models"].isdisjoint({e["name"] for e in _tf_entries})
          and info["expected_loras"].isdisjoint(
              {e["name"] for e in _tf_entries}))
    check("N27(2b): list_transformers count matches len(entries)",
          _tf_count == len(_tf_entries))


# ════════════════════════════════════════════════════════════════════════
print("\n== Slice 2b: list_transformers tool ==")
# ════════════════════════════════════════════════════════════════════════
#
# Invariants (docs/vision/slice-2b-mcp-list-transformers.md):
#   - Inv 2  response shape: strict-allowlist {name, kind, source[, model_family]};
#            NO abs_path / path / any filesystem string
#   - Inv 4  empty-input schema; audit-payload bounded to {} like the other
#            list_* tools; one audit line per invocation
#   - Inv 5  traceback strip carries forward (covered by the generic
#            _call_tool_impl outer-except; N22 pattern already exercises it)

# --- Tb1: list_transformers response shape (scan-derived omits model_family) ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    manifest, info = _build_step4_catalog(tmp_base)
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
        catalog=manifest,
    )
    result, count = _run(mcps._handle_list_transformers(cfg))
    check("Tb1: _handle_list_transformers returns (list[TextContent], int)",
          isinstance(result, list) and len(result) == 1
          and result[0].type == "text" and isinstance(count, int))
    entries = json.loads(result[0].text)
    check("Tb1: entry names match expected_transformers",
          {e["name"] for e in entries} == info["expected_transformers"],
          detail=f"got={sorted(e['name'] for e in entries)}")
    _allowed_keys = {"name", "kind", "source", "model_family"}
    _bad = [e for e in entries if not set(e.keys()).issubset(_allowed_keys)]
    check("Tb1: every entry's keys subset of {name, kind, source, model_family}",
          not _bad, detail=f"bad={_bad!r}")
    check("Tb1: NO entry contains 'abs_path' or 'path' key",
          not any("abs_path" in e or "path" in e for e in entries))
    check("Tb1: every entry has kind='transformer'",
          all(e["kind"] == "transformer" for e in entries))
    _scan_tf = next((e for e in entries if e["name"] == "scan-transformer"), None)
    check("Tb1: scan-derived transformer omits model_family (no inference)",
          _scan_tf is not None and "model_family" not in _scan_tf,
          detail=f"got={_scan_tf!r}")


# --- Tb2: manifest-declared model_family surfaces; abs_path never leaks ---
# Inject directly (mirror of N18) to prove the serializer's allowlist
# holds for a transformer entry carrying both a path and a family.

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    cfg.catalog["flux2-dit"] = {
        "abs_path": "/mnt/nvme-8tb/comfyui/diffusion_models/flux2-dit.safetensors",
        "kind": "transformer", "source": "manifest",
        "model_family": "flux2", "target_family": None,
    }
    result, _ = _run(mcps._handle_list_transformers(cfg))
    body = result[0].text
    check("Tb2: abs_path does NOT appear anywhere in list_transformers text",
          "/mnt/nvme-8tb" not in body
          and "flux2-dit.safetensors" not in body,
          detail=f"body={body!r}")
    _by_name = {e["name"]: e for e in json.loads(body)}
    check("Tb2: manifest-declared model_family surfaces on transformer",
          _by_name.get("flux2-dit", {}).get("model_family") == "flux2"
          and "abs_path" not in _by_name.get("flux2-dit", {}),
          detail=f"got={_by_name.get('flux2-dit')!r}")


# --- Tb3: empty catalog -> empty array, count 0 ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    _t_result, _t_count = _run(mcps._handle_list_transformers(cfg))
    check("Tb3: empty catalog -> list_transformers returns [] with count=0",
          json.loads(_t_result[0].text) == [] and _t_count == 0)


# --- Tb4: dispatch through _call_tool_impl + audit-payload bound to {} ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    manifest, info = _build_step4_catalog(tmp_base)
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
        catalog=manifest,
    )
    flood_blob = "Z" * 10_000
    captured_err = io.StringIO()
    with unittest.mock.patch.object(sys, "stderr", captured_err):
        tf_response = _run(mcps._call_tool_impl(
            cfg, "list_transformers", {"junk": flood_blob}))
    check("Tb4: _call_tool_impl('list_transformers') returns list[TextContent]",
          isinstance(tf_response, list) and len(tf_response) == 1
          and tf_response[0].type == "text"
          and isinstance(json.loads(tf_response[0].text), list))
    _stderr = captured_err.getvalue()
    check("Tb4: list_transformers audit line does NOT echo the flood blob",
          flood_blob not in _stderr and "junk" not in _stderr,
          detail=f"stderr[:200]={_stderr[:200]!r}")
    _audit_lines = [json.loads(ln) for ln in _stderr.splitlines()
                    if ln.startswith("{")]
    check("Tb4: exactly one list_transformers audit line, input={}, status=ok",
          len(_audit_lines) == 1
          and _audit_lines[0].get("tool") == "list_transformers"
          and _audit_lines[0].get("input") == {}
          and _audit_lines[0].get("status") == "ok"
          and "count" in _audit_lines[0],
          detail=f"lines={_audit_lines!r}")


# --- Empty-catalog edge case (N13 from Vision, list_* shape) ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    _m_result, _m_count = _run(mcps._handle_list_models(cfg))
    _l_result, _l_count = _run(mcps._handle_list_loras(cfg))
    check("Empty catalog: list_models returns empty JSON array, count=0",
          json.loads(_m_result[0].text) == [] and _m_count == 0)
    check("Empty catalog: list_loras returns empty JSON array, count=0",
          json.loads(_l_result[0].text) == [] and _l_count == 0)


# --- list_* through full _call_tool_impl dispatch (success path) ---

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    manifest, info = _build_step4_catalog(tmp_base)
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
        catalog=manifest,
    )
    captured_err = io.StringIO()
    with unittest.mock.patch.object(sys, "stderr", captured_err):
        models_response = _run(mcps._call_tool_impl(cfg, "list_models", {}))
        loras_response = _run(mcps._call_tool_impl(cfg, "list_loras", {}))
    check("Dispatch: _call_tool_impl('list_models') returns list[TextContent]",
          isinstance(models_response, list) and len(models_response) == 1
          and models_response[0].type == "text")
    check("Dispatch: _call_tool_impl('list_loras') returns list[TextContent]",
          isinstance(loras_response, list) and len(loras_response) == 1
          and loras_response[0].type == "text")
    # Sanity: both response bodies are valid JSON arrays
    check("Dispatch: list_models response body is a JSON array",
          isinstance(json.loads(models_response[0].text), list))
    check("Dispatch: list_loras response body is a JSON array",
          isinstance(json.loads(loras_response[0].text), list))


# --- Step-2 INFO-2 fold: catalog-name sanitization at build time ---
#
# Rejects C0/C1 controls, zero-width chars, bidi overrides/isolates,
# and LINE/PARAGRAPH SEPARATOR in catalog names (manifest keys here;
# scan-derived names cannot easily plant most of these via tempfile).
#
# Test inputs use \uXXXX escapes (NOT literal bidi/zw characters in
# source) so semgrep's bidi-detection rule does not flag this file on
# every edit. Python decodes the escapes at parse time, so the runtime
# strings DO contain the hostile codepoints that the catalog rejects.

def _build_with_manifest_key(mb: str, bad_name: str) -> None:
    """Helper: write a manifest with one entry keyed by `bad_name`, then
    call build_catalog. Should raise CatalogBuildError."""
    target = os.path.join(mb, "manifest-lora.safetensors")
    open(target, "wb").close()
    manifest_path = os.path.join(mb, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({bad_name: {"target": target, "kind": "lora"}}, f)
    cat_mod.build_catalog(mb, manifest_path)

# Sanity: a clean manifest key succeeds
with tempfile.TemporaryDirectory() as tmp_base:
    _ok = False
    try:
        _build_with_manifest_key(tmp_base, "clean-name_123")
        _ok = True
    except cat_mod.CatalogBuildError:
        pass
    check("Sanitize: clean manifest key 'clean-name_123' is accepted",
          _ok)

# Helper for sanitize cases: assert both exception type AND that the
# error message names the `_add_entry` gate. A future refactor that
# reorders gates (e.g., adding an earlier NUL-pre-check inside
# _parse_manifest_entry) could change WHICH gate caught the bad name,
# and the resulting message would no longer mention "forbidden
# character" \u2014 surfacing the regression here rather than silently. See
# code-reviewer slice-2 step-4 LOW-2 (2026-05-25).
def _check_sanitize_reject(name: str, bad_name: str, mb: str) -> None:
    raised = None
    try:
        _build_with_manifest_key(mb, bad_name)
    except cat_mod.CatalogBuildError as e:
        raised = e
    except BaseException as e:
        raised = f"UNEXPECTED-{type(e).__name__}: {e}"
    check(f"{name} -> CatalogBuildError",
          isinstance(raised, cat_mod.CatalogBuildError),
          detail=f"raised={raised!r}")
    check(f"{name} -> error message names 'forbidden character' "
          f"(proves the _add_entry gate fired, not some other reject)",
          isinstance(raised, cat_mod.CatalogBuildError)
          and "forbidden character" in str(raised),
          detail=f"raised={raised!r}")

# Reject NUL byte (C0)
with tempfile.TemporaryDirectory() as tmp_base:
    _check_sanitize_reject(
        "Sanitize: manifest key with NUL byte", "foo\x00bar", tmp_base
    )

# Reject DEL / C1 controls
with tempfile.TemporaryDirectory() as tmp_base:
    _check_sanitize_reject(
        "Sanitize: manifest key with C1 control", "foo\x9fbar", tmp_base
    )

# Reject zero-width chars (U+200B..U+200F)
with tempfile.TemporaryDirectory() as tmp_base:
    _check_sanitize_reject(
        "Sanitize: manifest key with zero-width space",
        "foo\u200bbar", tmp_base,
    )

# Reject bidi override formatting (U+202A..U+202E)
with tempfile.TemporaryDirectory() as tmp_base:
    _check_sanitize_reject(
        "Sanitize: manifest key with RLO bidi override",
        "foo\u202ebar", tmp_base,
    )

# Reject LINE SEPARATOR (U+2028)
with tempfile.TemporaryDirectory() as tmp_base:
    _check_sanitize_reject(
        "Sanitize: manifest key with LINE SEPARATOR",
        "foo\u2028bar", tmp_base,
    )

# Reject bidi isolate (U+2066..U+2069)
with tempfile.TemporaryDirectory() as tmp_base:
    _check_sanitize_reject(
        "Sanitize: manifest key with LRI bidi isolate",
        "foo\u2066bar", tmp_base,
    )

# Module-level: _FORBIDDEN_NAME_CHARS is a compiled regex
check("Sanitize: _FORBIDDEN_NAME_CHARS is a compiled regex",
      hasattr(cat_mod, "_FORBIDDEN_NAME_CHARS")
      and hasattr(cat_mod._FORBIDDEN_NAME_CHARS, "search"))


# ════════════════════════════════════════════════════════════════════════
print("\n== Slice 3 Step 1: resolve_reference (request-time resolver) ==")
# ════════════════════════════════════════════════════════════════════════
#
# Unit-level coverage of the pure resolver added to comfyless/catalog.py.
# Handler-integration coverage (uniform agent error, audit cause, notices,
# resolved_params-as-names) lands in Step 3 (Vision N1-N17). Here we prove
# the resolver returns the right ResolveResult / ResolveCause for every
# input shape — the building block the handler maps onto the uniform frame.
#
# Maps to Vision slice-3 invariants 1, 2 (cause set incl. KindMismatch),
# 4 (request-time existence + _within fail-closed), and the basename-strip
# / path_was_discarded flag that drives the §2 INFO notice.

_s3_root = tempfile.mkdtemp(prefix="s3-resolver-")
_s3_mb = os.path.join(_s3_root, "model-base")
os.makedirs(_s3_mb, exist_ok=True)
_make_model_fixture(_s3_mb, "qwen-image")
_make_loras_fixture(_s3_mb, ["anime-style"])
_make_transformer_fixture(_s3_mb, "diffusion_models", ["flux-dit"])
_s3_cat = cat_mod.build_catalog(_s3_mb, None)

# --- positive: bare name hit (no discard) ---
_r = cat_mod.resolve_reference(_s3_cat, "qwen-image", _s3_mb, expected_kind="model")
check("resolve: bare model name → ok", _r.ok, repr(_r))
check("resolve: bare model name → name is the catalog key",
      _r.name == "qwen-image", repr(_r))
check("resolve: bare model name → kind=model", _r.kind == "model", repr(_r))
check("resolve: bare model name → abs_path set + in model-base",
      bool(_r.abs_path) and _r.abs_path.startswith(os.path.realpath(_s3_mb)),
      repr(_r))
check("resolve: bare name → path_was_discarded False",
      _r.path_was_discarded is False, repr(_r))
check("resolve: hit has no cause", _r.cause is None, repr(_r))

# --- positive: path-shaped value → basename-strip + discard flag ---
_r = cat_mod.resolve_reference(
    _s3_cat, "/some/agent/dir/qwen-image", _s3_mb, expected_kind="model")
check("resolve: path-shaped value resolves via basename", _r.ok, repr(_r))
check("resolve: path-shaped → name is the catalog name (not the dir)",
      _r.name == "qwen-image", repr(_r))
check("resolve: path-shaped → path_was_discarded True",
      _r.path_was_discarded is True, repr(_r))
check("resolve: path-shaped → abs_path is catalog abs_path, NOT the supplied dir",
      _r.abs_path == _s3_cat["qwen-image"]["abs_path"], repr(_r))

# --- positive: lora + transformer kinds resolve under the right expected_kind ---
_r = cat_mod.resolve_reference(_s3_cat, "anime-style", _s3_mb, expected_kind="lora")
check("resolve: lora name under expected_kind=lora → ok + kind=lora",
      _r.ok and _r.kind == "lora", repr(_r))
_r = cat_mod.resolve_reference(_s3_cat, "flux-dit", _s3_mb, expected_kind="transformer")
check("resolve: transformer name under expected_kind=transformer → ok",
      _r.ok and _r.kind == "transformer", repr(_r))

# --- positive: expected_kind=None accepts any kind ---
_r = cat_mod.resolve_reference(_s3_cat, "anime-style", _s3_mb)
check("resolve: expected_kind=None accepts a lora", _r.ok and _r.kind == "lora",
      repr(_r))

# --- failure: UnknownName ---
_r = cat_mod.resolve_reference(_s3_cat, "does-not-exist", _s3_mb, expected_kind="model")
check("resolve: unknown name → not ok", not _r.ok, repr(_r))
check("resolve: unknown name → cause=UnknownName", _r.cause == "UnknownName", repr(_r))
check("resolve: unknown name → abs_path None (no leak)", _r.abs_path is None, repr(_r))

# --- failure: KindMismatch (lora name supplied where a model is expected) ---
_r = cat_mod.resolve_reference(_s3_cat, "anime-style", _s3_mb, expected_kind="model")
check("resolve: lora name as model → not ok", not _r.ok, repr(_r))
check("resolve: lora name as model → cause=KindMismatch",
      _r.cause == "KindMismatch", repr(_r))
check("resolve: KindMismatch → abs_path None (no leak)", _r.abs_path is None, repr(_r))

# --- failure: MalformedReference (null byte / empty-after-strip / forbidden char) ---
_r = cat_mod.resolve_reference(_s3_cat, "qwen-image\x00", _s3_mb, expected_kind="model")
check("resolve: NUL byte → cause=MalformedReference (no exception)",
      not _r.ok and _r.cause == "MalformedReference", repr(_r))
_r = cat_mod.resolve_reference(_s3_cat, "/trailing/dir/", _s3_mb, expected_kind="model")
check("resolve: path with trailing slash (empty basename) → MalformedReference",
      not _r.ok and _r.cause == "MalformedReference", repr(_r))
_r = cat_mod.resolve_reference(_s3_cat, "", _s3_mb, expected_kind="model")
check("resolve: empty string → MalformedReference",
      not _r.ok and _r.cause == "MalformedReference", repr(_r))
# Build the zero-width char programmatically — no literal hostile codepoint
# in this source file (mirrors catalog.py's bidi-detection-friendly policy).
_zw_name = "ze" + chr(0x200b) + "ro"
_r = cat_mod.resolve_reference(_s3_cat, _zw_name, _s3_mb, expected_kind="model")
check("resolve: zero-width char → MalformedReference",
      not _r.ok and _r.cause == "MalformedReference", repr(_r))
_r = cat_mod.resolve_reference(_s3_cat, 12345, _s3_mb, expected_kind="model")  # type: ignore[arg-type]
check("resolve: non-str input → MalformedReference (no exception)",
      not _r.ok and _r.cause == "MalformedReference", repr(_r))

# --- failure: PathMoved (catalog hit whose abs_path no longer exists) ---
_moved_root = tempfile.mkdtemp(prefix="s3-moved-")
_moved_mb = os.path.join(_moved_root, "model-base")
os.makedirs(_moved_mb, exist_ok=True)
_make_loras_fixture(_moved_mb, ["ghost"])
_moved_cat = cat_mod.build_catalog(_moved_mb, None)
os.remove(os.path.join(_moved_mb, "loras", "ghost.safetensors"))
_r = cat_mod.resolve_reference(_moved_cat, "ghost", _moved_mb, expected_kind="lora")
check("resolve: catalog hit whose path was deleted → not ok", not _r.ok, repr(_r))
check("resolve: deleted path → cause=PathMoved", _r.cause == "PathMoved", repr(_r))
check("resolve: PathMoved → no fallback abs_path returned", _r.abs_path is None, repr(_r))

# --- failure: WithinFailure (hand-built catalog entry pointing OUTSIDE model-base) ---
# build_catalog never produces this; construct it directly to prove the
# request-time _within net fires (Vision invariant 4 / MEDIUM-1).
_outside_root = tempfile.mkdtemp(prefix="s3-outside-")
_outside_file = os.path.join(_outside_root, "escaped.safetensors")
with open(_outside_file, "wb") as _f:
    _f.write(b"outside-model-base")
_evil_cat = {
    "escaped": {
        "abs_path": os.path.realpath(_outside_file),
        "kind": "lora", "source": "manifest",
        "model_family": None, "target_family": None,
    }
}
_r = cat_mod.resolve_reference(_evil_cat, "escaped", _s3_mb, expected_kind="lora")
check("resolve: entry abs_path outside model-base → not ok", not _r.ok, repr(_r))
check("resolve: outside model-base → cause=WithinFailure",
      _r.cause == "WithinFailure", repr(_r))
check("resolve: WithinFailure → no abs_path returned (no load)",
      _r.abs_path is None, repr(_r))

# --- malformed char in a DISCARDED directory component is stripped, not rejected ---
# Proves the malformed gate runs on the post-basename-strip candidate, not raw R:
# a NUL in the discarded dir must not poison a clean basename (code-reviewer gap 1).
_r = cat_mod.resolve_reference(
    _s3_cat, "/ev\x00il/qwen-image", _s3_mb, expected_kind="model")
check("resolve: NUL in discarded dir → clean basename still resolves",
      _r.ok and _r.name == "qwen-image", repr(_r))
check("resolve: NUL-in-dir resolve → path_was_discarded True",
      _r.path_was_discarded is True, repr(_r))

# --- path_was_discarded is preserved on a FAILING path-shaped input ---
# Step 2's audit line records that a path was supplied even on failure
# (code-reviewer gap 2).
_r = cat_mod.resolve_reference(
    _s3_cat, "/some/dir/does-not-exist", _s3_mb, expected_kind="model")
check("resolve: failing path-shaped input → cause=UnknownName",
      not _r.ok and _r.cause == "UnknownName", repr(_r))
check("resolve: failing path-shaped input → path_was_discarded preserved",
      _r.path_was_discarded is True, repr(_r))

# --- NFC-equivalence: a decomposed request resolves to a composed catalog key ---
# Proves request candidate and catalog key cannot disagree on normalization
# (invariant 1; code-reviewer gap 3). Build a lora whose name is composed "é".
_nfc_root = tempfile.mkdtemp(prefix="s3-nfc-")
_nfc_mb = os.path.join(_nfc_root, "model-base")
os.makedirs(_nfc_mb, exist_ok=True)
_composed = "caf" + chr(0x00e9)            # "café" with precomposed é (U+00E9)
_make_loras_fixture(_nfc_mb, [_composed])
_nfc_cat = cat_mod.build_catalog(_nfc_mb, None)
_decomposed = "caf" + "e" + chr(0x0301)    # "café" with combining acute (e + U+0301)
_r = cat_mod.resolve_reference(_nfc_cat, _decomposed, _nfc_mb, expected_kind="lora")
check("resolve: NFD request resolves to NFC catalog key (normalization symmetry)",
      _r.ok and _r.name == cat_mod.normalize_name(_composed), repr(_r))

# --- slice 3b: expected_kind accepts a TUPLE of kinds ----------------------
# Cascade stages resolve against {"model","transformer"} (a stage weight
# catalogs as transformer when single-file, model when a diffusers tree).
# Vision slice-3b invariants 2 (kind-set) + 8 (purely additive / byte-compat).
_kinds_mt = ("model", "transformer")
# model name accepted under the tuple
_r = cat_mod.resolve_reference(_s3_cat, "qwen-image", _s3_mb, expected_kind=_kinds_mt)
check("resolve: model name under ('model','transformer') → ok + kind=model",
      _r.ok and _r.kind == "model", repr(_r))
# transformer name accepted under the same tuple
_r = cat_mod.resolve_reference(_s3_cat, "flux-dit", _s3_mb, expected_kind=_kinds_mt)
check("resolve: transformer name under ('model','transformer') → ok + kind=transformer",
      _r.ok and _r.kind == "transformer", repr(_r))
# a lora name is NOT in the set → KindMismatch (folds into uniform not-available)
_r = cat_mod.resolve_reference(_s3_cat, "anime-style", _s3_mb, expected_kind=_kinds_mt)
check("resolve: lora name under ('model','transformer') → cause=KindMismatch",
      not _r.ok and _r.cause == "KindMismatch", repr(_r))
check("resolve: tuple KindMismatch → abs_path None (no leak)",
      _r.abs_path is None, repr(_r))
# byte-compat: a 1-tuple behaves identically to the bare str it wraps
_r_str = cat_mod.resolve_reference(_s3_cat, "qwen-image", _s3_mb, expected_kind="model")
_r_tup = cat_mod.resolve_reference(_s3_cat, "qwen-image", _s3_mb, expected_kind=("model",))
check("resolve: 1-tuple ('model',) == bare str 'model' (additive, no behavior drift)",
      _r_str.ok == _r_tup.ok and _r_str.kind == _r_tup.kind
      and _r_str.abs_path == _r_tup.abs_path, (repr(_r_str), repr(_r_tup)))
# a 1-tuple still REJECTS a wrong kind exactly as the bare str would
_r = cat_mod.resolve_reference(_s3_cat, "anime-style", _s3_mb, expected_kind=("model",))
check("resolve: 1-tuple ('model',) rejects a lora → KindMismatch",
      not _r.ok and _r.cause == "KindMismatch", repr(_r))

# --- module surface: ResolveResult dataclass + ResolveCause exported ---
check("resolve: ResolveResult exported", hasattr(cat_mod, "ResolveResult"))
check("resolve: resolve_reference is callable",
      callable(getattr(cat_mod, "resolve_reference", None)))


# ════════════════════════════════════════════════════════════════════════
print("\n== ADR-017: optional base64 image return (return_image) ==")
# ════════════════════════════════════════════════════════════════════════
#
# Gated, size-bounded base64 PNG return on generate + cascade. Invariants:
# default path byte-unchanged; bytes never in the audit line / stderr;
# size-bounded transport copy; on-disk PNG stays full-res; mime constant;
# fail-soft (never fails a successful gen); non-cascade + cascade identical.

import base64 as _b64  # noqa: E402


def _call_with_gen(cfg, args, gen_fn):
    """Like _call but with a caller-supplied generate() mock."""
    captured_err = io.StringIO()
    raised = None
    result = None
    with unittest.mock.patch.object(sys, "stderr", captured_err), \
         unittest.mock.patch.object(gen_mod, "_load_pipeline", _mock_load_pipeline), \
         unittest.mock.patch.object(gen_mod, "generate", gen_fn):
        try:
            result = _run(mcps._call_tool_impl(cfg, "generate", args))
        except ValueError as e:
            raised = str(e)
        except BaseException as e:
            raised = f"UNEXPECTED-{type(e).__name__}: {e}"
    return result, raised, captured_err.getvalue()


def _mock_generate_large(*, model_path, prompt, output_path, **kw):
    """generate() mock writing a 1536x768 PNG — exceeds the 1024 default cap."""
    Image.new("RGB", (1536, 768), "white").save(output_path)
    return {
        "prompt": prompt, "negative_prompt": kw.get("negative_prompt", ""),
        "model": model_path, "seed": kw.get("seed", 42),
        "steps": kw.get("steps", 28), "cfg_scale": kw.get("cfg_scale", 3.5),
        "transformer_path": kw.get("transformer_path", ""),
        "loras": list(kw.get("loras") or []),
        "elapsed_seconds": 0.01,
    }


def _mock_generate_noise(*, model_path, prompt, output_path, **kw):
    """generate() mock writing a 768x768 INCOMPRESSIBLE (random-noise) PNG so
    its base64 payload is large (~2 MB) — exercises the byte-cap downscale.
    White/flat images compress to a few hundred bytes and never trip it."""
    Image.frombytes("RGB", (768, 768), os.urandom(768 * 768 * 3)).save(
        output_path)
    return {
        "prompt": prompt, "negative_prompt": kw.get("negative_prompt", ""),
        "model": model_path, "seed": kw.get("seed", 42),
        "steps": kw.get("steps", 28), "cfg_scale": kw.get("cfg_scale", 3.5),
        "transformer_path": kw.get("transformer_path", ""),
        "loras": list(kw.get("loras") or []),
        "elapsed_seconds": 0.01,
    }


def _mock_generate_with_meta(*, model_path, prompt, output_path, **kw):
    """generate() mock that writes a PNG carrying a path-bearing tEXt chunk so
    the metadata-strip assertion is LOAD-BEARING: _encode_return_image re-saves
    with no pnginfo=, so the transport copy must NOT carry the planted chunk
    (invariant 5 — the data-egress control). A metadata-free source could not
    distinguish 'code strips' from 'source had none'."""
    from PIL import PngImagePlugin
    _meta = PngImagePlugin.PngInfo()
    _meta.add_text("abspath", "/home/gawkahn/secret/leak.png")
    _meta.add_text("prompt", prompt)
    Image.new("RGB", (16, 16), "white").save(output_path, pnginfo=_meta)
    return {
        "prompt": prompt, "negative_prompt": kw.get("negative_prompt", ""),
        "model": model_path, "seed": kw.get("seed", 42),
        "steps": kw.get("steps", 28), "cfg_scale": kw.get("cfg_scale", 3.5),
        "transformer_path": kw.get("transformer_path", ""),
        "loras": list(kw.get("loras") or []),
        "elapsed_seconds": 0.01,
    }


def _good_gen_args(**extra):
    a = {"prompt": "a cat", "model": "qwen-image"}
    a.update(extra)
    return a


# --- N1: return_image absent → response carries NO image fields (anchor) ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, _good_gen_args())
check("ADR-017 N1: return_image absent → success",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    check("ADR-017 N1: no image_b64 when return_image absent",
          "image_b64" not in resp)
    check("ADR-017 N1: no image_mime when return_image absent",
          "image_mime" not in resp)

# --- N2: return_image=false explicit → same as absent ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, _good_gen_args(return_image=False))
if result is not None:
    resp = _json.loads(result[0].text)
    check("ADR-017 N2: return_image=false → no image fields",
          "image_b64" not in resp and "image_mime" not in resp)

# --- N3: return_image=true → valid PNG base64, longest edge ≤ default cap ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, _good_gen_args(return_image=True))
check("ADR-017 N3: return_image=true → success",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    check("ADR-017 N3: image_b64 present", "image_b64" in resp)
    check("ADR-017 N3: image_mime == image/png",
          resp.get("image_mime") == "image/png")
    _pim = Image.open(io.BytesIO(_b64.b64decode(resp["image_b64"])))
    _pim.load()
    check("ADR-017 N3: image_b64 decodes to a valid PNG", _pim.format == "PNG")
    check("ADR-017 N3: returned longest edge ≤ 768 (default cap)",
          max(_pim.size) <= 768)
    # Invariant 5: the transport copy is re-encoded WITHOUT pnginfo, so it
    # carries no tEXt chunks — no on-disk metadata (or filesystem string)
    # can ride out through the returned image bytes. (A base64 substring
    # scan is meaningless here: base64's alphabet includes '/'.)
    check("ADR-017 N3: transport PNG carries NO text chunks (no metadata leak)",
          not getattr(_pim, "text", {}))

# --- N3b: metadata strip is LOAD-BEARING — source PNG carries a path-bearing
#          tEXt chunk; the transport copy must NOT (invariant 5) ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_with_gen(
    cfg, _good_gen_args(return_image=True), _mock_generate_with_meta)
check("ADR-017 N3b: meta-carrying gen succeeds",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    _disk = Image.open(resp["output_path"])
    _disk.load()
    # test sanity: the planted chunk really is on the source the encoder reads
    check("ADR-017 N3b: source PNG carries the planted tEXt chunk (sanity)",
          getattr(_disk, "text", {}).get("abspath")
          == "/home/gawkahn/secret/leak.png",
          detail=repr(getattr(_disk, "text", {})))
    _pim = Image.open(io.BytesIO(_b64.b64decode(resp["image_b64"])))
    _pim.load()
    # load-bearing: the re-encode (no pnginfo=) dropped ALL text chunks
    check("ADR-017 N3b: transport copy carries NO text chunks (strip proven)",
          not getattr(_pim, "text", {}), detail=repr(getattr(_pim, "text", {})))
    check("ADR-017 N3b: planted abspath value absent from image_b64 payload",
          "/home/gawkahn/secret" not in _b64.b64decode(resp["image_b64"])
          .decode("latin-1"))

# --- N4: image larger than cap → downscaled; on-disk PNG stays full-res ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_with_gen(
    cfg, _good_gen_args(return_image=True, max_return_px=1024),
    _mock_generate_large)
check("ADR-017 N4: large-image gen succeeds",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    _pim = Image.open(io.BytesIO(_b64.b64decode(resp["image_b64"])))
    _pim.load()
    check("ADR-017 N4: returned longest edge ≤ max_return_px (1024)",
          max(_pim.size) <= 1024)
    check("ADR-017 N4: aspect preserved (1536x768 → 1024x512)",
          _pim.size == (1024, 512), detail=repr(_pim.size))
    _disk = Image.open(resp["output_path"])
    _disk.load()
    check("ADR-017 N4: on-disk PNG stays FULL-RES (1536x768)",
          _disk.size == (1536, 768), detail=repr(_disk.size))

# --- N5: base64 payload NEVER appears in the audit line / stderr ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, _good_gen_args(return_image=True))
if result is not None:
    resp = _json.loads(result[0].text)
    _b = resp["image_b64"]
    check("ADR-017 N5: full image_b64 payload NOT in stderr/audit",
          _b not in stderr)
    check("ADR-017 N5: no 64-char b64 prefix leaked to stderr",
          _b[:64] not in stderr)
    check("ADR-017 N5: 'image_b64' key name absent from audit line",
          '"image_b64"' not in stderr)

# --- N6: non-bool return_image / non-int max_return_px → ValidationError ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, _good_gen_args(return_image="yes"))
check("ADR-017 N6: return_image non-bool → ValidationError before gen",
      result is None and err is not None
      and "return_image" in err and "bool" in err, detail=f"err={err!r}")
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(
    cfg, _good_gen_args(return_image=True, max_return_px=2.5))
check("ADR-017 N6: max_return_px non-int → ValidationError before gen",
      result is None and err is not None and "max_return_px" in err,
      detail=f"err={err!r}")
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(
    cfg, _good_gen_args(return_image=True, max_return_bytes="big"))
check("ADR-017 N6: max_return_bytes non-int → ValidationError before gen",
      result is None and err is not None and "max_return_bytes" in err,
      detail=f"err={err!r}")

# --- N7: cascade path with return_image=true → image_b64 present + valid ---
mb, out, _inside, cfg = _setup_mb_and_out()
_cargs = _good_cascade_args()
_cargs["return_image"] = True
result, err, stderr = _call_cascade(cfg, _cargs)
check("ADR-017 N7: cascade return_image=true → success",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    check("ADR-017 N7: cascade image_b64 present", "image_b64" in resp)
    check("ADR-017 N7: cascade image_mime == image/png",
          resp.get("image_mime") == "image/png")
    _pim = Image.open(io.BytesIO(_b64.b64decode(resp["image_b64"])))
    _pim.load()
    check("ADR-017 N7: cascade image_b64 decodes to valid bounded PNG",
          _pim.format == "PNG" and max(_pim.size) <= 1024)
    check("ADR-017 N7: cascade b64 NOT in stderr", resp["image_b64"] not in stderr)

# --- N8: encode failure → fail-soft (gen still returns; no image_b64) ---
def _boom_encoder(*a, **k):
    raise RuntimeError("encoder exploded")

mb, out, _inside, cfg = _setup_mb_and_out()
_captured_err = io.StringIO()
_raised = None
_result = None
with unittest.mock.patch.object(sys, "stderr", _captured_err), \
     unittest.mock.patch.object(gen_mod, "_load_pipeline", _mock_load_pipeline), \
     unittest.mock.patch.object(gen_mod, "generate", _mock_generate), \
     unittest.mock.patch.object(mcps, "_encode_return_image", _boom_encoder):
    try:
        _result = _run(mcps._call_tool_impl(
            cfg, "generate", _good_gen_args(return_image=True)))
    except BaseException as e:
        _raised = f"{type(e).__name__}: {e}"
check("ADR-017 N8: encode-failure does NOT raise (fail-soft)",
      _raised is None, detail=f"raised={_raised!r}")
if _result is not None:
    resp = _json.loads(_result[0].text)
    check("ADR-017 N8: generation still returns output_path",
          "output_path" in resp)
    check("ADR-017 N8: generation still returns resolved_params",
          "resolved_params" in resp)
    check("ADR-017 N8: no image_b64 on encode failure", "image_b64" not in resp)
    check("ADR-017 N8: fail-soft INFO notice present",
          any("transport copy" in n.get("message", "")
              for n in resp.get("notices", [])))

# --- N9: byte cap enforced — payload downscaled under a tight max_return_bytes;
#         on-disk PNG stays full-res (768x768) ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_with_gen(
    cfg, _good_gen_args(return_image=True, max_return_px=768,
                        max_return_bytes=65536),
    _mock_generate_noise)
check("ADR-017 N9: byte-capped gen succeeds",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    check("ADR-017 N9: image_b64 present under tight byte cap", "image_b64" in resp)
    check("ADR-017 N9: len(image_b64) ≤ max_return_bytes (65536)",
          len(resp["image_b64"]) <= 65536, detail=f"len={len(resp['image_b64'])}")
    _pim = Image.open(io.BytesIO(_b64.b64decode(resp["image_b64"])))
    _pim.load()
    check("ADR-017 N9: byte-capped payload still a valid PNG", _pim.format == "PNG")
    check("ADR-017 N9: byte-cap forced downscale below the 768 px bound",
          max(_pim.size) < 768, detail=repr(_pim.size))
    _disk = Image.open(resp["output_path"])
    _disk.load()
    check("ADR-017 N9: on-disk PNG stays FULL-RES (768x768) despite byte cap",
          _disk.size == (768, 768), detail=repr(_disk.size))

# --- N10: clamp — a max_return_bytes ABOVE the 1 MiB ceiling does not raise the
#          effective cap; an incompressible 768px image still comes back ≤ 1 MiB ---
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call_with_gen(
    cfg, _good_gen_args(return_image=True, max_return_px=768,
                        max_return_bytes=10_000_000),
    _mock_generate_noise)
check("ADR-017 N10: over-ceiling byte request succeeds",
      result is not None and err is None, detail=f"err={err!r}")
if result is not None:
    resp = _json.loads(result[0].text)
    # The agent asked for 10 MB; the server clamps to the 1 MiB ceiling, so an
    # incompressible 768px image (≈2 MB base64 unclamped) comes back downscaled.
    check("ADR-017 N10: requested 10MB clamped → payload ≤ 1 MiB ceiling",
          len(resp["image_b64"]) <= 1024 * 1024,
          detail=f"len={len(resp['image_b64'])}")
    _pim = Image.open(io.BytesIO(_b64.b64decode(resp["image_b64"])))
    _pim.load()
    check("ADR-017 N10: clamped payload still a valid PNG", _pim.format == "PNG")


# ════════════════════════════════════════════════════════════════════════
print("\n== Module hygiene ==")
# ════════════════════════════════════════════════════════════════════════

check("_MCP_PATH_TYPED_FIELDS is a tuple (immutable single-source-of-truth)",
      isinstance(mcps._MCP_PATH_TYPED_FIELDS, tuple))
check("_AUDIT_DROPPED_FIELDS is a frozenset",
      isinstance(mcps._AUDIT_DROPPED_FIELDS, frozenset))
check("_AUDIT_DROPPED_FIELDS contains prompt + negative_prompt",
      mcps._AUDIT_DROPPED_FIELDS == frozenset({"prompt", "negative_prompt"}))


# ════════════════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
