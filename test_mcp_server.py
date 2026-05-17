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
  - tools/list advertises exactly ["generate"]    (invariant 6)
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
import os
import re
import sys
import tempfile
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import comfyless.mcp_server as mcps  # noqa: E402
import comfyless.generate as gen_mod  # noqa: E402
from click.testing import CliRunner  # noqa: E402
from mcp.types import Tool  # noqa: E402


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
print("\n== Invariant 6: tools/list returns exactly ['generate'] ==")
# ════════════════════════════════════════════════════════════════════════

with tempfile.TemporaryDirectory() as tmp_out, \
     tempfile.TemporaryDirectory() as tmp_base:
    cfg = mcps._validate_startup_args(
        output_dir=tmp_out, model_base=tmp_base,
        default_model=None, mcp_max_iterations=100,
    )
    tools = _run(mcps._list_tools_impl(cfg))
    check("list_tools returns one element",
          isinstance(tools, list) and len(tools) == 1,
          detail=f"len={len(tools)}")
    check("list_tools[0] is a Tool",
          isinstance(tools[0], Tool))
    check("list_tools[0].name == 'generate' (invariant 6)",
          tools[0].name == "generate")
    check("list_tools[0] has tool-description steering text",
          tools[0].description and "qwen-image" in tools[0].description.lower(),
          detail="description should name model families per ADR-011 §2 amendment")


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
        except NotImplementedError:
            pass  # expected in step 1

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
        # error_class names the step-1 stub reason
        check("audit line error_class == 'NotImplementedYet' (step-1 stub)",
              audit_record.get("error_class") == "NotImplementedYet")
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
