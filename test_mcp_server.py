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
        # error_class names the step-2 rejection category
        # (model outside --model-base → PathAllowlist)
        check("audit line error_class == 'PathAllowlist' (step-2)",
              audit_record.get("error_class") == "PathAllowlist",
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
    # Write a fake PNG so callers that check the file see something.
    Image.new("RGB", (8, 8), "white").save(output_path)
    return {
        "prompt": prompt, "negative_prompt": kw.get("negative_prompt", ""),
        "model": model_path, "seed": kw.get("seed", 42),
        "steps": kw.get("steps", 28), "cfg_scale": kw.get("cfg_scale", 3.5),
        "elapsed_seconds": 0.01,
    }


def _setup_mb_and_out():
    """Return (model_base, output_dir, model_dir_inside_base, cfg)."""
    mb = tempfile.mkdtemp()
    out = tempfile.mkdtemp()
    inside_model = os.path.join(mb, "qwen-image")
    os.makedirs(inside_model)
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


# N5: absolute path OUTSIDE --model-base
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, {
    "prompt": "p", "model": "/etc/anything",
})
check("N5: model='/etc/anything' (outside --model-base) → MCP error",
      result is None and err is not None and "validation failed" in err,
      detail=f"err={err!r}")
check("N5: audit line written on rejection",
      "PathAllowlist" in stderr or "validation failed" in stderr)

# N6: traversal via .. segments — realpath collapses ..; if it lands outside,
# _check_paths rejects. Use a path like inside_model/../../../etc/passwd.
result, err, stderr = _call(cfg, {
    "prompt": "p",
    "model": os.path.join(_inside, "..", "..", "..", "etc", "passwd"),
})
check("N6: '..' traversal → MCP error after realpath",
      result is None and err is not None and "validation failed" in err)

# N7: symlink inside model_base pointing outside model_base
sym_target_outside = tempfile.mkdtemp()
symlink_in_base = os.path.join(mb, "evil-symlink")
os.symlink(sym_target_outside, symlink_in_base)
result, err, stderr = _call(cfg, {
    "prompt": "p", "model": symlink_in_base,
})
check("N7: symlink-inside-base pointing outside → MCP error",
      result is None and err is not None and "validation failed" in err)

# N8: savepath escapes --output-dir via `..` traversal. The daemon's
# template machinery (mirrored by _resolve_mcp_output_path) lstrips
# leading slashes so an "absolute" savepath becomes relative-from-output-
# dir (deliberate human-UX behavior in the daemon; preserved for MCP per
# invariant-3 wording — "rejects on _within failure", not "rejects on
# absolute-path"). A `..`-traversal savepath, however, DOES escape and
# must be rejected.
result, err, stderr = _call(cfg, {
    "prompt": "p", "model": _inside,
    "savepath": "../../etc/passwd",
})
check("N8: savepath '..' traversal outside --output-dir → MCP error",
      result is None and err is not None and "validation failed" in err,
      detail=f"err={err!r}")

# N9: loras[0].path outside --model-base
result, err, stderr = _call(cfg, {
    "prompt": "p", "model": _inside,
    "loras": [{"path": "/etc/bad.safetensors", "weight": 0.5}],
})
check("N9: loras[0].path outside --model-base → MCP error",
      result is None and err is not None and "validation failed" in err)


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 2: HF cache miss (N10) ==")
# ════════════════════════════════════════════════════════════════════════

# `snapshot_download` is imported INSIDE resolve_hf_path's body, so we
# must patch it on the source module (`huggingface_hub`) — patching
# nodes.eric_diffusion_utils.snapshot_download doesn't find an attribute
# since it's a local import.
import nodes.eric_diffusion_utils as eu  # noqa: E402
import huggingface_hub  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

def _mock_snapshot_local(repo_id, *, local_files_only=True, **_):
    # Always miss the local cache so resolve_hf_path raises ValueError
    # when allow_download is False.
    raise LocalEntryNotFoundError(repo_id)

mb, out, _inside, cfg = _setup_mb_and_out()
captured_err = io.StringIO()
raised = None
with unittest.mock.patch.object(sys, "stderr", captured_err), \
     unittest.mock.patch.object(huggingface_hub, "snapshot_download",
                                _mock_snapshot_local):
    try:
        _run(mcps._call_tool_impl(cfg, "generate", {
            "prompt": "p", "model": "Qwen/Qwen-Image",
        }))
    except ValueError as e:
        raised = str(e)
check("N10: HF repo ID not in local cache → MCP error (no network call)",
      raised is not None and "HF repo not in local cache" in raised,
      detail=f"raised={raised!r}")
check("N10: audit line records HFCacheMiss",
      "HFCacheMiss" in captured_err.getvalue())


# ════════════════════════════════════════════════════════════════════════
print("\n== Step 2: allow_hf_download=False regression (N11) ==")
# ════════════════════════════════════════════════════════════════════════

# Monkey-patch resolve_hf_path to record every (path, allow_download)
# call across the entire generate code path; assert every recorded
# allow_download is False.
mb, out, _inside, cfg = _setup_mb_and_out()
recorded_calls: list = []
original_resolve = eu.resolve_hf_path

def _spy_resolve(path, *, allow_download=False):
    recorded_calls.append((path, allow_download))
    return original_resolve(path, allow_download=allow_download)

with unittest.mock.patch.object(sys, "stderr", io.StringIO()), \
     unittest.mock.patch.object(eu, "resolve_hf_path", _spy_resolve), \
     unittest.mock.patch.object(gen_mod, "resolve_hf_path", _spy_resolve), \
     unittest.mock.patch.object(gen_mod, "_load_pipeline", _mock_load_pipeline), \
     unittest.mock.patch.object(gen_mod, "generate", _mock_generate):
    try:
        _run(mcps._call_tool_impl(cfg, "generate", {
            "prompt": "p", "model": _inside,
        }))
    except ValueError:
        pass
check("N11: resolve_hf_path was called at least once",
      len(recorded_calls) >= 1, detail=f"calls={len(recorded_calls)}")
check("N11: every recorded allow_download is False",
      all(ad is False for (_p, ad) in recorded_calls),
      detail=f"truthy: {[c for c in recorded_calls if c[1]]}")


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
    check("N25: response contains `resolved_params` (full blob)",
          "resolved_params" in response_obj)
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

# Security-auditor F1: null-byte in any path-typed field must be rejected
# BEFORE _check_paths' realpath would explode. Audit class must be
# ValidationError, NOT InternalError.
for nb_field in ("model", "transformer_path", "vae_path",
                 "text_encoder_path", "text_encoder_2_path", "savepath"):
    mb, out, _inside, cfg = _setup_mb_and_out()
    args = {"prompt": "p", "model": _inside}
    args[nb_field] = ((_inside if nb_field != "savepath" else "") + "\x00null")
    if nb_field == "savepath":
        # savepath doesn't need to start with _inside
        args["savepath"] = "out\x00null.png"
    result, err, stderr = _call(cfg, args)
    check(f"F1: null byte in `{nb_field}` → MCP error",
          result is None and err is not None and "null byte not allowed" in err,
          detail=f"err={err!r}")
    check(f"F1: null-byte audit class is ValidationError (not InternalError) for `{nb_field}`",
          "ValidationError" in stderr and "InternalError" not in stderr)

# Null byte in loras[i].path
mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, {
    "prompt": "p", "model": _inside,
    "loras": [{"path": _inside + "\x00null", "weight": 0.5}],
})
check("F1: null byte in loras[0].path → MCP error",
      result is None and err is not None and "null byte not allowed" in err)
check("F1: loras null-byte audit class is ValidationError",
      "ValidationError" in stderr)

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
print("\n== Step 2: cascade_config branch (step-3 deferral) ==")
# ════════════════════════════════════════════════════════════════════════

mb, out, _inside, cfg = _setup_mb_and_out()
result, err, stderr = _call(cfg, {
    "prompt": "p", "model": _inside,
    "cascade_config": {"stage_c": "/m/cascade_c", "stage_b": "/m/cascade_b"},
})
check("cascade_config branch: rejected with NotYetWired (step 3 lands)",
      result is None and err is not None and "step 3" in err.lower())
check("cascade_config branch: audit line records CascadeNotYetWired",
      "CascadeNotYetWired" in stderr)


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
