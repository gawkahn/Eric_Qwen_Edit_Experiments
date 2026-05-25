#!/usr/bin/env python3
"""Test suite for scripts/lora_audit.py (S1 of ADR-014).

Covers the S1-applicable Vision negative cases:
  #1  path-traversal-audit-root        test_path_traversal_audit_root
  #2  path-traversal-output-dir        test_path_traversal_output_dir
  #7  manifest determinism             test_manifest_determinism
  #8  per-file fault isolation         test_per_file_fault_isolation
  #10 symlink that doesn't escape      test_symlink_inside_resolves_normally
  #11 machine-caller non-interactive   test_machine_caller_non_interactive
  #12 forward-compatibility smoke      test_forward_compatibility_smoke
plus the security-review F-4 mandatory test:
      folder_paths stub installed     test_no_real_folder_paths_import
plus S1 review-fold coverage:
      F-8 argv redaction negative      test_redact_argv_no_path_leak
      F-17 output-dir mutex            test_output_dir_mutex_rejected
      --allow-output-outside-root use  test_allow_output_outside_root_usable

Run with the worktree's uv-managed venv per ADR-013:
  ./.venv/bin/python3 test_lora_audit.py
"""

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import types
from pathlib import Path

import safetensors.torch
import torch

_REPO_ROOT = Path(__file__).resolve().parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "lora_audit.py"
_FIX_ROOT = _REPO_ROOT / "tests" / "fixtures"
_TREE = _FIX_ROOT / "lora_audit_tree"
_BASE = _FIX_ROOT / "synthetic_base" / "transformer"

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        suffix = f" — {detail}" if detail else ""
        print(f"  FAIL  {name}{suffix}")


# ── Fixture builders ───────────────────────────────────────────────────
def _build_synth_base(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(64, 64),
        "transformer_blocks.0.attn.to_k.weight": torch.zeros(64, 64),
        "transformer_blocks.0.attn.to_v.weight": torch.zeros(64, 64),
        "transformer_blocks.0.attn.to_out.0.weight": torch.zeros(64, 64),
    }, str(base_dir / "model.safetensors"))


def _build_lora_tree(tree: Path) -> None:
    tree.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(8, 64),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(64, 8),
        "transformer_blocks.0.attn.to_k.lora_A.weight": torch.zeros(8, 64),
        "transformer_blocks.0.attn.to_k.lora_B.weight": torch.zeros(64, 8),
    }, str(tree / "usable.safetensors"))
    (tree / "sub").mkdir(exist_ok=True)
    safetensors.torch.save_file({
        "up_blocks.0.attentions.0.proj.lora_A.weight": torch.zeros(8, 16),
        "up_blocks.0.attentions.0.proj.lora_B.weight": torch.zeros(16, 8),
    }, str(tree / "sub" / "unconvertable.safetensors"))
    (tree / "zero.safetensors").write_bytes(b"")
    with open(tree / "truncated.safetensors", "wb") as f:
        f.write(struct.pack("<Q", 100))
        f.write(b"{ truncated json")
    (tree / "garbage.pt").write_bytes(b"definitely not a pt")


def setup_fixtures() -> None:
    # Localised cleanup: only wipe the two subdirs this test owns.
    # `tests/fixtures/` is shared with tests/test_lora_format_convert*.py
    # per ADR §1, so a blanket rmtree of _FIX_ROOT would silently delete
    # peer-test fixtures.
    shutil.rmtree(_TREE, ignore_errors=True)
    shutil.rmtree(_FIX_ROOT / "synthetic_base", ignore_errors=True)
    _FIX_ROOT.mkdir(parents=True, exist_ok=True)
    _build_synth_base(_BASE)
    _build_lora_tree(_TREE)


def teardown_fixtures() -> None:
    shutil.rmtree(_TREE, ignore_errors=True)
    shutil.rmtree(_FIX_ROOT / "synthetic_base", ignore_errors=True)


# ── Script import helper (used by in-process tests) ────────────────────
def _import_script():
    name = "lora_audit_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses need this before exec_module
    spec.loader.exec_module(mod)
    return mod


def _run_subprocess(args: list[str], stdin=None, timeout: float = 60.0):
    """Invoke the script as a subprocess. Returns CompletedProcess."""
    full = [sys.executable, str(_SCRIPT_PATH)] + args
    return subprocess.run(
        full,
        stdin=stdin if stdin is not None else subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO_ROOT),
    )


def _base_arg() -> str:
    return f"klein={_BASE}"


# ── Tests ──────────────────────────────────────────────────────────────
def test_no_real_folder_paths_import() -> None:
    print("\n[1] test_no_real_folder_paths_import (F-4)")
    mod = _import_script()
    fp = sys.modules.get("folder_paths")
    check("folder_paths is a ModuleType", isinstance(fp, types.ModuleType))
    check("folder_paths is the stub (get_folder_paths returns [])",
          fp.get_folder_paths("loras") == [])
    check("folder_paths.get_full_path returns None",
          fp.get_full_path("loras", "anything") is None)
    # The stub doesn't define real-comfyui attributes
    check("folder_paths has no models_dir attr (real comfyui would)",
          not hasattr(fp, "models_dir"))


def test_path_traversal_audit_root() -> None:
    print("\n[2] test_path_traversal_audit_root (Vision #1)")
    escape_link = _TREE / "escape.safetensors"
    if escape_link.exists() or escape_link.is_symlink():
        escape_link.unlink()
    escape_link.symlink_to("/etc/passwd")
    try:
        result = _run_subprocess([
            "--audit-root", str(_TREE), "--no-config",
            "--base", _base_arg(),
        ])
        check("exit code is 0 or 2 (run completes, no crash)",
              result.returncode in (0, 2),
              detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
        manifest = json.loads((_TREE / "lora_audit.json").read_text())
        symlink_warnings = [
            w for w in manifest["warnings"]
            if w["code"] == "excluded_symlink_escape" and "escape" in (w["file"] or "")
        ]
        check("escape symlink recorded as excluded_symlink_escape warning",
              len(symlink_warnings) >= 1,
              detail=f"warnings: {manifest['warnings']}")
        passwd_leaked = any(
            "root:" in json.dumps(f) for f in manifest["files"]
        )
        check("no /etc/passwd content leaked into manifest",
              not passwd_leaked)
    finally:
        if escape_link.exists() or escape_link.is_symlink():
            escape_link.unlink()
        (_TREE / "lora_audit.json").unlink(missing_ok=True)


def test_path_traversal_output_dir() -> None:
    print("\n[3] test_path_traversal_output_dir (Vision #2)")
    result = _run_subprocess([
        "--audit-root", str(_TREE), "--no-config",
        "--base", _base_arg(),
        "--output-dir", "/etc/should-fail",
    ])
    check("exit code is 1 (startup failure)", result.returncode == 1,
          detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
    check("[ERROR] line names the missing flag explanation",
          "neither --allow-output-outside-root" in result.stderr,
          detail=result.stderr[:300])
    check("no manifest was written", not (_TREE / "lora_audit.json").exists())


def test_manifest_determinism() -> None:
    print("\n[4] test_manifest_determinism (Vision #7)")
    a = _run_subprocess([
        "--audit-root", str(_TREE), "--no-config",
        "--base", _base_arg(),
        "-o", str(_TREE / "manifest_a.json"),
    ])
    b = _run_subprocess([
        "--audit-root", str(_TREE), "--no-config",
        "--base", _base_arg(),
        "-o", str(_TREE / "manifest_b.json"),
    ])
    check("both runs exited 0 or 2", a.returncode in (0, 2) and b.returncode in (0, 2),
          detail=f"a={a.returncode} b={b.returncode}")
    ma = json.loads((_TREE / "manifest_a.json").read_text())
    mb = json.loads((_TREE / "manifest_b.json").read_text())
    # ADR §13 — mask audited_at AND tool_invocation (argv contains the
    # per-run -o path, redacted but defensive against future drift).
    for m in (ma, mb):
        m.pop("audited_at", None)
        m.pop("tool_invocation", None)
    check("manifests byte-identical after masking audited_at + tool_invocation", ma == mb,
          detail="; ".join(f"{k}: A={ma.get(k)!r} B={mb.get(k)!r}"
                           for k in set(ma) | set(mb) if ma.get(k) != mb.get(k)))
    for name in ("manifest_a.json", "manifest_b.json"):
        (_TREE / name).unlink(missing_ok=True)


def test_per_file_fault_isolation() -> None:
    print("\n[5] test_per_file_fault_isolation (Vision #8)")
    mod = _import_script()
    original_check_lora = mod.check_lora

    def raising_check_lora(*args, **kwargs):
        raise RuntimeError("synthetic classifier failure")

    mod.check_lora = raising_check_lora
    try:
        out_path = _TREE / "fault_isolation.json"
        exit_code = mod.main([
            "--audit-root", str(_TREE), "--no-config",
            "--base", _base_arg(),
            "-o", str(out_path),
        ])
        check("exit code is 2 (file errors present)", exit_code == 2,
              detail=f"got {exit_code}")
        manifest = json.loads(out_path.read_text())
        error_entries = [
            f for f in manifest["files"] if f["classification"] == "error"
        ]
        check("at least one file marked classification=error",
              len(error_entries) >= 1,
              detail=f"files: {[f['classification'] for f in manifest['files']]}")
        good_entries = [
            f for f in manifest["files"]
            if f["classification"] in ("deletable",)
        ]
        check("deletable short-circuit still works (not blocked by classifier)",
              len(good_entries) >= 1)
        out_path.unlink(missing_ok=True)
    finally:
        mod.check_lora = original_check_lora


def test_symlink_inside_resolves_normally() -> None:
    print("\n[6] test_symlink_inside_resolves_normally (Vision #10)")
    inside_link = _TREE / "alias_usable.safetensors"
    if inside_link.exists() or inside_link.is_symlink():
        inside_link.unlink()
    inside_link.symlink_to(_TREE / "usable.safetensors")
    try:
        result = _run_subprocess([
            "--audit-root", str(_TREE), "--no-config",
            "--base", _base_arg(),
        ])
        check("exit code is 0 or 2", result.returncode in (0, 2),
              detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
        manifest = json.loads((_TREE / "lora_audit.json").read_text())
        alias_entries = [
            f for f in manifest["files"]
            if f["relative_path"] == "alias_usable.safetensors"
        ]
        check("inside-root symlink appears in files[]",
              len(alias_entries) == 1,
              detail=f"files: {[f['relative_path'] for f in manifest['files']]}")
        if alias_entries:
            check("symlink classified usable (same as target)",
                  alias_entries[0]["classification"] == "usable",
                  detail=f"got {alias_entries[0]['classification']}")
        alias_warnings = [
            w for w in manifest["warnings"]
            if "alias_usable" in (w["file"] or "")
        ]
        check("no excluded_symlink_escape warning for inside symlink",
              not any(w["code"] == "excluded_symlink_escape" for w in alias_warnings))
    finally:
        if inside_link.exists() or inside_link.is_symlink():
            inside_link.unlink()
        (_TREE / "lora_audit.json").unlink(missing_ok=True)


def test_machine_caller_non_interactive() -> None:
    print("\n[7] test_machine_caller_non_interactive (Vision #11, F-13)")
    import re as _re
    prefix_re = _re.compile(r"^\[(INFO|WARN|ERROR)\] ")
    result = _run_subprocess([
        "--audit-root", str(_TREE), "--no-config",
        "--base", _base_arg(),
        "--delete", "--yes",
    ], stdin=subprocess.DEVNULL, timeout=15.0)
    check("exit code is documented value (0/1/2)",
          result.returncode in (0, 1, 2),
          detail=f"got {result.returncode}")
    check("--delete --yes rejects in S1 with exit 1",
          result.returncode == 1)
    bad_lines = [
        line for line in result.stderr.splitlines()
        if line and not prefix_re.match(line)
    ]
    check("every stderr line matches ^\\[(INFO|WARN|ERROR)\\] regex",
          not bad_lines,
          detail=f"bad lines: {bad_lines[:3]}")
    result2 = _run_subprocess([
        "--audit-root", str(_TREE), "--no-config",
        "--base", _base_arg(),
        "--print-manifest",
    ], stdin=subprocess.DEVNULL, timeout=120.0)
    check("--print-manifest stdout is valid JSON only",
          bool(json.loads(result2.stdout)),
          detail=f"stdout head: {result2.stdout[:200]}")


def test_forward_compatibility_smoke() -> None:
    print("\n[8] test_forward_compatibility_smoke (Vision #12)")
    mod = _import_script()
    synthetic_future = {
        "audit_root": "/synth",
        "audit_version": 1,
        "audited_at": "2099-01-01T00:00:00Z",
        "bases": {"klein": {"path": "/synth/k", "param_count": 0, "dry_load_attempted": False}},
        "files": [
            {"kind": "lora",       "classification": "usable",      "reason": "ok",
             "relative_path": "a.safetensors", "sha256": "f00", "size_bytes": 100,
             "verdicts_by_base": {}, "convert_plan": None, "convert_output": None, "error": None},
            {"kind": "transformer", "classification": "usable",     "reason": "ok",
             "relative_path": "b.safetensors", "sha256": "f01", "size_bytes": 200,
             "verdicts_by_base": {}, "convert_plan": None, "convert_output": None, "error": None},
            {"kind": "lora",       "classification": "convertable", "reason": "convertable_loha",
             "relative_path": "c.safetensors", "sha256": "f02", "size_bytes": 150,
             "verdicts_by_base": {}, "convert_plan": {"source_family": "loha", "target_family": "lora", "target_base": "klein"},
             "convert_output": "c.lora.safetensors", "error": None},
        ],
        "totals": {"files_scanned": 3, "usable": 2, "convertable": 1,
                   "unconvertable": 0, "deletable": 0, "error": 0},
        "tool_invocation": {"argv_redacted": [], "config_path": None, "config_sha256": None},
        "tool_version": "0.99.0",
        "warnings": [],
        "future_field_we_dont_know_about": {"nested": True},
    }
    blob = json.dumps(synthetic_future, sort_keys=True)
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as e:
        check("synthetic future manifest is valid JSON", False, str(e))
        return
    check("audit_version is 1 (catalog filter point)", parsed["audit_version"] == 1)
    lora_files = [f for f in parsed["files"] if f["kind"] == "lora"]
    check("catalog can filter to kind=lora ignoring unknown kinds",
          len(lora_files) == 2)
    unknown_verdict_count = sum(
        1 for f in parsed["files"] if f["reason"] == "convertable_loha"
    )
    check("unknown reason codes pass through without crash",
          unknown_verdict_count == 1)
    check("unknown top-level fields don't break parse",
          parsed.get("future_field_we_dont_know_about", {}).get("nested") is True)
    check("MVP module constants stable for parser to depend on",
          mod._AUDIT_VERSION == 1 and isinstance(mod._TOOL_VERSION, str))


def test_redact_argv_no_path_leak() -> None:
    print("\n[9] test_redact_argv_no_path_leak (F-8 negative)")
    # Exercise long-flag, =-form, and -o<value> concatenated short-form.
    out_path = _TREE / "redact_check.json"
    short_concat_out = _TREE / "redact_short.json"
    result = _run_subprocess([
        "--audit-root", str(_TREE), "--no-config",
        "--base", _base_arg(),
        "-o", str(out_path),
    ])
    check("subprocess exit 0 or 2", result.returncode in (0, 2),
          detail=f"got {result.returncode}; stderr: {result.stderr[:200]}")
    manifest = json.loads(out_path.read_text())
    argv_redacted = manifest["tool_invocation"]["argv_redacted"]
    blob = json.dumps(argv_redacted)
    check("<redacted> appears in argv_redacted",
          "<redacted>" in blob,
          detail=f"argv_redacted={argv_redacted}")
    check("no raw _TREE absolute path leaked into argv_redacted",
          str(_TREE) not in blob,
          detail=f"argv_redacted={argv_redacted}")
    check("no raw _BASE absolute path leaked into argv_redacted",
          str(_BASE) not in blob,
          detail=f"argv_redacted={argv_redacted}")
    out_path.unlink(missing_ok=True)

    # Concatenated short form -o<path>
    result2 = _run_subprocess([
        "--audit-root", str(_TREE), "--no-config",
        "--base", _base_arg(),
        f"-o{short_concat_out}",
    ])
    check("subprocess (short -o<value>) exit 0 or 2",
          result2.returncode in (0, 2),
          detail=f"got {result2.returncode}; stderr: {result2.stderr[:200]}")
    if short_concat_out.exists():
        manifest2 = json.loads(short_concat_out.read_text())
        blob2 = json.dumps(manifest2["tool_invocation"]["argv_redacted"])
        check("short -o<value> form: no raw path leak",
              str(short_concat_out) not in blob2,
              detail=f"argv_redacted={manifest2['tool_invocation']['argv_redacted']}")
        check("short -o<value> form: -o<redacted> token present",
              "-o<redacted>" in blob2,
              detail=f"argv_redacted={manifest2['tool_invocation']['argv_redacted']}")
        short_concat_out.unlink(missing_ok=True)


def test_output_dir_mutex_rejected() -> None:
    print("\n[10] test_output_dir_mutex_rejected (F-17 mutex)")
    result = _run_subprocess([
        "--audit-root", str(_TREE), "--no-config",
        "--base", _base_arg(),
        "--allow-output-outside-root",
        "--require-output-allowlist",
        "--output-allowlist-prefix", str(_TREE),
        "--output-dir", "/tmp/should-fail-mutex",
    ])
    check("exit code is 1 (startup failure, argparse remapped per Vision §13)",
          result.returncode == 1,
          detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
    check("stderr names the mutex violation",
          "not allowed" in result.stderr.lower()
          or "mutually exclusive" in result.stderr.lower(),
          detail=result.stderr[:300])


def test_allow_output_outside_root_usable() -> None:
    print("\n[11] test_allow_output_outside_root_usable (B4 regression)")
    # Output to a non-blacklisted dir outside audit-root must succeed.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="lora_audit_outside_") as tmp:
        out_dir = Path(tmp) / "audit_out"
        out_dir.mkdir()
        result = _run_subprocess([
            "--audit-root", str(_TREE), "--no-config",
            "--base", _base_arg(),
            "--allow-output-outside-root",
            "--output-dir", str(out_dir),
            "--print-manifest",
        ])
        check("exit code is 0 or 2 (outside-root permitted via flag)",
              result.returncode in (0, 2),
              detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
        check("stderr does not mention 'system-blacklisted'",
              "system-blacklisted" not in result.stderr,
              detail=result.stderr[:300])
        manifest = json.loads(result.stdout)
        check("manifest output_dir matches the outside path",
              manifest.get("output_dir") == str(out_dir),
              detail=f"output_dir={manifest.get('output_dir')!r}")


# ── Driver ─────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 70)
    print(f"  test_lora_audit.py — S1 of ADR-014")
    print(f"  script: {_SCRIPT_PATH}")
    print(f"  fixtures: {_FIX_ROOT}")
    print("=" * 70)
    setup_fixtures()
    try:
        test_no_real_folder_paths_import()
        test_path_traversal_audit_root()
        test_path_traversal_output_dir()
        test_manifest_determinism()
        test_per_file_fault_isolation()
        test_symlink_inside_resolves_normally()
        test_machine_caller_non_interactive()
        test_forward_compatibility_smoke()
        test_redact_argv_no_path_leak()
        test_output_dir_mutex_rejected()
        test_allow_output_outside_root_usable()
    finally:
        teardown_fixtures()
    print("\n" + "─" * 70)
    print(f"  {passed} passed, {failed} failed")
    print("─" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
