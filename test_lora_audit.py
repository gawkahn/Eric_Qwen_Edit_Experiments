#!/usr/bin/env python3
"""Test suite for scripts/lora_audit.py (S1+S2 of ADR-014).

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
plus S2 dry-load coverage (ADR §7):
      default-off no pipeline import   test_dry_load_disabled_default_no_pipe_import
      per-file fault isolation         test_dry_load_per_file_fault_isolation
      base-load failure fallback       test_dry_load_base_load_failure_falls_back
      VRAM-cascade warning             test_dry_load_vram_cascade_warning
      manifest schema sub-object       test_manifest_schema_dry_load_subobject
plus S2 review-fold coverage:
      print-manifest stdout contract   test_dry_load_print_manifest_stdout_clean
      loaded=False nulls applied       test_dry_load_loaded_false_nulls_applied_modules
      pre-load containment recheck     test_dry_load_containment_recheck

Run with the worktree's uv-managed venv per ADR-013:
  ./.venv/bin/python3 test_lora_audit.py
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
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


# ── S2 dry-load helpers and tests ──────────────────────────────────────
def _build_dry_load_fixture(root: Path) -> tuple[Path, Path]:
    """Build a 2-LoRA tree + synthetic base under *root*.

    Both LoRAs target keys present in the synthetic base so each one is a
    non-WRONG_ARCH dry-load candidate. Returns (tree_dir, base_dir).
    """
    tree = root / "tree"
    base = root / "synth" / "transformer"
    _build_synth_base(base)
    tree.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file({
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(8, 64),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(64, 8),
    }, str(tree / "first.safetensors"))
    safetensors.torch.save_file({
        "transformer_blocks.0.attn.to_k.lora_A.weight": torch.zeros(8, 64),
        "transformer_blocks.0.attn.to_k.lora_B.weight": torch.zeros(64, 8),
    }, str(tree / "second.safetensors"))
    return tree, base


class _DryLoadPatch:
    """Context manager that swaps mod's dry-load surface for fakes
    and restores the originals on exit."""

    def __init__(self, mod, *,
                 pipeline_factory=None,
                 loader=None,
                 unloader=None):
        self.mod = mod
        self.pipeline_factory = pipeline_factory
        self.loader = loader
        self.unloader = unloader
        self._saved: dict = {}

    def __enter__(self):
        self._saved["pipe"] = self.mod._load_dry_load_pipeline
        self._saved["loader"] = self.mod.load_lora_with_key_fix
        self._saved["unloader"] = self.mod.unload_adapters
        if self.pipeline_factory is not None:
            self.mod._load_dry_load_pipeline = self.pipeline_factory
        if self.loader is not None:
            self.mod.load_lora_with_key_fix = self.loader
        if self.unloader is not None:
            self.mod.unload_adapters = self.unloader
        return self

    def __exit__(self, *exc):
        self.mod._load_dry_load_pipeline = self._saved["pipe"]
        self.mod.load_lora_with_key_fix = self._saved["loader"]
        self.mod.unload_adapters = self._saved["unloader"]


def test_dry_load_disabled_default_no_pipe_import() -> None:
    print("\n[12] test_dry_load_disabled_default_no_pipe_import (S2 ADR §7)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="dryload_off_") as tmp:
        tree, base = _build_dry_load_fixture(Path(tmp))

        def boom(_dir):
            raise AssertionError(
                "_load_dry_load_pipeline must not be invoked when "
                "--dry-load is off"
            )

        with _DryLoadPatch(mod, pipeline_factory=boom):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}",
                "-o", str(tree / "manifest.json"),
            ])
        check("exits cleanly without --dry-load",
              exit_code in (0, 2),
              detail=f"got {exit_code}")
        manifest = json.loads((tree / "manifest.json").read_text())
        check("bases[klein].dry_load_attempted is False",
              manifest["bases"]["klein"]["dry_load_attempted"] is False)
        no_subobj = all(
            "dry_load" not in v
            for f in manifest["files"]
            for v in (f.get("verdicts_by_base") or {}).values()
        )
        check("no dry_load sub-object on any per-base verdict", no_subobj)


def test_dry_load_per_file_fault_isolation() -> None:
    print("\n[13] test_dry_load_per_file_fault_isolation (Vision invariant 9)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="dryload_iso_") as tmp:
        tree, base = _build_dry_load_fixture(Path(tmp))

        def fake_loader(_pipe, lora_path, adapter_name=None, **kw):
            if "first.safetensors" in lora_path:
                print("[LoRA] LoRA direct merge (weight=1.0): "
                      "applied=4, skipped=0")
                return True
            raise RuntimeError("synthetic per-file fault")

        with _DryLoadPatch(
            mod,
            pipeline_factory=lambda _d: object(),
            loader=fake_loader,
            unloader=lambda _p, _names, **kw: None,
        ):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}",
                "--dry-load",
                "-o", str(tree / "manifest.json"),
            ])

        check("dry-load run completes despite per-file fault",
              exit_code in (0, 2),
              detail=f"got {exit_code}")
        manifest = json.loads((tree / "manifest.json").read_text())
        files_by_rel = {f["relative_path"]: f for f in manifest["files"]}
        first = files_by_rel.get("first.safetensors")
        second = files_by_rel.get("second.safetensors")
        check("first.safetensors present", first is not None)
        check("second.safetensors present", second is not None)
        if first and second:
            f_dl = first["verdicts_by_base"]["klein"].get("dry_load")
            s_dl = second["verdicts_by_base"]["klein"].get("dry_load")
            check("first.safetensors dry_load.loaded is True",
                  f_dl is not None and f_dl.get("loaded") is True,
                  detail=f"dry_load={f_dl}")
            check("first.safetensors dry_load.applied_modules == 4",
                  f_dl is not None and f_dl.get("applied_modules") == 4,
                  detail=f"dry_load={f_dl}")
            check("second.safetensors dry_load.loaded is False",
                  s_dl is not None and s_dl.get("loaded") is False,
                  detail=f"dry_load={s_dl}")
            check("second.safetensors dry_load.reason names the exception",
                  s_dl is not None
                  and "RuntimeError" in (s_dl.get("reason") or ""),
                  detail=f"dry_load={s_dl}")
        check("bases[klein].dry_load_attempted is True",
              manifest["bases"]["klein"]["dry_load_attempted"] is True)


def test_dry_load_base_load_failure_falls_back() -> None:
    print("\n[14] test_dry_load_base_load_failure_falls_back (ADR §7)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="dryload_bf_") as tmp:
        tree, base = _build_dry_load_fixture(Path(tmp))

        def fail_pipe(_dir):
            raise RuntimeError("synthetic OOM")

        def must_not_call_loader(*a, **kw):
            raise AssertionError(
                "load_lora_with_key_fix must not be invoked if base load "
                "failed"
            )

        with _DryLoadPatch(
            mod,
            pipeline_factory=fail_pipe,
            loader=must_not_call_loader,
        ):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}",
                "--dry-load",
                "-o", str(tree / "manifest.json"),
            ])

        check("run completes despite base-load failure",
              exit_code in (0, 2),
              detail=f"got {exit_code}")
        manifest = json.loads((tree / "manifest.json").read_text())
        check("bases[klein].dry_load_attempted is False",
              manifest["bases"]["klein"]["dry_load_attempted"] is False,
              detail=f"bases={manifest['bases']}")
        no_subobj = all(
            "dry_load" not in v
            for f in manifest["files"]
            for v in (f.get("verdicts_by_base") or {}).values()
        )
        check("no dry_load sub-object on any file's per-base verdict",
              no_subobj)
        codes = [w["code"] for w in manifest["warnings"]]
        check("dry_load_base_failed warning recorded",
              "dry_load_base_failed" in codes,
              detail=f"codes={codes}")
        check("no vram_cascade_possible (only one base)",
              "dry_load_vram_cascade_possible" not in codes,
              detail=f"codes={codes}")


def test_dry_load_vram_cascade_warning() -> None:
    print("\n[15] test_dry_load_vram_cascade_warning (ADR §7 F-3)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="dryload_casc_") as tmp:
        tree, base = _build_dry_load_fixture(Path(tmp))

        def fail_pipe(_dir):
            raise RuntimeError("synthetic OOM")

        with _DryLoadPatch(mod, pipeline_factory=fail_pipe):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}",
                "--base", f"sibling={base}",
                "--dry-load",
                "-o", str(tree / "manifest.json"),
            ])

        check("run completes despite both bases failing",
              exit_code in (0, 2),
              detail=f"got {exit_code}")
        manifest = json.loads((tree / "manifest.json").read_text())
        codes = [w["code"] for w in manifest["warnings"]]
        n_failed = sum(1 for c in codes if c == "dry_load_base_failed")
        n_cascade = sum(1 for c in codes if c == "dry_load_vram_cascade_possible")
        check("two dry_load_base_failed warnings (one per base)",
              n_failed == 2,
              detail=f"codes={codes}")
        check("exactly one vram_cascade_possible warning (fires on 2nd failure only)",
              n_cascade == 1,
              detail=f"codes={codes}")


def test_manifest_schema_dry_load_subobject() -> None:
    print("\n[16] test_manifest_schema_dry_load_subobject (ADR §5/§7)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="dryload_schema_") as tmp:
        tree, base = _build_dry_load_fixture(Path(tmp))

        def fake_loader(_pipe, lora_path, adapter_name=None, **kw):
            print("[LoRA] LoRA direct merge (weight=1.0): "
                  "applied=7, skipped=0")
            return True

        with _DryLoadPatch(
            mod,
            pipeline_factory=lambda _d: object(),
            loader=fake_loader,
            unloader=lambda _p, _names, **kw: None,
        ):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}",
                "--dry-load",
                "-o", str(tree / "manifest.json"),
            ])

        check("happy-path run exits 0/2",
              exit_code in (0, 2),
              detail=f"got {exit_code}")
        manifest = json.loads((tree / "manifest.json").read_text())
        check("bases[klein].dry_load_attempted is True",
              manifest["bases"]["klein"]["dry_load_attempted"] is True)
        lora_files = [
            f for f in manifest["files"]
            if f["kind"] == "lora" and f["classification"] == "usable"
        ]
        check("≥1 usable LoRA was dry-loaded",
              len(lora_files) >= 1)
        for f in lora_files:
            dl = (f.get("verdicts_by_base") or {}).get("klein", {}).get("dry_load")
            check(f"{f['relative_path']}: dry_load is a dict",
                  isinstance(dl, dict),
                  detail=f"dry_load={dl}")
            if isinstance(dl, dict):
                check(f"{f['relative_path']}: dry_load has exact keys "
                      f"{{loaded, applied_modules, reason}}",
                      set(dl.keys()) == {"loaded", "applied_modules", "reason"},
                      detail=f"keys={sorted(dl.keys())}")
                check(f"{f['relative_path']}: loaded is True",
                      dl.get("loaded") is True)
                check(f"{f['relative_path']}: applied_modules == 7",
                      dl.get("applied_modules") == 7,
                      detail=f"dl={dl}")
                check(f"{f['relative_path']}: reason is None on success",
                      dl.get("reason") is None,
                      detail=f"dl={dl}")


def test_dry_load_print_manifest_stdout_clean() -> None:
    print("\n[17] test_dry_load_print_manifest_stdout_clean "
          "(S2 security H-1 fold)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="dryload_pm_") as tmp:
        tree, base = _build_dry_load_fixture(Path(tmp))

        def noisy_pipe(_dir):
            # Simulate diffusers' from_pretrained banner output.
            print("Some weights of the model checkpoint at ... were not "
                  "used when initializing FluxTransformer2DModel: ...")
            print("- This IS expected if you are initializing FluxModel ...")
            return object()

        def fake_loader(_pipe, lora_path, adapter_name=None, **kw):
            print("[LoRA] LoRA direct merge (weight=1.0): "
                  "applied=3, skipped=0")
            return True

        original_pipe = mod._load_dry_load_pipeline
        original_loader = mod.load_lora_with_key_fix
        original_unloader = mod.unload_adapters
        mod._load_dry_load_pipeline = noisy_pipe
        mod.load_lora_with_key_fix = fake_loader
        mod.unload_adapters = lambda _p, _names, **kw: None
        try:
            captured_stdout = io.StringIO()
            with contextlib.redirect_stdout(captured_stdout):
                exit_code = mod.main([
                    "--audit-root", str(tree), "--no-config",
                    "--base", f"klein={base}",
                    "--dry-load",
                    "--print-manifest",
                ])
        finally:
            mod._load_dry_load_pipeline = original_pipe
            mod.load_lora_with_key_fix = original_loader
            mod.unload_adapters = original_unloader

        check("--dry-load --print-manifest exits 0/2",
              exit_code in (0, 2),
              detail=f"got {exit_code}")
        stdout_text = captured_stdout.getvalue().strip()
        check("stdout is non-empty (manifest present)",
              bool(stdout_text))
        try:
            parsed = json.loads(stdout_text)
            check("stdout parses as JSON (no from_pretrained chatter mixed in)",
                  True)
            check("parsed JSON has bases.klein.dry_load_attempted=True",
                  parsed["bases"]["klein"]["dry_load_attempted"] is True)
        except json.JSONDecodeError as e:
            check("stdout parses as JSON (no from_pretrained chatter mixed in)",
                  False, detail=f"JSONDecodeError: {e}; head={stdout_text[:200]}")


def test_dry_load_loaded_false_nulls_applied_modules() -> None:
    print("\n[18] test_dry_load_loaded_false_nulls_applied_modules "
          "(code-review H-1 fold)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="dryload_nulls_") as tmp:
        tree, base = _build_dry_load_fixture(Path(tmp))

        def zero_applied_loader(_pipe, lora_path, adapter_name=None, **kw):
            # Matches the direct-merge loader's "applied=0" log line on
            # zero-match. Return False to simulate the loader's
            # `return applied > 0` exit at lines 446 / 613 / 1042 in
            # nodes/eric_qwen_edit_lora.py.
            print("[LoRA] LoRA direct merge (weight=1.0): "
                  "applied=0, skipped=4")
            return False

        with _DryLoadPatch(
            mod,
            pipeline_factory=lambda _d: object(),
            loader=zero_applied_loader,
            unloader=lambda _p, _names, **kw: None,
        ):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}",
                "--dry-load",
                "-o", str(tree / "manifest.json"),
            ])

        check("run exits 0/2", exit_code in (0, 2),
              detail=f"got {exit_code}")
        manifest = json.loads((tree / "manifest.json").read_text())
        for f in manifest["files"]:
            if f["classification"] != "usable":
                continue
            dl = (f.get("verdicts_by_base") or {}).get("klein", {}).get("dry_load")
            if dl is None:
                continue
            check(f"{f['relative_path']}: loaded is False",
                  dl.get("loaded") is False,
                  detail=f"dl={dl}")
            check(f"{f['relative_path']}: applied_modules is None when "
                  f"loaded=False (not 0)",
                  dl.get("applied_modules") is None,
                  detail=f"dl={dl}")
            check(f"{f['relative_path']}: reason is None (no exception)",
                  dl.get("reason") is None,
                  detail=f"dl={dl}")


def test_dry_load_containment_recheck() -> None:
    print("\n[19] test_dry_load_containment_recheck "
          "(security M-1 fold — symlink TOCTOU)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="dryload_toctou_") as tmp:
        tree, base = _build_dry_load_fixture(Path(tmp))
        # `outside` exists outside the audit-root and outside the tmpdir
        # so the symlink target unambiguously escapes.
        outside = Path(tmp).parent / f"escape_target_{os.getpid()}.safetensors"
        # Copy a valid LoRA file there so the recheck (not the open) is
        # the failure point — this isolates the TOCTOU narrowing.
        outside.write_bytes((tree / "first.safetensors").read_bytes())

        def swapping_loader(_pipe, lora_path, adapter_name=None, **kw):
            # If recheck were absent, this loader would happily see the
            # swapped symlink target. The recheck must short-circuit
            # before we ever reach this point — assert as much.
            real = os.path.realpath(lora_path)
            if str(outside.resolve()) == real:
                raise AssertionError(
                    "containment recheck must reject this load — got "
                    f"realpath {real}"
                )
            print("[LoRA] LoRA direct merge (weight=1.0): "
                  "applied=2, skipped=0")
            return True

        original_loader = mod.load_lora_with_key_fix
        original_pipe = mod._load_dry_load_pipeline
        original_unloader = mod.unload_adapters

        # Hook between scan and dry-load: monkey-patch _scan to do the
        # swap *after* it returns (i.e., simulate the attacker swapping
        # the file between scan and dry-load reopen).
        original_scan = mod._scan

        def scan_then_swap(audit_root, bases, warnings):
            files, exit_code = original_scan(audit_root, bases, warnings)
            # Replace second.safetensors with a symlink to `outside`.
            second = tree / "second.safetensors"
            second.unlink()
            second.symlink_to(outside)
            return files, exit_code

        mod._scan = scan_then_swap
        mod._load_dry_load_pipeline = lambda _d: object()
        mod.load_lora_with_key_fix = swapping_loader
        mod.unload_adapters = lambda _p, _names, **kw: None
        try:
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}",
                "--dry-load",
                "-o", str(tree / "manifest.json"),
            ])
        finally:
            mod._scan = original_scan
            mod._load_dry_load_pipeline = original_pipe
            mod.load_lora_with_key_fix = original_loader
            mod.unload_adapters = original_unloader
            outside.unlink(missing_ok=True)

        check("run completes despite mid-flight symlink swap",
              exit_code in (0, 2),
              detail=f"got {exit_code}")
        manifest = json.loads((tree / "manifest.json").read_text())
        swapped = next(
            (f for f in manifest["files"]
             if f["relative_path"] == "second.safetensors"),
            None,
        )
        check("swapped file present in manifest", swapped is not None)
        if swapped is not None:
            dl = (swapped.get("verdicts_by_base") or {}).get("klein", {}).get("dry_load")
            check("swapped file dry_load is recorded",
                  isinstance(dl, dict),
                  detail=f"dl={dl}")
            if isinstance(dl, dict):
                check("swapped file dry_load.loaded is False",
                      dl.get("loaded") is False,
                      detail=f"dl={dl}")
                check("swapped file dry_load.reason == 'containment_changed'",
                      dl.get("reason") == "containment_changed",
                      detail=f"dl={dl}")
                check("swapped file applied_modules is None",
                      dl.get("applied_modules") is None,
                      detail=f"dl={dl}")
        # Non-swapped file should still have dry-loaded normally.
        first = next(
            (f for f in manifest["files"]
             if f["relative_path"] == "first.safetensors"),
            None,
        )
        if first is not None:
            dl = (first.get("verdicts_by_base") or {}).get("klein", {}).get("dry_load")
            check("non-swapped first.safetensors loaded normally",
                  isinstance(dl, dict) and dl.get("loaded") is True,
                  detail=f"dl={dl}")


# ── Driver ─────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 70)
    print(f"  test_lora_audit.py — S1+S2 of ADR-014")
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
        test_dry_load_disabled_default_no_pipe_import()
        test_dry_load_per_file_fault_isolation()
        test_dry_load_base_load_failure_falls_back()
        test_dry_load_vram_cascade_warning()
        test_manifest_schema_dry_load_subobject()
        test_dry_load_print_manifest_stdout_clean()
        test_dry_load_loaded_false_nulls_applied_modules()
        test_dry_load_containment_recheck()
    finally:
        teardown_fixtures()
    print("\n" + "─" * 70)
    print(f"  {passed} passed, {failed} failed")
    print("─" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
