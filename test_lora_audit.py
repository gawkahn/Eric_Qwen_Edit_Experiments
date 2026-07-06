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


@contextlib.contextmanager
def _temp_lora_tree():
    """A fresh, isolated copy of the standard fixture tree in a tempdir.

    Destructive --delete tests MUST use this, never the shared `_TREE`: an
    actual unlink against `_TREE` would wipe fixture files mid-suite and
    corrupt every test scheduled after it. Tree contents (same as
    `_build_lora_tree`): deletable = {zero, truncated, garbage.pt};
    non-deletable = {usable, sub/unconvertable}.
    """
    tmp = Path(tempfile.mkdtemp(prefix="lora_audit_del_"))
    try:
        tree = tmp / "tree"
        _build_lora_tree(tree)
        yield tree
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
    # Destructive: --delete --yes deletes the 3 deletable files. Use an
    # isolated temp tree so the shared _TREE survives for later tests.
    with _temp_lora_tree() as tree:
        result = _run_subprocess([
            "--audit-root", str(tree), "--no-config",
            "--base", _base_arg(),
            "--delete", "--yes",
        ], stdin=subprocess.DEVNULL, timeout=15.0)
        check("--delete --yes completes non-interactively (no hang/EOFError)",
              result.returncode in (0, 1, 2),
              detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
        check("--delete --yes exits 0 (run completed cleanly)",
              result.returncode == 0,
              detail=f"got {result.returncode}")
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


# ── S3 convert-path fixtures + tests (ADR §8, §10) ─────────────────────
def _build_convert_fixture(root: Path) -> tuple[Path, Path]:
    """Build a 2-LoRA tree + synthetic base under *root*.

    Both LoRAs use keys that do NOT match the synthetic base, so they are
    classified non-usable and fall through to the convertable probe (where
    `find_matching_plan` is patched in the tests). One sits under a subdir to
    exercise relative-path preservation. Returns (tree_dir, base_dir).
    """
    tree = root / "tree"
    base = root / "synth" / "transformer"
    _build_synth_base(base)
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "sub").mkdir(exist_ok=True)
    safetensors.torch.save_file({
        "double_blocks.0.img_attn.qkv.lora_A.weight": torch.zeros(4, 16),
        "double_blocks.0.img_attn.qkv.lora_B.weight": torch.zeros(16, 4),
    }, str(tree / "alpha.safetensors"))
    safetensors.torch.save_file({
        "double_blocks.1.img_attn.qkv.lora_A.weight": torch.zeros(4, 16),
        "double_blocks.1.img_attn.qkv.lora_B.weight": torch.zeros(16, 4),
    }, str(tree / "sub" / "beta.safetensors"))
    return tree, base


def _fake_plan(*_a, **_k):
    """A ConversionPlan stand-in carrying just the attributes the audit tool
    reads (source_family, target_family). The real convert_state_dict is
    patched out, so its qkv_splits/model_signature are never consulted."""
    return types.SimpleNamespace(
        source_family="bfl_chroma",
        target_family="diffusers_chroma",
        model_signature="",
        qkv_splits=[],
    )


class _ConvertPatch:
    """Swap the module's convert surface (find_matching_plan,
    convert_state_dict) for fakes; restore on exit. Mirrors _DryLoadPatch."""

    def __init__(self, mod, *, planner=None, converter=None):
        self.mod = mod
        self.planner = planner
        self.converter = converter
        self._saved: dict = {}

    def __enter__(self):
        self._saved["plan"] = self.mod.find_matching_plan
        self._saved["conv"] = self.mod.convert_state_dict
        if self.planner is not None:
            self.mod.find_matching_plan = self.planner
        if self.converter is not None:
            self.mod.convert_state_dict = self.converter
        return self

    def __exit__(self, *exc):
        self.mod.find_matching_plan = self._saved["plan"]
        self.mod.convert_state_dict = self._saved["conv"]


def _file_entry(manifest: dict, rel: str) -> dict:
    return next(
        (f for f in manifest["files"] if f["relative_path"] == rel), None
    )


def test_convert_writes_sibling() -> None:
    print("\n[20] test_convert_writes_sibling (S3 ADR §8)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="convert_ok_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))
        src_a = (tree / "alpha.safetensors").read_bytes()
        src_b = (tree / "sub" / "beta.safetensors").read_bytes()

        def good_convert(sd, plan, **kw):
            return {"converted.weight": torch.zeros(4, 4)}

        with _ConvertPatch(mod, planner=_fake_plan, converter=good_convert):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}", "--convert",
                "-o", str(tree / "manifest.json"),
            ])
        check("convert run exits cleanly", exit_code in (0, 2),
              detail=f"got {exit_code}")
        sib_a = tree / "alpha.diffusers_chroma.safetensors"
        sib_b = tree / "sub" / "beta.diffusers_chroma.safetensors"
        check("top-level sibling written", sib_a.is_file())
        check("subdir sibling written (relative path preserved)", sib_b.is_file())
        check("source alpha NOT modified",
              (tree / "alpha.safetensors").read_bytes() == src_a)
        check("source beta NOT modified",
              (tree / "sub" / "beta.safetensors").read_bytes() == src_b)
        manifest = json.loads((tree / "manifest.json").read_text())
        ea = _file_entry(manifest, "alpha.safetensors")
        eb = _file_entry(manifest, "sub/beta.safetensors")
        check("alpha manifest convert_output is relative sibling path",
              ea and ea["convert_output"] == "alpha.diffusers_chroma.safetensors",
              detail=f"got {ea and ea.get('convert_output')}")
        check("beta manifest convert_output preserves subdir",
              eb and eb["convert_output"] == "sub/beta.diffusers_chroma.safetensors",
              detail=f"got {eb and eb.get('convert_output')}")
        check("alpha convert_reason null on success",
              ea and ea["convert_reason"] is None)
        check("no stray .tmp left behind",
              not any(p.name.endswith(".tmp") for p in tree.rglob("*")))


def test_convert_collision_skipped() -> None:
    print("\n[21] test_convert_collision_skipped (S3 ADR §8)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="convert_coll_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))
        # Pre-create the target sibling for alpha with sentinel content.
        collide = tree / "alpha.diffusers_chroma.safetensors"
        collide.write_bytes(b"PRE-EXISTING-DO-NOT-CLOBBER")
        pre = collide.read_bytes()

        def good_convert(sd, plan, **kw):
            return {"converted.weight": torch.zeros(4, 4)}

        err = io.StringIO()
        with _ConvertPatch(mod, planner=_fake_plan, converter=good_convert):
            with contextlib.redirect_stderr(err):
                exit_code = mod.main([
                    "--audit-root", str(tree), "--no-config",
                    "--base", f"klein={base}", "--convert",
                    "-o", str(tree / "manifest.json"),
                ])
        check("convert run exits cleanly", exit_code in (0, 2),
              detail=f"got {exit_code}")
        check("collision target NOT overwritten",
              collide.read_bytes() == pre)
        check("WARN convert_skipped_collision emitted",
              "convert_skipped_collision" in err.getvalue(),
              detail=err.getvalue()[:300])
        manifest = json.loads((tree / "manifest.json").read_text())
        ea = _file_entry(manifest, "alpha.safetensors")
        check("alpha convert_reason == collision",
              ea and ea["convert_reason"] == "collision",
              detail=f"got {ea and ea.get('convert_reason')}")
        check("alpha convert_output null on collision",
              ea and ea["convert_output"] is None)
        # beta has no pre-existing collision → should still be written
        eb = _file_entry(manifest, "sub/beta.safetensors")
        check("non-colliding beta still converted",
              eb and eb["convert_output"] == "sub/beta.diffusers_chroma.safetensors")


def test_convert_atomic_tmp_cleanup() -> None:
    print("\n[22] test_convert_atomic_tmp_cleanup (S3 ADR §10)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="convert_tmp_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))

        def good_convert(sd, plan, **kw):
            return {"converted.weight": torch.zeros(4, 4)}

        # Break os.replace ONLY for our convert targets so the manifest's own
        # atomic write (lora_audit.json/.json.tmp) is unaffected.
        real_replace = os.replace

        def flaky_replace(src, dst, *a, **k):
            if str(dst).endswith(".diffusers_chroma.safetensors"):
                raise OSError("simulated disk-full during replace")
            return real_replace(src, dst, *a, **k)

        os.replace = flaky_replace
        try:
            with _ConvertPatch(mod, planner=_fake_plan, converter=good_convert):
                exit_code = mod.main([
                    "--audit-root", str(tree), "--no-config",
                    "--base", f"klein={base}", "--convert",
                    "-o", str(tree / "manifest.json"),
                ])
        finally:
            os.replace = real_replace

        check("convert run exits cleanly despite write failures",
              exit_code in (0, 2), detail=f"got {exit_code}")
        check("no target sibling left after failed replace",
              not (tree / "alpha.diffusers_chroma.safetensors").exists()
              and not (tree / "sub" / "beta.diffusers_chroma.safetensors").exists())
        check("no orphan .tmp left after cleanup",
              not any(p.name.endswith(".tmp") for p in tree.rglob("*")))
        check("source alpha still present and intact",
              (tree / "alpha.safetensors").is_file())
        manifest = json.loads((tree / "manifest.json").read_text())
        ea = _file_entry(manifest, "alpha.safetensors")
        check("alpha convert_reason == convert_failed",
              ea and ea["convert_reason"] == "convert_failed",
              detail=f"got {ea and ea.get('convert_reason')}")


def test_convert_per_file_fault_isolation() -> None:
    print("\n[23] test_convert_per_file_fault_isolation (Vision invariant 9)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="convert_iso_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))

        def half_convert(sd, plan, **kw):
            # alpha carries double_blocks.0 keys → raise; beta succeeds.
            if any("double_blocks.0" in k for k in sd):
                raise ValueError("boom on alpha")
            return {"converted.weight": torch.zeros(4, 4)}

        with _ConvertPatch(mod, planner=_fake_plan, converter=half_convert):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}", "--convert",
                "-o", str(tree / "manifest.json"),
            ])
        check("run completes despite one conversion raising",
              exit_code in (0, 2), detail=f"got {exit_code}")
        manifest = json.loads((tree / "manifest.json").read_text())
        ea = _file_entry(manifest, "alpha.safetensors")
        eb = _file_entry(manifest, "sub/beta.safetensors")
        check("failing alpha → convert_failed, no output",
              ea and ea["convert_reason"] == "convert_failed"
              and ea["convert_output"] is None,
              detail=f"alpha={ea and (ea.get('convert_reason'), ea.get('convert_output'))}")
        check("succeeding beta → output set, reason null",
              eb and eb["convert_output"] == "sub/beta.diffusers_chroma.safetensors"
              and eb["convert_reason"] is None,
              detail=f"beta={eb and (eb.get('convert_reason'), eb.get('convert_output'))}")
        check("beta sibling actually on disk",
              (tree / "sub" / "beta.diffusers_chroma.safetensors").is_file())
        check("alpha sibling NOT on disk",
              not (tree / "alpha.diffusers_chroma.safetensors").exists())


def test_convert_output_dir_directory() -> None:
    print("\n[24] test_convert_output_dir_directory (S3 ADR §8)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="convert_outdir_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))
        out_dir = tree / "converted"  # inside audit-root → containment OK

        def good_convert(sd, plan, **kw):
            return {"converted.weight": torch.zeros(4, 4)}

        with _ConvertPatch(mod, planner=_fake_plan, converter=good_convert):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}", "--convert",
                "--output-dir", str(out_dir),
                "-o", str(tree / "manifest.json"),
            ])
        check("convert run exits cleanly", exit_code in (0, 2),
              detail=f"got {exit_code}")
        check("top-level target under output-dir",
              (out_dir / "alpha.diffusers_chroma.safetensors").is_file())
        check("subdir mirrored under output-dir",
              (out_dir / "sub" / "beta.diffusers_chroma.safetensors").is_file())
        check("no sibling written next to source",
              not (tree / "alpha.diffusers_chroma.safetensors").exists())
        manifest = json.loads((tree / "manifest.json").read_text())
        eb = _file_entry(manifest, "sub/beta.safetensors")
        check("manifest convert_output is the relative subpath (base-independent)",
              eb and eb["convert_output"] == "sub/beta.diffusers_chroma.safetensors",
              detail=f"got {eb and eb.get('convert_output')}")


def test_convert_no_write_when_not_convertable() -> None:
    print("\n[25] test_convert_no_write_when_not_convertable (S3 guard)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="convert_noop_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))

        def never(*_a, **_k):
            raise AssertionError(
                "convert_state_dict must not be called for non-convertable files"
            )

        # Planner returns None → both LoRAs classify unconvertable; the convert
        # loop must skip them entirely (converter is the tripwire).
        with _ConvertPatch(mod, planner=lambda *a, **k: None, converter=never):
            exit_code = mod.main([
                "--audit-root", str(tree), "--no-config",
                "--base", f"klein={base}", "--convert",
                "-o", str(tree / "manifest.json"),
            ])
        check("run exits cleanly with nothing to convert",
              exit_code in (0, 2), detail=f"got {exit_code}")
        check("no converted siblings written",
              not any(".diffusers_chroma." in p.name for p in tree.rglob("*")))
        manifest = json.loads((tree / "manifest.json").read_text())
        all_null = all(
            f["convert_output"] is None and f["convert_reason"] is None
            for f in manifest["files"]
        )
        check("every entry has null convert_output and convert_reason", all_null)


def test_convert_print_manifest_stdout_clean() -> None:
    print("\n[26] test_convert_print_manifest_stdout_clean (S3 stdout contract)")
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="convert_stdout_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))

        def noisy_convert(sd, plan, **kw):
            # Real convert_state_dict prints [LoRA-Convert] chatter; simulate
            # that the convert loop must not let it pollute JSON-only stdout.
            print("noisy chatter from convert_state_dict that must be captured")
            return {"converted.weight": torch.zeros(4, 4)}

        out = io.StringIO()
        with _ConvertPatch(mod, planner=_fake_plan, converter=noisy_convert):
            with contextlib.redirect_stdout(out):
                exit_code = mod.main([
                    "--audit-root", str(tree), "--no-config",
                    "--base", f"klein={base}", "--convert", "--print-manifest",
                ])
        check("convert --print-manifest exits cleanly",
              exit_code in (0, 2), detail=f"got {exit_code}")
        captured = out.getvalue()
        check("no converter chatter leaked to stdout",
              "noisy chatter" not in captured)
        # stdout must be exactly one JSON document.
        try:
            parsed = json.loads(captured)
            ok = isinstance(parsed, dict) and "files" in parsed
        except json.JSONDecodeError:
            ok = False
        check("stdout is a single clean manifest JSON document", ok,
              detail=f"first 120 chars: {captured[:120]!r}")


def test_convert_output_dir_traversal_rejected() -> None:
    print("\n[27] test_convert_output_dir_traversal_rejected (S3 ADR §15)")
    # ADR §15 row S3 names "output-dir traversal rejection" as a deliverable.
    # An --output-dir outside --audit-root with neither escape flag must be
    # rejected at startup (exit 1) BEFORE any conversion writes occur.
    with tempfile.TemporaryDirectory(prefix="convert_trav_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))
        outside = Path(tmp) / "outside_root"
        outside.mkdir()
        result = _run_subprocess([
            "--audit-root", str(tree), "--no-config",
            "--base", f"klein={base}", "--convert",
            "--output-dir", str(outside),
        ])
        check("exit code 1 (startup rejection)", result.returncode == 1,
              detail=f"got {result.returncode}; stderr: {result.stderr[:200]}")
        check("stderr explains the output-dir containment failure",
              "output-dir" in result.stderr,
              detail=result.stderr[:200])
        check("no converted sibling written under the escape dir",
              not any(outside.rglob("*.safetensors")))
        check("no converted sibling written next to source either",
              not any(".diffusers_chroma." in p.name for p in tree.rglob("*")))


def test_convert_stale_tmp_surfaced_not_deleted() -> None:
    print("\n[28] test_convert_stale_tmp_surfaced_not_deleted (ADR §10 F-2)")
    # An interrupted convert leaves a *.safetensors.tmp orphan. The tool must
    # surface it as stale_tmp_file on the next scan and NEVER auto-delete it
    # (invariant 4; concurrent-invocation safety).
    mod = _import_script()
    with tempfile.TemporaryDirectory(prefix="convert_staletmp_") as tmp:
        tree, base = _build_convert_fixture(Path(tmp))
        orphan = tree / "sub" / "beta.diffusers_chroma.safetensors.tmp"
        orphan.write_bytes(b"interrupted convert payload")
        exit_code = mod.main([
            "--audit-root", str(tree), "--no-config",
            "--base", f"klein={base}",
            "-o", str(tree / "manifest.json"),
        ])
        check("scan exits cleanly with an orphan tmp present",
              exit_code in (0, 2), detail=f"got {exit_code}")
        check("orphan .tmp NOT deleted by the tool", orphan.is_file())
        manifest = json.loads((tree / "manifest.json").read_text())
        stale = [
            w for w in manifest["warnings"]
            if w["code"] == "stale_tmp_file"
            and "beta.diffusers_chroma.safetensors.tmp" in (w["file"] or "")
        ]
        check("orphan surfaced as stale_tmp_file warning", len(stale) == 1,
              detail=f"warnings: {manifest['warnings']}")


# ── S4 delete tests (ADR §9) ───────────────────────────────────────────
_DELETABLE = ("zero.safetensors", "truncated.safetensors", "garbage.pt")
_NON_DELETABLE = ("usable.safetensors", "sub/unconvertable.safetensors")


def test_delete_preview_no_io() -> None:
    print("\n[29] test_delete_preview_no_io (Vision #5, gate 3 preview)")
    with _temp_lora_tree() as tree:
        result = _run_subprocess([
            "--audit-root", str(tree), "--no-config", "--base", _base_arg(),
            "--delete",  # NO --yes
        ], stdin=subprocess.DEVNULL, timeout=15.0)
        check("preview exits 0", result.returncode == 0,
              detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
        for name in _DELETABLE:
            check(f"deletable file untouched in preview: {name}",
                  (tree / name).exists())
        check("preview prints 'would delete 3 files'",
              "would delete 3 files" in result.stderr,
              detail=result.stderr[:300])
        wd = [ln for ln in result.stderr.splitlines() if "would_delete:" in ln]
        check("preview lists each deletable path", len(wd) == 3,
              detail=f"{len(wd)} would_delete lines")


def test_delete_executes_with_yes() -> None:
    print("\n[30] test_delete_executes_with_yes (gate 3 execute)")
    with _temp_lora_tree() as tree:
        result = _run_subprocess([
            "--audit-root", str(tree), "--no-config", "--base", _base_arg(),
            "--delete", "--yes", "--print-manifest",
        ], stdin=subprocess.DEVNULL, timeout=15.0)
        check("execute exits 0", result.returncode == 0,
              detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
        for name in _DELETABLE:
            check(f"deletable file removed: {name}",
                  not (tree / name).exists())
        manifest = json.loads(result.stdout)
        by_path = {f["relative_path"]: f for f in manifest["files"]}
        for name in _DELETABLE:
            entry = by_path.get(name, {})
            check(f"manifest records deleted:true for {name}",
                  entry.get("deleted") is True and entry.get("delete_reason") is None,
                  detail=str(entry))


def test_delete_no_promotion() -> None:
    print("\n[31] test_delete_no_promotion (Vision #4, gate 1)")
    with _temp_lora_tree() as tree:
        result = _run_subprocess([
            "--audit-root", str(tree), "--no-config", "--base", _base_arg(),
            "--delete", "--yes", "--print-manifest",
        ], stdin=subprocess.DEVNULL, timeout=15.0)
        check("execute exits 0", result.returncode == 0,
              detail=f"got {result.returncode}; stderr: {result.stderr[:300]}")
        for name in _NON_DELETABLE:
            check(f"non-deletable file NOT removed even with --yes: {name}",
                  (tree / name).exists())
        manifest = json.loads(result.stdout)
        by_path = {f["relative_path"]: f for f in manifest["files"]}
        for name in _NON_DELETABLE:
            entry = by_path.get(name, {})
            check(f"manifest deleted:false for non-deletable {name}",
                  entry.get("deleted") is False, detail=str(entry))


def test_delete_reclassify_skip() -> None:
    print("\n[32] test_delete_reclassify_skip (ADR §9 F-5)")
    mod = _import_script()
    with _temp_lora_tree() as tree:
        # `usable.safetensors` is a valid LoRA — NOT deletable. Forge a
        # FileEntry that lies and claims it is deletable (simulating a
        # scan-time classification the file no longer matches at unlink time).
        target = tree / "usable.safetensors"
        entry = mod.FileEntry(
            relative_path="usable.safetensors",
            classification=mod.CLASS_DELETABLE,
            reason=mod.R_ZERO_BYTE,
        )
        warnings: list = []
        mod._run_delete(tree, [entry], warnings, confirmed=True)
        check("F-5: file with changed classification NOT deleted",
              target.exists())
        check("F-5: entry.deleted is False", entry.deleted is False)
        check("F-5: delete_reason == classification_changed",
              entry.delete_reason == mod.R_DELETE_CLASSIFICATION_CHANGED,
              detail=str(entry.delete_reason))
        check("F-5: emits delete_skipped_classification_changed warning",
              any(w.code == mod.W_DELETE_SKIPPED_RECLASSIFY for w in warnings))


def test_delete_containment_outside_root() -> None:
    print("\n[33] test_delete_containment_outside_root (ADR §9 gate 2)")
    mod = _import_script()
    with _temp_lora_tree() as tree:
        # A genuinely-deletable (zero-byte) file living OUTSIDE audit_root.
        # _safe_unlink's parent-containment gate must refuse to unlink it.
        outside = Path(tempfile.mkdtemp(prefix="lora_audit_outside_"))
        try:
            victim = outside / "zero.safetensors"
            victim.write_bytes(b"")
            check("precondition: outside file is deletable-signed",
                  mod._classify_deletable(victim, 0) == mod.R_ZERO_BYTE)
            warnings: list = []
            ok, reason = mod._safe_unlink(victim, tree, warnings)
            check("gate 2: unlink refused for parent outside audit_root",
                  ok is False)
            check("gate 2: reason == containment_failed",
                  reason == mod.R_DELETE_CONTAINMENT_FAILED,
                  detail=str(reason))
            check("gate 2: file outside root still exists", victim.exists())
        finally:
            shutil.rmtree(outside, ignore_errors=True)


def test_delete_per_file_fault_isolation() -> None:
    print("\n[34] test_delete_per_file_fault_isolation (Vision #9)")
    mod = _import_script()
    with _temp_lora_tree() as tree:
        # Two deletable entries; the first points at a now-missing path
        # (unlink will fail), the second is a real zero-byte file. The loop
        # must isolate the first failure and still delete the second.
        missing = mod.FileEntry(
            relative_path="gone.safetensors",
            classification=mod.CLASS_DELETABLE, reason=mod.R_ZERO_BYTE,
        )
        real = mod.FileEntry(
            relative_path="zero.safetensors",
            classification=mod.CLASS_DELETABLE, reason=mod.R_ZERO_BYTE,
        )
        warnings: list = []
        mod._run_delete(tree, [missing, real], warnings, confirmed=True)
        check("missing-path entry skipped, not deleted", missing.deleted is False)
        check("missing-path entry records delete_reason == unlink_failed",
              missing.delete_reason == mod.R_DELETE_UNLINK_FAILED,
              detail=str(missing.delete_reason))
        check("missing-path failure surfaces as a delete_failed manifest warning",
              any(w.code == mod.W_DELETE_FAILED for w in warnings))
        check("missing-path entry did not abort the loop; real file deleted",
              real.deleted is True and not (tree / "zero.safetensors").exists())


# ── Driver ─────────────────────────────────────────────────────────────
def test_transformer_audit() -> None:
    """ADR-021: kind:'transformer' entries — disjointness, prognosis
    mapping, shape matching, duplicate detection, report-only delete,
    manifest additivity + determinism."""
    print("\n[35] test_transformer_audit (ADR-021)")
    mod = _import_script()

    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        lora_root = td / "loras"
        troot = td / "checkpoints"
        troot2 = td / "checkpoints_old"  # sibling with shared name-prefix
        base_dir = td / "base" / "transformer"
        for d in (lora_root, troot, troot2, base_dir):
            d.mkdir(parents=True)

        # Base with 4 DISTINCT shapes so unique-(shape,dtype) pairing works.
        base_sd = {
            "transformer_blocks.0.q.weight": torch.arange(
                64 * 64, dtype=torch.float32).reshape(64, 64),
            "transformer_blocks.0.k.weight": torch.arange(
                32 * 64, dtype=torch.float32).reshape(32, 64),
            "transformer_blocks.0.v.weight": torch.arange(
                16 * 64, dtype=torch.float32).reshape(16, 64),
            "transformer_blocks.0.o.weight": torch.arange(
                8 * 64, dtype=torch.float32).reshape(8, 64),
        }
        safetensors.torch.save_file(base_sd, str(base_dir / "m.safetensors"))

        # (a) byte-duplicate under ComfyUI-style names (name-agnostic match)
        dup_sd = {f"model.diffusion_model.blk.{i}.weight": t.clone()
                  for i, t in enumerate(base_sd.values())}
        safetensors.torch.save_file(dup_sd, str(troot / "dup.safetensors"))
        # (b) same shapes, different content → finetune, not duplicate
        ft_sd = {k: v + 1.0 for k, v in dup_sd.items()}
        safetensors.torch.save_file(ft_sd, str(troot / "finetune.safetensors"))
        # (c) fp16 cast → shape-class match, dtype-exact pairing fails
        fp16_sd = {k: v.half() for k, v in dup_sd.items()}
        safetensors.torch.save_file(fp16_sd, str(troot / "fp16cast.safetensors"))
        # (d) bnb marker → quant_unsupported
        safetensors.torch.save_file(
            {"w.weight": torch.zeros(4, 4),
             "w.absmax": torch.zeros(4)}, str(troot / "bnb_nf4.safetensors"))
        # (e) DiT-family keys, non-matching shapes → no_matching_base
        safetensors.torch.save_file(
            {"transformer_blocks.0.x.weight": torch.zeros(7, 7)},
            str(troot / "nomatch.safetensors"))
        # (f) garbage under the transformer root → deletable, REPORT-ONLY
        (troot / "zero.safetensors").write_bytes(b"")
        # (g) second root: one valid file (root_index=1 + same-basename case)
        safetensors.torch.save_file(dup_sd, str(troot2 / "dup2.safetensors"))
        # lora root: one usable lora so the lora side is non-empty
        safetensors.torch.save_file({
            "transformer_blocks.0.q.lora_A.weight": torch.zeros(8, 64),
            "transformer_blocks.0.q.lora_B.weight": torch.zeros(64, 8),
        }, str(lora_root / "l.safetensors"))

        args = ["--audit-root", str(lora_root),
                "--transformer-root", str(troot),
                "--transformer-root", str(troot2),
                "--no-config", "--base", f"synth={base_dir}",
                "--print-manifest"]
        r = _run_subprocess(args)
        check("transformer run exits 0", r.returncode == 0,
              detail=r.stderr[-300:])
        m = json.loads(r.stdout)
        by_rel = {(e["kind"], e["relative_path"]): e for e in m["files"]}

        e = by_rel.get(("transformer", "dup.safetensors"))
        check("dup: usable via prognosis", e is not None
              and e["classification"] == "usable"
              and e["reason"].startswith("prognosis_"), detail=repr(e))
        check("dup: matched synth base", e and e["matched_bases"] == ["synth"])
        check("dup: duplicate_of == synth (byte-equal samples)",
              e and e["duplicate_of"] == "synth")
        check("dup: root_index 0 + display root", e
              and e["root_index"] == 0 and e["root"] == "checkpoints")

        e = by_rel.get(("transformer", "finetune.safetensors"))
        check("finetune: matched but NOT duplicate (content differs)",
              e and e["matched_bases"] == ["synth"]
              and e["duplicate_of"] is None, detail=repr(e))

        e = by_rel.get(("transformer", "fp16cast.safetensors"))
        check("fp16 cast: shape-class match, dtype-exact dup pairing fails "
              "(Vision neg-case 5)",
              e and e["matched_bases"] == ["synth"]
              and e["duplicate_of"] is None)

        e = by_rel.get(("transformer", "bnb_nf4.safetensors"))
        check("bnb: unconvertable quant_unsupported_bnb (Vision neg-case 3)",
              e and e["classification"] == "unconvertable"
              and e["reason"] == "quant_unsupported_bnb", detail=repr(e))

        e = by_rel.get(("transformer", "nomatch.safetensors"))
        check("no shape match: unconvertable no_matching_base",
              e and e["classification"] == "unconvertable"
              and e["reason"] == "no_matching_base")

        e = by_rel.get(("transformer", "zero.safetensors"))
        check("garbage transformer: deletable (report-only)",
              e and e["classification"] == "deletable"
              and e["reason"] == "zero_byte")
        check("transformer sha256 null (ADR-021 §5)",
              all(x["sha256"] is None for x in m["files"]
                  if x["kind"] == "transformer"))

        e = by_rel.get(("transformer", "dup2.safetensors"))
        check("second root: root_index 1", e and e["root_index"] == 1)
        check("manifest transformer_roots array (resolved, CLI order)",
              m.get("transformer_roots")
              == [str(troot.resolve()), str(troot2.resolve())])

        # additivity (F-3): lora entries carry NO transformer-only keys
        lora_entries = [x for x in m["files"] if x["kind"] == "lora"]
        check("lora entries unchanged (no root_index key — v1 shape)",
              lora_entries and all("root_index" not in x
                                   for x in lora_entries))
        # F-3 proof hooks: kind-filtering consumer sees exactly the v1 view;
        # naive consumer misparses (documented failure made visible)
        v1_view = [x for x in m["files"] if x["kind"] == "lora"]
        check("kind-filtering consumer sees only lora entries",
              all(x["kind"] == "lora" for x in v1_view) and v1_view)
        naive_usable = [x for x in m["files"]
                        if x["classification"] == "usable"]
        check("naive consumer WOULD ingest transformer rows as usable "
              "(documented misparse, F-3)",
              any(x["kind"] == "transformer" for x in naive_usable))

        # sort: all lora (-1) before transformer roots, roots in index order
        kinds_seq = [(x.get("root_index", -1) if x["kind"] == "transformer"
                      else -1) for x in m["files"]]
        check("files[] sorted by (root_index_or_-1, relative_path)",
              kinds_seq == sorted(kinds_seq))

        # determinism: second run byte-identical modulo audited_at
        r2 = _run_subprocess(args)
        m2 = json.loads(r2.stdout)
        for x in (m, m2):
            x.pop("audited_at", None)
        check("determinism: two runs identical modulo audited_at "
              "(Vision neg-case 7)", m == m2)

        # ── report-only delete (Vision neg-case 1): --delete --yes must
        # NOT unlink the garbage transformer file ──
        r3 = _run_subprocess(["--audit-root", str(lora_root),
                              "--transformer-root", str(troot),
                              "--no-config", "--base", f"synth={base_dir}",
                              "--delete", "--yes"])
        check("delete run exits 0", r3.returncode == 0,
              detail=r3.stderr[-200:])
        check("garbage transformer file SURVIVES --delete --yes "
              "(Vision invariants 7/11)",
              (troot / "zero.safetensors").exists())

        # ── disjointness startup aborts (F-1; Vision neg-case 8) ──
        for label, aroot, extra in (
            ("equal", troot, [str(troot)]),
            ("descendant", td, [str(troot)]),
            ("ancestor", troot, [str(td)]),
        ):
            r4 = _run_subprocess(["--audit-root", str(aroot),
                                  "--transformer-root", extra[0],
                                  "--no-config"])
            check(f"disjointness abort: transformer-root {label} of "
                  f"audit-root → exit 1", r4.returncode == 1,
                  detail=f"rc={r4.returncode} {r4.stderr[-150:]}")
        r4 = _run_subprocess(["--audit-root", str(lora_root),
                              "--transformer-root", str(troot),
                              "--transformer-root", str(troot / "sub_x"),
                              "--no-config"])
        check("pairwise overlap of transformer roots → exit 1",
              r4.returncode == 1)
        # NEW-1: sibling sharing a name-prefix is NOT nested — must run
        check("sibling name-prefix roots accepted (checkpoints vs "
              "checkpoints_old — NEW-1 component-boundary predicate)",
              mod._check_root_disjointness(
                  lora_root.resolve(),
                  [troot.resolve(), troot2.resolve()]) is None)

        # ── hostile-header sampler guards (F-4; Vision neg-case 10) ──
        hostile = td / "hostile.bin"
        hostile.write_bytes(b"\x00" * 64)
        check("F-4: hi < lo → None (no negative read)",
              mod._read_tensor_prefix(str(hostile), 8, 100, 50) is None)
        check("F-4: offsets past EOF → None",
              mod._read_tensor_prefix(str(hostile), 8, 10_000, 10_100)
              is None)
        check("F-4: hostile header_len → None",
              mod._read_tensor_prefix(str(hostile), 10**12, 0, 100) is None)

        # ── F-T1: crafted >100MB declared header → garbage verdict with
        # NO large read (the probe must cap before f.read(n)) ──
        crafted = troot / "hdrbomb.safetensors"
        with open(crafted, "wb") as f:
            f.write(struct.pack("<Q", 200_000_000))  # declares 200 MB hdr
            f.write(b"\x00" * 1024)                  # tiny actual body
        probe = mod._probe_safetensors_garbage(crafted, crafted.stat().st_size)
        check("F-T1: >100MB declared header → unparseable_header "
              "(capped, no unbounded read)",
              probe == mod.R_UNPARSEABLE_HEADER, detail=repr(probe))
        crafted.unlink()

        # ── review finding 9a: unreadable base shard → base excluded from
        # matching (partial |B| would inflate overlap), loud warning ──
        badbase = td / "badbase" / "transformer"
        badbase.mkdir(parents=True)
        safetensors.torch.save_file(
            {"transformer_blocks.0.q.weight": base_sd[
                "transformer_blocks.0.q.weight"].clone()},
            str(badbase / "a.safetensors"))
        with open(badbase / "b.safetensors", "wb") as f:  # corrupt shard
            f.write(struct.pack("<Q", 500))
            f.write(b"not json at all")
        r5 = _run_subprocess(["--audit-root", str(lora_root),
                              "--transformer-root", str(troot),
                              "--no-config", "--base", f"bad={badbase}",
                              "--print-manifest"])
        m5 = json.loads(r5.stdout)
        e = {(x["kind"], x["relative_path"]): x
             for x in m5["files"]}.get(("transformer", "dup.safetensors"))
        # (this fixture's keys give family hint "?", so the §2 precedence
        # yields format_unknown; the load-bearing assertions are
        # unconvertable + zero matches — NOT an inflated-overlap usable)
        check("9a: unreadable base shard → base excluded from matching "
              "(unconvertable/format_unknown, NOT inflated-overlap usable)",
              e is not None and e["classification"] == "unconvertable"
              and e["reason"] == "format_unknown"
              and e["matched_bases"] == [], detail=repr(e))
        check("9a: loud warning names the unreadable shard",
              any(w["code"] == "unreadable"
                  and "excluded from transformer matching" in w["detail"]
                  for w in m5["warnings"]), detail=repr(m5["warnings"])[:200])

        # ── review finding 9b: AIO bundle → dit-only matching, aio_bundle ──
        aio_sd = {f"model.diffusion_model.blk.{i}.weight": t.clone()
                  for i, t in enumerate(base_sd.values())}
        aio_sd["first_stage_model.decoder.conv.weight"] = torch.zeros(3, 3)
        aio_sd["cond_stage_model.emb.weight"] = torch.zeros(5, 5)
        safetensors.torch.save_file(aio_sd, str(troot / "aio.safetensors"))
        # ── review finding 9c: all-same-shape base → <2 unique pairs →
        # duplicate inconclusive + warning ──
        samebase = td / "samebase" / "transformer"
        samebase.mkdir(parents=True)
        same_sd = {f"transformer_blocks.{i}.w.weight":
                   torch.full((64, 64), float(i)) for i in range(4)}
        safetensors.torch.save_file(same_sd,
                                    str(samebase / "m.safetensors"))
        same_t = {f"model.diffusion_model.{i}.w.weight":
                  torch.full((64, 64), float(i)) for i in range(4)}
        safetensors.torch.save_file(same_t,
                                    str(troot / "sameshape.safetensors"))
        # ── review finding 9d: two roots with the SAME basename ──
        troot3 = td / "other" / "checkpoints"
        troot3.mkdir(parents=True)
        safetensors.torch.save_file(dup_sd,
                                    str(troot3 / "dup3.safetensors"))
        r6 = _run_subprocess(["--audit-root", str(lora_root),
                              "--transformer-root", str(troot),
                              "--transformer-root", str(troot3),
                              "--no-config",
                              "--base", f"synth={base_dir}",
                              "--base", f"same={samebase}",
                              "--print-manifest"])
        m6 = json.loads(r6.stdout)
        idx6 = {(x["kind"], x["relative_path"]): x for x in m6["files"]}
        e = idx6.get(("transformer", "aio.safetensors"))
        check("9b: AIO bundle → usable/aio_bundle via dit-only matching",
              e is not None and e["classification"] == "usable"
              and e["reason"] == "aio_bundle"
              and "synth" in (e["matched_bases"] or []), detail=repr(e))
        e = idx6.get(("transformer", "sameshape.safetensors"))
        check("9c: <2 unique-shape pairs → duplicate inconclusive "
              "(null + warning)",
              e is not None and e["duplicate_of"] is None
              and any(w["code"] == "dup_check_inconclusive"
                      for w in m6["warnings"]), detail=repr(e))
        e_dup = idx6.get(("transformer", "dup.safetensors"))
        e_dup3 = idx6.get(("transformer", "dup3.safetensors"))
        check("9d: same-basename roots → distinct root_index (F-2)",
              e_dup is not None and e_dup3 is not None
              and e_dup["root_index"] == 0 and e_dup3["root_index"] == 1
              and e_dup["root"] == e_dup3["root"] == "checkpoints")
        # cleanup extra fixtures so earlier-run manifests stay reproducible
        (troot / "aio.safetensors").unlink()
        (troot / "sameshape.safetensors").unlink()


def main() -> int:
    print("=" * 70)
    print(f"  test_lora_audit.py — S1+S2+S3+S4 of ADR-014")
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
        test_convert_writes_sibling()
        test_convert_collision_skipped()
        test_convert_atomic_tmp_cleanup()
        test_convert_per_file_fault_isolation()
        test_convert_output_dir_directory()
        test_convert_no_write_when_not_convertable()
        test_convert_print_manifest_stdout_clean()
        test_convert_output_dir_traversal_rejected()
        test_convert_stale_tmp_surfaced_not_deleted()
        test_delete_preview_no_io()
        test_delete_executes_with_yes()
        test_delete_no_promotion()
        test_delete_reclassify_skip()
        test_delete_containment_outside_root()
        test_delete_per_file_fault_isolation()
        test_transformer_audit()
    finally:
        teardown_fixtures()
    print("\n" + "─" * 70)
    print(f"  {passed} passed, {failed} failed")
    print("─" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
