#!/usr/bin/env python3
"""End-to-end dry-load integration test for scripts/lora_audit.py (S2 of ADR-014).

Gated on LORA_AUDIT_DRY_LOAD_E2E=1 because it loads a real diffusers
pipeline and applies a real LoRA against it (GPU + filesystem heavy).
Default skip keeps the worktree's 8-suite regression cheap and offline.

Run:
  LORA_AUDIT_DRY_LOAD_E2E=1 \
  LORA_AUDIT_E2E_BASE=/path/to/diffusers_root \
  LORA_AUDIT_E2E_LORA=/path/to/lora.safetensors \
      ./.venv/bin/python3 test_dry_load_integration.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "lora_audit.py"


def _skip(msg: str) -> int:
    print(f"SKIP: {msg}")
    return 0


def main() -> int:
    if os.environ.get("LORA_AUDIT_DRY_LOAD_E2E", "") != "1":
        return _skip("set LORA_AUDIT_DRY_LOAD_E2E=1 to run")

    base_env = os.environ.get("LORA_AUDIT_E2E_BASE")
    lora_env = os.environ.get("LORA_AUDIT_E2E_LORA")
    if not base_env or not lora_env:
        return _skip("LORA_AUDIT_E2E_BASE and LORA_AUDIT_E2E_LORA must be set")

    base_root = Path(base_env).resolve()
    lora_src = Path(lora_env).resolve()
    if not base_root.is_dir():
        return _skip(f"base dir does not exist: {base_root}")
    if not lora_src.is_file():
        return _skip(f"lora file does not exist: {lora_src}")

    transformer_dir = base_root / "transformer"
    if not transformer_dir.is_dir():
        return _skip(
            f"base {base_root} has no transformer/ subdir; this test "
            f"expects a diffusers model root that lora_audit's "
            f"build_param_dict_from_dir can read."
        )

    with tempfile.TemporaryDirectory(prefix="lora_audit_e2e_") as tmp:
        tree = Path(tmp) / "tree"
        tree.mkdir()
        staged_lora = tree / lora_src.name
        shutil.copy2(lora_src, staged_lora)
        out_path = tree / "manifest.json"

        result = subprocess.run(
            [
                sys.executable, str(_SCRIPT_PATH),
                "--audit-root", str(tree),
                "--no-config",
                "--base", f"e2e={transformer_dir}",
                "--dry-load",
                "-o", str(out_path),
            ],
            capture_output=True, text=True, timeout=1800.0,
            cwd=str(_REPO_ROOT),
        )
        print(f"exit code: {result.returncode}")
        if result.stderr:
            print(f"stderr (last 1KB):\n{result.stderr[-1024:]}")
        if result.returncode not in (0, 2):
            print(f"FAIL: unexpected exit code {result.returncode}")
            return 1

        manifest = json.loads(out_path.read_text())
        e2e_base = manifest["bases"].get("e2e")
        if e2e_base is None:
            print(f"FAIL: bases.e2e missing in manifest")
            return 1
        if not e2e_base.get("dry_load_attempted"):
            print(f"FAIL: bases.e2e.dry_load_attempted is False; "
                  f"base load likely failed (check stderr above)")
            return 1

        target = [f for f in manifest["files"]
                  if f["relative_path"] == staged_lora.name]
        if not target:
            print(f"FAIL: manifest has no entry for {staged_lora.name}")
            return 1
        entry = target[0]
        verdict = (entry.get("verdicts_by_base") or {}).get("e2e", {})
        dry = verdict.get("dry_load")
        if not isinstance(dry, dict):
            print(f"FAIL: dry_load sub-object missing on e2e verdict: "
                  f"verdict={verdict}")
            return 1
        if set(dry.keys()) != {"loaded", "applied_modules", "reason"}:
            print(f"FAIL: dry_load shape wrong: {sorted(dry.keys())}")
            return 1
        print(f"PASS: dry_load={dry}")
        print(f"  bases.e2e.dry_load_attempted = True")
        print(f"  manifest written: {out_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
