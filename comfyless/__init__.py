# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""
ComfyUI compatibility shims for comfyless operation.

This module MUST be imported before any ``nodes.*`` module.  It installs
stub modules for ``folder_paths``, ``comfy.utils``, and
``comfy.model_management`` so the existing node code can be imported
without a running ComfyUI instance.

The stubs are no-ops: progress bars do nothing, interrupt checks pass,
folder path lookups return empty/None.  All real paths are passed as
absolutes by the comfyless CLI.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _install_shims() -> None:
    """Install ComfyUI stub modules.  Idempotent."""

    # ── folder_paths ──────────────────────────────────────────────────
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.get_folder_paths = lambda *_a, **_k: []
        fp.get_full_path = lambda *_a, **_k: None
        sys.modules["folder_paths"] = fp

    # ── comfy namespace ───────────────────────────────────────────────
    if "comfy" not in sys.modules:
        sys.modules["comfy"] = types.ModuleType("comfy")

    # ── comfy.utils (ProgressBar) ─────────────────────────────────────
    if "comfy.utils" not in sys.modules:
        cu = types.ModuleType("comfy.utils")

        class _NoopProgressBar:
            def __init__(self, *_a, **_k):
                pass
            def update(self, *_a, **_k):
                pass
            def update_absolute(self, *_a, **_k):
                pass

        cu.ProgressBar = _NoopProgressBar
        sys.modules["comfy.utils"] = cu
        sys.modules["comfy"].utils = cu

    # ── comfy.model_management (interrupt check) ──────────────────────
    if "comfy.model_management" not in sys.modules:
        cmm = types.ModuleType("comfy.model_management")
        cmm.throw_exception_if_processing_interrupted = lambda: None
        sys.modules["comfy.model_management"] = cmm
        sys.modules["comfy"].model_management = cmm


_install_shims()
