"""ComfyUI stub modules for suites that import the node pack.

Until ADR-045 slice 5 these stubs lived in ``comfyless/__init__.py`` and were
installed as an import side effect, so any suite that did ``import comfyless``
before ``import nodes.*`` got them for free.  Slice 5 deleted them from the
runtime package (nothing under ``comfyless/`` has imported ``comfy.*`` or
``folder_paths`` since ADR-046), which left the test battery as their only
consumer — so they live here, on the node-pack side of the boundary, and are
installed explicitly by the suites that need them.

Explicit is the point: a suite that imports ``nodes.*`` now says so, instead of
depending on an unrelated import's side effect.  ``scripts/lora_audit.py``
already carried its own copy of this stub for the same reason (ADR §2, F-4).

Usage, before the first ``nodes.*`` / ``pipelines.*`` import::

    import comfy_stub; comfy_stub.install()

Idempotent, and never displaces a real ComfyUI that has already been
imported.  (A real ComfyUI that is importable but not yet imported IS masked —
the guard is ``sys.modules`` membership, exactly as the original shims were.)
"""

from __future__ import annotations

import sys
import types


def install() -> None:
    """Install no-op ``folder_paths`` / ``comfy`` stubs.  Idempotent."""

    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.get_folder_paths = lambda *_a, **_k: []
        fp.get_full_path = lambda *_a, **_k: None
        sys.modules["folder_paths"] = fp

    if "comfy" not in sys.modules:
        sys.modules["comfy"] = types.ModuleType("comfy")

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

    if "comfy.model_management" not in sys.modules:
        cmm = types.ModuleType("comfy.model_management")
        cmm.throw_exception_if_processing_interrupted = lambda: None
        sys.modules["comfy.model_management"] = cmm
        sys.modules["comfy"].model_management = cmm
