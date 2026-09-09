# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""comfyless — headless diffusion runtime.

Historically this module installed ComfyUI compatibility shims (``folder_paths``,
``comfy.utils``, ``comfy.model_management``) and inserted the repository root
into ``sys.path``, because comfyless reached its model code by importing
``nodes.*``.  ADR-045 slices 1-4 moved that code into :mod:`comfyless.core`, and
since slice 3c (ADR-046) nothing under ``comfyless/`` imports ``nodes.*`` or
``comfy.*`` at all.  Both mechanisms were therefore deleted in slice 5 rather
than ported — the ``sys.path`` insert is actively wrong under any real install
(it resolves to ``src/`` from the working tree and to ``site-packages/`` from a
wheel), and the shims had no remaining runtime consumer.

The one residual reference is :func:`comfyless.core.eric_diffusion_utils.
resolve_component_path`, whose ``import folder_paths`` sits inside a
``try/except`` and falls through to returning the caller's path unchanged — the
same result the no-op stub produced.

Test suites that import the ComfyUI node pack install the stubs themselves; see
``comfy_stub.py`` at the repository root.
"""

from __future__ import annotations
