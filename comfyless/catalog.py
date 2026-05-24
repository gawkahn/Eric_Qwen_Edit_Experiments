# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
comfyless/catalog.py — server-side reference catalog for the MCP surface.

This module is the security keystone for the MCP reference contract per
ADR-015. It will hold the spawn-time-built `name → abs_path` catalog that
every post-slice-2 MCP tool consults instead of accepting caller-supplied
paths. v1 (slice 2) is read-only after spawn; the catalog is built once and
held on `_StartupConfig.catalog` in `comfyless.mcp_server`.

The module is built up across the slice in four implementation steps:

  Step 1 (this commit) — scan-time `scan_model_family` helper only. The
    catalog data structure, name normalization, scan walker, manifest
    parser, and `build_catalog()` land in Step 2.

  Step 2 — catalog data structure, `normalize_name` (Unicode NFC +
    case-insensitive-collision-rejection), the scan walker
    (no-follow-symlinks; the kind-dispatch rule from Vision invariant 17),
    manifest parser + validator, `build_catalog()` orchestrator with
    fail-closed-at-startup on every ambiguity (Vision invariants 5–7,
    15–17). HF-cache resolution for manifest HF-source entries with
    cache-miss → fail-startup (Vision invariant 6).

  Step 3 — `_StartupConfig.catalog` field + `--catalog` click flag wired
    into `comfyless/mcp_server.py`. Nothing added to this module in
    Step 3.

  Step 4 — `list_models` / `list_loras` MCP tool handlers in
    `comfyless/mcp_server.py`. The catalog's outward surface; never
    serializes `abs_path`. Nothing added to this module in Step 4.

See `docs/decisions/ADR-015-mcp-catalog-reference-resolution.md` and
`docs/vision/slice-2-mcp-catalog.md`.
"""
from __future__ import annotations

import json
import os
from typing import Optional


# Hard cap on `model_index.json` size at read time. `model_index.json` in
# practice is a few KB; 1 MiB is generous. Caps spawn-time DoS risk before
# Step 2 fans this helper over every candidate directory under --model-base
# (security-auditor MEDIUM-1, slice 2 step 1, 2026-05-23). A file exceeding
# this cap is treated like every other malformed input — the helper returns
# `None` instead of attempting to parse it.
_MAX_INDEX_BYTES = 1024 * 1024  # 1 MiB


def scan_model_family(model_dir: str) -> Optional[str]:
    """Read `model_index.json` under `model_dir` and return its model_family.

    Scan-time companion to `nodes.eric_diffusion_utils.detect_pipeline_class`.
    `detect_pipeline_class` is the LOAD-time helper — it also instantiates
    the diffusers pipeline class and raises `ValueError` if the class is
    not importable in the running diffusers version. This helper is the
    SCAN-time companion: it returns the family string without requiring
    the diffusers pipeline class to be importable, so a model directory
    for a class the operator's diffusers install doesn't ship is still
    classifiable at scan time (the agent learns the model exists; whether
    the operator's diffusers can actually load it surfaces later when
    `generate` tries — via the existing slice-1 error path, unchanged in
    slice 2 per Vision invariant 14).

    Behavior on input that is not a usable diffusers directory: returns
    `None` rather than raising. The catalog scan walker (Step 2) uses
    this helper as the "is this a diffusers pipeline directory?"
    predicate against every directory it encounters; a permissive
    `None`-return keeps the scan loop simple and never causes a partial-
    catalog state to surface from a single bad file (Vision invariant 15
    requires no partial catalog ever runs, but build-failure-on-everything
    would be the wrong axis to enforce that — a bad `model_index.json`
    that's not even one we're supposed to find should not kill startup,
    while a *manifest* declaration referencing one should — that
    distinction lands in Step 2's `build_catalog()`).

    Returns `None` if any of:
      - `model_dir/model_index.json` does not exist or is not a regular file
      - the file exceeds `_MAX_INDEX_BYTES` (security-auditor MEDIUM-1)
      - the file is unreadable / malformed JSON / not UTF-8
      - the parsed top-level value is not a JSON object
      - the object lacks a non-empty string `_class_name` field
    """
    # Lazy import of `infer_model_family` to keep `comfyless.catalog`'s
    # module-import side effects to stdlib only — `nodes.eric_diffusion_utils`
    # imports torch at module top, which would otherwise be paid by every
    # consumer of this module regardless of whether they call this helper
    # (security-auditor HIGH-1, slice 2 step 1, 2026-05-23). Python caches
    # imports so subsequent calls within a process pay no extra cost.
    from nodes.eric_diffusion_utils import infer_model_family

    index_path = os.path.join(model_dir, "model_index.json")
    if not os.path.isfile(index_path):
        return None
    try:
        # Read in binary with an explicit size cap (MEDIUM-1) so a bloated
        # or hostile `model_index.json` cannot cause spawn-time OOM/stall
        # when Step 2 fans this over many directories. Decode with explicit
        # UTF-8 (MEDIUM-2) so the helper's "not UTF-8 → None" contract is
        # literally true regardless of host locale (`LANG=C`, misconfigured
        # systemd units, etc.).
        with open(index_path, "rb") as f:
            data = f.read(_MAX_INDEX_BYTES + 1)
        if len(data) > _MAX_INDEX_BYTES:
            return None
        index = json.loads(data.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(index, dict):
        return None
    class_name = index.get("_class_name")
    if not isinstance(class_name, str) or not class_name:
        return None
    return infer_model_family(class_name)
