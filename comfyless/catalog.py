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

  Step 1 (commit 049f7f7) — scan-time `scan_model_family` helper.

  Step 2 (this commit) — catalog data structure (`CatalogEntry`,
    `CatalogDict`), `CatalogBuildError`, `normalize_name` (Unicode NFC),
    the scan walker (no-follow-symlinks; the kind-dispatch rule from
    Vision invariant 17), manifest parser + validator, `build_catalog()`
    orchestrator with fail-closed-at-startup on every ambiguity (Vision
    invariants 5–7, 15–17). Case-insensitive name-collision rejection
    enforced at insert time. HF-cache resolution for manifest HF-source
    entries with cache-miss → fail-startup (Vision invariant 6).

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
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterator, Literal, Optional, Tuple, TypedDict


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


# ════════════════════════════════════════════════════════════════════════
# Step 2 — Catalog data structure, name normalization, scan walker,
# manifest parser, and build_catalog() orchestrator.
# ADR-015 §1; Vision invariants 1–7, 15–17.
# ════════════════════════════════════════════════════════════════════════

# Hard cap on the operator manifest file size at read time. Manifests are
# operator-supplied (not agent-supplied), so the threat surface is lower
# than for model_index.json, but the same fail-closed posture applies: a
# multi-MB manifest indicates operator error or attack and should fail
# startup with a clear error, not stall. 1 MiB is ~5000 entries at
# realistic densities; plenty.
_MAX_MANIFEST_BYTES = 1024 * 1024  # 1 MiB

# Recognized catalog kinds (ADR-015 §1; Vision invariant 16). Slice 2's
# scan + manifest only mints entries of these three kinds. `vae` /
# `text_encoder` / `text_encoder_2` and cascade `scaffolding_repo` are
# deferred per the Vision Out of Scope section.
_KINDS = ("model", "lora", "transformer")

# Subdirectory basenames the scan walker classifies as containing
# loadable safetensors (Vision invariant 17). Match is byte-exact on
# the immediate parent directory's basename; operators with
# non-conventional layouts declare entries via the manifest. The
# convention is anchored to this codebase's actual ComfyUI directory
# layout (the 2026-05-23 Vision amendment).
_SCAN_DIR_TO_KIND = {
    "loras":            "lora",
    "checkpoints":      "transformer",
    "diffusion_models": "transformer",
}


class CatalogEntry(TypedDict):
    """One catalog entry. The single source of truth for a reference's
    server-side state.

    `abs_path` is server-side only and never crosses the MCP boundary
    (Vision invariant 9 — `list_models` / `list_loras` strip it before
    serializing in Step 4).

    `model_family` is populated by `scan_model_family` for scan-derived
    `kind:"model"` entries; manifest entries may declare it for any
    kind. `target_family` is manifest-only on `kind:"lora"` entries
    (Vision OQ2 resolution, invariant 9).
    """

    abs_path: str
    kind: Literal["model", "lora", "transformer"]
    source: Literal["scan", "manifest"]
    model_family: Optional[str]
    target_family: Optional[str]


# Public catalog type alias: `normalize_name(original) -> CatalogEntry`.
CatalogDict = Dict[str, CatalogEntry]


class CatalogBuildError(ValueError):
    """Raised by `build_catalog()` on any fail-closed startup condition.

    Subclass of `ValueError` so the Step-3 wiring in
    `comfyless/mcp_server.py` can adapt this to `click.BadParameter`
    without needing new exception-class plumbing. The message is
    operator-facing — it names the offending entry, manifest path, or
    repo ID — which is the operator startup channel, distinct from the
    agent-facing uniform-error contract that slice 3 lands when
    reference-resolution failures become reachable through `generate`.
    """


# Forbidden characters in catalog names. Step-2 INFO-2 forward-pointer
# resolution (slice 2 step 4 / 2026-05-25): catalog names flow into the
# agent-facing `list_*` JSON responses AND form the round-trip key the
# slice-3 reference resolution will look up. Rejecting these classes at
# build time means every consumer downstream is safe by construction
# (no per-tool output sanitizer; the catalog key, error messages, audit
# lines, and MCP-frame `name` field all agree on the same value).
#
# Rejected codepoint ranges (codepoints listed by hex only -- no literal
# bidi/zw characters appear in this source file, so bidi-detection
# scanners do not need to flag this module):
#   - U+0000..U+001F  C0 controls (NUL, TAB, LF, CR, ...)
#   - U+007F..U+009F  DEL + C1 controls
#   - U+200B..U+200F  zero-width chars + LRM/RLM
#   - U+202A..U+202E  bidi override formatting (LRE/RLE/PDF/LRO/RLO)
#   - U+2028..U+2029  LINE / PARAGRAPH SEPARATOR
#   - U+2066..U+2069  bidi isolate formatting (LRI/RLI/FSI/PDI)
#
# Pattern is a raw string so Python does not pre-decode the \uXXXX
# escapes; the `re` engine parses them itself. This keeps the source
# free of literal hostile characters that would trip bidi-detection
# tools (semgrep CWE-94 rule) on every edit.
#
# Threat model (same-uid trust): the operator can already plant arbitrary
# filenames under `--model-base` and arbitrary keys in the manifest. The
# reject-at-build-time policy is not an escalation defense; it is an
# agent-UX / round-trip-property contract. A scan-derived name whose
# filesystem basename contains a forbidden char fails startup naming the
# file (operator self-harm channel); a manifest entry with such a key
# fails likewise.
_FORBIDDEN_NAME_CHARS = re.compile(
    r"["
    r"\x00-\x1f\x7f-\x9f"
    r"\u200b-\u200f"
    r"\u202a-\u202e"
    r"\u2028-\u2029"
    r"\u2066-\u2069"
    r"]"
)


def normalize_name(s: str) -> str:
    """Unicode NFC-normalize a catalog name (Vision invariant 3).

    The catalog lookup key is the NFC form. Case-insensitive collision
    rejection (also invariant 3) is applied at insert time by
    `_add_entry`, not here. This function is the SAME helper that the
    future slice-3 request-side resolution will call on agent-supplied
    reference names, so catalog keys and request candidates cannot
    disagree on normalization.

    Pure normalizer: does NOT enforce the catalog-name character allowlist
    (`_FORBIDDEN_NAME_CHARS`). That gate lives in `_add_entry` so it can
    raise `CatalogBuildError` naming the offending original name; slice-3
    caller-supplied names will get their own request-time gate.
    """
    return unicodedata.normalize("NFC", s)


def _add_entry(
    catalog: CatalogDict,
    entry: CatalogEntry,
    original_name: str,
) -> None:
    """Add `entry` to `catalog` under `normalize_name(original_name)`,
    failing closed on every collision shape (Vision invariant 5).

    Collision rules:
      - Same normalized name AND same `abs_path` → harmless alias;
        the existing entry is retained (operator declared what the
        scan would have found anyway, or a duplicate scan path
        normalized to the same name).
      - Same normalized name AND different `abs_path` → fail closed
        naming the conflict (one of: scan-internal collision per
        invariant 5(a); manifest-shadows-distinct-scanned-path per
        invariant 5(b)).
      - Different normalized names that case-insensitively collide
        (via `str.casefold()`) → fail closed. Defeats case-folding
        host confusion (HFS+/APFS, NTFS) even though the lookup key
        itself is case-sensitive.

    On collision, raises `CatalogBuildError` with a message naming the
    offending name (operator-visible debugging info).

    Catalog-name character allowlist: any `original_name` containing a
    forbidden character (C0/C1 controls, zero-width chars, bidi
    overrides/isolates, LINE/PARAGRAPH SEPARATOR — see
    `_FORBIDDEN_NAME_CHARS`) raises `CatalogBuildError` at insert time.
    Centralizing the check here means scan-derived and manifest-derived
    names both go through the same gate, and downstream consumers
    (`list_models` / `list_loras` JSON, slice-3 reference lookup) can
    rely on every catalog key being agent-presentable plain text. Step-2
    INFO-2 forward-pointer resolution, slice 2 step 4 / 2026-05-25.
    """
    # Gate runs BEFORE normalize_name (i.e., on the original byte form
    # the operator supplied). This is safe because every codepoint in
    # `_FORBIDDEN_NAME_CHARS` is NFC-stable: a format/control character
    # with no canonical decomposition, and no other BMP codepoint
    # NFC-decomposes INTO one of these ranges. If a future addition to
    # `_FORBIDDEN_NAME_CHARS` is NOT NFC-stable (e.g. a precomposed
    # character that decomposes to a forbidden-range component), this
    # check must move to AFTER `normalize_name` or be applied to both
    # forms — otherwise the gate would silently allow the decomposed
    # form. (code-reviewer slice-2 step-4 LOW-1, folded 2026-05-25.)
    if _FORBIDDEN_NAME_CHARS.search(original_name):
        raise CatalogBuildError(
            f"catalog name {original_name!r} contains a forbidden "
            f"character (control / bidi-override / zero-width / "
            f"line-separator); names must be agent-presentable plain "
            f"text"
        )

    name_norm = normalize_name(original_name)

    if name_norm in catalog:
        existing = catalog[name_norm]
        if existing["abs_path"] == entry["abs_path"]:
            # Harmless alias / same-realpath; existing entry retained.
            return
        raise CatalogBuildError(
            f"catalog name {original_name!r} maps to two distinct paths "
            f"(existing source={existing['source']}; "
            f"new source={entry['source']})"
        )

    # Case-insensitive collision check across the catalog.
    name_cf = name_norm.casefold()
    for existing_name in catalog:
        if existing_name.casefold() == name_cf:
            # existing_name != name_norm (we already checked exact-match
            # above and returned/raised). So this is a case-only
            # variation under casefold equality — reject.
            raise CatalogBuildError(
                f"catalog name {original_name!r} case-insensitively "
                f"collides with existing entry {existing_name!r}"
            )

    catalog[name_norm] = entry


def _scan(model_base_real: str) -> Iterator[Tuple[CatalogEntry, str]]:
    """Walk `model_base_real` (which must already be `realpath`-resolved)
    and yield `(entry, original_name)` tuples for each scan hit
    (Vision invariants 4, 17).

    - `os.walk(..., followlinks=False)`: symlinked DIRECTORIES under
      `model_base` are not descended into.
    - For each directory visited:
        * If it contains a `model_index.json` (regular file or symlink-
          to-file — `os.path.isfile` follows file symlinks, and HF
          cache snapshot directories contain symlinked metadata files
          we want to recognize), mint a single `kind:"model"` entry
          for the directory and DO NOT descend further (its subdirs
          are component subdirs like `vae/`, `text_encoder/`, etc. —
          not independent catalog entries in slice 2). `model_family`
          is populated via `scan_model_family()`.
        * Otherwise, classify each `.safetensors` file in the directory
          by the IMMEDIATE PARENT DIRECTORY basename, byte-exact-
          matched against `_SCAN_DIR_TO_KIND`. Symlink `.safetensors`
          files are skipped per invariant 4 (if the realpath also lives
          under the scanned tree it gets its own entry naturally; if
          not, the operator declares it via manifest).
        * Files and directories that don't match the dispatch table
          are skipped.

    Names are emitted unnormalized; `_add_entry` does NFC + case-
    insensitive collision at insertion time. Scan-internal collisions
    (two distinct paths normalizing to the same name) surface there.
    """
    for dirpath, dirnames, filenames in os.walk(
        model_base_real, followlinks=False
    ):
        # Diffusers-pipeline directory (kind:"model") — mint one entry,
        # do not descend further.
        if "model_index.json" in filenames:
            real_dir = os.path.realpath(dirpath)
            family = scan_model_family(dirpath)
            entry: CatalogEntry = {
                "abs_path": real_dir,
                "kind": "model",
                "source": "scan",
                "model_family": family,
                "target_family": None,
            }
            yield entry, os.path.basename(dirpath)
            dirnames.clear()  # don't descend into the pipeline's components
            continue

        # Otherwise classify .safetensors files by parent dir basename.
        parent_basename = os.path.basename(dirpath)
        kind = _SCAN_DIR_TO_KIND.get(parent_basename)
        if kind is None:
            # Unconventional location; operator declares via manifest if
            # desired. (Vision N25: unconventional .safetensors files
            # are skipped, not minted.)
            continue
        for fname in filenames:
            if not fname.endswith(".safetensors"):
                continue
            file_path = os.path.join(dirpath, fname)
            # Skip file-symlinks per invariant 4 (Vision N28).
            if os.path.islink(file_path):
                continue
            real_file = os.path.realpath(file_path)
            stem = fname[: -len(".safetensors")]
            entry = {
                "abs_path": real_file,
                "kind": kind,
                "source": "scan",
                "model_family": None,
                "target_family": None,
            }
            yield entry, stem


def _parse_manifest_entry(
    name: str,
    raw: object,
    model_base_real: str,
) -> CatalogEntry:
    """Validate one manifest entry and return a `CatalogEntry`.

    Raises `CatalogBuildError` on any validation failure, naming the
    offending entry (operator-visible).

    Validates:
      - `raw` is a JSON object (dict).
      - Only allowed keys: `target`, `kind`, `model_family`,
        `target_family`. Extras → reject.
      - Required `target`: non-empty string.
      - Required `kind`: one of `_KINDS`.
      - Optional `model_family`: string if present.
      - Optional `target_family`: string if present AND `kind` must be
        `"lora"` (Vision invariant 9 / OQ2).

    Resolution:
      - HF repo ID `target` (`_is_hf_repo_id` returns True):
        `resolve_hf_path(target, allow_download=False)`. `ValueError`
        on local-cache miss → `CatalogBuildError` naming the entry
        AND the repo ID (Vision invariant 6).
      - Otherwise: filesystem path. `os.path.realpath` it, then
        `_within(--model-base)`-check. Outside-root → fail closed
        (Vision invariant 2).
    """
    if not isinstance(raw, dict):
        raise CatalogBuildError(
            f"manifest entry {name!r} must be a JSON object "
            f"(got {type(raw).__name__})"
        )
    allowed_keys = {"target", "kind", "model_family", "target_family"}
    extra = set(raw) - allowed_keys
    if extra:
        raise CatalogBuildError(
            f"manifest entry {name!r} has unknown keys: {sorted(extra)}"
        )
    target = raw.get("target")
    if not isinstance(target, str) or not target:
        raise CatalogBuildError(
            f"manifest entry {name!r} is missing required 'target' string"
        )
    # NUL-byte pre-check before any os.path.realpath() call. realpath
    # raises bare `ValueError: embedded null byte` (not OSError), which
    # would escape `_parse_manifest`'s catch and surface as an
    # unstructured server crash rather than a CatalogBuildError naming
    # the offending entry — degrading the operator startup channel and
    # breaking the CatalogBuildError-only contract that Step 3 wraps
    # into click.BadParameter. Matches the established project pattern
    # at comfyless/server.py:138-149 and comfyless/mcp_server.py:482-502.
    # (security-auditor slice-2 step-2 MEDIUM-1, folded 2026-05-24.)
    if "\x00" in target:
        raise CatalogBuildError(
            f"manifest entry {name!r}: 'target' contains a null byte"
        )
    kind = raw.get("kind")
    if kind not in _KINDS:
        raise CatalogBuildError(
            f"manifest entry {name!r} has invalid 'kind' {kind!r}; "
            f"must be one of {list(_KINDS)}"
        )
    model_family = raw.get("model_family")
    if model_family is not None and not isinstance(model_family, str):
        raise CatalogBuildError(
            f"manifest entry {name!r}: 'model_family' must be a string"
        )
    target_family = raw.get("target_family")
    if target_family is not None:
        if not isinstance(target_family, str):
            raise CatalogBuildError(
                f"manifest entry {name!r}: 'target_family' must be a string"
            )
        if kind != "lora":
            raise CatalogBuildError(
                f"manifest entry {name!r}: 'target_family' is only allowed "
                f"on kind:'lora' entries (got kind:{kind!r})"
            )

    # Lazy imports for the resolvers — `nodes.eric_diffusion_utils`
    # imports torch at top-level (see Step 1 HIGH-1 / 2026-05-23
    # security review).
    from nodes.eric_diffusion_utils import _is_hf_repo_id, resolve_hf_path

    if _is_hf_repo_id(target):
        try:
            abs_path = resolve_hf_path(target, allow_download=False)
        except ValueError:
            raise CatalogBuildError(
                f"manifest entry {name!r}: HF repo {target!r} is not "
                f"in the local HF cache (set up via "
                f"`huggingface-cli download {target}` first; the MCP "
                f"server does not perform downloads)"
            ) from None
    else:
        abs_path = target

    abs_path_real = os.path.realpath(abs_path)
    # Defense-in-depth: reuse the existing slice-1 `_within` helper
    # verbatim per ADR-015 §1 (containment under `--model-base`).
    from comfyless.server import _within
    if not _within(abs_path_real, model_base_real):
        raise CatalogBuildError(
            f"manifest entry {name!r}: target resolves outside "
            f"--model-base after realpath"
        )

    return {
        "abs_path": abs_path_real,
        "kind": kind,
        "source": "manifest",
        "model_family": model_family,
        "target_family": target_family,
    }


def _parse_manifest(
    catalog_path: str,
    model_base_real: str,
) -> Iterator[Tuple[CatalogEntry, str]]:
    """Read + validate an operator manifest; yield `(entry, name)` tuples.

    Raises `CatalogBuildError` on file-level failures (missing, not a
    regular file, oversized, malformed JSON, non-object root) and on
    any per-entry validation failure (via `_parse_manifest_entry`).
    Vision invariants 6, 7.
    """
    if not os.path.isfile(catalog_path):
        raise CatalogBuildError(
            f"--catalog {catalog_path!r} does not resolve to a regular file"
        )
    try:
        with open(catalog_path, "rb") as f:
            data = f.read(_MAX_MANIFEST_BYTES + 1)
    except OSError as e:
        raise CatalogBuildError(
            f"--catalog {catalog_path!r} could not be read: "
            f"{type(e).__name__}"
        ) from None
    if len(data) > _MAX_MANIFEST_BYTES:
        raise CatalogBuildError(
            f"--catalog {catalog_path!r} exceeds "
            f"{_MAX_MANIFEST_BYTES} bytes; manifests are operator-"
            f"supplied and should be small"
        )
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CatalogBuildError(
            f"--catalog {catalog_path!r} is not valid UTF-8 JSON: "
            f"{type(e).__name__}"
        ) from None
    if not isinstance(manifest, dict):
        raise CatalogBuildError(
            f"--catalog {catalog_path!r} top-level must be a JSON object "
            f"(got {type(manifest).__name__})"
        )

    for name, raw_entry in manifest.items():
        if not isinstance(name, str) or not name:
            # JSON keys are always strings in Python's parser, but guard
            # explicitly against empty-string names which would collide
            # on `normalize_name("")` and produce confusing errors later.
            raise CatalogBuildError(
                f"--catalog {catalog_path!r} has an empty-string entry name"
            )
        entry = _parse_manifest_entry(name, raw_entry, model_base_real)
        yield entry, name


def build_catalog(
    model_base: str,
    catalog_path: Optional[str] = None,
) -> CatalogDict:
    """Build the catalog from a scan of `model_base` plus an optional
    operator manifest.

    No-partial-catalog (Vision invariant 15): either the complete
    catalog builds cleanly and this function returns it, OR catalog
    build raises `CatalogBuildError` and the caller (Step 3 wiring)
    fails startup. There is no "scan only, ignore broken manifest"
    fallback and no "skip the broken entry, continue" silent drop.

    Order:
      1. Scan `model_base` (no-follow-symlinks; the kind-dispatch
         rule). Each scan hit goes through `_add_entry` so a scan-
         internal collision (Vision invariant 5(a)) fails closed.
      2. If `catalog_path` is provided, parse and validate the
         manifest. Each manifest entry goes through `_add_entry` so:
           - A manifest-vs-scan collision at the SAME realpath is a
             harmless alias and retained (Vision N12).
           - A manifest-vs-scan collision at DIFFERENT realpaths fails
             closed (Vision invariant 5(b) / N9).
           - Manifest HF-source entries with a local-cache miss fail
             closed at parse time (Vision invariant 6 / N7).
           - Case-insensitive name collisions across scan and manifest
             fail closed (Vision invariant 3 / N10).

    Returns a `CatalogDict` mapping `normalize_name(original)` →
    `CatalogEntry`.
    """
    model_base_real = os.path.realpath(model_base)
    catalog: CatalogDict = {}

    for entry, name in _scan(model_base_real):
        _add_entry(catalog, entry, name)

    if catalog_path is not None:
        for entry, name in _parse_manifest(catalog_path, model_base_real):
            _add_entry(catalog, entry, name)

    return catalog


# ---------------------------------------------------------------------------
# Request-time reference resolution (ADR-015 slice 3).
#
# `generate` (and future `extract_params` / `iterate`) hands an agent-supplied
# reference value to `resolve_reference`, which turns it into a server-side
# `abs_path` OR a failure cause. This is the request-time half of the catalog;
# `build_catalog` above is the spawn-time half. The two share `normalize_name`
# and `_FORBIDDEN_NAME_CHARS` so a request candidate and a catalog key cannot
# disagree on normalization (ADR-015 §2 step 1).
#
# This module NEVER renders an agent-facing string. It returns a structured
# `ResolveResult` whose `cause` (on failure) is OPERATOR-AUDIT ONLY. The
# handler in `comfyless/mcp_server.py` maps every failure cause onto the single
# uniform agent-facing error `"reference not available"` (ADR-015 §2 step 2,
# HIGH-1) and writes the fine-grained cause to the stderr audit line only
# (§2 step 2, the load-bearing oracle-closure commitment). Keeping the
# uniform-error rendering in ONE place (the handler) — not scattered across
# this resolver — is what makes the byte-identical-frame property auditable.
# ---------------------------------------------------------------------------

# Operator-audit-only failure causes. NEVER agent-facing — the agent frame is
# uniform across all of these (ADR-015 §2 step 2). Differences exist solely so
# the operator's stderr audit line can name what actually went wrong.
#
# Mapping to ADR-015 §2's enumerated causes:
#   - UnknownName        — normalized candidate not a catalog key (catalog miss).
#   - KindMismatch       — candidate IS a catalog key but the entry's kind is not
#                          the kind this field accepts (e.g. a lora name supplied
#                          as `model`). Folded into the uniform "not available"
#                          frame so the wrong-kind case cannot be distinguished
#                          from a miss — without this, a kind mismatch would fall
#                          through to `_load_pipeline` and surface as a DIFFERENT
#                          (InternalError) frame, a mild existence oracle. This
#                          STRENGTHENS HIGH-1 beyond the ADR's five-cause list
#                          (Vision slice-3 §"Open questions"; flagged for review).
#   - MalformedReference — empty after basename-strip, NUL byte, or a
#                          `_FORBIDDEN_NAME_CHARS` codepoint. (ADR-015 §2 step 2
#                          "malformed-reference rejection".)
#   - PathMoved          — catalog hit, but `abs_path` no longer exists at
#                          request time (a drive remount / move / delete between
#                          spawn and request — MEDIUM-1). Request-time HF-cache
#                          eviction also surfaces here: catalog entries store the
#                          already-resolved LOCAL cache path (build-time HF
#                          resolution, Vision invariant 6), so a post-spawn cache
#                          clear makes that local path vanish — indistinguishable
#                          from any other PathMoved, which is correct (the agent
#                          frame is uniform regardless). The ADR's separate
#                          request-time `HFCacheMiss` audit label is therefore
#                          subsumed by PathMoved: the catalog does not retain the
#                          originating repo ID, so request-time re-resolution
#                          would be a no-op on an absolute local path. (Vision
#                          slice-3 §"Open questions"; flagged for review.)
#   - WithinFailure      — catalog hit, `abs_path` exists, but fails the
#                          request-time `_within(--model-base)` re-check
#                          (defense-in-depth, ADR-015 §2 step 3 / MEDIUM-1).
ResolveCause = Literal[
    "UnknownName",
    "KindMismatch",
    "MalformedReference",
    "PathMoved",
    "WithinFailure",
]


@dataclass(frozen=True)
class ResolveResult:
    """Outcome of `resolve_reference`. Discriminated on `ok`.

    On success (`ok=True`): `abs_path` is the server-side load target,
    `name` is the agent-presentable resolved catalog name (NFC; what the
    response renders — never `abs_path`), `kind` is the entry kind, and
    `path_was_discarded` is True iff the input was path-shaped and its
    directory component was stripped (drives the §2 INFO path-discard
    notice — interpolating `name`, NEVER the agent-supplied raw value).

    On failure (`ok=False`): `cause` names the OPERATOR-AUDIT-ONLY reason;
    `abs_path` / `name` / `kind` are None. `path_was_discarded` is still
    set so the audit line can record that a path was supplied.
    """

    ok: bool
    abs_path: Optional[str] = None
    name: Optional[str] = None
    kind: Optional[str] = None
    path_was_discarded: bool = False
    cause: Optional[ResolveCause] = None


def resolve_reference(
    catalog: CatalogDict,
    raw_ref: str,
    model_base: str,
    *,
    expected_kind: Optional[str] = None,
) -> ResolveResult:
    """Resolve one agent-supplied reference value to a server-side path.

    ADR-015 §2:
      step 1 — basename-strip a path-shaped value, then NFC-normalize.
      step 2 — catalog lookup; any failure → a single uniform agent error
               (rendered by the HANDLER, not here; this returns the cause).
      step 3 — request-time `realpath`/`_within` fail-closed on a hit; never
               fall back to a stale catalog path.

    `expected_kind` (optional): when set, a catalog hit of a different kind
    returns `ok=False, cause="KindMismatch"` — folded into the uniform
    not-available outcome so wrong-kind cannot be distinguished from a miss.
    `None` accepts any kind.

    Pure with respect to its inputs except for the request-time filesystem
    stat (`os.path.exists`) and `_within`'s `realpath` — both read-only.
    Raises nothing for agent-supplied input: malformed values (including a
    NUL byte) become `cause="MalformedReference"`, not an exception.
    """
    # 1 — basename-strip (ADR-015 §2 step 1). Path-shaped == contains any
    # separator; reduce to basename and flag the discard. `os.sep` and "/"
    # both checked so a "\\"-style value on a POSIX host is still treated as
    # path-shaped only via os.sep (POSIX sep is "/"); the "/" check is the
    # operative one here and matches `os.path.basename` semantics.
    if not isinstance(raw_ref, str):
        return ResolveResult(ok=False, cause="MalformedReference")
    path_was_discarded = False
    candidate = raw_ref
    if "/" in raw_ref or os.sep in raw_ref:
        candidate = os.path.basename(raw_ref)
        path_was_discarded = True

    # 2 — malformed gate. Empty after basename-strip (e.g. "/foo/bar/"), a
    # NUL byte, or any `_FORBIDDEN_NAME_CHARS` codepoint → MalformedReference.
    # Runs on the candidate BEFORE normalize_name so a control/zero-width
    # char cannot smuggle through (the build-side gate in `_add_entry` is the
    # symmetric guarantee that catalog KEYS are clean).
    if (
        candidate == ""
        or "\x00" in candidate
        or _FORBIDDEN_NAME_CHARS.search(candidate) is not None
    ):
        return ResolveResult(
            ok=False, path_was_discarded=path_was_discarded,
            cause="MalformedReference",
        )

    # 3 — normalize + lookup.
    name_norm = normalize_name(candidate)
    entry = catalog.get(name_norm)
    if entry is None:
        return ResolveResult(
            ok=False, path_was_discarded=path_was_discarded,
            cause="UnknownName",
        )

    # 4 — kind enforcement (folds into not-available; no kind oracle).
    if expected_kind is not None and entry["kind"] != expected_kind:
        return ResolveResult(
            ok=False, path_was_discarded=path_was_discarded,
            cause="KindMismatch",
        )

    abs_path = entry["abs_path"]

    # 5 — request-time existence (PathMoved; also subsumes post-spawn HF-cache
    # eviction — see the ResolveCause docstring). `_within` below does its own
    # `realpath` but does NOT verify existence, so this check is required to
    # distinguish a moved/deleted target from an in-base one.
    if not os.path.exists(abs_path):
        return ResolveResult(
            ok=False, path_was_discarded=path_was_discarded,
            cause="PathMoved",
        )

    # 6 — request-time `_within` fail-closed (ADR-015 §2 step 3 / MEDIUM-1).
    # Defense-in-depth: the catalog SHOULD only hold in-base paths (build-time
    # check), but a remount/move between spawn and request can invalidate that.
    # Reuse the daemon's `_within` (lazy import; keeps this module's import-time
    # side effects stdlib-only, matching `_parse_manifest_entry`).
    from comfyless.server import _within
    if not _within(abs_path, model_base):
        return ResolveResult(
            ok=False, path_was_discarded=path_was_discarded,
            cause="WithinFailure",
        )

    return ResolveResult(
        ok=True,
        abs_path=abs_path,
        name=name_norm,
        kind=entry["kind"],
        path_was_discarded=path_was_discarded,
    )
