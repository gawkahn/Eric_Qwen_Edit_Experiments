# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""Catalog build orchestration (ADR-022 S1): scan → audit-manifest join → DB.

`build()` is the no-network ingest (Vision invariant 8): it reuses the
ADR-018 kind-typed scan (`comfyless.catalog.build_catalog`) so DB names are
the SAME names the serving path resolves — the DB mirrors the serving
namespace, never forks it (ADR-022 §1). Audit manifests (ADR-014 LoRA /
ADR-021 transformer) join by realpath and contribute classification
evidence. Family resolution / exclusion policy land in S2; enrichment
(network) is a separate explicit step (S3+).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from . import catalog_db


# ── Family resolution (ADR-022 §5) ──────────────────────────────────────
# Ordered hint rules → CANDIDATE family token. Pony / Illustrious / NoobAI
# are SDXL-architecture derivatives: they resolve to arch family "sdxl"
# (that's what the loader cares about); the finer ecosystem string is
# preserved verbatim in the sidecar description row's model_name /
# base_model text. Candidates are validated against the families table
# (exact, then prefix, then substring) so hint text can only ever map to a
# family that actually exists in hf-local.
_FAMILY_HINT_RULES: Tuple[Tuple[str, str], ...] = (
    ("flux2klein", "flux2klein"), ("klein", "flux2klein"),
    ("flux.2", "flux2"), ("flux 2", "flux2"), ("flux2", "flux2"),
    ("flux", "flux"),
    ("qwenedit", "qwen-edit"), ("qwen edit", "qwen-edit"),
    ("qwen-edit", "qwen-edit"),
    ("qwen", "qwen-image"),
    ("chroma", "chroma"),
    ("krea", "krea"),
    ("pony", "sdxl"), ("illustrious", "sdxl"), ("noobai", "sdxl"),
    ("sdxl", "sdxl"), ("sd xl", "sdxl"), ("xl 1.0", "sdxl"),
    ("sd 3", "sd3"), ("sd3", "sd3"),
    ("sd 1", "sd1"), ("sd1", "sd1"), ("sd 1.5", "sd1"),
    ("zimage", "zimage"), ("z-image", "zimage"), ("z image", "zimage"),
    ("auraflow", "auraflow"),
    ("wan", "wan"), ("hunyuan", "hunyuan"), ("cascade", "cascade"),
    ("hidream", "hidream"), ("ltx", "ltx"),
)


#: Candidates allowed a fuzzy containment tier: ONLY families that
#: infer_model_family produces as lowercased-classname FALLBACKS (no
#: pattern-table entry), so the candidate token can never be a prefix/
#: substring of a DIFFERENT pattern-table family. flux/flux2/sd1/sd3/etc.
#: are exact-membership only — code-review S2 finding 1: the old prefix
#: tier mapped candidate "flux" → known "flux2", silently un-excluding a
#: Flux.1 LoRA into flux2 recommendations.
_FUZZY_CANDIDATES = frozenset({"wan", "hunyuan", "cascade", "hidream",
                               "ltx"})


def family_from_hint(text: Optional[str],
                     known_families: Iterable[str]) -> Optional[str]:
    """Map free text (sidecar base_model, folder name) to a KNOWN family.

    Returns None when no rule fires or the candidate matches nothing in
    `known_families` — unknown hints never mint families (fail toward
    exclusion + loud family_conflict, per ADR-022 §5). Exact membership
    only for pattern-table families (finding 1); short tokens match on
    word boundaries so "wan" cannot fire inside "wandering" (finding 5).
    """
    if not text:
        return None
    low = str(text).lower()
    known = list(known_families)
    for needle, candidate in _FAMILY_HINT_RULES:
        if len(needle) <= 3:
            if not re.search(rf"\b{re.escape(needle)}\b", low):
                continue
        elif needle not in low:
            continue
        if candidate in known:
            return candidate
        if candidate in _FUZZY_CANDIDATES:
            sub = sorted(f for f in known if candidate in f)
            if sub:
                return sub[0]
        return None
    return None


def load_sidecar(lora_abs_path: str) -> Optional[Dict[str, Any]]:
    """Read the Lora Manager `<stem>.metadata.json` beside a LoRA, if any.

    Returns the parsed dict or None (missing / unreadable / not-a-dict).
    All text fields are sanitized downstream by upsert_description — this
    function only parses.
    """
    stem, _ = os.path.splitext(lora_abs_path)
    side = stem + ".metadata.json"
    try:
        with open(side, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_manifest_file(path: str) -> Dict[str, Any]:
    """Single fail-closed parse of a manifest file (review finding 6 —
    one read, one error semantics; corrupt manifest aborts the build,
    matching no-partial-catalog)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"audit manifest is not a JSON object: {path!r}")
    return data


def _extract_bases(data: Dict[str, Any]) -> Dict[str, str]:
    """{base name: base path} from a parsed manifest's `bases` section."""
    out = {}
    for bname, binfo in (data.get("bases") or {}).items():
        if isinstance(binfo, dict) and binfo.get("path"):
            out[str(bname)] = str(binfo["path"])
    return out


def load_audit_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    """Parse one lora_audit.json manifest → {realpath: audit info}.

    KIND-BRANCHING per ADR-021 §5 (security F-3): entries are consumed only
    for kinds this consumer understands ('lora', 'transformer'); unknown
    kinds are skipped, never guessed at. LoRA paths resolve against the
    manifest's `audit_root`; transformer paths against
    `transformer_roots[root_index]` (index identity per ADR-021 F-2).
    """
    return _index_from_parsed(_parse_manifest_file(path))


def _index_from_parsed(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    audit_root = data.get("audit_root") or ""
    transformer_roots = data.get("transformer_roots") or []
    out: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("files", []):
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        rel = entry.get("relative_path") or ""
        if kind == "lora":
            base = audit_root
        elif kind == "transformer":
            ri = entry.get("root_index", -1)
            base = (transformer_roots[ri]
                    if isinstance(ri, int) and 0 <= ri < len(transformer_roots)
                    else None)
        else:
            # Unknown kind → skip (ADR-021 F-3: consumers MUST branch on
            # kind and skip kinds they do not understand).
            continue
        if not base or not rel:
            continue
        abs_path = os.path.realpath(os.path.join(base, rel))
        out[abs_path] = {
            "kind": kind,
            "classification": entry.get("classification"),
            "reason": entry.get("reason"),
            "sha256": entry.get("sha256"),
            "size_bytes": entry.get("size_bytes"),
            "duplicate_of": entry.get("duplicate_of"),
            # Family evidence (ADR-022 §5 tier 1): which audit bases this
            # file positively matched.
            "ok_bases": (
                sorted(entry.get("matched_bases") or [])
                if kind == "transformer" else
                sorted(bn for bn, v in
                       (entry.get("verdicts_by_base") or {}).items()
                       if isinstance(v, dict)
                       and v.get("verdict") in ("OK", "NORM_TARGETING"))
            ),
        }
    return out




def build(db_path: str,
          model_base: str,
          *,
          lora_paths: Sequence[str] = (),
          transformer_paths: Sequence[str] = (),
          audit_manifests: Sequence[str] = (),
          catalog_path: Optional[str] = None,
          force_fs: bool = False) -> Dict[str, int]:
    """Build/refresh the catalog DB from a scan + optional audit manifests.

    No network (Vision invariant 8). Upsert semantics: refresh paths and
    audit evidence, preserve first_seen/enrichment, mark vanished entries
    stale (never DROP — invariant 12). Returns counts for the operator.

    Scan failures (CatalogBuildError: collisions, bad roots) propagate —
    no-partial-catalog applies to the DB build exactly as it does to the
    serving catalog.
    """
    # Lazy import: catalog.py pulls nothing heavy, but keeping the load-plane
    # module out of OUR import graph at module level mirrors the structural
    # separation (the serving side must never import catalog_db; see module
    # docstring — the reverse direction, builder→catalog, is read-only reuse).
    from comfyless.catalog import build_catalog

    scan = build_catalog(
        model_base, catalog_path,
        lora_paths=tuple(lora_paths),
        transformer_paths=tuple(transformer_paths),
    )

    audit: Dict[str, Dict[str, Any]] = {}
    manifest_bases: Dict[str, str] = {}
    for mpath in audit_manifests:
        # Single parse per manifest, fail-closed (review finding 6).
        data = _parse_manifest_file(mpath)
        audit.update(_index_from_parsed(data))
        manifest_bases.update(_extract_bases(data))

    # Family registry from the scan's MODEL entries (ADR-022 §4 step 2 —
    # infer_model_family already ran inside the scan). Base-name → family
    # resolves via the model dir realpath (bases point at the model root or
    # its transformer/ subdir).
    model_dir_to_family = {
        e["abs_path"]: e["model_family"]
        for e in scan.values()
        if e["kind"] == "model" and e.get("model_family")
    }
    base_family: Dict[str, str] = {}
    for bname, bpath in manifest_bases.items():
        rp = os.path.realpath(bpath)
        fam = (model_dir_to_family.get(rp)
               or model_dir_to_family.get(os.path.dirname(rp)))
        if fam:
            base_family[bname] = fam

    conn = catalog_db.connect(db_path, force_fs=force_fs)
    stats = {"entries": 0, "audited": 0, "stale": 0, "fts_rows": 0,
             "families": 0, "sidecars": 0, "excluded": 0}
    try:
        with conn:
            for name, e in sorted(scan.items()):
                if e["kind"] == "model" and e.get("model_family"):
                    catalog_db.upsert_family(
                        conn, name=e["model_family"],
                        hf_local_path=e["abs_path"])
                    stats["families"] += 1
            known_families = list(model_dir_to_family.values())

            seen = []
            for name in sorted(scan):
                e = scan[name]
                info = audit.get(e["abs_path"], {})
                size = info.get("size_bytes")
                if size is None and os.path.isfile(e["abs_path"]):
                    try:
                        size = os.path.getsize(e["abs_path"])
                    except OSError:
                        size = None
                sidecar = (load_sidecar(e["abs_path"])
                           if e["kind"] == "lora" else None)
                entry_id = catalog_db.upsert_entry(
                    conn,
                    name=name,
                    kind=e["kind"],
                    abs_path=e["abs_path"],
                    size_bytes=size,
                    sha256=info.get("sha256")
                    or (sidecar or {}).get("sha256"),
                    classification=info.get("classification"),
                    reason=info.get("reason"),
                    duplicate_of=info.get("duplicate_of"),
                )
                seen.append((e["kind"], name))
                stats["entries"] += 1
                if info:
                    stats["audited"] += 1

                if e["kind"] == "model":
                    catalog_db.set_entry_family(
                        conn, entry_id, e.get("model_family"))
                    continue

                # ── Family resolution, evidence precedence (§5):
                # audit > sidecar declaration > path hint. Disagreement →
                # highest wins, losers recorded in family_conflict.
                ev_audit = next(
                    (base_family[b] for b in info.get("ok_bases", [])
                     if b in base_family), None)
                ev_sidecar = family_from_hint(
                    (sidecar or {}).get("base_model"), known_families)
                ev_path = family_from_hint(
                    os.path.dirname(e["abs_path"]), known_families)
                family = ev_audit or ev_sidecar or ev_path
                conflict = None
                losers = [
                    f"{label}={val}" for label, val in
                    (("audit", ev_audit), ("sidecar", ev_sidecar),
                     ("path", ev_path))
                    if val is not None and val != family
                ]
                if losers:
                    conflict = f"chose {family}; disagreeing: " \
                               f"{', '.join(losers)}"
                catalog_db.set_entry_family(conn, entry_id, family, conflict)

                # ── Sidecar → tier-1 description row (§4 step 4).
                if sidecar:
                    civ = sidecar.get("civitai") or {}
                    model_block = civ.get("model") or {}
                    desc_text = (civ.get("description")
                                 or model_block.get("description") or "")
                    notes = sidecar.get("notes") or ""
                    catalog_db.upsert_description(
                        conn, entry_id=entry_id, source="sidecar",
                        model_name=(sidecar.get("model_name")
                                    or model_block.get("name")),
                        description=desc_text,
                        usage_tips=notes or None,
                        trigger_words=civ.get("trainedWords"),
                        # explicit None-check: a genuine SFW level 0 must
                        # not fall through to civitai's value
                        nsfw_level=(civ.get("nsfwLevel")
                                    if sidecar.get("preview_nsfw_level")
                                    is None
                                    else sidecar.get("preview_nsfw_level")),
                        civitai_model_id=civ.get("modelId"),
                        civitai_version_id=civ.get("id"),
                        provenance_url=(
                            f"https://civitai.com/models/{civ['modelId']}"
                            if civ.get("modelId") else None),
                    )
                    stats["sidecars"] += 1

            stats["stale"] = catalog_db.mark_stale_except(conn, seen)
            excl = catalog_db.apply_exclusions(conn)
            stats["excluded"] = excl["excluded"]
            stats["fts_rows"] = catalog_db.rebuild_fts(conn)
    finally:
        conn.close()
    return stats
