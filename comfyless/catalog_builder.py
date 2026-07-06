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
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from . import catalog_db


def load_audit_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    """Parse one lora_audit.json manifest → {realpath: audit info}.

    KIND-BRANCHING per ADR-021 §5 (security F-3): entries are consumed only
    for kinds this consumer understands ('lora', 'transformer'); unknown
    kinds are skipped, never guessed at. LoRA paths resolve against the
    manifest's `audit_root`; transformer paths against
    `transformer_roots[root_index]` (index identity per ADR-021 F-2).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
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
    for mpath in audit_manifests:
        audit.update(load_audit_manifest(mpath))

    conn = catalog_db.connect(db_path, force_fs=force_fs)
    stats = {"entries": 0, "audited": 0, "stale": 0, "fts_rows": 0}
    try:
        with conn:
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
                catalog_db.upsert_entry(
                    conn,
                    name=name,
                    kind=e["kind"],
                    abs_path=e["abs_path"],
                    size_bytes=size,
                    sha256=info.get("sha256"),
                    classification=info.get("classification"),
                    reason=info.get("reason"),
                    duplicate_of=info.get("duplicate_of"),
                )
                seen.append((e["kind"], name))
                stats["entries"] += 1
                if info:
                    stats["audited"] += 1
            stats["stale"] = catalog_db.mark_stale_except(conn, seen)
            stats["fts_rows"] = catalog_db.rebuild_fts(conn)
    finally:
        conn.close()
    return stats
