# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""Catalog enrichment tier 2 — civitai SHA-256 hash lookup (ADR-022 §6).

THE ONLY NETWORK CODE IN THE CATALOG SERVICE. `build` is offline by
contract (Vision invariant 8); this module runs only under the explicit
`catalog_cli enrich` verb. Inference runtime (`local_files_only=True`)
is untouched — this is an offline batch step over the metadata DB.

Posture (ADR-022 §6):
- stdlib urllib only (no new dependency; §11 exact-pin discipline).
- Rate-limited (default 1 req/s), resumable (a recorded hit OR miss for
  (entry, civitai_api) is skipped on the next run unless --refresh).
- Per-entry fault isolation; N consecutive network failures abort the
  run as "network down" with partial stats (Vision neg-case 5).
- EVERY text field passes the catalog sanitizer via upsert_description —
  civitai text is untrusted web content headed for agent-visible fields.
- A definitive 404 on all hosts records a MISS row (description NULL,
  provenance_url = queried URL) so resume never re-queries known misses.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from . import catalog_db

#: Primary + mirror (post-split; project memory: civitai split, some
#: content only resolvable on one side).
CIVITAI_HOSTS: Tuple[str, ...] = ("https://civitai.com",
                                  "https://civitai.red")
USER_AGENT = "comfyless-catalog/0.1 (local desktop tool)"
_CONSECUTIVE_FAILURE_ABORT = 5


class EnrichError(Exception):
    """Operator-facing enrichment failure (network down, bad DB)."""


def _http_get_json(url: str, timeout: float = 15.0) -> Any:
    """GET → parsed JSON. Monkeypatch point for tests (no network in unit
    tests). Response size is bounded by a 4 MiB read cap — a hostile or
    misbehaving endpoint cannot balloon memory."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # URL is built from the hardcoded civitai API base + a quoted file hash —
    # never caller/model-controlled; response read is capped at 4 MiB.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        raw = resp.read(4 * 1024 * 1024)
    return json.loads(raw.decode("utf-8", "replace"))


def civitai_by_hash(sha256: str,
                    hosts: Tuple[str, ...] = CIVITAI_HOSTS,
                    ) -> Tuple[Optional[Dict[str, Any]], str]:
    """Look up a model version by file hash. Returns (payload|None, url).

    None = definitive miss (404). Host errors (5xx/429/timeouts) fall
    through to the mirror; if EVERY host errors, raises EnrichError so
    the caller can distinguish "not on civitai" from "couldn't ask".
    """
    sha = sha256.strip().lower()
    if not sha or any(c not in "0123456789abcdef" for c in sha):
        raise EnrichError(f"malformed sha256: {sha256!r}")
    last_err: Optional[Exception] = None
    last_url = ""
    saw_404 = False
    for host in hosts:
        url = f"{host}/api/v1/model-versions/by-hash/{sha}"
        last_url = url
        try:
            data = _http_get_json(url)
            if isinstance(data, dict):
                return data, url
            last_err = EnrichError(f"non-object response from {host}")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # NOT definitive yet — post-split, orphaned files exist
                # only on civitai.red (project_civitai_orphaned_files;
                # code-review S3 finding 1 HIGH). Try every host before
                # recording a persistent miss.
                saw_404 = True
                continue
            last_err = e
        except (urllib.error.URLError, TimeoutError, OSError,
                ValueError, RecursionError) as e:
            # ValueError covers JSONDecodeError; RecursionError covers a
            # deeply-nested JSON bomb under the 4 MiB cap (finding 7).
            last_err = e
    if saw_404 and last_err is None:
        return None, last_url  # 404 on ALL hosts — definitive miss
    raise EnrichError(f"all hosts failed for hash lookup "
                      f"({type(last_err).__name__}: {last_err}) "
                      f"last_url={last_url}")


def _int_or_none(v: Any) -> Optional[int]:
    """Coerce-or-drop at the trust boundary (security-audit S3 F-1):
    SQLite INTEGER affinity stores a non-numeric string VERBATIM —
    unsanitized, uncapped — so hostile payload values must never reach
    the bind as-is. bools rejected (JSON true would store as 1)."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _payload_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the description-row fields from a civitai model-version
    payload. Text sanitization happens inside upsert_description;
    integer-affinity fields are coerced-or-dropped HERE (F-1)."""
    model_block = payload.get("model") or {}
    return {
        "model_name": model_block.get("name") or payload.get("name"),
        "description": (payload.get("description")
                        or model_block.get("description")),
        "trigger_words": payload.get("trainedWords"),
        "nsfw_level": _int_or_none(payload.get("nsfwLevel")),
        "civitai_model_id": _int_or_none(payload.get("modelId")),
        "civitai_version_id": _int_or_none(payload.get("id")),
    }


def enrich(db_path: str = catalog_db.DEFAULT_DB_PATH, *,
           limit: Optional[int] = None,
           rate_s: float = 1.0,
           refresh: bool = False,
           include_excluded: bool = False,
           kinds: Tuple[str, ...] = ("lora",),
           force_fs: bool = False,
           _sleep=time.sleep) -> Dict[str, int]:
    """Tier-2 batch: civitai hash lookups for entries lacking a
    civitai_api description row. Returns stats; raises EnrichError only
    on total network failure (consecutive-failure abort) AFTER committing
    whatever completed — the run is resumable either way.
    """
    conn = catalog_db.connect(db_path, force_fs=force_fs)
    stats = {"queried": 0, "hits": 0, "misses": 0, "failures": 0,
             "skipped_existing": 0}
    consecutive = 0
    try:
        kind_ph = ",".join("?" * len(kinds))
        rows = conn.execute(
            f"""
            SELECT e.id, e.name, e.sha256,
                   (SELECT COUNT(*) FROM descriptions d
                     WHERE d.entry_id = e.id AND d.source = 'civitai_api')
                   AS have
            FROM entries e
            WHERE e.kind IN ({kind_ph}) AND e.sha256 IS NOT NULL
              AND e.stale = 0
              {"" if include_excluded else "AND e.excluded = 0"}
            ORDER BY e.name
            """, list(kinds)).fetchall()
        for r in rows:
            if limit is not None and stats["queried"] >= limit:
                break
            if r["have"] and not refresh:
                stats["skipped_existing"] += 1
                continue
            stats["queried"] += 1
            try:
                payload, url = civitai_by_hash(r["sha256"])
                consecutive = 0
            except EnrichError:
                stats["failures"] += 1
                consecutive += 1
                if consecutive >= _CONSECUTIVE_FAILURE_ABORT:
                    # FTS must include the hits committed BEFORE the abort
                    # (finding 5 — otherwise search misses them until the
                    # next completing run).
                    catalog_db.rebuild_fts(conn)
                    conn.commit()
                    raise EnrichError(
                        f"{consecutive} consecutive lookup failures — "
                        f"network down? Run is resumable; "
                        f"stats so far: {stats}")
                continue
            if payload is None:
                # Definitive miss: record the (entry, source) slot so
                # resume never re-queries; description stays NULL.
                catalog_db.upsert_description(
                    conn, entry_id=r["id"], source="civitai_api",
                    provenance_url=url)
                stats["misses"] += 1
            else:
                catalog_db.upsert_description(
                    conn, entry_id=r["id"], source="civitai_api",
                    provenance_url=url, **_payload_fields(payload))
                stats["hits"] += 1
            conn.commit()  # per-entry durability: kill-safe resume
            if rate_s > 0:
                _sleep(rate_s)
        catalog_db.rebuild_fts(conn)
        conn.commit()
    finally:
        conn.close()
    return stats


def hash_missing(db_path: str = catalog_db.DEFAULT_DB_PATH, *,
                 kinds: Tuple[str, ...] = ("lora",),
                 limit: Optional[int] = None,
                 force_fs: bool = False) -> Dict[str, int]:
    """Compute sha256 for entries that lack one (LoRAs by default;
    transformers stay opt-in per ADR-021 §5 — 2.2 TB). Local IO only."""
    import hashlib
    import os
    conn = catalog_db.connect(db_path, force_fs=force_fs)
    stats = {"hashed": 0, "errors": 0}
    try:
        kind_ph = ",".join("?" * len(kinds))
        rows = conn.execute(
            f"SELECT id, abs_path FROM entries WHERE kind IN ({kind_ph}) "
            f"AND sha256 IS NULL AND stale = 0 ORDER BY name",
            list(kinds)).fetchall()
        for r in rows:
            if limit is not None and stats["hashed"] >= limit:
                break
            try:
                h = hashlib.sha256()
                with open(r["abs_path"], "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                conn.execute("UPDATE entries SET sha256 = ? WHERE id = ?",
                             (h.hexdigest(), r["id"]))
                conn.commit()
                stats["hashed"] += 1
            except OSError:
                stats["errors"] += 1
    finally:
        conn.close()
    return stats
