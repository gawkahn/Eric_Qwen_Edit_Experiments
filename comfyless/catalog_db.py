# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""SQLite metadata plane for the LoRA/transformer catalog (ADR-022).

THE DB IS NEVER THE LOAD PLANE (ADR-022 §1 / Vision invariant 7): generation
resolves names through the in-memory ADR-015/018 scan catalog + _check_paths
union. This module stores and serves METADATA only — audit classifications,
descriptions with provenance, trigger words, usage tips, search. A corrupt,
stale, or poisoned DB row can affect a *recommendation*, never which weights
load. Enforced structurally: `comfyless/generate.py` and `comfyless/server.py`
must never import this module (test_catalog_db.py asserts it by AST grep).

Storage constraints (ADR-022 §2):
- Default path ~/.local/share/comfyless/catalog.sqlite — MUST be off the
  mergerfs union (FUSE breaks the fcntl locks SQLite WAL needs; see the
  filesystem-constraint section of the global CLAUDE.md). `connect()` refuses
  a FUSE-resident path unless force_fs=True (warn-don't-block escape).
- WAL mode, foreign keys ON, PRAGMA user_version carries the schema version.

Sanitization (ADR-022 §6): every piece of externally-sourced text (civitai
HTML, web findings, sidecar notes) passes sanitize_text() before storage —
tags stripped, entities decoded, control/zero-width chars removed, length
capped. Stored text is DATA for agents, never instructions; provenance is
mandatory on every description row (CHECK constraint on source).
"""
from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

DEFAULT_DB_PATH = os.path.join(
    os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
    "comfyless", "catalog.sqlite",
)

#: Description provenance tiers, strongest-facts-first (ADR-022 §3/§6).
DESCRIPTION_SOURCES = ("sidecar", "civitai_api", "web", "ai_authored")

#: Text caps (ADR-022 §6).
DESCRIPTION_CAP = 4096
USAGE_TIPS_CAP = 2048
TRIGGER_WORD_CAP = 64
TRIGGER_WORDS_MAX = 64


class CatalogDBError(Exception):
    """Operator-facing catalog-DB failure (bad path, FUSE target, schema)."""


# ════════════════════════════════════════════════════════════════════════
#  Filesystem guard (Vision invariant 10)
# ════════════════════════════════════════════════════════════════════════

def _existing_ancestor(path: str) -> str:
    """Deepest existing ancestor of `path` (path itself if it exists)."""
    p = os.path.realpath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p or "/"


def fs_is_fuse(path: str) -> bool:
    """True if `path` (or its deepest existing ancestor) sits on a FUSE fs.

    Reads /proc/mounts and matches the longest mountpoint prefix; any fstype
    starting with "fuse" (fuse, fuseblk, fuse.mergerfs, …) counts. On systems
    without /proc/mounts this returns False — the guard is a Linux-desktop
    foot-gun-dampener, not a portability contract.
    """
    target = _existing_ancestor(path)
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            mounts = f.read().splitlines()
    except OSError:
        return False
    best_len = -1
    best_type = ""
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mountpoint = parts[1].replace("\\040", " ")
        fstype = parts[2]
        if target == mountpoint or target.startswith(
                mountpoint.rstrip("/") + os.sep) or mountpoint == "/":
            if len(mountpoint) > best_len:
                best_len = len(mountpoint)
                best_type = fstype
    return best_type.startswith("fuse")


# ════════════════════════════════════════════════════════════════════════
#  Sanitizer (ADR-022 §6)
# ════════════════════════════════════════════════════════════════════════

_TAG_RE = re.compile(r"<[^>]*>")
# Control chars (minus \n\t), zero-width + BOM + bidi-control codepoints —
# spelled as \u escapes, never literal invisibles in source (bidi-attack
# hygiene; semgrep generic.unicode.security.bidi).
_CTRL_ZW_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"  # C0 controls (newline/tab kept)
    "\u200b-\u200f"                    # zero-widths, LRM/RLM
    "\u202a-\u202e"                    # bidi embedding/override
    "\u2060-\u2069"                    # word-joiner, invisibles, isolates
    "\ufeff]"                          # BOM / ZWNBSP
)
_WS_RE = re.compile(r"[ \t]+")


def sanitize_text(raw: Any, cap: int = DESCRIPTION_CAP) -> str:
    """External text → bounded plain text (tags out, entities decoded,
    control/zero-width stripped, whitespace collapsed, NFC, capped).

    Strip-unescape-strip: html.unescape can *mint* new tags out of entities
    (&lt;script&gt; → <script>), so tags are stripped again after decoding.
    Content (including hostile-looking prose) is preserved as inert text —
    sanitization removes markup and invisibles, not meaning; the injection
    posture is provenance + data-framing, per ADR-022 §6.
    """
    if raw is None:
        return ""
    s = str(raw)
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = _CTRL_ZW_RE.sub("", s)
    s = unicodedata.normalize("NFC", s)
    s = _WS_RE.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s[:cap]


def sanitize_trigger_words(raw: Any) -> str:
    """Trigger-word list → JSON array of ≤64 sanitized words (≤64 B each)."""
    words: List[str] = []
    if isinstance(raw, (list, tuple)):
        for w in raw[:TRIGGER_WORDS_MAX]:
            clean = sanitize_text(w, cap=TRIGGER_WORD_CAP)
            if clean:
                words.append(clean)
    return json.dumps(words, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════════
#  Schema
# ════════════════════════════════════════════════════════════════════════

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('lora', 'transformer', 'model')),
    abs_path TEXT NOT NULL,
    root TEXT,
    relative_path TEXT,
    size_bytes INTEGER,
    sha256 TEXT,
    model_family TEXT,
    classification TEXT,
    reason TEXT,
    duplicate_of TEXT,
    excluded INTEGER NOT NULL DEFAULT 0,
    excluded_reason TEXT,
    stale INTEGER NOT NULL DEFAULT 0,
    family_conflict TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    UNIQUE (kind, name)
);
CREATE TABLE IF NOT EXISTS descriptions (
    id INTEGER PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    source TEXT NOT NULL
        CHECK (source IN ('sidecar', 'civitai_api', 'web', 'ai_authored')),
    model_name TEXT,
    description TEXT,
    usage_tips TEXT,
    trigger_words TEXT,
    strength_rec TEXT,
    sampler_rec TEXT,
    nsfw_level INTEGER,
    civitai_model_id INTEGER,
    civitai_version_id INTEGER,
    provenance_url TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE (entry_id, source)
);
CREATE TABLE IF NOT EXISTS families (
    name TEXT PRIMARY KEY,
    hf_local_path TEXT NOT NULL,
    model_index_class TEXT,
    is_diffusers INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS gen_tests (
    id INTEGER PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    prompt TEXT,
    negative_prompt TEXT,
    params_json TEXT,
    image_path TEXT,
    verdict TEXT,
    notes TEXT,
    tested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5(
    name, model_name, description, usage_tips, trigger_words,
    entry_id UNINDEXED
);
CREATE INDEX IF NOT EXISTS idx_entries_family ON entries(model_family);
CREATE INDEX IF NOT EXISTS idx_entries_excluded ON entries(excluded, stale);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: str = DEFAULT_DB_PATH, *,
            force_fs: bool = False) -> sqlite3.Connection:
    """Open (creating if needed) the catalog DB, enforcing the FUSE guard.

    Raises CatalogDBError when the target sits on a FUSE filesystem unless
    force_fs=True (the mergerfs union eats fcntl locks — SQLite there hangs;
    Vision invariant 10 / negative case 1).
    """
    if fs_is_fuse(db_path):
        if not force_fs:
            raise CatalogDBError(
                f"catalog DB path {db_path!r} resolves onto a FUSE filesystem "
                f"(mergerfs?) — SQLite locking hangs there. Choose an ext4 "
                f"location (default: {DEFAULT_DB_PATH}) or pass force_fs/"
                f"--force-fs to proceed anyway."
            )
        print(f"[catalog-db] WARNING: {db_path!r} is on a FUSE filesystem; "
              f"SQLite may hang on locks. Proceeding under --force-fs.",
              flush=True)
    os.makedirs(os.path.dirname(os.path.realpath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ver = conn.execute("PRAGMA user_version").fetchone()[0]
    if ver == 0:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    elif ver != SCHEMA_VERSION:
        conn.close()
        raise CatalogDBError(
            f"catalog DB schema version {ver} != supported {SCHEMA_VERSION} "
            f"({db_path!r}); migrate or rebuild."
        )
    return conn


# ════════════════════════════════════════════════════════════════════════
#  Upserts (Vision invariants 5, 12)
# ════════════════════════════════════════════════════════════════════════

def upsert_entry(conn: sqlite3.Connection, *, name: str, kind: str,
                 abs_path: str, root: Optional[str] = None,
                 relative_path: Optional[str] = None,
                 size_bytes: Optional[int] = None,
                 sha256: Optional[str] = None,
                 classification: Optional[str] = None,
                 reason: Optional[str] = None,
                 duplicate_of: Optional[str] = None,
                 now: Optional[str] = None) -> int:
    """Insert or refresh one (kind, name) entry; returns entry id.

    Refresh updates path/size/audit fields + last_seen and clears stale;
    first_seen, exclusion decisions, and enrichment rows are preserved
    (rebuild-never-loses-enrichment, Vision invariant 5/12). sha256,
    classification, reason, and duplicate_of are only overwritten by
    non-NULL values (audit evidence, once computed, is not forgotten
    because a later manifest-less build didn't supply it — code-review S1
    finding 1). size_bytes IS a straight overwrite: it's recomputed from
    the live file, not persistent evidence.
    """
    ts = now or _utcnow()
    cur = conn.execute(
        """
        INSERT INTO entries (name, kind, abs_path, root, relative_path,
                             size_bytes, sha256, classification, reason,
                             duplicate_of, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (kind, name) DO UPDATE SET
            abs_path = excluded.abs_path,
            root = excluded.root,
            relative_path = excluded.relative_path,
            size_bytes = excluded.size_bytes,
            sha256 = COALESCE(excluded.sha256, entries.sha256),
            classification = COALESCE(excluded.classification,
                                      entries.classification),
            reason = COALESCE(excluded.reason, entries.reason),
            duplicate_of = COALESCE(excluded.duplicate_of,
                                    entries.duplicate_of),
            stale = 0,
            last_seen = excluded.last_seen
        """,
        (name, kind, abs_path, root, relative_path, size_bytes, sha256,
         classification, reason, duplicate_of, ts, ts),
    )
    # lastrowid is unreliable across ON CONFLICT DO UPDATE — always SELECT
    # (code-review S1 finding 3).
    del cur
    row = conn.execute(
        "SELECT id FROM entries WHERE kind = ? AND name = ?",
        (kind, name)).fetchone()
    return int(row["id"])


def mark_stale_except(conn: sqlite3.Connection,
                      seen: Iterable[Tuple[str, str]]) -> int:
    """Mark entries NOT in `seen` [(kind, name), …] as stale (never DROP —
    Vision invariant 12). Returns count marked."""
    seen_set = set(seen)
    rows = conn.execute(
        "SELECT id, kind, name FROM entries WHERE stale = 0").fetchall()
    stale_ids = [r["id"] for r in rows if (r["kind"], r["name"]) not in seen_set]
    for chunk_start in range(0, len(stale_ids), 500):
        chunk = stale_ids[chunk_start:chunk_start + 500]
        conn.execute(
            f"UPDATE entries SET stale = 1 WHERE id IN "
            f"({','.join('?' * len(chunk))})", chunk)
    return len(stale_ids)


def upsert_description(conn: sqlite3.Connection, *, entry_id: int,
                       source: str,
                       model_name: Optional[str] = None,
                       description: Optional[str] = None,
                       usage_tips: Optional[str] = None,
                       trigger_words: Optional[Any] = None,
                       strength_rec: Optional[str] = None,
                       sampler_rec: Optional[str] = None,
                       nsfw_level: Optional[int] = None,
                       civitai_model_id: Optional[int] = None,
                       civitai_version_id: Optional[int] = None,
                       provenance_url: Optional[str] = None,
                       now: Optional[str] = None) -> None:
    """Insert or refresh the (entry, source) description row. ALL free text
    is sanitized here — callers cannot bypass the sanitizer (ADR-022 §6)."""
    if source not in DESCRIPTION_SOURCES:
        raise CatalogDBError(f"unknown description source {source!r}")
    conn.execute(
        """
        INSERT INTO descriptions (entry_id, source, model_name, description,
                                  usage_tips, trigger_words, strength_rec,
                                  sampler_rec, nsfw_level, civitai_model_id,
                                  civitai_version_id, provenance_url,
                                  fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (entry_id, source) DO UPDATE SET
            model_name = excluded.model_name,
            description = excluded.description,
            usage_tips = excluded.usage_tips,
            trigger_words = excluded.trigger_words,
            strength_rec = excluded.strength_rec,
            sampler_rec = excluded.sampler_rec,
            nsfw_level = excluded.nsfw_level,
            civitai_model_id = excluded.civitai_model_id,
            civitai_version_id = excluded.civitai_version_id,
            provenance_url = excluded.provenance_url,
            fetched_at = excluded.fetched_at
        """,
        (entry_id, source,
         sanitize_text(model_name, cap=256) or None,
         sanitize_text(description) or None,
         sanitize_text(usage_tips, cap=USAGE_TIPS_CAP) or None,
         sanitize_trigger_words(trigger_words),
         sanitize_text(strength_rec, cap=128) or None,
         sanitize_text(sampler_rec, cap=128) or None,
         nsfw_level, civitai_model_id, civitai_version_id,
         provenance_url, now or _utcnow()),
    )


# ════════════════════════════════════════════════════════════════════════
#  FTS (full rebuild — ~1-2k rows, milliseconds; ADR-022 §8)
# ════════════════════════════════════════════════════════════════════════

def rebuild_fts(conn: sqlite3.Connection) -> int:
    """Rebuild the FTS index from entries × best-available description rows.
    Every description source is indexed (an ai_authored tip is findable even
    when a sidecar description exists). Returns row count."""
    conn.execute("DELETE FROM catalog_fts")
    rows = conn.execute(
        """
        SELECT e.id AS entry_id, e.name AS name,
               d.model_name, d.description, d.usage_tips, d.trigger_words
        FROM entries e LEFT JOIN descriptions d ON d.entry_id = e.id
        """).fetchall()
    n = 0
    for r in rows:
        conn.execute(
            "INSERT INTO catalog_fts (name, model_name, description, "
            "usage_tips, trigger_words, entry_id) VALUES (?, ?, ?, ?, ?, ?)",
            (r["name"], r["model_name"] or "", r["description"] or "",
             r["usage_tips"] or "", r["trigger_words"] or "", r["entry_id"]))
        n += 1
    return n
