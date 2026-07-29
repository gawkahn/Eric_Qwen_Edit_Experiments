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

SCHEMA_VERSION = 2  # 2: ADR-041 slice 1 — instruction_template column +
#                        porter-stemmed FTS. Migrated in place from 1 by
#                        _migrate_v1_to_v2 (preserves civitai enrichment;
#                        FTS is derived and simply rebuilt).

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
# ADR-041 D2: instruction TEMPLATES are trained-word entries that are whole
# sentences, not tokens. They are the single best search material an edit
# LoRA has AND the phrasing it was trained on, so truncating them to 64 B
# cost twice. They get their own column and their own, larger bound.
TRIGGER_TEMPLATE_CAP = 512
_TEMPLATE_MIN_SPACES = 4
# Long trained-word strings come in two shapes, and only one is a template.
# Instruction PROSE ("head_swap: start with Picture 1 as the base image,
# keeping its lighting…") runs 0.00-0.18 commas per word; comma-delimited
# TAG LISTS ("single braid, completely nude, rock, brown eyes…") run
# 0.62-3.20. Measured on the live corpus 2026-07-29 — 8 prose against 4 tag
# lists, with a >3x gap and nothing in between. Tag soup must not land in a
# column carrying a 6.0 bm25 weight meant for functional description.
_TEMPLATE_MAX_COMMA_RATIO = 0.3

# ADR-041 D3: bm25 column weights, in catalog_fts declaration order —
# name, model_name, description, usage_tips, trigger_words,
# instruction_template. (entry_id is UNINDEXED and takes no weight.)
# LOWER bm25 output = better match, and weights multiply a column's
# contribution, so a HIGHER weight here means "matches in this column
# matter more". `description` is civitai marketing prose — the least
# functionally informative field in the row (ADR-041 Context) — so it is
# damped rather than dropped; `instruction_template` and `name` say what
# the LoRA actually DOES. Note this is a RANKING change only: the same
# rows match, in a better order.
_BM25_WEIGHTS = ", 8.0, 2.0, 0.5, 2.0, 3.0, 6.0"
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


def sanitize_url(raw: Any, cap: int = 2048) -> Optional[str]:
    """Provenance-URL cleaner (S4 security F-1 — the raw bind bypassed the
    every-field-sanitized contract): control/zero-width/bidi stripped,
    whitespace rejected inside the URL, length-capped, http(s) schemes
    ONLY (ADR-022 §6: provenance is URLs only — javascript:/data: etc.
    are dropped, returning None)."""
    if raw is None:
        return None
    s = _CTRL_ZW_RE.sub("", str(raw)).strip()
    s = s.split()[0] if s.split() else ""
    low = s.lower()
    if not (low.startswith("https://") or low.startswith("http://")):
        return None
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


def is_instruction_template(raw: Any) -> bool:
    """Is this trained-word entry a prose INSTRUCTION TEMPLATE rather than a
    trigger token? (ADR-041 D2)

    Three conditions, all required:
      * it would be truncated by `TRIGGER_WORD_CAP` — a real trigger token
        fits comfortably;
      * it carries at least `_TEMPLATE_MIN_SPACES` spaces — it is a sentence,
        not a multi-word tag; and
      * its comma-per-word ratio is at most `_TEMPLATE_MAX_COMMA_RATIO` — it
        is PROSE, not a comma-delimited tag list.

    Grounded in the live corpus (2026-07-29, 222 enriched entries): 395
    trained words are 1-15 chars, and the 28 sitting at exactly 60-64 are all
    long text ("head swap face from Image 1 to Image 2, keep all facial
    details …") carrying 11-14 spaces. Nothing lands between those
    populations. The comma gate then splits the long ones: 8 instruction
    prose at 0.00-0.18 commas/word from 4 tag lists at 0.62-3.20. Both
    discriminators have a wide margin; neither is a close call.
    """
    if not isinstance(raw, str):
        return False
    s = _CTRL_ZW_RE.sub("", raw).strip()
    if len(s) <= TRIGGER_WORD_CAP or s.count(" ") < _TEMPLATE_MIN_SPACES:
        return False
    words = len(s.split())
    return s.count(",") / max(1, words) <= _TEMPLATE_MAX_COMMA_RATIO


def extract_instruction_template(raw: Any) -> Optional[str]:
    """The single longest template-shaped trained word, sanitized and capped
    at `TRIGGER_TEMPLATE_CAP` (ADR-041 D2), or None.

    At most ONE per entry: this is the phrasing the LoRA was trained on, and
    a checkpoint with several is choosing between near-identical wordings —
    the longest is the most complete. Kept in its own column rather than
    widening `trigger_words`, so bm25 can weight it independently (D3) and
    so the one-per-entry rule is visible in the schema rather than implied.
    """
    if not isinstance(raw, (list, tuple)):
        return None
    best: Optional[str] = None
    for w in raw[:TRIGGER_WORDS_MAX]:
        if not is_instruction_template(w):
            continue
        clean = sanitize_text(w, cap=TRIGGER_TEMPLATE_CAP)
        if clean and (best is None or len(clean) > len(best)):
            best = clean
    return best


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
    instruction_template TEXT,
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
CREATE INDEX IF NOT EXISTS idx_entries_family ON entries(model_family);
CREATE INDEX IF NOT EXISTS idx_entries_excluded ON entries(excluded, stale);
"""

# Kept separate from _SCHEMA so the fresh-install path and the v1->v2
# migration create BYTE-IDENTICAL FTS tables. Two copies of this DDL is
# exactly how a migrated DB and a fresh one end up silently differing in
# tokenizer — which would make search results depend on install history.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5(
    name, model_name, description, usage_tips, trigger_words,
    instruction_template,
    entry_id UNINDEXED,
    tokenize='porter unicode61'
);
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
        conn.executescript(_FTS_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    elif ver == 1 and SCHEMA_VERSION == 2:
        # ADR-041 slice 1. Migrate in place rather than asking for a
        # rebuild: `descriptions` holds civitai enrichment that costs a
        # network round-trip per row to regenerate (310 rows at adoption).
        # The FTS table is pure derived data, so it is dropped and rebuilt
        # with the new column and the porter tokenizer.
        _migrate_v1_to_v2(conn)
    elif ver != SCHEMA_VERSION:
        conn.close()
        raise CatalogDBError(
            f"catalog DB schema version {ver} != supported {SCHEMA_VERSION} "
            f"({db_path!r}); migrate or rebuild."
        )
    return conn


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2: add `descriptions.instruction_template`, recreate the FTS
    table with that column and `tokenize='porter unicode61'` (ADR-041 D2/D3).

    Existing rows get a NULL template — the raw trained words were already
    truncated to 64 B on the way in, so the untruncated text is simply not
    in the DB. Re-running `catalog_cli build` repopulates it from the source
    metadata JSON. Callers that need the templates must rebuild; callers
    that only need search keep working immediately, on stemmed FTS.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(descriptions)")}
    if "instruction_template" not in cols:
        conn.execute(
            "ALTER TABLE descriptions ADD COLUMN instruction_template TEXT")
    conn.execute("DROP TABLE IF EXISTS catalog_fts")
    conn.executescript(_FTS_SCHEMA)
    # ORDER IS LOAD-BEARING: the version bump goes LAST, inside the same
    # transaction as the FTS rows.
    #
    # Under Python sqlite3's legacy transaction control, DDL and PRAGMA run
    # in autocommit and are durable the instant they execute, while DML only
    # opens an implicit transaction. Bumping user_version before rebuilding
    # therefore made v2 durable on its own: a crash — or any exception
    # inside rebuild_fts — between the two left a DB reading v2 with an
    # EMPTY catalog_fts. The migration would never re-run (version already
    # v2) and every search would silently degrade to name-LIKE forever.
    # `rebuild_fts`'s DELETE opens the transaction, PRAGMA joins it, and the
    # single commit below makes rows-plus-version atomic. Everything above
    # is idempotent (guarded ALTER, DROP IF EXISTS, CREATE IF NOT EXISTS),
    # so a crash before the commit leaves v1 and the migration retries clean.
    rebuild_fts(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    print("[catalog-db] migrated schema v1 -> v2 (ADR-041): added "
          "instruction_template, rebuilt FTS with porter stemming. "
          "Re-run `catalog_cli build` to populate templates.", flush=True)


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open an EXISTING catalog DB read-only (sqlite URI mode=ro) —
    the MCP-surface accessor (ADR-022 S5): a reader that structurally
    cannot write, never creates files/dirs, and fail-closes on a missing
    file or schema mismatch."""
    real = os.path.realpath(db_path)
    if not os.path.isfile(real):
        raise CatalogDBError(f"catalog DB not found: {db_path!r}")
    import urllib.parse
    uri = "file:" + urllib.parse.quote(real) + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        raise CatalogDBError(f"catalog DB unreadable: {e}") from None
    conn.row_factory = sqlite3.Row
    ver = conn.execute("PRAGMA user_version").fetchone()[0]
    if ver != SCHEMA_VERSION:
        conn.close()
        raise CatalogDBError(
            f"catalog DB schema version {ver} != supported "
            f"{SCHEMA_VERSION} ({db_path!r})")
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
                                  usage_tips, trigger_words,
                                  instruction_template, strength_rec,
                                  sampler_rec, nsfw_level, civitai_model_id,
                                  civitai_version_id, provenance_url,
                                  fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (entry_id, source) DO UPDATE SET
            model_name = excluded.model_name,
            description = excluded.description,
            usage_tips = excluded.usage_tips,
            trigger_words = excluded.trigger_words,
            instruction_template = excluded.instruction_template,
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
         # Derived from the SAME raw list, so a caller cannot supply a
         # template that disagrees with the trigger words it came from
         # (ADR-022 §6: sanitization is not bypassable at the call site).
         extract_instruction_template(trigger_words),
         sanitize_text(strength_rec, cap=128) or None,
         sanitize_text(sampler_rec, cap=128) or None,
         nsfw_level, civitai_model_id, civitai_version_id,
         sanitize_url(provenance_url), now or _utcnow()),
    )


def upsert_family(conn: sqlite3.Connection, *, name: str, hf_local_path: str,
                  model_index_class: Optional[str] = None,
                  is_diffusers: bool = True) -> None:
    """Register one model family (from the hf-local scan's model entries)."""
    conn.execute(
        """
        INSERT INTO families (name, hf_local_path, model_index_class,
                              is_diffusers)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (name) DO UPDATE SET
            hf_local_path = excluded.hf_local_path,
            model_index_class = excluded.model_index_class,
            is_diffusers = excluded.is_diffusers
        """,
        (name, hf_local_path, model_index_class, 1 if is_diffusers else 0))


def set_entry_family(conn: sqlite3.Connection, entry_id: int,
                     family: Optional[str],
                     conflict: Optional[str] = None) -> None:
    conn.execute(
        "UPDATE entries SET model_family = ?, family_conflict = ? "
        "WHERE id = ?", (family, conflict, entry_id))


def apply_exclusions(conn: sqlite3.Connection) -> Dict[str, int]:
    """Recompute exclusion for every non-operator row (ADR-022 §4 step 5 /
    Grant's candidacy rule). Both directions: new evidence can un-exclude.
    Rows the operator excluded by hand (`excluded_reason = 'operator'`) are
    never touched. Model entries are never excluded (they ARE the bases).

    Precedence: audit_deletable > audit_unconvertable > duplicate >
    no_hf_local_base.
    """
    known = {r["name"] for r in conn.execute(
        "SELECT name FROM families WHERE is_diffusers = 1")}
    stats = {"excluded": 0, "included": 0}
    rows = conn.execute(
        "SELECT id, kind, model_family, classification, duplicate_of, "
        "excluded_reason FROM entries WHERE kind != 'model'").fetchall()
    for r in rows:
        if r["excluded_reason"] == "operator":
            continue
        if r["classification"] == "deletable":
            reason = "audit_deletable"
        elif r["classification"] == "unconvertable":
            reason = "audit_unconvertable"
        elif r["duplicate_of"]:
            reason = "duplicate"
        elif r["model_family"] is None or r["model_family"] not in known:
            reason = "no_hf_local_base"
        else:
            reason = None
        conn.execute(
            "UPDATE entries SET excluded = ?, excluded_reason = ? "
            "WHERE id = ?",
            (1 if reason else 0, reason, r["id"]))
        stats["excluded" if reason else "included"] += 1
    return stats


# ════════════════════════════════════════════════════════════════════════
#  Search (ADR-022 §8) — FTS over descriptions + LIKE over names
# ════════════════════════════════════════════════════════════════════════

def _like_escape(term: str) -> str:
    return (term.replace("\\", "\\\\").replace("%", r"\%")
            .replace("_", r"\_"))


def search(conn: sqlite3.Connection, term: str, *,
           kind: Optional[str] = None, family: Optional[str] = None,
           limit: int = 20,
           include_excluded: bool = False) -> List[Dict[str, Any]]:
    """Search by description terms (FTS) OR name/partial name (LIKE).

    The term is always treated as a QUOTED FTS string (no FTS5 query
    operators pass through — a hostile/odd term cannot inject MATCH
    syntax). Ranking: name-prefix hits first, then FTS bm25, then
    name-substring. Excluded/stale entries are hidden unless
    include_excluded (Vision invariants 2/11).
    """
    term = (term or "").strip()
    if not term:
        return []
    quoted = '"' + term.replace('"', '""') + '"'
    like_sub = f"%{_like_escape(term)}%"
    like_pre = f"{_like_escape(term)}%"

    filters = []
    args_tail: List[Any] = []
    if not include_excluded:
        filters.append("e.excluded = 0 AND e.stale = 0")
    if kind:
        filters.append("e.kind = ?")
        args_tail.append(kind)
    if family:
        filters.append("e.model_family = ?")
        args_tail.append(family)
    where_tail = (" AND " + " AND ".join(filters)) if filters else ""

    sql = f"""
    SELECT e.*, s.rank_class, s.score FROM entries e JOIN (
        SELECT id AS entry_id, 0 AS rank_class, 0.0 AS score
          FROM entries WHERE name LIKE ? ESCAPE '\\'
        UNION ALL
        SELECT CAST(entry_id AS INTEGER), 1, bm25(catalog_fts{_BM25_WEIGHTS})
          FROM catalog_fts WHERE catalog_fts MATCH ?
        UNION ALL
        SELECT id AS entry_id, 2, 0.0
          FROM entries WHERE name LIKE ? ESCAPE '\\'
    ) s ON s.entry_id = e.id
    WHERE 1=1{where_tail}
    GROUP BY e.id
    ORDER BY MIN(s.rank_class), MIN(s.score), e.name
    LIMIT ?
    """
    rows = conn.execute(
        sql, [like_pre, quoted, like_sub, *args_tail, limit]).fetchall()
    out = []
    for r in rows:
        d = {k: r[k] for k in r.keys() if k not in ("rank_class", "score")}
        desc = conn.execute(
            """SELECT source, model_name, description, usage_tips,
                      trigger_words, instruction_template,
                      strength_rec, sampler_rec
               FROM descriptions WHERE entry_id = ?
               ORDER BY CASE source
                   WHEN 'sidecar' THEN 0 WHEN 'civitai_api' THEN 1
                   WHEN 'web' THEN 2 ELSE 3 END LIMIT 1""",
            (r["id"],)).fetchone()
        d["best_description"] = dict(desc) if desc else None
        out.append(d)
    return out


def search_any(conn: sqlite3.Connection, terms: Sequence[str], *,
               kind: Optional[str] = None, family: Optional[str] = None,
               limit: int = 20,
               include_excluded: bool = False) -> List[Dict[str, Any]]:
    """OR-combine `terms` into ONE ranked FTS query (ADR-041 D3).

    Replaces "run N single-term queries and interleave their top hits
    tier-by-tier". The interleave gave every keyword equal weight no matter
    how discriminating it was, so a generic word contributed as many offers
    as a rare one. FTS5 ranks across the whole query — that is what bm25 is
    for — and a row matching three of the terms now outranks one matching a
    single common term.

    Injection posture is UNCHANGED from `search`: each term is individually
    quoted, so no term can smuggle a MATCH operator. The ` OR ` between them
    is code-owned, never caller-supplied.
    """
    # Fail-closed hygiene for callers outside the offer path (security review
    # 2026-07-29, INFO-1). A punctuation-only term quotes to an EMPTY FTS5
    # phrase and makes the whole OR'd MATCH raise — under the old per-keyword
    # loop only that one keyword was lost. An unbounded term list also blows
    # SQLite's bind-variable limit. `_offer_keywords` already guarantees both
    # for `search_loras`; `search_any` is public and must not inherit the
    # crash from a future caller.
    cleaned = [t.strip() for t in (terms or []) if t and t.strip()]
    cleaned = [t for t in cleaned if any(c.isalnum() for c in t)][:32]
    if not cleaned:
        return []
    quoted = " OR ".join('"' + t.replace('"', '""') + '"' for t in cleaned)

    filters = []
    args_tail: List[Any] = []
    if not include_excluded:
        filters.append("e.excluded = 0 AND e.stale = 0")
    if kind:
        filters.append("e.kind = ?")
        args_tail.append(kind)
    if family:
        filters.append("e.model_family = ?")
        args_tail.append(family)
    where_tail = (" AND " + " AND ".join(filters)) if filters else ""

    # All THREE tiers `search` has, per term. The substring arm is not
    # optional decoration: `unicode61` splits on separators only, so a
    # concatenated civitai name like `UltraRealPhoto` is one token and the
    # term "photo" reaches it ONLY via `%photo%`. Dropping this arm would
    # have been a silent recall regression on exactly the run-together names
    # third-party LoRAs favour, invisible to hyphenated ones.
    like_ors = " OR ".join(["name LIKE ? ESCAPE '\\'"] * len(cleaned))
    like_pre = [f"{_like_escape(t)}%" for t in cleaned]
    like_sub = [f"%{_like_escape(t)}%" for t in cleaned]

    sql = f"""
    SELECT e.*, s.rank_class, s.score FROM entries e JOIN (
        SELECT id AS entry_id, 0 AS rank_class, 0.0 AS score
          FROM entries WHERE {like_ors}
        UNION ALL
        SELECT CAST(entry_id AS INTEGER), 1, bm25(catalog_fts{_BM25_WEIGHTS})
          FROM catalog_fts WHERE catalog_fts MATCH ?
        UNION ALL
        SELECT id AS entry_id, 2, 0.0
          FROM entries WHERE {like_ors}
    ) s ON s.entry_id = e.id
    WHERE 1=1{where_tail}
    GROUP BY e.id
    ORDER BY MIN(s.rank_class), MIN(s.score), e.name
    LIMIT ?
    """
    rows = conn.execute(
        sql, [*like_pre, quoted, *like_sub, *args_tail, limit]).fetchall()
    out = []
    for r in rows:
        d = {k: r[k] for k in r.keys() if k not in ("rank_class", "score")}
        desc = conn.execute(
            """SELECT source, model_name, description, usage_tips,
                      trigger_words, instruction_template,
                      strength_rec, sampler_rec
               FROM descriptions WHERE entry_id = ?
               ORDER BY CASE source
                   WHEN 'sidecar' THEN 0 WHEN 'civitai_api' THEN 1
                   WHEN 'web' THEN 2 ELSE 3 END LIMIT 1""",
            (r["id"],)).fetchone()
        d["best_description"] = dict(desc) if desc else None
        out.append(d)
    return out


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
               d.model_name, d.description, d.usage_tips, d.trigger_words,
               d.instruction_template
        FROM entries e LEFT JOIN descriptions d ON d.entry_id = e.id
        """).fetchall()
    n = 0
    for r in rows:
        conn.execute(
            "INSERT INTO catalog_fts (name, model_name, description, "
            "usage_tips, trigger_words, instruction_template, entry_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (r["name"], r["model_name"] or "", r["description"] or "",
             r["usage_tips"] or "", r["trigger_words"] or "",
             r["instruction_template"] or "", r["entry_id"]))
        n += 1
    return n
