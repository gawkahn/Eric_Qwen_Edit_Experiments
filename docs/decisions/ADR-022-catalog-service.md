# ADR-022: LoRA/transformer catalog service — SQLite metadata layer, 4-tier enrichment, search

**Date:** 2026-07-05
**Status:** proposed (S1 implementation gated by `code-reviewer`; S3/S4/S5 additionally by `security-auditor` — untrusted web text + MCP surface)
**Risk:** L3 baseline; Red-Zone-adjacent at two seams named in §7.
**Related:** ADR-014/ADR-021 (audit manifests this ingests), ADR-015 (name-only reference resolution — the serving contract this must NOT disturb), ADR-018 (multi-root scan — shared name/stem policy), ADR-017 (OpenWebUI consumer), ADR-019 (quant verdicts in prognosis data).
**Vision:** `docs/vision/slice-catalog-service.md` (approved 2026-07-05).
**AI-Disclosure:** Claude (Fable 5) authored; Grant reviewed.

---

## Context

Grant's directive (2026-07-05): a complete catalog service — source of truth for LoRA/transformer full paths; `comfyless.generate` and MCP call **names only** (or `list_loras` with a model-family argument, then select); search by description terms or partial names; descriptions/usage tips sourced from civitai.com/civitai.red and the web; consumed later by the LLM iterative generator (generate → evaluate → adjust prompt/LoRAs → retry). AskUserQuestion settled: SQLite in `~/.local/share` (ext4 — verified `fuseblk` on `~/projects`, `ext2/ext3` family on `~/.local`); 4-tier enrichment; sample+unknowns live validation.

Assets already in hand:
- **ADR-018 scan** mints every LoRA/transformer as `(name → abs_path, kind)` — the serving-side namespace the DB must mirror, not fork.
- **ADR-014/021 audit manifests** carry per-file classification (usable/convertable/unconvertable/deletable), base matches, and `duplicate_of` — the candidacy evidence.
- **Lora Manager sidecars** (`<stem>.metadata.json` beside deployed LoRAs) carry full civitai payloads: sha256, base_model, `trainedWords`, HTML description, NSFW levels, stats. Tier-1 enrichment is a local parse for the bulk of deployed LoRAs.
- `infer_model_family` (nodes/eric_diffusion_utils) maps hf-local `model_index.json` dirs to family strings — the family registry.

## Decision

### 1. Architecture: DB is the metadata plane, never the load plane

The **serving** path (CLI/daemon/MCP `generate`) continues to resolve names via the in-memory ADR-015/018 catalog with `_check_paths`-union containment — byte-untouched by this ADR. The **DB** is a metadata/search layer consulted by `search`/`list_*`/the iterative generator for *choosing* names. Consequences:
- A corrupt/poisoned/stale DB cannot redirect a load (Vision invariant 7): worst case is a bad *recommendation*, and the recommended name still resolves through the reviewed load boundary.
- DB absence degrades discovery only; generation never depends on it.
- Name identity: DB `name` = ADR-018 stem via the same `normalize_name`; the DB stores what the scan minted, so agent-visible names are load-resolvable by construction. Divergence (DB name absent from a fresh scan) marks the row `stale`.

### 2. Storage

- Path: `~/.local/share/comfyless/catalog.sqlite` (override `--db`; XDG-respecting). WAL mode.
- **Mergerfs guard:** `build` refuses a `--db` on a FUSE filesystem (`os.statvfs`+`/proc/mounts` fstype check) with a loud explanation; `--force-fs` overrides (warn-don't-block).
- Module layout: `comfyless/catalog_db.py` (schema, connection, upserts, queries, sanitizer), `comfyless/catalog_builder.py` (build/ingest orchestration), `comfyless/catalog_cli.py` (click group: `build`, `enrich`, `search`, `show`, `annotate`, `exclude`). MCP wiring in `mcp_server.py` (S5).

### 3. Schema (v1; `PRAGMA user_version = 1`)

```sql
entries(id INTEGER PK, name TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('lora','transformer','model')),
        abs_path TEXT NOT NULL, root TEXT, relative_path TEXT,
        size_bytes INTEGER, sha256 TEXT,            -- null until hashed
        model_family TEXT,                          -- resolved per §5
        classification TEXT, reason TEXT,           -- from audit manifest
        duplicate_of TEXT,                          -- ADR-021 §4
        excluded INTEGER NOT NULL DEFAULT 0,
        excluded_reason TEXT,                       -- no_hf_local_base | audit_unconvertable | audit_deletable | duplicate | operator
        stale INTEGER NOT NULL DEFAULT 0,
        family_conflict TEXT,                       -- Vision neg-case 6
        first_seen TEXT NOT NULL, last_seen TEXT NOT NULL,
        UNIQUE(kind, name))
descriptions(id INTEGER PK, entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
        source TEXT NOT NULL CHECK(source IN ('sidecar','civitai_api','web','ai_authored')),
        description TEXT, usage_tips TEXT,
        trigger_words TEXT,                         -- JSON array
        strength_rec TEXT, sampler_rec TEXT,
        nsfw_level INTEGER,
        civitai_model_id INTEGER, civitai_version_id INTEGER,
        provenance_url TEXT, fetched_at TEXT NOT NULL,
        UNIQUE(entry_id, source))                   -- Vision neg-case 7
families(name TEXT PK, hf_local_path TEXT NOT NULL, model_index_class TEXT, is_diffusers INTEGER NOT NULL)
gen_tests(id INTEGER PK, entry_id INTEGER NOT NULL REFERENCES entries(id),
        prompt TEXT, negative_prompt TEXT, params_json TEXT,
        image_path TEXT, verdict TEXT, notes TEXT, tested_at TEXT NOT NULL)
catalog_fts (FTS5: name, model_name, description, usage_tips, trigger_words; content-synced by triggers)
meta(key TEXT PK, value TEXT)                       -- schema_version, last_build, manifest hashes
```

Best-description view: per entry, the single row whose source ranks `sidecar > civitai_api > web > ai_authored` for *facts* (trigger words, ids), while `search` FTS spans all rows (an ai_authored usage-tip is findable even when a sidecar description exists).

### 4. Build (no network)

`catalog build --model-base … --lora-path … --transformer-path … [--audit-manifest lora_audit.json]…`
1. Scan via `comfyless.catalog.build_catalog` (ADR-018 — same roots, same collision policy, same names as serving).
2. Family registry: hf-local dirs with `model_index.json` → `infer_model_family` → `families` (is_diffusers=1).
3. Join audit manifests (ADR-014 LoRA + ADR-021 transformer entries) by realpath: classification, reason, matched_bases, duplicate_of. Absent manifest → classification NULL (unaudited; candidacy pends).
4. Sidecar ingest: `<stem>.metadata.json` beside each LoRA → tier-1 description row + sha256 + base_model + trigger words (HTML sanitized per §6).
5. Family resolution per §5; exclusion per Grant's candidacy rule: `excluded=1` iff audit says `unconvertable`/`deletable`, or `duplicate_of` set, or resolved family has no `is_diffusers` hf-local entry, or operator `exclude` verb. Reason recorded; rows retained.
6. Upsert semantics: key `(kind, name)`; refresh paths/size/last_seen; vanished → `stale=1`; enrichment rows untouched (Vision invariant 12).

### 5. Family resolution (evidence precedence)

1. **Audit evidence** (strongest): LoRA `verdicts_by_base` OK/NORM base, transformer `matched_bases` → that base's family.
2. **Sidecar declaration**: `base_model` string mapped through a fixed table ("Flux.1 D"→flux, "Flux.2"→flux2, "Qwen"→qwen-image, "SDXL 1.0"→sdxl, "Pony"→pony, "Illustrious"→illustrious, …; unknown strings recorded verbatim, family NULL).
3. **Path convention** (weakest): `<root>/<family-folder>/…` directory-name hints.
Disagreement → highest-precedence wins, `family_conflict` records the loser (Vision neg-case 6). No evidence → family NULL → excluded (`no_hf_local_base`) until audited. *(Clarified per S2 review finding 4: a family-KNOWN but audit-less entry — e.g. sidecar-only evidence — is INCLUDED; missing audit is not an exclusion reason. Only the four §4-step-5 reasons exclude.)*

### 6. Enrichment (network; explicit `enrich` step) and the injection seam

Tiers, each filling only what's missing (idempotent upsert on `(entry, source)`):
1. `sidecar` — done at build.
2. `civitai_api` — `GET civitai.com/api/v1/model-versions/by-hash/{sha256}` (fallback mirror civitai.red). Needs sha256: sidecar value if present, else computed (bounded batch; LoRAs only by default — transformers per ADR-021 remain opt-in). Rate-limited (≤1 req/s), resumable, per-entry failure isolation (Vision neg-case 5).
3. `web` — operator/Claude-session-driven research for entries tiers 1–2 left bare (orphaned post-civitai-red files, `project_civitai_orphaned_files`): the `enrich --worklist` verb emits the bare-entry list; findings land via `annotate --source web --url …`.
4. `ai_authored` — synthesized from trigger words + live gen tests (gen_tests rows, task-8 loop) via `annotate --source ai_authored`.

**Sanitization (all stored text, every tier):** HTML → plain text (tags stripped, entities decoded), control/zero-width chars removed, length caps (description 4 KiB, tips 2 KiB, trigger words 64×64 B), URLs allowed in `provenance_url` only. Stored civitai text is DATA; **the prompt-injection posture** is: (a) sanitize, (b) provenance-tag, (c) MCP `search`/`list_*` return descriptions inside a clearly-data-shaped field with tool-doc language instructing the model to treat catalog text as content, never instructions. This does not *eliminate* the channel (an LLM can still be socially engineered by a description); it bounds it — and the load boundary is DB-independent (§1), so the blast radius of a hostile description is a bad generation, not a path escape. Named residual, accepted for the desktop/solo threat model; re-evaluate at any remote/multi-tenant transition (ADR-014 F-10 style trigger).

### 7. Red-Zone-adjacent seams (named)

- **Web text → agent-visible fields** (§6). Mitigations above; `security-auditor` gates S3/S4/S5.
- **Iterative-generator consumption**: when the LLM loop starts *acting* on catalog text (choosing LoRAs from descriptions), the catalog becomes input to an autonomous loop. The loop's authority stays bounded by the name-only MCP contract (worst action: generate an image with an odd LoRA). Any future widening of the loop's tool authority re-fires review.

### 8. Search

- `search TERM [--kind …] [--family …] [--limit N] [--include-excluded]`: FTS5 `MATCH` over name/model_name/description/tips/trigger-words, OR-fallback `LIKE %term%` on name for partial-name hits FTS tokenization misses (`search "mystic"` → `mystic_realism_v2`). Ranked: name-prefix > FTS bm25. Default hides excluded/stale.
- MCP tools (S5): `search` (same params), `list_loras`/`list_transformers` gain optional `model_family` + counts of hidden-excluded. Existing no-DB behavior preserved (fall back to scan-catalog listing with an INFO notice).

### 9. Slice plan

| Slice | Scope | Review |
|---|---|---|
| S1 | `catalog_db.py` schema + sanitizer + upserts + `build` (scan+manifest join) + `test_catalog_db.py` | code-reviewer |
| S2 | Sidecar ingest + family registry/resolution + exclusion policy + `search` CLI | code-reviewer |
| S3 | `enrich` tier-2 civitai hash API + sha256 batch | code-reviewer + security-auditor |
| S4 | `annotate` (web/ai_authored write-back) + worklist | code-reviewer + security-auditor |
| S5 | MCP `search`/family-filtered `list_*` | code-reviewer + security-auditor |
| S6 | gen_tests write-back wiring (feeds live-validation) | code-reviewer |

## Alternatives Rejected

- **DB in the load path** (resolve names from SQLite at generate time). Rejected hard: puts writable, web-fed state inside the reviewed load boundary; ADR-015/018's in-memory spawn-time catalog is the security posture. The DB recommends; the scan resolves.
- **JSON manifests instead of SQLite** (offered; Grant chose SQLite). FTS and incremental enrichment argue DB; mergerfs risk handled by location + guard.
- **Postgres**. Overkill for solo desktop; SQLite+WAL on ext4 suffices for ~1 k rows and single-writer batch jobs.
- **Extending Lora Manager itself.** It's a ComfyUI-coupled web app; the catalog must serve comfyless/MCP headless. Its *sidecars* are ingested as data instead.
- **Auto-deleting excluded files.** Audit/delete stays in ADR-014's triple-gated lane; the catalog only flags.
- **Embedding-based semantic search.** FTS5 first; embeddings are a later slice if term search proves insufficient for the iterative generator.

## Deferred / Out of Scope

- Preview-image ingestion (sidecar `preview_url` jpegs) — nice for OWUI, later slice.
- Embeddings/semantic search; HTTP transport; multi-host sync.
- Automatic re-audit scheduling (operator runs audit; build joins whatever manifests exist).
- civitai *download* integration (catalog is read-only over local files).

## Changelog

- **2026-07-05 (proposed):** Drafted per Grant's catalog-service directive + AskUserQuestion decisions. S1 may begin after `code-reviewer`; the ADR itself goes to `security-auditor` design review together with S3–S5 implementation (the seams are implementation-shaped; §6/§7 name the posture the review must hold me to).
