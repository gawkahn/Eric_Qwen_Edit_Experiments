# Vision slice — LoRA/transformer catalog service

**Date:** 2026-07-05
**Risk:** L3 baseline (SQLite writes, web-content ingestion into agent-visible text); **Red-Zone-adjacent** at the enrichment/MCP seams (untrusted web text surfaced to LLM agents — prompt-injection channel; named, bounded, provenance-tagged)
**Status:** approved (Grant directive 2026-07-05 + AskUserQuestion choices: SQLite in ~/.local/share; 4-tier enrichment local→hash→web→AI; sample+unknowns gen testing)
**AI-Disclosure:** Claude (Fable 5) authored; Grant reviewed.

## What must be true when done

1. A SQLite catalog at `~/.local/share/comfyless/catalog.sqlite` (ext4, off-mergerfs) is the **source of truth for LoRA/transformer metadata**: name, kind, path, family, classification (from the audit manifest), descriptions, trigger words, usage tips (strength/sampler recs), provenance, exclusion status.
2. **Candidacy rule (Grant):** only LoRAs/transformers whose family maps to an hf-local diffusers model are candidates. Everything else — `unconvertable`/`deletable` audit classifications, `duplicate_of` hits, no-hf-local-family — is `excluded` with a machine-readable reason, retained in the DB (not deleted) and hidden from default list/search output.
3. `comfyless` code can search: by description terms (`search "cinematic"` — FTS), by name or partial name (`search "mystic"`), optionally filtered by kind and model family. MCP exposes the same (`search`, `list_loras(family=…)`, `list_transformers(family=…)`).
4. Descriptions carry a 4-tier provenance tag: `sidecar` (Lora Manager .metadata.json) → `civitai_api` (SHA-256 hash lookup) → `web` (civitai.com/civitai.red + general search) → `ai_authored` (from trigger words + live gen tests). Every row: source, fetched_at, provenance URL where applicable.
5. Rebuilds are idempotent upserts; enrichment survives rebuilds (keyed by sha256 where present).
6. The eventual LLM iterative generator can consume the catalog through the same MCP surface (names + descriptions + tips in, names out to generate).

## What must never happen

7. **The DB is never in the load path.** `generate` (CLI, daemon, MCP) resolves names through the existing in-memory scan catalog + `_check_paths` union (ADR-015/018, security-reviewed). A poisoned/corrupt/stale DB row can NEVER redirect which weights load. DB absence degrades search/descriptions only — generation keeps working.
8. No network during `build` or at inference. Network happens only in the explicit `enrich` step.
9. No raw web HTML into the DB: sanitize to plain text, cap lengths, strip markup/URLs-as-instructions; provenance recorded. Agent-facing tool docs mark catalog text as untrusted data, not instructions.
10. The DB file never lands on mergerfs: `build` statfs-checks the target; FUSE target → loud warning + refusal unless `--force-fs` (warn-don't-block per Grant's standing preference, but the default protects against silent lock hangs).
11. Excluded entries are hidden, not deleted; `--include-excluded` reveals them. Requesting an excluded name via `generate` still resolves (serving catalog is independent) — at most a warning notice.
12. Rebuild never deletes enrichment rows; vanished files mark entries `stale`, never DROP.

## Proof hooks

- New `test_catalog_db.py`: schema, upsert idempotency, exclusion policy, FTS + name search, provenance constraint (description row without source rejected), sanitizer negatives (script/HTML/zero-width in civitai text → clean), mergerfs-refusal (mocked statfs), stale-not-deleted, enrichment-survives-rebuild.
- MCP tests: search/list tools hide excluded by default; generate path provably DB-independent (monkeypatch DB module away → generate still resolves).
- Full regression green.

## Negative cases required

1. DB path on FUSE fs → refusal + message (and `--force-fs` override works, loudly).
2. `generate` with no DB file → works unchanged.
3. Excluded LoRA: absent from `search`/`list_loras`; present with `--include-excluded`; generate-by-name unaffected.
4. Civitai description containing `<script>`, markdown links, and "ignore previous instructions" text → stored sanitized, surfaced with provenance tag; MCP payload contains it only inside the data field.
5. `enrich` with network down → per-entry failure recorded, run resumable, exit code distinguishes partial.
6. Family conflict (sidecar claims "SDXL", audit shape-match says flux) → audit evidence wins; conflict recorded in the entry's `notes`/warning field.
7. Two enrich runs → no duplicate description rows (upsert on (entry, source)).

## Out of scope

- Any change to the serving load boundary (`resolve_reference`, `_check_paths`) — untouched.
- HTTP transport for the catalog (CLI + MCP stdio only).
- Automated deletion of excluded files (report-only; ADR-014 delete stays LoRA-audit-scoped).
- Video-model workflow support beyond family tagging (Wan entries are tagged; the image-gen loop is the consumer).
