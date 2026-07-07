# Security Review — Slice 4a: default catalog-DB auto-discovery

**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored this security review; Grant to review. Red-Zone-adjacent per slice-4 Vision §Risk (L3).
**Date:** 2026-07-07
**Reviewer:** `security-auditor` (Opus), invoked per project CLAUDE.md review bar (MCP-surface change).
**Change under review:** `_discover_default_catalog_db()` + `main()` wiring in `comfyless/mcp_server.py` (Step 4a of ADR-011 slice 4; Vision `docs/vision/slice-4-mcp-extract-params.md` invariant 16 / N25–N28).
**Verdict:** **CLEAN** — no CRITICAL/HIGH/MEDIUM findings. Four INFO observations (below); INFO-1 and INFO-4 folded before commit.

---

## Summary

Step 4a changes what the comfyless MCP server adopts at spawn: when `--catalog-db` is unset, `main()` calls `_discover_default_catalog_db()`, which opens the SQLite metadata DB at `catalog_db.DEFAULT_DB_PATH` (`~/.local/share/comfyless/catalog.sqlite`) read-only via `connect_readonly`, probes `PRAGMA user_version`, closes, and returns the realpath if valid — else `None`. The result feeds `_validate_startup_args(catalog_db=...)` exactly as an explicit flag would. An explicit `--catalog-db` bypasses discovery entirely and keeps its fail-closed validation. Discovery is fail-open by design: a missing/corrupt/wrong-schema/unreadable file yields `None` and the server starts without a DB (`search` unadvertised, enrichment skipped). Threat model is a single-user desktop tool driven by a same-uid LLM agent over stdio, DB built offline by the user's own `catalog_cli`.

I traced the trust boundaries (agent/stdout vs operator/stderr vs the on-disk DB the operator built), the untrusted-data flow from the DB into `search`/enrichment, the fail-open error path in `_discover_default_catalog_db`, and the load-plane independence invariant. The change fails toward *less* capability (de-advertises a tool, skips enrichment) rather than granting authority; the discovered path is stored only in `cfg.catalog_db_path`, which is never folded into `all_roots` (the load boundary at mcp_server.py:720–722), so `generate`/`resolve_reference` remain on the live in-memory catalog. Read-only `mode=ro` with an `isfile` gate rejects device nodes/FIFOs and SQLite bounds how much of a non-DB file it reads. The `except (CatalogDBError, sqlite3.Error, OSError)` is appropriately narrow. Overall posture: **CLEAN** — no CRITICAL/HIGH/MEDIUM findings. Four INFO observations, the most notable being that auto-discovery makes the (already-mitigated) description prompt-injection surface default-on rather than operator-opted-in.

## Coverage

Reviewed:
- `comfyless/mcp_server.py` — `_discover_default_catalog_db` (the new helper)
- `comfyless/mcp_server.py` — `main()` call site (effective_catalog_db selection + stderr print)
- `comfyless/mcp_server.py` — `_validate_startup_args`, incl. the explicit `--catalog-db` fail-closed branch
- `comfyless/mcp_server.py` — `_StartupConfig` + `all_roots` derivation (load-boundary union)
- `comfyless/mcp_server.py` — `_list_tools_impl` conditional `search` advertisement
- `comfyless/mcp_server.py` — `_handle_search`, `_db_family_names`, `_validated_family_filter` (DB consumers)
- `comfyless/catalog_db.py` — `connect` (FUSE-guarded, writable) vs `connect_readonly` (the probe used by discovery)
- `comfyless/catalog_db.py` — `DEFAULT_DB_PATH`, `CatalogDBError`, `fs_is_fuse`
- `comfyless/catalog_db.py` — `sanitize_text` (storage-time description sanitizer, relevant to the injection posture)
- `docs/vision/slice-4-mcp-extract-params.md` — invariants 11/12/16 and N25–N28

Not reviewed (and why):
- `test_mcp_server.py` — test correctness is out of the security-truth scope for this change (N25–N28 confirmed present + green by the implementing session).
- `comfyless/catalog_db.py` `search()` SQL body and `catalog_builder.py` write path — DB *content* provenance is governed by ADR-022 and pre-exists this change; only its exposure-timing changes here.
- `comfyless/generate.py` load path — unchanged by this diff; relied on the two-catalog split as documented.

## Findings

No findings at CRITICAL/HIGH/MEDIUM. INFO observations below.

### [INFO-1] Auto-discovery makes the untrusted-description surface default-on (trust-posture shift) — **FOLDED**
Before this change, exposing DB `description`/`usage_tips`/`trigger_words` to the agent required the operator to explicitly pass `--catalog-db`. After it, any DB file present at the default path is adopted automatically, so the civitai/web-sourced description text — explicitly framed as a prompt-injection vector in `_SEARCH_TOOL_DESCRIPTION` — becomes agent-reachable by default whenever a DB has ever been built. Within the stated same-uid single-user threat model this is not a privilege escalation (anyone who can write that DB can do worse), and the existing mitigations hold: storage-time `sanitize_text` strips markup/bidi/control, the search tool description frames descriptions as DATA-not-instructions, and `trigger_words` is re-guarded at read.
**Resolution (folded 2026-07-07):** the operator stderr adoption line now names the posture change — `"… — \`search\` tool and metadata enrichment now active"` — so the activation is explicit in logs. Assumption: the DB at the default path is only writable by the same uid.

### [INFO-2] Double `realpath` between probe and stored path (benign same-uid TOCTOU) — accepted, no change
The path is realpath'd once inside `connect_readonly` for validation and again independently when returned, so a symlink swap between the two resolutions could store a path that differs from the one validated. Not exploitable: every downstream consumer (`_handle_search`, `_db_family_names`) calls `connect_readonly(cfg.catalog_db_path)` again per request, which re-runs the `isfile` + schema re-validation and fails closed ("catalog database unavailable") on a bad target — no load authority and no path egress. Same-uid is inside the trust boundary regardless. Cosmetic only; left as-is.

### [INFO-3] `connect_readonly` has no FUSE guard (default path is ext4, so not triggered here) — TECH_DEBT
A WAL-mode SQLite read on a FUSE/mergerfs filesystem can hang on fcntl locks. `DEFAULT_DB_PATH` lives under `~/.local/share` (home ext4, not the `/home/gawkahn/projects` mergerfs union), so discovery is unaffected. An explicit `--catalog-db` pointed at a FUSE path would also skip the guard — pre-existing behavior in `catalog_db.py`, not introduced by this diff. Logged to `TECH_DEBT.md` (Security 2026-07-07) for the explicit-path case; out of this slice's edit scope.

### [INFO-4] Discovery `except` omitted `ValueError` (unreachable for the default path) — **FOLDED**
The catch was `(CatalogDBError, sqlite3.Error, OSError)`. An embedded-NUL `db_path` would make `os.path.realpath` inside `connect_readonly` raise `ValueError`, propagating and crashing startup (fail-closed) rather than fail-open. Not reachable in the production call graph (`main()` always passes no argument → `DEFAULT_DB_PATH`, NUL-free), but the helper's `db_path` parameter is exercised by tests and future callers.
**Resolution (folded 2026-07-07):** `ValueError` added to the except tuple so the "never raises" contract holds for an explicitly-passed path too; regression test `4a N28: NUL-byte db_path -> None` added.

## Answers to the specific questions

1. **New attack surface?** No new code-execution/write/crash-loop/exhaustion surface — read-only, one-shot, `isfile`-gated, bounded reads. Fail-open is the correct posture here because the DB is a non-critical metadata plane and failing open drops *capability* (search/enrichment) rather than granting authority.
2. **`--catalog-db` fail-closed contract intact?** Yes — discovery runs only on `catalog_db is None`; an explicit value is never passed to discovery and retains its `click.BadParameter` fail-closed validation.
3. **Load-plane independence (ADR-022)?** Holds — `catalog_db_path` is not part of `all_roots`; generate/resolve stay on the live in-memory catalog.
4. **Information disclosure across MCP/stdout?** None — the discovered path is printed only to operator-facing stderr; `search`/enrichment responses are name + allowlisted-metadata only.

## Verdict

**CLEAN.** No CRITICAL/HIGH/MEDIUM findings. INFO-1 and INFO-4 folded before commit; INFO-2 accepted (benign); INFO-3 logged to TECH_DEBT (pre-existing, out of scope).
