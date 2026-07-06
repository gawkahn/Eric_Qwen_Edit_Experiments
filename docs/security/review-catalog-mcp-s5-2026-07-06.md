# Security review — ADR-022 S5 MCP catalog surface (search + family filters)

**Date:** 2026-07-06 · **Reviewers:** `code-reviewer` (APPROVED) + `security-auditor` (**CLEAN** — 0 CRITICAL/HIGH/MED, 1 LOW folded, 2 INFO recorded) · Both Opus, model pinned at invocation. Post-fold: test_mcp_server 597/597, test_catalog_db 125/125, full regression 2153+ green.

**AI-Disclosure:** Claude (Opus subagents) authored the reviews; Claude (Fable 5) drove the slice; Grant reviewed.

**Scope:** MCP `search` tool + `model_family` filters on `list_loras`/`list_transformers`, `--catalog-db` spawn option, `catalog_db.connect_readonly` (sqlite `mode=ro` per-request accessor), `start-mcpo.sh` conditional wiring. This is the seam ADR-022 §7 flagged: web-sourced catalog text reaching the LLM agent.

## Keystone verifications (security-auditor, verbatim conclusions)

- **Load-plane independence holds structurally:** the DB influences only which NAMES an agent hears about; names re-resolve through the reviewed in-memory ADR-015/018 catalog with request-time `os.path.exists` + `_within` against `all_roots` — the DB's `abs_path` column is never read on the load path. A poisoned DB row cannot inject a name that resolves out-of-root (non-catalog name → uniform `UnknownName`; catalog name → operator-curated in-root path). The family filter intersects DB names with the serving-catalog loop, so it can only hide or narrow, never inject.
- **Strict-allowlist projection:** name/kind/model_family/classification + fixed description sub-block only — no abs_path/root/relative_path/sha256/civitai ids/provenance_url/nsfw_level; excluded/stale never surface (`include_excluded` not on the MCP schema). Closes the S3 F-1 latent `nsfw_level` exposure concern at the read side too.
- **Injection posture:** FTS term force-quoted, LIKE escaped, all parameterized; query/kind/family/limit validated (bool-as-int rejected); audit lines bounded (schema'd keys, 256-char caps).
- **Read-only guarantee:** `connect_readonly` fail-closes on missing file/schema mismatch, creates nothing, URI-quotes hostile path characters (space/`?`/`#` — pinned by test), re-checks `user_version` per request; writes proven impossible (`mode=ro` OperationalError test).
- **Spawn probe fail-closed:** NUL pre-check, read-only probe, `click.BadParameter` on failure, realpath stored. `start-mcpo.sh` adds `--catalog-db` only when the file exists.
- **Data-framing:** the `search` tool description explicitly instructs the model to treat description/tips/trigger words as DATA, never instructions (ADR-022 §6 mitigation c).

## Findings + folds

- **[LOW, security] trigger_words projection trusted storage shape** — `json.loads` without a list-of-short-strings guard could pass a hand-tampered DB value (nested/oversized JSON) to the agent unbounded. **Folded:** read-boundary guard mirrors `sanitize_trigger_words` (list-only, `str`-coerced, 64×64 caps). Requires local-file tamper outside the sanitizing write path — largely out of the desktop/solo threat model; hardened anyway per the "treat the DB as poisoned" framing.
- **[INFO] search response not byte-capped** — bounded today by storage caps × limit≤50 (~≤350 KB worst case); add a per-response cap at any remote/multi-tenant transition (same trigger as the S3/S4 residuals).
- **[INFO] per-request ro connect cost** — negligible on local SQLite; MCP stdio serializes requests.
- **[code-review finding 1, LOW]** stale "No inputs." prose on the two list tools after S5 added the optional filter — **folded** (constants updated; byte-lock test compares to the constants so stays green). **finding 9 gap** — URI-quoting test added. **finding 3** — confirmed: the CLAUDE.md hunk is S5 counts only.

## Residual-risk register

1. Prompt-injection of the downstream LLM via sanitized, provenance-tagged, data-framed catalog text — the accepted ADR-022 §6 residual; blast radius bounded to a bad recommendation by load-plane independence.
2. Search response byte-cap + re-fire of this review at any remote/multi-tenant transition.
