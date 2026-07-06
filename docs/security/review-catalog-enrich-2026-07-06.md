# Security review — ADR-022 S3 catalog enrichment (civitai hash lookup)

**Date:** 2026-07-06 · **Reviewers:** `code-reviewer` + `security-auditor` (both Opus, model pinned at invocation) · **Both returned CHANGES REQUIRED; all blocking findings folded same-day; post-fold suite 98/98.**

**AI-Disclosure:** Claude (Opus subagents) authored the reviews; Claude (Fable 5) drove the slice; Grant reviewed.

**Scope:** `comfyless/catalog_enrich.py` (the catalog's ONLY network code — untrusted civitai JSON → sanitized DB rows later surfaced to LLM agents), `catalog_cli.py` enrich verb, S3 tests. Trust boundary per ADR-022 §6/§7: web text is prompt-injection-adjacent; the load plane is structurally DB-independent (§1, AST-enforced), so hostile text's blast radius is a bad recommendation, never a path escape.

## Findings + folds

**code-reviewer (CHANGES REQUIRED → folded):**
- **Finding 1 (HIGH):** `civitai_by_hash` recorded a *persistent* miss on the FIRST host's 404 without trying the mirror — but post-split, orphaned files exist only on civitai.red. **Fixed:** 404 now falls through to every host; a miss is definitive only when ALL hosts 404 with no errors; host errors keep the slot un-poisoned (resume retries).
- **Finding 3 (MED):** tests didn't discriminate the fix. **Added:** split-orphan fixture (404 on .com, hit on .red) asserting a HIT row.
- **Finding 4 (LOW):** load-plane import ban omitted `catalog_enrich` — the one module that would drag network code into the load boundary. **Fixed** (also security F-2).
- **Finding 5 (LOW):** consecutive-failure abort skipped `rebuild_fts`, leaving committed hits unsearchable until a later completing run. **Fixed:** FTS rebuilt before the abort raise.
- **Finding 7 (LOW):** JSON-bomb `RecursionError` uncaught. **Fixed:** `ValueError` + `RecursionError` in the except tuple.
- Finding 2 = this record + the security-auditor run below. Finding 6 (sha gate charset-only) accepted as noted. Finding 8: unrelated working-tree files kept out of the S3 commit.

**security-auditor (CHANGES REQUIRED, no CRITICAL/HIGH → folded):**
- **F-1 (MED):** `nsfwLevel`/`modelId`/`id` bound raw into INTEGER-affinity columns — SQLite stores non-numeric strings VERBATIM (unsanitized, uncapped), a genuine bypass of the "every field sanitized" contract, latent until S5 surfaces `nsfw_level` to agents. **Fixed:** `_int_or_none` coerce-or-drop at the boundary (bools rejected); hostile-payload negative test added (F-6).
- **F-2 (MED):** same as code-review finding 4 — **fixed**.
- **F-3 (INFO, accepted residual):** urllib follows redirects — a compromised/MITM'd origin could redirect the GET to a localhost service. Bounded by TLS verification, http/https-only redirect schemes, capped redirect count, read-only 4 MiB response handling. Accepted under the desktop/solo threat model; **re-fire at any remote/multi-tenant transition.**
- **F-4 (INFO, accepted residual):** 15 s timeout is per-recv, not wall-clock — slow-drip reads bounded by the 4 MiB cap + per-entry isolation + operator interrupt.
- **F-5 (INFO):** same substance as code-review finding 1 — **fixed**.
- Verified sound: URL construction gated by lowercase-hex sha validation + module-constant hosts (no payload value ever reaches URL/path construction — no SSRF from DB content); DoS bounds (consecutive-failure abort, per-entry commit, rate limit, 4 MiB cap); miss-marker rows cannot create hit/miss confusion with security consequence; zero load-plane interaction; `local_files_only` inference untouched.

## Residual-risk register

1. Redirect-following SSRF (F-3) — accepted, desktop/solo; trigger: remote/multi-tenant.
2. Per-recv timeout slow-drip (F-4) — accepted; trigger: hardening pass.
3. `--refresh` is coarse (re-queries hits too); misses-only refresh is a nice-to-have.
