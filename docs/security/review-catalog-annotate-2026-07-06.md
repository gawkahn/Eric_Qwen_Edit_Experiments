# Security review — ADR-022 S4 write-back verbs (worklist / annotate / exclude)

**Date:** 2026-07-06 · **Reviewers:** `code-reviewer` + `security-auditor` (both Opus, model pinned) · **Both CHANGES REQUIRED → all findings folded same-day; post-fold suite 120/120.**

**AI-Disclosure:** Claude (Opus subagents) authored the reviews; Claude (Fable 5) drove the slice; Grant reviewed.

**Scope:** the tier-3 (web-researched) / tier-4 (AI-authored) text write-back seam — untrusted-adjacent text entering agent-visible DB fields via `annotate`; operator exclusion toggling; the bare-candidate worklist.

## Findings + folds

**security-auditor:**
- **F-1 (MED)** — `provenance_url` bound RAW (no ctrl/zero-width strip, no cap, no scheme check) — same escapes-the-sanitizer class as S3 F-1, agent-visible via `show` today and any future S5 provenance projection. **Fixed:** `sanitize_url` inside `upsert_description` (applies to every caller): ctrl/zw/bidi strip, 2 KiB cap, **http(s) schemes only** — `javascript:`/`data:` drop to NULL. Hostile-URL negative tests added (F-3).
- **F-2 (LOW)** — `exclude --clear` could transiently un-exclude an AUDIT-excluded (deletable/duplicate) entry until the next build. **Fixed:** clear scoped `AND excluded_reason = 'operator'`; non-operator clear refused with exit 2 + explanation. Negative test added.
- Verified CLEAN: machine-tier spoofing blocked (click.Choice — `sidecar`/`civitai_api` unwritable by hand); full SQL parameterization; exclude touches only its two columns; zero network in S4 verbs; worklist leaks no filesystem paths (asserted by test).

**code-reviewer (CHANGES REQUESTED, minor):**
- Finding 1 (MED test gap) — worklist miss-marker semantics (civitai 404 row with NULL description stays ON the worklist) now pinned by test.
- Finding 2 (MED test gap) — exclude nonexistent/ambiguous → exit 2, tested.
- Finding 3 (LOW) — dead sub-expression in the exclude error message removed.
- Finding 4 (LOW) — standalone `worklist` verb vs the ADR's `enrich --worklist` phrasing recorded in the ADR-022 Changelog.
- Finding 5 (LOW) — re-annotate whole-row-replace semantics documented in the verb help + pinned by test.
- Finding 8 — this record is the mandated security-auditor artifact.
- Verified correct: worklist NULL semantics, exit codes, FTS placement (annotate rebuilds in-transaction; exclude correctly does not — search filters excluded at query time), machine-tier lockout, load-plane independence intact.

## Residuals

- `provenance_url` remains out of the FTS/search projection by design; the F-1 cleaner is the guard if S5 adds provenance to any agent surface.
- Re-annotate replace semantics is a documented operator footgun (not a security surface).
