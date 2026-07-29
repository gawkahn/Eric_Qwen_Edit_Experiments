# Security review — ADR-041 slice 1 (D2 + D3), 2026-07-29

AI-Disclosure: `security-auditor` run on Fable 5 (`claude-fable-5`), invoked with
an explicit `model: "fable"` override after Grant confirmed the Fable budget had
reset (2026-07-28 evening). The reviewer notes it executed directly rather than
spawning a sub-agent, so there is **no subagent transcript to grep for a
model fallback** — the usual `feedback_reviewer_model_fallback_check` step is
not applicable to this review and its absence is recorded here deliberately
rather than left as an unexplained gap.

**Verdict: CLEAN — no blocking findings.**

Trigger: `comfyless/refine.py` is listed in `scripts/git-policy/_red-zone-paths.sh`
(ADR-027 judge/seed surfaces) and this slice modifies `search_loras`. Caught by
`code-reviewer` as a process note — the Red Zone gate was initially missed when
the slice was planned.

## Scope

ADR-041 slice 1 = D2 + D3:

- **D2** — new `descriptions.instruction_template` column, cap 512 B, versus the
  existing 64 B per-trigger-word cap. Holds THIRD-PARTY civitai `trainedWords`
  prose, so this stores ~8x more attacker-controllable text per entry than
  before.
- **D3** — FTS5 `tokenize='porter unicode61'`, bm25 column weights, the new
  `instruction_template` FTS column, and `catalog_db.search_any()` which
  OR-combines terms into ONE ranked query (replacing N interleaved per-keyword
  queries in `refine.search_loras`).
- Schema v1 → v2 in-place migration.

## Threat model

1. A hostile civitai uploader controls `trainedWords` and `description` for any
   LoRA the operator has downloaded locally.
2. The judge's LLM-authored critique feeds `_offer_keywords` → `search_any`, so
   **query terms are attacker-influenceable** if a hostile description or seed
   image steers the judge.
3. Migration failure must not silently degrade search or fail open.
4. The catalog must stay off the load plane (ADR-022 invariant 7 / ADR-041 D6).

## Surfaces examined and verdicts

**FTS5 injection via `search_any` — HOLDS.** Each term is individually
double-quoted with `""` doubling and joined by a code-owned ` OR `
(`catalog_db.py:732`). FTS5 operators (`NEAR`, `*`, `^`, `AND`/`OR`/`NOT`,
`col:`) are only meaningful OUTSIDE quotes, and quote-doubling is FTS5's only
string escape, so no term can close its own phrase. Defence in depth: on the
refine path terms are structurally `[a-z][a-z-]{3,}` (`refine.py:772`) — the
operator characters are unrepresentable — and the call is wrapped in
`except Exception` (`refine.py:2583`), degrading to no offers rather than
failing the run. Both LIKE arms bind parameters through `_like_escape`
(escapes `\`, `%`, `_`) paired with `ESCAPE '\'` (`catalog_db.py:752-765`) —
no wildcard injection.

**Third-party text exposure — STRUCTURALLY CONTAINED.** `instruction_template`
reaches NO LLM context in this slice, and the exclusion is allowlist-by-omission
rather than incidental. Verified independently at all three projections:
`refine._SAFE_DESC_FIELDS` (`refine.py:630`) omits it; `refine.lora_metadata`
SELECTs explicit columns without it (`refine.py:734`); MCP `_handle_search`
(`mcp_server.py:2630-2640`) and `_enrich_from_catalog_db`
(`mcp_server.py:575-585`) both project through explicit key tuples that omit it.
The only LLM-visible effect is INDIRECT ranking influence via the 6.0 bm25
weight.

**Sanitization — NOT BYPASSABLE.** The template is derived inside
`upsert_description` from the same raw `trainedWords` list
(`catalog_db.py:568`), through the same `sanitize_text` path as `description`
(tag strip, entity re-strip, zero-width/bidi removal, NFC, cap). No parameter
lets a caller supply a template directly (ADR-022 §6).

**Migration — CORRECTLY ORDERED AND FAIL-CLOSED.** The version bump joins
`rebuild_fts`'s implicit DML transaction and commits last
(`catalog_db.py:415-417`); a crash leaves v1 so the migration retries, pinned by
the simulated-crash test. `connect_readonly` genuinely fail-closes on a v1 DB
(`catalog_db.py:439-443`) — `mode=ro` cannot migrate.

**Load plane — INTACT.** The AST test that `generate.py` / `server.py` never
import `catalog_db` is unchanged. `search_any` returns `e.*` including
`abs_path` exactly as `search` does, and every planner/MCP consumer projects it
away; name→path resolution stays with the ADR-015 in-memory resolver.

## Findings

**[LOW-1] The `instruction_template` exclusion is structural but UNPINNED.**
`refine.py:630`, `mcp_server.py:2630-2640`, `mcp_server.py:575-585`; no test.
The entire security posture of this slice is "512 B of third-party prose is
stored and indexed but never rendered into an LLM context" — and that holds only
because three independent allowlist tuples omit the key. Adding
`"instruction_template"` to any of them (the obvious slice-2 temptation, since
the ADR itself calls it "the phrasing the LoRA was trained on") would silently
promote third-party text into planner/agent context **with every existing test
green**. The ADR-037 review's lesson — "reviewed while dead ≠ reviewed while
live" — applies in reverse: the channel is deliberately dead and nothing pins it
dead.
*Fix:* negative assertions mirroring the existing `/secret` path sentinel.
**FOLDED — see Resolution below.**

**[LOW-2] Hostile-uploader ranking-steering channel widened.**
`catalog_db.py:79`, `:325-332`, `:815`. A hostile uploader whose LoRA the
operator holds locally can stuff a template-shaped `trainedWords` entry
(>64 chars, ≥4 spaces, low comma ratio — trivially satisfiable) with common
critique vocabulary ("realism skin texture detail lighting"), now stemmed and
weighted 12x above `description`, making their LoRA the top offer for most
critiques. Impact is bounded exactly as ADR-037's review bounded the description
channel: a bad OFFER whose name the planner may act on — image-content steering
only. F1/F2/F3 keep it off paths, config, and the load plane, and only
locally-present files have catalog entries at all. This is an escalation in
steering REACH (previously the top-weighted attacker text was 64 B per trigger
word at weight 3.0), not in AUTHORITY.
*Fix:* record the accepted bound. **FOLDED — ADR-041 Changelog.**

**[INFO-1] `search_any` can raise on degenerate terms / oversized term lists.**
`catalog_db.py:729-732`, `:752`. A punctuation-only term quotes to an empty FTS5
phrase and raises `OperationalError`; a very long `terms` list exceeds SQLite's
bind-variable limit. Both FAIL CLOSED (exception, never wrong rows).
Unreachable from `search_loras` and swallowed there, but `search_any` is a
public function and a future caller inherits the crash.
*Fix:* charset filter + clamp. **FOLDED.**

**[INFO-2] Concurrent v1→v2 migrators race without a busy timeout.**
Loud failure, correct end state — WAL's single-writer rule makes the loser raise
`database is locked`; the DB ends at v2 with populated FTS either way.
Operational noise, not a boundary failure. *Not folded* — adding a
`busy_timeout` is a catalog-wide behaviour change beyond this slice's scope.

**[INFO-3] `catalog_cli search` operator output includes the template.**
Same trust class rendered to the same human-only sink that already prints
`abs_path` and full `description`. No change in posture. Listed to complete the
exposure trace: planner (excluded), MCP (excluded), operator CLI (included,
acceptable), FTS ranking (included — LOW-2), logs/verdict records (not reached).

## Resolution (same-day fold)

- **LOW-1** — tripwire tests added in `test_refine.py` and `test_mcp_server.py`:
  a catalog row carrying `instruction_template` with a sentinel value must not
  appear in any offer view, in `build_judge_user_text` output, or in the MCP
  search payload. `_SAFE_DESC_FIELDS` carries a comment naming the ADR-041 D5
  condition that must be satisfied before the field is ever added.
- **LOW-2** — accepted bound recorded in the ADR-041 Changelog so slice 2's
  enrichment work inherits it.
- **INFO-1** — `search_any` now keeps only terms containing an alphanumeric and
  clamps the term list.
- **INFO-2 / INFO-3** — accepted as-is, reasons above.

No scope creep: the diff touches exactly the declared D2+D3 files. No auth,
crypto, or audit surface is involved.
