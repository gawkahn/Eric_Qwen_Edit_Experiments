# Security review — critique-driven LoRA offers (ADR-037 addendum, 2026-07-25)

AI-Disclosure: security-auditor and code-reviewer subagents ran on Claude Fable 5
(`claude-fable-5`), both explicitly confirmed — no model fallback occurred.
Parent session (Fable 5) folded the findings; Grant owns the decision.

Scope: uncommitted diff to `comfyless/refine.py` (Red Zone path) + tests + the
ADR-037 changelog addendum. The refine loop now feeds the PREVIOUS iteration's
judge critique — untrusted LLM output, F7-validated to string values, sliced
to 2000 chars — into `search_loras` as prepended FTS keyword material (cap
8→10), so quality/fix-it LoRAs surface on flaw words the operator prompt never
contains. No topical filtering by design (Grant: body/NSFW LoRAs often
genuinely improve skin texture/realism; the judge decides relevance).
Enhance-based prompt rewording evaluated and deliberately NOT wired.

## Disposition summary (folded same day)

| Finding | Severity | Disposition |
|---|---|---|
| Judge-steered catalog-description feedback loop: critique keywords can crowd out prompt keywords and pull chosen third-party descriptions into the judge's own next context | INFO | **Accepted by design** (Grant's no-filtering call; provenance label + F1/F2/F3/F6 boundaries hold; offers stay advisory). Hardening option recorded: reserve a floor of prompt-derived keyword slots. |
| No length cap on critique before tokenization (hostile endpoint could deliver ~8 MiB; re.findall materializes full match list) | INFO | **Fixed** — `prev_critique_text` sliced to 2000 chars at assignment. |
| New critique-path test lacked the `/secret` path sentinel | INFO | **Fixed** — sentinel added. |
| 10-term cap unpinned (code-reviewer PROMISE DRIFT) | SHOULD | **Fixed** — 8-filler-critique test fails under a cap of 8. |
| Threading test: dead primary path + spy restore not in finally (code-reviewer LOW×2) | NIT | **Fixed** — single direct-drive path, try/finally restore. |

## Verdicts (condensed)

**Q1 — injection: NO.** Two independent sufficient layers: `_offer_keywords`
emits only `[a-z][a-z-]{3,}` tokens (FTS5 operators, quotes, LIKE wildcards,
uppercase keywords unrepresentable), and `catalog_db.search` phrase-quotes +
escapes + parameterizes on a `mode=ro` connection. The prior review's
operator-prompt analysis (review-adr-037-lora-offers-2026-07-25.md) transfers
verbatim — same tokenizer, same call site, source-blind.

**Q2 — self-steering: no new AUTHORITY (the judge could already propose any
catalog name directly per F2); one new DISCOVERY capability (a ~10-query×5-row
read-only catalog search oracle over non-sensitive metadata), rated INFO and
accepted. Laundering closed: `include_excluded` unreachable (excluded/stale
entries stay hidden); paths dropped pre-merge by `_safe_lora_view` allowlist.**

**Q3 — lifecycle: clean.** `prev_critique_text` has exactly one consumer (the
search call); it never enters history records, judge context as raw text,
sidecars (the critique in `*.verdict.json` predates this diff), or logs.
Judge-error iterations retain the last USABLE critique — correct, same trust
class.

**Q4 — resource: bounded.** ≤ ~60 read-only queries/iteration, iterations
capped at 100, critique input now capped at 2000 chars.

Test state at fold: test_refine.py 410 passed / 0 failed (re-run at commit);
battery 29/29.
