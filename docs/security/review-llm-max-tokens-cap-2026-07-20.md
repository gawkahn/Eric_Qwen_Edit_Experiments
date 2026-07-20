# Security Review — LLM response cap (`max_tokens`) on enhance + refine judge wires

AI-Disclosure: Claude (Fable 5) performed this review via the `security-auditor`
agent; Claude (Fable 5) authored the reviewed change; Grant reviewed.

- **Date:** 2026-07-20
- **Scope:** uncommitted slice touching `comfyless/enhance.py`,
  `comfyless/refine.py` (Red Zone path), `test_enhance.py`, `test_refine.py`
- **Change:** always-emitted OpenAI `max_tokens` cap (default 1024) on
  (a) the enhance openai-endpoint payload via the `_resolve_endpoint_sampling`
  recipe > cfg > default knob machinery and (b) the refine judge/planner
  payload via `build_judge_payload` + a backend-cfg override validated in
  `judge_candidate`. Motivation: a model that misses its stop token churns KV
  cache until the 120 s HTTP timeout.
- **Verdict:** **approve-with-notes.** Net security improvement; both MEDIUM
  findings were fixed in the same slice (see Disposition).

## Auditor findings (verbatim summary)

**Trust boundary (Q1):** No new injection/type-confusion path. The value
originates only from operator-owned config (recipe TOMLs, `enhancers.toml`);
`parse_verdict`'s override allowlist (`prompt`, `loras` only) means LLM output
cannot reach `backend_cfg` or the sampling resolver. The value is a validated
`int` serialized by `json.dumps` under a fixed key.

**F-series invariants (Q2):** Respected. No load-plane keys added (F3); the
new `RefineError` raises inside `judge_candidate` and is consumed by
`refine_loop`'s per-iteration `except RefineError` (F7).

**Truncation (Q3):** Fails closed. A mid-JSON cut leaves `_extract_json_block`'s
brace slice unbalanced → `json.JSONDecodeError` → `RefineError` → iteration
consumed. The only parseable cut lands on a balanced-brace boundary, where a
missing `verdict` coerces to `"revise"` (conservative) and missing `overrides`
apply nothing. No path to accepting a partial verdict as `pass`. (No
`finish_reason == "length"` check — observability nicety, not a security gap.)

**Cap disablement (Q4):** Cannot be disabled: always emitted on both wires,
TOML cannot express null, 0/negative/bool rejected loudly.

### [MEDIUM] Huge recipe/cfg `max_tokens` crashes enhance with raw OverflowError
`_coerce_sampling_value`: TOML ints are arbitrary-precision; `float(10**400)`
raises `OverflowError`, uncaught by the `(TypeError, ValueError)` tuple —
escapes the EnhanceError contract. Pre-existing latent bug for `top_k`,
extended by this slice to a knob where large values are plausible.

### [MEDIUM] Invalid judge `max_tokens` burns full GPU generations per iteration
Config typo detected only inside `judge_candidate`, after generation; with
default `patience=0` the loop burns all `max_iterations` generations against a
statically-knowable config error. Remediation: mirror the check in `main()`'s
startup validation block. (Same failure shape as review slice-3 MEDIUM-2.)

### [INFO] No upper bound on the cap
`max_tokens = 100_000_000` passes silently, reinstating the KV-churn exposure.
Operator-owned files, matches the project's "warn, don't block" preference —
footgun, not an exploit path. No action taken.

### [INFO] Coercion-layer label misattributes recipe-sourced values
`_coerce_sampling_value(..., "backend cfg:")` labels recipe-sourced errors as
cfg-sourced. Pre-existing pattern (temperature identical); recipe values are
already coerced in `load_recipe`, so the mislabel is unreachable in practice.
No action required.

## Disposition (fixes applied in this slice, post-review)

1. **MEDIUM-1 fixed:** `OverflowError` added to `_coerce_sampling_value`'s
   except tuple (`comfyless/enhance.py`); negative test added
   (`max_tokens = 10**400` → `EnhanceError`, never raw `OverflowError`).
2. **MEDIUM-2 fixed:** `main()` now validates `backend_cfg["max_tokens"]` in
   the startup judge-backend block (exit 2 before any generation), keeping the
   `judge_candidate` check as defense in depth.
3. **code-reviewer LOW fixed:** `judge_candidate`'s validation moved above the
   model-autodetect fallback so a static config error never costs a live
   `GET /models`.
4. **code-reviewer LOW (test gap) fixed:** recipe-layer non-positive
   `max_tokens` negative added alongside the cfg-layer negatives.

Companion code review (`code-reviewer`, Fable, same date): no blocking
findings; scope clean; `build_judge_payload` signature change backward
compatible; hunyuan-reprompt confirmed untouched (`_REPROMPT_MAX_NEW_TOKENS`
path); always-emitting is wire-safe (`max_tokens` is OpenAI-standard, unlike
the conditional vLLM-extension knobs). Accepted asymmetry: enhance's shared
coercer keeps its long-standing numeric-string/integer-float acceptance; the
judge requires a bare TOML integer (stricter, safe direction; documented in
the vault docs).

## References

- Design: `docs/decisions/ADR-027-comfyless-refinement-loop.md` (judge surface),
  ADR-026 (enhancer registry/knob machinery)
- Prior chain: `docs/security/review-refinement-loop-*.md`,
  `review-comfyless-server-2026-04-23.md`
