# Security review — parity slice 1: shared family-defaults applier + refine --schedule (2026-07-25)

AI-Disclosure: security-auditor and code-reviewer subagents ran on Claude Fable 5
(`claude-fable-5`), both explicitly confirmed — no model fallback occurred.
Parent session (Fable 5) folded the findings; Grant owns the decision.

Scope: uncommitted diff unifying the ADR-009 family-defaults fill loop into
`family_defaults.apply_family_defaults()` (both `generate._apply_family_defaults`
and `refine._overlay_family_defaults` become adapters supplying `is_pinned` /
`has_value` / `is_eligible` predicates), plus the `--schedule` port to refine.
`comfyless/refine.py` is a Red Zone path; `generate.py`'s function-scoped Red
Zone surfaces (`_run_json_mode`, `resolve_hf_path`) verified untouched.

Motivation: the CFG-knob aliasing bug (2026-07-24) had to be found and fixed
in BOTH implementations with subtly different predicates — the values were
shared via FAMILY_DEFAULTS but the LOGIC was not.

## Disposition summary (folded same day)

| Finding | Severity | Disposition |
|---|---|---|
| `schedule` has no VALUE allowlist at the machine boundary (`_KIND_STR` only): an out-of-set name from `--json`/MCP/sidecar passes, silently shapes as linear, and the log + sidecar then RECORD a schedule that did not run | LOW (pre-existing; slice widens the schedule-carrying request population) | **Deferred with a named trigger** — TECH_DEBT 2026-07-25. CLI is unaffected (argparse choices-gated both sides). Fix has two shapes worth choosing deliberately (boundary value-allowlist vs returning the EFFECTIVE name). |
| Both-knobs invariant (`--cfg` + `--true-cfg`) now rests purely on statement order inside the shared applier; the old refine guard is gone and nothing pinned it | SHOULD (code review) | **Fixed** — test pins that both pinned knobs are excluded from `applied` AND that zero suppression lines are logged. |
| Comment claimed refine "had NO --schedule so every generation ran linear" — inaccurate: seed-image replays already carried the sidecar's recorded schedule through both the cold call and the wire builder; only fresh CLI entry was schedule-blind | NIT (code review) | **Fixed** — comment corrected so future archaeology doesn't conclude seed replays were broken pre-slice. |
| "local: avoid import cost" comment misleading — the cost is MOVED to parser-build, and `--help` now pays generate's heavy import | NIT (code review) | **Fixed** — comment corrected. |
| `--schedule` port had no end-to-end pin (a regression dropping `"schedule": args.schedule` from the base dict would pass the whole suite) | NIT (code review) | **Fixed** — pins the value through `build_config_from_args` → `to_generate_params()`, which is exactly what `_build_server_request` and the cold call read. |
| Applied-keys log line changed from insertion order to `sorted()` | INFO | Accepted — no test pins order; deterministic ordering is preferable. |
| In seed mode `--schedule` is silently ignored (only `--model` overrides the seed) | INFO | Pre-existing pattern, consistent with `--steps`/`--cfg`. |

## Auditor verdicts (condensed)

**Q1 — no trust/authority delta.** The premise that seed-sidecar keys feed
`is_eligible` is FALSE in the current tree: `build_config_from_seed` never
invokes the overlay — its sole call site is `build_config_from_args`, where
`base` is a fixed argparse-shaped key set. Even hypothetically, `is_eligible`
is byte-equivalent to the old test, and eligibility only gates which
CODE-OWNED FAMILY_DEFAULTS values land on keys the sidecar already has full
schema authority to set directly — strictly weaker than authority it holds.
Generate-side predicates reproduce the deleted ones exactly, including the
value-aware cfg-null masking rule. **Delta: zero.**

**Q2 — `--schedule` is gated; the ungated sidecar path pre-dates the diff.**
argparse choices-gated to `SCHEDULE_NAMES`. A seed sidecar could already carry
an arbitrary schedule string before this diff (the cold path and
`_build_server_request` both read it from the params dict), and
`build_sigma_schedule` falls back to linear for unrecognized values — fail-safe,
no dispatch, no eval. **New unvalidated paths introduced: zero.**

**Q3 — no injection/leakage via `log`/`prefix`/`family`.** The `family={...}`
interpolation is reachable only when the family is a member of the closed,
code-owned FAMILY_DEFAULTS key set (an attacker-shaped `_class_name` yields no
entry → early return, no log). Applied values are code-owned constants rendered
with `!r`; `prefix` is a caller literal at both sites.

**Q4 — scope clean.** generate.py hunks touch only the import block and
`_apply_family_defaults`; `_run_json_mode` and `resolve_hf_path` unmodified;
MCP behavior preserved by predicate equivalence.

Code-reviewer verdict: no PROMISE DRIFT, no SECURITY REGRESSION, no BOUNDARY
VIOLATION, no SCOPE CREEP; "code-level parity is exact on both sides."

Test state at fold: test_refine.py 432 passed, test_params_schema.py 326
passed, full battery 29/29.
