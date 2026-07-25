# Security review — ADR-009 CFG-knob aliasing fix (cfg_scale / true_cfg_scale)

AI-Disclosure: security-auditor and code-reviewer subagents ran on Claude Fable 5
(`claude-fable-5`), both explicitly confirmed in their reports — no model
fallback occurred. Parent session (Fable 5) folded the findings; Grant owns the
decision.

Scope: uncommitted diff making family-default application alias-aware in both
appliers — `generate._apply_family_defaults` and
`refine._overlay_family_defaults` skip the `true_cfg_scale` family default
when `cfg_scale` is explicit (value-aware) or iterated, so the router's
existing `--cfg` → true-CFG mapping is no longer defeated by the default.
`comfyless/refine.py` is a Red Zone path; `generate.py`'s Red Zone surface
`_run_json_mode` is untouched (verified — not a caller of the changed
function).

Live incident: `--cfg 1` on qwen-edit + a Lightning 8-step LoRA silently ran
double-pass true-CFG 4.0 (the family default won the slot the router
prefers) — CFG burn on a distilled setup, through every edit-mode refine
smoke that day.

## Disposition summary (parent session, folded same day)

| Finding | Severity | Disposition |
|---|---|---|
| Generate-side explicit test was key-PRESENCE-based: a replayed sidecar `"cfg_scale": null` (kept by `_validate_params`) would suppress the default and ride None into the pipeline — new crash corner (fail-closed, operator-initiated replay only) | LOW | **Fixed** — value-aware test (`cfg_scale in explicit_keys AND p_cur.get("cfg_scale") is not None`); iterated-axis presence stays as-is (elements shape-validated, never None). Pinned. |
| No end-to-end routing pin at the incident junction (`_build_call_kwargs` qwen branch prefers non-None true_cfg) | SHOULD | **Fixed** — pins: cfg 1.0 + suppressed default → `true_cfg_scale == 1.0`; explicit --true-cfg 6.0 outranks --cfg 1.0. |
| Refine suppression log fired misleadingly when BOTH knobs explicit | NIT | **Fixed** — log/skip gated on `base["true_cfg_scale"] is None`. |
| No structural guard against a future FAMILY_DEFAULTS entry declaring both knobs | NIT | **Fixed** — test asserts no family sets both. |
| Log wording said "explicit --cfg" for iterated/sidecar triggers | NIT | **Fixed** — "explicit/iterated". |

## Auditor verdicts (condensed)

**Q1 — no new authority on any untrusted surface.** MCP: an agent-supplied
`cfg_scale` now triggers suppression, but the machine boundary already lets
the agent set `true_cfg_scale` to any float or explicit null directly —
suppression is a strict subset; null `cfg_scale` is rejected at the boundary
(`_KIND_FLOAT`). Refine: the overlay's sole caller is
`build_config_from_args` (argparse-typed operator input);
`build_config_from_seed` never calls it, so a crafted sidecar/PNG chunk gains
zero suppression authority (and t2i seed mode's F4 full-schema authority
already covers `true_cfg_scale` directly). Planner output cannot reach
`cfg_scale` (F1 two-key allowlist) and the overlay runs before any LLM output
exists.

**Q2 — fail direction.** Every well-formed suppression leaves
`true_cfg_scale=None` with a non-None `cfg_scale`, which the qwen router maps
onto true CFG (the intended fix). The only degenerate pair was the LOW above
— closed by the value-aware guard.

**Q3 — logs injection-clean.** Both new lines interpolate repo-constant
family-default values and an int index only.

**Q4 — scope.** Exactly the five declared files; `_run_json_mode` and
`resolve_hf_path` untouched; ADR-009 mutation is a Changelog append.

Code-reviewer consumer sweep (condensed): CLI in-process, daemon delegation
(server.py applies no family defaults — pure pass-through of the
post-suppression wire value), `--iterate` (canonical axis names; bonus: qwen
`cfg_scale` sweeps were previously inert and now work), MCP (shared helper),
refine fresh-prompt + edit entries (the incident path), seed entry (overlay
not called — replay stays sidecar-authoritative). Fallback ordering verified:
refine's `_GEN_KEY_FALLBACKS` backfills after the overlay, so the 3.5
backstop cannot falsely suppress. No stale test pins of the defeated
behavior anywhere.

Test state at fold: test_params_schema 326 / test_refine 394, 0 failed;
battery re-run at commit.
