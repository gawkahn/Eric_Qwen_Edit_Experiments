# Security Review — refine family-defaults overlay + scan announce gating

AI-Disclosure: security-auditor subagent (Fable 5) performed the audit; Claude (Fable 5) authored this record; Grant reviewed.

**Date:** 2026-07-18
**Scope:** uncommitted diff to `comfyless/refine.py` (Red Zone path — ADR-027
refinement loop), `nodes/eric_diffusion_utils.py`, `comfyless/catalog.py`
**Change:** (1) refine's `--steps/--cfg/--width/--height` CLI defaults become
`None` sentinels; new `_overlay_family_defaults` fills unset keys from
`FAMILY_DEFAULTS` (family via `detect_pipeline_class(base["model"])`), then
backstops from `_GEN_KEY_FALLBACKS`; called only from `build_config_from_args`
(fresh-CLI entry). (2) `infer_model_family` gains keyword-only
`announce: bool = True` gating the "Z-Image Turbo inferred" INFO print;
`catalog.scan_model_family` passes `announce=False`.
**Prior chain:** `review-refinement-loop-*.md`, ADR-027, ADR-009.

## Verdict

**Security-neutral. No findings at CRITICAL/HIGH/MEDIUM.** Two INFO
observations, neither requiring changes in this slice.

## Audit-question answers

1. **No new untrusted channel.** Overlay values are in-repo constants
   (`FAMILY_DEFAULTS` / `_GEN_KEY_FALLBACKS`); file content
   (`model_index.json`) can at most *select among* code-constant parameter
   sets, never inject arbitrary values (unknown `_class_name` → lowercased
   fallback → `.get()` → `{}` → backstops; closed on both ends). Selection is
   gated only by the operator-supplied `--model` path. `build_config_from_seed`
   does **not** call the overlay — seed-path authority unchanged; missing seed
   gen keys still default inside `run_generation`. The planner F1 two-key
   allowlist (prompt + loras) is untouched, so overlaid keys are immutable
   post-startup. Family keys refine has no flag for (`refiner_steps`,
   `refiner_cfg`) are excluded by the `key in base` guard and cannot ride into
   the daemon request.
2. **File-read risk:** the overlay adds an earlier read of the same
   `model_index.json` the load path reads moments later — no material delta
   (see INFO-1).
3. **Announce suppression is safe.** The false-positive alarm (code-review
   finding 3) was always about the *load* path; `detect_pipeline_class` passes
   no `announce` kwarg and still announces. Scan-derived family strings feed
   only catalog metadata, never generation defaults.
4. **F4/F5/root-containment unweakened.** Loud echo, `_root_flag` flagging,
   byte/pixel caps, seed-prompt char cap, and ADR-015-only LoRA resolution are
   byte-identical to the pre-change file.

## INFO findings

- **INFO-1 — earlier uncapped `model_index.json` read.**
  `detect_pipeline_class` has no size cap / symlink check (unlike
  `scan_model_family`'s `_MAX_INDEX_BYTES` gate). Pre-existing and shared with
  every load path; `--model` is operator CLI input, not an LLM/seed channel.
  Worst-case TOCTOU between overlay-time and load-time reads is wrong family
  defaults — bad image params, never a path/exec channel. If ever closed, fix
  once in `detect_pipeline_class` (adopt the capped-read pattern). Related
  debt: `resolve_hf_path` §12 review still outstanding (CLAUDE.md Review bar).
- **INFO-2 — Turbo-inference notice prints twice on the refine fresh-CLI
  path** (overlay-time + load-time, and the overlay-time print goes to stdout
  via `print`, bypassing refine's `log` callable). Redundant, not weakened.
  Optional polish, out of this slice's scope.

## Error-path note

`_overlay_family_defaults` catches `(ValueError, OSError, AttributeError)` and
degrades to backstops — fail-safe for this surface (worse image params, never
a path or execution channel). The `AttributeError` arm (non-object JSON top
level) was added post-audit on the code-reviewer's LOW finding; it narrows an
operator-facing crash, no security delta.

## Patience addendum (same day, separate slice)

**Change:** `DEFAULT_PATIENCE` 2 → 0; `patience <= 0` disables the
no-improvement early stop at both check sites in `refine_loop` (composite
stall + unusable-verdict). Loop is then bounded by `verdict_passes` and
`max_iterations` only; `--patience N` opts back in.

**Verdict: security-neutral** (security-auditor, Fable, 2026-07-18).
- **Spend:** all four loop paths (pass / cap / judge-error continue /
  apply_overrides) terminate inside the structural
  `for i in range(max_iterations)`; no retry logic exists anywhere in
  refine.py; LLM output cannot reach `max_iterations` or `patience` through
  the F1 allowlist. Worst case is exactly `max_iterations` generations and
  judge calls.
- **Authority:** more iterations = more applications of the same per-verdict
  authority, not new authority — F1/F2/F3/F6 gates apply identically each
  iteration, and a score-crafting adversary could already reach the cap under
  `patience=2` by reporting monotonic sub-threshold improvements. Malformed
  verdicts never reach `apply_overrides`.
- **Daemon:** one unchanged validated wire request per iteration; containment
  identical.
- **INFO (accepted):** a persistently broken judge now burns up to
  `max_iterations` identical generations (seed pinned, config unchanged)
  instead of stopping after 2 — bounded, loud in logs, ^C-recoverable;
  documented in the ADR-027 changelog. Optional future hardening: a
  consecutive-judge-failure fail-fast distinct from patience.
- **INFO (accepted as documented behavior):** negative `--patience` now
  disables rather than stopping after iteration 1; the `<= 0` contract is
  documented in the docstring/help and pinned by a `patience=-1` test
  (warn-don't-block house style — no argparse rejection).

## Post-audit deltas folded into the slice (code-reviewer findings)

- Guard test in `test_refine.py`: every `FAMILY_DEFAULTS` key must be known to
  the refine overlay (prevents silent refine/generate divergence when a family
  gains a new key). (MEDIUM)
- `AttributeError` in the overlay's except tuple. (LOW)
- Docstring records that generate's distilled-transformer warning is
  intentionally absent (refine exposes no transformer override). (LOW)
