# Security Review — ADR-015 Slice 3b: MCP cascade `generate` catalog-name migration

**AI-Disclosure:** Reviews authored by Claude (Opus 4.8, 1M context) acting as
`code-reviewer` and `security-auditor`; Grant reviewed. Both agents Opus,
model pinned at invocation per global §5A (the frontmatter pin is known-broken
in Claude Code 2.1.117).

**Date:** 2026-06-24
**Slice:** ADR-015 slice 3b (cascade catalog-name migration), two steps.
**Surface:** comfyless MCP server — `comfyless/catalog.py` `resolve_reference`
(step 1) and `comfyless/mcp_server.py` `_handle_generate_cascade` (step 2).
Red Zone (the ADR-015 catalog reference contract — the LLM-agent input/output
boundary).
**Commits:** step 1 `7a92e2f`, step 2 `48a0833`. Vision:
`docs/vision/slice-3b-mcp-cascade-catalog.md`.

---

## Step 1 — `resolve_reference` tuple-kind extension

**code-reviewer: APPROVED.** Bare-`str` path provably byte-identical
(`x not in (s,)` ≡ `x != s`); the `isinstance(expected_kind, str)` guard keeps
a str from being iterated into characters; all three production callers pass
bare-str literals so the slice-3 non-cascade path is unaffected. Degenerate
`expected_kind` (empty tuple, `None`-in-tuple) fails closed. Uniform-error /
no-leak / no-kind-oracle invariants preserved for the tuple path. Six new unit
cases adequate.

**security-auditor: CLEAN** (no HIGH/MEDIUM/LOW). The membership rewrite only
ever *narrows* what resolves; `KindMismatch` is emitted identically for str and
tuple and folds into the single uniform frame at the handler chokepoint, so
wrong-kind stays byte-indistinguishable from a miss — no new oracle. Failure
path returns `abs_path=None` (no leak). Empty tuple → deny-all (fail-closed).
Request-time existence + `_within` (steps 5–6) unchanged and unconditionally
reached once the kind gate passes. INFO only: a non-iterable `expected_kind`
would raise `TypeError`, but `expected_kind` is handler-controlled (never
agent-supplied) and a crash on operator misconfiguration is the correct
fail-closed posture; no action.

---

## Step 2 — `_handle_generate_cascade` catalog-name migration

**code-reviewer: CHANGES REQUIRED → folded.** One finding: the Vision declared
9 cascade negative cases; 8 were present — the cascade-level **PathMoved** case
(catalog hit whose `abs_path` vanished post-spawn) was missing (the existing
PathMoved coverage was resolver-level / non-cascade). Folded by adding one
cascade test (`os.remove` the resolved stage fixture, assert uniform message +
`PathMoved` audit cause); no production-code change. Verification points all
cleared: `catalog.py` was step 1 (already committed `7a92e2f`), not an
undeclared third file in the step-2 diff; the latent loras-in-cascade
validator inconsistency is pre-existing and harmless (cascade never consumed
loras). Confirmed correct: handler mirrors slice-3 `_handle_generate`;
removed-field guard checked on `raw_cc` before `validate_config` setdefault; no
abs_path leak to response; stage_a optionality; load-boundary `_within`;
`allow_hf_download=False`; no scope creep.

**security-auditor: CLEAN** (no HIGH/MEDIUM/LOW/INFO). Verdict reproduced:

> This diff migrates `_handle_generate_cascade` from the slice-1 raw-path
> contract to opaque catalog-name reference resolution, mirroring the
> already-audited non-cascade `_handle_generate`. **Overall posture: the diff
> upholds every security claim. CLEAN.**

Claims verified against the code (not the comments):

1. **No abs_path crosses the boundary.** Traced `resolved_cc` (abs_path-bearing)
   → `metadata["cascade_config"]` → both sinks. The response renderer
   `_resolved_cascade_params_as_names` overwrites each `stage_*` present in
   `stage_names` with the catalog name and pops the rest plus `scaffolding_repo`;
   the only four path-carrying keys a validated cascade config can hold
   (`stage_c/b/a`, `scaffolding_repo`) are all name-substituted or dropped.
   Empirically confirmed by the "no `/`-bearing value" test. PNG sink
   (`redact_metadata_for_png`, basenames) is a separate on-disk sink, unchanged.
2. **Uniform-error contract (HIGH-1) extended to cascade.** Every
   `resolve_reference` failure → `_reference_error(rr.cause)` → single
   `_UNIFORM_REFERENCE_ERROR` constant; fine cause only on the stderr audit. The
   slice-1 distinct agent-facing errors are gone. The pre-resolution
   `validate_config` failure reports only the exception class on cascade-config
   *structure*, fires before resolution, and cannot distinguish two reference
   values — not a reference oracle.
3. **scaffolding_repo input surface closed.** Rejected on `raw_cc` before the
   `setdefault` masks it; `resolved_cc` carries only the operator default; the
   resolution loop never passes `scaffolding_repo` to `resolve_reference`.
4. **Notice sanitization (INFO-2).** `_discard_notice(rr.name)` — never the raw
   value. Proven non-echoing by test (`/etc/passwd` and `/some/agent/dir` cases).
5. **Defense-in-depth retained.** `allow_hf_download=False` hard-coded at
   `build_pipelines`; load-boundary `_within` re-check on every resolved stage;
   resolver's own request-time `os.path.exists` + `_within` still reached.
6. **No new kind-set oracle.** Catalog kinds are exactly {model, lora,
   transformer}; the stage set excludes lora; no `component` kind exists.

> **Adversarial constructions attempted, all blocked:** (a) abs_path/full
> directory into the response; (b) distinguishing two failure causes from the
> agent frame; (c) smuggling attacker directory text via notice or error;
> (d) influencing the scaffolding path. I could not construct a working input
> for any of the four.

---

## Disposition

Both steps land CLEAN after the one code-reviewer finding (PathMoved coverage)
was folded. Slice 3b extends the ADR-015 §2 step-2 / HIGH-1 uniform-error
contract to the cascade path and closes the slice-1 cascade-side enumeration
oracle as designed. No deferred findings; no TECH_DEBT entries opened.
