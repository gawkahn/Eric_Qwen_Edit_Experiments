# Security Review — ADR-027 Refinement Loop, slice 3 (loop controller)

**AI-Disclosure:** Reviewed by the `security-auditor` agent (Claude Fable 5, pinned per §5A); Grant reviewed.
**Date:** 2026-07-13
**Subject:** `comfyless/refine.py` slice-3 section (loop controller, daemon-aware generation, CLI) + slice-3 tests in `test_refine.py`
**Verdict:** **APPROVE** — all six keystone invariants and both slice-2 forward constraints hold in slice 3. One LOW (contract deviation in the transport error path, fail-closed direction) and INFO observations only. No CRITICAL/HIGH/MEDIUM security findings.

> Note: a parallel `code-reviewer` (Fable) pass returned **needs-changes** on correctness grounds (1 HIGH daemon-savepath re-rooting, 2 MEDIUM, test gap, LOWs). Those are NOT security findings but were all folded before commit — see the ADR-027 Changelog "slice 3 landed" entry. This document is the security record.

## Summary

Slice 3 adds the greedy hill-climb controller: `WorkingConfig` (prompt + LoRA slots + opaque trusted `base`), `apply_overrides` (pure merge of resolver-validated ops), `run_generation` (daemon-first via the canonical `generate._build_server_request` wire builder, cold in-process fallback), `judge_candidate` (downscale → data-URI → POST → `parse_verdict`), path-free `verdict_record`, `refine_loop`, and the CLI. Threat model per ADR-027: §12 machine-driven-params surface, not §5 Red Zone — the only untrusted-in-the-relevant-sense actor is the LLM judge's output (and transitively, image content steering it); the judge endpoint, CLI args, catalog roots, and the daemon are operator-trusted. The two security-truth boundaries slice 3 touches are (1) verdict → next generation config, and (2) in-memory state → LLM-visible / disk artifacts.

## Keystone verification (the six questions)

**1. F1 — closed two-key allowlist: HOLDS.** Slice 3 never accepts a path from the verdict. `apply_overrides` mutates only `prompt` (from `verdict.override_prompt`, length/type-gated at parse) and the LoRA slot list, and only from `resolved_ops`; its only caller passes `resolve_lora_ops` output. `cfg.base` — carrying `model` and every other path-adjacent field — is built exclusively from CLI args, copied unmodified on every merge, and its single later mutation is the seed pin from generation metadata (trusted daemon/in-process generate, not the LLM). `to_generate_params` constructs `loras[].path` from `LoraSlot.abs_path`, only ever populated from `ResolvedLoraOp` output. The LLM cannot influence model/transformer/vae/te paths.

**2. F2 — resolver-only name→path: HOLDS.** Both name→path sites (`refine_loop`, `_resolve_startup_loras`) go through `resolve_lora_ops` → ADR-015 `resolve_reference(..., expected_kind="lora")`. Slice-3 code does not defeat the AST guard: its only constant-subscript reads are `params["model"]`, `params["prompt"]`, `cfg.base["seed"]`, `outcome.metadata["seed"]` — none in `{abs_path, root, relative_path}`; it adds no SQL.

**3. F3 + forward-constraint (a) — RULING: the load-plane sidecar is NOT a violation.** The two artifacts are correctly separated: `*.verdict.json` (built by `verdict_record`) carries the raw `LoraOp.name`, never `ResolvedLoraOp.abs_path`, and is gated by `_assert_no_paths` before write; `build_judge_user_text` renders active LoRAs as name+weight only and is gated the same way. The `<stem>.json` sidecar legitimately carries `loras[].path` — but it is the pre-existing generate-plane replay artifact (identical to what `_delegate_to_server` writes for a manual run), and nothing in the loop ever reads it back: the next judge call's context is assembled solely from in-memory `cfg` + DB-allowlisted metadata. "Planner-visible" is the operative word in constraint (a); this artifact is not planner-visible. Stripping paths from it would break the human `--params` replay channel for no security gain. Condition attached: slice 4 must ingest sidecars only through the F4 trusted-human channel and never route sidecar content into judge context.

**4. Forward-constraint (b) — HOLDS.** `resolve_lora_ops` notices (which embed `res.cause`, i.e. `PathMoved`/`WithinFailure` filesystem-drift state) go only to `log`; they are never appended to `verdict.notices`, never written to `*.verdict.json`, and never enter `build_judge_user_text`, whose inputs are `target_prompt` (CLI), `cfg` (prompt/names/weights), and `assemble_planner_loras` / `search_loras` output (DB-allowlisted metadata).

**5. F5/F6 — HOLD.** `judge_candidate` unconditionally routes through `downscale_for_judge` before `image_to_data_uri`; the timeout flows CLI → `refine_loop` → `judge_candidate` → `_post_judge`; the response read is byte-capped. The pass gate is numeric (`verdict_passes`) on F6-validated clamped ints, and deliberately ignores the advisory `verdict` string — a lying "pass" cannot self-promote. `load_seed_image_capped` is correctly unused in slice 3 (seed-image entry is slice 4).

**6. New daemon-wire surface — ACCEPTABLE; reliance is defense-in-depth, not primary.** LLM-influenced fields reaching the daemon: `prompt` (schema-validated at the socket boundary), LoRA `weight` (clamped |w|≤4 at parse), and LoRA `path` — but the last is resolver output, not LLM text. `_daemon_namespace` supplies exactly the attribute set `_build_server_request` reads. The cold in-process fallback, which skips daemon validation, receives paths exclusively from the resolver + trusted CLI, so the invariant does not depend on which plane executes. Named assumption: the daemon serving `--device` was started with a `--model-base` compatible with refine's; a mismatch fails closed server-side.

## Findings

**[LOW] Judge-endpoint JSON decode escapes the F7 "consume an iteration" contract** — `comfyless/refine.py` `_post_judge`. `json.loads(resp.read(...))` was inside the transport `try` catching only HTTPError/URLError/OSError — a non-JSON or truncated (byte-capped) body raises `json.JSONDecodeError` (a `ValueError`), which `refine_loop`'s `except RefineError` does not catch; the run would crash instead of consuming an iteration. Fail direction is closed. **RESOLVED 2026-07-13:** decode moved outside the transport `try`; `json.JSONDecodeError`/`UnicodeDecodeError` → `RefineError`.

**[INFO] Constraint (a) wording disambiguated in the ADR** — constraint (a)'s parenthetical "(sidecar, `*.verdict.json`, next-call context)" can be read as banning paths from the load-plane `<stem>.json` sidecar, which this review rules acceptable. **RESOLVED:** ADR-027 Changelog "slice 3 landed" clarifies constraint (a) means planner-visible artifacts only; the load-plane sidecar is exempt on condition it is never read back into judge context.

**[INFO] `_assert_no_paths` is key-shaped; string values are unchecked** — a filesystem path embedded in a metadata *value* (e.g. an operator/civitai `description` containing an absolute path) would pass into LLM context. Impact is information-leak-shaped only — F1 makes path knowledge useless to the planner. Acceptable defense-in-depth layering; catalog description text is operator-curated. No change required.

**[INFO] Daemon "ok-but-empty-output_path" fall-through** — if the daemon replied `status: ok` with no `output_path` (shouldn't occur), execution fell into the in-process path and generated a second time. **RESOLVED:** now raises `RefineError`; the `resp is None` connect-failure fallback now logs.

**[INFO] CLI help over-promised `--model` containment** — help said "must be within --model-base" but refine performs no such check (enforced only on the daemon plane; identical trust posture to `generate`). **RESOLVED:** help reworded to "validated against --model-base when a daemon is running".

**Scope creep:** none — all changes sit within the declared slice-3 section of `refine.py` plus tests.

## Bottom line

Approve. F1/F2/F3/F5/F6 and forward-constraints (a)/(b) all verifiably hold in slice 3; the pass gate is numeric and lie-proof; the two artifact planes are correctly separated. The single LOW and all INFO items were resolved before commit.
