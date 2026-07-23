AI-Disclosure: Claude (Fable 5) authored — security-auditor subagent, verified
no model fallback (all 23 transcript API calls on claude-fable-5); companion
code-reviewer likewise fallback-free (31 calls). Grant reviewed.

Fold status: LOW-1 fixed (deepcopy in apply_overrides + derived-direction
covered by the snapshot tests), LOW-2 fixed (critique-sentinel negative),
coverage gap fixed (producing-prompt pin). Reviewer SHOULD-1 folded
(LoopOutcome.aborted + exit 3), SHOULD-2 folded (_resolve_max_iterations seam
+ tests), NIT-1/2/3/5 folded. INFO-3/4/5/6 accepted without action (INFO-4's
label alignment applied to the ADR text). Post-fold: test_refine 313/313,
battery 29/29, pyright at baseline.

# Security Review — ADR-037 Slice A Implementation (trajectory core, t2i)

**Date:** 2026-07-23
**Type:** §12 Red Zone implementation review (pre-commit), `comfyless/refine.py` (`_red-zone-paths.sh`-gated)
**Scope:** Uncommitted working-tree state of `comfyless/refine.py`, `comfyless/judge_recipes/generic.toml`, `test_refine.py`. Other dirty files out of slice per the brief.
**Authorities:** ADR-037 (D1–D3, D6, F8-P), `review-adr-037-design-2026-07-23.md` (Findings 1, 3, 6, 8, 9, 10, 13 binding on slice A), ADR-027 F1–F8.

## Threat model and review approach

Adversaries per the brief: (a) a compromised/steered judge LLM whose verdict JSON is the only planner channel; (b) crafted seed images/`--params` sidecars (the F4 channel); (c) injection text rendered inside generated images and read back by the multimodal judge — which in v2 gains a *persistence* vector, because iteration history re-enters every future judge call for the rest of the run. The slice's security surface is therefore: what gets written into the in-memory `history` list, how it is bounded and labeled before serialization into judge context, whether `best`'s config can be desynced or rebuilt from disk, and whether the new stop modes change the accepted spend bound. The F1 keystone (closed two-key override allowlist) and the F2/F3 planes were re-verified as unchanged, not re-derived.

What I actually did: read `comfyless/refine.py` in full (lines 1–1950) in its current working-tree state; read `generic.toml` in full; read the ADR-037 D1–D3 obligations and the seven binding design-review findings; traced every write into `history` (exactly two append sites plus one in-place flag mutation), every value source flowing into `history_record`, the serialization path through `prepare_history_for_context` → `build_judge_user_text` → `_assert_no_paths`, the snapshot/lineage machinery in `refine_loop`, the abort counter, and the CLI cap validation in `main()`; then read `test_refine.py:1286–1555` (the slice-A battery) and grepped the contract-pinning tests (`test_refine.py:1069–1131`). No Bash tool is available in this session, so this is an audit of the working-tree state of the three named files rather than a literal `git diff` walk; the v2 additions are cleanly delimited in the file (constants at 97–115, history machinery at 864–958, loop rewiring at 1263–1391, CLI at 1473–1483, 1807–1822).

## Coverage

Reviewed:
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/refine.py` — full file (1–1950)
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/comfyless/judge_recipes/generic.toml` — full file
- `/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments/test_refine.py` — lines 1286–1555 in full; contract/recipe tests (1069–1131) and loop fakes (647–770) via targeted grep+read
- `docs/decisions/ADR-037-refine-loop-v2-trajectory-edit.md`, `docs/security/review-adr-037-design-2026-07-23.md` — full

Not reviewed (and why):
- `comfyless/generate.py` / `comfyless/server.py` internals — unchanged in this slice; behavior taken from prior reviews
- Other dirty working-tree files — out of slice per the brief
- Literal `git diff` — no shell access in this session; audited file state instead

## Checklist verification (brief items 1–8)

1. **Finding 1 / F8-P:** `prompt_excerpt` provenance is decided by `prompt != target_prompt` (`refine.py:885`). This discriminator is *content-based*, and that makes it sound where it matters: a planner override that is byte-identical to the operator's target carries no payload the target didn't already carry, so labeling it "operator" grants nothing; any planner text that *differs* is labeled `planner-proposed (untrusted)`. Seed-image entry: `target_prompt` is the seed's embedded prompt (`refine.py:1800`), so iteration 0 is labeled "operator" for file-derived text — consistent with the ADR-027 F4 disposition (seed params are user-initiated authority, and the same text already enters judge context unlabeled as `target_prompt`), but see INFO-4. The 8 KiB planner-text budget is enforced oldest-first (`prepare_history_for_context`, 936–946) and only counts untrusted-labeled excerpts; operator-labeled text is bounded by the 64 KiB byte cap. No critique text is constructible into a history record (`history_record` never receives `verdict.critique`). **Holds.**
2. **Finding 9:** exactly two writes into `history`: `history.append(history_error_record(i))` at 1312 ({iteration, judge_error} only) and `history.append(history_record(...))` at 1388 (scores/excerpt/provenance/resolved-ops/flags). The full `str(e)` — endpoint URL + up to 300 endpoint-controlled bytes (`refine.py:449–450`) — goes only to the on-disk `*.verdict.json` (1311), which is never read back. History is in-memory only, never loaded from disk. Sentinel test pins it (`test_refine.py:1494–1508`). **Holds.**
3. **Finding 10:** `history_record(applied_ops=resolved_ops)` records `r.resolved_name` (NFC catalog name from the ADR-015 resolver) + parse-clamped weight only (894–896). Unresolvable names surface only in `resolve_lora_ops` notices, which go to `log`/stderr (1371–1373) and to the on-disk `verdict_record` — never into `history`. Steering-name negative pinned (`test_refine.py:1510–1521`). **Holds.**
4. **Finding 3:** `history_record` inputs are verdict ints, a float, `prev_prompt`/`target_prompt` strings, resolved ops, and two booleans — `WorkingConfig.base`, `Candidate.metadata`, sidecars, and `abs_path` are structurally unreachable (the record projection at 886–900 uses `resolved_name`, never `abs_path`). The history block rides `payload["iteration_history"]` and passes `_assert_no_paths(payload)` inside `build_judge_user_text` (978–980); record keys and `_history_stub` keys are disjoint from `_FORBIDDEN_CONTEXT_KEYS`. Construction test pins `/root/detail` absence (`test_refine.py:1325–1326`). **Holds.**
5. **Finding 6:** `snapshot_config` (653–663) deep-copies `base` and rebuilds `LoraSlot`s; `best_cfg` is assigned only from `snapshot_config(cfg)` at 1351 and never from disk. Ties are non-promotions (`comp > best.composite`, strict, 1345) and revert to `best_cfg` (1374) — lineage cannot fork; both pinned (`test_refine.py:1411–1428`). **Holds**, with one latent aliasing residue — LOW-1 below.
6. **Finding 8:** `consecutive_judge_errors` resets to 0 immediately after a usable verdict (1325); the abort check fires inside the error branch (1314–1319) *before* the next generation; patience (also incremented on error, 1320–1323) can only stop *earlier* — both stops fail closed with candidates kept. Exact-count and reset negatives pinned (`test_refine.py:1476–1492`). **Holds.**
7. **New surfaces:** elision markers (`"[elided: planner-text budget]"`, `…[truncated]`, `compacted: true`) are code-owned constants carrying no attacker text; `_history_stub` (912–921) carries scores+flags only. `generic.toml` contains rubric text only — no JSON output shape — and the code-owned `_JUDGE_OUTPUT_CONTRACT` is still appended unconditionally by `compose_judge_system_prompt` (797–802); `test_refine.py:1082–1096` still pins that a recipe cannot carry or strip the contract. `--until-score` is bounded: `main()` refuses `--max-iterations` outside 1..100 (1810–1815) and defaults until-score to the sanity cap (1818); the loop is `range(max_iterations)`. No exhaustion beyond the documented cap — but see INFO-5.
8. **F1/F7 regression sweep:** `parse_verdict` (261–352), `_coerce_score`/`_parse_lora_op` numeric hygiene, `_JUDGE_OUTPUT_CONTRACT` (751–763), and `apply_overrides` authority (697–741: prompt + add/remove/set_weight on resolved names only) are semantically identical to the ADR-027-reviewed surface; history adds zero planner-mutable keys (it is read-only context). **Holds.**

## Findings

**[LOW] 1 — `apply_overrides` shallow-copies `base`, so a reverted config aliases nested members of `best_cfg`'s snapshot — the D2 by-value invariant survives by accident, not by construction**
Location: `comfyless/refine.py:741` (`base=dict(cfg.base)`), interacting with `refine.py:1374,1383` and `snapshot_config` at 653–663
Risk: On the non-promoted path, `cfg = apply_overrides(best_cfg, ...)` builds the next working config with `dict(best_cfg.base)` — a *shallow* copy. Any nested mutable inside `base` (seed-image entry populates `base` from an arbitrary extracted sidecar dict, which legitimately carries nested lists/dicts such as `quant_skip`) is now shared between the live `cfg` and the by-value snapshot the D2 invariant depends on. Today no code mutates a nested `base` member in place (the only in-place write, `cfg.base["seed"]` at 1283, is top-level and occurs only at i==0, before any `best_cfg` exists), so there is no live exploit — but the invariant "snapshot immune to later mutation" currently holds only against top-level writes, and the negative test (`test_refine.py:1294–1306`) mutates the *original* config, never a config *derived from* the snapshot. A future slice that appends to a nested `base` list in place would silently desync `best` — exactly the audit-trail lie Finding 6 was written to prevent, and in slice B it would mean editing image X under a config recorded for Y.
Remediation: One line — `base=copy.deepcopy(cfg.base)` in `apply_overrides` (the module already imports `copy`; `base` is small). Add the derived-direction negative: snapshot, `apply_overrides` from it, mutate the derived config's nested `base` member, assert the snapshot unchanged.

**[LOW] 2 — D1's "past judge critiques are never replayed" is enforced by construction but has no sentinel negative test**
Location: `comfyless/refine.py:871–900` (construction); gap in `test_refine.py` (slice-A block, 1286–1555)
Risk: The binding D1 constraint — critiques are LLM-authored free text and must never re-enter future LLM context — currently holds because `history_record` simply doesn't take `verdict.critique`. The slice-A battery pins the two sibling constraints with sentinels (endpoint error text at 1494–1508, unresolvable op names at 1510–1521) but not this one. A future "give the planner more context" change that adds `critique` to the record (the single most natural enhancement to reach for) would reopen the persistent-steering channel with zero red tests — attack chain: image-rendered injection → steered critique → replayed into every remaining judge call for up to 100 iterations.
Remediation: Add one loop-level negative mirroring the Finding-9 test: a scripted verdict whose critique carries a sentinel string; assert the sentinel absent from every `histories_seen` entry.

**[INFO] 3 — `current_prompt` in the judge payload is unlabeled planner-authored text outside the F8-P budget (v1 carryover, ADR-acknowledged; label it for consistency)**
Location: `comfyless/refine.py:972` (`"current_prompt": cfg.prompt`)
Risk: The F8-P mitigations label and budget planner text *inside the history block*, but the payload's `current_prompt` field — planner-authored at every iteration after 0, up to `OVERRIDE_PROMPT_MAX_CHARS` (20 000 chars) — rides unlabeled and uncounted, so total planner-authored characters in one judge call can reach ~28 KiB against the nominal 8 KiB budget. ADR-037 D1 explicitly acknowledges the v1 `current_prompt` re-entry, and the rubric's "target_prompt is the only authority" line partially covers it, so this is a consistency note, not a new channel.
Remediation: Cheapest: add a fixed provenance note alongside the field (e.g. a `current_prompt_provenance` sibling using the same label string), or one clarifying sentence in the ADR that the F8-P budget scopes the history block only, with `current_prompt` bounded separately by `OVERRIDE_PROMPT_MAX_CHARS`.

**[INFO] 4 — Seed-image entry labels a file-derived prompt "operator"; ADR label string also differs slightly from the implementation**
Location: `comfyless/refine.py:885–893,1800`
Risk: In seed entry the iteration-0 excerpt is labeled `"operator"` although its true provenance is the seed image's embedded chunk (F4 channel). This grants nothing new — the identical text already enters every judge call unlabeled as `target_prompt`, and F4 disposes seed params as user-initiated authority — but the label overstates provenance precision. Separately, ADR-037 D1 specifies the label `"planner-proposed prompt (untrusted)"`; code, rubric, and tests consistently use `"planner-proposed (untrusted)"`.
Remediation: No code change needed; optionally note in the ADR Changelog that seed-derived targets are labeled "operator" per the F4 trust decision, and align the ADR's quoted label string with the implemented one (or vice versa) so future greps for the exact string don't miss.

**[INFO] 5 — `--pass-threshold` is unvalidated; an unreachable threshold under `--until-score` guarantees the full 100-iteration burn**
Location: `comfyless/refine.py:1471–1472` (no range check), `1810–1822`
Risk: Scores clamp to 1–10, so `--pass-threshold 11 --until-score` can never pass and deterministically burns 100 generations (bounded, within the D3-accepted spend envelope; judge-error abort does not fire because verdicts are usable). `--pass-threshold 0` passes on iteration 1. Operator footgun, not an attacker channel.
Remediation: Reject `--pass-threshold` outside `SCORE_MIN..SCORE_MAX` at startup (mirrors the `--max-iterations` check), or warn loudly per the project's warn-don't-block preference.

**[INFO] 6 — `lora_ops_applied` records resolved ops even when application was a no-op**
Location: `comfyless/refine.py:1371,1388–1391` vs `apply_overrides` no-op branches at 722, 725
Risk: An op that resolves but doesn't change the config (add of an already-active LoRA, remove of an inactive one) still enters history as "applied." Names remain catalog-bounded (no injection surface), but the trajectory context slightly overstates what changed, which can mislead the planner into believing a failed change was tried — a mild spend inefficiency against the exact signal D1 exists to provide.
Remediation: Pass only the ops that mutated the config (have `apply_overrides` return them, or filter on the notices), or accept and note in D1 that `lora_ops_applied` means "resolved and submitted," not "state-changing."

## Slice-A negative-test coverage assessment (brief item 9)

Well pinned: snapshot immutability (original-direction), tie reversion, climb-from-best re-derivation, no-history-at-iter-0, single `is_best`, budget oldest-first elision + loudness + non-mutation of originals, byte-cap stub compaction, error-record shape, endpoint-sentinel exclusion, unresolvable-name exclusion, abort count + counter reset, sanity-cap CLI refusal (both bounds), contract-in-code survival against a hostile recipe. Gaps: the derived-config aliasing direction (LOW-1), a critique-sentinel negative (LOW-2), and no assertion that a record's `prompt_excerpt` is the prompt that *produced* the candidate (`prev_prompt`, captured at 1381) rather than the post-override prompt — a regression there would mislabel provenance timing; cheap to pin alongside LOW-2.

## Verdict

**Findings requiring fold — LOW only.** No CRITICAL, HIGH, or MEDIUM. Every binding design-review obligation (Findings 1, 3, 6, 8, 9, 10, 13) is implemented as specified and negatively tested, the F1/F2/F3/F7 keystones are unchanged, and the recipe file cannot touch the code-owned contract. The two LOW items are a one-line `deepcopy` in `apply_overrides` plus two small sentinel tests — small enough to land inside this slice before commit; if deferred instead, each needs a TECH_DEBT entry per §12. The four INFO items require no action for merge.
