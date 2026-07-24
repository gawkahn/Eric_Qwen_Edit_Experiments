# Security review — ADR-037 D2 amendment (tie-promotion + no-op seed resample)

AI-Disclosure: security-auditor and code-reviewer subagents ran on Claude Fable 5
(`claude-fable-5`), both explicitly confirmed — no Fable→Opus/Sonnet fallback
occurred in either transcript (each report carries its own model statement).
Parent session (Fable 5) folded the findings; Grant owns the decision.

Scope: uncommitted working diff (comfyless/refine.py loop-controller lineage
block, test_refine.py pins, comfyless/judge_recipes/edit-generic.toml rewrite,
ADR-037 changelog entry 2026-07-24). refine.py is a declared Red Zone path
(`scripts/git-policy/_red-zone-paths.sh`).

## Disposition summary (parent session, folded same day)

| Finding | Severity | Disposition |
|---|---|---|
| Tie-promotion lowers F8-E propagation bar to parity; patience=0 default means only --max-iterations bounds a constant-parity tie chain; winner shifts earliest-peak → last-tied | MEDIUM | **Accepted as documented risk** — recorded in ADR changelog + corrected code comment (refine.py promotion block). Deferred option: tie-streak cap (revert to best after N consecutive non-improving promotions) — Grant to decide if drift is observed live. |
| Non-monotonic no-op seed bump on decline cycles regenerates the identical image to the cap (code-reviewer SHOULD, borderline CRITICAL; auditor INFO "residual plateau") | SHOULD | **Fixed** — monotonic loop-level `noop_resamples` counter (`seed = source_seed + Nth_noop`); pinned by strictly-increasing-seeds decline-cycle test. |
| Edit-mode tie lineage + divergent history flags unpinned (code-reviewer SHOULD) | SHOULD | **Fixed** — test pins tie-advances-source and improved=False/is_best=True/accepted=True. |
| Rubric preamble + JSON vs 1024-token response cap: truncation fails closed but burns iterations | LOW | **Documented** — recipe header note; raise backend-cfg `max_tokens` on observed truncation. |
| `isinstance(_seed, int)` admitted bool | INFO | **Fixed** — bool excluded, matching `_coerce_score`/`_parse_lora_op`. |
| No test for preamble-shaped judge output through `parse_verdict` | INFO | **Fixed** — positive preamble parse + stray-`{` fail-closed negative added. |
| Code-owned "STRICT JSON and nothing else" vs recipe-mandated preamble contradiction | INFO | **Left standing** (parse-safe both ways; contract is code-owned and shared with t2i). Watch live smokes for preamble omission; softening the contract wording is its own future slice. |

Audit verdicts on the framed questions: Q1 tie-promotion — bar genuinely
lowered (the MEDIUM above); F1/F2/F7 boundaries intact. Q2 seed resample — no
planner channel into base params beyond a 1-bit no-op/not choice; bump value
code-computed; alias-safe vs best's snapshot. Q3 rubric preamble — discarded
by `_extract_json_block`, never persisted or re-injected; recipe cannot
override the code-owned output contract. Q4 — no log injection, no history
shape change beyond divergent boolean semantics, no scope creep beyond the
bundled D6 rubric rewrite (acknowledged in the ADR changelog).

Test state at fold: test_refine.py 362 passed / 0 failed; full battery 29/29
(re-run pending at commit).

---

## Full security-auditor report (verbatim)

The diff amends ADR-037 D2 in `comfyless/refine.py` (Red Zone path): (1) lineage promotion relaxes from strict `composite > best` to `>=` — a tie now advances both config lineage and, in edit mode, image lineage; (2) after `apply_overrides`, a derived config identical to its lineage source bumps `cfg.base["seed"]` by +1 so the loop explores instead of regenerating byte-identical images; (3) the operator-owned judge recipe `edit-generic.toml` is rewritten to DECOMPOSE-THEN-VERIFY, instructing the judge to emit a plain-text preamble before the strict-JSON verdict, relying on `parse_verdict._extract_json_block`'s first-`{`/last-`}` slicing. Threat model: the judge/planner is an untrusted LLM whose only sanctioned authority is prompt + LoRA-by-name (D4/F1/F2); text inside judged images is adversarial (F8-E); the recipe file and CLI are operator-owned. I traced the promotion path, the seed-bump guard, the verdict parse/persistence surfaces, and the history records bound for future judge context (F8-P). Overall the amendment preserves the parse boundary and planner-authority allowlists; the material change is a genuine weakening of the lineage-advancement bar, whose stated compensating control (patience) is disabled by default.

### Findings

[MEDIUM] Tie-promotion advances edit-image lineage with no demonstrated improvement, and the claimed compensating control is off by default
Location: comfyless/refine.py:1536-1552 (amended promotion), 89 (`DEFAULT_PATIENCE = 0`), 1530-1535 (comment)
Risk: Previously a candidate had to strictly beat `best` for its image to become the next edit source; F8-E (directive-looking text baked into candidate pixels swaying the judge) therefore had to *win*, not merely *hold parity*, to propagate. Now a sycophantic or adversarially-influenced judge that scores constant parity (the exact 9/9 checklist-echo behavior that motivated the rubric rewrite) promotes **every** iteration: `current_source = cand.image_path` fires on each tie, so cumulative unrequested drift — including F8-E payload content, which edit models preserve — compounds through the whole run, and the winner becomes the *last* (most-drifted) tied candidate instead of the earliest peak. The code comment and ADR changelog both lean on "patience still counts ties as non-improvement, so tie-promotion cannot defeat the early stop" — but `DEFAULT_PATIENCE = 0` disables the early stop entirely (default since 2026-07-18), so on a default invocation the tie-chain runs to `--max-iterations` with nothing bounding it but the cap. This is a real risk-profile change to F8-E (propagation bar lowered from strict-improvement to parity) that the ADR changelog does not acknowledge as such.
Remediation: Smallest change — record the F8-E bar-lowering explicitly in the ADR changelog and correct the refine.py:1533-1535 comment to state patience defaults to 0 (no early stop). If a mechanical bound is wanted: cap consecutive tie-promotions (e.g. a tie-streak counter that reverts to `best` after N consecutive non-improving promotions), which restores a drift ceiling without reverting the amendment.

[LOW] Rubric-mandated preamble vs unchanged 1024-token response cap — systematic truncation risk
Location: comfyless/judge_recipes/edit-generic.toml (MANDATORY PROCEDURE / DESCRIPTION / VERIFICATION sections) vs comfyless/refine.py:95 (`DEFAULT_JUDGE_MAX_TOKENS = 1024`, comment: "A verdict JSON is a few hundred tokens")
Risk: The rewritten rubric requires per-region DESCRIPTION lines plus per-requirement VERIFICATION lines *before* the few-hundred-token JSON. On a compound instruction this can routinely approach or exceed 1024 tokens; truncation mid-JSON fails closed in `_extract_json_block` (unbalanced slice → `RefineError`), but each such failure burns an iteration and `JUDGE_ERROR_ABORT_AFTER` consecutive ones abort the run — an availability failure introduced by the recipe change, not an integrity one. (I verified truncation cannot yield a *valid* smaller JSON: the first `{` is the unclosed outer brace, so any mid-JSON cut parses invalid.)
Remediation: Raise the default `max_tokens` for the edit recipe path (or document in the recipe header that backend-cfg `max_tokens` must be raised for this rubric).

[INFO] Contradictory code-owned contract vs recipe instructions
Location: comfyless/refine.py:800 (`"Respond with STRICT JSON and nothing else"` in `_JUDGE_OUTPUT_CONTRACT`) vs edit-generic.toml ("Begin your response with these plain-text sections — before the JSON object required at the end of this prompt")
Risk: The composed system prompt now tells the judge both "nothing else" and "preamble first." The recipe cannot *override* the contract (it is appended by `compose_judge_system_prompt`, refine.py:877-882, never recipe-editable — confirmed), and any confusion fails closed at parse; but contradictory instructions measurably raise malformed-output rates on some judge models, feeding the LOW finding above.
Remediation: None required for security. If churn is observed, soften the contract sentence to "end your response with exactly one JSON object of this shape" — a code change, own slice.

[INFO] `isinstance(_seed, int)` admits bool; no judge path exists to exploit it
Location: comfyless/refine.py:1606-1608
Risk: `isinstance(True, int)` is `True` and `True >= 0` holds, so a boolean seed would be "bumped" to `2`. However, I traced every writer of `base["seed"]`: operator CLI/sidecar entry, the iter-0 pin from `outcome.metadata["seed"]` (produced by this codebase's own generation path), and this bump itself. `parse_verdict`'s override allowlist is `prompt`/`loras` only and `apply_overrides` copies `base` untouched (refine.py:783-785) — a judge-supplied value cannot reach `base["seed"]`. Bool admission is an operator-input hygiene nit only.
Remediation: If desired, mirror the codebase's own convention (`_coerce_score`, `_parse_lora_op` both exclude bool explicitly): `isinstance(_seed, int) and not isinstance(_seed, bool)`.

[INFO] Residual plateau: decline-then-no-op regenerates the identical image repeatedly
Location: comfyless/refine.py:1579, 1602-1611
Risk: When an iteration *declines*, `source_cfg = best_cfg` (snapshot holds the original pinned seed S — deep-copied, so the prior bump is discarded). A no-op then bumps to S+1 every time, regenerating the byte-identical S+1 image each iteration until a tie/pass escapes or the cap hits — the same GPU-burn failure the amendment targets, surviving on the strict-decline path. Bounded by `--max-iterations` (default 10; 100 under `--until-score`), operator-local resource waste only.
Remediation: None required now; note it in the ADR changelog so the next plateau report isn't re-diagnosed from scratch.

[INFO] No test exercises the new preamble-shaped response through `parse_verdict`
Location: test_refine.py (diff hunks); comfyless/refine.py:174-190
Risk: The rubric now guarantees leading prose in every live response, but the diff's test fixtures (`_mkverdict*`) still feed bare JSON; there is no pin that a DESCRIPTION/VERIFICATION preamble parses, nor a negative pin that a preamble containing a stray `{`/`}` fails closed (the recipe's brace ban is an unenforced LLM instruction). Prose-tolerance may be covered by pre-existing ADR-027 tests, but the *specific* new output shape is unpinned.
Remediation: Two small checks: `parse_verdict("DESCRIPTION\n- x\nVERIFICATION\nR1: ... -> MET\nPRESERVATION: ok\n" + valid_json)` succeeds; same input with a `}` inside a VERIFICATION line raises `RefineError`.

### Verdicts on the audit questions

**Q1 (tie-promotion):** Yes — the disposition is weakened, and it does change F8-E's risk profile: image-lineage propagation of judge-swaying pixel content now requires only score *parity* instead of a strict win, and a constant-scoring (hallucinating/sycophantic/influenced) judge promotes every candidate to the iteration cap because the compensating control cited in the code comment (patience) defaults to 0/disabled. Winner selection also shifts from earliest-peak to last-tied, i.e. most-drifted. See the MEDIUM finding. The strict-decline revert, `no_improve` accounting, pass gate, and F1/F2/F7 parse boundaries are unchanged and intact.

**Q2 (no-op seed resample):** No new planner authority channel beyond the intended +1-per-noop. The `cfg.base == source_cfg.base` clause is structurally always-true (planner has no base channel; `apply_overrides` deep-copies `base` unmodified — refine.py:783-785), so the comparison reduces to prompt + LoRA (name, weight) — surfaces the planner already fully owns. The planner's only lever is a 1-bit per-iteration choice (make the derived config equal/unequal to its source), and the bump value is code-computed (+1 from the source's pinned seed), never planner-supplied. A judge-supplied value can never reach `base["seed"]` — the override allowlist (refine.py:328-330) and `apply_overrides` guarantee it. The bool subclass hole in the guard exists but is unreachable from judge input (INFO finding). The bump is alias-safe w.r.t. `best_cfg`: promotion snapshots (refine.py:1547, `snapshot_config` deep-copies base per 689-695) *before* `apply_overrides` re-binds `cfg`, and the new cfg's base is itself a fresh deepcopy.

**Q3 (rubric rewrite / preamble):** Confirmed clean. The preamble is discarded by `_extract_json_block` (first-`{`/last-`}`); nothing in the diff or in the surviving code persists or re-injects it: `history_record` is built only from validated `Verdict` fields + resolved ops (refine.py:951-986), `verdict_record` from the same (1336+), notices are code-authored strings. The only raw-text echo path is pre-existing and unchanged: `raw[:200]` inside the "no JSON object found" `RefineError` (refine.py:189), which lands in the on-disk `*.verdict.json` operator artifact and stderr log — never in judge-bound history (Finding 9 discipline, `history_error_record`). The recipe cannot override the code-owned contract: `compose_judge_system_prompt` unconditionally appends `_JUDGE_OUTPUT_CONTRACT` (refine.py:877-882) and `load_judge_recipe` reads only `system_prompt` from the operator-owned TOML (F1). A preamble that violates the recipe's brace ban fails closed (invalid slice → consumed iteration → consecutive-error abort). Residual observations: contract/recipe contradiction (INFO) and token-budget interplay (LOW).

**Q4 (anything else):** Log injection — none: the three new log lines interpolate only ints/floats (`comp:.2f`, `best.index`, `_seed`). History shape — no new keys; `improved` (strict) and `is_best` (promotion) now diverge on ties, but both fields pre-exist in the record and in `_history_stub`, so the F8-P byte budget and `_assert_no_paths` posture are unchanged; the divergent semantics leak nothing sensitive to the judge. `accepted` reuses the existing edit-mode key with tie-inclusive semantics — consistent with the promotion rule. Test fixtures — tie/resample/pinned-seed behavior is well-pinned, but the new judge output *shape* is not (INFO finding). Scope — the diff stays inside the declared D2-amendment scope (loop lineage block, no-op block, recipe, matching tests); no scope creep found.

Model statement: this audit was run on **Fable 5** (model ID `claude-fable-5`). No fallback to a weaker model occurred.
