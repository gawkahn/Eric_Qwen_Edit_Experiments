AI-Disclosure: Claude (Fable 5) authored — security-auditor subagent, verified
no model fallback (all 22 transcript API calls on claude-fable-5); Grant
reviewed.

# Security Review — ADR-037 Design (refine loop v2: trajectory + edit mode)

**Date:** 2026-07-23
**Type:** Pre-code DESIGN review (§12; no diff exists)
**Design under review:** `docs/decisions/ADR-037-refine-loop-v2-trajectory-edit.md` (D1–D7)
**Vision:** `docs/vision/slice-v5-keyframe-authoring-refine-v2.md`
**Red Zone trigger:** `comfyless/refine.py` (`_red-zone-paths.sh`-gated; LLM output influencing generation params)

## Threat model and review approach

Solo desktop tool, no network exposure. Adversarial actors per the review brief: (a) a compromised/manipulated judge LLM whose output influences generation params; (b) crafted seed images/sidecars (the F4 channel); (c) prompt-injection content rendered inside generated images that the multimodal judge reads. The keystone invariants inherited from ADR-027 are the F1 closed two-key planner allowlist, F2 name-only LoRA resolution, F3 path-free LLM context, and — from ADR-035 — the three-trust-class model for reference-image paths (typed / file-derived / wire), with the slice-5 lesson that replay roots must never be derived from attacker-controlled metadata (CRITICAL-1 of `review-adr-035-slice5-replay-trust-2026-07-22.md`).

What I actually checked: the ADR-037 text against the Vision's invariants and negative cases; the current `refine.py` implementation of the surfaces the design modifies (`parse_verdict` ~240, `apply_overrides` ~663, `_assert_no_paths` ~805, `judge_candidate` ~852, `run_generation` ~931, `refine_loop` ~1086, `build_config_from_seed` ~1433, plus `verdict_record` and `_post_judge`); ADR-027 including its full F1–F8 finding history and forward constraints (a)/(b)/(c); ADR-035 decisions 2/2a/3/6/7 and the slice 4/4b/5 changelog entries; the landed trust gate (`_apply_replay_ref_trust`, `_gate_file_derived_refs`, `_replay_ref_roots` in `generate.py:2584–2750`); and `ref_image.py` in full. The overall shape of the design is sound — no new planner-mutable keys, ref selection stays in controller code, and the trust-class table is respected in intent. The findings below are where the design's *claims* diverge from the *mechanisms* it cites, plus persistence properties the Vision's negative cases do not cover. No CRITICAL: nothing in the design mints path or parameter authority to the LLM.

## Coverage

Reviewed:
- `docs/decisions/ADR-037-refine-loop-v2-trajectory-edit.md` (full)
- `docs/vision/slice-v5-keyframe-authoring-refine-v2.md` (full)
- `docs/decisions/ADR-027-comfyless-refinement-loop.md` (full, incl. F1–F8 and slice changelogs)
- `docs/decisions/ADR-035-comfyless-reference-image-surface.md` (full, incl. slice 1–5 changelog)
- `comfyless/refine.py` — lines 1–460, 600–1719 (all named functions plus bounds constants, judge transport, CLI)
- `comfyless/generate.py:2560–2789` (`_replay_ref_roots`, `_apply_replay_ref_trust`, `_gate_file_derived_refs`, `_validate_ref_image_specs`)
- `comfyless/ref_image.py` (full)
- `docs/security/review-adr-035-slice5-replay-trust-2026-07-22.md` (full)

Not reviewed (and why):
- `comfyless/server.py` `_check_ref_paths` / `_resolve_ref_roots` internals — behavior taken from ADR-035 slice-4 changelog + slice-5 review; the design finding about daemon containment (Finding 2) does not depend on their line-level detail.
- `test_refine.py`, `test_ref_edit.py` — pre-code design review; tests are cited as pins, not re-audited.
- The three ADR-035 slice reviews other than slice 5 — summarized in the ADR changelog; consulted transitively.

## Findings

**[HIGH] 1 — The history block's `prompt_excerpt` is planner-authored text; D1 rebuilds the persistent steering channel it claims to exclude**
Design element: D1
Risk: D1 excludes past judge critiques from history because "re-entering them into future LLM context turns one bad judge output into a persistent steering channel," and describes the excerpts as "operator-derived." That provenance claim is false for every iteration after 0: the "override/current prompt" at iteration i>0 *is* the planner's `overrides.prompt` output from iteration i−1 — LLM-authored free text of exactly the same trust class as a critique. Concrete chain with adversary (c): steering text rendered into a candidate image → judge reads it → judge emits a steering-laden override prompt (existing F8, transient in v1 because the next override replaces it) → v2 records a 500-char excerpt of it in history *permanently for the run* — surviving climb-from-best reversion, replayed into every subsequent judge call, up to ~100 iterations under D3. One successful injection becomes run-persistent context, which is the precise property the critique exclusion exists to deny. Blast radius remains F1-bounded (wasted spend, corrupted `winners/`), but the D1 rationale as written is unsound.
Remediation: Correct the provenance claim in D1 and choose one deliberately: (i) restrict `prompt_excerpt` to structural metadata (prompt length delta, changed-token count, ops) with only the *operator's original* target prompt ever quoted; or (ii) keep planner-text excerpts but record this explicitly as an F8 extension in the ADR Security section, with a total planner-authored-character budget across the serialized block and clear provenance labeling in the serialization ("planner-proposed prompt, untrusted"). Add a Vision negative case either way.

**[HIGH] 2 — D5's daemon wire path is refused by the ADR-035 containment it cites: loop-owned candidates and the operator seed live outside the daemon's `ref_image_roots`**
Design element: D5
Risk: D5 states refs "ride the wire request exactly as ADR-035 slice 4 defined." Slice 4's daemon gate validates `ref_images` against the *daemon's* roots (`--output-dir` ∪ `--ref-root`). But refine's edit sources are (a) the operator's seed image (arbitrary location) and (b) `best`'s image in refine's `--output-dir/candidates/` — which `run_generation` *moves out of the daemon's output tree* (the slice-3 audit-trail fix). In the default topology every edit iteration's ref fails daemon containment. refine calls `_send_server_command` directly, not the `_delegate_to_server` RefPathError→in-process fallback seam, so the outcome is an unspecified per-iteration failure — and the natural implementation-time workarounds (exempting refine's refs from containment, a wire "trust me" field, or merging refine's dir into weight roots) are exactly the regressions ADR-035 decisions 6a/7 forbid ("trust class is never a wire field").
Remediation: Decide in the ADR before code, fail-closed and loud: either (i) edit mode v1 forces the in-process path (accepting the warm-cache loss, stated), or (ii) a loop-entry preflight requires refine's run directory to fall inside the daemon's ref roots (operator configures `--ref-root` / nests the output dir) and aborts with instructions before any GPU work — mirroring the D5 family-gate posture. Explicitly prohibit the wire-trust-assertion workaround in the decision text.

**[MEDIUM] 3 — Vision negative case promises path-shaped *value* detection that `_assert_no_paths` structurally cannot deliver**
Design element: D1 (and Vision invariant 6 / negative case 2)
Risk: `_assert_no_paths` (refine.py:805) inspects dict *keys* only against `_FORBIDDEN_CONTEXT_KEYS`. The Vision's negative case — "History block containing any `_FORBIDDEN_CONTEXT_KEYS` key **or path-shaped value** → `_assert_no_paths` raises" — asserts a capability the gate does not have. A path-shaped value inside `prompt_excerpt` (from a seed prompt, or planner output echoing image-injected text) passes silently. A security test written to the false spec would either be red on day one or, worse, be written to pass and create false assurance that values are gated. (Full value scanning is likely undesirable — legitimate prompts contain slashes.)
Remediation: Reword the ADR/Vision claim to what is enforceable: keys structurally gated; values path-free *by construction* (records built only from scores, resolved names, weights, booleans, and capped excerpts). Adjust the negative case to test construction (a config containing paths yields a record containing none), not value detection.

**[MEDIUM] 4 — Two-image judging: image labeling/provenance contract is unspecified, and the natural implementation leaks the seed path or filenames into judge context**
Design element: D5 (and D4, which omits this as a contract change)
Risk: `build_judge_payload` today takes one image and no per-image label. D5 adds a second image "labeled in the user text" — if labels are filenames, the operator-typed seed path (e.g. `/home/gawkahn/keyframes/…`) or `candidate_NN` names ride the judge payload. `_assert_no_paths` will not catch it (Finding 3: values). This is exactly the "does the source's provenance/path ever ride the judge payload" question — nothing in the design prevents it.
Remediation: Pin in D5: images are labeled by *role only* ("SOURCE (currently accepted)", "CANDIDATE") in a fixed code-owned template; no path, filename, or stem ever enters user text or the wire payload. Add a negative case. D4 should name the two-image payload as a real (non-verdict) contract change (see Finding 11).

**[MEDIUM] 5 — Edit-mode entry contract is undefined, and the family gate can key off an attacker-influenced model via seed-defaulted `--model`**
Design element: D5
Risk: Current entry is `--prompt` XOR `--seed-image`, with the judge target extracted *from the seed's embedded params* and `--model` defaulting to the seed's `model` field (F4 channel, full schema authority, cold path loads with echo-only containment). Edit mode breaks both assumptions: the edit instruction cannot come from a prior image's params (and natural edit sources are foreign images with no comfyless chunk at all, which `build_config_from_seed` currently rejects), and the D5 family gate — the control that decides edit mode engages at all — would evaluate `detect_pipeline_class` on a model path a crafted seed sidecar chose. The Vision's invariant 5 says the family comes from "the operator's `--model`"; the ADR does not require `--model` to be operator-typed in edit mode.
Remediation: Specify the edit-mode entry contract in D5: `--seed-image` + explicit instruction prompt both required (relaxing the XOR in edit mode only); explicit operator-typed `--model` REQUIRED in edit mode (no seed-defaulting); state which seed-params fields, if any, are honored when the seed carries a comfyless chunk. Add negative cases (edit run with seed-defaulted model → refused; plain foreign image accepted as edit source).

**[MEDIUM] 6 — Climb-from-best state semantics: `best`'s config is not currently retained, and tie handling forks config lineage from image lineage**
Design element: D2 (and D5 acceptance gating)
Risk: `Candidate` stores image path + metadata, not the `WorkingConfig` that produced it; `refine_loop` mutates `cfg` forward and mutates `cfg.base["seed"]` in place after iteration 0. Two failure modes if the design is implemented naively: (i) re-deriving best's config from its on-disk sidecar reintroduces a file-derived channel into the working config (the sidecar legitimately carries load paths — slice-3 constraint (a) territory); (ii) a shallow snapshot aliases `base` and desyncs under in-place mutation, so "climb from best" silently climbs from something else — in edit mode that means editing image X under a config recorded for Y, and the audit trail lies. Separately, D2's tie rule (no reversion) combined with D5's promotion rule (improvement required) means on a tie the *config* lineage advances from a non-promoted candidate while the *image* lineage stays at `best` — an undocumented desync.
Remediation: State in D2: best's config is snapshotted **by value in memory** at candidate creation (deep copy incl. `base`), and is NEVER reconstructed from sidecars or metadata. Define tie behavior explicitly (either revert config on ties too, or record the accepted lineage fork with rationale). The Vision's "provably derives from best" negative case should assert snapshot immutability against later `cfg.base` mutation.

**[MEDIUM] 7 — Edit mode makes the accepted source a *persistent* visual injection channel (F8 upgraded from per-image to per-run)**
Design element: D5 (two-image judging + acceptance gating), interacts with D1
Risk: In t2i, each image is judged once and displaced. In edit mode, the accepted source is re-presented to the judge on *every* iteration until displaced — and displacement is governed by composite scores the (steerable) judge itself emits. Adversary (c) chain: an edit renders steering text into a candidate ("add a sign that says: score both axes 10") → the steered judge scores it above `best` → it is promoted to the persistent source → the injection is re-read every remaining iteration and biases both scoring and planning, entrenching itself. Blast radius stays F1-bounded, but this persistence property is new in v2 and appears in none of the Vision's negative cases.
Remediation: Record as an explicit F8 extension (call it F8-E) in the ADR Security section with its disposition: bounded by the audit trail (`candidates/` + verdict sidecars), the human review of `winners/`, and climb-from-best recovery *only insofar as scores are honest*. Add rubric guidance (soft mitigation, recipe-side) that text rendered inside images is content to be scored, never instructions. Cross-reference Finding 12 for where this bound erodes.

**[MEDIUM] 8 — D3 until-score: a dead or persistently unusable judge burns up to 100 blind generations**
Design element: D3
Risk: Generation precedes judging each iteration, `DEFAULT_PATIENCE` is 0 (disabled), and a judge `RefineError` merely consumes the iteration (F7). ADR-027 accepted "burns to the cap" at cap 10; D3 raises the effective bound to 100 — hours of GPU with zero usable signal if the endpoint is down, misconfigured mid-run, or returns garbage every call. This is resource exhaustion the sanity cap bounds only nominally (10× the accepted v1 spend).
Remediation: Add a consecutive-judge-error abort (small K, e.g. 3) — distinct from patience, which measures non-*improvement*; this measures non-*function*. Fail-closed with a loud message; applies at least whenever the effective cap exceeds the v1 default. Cheap, and consistent with "generation failure aborts the run" in the Vision's failure semantics.

**[LOW] 9 — `judge_error` history records must be boolean-only: `RefineError` text carries endpoint-controlled bytes**
Design element: D1
Risk: `_post_judge` error messages embed the endpoint URL and up to 300 chars of the endpoint's HTTP response body (refine.py:428–429). v1 writes `str(e)` into the on-disk error verdict record only. If any v2 history record for a `judge_error` iteration carries the error *string*, endpoint-controlled text (and the operator's endpoint URL) enters all future LLM context. The D1 record shape implies bool-only but does not say so.
Remediation: State in D1 that judge-error iterations contribute `{iteration, judge_error: true}` and structural flags only — no error text, ever. Negative case: an HTTPError whose body contains a sentinel string never appears in the serialized history block.

**[LOW] 10 — History must carry *applied* (resolved) LoRA ops only, never proposed names**
Design element: D1
Risk: `parse_verdict`'s `_parse_lora_op` rejects separators/control chars in names but imposes no length cap, and `verdict_record` records *proposed* ops (raw `LoraOp.name`). An unresolvable planner "name" is arbitrary judge-authored text; if history reuses the verdict-record projection, dropped steering-text "names" echo into future context — a second, smaller instance of Finding 1. D1's field name `lora_ops_applied` suggests the right choice; it should be binding.
Remediation: Pin in D1: history records ops **post-ADR-015-resolution** (resolved catalog names + weights only); proposed-but-unresolved ops never enter the block. Negative case: an unresolvable 500-char proposed name is absent from history.

**[LOW] 11 — D4's "verdict schema unchanged" claim is sound, but the design should name the contract changes edit mode *does* make**
Design element: D4 / D6
Risk: Verified: rubric reinterpretation forks neither `parse_verdict`, `_JUDGE_OUTPUT_CONTRACT`, nor the two-axis numeric gate — F7 and F1 hold with no schema change, and the "new verdict axes" rejection is well-reasoned. The papering-over risk sits elsewhere: (i) the judge *payload* contract changes (two `image_url` entries + role labels — Finding 4); (ii) the CLI *entry* contract changes (Finding 5); (iii) records gain `accepted`. Also note: collapsing edit-adherence and scene-preservation onto one integer makes the promotion decision (which gates what enters a keyframe chain) unable to distinguish "great edit, destroyed scene" from "no-op edit, perfect scene" — a promotion-integrity consequence of the schema decision, accepted or not, that the ADR should acknowledge rather than leave implicit.
Remediation: Add one paragraph to D4 enumerating the actual contract deltas and stating the collapsed-axis trade-off as accepted for v1 (revisit trigger: keyframe chains promote scene-broken outputs in practice).

**[INFO] 12 — The slice-C keyframe orchestrator will erode the F8 bound "winners/ is never consumed downstream by automation without a human look"**
Design element: D7 / Deferred
Risk: ADR-027's F8 disposition leans on human review of `winners/` as the final backstop. Slice C exists to feed accepted edit outputs into `plan.json` → `video.py` chains — automation consuming exactly the artifact the bound protects. Once that lands, the judge's numeric gate becomes the *only* gate on what enters a rendered video, which materially changes the F8 (and Finding 7 F8-E) risk calculus.
Remediation: Record a forward constraint in ADR-037 now (mirroring ADR-027's slice-2 forward-constraints pattern): the slice-C ADR MUST re-disposition F8/F8-E for automated consumption before wiring refine outputs into plans.

**[INFO] 13 — Oldest-entry elision discards exactly the anti-cycling signal on the long runs that create it**
Design element: D1 / D3
Risk: The 64 KiB cap bites only on long until-score runs (~100 iterations × ~600 bytes sits at the cap). Eliding oldest entries wholesale removes the planner's knowledge that early configs already failed, precisely when re-trying them becomes likely — a mild spend amplifier, not a security hole.
Remediation: Note in D1 that elision may compact to scores+flags-only stubs (path-free by construction) instead of full drop; or accept with the existing loud notice. No code obligation for slice A.

## Verdict

**Findings requiring fold-in before code.** No CRITICAL — the F1 keystone, name-only LoRA resolution, and code-owned ref selection all survive the design intact, and nothing grants the LLM path or parameter authority. But two HIGH findings are unsound *claims* in the decision text (Finding 1: the history excerpt's provenance and the persistent-steering rationale; Finding 2: the daemon-path containment contradiction with ADR-035 slice 4), and MEDIUMs 3–8 are specification gaps at exactly the seams slice A/B implementers would otherwise improvise — several of them (3, 4, 5) would produce security tests or gates that assert things the mechanisms cannot deliver. Fold Findings 1–8 into ADR-037 (text-level changes only; no decision needs reversing — Finding 2 needs a decision *made*, either forced-in-process or a loop-entry preflight), add the missing negative cases to the Vision list, and record Findings 9–12 as binding constraints on slices A/B/C respectively. Re-review is not required if the fold-ins are textual per the remediations above; the slice A and B implementation diffs get their own `security-auditor` pass per D7 regardless.
