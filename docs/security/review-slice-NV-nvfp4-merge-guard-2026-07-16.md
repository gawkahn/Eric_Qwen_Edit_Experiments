# Security review — slice NV nvfp4 merge guard (2026-07-16)

AI-Disclosure: security-auditor subagent (Fable 5) authored the review; Claude (Fable 5) implemented; Grant reviews.

**Scope:** the Red Zone half of ADR-019 slice NV — `nodes/eric_diffusion_fp8_ops.py` (`_requant_config_matching_base` class-allowlist gate + consequences). Design-phase review per §12 order (ADR entry 2026-07-16 → this review → code). Verdict: **approvable-with-changes**, requirements 61–66 (continuing the file's chain from req 60). All six were implemented in the same slice; compliance notes follow the verbatim review at the bottom.

---

## Verbatim reviewer output

### Summary

Design-phase review of the slice-NV change to `nodes/eric_diffusion_fp8_ops.py` (Red Zone, review chain reqs 1–60): a class-identity gate in `_requant_config_matching_base` so that only torchao `Float8Tensor` bases proceed to the `act_quant_kwargs` recipe sniff, and every other torchao representation (specifically the new `NVFP4Tensor` once `--quant nvfp4` lands) refuses loudly instead of being silently requantized as fp8 on a LoRA direct merge. Threat model: the adversary-shaped inputs here are (a) caller-supplied adapter files whose key ordering and deltas are attacker-controlled, and (b) the operator's own footgun surface — a mixed-representation model produced silently, which is exactly the class of defect (well-typed but wrong) this file's review chain exists to block. The change itself parses no new caller-supplied bytes; it discriminates between already-instantiated in-process tensor objects, so the descriptor-parsing hot surface (reqs 1–56) is untouched.

What I actually checked: the full dispatch path `apply_merge_delta` → `_merge_into_torchao` → `_requant_config_matching_base` → `restore_merge_backup` (fp8_ops:1202–1438); all four merge call sites (`eric_qwen_edit_lora.py:456/631/1065`, `eric_lora_format_convert_apply.py:755`) — `_requant_config_matching_base` has exactly one caller, so the gate is at a true choke point; the caller-side exception handling (`comfyless/generate.py:_apply_loras`, line 1143 broad `except Exception`); the retained-but-uncalled `guard_direct_merge` entry gate (`eric_diffusion_utils.py:1868`); the DMR review (reqs 21–30) and the DMR partial-merge TECH_DEBT entry (TECH_DEBT.md:438–452), whose stated trigger — "extending the DMR surface (new quantized reps)" — this slice fires. The empirical claims (NVFP4Tensor module path satisfies the branch-(b) sniff; carries `act_quant_kwargs`; `Float8Tensor` importable from `torchao.quantization` on the pinned 0.17.0) are taken from the session's stated verification, not re-run here.

### Coverage

Reviewed:
- `nodes/eric_diffusion_fp8_ops.py` — lines 1150–1449 (DMR block: dispatcher, both merge sinks, recipe matcher, restore, ledger) plus `_MISSING`/`_safe_name`/`_validate_scale` definitions
- `docs/vision/slice-NV-nvfp4-quant-load.md` — full
- `docs/decisions/ADR-019-native-quantization-support.md` — 2026-07-10 and 2026-07-16 changelog entries, §Deferred
- `docs/security/review-slice-DMR-quantized-merge-2026-07-03.md` — full (reqs 21–30 baseline)
- `nodes/eric_qwen_edit_lora.py` — LoKR merge loop 380–479 (representative of the three sites), restore path 1130–1146
- `nodes/eric_diffusion_utils.py` — 1824–1902 (`is_quantized_module` structural-check rationale, `guard_direct_merge`)
- `comfyless/generate.py` — `_apply_loras` 1104–1149
- `TECH_DEBT.md` — DMR partial-merge entry 438–452

Not reviewed (and why):
- torchao 0.17.0 source (`NVFP4Tensor` internals, `Float8Tensor` import path stability) — dependency source out of scope; the session's empirical verification on the repo `.venv` is relied on and named as an assumption in findings 1 and 3.
- The nvfp4 config-routing / hardware-gate / mslk-probe half of the slice (`eric_diffusion_utils.py`, `params_validation.py` edits) — prompt scopes this review to the one Red Zone file change; those go to `code-reviewer`.
- `eric_lora_format_convert_apply.py` beyond confirming its call site routes through `apply_merge_delta`.

### Findings

**1. [BLOCKER] The gate must exist and must be an allowlist — the finding is confirmed against the code as written (req 61)**
Location: `nodes/eric_diffusion_fp8_ops.py:1328–1354` (`_requant_config_matching_base`)
Risk: As written, ANY torchao tensor subclass carrying `act_quant_kwargs` passes the sniff: an NVFP4 weight-only base (akw `None`) silently maps to `Float8WeightOnlyConfig`, a dynamic-activation NVFP4 base maps to `Float8DynamicActivationFloat8WeightConfig` — either way `quantize_(tmp, _cfg)` at line 1389 produces a Float8 layer inside an nvfp4 model with no error, notice, or cache-key discrimination. Mixed-representation state is exactly the silent-corruption class DMR-4/req 24 ("positive-match-then-explicit-raise, never a fallthrough") exists to prevent; the akw sniff is a fallthrough in disguise once a second akw-carrying class exists.
**Req 61:** `_requant_config_matching_base` proceeds to the `act_quant_kwargs` sniff ONLY when the base tensor is a torchao `Float8Tensor` (allowlist by class, `from torchao.quantization import Float8Tensor`); every other type — including but not limited to `NVFP4Tensor` — raises `RuntimeError` with ADR-019 §4 actionable wording naming the actual tensor class (`type(data).__name__`) and directing to the PEFT adapter path / generating without `--quant nvfp4`. It must be an allowlist (refuse everything not-Float8), never an NVFP4 blocklist — a third torchao rep (int8 AQT, MX fp8, future classes) must hit the same refusal. The existing `_MISSING` raise for a genuine Float8Tensor missing the attribute stays behind the gate unchanged.
The proposed design satisfies this. Verdict on review question 1's core: yes, the gate is sufficient for the recipe-matching hole, subject to placement (finding 2) and the one other dispatch path (finding 4).

**2. [CONDITION] Gate placement: fire before the backup record and before dequantize (req 62)**
Location: `nodes/eric_diffusion_fp8_ops.py:1360–1383` (`_merge_into_torchao` sequence: backup → `dequantize()` → merged compute → finite/zero checks → `_requant_config_matching_base`)
Risk: With the gate inside `_requant_config_matching_base` at its current call position, a refused NVFP4 merge still (a) records a `{"kind": "torchao_param", "param": p}` backup entry for a target that was never merged, and (b) executes `data.dequantize()` + a full-precision merged-tensor allocation on a representation the code is about to declare unsupported. On (a): I traced it — the stale entry is *benign* today (it holds the still-live original Parameter; the adapter's `peft_config`/ledger registration happens after the loop and never runs, so `unload_adapters` never restores it; even if restored, the verbatim swap is an identity no-op). On (b): wasted allocation only, plus an assumption that `NVFP4Tensor.dequantize()` exists and behaves — an assumption the slice otherwise never needs to make. But "benign stale state + unvetted-rep compute on the refusal path" is looser than the file's own posture (req 30: "nothing has been persisted yet, so a failure here is a clean raise"), and it costs one line to be exact.
**Req 62:** the class gate fires before any state is touched in `_merge_into_torchao` — either hoist the `_cfg = _requant_config_matching_base(data, target_key, log_prefix)` call above the backup record at line 1360, or perform the isinstance check at function entry. Post-refusal invariants, pinned by test: model weights bit-identical, backup dict unchanged (no entry for the refused key), no swap performed. This also fully answers review question 2: with the hoist, no partial mutation and no stale entry exist at all; without it, the stale entry is benign but the negative test for "backup unchanged" becomes unwritable.

**3. [CONDITION] Class-identity pins so a torchao bump can't silently defeat the gate (req 63)**
Location: `nodes/eric_diffusion_fp8_ops.py:1341–1345` (import + sniff); tests in `test_fp8_single_file.py` / `test_quant.py`
Risk: On review question 3 — class identity IS the right mechanism here, and the `is_quantized_module` structural precedent does not apply: that function avoids identity checks for the repo's *own* classes because the test harness spec-loads modules via `spec_from_file_location`, duplicating class objects. torchao is a normal pinned import with a single `sys.modules` identity; `isinstance` holds. The residual risks are all at dependency-bump time: (a) a future torchao making `NVFP4Tensor` a `Float8Tensor` subclass would silently re-enable the sniff (the exact CRITICAL this slice closes); (b) a moved/renamed `Float8Tensor` import raises `ImportError` inside the matcher — which propagates and fails closed, acceptable, but should trip the suite at bump time, not at a user's merge. On review question 5: with the gate ordered first (req 62), the akw sniff is unreachable for NVFP4, so pinning "NVFP4Tensor has act_quant_kwargs" is no longer load-bearing and should NOT be the pin — the load-bearing relationship is the *class hierarchy*, not the attribute.
**Req 63:** tests pin, against the real pinned torchao: (a) the refusal negative uses a REAL CPU-quantized weight-only NVFP4 parameter (akw `None` — the arm that old code silently mapped to `Float8WeightOnlyConfig`) through `apply_merge_delta` → `RuntimeError`, weights untouched, backup unchanged; (b) `isinstance(<nvfp4 instance>, Float8Tensor) is False` asserted directly, so a subclassing/renaming torchao bump fails CI before it ships; (c) a generic-allowlist arm — a torchao-module tensor that is neither Float8 nor NVFP4 (synthetic class with `__module__` under `torchao.` is acceptable for this arm) also refuses; (d) the existing Float8 positive controls (both recipes merge, existing akw-`_MISSING` raise) unchanged. The existing test pins on Float8Tensor's akw attribute stay — they still guard the genuine-Float8 arm.

**4. [CONDITION] Close the branch-(d) fallthrough for torchao tensors not held as `nn.Parameter` (req 64)**
Location: `nodes/eric_diffusion_fp8_ops.py:1249–1268` (`apply_merge_delta` branches b–d)
Risk: Answering review question 1's "any other path": branch (b) requires `isinstance(t, nn.Parameter)`. A torchao tensor subclass reached as a *buffer* (via `merge_resolution_map`'s `.weight`-buffer union, DMR-8) skips (b); because torchao subclasses report their logical dtype (bf16), it also skips the fp8-dtype check in (c) and lands in branch (d)'s in-place `data.add_(delta)` — a plain mutation of a quantized subclass, the precise operation DMR-4 forbids. Not reachable via current load paths (torchao `quantize_` only produces Parameters; `ScaledFp8Linear` buffers are raw fp8 dtype and hit (c)), so this is latent, pre-existing — but slice NV is widening the diversity of torchao reps in loaded models, and the file's own invariant is "explicit raise, never a fallthrough."
**Req 64:** before branch (d), any tensor with `type(data).__module__.startswith("torchao")` that did not already dispatch raises the ADR-019 §4 refusal (do not instead route buffers into `_merge_into_torchao` — its backup/swap contract assumes a Parameter). One guard clause in the in-scope file; negative test with a torchao-module tensor registered as a buffer.

**5. [CONDITION] The DMR partial-merge debt trigger has fired: mid-loop refusal is now a NORMAL flow, and `_apply_loras` swallows it (req 65)**
Location: `nodes/eric_qwen_edit_lora.py:407–458` (per-target loop), `comfyless/generate.py:1143` (`except Exception`), `TECH_DEBT.md:438–452`
Risk: Answering review question 2's deeper half. The DMR posture ("fail-loud NARROWS, never lapses") is preserved at the *dispatcher* level — this change only converts a silent success into a raise, which is monotone-restrictive. But the DMR TECH_DEBT entry accepted partial-merge-on-raise on the explicit grounds that it was "only reachable on adversarial/degenerate adapters," and named "extending the DMR surface (new quantized reps)" as its revisit trigger. Slice NV is that trigger. Under `--quant nvfp4` + any direct-merge-only adapter (LoKR/`.diff` — Grant's snofs class), the refusal fires routinely: plain targets preceding the first NVFP4 target in the adapter's own key order (author/attacker-controlled) merge and PERSIST, then the raise propagates to `_apply_loras`, whose broad `except Exception` logs "LoRA load failed," records `applied: False`, and **continues generating on the partially-merged transformer** — with a notice that is factually wrong (deltas from the merged prefix are in the weights) and, on the daemon/MCP path, a *cached* pipeline that serves that half-merged state to subsequent identical requests (slice DQ's LoRA-set cache key evicts on a LoRA *change*, not on a same-key repeat). The Vision's invariant-6 negative ("actionable RuntimeError, weights untouched") is only true per-target; it is not true per-adapter.
**Req 65:** the slice must resolve this deliberately, one of: **(a)** restore an all-or-nothing entry gate for the nvfp4 case at the direct-merge call sites — `guard_direct_merge` (`eric_diffusion_utils.py:1868`) already exists, is uncalled, and carries the right message shape; calling it (or an nvfp4-specific pre-scan of `merge_resolution_map` targets) before the loop makes "weights untouched" true per-adapter. This touches `eric_qwen_edit_lora.py` / `eric_lora_format_convert_apply.py`, which are OUTSIDE the declared §2 edit scope — it requires a scope amendment, flagged here as required by §4 discipline, not smuggled in. Or **(b)** documented acceptance: amend the TECH_DEBT.md:438 entry (the "adversarial-only" claim is now false; its trigger fired and the resolution was "accepted for NV, entry-gate deferred"), AND the req-61 refusal message states that earlier targets of this adapter may already be merged and the pipeline must be reloaded — so the operator-facing sidecar warning (`lora_failure_warnings`) carries the true state. Option (a) is the sound one; (b) is the floor below which this is not approvable. Assumption named: I did not trace the daemon's exception-path cache handling beyond the DQ changelog; if the server evicts the cached pipeline whenever any LoRA outcome has `applied: False`, the cross-request half of the risk is already closed — verify and record which it is.

**6. [INFO] No injection/DoS surface change; error-message hygiene (req 66)**
Location: `nodes/eric_diffusion_fp8_ops.py:1328` (new raise site)
Answering review question 4: the change parses no caller-supplied file content — it discriminates among tensor objects torchao already instantiated; the descriptor-parsing surface (reqs 1–56) and its fail-closed posture are untouched, and nvfp4 single-file parsing still rejects at header (unchanged, per ADR). The one attacker-influenced string near the new raise is the target key.
**Req 66:** the refusal message passes the target key through `_safe_name` (F7 pattern, as every neighboring raise does) and names the class via `type(data).__name__` — never `repr(data)` or tensor contents.

**7. [INFO] Single choke point confirmed; restore path is representation-agnostic**
`_requant_config_matching_base` has exactly one caller (`_merge_into_torchao:1383`); no other code path reaches the Float8 recipe assumption. `restore_merge_backup`'s `torchao_param` arm (fp8_ops:1428–1435) is a verbatim object swap with no Float8-specific logic; with req 62 in place no NVFP4 backup entry can exist, and the cross-device `nn.Parameter(old_p.data.to(...))` branch is never exercised for a refused rep. `record_direct_merge`/LIFO warnings are name-only. No further Float8 assumptions found in the reviewed range.

### Verdict

**Approvable-with-changes.** The core design — allowlist class gate, fail-loud refusal, no nvfp4 requant-matching built this slice — is sound and correctly identified as the minimal fail-closed fix for a real CRITICAL-class silent-corruption path (finding 1 / req 61). Conditions: gate placement before backup/dequantize (req 62), class-hierarchy test pins rather than attribute pins (req 63), the branch-(d) torchao-buffer fallthrough closed (req 64), the message/`_safe_name` hygiene (req 66), and — the substantive one — an explicit, recorded resolution of the fired DMR partial-merge trigger (req 65), which either amends the edit scope to add an entry gate at the merge call sites or, at minimum, amends TECH_DEBT.md:438 and makes the refusal message truthful about partial state. Requirements numbered 61–66, continuing the file's chain from req 60.

---

## Implementation compliance (same slice, same day)

- **Req 61** — implemented: `from torchao.quantization import Float8Tensor` + `isinstance` allowlist at the top of `_requant_config_matching_base`; refusal names `type(data).__name__`, uses `_safe_name`, directs to the PEFT path / no `--quant nvfp4`. Tests: NVFP4 refusal + generic duck-typed-akw refusal (`test_fp8_single_file.py` §slice NV).
- **Req 62** — implemented: the recipe match (and thus the gate) hoisted to the first statement of `_merge_into_torchao`, before the backup record and dequantize. Tests: weights bit-identical + backup dict empty after refusal.
- **Req 63** — implemented: (a) real CPU-quantized weight-only NVFP4 param through `apply_merge_delta` → RuntimeError; (b) `isinstance(nvfp4, Float8Tensor) is False` tripwire; (c) generic non-Float8 refusal arm; (d) Float8 weight-only positive control merges end-to-end (kind `torchao`, kind-tagged backup). The pre-existing three requant-match arms were converted from duck-typed fakes to REAL Float8Tensors (the fakes now exercise the allowlist refusal instead).
- **Req 64** — implemented: branch (c2) in `apply_merge_delta` refuses any `torchao`-module tensor that did not dispatch as a Parameter or ScaledFp8Linear. Negative test registers a real NVFP4Tensor as a `.weight` buffer.
- **Req 65** — implemented as option **(a)**: new `refuse_unmergeable_base(root, model_sd, log_prefix)` in fp8_ops; all four merge call sites call it immediately after `merge_resolution_map`, before any mutation. Edit-scope amendment recorded in the Vision doc §2 (2026-07-16). TECH_DEBT.md DMR partial-merge entry amended: closed for the unmergeable-rep class, stays open for the other per-target raise paths (transactional merge still deferred). On the auditor's named assumption (daemon cache on `applied: False`): with the entry gate, a refused adapter mutates nothing, so the daemon-cached pipeline is clean **by construction** — the cross-request risk does not arise regardless of the server's eviction behavior; not separately traced.
- **Req 66** — implemented as specified (`_safe_name` + class name only).

Proof: `test_fp8_single_file.py` 225→239 (0 failures), `test_quant.py` 117→147 (0 failures); full battery run recorded in the commit.

Post-implementation code review (code-reviewer, Fable 5) additionally required: fail-closed `refuse_unmergeable_base` when torchao is present but `Float8Tensor` unimportable (dep-bump case — fail-open stays only for torchao-absent); explicit-branch `_torchao_quant_config` with a loud raise on unknown modes; strictest-floor default in the capability table; and the addendum below for the `mcp_server.py` half of the diff. All folded.

---

# Addendum — comfyless/mcp_server.py quant-enum derivation (2026-07-16)

AI-Disclosure: security-auditor subagent (Fable 5) authored; Claude (Fable 5) integrated; Grant reviews.

**Trigger:** code-reviewer BLOCKER — the slice's 4-line diff to `comfyless/mcp_server.py` (path-gated Red Zone) had no auditor coverage in the review above.

## Verbatim reviewer output

### Summary

Addendum scope: the uncommitted slice-NV diff to `comfyless/mcp_server.py` (path-gated Red Zone — LLM-agent tool surface), consisting of (1) a module-level `from comfyless.params_validation import QUANT_MODES` and (2) `_GENERATE_INPUT_SCHEMA["quant"]["enum"]` changed from a hardcoded two-value list to `list(QUANT_MODES)` plus extended description text. Threat model: a hostile or confused MCP client (the LLM) supplies arbitrary `arguments`; the schema is advisory only, so the security question is whether this diff changes what the server *accepts* or *emits*, not what it advertises. I traced the quant value through the MCP path: `validate_machine_request` (type-only, `_KIND_STR`) → `_get_or_load_cached_pipeline` cache key (mcp_server.py:1858) → `generate()` → `build_quant_config` membership gate (nodes/eric_diffusion_utils.py:1749, raises `ValueError` on off-enum values — fail closed). I checked for residual two-value assumptions, import-graph weight/cycles, and audit/redaction set membership.

Net: the diff is a strict single-source-of-truth improvement to an advisory surface. Acceptance at the MCP boundary is unchanged — the old hardcoded enum was never enforced by server code (schema enums are client-side hints; `validate_machine_request` never checked membership), so a client sending `"nvfp4"` before this diff already reached the same downstream gates. Enforcement remains where it was: daemon membership check (server.py path) and `build_quant_config`'s raise (in-process MCP path).

### Coverage

Reviewed:
- `comfyless/mcp_server.py:1-125` (imports, redaction/audit constants), `760-830` (quant/quant_skip/quant_only/nag schema properties), `2100-2199` (in-process generate path, quant carriage), plus full-file grep for `quant`, `"fp8"` literals, `len(QUANT_MODES)`, redaction sets
- `comfyless/params_validation.py:1-375` (full file — import graph, QUANT_MODES definition, quant field validation)
- `comfyless/generate.py:920-955, 2815-2844` (build_quant_config call site; CLI membership gate for contrast)
- `nodes/eric_diffusion_utils.py:1730-1790` (`build_quant_config` membership + hardware fallback)
- `test_quant.py:240-246, 454-461, 545-548` (sync pin params_validation↔utils; schema-enum pin; nvfp4 value pin)

Not reviewed (and why):
- The `nodes/eric_diffusion_fp8_ops.py` half of slice NV — covered by the main review (reqs 61–66), out of this addendum's scope.
- I could not run `git diff`; I verified the *current file state* matches the two declared edits and found no other uncommitted-looking quant changes in `mcp_server.py`. Assumption: the working tree I read is the diff under review.

### Findings

**[INFO] (a) Enum derivation adds no acceptance; no stale two-value assumptions remain**
Location: `comfyless/mcp_server.py:780`
Risk: none — verified. The schema enum was never load-bearing: `validate_machine_request` type-checks `quant` as bare `str` (params_validation.py:79) with no membership check, so pre-diff the MCP boundary already passed any string downstream. Grep confirms no `"fp8"` literals, no `len(QUANT_MODES)` or exhaustive-match logic anywhere in `mcp_server.py`; the cascade ignore-loudly guard (line 2319) compares only against `"none"` and handles a third mode correctly; the pipeline cache key (line 1858) stringifies the mode and discriminates any new value automatically. Single-source derivation is a strict improvement, and `test_quant.py:457` pins schema == constant.
Remediation: none required.

**[INFO] (b) Import is light and cycle-free**
Location: `comfyless/mcp_server.py:49`; `comfyless/params_validation.py:22-26`
Risk: none — verified. `params_validation` imports only stdlib (`types`, `dataclasses`, `typing`); it imports no comfyless module, so no cycle with `mcp_server` or `server` is possible. `mcp_server` already imports `comfyless.server` at module level (line 45), which is strictly heavier, so startup-path weight is unchanged. The slice-DQ F1 rationale (no torch on the validation/startup path) is preserved.
Remediation: none required.

**[INFO] (c) Audit/redaction sets correctly untouched**
Location: `comfyless/mcp_server.py:60-98`
Risk: none — verified. `quant` is absent from `_MCP_PATH_TYPED_FIELDS`, `_MCP_CASCADE_PATH_TYPED_FIELDS`, `_GENERATE_REMOVED_FIELDS`, and `_AUDIT_DROPPED_FIELDS`. Correct on all counts: it is a bare mode token (not path-typed, so basenaming would be wrong), not sensitive (so audit lines *should* retain it — they do, via the `_AUDIT_DROPPED_FIELDS` filter at line 1075), and `quant_skip`/`quant_only` entries are path-shape-rejected at the validator (params_validation.py:368) so the audit echo stays path-free.
Remediation: none required.

**[INFO] (d) Description text is clean**
Location: `comfyless/mcp_server.py:781-790`
Risk: none — verified. No filesystem paths, hostnames, or secrets; ADR/slice references and the "Blackwell" hardware mention match the established style of neighboring descriptions (e.g. `nag_scale` cites ADR-023/024). The "prefer fp8 until the live smoke lands" steering is appropriate advisory hygiene for a not-yet-quality-validated mode.
Remediation: none required.

**[INFO] Pre-existing, named for the record: MCP in-process path has no boundary-time quant membership check**
Location: `comfyless/mcp_server.py:2126-2134` → `nodes/eric_diffusion_utils.py:1749`
Risk: an off-enum quant string from a schema-ignoring client travels past validation, through catalog/model resolution and cache-key construction, and is only rejected by `build_quant_config`'s `ValueError` deep in the load path (the daemon surface checks membership earlier, at server.py ~190; the CLI at generate.py:2832; the MCP surface has no equivalent early gate). This fails closed and predates this diff — it is not introduced or worsened by deriving the enum. Not issuing a numbered requirement; if a cheap early gate is ever wanted, a two-line `if q not in QUANT_MODES: raise MCPValidationError` before step 7.5 would mirror the daemon.
Remediation: optional, out of this slice's scope — record here only.

**Dual-constant drift note:** the diff binds the advertised enum to `params_validation.QUANT_MODES`, while runtime enforcement on the in-process path uses `nodes.eric_diffusion_utils.QUANT_MODES`. Drift between them would fail closed (advertised mode rejected at load), and `test_quant.py:244` pins the sync in the CI-gated battery. Acceptable.

### Verdict

**PASS — no new requirements issued (numbering stays at req 66).** The diff is advisory-surface only: it changes what the MCP tool schema *tells* the client, sourced from the same constant the validators use, and changes nothing about what the server accepts, emits, audits, or redacts. Safe to commit alongside the main slice-NV review.
