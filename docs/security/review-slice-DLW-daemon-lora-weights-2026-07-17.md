AI-Disclosure: security-auditor (Claude Fable 5) authored the review; Claude (Fable 5) transcribed and folded the conditions; Grant reviews at merge.

# Delta security review — slice DLW: daemon LoRA weight application

Reviewed diff: `comfyless/server.py` (Red Zone) + `comfyless/generate.py` +
the two test suites, against `docs/vision/slice-DLW-daemon-lora-weights.md`
and the DQ contract (`review-slice-DQ-daemon-quant-2026-07-03.md` F2/F3/F4).
Numbering continues from the DQ review (F9+).

## Summary

Slice DLW makes the daemon's incremental LoRA diff weight-aware and gives the daemon the cumulative `set_adapters` weight application the CLI path gained in 1f52672, via a shared helper `apply_adapter_weights` extracted from `_apply_loras`. Threat model per ADR-001: same-uid unix socket (0700 dir / 0600 socket), single-threaded accept loop; primary assets are accept-loop availability, path confinement, and the never-serve-the-wrong-artifact cache invariant — extended by the DQ contract (quantized requests carry the LoRA `(path, weight)` set in the cache key and full-evict on any change; only unquantized pipelines take the incremental diff). Overall posture: the delta is sound. The quantized full-evict contract is structurally preserved (the weight-update branch is unreachable on quantized pipelines — F13), state desync between bookkeeping and pipe adapter scale is bounded to one request and always accompanied by a surfaced warning (F9), and the direct-merge warning is consistent with existing operator-facing `lora_warnings` semantics (F12).

## What was checked

The request `loras` list from the socket parse (`json.loads`, default) through `validate_lora_entry` (float type check, no finiteness check), into `_handle_generate`'s diff, the cumulative `apply_adapter_weights` call, and the cache key (unchanged). Daemon verified single-threaded (one `srv.accept()`, no threading); `generate()` with `_cached_pipeline` skips `_apply_loras`, so weights apply exactly once per request on the daemon path; `mcp_server.py` imports only `_within` from the server — daemon path-bearing `lora_warnings` cannot reach the agent surface. Every failure branch checked for fail-open vs fail-closed and record/pipe desync.

## Findings

**[MEDIUM] F10 — Silent `_te`-half deactivation when adapter discovery fails (fail-open in the shared helper).**
`apply_adapter_weights` swallowed `get_list_adapters()` exceptions into an empty set; `set_adapters` then REPLACES the active set without the Kohya `<name>_te` halves — silently deactivating them, the exact silent state the function's docstring forbids, with no warning returned. On the daemon this repeats every request.
**Condition 1 (in-scope): fix in this slice. FOLDED:** the except branch now logs a WARNING and returns a `discovery_warn` string ("Kohya text-encoder LoRA halves, if any, may be inactive") that reaches CLI logs and daemon `lora_warnings`; combined with the set_adapters-failure message when both fire.

**[MEDIUM] F11 — Nonfinite LoRA weight passes the boundary and now reaches `set_adapters` and the diff/cache-key logic.**
Socket `json.loads` accepts `NaN`/`Infinity`; `validate_lora_entry` has no `isfinite` gate. `"weight": NaN`: unquantized → perpetual weight-change churn (`abs(rec - NaN) <= eps` always False) + `set_adapters(NaN)` garbage; quantized → `NaN != NaN` cache key never matches → full evict+reload every request (DQ F7 availability-churn class). Predates DLW; DLW makes weight an effective per-request compute input. `refine.py` already rejects nonfinite at its LLM boundary — the daemon boundary is the odd one out.
**Condition 2 (out of scope — do NOT fold): TECH_DEBT entry appended** with binding trigger: precondition of the `--json`/LLM-agent wiring commit, or the next `params_validation.py` slice, whichever first. Fix shape: `math.isfinite` in `validate_lora_entry` + `parse_constant` rejection at the socket parse.

**[INFO] F9 — Record/pipe weight desync is bounded, self-healing, never silent.** `rec["weight"]` updates before the cumulative call; on failure a warning lands in response metadata and the unconditional per-request re-apply heals or re-warns. Cosmetic: the "remain at full strength" wording was inaccurate on the daemon — **reworded in-slice** to "remain at their previously applied scale (full trained strength if never scaled)". Warnings are dropped from ERROR responses (stderr only) — pre-existing.

**[INFO] F12 — Direct-merge warning path embedding is consistent, not a new leak.** Same class as existing `lora_warnings` (reflects the caller's own path over the same-uid socket). Scope-change condition: if a daemon-backed MCP/agent client ever lands, daemon `lora_warnings` joins the ADR-015 path→catalog-name mapping set (fold into the existing `--json` scope-change gate; DQ F6 tracks the sibling item).

**[INFO] F13 — DQ F2/F3/F4 quantized posture structurally intact.** The cache key's quant LoRA tuple uses identical float coercion to the diff loop; any weight change under quant flips the key → evict BEFORE the diff → `loaded_loras == []` → the weight-update branch only ever sees exactly-equal floats on quantized pipelines. Epsilon-vs-exact divergence resolves safe-direction only (quant over-evicts on sub-epsilon changes; unquantized no-ops them). The per-request cumulative call on a warm quantized pipeline is an idempotent re-pin — no adapter add/remove/merge; nvfp4 refusal and DMR posture untouched.

**[INFO] F14 — No reentrancy** with NAG per-call processors or the refiner cache (single-threaded; apply runs strictly before `generate()`). Pre-existing property: the Hunyuan refiner shares the base text encoder, so a scaled `_te` half is scaled for the refiner too — a property of shared-encoder LoRAs, not this slice.

**[INFO] F15 — The 1e-9 epsilon has no drift or abuse surface for finite inputs** (exact request floats stored; no accumulation; absolute epsilon far below perceptual granularity on |w|≲4). Only nonfinite abuses it (F11).

Scope creep: none. Request schema, path validation, cache key, socket framing untouched (Vision invariant 5).

## Verdict

**PASS-with-conditions — both conditions discharged:** F10 fixed in-slice (discovery-failure warning + message reword); F11 recorded in TECH_DEBT with the binding `--json`-precondition trigger.

Session confirmations requested by the auditor: `test_apply_loras.py` assertions were NOT modified in this diff (verified: `git diff --stat test_apply_loras.py` is empty), so invariant 1's "CLI behavior unchanged" claim rests on the unmodified suite passing — it does (24/24 battery).

## Post-review compliance (code-reviewer, Fable, same day)

Code review returned CHANGES REQUIRED: finding 1 — duplicate LoRA paths in one request double-loaded under the same adapter name (pre-existing behavior, not a regression: the old `loaded_paths` set was never updated mid-loop either) → fixed by registering each successful load in `loaded_by_path` immediately (second occurrence takes the weight-update branch, last wins) + the now-dead `loaded_paths` writes removed; finding 2 — this review doc + TECH_DEBT resolution were required before commit (this document). Optional seam-test suggestion adopted: one daemon scenario runs the REAL `apply_adapter_weights` against a recording fake pipe, pinning the `[(name, weight)]` contract end-to-end.
