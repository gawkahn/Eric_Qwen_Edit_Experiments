AI-Disclosure: security-auditor subagent (Claude Fable 5) authored; Grant reviewed. Delta review (design phase, pre-code) — third review of the gated fp8 surface; baselines review-slice-C (1-10) and review-slice-Cd (11-20) carry forward.

# Delta security review — ADR-019 slice DMR (dequant→merge→requant)

## Summary

DMR replaces the four guard_direct_merge entry-gates with a per-target apply_merge_delta dispatcher (plain / torchao Float8Tensor / ScaledFp8Linear / raise) plus kind-tagged exact-restore backups. The delta trust class is unchanged (LoRA tensors were always untrusted and always merged on unquantized bases); NEW surfaces are the requant math (amax/448 persisted as scale state), the Parameter-object swap lifecycle, and the named_buffers() union in resolution maps — the last creating a genuinely new adversarial resolution target.

## Findings

**[HIGH] DMR-1 — Non-finite delta poisons the PERSISTED SCALE on quantized paths** (strictly worse than the plain path's value-only poisoning: NaN amax → NaN weight_scale → whole tensor unrecoverable except via backup). Mitigate: torch.isfinite(delta).all() gate at dispatcher entry, loud raise with target + non-finite count.

**[HIGH] DMR-2 — amax==0 divides by zero in ScaledFp8Linear requant** (all-zero merged tensor → scale 0 → poisoned buffer + future div-by-zero). Mitigate: short-circuit — store scale sentinel (>= F32 min normal), write zeros, log. Case 2 defensively too.

**[HIGH] DMR-3 — named_buffers() union lets adversarial LoRA keys land on scale buffers.** `foo.weight_scale.diff` resolves (via _adapter_module_path .diff marker + resolve_merge_target bare-base fallback) to the ScaledFp8Linear scale BUFFER, which the dispatcher would see as a plain fp32 tensor → case-1 add_ silently poisons the layer's quantization scale. Did not exist pre-DMR (named_parameters never exposed buffers). Mitigate: restrict merged buffers to k.endswith(".weight"); negative tests for .weight_scale/.input_scale/.comfy_quant-targeting keys.

**[HIGH] DMR-4 — Case-4 must be positive-match-then-explicit-raise, not a fallthrough else-plain.** An fp8-dtype tensor NOT owned by ScaledFp8Linear, or a torchao subclass under a renamed namespace, must raise — never take the plain add_. Source-inspection test: no direct param.data.add_ remains at any merge site.

**[MED] DMR-5 — LIFO unload assumption becomes structurally binding under Parameter-object swaps.** Same ordering semantics as today's plain backups, heavier failure. Mitigate: state LIFO in Vision invariant 2; WARN on non-LIFO unload (warn-don't-block).

**[MED] DMR-6 — Parameter swap aliasing (accelerate hooks, tied weights).** Mitigate: post-swap assert named_parameters()[target_key] IS the new object; document the no-tying/no-offload-hooks assumption.

**[MED] DMR-7 — Scale-coarsening: keep log-and-proceed at ratio>2; ADD a loud 'adapter likely corrupt or crafted' warning at ratio>1000** (still proceed — operator-initiated). --strict-lora blocking mode is follow-on TECH_DEBT if wanted.

**[MED] DMR-8 — Resolution-map merge order spec'd:** {**named_parameters(), **named_buffers-.weight-only} (buffers override), commented at each site; unit test with PEFT-wrapped bf16 + ScaledFp8Linear coexistence.

**[MED] DMR-9 — restore_merge_backup must invalidate ScaledFp8Linear caches** (_fallback_weight=None, _warned_fallback=False) or post-restore forwards serve the MERGED weights from stale cache.

**[MED] DMR-10 — Requant OUTPUT scales validated through _validate_scale before persisting** (closes the F2 finite-positive-normal gap at the merge boundary; on failure restore from backup and raise).

## Requirements for implementation (21-30)

21. Delta finiteness gate at dispatcher entry (raise: target, count).
22. amax==0 short-circuit (case 3 sentinel scale + zeros + log; case 2 defensively).
23. .weight-only buffer filter in resolution maps + adversarial-suffix negative tests.
24. Positive-match-then-explicit-raise dispatch precedence (ScaledFp8Linear.weight → torchao → orphan-fp8 RAISE → plain dtypes → RAISE); source-inspection: no direct param.data.add_ at merge sites.
25. LIFO unload order guard: Vision states it; WARN on non-LIFO.
26. Post-swap aliasing assert + documented no-tying/no-hooks assumption.
27. Coarsening ceiling: log>2, loud corrupt-warning>1000, proceed.
28. Resolution-map merge order spec'd + commented + coexistence unit test.
29. Restore invalidates ScaledFp8Linear caches; negative test (merge→forward→unload→forward == original).
30. _validate_scale on requant output before persist; failure → restore + raise.
