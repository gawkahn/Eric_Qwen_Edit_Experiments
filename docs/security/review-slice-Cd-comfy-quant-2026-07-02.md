AI-Disclosure: security-auditor subagent (Claude Fable 5) authored; Grant reviewed. Delta review (design phase, pre-code) — extends review-slice-C-fp8-single-file-2026-07-02.md; baseline findings F1-F12 carry forward unchanged.

# Delta security review — ADR-019 Slice C-d (comfy_quant descriptor)

## Summary

C-d turns the blanket `.comfy_quant` reject into a bounded JSON-parse-then-allowlist unlocking three real fp8 finetunes (two weight-only). It preserves reviewed C-a semantics for descriptor-plus-both-scales files (`cq-a`) and adds a weight-only variant (`cq-w`) routed through the existing dequant-matmul path with `_scaled_mm` never invoked. Small code footprint, non-trivial threat delta: attacker-controlled JSON bytes read at classify time, and a relaxation of the F6 per-tensor coverage rule for one variant.

## Findings

**[HIGH] D1 — Dangling `.comfy_quant` on a non-fp8 base must reject.** Every descriptor must pair with an fp8 `<base>.weight`; assert at classify AND load (extend the stray scan).

**[HIGH] D2 — "Header-only detection" invariant is broken by descriptor reads; the exception must be gated from the HEADER.** Assert dtype U8, 1-D, numel <= 4096 from header shape BEFORE get_tensor (a crafted 2^30-numel descriptor must reject without materialization); strict-UTF-8 decode; json root must be dict; format must be str; literal allowlist {"float8_e4m3fn","float8_e5m2"}. Amend the module docstring's header-only claim to spell out this bounded exception.

**[HIGH] D3 — cq/plain mixing within one file must reject.** Either EVERY fp8 weight carries a descriptor or NONE do; partial coverage → loud reject naming one example of each. Also reject cq + `scaled_fp8` (C-b) marker co-presence.

**[HIGH] D4 — JSON parse strictness enumerated.** Non-dict roots (1/null/list), non-string format, UnicodeDecodeError (strict), empty tensor, trailing garbage → each a loud reject. Allowlist check is literal equality — no normalization/case-fold/startswith. Unknown fields beyond {format, full_precision_matrix_mult} log-once at INFO via _safe_name.

**[MED] D5 — Descriptor semantic scope lock.** The JSON is consumed for exactly ONE decision: format allowlist pass/fail. `full_precision_matrix_mult` is NOT read (weight-only mode is inferred from input_scale ABSENCE — tensor presence, not attacker JSON). Any future code reading additional descriptor fields requires a fresh security review. cq-a and cq-w produce numerically equivalent outputs, so the inference choice is safe.

**[MED] D6 — F6 waiver has no downgrade path, but make the enforcement explicit.** cq-a iff EVERY fp8 weight has both scales; cq-w iff EVERY fp8 weight has ONLY weight_scale; mixed input_scale coverage within a cq file → loud reject. A crafted ca file gaining descriptors becomes cq-a (both scales still validated); removing input_scale yields cq-w where validation is skipped only because the value is absent — no bypass of present-but-crafted values.

**[MED] D7 — Header-time scalar-F32 scale sanity extends to cq variants** (the existing ca/cb loop must also cover cq-mode scale keys).

**[LOW] D8 — Bounded logging:** one INFO line per file (descriptor count + unique formats), not per-layer.

**[INFO] D9 — cq-w VRAM economics** (fp8 + bf16 dequant cache after first forward) — documented, not a security issue.

## Requirements delta (11-20, additive to baseline 1-10)

11. Bounded .comfy_quant read: header-gated (U8, 1-D, numel<=4096) BEFORE get_tensor; docstring exception spelled out.
12. Descriptor→fp8-weight pairing enforced at classify and load.
13. cq/plain mutual exclusion per file (extends F5).
14. JSON strictness: strict UTF-8, dict root, str format, literal allowlist; empty/trailing-garbage reject; unknown fields log-once.
15. Descriptor scope lock docstring: ONE decision only; new fields = new review.
16. Variant from tensor presence: cq-a = all both-scales; cq-w = all weight-scale-only; mixed → reject.
17. Header scalar-F32 sanity covers cq scale keys.
18. ScaledFp8Linear weight-only mode: input_scale=None → existing dequant path, _scaled_mm never called; no new compute path.
19. Negative tests, each its own case: non-str format; list/scalar/null JSON root; unknown formats (fp4/nvfp4/e4m3fnuz); dangling descriptor on bf16 layer; mixed cq+plain; oversize descriptor rejected without materialization; non-U8 dtype; UTF-8 failure; empty tensor; trailing garbage; two different allowlisted formats in one file (allowed — tested); unknown JSON field logs without rejecting.
20. Module docstring invariant-1 wording amended (bounded descriptor-bytes exception).
