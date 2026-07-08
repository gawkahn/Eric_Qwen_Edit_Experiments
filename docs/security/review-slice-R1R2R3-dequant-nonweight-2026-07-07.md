AI-Disclosure: security-auditor (Claude Opus 4.8) authored; Grant reviewed. Design-phase review, pre-code — extends review-slice-C-fp8-single-file-2026-07-02.md (F1-F12), review-slice-Cd-comfy-quant-2026-07-02.md (D1-D9, reqs 11-20), and review-slice-PQ-partial-quant-2026-07-07.md (PQ-1..6, reqs 31-38). New requirements numbered from 39.

# Delta security review — ADR-019 Slice R1/R2/R3 (non-weight fp8 upcast + dequant-to-bf16 load mode + quant wiring)

## Summary

Three changes to the caller-supplied fp8 weight-file parser, all inside the existing Review-bar surface (`nodes/eric_diffusion_fp8_ops.py` + the slice-C routing in `eric_diffusion_utils.py` + the quant wiring in `comfyless/generate.py`). **R1** relaxes the loader's unconditional "non-`.weight` fp8 → reject" (line 574-577) so that a *fully-bare* non-`.weight` fp8 tensor (norm scales, biases, modulation params) is skipped and upcast to bf16 in step 2, while any non-`.weight` fp8 tensor carrying a scale/descriptor binding still rejects — the direct analog of PQ-1/PQ-2's present-value principle applied to the non-weight namespace. **R2** adds a `dequant_fp8` flag that, when set, returns the fully-built bf16 model after step 2 (dequant via validated weight_scale + naked/non-weight upcast) and skips step 3 (the sm89 gate + fingerprint-swap to `ScaledFp8Linear`), so the existing torchao `quantize_module` can convert a clean all-bf16 model — routing around the NaN-producing LoRA-merge-into-fp8-resident path. **R3** hoists `build_quant_config` above the component-override block so `quant_selected` is known when the transformer override loads, enabling `dequant_fp8=(transformer_slot in quant_selected)`.

Threat model is unchanged from the whole chain: same-uid solo-desktop operator loading a community civitai checkpoint they chose; ceiling is silent numerical corruption of their own generation (safetensors is unpickled, mmap-bounded — no RCE, no cross-user boundary). Assessment: **all three add no new attack primitive.** R1 introduces no new parse surface (non-`.weight` fp8 already flows through the step-2 bf16 upcast at lines 650-658; the only change is relaxing the step-1 raise). R2 skips a step that contains *zero* validation — all scale validation and coverage checks live in step 1, which R2 leaves untouched — so the residency swap is a pure optimization whose omission loses no control. R3 is pure computation from CLI args + `model_index.json` in one process, so no TOCTOU. The risk is entirely in *how* the relaxations are coded: R1 must reject on any present binding (the PQ-2 silent-448×-discard lesson), and R2's flag must gate ONLY the step-3 return and never leak into steps 1-2 where validation lives.

## Coverage
Reviewed:
- `docs/security/review-slice-C-fp8-single-file-2026-07-02.md` (F1-F12) — whole file
- `docs/security/review-slice-Cd-comfy-quant-2026-07-02.md` (D1-D9, reqs 11-20) — whole file
- `docs/security/review-slice-PQ-partial-quant-2026-07-07.md` (PQ-1..6, reqs 31-38) — whole file
- `nodes/eric_diffusion_fp8_ops.py:1-1034` — full module; specifically `classify_fp8_single_file` (94-199), `_classify_cq` (202-343, incl. PQ naked block 240-271), `_validate_scale` (346-373), `ScaledFp8Linear` (376-488), `load_scaled_fp8_component` (506-728, pairing loop 561-644, dequant 648-678, swap 680-728)
- `comfyless/generate.py:780-981` — `_load_pipeline`, override block (831-905), quant-config block (913-941)
- `nodes/eric_diffusion_utils.py:580-699` (`_load_single_weights` fp8 routing), `1211-1270` (`load_component`), `1488-1573` (`build_quant_config`, `quantize_module`)

Not reviewed (and why):
- The proposed R1/R2/R3 code — does not exist (design phase); the current tree implements PQ but not R1/R2/R3 (line 574 still raises unconditionally; `load_scaled_fp8_component` has no `dequant_fp8` param — confirms pre-code)
- `nodes/eric_krea2_convert.py` `build_krea2_transformer` — invoked on the post-upcast bf16 dict; non-`.weight` upcast tensors reach it as bf16 identically to any plain-cast Krea file, no new surface; internals not re-read
- `torchao` `Float8DynamicActivationFloat8WeightConfig` internals — the dequant→requant numerics are torchao's, out of this diff's scope; treated as an assumption where relevant

## Findings

**[HIGH] 39 — R1 bare-check MUST reject on any present binding (PQ-2 present-value principle, non-weight namespace)**
Location: `load_scaled_fp8_component` pairing loop, replacing the `if not k.endswith(".weight"): raise` at `nodes/eric_diffusion_fp8_ops.py:574-577`
Risk: If R1 is coded as "non-`.weight` fp8 → skip/upcast" without re-checking for a bound scale, a non-`.weight` fp8 tensor carrying one of the recognized scale suffixes is upcast to bf16 with its scale silently discarded — the same silent-corruption class PQ-2 closes for `.weight` bases. "Bare" must be defined by scale-ABSENCE, never by "it isn't a `.weight`."
Remediation: For a non-`.weight` fp8 tensor `k`, assert none of `{k+".weight_scale", k+".input_scale", k+".scale_weight", k+".scale_input", k+".comfy_quant"}` is in `sd`; fully bare → `continue` (step 2 upcasts); any hit → `raise ScaledFp8FormatError` naming `k` and the offending suffix. Keep this branch strictly AFTER any `.weight`-based logic so it only handles the non-`.weight` case, mirroring the existing `_naked_forbidden` set.

**[HIGH] 40 — R2 `dequant_fp8` MUST gate ONLY the step-3 return; steps 1-2 run identically**
Location: `load_scaled_fp8_component`, new flag; the return point must sit after model construction (after both the krea branch and the `from_single_file` branch, before the sm89 gate)
Risk: All scale validation (`_validate_scale`, F2/F3), per-tensor coverage (F6), stray-scale (F1), and dangling-descriptor (D1) checks live in step 1; the naked/non-weight upcast lives in step 2. If the implementer threads `dequant_fp8` into steps 1-2 — e.g. skipping input_scale validation because it "won't feed `_scaled_mm`" in dequant mode — a crafted `input_scale` (0/NaN/Inf/subnormal, F2 class) slips past validation. The value is unused in dequant mode today, but a future resident-fallback or a mislabeled flag would then hit an unvalidated scale.
Remediation: Implement `dequant_fp8` as a single `if dequant_fp8: return model` immediately after model construction. Steps 1-2 must be byte-identical regardless of the flag — including `_validate_scale(is_key, ...)` on cq-a/ca/cb. Default `dequant_fp8=False` so today's resident behavior is the fail-closed default.

**[MED] 41 — R1 non-`.weight` naked set must be enumerated/logged and asserted out of residency (PQ-4/PQ-5 analog)**
Location: `load_scaled_fp8_component` (the new R1 skip) and the fingerprint-swap step 3
Risk: The design states these tensors "never enter `fp8_entries`, never get residency," but that is a claim, not an enforced invariant. A non-`.weight` fp8 tensor that is 2D (e.g. a mislabeled matrix) must still be upcast raw and must NOT be swapped to `ScaledFp8Linear`. Without an enumerated log, an operator loading a crafted file where a sensitive param was silently demoted has no visibility (the PQ-4 finding, re-raised for the non-weight set).
Remediation: (a) The R1 skip must `continue` before any `fp8_entries` insertion, so non-`.weight` tensors never reach the step-3 swap (step 3 already only indexes `nn.Linear`, so a bf16 upcast param is inert — assert this holds). (b) Emit one aggregate INFO line via `_safe_name` naming the count and a bounded sample of the skipped non-`.weight` fp8 tensors, matching the existing naked-`.weight` log. One line per file, not per tensor.

**[MED] 42 — R2 dequant mode drops the file's `input_scale` (cq-a); numerically correct but state the invariant, don't skip its validation**
Location: step-2 dequant `bf16_sd[k] = (w.to(float32) * ws).to(dtype)` vs the design claim "scales are still validated AND applied"
Risk: For cq-a/ca files the design's claim is imprecise: `weight_scale` is applied (dequant multiplies by it), but `input_scale` is validated then DROPPED — the bf16 weight is fully captured by `W_fp8 * weight_scale`, and torchao then does dynamic activation quantization, so the file's static `input_scale` is discarded. This is correct numerics (dynamic activation quant is at least as accurate as a fixed file scale) and NOT a corruption path — but a future reviewer reading "scales applied" could wrongly conclude `input_scale` corruption is impossible for the wrong reason, and an implementer could "optimize" by skipping the now-unused `input_scale` validation (see finding 40).
Remediation: Document in the module docstring's PQ/R2 section that in dequant mode `input_scale` is validated-then-dropped (activation-quant param, correctly superseded by torchao dynamic activation quant), and that `weight_scale` is the only file scale that reaches numerics. Keep `_validate_scale(is_key, ...)` unconditional (finding 40).

**[LOW] 43 — Aliasing analysis: no accept-side collision; C-b suffix names cause safe false-rejects; one residual assumption**
Location: classify `_any_marker`/`cb_hits`, `_classify_cq` cb-copresence reject, R1 forbidden-suffix set
Risk/analysis: The norm-scale names in the failing file (`...qnorm.scale`, `prenorm.scale`, `postnorm.scale`, `last.norm.scale`) end in `.scale` — none of the four scale suffixes (`.weight_scale`, `.input_scale`, `.scale_weight`, `.scale_input`) is a suffix of `.scale`, so `str.endswith` never matches them; they are neither collected as scale keys nor mistaken for bindings. The residual: if a community file ever named a param literally ending in `.scale_weight`/`.scale_input` (fp8 or not), classify's `cb_hits` fires and — for a cq file — the cb-copresence guard REJECTS the whole file. That is a false-reject (availability), never a false-accept — the safe direction. So R1 introduces no accept-side aliasing.
Remediation: None required for correctness. Name the residual assumption explicitly in the docstring: **non-`.weight` fp8 tensors are assumed to carry no scale under any recognized OR future convention** — a format that quantized biases/norms with a scale bound under a name outside the 5-suffix set would be silently upcast at scale 1.0. This is the identical residual PQ accepted (PQ-6), extended to the non-weight namespace where the suffix guess is more speculative because no real convention scales non-weight tensors.

**[INFO] 44 — R3 hoist is TOCTOU-free and fail-closed; confirm the fallback path**
Location: `comfyless/generate.py` — moving `build_quant_config` above the override block
Risk/analysis: `build_quant_config` is pure over `model_path`, the CLI quant args, `device`, and `model_index.json` — it mutates nothing and is deterministic in-process, so the hoist introduces no check-then-use gap between computing `quant_selected` and loading the transformer file. The `ValueError` for an unknown quant mode simply fires earlier (fail-fast, still fail-closed). On any fallback — unsupported hardware, missing torchao, no eligible components — `build_quant_config` returns `selected={}`, so `dequant_fp8=(transformer_slot in {})` is `False` → today's fp8-resident behavior is preserved. That is the correct fail-closed default.
Remediation: None. Verify in code review that the `if sequential_offload: warn` and the `quant_config.quant_mapping.pop(slot, ...)` / in-place `quantize_module` logic are preserved verbatim across the hoist — the hoist must move only the `build_quant_config` call site, not the mapping-pop or the in-place quantize.

## Required negative/positive tests (R1 + R2), each its own case

R1 (amends/replaces PQ test 9):
1. Fully-bare non-`.weight` fp8 tensor (`mod.lin.bias` fp8, no suffixes) alongside a valid cq set → loads; assert the bias lands in the model as bf16 and never appears in `fp8_entries`.
2. Non-`.weight` fp8 tensor carrying `k+".weight_scale"` → reject at load, message names `k` + suffix (present-value guard, finding 39).
3. Non-`.weight` fp8 tensor carrying `k+".input_scale"` → reject.
4. Non-`.weight` fp8 tensor carrying `k+".scale_weight"` or `k+".scale_input"` → (for a cq file) whole-file reject via the cb-copresence guard; assert the reject fires (finding 43, safe false-reject).
5. Non-`.weight` fp8 tensor carrying `k+".comfy_quant"` → reject.
6. 2D non-`.weight` fp8 tensor, fully bare → upcast to bf16, NOT swapped to `ScaledFp8Linear` (finding 41).
7. Positive: the target layout — cq-w descriptored set + naked `.weight` set (PQ) + fully-bare non-`.weight` set (norm/bias/mod tensors) → loads; assert descriptored swap to `ScaledFp8Linear` (CUDA-gated), naked `.weight` and non-`.weight` land bf16, log line enumerates the non-`.weight` count (finding 41).

R2:
8. `dequant_fp8=True` on a cq-a file → returns an all-bf16 model (no `ScaledFp8Linear` present); assert step-1 `_validate_scale` still ran on both `weight_scale` and `input_scale` (feed a NaN `input_scale` → still rejects with `dequant_fp8=True`, proving finding 40).
9. `dequant_fp8=True` with a bad `weight_scale` (0/Inf/subnormal) → rejects identically to resident mode (validation not gated by the flag).
10. `dequant_fp8=False` (default) on the same file → today's fp8-resident behavior unchanged (regression: flag default is fail-closed).
11. `dequant_fp8=True` on the native-Krea branch (`is_krea2_comfy_checkpoint` true) → returns bf16 model after `build_krea2_transformer`, skips step 3 (return placement covers both build branches, finding 40).
12. Wiring (R3): with `--quant fp8` and a transformer override, assert `load_component` receives `dequant_fp8=True`; with `--quant none` (or a fallback that empties `quant_selected`), assert `dequant_fp8=False` (finding 44).

## Requirements delta (39-45, additive; 31-38 are PQ's, 21-30 DMR's, 11-20 C-d's, 1-10 baseline)

39. R1 relaxes the non-`.weight` raise to a fully-bare skip: a non-`.weight` fp8 tensor is upcast to bf16 iff it carries none of `{.weight_scale, .input_scale, .scale_weight, .scale_input, .comfy_quant}`; any binding present → loud reject naming tensor + suffix. Bare = scale-absence, never "not a `.weight`" (finding 39).
40. R2 `dequant_fp8` gates ONLY a `return model` placed after model construction (both krea and from_single_file branches) and before the sm89 gate; steps 1-2 (all validation, coverage, pairing, naked/non-weight upcast) run byte-identically regardless of the flag; default `False` (findings 40, 42).
41. R1-skipped non-`.weight` fp8 tensors never enter `fp8_entries` / step-3 residency; the skipped set is enumerated in one aggregate INFO line via `_safe_name`, not per-tensor (finding 41).
42. Module docstring updated: in R2 dequant mode `weight_scale` is applied and `input_scale` is validated-then-dropped (superseded by torchao dynamic activation quant); the dequant→requant round-trip is lossy-but-intended, not corruption (finding 42).
43. Docstring names the residual assumption: non-`.weight` fp8 tensors are assumed unscaled under all conventions; airtightness comes from the finding-39 reject, not from proving the class — the PQ-6 posture extended to the non-weight namespace (finding 43).
44. R3 moves only the `build_quant_config` call site above the override block; the `quant_mapping.pop(slot)` and in-place `quantize_module` logic are preserved verbatim; `dequant_fp8=(transformer_slot in quant_selected)` with an empty `quant_selected` on any fallback → `False` (finding 44).
45. PQ test 9 amended and tests 1-12 above land in `test_fp8_single_file.py` in the same slice as the code.

## Verdict

**APPROVABLE WITH CHANGES.** None of R1/R2/R3 adds an attack primitive, and all three are within the existing content-parser Review-bar surface. Merge is gated on the two load-bearing items — **finding 39** (R1 must reject on any present binding, the PQ-2 silent-discard lesson applied to the non-weight namespace) and **finding 40** (R2's flag must gate only the step-3 return, never the validation in steps 1-2) — plus the enumeration/docstring/test requirements 41-45. Findings 42-44 are precision and confirmation, not blockers. The single highest-consequence mistake an implementer can make here is defining R1's "bare" by "not a `.weight`" instead of by scale-absence, or letting `dequant_fp8` short-circuit `_validate_scale` — either reintroduces the ~448× silent-corruption class the whole chain exists to close.
