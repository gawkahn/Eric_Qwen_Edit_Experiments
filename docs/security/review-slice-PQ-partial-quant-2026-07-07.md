AI-Disclosure: security-auditor (Claude Opus 4.8) authored; Grant reviewed. Design-phase review, pre-code — extends review-slice-C-fp8-single-file-2026-07-02.md (F1-F12) and review-slice-Cd-comfy-quant-2026-07-02.md (D1-D9, reqs 11-20).

# Delta security review — ADR-019 Slice PQ (partial-quant / naked-fp8 coexistence)

## Summary

The proposal relaxes the D3 all-or-nothing descriptor rule (`_classify_cq`, `nodes/eric_diffusion_fp8_ops.py:228-239`) so that "naked" fp8 `.weight` tensors — no `.comfy_quant`, no `.weight_scale`, no `.input_scale`, no C-b marker — may coexist with the descriptored set and be upcast to bf16 (the already-reviewed `cc` plain-cast operation). The file in scope (`krea2turbobadmilkmela_v10`) has 256 descriptored + 8 fully-bare fp8 layers. Threat model is unchanged from the baseline: same-uid solo-desktop operator loading a community checkpoint they chose; the ceiling is silent numerical corruption of their own generation, not RCE or cross-user compromise (safetensors is mmap-bounded, unpickled).

My assessment: the relaxation adds **no new attack primitive** — naked fp8 carries no JSON and no scale, so it expands neither the descriptor-parse surface (D2 numel gate untouched — naked layers have no descriptors) nor the scale-validation surface (`_validate_scale` is only reached for entries that have scales), and the memory ceiling stays at "a bf16 model," so no new DoS. **BUT** the "bounded to fully-bare only" claim is **NOT airtight as the current code stands**. The danger is entirely in *how* "naked" is detected. The current D3 line is the only thing that today prevents a fp8 base that has a scale-but-no-descriptor. If the relaxation defines naked as "no descriptor" (descriptor-absence), then a crafted layer carrying both scales but no descriptor gets upcast **ignoring its present scale** — an ~448× silent weight corruption, exactly the F6/D3 class. If naked is defined as "no scale of any kind" (scale-absence) but classify adds no explicit reject for the scale-present-descriptor-absent combos, those combos leak through to the loader where behavior diverges by variant (stray-reject in some, silently-loaded-as-scaled in others). The design is therefore **APPROVABLE WITH CHANGES**: the naked set must be enumerated and each member asserted **fully bare at classify** (reject any C-a/C-b scale suffix or descriptor on a naked base), the loader must re-assert by **scale-absence** (never descriptor-absence), the existing C-b co-presence reject at line 199 must be preserved untouched, and the negative-test battery below must land with the code.

## Coverage

Reviewed:
- `docs/security/review-slice-C-fp8-single-file-2026-07-02.md` (F1-F12, reqs 1-10) — whole file
- `docs/security/review-slice-Cd-comfy-quant-2026-07-02.md` (D1-D9, reqs 11-20) — whole file
- `nodes/eric_diffusion_fp8_ops.py:1-989` — full module; specifically `classify_fp8_single_file` (82-187), `_classify_cq` (190-311), `_validate_scale` (314-341), `ScaledFp8Linear` (344-456), `load_scaled_fp8_component` (474-683)
- `CLAUDE.md` Review bar section

Not reviewed (and why):
- The proposed code — does not exist (design phase, per prompt)
- `nodes/eric_diffusion_utils.py` classify/route caller — not in the stated edit scope; assumption noted below
- `nodes/eric_krea2_convert.py` `build_krea2_transformer` — invoked at `:627` on the post-upcast bf16 dict; naked layers reach it as bf16 identically to any plain-cast Krea file, so no new surface, but its internals were not re-read
- `torch._scaled_mm` device behavior — naked layers never reach it (they stay bf16), so out of scope for this delta

**Numbering note:** reqs 21-30 are already consumed by the DMR review (`review-slice-DMR-quantized-merge-2026-07-03.md`, referenced at `:688`). To avoid collision I number this delta **31-38**, not 21+.

## Combination enumeration (question 2)

For a single fp8 `.weight` base inside a file that reaches `_classify_cq` (i.e. at least one `.comfy_quant` exists). Variables: D=descriptor, W=`.weight_scale`, I=`.input_scale`, B=C-b marker. **Any B present anywhere in the file → whole-file REJECT at `:199` (unchanged; question 4 answered — that guard runs before the D3 block and must not be touched).** So the table assumes B=0:

| D | W | I | Required outcome | Enforced by the CURRENT code after a naive relaxation? |
|---|---|---|------------------|--------------------------------------------------------|
| 0 | 0 | 0 | accept-plain (upcast bf16) | ✓ — this is the target case |
| 0 | 0 | 1 | **REJECT** | ✗ classify passes; load: stray-reject (line 576-587) in most variants |
| 0 | 1 | 0 | **REJECT** | ✗ classify passes; load: cq-a F6-rejects (`:548`), **cq-w silently loads it as weight-only scaled** |
| 0 | 1 | 1 | **REJECT** | ✗ classify passes; load: **cq-a silently loads it as fully scaled** (`:548` both scales present → added to `fp8_entries`) |
| 1 | 0 | 0 | REJECT (missing weight_scale) | ✓ existing missing_ws (`:282-287`) |
| 1 | 0 | 1 | REJECT | ✓ existing missing_ws (`:282-287`) |
| 1 | 1 | 0 | accept cq-w (scaled) | ✓ existing (D6 all-or-none, `:288-296`) |
| 1 | 1 | 1 | accept cq-a (scaled) | ✓ existing (D6 all-or-none) |

The three rows that need the new control are **{0,0,1}, {0,1,0}, {0,1,1}**. They are NOT uniformly rejected: the loader binds by scale presence, so a descriptor-less base that carries its scales is silently accepted as a scaled layer (rows {0,1,0} in cq-w, {0,1,1} in cq-a) — numerically correct only by luck (the descriptor's sole job is the format allowlist, which is derivable from the fp8 dtype), but it means the "descriptor required" and "naked = fully bare" invariants are both violated with no error. Hence: **not airtight without PQ-1.**

## Findings

**[HIGH] PQ-1 — "Fully-bare" is not enforced at classify; scale-present/descriptor-absent combos leak through**
Location: `_classify_cq`, replacing `nodes/eric_diffusion_fp8_ops.py:228-239` (the D3 naked block)
Risk: Removing the D3 naked reject without a replacement lets combos {0,0,1}/{0,1,0}/{0,1,1} (naked base carrying a C-a scale) pass classification. They then load with variant-dependent behavior — silently accepted as a scaled `ScaledFp8Linear` in cq-w/cq-a — so the design's "bounded to fully-bare only" claim is false and the descriptor gate is bypassable.
Remediation: After computing `naked = fp8_bases - cq_bases`, do not accept it silently. Enumerate it and assert each naked base carries **none** of `{base+".weight_scale", base+".input_scale", base+".scale_weight", base+".scale_input", base+".comfy_quant"}` in `key_set`; any hit → `ScaledFp8FormatError` naming the base and the offending suffix. Only a base with zero scale/descriptor keys is a legal naked layer.

**[HIGH] PQ-2 — Loader naked-detection must key on scale ABSENCE, never descriptor absence**
Location: `load_scaled_fp8_component` pairing loop, `nodes/eric_diffusion_fp8_ops.py:539-571`
Risk: The loader is the D1-pattern re-assertion point (req 12). If the relaxation is implemented as "no descriptor → upcast," a base carrying both scales but no descriptor (row {0,1,1}) is upcast to bf16 with its `weight_scale` silently discarded — the fp8 storage values (which are `W_true / scale`, magnitude ~1/448 of true) load as-is, corrupting that layer by hundreds× with no error. This is the exact silent-corruption class F6/D3 exist to close.
Remediation: In the pairing loop, treat a base as naked **iff** `ws_key not in sd and is_key not in sd and base+_CQ_SUFFIX not in sd` (fully bare) → `continue` (skip `fp8_entries`; it falls through to the `else` bf16 upcast at `:613`). Any partial coverage (some but not all of ws/is present, or a scale without... ) must reject, not skip. Do not add a branch that skips on descriptor absence. The existing stray-scale check (`:575-587`) and D1-at-load (`:589-594`) then remain the backstop for rows {0,0,1}/{0,1,0}.

**[MED] PQ-3 — Scope the edit to the D3 naked block; do not touch the C-b co-presence reject**
Location: `nodes/eric_diffusion_fp8_ops.py:199-203` (cb reject) vs `:228-239` (D3 naked)
Risk: F5/D3's other job — rejecting `.comfy_quant` co-present with any C-b marker (`.scale_weight`/`.scale_input`/`scaled_fp8`) — lives at `:199`, ahead of the D3 naked block. A refactor that merges or reorders these could weaken it, allowing a mixed-convention file that the F1 collision class depends on rejecting.
Remediation: Keep the `if cb_hits: raise` at `:199` byte-identical. Confine the PQ change to the `naked` block at `:228-239`. Add a regression test asserting cq + `.scale_weight` still rejects (see PQ tests).

**[MED] PQ-4 — Naked set must be enumerated and logged (D8 pattern), never silent**
Location: after the new naked assertion in `_classify_cq`; alongside the existing D8 log at `:276-277`
Risk: The design states the naked set is "logged, never silent," but the current single D8 line reports only descriptor count/formats. An operator loading a crafted file where a sensitive layer was silently demoted to naked has no visibility.
Remediation: Emit one aggregate INFO line naming the naked count and, bounded, the first few base names via `_safe_name` (e.g. `"8 plain-fp8 (naked) layers upcast to bf16: [...]"`). One line per file, not per layer (D8 economics).

**[LOW] PQ-5 — Naked fp8 tensor not ending in `.weight` — confirm the loader backstop and test it**
Location: `nodes/eric_diffusion_fp8_ops.py:230` (classify `.weight` filter) vs `:542-545` (loader)
Risk: classify's `fp8_bases` only sees `.weight` fp8 tensors, so a naked fp8 tensor like `foo.bias` is invisible at classify. It is caught at load by the `if not k.endswith(".weight"): raise` at `:542`, but only because that raise is unconditional; the relaxation must not move the naked `continue` above it.
Remediation: Ensure the naked-skip branch sits *after* the `.weight` check, so a non-`.weight` fp8 tensor still reaches the raise. Add a negative test.

**[INFO] PQ-6 — A naked-fraction cap is NOT required; state it explicitly**
Risk/assessment: A crafted file that is 99% naked + one descriptor loads a mostly-bf16 model — the memory ceiling is a normal bf16 checkpoint, no new DoS, and the attacker gains nothing over editing weights directly (their own file). A cap would be defense-in-depth only and is not warranted given PQ-1 makes the boundary airtight. Recommend documenting "no cap; airtightness comes from PQ-1, not from bounding the fraction" so a future reviewer doesn't assume a missing cap is an oversight.
Assumption: the caller/router in `eric_diffusion_utils.py` does not itself re-derive variant or bypass `_classify_cq`. If that assumption is wrong, PQ-1/PQ-2 alone are insufficient — flag for verification when the router is next touched.

## Answers to the posed questions

1. **New attack primitive?** No — provided PQ-1/PQ-2 land. Naked fp8 has no JSON (no descriptor-parse expansion; D2 numel gate untouched) and no scale (no `_validate_scale` expansion). The only new operation is the fp8→bf16 upcast, identical to the already-reviewed `cc` cast. No new DoS bound (ceiling = bf16 model size). Corrupting a naked layer is self-inflicted on the operator's own file, crossing no trust boundary.
2. **Airtight?** Not as-is — see the table. Rows {0,0,1}/{0,1,0}/{0,1,1} are the gaps; PQ-1 (classify reject) closes them and is the *only* airtight control, because the loader binds by scale presence and will silently accept a descriptor-less-but-scaled layer.
3. **New compensating controls needed?** Yes: (a) PQ-1 explicit fully-bare assertion at classify; (b) PQ-2 loader re-assertion keyed on scale-absence (D1 pattern); (c) PQ-4 enumerated log. A naked-fraction cap is *not* required (PQ-6).
4. **F5/D3 cb co-presence job?** Preserved — it lives at `:199`, before the D3 naked block, and PQ-3 requires it stay untouched.
5. **Negative tests** — see below.

## Required negative tests (each its own case)

1. cq-a set + naked base carrying `.weight_scale`, no descriptor ({0,1,0}) → reject at classify.
2. cq-w set + naked base carrying `.weight_scale`, no descriptor ({0,1,0}) → reject at classify (must NOT silently load as weight-only).
3. Naked base carrying `.input_scale` only, no descriptor ({0,0,1}) → reject at classify.
4. Naked base carrying BOTH scales, no descriptor ({0,1,1}) → reject at classify (the silent-corruption combo; must NOT load as scaled).
5. cq set + any `.scale_weight`/`.scale_input`/`scaled_fp8` present → still reject via the `:199` cb guard (regression for PQ-3).
6. **Positive:** cq-a descriptored set + fully-bare naked set → accept; assert naked layers land in `bf16_sd` as bf16, descriptored swap to `ScaledFp8Linear`, naked never appears in `fp8_entries`.
7. **Positive:** cq-w descriptored set + fully-bare naked set → accept (mirrors the target `krea2turbo` file: 256 cq-w + 8 naked).
8. Loader re-assertion: a state dict where a naked base carries a scale (simulated classify/loader divergence) → loader rejects (stray-scale or the new fully-bare re-check), never upcasts ignoring the scale.
9. Naked fp8 tensor not ending in `.weight` (e.g. `foo.bias` fp8) → loader rejects at `:542` (PQ-5 backstop).
10. Enumeration/log: assert the naked-set INFO line fires and names the count (PQ-4).
11. Boundary: all-fp8-naked-except-one-descriptor → loads (one scaled layer + rest bf16), confirming no implicit cap (PQ-6).

## Requirements delta (31-38, additive; 21-30 are DMR's)

31. `_classify_cq` replaces the D3 naked reject with a fully-bare assertion: every base in `fp8_bases - cq_bases` must carry none of `{.weight_scale, .input_scale, .scale_weight, .scale_input, .comfy_quant}`; any hit = loud reject naming base + suffix (PQ-1).
32. Loader treats a base as naked iff no C-a weight_scale AND no input_scale AND no descriptor are present in the state dict; naked → skip `fp8_entries`, fall to bf16 upcast; partial coverage → reject. Naked detection keyed on scale-absence, never descriptor-absence (PQ-2).
33. The `if cb_hits: raise` at classify (`:199`) is preserved byte-identical; the PQ change is confined to the D3 naked block (PQ-3).
34. Naked set enumerated and logged in one aggregate INFO line via `_safe_name`, never silent (PQ-4).
35. The naked-skip branch sits after the `.weight`-suffix check in the loader so non-`.weight` fp8 still rejects (PQ-5).
36. No naked-fraction cap; airtightness derives from req 31, documented in the module docstring's C-d section (PQ-6).
37. Module docstring updated: the D3 "EVERY fp8 weight carries a descriptor, or NONE" wording (`:228-229` comment) is amended to the per-layer rule — each fp8 base is fully descriptored+scaled OR fully bare, with partial coverage rejected.
38. Negative tests 1-11 above land in `test_fp8_single_file.py` in the same slice as the code.

## Verdict

**APPROVABLE WITH CHANGES.** The relaxation introduces no new attack primitive and the target layout is legitimate, but the "bounded to fully-bare only" invariant is not airtight in the current structure. Merge is gated on reqs 31-35 and 37-38 (PQ-1 through PQ-5, docstring, tests); req 36 (PQ-6) is documentation. The single most important item is **PQ-2**: if the implementer keys naked-detection on descriptor-absence rather than scale-absence, the change ships a silent ~448× weight-corruption path for the {0,1,1} combo.
