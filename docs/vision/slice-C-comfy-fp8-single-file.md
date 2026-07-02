# Slice C — ComfyUI scaled-fp8 single-file consumption — Vision

**Date:** 2026-07-02 · **Author:** Claude (Fable 5) · **Status:** proposed — security-auditor review REQUIRED before code (this slice modifies the caller-supplied single-file weight-loading path, the §12 surface flagged in the repo review bar).
**Implements:** [ADR-019](../decisions/ADR-019-native-quantization-support.md) Slice C ("scaled-fp8 ops from the start" per Grant's decision). Slice B (GGUF) is deferred behind this (ADR-019 Changelog 2026-07-02).

---

## 0. Ground truth (2026-07-02 header survey of the actual collection)

Three fp8 single-file variants exist on disk; "ComfyUI scaled-fp8" is not one format:

| Variant | Example file | Layout |
|---|---|---|
| **C-a: scaled, `weight_scale` suffixes** | `Flux.2-Klein-9B-base/flux-2-klein-base-9b-fp8.safetensors` (the file whose KeyError motivated this); also `ltx-2-19b-distilled-fp8.safetensors` (same layout under a `model.diffusion_model.` prefix — *corrected 2026-07-02: initial survey misread its markers as comfy_quant*) | per-Linear `weight` F8_E4M3 `[out,in]` + `weight_scale` F32 scalar + `input_scale` F32 scalar; BFL key layout (`double_blocks.*`) |
| **C-b: scaled, `scale_weight` suffixes** | `wan2.2_t2v_*_14B_fp8_scaled.safetensors` | per-Linear `weight` F8_E4M3 + `scale_weight`/`scale_input` F32 scalars + global `scaled_fp8` marker tensor; biases f16 |
| **C-c: plain fp8 cast (no scales)** | most civitai Flux.1 fp8 checkpoints (`colossusProject…FP8UNET`, etc.) | fp8 tensors, no scale keys — storage compression only; already loads via the standard path + dtype upcast (no new code) |
| (observed, out of scope) | `ZImageTurbo-nvfp4_FP32.safetensors` | nvfp4 block layout: `weight` U8 packed + `weight_scale` F8_E4M3 `[out,blocks]` + `weight_scale_2` F32 scalar — header-time reject on the `.weight_scale_2` signature; ADR-019-deferred |

Semantics (matches ComfyUI fp8_ops): `W ≈ W_fp8 · weight_scale`; forward computes fp8 GEMM with per-tensor scales. **This maps directly onto `torch._scaled_mm(x_fp8, W_fp8.T, scale_a=input_scale, scale_b=weight_scale)`** — the native fp8 tensor-core op on sm89+. The "ops port" is therefore ONE custom Linear, not a kernel port.

## 1. Vision

**Outcome:** a community single-file fp8 checkpoint (any of C-a/C-b/C-c) passed via `--transformer-path` / `--transformer` (MCP catalog name) loads and generates — weights staying fp8 with true scaled-GEMM compute for C-a/C-b — instead of today's hard-reject at `_diagnose_slot_mismatch` (`eric_diffusion_utils.py:337`).

**Invariants:**
1. **Detection is read-only and header-only** — variant classification reads the safetensors header (as `_diagnose_slot_mismatch` does today); no weight bytes load before the format is decided.
2. **Non-fp8 paths are byte-identical** — a bf16 single file loads exactly as before; the new code is reached only when fp8 dtypes are present in the header. (Negative test: bf16 checkpoint → new code never invoked.)
3. **Scale integrity — no missing AND no colliding scales** *(amended per security review F1/F6)* — every fp8 weight that has a scale in the file keeps that exact scale through key-remap; a quantized layer whose scale went missing is a LOUD load error, never a silent scale=1.0; two source keys canonicalizing to the same target, or a scale name colliding with a remapped weight name, are LOUD load errors naming the offending keys; scale coverage is checked per-tensor (partial C-a/C-c mixtures reject). (Negative tests: missing scale; duplicate canonical targets; scale/weight name collision; partial coverage.)
4. **C-c (scale-less) upcasts to the requested dtype** — plain fp8 casts are storage compression; upcast to bf16 on load (documented; the file carries no scale so there is nothing to run scaled-GEMM with).
5. **Hardware gate mirrors slice A** — `_scaled_mm` needs sm89+; below that, warn loudly and dequantize to bf16 at load (warn-don't-block). (Negative test: gate function forced false → dequant fallback, no crash.)
6. **LoRA guard extends** — `is_quantized_module` must detect the new `ScaledFp8Linear` weights so `guard_direct_merge` fires for tier-3 merges onto scaled-fp8 bases exactly as for torchao bases. (Negative test: guard raises on a scaled-fp8 module.)
7. **Caller trust unchanged; file-content parsing is a NEW surface** *(reworded per security review F4)* — MCP/agent callers still supply catalog NAMES only; the new loader runs on operator-curated files under the scanned roots (ADR-015/018 containment unchanged); nothing adds a caller-controllable path input. HOWEVER, this slice adds in-house parsing of untrusted file CONTENT (header key patterns, scale tensors wired into compute ops) — a new §12 surface with its own Review bar row; future changes to the fp8 file-content parser are security-auditor-gated by default.
8. **`--quant` and scaled-fp8 files compose predictably** — a pre-quantized single file ignores `--quant` for that component with an INFO notice (it is already quantized); no double quantization.

**Out of scope:** GGUF (slice B, deferred); nvfp4 single-file (log-and-reject with actionable message); text-encoder/VAE single-file fp8 (transformer/denoiser slot only in v1 — matches the `--transformer-path` surface); ComfyUI checkpoints whose fp8 layers use per-CHANNEL scale vectors (none found in the survey; detect shape != scalar and reject loudly naming the layer).

## 2. Change boundary / edit scope

May change: `nodes/eric_diffusion_utils.py` (detection flip at `_diagnose_slot_mismatch`, new loader path in the single-file component machinery, `is_quantized_module` extension); a NEW module `nodes/eric_diffusion_fp8_ops.py` (`ScaledFp8Linear` + module-swap helper); `comfyless/generate.py` only if the override plumbing needs a hook (expect none — the component loader is shared); tests (`test_quant.py` growth or a `test_fp8_single_file.py`); docs.
**STOP boundaries unchanged:** `comfyless/server.py`, `resolve_hf_path`, `_run_json_mode`. The catalog/MCP surface needs NO schema change (files are already minted as `kind:"transformer"` by the ADR-018 scan).

## 3. Design sketch (for the security review)

1. **Detect** (header-only): classify {C-a, C-b, C-c, nvfp4-reject} from dtype + scale-suffix patterns. Extend, don't replace, the existing detection helpers.
2. **Load**: read state dict; for C-a/C-b, pair each fp8 weight with its scales under a canonical internal name (`weight_scale`/`input_scale`); strip scale keys BEFORE the existing key-remap/converters walk the dict (this is precisely the KeyError from the Klein file), remap the base keys with the existing machinery, re-attach scales under the remapped names.
3. **Instantiate**: build the transformer skeleton (existing config-detection path), then swap eligible `nn.Linear`s for `ScaledFp8Linear` (weight buffer F8_E4M3 + two F32 scalar buffers; forward = quantize x to fp8 with input_scale → `torch._scaled_mm` → bf16 out; fallback path dequantizes when the gate fails).
4. **Wire**: the loaded module flows through the existing `comp_kwargs` override path untouched; slice-A's `quantize_module` skips it (invariant 8 notice).

## 4. Proof

- Unit: header-classifier on synthetic headers for all four variants (+ per-channel-scale reject, missing-scale reject); scale-carry-through under key-remap; `ScaledFp8Linear` forward vs dequantized reference matmul (tolerance assert, CPU-fallback path); guard extension negative.
- GPU smoke: load `flux-2-klein-base-9b-fp8.safetensors` as `--transformer` over the Flux.2-Klein base model dir, generate P1/P2-style anchored prompts; compare against the bf16 `flux-2-klein-base-9b.safetensors` sibling (same directory — a natural A/B pair); VRAM + timing table. A wan2.2 file proves C-b detection only (video model — no image generation path here).
- All suites green.

## 5. Gates

**Changelog note (post-implementation review, 2026-07-02):** Req-9's two
name-canonicalization negatives ("duplicate canonical scale targets" and
"scale name collides with remapped weight name") are MOOT BY CONSTRUCTION —
the implementation binds scales to weights by source-key pairing + value
fingerprint and never re-keys a scale by name, so the crafted inputs those
tests describe cannot reach a re-attachment step. The adapted property IS
tested (an fp8 tensor under a scale-suffix name rejects on the F32 dtype
gate; dangling scales reject; double-bind of one model slot is prevented by
a claimed-set in the fingerprint swap). Reviewer accepted the deviation
("legitimate structural closure of F1"). Finding 8 (module-hook footgun in
`_apply`'s zero-probe) documented as accepted risk — `.to()`/`.cuda()`
/`.half()` all behave identically on zero-element probes.

`security-auditor` on THIS DESIGN before code — **DONE 2026-07-02**: `docs/security/review-slice-C-fp8-single-file-2026-07-02.md` (4 HIGH / 5 MED / 3 LOW, all with mitigations). Its "Requirements for implementation" list (10 points: safe_open-only inspection, names+dtypes-only classification, scalar-F32 shape/dtype asserts, finite-and-positive scale validation at load, collision-safe strip/remap/re-attach, sanitized tensor names, isinstance-walker guard extension, doc amendments, per-finding negative tests) is BINDING on the coding phase. Then `code-reviewer` on the diff before commit, per slice-A pattern.
