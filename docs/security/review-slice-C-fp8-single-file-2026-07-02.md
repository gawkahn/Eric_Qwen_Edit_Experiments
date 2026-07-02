AI-Disclosure: security-auditor subagent (Claude Fable 5) authored; Grant reviewed. Design-phase review — no code existed at review time.

# Security review — ADR-019 Slice C (design, pre-code)

## Summary

Slice C flips the current hard-reject at `_diagnose_slot_mismatch` into a custom parser for community ComfyUI scaled-fp8 single-file checkpoints, then wires attacker-supplied `weight` bytes and F32 scale scalars into a new `ScaledFp8Linear` whose forward calls `torch._scaled_mm`. The caller-facing trust boundary (MCP catalog names, ADR-015/018 containment) does not move, but a **new file-content parsing surface** is added: header inspection, per-file variant classification, and — the risky step — a strip-scales → key-remap → re-attach dance that must preserve the exact scale↔layer binding under adversarial key names. Threat model is same-uid solo-desktop with files sourced from civitai/community. The attacker's ceiling is (a) silent numerical corruption of generation, (b) log injection via tensor names, and (c) at worst NaN/Inf hitting a `_scaled_mm` tensor-core path (low but nonzero device-side risk on some CUDA versions). RCE is not a realistic ceiling — `safetensors` is unpickled and mmap-bounded — but silent-wrong-answer paths are the natural failure mode the design must close.

Design posture is largely sound: invariants 1 (header-only detection), 2 (byte-identical non-fp8), 3 (loud on missing scale), 5 (hardware fallback), 6 (LoRA guard extension) all name the right rules. Two are underspecified — invariant 3 covers *missing* scales but not *colliding* scales after remap; invariant 7's claim of "no trust-class change" is technically wrong at the file-content review-bar level. The strip → remap → re-attach step in §3.2 is where the most consequential silent-corruption paths live and it has no listed collision-integrity check.

## Coverage

Reviewed:
- `docs/vision/slice-C-comfy-fp8-single-file.md` (whole file)
- `docs/decisions/ADR-019-native-quantization-support.md` (whole file)
- `nodes/eric_diffusion_utils.py` lines 1–1475 (`_diagnose_slot_mismatch` 306–481, `_load_single_weights` 484–1006, `load_component` 1053–1147, `_try_from_single_file` 1017–1050, slice-A quant section 1150–1475)
- `CLAUDE.md` Review bar section
- Prior security-review index under `docs/security/` (filenames only; no prior slice-C review exists)

Not reviewed:
- `nodes/eric_diffusion_loader.py` — cache-key wiring is slice-A/ADR-019 §5, out of slice-C design scope
- `comfyless/generate.py` `_apply_loras` — LoRA path integration is slice A's concern
- `torch._scaled_mm` documented behavior on non-finite / negative scales — cannot verify from repo; findings mark this as an assumption where relevant
- `safetensors` library version pin — not extracted

## Findings

**[HIGH] F1 — Strip → remap → re-attach has no collision-integrity check; wrong scale can bind to a layer**
Location: Vision §3.2 (design)
Risk: An attacker who controls source key names can craft (a) two source scale keys that canonicalize to the same post-remap name (`x.weight_scale` and `x.scale_weight` both landing on `x.weight_scale`), or (b) a source scale key that after canonicalization collides with an actual remapped weight name. In case (a) whichever is written second silently wins — the wrong scale binds to a real layer and generation is subtly wrong forever with no load-time error. In case (b) a scale scalar can silently overwrite a `[out,in]` weight tensor. Invariant 3 covers *missing* scales but says nothing about *colliding* scales.
Remediation: Build a `canonical → source_key` map during strip; two sources canonicalizing to the same target = loud reject naming both. Before re-attach, assert the canonical scale name is not already a key in the remapped dict.

**[HIGH] F2 — Scale values are not validated for finiteness or positivity before feeding `_scaled_mm`**
Location: Vision §3.3; consumer is `torch._scaled_mm(x_fp8, W_fp8.T, scale_a=input_scale, scale_b=weight_scale)`
Risk: A crafted file can set scales to 0.0, negative, NaN, subnormal, or ±Inf. Zero produces all-zero output; negative flips signs (silent garbage); NaN/Inf cascade through the denoising loop and — on some CUDA driver + tensor-core paths — have historically triggered device-side asserts requiring context reset. `_scaled_mm` performs no such validation.
Remediation: At load, immediately after reading each scale tensor, assert `torch.isfinite(s).all() and (s > 0).all()`; on failure raise a load error naming the layer and value(s). Same validation on the dequantize-to-bf16 hardware-fallback path.

**[HIGH] F3 — Scale-tensor shape/dtype is trusted from the header without enforcement**
Location: Vision §0 (declares scales "F32 scalar") and invariant 3; no enforcement point named in §3
Risk: A crafted file can declare a "scale" key as a per-channel `[out]` vector or a large `[N,M]` tensor. If the shape check is absent, per-channel semantics silently take effect (compute changes meaning) or a huge tensor loads into RAM before rejection.
Remediation: Before accepting a scale into the per-layer binding, assert `scale.numel() == 1` and `scale.dtype == torch.float32`. Reject any other shape/dtype loudly, naming the layer and observed shape.

**[HIGH] F4 — Invariant 7 is inaccurate; the CLAUDE.md Review bar needs a new §12 entry for the file-content parser**
Location: Vision §1 invariant 7; CLAUDE.md Review bar table
Risk: "No trust-class change" is correct at the *caller* boundary but wrong at the *file-content* boundary: today untrusted `.safetensors` bytes are consumed only by the safetensors library, diffusers converters, and a hard-reject. After this slice, in-house code inspects headers, matches key patterns, strips/remaps/re-attaches scales, and wires bytes into a compute op. That is a new §12 surface deserving its own Review bar line so future changes are `security-auditor`-gated by default.
Remediation: (a) Reword invariant 7 to distinguish caller trust (unchanged) from file-content parsing (new attack surface). (b) Add a Review bar row: `nodes/eric_diffusion_fp8_ops.py` + slice-C detection/remap in `eric_diffusion_utils.py`; trigger = "custom parsing of caller-supplied weight file content fed to compute ops".

**[MED] F5 — C-a/C-b marker-suffix mixing is not required to be mutually exclusive**
Risk: A file carrying both `weight_scale` (C-a) and `scale_weight` (C-b) suffixes is either an F1-style collision after canonicalization or an arbitrary classifier pick with wrong downstream semantics.
Remediation: If both suffix conventions appear in one file, reject loudly naming an example key from each.

**[MED] F6 — Missing-scale enforcement must run per-tensor, not per-file**
Risk: A file mostly fp8+scales but with ONE fp8 weight lacking its scale (or mixing C-a and C-c) risks misclassification as C-c and silent upcast with scale=1.0 for affected layers — the exact silent-corruption path invariant 3 closes.
Remediation: After classification, walk every fp8 weight and assert it has its paired scale (C-a/C-b) or that NO fp8 tensor has a scale key anywhere (C-c). Partial coverage → loud reject naming the first offender.

**[MED] F7 — Tensor names are used in log lines and errors without sanitization**
Risk: Tensor names are attacker-controlled UTF-8. Terminal escapes, NUL, backspace, CR can rewrite operator-visible logs; `k.split('.')` on adversarial names may produce empty segments passing canonicalization.
Remediation: Print/raise tensor names through `repr()` (or control-char-stripping, length-capped helper). Reject keys containing NUL or control chars (< 0x20) at strip time.

**[MED] F8 — `is_quantized_module` extension must be by isinstance walker, not parameter-type sniff**
Location: `eric_diffusion_utils.py:1413-1431` vs Vision invariant 6
Risk: The existing guard keys off `type(p.data).__module__.startswith("torchao")`. `ScaledFp8Linear` stores plain `torch.Tensor` F8_E4M3 weights — not a torchao subclass. Without an explicit module-type walker the guard misses it and permits tier-3 direct-merge writes into fp8 buffers.
Remediation: Extend `is_quantized_module` with `isinstance(m, ScaledFp8Linear)` over `module.modules()` before the parameter loop. Negative test: `guard_direct_merge` raises on a ScaledFp8Linear-containing module.

**[MED] F9 — safetensors `__metadata__` is untrusted; never use it for variant decisions, bound it in logs**
Remediation: Classification uses ONLY tensor names and dtypes. If metadata is logged, bound (first 256 bytes) + `repr()`.

**[LOW] F10 — Restrict the new path to `.safetensors`; reject fp8 markers on `.pt/.bin/.pth`**
Remediation: Refuse the scaled-fp8 loader unless the extension is `.safetensors`; message points at re-saving as safetensors.

**[LOW] F11 — nvfp4 reject must exit before any weight-byte load and before instantiation**
Remediation: nvfp4/unknown-layout detection fires from header inspection only, before tensor materialization or `ScaledFp8Linear` construction.

**[LOW] F12 — Hardware-fallback dequant path reuses the same scales — validate once at load, not at compute**
Remediation: Perform F2 validation once, at load, before either compute path is chosen.

## Requirements for implementation

1. Detection/header inspection via `safetensors.safe_open()` APIs only; no raw header parsing. fp8-marker keys on non-`.safetensors` = reject without loading (F10).
2. Classification from tensor names + dtypes only, never `__metadata__` (F9). Mixed C-a/C-b conventions = loud reject (F5). nvfp4/unknown layouts reject from header inspection before any construction (F11).
3. Scale shape/dtype asserted at load: `numel()==1`, `dtype==torch.float32`; anything else = loud reject naming layer + shape (F3).
4. Scale values validated at load: finite AND > 0; failure = loud load error naming layer + value. Once, covering both `_scaled_mm` and bf16-fallback paths (F2, F12).
5. Strip → remap → re-attach collision-safe (F1): canonical→source map with duplicate-target reject; pre-attach assert against remapped weight names; per-tensor paired-scale coverage check, partial coverage = reject (F6).
6. Tensor names sanitized (`repr()`/control-char strip) in all logs/errors; NUL/control-char keys rejected at strip (F7).
7. `is_quantized_module` extended with `isinstance(m, ScaledFp8Linear)` walker; negative test on `guard_direct_merge` (F8).
8. Vision invariant 7 reworded (caller trust vs file-content parsing); CLAUDE.md Review bar row added for the fp8 file-content parser (F4).
9. Unit-test negatives, each its own test: duplicate canonical scale targets; scale-name collides with weight name; mixed conventions; scale = 0.0 / +Inf / -Inf / NaN / -1.0 (separate cases); per-channel scale shape; non-F32 scale dtype; fp8 weight missing its scale in C-a file; tensor names with `\x00` / `\x1b[31m` / `\n`; non-safetensors extension with fp8 markers; guard raises on ScaledFp8Linear module.
10. Vision invariant 3 amended to "no missing AND no colliding scales" (F1/F6).
