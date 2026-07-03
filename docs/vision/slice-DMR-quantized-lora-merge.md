# Slice DMR — dequant→merge→requant: direct-merge LoRAs on quantized bases — Vision

**Date:** 2026-07-03 · **Author:** Claude (Fable 5) · **Status:** proposed — security-auditor delta review REQUIRED before code (modifies `nodes/eric_diffusion_fp8_ops.py`, the Review-bar-gated surface, plus the four merge sites).
**Implements:** the deferred item ADR-019 §4 explicitly reserved: *"The real fix — a dequant→merge→requant path … is deferred, to be built only if fail-loud proves too limiting in practice."* **The trigger has fired:** Grant's Krea filter-bypass (`.diff`) and snofs (LoKR) adapters apply ONLY via direct merge, and his OOM relief REQUIRES `--quant fp8` — the §4 fail-loud makes the two mutually exclusive exactly where both are needed most.
**Precondition (done 2026-07-03):** krea-testing's LoRA commits cherry-picked to main (`a59c990`, `4899def`) so DMR builds once against the unified merge code (`resolve_merge_target` et al.) and merges cleanly back to krea-testing.

---

## 1. Vision

**Outcome:** a direct-merge-only adapter (LoKR/LoHa direct, converted `.diff`, standard-LoRA tier-3) applies onto a quantized base — both quantized representations — with quantization-step-bounded error, exact unload, and zero change to unquantized behavior. `--quant fp8` + filter-bypass LoRA on Krea becomes a working combination.

**The two quantized representations covered:**
| Rep | Origin | Weight storage | Merge strategy |
|---|---|---|---|
| torchao `Float8Tensor` param | slice A quantize-on-load | `Parameter` whose `.data` is a torchao subclass | dequant → +delta (fp32) → requantize via the SAME `_torchao_fp8_config()` recipe on a temp holder → **Parameter object swap** (never in-place `.data` mutation) |
| `ScaledFp8Linear` | slices C/C-d single-file loader | fp8 `weight` BUFFER + `weight_scale` (+ optional `input_scale`) | dequant → +delta (fp32) → per-tensor requant (`new_scale = amax/448`, clamp, cast e4m3) → buffer update + **`_fallback_weight` cache invalidation** |

**Invariants:**
1. **Unquantized path byte-identical** — for a plain bf16/fp16 param the dispatcher performs exactly today's `backup clone + param.data.add_(delta)`; no behavior change. (Negative test: bf16 merge output bit-equal to the pre-DMR implementation's.)
2. **Exact restore on unload** — quantized backups store the ORIGINAL representation verbatim (torchao: the original `Parameter` object; ScaledFp8Linear: `(w_fp8, weight_scale)` clones — fp8-sized, cheaper than today's bf16 clones) and restore by swap, never by re-derivation. Unload returns the model to bit-identical pre-merge weights. (Negative test: merge → unload → weights bit-equal originals, both reps.)
3. **Requant error bounded** — `dequant(requant(W+Δ))` is within one e4m3 quantization step of the exact merged `W+Δ` (relative tolerance test). Documented caveat: a delta that raises the tensor's amax coarsens the per-tensor scale for the whole tensor — same error class as the file's own quantization, logged when `new_scale/old_scale > 2`.
4. **Loud failure preserved for unmergeable reps** — a quantized representation the dispatcher doesn't handle (unknown torchao subclass, future nvfp4 module) still raises the actionable ADR-019 §4 error; DMR narrows the fail-loud, never silently widens acceptance. (Negative test: fake unknown-quant param → raise.)
5. **`input_scale` untouched** — activation scaling is calibrated to layer INPUT statistics, which a weight merge does not change. Weight-only (`input_scale=None`) modules merge identically.
6. **Cache coherence** — `ScaledFp8Linear._fallback_weight` is invalidated on merge AND on restore; a stale dequant cache serving pre-merge weights is silent corruption. (Negative test: merge → forward reflects delta on the dequant path.)
7. **Guard evolution is explicit** — the four merge sites stop calling `guard_direct_merge` at entry and instead route EVERY per-target write through the dispatcher (which owns the raise for case-4). The `test_quant.py` source-inspection test is updated in the same commit to assert dispatcher usage — the protection MOVES, it does not lapse.

**Out of scope:** fused-adapter (`fuse_lora`) support (still zero call sites); per-row scale preservation for ScaledFp8Linear (per-tensor requant only — matches the file format); merging INTO nvfp4/GGUF reps (formats not loadable yet); daemon-side quant (separate gated slice, TECH_DEBT).

## 2. Change boundary / edit scope

- `nodes/eric_diffusion_fp8_ops.py` — new `apply_merge_delta(root, target_key, delta, backup, log_prefix)` dispatcher + `restore_merge_backup(root, live_key, entry, log_prefix)`; requant helpers. (Gated file — auditor delta first.)
- `nodes/eric_qwen_edit_lora.py` — the three `_load_*_adapter_direct` merge loops route through the dispatcher; `model_sd` gains `named_buffers()` so `ScaledFp8Linear` weights resolve; `unload_adapters` restore branch dispatches kind-tagged backups.
- `nodes/eric_lora_format_convert_apply.py` — `_apply_converted_lora_as_delta` same treatment; `resolve_merge_target`/`resolve_restore_target` unchanged in signature (they consult the caller-supplied name→tensor mapping, which now includes buffers).
- Tests: `test_quant.py` (source-inspection update, invariant 4), new DMR cases in `test_fp8_single_file.py` or a focused suite.
- **STOP boundaries unchanged:** `comfyless/server.py`, `resolve_hf_path`, `_run_json_mode`.

## 3. Proof

- Unit: invariant negatives above; dispatcher case coverage (plain/torchao/ScaledFp8Linear/unknown); backup round-trips; scale-coarsening log.
- GPU smoke (main): Qwen-Image `--quant fp8` + a LoKR forced down the direct path — merge applies (nonzero modules), generates, unloads clean. Klein fp8 single-file (cq-a) + converted `.diff` adapter — generate before/after delta visibly differs.
- The REAL target smoke — Krea + `--quant fp8` + filter-bypass/snofs — runs on krea-testing after main→krea merge (needs `Krea2Pipeline`); Grant drives that verification.

## 4. Gates

`security-auditor` delta on THIS design before code (the dispatcher writes attacker-influenceable deltas into quantized storage and swaps Parameters — same trust class as existing direct merge, but the requant math and backup/restore lifecycle are new). Then `code-reviewer` on the diff. ADR-019 Changelog records the §4 amendment (fail-loud → DMR) in the same batch.
