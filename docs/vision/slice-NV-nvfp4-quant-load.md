# Slice NV — nvfp4 quantize-on-load — Vision

**Date:** 2026-07-16 · **Author:** Claude (Fable 5) · **Approved:** Grant (session directive 2026-07-16, referencing `session-handoff-nvfp4-unblocked.md`)
**Implements:** [ADR-019](../decisions/ADR-019-native-quantization-support.md) — the deferred nvfp4 half of slice A. TECH_DEBT trigger ("stable torch/torchao/mslk triad") met 2026-07-16 on **easier terms than recorded**: the stable triad works on current pins (torch 2.11.0 + torchao 0.17.0 + mslk-cuda 1.1.0), no torch 2.12 bump. Empirically smoke-tested on Blackwell sm_120 (see the handoff doc for the verification record and the pin-choice analysis).

---

## 1. Vision

**Outcome when done:** `--quant nvfp4` (CLI / daemon / MCP / loader node dropdown) loads eligible components as torchao `NVFP4Tensor` weights via `NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True)` from `torchao.prototype.mx_formats`, with the same eligibility policy, cache-key discrimination, and warn-don't-block environment handling as fp8. **Live GPU quality smoke is explicitly deferred** (both GPUs busy on a long iterate run) — until that gate passes, nvfp4 is wired-but-unvalidated and fp8 remains the recommended quant mode.

**Invariants (each gets at least one negative test):**

1. **Default and fp8 paths unchanged** — with `--quant` absent or `fp8`, config objects, recipe selection (incl. the zimage weight-only gate), notices, and cache fragments are identical to today.
2. **nvfp4 requires Blackwell** — `quant_hardware_ok("nvfp4", dev)` requires CUDA compute capability ≥ 10.0; fp8 keeps its ≥ 8.9 gate. Below-threshold → loud warn + unquantized fallback, never a crash. (Negative: cap 8.9 device accepts fp8, rejects nvfp4.)
3. **Missing mslk is an environment problem, not a crash** — `use_triton_kernel=True` asserts mslk at quantize time *inside* `from_pretrained`; `build_quant_config` / `quantize_module` must probe `import mslk` first and warn-fallback to unquantized, so the load never dies mid-flight on the known `AssertionError: mslk is required for NVFP4 triton quantization`. (Negative: hidden-mslk import → None config + notice.)
4. **Weight-only family gate applies to nvfp4** — families in `_FP8_WEIGHT_ONLY_FAMILIES` (zimage) get `NVFP4WeightOnlyConfig()`; the outlier-sensitivity rationale (ADR-019 2026-07-10) transfers until live testing says otherwise.
5. **QUANT_MODES stays in sync** — `("none", "fp8", "nvfp4")` in BOTH `nodes/eric_diffusion_utils.py` and `comfyless/params_validation.py`; the existing sync test enforces equality, argparse/daemon/MCP/node dropdown all derive from these constants.
6. **No silent fp8 requant of an nvfp4 base (Red Zone finding, scoping 2026-07-16)** — `NVFP4Tensor` instances carry `act_quant_kwargs` (None when weight-only), so `_requant_config_matching_base` in `eric_diffusion_fp8_ops.py` would MISCLASSIFY an nvfp4 base as fp8 and silently swap representation on a LoRA direct merge. It must discriminate by tensor class and **refuse loudly** (ADR-019 §4 wording) for NVFP4Tensor bases — PEFT adapter path is the only supported LoRA route under nvfp4 this slice. (Negative: NVFP4Tensor param through `apply_merge_delta` → actionable RuntimeError, weights untouched.)
7. **Cache/sidecar/replay carry nvfp4 for free** — quant mode is already a string in the cache key and SCHEMA_KIND; "nvfp4" must round-trip like "fp8" (covered by existing structure + the QUANT_MODES sync).

**Out of scope:** live GPU quality smoke (nvfp4 vs fp8 vs bf16 — REQUIRED before recommending nvfp4; separate follow-up when a GPU frees); `torch.compile` (where the 1.39–1.49× throughput win lives; without it nvfp4 is mainly a VRAM play); nvfp4 requant-on-merge support (invariant 6 refuses instead); single-file nvfp4 parsing; enhance/hunyuan-reprompt nvfp4 (its `quant != "fp8"` hard error is correct — unsupported there); catalog per-model default quant.

## 2. Change boundary / edit scope (hard)

May change: `docs/vision/slice-NV-nvfp4-quant-load.md` (this doc), `docs/decisions/ADR-019-*.md` (Changelog append), `pyproject.toml` + `requirements.txt` + `uv.lock` (dep slice), `nodes/eric_diffusion_utils.py`, `comfyless/params_validation.py`, `nodes/eric_diffusion_fp8_ops.py` (**Red Zone — security-auditor gate**), `comfyless/mcp_server.py` (**Red Zone** — found during implementation: the tool schema hardcoded the quant enum at line ~776; now derived from `QUANT_MODES`), `test_quant.py`, `test_fp8_single_file.py` (if the merge-refusal negative fits there better), `TECH_DEBT.md` (Resolved append, entry at line ~348), `docs/security/review-slice-NV-*.md` (new). Anything else → STOP and split. `resolve_hf_path`, `comfyless/server.py`, `comfyless/generate.py` must NOT need edits (they consume QUANT_MODES).

**Amendment (2026-07-17, first live-gate finding — nvfp4 shape screen):**
The deferred live smoke's first Qwen-Image-2512 load crashed inside
`from_pretrained`: torchao's `_nvfp4_inference_linear_transform` HARD-RAISES
on any Linear weight whose last two dims aren't divisible by 16 (NVFP4 packs
16-element blocks along both dims), and the Qwen2.5-VL text encoder's vision
tower is hidden-size 3420 — all 96 `visual.blocks.*.mlp.{up,gate,down}_proj`
weights violate the constraint. fp8 never hit this (rowwise scales, no shape
requirement). Fix, within the existing edit scope (`eric_diffusion_utils.py`
+ `test_quant.py`): `_nvfp4_incompatible_weight_keys` scans each selected
component's safetensors HEADERS (official API, metadata only, ms per file)
and routes offenders through `modules_to_not_convert` — full weight keys
including `.weight`, the one form both quantizer gates match on current pins
(transformers `should_convert_module` endswith rule, surviving the
checkpoint→runtime `model.` prefix remap; diffusers exact-equality rule).
`quantize_module` (override path) applies the same screen as a `filter_fn`.
Warn-don't-block both ways: offenders stay bf16 with a loud notice; an
unscannable component (no readable safetensors header) warns that screening
was impossible and proceeds. Invariant 1 unaffected — fp8 is deliberately
unscreened (negative-tested).

**Scope amendment (2026-07-16, security-auditor req 65):** `nodes/eric_qwen_edit_lora.py` and `nodes/eric_lora_format_convert_apply.py` added — one call-site line after each of the four `merge_resolution_map` computations. The auditor found the DMR partial-merge debt trigger ("extending the DMR surface — new quantized reps") FIRED: a per-target refusal mid-adapter would leave earlier targets merged while `_apply_loras`' broad except reports "LoRA load failed" and generation continues on a half-merged, possibly daemon-cached transformer. Resolution is the auditor's option (a): an all-or-nothing entry gate (`refuse_unmergeable_base` in fp8_ops) scanning the resolution map BEFORE the first mutation, making invariant 6's "weights untouched" true per-adapter. Deliberately conservative: refuses direct merge into any model holding an unmergeable torchao rep, even if the adapter might have touched only plain targets.

## 3. Design (condensed)

- **Dep:** `mslk-cuda==1.1.0` (plain PyPI, no custom index, declares only numpy — see handoff §"Pin choice" for why 1.1.0 over 1.1.1+cu130: the load-bearing int32→int64 offset fix for >2³¹-element activation tensors is already in 1.1.0). 17→18 pins, same order in pyproject + requirements, `uv lock`, one commit.
- **Config routing:** generalize `_torchao_fp8_config(family)` → mode-aware selection (`_torchao_quant_config(quant_mode, family)` or equivalent minimal shape): fp8 branch byte-identical to today; nvfp4 branch imports from `torchao.prototype.mx_formats`, honors the weight-only family gate. `build_quant_config` + `quantize_module` route by mode; notice strings become mode-aware.
- **Hardware gate:** `quant_hardware_ok` takes the mode into account: fp8 ≥ (8, 9); nvfp4 ≥ (10, 0).
- **Merge guard:** `_requant_config_matching_base` gains a class check — only `Float8Tensor` proceeds to the akw sniff; NVFP4Tensor (and any other torchao class) raises the ADR-019 §4 actionable error.

## 4. Build order (each a commit; conventional prefix; both disclosure trailers)

1. `docs:` this Vision doc + ADR-019 Changelog append (spec-first).
2. `deps:` `mslk-cuda==1.1.0` — pyproject + requirements + `uv lock`; verify `./.venv/bin/python3 -c "import mslk"`.
3. `feat:` mode wiring (utils + params_validation) + merge guard (fp8_ops) + tests. **security-auditor (Fable) on the fp8_ops diff** → `docs/security/review-slice-NV-nvfp4-merge-guard-2026-07-16.md`, referenced in the commit body; code-reviewer (Fable) on the whole diff.
4. `docs:` TECH_DEBT Resolved append + memory update.

## 5. Proof

- `just tests` battery green (0 failures), CPU-only safe (CI has no GPU): config-selection tests construct configs and — where CPU allows (weight-only nvfp4 quantizes on CPU, verified in scoping) — real `quantize_` round-trips; hardware-gate tests monkeypatch `get_device_capability`; mslk-fallback test hides the module; merge-refusal negative uses a real CPU-quantized NVFP4 weight.
- Live GPU smoke: **deferred, tracked** — TECH_DEBT follow-up entry with trigger "a GPU frees up"; quality gate per handoff §4 (nvfp4 vs fp8 vs bf16 on QwenImage, detailed idiosyncratic prompts).

## 6. Acceptance

- [ ] Battery green; every §1 invariant has a passing negative test.
- [ ] security-auditor review saved + referenced (fp8_ops is Red Zone).
- [ ] code-reviewer run, findings addressed.
- [ ] TECH_DEBT nvfp4-blocked entry gets `Resolved:`; new deferred-smoke entry added.
- [ ] Commits with trailers; push held for batch approval per repo convention.

AI-Disclosure: Claude (Fable 5) authored; Grant approved the build direction 2026-07-16.
