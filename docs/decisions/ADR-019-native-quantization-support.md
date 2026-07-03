# ADR-019: Native quantization support — quantize-on-load (fp8) + community single-file quant consumption (GGUF, ComfyUI scaled-fp8)

**Date:** 2026-07-02
**Status:** proposed (implementation gated by `code-reviewer` + `security-auditor`, both Opus — the loader/component-loader path loads model weights from caller-supplied paths, a §12 security-review trigger already flagged in this repo's CLAUDE.md review bar).
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant reviewed.
**Relates to / supersedes:** Supersedes the offline-dequantization direction (`dequantize_nf4.py`, unbuilt `dequantize_comfy.py`; backlog task #52) — this ADR parks that as a dead end. Extends the generic loader (`nodes/eric_diffusion_loader.py`, `nodes/eric_diffusion_utils.py`) and the comfyless CLI/MCP surface (ADR-011, ADR-015, ADR-018).

---

## Context

We repeatedly punted on quantized models by proposing offline scripts that dequantize them back to bf16 (task #52). That direction is backwards: dequantizing to bf16 throws away the entire reason the quantized artifact exists — you pay full VRAM *and* run bf16 GEMM, forfeiting the hardware's low-precision tensor cores. On this workstation (2× RTX PRO 6000 Blackwell, sm_120, 96 GB each) that waste is acute: Blackwell has native FP8 **and** NVFP4 tensor-core paths.

Two distinct capabilities are wanted, and they are different engineering tasks:

1. **Quantize-on-load from bf16** — take the clean diffusers folders we already have and quantize at load time. We own the whole flow; cleanest integration; the path that yields real Blackwell fp8/nvfp4 compute.
2. **Consume community single-file quantized checkpoints** on the `--transformer-path` override. This is a hard requirement, not a nice-to-have: the whole point of comfyless is a ComfyUI-*independent* CLI/MCP generation path (see the API/LLM endgame). "It runs in ComfyUI" does not make a useful community finetune usable to *us*. Community single-file quant is predominantly two formats: **GGUF** (city96-style) and **ComfyUI scaled-fp8** (`.comfy_quant`/`.weight_scale` markers — the format `nodes/eric_diffusion_utils.py:337` currently hard-rejects).

### Verified viability spike (2026-07-02)

Before committing the plan, `torchao==0.17.0` was installed (`--no-deps`, torch 2.11/cu130 pin untouched) and tested on the real target:

| Test | Result |
|---|---|
| torchao 0.17.0 import vs torch 2.11.0+cu130 | ✅ clean |
| FP8 kernel on sm_120 (`Float8DynamicActivationFloat8WeightConfig`, tiny linear + forward) | ✅ `Float8Tensor`, bf16 out |
| **FP8 quantize-on-load, real 20B Qwen-Image transformer via diffusers `TorchAoConfig(quant_type=<AOBaseConfig obj>)`** | ✅ **20.4 GB vs ~40 GB bf16, block Linears are `Float8Tensor`, 53 s load** |
| NVFP4 default path (`NVFP4DynamicActivationNVFP4WeightConfig`) | ⚠️ blocked — quantizer routes through external `pytorch/MSLK` triton kernel, not installed |
| NVFP4 `use_triton_kernel=False` reference path | ✅ functional (`NVFP4Tensor`) but the reference quantizer is not a throughput win |

Key facts this establishes:
- diffusers 0.37.1's `TorchAoConfig` accepts a torchao **`AOBaseConfig` object** (not just quant-type strings), so nvfp4 configs are reachable. The "Unable to import torchao Tensor objects" warning it emits affects only loading *pre-serialized* torchao checkpoints, **not** quantize-on-load — our path is unaffected.
- **fp8 quantize-on-load is fully de-risked** on the real model.
- **fast nvfp4 for diffusion is officially supported but nightly-only.** Post-spike web verification (2026-07-02) corrected the initial read: nvfp4 quantize-on-load *works* — PyTorch's "Faster Diffusion on Blackwell" blog, diffusers `TorchAoConfig`, and the `sayakpaul/diffusers-blackwell-quants` recipes cover QwenImage specifically (1.39–1.49× over bf16 at batch 1–8, 62→52 GB). The fast `to_nvfp4` quantizer needs the `mslk` kernel (pytorch/ao PR #4031). **`mslk` is a pip wheel** (`--pre --index-url .../whl/nightly/cu130`), *not* a source build — but it is nightly-only and version-locked to nightly torch (`2.12.0.dev`) + torchao (`0.17.0.dev`). Our stack is torch `2.11.0` stable. So nvfp4 is deferred **for the same reason as Krea-2 (§Deferred): the working path exists only on non-release builds** — not because a kernel is missing or must be built. Building our own is unnecessary (nothing to build) and wrong (would just be nightly-equivalent code, not a stable pin).

## Decision

### 1. Backends, one per storage class

| Capability | Backend | Status |
|---|---|---|
| Quantize-on-load (bf16 → fp8) | torchao `Float8DynamicActivationFloat8WeightConfig` via diffusers `TorchAoConfig` | **v1** |
| Quantize-on-load (bf16 → nvfp4) | torchao `NVFP4DynamicActivationNVFP4WeightConfig` + `mslk` | **deferred** — works but nightly-only (see spike facts); ship when the stable torch/torchao/mslk triad releases |
| GGUF single-file consumption | diffusers-native `GGUFQuantizationConfig` + `from_single_file` | **v1 (slice B)** |
| ComfyUI scaled-fp8 single-file | custom key-remap + ported scaled-fp8 `Linear` ops (weights stay fp8, Blackwell fp8 GEMM) | **v1 (slice C)** — user chose scaled ops from the start over in-memory upcast |

### 2. Component eligibility policy — quantization is per-component-*type*, not per-file

"Anything that's a `.safetensors` is eligible" is the wrong model. Eligibility is "a large stack of `Linear` layers run every step." The policy the loader applies:

| Component | Quantize? | Reason |
|---|---|---|
| Denoiser (transformer / UNet) | **Yes — primary** | 20B of Linears; proven 40→20 GB, negligible loss. |
| Large-LM text encoders — Qwen2.5-VL (qwen-image/edit, flux2/krea VL path), Mistral (flux2 ~24B), T5-XXL (flux) | **Yes** | Biggest single VRAM item on some models; decoder-LM fp8 is the lowest-loss quantization there is. Skipping these leaves the largest win on the table. |
| CLIP-class encoders (CLIP-L/G) | **No, by default** | Tiny (≈zero VRAM win) *and* sensitive — low-dim embeddings, fp8 measurably shifts prompt adherence. Opt-in only. |
| VAE | **Never** | Small; quantization causes visible decode artifacts. Hard-excluded, not overridable. |
| LoRAs | **Never a quant target** | Rank-16–128 deltas, a few MB — pure loss, zero benefit. The LoRA issue is *base-compat*, not quantizing the LoRA — see §4. |

### 3. UX surface — global flag over a role-based eligible set, with per-component opt-out

- **`--quant <mode>`** (value flag, not boolean `--fp8`, so `--quant nvfp4` slots in later): quantize the *eligible* components — denoiser + large-LM text encoders — with VAE **hard-excluded** and CLIP-class **excluded by default**. "Eligible" is decided by component role/class (the loader already detects these), never by "it's a file."
- **`--quant-skip <component>` / `--quant-only <component>`**: per-component override, for isolating a regression. This control must exist even though the default is global — it is the direct answer to "debugging which component degraded would suck." A subtly-off image is bisected by re-running with the text encoder excluded.
- **Precedence:** catalog per-model default → `--quant` global → `--quant-skip`/`--quant-only`. A catalog entry may declare a recommended policy (e.g. Flux.2: quantize transformer + Mistral).
- **MCP / JSON form:** `quantization: {mode: "fp8", exclude: ["text_encoder"]}` — structured, no positional semantics.

**Positional `--fp8`-after-a-component-arg is rejected:** the endgame surface is MCP/JSON, which has no argument ordering. A CLI grammar that can't round-trip to JSON is the wrong grammar.

Mixed precision across components is numerically safe — modules hand off in bf16 activations, so a quantized transformer feeding a bf16 VAE has no dtype-boundary problem.

### 4. LoRA + quantized-base compatibility (the load-bearing resolution)

This is the ongoing hassle and the reason the section is explicit. With a torchao-quantized base (`Float8Tensor` weights), the repo's three-tier LoRA loader (fast `load_lora_weights` / manual PEFT injection / direct state-dict merge — see CLAUDE.md) splits cleanly:

- **Tier 1 & 2 (PEFT adapter, unfused) — supported.** PEFT keeps `lora_A`/`lora_B` as separate bf16 adapters: `y = base(x) + scale·B(A(x))`. The base Linear's `Float8Tensor` weight dequantizes inside its own matmul via torchao's dispatch; the adapter path is bf16 and independent of base quantization. **This is the path quantized runs use.**
- **`fuse_lora()` is disabled under quant.** Fusing merges the delta into the base weight — the same in-place mutation of a `Float8Tensor` that tier 3 does. Never call it when quantized.
- **Tier 3 (direct merge into params) — disabled under quant, fails loud.** You cannot `weight.data += (B@A)·scale` into a `Float8Tensor` in place. When a LoRA can *only* load via tier 3 (exotic LoKR/LoHa or prefix layouts PEFT can't ingest — the case tier 3 exists for) **and** `--quant` is set, we raise a clear, diagnosable error: *"LoRA <name> requires the direct-merge path, which is incompatible with `--quant <mode>`. Load it without `--quant`, or use a PEFT-loadable version."* Not a silent crash, not silent corruption.

**The honest tradeoff this documents:** `--quant` + a tier-3-only LoRA are not simultaneously supported in v1. The user picks: run that LoRA (base stays bf16) or quantize (that LoRA unavailable). The *real* fix — a **dequant→merge→requant** path that dequantizes the affected base layers to bf16, merges the LoRA, and re-quantizes — is deferred (§Deferred), to be built only if fail-loud proves too limiting in practice.

**Must-verify before shipping fp8 + LoRA together:** that diffusers `load_lora_weights` on an fp8-quantized transformer works via the unfused adapter path on our stack (torchao + PEFT LoRA is documented-supported in diffusers, but unproven here).

### 5. Cache key

`nodes/eric_diffusion_loader.py:131` `cache_key` gains the quant mode **and** the effective per-component policy (skip/only), or switching fp8↔bf16 (or changing which components are quantized) silently returns the wrong cached pipeline.

### 6. Detection-layer flip

`_diagnose_slot_mismatch` (`nodes/eric_diffusion_utils.py:317-375`) currently hard-rejects quantized single-file checkpoints and points at the dequantize scripts. Under this ADR:
- Recognized **GGUF** and **pre-quantized diffusers folders** route into the native quant loader instead of rejecting.
- **ComfyUI scaled-fp8** routes into the slice-C loader (key-remap + scaled-fp8 ops).
- Genuinely-unsupported inputs still reject, but the message points at `--quant` / the native path, **not** the (now-dead) dequantize scripts.

### 7. Slice plan & dependencies

- **Slice A** — quantize-on-load, fp8 only. Loader `--quant` input + eligibility policy + cache key + `--quant-skip`/`--quant-only`. Dep: `torchao==0.17.0` (exact pin; the "16 direct deps agree" rule means the pin lands in **both** `pyproject.toml` and `requirements.txt`, then `uv lock`). Verify fp8+LoRA PEFT path.
- **Slice B** — GGUF single-file consumption via diffusers. Dep: `gguf` (exact pin). Verify Qwen single-file→diffusers conversion maturity.
- **Slice C** — ComfyUI scaled-fp8 single-file: **C1** key-remap, **C2** ported scaled-fp8 `Linear` ops (weights stay fp8). Heaviest slice; the one that most needs `security-auditor` (custom loader on the caller-supplied model surface).
- **MCP/comfyless wiring** (after A) — `COMFYLESS_SCHEMA` + `generate.py` + `catalog.py` gain the `quantization` field and per-model default. Its own slice.

Each dep add is its own commit (§11); nvfp4 is *not* a dep in any v1 slice.

## Alternatives Rejected

- **Offline dequantize-to-bf16 scripts (task #52).** The original punt. Forfeits the entire point of the quantized artifact; on Blackwell it wastes the hardware. Superseded.
- **Pure-global `--fp8` (quantize everything).** Would sweep in VAE (visible artifacts) and CLIP (prompt drift) and give no way to isolate a regression. Rejected for the curated eligible-set + opt-out.
- **Pure per-component flags, no global.** Safe and debuggable but verbose, and forces "which components are eligible" onto the user every invocation. Rejected for global-default-with-override.
- **Positional `--fp8` after a component arg.** MCP/JSON has no argument ordering; can't round-trip. Rejected.
- **In-memory upcast for ComfyUI scaled-fp8 (dequant on load, bf16 compute).** Ships faster but discards the format's Blackwell fp8 speed. User chose scaled-fp8 ops from the start.
- **nvfp4 in v1.** The working diffusion nvfp4 path is nightly-only (torch 2.12.dev + torchao 0.17.dev + mslk); adopting it now would drag torch 2.11 stable → 2.12 nightly across the whole stack (every model + all tests), violating §11. Not a "build our own kernel" gap — `mslk` is already a pip wheel; the blocker is release cadence, which self-building cannot fix. Deferred like Krea-2.

## Deferred / Out of Scope

- **nvfp4 quantize-on-load.** Works today but only on the nightly triad (torch 2.12.dev + torchao 0.17.dev + mslk 2026.3.15, cu130); pinning it crosses the no-nightly-pin line (same as Krea-2). Trigger: stable torch ≥2.12/cu130 + matching stable torchao + stable mslk all released. Payoff when it lands: 1.39–1.49× on QwenImage. Tracked in TECH_DEBT.md §Dependencies.
- **nvfp4 / bnb-NF4 single-file consumption.** Near-zero community volume (nvfp4 files are mostly TensorRT engines, not diffusers-loadable); NF4 single-file is spotty in diffusers. Trigger: a concrete model we actually want that ships only in one of these.
- **dequant→merge→requant LoRA path** (tier-3-only LoRAs under quant). Build only if the §4 fail-loud proves too limiting.
- **Quantizing VAE / CLIP-class encoders beyond opt-in.** No plan to make these defaults.

## Changelog

- 2026-07-02 — Initial. Proposed. Records the 2026-07-02 torchao-0.17 viability spike (fp8 proven end-to-end on the 20B Qwen-Image transformer; nvfp4 deferred). Supersedes dequantize-script direction (task #52 → parked). Awaiting `security-auditor` before slice-A code.
- 2026-07-02 — Corrected the nvfp4 deferral rationale after web verification. Earlier text called `mslk` a "build-from-source, not-on-PyPI" gap — both wrong. nvfp4 diffusion is officially supported (PyTorch Blackwell blog + diffusers + `diffusers-blackwell-quants`, 1.39–1.49× on QwenImage); `mslk` is a pip wheel; the real blocker is that the working path is nightly-only (torch 2.12.dev triad), making it a Krea-2-style "wait for stable release," not a build-our-own. tech-debt entry added (TECH_DEBT.md §Dependencies).
- 2026-07-02 — Slice A shipped on main (b2bee68..1026d99: dep, utils, loader, comfyless/MCP surface, LoRA guard, 88-test suite, review F1/F3/F4 closed, F2 → TECH_DEBT) and merged onto krea-testing (3d5f82a, green against diffusers 0.39.0.dev0). Slice B (GGUF) deprioritized behind slice C per Grant: collection contains almost no GGUF files — a .safetensors variant has always existed, and VRAM abundance removes the motive; revisit for completeness after C.
- 2026-07-02 — Slice C shipped on main (d732b08 core, 1c2750f tests, a215b8e review fixes). Design-phase security-auditor review gated the code (docs/security/review-slice-C-fp8-single-file-2026-07-02.md, 12 findings → 10-point binding contract); code-reviewer verified per-requirement compliance (0 HIGH). The 'port ComfyUI ops' plan collapsed to one ScaledFp8Linear on torch._scaled_mm — per-tensor F32 scalar scales map directly onto the native fp8 GEMM. Klein-fp8 (the motivating KeyError file) now generates: 94/108 Linears fp8-resident (86.9% of params), 15.2s vs 17.4s bf16, 31.4 vs 35.3 GB peak, near-identical output. Three variants on disk: C-a weight_scale/input_scale (Klein, ltx-2-under-prefix), C-b scale_weight/scale_input (wan2.2), C-c plain-cast (civitai Flux — already loaded via the standard path, no new code). nvfp4 single-file rejects at header on the .weight_scale_2 signature.
- 2026-07-02 — Slice C-d shipped (b248f17 delta security review, 59e221e implementation, 4f361ee review fixes): comfy_quant descriptor support. The blanket comfy_quant reject became a bounded, allowlisted descriptor parse (delta reqs 11-20, all verified item-by-item at code review); new variants cq-a (descriptor + both scales, C-a semantics) and cq-w (weight_scale only — ScaledFp8Linear weight-only mode, dequant matmul, _scaled_mm never called). Unlocks Grant's three named problem children: pornmasterFlux2Klein_v2 (cq-a, 94/108 fp8-resident, generates 15.3s — THE file whose KeyError motivated slice C), absoluteRealismV01_qwenV10 (cq-w, 839/839, 10.0s ≈ bf16 speed), krea2TurboUncensored_v1 (cq-w; full generation awaits the krea-testing branch venv). fp4/nvfp4 descriptor files still reject. Also: audit_single_files.py tool committed (a3224e4) for collection-wide loading prognosis; bnb-NF4 revisit trigger + pure-torch dequant path recorded in TECH_DEBT (two projectGaia files, deferred).

