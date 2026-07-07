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
- 2026-07-03 — §4 amendment: the deferred dequant->merge->requant path is ACTIVATED — its own trigger fired (Grant's Krea filter-bypass/.diff and snofs/LoKR adapters are direct-merge-only, and OOM relief requires --quant fp8; fail-loud made them mutually exclusive). Slice DMR: central apply_merge_delta dispatcher covering plain params (byte-identical legacy), torchao Float8Tensor params (dequant + delta + requant via the same _torchao_fp8_config recipe + Parameter swap), and ScaledFp8Linear buffers (per-tensor requant + cache invalidation); kind-tagged exact-restore backups. Fail-loud NARROWS to genuinely unmergeable reps rather than lapsing. Precondition: krea-testing LoRA commits cherry-picked to main (a59c990, 4899def) so the merge code is unified before restructuring. See docs/vision/slice-DMR-quantized-lora-merge.md.
- 2026-07-03 — Slice DMR shipped (a59c990/4899def cherry-picks, 3b20bd0 docs, 9d15a6a security delta, 812ac27 implementation + review-fix follow-up): direct-merge LoRAs now apply onto quantized bases via the apply_merge_delta dispatcher. code-reviewer verdict APPROVED, requirements 21-30 all satisfied; one LOW ledger-hygiene fix applied; partial-merge-on-adversarial-raise noted as TECH_DEBT. Live proof: Klein --quant fp8 + klein_snofs LoKR merged 144/144 diffs into fp8 weights and generated coherently at the full quant VRAM footprint — the exact combination §4's fail-loud previously forbade. Grant's Krea filter-bypass/snofs + fp8 case activates when main merges to krea-testing.
- 2026-07-03 — fix: Krea-2's model_index.json carries a non-component list at top level (text_encoder_select_layers, ints); resolve_quant_components crashed classifying it (TypeError on Grant's first krea-testing --quant fp8 run). Component entries are now required to be [library, class] string pairs; other lists are skipped as index metadata. df71b8a on main, 8a66d1b on krea-testing.
- 2026-07-03 — Slice DQ shipped: quant carried over the daemon protocol (closes slice-A reviewer F2 / TECH_DEBT "daemon silently drops quant"). Wire request sends quant/quant_skip/quant_only; _validate_request semantically rejects unknown modes via a light QUANT_MODES constant in params_validation (security review F1: no torch import on the unguarded accept-loop path — sync with eric_diffusion_utils pinned by test); _request_cache_key discriminates on the quant triple always and on the LoRA (path, weight) set when quant is active, so quantized pipelines evict+reload on ANY LoRA change instead of taking the incremental diff (delete_adapters cannot undo DMR direct merges); client delegation-skip removed. Bonus: hardening-review H-1 (_socket_dir symlink refusal) landed — its trigger (server-touching commit) fired. Design-phase security review: docs/security/review-slice-DQ-daemon-quant-2026-07-03.md (verdict: sound; F1 blocker + F2/F3 conditions all implemented). Vision: docs/vision/slice-DQ-daemon-quant.md.

- 2026-07-07 — ComfyUI-native Krea-2 single-file loading (community civitai checkpoints). Grant's target files (RedCraft, Dark Beast — both Krea-2 finetunes despite "int8 convrot" filenames; actually scaled-fp8) failed to load: released diffusers 0.39.0 ships `Krea2Transformer2DModel` but NO `from_single_file` for it, and the checkpoints use ComfyUI-native key names (`model.diffusion_model.blocks.N.attn.wq`, `qknorm`, `mod.lin`, `first`, `last`, `tmlp`, `txtfusion`…) vs diffusers' `transformer_blocks.N.attn.to_q` / `norm_q` / `scale_shift_table` / `img_in` / `final_layer` / `time_embed` / `text_fusion`. **New `nodes/eric_krea2_convert.py`** converts native→diffusers — every mapping a verified pure 1:1 RENAME (0 missing / 0 unexpected vs `Krea2Transformer2DModel.from_config`) plus ONE numel-safe reshape (native block `mod.lin` is a flat `(6·dim,)` scale_shift_table; diffusers stores `(6, dim)`). Wired into `load_scaled_fp8_component` (the cq-w path, since these are comfy_quant `float8_e4m3fn`): detect native-Krea via key signature → convert → `from_config` + `load_state_dict(assign=True)` → **loud missing-key assertion** (refuse rather than generate from random-init weights) → the existing fp8-residency re-swap. Verified end-to-end on `main`'s released-diffusers venv: RedCraft on Krea-2-Turbo generates a clean, coherent image, 256/256 Linears fp8-resident, weights magnitude-matched to base. **Chose the self-contained converter over an upstream-PR pin** (the earlier `krea-singlefile` branch pinned diffusers PR #14126, which converts only the Linears — 173/430 norm/embed params unmapped for the ComfyUI format → noise): the converter works on the released pin with no branch/dependency, and PR #14126 was reported/noted as incomplete for ComfyUI-export naming. **Future-proofing (Grant's maintenance concern):** the whole shim is one isolated module with Linear-vs-norm rule groups; the missing-key assertion makes rules provably dead the day a diffusers release covers a group (delete at leisure). The ComfyUI-name half may stay permanently bespoke (diffusers single-file converters target the reference layout, not ComfyUI's). **Reviews (both Opus):** `security-auditor` CLEAN (caller-supplied key-name parsing; no primitive beyond the "load a checkpoint you chose" baseline; fail-closed) — saved to `docs/security/review-krea2-comfy-convert-2026-07-07.md`. `code-reviewer` CHANGES REQUIRED → folded before commit: (a) `assign=True` had coerced Krea2's `_keep_in_fp32_modules` norms to bf16 — now restored to fp32 after load to match the base model's precision contract (re-verified: RedCraft still coherent); (b) the numel-safe reshape is allowlisted to `*scale_shift_table` targets (a coincidental-numel mis-rename now falls through to PyTorch's size-mismatch raise instead of being silently reshaped); (c) the load now raises on BOTH missing AND unexpected keys (locks the pure-1:1 contract); (d) added reshape + self-guard unit tests; (e) this security artifact saved (closing the reviewer's #9). The converter also self-guards against control-char key names (security INFO-1). **Out of scope (follow-on):** Dark Beast is an all-in-one bundle (Qwen3-VL TE + VAE + transformer) — needs component-splitting on top of this converter; the plain-fp8-cast (`cc`) + non-fp8 native-Krea paths still route through `from_single_file` and would need the same converter hook. krea-singlefile branch/worktree torn down (PR path abandoned).
- 2026-07-07 (follow-on — "transformer load in general"): the 2026-07-07 Krea converter was wired only into the scaled-fp8 path, so only comfy_quant native-Krea files loaded (RedCraft, Moody, TurboUncensored). Grant's other community Krea-2 checkpoints route through the GENERAL single-file path and still failed. Extended: the Krea build logic is factored into shared `build_krea2_transformer` (convert → `from_config` → scale_shift_table reshape → strict load → fp32-norm restore, raising `Krea2ConversionError`), reused by both the scaled-fp8 loader and a new native-Krea branch in `eric_diffusion_utils._load_single_weights` (after the scaled-fp8 classify, before `from_single_file`): peek keys → `is_krea2_comfy_checkpoint` → load + in-place fp8→bf16 upcast → build. **Bundle support:** `extract_krea2_transformer_sd` drops bundled TE/VAE (Dark Beast = Qwen3-VL TE + VAE + transformer), keeping only the transformer; **prefix-robust** because a bundle dilutes `model.diffusion_model.` below the loader's 50% dominant-prefix threshold (→ strip_prefix None), so extraction + `convert_krea2_comfy_state_dict` fall back to per-key `_strip_known_prefix`. **All 8 community Krea-2 safetensors now load + generate coherently** — scaled-fp8 (RedCraft/Moody/TurboUncensored), plain-fp8 prefixed (dafK2T) + bare-blocks (fascium/Muse), bf16 (unstableDissolution), and the TE+VAE bundle (Dark Beast). +5 unit tests (bundle detect/extract, prefix-robust convert). code-reviewer + security-auditor (Opus) delta on the general-path + bundle handling. **Perf note (accepted, TECH_DEBT-able):** a bundle loads the full file to RAM (~22GB) then drops TE/VAE — a selective transformer-only load would save it; deferred.
