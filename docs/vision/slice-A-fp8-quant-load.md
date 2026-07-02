# Slice A — fp8 quantize-on-load — Vision & Autonomous-Build Handoff

**Date:** 2026-07-02 · **Author:** Claude (Opus 4.8) · **Executor:** intended for an autonomous run (Fable under eval)
**Implements:** [ADR-019](../decisions/ADR-019-native-quantization-support.md) Slice A. **Do not** implement Slice B (GGUF) or C (ComfyUI scaled-fp8) in this run.

---

## 0. Orientation (read first — you may be a different model than authored the design)

- The full design is in **ADR-019** and the `project_native_quant_support` memory. Read both before writing code.
- **Committed so far:** `5124dc9` (docs: ADR-019 + TECH_DEBT). HEAD is there, **unpushed**. Working tree also has a pre-existing unrelated `comfyless/catalog.py` modification and untracked dirs — **do not touch or stage those.**
- **Spike already proved** (2026-07-02): `torchao==0.17.0` imports against torch 2.11/cu130; fp8 quantize-on-load works end-to-end on the real 20B Qwen-Image transformer (40→20 GB, weights become `Float8Tensor`) via `TorchAoConfig(quant_type=Float8DynamicActivationFloat8WeightConfig())`. diffusers 0.37.1 `TorchAoConfig` accepts an `AOBaseConfig` **object**. `torchao` is currently installed in `.venv` via `--no-deps` but NOT in the lockfile — step 1 formalizes it.
- **GPU:** two Blackwell sm_120, 96 GB. **GPU 0 is usually busy; use `CUDA_VISIBLE_DEVICES=1`** for smoke tests (verify it's free with `nvidia-smi` first).
- **Test runner:** `./.venv/bin/python3` (uv-managed, per ADR-013). Nine unit suites, 1412 tests, expect 0 failures (see CLAUDE.md).

## 1. Vision

**Outcome when done:** a user can pass `--quant fp8` (CLI / MCP) or select fp8 on the loader node and the eligible components load quantized — cutting VRAM ~half — with no change to the default (unquantized) path, and LoRAs still apply.

**Invariants (must always hold):**
1. **Default path unchanged** — with no `--quant`, behavior and outputs are byte-identical to today. (Negative test: quant-absent load produces the same pipeline config as before.)
2. **VAE is never quantized** — not even with `--quant-only vae`. Hard exclusion. (Negative test: VAE weights stay bf16 under every flag combination.)
3. **CLIP-class encoders are excluded by default** — quantized only if explicitly named via `--quant-only`. (Negative test: `--quant fp8` leaves a CLIP text encoder in bf16.)
4. **Quant state is in the cache key** — switching quant mode or component set evicts and reloads; never returns a mismatched cached pipeline. (Negative test: load fp8 then bf16 → second load does NOT return the fp8 pipeline.)
5. **LoRA under quant uses the PEFT adapter path only** — `fuse_lora()` and tier-3 direct-merge are disabled; a tier-3-only LoRA fails with a clear, actionable error (not a crash, not silent corruption). (Negative test: forcing tier-3 under quant raises the documented error message.)
6. **Unsupported hardware warns and falls back, never hard-crashes** — an fp8 request on a non-fp8 GPU warns loudly and proceeds in bf16 (warn-don't-block).

**Out of scope:** GGUF (Slice B), ComfyUI scaled-fp8 (Slice C), nvfp4 (deferred — nightly-only), catalog per-model default quant (thin follow-up, optional). Do not add `gguf` or any other dep.

## 2. Change boundary / edit scope

May change: `pyproject.toml`, `requirements.txt`, `uv.lock`; `nodes/eric_diffusion_utils.py`; `nodes/eric_diffusion_loader.py`; the comfyless CLI generate path (`comfyless/generate.py`), the params schema (`COMFYLESS_SCHEMA` — the thing `test_params_schema.py` exercises), the MCP `generate` tool (`comfyless/mcp_server.py`); the shared LoRA apply path (`generate._apply_loras` / the tier dispatch); and the test files. **Locate schema/LoRA symbols by grep — do not assume paths.** If a change requires editing `resolve_hf_path` or `comfyless/server.py`, **STOP** — that re-triggers the mandatory security-auditor gate and is likely out of scope for A.

## 3. Design (from ADR-019 — condensed)

- **Eligibility is by component ROLE, not by file.** Build a small policy: denoiser (`transformer`/`unet`) = quantize; text encoder = quantize **iff** it's a large transformer LM (Qwen2.5-VL, Mistral, T5) — detect by class/param-count, NOT CLIP-class; VAE = never; CLIP-class = only if `--quant-only` names it. The loader already detects component classes — reuse that.
- **Surface:** `--quant <mode>` (value flag; `fp8` the only valid value this slice, but keep the enum open for `nvfp4` later), `--quant-skip <component>` (repeatable), `--quant-only <component>` (repeatable). MCP form: `quantization: {mode, exclude:[...], only:[...]}`. **No positional flags.** Node: a `quant` dropdown (`none`/`fp8`) + the shared util does eligibility.
- **Construct** `TorchAoConfig(quant_type=Float8DynamicActivationFloat8WeightConfig())` and pass via `quantization_config` into the existing `from_pretrained` `load_kwargs` in `eric_diffusion_loader.py:159`. For per-component control use diffusers `PipelineQuantizationConfig(..., components_to_quantize=[...])` if quantizing at the pipeline level, or apply per-component when loading components individually — pick whichever fits the existing load structure; document the choice.
- **Cache key:** extend `eric_diffusion_loader.py:131` `cache_key` with the mode + the resolved eligible-component set.
- **LoRA:** in the tier dispatch, when the base is quantized (detect `Float8Tensor` params), force tiers 1/2 unfused, skip `fuse_lora`, and make tier-3 raise the ADR-019 §4 error.

## 4. Build order (each a commit; conventional prefix; both disclosure trailers)

1. **`feat(deps): pin torchao==0.17.0`** — add to `pyproject.toml` + `requirements.txt` (17th direct dep — **honor the same-order rule** in CLAUDE.md), `uv lock`, commit all three together. Verify `./.venv/bin/python3 -c "import torchao"` still works.
2. **utils** — quant-config builder + role-based eligibility + cache-key extension in `eric_diffusion_utils.py`.
3. **loader** — `quant` input(s), thread `quantization_config` into `from_pretrained`, hardware gate + warn-fallback.
4. **comfyless** — `--quant`/`--quant-skip`/`--quant-only` on the CLI, `quantization` in `COMFYLESS_SCHEMA`, the MCP `generate` param.
5. **LoRA guard** — PEFT-path-under-quant, tier-3/fuse disabled + loud error.
6. **tests** — see §5.

Keep steps as separate commits where they're separable; bundle test+code for a symbol if the test would otherwise import a not-yet-committed module (CLAUDE.md staging note).

## 5. Proof — tests + real generation

**Unit (must stay green + add coverage):**
- All 9 suites, 0 failures. Extend `test_params_schema.py` (the `quantization` schema + adapters) and `test_mcp_server.py` (the MCP `generate` param). Add unit tests for: eligibility policy (each component role → quantized or not), cache-key discrimination (invariant 4), and the tier-3-under-quant loud failure (invariant 5). **At least one negative case per invariant in §1.**

**GPU smoke (the real gate — this is what "works" means):**
- On `CUDA_VISIBLE_DEVICES=1`, generate the SAME prompt+seed+steps twice: once bf16, once `--quant fp8`. Save both PNGs to the scratchpad. Read both with the Read tool and compare the checkable details. fp8 should be near-indistinguishable; note any degradation.
- Then repeat with a LoRA applied, to prove LoRA-on-fp8 works (invariant 5, positive case).
- **Use detailed, idiosyncratic prompts with verifiable specifics** — NOT generic scenes. Degradation must be *detectable*. Two required prompts:

  > **P1 (materials + count + legible text):** "A weathered brass astrolabe on an open leather-bound journal. The left page shows a hand-drawn crescent-moon diagram labeled '3rd house' in faded sepia ink; the right page has exactly three pressed maple leaves arranged in a fan. A half-empty glass of amber tea sits on a hexagonal cork coaster. Raking morning light from the left casts long shadows; a small brass key with a triangular bow lies beside the astrolabe." — check: is '3rd house' legible? exactly three leaves? hexagonal (not round) coaster? brass specularity intact?

  > **P2 (fine texture + color precision + spatial):** "A macro photograph of a hummingbird mid-hover to the right of a single foxglove stalk with seven open magenta blooms, each throat freckled with maroon spots. Iridescent teal-to-violet gorget on the bird's throat. A brass garden tag reading 'Digitalis' hangs from the stalk on red twine. Shallow depth of field, blurred greenhouse glass behind." — check: seven blooms? maroon throat-spots present? 'Digitalis' legible? gorget iridescence preserved?

## 6. Acceptance ("works" → hand back to Grant)

- [ ] All 9 suites green (report count) + new negative tests pass.
- [ ] fp8 generation succeeds on GPU 1; VRAM roughly halved vs bf16 (report both).
- [ ] P1 & P2 rendered bf16 and fp8, saved, and compared with the Read tool — degradation characterized, not just "looks fine."
- [ ] LoRA-on-fp8 generation succeeds.
- [ ] `code-reviewer` run on the final diff, findings addressed. `security-auditor` **only** if any change touched `resolve_hf_path`/`server.py`/path handling (it shouldn't).
- [ ] All slices committed with trailers; working tree clean except the pre-existing `catalog.py`/untracked items. **Do not push** — Grant verifies first.
- [ ] Hand back with: commit list, test counts, VRAM numbers, and the four comparison images' paths + your read of them.

## 7. Gates & disclosure

`code-reviewer` is mandatory (non-trivial slice). Reviewer model: current pins say Opus; Grant is evaluating Fable and may accept Fable-authored review here — follow whatever the session model/settings dictate, don't hard-fail on it. Every commit carries `AI-disclosure:` + `Co-Authored-By:` trailers per CLAUDE.md (use the tier that actually wrote the code). Commit THIS Vision doc as the spec-first opening move (`docs:`), before step 1.
