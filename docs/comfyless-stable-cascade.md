# Stable Cascade in Comfyless

Stable Cascade is the only family in comfyless that doesn't follow the standard `--model <repo>` convention. It loads three independent weight files (Stage C, Stage B, Stage A) and runs them as a chained pipeline. To keep the standard CLI surface uncluttered, Cascade has its own dispatch path driven by JSON config files.

If you've used the rest of comfyless, the only flag that's different is `--model`. Everything else (`--prompt`, `--seed`, `--output`, `--batch`, `--limit`) works the same.

See **ADR-010** for the design rationale.

---

## Quick start

```bash
PY=/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3
cd /home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments

$PY -m comfyless.generate \
    --model stablecascade comfyless/examples/cascade_default.json \
    --prompt "an astronaut riding a horse, photorealistic, dusk light" \
    --seed 42 \
    --output /tmp/cascade_hello.png \
    --device cuda:0
```

`cascade_default.json` is shipped in `comfyless/examples/` and points at the SAI `stabilityai/stable-cascade` repo with full-variant defaults. Edit a copy to swap stages or tune params.

---

## Three stages

| Stage | What it does                                  | Native filename               |
|-------|-----------------------------------------------|-------------------------------|
| **C** | Text → 24×24 image-embedding latents (prior)  | `stage_c_bf16.safetensors`    |
| **B** | Embedding latents → intermediate features     | `stage_b_bf16.safetensors`    |
| **A** | Features → final RGB image (Paella VQ-VAE)    | `stage_a.safetensors`         |

Inference order is **C → B → A**. Stages C and B are large diffusion networks (3.6B and 1.5B params); Stage A is a 20M-param fixed-quality decoder. The C/B/A naming tracks abstraction depth (A is closest to pixels), not execution order — a Würstchen-original quirk.

---

## JSON config schema

```jsonc
{
  // ── Required ──────────────────────────────────────────────────────────
  "stage_c": "/abs/path/stage_c_bf16.safetensors",
  "stage_b": "/abs/path/stage_b_bf16.safetensors",

  // ── Optional, with defaults ──────────────────────────────────────────
  "stage_a":          "/abs/path/stable_cascade/vqgan",  // default: scaffolding_repo's vqgan/
  "scaffolding_repo": "stabilityai/stable-cascade",      // provides text_encoder/, tokenizer/, scheduler/

  "prior_dtype":   "bf16",   // safe default
  "decoder_dtype": "bf16",   // safe default; SAI model card suggests fp16, see note below
  "vae_dtype":     "bf16",   // safe default; matches decoder for uniform dtype throughout

  "prior_steps":       20,
  "prior_cfg_scale":   4.0,
  "decoder_steps":     10,
  "decoder_cfg_scale": 0.0,  // SAI: decoder doesn't benefit from CFG

  "width":  1024,
  "height": 1024              // both rounded DOWN to nearest multiple of 128
}
```

**Path semantics.** Each `stage_*` field accepts:
- An absolute path to a Würstchen-native single-file safetensors, OR
- An absolute path to a diffusers tree directory.

The loader branches on `os.path.isfile` vs `os.path.isdir`. The JSON field name is the only signal about which stage a file belongs to — there is no filename sniffing.

**`scaffolding_repo`** provides `text_encoder/`, `tokenizer/`, `scheduler/`. It can be a HuggingFace repo ID (resolved via `resolve_hf_path`) or a local directory. The default `stabilityai/stable-cascade` carries everything needed for text-to-image — the `-prior` companion repo is only required if you want image-variation, which v1 does not support.

**Resolution alignment.** Cascade compresses 128× in image space (`resolution_multiple = 42.67` × VAE's 3 downsample stages). Width and height are rounded down to multiples of 128 with a `[comfyless]` warning, never hard-failed. So `1023 → 896`, `1100 → 1024`, `1200 → 1152`.

---

## Iteration

Two ways to expand a single invocation into many runs.

**Topology iteration — pass multiple configs.** Each positional config is a complete cascade (paths, dtypes, denoising params). Configs run left-to-right; pipelines tear down and rebuild between configs.

```bash
$PY -m comfyless.generate \
    --model stablecascade full.json lite.json \
    --prompt "..." --seed 42 \
    --output /tmp/cascade-output
```

**Prompt / seed iteration — `--iterate prompt` or `--iterate seed`.** Cascade-supported subset of the standard `--iterate` axis system (ADR-008). Both accept a JSON list file. Cartesian-product when both are given.

```bash
$PY -m comfyless.generate \
    --model stablecascade full.json \
    --iterate prompt /path/to/prompts.json \
    --seed 42 \
    --output /tmp/cascade-sweep \
    --max-iterations 1500
```

Other `--iterate` axes (`cfg_scale`, `model`, `transformer`, …) are rejected — those are JSON-config concerns. Within a sweep, pipelines load once per config and are reused across all prompts × seeds × batch repetitions of that config before disposal.

**`--batch N`** runs N repetitions per (config, prompt, seed) tuple. Pair with `--seed -1` for fresh random seeds per repeat.

**`--limit M`** caps the total run count across all expanded combinations (silent truncation, ceiling not requirement). **`--max-iterations N`** is a hard fail-closed cap (default 500).

---

## Swapping alternative Stage B / Stage C weights

The handful of community-published Cascade alternatives (Reson4nce r35*, AltCascade) come as **ComfyUI all-in-one bundles**: a single `.safetensors` containing the diffusion stage *plus* embedded copies of the VAE and text encoder, with key prefixes `model.diffusion_model.*`, `vae.*`, `text_encoder.*`. These do not load via `StableCascadeUNet.from_single_file()` directly.

Run the conversion utility once per alt file:

```bash
$PY convert_cascade_comfyui.py \
    --in  ~/projects/ai-lab/ai-base/models/comfyui/models/checkpoints/StableCascade/r35on4nce_r35MCstageBf16.safetensors \
    --out ~/projects/ai-lab/ai-base/models/comfyui/models/checkpoints/StableCascade/r35on4nce_r35MCstageBf16-comfyless.safetensors \
    --stage c
```

The tool strips the `model.diffusion_model.` prefix, writes a Würstchen-native single-file safetensors next to the original (with `-comfyless` suffix per the parallel-library convention), and runs a strict-load smoke test against `StableCascadeUNet`. Any missing or unexpected keys are reported. Conversion is one-shot per file; thereafter the JSON config just points at the converted file.

This is the first concrete instance of the broader **parallel `-comfyless` library** convention (Backlog, Queued): pre-validated cleanly-loading siblings to avoid load-time retry chains.

---

## Recommended configs

For the user's hardware (24 GB consumer GPU) and the SAI guidance:

- **Full variants**: full Stage C (3.6B) + full Stage B (1.5B). The lite variants exist (`stage_c_lite_*`, `stage_b_lite_*`) and will load if the JSON points at them, but the SAI model card explicitly says they're inferior. By policy: don't use them. By code: nothing prevents it.
- **bf16 everywhere.** The SAI model card recommends bf16 prior + fp16 decoder for a slight speed/VRAM win, but the (deprecated upstream as of diffusers 0.35.2) Cascade pipelines mishandle the bf16→fp16 boundary when the two pipelines are composed manually. Defaulting both to bf16 sidesteps the integration bug. To opt into the model-card recipe explicitly, set `"decoder_dtype": "fp16"` in the JSON — `run_one` casts `image_embeddings` to the decoder's dtype at the boundary, so it works, just on a less-walked code path.
- **20 prior steps / 10 decoder steps / prior CFG 4.0 / decoder CFG 0.0.** SAI's documented sweet spot. Stable Cascade also has a fast path at 10 prior / 4 decoder steps if you want to trade quality for speed.

---

## Memory floor

Both stages live on GPU during their respective denoising loops, but only one at a time. Peak VRAM is dominated by Stage C (full variant in bf16, ~7.2 GB weights) plus the text encoder (CLIP-G, ~700 MB) plus working tensors. Expect ~12–16 GB peak at 1024×1024. The decoder loop is lighter. Stage A is negligible.

If you push above 1024×1024, Stage B's working memory grows quadratically with output area; OOMs surface there first.

---

## Lite-variant policy

Permitted. Discouraged. Not enforced. Pointing `stage_c` at `stage_c_lite_bf16.safetensors` will work fine; the SAI model card considers it lower quality, and on any hardware that can run the full variants there is no reason to use lite. We don't filter or warn at load.

---

## Where files live

| Artifact                                                                      | Path                                                                |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------|
| SAI main repo (decoder + scaffolding + flat stage_a/b/c safetensors)          | `~/projects/ai-lab/ai-base/models/hf-local/stable_cascade/`         |
| SAI prior repo (diffusers-tree prior, optional)                               | `~/projects/ai-lab/ai-base/models/hf-local/stable_cascade_prior/`   |
| ComfyUI alt files (require conversion before use)                             | `~/projects/ai-lab/ai-base/models/comfyui/models/checkpoints/StableCascade/` |
| Conversion tool                                                               | `convert_cascade_comfyui.py` (repo root)                            |
| Default config example                                                        | `comfyless/examples/cascade_default.json`                           |
| Cascade dispatch + loader                                                     | `comfyless/cascade.py`                                              |
| Design rationale                                                              | `docs/decisions/ADR-010-stable-cascade-json-config.md`              |
