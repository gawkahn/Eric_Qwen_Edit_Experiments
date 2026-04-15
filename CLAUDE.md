# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI custom node set wrapping two 20B-parameter Qwen models from Alibaba:
- **Qwen-Image-Edit-2511** — Image editing up to 17 MP
- **Qwen-Image-2512** — Text-to-image generation up to 50+ MP

This is a ComfyUI extension; there is no standalone executable, test suite, or build step. Development is done by editing node files and reloading ComfyUI.

## Development Workflow

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Reload nodes in ComfyUI:** Restart ComfyUI or use the ComfyUI Manager "Reload" button after editing node files.

**Lint (no CI configured — run manually):**
```bash
python -m py_compile nodes/<file>.py   # syntax check a single file
```

**Test suites (no CI, run manually):**
```bash
python3 test_manual_loop.py   # 186 tests: samplers, manual loop, encode helper, Qwen edit
python3 test_multistage.py    # 141 tests: multistage infrastructure
python3 test_samplers.py      # 41 tests: custom schedulers / sampler swap
```
All three suites use the ComfyUI venv's Python; run them via `/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3` when working outside of ComfyUI. Total 368 tests; expect 0 failures.

## Git commit conventions for this repo

See the general `Git Commit Discipline` rule in `~/.claude/CLAUDE.md` for the cadence and staging rules that apply to every Claude session. The additions below are the repo-specific conventions layered on top.

**Commit message style** — matches the existing history shown by `git log --oneline`:

- Prefix: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `update:`, `tool:` for standalone CLI tools, `workflows:` for workflow JSON artifacts
- Imperative mood, lowercase after the prefix
- Short first line (≤72 chars), optional body explaining the _why_ not the _what_
- When a change was AI-produced, append a Co-Authored-By trailer referencing Claude. Match the existing repo pattern — check `git log --format="%an|%cn" -20` before starting if unsure.

**Files that belong in commits:**

- Node files (`nodes/`), pipelines (`pipelines/`), tests (`test_*.py`), docs (`*.md`) — always committed
- Workflow JSON files (`workflows/*.png`, `*.json`) — commit in their own slice separate from code changes; they're artifacts, not logic, and should be readable in history as "workflows: …"
- Standalone CLI tools (`analyze_checkpoint.py`, `dequantize_nf4.py`) — commit individually with `tool:` prefix since they're self-contained utilities not imported by nodes
- `CLAUDE.md`, `.gitignore`, `requirements.txt`, `README.md` — committed with their respective content changes

**Files that must NEVER be committed:**

- `session-handoff-*.md` — ephemeral scratch notes from prior sessions; belong in `.gitignore`
- `memory/` directory — Claude's internal persistent memory system (user/feedback/project/reference notes); session-specific and should stay in `.gitignore`
- `api_keys.ini`, `.env` — already gitignored, never bypass
- `__pycache__/`, `*.pyc`, `*.pyo` — already gitignored

**Staging discipline for this repo specifically:**

- Before touching `eric_diffusion_*.py` or `eric_diffusion_advanced_*.py`, run `git diff <file>` on every file you intend to change so you see the full starting state
- The manual loop module `eric_diffusion_manual_loop.py` is large (>2600 lines) — when committing changes to it, write the commit message based on the _semantic_ change (which function/path you touched), not the line count
- The test files import from `nodes/eric_diffusion_manual_loop.py` via `importlib.util.spec_from_file_location` — they DON'T fail if the module file is missing at import time, but they fail silently with confusing errors. When committing a test-only change, verify the tested module is already in git, or bundle test+module changes in the same commit

## Architecture

### ComfyUI Registration
`nodes/__init__.py` imports all node classes and defines `NODE_CLASS_MAPPINGS` (internal key → class) and `NODE_DISPLAY_NAME_MAPPINGS` (internal key → UI label). Adding a new node requires: create file in `nodes/`, import class in `nodes/__init__.py`, add both mappings.

### Node Groups & Naming Conventions
- `eric_qwen_edit_*.py` — Edit pipeline nodes (loader, inpaint, LoRA, spectrum, etc.)
- `eric_qwen_image_*.py` — Generation pipeline nodes (loader, UltraGen, ControlNet, etc.)
- `eric_qwen_*.py` — Shared utilities (prompt rewriter, VAE loader)

Each node file defines a class with:
- `CATEGORY`, `FUNCTION`, `RETURN_TYPES`, `RETURN_NAMES` class attributes
- `INPUT_TYPES(cls)` classmethod returning required/optional input dicts
- A main method matching `FUNCTION` that does the work

### Pipeline Objects Passed Between Nodes
Nodes communicate via typed pipeline dicts, not raw model objects:
- `"QWEN_EDIT_PIPELINE"` — `{"pipeline": <QwenEditPipeline>, "model_path": str, ...}`
- `"QWEN_IMAGE_PIPELINE"` — `{"pipeline": <QwenImagePipeline>, "model_path": str, "offload_vae": bool}`
- `"QWEN_CONTROLNET_PIPELINE"` — wraps ControlNet model reference

### Custom Pipeline (`pipelines/`)
`pipeline_qwen_edit.py` extends the diffusers `QwenImageEditPlusPipeline` with:
- Native resolution preservation (aligned to 32px, capped at `DEFAULT_MAX_PIXELS = 16 MP`)
- True CFG support (`true_cfg_scale`) with norm-preserving rescaling
- Dual conditioning: VL path (Qwen2.5-VL tokens) + VAE/ref path (pixel latents)
- Spectrum acceleration hooks (Chebyshev feature forecasting, CVPR 2026)

`spectrum_utils.py` / `spectrum_forward.py` — patch the transformer's forward pass for 3–5× speedup without retraining.

### Model Caching
Loaders use module-level cache dicts (in `eric_qwen_edit_loader.py` and `eric_qwen_image_loader.py`). The cache stores a single pipeline + its config key; a different config triggers eviction and reload. Functions `get_gen_pipeline_cache()` / `clear_gen_pipeline_cache()` are imported by the component loaders.

### LoRA Loading (Three-Tier Fallback)
See `eric_qwen_edit_lora.py` / `eric_qwen_image_lora.py`:
1. **Fast path** — `pipeline.load_lora_weights()` (PEFT / diffusers native)
2. **PEFT injection** — inject adapter layers manually, then load
3. **Direct merge** — load state dict and merge weights into model parameters

Supports LoRA, LoKR, and LoHa formats with auto prefix detection.

### Guidance: Embedding vs. True CFG
**Critical distinction** (documented in `DEV_NOTES.md`):
- Guidance-distilled models (Flux.1-dev, SD3.5-Medium): `guidance_scale` is fed as a transformer input embedding — one forward pass per step, requires trained distillation.
- Qwen-Image-2512: `transformer.config.guidance_embeds = False` — guidance embedding is dead code. Use `true_cfg_scale` (standard CFG, 2× forward passes). Official recommendation: 50 steps, `true_cfg_scale = 4.0`.

### UltraGen Multi-Stage
`eric_qwen_image_ultragen.py` runs up to 3 progressive upscale stages. Each stage independently controls: steps, CFG scale, denoise strength, sigma schedule (`linear`/`balanced`/`karras`), seed mode, and LoRA weight. ControlNet variant in `eric_qwen_image_ultragen_cn.py`.

## Generic Multi-Model Nodes (`GEN_PIPELINE` type)

Three new nodes in `nodes/eric_diffusion_*.py` support any diffusers text-to-image model without model-specific code:

| Node | File |
|------|------|
| Eric Diffusion Load Model | `eric_diffusion_loader.py` |
| Eric Diffusion Unload | `eric_diffusion_loader.py` |
| Eric Diffusion Generate | `eric_diffusion_generate.py` |
| (shared helpers) | `eric_diffusion_utils.py` |

**Auto-detection:** loader reads `model_index.json → _class_name`, maps to a short `model_family` string (`"qwen-image"`, `"flux"`, `"flux2"`, etc.), and dynamically instantiates the pipeline class via `getattr(diffusers, class_name)`. New model families in diffusers work automatically.

**`GEN_PIPELINE` dict:**
```python
{
    "pipeline":        <pipeline obj>,
    "model_path":      str,
    "model_family":    "qwen-image" | "flux" | "flux2" | ...,
    "offload_vae":     bool,
    "guidance_embeds": bool,   # from transformer.config.guidance_embeds
}
```

**CFG routing in generate node:**
- `qwen-image` → `true_cfg_scale` (double-pass CFG), negative prompt used
- `flux` / `flux2` → `guidance_scale` (guidance embedding, single pass), negative prompt ignored
- unknown → `inspect.signature(pipe.__call__)` introspection, passes only accepted params

Existing `QWEN_IMAGE_PIPELINE` / `QWEN_EDIT_PIPELINE` nodes are untouched and use their own cache. The new nodes have a separate cache in `eric_diffusion_utils.py`.

**Edit nodes for Flux:** Flux.2 has no native edit variant. Future options: image-to-image via reference latent conditioning, or dedicated nodes using Flux.2-Klein-9B.

## Key Files for Common Tasks

| Task | File |
|------|------|
| Add a new Edit node | `nodes/eric_qwen_edit_<name>.py` + register in `nodes/__init__.py` |
| Add a new Generation node | `nodes/eric_qwen_image_<name>.py` + register in `nodes/__init__.py` |
| Modify pipeline inference logic | `pipelines/pipeline_qwen_edit.py` |
| Change Spectrum acceleration | `pipelines/spectrum_utils.py`, `pipelines/spectrum_forward.py` |
| Modify LoRA loading | `nodes/eric_qwen_edit_lora.py` or `nodes/eric_qwen_image_lora.py` |
| Prompt rewriting / LLM API | `nodes/eric_qwen_prompt_rewriter.py` (reads `api_keys.ini`) |
| 2× VAE upscale (Wan2.1) | `nodes/eric_qwen_upscale_vae.py` |

## Important Constraints

- All model loading uses `local_files_only=True` — no internet access during inference.
- Dimension alignment is 32px throughout; violating this causes transformer shape errors.
- `pipeline.vae.enable_tiling()` is always called on generation pipelines — required for >2 MP decode without OOM.
- The Edit pipeline takes a `Qwen2VLProcessor` (vision-language processor); the Generation pipeline does **not** — it uses only a tokenizer.
