# Comfyless User Manual

**Comfyless** is the pure-Python / CLI path to the same diffusers pipeline code the ComfyUI nodes in this repo drive. No ComfyUI, no graph, no browser — just a command line (or a Python function call, or a JSON bridge for agents).

Use it when you want to:
- Script a parameter sweep (e.g. iterate over every transformer in a directory with the same prompt + seed)
- Replay a generation from a sidecar JSON with one or two overrides
- Drive image gen from an LLM agent via a JSON stdin/stdout contract
- Run unattended overnight batches

Every feature reachable from the ComfyUI loader/generate node chain is reachable here: CFG routing for every supported model family, component-level overrides (custom transformer, VAE, text encoders), LoRA stacking, sampler swap, VRAM optimizations.

---

## Contents

1. [Prerequisites](#prerequisites)
2. [Quick start](#quick-start)
3. [Invocation modes](#invocation-modes)
4. [CLI flag reference](#cli-flag-reference)
5. [Sidecar JSON and `--params` replay](#sidecar-json-and---params-replay)
6. [Example walkthrough (Qwen-Image)](#example-walkthrough-qwen-image)
7. [Model families and what they accept](#model-families-and-what-they-accept)
8. [Component overrides](#component-overrides)
9. [LoRA](#lora)
10. [VRAM knobs](#vram-knobs)
11. [Python function API](#python-function-api)
12. [JSON bridge mode](#json-bridge-mode)
13. [Server mode](#server-mode)
14. [Troubleshooting](#troubleshooting)

---

## Prerequisites

**Python interpreter** — use the ComfyUI venv so the diffusers/torch/accelerate versions match what the nodes use:

```
/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3
```

Set it up as a shell alias or just paste it in full. Everywhere below I'll write this as `$PY` for brevity.

**Working directory** — always invoke from the repo root (so `-m comfyless.generate` resolves):

```
/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments
```

**Model paths** — comfyless runs on the host, so use host paths, not the container paths you see in ComfyUI workflow JSONs. The mapping:

| In ComfyUI workflow | On host for comfyless |
|---|---|
| `/comfy/mnt/hf-local/<name>` | `/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/<name>` |
| (LoRAs in ComfyUI Loader dropdown) | `/home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models/loras/<name>.safetensors` |
| (HF hub snapshots) | `/home/gawkahn/projects/ai-lab/ai-base/models/hf-cache/hub/models--<org>--<name>/snapshots/<hash>` |

---

## Quick start

Generate a single image using the provided example sidecar:

```bash
PY=/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3
cd /home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments

$PY -m comfyless.generate \
    --params comfyless/examples/qwen_image_hello_world.json \
    --output /tmp/hello_qwen.png \
    --device cuda:1
```

That writes two files:
- `/tmp/hello_qwen.png` — the image
- `/tmp/hello_qwen.json` — a sidecar with every param used, plus `seed`, `timestamp`, `elapsed_seconds`

The sidecar is replayable: pass it back in via `--params` and you get the same generation. Change one thing via `--override` without editing the file.

---

## Invocation modes

All invocations are `python -m comfyless.generate …`. There are three ways to supply params:

### 1. Plain CLI flags

```bash
$PY -m comfyless.generate \
    --model /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Qwen-Image-2512 \
    --prompt "a golden retriever puppy on autumn leaves" \
    --seed 42 --steps 50 --cfg 4.0 \
    --output /tmp/dog.png
```

Use this for interactive one-offs.

### 2. Sidecar JSON + optional overrides (`--params` / `--override`)

```bash
$PY -m comfyless.generate \
    --params comfyless/examples/qwen_image_hello_world.json \
    --override seed=1337 \
    --override steps=25 \
    --output /tmp/dog_fast.png
```

Sidecar provides the base; explicit CLI flags (non-`None`) and `--override key=value` patches win over it. Use this for sweeps and for replaying saved generations.

### 3. JSON bridge (`--json`)

```bash
echo '{"model":"...","prompt":"...","contract_version":1,"params":{"seed":42}}' \
    | $PY -m comfyless.generate --json
```

Structured input on stdin, structured output on stdout, human-readable progress on stderr. For agents / tool-calling backbones. Schema: see [`../contracts/image_gen_bridge.md`](../contracts/image_gen_bridge.md) (if present) or the `_run_json_mode()` function in `generate.py`.

---

## CLI flag reference

All flags are optional except `--model` and `--prompt` (which can alternatively be supplied via `--params`).

### Core generation

| Flag | Default | What it does |
|---|---|---|
| `--model PATH` | — (required) | Base diffusers model directory (has `model_index.json`) |
| `--prompt STR` | — (required) | Text prompt |
| `--negative-prompt STR` | `""` | Negative prompt. **Used by:** qwen-image, sdxl, sd3, sd1, auraflow. **Silently ignored by:** flux, flux2, flux2klein, chroma (guidance-embedded models) |
| `--seed INT` | `-1` | Seed. `-1` = random, logged to stderr |
| `--steps INT` | `28` | Denoising steps |
| `--cfg FLOAT` | `3.5` | CFG / guidance scale. Routes to `guidance_scale` or `true_cfg_scale` depending on model family — see [Model families](#model-families-and-what-they-accept) |
| `--true-cfg FLOAT` | `None` | Explicit override for `true_cfg_scale` (qwen-image only). When omitted, `--cfg` is used |
| `--width INT` | `1024` | Width in px. Rounded down to nearest multiple of 32 |
| `--height INT` | `1024` | Height in px. Rounded down to nearest multiple of 32 |
| `--max-seq-len INT` | `512` | Text-encoder max sequence length. Relevant to flux/flux2/chroma/auraflow |
| `--output PATH`, `-o` | `output.png` | Output image path. Sidecar JSON is written alongside with the same stem. **Ignored when a server is running** (server owns the output path) — use `--savepath` instead |

### Sampler / schedule

| Flag | Default | What it does |
|---|---|---|
| `--sampler NAME` | `default` | One of: `default`, `multistep2`, `multistep3`. `default` = pipeline's native scheduler (usually FlowMatchEuler). Multistep variants are for flow-match models only; on SDXL/SD1 they auto-fall-back to `default` with a warning |
| `--schedule NAME` | `linear` | One of: `linear`, `balanced`, `karras`. **Currently reserved** — accepted and saved to sidecar, but not yet applied to the pipeline scheduler. Will become live when the manual-loop path lands |

### Component overrides (optional — leave empty to use base model's own components)

| Flag | What it loads |
|---|---|
| `--transformer PATH` | Custom transformer weights (or UNet for SDXL/SD1). Accepts a directory, a dir with a `transformer/` subfolder, or a single `.safetensors` file |
| `--vae PATH` | Custom VAE weights |
| `--te1 PATH` | Custom text encoder slot 1 (CLIP-L for Flux/Chroma; Qwen2.5-VL for Qwen-Image) |
| `--te2 PATH` | Custom text encoder slot 2 (T5-XXL for Flux/Chroma). No-op for Qwen |
| `--vae-from-transformer` | Extract the VAE from the `--transformer` checkpoint (for AIO single-file safetensors that bundle UNet + VAE). Ignored if `--vae` is also set |

### LoRA

| Flag | What it does |
|---|---|
| `--lora PATH[:WEIGHT]` | Apply a LoRA. Repeatable. `WEIGHT` defaults to `1.0`. Example: `--lora /path/a.safetensors:0.8 --lora /path/b.safetensors` |

### Device / VRAM

| Flag | Default | What it does |
|---|---|---|
| `--device STR` | `cuda` | `cuda`, `cuda:0`, `cuda:1`, `cpu`. Comfyless is single-GPU; multi-GPU `balanced` is a ComfyUI-only node feature |
| `--precision CHOICE` | `bf16` | `bf16`, `fp16`, `fp32`. `bf16` recommended for RTX 40/50 |
| `--offload-vae` | off | Move VAE to CPU during transformer inference, move back for decode. Saves ~1 GB |
| `--attention-slicing` | off | `enable_attention_slicing("auto")`. Lower peak VRAM, slower |
| `--sequential-offload` | off | `enable_sequential_cpu_offload()` via accelerate. Extreme VRAM savings, very slow. Incompatible with `--offload-vae` (sequential mode manages placement itself) |

### Output naming

| Flag | What it does |
|---|---|
| `--savepath TEMPLATE` | Output path template. Auto-creates parent dirs; appends a 4-digit counter (`0001`, `0002`, …) so files never overwrite. Wins over `--output` when set. In server mode the template is applied within `--output-dir`. See [savepath tokens](#savepath-tokens) |

**Savepath tokens** — all case-insensitive, `%name%` or `%name:spec%`:

| Token | Expands to |
|---|---|
| `%model%` | Transformer filename if `--transformer` is set, otherwise base model directory name. Mirrors ComfyUI "show what was actually used" convention |
| `%transformer%` | Always the transformer filename (or base model if none) |
| `%base_model%` | Always the base model directory name |
| `%date%` | `YYYY-MM-DD` |
| `%date:FORMAT%` | `strftime`-formatted date, e.g. `%date:MM-dd-YY%` → `04-22-26` |
| `%seed%` | Resolved seed integer |
| `%steps%` | Step count |
| `%cfg%` | CFG scale |
| `%sampler%` | Sampler name |
| `%model:N%` | First N characters of the model name (e.g. `%model:12%`) |

Example: `--savepath ~/gen/%date%/%model%_s%seed%` → `~/gen/2026-04-22/Qwen-Image-25120001.png`

### Replay mode

| Flag | What it does |
|---|---|
| `--params SIDECAR_JSON` | Load base params from a comfyless sidecar JSON. Any explicit CLI flag wins over the sidecar |
| `--override KEY=VALUE` | Patch a sidecar key. Repeatable. Value is coerced to `int`, `float`, `bool`, or `str` |

### Server mode flags

| Flag | What it does |
|---|---|
| `--serve` | Start the persistent model server. Blocks until `--unload` is received. Requires `--model-base` and `--output-dir` |
| `--unload` | Send a shutdown command to the running server |
| `--output-dir DIR` | `[--serve]` Directory where the server saves generated images. Created if absent |
| `--model-base DIR` | `[--serve]` Security boundary: all model and LoRA paths in incoming requests must resolve within this directory |

### Agent bridge

| Flag | What it does |
|---|---|
| `--json` | Agent bridge mode: JSON on stdin, JSON on stdout, progress on stderr |

---

## Sidecar JSON and `--params` replay

Every successful run writes a JSON sidecar alongside the output image. For `--output /tmp/dog.png`, the sidecar is `/tmp/dog.json`. Contents:

```json
{
  "prompt": "...",
  "negative_prompt": "...",
  "model": "/abs/path/to/model",
  "model_family": "qwen-image",
  "transformer_path": "",
  "vae_path": "",
  "text_encoder_path": "",
  "text_encoder_2_path": "",
  "vae_from_transformer": false,
  "loras": [{"path": "...", "weight": 1.0}],
  "seed": 42,
  "steps": 50,
  "cfg_scale": 4.0,
  "true_cfg_scale": 4.0,
  "width": 1024,
  "height": 1024,
  "sampler": "default",
  "schedule": "linear",
  "timestamp": "2026-04-21T12:34:56+00:00",
  "elapsed_seconds": 42.3,
  "contract_version": 1
}
```

### Replay

Same image, guaranteed (same seed + same model + same params + same torch/CUDA build = same pixels):

```bash
$PY -m comfyless.generate \
    --params /tmp/dog.json \
    --output /tmp/dog_again.png
```

### Replay with tweaks

Keep everything, bump CFG:

```bash
$PY -m comfyless.generate \
    --params /tmp/dog.json \
    --override cfg_scale=6.0 \
    --output /tmp/dog_cfg6.png
```

### Sweep example (bash)

Iterate over a directory of custom transformer checkpoints, same everything else:

```bash
for t in /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/qwen-finetunes/*/; do
    stem=$(basename "$t")
    $PY -m comfyless.generate \
        --params comfyless/examples/qwen_image_hello_world.json \
        --override transformer_path="$t" \
        --output "/tmp/sweep_${stem}.png" \
        --device cuda:1
done
```

Resolution of conflicting settings is:
1. `--params` loads the sidecar (if given)
2. `--override` patches apply
3. Any explicit `--flag VALUE` on the command line wins over both

Meaning: the sidecar is the "recipe," `--override` is inline edits, and explicit flags are per-invocation trumps.

---

## Example walkthrough (Qwen-Image)

### Step 1 — verify the model is present

```bash
ls /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Qwen-Image-2512/
# expected: model_index.json  scheduler  text_encoder  tokenizer  transformer  vae  README.md
```

### Step 2 — run the example sidecar

```bash
PY=/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3
cd /home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments

$PY -m comfyless.generate \
    --params comfyless/examples/qwen_image_hello_world.json \
    --output /tmp/hello_qwen.png \
    --device cuda:1
```

What you'll see on stderr (roughly):

```
[comfyless] Loading model: /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Qwen-Image-2512
[comfyless] Detected: QwenImagePipeline (family: qwen-image)
[comfyless] Ready — family=qwen-image, guidance_embeds=False
[comfyless] Generating: 1024x1024, steps=50, cfg=4.0, seed=42, sampler=default
[comfyless] Generated in ~40–80s (depends on GPU)
[comfyless] Saved: /tmp/hello_qwen.png
[comfyless] Metadata: /tmp/hello_qwen.json

Done. seed=42, time=XX.XXs
```

### Step 3 — inspect the sidecar

```bash
cat /tmp/hello_qwen.json
```

Notice `model_family: "qwen-image"`, `seed: 42`, and the exact params used. This is the replay spec.

### Step 4 — replay it

```bash
$PY -m comfyless.generate \
    --params /tmp/hello_qwen.json \
    --output /tmp/hello_qwen_replay.png \
    --device cuda:1
```

Compare the two PNGs — they should be byte-identical (same torch build + same GPU).

### Step 5 — one-variable tweak

Bump true_cfg_scale to 5.0, keep everything else:

```bash
$PY -m comfyless.generate \
    --params /tmp/hello_qwen.json \
    --override true_cfg_scale=5.0 \
    --output /tmp/hello_qwen_cfg5.png \
    --device cuda:1
```

The new sidecar at `/tmp/hello_qwen_cfg5.json` will show `true_cfg_scale: 5.0` and everything else unchanged.

### Step 6 — try a different seed

```bash
$PY -m comfyless.generate \
    --params /tmp/hello_qwen.json \
    --override seed=1337 \
    --output /tmp/hello_qwen_s1337.png \
    --device cuda:1
```

---

## Model families and what they accept

On load, comfyless reads `model_index.json` and detects a `model_family`. CFG routing branches accordingly:

| Family | Detected from | CFG param | Negative prompt | Notes |
|---|---|---|---|---|
| `qwen-image` | `QwenImagePipeline` | `true_cfg_scale` (double-pass CFG) | ✅ used | Official rec: 50 steps, true_cfg=4.0 |
| `flux` | `FluxPipeline` | `guidance_scale` (single-pass, embedded) | ignored | Guidance-distilled |
| `flux2` / `flux2klein` | `Flux2Pipeline` / variants | `guidance_scale` | ignored | Mistral3 text encoder, max_seq_len relevant |
| `chroma` | `ChromaPipeline` | `guidance_scale` | ignored | |
| `sdxl` | `StableDiffusionXLPipeline` | `guidance_scale` | ✅ used | UNet-based; flow-match samplers auto-fall-back to default Euler |
| `sd3` | `StableDiffusion3Pipeline` | `guidance_scale` | ✅ used | |
| `sd1` | `StableDiffusionPipeline` | `guidance_scale` | ✅ used | UNet-based |
| `auraflow` | `AuraFlowPipeline` | `guidance_scale` | ✅ used | `max_sequence_length` honored |
| (unknown) | any other diffusers class | introspection | depends | Passes any kwarg that matches the pipeline's `__call__` signature |

If you're loading a model not in the table, comfyless will still try — it introspects `pipeline.__call__` and only passes kwargs the pipeline accepts. Watch stderr for `[comfyless] Unknown model_family=…` to know this path fired.

---

## Component overrides

You can load a base pipeline and then swap in individual components. This mirrors the `EricDiffusionComponentLoader` ComfyUI node.

Example: use the SDXL base pipeline config/scheduler/tokenizer, but swap in a fine-tuned UNet from a single `.safetensors`:

```bash
$PY -m comfyless.generate \
    --model /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/stable-diffusion-xl-base-1.0 \
    --transformer /path/to/my_custom_unet.safetensors \
    --prompt "..." \
    --sampler default \
    --device cuda:1 \
    --output /tmp/custom.png
```

The `--transformer` path accepts:
- A diffusers model directory (has `transformer/` or `unet/` subfolder)
- A subfolder path directly (`/path/to/model/transformer/`)
- A single `.safetensors` file (AIO checkpoint — let the loader autodetect slot)

The `--vae-from-transformer` flag is specifically for AIO `.safetensors` checkpoints that bundle VAE weights alongside UNet weights (common with Illustrious / NoobAI / AllInOne-style SDXL merges). Without it, comfyless uses the base pipeline's VAE.

For abliterated / custom text encoders (e.g. abliterated Qwen2.5-VL for Qwen-Image):

```bash
$PY -m comfyless.generate \
    --model /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Qwen-Image-2512 \
    --te1 /path/to/abliterated_qwen2.5_vl \
    --prompt "..." \
    --output /tmp/abliterated.png
```

---

## LoRA

Stack LoRAs via `--lora PATH[:WEIGHT]`, repeatable:

```bash
$PY -m comfyless.generate \
    --model /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Qwen-Image-2512 \
    --prompt "..." \
    --lora /home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models/loras/style_a.safetensors:0.8 \
    --lora /home/gawkahn/projects/ai-lab/ai-base/models/comfyui/models/loras/subject_b.safetensors:1.0 \
    --output /tmp/stacked.png
```

Load failures (bad LoRA, keys don't match the architecture, etc.) are **non-fatal**. You get a warning on stderr and the warning is also recorded in the sidecar under `lora_warnings`. The run continues with whatever LoRAs loaded successfully.

In sidecar JSON the same info is:

```json
"loras": [
  {"path": "/abs/path/style_a.safetensors", "weight": 0.8},
  {"path": "/abs/path/subject_b.safetensors", "weight": 1.0}
]
```

---

## VRAM knobs

In rough order of impact:

1. **`--offload-vae`** — move VAE to CPU during transformer inference, back to GPU for decode. ~1 GB savings. Near-zero speed cost (one CPU↔GPU VAE copy per run).
2. **`--attention-slicing`** — slice attention ops. Lowers peak VRAM in transformer layers. Modest slowdown.
3. **`--sequential-offload`** — accelerate's full sequential CPU offload. Entire model is cycled CPU↔GPU per step. Massive VRAM savings, **very slow** (often >10× slower). Incompatible with `--offload-vae` (sequential mode manages all placement).

For Flux.2 on a single 24 GB card: start with `--offload-vae`, add `--attention-slicing` if still OOM, resort to `--sequential-offload` only if nothing else fits.

---

## Python function API

If you'd rather call `generate()` directly from a Python script:

```python
import sys
sys.path.insert(0, "/home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments")

import comfyless  # installs the ComfyUI-compatibility shims
from comfyless.generate import generate

metadata = generate(
    model_path="/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Qwen-Image-2512",
    prompt="A close-up photograph of a golden retriever puppy on autumn leaves",
    output_path="/tmp/puppy.png",
    seed=42,
    steps=50,
    cfg_scale=4.0,
    true_cfg_scale=4.0,
    width=1024,
    height=1024,
    device="cuda:1",
)

print(metadata["seed"], metadata["elapsed_seconds"])
```

The function signature matches the CLI flags one-to-one. See the `def generate(...)` block in `generate.py` for the full list.

**Caveat:** each `generate()` call loads the pipeline fresh — there is no cross-call cache (unlike the ComfyUI nodes, which keep the pipeline in VRAM across runs). If you're calling `generate()` in a loop, expect model-load overhead each time. For in-process sweeps, consider loading the pipeline yourself once and calling it directly (see `_build_call_kwargs()` for reference).

---

## JSON bridge mode

For agent / tool-calling integrations. Input is a JSON object on stdin, output is a JSON object on stdout, progress logs go to stderr.

**Request schema** (minimum):

```json
{
  "contract_version": 1,
  "model": "/abs/path/to/model",
  "prompt": "...",
  "output_dir": "/tmp",
  "output_stem": "agent_run_001",
  "params": {
    "seed": 42,
    "steps": 50,
    "cfg_scale": 4.0,
    "width": 1024,
    "height": 1024
  },
  "loras": []
}
```

**Response on success:**

```json
{
  "status": "ok",
  "output_paths": {
    "image":    "/tmp/agent_run_001.png",
    "metadata": "/tmp/agent_run_001.json"
  },
  "metadata": { /* full sidecar */ },
  "contract_version": 1
}
```

**Response on failure:**

```json
{
  "status": "error",
  "error": "...",
  "error_type": "ModelNotFound" | "InferenceError" | "InvalidParams" | "ContractVersionMismatch",
  "contract_version": 1
}
```

Non-zero exit code on failure, zero on success.

---

## Server mode

The server keeps a diffusers pipeline loaded in VRAM between invocations, eliminating the 30–90 second model-load overhead on every run. Once the server is up, any normal `comfyless.generate` invocation auto-detects the socket and delegates to it — no flag changes needed on the client side.

### Start the server

```bash
PY=/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/venv/bin/python3
cd /home/gawkahn/projects/ai-lab/code/Eric_Qwen_Edit_Experiments

$PY -m comfyless.generate --serve \
    --model-base /home/gawkahn/projects/ai-lab/ai-base/models \
    --output-dir /home/gawkahn/gen-output \
    --device cuda:1 \
    --precision bf16 &
```

The server prints its socket path and config to stderr, then blocks waiting for requests. Run it in a background shell or a tmux pane — it logs to stderr.

**Socket location:** `$XDG_RUNTIME_DIR/comfyless.sock` when systemd has set `XDG_RUNTIME_DIR` (typical on Linux desktop); otherwise `/tmp/comfyless-$UID/comfyless.sock`. The directory is created at mode `0700`; the socket at `0600` — inaccessible to other users.

### Using the server (auto-detect)

After the server starts, run generation normally:

```bash
$PY -m comfyless.generate \
    --model /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/Qwen-Image-2512 \
    --prompt "..." \
    --savepath /home/gawkahn/gen-output/%date%/%model%_s%seed% \
    --device cuda:1
```

Auto-detect fires when the socket exists **and** you are using `--savepath` or the default `--output` path. If you pass an explicit `--output` path, comfyless assumes you want to control the destination yourself and runs in-process instead.

The server handles:
- **Model caching** — first request loads the model; subsequent requests with the same config (`model`, `device`, `precision`, all component override paths) reuse the cached pipeline with no load time.
- **Config eviction** — if the model, precision, device, or any component override changes, the server evicts the old pipeline, frees GPU memory, and loads the new one.
- **Incremental LoRA diff** — the server tracks which LoRAs are applied. When a request adds or removes a LoRA, only the delta is applied (`delete_adapters` / `load_lora_weights`). A fresh load for every run is not needed.

### Output naming in server mode

The server owns the output path — `--output` on the client is ignored. Use `--savepath` to control naming within the server's `--output-dir`:

```bash
$PY -m comfyless.generate \
    --model /home/.../Qwen-Image-2512 \
    --prompt "..." \
    --savepath "%date:YYYY-MM-dd%/%model:12%_s%seed%"
```

The savepath template is relative (leading `/` stripped); the server resolves it within `--output-dir` and validates it doesn't escape that directory.

When no `--savepath` is given, the server auto-names files `comfyless0001.png`, `comfyless0002.png`, … in `--output-dir`.

### Stop the server

```bash
$PY -m comfyless.generate --unload
```

This sends a shutdown command over the socket. The server unloads the pipeline from VRAM, frees the socket file, and exits cleanly. Prints `No server found` if no socket is present.

### Security model

- All model and LoRA paths in requests are validated against `--model-base` before any load. Requests with paths outside that root are rejected.
- Output paths are validated against `--output-dir` after template expansion; templates that would escape the directory are rejected.
- LoRA adapter names are sanitized to `[a-zA-Z0-9_-]` before being passed to diffusers.
- Full design rationale: `docs/decisions/ADR-001-daemon-socket-security.md`.

---

## Troubleshooting

### `Model not found: /path`
`--model` must be a directory containing `model_index.json`. For single-file `.safetensors` checkpoints use `--transformer` with an appropriate `--model` base.

### `AttributeError: 'SomePipeline' object has no attribute 'transformer'`
Means a LoRA or component override was built for a transformer-based model but you loaded a UNet-based one (SDXL/SD1), or vice versa. Check the `model_family` in stderr output and the LoRA's target architecture.

### `AttributeError: 'FlowMultistep2Scheduler' object has no attribute 'init_noise_sigma'`
You set `--sampler multistep2` or `multistep3` on an SDXL/SD1 model. These samplers only work on flow-match models (Flux, Chroma, Qwen). Comfyless now auto-falls-back to `default` with a warning — if you see this error, it means you're on an older revision; update.

### LoRA applies but has no visible effect
Check stderr for `LoRA skipped (0 modules applied)` — means key mapping failed silently. Also check the sidecar `lora_warnings` field. Likely causes: Kohya format with keys we don't yet remap (e.g. `lora_te1_*` text-encoder LoRAs are currently dropped), or architecture mismatch.

### OOM on load
Try `--offload-vae` first, then `--attention-slicing`, then `--sequential-offload` as last resort.

### OOM during decode
Only VAE tiling handles this and it's always on. If still OOM at decode, you're genuinely out of VRAM at that resolution; drop dimensions or use `--sequential-offload`.

### `--json` mode hangs
It's waiting for stdin. Either pipe JSON in (`echo '{...}' | $PY -m ...`) or use heredoc:

```bash
$PY -m comfyless.generate --json <<'EOF'
{"contract_version": 1, "model": "...", "prompt": "...", "params": {"seed": 42}}
EOF
```

### Output file exists but image looks wrong / blotchy
A handful of model merges produce garbage output regardless of params (see the `AllInOne-XL-V12.5` case in the project Backlog). Verify with a known-good model first (the Qwen-Image example in this manual is a good baseline) before troubleshooting the merge itself.

---

*For ComfyUI node usage (same pipelines, graph UI), see [`../README.md`](../README.md). For developer notes on the Qwen pipeline internals, see [`../DEV_NOTES.md`](../DEV_NOTES.md).*
