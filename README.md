 # Eric Qwen-Edit & Qwen-Image Nodes

## 🖼️ Up to 17 MP image editing · 50 MP+ text-to-image generation

ComfyUI custom nodes for **Qwen-Image-Edit-2511** (image editing) and **Qwen-Image-2512** (text-to-image generation) — 20-billion-parameter MMDiT models by Qwen (Alibaba).  
32 nodes covering loading, image editing (single + multi-reference), style transfer, inpainting, inpaint-with-transfer, LoRA stacking, Spectrum acceleration, delta overlay, mask utilities, **text-to-image generation** (Qwen-Image, Flux, Chroma, SDXL, SD3.5, Pony, Illustrious — auto-detected via `model_index.json`), multi-stage generation, prompt rewriting, **2× VAE super-resolution upscaling**, **ControlNet-guided generation**, and **ControlNet inpainting** *(experimental)*.

![8 MP image editing in just a few nodes](examples/FireRed11-8mp.png)
*Edit images at up to 16 MP resolution — just a loader, LoRA, and edit node.*

![Advanced Qwen-Edit worfflow up to 16mp](workflows\Qwen-Edit-HiRes-Adv.png)
*Advanced Qwen-Edit workflow - just drop into ComfyUI - I use a lot of my own custom nodes which are all available but you can use the basic workflow too - just check the workflows folder*

## Features

- **Text-to-image generation** — Generate images from text prompts using Qwen-Image-2512
- **Preserves input resolution** — No forced upscaling to fill a pixel budget (edit nodes)
- **Configurable max_mp cap** — Control maximum output size for VRAM safety
- **Resolution presets** — Quick selection of common aspect ratios for generation
- **VAE tiling** — Automatic high-resolution decode without OOM
- **Supports up to 16 MP** — Edit or generate large images directly
- **True CFG** — Two full transformer forward passes per step (conditional + unconditional)
- **Dual conditioning paths** — VL path (~384 px semantic tokens via Qwen2.5-VL) + VAE/ref path (output-resolution pixel latents), individually controllable per image (edit nodes)
- **Multi-stage generation** — Progressive upscale + re-denoise across up to 3 stages with per-stage control over steps, CFG, denoise, and sigma schedule
- **UltraGen** — Quality-focused v2 multi-stage node with Qwen-Image-2512 best practices, per-stage seeds, sigma schedules, and upscale VAE integration
- **ControlNet-guided generation** — UltraGen CN node with [InstantX/Qwen-Image-ControlNet-Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union) for Canny, SoftEdge, Depth, and Pose guided generation up to 50 MP+, with auto-scaling CN strength
- **Spectrum acceleration** — Training-free CVPR 2026 Chebyshev feature forecaster for ~3–5× speedup (both edit and generation)
- **Prompt rewriting** — Local or remote LLM-powered prompt enhancement via any OpenAI-compatible API (Ollama, LM Studio, DeepSeek, etc.)
- **LoRA support** — Apply and unload LoRAs on both edit and generation pipelines with chainable weight control
- **2× VAE super-resolution** — Optional [Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x) integration for free 2× upscale during VAE decode, with inter-stage and final-decode modes
- **Extended prompt token length** — Configurable `max_sequence_length` (up to 1024 tokens) in UltraGen for highly detailed prompts — not exposed by other Qwen-Image nodes or workflows
- **Progress bars** — Native ComfyUI progress display during denoising on every generation/edit node

## What Makes This Different

Most ComfyUI Qwen nodes decompose the model into ComfyUI's generic UNET → Scheduler → Sampler graph. These nodes take a fundamentally different approach — running the **real Hugging Face diffusers pipeline** end-to-end, with targeted patches that unlock capabilities no other Qwen node set provides.

### 1. Native-resolution editing up to 17 MP

The stock diffusers `QwenImageEditPlusPipeline` forces all outputs to ~1 MP regardless of input size — a 12 MP photo gets crushed to 1 MP and fine details are lost. This node set patches the pipeline to **preserve your input resolution** (aligned to 32 px) up to a configurable cap (default 16 MP, supports 17 MP). No other ComfyUI Qwen-Edit implementation does this.

| Input | Stock Pipeline | Eric Qwen-Edit (max_mp=16) |
|-------|----------------|----------------------------|
| 2 MP  | 1 MP output    | 2 MP output                |
| 6 MP  | 1 MP output    | 6 MP output                |
| 20 MP | 1 MP output    | 16 MP output (capped)      |

### 2. Real FlowMatch diffusers pipeline — not the UNET/KSampler abstraction

ComfyUI's native approach treats every diffusion model as a generic UNET with a separate sampler and scheduler. Users must manually add an **"Aura Flow Shift"** node and guess shift values. This loses model-specific details and produces inferior results.

These nodes call the **`FlowMatchEulerDiscreteScheduler`** pipeline directly, so every sigma shift, timestep, and conditioning step matches exactly what the model was trained with:

| Aspect | ComfyUI native (UNET + KSampler) | Eric Qwen-Edit / Qwen-Image (diffusers) |
|--------|-----------------------------------|------------------------------------------|
| Sigma shifting | Manual — requires an extra "Aura Flow Shift" node with a user-chosen shift value | Automatic — `FlowMatchEulerDiscreteScheduler` with `use_dynamic_shifting` reads parameters from the model config |
| Resolution-aware | No — fixed shift regardless of output size | Yes — time-shift μ is interpolated from the output resolution's latent sequence length |
| Shift formula | `α·t / (1 + (α-1)·t)` with a single hand-tuned α | Exponential: `exp(μ) / (exp(μ) + (1/t - 1))` + terminal stretch, where μ adapts per resolution |
| Dual conditioning | Lost — UNET abstraction has no concept of separate VL + VAE/ref paths | Preserved — VL path (~384 px semantic tokens via Qwen2.5-VL) + VAE/ref path (output-resolution pixel latents), individually controllable per image |
| Configuration | User must wire shift nodes and pick values | Zero-config — parameters come from `scheduler_config.json` shipped with the model |

**You do not need any extra shift nodes with these nodes.**

### 3. Spectrum acceleration — training-free 3–5× speedup (CVPR 2026)

Implements adaptive spectral feature forecasting from a CVPR 2026 paper. Instead of running all transformer blocks on every denoising step, Spectrum predicts outputs on skipped steps using Chebyshev polynomial regression with Newton forward-difference blending. The flexible-window schedule caches more aggressively in later steps where changes are smaller. Applies to both edit and generation nodes. No other ComfyUI node set ships this.

### 4. Cross-architecture 2× upscale VAE — 50 MP+ generation

Exploits a discovery that the **Wan2.1** and **Qwen-Image** VAEs are architecturally identical (`AutoencoderKLWan` / `AutoencoderKLQwenImage`) and share the same latent space. This lets us use [spacepxl's Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x) decoder on Qwen-Image latents for free 2× super-resolution during VAE decode. Four modes:

| Mode | Effect |
|------|--------|
| `disabled` | Standard Qwen VAE decode |
| `inter_stage` | Decode S2 at 2× via upscale VAE, re-encode for S3 input |
| `final_decode` | Replace final VAE decode with 2× upscale VAE |
| `both` | Inter-stage + final decode — stacks for **4× total** (50+ MP output) |

### 5. Extended prompt token length (`max_sequence_length` up to 1024)

The `max_sequence_length` parameter is buried inside the diffusers pipeline and hardcoded to 512 everywhere else. UltraGen exposes it with range 128–1024. Padding positions are masked out via attention masks, so there's **zero quality penalty** for setting it higher — only a negligible compute increase (~8 MB VRAM). Long, detailed prompts (~200 words) that would be silently truncated at 512 tokens now reach the model in full.

### 6. Built-in LLM prompt rewriting

A dedicated node that calls any **OpenAI-compatible API** (Ollama, LM Studio, DeepSeek, OpenAI) to auto-expand terse prompts into rich ~200-word descriptions following Qwen's own recommended prompt methodology. API keys are loaded securely from environment variables or `api_keys.ini` — never stored in the workflow JSON. Includes language selection (English/Chinese), temperature control, custom instructions, and a passthrough toggle for A/B testing.

### 7. Multi-stage progressive generation with per-stage control

Up to 3 stages of progressive upscale → re-denoise, each with independent control over steps, CFG scale, denoise strength, sigma schedule (`linear` / `balanced` / `karras`), and seed mode (`same_all_stages` / `offset_per_stage` / `random_per_stage`). The UltraGen node combines all of this with tuned defaults that incorporate Qwen's official best practices — including the Chinese negative prompt that materially improves results.

### 8. True CFG with norm-preserving guidance

Two full transformer forward passes per step (conditional + unconditional) for genuine classifier-free guidance — not the approximations that single-pass "CFG-like" implementations use. UltraGen uses norm-preserving CFG rescaling that makes high CFG values (8–10) safe at low resolution for locking in composition, with lower CFG (2–4) at higher resolution stages for refinement.

### 9. Automatic VRAM management

Transformer is automatically offloaded to CPU before upscale VAE decode at every exit point. Tiled decode is used for large images. The pipeline manages device placement so you don't have to wire manual offload nodes.

### 10. Chainable LoRA with weight control

Apply multiple LoRAs in sequence with independent weight control (−2.0 to 2.0), and cleanly unload all LoRAs to restore the base model. Works on both edit and generation pipelines.

## Installation

### Option 1: ComfyUI Manager

Search for "Eric Qwen-Edit" in ComfyUI Manager.

### Option 2: Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EricRollei/Eric_Qwen_Edit_Experiments.git
```

## Requirements

- **Edit Model**: Download Qwen-Image-Edit-2511 (recommended) or 2509
  - https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- **Generation Model**: Download Qwen-Image-2512 (recommended) or Qwen-Image
  - https://huggingface.co/Qwen/Qwen-Image-2512
  - https://huggingface.co/Qwen/Qwen-Image
- **Upscale VAE** *(optional)*: spacepxl/Wan2.1-VAE-upscale2x (~0.5 GB)
  - https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x
  - Only needed for the 2× VAE super-resolution feature in UltraGen
- **ControlNet** *(optional)*: InstantX/Qwen-Image-ControlNet-Union (~2.3 GB)
  - https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union
  - Canny, SoftEdge, Depth, Pose guided generation
- **VRAM**:
  - 24 GB for up to 2 MP
  - 48 GB for up to 6 MP
  - 96 GB for up to 16 MP

---

## Nodes

### Eric Qwen-Edit Load Model

Loads the Qwen-Image-Edit pipeline from a local directory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | STRING | — | Path to the Qwen-Image-Edit model directory |
| `precision` | COMBO | `bf16` | Weight precision: bf16 (recommended), fp16, fp32 |
| `device` | COMBO | `cuda` | Device: cuda, cuda:0, cuda:1, cpu |
| `keep_in_vram` | BOOLEAN | `True` | Cache pipeline between runs to avoid reload |
| `offload_vae` | BOOLEAN | `False` | Move VAE to CPU when not in use (saves ~1 GB) |
| `attention_slicing` | BOOLEAN | `False` | Trade speed for lower peak VRAM |
| `sequential_offload` | BOOLEAN | `False` | Extreme VRAM savings via sequential CPU offload |

**Output:** `QWEN_EDIT_PIPELINE`

---

### Eric Qwen-Edit Component Loader

Advanced loader that lets you swap individual sub-models (transformer, VAE, or text encoder) from different directories. Useful for testing fine-tuned components without duplicating the full ~54 GB model.

> **Important — architecture constraints:** Every component must be architecture-compatible with Qwen-Image-Edit. The text encoder is **Qwen2.5-VL** (`Qwen2_5_VLForConditionalGeneration`), **not** CLIP. You cannot plug in a Stable Diffusion UNet, a standard CLIP model, or an unrelated VAE. You *can* use different fine-tuned or quantised versions of the same Qwen-Image-Edit components.

> **`base_pipeline_path` is always required**, even if you override all three components. The base path provides the scheduler config, tokenizer, and processor files that have no separate override.

#### What the base path must contain

The minimum viable `base_pipeline_path` folder needs these files (the small config/tokenizer files, not the large weights):

```
base_pipeline_path/
├── model_index.json                 ← pipeline class mapping (required)
├── scheduler/
│   └── scheduler_config.json        ← FlowMatchEulerDiscreteScheduler config
├── tokenizer/
│   ├── vocab.json
│   ├── merges.txt
│   ├── tokenizer_config.json
│   ├── added_tokens.json
│   ├── special_tokens_map.json
│   └── chat_template.jinja
└── processor/
    ├── tokenizer.json
    ├── preprocessor_config.json
    ├── video_preprocessor_config.json
    ├── vocab.json
    ├── merges.txt
    ├── tokenizer_config.json
    ├── added_tokens.json
    ├── special_tokens_map.json
    └── chat_template.jinja
```

If you don't override a component, its weights are also loaded from the base path.

#### Component folder structures

Each override path must contain a `config.json` plus the weight files for that component:

**Transformer** (~38 GB, `QwenImageTransformer2DModel` — 20B-parameter MMDiT):
```
transformer_path/
├── config.json
├── diffusion_pytorch_model.safetensors.index.json
├── diffusion_pytorch_model-00001-of-00005.safetensors
├── diffusion_pytorch_model-00002-of-00005.safetensors
├── diffusion_pytorch_model-00003-of-00005.safetensors
├── diffusion_pytorch_model-00004-of-00005.safetensors
└── diffusion_pytorch_model-00005-of-00005.safetensors
```
Also accepts: a parent folder with a `transformer/` subfolder, or a single `.safetensors` file (loaded as state dict into the base architecture).

**VAE** (~0.24 GB, `AutoencoderKLQwenImage`):
```
vae_path/
├── config.json
└── diffusion_pytorch_model.safetensors
```
Also accepts a parent folder with a `vae/` subfolder.

**Text Encoder** (~15.5 GB, `Qwen2_5_VLForConditionalGeneration` — Qwen2.5-VL 7B):
```
text_encoder_path/
├── config.json
├── generation_config.json
├── model.safetensors.index.json
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
└── model-00004-of-00004.safetensors
```
Also accepts a parent folder with a `text_encoder/` subfolder.

#### Typical use cases

| Scenario | What to set |
|----------|-------------|
| Fine-tuned transformer only | `base_pipeline_path` = full model, `transformer_path` = fine-tune dir |
| Quantised text encoder | `base_pipeline_path` = full model, `text_encoder_path` = quantised dir |
| Everything stock | Just use the standard **Load Model** node instead |

#### Node parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_pipeline_path` | STRING | — | Path to complete Qwen-Image-Edit model (always required — provides scheduler, tokenizer, processor, and defaults for unset components) |
| `transformer_path` | STRING | *(empty)* | Optional override — transformer weights directory or single `.safetensors` file |
| `vae_path` | STRING | *(empty)* | Optional override — VAE weights directory |
| `text_encoder_path` | STRING | *(empty)* | Optional override — text encoder weights directory |
| `precision` | COMBO | `bf16` | bf16, fp16, fp32 |
| `device` | COMBO | `cuda` | cuda, cuda:0, cuda:1, cpu |
| `keep_in_vram` | BOOLEAN | `True` | Cache between runs |
| `offload_vae` | BOOLEAN | `False` | Offload VAE to CPU when idle |
| `attention_slicing` | BOOLEAN | `False` | Attention slicing for lower VRAM |
| `sequential_offload` | BOOLEAN | `False` | Sequential CPU offload |

**Output:** `QWEN_EDIT_PIPELINE`

> **Note for ComfyUI users:** The standard ComfyUI "Load Diffusion Model" / "Load CLIP" / "Load VAE" nodes produce ComfyUI-internal model wrappers and **will not work** with these nodes. Qwen-Image-Edit requires the diffusers `from_pretrained` loading path, which is what both the Load Model and Component Loader nodes provide.

---

### Eric Qwen-Edit Unload

Free VRAM by unloading the pipeline. Connect after the last generation node.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | *(optional)* | Pipeline to unload |
| `images` | IMAGE | *(optional)* | Passthrough — connect to trigger unload after generation |

**Output:** `status` (STRING)

---

### Eric Qwen-Edit Image

Edit a single image using a text prompt.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | — | From any loader node |
| `image` | IMAGE | — | Image to edit |
| `prompt` | STRING | — | Describe the edit |
| `negative_prompt` | STRING | *(empty)* | What to avoid |
| `steps` | INT | `8` | Inference steps (8 for lightning LoRA, 50 for base model) |
| `true_cfg_scale` | FLOAT | `4.0` | True CFG strength (1.0–20.0) |
| `seed` | INT | `0` | Random seed |
| `max_mp` | FLOAT | `8.0` | Maximum output megapixels (0.5–16.0) |

**Output:** `IMAGE`

---

### Eric Qwen-Edit Inpaint

Inpaint masked regions of an image. The model has no native mask input — this node blanks the masked area, lets the model regenerate it, then composites the result back onto the original with feathered blending.

**Strategy:** blank masked region → model sees hole and prompt → post-composite with Gaussian-feathered mask.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | — | From loader |
| `image` | IMAGE | — | Image to inpaint |
| `mask` | MASK | — | White = inpaint, black = keep |
| `prompt` | STRING | — | Describe what to generate in masked area |
| `mask_mode` | COMBO | `blank_white` | How to blank the mask: blank_white, blank_gray, color_overlay |
| `feather` | INT | `8` | Gaussian blur radius for mask edge blending |
| `negative_prompt` | STRING | *(empty)* | What to avoid |
| `steps` | INT | `8` | Inference steps |
| `true_cfg_scale` | FLOAT | `4.0` | True CFG strength |
| `seed` | INT | `0` | Random seed |
| `max_mp` | FLOAT | `8.0` | Maximum output megapixels |

**Output:** `IMAGE`

---

### Eric Qwen-Edit Inpaint Transfer

Transfer content from a reference image into the masked region of the original. Combines pre-compositing, model harmonisation, and post-compositing for seamless results.

**Strategy:**
1. Scale the transfer image (+ optional transfer mask) proportionally so the source region fits inside the target mask bounding box
2. Pre-composite the transfer into the masked area — model sees content already in place
3. Model harmonises lighting, color, and edges via the prompt
4. Post-composite with feathered mask to preserve the original outside the mask

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | — | From loader |
| `image` | IMAGE | — | Original image (target) |
| `mask` | MASK | — | Target region (white = where to place transfer) |
| `transfer_image` | IMAGE | — | Reference image containing the content to transfer |
| `prompt` | STRING | — | Describe what you want (e.g. "harmonise the pasted element with its surroundings") |
| `transfer_mask` | MASK | *(optional)* | Mark which part of the transfer image to use (white = keep). When provided, both masks' bounding boxes are used for proportional scaling. |
| `transfer_vl_ref` | BOOLEAN | `True` | Also send full transfer image as a VL semantic reference |
| `blend_strength` | FLOAT | `1.0` | Pre-composite alpha (0.0–1.0) |
| `feather` | INT | `8` | Gaussian blur radius for post-composite blending |
| `negative_prompt` | STRING | *(empty)* | What to avoid |
| `steps` | INT | `8` | Inference steps |
| `true_cfg_scale` | FLOAT | `4.0` | True CFG strength |
| `seed` | INT | `0` | Random seed |
| `max_mp` | FLOAT | `8.0` | Maximum output megapixels |

**Output:** `IMAGE`

---

### Eric Qwen-Edit Multi-Image Fusion

Combine 2–4 images with composition modes and per-image conditioning control over both the VL (semantic) and VAE/ref (pixel) paths.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | — | From loader |
| `image_1` – `image_4` | IMAGE | — (2 required) | Input images (image_3, image_4 optional) |
| `prompt` | STRING | — | Describe the desired composition |
| `composition_mode` | COMBO | `group` | group / scene / merge / raw |
| `subject_label` | STRING | *(empty)* | Optional label for subject identification |
| `main_image` | COMBO | `image_1` | Which image seeds the output resolution and denoising |
| `vae_target_size` | INT | `0` | VAE encoding resolution for ref images (0 = match output) |
| `vl_1` – `vl_4` | BOOLEAN | `True` | Include each image in the VL semantic path |
| `ref_1` | BOOLEAN | `True` | Include image_1 in the VAE/ref pixel path |
| `ref_2` – `ref_4` | BOOLEAN | `False` | Include secondary images in VAE/ref path (default off — VL-only) |
| `negative_prompt` | STRING | *(empty)* | What to avoid |
| `steps` | INT | `8` | Inference steps |
| `true_cfg_scale` | FLOAT | `4.0` | True CFG strength |
| `seed` | INT | `0` | Random seed |
| `max_mp` | FLOAT | `8.0` | Maximum output megapixels |

**Output:** `IMAGE`

---

### Eric Qwen-Edit Style Transfer

Apply the visual style of one image to the content of another, with fine-grained control over which aspects of style are transferred.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | — | From loader |
| `style_image` | IMAGE | — | Reference providing the aesthetic |
| `content_image` | IMAGE | — | Image to restyle |
| `style_mode` | COMBO | `full_style` | full_style / color_palette / lighting / artistic_medium / texture / custom |
| `custom_prompt` | STRING | *(empty)* | When non-empty, always overrides the style_mode template |
| `additional_guidance` | STRING | *(empty)* | Extra instructions appended to the auto-generated prompt |
| `style_strength` | FLOAT | `1.0` | Scales CFG for stronger/weaker style (0.1–3.0) |
| `vae_target_size` | INT | `1024` | VAE encoding resolution for style image |
| `vl_style` | BOOLEAN | `True` | Style image in VL semantic path |
| `vl_content` | BOOLEAN | `True` | Content image in VL semantic path |
| `ref_style` | BOOLEAN | `False` | Style image in VAE/ref pixel path (off by default — avoids pixel bleed) |
| `ref_content` | BOOLEAN | `True` | Content image in VAE/ref pixel path |
| `negative_prompt` | STRING | *(empty)* | What to avoid |
| `steps` | INT | `8` | Inference steps |
| `true_cfg_scale` | FLOAT | `4.0` | True CFG strength |
| `seed` | INT | `0` | Random seed |
| `max_mp` | FLOAT | `8.0` | Maximum output megapixels |

**Output:** `IMAGE`

---

### Eric Qwen-Edit Spectrum Accelerator

Training-free diffusion acceleration based on the **Spectrum** method (CVPR 2026). Uses Chebyshev polynomial feature forecasting to skip redundant transformer forward passes, achieving ~3–5× speedup with minimal quality loss.

Attach this node between the loader and any generation node. The config is stored on the pipeline and takes effect during the next denoising run. Automatically disabled when total steps < `min_steps`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | — | From loader |
| `enable` | BOOLEAN | `True` | Toggle acceleration on/off |
| `warmup_steps` | INT | `3` | Full-compute warm-up steps before forecasting begins |
| `window_size` | INT | `2` | History window for Chebyshev polynomial fitting |
| `flex_window` | FLOAT | `0.75` | Fraction of remaining steps to recompute vs. forecast (0.0–1.0) |
| `w` | FLOAT | `0.5` | Blend weight between forecast and previous features |
| `lam` | FLOAT | `0.1` | Regularisation coefficient for the forecaster |
| `M` | INT | `4` | Chebyshev polynomial degree |
| `min_steps` | INT | `15` | Spectrum auto-disables below this step count |

**Output:** `QWEN_EDIT_PIPELINE` (same pipeline with spectrum config attached)

---

### Eric Qwen-Edit LoRA

Load a LoRA adapter into the pipeline. Use the lightning LoRA for 8-step inference.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | — | From loader |
| `lora_name` | COMBO | — | Dropdown of `.safetensors` files in `ComfyUI/models/loras/` |
| `weight` | FLOAT | `1.0` | LoRA scale (0.0–2.0) |
| `lora_path_override` | STRING | *(empty, optional)* | Full path to a LoRA file outside the standard loras folder |

**Output:** `QWEN_EDIT_PIPELINE`

---

### Eric Qwen-Edit Unload LoRA

Remove all LoRA adapters from the pipeline, restoring base weights.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_EDIT_PIPELINE | — | Pipeline with LoRA loaded |

**Output:** `QWEN_EDIT_PIPELINE`

---

### Eric Qwen-Edit Delta Overlay

Compare an edited image with the original, extract a change mask, and composite the edit onto the original only where changes occurred. Useful for upscaling an edit at full resolution and applying it precisely.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `original_image` | IMAGE | — | Original (before edit) |
| `edited_image` | IMAGE | — | Edited (after edit) — may be a different resolution |
| `threshold` | FLOAT | `0.05` | Minimum per-pixel difference to count as a change (0.0–1.0) |
| `blur_radius` | INT | `5` | Gaussian blur on the change mask for softer edges |
| `expand_mask` | INT | `3` | Dilate the mask by this many pixels |
| `upscale_method` | COMBO | `lanczos` | Resampling method when resizing: lanczos, bicubic, bilinear, nearest |
| `input_mask` | MASK | *(optional)* | If provided, intersected with the auto-detected change mask |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| `composite` | IMAGE | Original with edit applied only where changes were detected |
| `change_mask` | MASK | Binary mask of detected changes |
| `upscaled_edit` | IMAGE | Edited image resized to match original resolution |

---

### Eric Qwen-Edit Apply Mask

Simple mask-based compositing utility. Blends a foreground and background image using a mask.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `foreground` | IMAGE | — | Image shown in white areas of the mask |
| `background` | IMAGE | — | Image shown in black areas of the mask |
| `mask` | MASK | — | Blend mask: white = foreground, black = background |
| `blur_mask` | INT | `0` | *(optional)* Additional Gaussian blur on the mask (0–50) |

**Output:** `IMAGE`

---

## Qwen-Image Generation Nodes

These nodes use **Qwen-Image / Qwen-Image-2512** for text-to-image generation. They share the same 20B MMDiT transformer and VAE architecture as the edit model, but take only text input — no source image required.

> Generation nodes use a **separate pipeline type** (`QWEN_IMAGE_PIPELINE`) that is not interchangeable with the edit pipeline (`QWEN_EDIT_PIPELINE`). You need separate loader nodes for each.

### Eric Qwen-Image Load Model

Loads the Qwen-Image-2512 (or Qwen-Image) text-to-image pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | STRING | — | Path to the Qwen-Image model directory |
| `precision` | COMBO | `bf16` | Weight precision: bf16 (recommended), fp16, fp32 |
| `device` | COMBO | `cuda` | Device: cuda, cuda:0, cuda:1, cpu |
| `keep_in_vram` | BOOLEAN | `True` | Cache pipeline between runs |
| `offload_vae` | BOOLEAN | `False` | Move VAE to CPU when not in use |
| `attention_slicing` | BOOLEAN | `False` | Trade speed for lower peak VRAM |
| `sequential_offload` | BOOLEAN | `False` | Sequential CPU offload for extreme VRAM savings |

**Output:** `QWEN_IMAGE_PIPELINE`

---

### Eric Qwen-Image Component Loader

Advanced loader that lets you swap individual sub-models (transformer, VAE, or text encoder) from different directories.

> The generation pipeline has **no processor** component (unlike the edit pipeline). The base path must provide `model_index.json`, `scheduler/`, and `tokenizer/`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_pipeline_path` | STRING | — | Path to complete Qwen-Image model (always required) |
| `transformer_path` | STRING | *(empty)* | Optional override — transformer weights directory or `.safetensors` |
| `vae_path` | STRING | *(empty)* | Optional override — VAE weights directory |
| `text_encoder_path` | STRING | *(empty)* | Optional override — text encoder weights directory |
| `precision` | COMBO | `bf16` | bf16, fp16, fp32 |
| `device` | COMBO | `cuda` | cuda, cuda:0, cuda:1, cpu |
| `keep_in_vram` | BOOLEAN | `True` | Cache between runs |
| `offload_vae` | BOOLEAN | `False` | Offload VAE to CPU when idle |
| `attention_slicing` | BOOLEAN | `False` | Attention slicing for lower VRAM |
| `sequential_offload` | BOOLEAN | `False` | Sequential CPU offload |

**Output:** `QWEN_IMAGE_PIPELINE`

---

### Eric Qwen-Image Generate

Generate images from text prompts. Choose a resolution preset or set custom dimensions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_IMAGE_PIPELINE | — | From any generation loader node |
| `prompt` | STRING | — | Describe the image to generate |
| `negative_prompt` | STRING | *(empty)* | What to avoid |
| `resolution` | COMBO | `1024×1024 (1:1)` | Resolution preset (9 common aspect ratios, or "custom") |
| `width` | INT | `1024` | Custom width — only used when resolution = "custom" |
| `height` | INT | `1024` | Custom height — only used when resolution = "custom" |
| `steps` | INT | `50` | Inference steps |
| `true_cfg_scale` | FLOAT | `4.0` | True CFG strength (>1 enables dual forward passes) |
| `seed` | INT | `0` | Random seed (0 = random) |
| `max_mp` | FLOAT | `1.0` | Maximum output megapixels |

**Resolution presets available:**
`1024×1024 (1:1)`, `1152×896 (9:7)`, `896×1152 (7:9)`, `1216×832 (19:13)`, `832×1216 (13:19)`, `1344×768 (7:4)`, `768×1344 (4:7)`, `1536×640 (12:5)`, `640×1536 (5:12)`, `custom`

**Output:** `IMAGE`

---

### Eric Qwen-Image Unload

Free VRAM by unloading the generation pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_IMAGE_PIPELINE | *(optional)* | Pipeline to unload |
| `images` | IMAGE | *(optional)* | Passthrough — connect to trigger unload after generation |

**Output:** `status` (STRING)

---

### Eric Qwen-Image Apply LoRA

Apply a LoRA to the Qwen-Image generation pipeline. Loads LoRA weights onto the transformer. Multiple Apply LoRA nodes can be chained to stack several LoRAs with different weights. LoRAs are loaded from `ComfyUI/models/loras/`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_IMAGE_PIPELINE | — | From any generation loader node |
| `lora_name` | COMBO | — | Select LoRA from `ComfyUI/models/loras/` |
| `weight` | FLOAT | `1.0` | LoRA weight strength (−2.0 to 2.0, step 0.05). 1.0 = full, 0.5 = half |
| `lora_path_override` | STRING | *(empty)* | Optional: custom path override (leave empty to use dropdown) |

**Output:** `QWEN_IMAGE_PIPELINE`

---

### Eric Qwen-Image Unload LoRA

Unload all LoRAs from the Qwen-Image generation pipeline. Use to reset the model to its base state before applying different LoRAs, or to free memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_IMAGE_PIPELINE | — | Pipeline with LoRAs to unload |

**Output:** `QWEN_IMAGE_PIPELINE`

---

### Eric Qwen-Image Multi-Stage Generate

Progressive multi-stage text-to-image generation with full per-stage control. Up to 3 stages with independent steps, CFG, resolution, and denoise settings. Latents are upscaled between stages via bislerp and re-noised according to the per-stage denoise strength before re-sampling.

- Set `upscale_to_stage2 = 0` → output Stage 1 only (single-stage).
- Set `upscale_to_stage3 = 0` → stop after Stage 2 (two-stage).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_IMAGE_PIPELINE | — | From any generation loader node |
| `prompt` | STRING | — | Describe the image you want to generate |
| `negative_prompt` | STRING | *(empty)* | What to avoid in the output |
| `aspect_ratio` | COMBO | `1:1 Square` | Aspect ratio applied at every stage |
| `seed` | INT | `0` | Random seed (0 = random) |
| **Stage 1** | | | |
| `s1_mp` | FLOAT | `0.5` | Stage 1 resolution in megapixels (0.3–2.0) |
| `s1_steps` | INT | `15` | Stage 1 inference steps (txt2img from noise) |
| `s1_cfg` | FLOAT | `8.0` | Stage 1 true CFG scale |
| **Stage 2** | | | |
| `upscale_to_stage2` | FLOAT | `2.0` | Upscale factor (area) S1→S2. 0 = skip S2 & S3, output S1 |
| `s2_steps` | INT | `20` | Stage 2 inference steps |
| `s2_cfg` | FLOAT | `4.0` | Stage 2 true CFG scale |
| `s2_denoise` | FLOAT | `1.0` | Stage 2 denoise (1.0 = full, lower preserves prior detail) |
| **Stage 3** | | | |
| `upscale_to_stage3` | FLOAT | `2.0` | Upscale factor (area) S2→S3. 0 = skip S3, output S2 |
| `s3_steps` | INT | `15` | Stage 3 inference steps |
| `s3_cfg` | FLOAT | `2.0` | Stage 3 true CFG scale |
| `s3_denoise` | FLOAT | `1.0` | Stage 3 denoise |

**Output:** `IMAGE`

---

### Eric Qwen-Image UltraGen

Quality-focused multi-stage text-to-image generation (v2). Incorporates all Qwen-Image-2512 best practices: official Chinese negative prompt as default, `max_sequence_length` up to 1024 for detailed prompts, Spectrum acceleration on Stage 1, tuned defaults (0.5 MP s1 → 4× upscale → 26-step s2 refinement), per-stage seed modes, sigma schedule selection, and optional upscale VAE for 2× super-resolution decode.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_IMAGE_PIPELINE | — | From any generation loader node |
| `prompt` | STRING | — | Describe the image. For best results ~200 words. Connect Prompt Rewriter to auto-enhance. |
| `negative_prompt` | STRING | *(official Chinese default)* | Official Qwen-Image-2512 negative prompt |
| `aspect_ratio` | COMBO | `1:1 Square` | Aspect ratio applied at every stage |
| `seed` | INT | `0` | Random seed (0 = random) |
| `seed_mode` | COMBO | `offset_per_stage` | `same_all_stages`, `offset_per_stage` (S2=seed+1, S3=seed+2), or `random_per_stage` |
| `max_sequence_length` | INT | `1024` | Max prompt token length (128–1024, step 64). Full capacity by default. |
| **Stage 1** | | | |
| `s1_mp` | FLOAT | `0.5` | Stage 1 resolution in megapixels |
| `s1_steps` | INT | `15` | Stage 1 inference steps |
| `s1_cfg` | FLOAT | `10.0` | Stage 1 true CFG. High CFG at low res locks in composition. |
| **Stage 2** | | | |
| `upscale_to_stage2` | FLOAT | `4.0` | Upscale factor (area) S1→S2. 0 = skip S2 & S3. |
| `s2_steps` | INT | `26` | Stage 2 inference steps (main refinement) |
| `s2_cfg` | FLOAT | `4.0` | Stage 2 true CFG (matches official recommendation) |
| `s2_denoise` | FLOAT | `0.85` | Stage 2 denoise |
| `s2_sigma_schedule` | COMBO | `linear` | `linear`, `balanced` (Karras ρ=3), or `karras` (Karras ρ=7) |
| **Stage 3** | | | |
| `upscale_to_stage3` | FLOAT | `2.0` | Upscale factor (area) S2→S3. 0 = disabled. |
| `s3_steps` | INT | `18` | Stage 3 inference steps |
| `s3_cfg` | FLOAT | `2.0` | Stage 3 true CFG |
| `s3_denoise` | FLOAT | `0.45` | Stage 3 denoise (0.3–0.5 recommended for final polish) |
| `s3_sigma_schedule` | COMBO | `karras` | Sigma schedule for S3 (karras recommended for fine micro-texture) |
| **Upscale VAE** | | | |
| `upscale_vae` | UPSCALE_VAE | *(optional)* | From Eric Qwen Upscale VAE Loader |
| `upscale_vae_mode` | COMBO | `both` | `disabled`, `inter_stage`, `final_decode`, or `both` (see Upscale VAE section below) |

**Output:** `IMAGE`

---

### Eric Qwen-Image ControlNet Loader

Loads an InstantX Qwen-Image ControlNet model. Supports both the **Union** model (Canny, SoftEdge, Depth, Pose) and the **Inpainting** model. The model is kept on CPU and moved to GPU automatically when called by UltraGen CN or UltraGen Inpaint CN.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | STRING | `InstantX/Qwen-Image-ControlNet-Union` | HuggingFace model ID or local path |
| `dtype` | COMBO | `bfloat16` | Model precision: bfloat16, float16, float32 |

**Output:** `QWEN_IMAGE_CONTROLNET`

> **Models:**
> - [InstantX/Qwen-Image-ControlNet-Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union) — Canny, SoftEdge, Depth, Pose (recommended for generation)
> - [InstantX/Qwen-Image-ControlNet-Inpainting](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting) — Mask-based inpainting (experimental, see below)

---

### Eric Qwen-Image UltraGen CN

ControlNet-guided multi-stage text-to-image generation. Same architecture as UltraGen but uses the [InstantX/Qwen-Image-ControlNet-Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union) model on Stage 1 (and optionally Stage 2) to guide composition and structure from a control image. Supports Canny edge maps, SoftEdge/HED, depth maps, and OpenPose skeletons. Output up to **50 MP+** with upscale VAE.

Includes ControlNet auto-scaling that calibrates CN signal magnitude to match the transformer's hidden states, so the same `cn_target_strength` value works across different fine-tuned transformers without manual scale hunting.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_IMAGE_PIPELINE | — | From any generation loader node |
| `controlnet` | QWEN_IMAGE_CONTROLNET | — | From the ControlNet Loader |
| `control_image` | IMAGE | — | Control image (Canny, depth, pose, or soft edge map) |
| `cn_type` | COMBO | `canny` | ControlNet type: `canny`, `soft_edge`, `depth`, `pose` |
| `prompt` | STRING | — | Describe the image |
| `negative_prompt` | STRING | *(official default)* | Negative prompt |
| **ControlNet** | | | |
| `cn_auto_scale` | BOOLEAN | `True` | Auto-calibrate CN strength to transformer magnitude |
| `cn_target_strength` | FLOAT | `1.0` | CN influence (1.0 = standard, higher = stronger guidance) |
| `controlnet_conditioning_scale` | FLOAT | `1.0` | Manual CN scale (when auto-scale OFF) |
| `control_guidance_start` | FLOAT | `0.0` | When CN guidance begins (fraction of steps) |
| `control_guidance_end` | FLOAT | `1.0` | When CN guidance ends |
| **S2 ControlNet** | | | |
| `s2_cn_scale` | FLOAT | `1.0` | CN strength on Stage 2 (0 = disable CN for S2) |
| `s2_cn_start` | FLOAT | `0.0` | S2 CN guidance start |
| `s2_cn_end` | FLOAT | `1.0` | S2 CN guidance end |
| **Stages** | | | *(Same stage parameters as UltraGen — s1_mp, s1_steps, s1_cfg, upscale_to_stage2, s2_steps, etc.)* |
| **Upscale VAE** | | | *(Same upscale VAE parameters as UltraGen)* |

**Output:** `IMAGE`

---

### Eric Qwen-Image Spectrum Accelerator

Training-free diffusion sampling speedup using adaptive spectral feature forecasting (CVPR 2026). Predicts transformer outputs on skipped steps via Chebyshev polynomial regression instead of running all transformer blocks. Best for ≥20 inference steps and true CFG runs (2× transformer passes per step → double the savings). Wire between the Image Loader and any generation node.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | QWEN_IMAGE_PIPELINE | — | Pipeline to accelerate |
| `enable` | BOOLEAN | `True` | Enable/disable Spectrum acceleration |
| `warmup_steps` | INT | `3` | Initial denoising steps that always run the full transformer (2–4 recommended) |
| `window_size` | INT | `2` | Base period between actual transformer evaluations. 2 = every other step cached. |
| `flex_window` | FLOAT | `0.75` | Window growth rate. Later steps change less, so larger windows are safe. 0 = fixed window. |
| `w` | FLOAT | `0.5` | Blend between Chebyshev predictor (1.0) and Newton forward-difference predictor (0.0) |
| `lam` | FLOAT | `0.1` | Ridge regularization for Chebyshev regression. Higher = smoother predictions. |
| `M` | INT | `4` | Chebyshev polynomial degree (1–8). Higher captures complex trajectories but risks overfitting. |
| `min_steps` | INT | `15` | Auto-disable when `num_inference_steps` < this (low step counts don't benefit) |

**Output:** `QWEN_IMAGE_PIPELINE`

---

### Eric Qwen Prompt Rewriter

Enhance image prompts using a local or remote LLM. Rewrites terse prompts into rich ~200-word descriptions following Qwen-Image-2512 recommended methodology. Connects to any OpenAI-compatible API (Ollama, LM Studio, DeepSeek, OpenAI, etc.). API keys are loaded securely from environment variables or `api_keys.ini` — never stored in the workflow file. Output connects to the prompt input of any generation node.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | STRING | — | Original image description to enhance |
| `api_url` | STRING | `http://localhost:11434/v1` | OpenAI-compatible API base URL |
| `model` | STRING | `qwen3:8b` | Model name on the API server |
| `language` | COMBO | `English` | Language for the rewritten prompt (`English` or `Chinese`) |
| `temperature` | FLOAT | `0.7` | LLM temperature — lower = more faithful, higher = more creative |
| `max_tokens` | INT | `2048` | Max tokens for LLM response |
| `custom_instructions` | STRING | *(empty)* | Additional instructions appended to the system prompt |
| `lora_triggers` | STRING | *(empty)* | LoRA trigger words/phrases, one per line or comma-separated |
| `trigger_mode` | COMBO | `off` | How to apply trigger words: `incorporate`, `prepend`, `append`, or `off` |
| `passthrough` | BOOLEAN | `False` | Skip rewriting and pass prompt through unchanged (for A/B testing) |

**Output:** `enhanced_prompt` (STRING)

#### LoRA Trigger Words

Many LoRAs require specific trigger words or phrases in the prompt to activate their trained style or concept. The `lora_triggers` and `trigger_mode` inputs let you inject these automatically:

| Mode | Behavior |
|------|----------|
| `off` | Trigger words are ignored |
| `incorporate` | The LLM is instructed to weave the trigger words **verbatim** into the rewritten prompt naturally. Falls back to `prepend` when `passthrough` is enabled (no LLM call). |
| `prepend` | Trigger words are prepended to the prompt (before the rewritten text). Works even in passthrough mode. |
| `append` | Trigger words are appended to the prompt (after the rewritten text). Works even in passthrough mode. |

**Usage:** Enter one trigger per line, or separate with commas. For example:
```
ohwx woman
cinematic lighting
film grain
```

When using `incorporate` mode, the LLM receives an additional system instruction requiring the trigger words to appear verbatim in the output, so they blend naturally into the description rather than being awkwardly tacked on.

---

## ⚠️ Experimental: ControlNet Inpainting Nodes

> **Motivation:** Qwen-Image-Edit redraws the *entire* image on every edit, which progressively degrades areas outside the edit region — fine details, textures, and sharpness are lost across the whole canvas. A true inpainting pipeline would regenerate *only* the masked region while leaving the rest of the image completely untouched, preserving full original quality. That is the goal of these ControlNet inpainting nodes.
>
> **Status: Experimental — not fully working.** These nodes are functional but produce visible halos and ghosting artifacts from double-sampling at mask boundaries. The multi-stage pipeline generates the full image from noise while the ControlNet conditions on the masked source, but compositing the result back onto the original creates noticeable seams that the harmonization pass has not yet fully resolved. We believe techniques from the Qwen-Edit inpaint nodes (which use a fundamentally different conditioning approach) may help, but this has not been explored yet.
>
> **Alternative:** The **Eric Qwen-Edit Inpaint** node provides a separate experimental approach to masked inpainting using the Qwen-Image-Edit model. It blanks out the masked region before sending the image through both the VL and VAE encoders, then composites the generated output back onto the original using the mask with edge feathering. This approach currently produces better results than the ControlNet inpainting nodes, though it still relies on Qwen-Edit (which reprocesses the full image internally) and is itself experimental.
>
> These nodes are included for experimentation.

### Eric Qwen-Image UltraGen Inpaint CN

ControlNet-guided multi-stage inpainting and outpainting using the [InstantX/Qwen-Image-ControlNet-Inpainting](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting) model. Uses `QwenImageControlNetInpaintPipeline` with 17-channel conditioning (16ch VAE-encoded masked image + 1ch mask). Supports object replacement, background replacement, text modification, and outpainting.

**Architecture:** Up to 3 stages — S1 (CN draft), S2A (CN refine + dilated mask), S2B (whole-image harmonize, no CN), S3 (polish upscale, no CN). Smart stage selection (`auto_stages`) skips S1 when input is already large enough. Final feathered composite preserves original pixels outside the mask.

**Known issues:**
- Halo artifacts at mask boundaries due to double-sampling
- Ghosting where generated content overlaps original pixels
- Harmonization pass (S2B) reduces but does not eliminate boundary artifacts

### Eric Qwen Inpaint Prompt Rewriter

VLM-powered prompt rewriter for inpainting. Analyzes the source image and mask to generate short, change-focused prompts (40–80 words) describing the desired edit. Uses mask outline overlay for spatial awareness.

### Eric Qwen ControlNet Prompt Rewriter

VLM-powered prompt rewriter for ControlNet-guided generation. Generates full scene descriptions (200–400 words) with CN-type awareness (Canny, SoftEdge, Depth, Pose). Outputs both the prompt and a `cn_type_index` integer.

---

## 2× Upscale VAE (Super-Resolution Decode)

The **Wan2.1-VAE-upscale2x** by [spacepxl](https://huggingface.co/spacepxl) is a decoder-only finetune of the Wan2.1 VAE that outputs 12 channels instead of 3. After decode, `pixel_shuffle(12→3, 2×)` produces a **2× upscaled image** — effectively free super-resolution during VAE decode with no extra diffusion steps.

The Wan2.1 and Qwen-Image VAEs are architecturally identical (`AutoencoderKLWan` / `AutoencoderKLQwenImage`) and share the same latent space, so the upscale VAE works directly with Qwen-Image latents.

### How it works

1. **Load the upscale VAE** with the **Eric Qwen Upscale VAE Loader** node
2. **Connect it** to the `upscale_vae` input on the **Eric Qwen-Image UltraGen** node
3. **Choose a mode** via the `upscale_vae_mode` dropdown

### Upscale VAE Modes

| Mode | Description |
|------|-------------|
| `disabled` | Upscale VAE ignored even if connected (safe default) |
| `inter_stage` | Decode S2 latents at 2× via the upscale VAE, re-encode back to latents, and feed the 2× canvas to S3. Replaces the bislerp inter-stage upscale with a higher-quality decode→2×→re-encode round trip. Requires 3 active stages. |
| `final_decode` | Replace the final stage's normal VAE decode with the 2× upscale decode. The output image is 2× the resolution of the final denoising stage. |
| `both` | Inter-stage S2→S3 **and** 2× final decode. These stack: S3 runs on a 2× canvas from inter-stage, then the output gets another 2× from final decode = **4× total** vs. S2. |

### VRAM Management

The upscale VAE is kept on CPU until needed. Before decode, the diffusion transformer is automatically offloaded to CPU to free VRAM. For large images, **tiled VAE decoding** is automatically enabled when latent spatial dimensions exceed 128 (roughly ≥1024 px per side before the 2× upscale).

### Typical workflow

```
[Qwen-Image Loader] → [LoRA] → [Spectrum] → [Upscale VAE Loader] → [UltraGen]
                                                     ↑                    ↑
                                              upscale_vae ──────── upscale_vae
                                                              upscale_vae_mode = final_decode
```

---

### Eric Qwen Upscale VAE Loader

Load the Wan2.1 2× upscale VAE. The model is kept on CPU until decode is requested.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | STRING | `spacepxl/Wan2.1-VAE-upscale2x` | HuggingFace model ID or local path |
| `subfolder` | STRING | `diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1` | Subfolder within the repo containing config.json + weights. Leave blank if model_path already points to the correct directory. |
| `dtype` | COMBO | `bfloat16` | Model precision: bfloat16 (recommended), float16, float32 |

**Output:** `UPSCALE_VAE`

The first run downloads the model from HuggingFace (~0.5 GB). Subsequent runs load from the local HuggingFace cache. You can also download the model manually and point `model_path` to the local directory.

> **Model source:** [spacepxl/Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x) — a decoder-only finetune of the Wan2.1 VAE by spacepxl. The specific subfolder used is `diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1` (image-only variant, trained on real images).

---

## Architecture Notes

Qwen-Image-Edit-2511 uses a **dual conditioning path**:

1. **VL path** — Each input image is processed by the built-in Qwen2.5-VL vision-language encoder at ~384 px to produce semantic token embeddings. These tell the model *what* is in each image.
2. **VAE/ref path** — Each input image is VAE-encoded at output resolution to produce pixel-level latents. These tell the model *how* to render the pixels.

Most multi-image nodes expose per-image `vl_*` and `ref_*` toggles so you can control which path each image participates in. For example, in Style Transfer, the style image defaults to VL-only (semantic style cues) while the content image defaults to both VL + ref (preserving pixel structure).

## Example Workflows

All workflow PNGs below have the full ComfyUI workflow embedded — drag them directly into ComfyUI to load.

### Qwen-Edit Hi-Res — Simple

A minimal editing workflow: loader → LoRA → edit node → output. Quick to set up and great for getting started.

![Qwen-Edit Hi-Res Simple workflow](workflows/Qwen-Edit-HiRes-Simple.png)

### Qwen-Edit Hi-Res — Advanced

A full-featured editing workflow with multi-stage generation, Spectrum acceleration, upscale VAE, and fine-grained stage controls.

![Qwen-Edit Hi-Res Advanced workflow](workflows/Qwen-Edit-HiRes-Adv.png)

### Qwen-Image Hi-Res with ControlNet

Text-to-image generation guided by ControlNet (Canny, SoftEdge, Depth, or Pose) using the InstantX Union model, with multi-stage UltraGen upscaling.

![Qwen-Image Hi-Res ControlNet workflow](workflows/Qwen-Image-HiRes-Controlnet.png)

### Qwen-Image UltraGen Hi-Res (30 MP+)

Text-to-image generation without ControlNet using the UltraGen multi-stage pipeline. Produces 30 MP+ output with Spectrum acceleration and upscale VAE.

![Qwen-Image UltraGen Hi-Res 30MP+ workflow](workflows/Qwen-image-UltraGen-HiRes-30mp-plus.png)

### Qwen-Image UltraGen — Advanced

Advanced UltraGen workflow with Prompt Rewriter, selective sharpening, and several other features for high-quality text-to-image generation.

![Qwen-Image UltraGen Advanced workflow](workflows/Qwen-UltraGen-Adv.png)

See the `examples/` and `workflows/` folders for additional workflow files and screenshots.

## Example Prompts

- "Change the background to a sunset over the ocean"
- "Make the person smile"
- "Add a red hat to the person"
- "Change the car color from blue to red"
- "Remove the text from the image"
- "Make it look like a painting"
- "Harmonise the pasted element with its surroundings" *(inpaint transfer)*
- "Apply the watercolor style of Picture 1 to Picture 2" *(style transfer)*

## Tips

1. **Start with lower max_mp** (4–6) to test edits, then increase
2. **Use the lightning LoRA** with 8 steps for fast iteration on edit nodes (50 steps without)
3. **Use negative prompts** to avoid unwanted elements
4. **VAE tiling is automatic** — no configuration needed
5. **Progress bars** appear in ComfyUI during denoising on all edit and generation nodes
6. **Spectrum accelerator** can cut generation time by 3–5× with ≥15 steps (works with both edit and generation pipelines)
7. **For inpaint transfer**, provide a `transfer_mask` to select exactly which part of the reference image to use. The node handles all scaling and positioning automatically.
8. **Delta Overlay** is great for up-res workflows: edit at low resolution, upscale the original, then apply only the changed pixels at full resolution.
9. **Generation resolution presets** let you quickly choose common aspect ratios without doing pixel math.
10. **Edit and generation pipelines are separate** — you can load both simultaneously if you have enough VRAM.
11. **Increase `max_sequence_length`** in UltraGen if you use very detailed prompts or the Prompt Rewriter node (see below).

### Extended Prompt Token Length (`max_sequence_length`)

Most Qwen-Image ComfyUI workflows and the default diffusers pipeline hard-code the prompt token budget at 512 tokens. The UltraGen node exposes this as a configurable parameter (`max_sequence_length`, 128–1024) — a feature **not available in other Qwen-Image nodes or workflows**.

**How it works:** After the Qwen2.5-VL text encoder produces token embeddings from your prompt, the sequence is truncated to `max_sequence_length` before being fed to the transformer. If your prompt is shorter than the limit, the extra positions are zero-padded and ignored via the attention mask — so there is no quality penalty for setting it higher than needed.

| Consideration | Impact |
|---------------|--------|
| **Prompt fidelity** | Higher values preserve more detail from long prompts. At 512, prompts over ~200 words may be silently truncated. |
| **Generation time** | Slightly more cross-attention compute per step. Negligible for most prompts — the image latent sequence dominates. |
| **VRAM** | ~8 MB extra per batch item at 1024 vs 512 (trivial vs. the 38 GB transformer). |
| **Quality** | No degradation — unused positions are masked out. |

**Recommendation:** Leave at **512** for typical prompts. Increase to **768–1024** when using the Prompt Rewriter node or manually writing very detailed descriptions (300+ words). The maximum is 1024 (hard limit in the model architecture).

## Credits

- **Qwen-Image-Edit / Qwen-Image**: Developed by Qwen Team (Alibaba)
- **Wan2.1-VAE-upscale2x**: 2× super-resolution VAE by [spacepxl](https://huggingface.co/spacepxl) — model weights: [Apache-2.0](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x), reference code: [MIT](https://github.com/spacepxl/ComfyUI-VAE-Utils)
- **Spectrum**: Han *et al.*, "Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration" (CVPR 2026)
- **ComfyUI Nodes**: Eric Hiss (GitHub: [EricRollei](https://github.com/EricRollei))

## License

Dual licensed: **CC BY-NC 4.0** for non-commercial use, separate commercial license available. See [LICENSE.txt](LICENSE.txt) for full terms.

Contact: eric@rollei.us / eric@historic.camera

## Related

- [Eric UniPic3 Nodes](https://github.com/EricRollei/Eric_UniPic3) — Similar nodes for UniPic3 model
- [Qwen-Image GitHub](https://github.com/QwenLM/Qwen-Image)
