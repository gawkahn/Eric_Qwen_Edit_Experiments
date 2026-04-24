# Removed Nodes

Record of nodes that have been unregistered and (where applicable) deleted. Git history is the authoritative archive — this file is a pointer for "what was that node, again?" without needing to browse the log.

To resurrect any node file verbatim:
```bash
git show <COMMIT-BEFORE-REMOVAL>:nodes/<filename>.py
```

---

## 2026-04-24 — Legacy node cull

**Context:** The ComfyUI node pack evolved two tracks — a Qwen-specific track (`Eric Qwen-Image *`, `Eric Qwen-Edit *`) and a unified multi-model track (`Eric Diffusion *`) that supports Qwen, Flux, Chroma, SDXL, SD3.5, Pony, Illustrious, etc. through `model_index.json` auto-detection. The unified track absorbed Qwen-Image generation entirely and most Qwen-Edit functionality. This commit unregistered 12 superseded node classes and deleted 5 orphan files. Remove-commit SHA referenced below as `HEAD~1` relative to this file's introduction.

### Unregistered AND file deleted

| Node (registration key) | Class | File | Superseded by |
|---|---|---|---|
| Eric Qwen-Image Loader | `EricQwenImageLoader` | `eric_qwen_image_loader.py` | Eric Diffusion Loader (or Component Loader) |
| Eric Qwen-Image Unload | `EricQwenImageUnload` | `eric_qwen_image_loader.py` | Eric Diffusion Unload |
| Eric Qwen-Image Component Loader | `EricQwenImageComponentLoader` | `eric_qwen_image_component_loader.py` | Eric Diffusion Component Loader |
| Eric Qwen-Image Apply LoRA | `EricQwenImageApplyLoRA` | `eric_qwen_image_lora.py` | Eric Diffusion LoRA Stacker |
| Eric Qwen-Image Unload LoRA | `EricQwenImageUnloadLoRA` | `eric_qwen_image_lora.py` | Eric Diffusion LoRA Stacker |
| Eric Qwen-Edit Image | `EricQwenEditImage` | `eric_qwen_edit_node.py` | Eric Diffusion Advanced Edit |
| Eric Qwen-Edit Multi-Image | `EricQwenEditMultiImage` | `eric_qwen_edit_multi_image.py` | Eric Diffusion Advanced Edit (multi-image support folded in) |

Resurrect any of the above: `git show HEAD~1:nodes/<filename>.py`

### Unregistered, file retained for shared helpers

| Node (registration key) | Class (removed from file) | File (kept) | Why the file stays |
|---|---|---|---|
| Eric Qwen-Image Generate | `EricQwenImageGenerate` | `eric_qwen_image_generate.py` | `ASPECT_RATIOS`, `_align`, `compute_dimensions_from_ratio` used by ControlNet subsystem |
| Eric Qwen-Image Multi-Stage | `EricQwenImageMultiStage` | `eric_qwen_image_multistage.py` | `_pack_latents` / `_unpack_latents` used by live diffusion nodes; sigma-schedule helpers used by ControlNet subsystem |
| Eric Qwen-Image UltraGen | `EricQwenImageUltraGen` | `eric_qwen_image_ultragen.py` | `_apply_lora_stage_weights`, `QWEN_OFFICIAL_RESOLUTIONS`, `DEFAULT_NEGATIVE_PROMPT` used by ControlNet subsystem |
| Eric Qwen-Edit Apply LoRA | `EricQwenEditApplyLoRA` | `eric_qwen_edit_lora.py` | `load_lora_with_key_fix`, `_set_adapters_safe`, `get_lora_list`, and all adapter-format loaders are the pack-wide LoRA backbone |
| Eric Qwen-Edit Unload LoRA | `EricQwenEditUnloadLoRA` | `eric_qwen_edit_lora.py` | (same file as above) |

Resurrect any of the above class definitions: `git show HEAD~1:nodes/<filename>.py` — the classes are at the end of the original file, below all the helpers.

### Still in the pack but flagged for follow-up (see `Backlog.md`)

Not removed in this slice — these need investigation first:

- Qwen-Image ControlNet subsystem (`Eric Qwen-Image ControlNet Loader/Unload`, `UltraGen CN`, `UltraGen Inpaint CN`, `Spectrum`)
- Qwen-Edit auxiliary nodes (Delta, Inpaint, Inpaint Transfer, Spectrum, Style Transfer)
- Qwen prompt rewriters (plain, inpaint, ControlNet — likely duplicative with standalone LLM nodes)
