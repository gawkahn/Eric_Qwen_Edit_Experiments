# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit & Qwen-Image Node Definitions
"""

# ── Generic multi-model nodes (GEN_PIPELINE type) ───────────────────────
from .eric_diffusion_loader import EricDiffusionLoader, EricDiffusionUnload
from .eric_diffusion_generate import EricDiffusionGenerate
from .eric_diffusion_component_loader import EricDiffusionComponentLoader
from .eric_diffusion_multistage import EricDiffusionMultiStage
from .eric_diffusion_ultragen import EricDiffusionUltraGen
from .eric_diffusion_advanced_generate import EricDiffusionAdvancedGenerate
from .eric_diffusion_advanced_multistage import EricDiffusionAdvancedMultiStage
from .eric_diffusion_advanced_edit import EricDiffusionAdvancedEdit
from .eric_diffusion_advanced_edit_multistage import EricDiffusionAdvancedEditMultistage
from .eric_diffusion_lora_stacker import (
    EricDiffusionLoRAStacker,
    EricQwenEditLoRAStacker,
)
from .eric_diffusion_save import EricDiffusionSave

# ── Edit nodes ──────────────────────────────────────────────────────────
from .eric_qwen_edit_loader import EricQwenEditLoader, EricQwenEditUnload
from .eric_qwen_edit_node import EricQwenEditImage
from .eric_qwen_edit_inpaint import EricQwenEditInpaint
from .eric_qwen_edit_inpaint_transfer import EricQwenEditInpaintTransfer
from .eric_qwen_edit_lora import EricQwenEditApplyLoRA, EricQwenEditUnloadLoRA
from .eric_qwen_edit_component_loader import EricQwenEditComponentLoader
from .eric_qwen_edit_multi_image import EricQwenEditMultiImage
from .eric_qwen_edit_style_transfer import EricQwenEditStyleTransfer
from .eric_qwen_edit_delta import EricQwenEditDelta, EricQwenEditApplyMask
from .eric_qwen_edit_spectrum import EricQwenEditSpectrum

# ── Generation nodes (Qwen-Image / Qwen-Image-2512) ────────────────────
from .eric_qwen_image_loader import EricQwenImageLoader, EricQwenImageUnload
from .eric_qwen_image_component_loader import EricQwenImageComponentLoader
from .eric_qwen_image_generate import EricQwenImageGenerate
from .eric_qwen_image_lora import EricQwenImageApplyLoRA, EricQwenImageUnloadLoRA
from .eric_qwen_image_multistage import EricQwenImageMultiStage
from .eric_qwen_image_ultragen import EricQwenImageUltraGen
from .eric_qwen_image_spectrum import EricQwenImageSpectrum
from .eric_qwen_image_controlnet_loader import EricQwenImageControlNetLoader, EricQwenImageControlNetUnload
from .eric_qwen_image_ultragen_cn import EricQwenImageUltraGenCN
from .eric_qwen_image_ultragen_inpaint_cn import EricQwenImageUltraGenInpaintCN
from .eric_qwen_prompt_rewriter import EricQwenPromptRewriter
from .eric_qwen_inpaint_prompt_rewriter import EricQwenInpaintPromptRewriter
from .eric_qwen_controlnet_prompt_rewriter import EricQwenControlNetPromptRewriter
from .eric_qwen_upscale_vae import EricQwenUpscaleVAELoader

NODE_CLASS_MAPPINGS = {
    # Generic multi-model (GEN_PIPELINE)
    "Eric Diffusion Loader":            EricDiffusionLoader,
    "Eric Diffusion Unload":            EricDiffusionUnload,
    "Eric Diffusion Generate":          EricDiffusionGenerate,
    "Eric Diffusion Component Loader":  EricDiffusionComponentLoader,
    "Eric Diffusion Multi-Stage":       EricDiffusionMultiStage,
    "Eric Diffusion UltraGen":          EricDiffusionUltraGen,
    "Eric Diffusion Advanced Generate": EricDiffusionAdvancedGenerate,
    "Eric Diffusion Advanced Multi-Stage": EricDiffusionAdvancedMultiStage,
    "Eric Diffusion Advanced Edit":     EricDiffusionAdvancedEdit,
    "Eric Diffusion Advanced Edit Multi-Stage": EricDiffusionAdvancedEditMultistage,
    "Eric Diffusion LoRA Stacker":      EricDiffusionLoRAStacker,
    "Eric Diffusion Save":              EricDiffusionSave,
    # Edit
    "Eric Qwen-Edit Loader": EricQwenEditLoader,
    "Eric Qwen-Edit Unload": EricQwenEditUnload,
    "Eric Qwen-Edit Image": EricQwenEditImage,
    "Eric Qwen-Edit Inpaint": EricQwenEditInpaint,
    "Eric Qwen-Edit Inpaint Transfer": EricQwenEditInpaintTransfer,
    "Eric Qwen-Edit Apply LoRA": EricQwenEditApplyLoRA,
    "Eric Qwen-Edit Unload LoRA": EricQwenEditUnloadLoRA,
    "Eric Qwen-Edit Component Loader": EricQwenEditComponentLoader,
    "Eric Qwen-Edit Multi-Image": EricQwenEditMultiImage,
    "Eric Qwen-Edit Style Transfer": EricQwenEditStyleTransfer,
    "Eric Qwen-Edit Delta": EricQwenEditDelta,
    "Eric Qwen-Edit Apply Mask": EricQwenEditApplyMask,
    "Eric Qwen-Edit Spectrum": EricQwenEditSpectrum,
    "Eric Qwen-Edit LoRA Stacker": EricQwenEditLoRAStacker,
    # Generation
    "Eric Qwen-Image Loader": EricQwenImageLoader,
    "Eric Qwen-Image Unload": EricQwenImageUnload,
    "Eric Qwen-Image Component Loader": EricQwenImageComponentLoader,
    "Eric Qwen-Image Generate": EricQwenImageGenerate,
    "Eric Qwen-Image Apply LoRA": EricQwenImageApplyLoRA,
    "Eric Qwen-Image Unload LoRA": EricQwenImageUnloadLoRA,
    "Eric Qwen-Image Multi-Stage": EricQwenImageMultiStage,
    "Eric Qwen-Image UltraGen": EricQwenImageUltraGen,
    "Eric Qwen-Image Spectrum": EricQwenImageSpectrum,
    "Eric Qwen-Image ControlNet Loader": EricQwenImageControlNetLoader,
    "Eric Qwen-Image ControlNet Unload": EricQwenImageControlNetUnload,
    "Eric Qwen-Image UltraGen CN": EricQwenImageUltraGenCN,
    "Eric Qwen-Image UltraGen Inpaint CN": EricQwenImageUltraGenInpaintCN,
    "Eric Qwen Prompt Rewriter": EricQwenPromptRewriter,
    "Eric Qwen Inpaint Prompt Rewriter": EricQwenInpaintPromptRewriter,
    "Eric Qwen ControlNet Prompt Rewriter": EricQwenControlNetPromptRewriter,
    # Utility
    "Eric Qwen Upscale VAE Loader": EricQwenUpscaleVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Generic multi-model
    "Eric Diffusion Loader":            "Eric Diffusion Load Model",
    "Eric Diffusion Unload":            "Eric Diffusion Unload",
    "Eric Diffusion Generate":          "Eric Diffusion Generate",
    "Eric Diffusion Component Loader":  "Eric Diffusion Component Loader",
    "Eric Diffusion Multi-Stage":       "Eric Diffusion Multi-Stage Generate",
    "Eric Diffusion UltraGen":          "Eric Diffusion UltraGen",
    "Eric Diffusion Advanced Generate": "Eric Diffusion Advanced Generate (Flux/Chroma)",
    "Eric Diffusion Advanced Multi-Stage": "Eric Diffusion Advanced Multi-Stage (Flux/Chroma)",
    "Eric Diffusion Advanced Edit":     "Eric Diffusion Advanced Edit (Qwen Edit)",
    "Eric Diffusion Advanced Edit Multi-Stage": "Eric Diffusion Advanced Edit Multi-Stage (Qwen Edit)",
    "Eric Diffusion LoRA Stacker":      "Eric Diffusion LoRA Stacker",
    "Eric Diffusion Save":              "Eric Diffusion Save Image",
    # Edit
    "Eric Qwen-Edit Loader": "Eric Qwen-Edit Load Model",
    "Eric Qwen-Edit Unload": "Eric Qwen-Edit Unload",
    "Eric Qwen-Edit Image": "Eric Qwen-Edit Image",
    "Eric Qwen-Edit Inpaint": "Eric Qwen-Edit Inpaint",
    "Eric Qwen-Edit Inpaint Transfer": "Eric Qwen-Edit Inpaint Transfer",
    "Eric Qwen-Edit Apply LoRA": "Eric Qwen-Edit Apply LoRA",
    "Eric Qwen-Edit Unload LoRA": "Eric Qwen-Edit Unload LoRA",
    "Eric Qwen-Edit Component Loader": "Eric Qwen-Edit Component Loader",
    "Eric Qwen-Edit Multi-Image": "Eric Qwen-Edit Multi-Image Fusion",
    "Eric Qwen-Edit Style Transfer": "Eric Qwen-Edit Style Transfer",
    "Eric Qwen-Edit Delta": "Eric Qwen-Edit Delta Overlay",
    "Eric Qwen-Edit Apply Mask": "Eric Qwen-Edit Apply Mask",
    "Eric Qwen-Edit Spectrum": "Eric Qwen-Edit Spectrum Accelerator",
    "Eric Qwen-Edit LoRA Stacker": "Eric Qwen-Edit LoRA Stacker (8 slots)",
    # Generation
    "Eric Qwen-Image Loader": "Eric Qwen-Image Load Model",
    "Eric Qwen-Image Unload": "Eric Qwen-Image Unload",
    "Eric Qwen-Image Component Loader": "Eric Qwen-Image Component Loader",
    "Eric Qwen-Image Generate": "Eric Qwen-Image Generate",
    "Eric Qwen-Image Apply LoRA": "Eric Qwen-Image Apply LoRA",
    "Eric Qwen-Image Unload LoRA": "Eric Qwen-Image Unload LoRA",
    "Eric Qwen-Image Multi-Stage": "Eric Qwen-Image Multi-Stage Generate",
    "Eric Qwen-Image UltraGen": "Eric Qwen-Image UltraGen",
    "Eric Qwen-Image Spectrum": "Eric Qwen-Image Spectrum Accelerator",
    "Eric Qwen-Image ControlNet Loader": "Eric Qwen-Image ControlNet Loader",
    "Eric Qwen-Image ControlNet Unload": "Eric Qwen-Image ControlNet Unload",
    "Eric Qwen-Image UltraGen CN": "Eric Qwen-Image UltraGen (ControlNet)",
    "Eric Qwen-Image UltraGen Inpaint CN": "Eric Qwen-Image UltraGen Inpaint (ControlNet)",
    "Eric Qwen Prompt Rewriter": "Eric Qwen Prompt Rewriter",
    "Eric Qwen Inpaint Prompt Rewriter": "Eric Qwen Inpaint Prompt Rewriter (Vision)",
    "Eric Qwen ControlNet Prompt Rewriter": "Eric Qwen ControlNet Prompt Rewriter (Vision)",
    # Utility
    "Eric Qwen Upscale VAE Loader": "Eric Qwen Upscale VAE Loader (2×)",
}
