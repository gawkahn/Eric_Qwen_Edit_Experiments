"""Hunyuan-Image 2.1 base + refiner chained dispatch for comfyless.

Tencent designed Hunyuan-Image 2.1 as a two-stage pipeline where the
refiner explicitly "further enhances image quality and clarity, while
minimizing artifacts" (Tencent README §Architecture). ADR-014's
2026-05-24 amendment retracted the original "refiner as SDXL-style
optional polish" framing in favor of a Cascade-coupled dispatch shape:
both stages are required for clean output, even though the data
exchanged between them is images (structurally edit-shape) rather than
latents (structurally Cascade-shape).

Activated when comfyless.generate detects `model_family == "hunyuan-image"`
AND `refiner_path` is non-empty. Path resolution rides the existing
`resolve_hf_path` + `--allow-hf-download` surface — no new resolver
code, no filesystem auto-discovery (rejected at Vision review pass per
ADR-016 Alternative A: TOCTOU / containment-escape / symlink-traversal
risk on a caller-supplied path).

See:
- ADR-016: design rationale (this slice)
- docs/vision/slice-hunyuan-image-2-1-refiner.md: invariants
- ADR-014 §3 (2026-05-24 amendment): original framing retraction
- comfyless/cascade.py: precedent for family-specific dispatch forks
  (ADR-010)
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional


_REFINER_PIPELINE_CLASS_NAME = "HunyuanImageRefinerPipeline"
"""Locked at load time. ADR-016 §(b) — load_refiner_pipeline rejects any
other pipeline class to defend Vision §"Failure semantics" (clean error
on a wrong-class --refiner path; no silent fallback)."""


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _resolve_torch_dtype(precision: str):
    import torch
    name = (precision or "bf16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision!r}")


def load_refiner_pipeline(
    refiner_path: str,
    base_pipe,
    *,
    precision: str = "bf16",
    device: str = "cuda",
    vae_tiling: str = "auto",
    allow_hf_download: bool = False,
):
    """Load HunyuanImageRefinerPipeline alongside an already-loaded base.

    Asymmetric shared-encoder optimization per ADR-016 §(e):
    - `text_encoder` (Qwen2.5-VL, ~14 GB bf16) and `tokenizer` are
      injected from `base_pipe` via kwarg-override — saves a full
      ~14 GB re-load of the VL encoder on the refiner side.
    - The refiner pipeline class has no `text_encoder_2` slot (no
      T5/ByT5 stack) and no `guider` / `ocr_guider` — those base-side
      components are not shared because they have no refiner-side
      target.
    - `id(base.text_encoder) == id(refiner.text_encoder)` after load
      (Vision invariant 9).

    The refiner's `model_index.json._class_name` is checked at load
    time; any value other than `HunyuanImageRefinerPipeline` raises a
    clean ValueError (no silent fallback — the opt-in signal was
    explicit, the operator wants the refiner, masking the misconfig
    would defeat the point).

    Family-aware VAE tiling resolution per ADR-014's tile-VAE-skip
    amendment: the refiner declares its own family
    (`hunyuan-image-refiner`) which is NOT in the
    `_VAE_TILING_FAMILIES_DEFAULT_OFF` set today, so `auto` resolves to
    tiling-on. Operator override (`--vae-tiling off`) still applies.
    """
    from nodes.eric_diffusion_utils import (
        detect_pipeline_class,
        resolve_hf_path,
        resolve_vae_tiling,
    )

    refiner_path = resolve_hf_path(refiner_path, allow_download=allow_hf_download)
    _log(f"[comfyless] Loading refiner: {refiner_path}")

    pipeline_class, class_name, family = detect_pipeline_class(refiner_path)
    if class_name != _REFINER_PIPELINE_CLASS_NAME:
        raise ValueError(
            f"--refiner path resolves to pipeline class {class_name!r}; "
            f"expected {_REFINER_PIPELINE_CLASS_NAME!r}. Refiner chaining "
            f"requires a HunyuanImage-2.1-Refiner-Diffusers checkpoint."
        )
    _log(f"[comfyless] Detected refiner: {class_name} (family: {family})")

    dtype = _resolve_torch_dtype(precision)

    refiner = pipeline_class.from_pretrained(
        refiner_path,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        torch_dtype=dtype,
        local_files_only=True,
    )
    refiner = refiner.to(device)

    if hasattr(refiner, "vae") and hasattr(refiner.vae, "enable_tiling"):
        if resolve_vae_tiling(family, vae_tiling):
            refiner.vae.enable_tiling()
            _log(f"[comfyless] Refiner VAE tiling enabled (vae_tiling={vae_tiling})")
        else:
            if hasattr(refiner.vae, "disable_tiling"):
                refiner.vae.disable_tiling()
            _log(f"[comfyless] Refiner VAE tiling disabled (vae_tiling={vae_tiling})")

    _log(f"[comfyless] Refiner ready (shared text_encoder with base)")
    return refiner


def build_refiner_call_kwargs(
    refiner_pipe,
    image,
    prompt: str,
    negative_prompt: Optional[str],
    refiner_steps: int,
    refiner_cfg: float,
    generator,
) -> Dict[str, Any]:
    """Build refiner-side __call__ kwargs (ADR-016 §(f) CFG parity).

    Mirrors the base's distilled-guidance routing from ADR-014 §2:
    `refiner_cfg` → `distilled_guidance_scale` (the refiner is also a
    guidance-distilled model — single-pass CFG, not 2× true-CFG).

    `negative_prompt` is shared between stages per ADR-016 §(f); when
    set, forwarded verbatim. v1 has no `--refiner-negative` — a future
    slice can add it if empirical evidence shows stage-specific
    negatives improve output.

    The refiner reads its CFG from `refiner_cfg`, NOT `cfg_scale` — the
    two stages route their CFG independently through the precedence
    ladder (ADR-016 §(d), Vision invariant 5).
    """
    kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "image": image,
        "num_inference_steps": refiner_steps,
        "distilled_guidance_scale": refiner_cfg,
        "generator": generator,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    return kwargs


def run_chain(
    base_pipe,
    refiner_pipe,
    base_kwargs: Dict[str, Any],
    *,
    prompt: str,
    negative_prompt: Optional[str],
    refiner_steps: int,
    refiner_cfg: float,
    generator,
):
    """Run base → PIL → refiner; return the final PIL image.

    PIL roundtrip transport per ADR-016 §(d): base's VAE class
    (`AutoencoderKLHunyuanImage`) and refiner's
    (`AutoencoderKLHunyuanImageRefiner`) are not interchangeable, so
    latents from base's VAE would not decode cleanly through refiner's
    VAE. Latent passthrough is ADR-016 Alternative H, deferred.

    The caller owns the sampler-swap context around this call: the
    base-side sampler swap MUST NOT leak to the refiner pipeline (Vision
    invariant 8 — refiner scheduler pinned per-pipeline). The
    `swap_sampler` context manager is per-pipe, so the comfyless
    caller wraps run_chain with one swap context targeting the base
    pipe only; the refiner's scheduler is untouched.

    `base_kwargs` is the result of `comfyless.generate._build_call_kwargs`
    for the base pipeline. The refiner kwargs are built internally by
    `build_refiner_call_kwargs` because the refiner's `image=` input
    depends on the base call's output PIL.
    """
    base_result = base_pipe(**base_kwargs)
    base_pil = base_result.images[0]

    refiner_kwargs = build_refiner_call_kwargs(
        refiner_pipe, base_pil, prompt, negative_prompt,
        refiner_steps, refiner_cfg, generator,
    )
    refiner_result = refiner_pipe(**refiner_kwargs)
    return refiner_result.images[0]
