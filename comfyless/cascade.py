"""Stable Cascade dispatch for comfyless.

Stable Cascade is the only family in comfyless that doesn't follow the
standard "one repo / one pipeline" load shape. It has three independent
weight files (Stage C, Stage B, Stage A) chained as prior → decoder → VAE.

Activated via the literal sentinel `--model stablecascade <config.json>
[config2.json] ...`. Multiple positional configs auto-iterate (dump-and-
reload between configs). The JSON owns topology (paths) and denoising
params; the CLI owns shared concerns (prompt, seed, output, --batch,
--limit, --negative-prompt).

See:
- ADR-010: design rationale (why this is a separate dispatch, not a schema extension)
- docs/comfyless-stable-cascade.md: user-facing reference
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════

CASCADE_SENTINEL = "stablecascade"
"""The literal --model token that activates this dispatch."""

_ALIGN = 128
"""Cascade resolution_multiple = 42.67 → 128px in image space (3 VAE downsamples)."""

_DEFAULT_SCAFFOLDING_REPO = "stabilityai/stable-cascade"

# Default config-dir paths used by the conversion utility's smoke test.
# Single source of truth — convert_cascade_comfyui.py imports this.
PRIOR_CONFIG_DIRS = {
    # (stage, lite) → diffusers tree dir holding the architecture config.json
    ("c", False): "/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/stable_cascade_prior/prior",
    ("c", True):  "/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/stable_cascade_prior/prior_lite",
    ("b", False): "/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/stable_cascade/decoder",
    ("b", True):  "/home/gawkahn/projects/ai-lab/ai-base/models/hf-local/stable_cascade/decoder_lite",
}

_DTYPE_DEFAULTS = {
    # SAI model card recommends bf16 prior + fp16 decoder for a slight speed/VRAM
    # win, but the deprecated diffusers Cascade integration mishandles the
    # bf16→fp16 boundary when the two pipelines are composed manually (as we do
    # to support per-stage path swaps). Default to bf16 everywhere — overridable
    # in JSON for users who explicitly want the model-card recipe.
    "prior_dtype":   "bf16",
    "decoder_dtype": "bf16",
    "vae_dtype":     "fp32",   # Paella VAE always fp32
}

_PARAM_DEFAULTS = {
    "prior_steps":       20,
    "prior_cfg_scale":   4.0,
    "decoder_steps":     10,
    "decoder_cfg_scale": 0.0,
    "width":  1024,
    "height": 1024,
}

_KNOWN_KEYS = {
    # ── Configuration: paths + scaffolding ────────────────────────────
    "stage_c", "stage_b", "stage_a", "scaffolding_repo",
    "prior_dtype", "decoder_dtype", "vae_dtype",
    # ── Configuration: denoising params ───────────────────────────────
    "prior_steps", "prior_cfg_scale", "decoder_steps", "decoder_cfg_scale",
    "width", "height",
    # ── Optional replay fields (CLI overrides if present) ─────────────
    "prompt", "negative_prompt", "seed",
    # ── Sidecar runtime metadata — written by dispatch(), accepted on
    #    replay so a saved sidecar round-trips cleanly without warnings.
    "model_family", "config_source", "output_path",
    "iterate_batch_id", "run_index", "total_runs",
    "timestamp", "elapsed_seconds", "prior_seconds", "decoder_seconds",
}


# ════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _align_cascade_dim(x: int) -> int:
    """Round DOWN to the nearest multiple of 128 (matches _align_dim semantics)."""
    return (x // _ALIGN) * _ALIGN


def _resolve_torch_dtype(name: str):
    import torch
    name = (name or "bf16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name!r}")


def validate_config(raw: Dict[str, Any], source: str = "") -> Dict[str, Any]:
    """Apply defaults, type-check, align dimensions. Returns a fully populated
    config dict ready to drive a generation. Unknown keys are warned, not rejected.

    Required: stage_c, stage_b. Everything else has defaults.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"Cascade config {source!r}: expected JSON object, got {type(raw).__name__}")

    cfg = dict(raw)  # don't mutate caller

    # Required.
    for required in ("stage_c", "stage_b"):
        if not cfg.get(required):
            raise ValueError(f"Cascade config {source!r}: missing required field {required!r}")
        if not isinstance(cfg[required], str):
            raise TypeError(
                f"Cascade config {source!r}: {required!r} must be a string path, "
                f"got {type(cfg[required]).__name__}"
            )

    # Defaults — dtypes.
    for key, default in _DTYPE_DEFAULTS.items():
        cfg.setdefault(key, default)
        # Validate the dtype name eagerly.
        _resolve_torch_dtype(cfg[key])

    # Defaults — denoising params.
    for key, default in _PARAM_DEFAULTS.items():
        cfg.setdefault(key, default)

    # Type coercion / range checks for the numeric params.
    for int_key in ("prior_steps", "decoder_steps", "width", "height"):
        try:
            cfg[int_key] = int(cfg[int_key])
        except (TypeError, ValueError):
            raise TypeError(f"Cascade config {source!r}: {int_key!r} must be an integer")
        if cfg[int_key] <= 0:
            raise ValueError(f"Cascade config {source!r}: {int_key!r} must be positive (got {cfg[int_key]})")
    for float_key in ("prior_cfg_scale", "decoder_cfg_scale"):
        try:
            cfg[float_key] = float(cfg[float_key])
        except (TypeError, ValueError):
            raise TypeError(f"Cascade config {source!r}: {float_key!r} must be a number")
        if cfg[float_key] < 0:
            raise ValueError(f"Cascade config {source!r}: {float_key!r} must be >= 0 (got {cfg[float_key]})")

    # Default scaffolding repo.
    cfg.setdefault("scaffolding_repo", _DEFAULT_SCAFFOLDING_REPO)

    # Align dimensions (round down, warn).
    aligned_w = _align_cascade_dim(cfg["width"])
    aligned_h = _align_cascade_dim(cfg["height"])
    if aligned_w != cfg["width"] or aligned_h != cfg["height"]:
        _log(
            f"[comfyless] Dimensions aligned to {_ALIGN}px: "
            f"{cfg['width']}x{cfg['height']} -> {aligned_w}x{aligned_h}"
        )
        cfg["width"], cfg["height"] = aligned_w, aligned_h
    if aligned_w == 0 or aligned_h == 0:
        raise ValueError(
            f"Cascade config {source!r}: width/height after 128px alignment is 0; "
            f"raise width/height to at least 128"
        )

    # Warn on unknown keys (don't reject — forward-compatibility for future fields).
    # Keys starting with `_` are treated as comments per JSON convention.
    unknown = {k for k in (set(cfg) - _KNOWN_KEYS) if not k.startswith("_")}
    if unknown:
        _log(f"[comfyless] Cascade config {source!r}: unknown keys ignored: {sorted(unknown)}")

    return cfg


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a cascade config from a JSON file path."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cascade config not found: {path}")
    with open(path, "r") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Cascade config {path!r}: invalid JSON: {e}")
    return validate_config(raw, source=path)


# ════════════════════════════════════════════════════════════════════════
#  Pipeline assembly
# ════════════════════════════════════════════════════════════════════════

def _resolve_scaffolding(repo: str, allow_hf_download: bool) -> str:
    """Resolve scaffolding_repo to a local path. Goes through resolve_hf_path so
    HF repo IDs and local absolute paths are both accepted, matching the rest of
    comfyless."""
    from nodes.eric_diffusion_utils import resolve_hf_path
    return resolve_hf_path(repo, allow_download=allow_hf_download)


def _load_unet(path: str, torch_dtype, *, label: str, config_subfolder: str, scaffolding_dir: str):
    """Load a StableCascadeUNet from either a single safetensors file or a diffusers tree."""
    from diffusers import StableCascadeUNet
    if os.path.isdir(path):
        _log(f"[comfyless] {label}: loading diffusers tree from {path}")
        return StableCascadeUNet.from_pretrained(path, torch_dtype=torch_dtype)
    if os.path.isfile(path):
        # Need a config dir for from_single_file. The scaffolding repo's prior/
        # or decoder/ subfolder has it. Prefer using the prior tree for stage_c
        # (prior config) — but the SAI scaffolding repo only has decoder/, not
        # prior/. So for stage_c we look in stable_cascade_prior/prior or
        # stable_cascade_prior/prior_lite if available; otherwise scaffolding/decoder.
        config_path = os.path.join(scaffolding_dir, config_subfolder)
        if not os.path.isdir(config_path):
            # Cascade's main repo only has decoder/; for prior we try the sibling
            # -prior repo by convention.
            sibling = scaffolding_dir.rstrip("/") + "_prior"
            alt = os.path.join(sibling, config_subfolder)
            if os.path.isdir(alt):
                config_path = alt
            else:
                raise FileNotFoundError(
                    f"{label}: cannot find a {config_subfolder!r} config dir. "
                    f"Tried {os.path.join(scaffolding_dir, config_subfolder)!r} and {alt!r}. "
                    f"Either point stage_c/stage_b at a diffusers tree dir, or download "
                    f"the scaffolding repo with the appropriate sub-tree."
                )
        _log(f"[comfyless] {label}: loading single-file from {path}")
        _log(f"[comfyless] {label}: using config from {config_path}")
        return StableCascadeUNet.from_single_file(path, config=config_path, torch_dtype=torch_dtype)
    raise FileNotFoundError(f"{label}: path not found: {path}")


def _load_stage_a(path: Optional[str], torch_dtype, *, scaffolding_dir: str):
    """Load the Paella VQ-VAE (Stage A). Defaults to scaffolding/vqgan/."""
    from diffusers.pipelines.wuerstchen import PaellaVQModel  # transitive of StableCascadeDecoderPipeline
    target = path or os.path.join(scaffolding_dir, "vqgan")
    if os.path.isdir(target):
        _log(f"[comfyless] stage_a: loading diffusers tree from {target}")
        return PaellaVQModel.from_pretrained(target, torch_dtype=torch_dtype)
    if os.path.isfile(target):
        # Single-file Stage A is a plain state dict (122 F32 tensors) — we wrap it
        # with the architecture from the scaffolding tree.
        from safetensors.torch import load_file
        proto_dir = os.path.join(scaffolding_dir, "vqgan")
        if not os.path.isdir(proto_dir):
            raise FileNotFoundError(
                f"stage_a single-file load needs a vqgan/ tree alongside for the architecture; "
                f"none at {proto_dir!r}. Either point stage_a at a tree dir or download "
                f"the scaffolding repo's vqgan subfolder."
            )
        _log(f"[comfyless] stage_a: loading single-file from {target} (architecture from {proto_dir})")
        proto = PaellaVQModel.from_pretrained(proto_dir, torch_dtype=torch_dtype)
        proto.load_state_dict(load_file(target), strict=False)
        return proto
    raise FileNotFoundError(f"stage_a: path not found: {target}")


def build_pipelines(cfg: Dict[str, Any], device: str, allow_hf_download: bool):
    """Construct prior + decoder pipelines from a validated config.

    Returns (prior_pipeline, decoder_pipeline). Both moved to `device`.
    """
    from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers.schedulers import DDPMWuerstchenScheduler

    prior_dtype   = _resolve_torch_dtype(cfg["prior_dtype"])
    decoder_dtype = _resolve_torch_dtype(cfg["decoder_dtype"])
    vae_dtype     = _resolve_torch_dtype(cfg["vae_dtype"])

    scaffolding = _resolve_scaffolding(cfg["scaffolding_repo"], allow_hf_download)
    if not os.path.isdir(scaffolding):
        raise FileNotFoundError(
            f"scaffolding_repo {cfg['scaffolding_repo']!r} did not resolve to a local "
            f"directory ({scaffolding!r}); pass --allow-hf-download or pre-fetch the repo."
        )
    # HF Hub snapshots can be partial — e.g. an incidental prefetch of just
    # model_index.json leaves a snapshot dir with only that file symlinked.
    # `resolve_hf_path` returns the snapshot regardless. Verify the subdirs we
    # actually need are present, with a clear error if not.
    missing = [
        sub for sub in ("text_encoder", "tokenizer", "scheduler")
        if not os.path.isdir(os.path.join(scaffolding, sub))
    ]
    if missing:
        raise FileNotFoundError(
            f"scaffolding_repo {cfg['scaffolding_repo']!r} resolved to {scaffolding!r}, "
            f"but the following required subdirs are missing: {missing}. "
            f"This usually means an incomplete HF Hub fetch — only some files were "
            f"cached. Either: (a) point scaffolding_repo at a fully-mirrored local "
            f"directory, or (b) re-run with --allow-hf-download to fetch the full repo."
        )

    _log(f"[comfyless] cascade: scaffolding from {scaffolding}")

    # Shared scaffolding components.
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        os.path.join(scaffolding, "text_encoder"),
        torch_dtype=prior_dtype,  # text encoder runs at prior dtype (it conditions the prior)
    )
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(scaffolding, "tokenizer"))
    scheduler = DDPMWuerstchenScheduler.from_pretrained(os.path.join(scaffolding, "scheduler"))

    # Stage C (prior) and Stage B (decoder) UNets.
    prior_unet = _load_unet(
        cfg["stage_c"], prior_dtype,
        label="stage_c (prior)",
        config_subfolder="prior",
        scaffolding_dir=scaffolding,
    )
    decoder_unet = _load_unet(
        cfg["stage_b"], decoder_dtype,
        label="stage_b (decoder)",
        config_subfolder="decoder",
        scaffolding_dir=scaffolding,
    )

    # Stage A (VAE).
    vqgan = _load_stage_a(cfg.get("stage_a"), vae_dtype, scaffolding_dir=scaffolding)

    # Assemble the two pipelines manually — bypasses from_pretrained's repo-layout assumptions.
    prior_pipe = StableCascadePriorPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prior=prior_unet,
        scheduler=scheduler,
        feature_extractor=None,
        image_encoder=None,
    )
    # Decoder reuses the same tokenizer + a separate text_encoder cast to decoder_dtype
    # (model card recommends decoder runs in fp16; loading at decoder_dtype lets us match).
    decoder_text_encoder = CLIPTextModelWithProjection.from_pretrained(
        os.path.join(scaffolding, "text_encoder"),
        torch_dtype=decoder_dtype,
    )
    decoder_pipe = StableCascadeDecoderPipeline(
        decoder=decoder_unet,
        tokenizer=tokenizer,
        text_encoder=decoder_text_encoder,
        scheduler=scheduler,
        vqgan=vqgan,
        latent_dim_scale=10.67,
    )

    # `.to(device, dtype=...)` forces a uniform dtype across every parameter and
    # buffer in each pipeline. Without the explicit dtype, `from_single_file`'s
    # config-based architecture instantiation can leave a few bias tensors at
    # fp16 even when torch_dtype=bfloat16 was requested at load time, producing
    # the "Input type BFloat16 and bias type Half should be the same" error
    # mid-forward. Pipeline-level cast eliminates the cross-precision mix.
    # (vqgan stays fp32 inside the decoder pipeline because Cascade's
    # PaellaVQModel must operate in fp32 — we explicitly recast it back below.)
    prior_pipe   = prior_pipe.to(device, dtype=prior_dtype)
    decoder_pipe = decoder_pipe.to(device, dtype=decoder_dtype)
    decoder_pipe.vqgan = decoder_pipe.vqgan.to(device=device, dtype=vae_dtype)
    return prior_pipe, decoder_pipe


def dispose_pipelines(prior_pipe, decoder_pipe) -> None:
    """Tear down both pipelines and free GPU memory between configs."""
    try:
        prior_pipe.to("cpu")
        decoder_pipe.to("cpu")
    except Exception:
        pass
    del prior_pipe, decoder_pipe
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════
#  Generation
# ════════════════════════════════════════════════════════════════════════

def _resolve_seed(seed: Optional[int]) -> int:
    if seed is None or seed < 0:
        import torch
        return int(torch.randint(0, 2**32 - 1, (1,)).item())
    return int(seed)


def run_one(
    prior_pipe, decoder_pipe,
    cfg: Dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    seed: int,
    device: str,
) -> Tuple[Any, Dict[str, Any]]:
    """Run one (prior → decoder) pass. Returns (PIL.Image, runtime_meta)."""
    import torch
    generator = torch.Generator(device=device).manual_seed(seed)

    t0 = time.time()
    prior_out = prior_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=cfg["height"],
        width=cfg["width"],
        num_inference_steps=cfg["prior_steps"],
        guidance_scale=cfg["prior_cfg_scale"],
        generator=generator,
    )
    prior_seconds = time.time() - t0

    # Cast image_embeddings to the decoder's actual dtype. Prior may run at a
    # different dtype than decoder (e.g. SAI model card: bf16 prior + fp16
    # decoder); the embeddings tensor flows raw between them, so without an
    # explicit cast at the boundary the decoder's first Linear sees mismatched
    # input/bias dtypes ("Input type BFloat16 and bias type Half should be
    # the same"). Probe the decoder for whatever dtype it ended up at.
    decoder_actual_dtype = next(decoder_pipe.decoder.parameters()).dtype
    image_embeddings = prior_out.image_embeddings.to(decoder_actual_dtype)

    t0 = time.time()
    decoder_out = decoder_pipe(
        image_embeddings=image_embeddings,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=cfg["decoder_steps"],
        guidance_scale=cfg["decoder_cfg_scale"],
        output_type="pil",
        generator=generator,
    )
    decoder_seconds = time.time() - t0

    pil = decoder_out.images[0]
    return pil, {
        "prior_seconds":   round(prior_seconds, 2),
        "decoder_seconds": round(decoder_seconds, 2),
    }


# ════════════════════════════════════════════════════════════════════════
#  Output path resolution + sidecar write
# ════════════════════════════════════════════════════════════════════════

def _resolve_output_path(base: str, total_iterations: int, run_index: int) -> str:
    """Pick the output filename for one run.

    Rules:
    - `base` is a directory: always emit `<base>/cascade.png` (single iter) or
      `<base>/cascade_NNNN.png` (multi iter).
    - `base` is a file path AND total_iterations == 1: write exactly to `base`.
    - `base` is a file path AND total_iterations > 1: treat `base` as a stem;
      emit `<dir>/<stem>_NNNN<.ext>` with a 4-digit run index.

    Single iter to a directory base is symmetric with multi iter (was a gap
    that would have crashed at PIL save).
    """
    if os.path.isdir(base):
        Path(base).mkdir(parents=True, exist_ok=True)
        if total_iterations == 1:
            return os.path.join(base, "cascade.png")
        return os.path.join(base, f"cascade_{run_index + 1:04d}.png")

    if total_iterations == 1:
        Path(base).parent.mkdir(parents=True, exist_ok=True)
        return base

    p = Path(base)
    p.parent.mkdir(parents=True, exist_ok=True)
    stem = p.stem
    suffix = p.suffix or ".png"
    return str(p.parent / f"{stem}_{run_index + 1:04d}{suffix}")


def _write_sidecar(image_path: str, sidecar: Dict[str, Any]) -> str:
    """Write a JSON sidecar next to the image. Returns the sidecar path."""
    sidecar_path = os.path.splitext(image_path)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    return sidecar_path


def _save_with_metadata(pil_image, path: str, metadata: Dict[str, Any]) -> None:
    """Save PNG with a comfyless tEXt chunk. Mirrors generate.py's helper."""
    from PIL.PngImagePlugin import PngInfo
    pnginfo = PngInfo()
    pnginfo.add_text("comfyless", json.dumps(metadata, default=str))
    pil_image.save(path, pnginfo=pnginfo)


# ════════════════════════════════════════════════════════════════════════
#  Top-level dispatch
# ════════════════════════════════════════════════════════════════════════

def _reject_unsupported_flags(args: argparse.Namespace) -> Optional[str]:
    """Cascade dispatch ignores most family-specific knobs. Surface as errors so
    users don't think they've taken effect.

    `--limit` is supported (slices the post-Cartesian plan). `--max-iterations`
    is honored (hard cap). Everything else family-specific is rejected.
    """
    bad = []
    if args.lora:
        bad.append("--lora (Cascade LoRA support is out of scope per ADR-010)")
    if args.transformer or args.vae or args.te1 or args.te2 or args.vae_from_transformer:
        bad.append("--transformer / --vae / --te1 / --te2 / --vae-from-transformer "
                   "(Cascade uses per-stage paths in the JSON config — see docs/comfyless-stable-cascade.md)")
    if args.true_cfg is not None:
        bad.append("--true-cfg (Cascade has prior_cfg_scale + decoder_cfg_scale in JSON)")
    if args.cfg is not None:
        bad.append("--cfg (Cascade has prior_cfg_scale + decoder_cfg_scale in JSON)")
    if args.steps is not None:
        bad.append("--steps (Cascade has prior_steps + decoder_steps in JSON)")
    if args.width is not None or args.height is not None:
        bad.append("--width / --height (Cascade dimensions are JSON-owned)")
    if args.sampler is not None or args.schedule is not None:
        bad.append("--sampler / --schedule (Cascade scheduler is fixed: DDPMWuerstchenScheduler)")
    # --iterate: only `prompt` and `seed` axes are meaningful for cascade. Topology
    # iteration uses positional configs instead.
    bad_axes = [param for param, _ in (args.iterate or []) if param not in ("prompt", "seed")]
    if bad_axes:
        bad.append(
            f"--iterate {bad_axes[0]} (Cascade --iterate supports only `prompt` and `seed`; "
            f"topology variations use positional configs: --model stablecascade c1.json c2.json ...)"
        )
    if args.params is not None or args.override:
        bad.append("--params / --override (Cascade replays via positional config paths; "
                   "edit a saved sidecar JSON and pass it as a positional arg instead)")
    if args.savepath is not None:
        bad.append("--savepath (Cascade output paths use --output directly; auto-suffix on multi-run)")
    if args.max_seq_len is not None:
        bad.append("--max-seq-len (not applicable to Cascade's CLIP-G text encoder)")
    if args.attention_slicing or args.sequential_offload or args.offload_vae:
        bad.append("--attention-slicing / --sequential-offload / --offload-vae "
                   "(VRAM optimizations not yet wired into the cascade dispatch)")
    # --precision: argparse default is "bf16"; reject any non-default since the JSON
    # owns per-stage dtypes. This catches users who pass --precision fp16 expecting
    # it to apply.
    if args.precision is not None and args.precision != "bf16":
        bad.append(f"--precision={args.precision!r} (Cascade dtypes live in JSON: "
                   f"prior_dtype / decoder_dtype / vae_dtype)")
    if bad:
        return "Cascade-incompatible flags:\n  - " + "\n  - ".join(bad)
    return None


def _interactive_confirm(total: int, auto_yes: bool) -> bool:
    """Re-uses the same threshold as the existing iterate confirmation (5)."""
    if auto_yes or total <= 5:
        return True
    sys.stderr.write(f"About to run {total} cascade generations. Continue? [y/N] ")
    sys.stderr.flush()
    try:
        line = sys.stdin.readline().strip().lower()
    except EOFError:
        return False
    return line in ("y", "yes")


def _emit_json_error(error_type: str, message: str) -> int:
    """Structured JSON error to stdout for --json mode. Returns exit code 1."""
    json.dump({
        "status": "error",
        "error": message,
        "error_type": error_type,
    }, sys.stdout, indent=2)
    return 1


def _effective_seed_for_batch(raw_seed: int, batch_index: int) -> int:
    """For multi-batch runs with an explicit seed: batch 0 uses seed, subsequent
    batches use seed+batch_index. Random seed (-1) stays random per call.
    """
    if raw_seed < 0:
        return _resolve_seed(raw_seed)
    if batch_index == 0:
        return _resolve_seed(raw_seed)
    return _resolve_seed(raw_seed + batch_index)


def dispatch(args: argparse.Namespace, config_paths: List[str]) -> int:
    """Top-level cascade dispatch. Returns process exit code.

    Args:
        args: full argparse Namespace from comfyless.generate's parser.
        config_paths: positional config paths after the `stablecascade` sentinel.
    """
    # ── --json mode: emit structured JSON for any rejection. Comes FIRST so
    # the contract surface stays clean for tools/agents that parse stdout.
    if args.json:
        return _emit_json_error(
            "CascadeNotSupportedInJsonMode",
            "Stable Cascade is not yet wired into the --json bridge contract",
        )

    # ── --serve / --unload: cascade does not support persistent server mode.
    if args.serve or args.unload:
        print("Error: Stable Cascade is not supported in --serve / --unload modes (see ADR-010)",
              file=sys.stderr)
        return 2

    # ── Reject conflicting CLI flags ──────────────────────────────────
    err = _reject_unsupported_flags(args)
    if err is not None:
        print(f"Error: {err}", file=sys.stderr)
        return 2

    # ── Validate config files ─────────────────────────────────────────
    if not config_paths:
        print("Error: --model stablecascade requires at least one config JSON path",
              file=sys.stderr)
        return 2
    configs: List[Tuple[str, Dict[str, Any]]] = []
    for p in config_paths:
        try:
            cfg = load_config(p)
        except (FileNotFoundError, ValueError, TypeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        configs.append((p, cfg))

    # ── Read --iterate axes (cascade-supported subset: prompt, seed) ──
    iterate_lists: Dict[str, List[Any]] = {}
    for param, file_path in (args.iterate or []):
        if not os.path.isfile(file_path):
            print(f"Error: --iterate {param}: file not found: {file_path}", file=sys.stderr)
            return 2
        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: --iterate {param}: {file_path!r}: invalid JSON: {e}", file=sys.stderr)
            return 2
        if not isinstance(data, list) or not data:
            print(f"Error: --iterate {param}: {file_path!r}: must be a non-empty JSON list",
                  file=sys.stderr)
            return 2
        iterate_lists[param] = data

    # ── Build flat run plan ───────────────────────────────────────────
    # Each plan entry = (cfg_path, cfg_dict, batch_index, prompt_override, seed_override).
    # _override slots are None when no --iterate axis covers that field, in which
    # case the per-run seed/prompt resolution falls back to CLI > sidecar > random.
    # Plan order groups by cfg first so pipelines load once per cfg, then expands
    # batch × prompt × seed within each cfg.
    batch = max(int(args.batch or 1), 1)
    prompts_axis = iterate_lists.get("prompt", [None])  # [None] = no override
    seeds_axis   = iterate_lists.get("seed",   [None])
    plan: List[Tuple[str, Dict[str, Any], int, Any, Any]] = []
    for cfg_path, cfg in configs:
        for batch_index in range(batch):
            for prompt_iter in prompts_axis:
                for seed_iter in seeds_axis:
                    plan.append((cfg_path, cfg, batch_index, prompt_iter, seed_iter))

    # Apply --limit: silent truncation (ceiling, not requirement). Matches ADR-008
    # and ADR-010 §2 semantics.
    if args.limit is not None and len(plan) > args.limit:
        _log(f"[comfyless] --limit {args.limit}: truncating plan from {len(plan)} to {args.limit}")
        plan = plan[: args.limit]

    total = len(plan)

    # Honor --max-iterations as a hard fail-closed safety cap (matches ADR-008).
    if args.max_iterations is not None and total > args.max_iterations:
        print(f"Error: planned {total} runs exceeds --max-iterations {args.max_iterations}. "
              f"Either raise --max-iterations or reduce configs/batch/limit.",
              file=sys.stderr)
        return 2

    if total == 0:
        print("Error: nothing to do (--limit 0 or empty plan)", file=sys.stderr)
        return 2

    if not _interactive_confirm(total, args.yes):
        _log("[comfyless] cancelled by user")
        return 1

    # ── CLI shared inputs (CLI wins; JSON is a replay fallback) ───────
    cli_prompt   = args.prompt
    cli_neg      = args.negative_prompt or ""
    cli_seed_raw = args.seed

    # ── Run loop ──────────────────────────────────────────────────────
    iterate_batch_id = uuid.uuid4().hex
    fail_count = 0
    current_cfg_path: Optional[str] = None
    prior_pipe = None
    decoder_pipe = None

    for run_index, (cfg_path, cfg, batch_index, prompt_iter, seed_iter) in enumerate(plan):
        # Prompt resolution: --iterate prompt wins > CLI --prompt > JSON `prompt`.
        if prompt_iter is not None:
            prompt = prompt_iter
        elif cli_prompt is not None:
            prompt = cli_prompt
        else:
            prompt = cfg.get("prompt")
        if not prompt:
            print(f"Error: no prompt provided (--prompt, --iterate prompt, "
                  f"or {cfg_path!r}'s 'prompt' field)",
                  file=sys.stderr)
            fail_count += 1
            continue
        negative = cli_neg if cli_neg else cfg.get("negative_prompt", "")

        # Build pipelines on the first run for this config; dispose when moving to next.
        if cfg_path != current_cfg_path:
            if prior_pipe is not None:
                dispose_pipelines(prior_pipe, decoder_pipe)
                prior_pipe = decoder_pipe = None
            _log(f"[comfyless] cascade config: {cfg_path}")
            try:
                prior_pipe, decoder_pipe = build_pipelines(
                    cfg, device=args.device, allow_hf_download=args.allow_hf_download,
                )
                current_cfg_path = cfg_path
            except Exception as e:
                # Build-fail: count it and skip remaining batch slots for this config.
                # We advance run_index normally; the next iteration's cfg_path comparison
                # decides whether to attempt rebuild (next config) or skip-equivalent
                # (more batch slots of the same broken config).
                print(f"Error building cascade pipelines for {cfg_path!r}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                fail_count += 1
                # Mark current_cfg_path so subsequent same-config batch slots see "already
                # tried, skip" without re-attempting the build.
                current_cfg_path = cfg_path
                prior_pipe = decoder_pipe = None
                continue

        # Pipelines may be None if the build above failed for this config.
        if prior_pipe is None or decoder_pipe is None:
            continue

        # Seed resolution: --iterate seed wins > CLI --seed > JSON `seed` > random.
        # When --iterate seed is supplying the value, batch_index does NOT add to it
        # (the user explicitly listed each seed they wanted).
        if seed_iter is not None:
            try:
                eff_seed = _resolve_seed(int(seed_iter))
            except (TypeError, ValueError):
                print(f"Error: --iterate seed value {seed_iter!r} is not an int", file=sys.stderr)
                fail_count += 1
                continue
        else:
            if cli_seed_raw is not None:
                raw_seed = cli_seed_raw
            elif "seed" in cfg:
                raw_seed = cfg["seed"]
            else:
                raw_seed = -1
            eff_seed = _effective_seed_for_batch(raw_seed, batch_index)

        out_path = _resolve_output_path(args.output, total, run_index)
        _log(f"[comfyless] cascade run {run_index + 1}/{total} : seed={eff_seed} → {out_path}")

        try:
            t0 = time.time()
            pil, runtime = run_one(
                prior_pipe, decoder_pipe, cfg,
                prompt=prompt,
                negative_prompt=negative,
                seed=eff_seed,
                device=args.device,
            )
            elapsed = time.time() - t0
        except Exception as e:
            print(f"Error during cascade generation (run {run_index+1}/{total}): "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            fail_count += 1
            continue

        # Sidecar = full cascade config + runtime metadata. Round-trips through
        # load_config without warnings (runtime keys are in _KNOWN_KEYS).
        sidecar = dict(cfg)
        sidecar.update({
            "model_family": "stable-cascade",
            "config_source": cfg_path,
            "prompt": prompt,
            "negative_prompt": negative,
            "seed": eff_seed,
            "output_path": out_path,
            "elapsed_seconds":  round(elapsed, 2),
            "prior_seconds":    runtime["prior_seconds"],
            "decoder_seconds":  runtime["decoder_seconds"],
            "iterate_batch_id": iterate_batch_id,
            "run_index":        run_index,
            "total_runs":       total,
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        _save_with_metadata(pil, out_path, sidecar)
        sidecar_path = _write_sidecar(out_path, sidecar)
        _log(f"[comfyless] saved {out_path}  (sidecar: {sidecar_path})")

    # Final dispose.
    if prior_pipe is not None:
        dispose_pipelines(prior_pipe, decoder_pipe)

    if fail_count:
        return 3
    return 0
