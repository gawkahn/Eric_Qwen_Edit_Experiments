"""comfyless video — single-segment Wan 2.2 video generation (ADR-033, slice V1).

Standalone dispatch module in the cascade.py mold: NOT part of the
GEN_PIPELINE image family system. Renders one ~5 s segment from a start
keyframe (optionally anchored to an end keyframe via Wan's `last_image`
conditioning) and writes an H.264 mp4 plus a replayable JSON sidecar.

Usage:
    python -m comfyless.video --keyframe-start A.png [--keyframe-end B.png] \
        --prompt "..." --output seg.mp4

Defaults follow ADR-033: Lightning 4-step distill LoRAs on both MoE experts
with cfg 1.0 (override with --no-lightning for the 40-step quality tier).
Multi-segment plan.json chaining is slice V2, not here.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

SIDECAR_SCHEMA = "comfyless-video/1"

DEFAULT_MODEL_NAME = "Wan2.2-I2V-A14B-Diffusers"
DEFAULT_LIGHTNING_SUBDIR = os.path.join(
    "Wan2.2-Lightning", "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1")
LIGHTNING_HIGH = "high_noise_model.safetensors"
LIGHTNING_LOW = "low_noise_model.safetensors"

# Wan-recommended negative prompt (same as ComfyUI workflows use).
WAN_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# (steps, cfg) defaults per mode; explicit flags override (ADR-009 sentinel
# pattern: argparse defaults are None, resolved here).
MODE_DEFAULTS = {"lightning": (4, 1.0), "base": (40, 3.5)}


class VideoParamError(ValueError):
    """Parameter validation failure — message is user-facing."""


def _log(msg: str) -> None:
    print(f"[comfyless.video] {msg}", file=sys.stderr)


def align_dim(x: int) -> int:
    """Align a pixel dimension down to the /16 grid Wan requires."""
    aligned = (int(x) // 16) * 16
    if aligned < 16:
        raise VideoParamError(f"dimension {x} is below the 16 px minimum")
    return aligned


def validate_frames(n: int) -> int:
    """Wan requires (frames - 1) % 4 == 0 (temporal VAE stride)."""
    if n < 5:
        raise VideoParamError(f"--frames {n}: minimum is 5")
    if (n - 1) % 4 != 0:
        lower = ((n - 1) // 4) * 4 + 1
        raise VideoParamError(
            f"--frames {n}: Wan needs (frames-1) divisible by 4 — "
            f"nearest valid are {lower} and {lower + 4}")
    return n


def resolve_mode_defaults(lightning: bool, steps: Optional[int],
                          cfg: Optional[float]) -> tuple:
    """Fill steps/cfg from the mode unless explicitly overridden."""
    d_steps, d_cfg = MODE_DEFAULTS["lightning" if lightning else "base"]
    return (steps if steps is not None else d_steps,
            cfg if cfg is not None else d_cfg)


def load_keyframe(path: str, label: str):
    """Open a keyframe image; loud failure on missing/undecodable files."""
    from PIL import Image
    if not os.path.isfile(path):
        raise VideoParamError(f"{label} not found: {path}")
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:
        raise VideoParamError(f"{label} is not a readable image: {path} ({exc})")
    return img.convert("RGB")


def prepare_keyframes(start_path: str, end_path: Optional[str]):
    """Load keyframes; dims come from the start frame aligned to /16.

    An end keyframe with different dimensions is resized to match with a loud
    warning (warn-don't-block).
    """
    from PIL import Image
    start = load_keyframe(start_path, "--keyframe-start")
    w, h = align_dim(start.width), align_dim(start.height)
    if (w, h) != start.size:
        _log(f"WARNING: start keyframe {start.size} aligned to {w}x{h} (/16 grid)")
        start = start.resize((w, h), Image.Resampling.LANCZOS)
    end = None
    if end_path is not None:
        end = load_keyframe(end_path, "--keyframe-end")
        if end.size != (w, h):
            _log(f"WARNING: end keyframe {end.size} resized to {w}x{h} "
                 f"to match the start keyframe")
            end = end.resize((w, h), Image.Resampling.LANCZOS)
    return start, end, w, h


def encode_mp4(frames, path: str, fps: int, crf: int = 16) -> None:
    """Encode a list of PIL images (or HxWx3 uint8 arrays) to H.264 mp4."""
    import av
    import numpy as np
    if not frames:
        raise VideoParamError("no frames to encode")
    first = np.asarray(frames[0])
    height, width = first.shape[0], first.shape[1]
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        for frame in frames:
            arr = np.asarray(frame, dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(vf):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def build_sidecar(args: argparse.Namespace, *, model: str, lightning_dir:
                  Optional[str], steps: int, cfg: float, seed: int, width: int,
                  height: int, gen_time_s: float) -> Dict[str, Any]:
    return {
        "schema": SIDECAR_SCHEMA,
        "model": os.path.abspath(model),
        "lightning": lightning_dir is not None,
        "lightning_dir": (os.path.abspath(lightning_dir)
                          if lightning_dir is not None else None),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "keyframe_start": os.path.abspath(args.keyframe_start),
        "keyframe_end": (os.path.abspath(args.keyframe_end)
                         if args.keyframe_end else None),
        "seed": seed,
        "width": width,
        "height": height,
        "frames": args.frames,
        "fps": args.fps,
        "steps": steps,
        "cfg": cfg,
        "crf": args.crf,
        "device": args.device,
        "gen_time_s": round(gen_time_s, 1),
    }


def write_sidecar(video_path: str, sidecar: Dict[str, Any]) -> str:
    sidecar_path = os.path.splitext(video_path)[0] + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    return sidecar_path


def _resolve_model_paths(args: argparse.Namespace) -> tuple:
    """Resolve model dir + lightning LoRA dir from flags / model-base."""
    base = args.model_base or os.environ.get("COMFYLESS_MODEL_BASE")
    model = args.model
    if model is None:
        if base is None:
            raise VideoParamError(
                "--model not given and no --model-base / COMFYLESS_MODEL_BASE "
                f"to resolve the default ({DEFAULT_MODEL_NAME}) against")
        model = os.path.join(base, DEFAULT_MODEL_NAME)
    if not os.path.isdir(model):
        raise VideoParamError(f"model directory not found: {model}")
    lightning_dir = None
    if not args.no_lightning:
        lightning_dir = args.lightning_dir
        if lightning_dir is None:
            if base is None:
                raise VideoParamError(
                    "lightning mode needs --lightning-dir or --model-base / "
                    "COMFYLESS_MODEL_BASE (or pass --no-lightning)")
            lightning_dir = os.path.join(base, DEFAULT_LIGHTNING_SUBDIR)
        for fname in (LIGHTNING_HIGH, LIGHTNING_LOW):
            if not os.path.isfile(os.path.join(lightning_dir, fname)):
                raise VideoParamError(
                    f"lightning LoRA missing: {os.path.join(lightning_dir, fname)}")
    return model, lightning_dir


def run(args: argparse.Namespace) -> int:
    # Pure validation first — loud, fast failure before the heavy torch import.
    validate_frames(args.frames)
    model, lightning_dir = _resolve_model_paths(args)
    steps, cfg = resolve_mode_defaults(lightning_dir is not None,
                                       args.steps, args.cfg)
    start, end, width, height = prepare_keyframes(
        args.keyframe_start, args.keyframe_end)

    import torch
    from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline

    seed = args.seed
    if seed < 0:
        seed = torch.seed() % (2 ** 31)

    t0 = time.time()
    pipe = WanImageToVideoPipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, local_files_only=True)
    if lightning_dir is not None:
        pipe.load_lora_weights(os.path.join(lightning_dir, LIGHTNING_HIGH),
                               adapter_name="lightning_high")
        pipe.load_lora_weights(os.path.join(lightning_dir, LIGHTNING_LOW),
                               adapter_name="lightning_low",
                               load_into_transformer_2=True)
    if args.offload:
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        pipe.to(args.device)
    _log(f"model ready in {time.time() - t0:.1f}s — "
         f"{'lightning' if lightning_dir else 'base'} mode, "
         f"{width}x{height}, frames={args.frames}, steps={steps}, cfg={cfg}, "
         f"seed={seed}, anchored={'yes' if end is not None else 'no'}")

    call_kwargs: Dict[str, Any] = dict(
        image=start, prompt=args.prompt, negative_prompt=args.negative_prompt,
        height=height, width=width, num_frames=args.frames,
        num_inference_steps=steps, guidance_scale=cfg, guidance_scale_2=cfg,
        generator=torch.Generator(args.device).manual_seed(seed))
    if end is not None:
        call_kwargs["last_image"] = end

    t0 = time.time()
    result = pipe(**call_kwargs)
    gen_time = time.time() - t0

    import numpy as np
    raw = getattr(result, "frames")[0]  # union-typed pipeline output; return_dict path
    frames = [np.clip(np.asarray(f) * 255.0, 0, 255).astype("uint8")
              for f in raw]
    encode_mp4(frames, args.output, fps=args.fps, crf=args.crf)
    sidecar = build_sidecar(args, model=model, lightning_dir=lightning_dir,
                            steps=steps, cfg=cfg, seed=seed, width=width,
                            height=height, gen_time_s=gen_time)
    sidecar_path = write_sidecar(args.output, sidecar)
    _log(f"generated {len(frames)} frames in {gen_time:.1f}s")
    _log(f"saved: {args.output} (+ {sidecar_path})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comfyless.video",
        description="Single-segment Wan 2.2 video generation (ADR-033).")
    p.add_argument("--keyframe-start", required=True,
                   help="Start keyframe image (dims are taken from it, /16-aligned)")
    p.add_argument("--keyframe-end",
                   help="Optional end keyframe — enables last_image anchoring")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=WAN_NEGATIVE)
    p.add_argument("--model",
                   help=f"Wan 2.2 I2V diffusers dir (default: <model-base>/{DEFAULT_MODEL_NAME})")
    p.add_argument("--model-base",
                   help="Base dir for default model/LoRA resolution "
                        "(env: COMFYLESS_MODEL_BASE)")
    p.add_argument("--no-lightning", action="store_true",
                   help="Quality tier: full 40-step CFG run instead of the "
                        "4-step Lightning distill")
    p.add_argument("--lightning-dir",
                   help="Dir holding the Lightning high/low-noise LoRAs")
    p.add_argument("--frames", type=int, default=81,
                   help="Frame count; (frames-1) must divide by 4 (default 81 = 5 s)")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--steps", type=int, default=None,
                   help="Override mode default (lightning 4 / base 40)")
    p.add_argument("--cfg", type=float, default=None,
                   help="Override mode default (lightning 1.0 / base 3.5)")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--crf", type=int, default=16, help="x264 quality (default 16)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--offload", action="store_true",
                   help="CPU-offload components (peak VRAM ~30-40 GB instead "
                        "of ~80 GB; small per-step transfer cost)")
    p.add_argument("--output", default="video_segment.mp4")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except VideoParamError as exc:
        _log(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
