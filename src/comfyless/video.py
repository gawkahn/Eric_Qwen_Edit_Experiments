"""comfyless video — Wan 2.2 video generation (ADR-033, slices V1 + V2).

Standalone dispatch module in the cascade.py mold: NOT part of the
GEN_PIPELINE image family system.

Single-segment mode (slice V1): renders one ~5 s segment from a start
keyframe (optionally anchored to an end keyframe via Wan's `last_image`
conditioning) and writes an H.264 mp4 plus a replayable JSON sidecar.

    python -m comfyless.video --keyframe-start A.png [--keyframe-end B.png] \
        --prompt "..." --output seg.mp4

Plan mode (slice V2): renders K keyframe-anchored segments from a
`comfyless-video-plan/1` JSON plan, keeps per-segment mp4s + sidecars in
`<output-stem>_segments/`, and stitches one master mp4 with boundary-frame
dedup and measure-then-apply color correction at continuous joins. Segments
shard across GPUs via per-device worker subprocesses (`--devices`).

    python -m comfyless.video --plan plan.json --output movie.mp4

In plan mode, creative params (prompts, keyframes, seeds, frames, fps,
steps, cfg, lightning tier) come from the plan; the CLI carries only
operational flags (model paths, devices, offload, crf, output).

Defaults follow ADR-033: Lightning 4-step distill LoRAs on both MoE experts
with cfg 1.0 (override with --no-lightning for the 40-step quality tier).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

# ── plan.json (slice V2) ─────────────────────────────────────────────────
# Machine-boundary format (ADR-012 patterns): byte cap, unknown-key
# rejection at every level, strict types/ranges — all before any torch
# import. This is the contract the future LLM planner (slice 6) must emit.
PLAN_SCHEMA = "comfyless-video-plan/1"
PLAN_MAX_BYTES = 1_000_000
MAX_SEGMENTS = 200
MAX_PROMPT_CHARS = 5000

_PLAN_TOP_KEYS = {"schema", "defaults", "lightning", "segments"}
_PLAN_DEFAULT_KEYS = {"frames", "fps", "steps", "cfg", "negative_prompt",
                      "seed"}
_PLAN_SEGMENT_KEYS = {"prompt", "keyframe_start", "keyframe_end", "seed",
                      "frames", "steps", "cfg", "negative_prompt"}


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


class Mp4Writer:
    """Incremental H.264 mp4 encoder — frames stream in, one at a time.

    Lets the stitch path hold at most one segment in memory instead of the
    whole master.
    """

    def __init__(self, path: str, fps: int, crf: int = 16):
        import av
        self._av = av
        self.frame_count = 0
        # explicit format: the stitch path writes to a ".tmp" name first
        self._container = av.open(path, mode="w", format="mp4")
        self._stream = None
        self._fps = fps
        self._crf = crf
        self._closed = False

    def add(self, frame) -> None:
        import numpy as np
        arr = np.asarray(frame, dtype=np.uint8)
        if self._stream is None:
            stream = self._container.add_stream("libx264", rate=self._fps)
            stream.width = arr.shape[1]
            stream.height = arr.shape[0]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(self._crf), "preset": "medium"}
            self._stream = stream
        vf = self._av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in self._stream.encode(vf):
            self._container.mux(packet)
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stream is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
        self._container.close()


def encode_mp4(frames, path: str, fps: int, crf: int = 16) -> None:
    """Encode a list of PIL images (or HxWx3 uint8 arrays) to H.264 mp4."""
    if not frames:
        raise VideoParamError("no frames to encode")
    writer = Mp4Writer(path, fps=fps, crf=crf)
    try:
        for frame in frames:
            writer.add(frame)
    finally:
        writer.close()


def decode_frames(path: str) -> list:
    """Decode a segment mp4 back to a list of HxWx3 uint8 rgb24 arrays."""
    import av
    if not os.path.isfile(path):
        raise VideoParamError(f"segment file not found: {path}")
    with av.open(path) as container:
        stream = container.streams.video[0]
        return [f.to_ndarray(format="rgb24") for f in container.decode(stream)]


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


# ── plan.json ingestion (slice V2, ADR-012 machine-boundary style) ───────

def _require_int(val, name: str, lo: int, hi: int) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise VideoParamError(f"{name}: expected an integer, got {val!r}")
    if not lo <= val <= hi:
        raise VideoParamError(f"{name}: {val} outside valid range {lo}..{hi}")
    return val


def _require_float(val, name: str, lo: float, hi: float) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise VideoParamError(f"{name}: expected a number, got {val!r}")
    val = float(val)
    if not lo <= val <= hi:
        raise VideoParamError(f"{name}: {val} outside valid range {lo}..{hi}")
    return val


def _require_str(val, name: str, max_chars: int,
                 allow_empty: bool = False) -> str:
    if not isinstance(val, str):
        raise VideoParamError(f"{name}: expected a string, got {val!r}")
    if not allow_empty and not val.strip():
        raise VideoParamError(f"{name}: must not be empty")
    if len(val) > max_chars:
        raise VideoParamError(
            f"{name}: {len(val)} chars exceeds the {max_chars} cap")
    return val


def _reject_unknown_keys(d: dict, allowed: set, where: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise VideoParamError(
            f"{where}: unknown key(s) {', '.join(unknown)} — "
            f"allowed: {', '.join(sorted(allowed))}")


def load_plan(path: str) -> Dict[str, Any]:
    """Parse + validate a comfyless-video-plan/1 file. Fail-closed, pre-torch.

    Returns a normalized dict: {lightning, fps, width, height, segments:
    [{prompt, negative_prompt, seed, frames, steps, cfg, keyframe_start,
    keyframe_end, continuous}]} with keyframe paths absolute (resolved
    against the plan file's directory) and `continuous` marking whether
    segment i's start keyframe is segment i-1's end keyframe (realpath
    equality; segment 0 is always False).
    """
    if not os.path.isfile(path):
        raise VideoParamError(f"plan not found: {path}")
    size = os.path.getsize(path)
    if size > PLAN_MAX_BYTES:
        raise VideoParamError(
            f"plan file is {size} bytes — cap is {PLAN_MAX_BYTES}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except (ValueError, UnicodeDecodeError) as exc:
            raise VideoParamError(f"plan is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        raise VideoParamError("plan: top level must be a JSON object")
    if raw.get("schema") != PLAN_SCHEMA:
        raise VideoParamError(
            f"plan schema must be {PLAN_SCHEMA!r}, got {raw.get('schema')!r}")
    _reject_unknown_keys(raw, _PLAN_TOP_KEYS, "plan")

    defaults = raw.get("defaults", {})
    if not isinstance(defaults, dict):
        raise VideoParamError("plan defaults: must be a JSON object")
    _reject_unknown_keys(defaults, _PLAN_DEFAULT_KEYS, "plan defaults")

    lightning = raw.get("lightning", True)
    if not isinstance(lightning, bool):
        raise VideoParamError(
            f"plan lightning: expected true/false, got {lightning!r}")

    fps = _require_int(defaults.get("fps", 16), "plan defaults.fps", 1, 60)
    d_frames = validate_frames(
        _require_int(defaults.get("frames", 81), "plan defaults.frames",
                     5, 100_000))
    d_seed = _require_int(defaults.get("seed", -1), "plan defaults.seed",
                          -1, 2 ** 63 - 1)
    d_steps = defaults.get("steps")
    if d_steps is not None:
        d_steps = _require_int(d_steps, "plan defaults.steps", 1, 100)
    d_cfg = defaults.get("cfg")
    if d_cfg is not None:
        d_cfg = _require_float(d_cfg, "plan defaults.cfg", 0.0, 20.0)
    d_neg = defaults.get("negative_prompt")
    if d_neg is not None:
        d_neg = _require_str(d_neg, "plan defaults.negative_prompt",
                             MAX_PROMPT_CHARS, allow_empty=True)

    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments:
        raise VideoParamError("plan segments: must be a non-empty list")
    if len(segments) > MAX_SEGMENTS:
        raise VideoParamError(
            f"plan has {len(segments)} segments — cap is {MAX_SEGMENTS}")

    plan_dir = os.path.dirname(os.path.abspath(path))
    norm: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        where = f"plan segment {i}"
        if not isinstance(seg, dict):
            raise VideoParamError(f"{where}: must be a JSON object")
        _reject_unknown_keys(seg, _PLAN_SEGMENT_KEYS, where)
        for req in ("prompt", "keyframe_start", "keyframe_end"):
            if req not in seg:
                raise VideoParamError(f"{where}: missing required key {req!r}")
        prompt = _require_str(seg["prompt"], f"{where}.prompt",
                              MAX_PROMPT_CHARS)
        kf_paths = {}
        for k in ("keyframe_start", "keyframe_end"):
            p = _require_str(seg[k], f"{where}.{k}", 4096)
            if not os.path.isabs(p):
                p = os.path.join(plan_dir, p)
            kf_paths[k] = os.path.abspath(p)
            # existence + decodability, loud per-segment (pre-GPU)
            load_keyframe(kf_paths[k], f"{where}.{k}")
        frames = seg.get("frames")
        frames = (d_frames if frames is None else validate_frames(
            _require_int(frames, f"{where}.frames", 5, 100_000)))
        seed = seg.get("seed")
        seed = (d_seed if seed is None else _require_int(
            seed, f"{where}.seed", -1, 2 ** 63 - 1))
        steps = seg.get("steps", d_steps)
        if steps is not None:
            steps = _require_int(steps, f"{where}.steps", 1, 100)
        cfg = seg.get("cfg", d_cfg)
        if cfg is not None:
            cfg = _require_float(cfg, f"{where}.cfg", 0.0, 20.0)
        neg = seg.get("negative_prompt", d_neg)
        if neg is None:
            neg = WAN_NEGATIVE
        else:
            neg = _require_str(neg, f"{where}.negative_prompt",
                               MAX_PROMPT_CHARS, allow_empty=True)
        continuous = (i > 0 and os.path.realpath(kf_paths["keyframe_start"])
                      == os.path.realpath(norm[i - 1]["keyframe_end"]))
        norm.append({
            "prompt": prompt, "negative_prompt": neg, "seed": seed,
            "frames": frames, "steps": steps, "cfg": cfg,
            "keyframe_start": kf_paths["keyframe_start"],
            "keyframe_end": kf_paths["keyframe_end"],
            "continuous": continuous,
        })

    # Global dims from segment 0's start keyframe (deterministic across
    # worker shards — every worker derives the same dims the same way).
    kf0 = load_keyframe(norm[0]["keyframe_start"], "plan segment 0.keyframe_start")
    width, height = align_dim(kf0.width), align_dim(kf0.height)
    return {"lightning": lightning, "fps": fps, "width": width,
            "height": height, "segments": norm}


def parse_only_segments(spec: str, n_segments: int) -> List[int]:
    """Parse --only-segments '0,2,5' into validated sorted unique indices."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            raise VideoParamError(f"--only-segments: {part!r} is not an integer")
        if not 0 <= idx < n_segments:
            raise VideoParamError(
                f"--only-segments: {idx} outside 0..{n_segments - 1}")
        out.add(idx)
    if not out:
        raise VideoParamError("--only-segments: no indices given")
    return sorted(out)


def shard_round_robin(indices: List[int], n: int) -> List[List[int]]:
    """Deal segment indices across n workers round-robin; drops empty shards."""
    shards: List[List[int]] = [[] for _ in range(n)]
    for pos, idx in enumerate(indices):
        shards[pos % n].append(idx)
    return [s for s in shards if s]


def segment_paths(seg_dir: str, idx: int) -> Tuple[str, str]:
    stem = os.path.join(seg_dir, f"seg_{idx:03d}")
    return stem + ".mp4", stem + ".json"


def pending_indices(seg_dir: str, indices: List[int],
                    resume: bool) -> List[int]:
    """With --resume, drop segments whose mp4 + sidecar already exist."""
    if not resume:
        return list(indices)
    pending = []
    for idx in indices:
        mp4, sidecar = segment_paths(seg_dir, idx)
        if os.path.isfile(mp4) and os.path.isfile(sidecar):
            _log(f"seg {idx:03d}: exists, skipped (--resume)")
        else:
            pending.append(idx)
    return pending


def _load_pipeline(model: str, lightning_dir: Optional[str], device: str,
                   offload: bool):
    import torch
    from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
    t0 = time.time()
    pipe = WanImageToVideoPipeline.from_pretrained(
        model, torch_dtype=torch.bfloat16, local_files_only=True)
    if lightning_dir is not None:
        pipe.load_lora_weights(os.path.join(lightning_dir, LIGHTNING_HIGH),
                               adapter_name="lightning_high")
        pipe.load_lora_weights(os.path.join(lightning_dir, LIGHTNING_LOW),
                               adapter_name="lightning_low",
                               load_into_transformer_2=True)
    if offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    _log(f"model ready in {time.time() - t0:.1f}s — "
         f"{'lightning' if lightning_dir else 'base'} mode")
    return pipe


def _render_segment(pipe, device: str, *, start, end, prompt: str,
                    negative_prompt: str, width: int, height: int,
                    num_frames: int, steps: int, cfg: float,
                    seed: int) -> Tuple[list, float]:
    """One pipeline call → (uint8 frame arrays, gen seconds)."""
    import torch
    call_kwargs: Dict[str, Any] = dict(
        image=start, prompt=prompt, negative_prompt=negative_prompt,
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=steps, guidance_scale=cfg, guidance_scale_2=cfg,
        generator=torch.Generator(device).manual_seed(seed))
    if end is not None:
        call_kwargs["last_image"] = end
    t0 = time.time()
    result = pipe(**call_kwargs)
    gen_time = time.time() - t0
    import numpy as np
    raw = getattr(result, "frames")[0]  # union-typed pipeline output; return_dict path
    frames = [np.clip(np.asarray(f) * 255.0, 0, 255).astype("uint8")
              for f in raw]
    return frames, gen_time


def _resolve_seed(seed: int) -> int:
    if seed >= 0:
        return seed
    import torch
    return torch.seed() % (2 ** 31)


def run(args: argparse.Namespace) -> int:
    # Pure validation first — loud, fast failure before the heavy torch import.
    if not args.keyframe_start or not args.prompt:
        raise VideoParamError(
            "--keyframe-start and --prompt are required (or use --plan)")
    validate_frames(args.frames)
    model, lightning_dir = _resolve_model_paths(args)
    steps, cfg = resolve_mode_defaults(lightning_dir is not None,
                                       args.steps, args.cfg)
    start, end, width, height = prepare_keyframes(
        args.keyframe_start, args.keyframe_end)

    seed = _resolve_seed(args.seed)
    pipe = _load_pipeline(model, lightning_dir, args.device, args.offload)
    _log(f"{width}x{height}, frames={args.frames}, steps={steps}, cfg={cfg}, "
         f"seed={seed}, anchored={'yes' if end is not None else 'no'}")

    frames, gen_time = _render_segment(
        pipe, args.device, start=start, end=end, prompt=args.prompt,
        negative_prompt=args.negative_prompt, width=width, height=height,
        num_frames=args.frames, steps=steps, cfg=cfg, seed=seed)
    encode_mp4(frames, args.output, fps=args.fps, crf=args.crf)
    sidecar = build_sidecar(args, model=model, lightning_dir=lightning_dir,
                            steps=steps, cfg=cfg, seed=seed, width=width,
                            height=height, gen_time_s=gen_time)
    sidecar_path = write_sidecar(args.output, sidecar)
    _log(f"generated {len(frames)} frames in {gen_time:.1f}s")
    _log(f"saved: {args.output} (+ {sidecar_path})")
    return 0


# ── stitch: dedup + measure-then-apply color correction (slice V2) ───────

def frame_stats(arr) -> Tuple[Any, Any]:
    """Per-channel (mean, std) of one HxWx3 uint8 frame, float64."""
    import numpy as np
    a = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
    return a.mean(axis=0), a.std(axis=0)


def apply_adain(frames: list, src_stats, dst_stats) -> list:
    """Affine-match each frame's per-channel stats from src to dst (AdaIN).

    Stats come from single boundary frames but apply to a whole segment, so
    the std ratio is clamped to [0.5, 2.0] — a near-flat boundary frame
    (fade, heavy compression) must not blow out contrast across the segment.
    """
    import numpy as np
    m_src, s_src = src_stats
    m_dst, s_dst = dst_stats
    scale = np.clip(s_dst / np.maximum(s_src, 1e-6), 0.5, 2.0)
    out = []
    for f in frames:
        a = np.asarray(f, dtype=np.float64)
        a = (a - m_src) * scale + m_dst
        out.append(np.clip(a, 0, 255).astype(np.uint8))
    return out


def stitch_master(plan: Dict[str, Any], plan_path: str, seg_dir: str,
                  output: str, *, crf: int = 16, color_correct: bool = True,
                  threshold: float = 2.0) -> Dict[str, Any]:
    """Decode segments → dedup boundary frames → color-correct → master mp4.

    Streaming: at most one segment's frames are in memory. Dedup and
    correction happen ONLY at continuous joins (shared boundary keyframe);
    cut joins keep both frames untouched. Correction is measure-then-apply:
    join deltas are always measured and recorded; frames change only when
    color_correct is on AND the delta exceeds the threshold (max abs
    per-channel mean difference, 0–255 scale).
    """
    import numpy as np
    segments = plan["segments"]
    joins: List[Dict[str, Any]] = []
    seg_meta: List[Dict[str, Any]] = []
    n_continuous = 0
    prev_last = None
    # Encode to a tmp name; the deliverable path only ever holds a master
    # that passed the frame-count invariant (ADR-020 atomic-output pattern).
    tmp = output + ".tmp"
    writer = Mp4Writer(tmp, fps=plan["fps"], crf=crf)
    try:
        for i, seg in enumerate(segments):
            mp4, sidecar_path = segment_paths(seg_dir, i)
            frames = decode_frames(mp4)
            if len(frames) != seg["frames"]:
                raise VideoParamError(
                    f"seg {i:03d}: {mp4} decodes to {len(frames)} frames but "
                    f"the plan says {seg['frames']} — re-render it")
            drop = applied = False
            if i > 0:
                m_prev, s_prev = frame_stats(prev_last)
                m_cur, s_cur = frame_stats(frames[0])
                delta = float(np.max(np.abs(m_cur - m_prev)))
                if seg["continuous"]:
                    n_continuous += 1
                    drop = True
                    if color_correct and delta > threshold:
                        frames = apply_adain(frames, (m_cur, s_cur),
                                             (m_prev, s_prev))
                        applied = True
                        _log(f"seg {i:03d}: join delta {delta:.2f} > "
                             f"{threshold} — AdaIN correction applied")
                else:
                    _log(f"seg {i:03d}: cut join (start keyframe differs "
                         f"from previous end) — no dedup, no correction")
                joins.append({"index": i, "continuous": seg["continuous"],
                              "delta": round(delta, 3),
                              "mean_prev": [round(x, 2) for x in m_prev],
                              "mean_cur": [round(x, 2) for x in m_cur],
                              "std_prev": [round(x, 2) for x in s_prev],
                              "std_cur": [round(x, 2) for x in s_cur],
                              "dropped_duplicate_frame": drop,
                              "corrected": applied})
            if drop:
                frames = frames[1:]
            for f in frames:
                writer.add(f)
            prev_last = frames[-1]
            meta: Dict[str, Any] = {"index": i, "file": os.path.basename(mp4),
                                    "frames": seg["frames"]}
            if os.path.isfile(sidecar_path):
                try:
                    with open(sidecar_path) as sf:
                        sc = json.load(sf)
                    meta["seed"] = sc.get("seed")
                    meta["gen_time_s"] = sc.get("gen_time_s")
                except (ValueError, OSError) as exc:
                    _log(f"WARNING: seg {i:03d} sidecar unreadable "
                         f"({exc}) — stitching without its metadata")
            seg_meta.append(meta)
        writer.close()
        expected = sum(s["frames"] for s in segments) - n_continuous
        if writer.frame_count != expected:
            raise VideoParamError(
                f"master has {writer.frame_count} frames, expected {expected}"
                f" — stitch invariant violated, not trusting the output")
        os.replace(tmp, output)
    except BaseException:
        writer.close()
        if os.path.isfile(tmp):
            os.unlink(tmp)
        raise
    sidecar = {
        "schema": PLAN_SCHEMA,
        "plan": os.path.abspath(plan_path),
        "lightning": plan["lightning"],
        "fps": plan["fps"],
        "width": plan["width"],
        "height": plan["height"],
        "crf": crf,
        "color_correct": {"enabled": color_correct, "threshold": threshold},
        "segments": seg_meta,
        "joins": joins,
        "master_frames": writer.frame_count,
    }
    write_sidecar(output, sidecar)
    return sidecar


# ── plan mode driver (slice V2) ──────────────────────────────────────────

def _load_keyframe_at(path: str, label: str, w: int, h: int):
    from PIL import Image
    img = load_keyframe(path, label)
    if img.size != (w, h):
        _log(f"WARNING: {label} {img.size} resized to {w}x{h} (plan dims)")
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    return img


def _render_plan_segments(args: argparse.Namespace, plan: Dict[str, Any],
                          indices: List[int], device: str,
                          seg_dir: str) -> None:
    """Render a shard of plan segments on one device, one pipeline load."""
    args = argparse.Namespace(**vars(args))  # keep the caller's args pristine
    args.no_lightning = not plan["lightning"]  # the plan owns the tier
    model, lightning_dir = _resolve_model_paths(args)
    pipe = _load_pipeline(model, lightning_dir, device, args.offload)
    w, h = plan["width"], plan["height"]
    for idx in indices:
        seg = plan["segments"][idx]
        start = _load_keyframe_at(seg["keyframe_start"],
                                  f"seg {idx:03d} keyframe_start", w, h)
        end = _load_keyframe_at(seg["keyframe_end"],
                                f"seg {idx:03d} keyframe_end", w, h)
        steps, cfg = resolve_mode_defaults(plan["lightning"], seg["steps"],
                                           seg["cfg"])
        seed = _resolve_seed(seg["seed"])
        _log(f"seg {idx:03d}: rendering on {device} — {w}x{h}, "
             f"frames={seg['frames']}, steps={steps}, cfg={cfg}, seed={seed}")
        frames, gen_time = _render_segment(
            pipe, device, start=start, end=end, prompt=seg["prompt"],
            negative_prompt=seg["negative_prompt"], width=w, height=h,
            num_frames=seg["frames"], steps=steps, cfg=cfg, seed=seed)
        mp4, _ = segment_paths(seg_dir, idx)
        encode_mp4(frames, mp4, fps=plan["fps"], crf=args.crf)
        ns = argparse.Namespace(
            prompt=seg["prompt"], negative_prompt=seg["negative_prompt"],
            keyframe_start=seg["keyframe_start"],
            keyframe_end=seg["keyframe_end"], frames=seg["frames"],
            fps=plan["fps"], crf=args.crf, device=device)
        sidecar = build_sidecar(ns, model=model, lightning_dir=lightning_dir,
                                steps=steps, cfg=cfg, seed=seed, width=w,
                                height=h, gen_time_s=gen_time)
        write_sidecar(mp4, sidecar)
        _log(f"seg {idx:03d}: done in {gen_time:.1f}s → {mp4}")


def build_child_cmd(args: argparse.Namespace, shard: List[int],
                    device: str) -> List[str]:
    """CLI for one per-device worker: same plan, its shard, no stitch."""
    cmd = [sys.executable, "-m", "comfyless.video", "--plan", args.plan,
           "--only-segments", ",".join(str(i) for i in shard),
           "--device", device, "--no-stitch", "--output", args.output,
           "--crf", str(args.crf)]
    for flag, val in (("--model", args.model),
                      ("--model-base", args.model_base),
                      ("--lightning-dir", args.lightning_dir)):
        if val:
            cmd += [flag, val]
    if args.offload:
        cmd.append("--offload")
    return cmd


def run_plan(args: argparse.Namespace) -> int:
    if args.no_stitch and args.stitch_only:
        raise VideoParamError("--no-stitch and --stitch-only are exclusive")
    # Creative params live in the plan; reject single-segment flags loudly
    # rather than silently ignoring them. Flags with real defaults are
    # detected by default-comparison — an explicit flag that happens to
    # match its default is undetectable, which is an acceptable residue.
    banned = [(args.keyframe_start, "--keyframe-start"),
              (args.keyframe_end, "--keyframe-end"),
              (args.prompt, "--prompt"),
              (args.steps is not None, "--steps"),
              (args.cfg is not None, "--cfg"),
              (args.no_lightning, "--no-lightning"),
              (args.frames != 81, "--frames"),
              (args.fps != 16, "--fps"),
              (args.seed != -1, "--seed"),
              (args.negative_prompt != WAN_NEGATIVE, "--negative-prompt")]
    bad = [name for given, name in banned if given]
    if bad:
        raise VideoParamError(
            f"{', '.join(bad)} not valid with --plan — creative params "
            f"(prompts, keyframes, frames, fps, seed, steps, cfg, tier) "
            f"come from the plan")
    plan = load_plan(args.plan)
    n = len(plan["segments"])
    seg_dir = os.path.splitext(args.output)[0] + "_segments"
    indices = (parse_only_segments(args.only_segments, n)
               if args.only_segments else list(range(n)))

    if not args.stitch_only:
        os.makedirs(seg_dir, exist_ok=True)
        pending = pending_indices(seg_dir, indices, args.resume)
        devices = ([d.strip() for d in args.devices.split(",") if d.strip()]
                   if args.devices else [args.device])
        if not devices:
            raise VideoParamError("--devices: no devices given")
        if pending:
            if len(devices) == 1:
                _render_plan_segments(args, plan, pending, devices[0],
                                      seg_dir)
            else:
                shards = shard_round_robin(pending, len(devices))
                procs = []
                for shard, device in zip(shards, devices):
                    cmd = build_child_cmd(args, shard, device)
                    _log(f"worker {device}: segments {shard}")
                    procs.append((device, shard, subprocess.Popen(cmd)))
                failures = []
                try:
                    for device, shard, proc in procs:
                        rc = proc.wait()
                        if rc != 0:
                            failures.append(
                                f"{device} (segments {shard}) rc={rc}")
                except KeyboardInterrupt:
                    for _, _, proc in procs:
                        proc.terminate()
                    raise
                if failures:
                    raise VideoParamError(
                        "worker failure — not stitching: "
                        + "; ".join(failures)
                        + " — completed segments remain, re-run with --resume")
        missing = [i for i in indices
                   if not os.path.isfile(segment_paths(seg_dir, i)[0])]
        if missing:
            raise VideoParamError(
                f"segments {missing} missing after render — not stitching")

    if args.no_stitch:
        _log(f"segments done in {seg_dir} (--no-stitch)")
        return 0

    missing = [i for i in range(n)
               if not os.path.isfile(segment_paths(seg_dir, i)[0])]
    if missing:
        raise VideoParamError(
            f"cannot stitch: segments {missing} missing from {seg_dir}")
    sidecar = stitch_master(
        plan, args.plan, seg_dir, args.output, crf=args.crf,
        color_correct=args.color_correct != "off",
        threshold=args.color_correct_threshold)
    _log(f"master: {args.output} ({sidecar['master_frames']} frames, "
         f"{len(sidecar['joins'])} joins)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comfyless.video",
        description="Wan 2.2 video generation: single segment or plan.json "
                    "multi-segment chaining + stitch (ADR-033).")
    p.add_argument("--keyframe-start",
                   help="Start keyframe image (dims are taken from it, "
                        "/16-aligned; single-segment mode)")
    p.add_argument("--keyframe-end",
                   help="Optional end keyframe — enables last_image anchoring")
    p.add_argument("--prompt", help="Prompt (single-segment mode)")
    p.add_argument("--plan",
                   help=f"{PLAN_SCHEMA} JSON — multi-segment chaining + "
                        f"stitch; creative params come from the plan")
    p.add_argument("--devices",
                   help="Comma list (e.g. cuda:0,cuda:1) — plan segments "
                        "shard across per-device worker subprocesses")
    p.add_argument("--only-segments",
                   help="Comma list of plan segment indices to render "
                        "(worker/manual re-render)")
    p.add_argument("--no-stitch", action="store_true",
                   help="Plan mode: render segments only, skip the master")
    p.add_argument("--stitch-only", action="store_true",
                   help="Plan mode: skip rendering, stitch existing segments")
    p.add_argument("--resume", action="store_true",
                   help="Plan mode: skip segments whose mp4 + sidecar exist")
    p.add_argument("--color-correct", choices=["auto", "off"], default="auto",
                   help="Join color correction: auto = apply above threshold "
                        "(deltas are always measured + recorded)")
    p.add_argument("--color-correct-threshold", type=float, default=2.0,
                   help="Max abs per-channel mean delta (0-255) above which "
                        "AdaIN correction applies (default 2.0)")
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
        if args.plan:
            return run_plan(args)
        return run(args)
    except VideoParamError as exc:
        _log(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
