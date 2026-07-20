#!/usr/bin/env python3
"""Tests for comfyless/video.py — single-segment video CLI (ADR-033 slice V1).

No GPU required: covers param validation, mode defaults, keyframe prep,
sidecar round-trip, and the av encode round-trip with synthetic frames.
Run: ./.venv/bin/python3 test_video.py
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyless.video as v  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name} {detail}")


def expect_error(name, fn, *args, needle=""):
    try:
        fn(*args)
    except v.VideoParamError as exc:
        check(name, needle in str(exc), f"(message {exc!r} lacks {needle!r})")
        return
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"(wrong exception type: {type(exc).__name__})")
        return
    check(name, False, "(no error raised)")


def make_png(path, w, h, color=(200, 30, 30)):
    from PIL import Image
    Image.new("RGB", (w, h), color).save(path)
    return path


# ── align_dim ────────────────────────────────────────────────────────────
check("align exact", v.align_dim(1280) == 1280)
check("align down", v.align_dim(1283) == 1280)
check("align 720->704", v.align_dim(719) == 704)
check("align min ok", v.align_dim(16) == 16)
expect_error("align below min", v.align_dim, 15, needle="16 px minimum")
expect_error("align zero", v.align_dim, 0, needle="16 px minimum")

# ── validate_frames ──────────────────────────────────────────────────────
check("frames 81 ok", v.validate_frames(81) == 81)
check("frames 5 ok", v.validate_frames(5) == 5)
check("frames 121 ok", v.validate_frames(121) == 121)
expect_error("frames 80 rejected", v.validate_frames, 80, needle="77")
expect_error("frames 80 names next", v.validate_frames, 80, needle="81")
expect_error("frames 4 too low", v.validate_frames, 4, needle="minimum is 5")
expect_error("frames 0 too low", v.validate_frames, 0, needle="minimum is 5")

# ── resolve_mode_defaults (ADR-009 sentinel pattern) ─────────────────────
check("lightning defaults", v.resolve_mode_defaults(True, None, None) == (4, 1.0))
check("base defaults", v.resolve_mode_defaults(False, None, None) == (40, 3.5))
check("steps override kept", v.resolve_mode_defaults(True, 8, None) == (8, 1.0))
check("cfg override kept", v.resolve_mode_defaults(False, None, 5.0) == (40, 5.0))
check("both overridden", v.resolve_mode_defaults(True, 6, 2.0) == (6, 2.0))
check("explicit zero cfg honored", v.resolve_mode_defaults(True, None, 0.0) == (4, 0.0))

# ── keyframe loading / preparation ───────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    a = make_png(os.path.join(td, "a.png"), 1280, 704)
    b_off = make_png(os.path.join(td, "b.png"), 640, 360, color=(30, 30, 200))
    misaligned = make_png(os.path.join(td, "m.png"), 1283, 719)
    notimg = os.path.join(td, "x.png")
    with open(notimg, "w") as f:
        f.write("not a png")

    expect_error("missing start keyframe", v.load_keyframe,
                 os.path.join(td, "nope.png"), "--keyframe-start",
                 needle="not found")
    expect_error("undecodable keyframe", v.load_keyframe, notimg,
                 "--keyframe-end", needle="not a readable image")

    start, end, w, h = v.prepare_keyframes(a, None)
    check("dims from start", (w, h) == (1280, 704))
    check("no end -> None", end is None)

    import contextlib
    import io
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
        start, end, w, h = v.prepare_keyframes(a, b_off)
    check("end resized to start dims", end is not None and end.size == (1280, 704))
    check("end resize warning emitted", "WARNING" in captured.getvalue(),
          f"(stderr: {captured.getvalue()!r})")

    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
        start, end, w, h = v.prepare_keyframes(misaligned, None)
    check("misaligned start aligned down", (w, h) == (1280, 704))
    check("start actually resized", start.size == (1280, 704))
    check("align warning emitted", "WARNING" in captured.getvalue())

# ── encode round-trip ────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    import numpy as np
    frames = [np.full((64, 96, 3), i * 12, dtype=np.uint8) for i in range(9)]
    out = os.path.join(td, "clip.mp4")
    v.encode_mp4(frames, out, fps=16)
    check("mp4 written", os.path.isfile(out) and os.path.getsize(out) > 0)
    import av
    with av.open(out) as container:
        vstream = container.streams.video[0]
        decoded = [f for f in container.decode(vstream)]
    check("mp4 frame count", len(decoded) == 9, f"(got {len(decoded)})")
    check("mp4 dims", (decoded[0].width, decoded[0].height) == (96, 64))
    check("mp4 fps", vstream.average_rate == 16,
          f"(got {vstream.average_rate})")
    expect_error("encode empty", v.encode_mp4, [], out, 16, needle="no frames")

# ── sidecar round-trip ───────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    args = argparse.Namespace(
        prompt="p", negative_prompt="n", keyframe_start="a.png",
        keyframe_end="b.png", frames=81, fps=16, crf=16, device="cuda:0")
    sc = v.build_sidecar(args, model="/m", lightning_dir="/l", steps=4,
                         cfg=1.0, seed=7, width=1280, height=704,
                         gen_time_s=12.34)
    video_path = os.path.join(td, "seg.mp4")
    with open(video_path, "wb") as f:
        f.write(b"stub")
    sc_path = v.write_sidecar(video_path, sc)
    check("sidecar path", sc_path == os.path.join(td, "seg.json"))
    with open(sc_path) as f:
        back = json.load(f)
    check("sidecar schema", back["schema"] == v.SIDECAR_SCHEMA)
    check("sidecar lightning flag", back["lightning"] is True)
    check("sidecar replay fields",
          all(k in back for k in ("model", "prompt", "negative_prompt", "seed",
                                  "width", "height", "frames", "fps", "steps",
                                  "cfg", "crf", "keyframe_start", "keyframe_end",
                                  "lightning_dir", "device", "gen_time_s")))
    check("sidecar keyframes absolute",
          os.path.isabs(back["keyframe_start"]))
    check("sidecar model absolute", os.path.isabs(back["model"]))
    check("sidecar lightning_dir absolute", os.path.isabs(back["lightning_dir"]))
    sc2 = v.build_sidecar(args, model="/m", lightning_dir=None, steps=40,
                          cfg=3.5, seed=7, width=1280, height=704,
                          gen_time_s=1.0)
    check("sidecar base mode", sc2["lightning"] is False
          and sc2["lightning_dir"] is None)

# ── model/LoRA path resolution ───────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    base = os.path.join(td, "models")
    model_dir = os.path.join(base, v.DEFAULT_MODEL_NAME)
    light_dir = os.path.join(base, v.DEFAULT_LIGHTNING_SUBDIR)
    os.makedirs(model_dir)
    os.makedirs(light_dir)

    def ns(**kw):
        d = dict(model=None, model_base=None, no_lightning=False,
                 lightning_dir=None)
        d.update(kw)
        return argparse.Namespace(**d)

    os.environ.pop("COMFYLESS_MODEL_BASE", None)
    expect_error("no model no base", v._resolve_model_paths, ns(),
                 needle="--model not given")
    expect_error("lightning loras missing", v._resolve_model_paths,
                 ns(model_base=base), needle="lightning LoRA missing")
    for fname in (v.LIGHTNING_HIGH, v.LIGHTNING_LOW):
        with open(os.path.join(light_dir, fname), "wb") as f:
            f.write(b"x")
    m, l = v._resolve_model_paths(ns(model_base=base))
    check("default model under base", m == model_dir)
    check("default lightning under base", l == light_dir)
    m, l = v._resolve_model_paths(ns(model_base=base, no_lightning=True))
    check("no-lightning skips loras", l is None)
    expect_error("bad model dir", v._resolve_model_paths,
                 ns(model=os.path.join(td, "missing")),
                 needle="model directory not found")
    os.environ["COMFYLESS_MODEL_BASE"] = base
    try:
        m, l = v._resolve_model_paths(ns())
        check("env model base honored", m == model_dir and l == light_dir)
    finally:
        os.environ.pop("COMFYLESS_MODEL_BASE", None)

# ── CLI parser / main error path ─────────────────────────────────────────
parser = v.build_parser()
args = parser.parse_args(["--keyframe-start", "a.png", "--prompt", "p"])
check("parser defaults frames", args.frames == 81)
check("parser defaults fps", args.fps == 16)
check("parser steps sentinel", args.steps is None)
check("parser cfg sentinel", args.cfg is None)
check("parser lightning default on", args.no_lightning is False)
check("parser offload default off", args.offload is False)
rc = v.main(["--keyframe-start", "/nonexistent/a.png", "--prompt", "p",
             "--frames", "80"])
check("main exit 2 on validation error", rc == 2)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
