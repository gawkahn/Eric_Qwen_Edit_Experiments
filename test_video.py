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
rc = v.main(["--prompt", "p"])
check("main exit 2 missing keyframe-start", rc == 2)

# ═════════════════════════ slice V2: plan mode ═══════════════════════════

def write_plan(td, plan_dict, name="plan.json"):
    path = os.path.join(td, name)
    with open(path, "w") as f:
        json.dump(plan_dict, f)
    return path


def base_plan(kf, chained=True):
    """Two-segment plan over keyframes kf = [a, b, c]; chained shares b."""
    return {
        "schema": v.PLAN_SCHEMA,
        "segments": [
            {"prompt": "seg one", "keyframe_start": kf[0],
             "keyframe_end": kf[1]},
            {"prompt": "seg two",
             "keyframe_start": kf[1] if chained else kf[2],
             "keyframe_end": kf[2]},
        ],
    }


# ── load_plan: normalization ─────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    kf = [make_png(os.path.join(td, f"kf{i}.png"), 1280, 704) for i in range(3)]
    p = write_plan(td, base_plan(kf))
    plan = v.load_plan(p)
    check("plan lightning default", plan["lightning"] is True)
    check("plan fps default", plan["fps"] == 16)
    check("plan dims from kf0", (plan["width"], plan["height"]) == (1280, 704))
    check("plan two segments", len(plan["segments"]) == 2)
    s0, s1 = plan["segments"]
    check("plan frames default", s0["frames"] == 81)
    check("plan seed default", s0["seed"] == -1)
    check("plan steps sentinel", s0["steps"] is None and s0["cfg"] is None)
    check("plan negative default", s0["negative_prompt"] == v.WAN_NEGATIVE)
    check("plan kf absolute", os.path.isabs(s0["keyframe_start"]))
    check("plan seg0 not continuous", s0["continuous"] is False)
    check("plan chained join continuous", s1["continuous"] is True)

    # relative paths resolve against the plan dir
    rel = base_plan(["kf0.png", "kf1.png", "kf2.png"])
    plan = v.load_plan(write_plan(td, rel, "rel.json"))
    check("plan relative kf resolved",
          plan["segments"][0]["keyframe_start"] == os.path.join(td, "kf0.png"))

    # cut join
    plan = v.load_plan(write_plan(td, base_plan(kf, chained=False), "cut.json"))
    check("plan cut join not continuous",
          plan["segments"][1]["continuous"] is False)

    # defaults + per-segment override
    d = base_plan(kf)
    d["lightning"] = False
    d["defaults"] = {"frames": 33, "fps": 24, "steps": 30, "cfg": 4.0,
                     "negative_prompt": "bad", "seed": 5}
    d["segments"][1].update({"frames": 49, "steps": 8, "cfg": 2.0,
                             "negative_prompt": "worse", "seed": 9})
    plan = v.load_plan(write_plan(td, d, "ovr.json"))
    check("plan lightning off", plan["lightning"] is False)
    check("plan fps override", plan["fps"] == 24)
    s0, s1 = plan["segments"]
    check("plan defaults applied", (s0["frames"], s0["steps"], s0["cfg"],
                                    s0["negative_prompt"], s0["seed"])
          == (33, 30, 4.0, "bad", 5))
    check("plan segment override wins", (s1["frames"], s1["steps"], s1["cfg"],
                                         s1["negative_prompt"], s1["seed"])
          == (49, 8, 2.0, "worse", 9))

# ── load_plan: negative cases ────────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    kf = [make_png(os.path.join(td, f"kf{i}.png"), 640, 352) for i in range(3)]

    expect_error("plan missing file", v.load_plan,
                 os.path.join(td, "nope.json"), needle="plan not found")

    big = os.path.join(td, "big.json")
    with open(big, "w") as f:
        f.write(" " * (v.PLAN_MAX_BYTES + 1))
    expect_error("plan oversize rejected", v.load_plan, big, needle="cap is")

    notjson = os.path.join(td, "nj.json")
    with open(notjson, "w") as f:
        f.write("{nope")
    expect_error("plan invalid json", v.load_plan, notjson,
                 needle="not valid JSON")

    expect_error("plan top-level list", v.load_plan,
                 write_plan(td, [1, 2], "lst.json"), needle="top level")

    d = base_plan(kf); d["schema"] = "comfyless-video-plan/9"
    expect_error("plan wrong schema", v.load_plan,
                 write_plan(td, d, "ws.json"), needle=v.PLAN_SCHEMA)

    d = base_plan(kf); d["surprise"] = 1
    expect_error("plan unknown top key", v.load_plan,
                 write_plan(td, d, "uk.json"), needle="surprise")

    d = base_plan(kf); d["defaults"] = {"width": 512}
    expect_error("plan unknown defaults key", v.load_plan,
                 write_plan(td, d, "ud.json"), needle="width")

    d = base_plan(kf); d["segments"][0]["fps"] = 8
    expect_error("plan unknown segment key (fps is global-only)", v.load_plan,
                 write_plan(td, d, "us.json"), needle="fps")

    d = base_plan(kf); d["lightning"] = "yes"
    expect_error("plan lightning non-bool", v.load_plan,
                 write_plan(td, d, "lb.json"), needle="true/false")

    d = base_plan(kf); d["segments"] = []
    expect_error("plan empty segments", v.load_plan,
                 write_plan(td, d, "es.json"), needle="non-empty")

    d = base_plan(kf); d["segments"] = [{}] * (v.MAX_SEGMENTS + 1)
    expect_error("plan too many segments", v.load_plan,
                 write_plan(td, d, "ts.json"), needle=str(v.MAX_SEGMENTS))

    d = base_plan(kf); del d["segments"][1]["prompt"]
    expect_error("plan missing prompt", v.load_plan,
                 write_plan(td, d, "mp.json"), needle="prompt")

    d = base_plan(kf); d["segments"][0]["prompt"] = "  "
    expect_error("plan empty prompt", v.load_plan,
                 write_plan(td, d, "ep.json"), needle="empty")

    d = base_plan(kf); d["segments"][0]["prompt"] = "x" * 5001
    expect_error("plan prompt cap", v.load_plan,
                 write_plan(td, d, "pc.json"), needle="cap")

    d = base_plan(kf); d["defaults"] = {"frames": 80}
    expect_error("plan bad default frames", v.load_plan,
                 write_plan(td, d, "bf.json"), needle="divisible by 4")

    d = base_plan(kf); d["segments"][1]["frames"] = 80
    expect_error("plan bad segment frames", v.load_plan,
                 write_plan(td, d, "bsf.json"), needle="divisible by 4")

    d = base_plan(kf); d["segments"][0]["frames"] = True
    expect_error("plan bool frames rejected", v.load_plan,
                 write_plan(td, d, "bool.json"), needle="integer")

    d = base_plan(kf); d["defaults"] = {"steps": 0}
    expect_error("plan steps too low", v.load_plan,
                 write_plan(td, d, "s0.json"), needle="range")

    d = base_plan(kf); d["segments"][0]["cfg"] = 21
    expect_error("plan cfg too high", v.load_plan,
                 write_plan(td, d, "c21.json"), needle="range")

    d = base_plan(kf); d["defaults"] = {"fps": 0}
    expect_error("plan fps too low", v.load_plan,
                 write_plan(td, d, "f0.json"), needle="range")

    d = base_plan(kf)
    d["segments"][1]["keyframe_end"] = os.path.join(td, "missing.png")
    expect_error("plan missing keyframe file", v.load_plan,
                 write_plan(td, d, "mk.json"), needle="not found")

# ── shard / only-segments / paths / pending ──────────────────────────────
check("shard round robin", v.shard_round_robin([0, 1, 2, 3, 4], 2)
      == [[0, 2, 4], [1, 3]])
check("shard drops empty", v.shard_round_robin([7], 3) == [[7]])
check("only-segments parse", v.parse_only_segments("2, 0", 5) == [0, 2])
check("only-segments dedup", v.parse_only_segments("1,1", 5) == [1])
expect_error("only-segments out of range", v.parse_only_segments, "5", 5,
             needle="outside")
expect_error("only-segments garbage", v.parse_only_segments, "a", 5,
             needle="not an integer")
expect_error("only-segments empty", v.parse_only_segments, ",", 5,
             needle="no indices")
check("segment paths", v.segment_paths("/d", 3)
      == ("/d/seg_003.mp4", "/d/seg_003.json"))

with tempfile.TemporaryDirectory() as td:
    mp4, sc = v.segment_paths(td, 0)
    for f in (mp4, sc):
        with open(f, "wb") as fh:
            fh.write(b"x")
    mp4_only, _ = v.segment_paths(td, 1)
    with open(mp4_only, "wb") as fh:
        fh.write(b"x")
    check("pending no resume keeps all",
          v.pending_indices(td, [0, 1, 2], False) == [0, 1, 2])
    import contextlib
    import io
    cap = io.StringIO()
    with contextlib.redirect_stderr(cap):
        got = v.pending_indices(td, [0, 1, 2], True)
    check("pending resume skips complete", got == [1, 2])
    check("pending resume notice", "skipped" in cap.getvalue())

# ── child worker command ─────────────────────────────────────────────────
args = v.build_parser().parse_args(
    ["--plan", "/p/plan.json", "--output", "/o/m.mp4", "--model-base", "/mb",
     "--offload", "--crf", "14"])
cmd = v.build_child_cmd(args, [0, 2], "cuda:1")
check("child cmd module", "-m" in cmd and "comfyless.video" in cmd)
check("child cmd shard", cmd[cmd.index("--only-segments") + 1] == "0,2")
check("child cmd device", cmd[cmd.index("--device") + 1] == "cuda:1")
check("child cmd no-stitch", "--no-stitch" in cmd)
check("child cmd forwards model-base", cmd[cmd.index("--model-base") + 1] == "/mb")
check("child cmd forwards offload", "--offload" in cmd)
check("child cmd forwards crf", cmd[cmd.index("--crf") + 1] == "14")
check("child cmd no creative flags",
      "--prompt" not in cmd and "--steps" not in cmd)

# ── frame stats / AdaIN ──────────────────────────────────────────────────
import numpy as np
flat100 = np.full((32, 48, 3), 100, dtype=np.uint8)
flat140 = np.full((32, 48, 3), 140, dtype=np.uint8)
m, s = v.frame_stats(flat100)
check("frame stats mean", np.allclose(m, 100) and np.allclose(s, 0))
corrected = v.apply_adain([flat100], v.frame_stats(flat100),
                          v.frame_stats(flat140))
check("adain flat maps to dst mean", np.allclose(corrected[0], 140))
grad = np.tile(np.arange(48, dtype=np.uint8) * 2, (32, 1))[..., None]
grad = np.repeat(grad, 3, axis=2)
out = v.apply_adain([grad], v.frame_stats(grad), v.frame_stats(grad))
check("adain identity is lossless-ish",
      np.max(np.abs(out[0].astype(int) - grad.astype(int))) <= 1)

# near-flat src boundary vs textured dst: std ratio is clamped so the
# correction cannot blow out the segment's contrast
near_flat = np.full((32, 48, 3), 100, dtype=np.uint8)
near_flat[0, 0] = 104  # tiny std, huge unclamped ratio vs grad
out = v.apply_adain([near_flat], v.frame_stats(near_flat),
                    v.frame_stats(grad))
_, s_in = v.frame_stats(near_flat)
_, s_out = v.frame_stats(out[0])
check("adain scale clamped on flat boundary",
      np.all(s_out <= s_in * 2.0 + 0.5), f"(std {s_in} -> {s_out})")

# ── stitch_master (no GPU) ───────────────────────────────────────────────
def stitch_fixture(td, chained=True, seg2_val=100, frames2=9):
    kf = [make_png(os.path.join(td, f"kf{i}.png"), 96, 64) for i in range(3)]
    plan_dict = base_plan(kf, chained=chained)
    plan_dict["defaults"] = {"frames": 9, "fps": 16}
    plan_path = write_plan(td, plan_dict)
    plan = v.load_plan(plan_path)
    seg_dir = os.path.join(td, "out_segments")
    os.makedirs(seg_dir, exist_ok=True)
    seg1 = [np.full((64, 96, 3), 100, dtype=np.uint8)] * 9
    seg2 = [np.full((64, 96, 3), seg2_val, dtype=np.uint8)] * frames2
    v.encode_mp4(seg1, v.segment_paths(seg_dir, 0)[0], fps=16)
    v.encode_mp4(seg2, v.segment_paths(seg_dir, 1)[0], fps=16)
    return plan, plan_path, seg_dir


with tempfile.TemporaryDirectory() as td:
    plan, plan_path, seg_dir = stitch_fixture(td)
    with open(v.segment_paths(seg_dir, 0)[1], "w") as f:
        json.dump({"seed": 42, "gen_time_s": 7.5}, f)
    out = os.path.join(td, "out.mp4")
    sc = v.stitch_master(plan, plan_path, seg_dir, out)
    check("stitch master written", os.path.isfile(out))
    check("stitch master frames dedup", sc["master_frames"] == 17)
    check("stitch master decode count", len(v.decode_frames(out)) == 17)
    check("stitch join recorded", len(sc["joins"]) == 1
          and sc["joins"][0]["dropped_duplicate_frame"] is True)
    check("stitch small delta uncorrected",
          sc["joins"][0]["corrected"] is False)
    check("stitch seg sidecar meta", sc["segments"][0]["seed"] == 42
          and sc["segments"][0]["gen_time_s"] == 7.5)
    check("stitch master sidecar file",
          json.load(open(os.path.join(td, "out.json")))["schema"]
          == v.PLAN_SCHEMA)

with tempfile.TemporaryDirectory() as td:
    plan, plan_path, seg_dir = stitch_fixture(td, seg2_val=150)
    out = os.path.join(td, "cc.mp4")
    sc = v.stitch_master(plan, plan_path, seg_dir, out)
    check("stitch big delta measured", sc["joins"][0]["delta"] > 40)
    check("stitch correction applied", sc["joins"][0]["corrected"] is True)
    last = v.decode_frames(out)[-1]
    check("stitch corrected toward prev boundary",
          abs(float(last.mean()) - 100) < 8, f"(mean {last.mean():.1f})")

    out2 = os.path.join(td, "cc_off.mp4")
    sc2 = v.stitch_master(plan, plan_path, seg_dir, out2,
                          color_correct=False)
    check("stitch off still measures", sc2["joins"][0]["delta"] > 40)
    check("stitch off leaves frames", sc2["joins"][0]["corrected"] is False)
    last2 = v.decode_frames(out2)[-1]
    check("stitch off frames unchanged",
          abs(float(last2.mean()) - 150) < 8, f"(mean {last2.mean():.1f})")

with tempfile.TemporaryDirectory() as td:
    plan, plan_path, seg_dir = stitch_fixture(td, chained=False,
                                              seg2_val=150)
    out = os.path.join(td, "cut.mp4")
    import contextlib
    import io
    cap = io.StringIO()
    with contextlib.redirect_stderr(cap):
        sc = v.stitch_master(plan, plan_path, seg_dir, out)
    check("cut join no dedup", sc["master_frames"] == 18)
    check("cut join not corrected", sc["joins"][0]["corrected"] is False
          and sc["joins"][0]["dropped_duplicate_frame"] is False)
    check("cut join notice", "cut join" in cap.getvalue())
    last = v.decode_frames(out)[-1]
    check("cut join frames untouched",
          abs(float(last.mean()) - 150) < 8)

with tempfile.TemporaryDirectory() as td:
    plan, plan_path, seg_dir = stitch_fixture(td, frames2=5)
    bad_out = os.path.join(td, "bad.mp4")
    expect_error("stitch frame-count mismatch", v.stitch_master, plan,
                 plan_path, seg_dir, bad_out, needle="re-render")
    check("failed stitch leaves no master", not os.path.isfile(bad_out))
    check("failed stitch leaves no tmp",
          not os.path.isfile(bad_out + ".tmp"))

# corrupt segment sidecar: stitch proceeds without its metadata, loudly
with tempfile.TemporaryDirectory() as td:
    plan, plan_path, seg_dir = stitch_fixture(td)
    with open(v.segment_paths(seg_dir, 0)[1], "w") as f:
        f.write("{corrupt")
    out = os.path.join(td, "csc.mp4")
    import contextlib
    import io
    cap = io.StringIO()
    with contextlib.redirect_stderr(cap):
        sc = v.stitch_master(plan, plan_path, seg_dir, out)
    check("corrupt sidecar tolerated", os.path.isfile(out)
          and "seed" not in sc["segments"][0])
    check("corrupt sidecar warned", "sidecar unreadable" in cap.getvalue())
    check("join records channel stats",
          len(sc["joins"][0]["mean_prev"]) == 3
          and len(sc["joins"][0]["std_cur"]) == 3)
    check("success leaves no tmp", not os.path.isfile(out + ".tmp"))

# ── run_plan via main() (no-GPU paths) ───────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    plan, plan_path, seg_dir = stitch_fixture(td)
    out = os.path.join(td, "out.mp4")

    check("plan+prompt conflict",
          v.main(["--plan", plan_path, "--prompt", "x"]) == 2)
    check("plan+no-lightning conflict",
          v.main(["--plan", plan_path, "--no-lightning"]) == 2)
    check("plan+steps conflict",
          v.main(["--plan", plan_path, "--steps", "4"]) == 2)
    check("plan+frames conflict",
          v.main(["--plan", plan_path, "--frames", "49"]) == 2)
    check("plan+fps conflict",
          v.main(["--plan", plan_path, "--fps", "30"]) == 2)
    check("plan+seed conflict",
          v.main(["--plan", plan_path, "--seed", "7"]) == 2)
    check("no-stitch + stitch-only exclusive",
          v.main(["--plan", plan_path, "--no-stitch", "--stitch-only"]) == 2)
    check("only-segments out of range via main",
          v.main(["--plan", plan_path, "--stitch-only",
                  "--only-segments", "9", "--output", out]) == 2)

    # stitch-only end-to-end: no model, no torch, no GPU
    os.environ.pop("COMFYLESS_MODEL_BASE", None)
    rc = v.main(["--plan", plan_path, "--stitch-only", "--output", out])
    check("stitch-only exit 0", rc == 0)
    check("stitch-only master written", os.path.isfile(out)
          and os.path.isfile(os.path.join(td, "out.json")))

    # stitch-only with a missing segment refuses
    os.remove(v.segment_paths(seg_dir, 1)[0])
    check("stitch-only missing segment refused",
          v.main(["--plan", plan_path, "--stitch-only", "--output", out]) == 2)

# worker-dispatch failure: real child processes fail fast (no model base,
# pre-torch) → parent names the failure, does not stitch, no master
with tempfile.TemporaryDirectory() as td:
    plan, plan_path, seg_dir = stitch_fixture(td)
    for i in (0, 1):
        os.remove(v.segment_paths(seg_dir, i)[0])
    out = os.path.join(td, "wf.mp4")
    os.environ.pop("COMFYLESS_MODEL_BASE", None)
    old_pp = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(Path(__file__).parent) + (
        os.pathsep + old_pp if old_pp else "")
    try:
        import contextlib
        import io
        cap = io.StringIO()
        with contextlib.redirect_stderr(cap):
            rc = v.main(["--plan", plan_path, "--output", out,
                         "--devices", "cuda:0,cuda:1"])
    finally:
        if old_pp is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pp
    check("worker failure exit 2", rc == 2)
    check("worker failure names workers", "worker failure" in cap.getvalue())
    check("worker failure no master", not os.path.isfile(out))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
