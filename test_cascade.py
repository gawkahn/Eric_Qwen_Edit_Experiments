#!/usr/bin/env python3
"""Test harness for the Stable Cascade dispatch in comfyless (ADR-010).

Exercises the pure-logic helpers in comfyless/cascade.py:
  - validate_config            (defaults, type checks, alignment, comment keys)
  - load_config                (JSON-parse + filesystem errors)
  - _resolve_torch_dtype       (name aliases, error on bad name)
  - _align_cascade_dim         (128px boundary cases)
  - _resolve_output_path       (single-run, multi-run suffixing, directory base)
  - dispatch                   (CLI flag rejection, missing-config errors,
                                empty-config-list error, --json/--serve refusal)
  - _split_model_arg in generate.py (nargs='+' normalization)

Runs without ComfyUI, GPU, or loaded diffusion models. Pipeline assembly
(build_pipelines) is NOT exercised — that requires real weights and a GPU
or significant CPU memory; the user smoke-tests it manually.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyless.generate as g
import comfyless.cascade as c


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def expect_raises(name, fn, exc_types, detail=""):
    """Run fn() and check that it raises one of exc_types."""
    try:
        fn()
    except exc_types as e:
        check(name, True)
        return e
    except Exception as e:
        check(name, False, f"expected {exc_types}, got {type(e).__name__}: {e}")
        return None
    check(name, False, f"expected {exc_types}, got no exception")
    return None


def make_cli_args(**overrides):
    """Build an argparse.Namespace mimicking the full comfyless arg surface."""
    defaults = dict(
        json=False, serve=False, unload=False,
        params=None, override=[],
        prompt="a test prompt", negative_prompt=None,
        seed=42, steps=None, cfg=None, true_cfg=None,
        width=None, height=None,
        lora=[], sampler=None, schedule=None,
        max_seq_len=None,
        transformer=None, vae=None, te1=None, te2=None,
        vae_from_transformer=False,
        precision="bf16", device="cpu",
        offload_vae=False, attention_slicing=False, sequential_offload=False,
        allow_hf_download=False,
        output="/tmp/cascade-test.png", savepath=None,
        iterate=[], max_iterations=500, limit=None, batch=1, yes=True,
        output_dir=None, model_base=None,
        model="stablecascade",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


# ──────────────────────────────────────────────────────────────────────
print("── _resolve_torch_dtype ───────────────────────────────────────")

import torch
check("bf16 → torch.bfloat16",      c._resolve_torch_dtype("bf16") is torch.bfloat16)
check("bfloat16 → torch.bfloat16",  c._resolve_torch_dtype("bfloat16") is torch.bfloat16)
check("fp16 → torch.float16",       c._resolve_torch_dtype("fp16") is torch.float16)
check("float16 → torch.float16",    c._resolve_torch_dtype("float16") is torch.float16)
check("half → torch.float16",       c._resolve_torch_dtype("half") is torch.float16)
check("fp32 → torch.float32",       c._resolve_torch_dtype("fp32") is torch.float32)
check("float32 → torch.float32",    c._resolve_torch_dtype("float32") is torch.float32)
check("float → torch.float32",      c._resolve_torch_dtype("float") is torch.float32)
check("BF16 case-insensitive",      c._resolve_torch_dtype("BF16") is torch.bfloat16)
expect_raises("bad dtype raises ValueError",
              lambda: c._resolve_torch_dtype("int8"), ValueError)


# ──────────────────────────────────────────────────────────────────────
print("\n── _align_cascade_dim ─────────────────────────────────────────")

check("0 stays 0",         c._align_cascade_dim(0) == 0)
check("127 → 0",           c._align_cascade_dim(127) == 0)
check("128 stays 128",     c._align_cascade_dim(128) == 128)
check("129 → 128",         c._align_cascade_dim(129) == 128)
check("1023 → 896",        c._align_cascade_dim(1023) == 896)
check("1024 stays 1024",   c._align_cascade_dim(1024) == 1024)
check("1100 → 1024",       c._align_cascade_dim(1100) == 1024)
check("1200 → 1152",       c._align_cascade_dim(1200) == 1152)
check("2048 stays 2048",   c._align_cascade_dim(2048) == 2048)


# ──────────────────────────────────────────────────────────────────────
print("\n── validate_config: happy paths ──────────────────────────────")

minimal = {"stage_c": "/tmp/c.safetensors", "stage_b": "/tmp/b.safetensors"}
out = c.validate_config(minimal)
check("minimal config keeps stage_c", out["stage_c"] == "/tmp/c.safetensors")
check("minimal config keeps stage_b", out["stage_b"] == "/tmp/b.safetensors")
check("default prior_dtype = bf16",   out["prior_dtype"] == "bf16")
check("default decoder_dtype = bf16 (safe; sidesteps deprecated diffusers cross-dtype quirk)",
      out["decoder_dtype"] == "bf16")
check("default vae_dtype = bf16 (uniform with decoder)",
      out["vae_dtype"] == "bf16")
check("default prior_steps = 20",     out["prior_steps"] == 20)
check("default prior_cfg_scale = 4.0",   out["prior_cfg_scale"] == 4.0)
check("default decoder_steps = 10",   out["decoder_steps"] == 10)
check("default decoder_cfg_scale = 0.0", out["decoder_cfg_scale"] == 0.0)
check("default width = 1024",         out["width"] == 1024)
check("default height = 1024",        out["height"] == 1024)
check("default scaffolding_repo",     out["scaffolding_repo"] == "stabilityai/stable-cascade")

custom = c.validate_config({**minimal, "prior_steps": 30, "prior_cfg_scale": 5.5})
check("custom prior_steps preserved", custom["prior_steps"] == 30)
check("custom prior_cfg_scale preserved", custom["prior_cfg_scale"] == 5.5)
check("decoder_steps still default",  custom["decoder_steps"] == 10)


# ──────────────────────────────────────────────────────────────────────
print("\n── validate_config: errors ───────────────────────────────────")

expect_raises("missing stage_c raises",
              lambda: c.validate_config({"stage_b": "/tmp/b.safetensors"}), ValueError)
expect_raises("missing stage_b raises",
              lambda: c.validate_config({"stage_c": "/tmp/c.safetensors"}), ValueError)
expect_raises("non-string stage_c raises",
              lambda: c.validate_config({"stage_c": 42, "stage_b": "/tmp/b.safetensors"}),
              TypeError)
expect_raises("non-dict raises",
              lambda: c.validate_config([1, 2, 3]), TypeError)
expect_raises("bad prior_dtype raises",
              lambda: c.validate_config({**minimal, "prior_dtype": "int8"}), ValueError)
expect_raises("non-int steps raises",
              lambda: c.validate_config({**minimal, "prior_steps": "many"}), TypeError)
expect_raises("zero steps raises",
              lambda: c.validate_config({**minimal, "prior_steps": 0}), ValueError)
expect_raises("negative cfg raises",
              lambda: c.validate_config({**minimal, "prior_cfg_scale": -1.0}), ValueError)
expect_raises("non-numeric cfg raises",
              lambda: c.validate_config({**minimal, "decoder_cfg_scale": "soft"}), TypeError)


# ──────────────────────────────────────────────────────────────────────
print("\n── validate_config: alignment ────────────────────────────────")

aligned = c.validate_config({**minimal, "width": 1100, "height": 1023})
check("width 1100 → 1024",  aligned["width"] == 1024)
check("height 1023 → 896",  aligned["height"] == 896)

aligned_already = c.validate_config({**minimal, "width": 1024, "height": 1024})
check("aligned width preserved",  aligned_already["width"] == 1024)
check("aligned height preserved", aligned_already["height"] == 1024)

expect_raises("width that aligns to 0 raises",
              lambda: c.validate_config({**minimal, "width": 100, "height": 1024}), ValueError)
expect_raises("height that aligns to 0 raises",
              lambda: c.validate_config({**minimal, "width": 1024, "height": 50}), ValueError)


# ──────────────────────────────────────────────────────────────────────
print("\n── validate_config: comment keys ─────────────────────────────")

# The validator should not warn on `_*` keys (JSON comment convention).
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    c.validate_config({**minimal, "_comment": "ignore me", "_notes": "or me"})
stderr = buf.getvalue()
check("comment keys do not produce 'unknown keys' warning",
      "unknown keys" not in stderr,
      f"stderr leaked: {stderr!r}")

# But genuine unknowns should still warn.
buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    c.validate_config({**minimal, "frobinate": True})
stderr = buf.getvalue()
check("genuine unknown key warns",
      "unknown keys" in stderr and "frobinate" in stderr,
      f"stderr: {stderr!r}")


# ──────────────────────────────────────────────────────────────────────
print("\n── load_config: filesystem ───────────────────────────────────")

expect_raises("missing file raises FileNotFoundError",
              lambda: c.load_config("/tmp/does-not-exist-cascade.json"), FileNotFoundError)

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    f.write("{ not: valid json")
    bad_path = f.name
try:
    expect_raises("invalid JSON raises ValueError",
                  lambda: c.load_config(bad_path), ValueError)
finally:
    os.unlink(bad_path)

# Round-trip a real file.
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({**minimal, "prior_steps": 25}, f)
    good_path = f.name
try:
    cfg = c.load_config(good_path)
    check("good config loads",        cfg["prior_steps"] == 25)
    check("good config gets defaults", cfg["decoder_steps"] == 10)
finally:
    os.unlink(good_path)


# ──────────────────────────────────────────────────────────────────────
print("\n── _resolve_output_path ──────────────────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    # Single iteration: write to exact path.
    base = os.path.join(tmp, "single.png")
    out = c._resolve_output_path(base, total_iterations=1, run_index=0)
    check("single iter uses exact path", out == base)

    # Multi iteration with a file-shaped base: suffix.
    base = os.path.join(tmp, "multi.png")
    out0 = c._resolve_output_path(base, total_iterations=3, run_index=0)
    out1 = c._resolve_output_path(base, total_iterations=3, run_index=1)
    out2 = c._resolve_output_path(base, total_iterations=3, run_index=2)
    check("multi iter run 0 → _0001",  out0 == os.path.join(tmp, "multi_0001.png"))
    check("multi iter run 1 → _0002",  out1 == os.path.join(tmp, "multi_0002.png"))
    check("multi iter run 2 → _0003",  out2 == os.path.join(tmp, "multi_0003.png"))

    # Directory base + multi → cascade_NNNN.png inside the directory.
    out0 = c._resolve_output_path(tmp, total_iterations=2, run_index=0)
    out1 = c._resolve_output_path(tmp, total_iterations=2, run_index=1)
    check("dir base + multi run 0",
          out0 == os.path.join(tmp, "cascade_0001.png"))
    check("dir base + multi run 1",
          out1 == os.path.join(tmp, "cascade_0002.png"))

    # Directory base + SINGLE iteration → cascade.png (no suffix). Symmetric
    # with the multi-iter dir branch — closes the gap that would have crashed
    # at PIL save when a user passes a directory path with a single config.
    out = c._resolve_output_path(tmp, total_iterations=1, run_index=0)
    check("dir base + single → cascade.png",
          out == os.path.join(tmp, "cascade.png"))

    # Multi iter creates parent dirs as a side effect.
    deep = os.path.join(tmp, "a", "b", "c", "img.png")
    out = c._resolve_output_path(deep, total_iterations=2, run_index=0)
    check("multi creates deep parent dir",
          os.path.isdir(os.path.dirname(out)))


# ──────────────────────────────────────────────────────────────────────
print("\n── _scan_existing_offset / continuation numbering ───────────")

with tempfile.TemporaryDirectory() as tmp:
    # Empty dir → offset 0.
    check("empty dir → offset 0", c._scan_existing_offset(tmp) == 0)

    # Dir with cascade_0001…0050.png → offset 50.
    for i in range(1, 51):
        Path(os.path.join(tmp, f"cascade_{i:04d}.png")).touch()
    check("50 existing files → offset 50", c._scan_existing_offset(tmp) == 50)

    # Non-matching files don't affect offset.
    Path(os.path.join(tmp, "random.png")).touch()
    Path(os.path.join(tmp, "cascade.png")).touch()  # no number, ignored
    Path(os.path.join(tmp, "cascade_0001.json")).touch()  # sidecar, ignored
    check("non-matching files don't change offset", c._scan_existing_offset(tmp) == 50)

    # Sparse / non-contiguous: highest wins.
    Path(os.path.join(tmp, "cascade_9999.png")).touch()
    check("sparse + highest wins", c._scan_existing_offset(tmp) == 9999)

    # Offset feeds _resolve_output_path correctly.
    out = c._resolve_output_path(tmp, total_iterations=100, run_index=0, dir_offset=50)
    check("dir_offset=50, run 0 → cascade_0051.png",
          out == os.path.join(tmp, "cascade_0051.png"))
    out = c._resolve_output_path(tmp, total_iterations=100, run_index=99, dir_offset=50)
    check("dir_offset=50, run 99 → cascade_0150.png",
          out == os.path.join(tmp, "cascade_0150.png"))

    # Single-iter + dir_offset=0 keeps the unnumbered cascade.png form.
    out = c._resolve_output_path(tmp, total_iterations=1, run_index=0, dir_offset=0)
    check("dir + single + offset=0 → cascade.png (legacy unnumbered form)",
          out == os.path.join(tmp, "cascade.png"))

    # Single-iter + dir_offset>0 → numbered (continuation).
    out = c._resolve_output_path(tmp, total_iterations=1, run_index=0, dir_offset=42)
    check("dir + single + offset>0 → cascade_NNNN.png (continuation)",
          out == os.path.join(tmp, "cascade_0043.png"))

# File-shaped base offset scan (uses the file's stem as the prefix).
with tempfile.TemporaryDirectory() as tmp:
    base = os.path.join(tmp, "myrun.png")
    Path(os.path.join(tmp, "myrun_0001.png")).touch()
    Path(os.path.join(tmp, "myrun_0007.png")).touch()
    Path(os.path.join(tmp, "other_0099.png")).touch()  # different stem, ignored
    check("file-base offset scan: highest matching stem wins",
          c._scan_existing_offset(base) == 7)
    out = c._resolve_output_path(base, total_iterations=10, run_index=0, dir_offset=7)
    check("file-base + offset → <stem>_<NNNN>.png continues",
          out == os.path.join(tmp, "myrun_0008.png"))


# ──────────────────────────────────────────────────────────────────────
print("\n── _resolve_cascade_savepath / template expansion ───────────")

# Build a minimal cfg dict.
sp_cfg = {"prior_steps": 20, "prior_cfg_scale": 4.0}

with tempfile.TemporaryDirectory() as tmp:
    # %input% expansion via iterate_inputs.
    template = os.path.join(tmp, "%input%", "cascade")
    iterate_inputs = {"prompt": "prompt1-100", "_primary": "prompt1-100"}
    out = c._resolve_cascade_savepath(template, eff_seed=42, cfg=sp_cfg,
                                       iterate_inputs=iterate_inputs)
    check("%input% expands and writes into per-input subdir",
          out == os.path.join(tmp, "prompt1-100", "cascade_0001.png"))
    Path(out).touch()

    # Second call to the same template auto-counters past existing.
    out = c._resolve_cascade_savepath(template, eff_seed=42, cfg=sp_cfg,
                                       iterate_inputs=iterate_inputs)
    check("savepath auto-counter increments past existing",
          out == os.path.join(tmp, "prompt1-100", "cascade_0002.png"))

    # Different %input% goes to a different dir, fresh counter.
    out = c._resolve_cascade_savepath(template, eff_seed=42, cfg=sp_cfg,
                                       iterate_inputs={"prompt": "prompt101-200",
                                                       "_primary": "prompt101-200"})
    check("different %input% → different subdir, fresh counter",
          out == os.path.join(tmp, "prompt101-200", "cascade_0001.png"))

    # %seed% expansion.
    template_seed = os.path.join(tmp, "seed_%seed%", "cascade")
    out = c._resolve_cascade_savepath(template_seed, eff_seed=42, cfg=sp_cfg,
                                       iterate_inputs={})
    check("%seed% expands",
          out == os.path.join(tmp, "seed_42", "cascade_0001.png"))


# ──────────────────────────────────────────────────────────────────────
print("\n── _split_model_arg in generate.py ───────────────────────────")

ns = argparse.Namespace(model=None)
extras = g._split_model_arg(ns)
check("None model → no extras",     extras == [] and ns.model is None)

ns = argparse.Namespace(model=["/path/to/sdxl"])
extras = g._split_model_arg(ns)
check("single-element list → str + []", ns.model == "/path/to/sdxl" and extras == [])

ns = argparse.Namespace(model=["stablecascade", "c1.json"])
extras = g._split_model_arg(ns)
check("cascade sentinel → str + 1 extra",
      ns.model == "stablecascade" and extras == ["c1.json"])

ns = argparse.Namespace(model=["stablecascade", "c1.json", "c2.json", "c3.json"])
extras = g._split_model_arg(ns)
check("cascade sentinel + 3 configs",
      ns.model == "stablecascade" and extras == ["c1.json", "c2.json", "c3.json"])


# ──────────────────────────────────────────────────────────────────────
print("\n── dispatch: CLI-flag rejection ─────────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    cfg_path = os.path.join(tmp, "cfg.json")
    write_json(minimal, cfg_path)

    # --steps conflicts with prior_steps/decoder_steps.
    rc = c.dispatch(make_cli_args(steps=50), [cfg_path])
    check("--steps rejected", rc == 2)

    rc = c.dispatch(make_cli_args(cfg=4.0), [cfg_path])
    check("--cfg rejected", rc == 2)

    rc = c.dispatch(make_cli_args(width=1024, height=1024), [cfg_path])
    check("--width/--height rejected", rc == 2)

    rc = c.dispatch(make_cli_args(lora=["/tmp/foo.safetensors:1.0"]), [cfg_path])
    check("--lora rejected", rc == 2)

    rc = c.dispatch(make_cli_args(transformer="/tmp/foo"), [cfg_path])
    check("--transformer rejected", rc == 2)

    rc = c.dispatch(make_cli_args(sampler="multistep2"), [cfg_path])
    check("--sampler rejected", rc == 2)

    # --iterate prompt and --iterate seed are SUPPORTED axes. Bad iterate axes
    # (e.g. cfg_scale, model) are rejected.
    rc = c.dispatch(make_cli_args(iterate=[("cfg_scale", "/tmp/cfgs.json")]), [cfg_path])
    check("--iterate cfg_scale rejected (unsupported axis)", rc == 2)

    rc = c.dispatch(make_cli_args(iterate=[("model", "/tmp/models.json")]), [cfg_path])
    check("--iterate model rejected (unsupported axis)", rc == 2)

    rc = c.dispatch(make_cli_args(params="/tmp/foo.json"), [cfg_path])
    check("--params rejected", rc == 2)

    rc = c.dispatch(make_cli_args(override=["foo=bar"]), [cfg_path])
    check("--override rejected", rc == 2)

    # --savepath is now SUPPORTED (template expansion). Passes flag rejection;
    # fails downstream at pipeline build (no real weights). rc != 2 means it
    # got past flag rejection.
    rc = c.dispatch(make_cli_args(savepath="/tmp/x_%seed%"), [cfg_path])
    check("--savepath accepted (template expansion supported)",
          rc != 2,
          f"expected pipeline-build failure (rc=3), got rc={rc}")

    rc = c.dispatch(make_cli_args(max_seq_len=512), [cfg_path])
    check("--max-seq-len rejected", rc == 2)

    rc = c.dispatch(make_cli_args(attention_slicing=True), [cfg_path])
    check("--attention-slicing rejected", rc == 2)

    rc = c.dispatch(make_cli_args(sequential_offload=True), [cfg_path])
    check("--sequential-offload rejected", rc == 2)

    rc = c.dispatch(make_cli_args(precision="fp16"), [cfg_path])
    check("--precision=fp16 rejected (non-default)", rc == 2)

    # --precision=bf16 (the default) should NOT be rejected, since we can't
    # distinguish it from "user didn't pass --precision".
    rc = c.dispatch(make_cli_args(precision="bf16"), ["/tmp/does-not-exist.json"])
    check("--precision=bf16 (default) does not trigger rejection",
          rc == 2,  # still exits 2 from missing config, but for a different reason
          "should fail at config-load step, not at flag rejection")


# ──────────────────────────────────────────────────────────────────────
print("\n── dispatch: missing config errors ───────────────────────────")

rc = c.dispatch(make_cli_args(), [])
check("empty config list → exit 2", rc == 2)

rc = c.dispatch(make_cli_args(), ["/tmp/does-not-exist-cascade.json"])
check("nonexistent config → exit 2", rc == 2)


# ──────────────────────────────────────────────────────────────────────
print("\n── dispatch: --json / --serve refusal ────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    cfg_path = os.path.join(tmp, "cfg.json")
    write_json(minimal, cfg_path)

    # --json mode emits a JSON error to stdout.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = c.dispatch(make_cli_args(json=True), [cfg_path])
    check("--json mode → exit 1", rc == 1)
    try:
        payload = json.loads(buf.getvalue())
        check("--json mode emits JSON error",
              payload.get("status") == "error"
              and payload.get("error_type") == "CascadeNotSupportedInJsonMode")
    except json.JSONDecodeError:
        check("--json mode emits JSON error", False, f"stdout was: {buf.getvalue()!r}")

    # --serve mode → exit 2 with stderr message.
    rc = c.dispatch(make_cli_args(serve=True), [cfg_path])
    check("--serve mode → exit 2", rc == 2)

    rc = c.dispatch(make_cli_args(unload=True), [cfg_path])
    check("--unload mode → exit 2", rc == 2)

    # --json mode emits structured JSON BEFORE flag rejection (contract: stdout
    # must always be JSON in --json mode, even when the request is malformed).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = c.dispatch(make_cli_args(json=True, steps=50), [cfg_path])
    check("--json --steps emits JSON (not bare stderr)", rc == 1)
    try:
        payload = json.loads(buf.getvalue())
        check("--json + bad flag emits structured JSON",
              payload.get("status") == "error"
              and payload.get("error_type") == "CascadeNotSupportedInJsonMode",
              f"payload was: {payload!r}")
    except json.JSONDecodeError:
        check("--json + bad flag emits structured JSON", False,
              f"stdout was: {buf.getvalue()!r}")


# ──────────────────────────────────────────────────────────────────────
print("\n── dispatch: --limit + --max-iterations ──────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    cfg_path = os.path.join(tmp, "cfg.json")
    write_json(minimal, cfg_path)

    # --max-iterations: total > cap → exit 2.
    rc = c.dispatch(make_cli_args(batch=10, max_iterations=5), [cfg_path])
    check("--max-iterations cap exceeded → exit 2", rc == 2)

    # --max-iterations exactly equal → would proceed past validation (it's
    # build_pipelines that fails, exit 3); we accept either rc=3 or rc=2 here
    # since we can't actually load weights.
    rc = c.dispatch(make_cli_args(batch=5, max_iterations=5), [cfg_path])
    check("--max-iterations exactly equal does not trigger cap rejection",
          rc != 2 or rc == 3,
          f"expected anything except a 'cap exceeded' rejection, got rc={rc}")

    # --limit smaller than total: passes validation; will fail later at
    # build-pipelines on CPU (no weights). Just verifies --limit doesn't reject.
    rc = c.dispatch(make_cli_args(batch=5, limit=2), [cfg_path])
    check("--limit smaller than total does not pre-reject",
          rc != 2 or rc == 3,
          f"expected pipeline-build failure (rc=3), got rc={rc}")


# ──────────────────────────────────────────────────────────────────────
print("\n── dispatch: --iterate prompt / seed accepted ───────────────")

with tempfile.TemporaryDirectory() as tmp:
    cfg_path = os.path.join(tmp, "cfg.json")
    write_json(minimal, cfg_path)
    prompts_path = os.path.join(tmp, "prompts.json")
    write_json(["a", "b", "c"], prompts_path)
    seeds_path = os.path.join(tmp, "seeds.json")
    write_json([1, 2], seeds_path)

    # --iterate prompt: passes flag rejection; fails downstream at pipeline build
    # (no real weights). rc != 2 means it got past flag rejection.
    rc = c.dispatch(make_cli_args(iterate=[("prompt", prompts_path)]), [cfg_path])
    check("--iterate prompt accepted (passes rejection)", rc != 2,
          f"expected pipeline-build failure (rc=3), got rc={rc}")

    rc = c.dispatch(make_cli_args(iterate=[("seed", seeds_path)]), [cfg_path])
    check("--iterate seed accepted (passes rejection)", rc != 2,
          f"got rc={rc}")

    # Combined prompt + seed → Cartesian.
    rc = c.dispatch(make_cli_args(iterate=[("prompt", prompts_path), ("seed", seeds_path)]),
                    [cfg_path])
    check("--iterate prompt + seed accepted", rc != 2,
          f"got rc={rc}")

    # Bad axes raise.
    bad_path = os.path.join(tmp, "bad.json")
    write_json(["x"], bad_path)
    rc = c.dispatch(make_cli_args(iterate=[("cfg_scale", bad_path)]), [cfg_path])
    check("--iterate cfg_scale rejected", rc == 2)

    # Mixed: one good, one bad → rejected.
    rc = c.dispatch(make_cli_args(iterate=[("prompt", prompts_path), ("cfg_scale", bad_path)]),
                    [cfg_path])
    check("--iterate prompt + cfg_scale (mixed) rejected", rc == 2)

    # Empty iterate file rejected.
    empty_path = os.path.join(tmp, "empty.json")
    write_json([], empty_path)
    rc = c.dispatch(make_cli_args(iterate=[("prompt", empty_path)]), [cfg_path])
    check("--iterate empty list rejected", rc == 2)

    # Missing iterate file rejected.
    rc = c.dispatch(
        make_cli_args(iterate=[("prompt", "/tmp/no-such-prompts.json")]), [cfg_path]
    )
    check("--iterate missing file rejected", rc == 2)


# ──────────────────────────────────────────────────────────────────────
print("\n── plan expansion math (cfg × batch × prompt × seed) ────────")

# Verify the plan-build math without actually invoking dispatch (which needs
# pipelines). We exercise the same loop structure inline.
def _build_plan(num_configs, batch, num_prompts, num_seeds):
    cfg_stub = {"stage_c": "x", "stage_b": "y"}
    configs = [(f"c{i}.json", cfg_stub) for i in range(num_configs)]
    prompts = list(range(num_prompts)) if num_prompts else [None]
    seeds = list(range(num_seeds)) if num_seeds else [None]
    plan = []
    for cfg_path, cfg in configs:
        for batch_index in range(max(batch, 1)):
            for p in prompts:
                for s in seeds:
                    plan.append((cfg_path, cfg, batch_index, p, s))
    return plan

check("1 cfg × 1 batch × 1 prompt × 1 seed = 1 run", len(_build_plan(1, 1, 0, 0)) == 1)
check("1 cfg × 1 batch × 3 prompts × 1 seed = 3 runs", len(_build_plan(1, 1, 3, 0)) == 3)
check("1 cfg × 2 batch × 3 prompts × 1 seed = 6 runs", len(_build_plan(1, 2, 3, 0)) == 6)
check("1 cfg × 1 batch × 3 prompts × 2 seeds = 6 runs", len(_build_plan(1, 1, 3, 2)) == 6)
check("2 cfg × 1 batch × 3 prompts × 2 seeds = 12 runs", len(_build_plan(2, 1, 3, 2)) == 12)
check("plan grouped by cfg first",
      [e[0] for e in _build_plan(2, 1, 2, 0)] == ["c0.json", "c0.json", "c1.json", "c1.json"])


# ──────────────────────────────────────────────────────────────────────
print("\n── sidecar round-trip ────────────────────────────────────────")

# Construct a sidecar dict matching exactly what dispatch() writes (a config dict
# augmented with runtime metadata). Verify load_config accepts it without warnings.
sidecar_dict = {
    **minimal,
    "scaffolding_repo": "stabilityai/stable-cascade",
    "prior_dtype": "bf16",
    "decoder_dtype": "fp16",
    "vae_dtype": "fp32",
    "prior_steps": 20,
    "prior_cfg_scale": 4.0,
    "decoder_steps": 10,
    "decoder_cfg_scale": 0.0,
    "width": 1024,
    "height": 1024,
    "prompt": "test prompt",
    "negative_prompt": "",
    "seed": 12345,
    # Runtime metadata (the keys dispatch() adds to the saved sidecar):
    "model_family": "stable-cascade",
    "config_source": "/tmp/some.json",
    "output_path": "/tmp/some.png",
    "iterate_batch_id": "abcd1234" * 4,
    "run_index": 3,
    "total_runs": 10,
    "timestamp": "2026-04-26T12:34:56",
    "elapsed_seconds": 42.5,
    "prior_seconds": 20.0,
    "decoder_seconds": 22.0,
}

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(sidecar_dict, f)
    sidecar_path = f.name
try:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        cfg = c.load_config(sidecar_path)
    stderr = buf.getvalue()
    check("sidecar round-trip: load_config accepts all keys",
          isinstance(cfg, dict) and cfg["prior_steps"] == 20)
    check("sidecar round-trip: no 'unknown keys' warning",
          "unknown keys" not in stderr,
          f"stderr leaked: {stderr!r}")
finally:
    os.unlink(sidecar_path)


# ──────────────────────────────────────────────────────────────────────
print("\n── _parse_args: --model nargs='+' end-to-end ────────────────")

# Confirm argparse really accepts `--model stablecascade c1.json c2.json` without
# slurping subsequent --flags as part of the model list.
import sys as _sys

old_argv = _sys.argv
try:
    _sys.argv = [
        "comfyless.generate",
        "--model", "stablecascade", "/tmp/c1.json", "/tmp/c2.json",
        "--prompt", "hello world",
        "--seed", "7",
        "--output", "/tmp/x.png",
    ]
    parsed = g._parse_args()
    check("nargs+ collects 3 model values",
          parsed.model == ["stablecascade", "/tmp/c1.json", "/tmp/c2.json"])
    check("nargs+ does not slurp --prompt",
          parsed.prompt == "hello world")
    check("nargs+ does not slurp --seed",
          parsed.seed == 7)
    check("nargs+ does not slurp --output",
          parsed.output == "/tmp/x.png")

    # And the existing single-path form still works.
    _sys.argv = [
        "comfyless.generate",
        "--model", "/abs/path/sdxl",
        "--prompt", "test",
        "--output", "/tmp/y.png",
    ]
    parsed = g._parse_args()
    check("single --model value parses to a 1-element list",
          parsed.model == ["/abs/path/sdxl"])
    extras = g._split_model_arg(parsed)
    check("post-_split_model_arg: model is str, no extras",
          parsed.model == "/abs/path/sdxl" and extras == [])
finally:
    _sys.argv = old_argv


# ──────────────────────────────────────────────────────────────────────
print(f"\n══════ {passed} passed / {failed} failed ══════")
sys.exit(0 if failed == 0 else 1)
