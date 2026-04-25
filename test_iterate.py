#!/usr/bin/env python3
"""Test harness for comfyless iteration mode (ADR-008).

Exercises the pure-logic helpers in comfyless/generate.py:
  - _validate_iterate_value      (element-shape validator)
  - _plan_iterations             (argparse → plan dict; error cases)
  - _iteration_combos            (Cartesian expansion)
  - _expand_iterate_tokens       (%input% / %input_<param>% client-side)
  - _expand_savepath_template    (full template + iterate_inputs kwarg)

Runs without ComfyUI, GPU, or loaded diffusion models — the comfyless
package installs its own shims for folder_paths / comfy.utils so the
module imports cleanly.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyless.generate as g


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


def make_args(**overrides):
    """argparse.Namespace with defaults matching _parse_args surface for iter-planning."""
    defaults = dict(iterate=[], max_iterations=500, yes=True)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


# ──────────────────────────────────────────────────────────────────────
print("── _validate_iterate_value ────────────────────────────────────")

check("str shape accepts strings",
      g._validate_iterate_value("hello", str) is True)
check("str shape rejects non-str",
      g._validate_iterate_value(42, str) is False)

check("int shape accepts plain int",
      g._validate_iterate_value(42, int) is True)
check("int shape rejects bool (subclass trap)",
      g._validate_iterate_value(True, int) is False,
      "bool is a subclass of int — validator must explicitly exclude it")
check("int shape rejects float",
      g._validate_iterate_value(3.14, int) is False)

check("number shape accepts int",
      g._validate_iterate_value(4, "number") is True)
check("number shape accepts float",
      g._validate_iterate_value(4.5, "number") is True)
check("number shape rejects bool",
      g._validate_iterate_value(False, "number") is False)
check("number shape rejects str",
      g._validate_iterate_value("4.5", "number") is False)

check("lora_stack accepts empty list",
      g._validate_iterate_value([], "lora_stack") is True,
      "empty stack == 'no LoRA this iteration'")
check("lora_stack accepts single-item stack",
      g._validate_iterate_value(
          [{"path": "/x.safetensors", "weight": 0.8}],
          "lora_stack",
      ) is True)
check("lora_stack accepts stack without weight",
      g._validate_iterate_value(
          [{"path": "/x.safetensors"}],
          "lora_stack",
      ) is True)
check("lora_stack rejects non-list top-level",
      g._validate_iterate_value({"path": "/x.safetensors"}, "lora_stack") is False)
check("lora_stack rejects item without path",
      g._validate_iterate_value([{"weight": 0.8}], "lora_stack") is False)
check("lora_stack rejects non-str path",
      g._validate_iterate_value([{"path": 42}], "lora_stack") is False)
check("lora_stack rejects bool weight",
      g._validate_iterate_value(
          [{"path": "/x.safetensors", "weight": True}],
          "lora_stack",
      ) is False)


# ──────────────────────────────────────────────────────────────────────
print("\n── _plan_iterations ───────────────────────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    prompts = os.path.join(tmp, "prompts.json")
    write_json(["a forest at dawn", "a mountain at dusk", "a river at noon"], prompts)

    seeds = os.path.join(tmp, "seeds.json")
    write_json([42, 1337, 9999], seeds)

    loras = os.path.join(tmp, "loras.json")
    write_json([
        [],
        [{"path": "/loras/style_a.safetensors", "weight": 0.8}],
    ], loras)

    bad_shape = os.path.join(tmp, "bad_shape.json")
    write_json([42, "not a number", 100], bad_shape)

    not_a_list = os.path.join(tmp, "not_a_list.json")
    write_json({"prompts": ["a", "b"]}, not_a_list)

    empty_list = os.path.join(tmp, "empty.json")
    write_json([], empty_list)

    # Happy path: single axis
    plan = g._plan_iterations(make_args(iterate=[["prompt", prompts]]))
    check("single-axis plan: total=len(values)",
          plan is not None and plan["total"] == 3)
    check("single-axis plan: axes recorded",
          plan["axes"][0][0] == "prompt" and plan["axes"][0][1] == "prompts")
    check("single-axis plan: _primary input = first axis stem",
          plan["input_tokens"]["_primary"] == "prompts")
    check("single-axis plan: per-axis input recorded",
          plan["input_tokens"]["prompt"] == "prompts")

    # Happy path: multi-axis Cartesian
    plan = g._plan_iterations(make_args(iterate=[["prompt", prompts], ["seed", seeds]]))
    check("multi-axis plan: total = product",
          plan["total"] == 9,
          f"expected 3*3=9, got {plan['total'] if plan else None}")
    check("multi-axis plan: _primary = first axis",
          plan["input_tokens"]["_primary"] == "prompts")
    check("multi-axis plan: per-axis tokens both present",
          plan["input_tokens"]["prompt"] == "prompts"
          and plan["input_tokens"]["seed"] == "seeds")

    # Happy path: lora axis
    plan = g._plan_iterations(make_args(iterate=[["lora", loras]]))
    check("lora-axis plan accepts stacks",
          plan is not None and plan["total"] == 2)

    # No --iterate → None
    check("no --iterate returns None",
          g._plan_iterations(make_args()) is None)

    # Error: unknown param
    try:
        g._plan_iterations(make_args(iterate=[["totally_unknown", prompts]]))
        check("unknown param raises ValueError", False, "did not raise")
    except ValueError as e:
        check("unknown param raises ValueError",
              "totally_unknown" in str(e) and "not supported" in str(e))

    # Error: file doesn't exist
    try:
        g._plan_iterations(make_args(iterate=[["prompt", "/nonexistent.json"]]))
        check("missing file raises ValueError", False, "did not raise")
    except ValueError as e:
        check("missing file raises ValueError",
              "/nonexistent.json" in str(e))

    # Error: file not JSON list
    try:
        g._plan_iterations(make_args(iterate=[["prompt", not_a_list]]))
        check("non-list top-level raises ValueError", False, "did not raise")
    except ValueError as e:
        check("non-list top-level raises ValueError",
              "must be a JSON list" in str(e))

    # Error: empty list
    try:
        g._plan_iterations(make_args(iterate=[["prompt", empty_list]]))
        check("empty list raises ValueError", False, "did not raise")
    except ValueError as e:
        check("empty list raises ValueError", "empty list" in str(e))

    # Error: wrong element shape
    try:
        g._plan_iterations(make_args(iterate=[["seed", bad_shape]]))
        check("wrong element shape raises ValueError", False, "did not raise")
    except ValueError as e:
        check("wrong element shape raises ValueError",
              "expected int" in str(e) and "element [1]" in str(e))

    # Error: max_iterations exceeded
    try:
        g._plan_iterations(make_args(
            iterate=[["prompt", prompts], ["seed", seeds]],
            max_iterations=5,
        ))
        check("cap exceeded raises ValueError", False, "did not raise")
    except ValueError as e:
        check("cap exceeded raises ValueError",
              "exceeds --max-iterations=5" in str(e))


# ──────────────────────────────────────────────────────────────────────
print("\n── _iteration_combos ──────────────────────────────────────────")

plan_mock = {
    "axes": [("prompt", "prompts", ["a", "b", "c"]),
             ("seed",   "seeds",   [1, 2])],
    "total": 6,
    "input_tokens": {"prompt": "prompts", "seed": "seeds", "_primary": "prompts"},
}
combos = list(g._iteration_combos(plan_mock))
check("combos: total count = product",
      len(combos) == 6)
check("combos: first combo is first-of-each",
      combos[0] == {"prompt": "a", "seed": 1})
check("combos: axis order preserved (prompt before seed)",
      list(combos[0].keys()) == ["prompt", "seed"])
check("combos: last combo is last-of-each",
      combos[-1] == {"prompt": "c", "seed": 2})

single_axis = {
    "axes": [("lora", "loras", [[], [{"path": "/a.sft", "weight": 1.0}]])],
    "total": 2,
    "input_tokens": {"lora": "loras", "_primary": "loras"},
}
combos = list(g._iteration_combos(single_axis))
check("combos: lora axis values preserved (list-of-dicts)",
      combos[0] == {"lora": []} and combos[1] == {"lora": [{"path": "/a.sft", "weight": 1.0}]})


# ──────────────────────────────────────────────────────────────────────
print("\n── _expand_iterate_tokens ────────────────────────────────────")

inputs = {"prompt": "my_prompts", "seed": "my_seeds", "_primary": "my_prompts"}

check("%input% → _primary",
      g._expand_iterate_tokens("out/%input%/gen", inputs) == "out/my_prompts/gen")
check("%input_prompt% → that axis stem",
      g._expand_iterate_tokens("%input_prompt%-%input_seed%", inputs)
      == "my_prompts-my_seeds")
check("unknown axis token → empty string",
      g._expand_iterate_tokens("%input_unknown%", inputs) == "")
check("non-iterate tokens left untouched",
      g._expand_iterate_tokens("%date:YYYY-MM-dd%/%model%/%input%", inputs)
      == "%date:YYYY-MM-dd%/%model%/my_prompts")
check("empty iterate_inputs → empty for %input%",
      g._expand_iterate_tokens("%input%", {}) == "")
check("case-insensitive match",
      g._expand_iterate_tokens("%INPUT%-%Input_Prompt%", inputs)
      == "my_prompts-my_prompts")


# ──────────────────────────────────────────────────────────────────────
print("\n── _expand_savepath_template (with iterate_inputs) ────────────")

out = g._expand_savepath_template(
    template="%date:YYYY-MM-dd%/%input%/%model:6%-seed%seed%",
    model_path="/models/Qwen-Image-2512",
    seed=42,
    steps=50,
    cfg_scale=4.0,
    sampler="default",
    transformer_path="",
    iterate_inputs=inputs,
)
# Only assert the iterate-token expansion; date is time-dependent.
check("savepath template: %input% expanded with iterate_inputs",
      "/my_prompts/" in out and "-seed42" in out)
check("savepath template: %model:6% truncated",
      "Qwen-I" in out)

out2 = g._expand_savepath_template(
    template="%input_unknown%-%seed%",
    model_path="/m",
    seed=42,
    steps=50,
    cfg_scale=4.0,
    sampler="default",
)
check("savepath template: iterate_inputs omitted → %input_*% → empty",
      out2 == "-42")


# ──────────────────────────────────────────────────────────────────────
print("\n── _iteration_replaces_loras ──────────────────────────────────")

plan_with_lora = {
    "axes": [("lora", "loras", [[], [{"path": "/a.sft", "weight": 1.0}]])],
    "total": 2,
    "input_tokens": {"lora": "loras", "_primary": "loras"},
}
plan_without_lora = {
    "axes": [("prompt", "prompts", ["a", "b"])],
    "total": 2,
    "input_tokens": {"prompt": "prompts", "_primary": "prompts"},
}
some_base_loras = [{"path": "/base.sft", "weight": 0.5}]

check("replaces: plan with lora axis + base loras → warn",
      g._iteration_replaces_loras(plan_with_lora, some_base_loras) is True)
check("replaces: plan with lora axis + empty base loras → no warn",
      g._iteration_replaces_loras(plan_with_lora, []) is False,
      "no base loras means there's nothing to replace")
check("replaces: plan without lora axis + base loras → no warn",
      g._iteration_replaces_loras(plan_without_lora, some_base_loras) is False,
      "base loras pass through unchanged when no lora axis iterates")
check("replaces: plan is None → no warn",
      g._iteration_replaces_loras(None, some_base_loras) is False)


# ──────────────────────────────────────────────────────────────────────
print("\n── --iterate satisfies required fields (subprocess) ──────────")

# Regression: `--iterate prompt` with no --prompt used to fail the
# "-- prompt is required" gate because the planning step ran too late.
# Likewise "--iterate model" should satisfy --model on its own.
with tempfile.TemporaryDirectory() as tmp:
    prompts = os.path.join(tmp, "prompts.json")
    write_json(["hello", "world"], prompts)

    # Without --prompt on CLI but with --iterate prompt: the gate must NOT
    # fail with "--prompt is required". Use a nonexistent model so the run
    # exits before any real generation; the required-field check fires
    # before the path-resolve step, so its absence in stderr proves the
    # gate accepts iterated prompts.
    proc = subprocess.run(
        [sys.executable, "-m", "comfyless.generate",
         "--model", "/nonexistent/model/path",
         "--iterate", "prompt", prompts,
         "--yes"],
        input="",
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    check("--iterate prompt: does NOT fail with '--prompt is required'",
          "--prompt is required" not in proc.stderr,
          f"stderr={proc.stderr[:300]!r}")

    # Symmetrically, --iterate model should satisfy --model.
    models = os.path.join(tmp, "models.json")
    write_json(["/a", "/b"], models)
    proc = subprocess.run(
        [sys.executable, "-m", "comfyless.generate",
         "--prompt", "hello",
         "--iterate", "model", models,
         "--yes"],
        input="",
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    check("--iterate model: does NOT fail with '--model is required'",
          "--model is required" not in proc.stderr,
          f"stderr={proc.stderr[:300]!r}")

    # Negative: iterating an unrelated axis (seed) still requires --prompt/--model.
    seeds = os.path.join(tmp, "seeds.json")
    write_json([1, 2, 3], seeds)
    proc = subprocess.run(
        [sys.executable, "-m", "comfyless.generate",
         "--iterate", "seed", seeds,
         "--yes"],
        input="",
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    check("--iterate seed (no --model/--prompt): still fails with required-field error",
          "--model is required" in proc.stderr,
          f"stderr={proc.stderr[:300]!r}")


# ──────────────────────────────────────────────────────────────────────
print("\n── --json + --iterate rejection (subprocess) ──────────────────")

with tempfile.TemporaryDirectory() as tmp:
    prompts = os.path.join(tmp, "prompts.json")
    write_json(["a", "b", "c"], prompts)

    # Feed a minimal JSON request on stdin so --json mode has something to parse
    # in the no-iterate control case; it won't get used when --iterate is set.
    stdin_req = json.dumps({"contract_version": 1,
                            "model": "/nonexistent",
                            "prompt": "test"})

    proc = subprocess.run(
        [sys.executable, "-m", "comfyless.generate",
         "--json", "--iterate", "prompt", prompts],
        input=stdin_req,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    check("--json + --iterate: exit code 1",
          proc.returncode == 1,
          f"rc={proc.returncode}, stderr={proc.stderr[:200]!r}")
    try:
        payload = json.loads(proc.stdout)
        ok_shape = (payload.get("status") == "error"
                    and payload.get("error_type") == "IterationNotSupported"
                    and "not supported in --json mode" in payload.get("error", "")
                    and "contract_version" in payload)
    except json.JSONDecodeError:
        ok_shape = False
        payload = proc.stdout
    check("--json + --iterate: stdout is contract-shaped error",
          ok_shape,
          f"stdout={proc.stdout[:300]!r}")
    check("--json + --iterate: message includes the ADR-specified phrasing",
          "iteration semantics will be added" in proc.stdout)


# ──────────────────────────────────────────────────────────────────────
print("\n── _plan_iterations: --limit ────────────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    prompts = os.path.join(tmp, "prompts.json")
    write_json(["a", "b", "c", "d", "e"], prompts)

    # --limit truncates Cartesian to first N; ceiling, not requirement.
    plan = g._plan_iterations(make_args(
        iterate=[["prompt", prompts]], limit=3,
    ))
    check("--limit 3 on 5 prompts: cartesian = 5",
          plan["cartesian"] == 5)
    check("--limit 3 on 5 prompts: effective_combos = 3",
          plan["effective_combos"] == 3)
    check("--limit 3 on 5 prompts: total = 3 (batch default 1)",
          plan["total"] == 3)

    # limit > cartesian: clamps, no error.
    plan = g._plan_iterations(make_args(
        iterate=[["prompt", prompts]], limit=10,
    ))
    check("--limit 10 on 5 prompts: effective = 5 (clamps, no error)",
          plan["effective_combos"] == 5 and plan["total"] == 5)

    # limit applies BEFORE max-iterations check.
    plan = g._plan_iterations(make_args(
        iterate=[["prompt", prompts]], limit=2, max_iterations=2,
    ))
    check("--limit 2 + --max-iterations 2: succeeds (limit caps first)",
          plan["total"] == 2)

    # limit = cartesian: no-op.
    plan = g._plan_iterations(make_args(
        iterate=[["prompt", prompts]], limit=5,
    ))
    check("--limit equal to cartesian: total unchanged",
          plan["total"] == 5)


# ──────────────────────────────────────────────────────────────────────
print("\n── _plan_iterations: --batch ────────────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    prompts = os.path.join(tmp, "prompts.json")
    write_json(["a", "b", "c"], prompts)

    # --batch alone (no --iterate): degenerate plan, axes empty, total = batch.
    plan = g._plan_iterations(make_args(batch=5))
    check("--batch 5 alone: plan returned (not None)",
          plan is not None)
    check("--batch 5 alone: axes empty",
          plan["axes"] == [])
    check("--batch 5 alone: cartesian = 1",
          plan["cartesian"] == 1)
    check("--batch 5 alone: effective_combos = 1",
          plan["effective_combos"] == 1)
    check("--batch 5 alone: total = 5",
          plan["total"] == 5)
    check("--batch 5 alone: input_tokens has no _primary",
          "_primary" not in plan["input_tokens"])

    # --batch 1 with no --iterate: no plan (back to single-gen path).
    plan = g._plan_iterations(make_args(batch=1))
    check("--batch 1 alone: plan is None (no plan needed)",
          plan is None)

    # --batch with --iterate: multiplies.
    plan = g._plan_iterations(make_args(
        iterate=[["prompt", prompts]], batch=4,
    ))
    check("--batch 4 + 3-prompt iterate: cartesian = 3",
          plan["cartesian"] == 3)
    check("--batch 4 + 3-prompt iterate: effective_combos = 3",
          plan["effective_combos"] == 3)
    check("--batch 4 + 3-prompt iterate: total = 12 (3 × 4)",
          plan["total"] == 12)


# ──────────────────────────────────────────────────────────────────────
print("\n── _plan_iterations: --limit + --batch interaction ──────────")

with tempfile.TemporaryDirectory() as tmp:
    prompts = os.path.join(tmp, "prompts.json")
    write_json(list("abcdefghij"), prompts)  # 10 prompts

    # --limit 2 --batch 3 on 10 prompts: 2 × 3 = 6 total.
    plan = g._plan_iterations(make_args(
        iterate=[["prompt", prompts]], limit=2, batch=3,
    ))
    check("--limit 2 --batch 3: effective_combos = 2 (limit applies first)",
          plan["effective_combos"] == 2)
    check("--limit 2 --batch 3: total = 6 (limit × batch)",
          plan["total"] == 6)

    # max-iterations applies to TOTAL, not pre-batch.
    try:
        g._plan_iterations(make_args(
            iterate=[["prompt", prompts]], limit=2, batch=10,
            max_iterations=5,
        ))
        check("--limit 2 --batch 10 vs --max-iterations 5: raises ValueError",
              False, "did not raise")
    except ValueError as e:
        check("--limit 2 --batch 10 vs --max-iterations 5: raises ValueError",
              "20 iterations exceeds --max-iterations=5" in str(e))


# ──────────────────────────────────────────────────────────────────────
print("\n── _iteration_combos: --batch repetition ────────────────────")

# Pure-batch plan: empty axes, batch > 1 → yield empty dicts.
batch_only_plan = {
    "axes": [],
    "cartesian": 1,
    "effective_combos": 1,
    "batch": 4,
    "total": 4,
    "input_tokens": {},
}
combos = list(g._iteration_combos(batch_only_plan))
check("pure --batch 4: yields 4 combos",
      len(combos) == 4)
check("pure --batch 4: each combo is empty dict (use base config)",
      all(c == {} for c in combos))

# Iterate + batch: each combo yielded `batch` times consecutively.
batch_iterate_plan = {
    "axes": [("prompt", "p", ["a", "b"])],
    "cartesian": 2,
    "effective_combos": 2,
    "batch": 3,
    "total": 6,
    "input_tokens": {"prompt": "p", "_primary": "p"},
}
combos = list(g._iteration_combos(batch_iterate_plan))
check("--batch 3 + 2-prompt iterate: yields 6 combos",
      len(combos) == 6)
check("--batch 3 + 2-prompt iterate: first 3 combos are prompt 'a'",
      [c["prompt"] for c in combos[:3]] == ["a", "a", "a"])
check("--batch 3 + 2-prompt iterate: last 3 combos are prompt 'b'",
      [c["prompt"] for c in combos[3:]] == ["b", "b", "b"])

# Limit + batch: only `effective_combos` distinct combos, each repeated `batch` times.
limit_batch_plan = {
    "axes": [("prompt", "p", ["a", "b", "c", "d", "e"])],
    "cartesian": 5,
    "effective_combos": 2,  # --limit 2
    "batch": 3,
    "total": 6,
    "input_tokens": {"prompt": "p", "_primary": "p"},
}
combos = list(g._iteration_combos(limit_batch_plan))
check("--limit 2 + --batch 3 + 5-prompt iterate: yields 6 combos",
      len(combos) == 6)
check("--limit 2 + --batch 3: only first 2 prompts appear",
      set(c["prompt"] for c in combos) == {"a", "b"})


# ──────────────────────────────────────────────────────────────────────
print("\n── _positive_int argparse type ──────────────────────────────")

import argparse as _argparse
try:
    g._positive_int("5")
    check("_positive_int('5') = 5", g._positive_int("5") == 5)
except _argparse.ArgumentTypeError:
    check("_positive_int('5') = 5", False, "raised unexpectedly")

try:
    g._positive_int("0")
    check("_positive_int('0') raises", False, "did not raise")
except _argparse.ArgumentTypeError:
    check("_positive_int('0') raises", True)

try:
    g._positive_int("-3")
    check("_positive_int('-3') raises", False, "did not raise")
except _argparse.ArgumentTypeError:
    check("_positive_int('-3') raises", True)

try:
    g._positive_int("notanumber")
    check("_positive_int('notanumber') raises", False, "did not raise")
except _argparse.ArgumentTypeError:
    check("_positive_int('notanumber') raises", True)


# ──────────────────────────────────────────────────────────────────────
print("\n── --json + --batch / --limit rejection (subprocess) ──────")

# Same rejection shape as --iterate: ADR-008 says iteration semantics
# (including --batch and --limit) aren't expressible in the JSON contract v1.
stdin_req = json.dumps({"contract_version": 1,
                        "model": "/nonexistent",
                        "prompt": "test"})

for flag, value in [("--batch", "5"), ("--limit", "5")]:
    proc = subprocess.run(
        [sys.executable, "-m", "comfyless.generate",
         "--json", flag, value],
        input=stdin_req,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    check(f"--json + {flag} {value}: exit code 1",
          proc.returncode == 1)
    try:
        payload = json.loads(proc.stdout)
        ok_shape = (payload.get("status") == "error"
                    and payload.get("error_type") == "IterationNotSupported"
                    and "contract_version" in payload)
    except json.JSONDecodeError:
        ok_shape = False
    check(f"--json + {flag} {value}: contract-shaped IterationNotSupported error",
          ok_shape, f"stdout={proc.stdout[:300]!r}")


# ──────────────────────────────────────────────────────────────────────
print("\n── --limit / --batch CLI integration (subprocess) ───────────")

with tempfile.TemporaryDirectory() as tmp:
    prompts = os.path.join(tmp, "prompts.json")
    write_json(["a", "b", "c"], prompts)

    # --limit 0 → argparse error
    proc = subprocess.run(
        [sys.executable, "-m", "comfyless.generate",
         "--prompt", "x", "--model", "/nonexistent",
         "--iterate", "prompt", prompts, "--limit", "0", "--yes"],
        input="", capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    check("--limit 0: argparse rejects",
          proc.returncode != 0 and "positive integer" in proc.stderr)

    # --batch 0 → argparse error
    proc = subprocess.run(
        [sys.executable, "-m", "comfyless.generate",
         "--prompt", "x", "--model", "/nonexistent",
         "--batch", "0", "--yes"],
        input="", capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    check("--batch 0: argparse rejects",
          proc.returncode != 0 and "positive integer" in proc.stderr)

    # --batch alone with no --iterate, no --prompt: still hits required-field gate.
    # This pins that --batch (unlike --iterate prompt) does NOT supply the prompt.
    proc = subprocess.run(
        [sys.executable, "-m", "comfyless.generate",
         "--model", "/nonexistent",
         "--batch", "3", "--yes"],
        input="", capture_output=True, text=True,
        cwd=str(Path(__file__).parent),
    )
    check("--batch 3 with no --prompt: still hits required-field gate",
          "--prompt is required" in proc.stderr)


# ──────────────────────────────────────────────────────────────────────
print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
