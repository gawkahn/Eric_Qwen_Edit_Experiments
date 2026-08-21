#!/usr/bin/env python3
"""Test harness for resolve_override_component_class (VAE class detection).

Covers the invariant behind letting a latent-compatible but differently-
classed VAE (e.g. AutoencoderKLWan) load as a --vae override onto a
Qwen-latent model like Krea-2 (whose own VAE is AutoencoderKLQwenImage),
instead of failing the base class's key-match guard at 0%.

The helper is pure config-reading — no weights, no GPU. Each promise has at
least one negative case:

  1. Override's own class preferred   — dir config._class_name overrides base.
  2. Single-file → base unchanged     — a lone .safetensors carries no config.
  3. Same class is a no-op            — override == base class name.
  4. vae/ subfolder detection         — full model dir, no top-level config.
  5. Unknown class → base fallback    — declared class absent from diffusers.
  6. No config anywhere → base        — bare dir with weights only.
  7. Top-level config wins over vae/  — resolution order is deterministic.
  8. Missing/blank _class_name skips  — falls through to the next candidate.
  9. Malformed JSON skips             — never raises, falls back.

Run: ./.venv/bin/python3 test_vae_override_class.py   (expect 0 failures)
"""

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import diffusers

# Load the utils module directly — dodges nodes/__init__.py (which imports
# every node class and expects a ComfyUI environment). Established pattern
# from the other suites.
_spec = importlib.util.spec_from_file_location(
    "edu", Path(__file__).parent / "comfyless" / "core" / "eric_diffusion_utils.py")
edu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(edu)

resolve = edu.resolve_override_component_class

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


# A sentinel base — the helper must return THIS object untouched when it
# falls back, so identity comparison proves no re-resolution happened.
BASE_SENTINEL = object()
BASE_NAME = "AutoencoderKLQwenImage"

# Real, always-present diffusers classes used as the "override's own" class.
assert hasattr(diffusers, "AutoencoderKL"), "diffusers.AutoencoderKL missing"


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


print("── resolve_override_component_class ──────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    # 1. Bare diffusers VAE repo whose config declares a DIFFERENT class.
    d = os.path.join(tmp, "wan_vae")
    os.makedirs(d)
    _write(os.path.join(d, "config.json"), {"_class_name": "AutoencoderKL"})
    cls, name = resolve(d, "vae", BASE_SENTINEL, BASE_NAME)
    check("1 override's own class preferred",
          cls is diffusers.AutoencoderKL and name == "AutoencoderKL",
          f"got ({cls}, {name})")

    # 2. Single-file override — no config → base returned unchanged.
    f = os.path.join(tmp, "vae.safetensors")
    Path(f).write_bytes(b"\x00")
    cls, name = resolve(f, "vae", BASE_SENTINEL, BASE_NAME)
    check("2 single-file falls back to base",
          cls is BASE_SENTINEL and name == BASE_NAME, f"got ({cls}, {name})")

    # 3. Override declares the SAME class as base — resolves to a real class,
    #    name unchanged (no spurious "differs" behavior downstream).
    d3 = os.path.join(tmp, "same_vae")
    os.makedirs(d3)
    _write(os.path.join(d3, "config.json"), {"_class_name": "AutoencoderKL"})
    cls, name = resolve(d3, "vae", BASE_SENTINEL, "AutoencoderKL")
    check("3 same class name resolves, name stable",
          cls is diffusers.AutoencoderKL and name == "AutoencoderKL",
          f"got ({cls}, {name})")

    # 4. Full model directory: no top-level config, class lives in vae/.
    d4 = os.path.join(tmp, "full_model")
    os.makedirs(os.path.join(d4, "vae"))
    _write(os.path.join(d4, "vae", "config.json"),
           {"_class_name": "AutoencoderKL"})
    cls, name = resolve(d4, "vae", BASE_SENTINEL, BASE_NAME)
    check("4 vae/ subfolder config detected",
          cls is diffusers.AutoencoderKL and name == "AutoencoderKL",
          f"got ({cls}, {name})")

    # 5. Declared class absent from installed diffusers → base fallback.
    d5 = os.path.join(tmp, "unknown_vae")
    os.makedirs(d5)
    _write(os.path.join(d5, "config.json"),
           {"_class_name": "TotallyNotARealVAE"})
    cls, name = resolve(d5, "vae", BASE_SENTINEL, BASE_NAME)
    check("5 unknown class falls back to base",
          cls is BASE_SENTINEL and name == BASE_NAME, f"got ({cls}, {name})")

    # 6. Bare directory with weights only, no config → base.
    d6 = os.path.join(tmp, "no_config")
    os.makedirs(d6)
    Path(os.path.join(d6, "diffusion_pytorch_model.safetensors")).write_bytes(b"\x00")
    cls, name = resolve(d6, "vae", BASE_SENTINEL, BASE_NAME)
    check("6 no config anywhere falls back to base",
          cls is BASE_SENTINEL and name == BASE_NAME, f"got ({cls}, {name})")

    # 7. Top-level config takes precedence over a vae/ subfolder config.
    d7 = os.path.join(tmp, "both_configs")
    os.makedirs(os.path.join(d7, "vae"))
    _write(os.path.join(d7, "config.json"), {"_class_name": "AutoencoderKL"})
    _write(os.path.join(d7, "vae", "config.json"),
           {"_class_name": "AutoencoderTiny"})
    cls, name = resolve(d7, "vae", BASE_SENTINEL, BASE_NAME)
    check("7 top-level config wins over vae/ subfolder",
          name == "AutoencoderKL", f"got ({cls}, {name})")

    # 8. Top-level config lacks _class_name → skip it, fall through to vae/.
    d8 = os.path.join(tmp, "blank_then_sub")
    os.makedirs(os.path.join(d8, "vae"))
    _write(os.path.join(d8, "config.json"), {"scaling_factor": 0.5})
    _write(os.path.join(d8, "vae", "config.json"),
           {"_class_name": "AutoencoderKL"})
    cls, name = resolve(d8, "vae", BASE_SENTINEL, BASE_NAME)
    check("8 blank _class_name skips to next candidate",
          cls is diffusers.AutoencoderKL and name == "AutoencoderKL",
          f"got ({cls}, {name})")

    # 9. Malformed JSON must not raise — fall back to base.
    d9 = os.path.join(tmp, "bad_json")
    os.makedirs(d9)
    with open(os.path.join(d9, "config.json"), "w") as fh:
        fh.write("{ not valid json ")
    try:
        cls, name = resolve(d9, "vae", BASE_SENTINEL, BASE_NAME)
        ok = cls is BASE_SENTINEL and name == BASE_NAME
    except Exception as e:
        ok = False
        name = f"raised {e!r}"
    check("9 malformed JSON falls back without raising", ok, f"got name={name}")

    # 10. _class_name resolves to a real diffusers attribute that is NOT a
    #     class (here the version string) → base fallback, no crash.
    d10 = os.path.join(tmp, "nonclass_attr")
    os.makedirs(d10)
    _write(os.path.join(d10, "config.json"), {"_class_name": "__version__"})
    cls, name = resolve(d10, "vae", BASE_SENTINEL, BASE_NAME)
    check("10 non-class diffusers attr falls back to base",
          cls is BASE_SENTINEL and name == BASE_NAME, f"got ({cls}, {name})")


print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
