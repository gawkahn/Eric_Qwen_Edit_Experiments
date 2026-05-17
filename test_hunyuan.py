#!/usr/bin/env python3
"""Test harness for the Hunyuan-Image 2.1 family (ADR-014).

Step 2 invariants (auto-detection + non-regression):
  - infer_model_family("HunyuanImagePipeline") == "hunyuan-image"
  - infer_model_family("HunyuanImageRefinerPipeline") == "hunyuan-image-refiner"
    (refiner string must NOT collapse into the base "hunyuan-image" slot —
    proves the _FAMILY_PATTERNS ordering per ADR-014 §3)
  - detect_pipeline_class on a fixture model_index.json with
    _class_name: "HunyuanImagePipeline" returns the right (class, name,
    family) triple
  - No existing pipeline-class string regressed by the new patterns
    (sweep of Qwen, Flux, Chroma, AuraFlow, SD*, ZImage classes)
  - Other Hunyuan classes (DiT, Video, Skyreels, Framepack) do NOT
    collapse into the Hunyuan-Image family slots — defends the ADR-014
    §3 class-roster audit at runtime

Later steps (3 + 4) will extend this file with CFG-routing and
family-defaults invariants.

Runs without ComfyUI, GPU, or loaded diffusion models. Uses the real
diffusers package (for class resolution in detect_pipeline_class) but
synthetic model_index.json fixtures only.
"""

import importlib.util
import json
import os
import sys
import tempfile
import types

import diffusers


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
    try:
        fn()
    except exc_types:
        check(name, True)
        return
    except Exception as e:
        check(name, False, f"expected {exc_types}, got {type(e).__name__}: {e}")
        return
    check(name, False, f"expected {exc_types}, got no exception")


# ── Defensive ComfyUI-side stubs (mirror test_samplers.py pattern) ─────────
# eric_diffusion_utils.py doesn't import folder_paths/comfy at module level —
# both are lazy-imported inside specific functions we don't call here — but
# the mocks are cheap insurance against a future import-site move.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
folder_paths_mock = types.ModuleType("folder_paths")
folder_paths_mock.get_folder_paths = lambda *a, **kw: []
folder_paths_mock.get_full_path = lambda *a, **kw: None
sys.modules["folder_paths"] = folder_paths_mock
for m in ("comfy", "comfy.utils", "comfy.model_management"):
    if m not in sys.modules:
        sys.modules[m] = types.ModuleType(m)

spec = importlib.util.spec_from_file_location(
    "eric_diffusion_utils", "nodes/eric_diffusion_utils.py"
)
utils_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_mod)


# ──────────────────────────────────────────────────────────────────────
print("── infer_model_family: Hunyuan-Image base + refiner ───────────")

check(
    "HunyuanImagePipeline → hunyuan-image",
    utils_mod.infer_model_family("HunyuanImagePipeline") == "hunyuan-image",
)
check(
    "HunyuanImageRefinerPipeline → hunyuan-image-refiner (NOT hunyuan-image)",
    utils_mod.infer_model_family("HunyuanImageRefinerPipeline") == "hunyuan-image-refiner",
)
# Defensive: make sure the refiner string truly does not collapse into the
# base slot. The first-match-wins ordering in _FAMILY_PATTERNS is the only
# thing standing between "ship correctly" and "ship the refiner routed
# through the base CFG branch."
check(
    "refiner does NOT collapse into hunyuan-image",
    utils_mod.infer_model_family("HunyuanImageRefinerPipeline") != "hunyuan-image",
)


# ──────────────────────────────────────────────────────────────────────
print("── infer_model_family: existing family non-regression sweep ───")

# One assertion per currently-supported diffusers pipeline class that
# round-trips through _FAMILY_PATTERNS. If a future entry mis-orders the
# list, one of these assertions catches it before commit.
EXISTING_FAMILY_FIXTURES = [
    ("QwenImageEditPlusPipeline",   "qwen-edit"),
    ("QwenImagePipeline",           "qwen-image"),
    ("Flux2KleinPipeline",          "flux2klein"),
    ("Flux2Pipeline",               "flux2"),
    ("ChromaPipeline",              "chroma"),
    ("FluxPipeline",                "flux"),
    ("AuraFlowPipeline",            "auraflow"),
    ("StableDiffusion3Pipeline",    "sd3"),
    ("StableDiffusionXLPipeline",   "sdxl"),
    ("StableDiffusionPipeline",     "sd1"),
    ("ZImagePipeline",              "zimage"),
]

for class_name, expected_family in EXISTING_FAMILY_FIXTURES:
    actual = utils_mod.infer_model_family(class_name)
    check(
        f"{class_name} → {expected_family}",
        actual == expected_family,
        f"got {actual!r}",
    )


# ──────────────────────────────────────────────────────────────────────
print("── infer_model_family: unrelated Hunyuan classes do NOT collide")

# Per ADR-014 §3's class-roster audit: every other Hunyuan* class shipped
# by diffusers 0.37.1 must NOT collapse into either Hunyuan-Image family
# slot. This locks the audit at runtime so a future diffusers release that
# ships a new Hunyuan* class can't silently introduce a collision.
UNRELATED_HUNYUAN_CLASSES = [
    "HunyuanDiTPipeline",
    "HunyuanDiTControlNetPipeline",
    "HunyuanDiTPAGPipeline",
    "HunyuanVideoPipeline",
    "HunyuanVideo15Pipeline",
    "HunyuanVideoImageToVideoPipeline",
    "HunyuanVideo15ImageToVideoPipeline",
    "HunyuanSkyreelsImageToVideoPipeline",
    "HunyuanVideoFramepackPipeline",
]

for class_name in UNRELATED_HUNYUAN_CLASSES:
    fam = utils_mod.infer_model_family(class_name)
    check(
        f"{class_name} does not collapse into Hunyuan-Image slots",
        fam not in ("hunyuan-image", "hunyuan-image-refiner"),
        f"got {fam!r}",
    )


# ──────────────────────────────────────────────────────────────────────
print("── infer_model_family: fully-unknown class falls back gracefully")

# Documented behavior (ADR-003 introspection-fallback boundary): an
# unknown class name returns its lowercase-stripped form, not a raise.
# Locks the "no hard error on unknown family" guarantee that the
# introspection fallback in _build_call_kwargs depends on.
check(
    "FooBarPipeline returns lowercase fallback",
    utils_mod.infer_model_family("FooBarPipeline") == "foobarpipeline",
)


# ──────────────────────────────────────────────────────────────────────
print("── detect_pipeline_class: end-to-end via model_index.json fixture")

with tempfile.TemporaryDirectory() as tmpdir:
    # Minimal model_index.json — detect_pipeline_class only reads
    # _class_name; the full diffusers manifest isn't needed because we
    # don't instantiate the pipeline here.
    fixture_path = os.path.join(tmpdir, "model_index.json")
    with open(fixture_path, "w") as f:
        json.dump({"_class_name": "HunyuanImagePipeline"}, f)

    pipeline_class, class_name, family = utils_mod.detect_pipeline_class(tmpdir)
    check(
        "detect returns diffusers.HunyuanImagePipeline class",
        pipeline_class is diffusers.HunyuanImagePipeline,
        f"got {pipeline_class}",
    )
    check(
        "detect returns class_name='HunyuanImagePipeline'",
        class_name == "HunyuanImagePipeline",
        f"got {class_name!r}",
    )
    check(
        "detect returns family='hunyuan-image'",
        family == "hunyuan-image",
        f"got {family!r}",
    )

# Negative — directory with no model_index.json must raise ValueError
# (unchanged fail-closed behavior; defends Vision invariant 5's "no new
# caller-surface widening" by proving the existing failure semantics
# still hold under the new family entries).
with tempfile.TemporaryDirectory() as tmpdir:
    expect_raises(
        "detect raises ValueError on missing model_index.json",
        lambda: utils_mod.detect_pipeline_class(tmpdir),
        ValueError,
    )

# Negative — model_index.json with an unknown _class_name must raise
# ValueError. This is the diffusers-not-installed path; it must continue
# to raise even after the new family entries land.
with tempfile.TemporaryDirectory() as tmpdir:
    bad_fixture = os.path.join(tmpdir, "model_index.json")
    with open(bad_fixture, "w") as f:
        json.dump({"_class_name": "FictitiousNonexistentPipeline"}, f)
    expect_raises(
        "detect raises ValueError on unknown class name",
        lambda: utils_mod.detect_pipeline_class(tmpdir),
        ValueError,
    )


# ──────────────────────────────────────────────────────────────────────
print(f"\n────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print(f"────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
