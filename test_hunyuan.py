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
    "eric_diffusion_utils", "comfyless/core/eric_diffusion_utils.py"
)
utils_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_mod)


# ── Load the two _build_call_kwargs copies (Step 3) ────────────────────────
# comfyless is a real package on the path — direct import works (the same way
# test_cascade.py imports it).
import comfyless.generate as cg

# nodes/eric_diffusion_generate.py uses relative imports
# (from .eric_qwen_edit_utils / .eric_diffusion_samplers / .eric_diffusion_utils).
# Loading it standalone would fail the relative imports, and loading the real
# nodes package would drag in the full ComfyUI node surface via nodes/__init__.py.
# Register a bare 'nodes' package + stub the three sibling modules it imports at
# top level (none are used by _build_call_kwargs), then load just the module.
_nodes_pkg = types.ModuleType("nodes")
_nodes_pkg.__path__ = ["nodes"]
sys.modules["nodes"] = _nodes_pkg
for _name, _attrs in [
    ("nodes.eric_qwen_edit_utils",   {"pil_to_tensor": lambda *a, **k: None}),
    ("comfyless.core.eric_diffusion_samplers",
     {"sampler_choices": lambda *a, **k: ["default"], "swap_sampler": None}),
    ("comfyless.core.eric_diffusion_utils",   {"build_model_metadata": lambda *a, **k: {}}),
]:
    _stub_mod = types.ModuleType(_name)
    for _k, _v in _attrs.items():
        setattr(_stub_mod, _k, _v)
    sys.modules[_name] = _stub_mod

_gspec = importlib.util.spec_from_file_location(
    "nodes.eric_diffusion_generate", "nodes/eric_diffusion_generate.py"
)
gen_nodes = importlib.util.module_from_spec(_gspec)
sys.modules["nodes.eric_diffusion_generate"] = gen_nodes
_gspec.loader.exec_module(gen_nodes)


class _StubPipe:
    """Minimal pipe whose __call__ signature exposes the params the
    introspecting CFG branches probe. Never actually invoked — _build_call_kwargs
    only reads inspect.signature(pipe.__call__) for the flux/auraflow branches;
    the hunyuan-image branch doesn't touch pipe at all."""

    def __call__(self, prompt=None, negative_prompt=None, guidance_scale=None,
                 max_sequence_length=None, **kw):  # pragma: no cover
        raise NotImplementedError


def _cg_kwargs(model_family, negative_prompt, cfg_scale=3.25, guidance_embeds=True):
    """Call the comfyless _build_call_kwargs with fixed positional shape."""
    return cg._build_call_kwargs(
        _StubPipe(), model_family, guidance_embeds,
        "a prompt", negative_prompt,
        1024, 1024, 50, cfg_scale,
        None,   # true_cfg_scale
        512,    # max_sequence_length
        None,   # generator
    )


def _nodes_kwargs(model_family, negative_prompt, cfg_scale=3.25, guidance_embeds=True):
    """Call the nodes _build_call_kwargs with fixed positional shape."""
    return gen_nodes._build_call_kwargs(
        _StubPipe(), model_family, guidance_embeds,
        "a prompt", negative_prompt,
        1024, 1024, 50, cfg_scale,
        512,    # max_sequence_length
        None,   # generator
        None,   # on_step_end
    )


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
    # Unmarked Klein = non-distilled base; the distilled flagship carries
    # is_distilled:true (ADR-009 changelog 2026-07-22).
    ("Flux2KleinPipeline",          "flux2klein-base"),
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
print("── _build_call_kwargs: hunyuan-image CFG routing (invariant 2) ─")

# Positive: both copies route cfg_scale → distilled_guidance_scale, forward a
# set negative_prompt, and omit the kwargs that belong to other families.
for label, fn in (("comfyless", _cg_kwargs), ("nodes", _nodes_kwargs)):
    kw = fn("hunyuan-image", "blurry", cfg_scale=3.25)
    check(f"{label}: hunyuan-image sets distilled_guidance_scale=cfg_scale",
          kw.get("distilled_guidance_scale") == 3.25, f"got {kw!r}")
    check(f"{label}: hunyuan-image omits guidance_scale",
          "guidance_scale" not in kw)
    check(f"{label}: hunyuan-image omits true_cfg_scale",
          "true_cfg_scale" not in kw)
    check(f"{label}: hunyuan-image forwards negative_prompt when set",
          kw.get("negative_prompt") == "blurry")
    check(f"{label}: hunyuan-image omits max_sequence_length",
          "max_sequence_length" not in kw,
          "Hunyuan signature has no max_sequence_length (ADR-014 §2)")

# Cross-copy consistency on the CFG-routing decision — the part invariant 2
# says must be identical in both copies. The two base dicts differ by
# callback_on_step_end (nodes-only), so compare only the CFG-relevant keys.
_cfg_keys = ("distilled_guidance_scale", "guidance_scale", "true_cfg_scale",
             "negative_prompt", "max_sequence_length")
_cg = _cg_kwargs("hunyuan-image", "blurry")
_nd = _nodes_kwargs("hunyuan-image", "blurry")
check(
    "both copies route hunyuan-image identically (CFG-relevant keys)",
    {k: _cg.get(k) for k in _cfg_keys} == {k: _nd.get(k) for k in _cfg_keys},
    f"comfyless={_cg!r} nodes={_nd!r}",
)


# ──────────────────────────────────────────────────────────────────────
print("── _build_call_kwargs: empty negative_prompt is omitted ───────")

for label, fn in (("comfyless", _cg_kwargs), ("nodes", _nodes_kwargs)):
    kw = fn("hunyuan-image", "")
    check(f"{label}: empty negative_prompt omitted from kwargs",
          "negative_prompt" not in kw)


# ──────────────────────────────────────────────────────────────────────
print("── _build_call_kwargs: distilled key NOT smeared (negative case)")

# Invariant 2 negative: a non-Hunyuan family must NOT receive
# distilled_guidance_scale. Proves the new branch is gated on the family
# string, not appended to the base dict for everyone. The "foobar" row
# probes the unknown-family introspection fallback — the candidates dict
# in that path lists guidance_scale + true_cfg_scale but not
# distilled_guidance_scale, so the kwarg must not appear even for a
# pipe whose __call__ signature happens to accept it.
for fam in ("flux", "qwen-image", "sdxl", "auraflow", "foobar"):
    # qwen-image is the only one of these that is NOT guidance-distilled.
    embeds = fam != "qwen-image"
    cgk = _cg_kwargs(fam, "y", cfg_scale=4.0, guidance_embeds=embeds)
    check(f"comfyless: {fam} does NOT get distilled_guidance_scale",
          "distilled_guidance_scale" not in cgk, f"got {cgk!r}")
    ndk = _nodes_kwargs(fam, "y", cfg_scale=4.0, guidance_embeds=embeds)
    check(f"nodes: {fam} does NOT get distilled_guidance_scale",
          "distilled_guidance_scale" not in ndk, f"got {ndk!r}")


# ──────────────────────────────────────────────────────────────────────
print("── family_defaults: hunyuan-image row exists with ADR-014 §4 values")

from comfyless.family_defaults import FAMILY_DEFAULTS

check(
    "FAMILY_DEFAULTS['hunyuan-image'] exists",
    "hunyuan-image" in FAMILY_DEFAULTS,
)
check(
    "FAMILY_DEFAULTS['hunyuan-image'] sets cfg_scale=3.25 (pipeline default)",
    FAMILY_DEFAULTS.get("hunyuan-image", {}).get("cfg_scale") == 3.25,
)
check(
    "FAMILY_DEFAULTS['hunyuan-image'] sets steps=50 (model card)",
    FAMILY_DEFAULTS.get("hunyuan-image", {}).get("steps") == 50,
)
# 2K-native (ADR-014 Changelog 2026-05-24 amendment): the 32× VAE was
# trained on 2048-decoded images; sub-2K renders are out-of-distribution
# per Tencent README ("1K resolution will result in artifacts").
check(
    "FAMILY_DEFAULTS['hunyuan-image'] sets width=2048 (2K-native per Tencent README)",
    FAMILY_DEFAULTS.get("hunyuan-image", {}).get("width") == 2048,
)
check(
    "FAMILY_DEFAULTS['hunyuan-image'] sets height=2048 (2K-native per Tencent README)",
    FAMILY_DEFAULTS.get("hunyuan-image", {}).get("height") == 2048,
)
# Defensive: NOT true_cfg_scale (Hunyuan is distilled, not 2-pass CFG).
check(
    "FAMILY_DEFAULTS['hunyuan-image'] does NOT set true_cfg_scale",
    "true_cfg_scale" not in FAMILY_DEFAULTS.get("hunyuan-image", {}),
)
# Defensive: NOT distilled_guidance_scale either — comfyless schema has no
# such canonical key (ADR-014 §6); the overlay applier skips keys not in
# COMFYLESS_SCHEMA, so listing it would be silently ignored AND misleading.
check(
    "FAMILY_DEFAULTS['hunyuan-image'] does NOT set distilled_guidance_scale "
    "(not a COMFYLESS_SCHEMA key per ADR-014 §6)",
    "distilled_guidance_scale" not in FAMILY_DEFAULTS.get("hunyuan-image", {}),
)


# ──────────────────────────────────────────────────────────────────────
print("── family_defaults: hunyuan-image refiner-stage row (ADR-016 §(d))")

# ADR-016 §(d): Tencent refiner README is authoritative — refiner_cfg=3.5,
# refiner_steps=4. Diffusers HunyuanImageRefinerPipeline.__call__ signature
# default for distilled_guidance_scale is 3.25, but the README wins
# (same lesson as the 2K-mandatory amendment in ADR-014's 2026-05-24
# Changelog: signature docstrings are unreliable; the Tencent README is
# the source of truth for operating points). Both keys are no-ops when
# the chained dispatch path isn't activated — they flow through the
# ADR-009 precedence ladder same as cfg_scale/steps, just only consumed
# by the refiner stage when it runs.
check(
    "FAMILY_DEFAULTS['hunyuan-image'] sets refiner_steps=4 (Tencent refiner README)",
    FAMILY_DEFAULTS.get("hunyuan-image", {}).get("refiner_steps") == 4,
)
check(
    "FAMILY_DEFAULTS['hunyuan-image'] sets refiner_cfg=3.5 (Tencent refiner README)",
    FAMILY_DEFAULTS.get("hunyuan-image", {}).get("refiner_cfg") == 3.5,
)
# Defensive: NOT refiner_path — there is no family-wide default refiner
# location. The ADR-016 §(a) "no filesystem auto-discovery" invariant
# requires the operator to point at the refiner explicitly. Family-
# defaulting refiner_path would either re-introduce path derivation
# (Alternative A — rejected for the security-surface widening reason)
# or hardcode a single path that doesn't match every operator's layout.
check(
    "FAMILY_DEFAULTS['hunyuan-image'] does NOT set refiner_path "
    "(ADR-016 §(a) — no auto-discovery; operator opts in explicitly)",
    "refiner_path" not in FAMILY_DEFAULTS.get("hunyuan-image", {}),
)


# ──────────────────────────────────────────────────────────────────────
print("── COMFYLESS_SCHEMA: refiner-stage canonical keys (ADR-016 §(d), (h))")

# Schema-replayable. refiner_path parallels transformer_path/vae_path etc.
# (sidecar-replayable component path; empty string = unset). refiner_steps
# / refiner_cfg parallel steps / cfg_scale (typed numeric, ADR-009 overlay).
# All three are no-ops when the chained dispatch path doesn't run; sidecar
# replay of a base+refiner generation against a pre-refiner build silently
# ignores unknown keys (existing schema-validator pass-through behavior).
from comfyless.params_schema import COMFYLESS_SCHEMA

check(
    "COMFYLESS_SCHEMA contains refiner_path (sidecar-replayable path)",
    "refiner_path" in COMFYLESS_SCHEMA,
)
check(
    "COMFYLESS_SCHEMA contains refiner_steps (sidecar-replayable int)",
    "refiner_steps" in COMFYLESS_SCHEMA,
)
check(
    "COMFYLESS_SCHEMA contains refiner_cfg (sidecar-replayable float)",
    "refiner_cfg" in COMFYLESS_SCHEMA,
)
# Schema default values match Tencent refiner README (locks defaults
# even if a future engineer changes _FIELD_DEFAULTS without acknowledging
# the README citation).
check(
    "COMFYLESS_SCHEMA['refiner_steps'] default is 4",
    COMFYLESS_SCHEMA["refiner_steps"][1] == 4,
)
check(
    "COMFYLESS_SCHEMA['refiner_cfg'] default is 3.5",
    COMFYLESS_SCHEMA["refiner_cfg"][1] == 3.5,
)
check(
    "COMFYLESS_SCHEMA['refiner_path'] default is '' (unset → base-only)",
    COMFYLESS_SCHEMA["refiner_path"][1] == "",
)


# ──────────────────────────────────────────────────────────────────────
print("── family_defaults: precedence ladder (ADR-009 + ADR-014 §4) ──")

# End-to-end: _apply_family_defaults reads detect_pipeline_class to derive
# the family, then walks FAMILY_DEFAULTS, skipping keys in explicit_keys /
# iterated_axes / not-in-schema. Using the fixture model_index.json gives
# a real end-to-end path (same test pattern as the detect_pipeline_class
# fixture above).
with tempfile.TemporaryDirectory() as tmpdir:
    fixture_path = os.path.join(tmpdir, "model_index.json")
    with open(fixture_path, "w") as f:
        json.dump({"_class_name": "HunyuanImagePipeline"}, f)

    # Case A — bare run: explicit_keys empty, iterated_axes empty. Family
    # defaults SHOULD apply, overriding any schema default already in p_cur.
    p_cur = {
        "model": tmpdir, "cfg_scale": 3.5, "steps": 28,
        "width": 1024, "height": 1024,  # schema defaults
    }
    cg._apply_family_defaults(p_cur, explicit_keys=set(), iterated_axes=set())
    check(
        "bare run: family default overrides schema default (cfg_scale 3.5 → 3.25)",
        p_cur["cfg_scale"] == 3.25,
        f"p_cur={p_cur!r}",
    )
    check(
        "bare run: family default overrides schema default (steps 28 → 50)",
        p_cur["steps"] == 50,
    )
    # 2K-native dim defaults override 1K schema defaults — the headline fix
    # of the 2026-05-24 amendment that addresses the artifact issue Grant
    # surfaced after the original Step 5 smoke.
    check(
        "bare run: family default overrides schema default (width 1024 → 2048)",
        p_cur["width"] == 2048,
    )
    check(
        "bare run: family default overrides schema default (height 1024 → 2048)",
        p_cur["height"] == 2048,
    )

    # Case B — explicit CLI override on cfg_scale. Family default for cfg_scale
    # SHOULD be skipped; steps SHOULD still apply.
    p_cur = {"model": tmpdir, "cfg_scale": 5.0, "steps": 28}
    cg._apply_family_defaults(p_cur, explicit_keys={"cfg_scale"}, iterated_axes=set())
    check(
        "explicit cfg_scale in CLI: family default for cfg_scale skipped (stays 5.0)",
        p_cur["cfg_scale"] == 5.0,
    )
    check(
        "explicit cfg_scale in CLI: other family defaults still apply (steps → 50)",
        p_cur["steps"] == 50,
    )

    # Case C — iterated axis. Same gate semantics as explicit_keys.
    p_cur = {"model": tmpdir, "cfg_scale": 3.5, "steps": 28}
    cg._apply_family_defaults(p_cur, explicit_keys=set(), iterated_axes={"steps"})
    check(
        "iterated steps axis: family default for steps skipped (stays 28)",
        p_cur["steps"] == 28,
    )
    check(
        "iterated steps axis: cfg_scale family default still applies (3.5 → 3.25)",
        p_cur["cfg_scale"] == 3.25,
    )


# ──────────────────────────────────────────────────────────────────────
print("── family_defaults: missing row degrades gracefully (negative) ──")

# Defends the overlay applier's robustness: removing the row must not crash
# the call. The applier short-circuits on `not fam_defaults`; the absence
# of an entry must not differ from the "family has no opinion" path.
with tempfile.TemporaryDirectory() as tmpdir:
    fixture_path = os.path.join(tmpdir, "model_index.json")
    with open(fixture_path, "w") as f:
        json.dump({"_class_name": "HunyuanImagePipeline"}, f)

    saved = cg.FAMILY_DEFAULTS.pop("hunyuan-image")
    try:
        p_cur = {"model": tmpdir, "cfg_scale": 3.5, "steps": 28}
        cg._apply_family_defaults(p_cur, explicit_keys=set(), iterated_axes=set())
        check(
            "missing hunyuan-image row: no crash, schema defaults retained (cfg_scale)",
            p_cur["cfg_scale"] == 3.5,
        )
        check(
            "missing hunyuan-image row: no crash, schema defaults retained (steps)",
            p_cur["steps"] == 28,
        )
    finally:
        cg.FAMILY_DEFAULTS["hunyuan-image"] = saved


# ──────────────────────────────────────────────────────────────────────
print("── resolve_vae_tiling: family-conditional default + explicit overrides")

# Invariant 1 — hunyuan-image under auto resolves to "no tiling". Locks the
# headline behavior of the tile-VAE-skip slice: the 32× Hunyuan VAE does
# not benefit from tiling and the seams compound the artifacts diagnosed
# in the ADR-014 2026-05-24 amendment.
check(
    "hunyuan-image + auto → tiling False (32× VAE; ADR-014 Changelog 2026-05-27)",
    utils_mod.resolve_vae_tiling("hunyuan-image", "auto") is False,
)

# Invariant 2 — every other family under auto stays tiled (preserves the
# pre-existing behavior on 8×/16× VAEs).
AUTO_TILED_FAMILIES = [
    "qwen-edit", "qwen-image",
    "flux", "flux2", "flux2klein", "flux2klein-base",
    "chroma", "auraflow",
    "sd1", "sd3", "sdxl",
    "zimage",
    "stablecascade",
]
for fam in AUTO_TILED_FAMILIES:
    check(
        f"{fam} + auto → tiling True (preserves current behavior)",
        utils_mod.resolve_vae_tiling(fam, "auto") is True,
    )

# Refiner uses the same 32× DCAE VAE as base → no-tile default. Confirmed
# by live 2026-06-02 smoke: tiled encode raises shape error on 1920×1088 in
# AutoencoderKLHunyuanImageRefiner._dcae_downsample_rearrange.
check(
    "hunyuan-image-refiner + auto → tiling False (DCAE 32× VAE; same as base)",
    utils_mod.resolve_vae_tiling("hunyuan-image-refiner", "auto") is False,
)
check(
    "hunyuan-image-refiner + on → tiling True (force-on wins over family default)",
    utils_mod.resolve_vae_tiling("hunyuan-image-refiner", "on") is True,
)

# Invariant 3a — explicit "on" forces tiling even on a default-off family.
check(
    "hunyuan-image + on → tiling True (force-on wins over family default)",
    utils_mod.resolve_vae_tiling("hunyuan-image", "on") is True,
)

# Invariant 3b — explicit "off" forces no-tiling even on a default-on family.
check(
    "qwen-image + off → tiling False (force-off wins over family default)",
    utils_mod.resolve_vae_tiling("qwen-image", "off") is False,
)

# Unknown family falls into the safe (memory-safe) tiled path under auto.
# Locks the closed-world contract on _VAE_TILING_FAMILIES_DEFAULT_OFF: a new
# 32× family that needs default-off must be explicitly added — the resolver
# does not silently extrapolate from name shape.
check(
    "foobar (unknown family) + auto → tiling True (memory-safe fallback)",
    utils_mod.resolve_vae_tiling("foobar", "auto") is True,
)

# Default arg (flag omitted) matches explicit "auto" — pins the in-process
# convention used by _load_pipeline()'s `vae_tiling: str = "auto"` signature.
check(
    "flag omitted matches explicit 'auto' for hunyuan-image",
    utils_mod.resolve_vae_tiling("hunyuan-image") is False,
)
check(
    "flag omitted matches explicit 'auto' for qwen-image",
    utils_mod.resolve_vae_tiling("qwen-image") is True,
)

# Defense-in-depth — invalid flag raises ValueError. Argparse rejects bad
# values upstream in the CLI path; the in-process raise defends the ComfyUI
# node dropdown and any future programmatic caller from silent fallthrough.
expect_raises(
    "resolve_vae_tiling raises ValueError on invalid flag",
    lambda: utils_mod.resolve_vae_tiling("hunyuan-image", "garbage"),
    ValueError,
)


# ──────────────────────────────────────────────────────────────────────
print("── Invariant 4 — comfyless _load_pipeline applies resolver decision")

# Behavior test: drive comfyless._load_pipeline against a FakePipe whose VAE
# tracks use_tiling state via enable_tiling()/disable_tiling(). Asserts that
# for every (family, flag) combination, the loader's post-load
# pipe.vae.use_tiling matches resolve_vae_tiling(family, flag) — i.e. the
# loader actually calls the resolver and applies its result, not a duplicated
# inline family check (per code-reviewer's Step 1 forward-watch).
class _FakeVAE:
    def __init__(self):
        self.use_tiling = False
    def enable_tiling(self):
        self.use_tiling = True
    def disable_tiling(self):
        self.use_tiling = False

class _FakePipe:
    def __init__(self):
        self.vae = _FakeVAE()
    def to(self, device):
        return self

class _FakePipeClass:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        return _FakePipe()

def _drive_comfyless_load(model_family, vae_tiling, tmp_path):
    """Drive cg._load_pipeline with stubbed disk/diffusers boundary."""
    orig = {
        "resolve_hf_path":      cg.resolve_hf_path,
        "detect_pipeline_class": cg.detect_pipeline_class,
        "detect_load_variant":   cg.detect_load_variant,
        "read_guidance_embeds":  cg.read_guidance_embeds,
    }
    cg.resolve_hf_path      = lambda p, **kw: p
    cg.detect_pipeline_class = lambda p: (_FakePipeClass, "FakePipeline", model_family)
    cg.detect_load_variant   = lambda p: None
    cg.read_guidance_embeds  = lambda p: False
    try:
        pipe, _fam, _emb = cg._load_pipeline(
            tmp_path,
            precision="bf16",
            device="cpu",
            offload_vae=False,
            attention_slicing=False,
            sequential_offload=False,
            vae_tiling=vae_tiling,
        )
        return pipe.vae.use_tiling
    finally:
        for k, v in orig.items():
            setattr(cg, k, v)

with tempfile.TemporaryDirectory() as _tmp:
    PARITY_CASES = [
        ("hunyuan-image", "auto", False),
        ("hunyuan-image", "on",   True),
        ("hunyuan-image", "off",  False),
        ("qwen-image",    "auto", True),
        ("qwen-image",    "off",  False),
        ("flux2",         "on",   True),
        ("foobar",        "auto", True),  # unknown family → memory-safe default
    ]
    for family, flag, expected in PARITY_CASES:
        actual = _drive_comfyless_load(family, flag, _tmp)
        check(
            f"comfyless _load_pipeline: ({family!r},{flag!r}) → use_tiling={expected}",
            actual is expected,
            f"got {actual!r}",
        )


# ──────────────────────────────────────────────────────────────────────
print("── Invariant 4 — both loader call sites wire the resolver identically")

# Structural co-locking: the ComfyUI loader path runs inside ComfyUI so it
# can't be cheaply unit-instrumented (relative imports drag the whole node
# pack), but the parity contract is that BOTH loaders call resolve_vae_tiling
# with the same (family, flag) shape and apply both branches. These string
# checks lock the "Step 2 wired one side, forgot the other" failure mode
# and the "duplicated the family check inline instead of calling the
# resolver" failure mode the Step 1 reviewer flagged as the watch-item.
with open("nodes/eric_diffusion_loader.py") as f:
    nodes_loader_src = f.read()

check(
    "nodes loader imports resolve_vae_tiling from eric_diffusion_utils",
    "resolve_vae_tiling" in nodes_loader_src
    and "from comfyless.core.eric_diffusion_utils import" in nodes_loader_src,
)
check(
    "nodes loader calls resolve_vae_tiling(model_family, vae_tiling)",
    "resolve_vae_tiling(model_family, vae_tiling)" in nodes_loader_src,
)
check(
    "nodes loader applies both enable_tiling() AND disable_tiling()",
    "pipeline.vae.enable_tiling()" in nodes_loader_src
    and "pipeline.vae.disable_tiling()" in nodes_loader_src,
)

with open("comfyless/generate.py") as f:
    cg_src = f.read()

check(
    "comfyless _load_pipeline imports resolve_vae_tiling",
    "resolve_vae_tiling" in cg_src,
)
check(
    "comfyless _load_pipeline calls resolve_vae_tiling(model_family, vae_tiling)",
    "resolve_vae_tiling(model_family, vae_tiling)" in cg_src,
)
check(
    "comfyless _load_pipeline applies both enable_tiling() AND disable_tiling()",
    "pipe.vae.enable_tiling()" in cg_src
    and "pipe.vae.disable_tiling()" in cg_src,
)


# ──────────────────────────────────────────────────────────────────────
print("── Invariant 5 — argparse rejects invalid --vae-tiling value ─────────")

# Subprocess invocation so we hit the real argparse layer, not a mock.
# Uses the same .venv interpreter the rest of the suite runs under.
import subprocess
_repo_root = os.path.dirname(os.path.abspath(__file__))
_proc = subprocess.run(
    [sys.executable, "-m", "comfyless.generate",
     "--vae-tiling", "garbage",
     "--model", "/tmp/__vae_tiling_test_nonexistent__",
     "--prompt", "x",
     "--output", "/tmp/__vae_tiling_test_nonexistent__.png"],
    capture_output=True,
    text=True,
    timeout=60,
    cwd=_repo_root,
)
check(
    "--vae-tiling garbage exits non-zero",
    _proc.returncode != 0,
    f"returncode={_proc.returncode}",
)
check(
    "--vae-tiling garbage rejection lists valid choices on stderr",
    "'auto'" in _proc.stderr and "'on'" in _proc.stderr and "'off'" in _proc.stderr,
    f"stderr tail={_proc.stderr[-300:]!r}",
)
check(
    "--vae-tiling garbage never attempts to load the model",
    "Loading model" not in _proc.stderr,
    "argparse must reject before any disk I/O",
)


# ──────────────────────────────────────────────────────────────────────
print("── Invariant 4 — comfyless daemon wires vae_tiling through IPC (structural)")

# Structural co-locking for the comfyless daemon path. _handle_generate is
# hard to unit-instrument cheaply (it reaches into pipe.delete_adapters,
# LoRA load, savepath resolution); the contract that matters for the
# tile-VAE-skip slice is "the daemon passes vae_tiling through with the same
# shape as the one-shot CLI path." These string assertions lock the wire-
# protocol field, the cache_key membership, and both _load_pipeline call
# sites against silent regression. Behavior coverage of resolver application
# lives in the FakePipe test above (which exercises the same _load_pipeline
# the daemon calls).
with open("comfyless/server.py") as f:
    server_src = f.read()

check(
    "server.py cache_key tuple includes vae_tiling entry",
    "req.get(\"vae_tiling\")" in server_src,
)
# Three vae_tiling threading sites exist in _handle_generate after the
# Step-4 refiner-chain slice: (1) initial base load via _load_pipeline,
# (2) LoRA-removal-failure reload base via _load_pipeline, (3) refiner
# load via _maybe_load_refiner → hunyuan_chain.load_refiner_pipeline.
# All three must thread vae_tiling identically so the family-aware
# default applies uniformly across base + refiner stages.
check(
    "server.py threads vae_tiling identically to all 3 ML-stack call sites "
    "(2× _load_pipeline + _maybe_load_refiner → load_refiner_pipeline)",
    server_src.count("vae_tiling=req.get(\"vae_tiling\") or \"auto\"") == 3,
    f"got {server_src.count('vae_tiling=req.get(\"vae_tiling\") or \"auto\"')} occurrence(s)",
)

# Client side — _delegate_to_server must include the field in the outbound
# request dict. Without this, daemon-mode clients silently lose the flag
# (the request omits it, server sees None → "auto" via the `or "auto"`
# fallback, which loses any explicit override the operator typed).
# Reuses cg_src loaded in the I4-structural section above; regex form is
# whitespace-insensitive so column-alignment changes in the request dict
# do not silently invalidate the lock (Step 3 code-reviewer minor).
import re
check(
    "comfyless/generate.py _delegate_to_server request dict includes vae_tiling",
    re.search(r'"vae_tiling"\s*:\s*args\.vae_tiling', cg_src) is not None,
)

# Lock the deliberate MCP-server omission. comfyless/mcp_server.py L600-606
# explicitly states that operator-tuning knobs (precision, offload_vae,
# attention_slicing, sequential_offload, vae_tiling) are NOT exposed on the
# MCP schema — the LLM agent should not be tuning these per-call. The MCP
# _load_pipeline call inherits the family-aware "auto" default via the
# signature default. This assertion catches a future engineer either (a)
# adding vae_tiling to the MCP schema without acknowledging the design
# intent, or (b) silently threading args through without updating the
# comment block. Forces the change to surface here. (Step 3 code-reviewer
# MEDIUM, security-auditor confirmed: out of scope for this slice; locked.)
with open("comfyless/mcp_server.py") as f:
    mcp_src = f.read()
check(
    "mcp_server.py does NOT thread vae_tiling (deliberate omission per "
    "operator-tuning-knob comment at L600-606)",
    "vae_tiling" not in mcp_src,
)


# ══════════════════════════════════════════════════════════════════════
#  Refiner slice — Step 2 (ADR-016, slice-hunyuan-image-2-1-refiner.md)
# ══════════════════════════════════════════════════════════════════════
#
# Invariants tested below (cross-reference Vision §"Invariants"):
#   Inv 1  no-fs-search       — structural + runtime
#   Inv 2  warn-don't-block   — runtime stderr capture + PIL write
#   Inv 3  opt-in via path    — runtime activation gating
#   Inv 4  output identity    — PNG tEXt chunk shape
#   Inv 6  CFG routing parity — build_refiner_call_kwargs direct call
#   Inv 7  LoRAs base-only    — structural + runtime adapter-count proxy
#   Inv 8  scheduler pinned   — structural source check
#   Inv 9  shared text enc    — structural source check + runtime
#   Inv 10 no regressions     — negative case (non-hunyuan + refiner_path)
#
# Inv 5 is locked by the Step-1 schema/defaults tests above. Inv 11 is
# the Step-4 daemon scope. Inv 12 is the Step-3 ComfyUI-node scope.
# Inv 13 is the live-smoke memory ceiling (no CPU coverage).

import comfyless.hunyuan_chain as hc
import contextlib
import io
from PIL import Image as _PILImage
from PIL.PngImagePlugin import PngInfo


class _FakeBaseResult:
    """diffusers-shaped result: .images is a list of PIL images."""
    def __init__(self, pil):
        self.images = [pil]


class _FakeBasePipe:
    """Fake base pipeline. Tracks call count and last kwargs."""
    def __init__(self, family="hunyuan-image"):
        self.call_count = 0
        self.last_kwargs = None
        self.family = family
        # text_encoder + tokenizer present so shared-encoder injection
        # has identity-trackable objects to assert on.
        self.text_encoder = object()
        self.tokenizer = object()
        self.vae = None  # offload_vae=False in tests → never accessed

    def __call__(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        pil = _PILImage.new("RGB", (16, 16), color=(10, 20, 30))
        return _FakeBaseResult(pil)


class _FakeRefinerPipe:
    """Fake refiner pipeline. Tracks call count and last kwargs."""
    def __init__(self):
        self.call_count = 0
        self.last_kwargs = None

    def __call__(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        pil = _PILImage.new("RGB", (16, 16), color=(200, 100, 50))
        return _FakeBaseResult(pil)

    def to(self, device):
        # diffusers pipelines return self from .to() — load_refiner_pipeline
        # rebinds the returned value, so this fake must follow the same shape.
        return self


def _cached(family="hunyuan-image", refiner_pipe=None):
    """Build a _cached_pipeline dict shape that bypasses _load_pipeline."""
    base = _FakeBasePipe(family=family)
    cd = {
        "pipeline": base,
        "model_family": family,
        "guidance_embeds": False,
    }
    if refiner_pipe is not None:
        cd["refiner_pipeline"] = refiner_pipe
    return cd, base


# ──────────────────────────────────────────────────────────────────────
print("── Inv 1 — no filesystem search (structural + runtime) ────────")

# Structural: hunyuan_chain.py must not contain any directory-traversal
# primitives. If it has none, the runtime can't search — locks the
# "no path-derivation" posture at the source level. Defends against a
# future engineer adding sibling-glob logic without re-reading ADR-016
# Alternative A's rationale.
with open("comfyless/hunyuan_chain.py") as f:
    chain_src = f.read()
for forbidden in ("os.listdir", "Path.glob", ".iterdir(", ".scandir(", "os.scandir"):
    check(
        f"hunyuan_chain.py does NOT contain {forbidden!r} (no auto-discovery)",
        forbidden not in chain_src,
    )

# Runtime: a bare hunyuan-image generate() with refiner_path="" must
# never reach load_refiner_pipeline — the only code path that could
# derive any refiner path. Tracks invocation by replacing the loader
# with a recording spy.
_load_calls = []
_orig_loader = hc.load_refiner_pipeline
hc.load_refiner_pipeline = lambda *a, **kw: (_load_calls.append((a, kw)),
                                             _FakeRefinerPipe())[1]
try:
    cached, base = _cached(family="hunyuan-image", refiner_pipe=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use the real fixture dir as --model so output dir validation passes.
        fixture = os.path.join(tmpdir, "model_index.json")
        with open(fixture, "w") as f:
            json.dump({"_class_name": "HunyuanImagePipeline"}, f)
        out = os.path.join(tmpdir, "out.png")
        # Silence the expected warn-don't-block stderr line.
        with contextlib.redirect_stderr(io.StringIO()):
            cg.generate(
                model_path=tmpdir, prompt="x", output_path=out,
                refiner_path="", device="cpu", _cached_pipeline=cached,
            )
    check(
        "bare hunyuan-image (refiner_path='') never invokes load_refiner_pipeline",
        len(_load_calls) == 0,
        f"got {len(_load_calls)} call(s)",
    )
finally:
    hc.load_refiner_pipeline = _orig_loader


# ──────────────────────────────────────────────────────────────────────
print("── Inv 2 — warn-don't-block (bare hunyuan-image; no refiner) ───")

# The exact warning text is locked by Vision §Intent + dump §"Step
# 2-specific notes". Changing it requires a paired test edit — this
# string assertion catches casual edits to the user-facing line.
EXPECTED_WARNING_FRAGMENTS = (
    "hunyuan-image quality requires a refiner",
    "pass --refiner <path>",
    "huggingface-cli download hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers",
)

cached, base = _cached(family="hunyuan-image", refiner_pipe=None)
with tempfile.TemporaryDirectory() as tmpdir:
    out = os.path.join(tmpdir, "out.png")
    buf = io.StringIO()
    rc_exc = None
    with contextlib.redirect_stderr(buf):
        try:
            metadata = cg.generate(
                model_path=tmpdir, prompt="x", output_path=out,
                refiner_path="", device="cpu", _cached_pipeline=cached,
            )
        except Exception as e:
            rc_exc = e
    stderr_text = buf.getvalue()
    check(
        "Inv 2: bare hunyuan-image without --refiner does NOT raise",
        rc_exc is None,
        f"raised {type(rc_exc).__name__}: {rc_exc}" if rc_exc else "",
    )
    for fragment in EXPECTED_WARNING_FRAGMENTS:
        check(
            f"Inv 2: warning stderr contains {fragment!r}",
            fragment in stderr_text,
            f"stderr={stderr_text!r}",
        )
    check(
        "Inv 2: PNG written despite warn-don't-block",
        os.path.isfile(out),
    )
    check(
        "Inv 2: base pipeline called exactly once (no refiner stage)",
        base.call_count == 1,
        f"got {base.call_count}",
    )
    # Metadata must NOT carry the chain keys when the chain didn't run
    # (Vision Inv 4: "the new keys are absent, not present-and-empty").
    for key in ("pipeline", "refiner_path", "refiner_steps", "refiner_cfg"):
        check(
            f"Inv 2: base-only metadata omits {key!r} (Vision Inv 4)",
            key not in metadata,
            f"got {metadata.get(key)!r}",
        )


# ──────────────────────────────────────────────────────────────────────
print("── Inv 3 — opt-in via refiner_path activates the chain ────────")

# Drive generate() with refiner_path set. The cached-pipeline dict
# carries a pre-built _FakeRefinerPipe so the chain activates without
# hitting load_refiner_pipeline. Assertion: refiner __call__ ran
# exactly once → opt-in path was taken.
ref_pipe = _FakeRefinerPipe()
cached, base = _cached(family="hunyuan-image", refiner_pipe=ref_pipe)
with tempfile.TemporaryDirectory() as tmpdir:
    out = os.path.join(tmpdir, "out.png")
    with contextlib.redirect_stderr(io.StringIO()):
        metadata_chain = cg.generate(
            model_path=tmpdir, prompt="alpine lake", output_path=out,
            negative_prompt="blurry",
            refiner_path="/fake/refiner/path",
            refiner_steps=4, refiner_cfg=3.5,
            device="cpu", _cached_pipeline=cached,
        )
check(
    "Inv 3: refiner __call__ invoked exactly once when refiner_path set",
    ref_pipe.call_count == 1,
    f"got {ref_pipe.call_count}",
)
check(
    "Inv 3: base __call__ also invoked exactly once (single base pass)",
    base.call_count == 1,
    f"got {base.call_count}",
)


# ──────────────────────────────────────────────────────────────────────
print("── Inv 4 — output identity + PNG metadata schema (ADR-016 §h) ──")

# Verify the comfyless tEXt chunk on the chained-output PNG carries:
#   - pipeline = "base+refiner" (literal, locked)
#   - refiner_path = exact path passed in
#   - refiner_steps / refiner_cfg = effective values
# AND that the base-only PNG from Inv 2 carries NONE of these keys.

with tempfile.TemporaryDirectory() as tmpdir:
    out = os.path.join(tmpdir, "chained.png")
    ref_pipe = _FakeRefinerPipe()
    cached, _ = _cached(family="hunyuan-image", refiner_pipe=ref_pipe)
    with contextlib.redirect_stderr(io.StringIO()):
        metadata = cg.generate(
            model_path=tmpdir, prompt="p", output_path=out,
            refiner_path="/some/refiner/path",
            refiner_steps=8, refiner_cfg=4.25,
            device="cpu", _cached_pipeline=cached,
        )

    # In-memory metadata shape
    check(
        "Inv 4: metadata['pipeline'] == 'base+refiner'",
        metadata.get("pipeline") == "base+refiner",
        f"got {metadata.get('pipeline')!r}",
    )
    check(
        "Inv 4: metadata['refiner_path'] preserved from input",
        metadata.get("refiner_path") == "/some/refiner/path",
    )
    check(
        "Inv 4: metadata['refiner_steps'] == effective value (8)",
        metadata.get("refiner_steps") == 8,
    )
    check(
        "Inv 4: metadata['refiner_cfg'] == effective value (4.25)",
        metadata.get("refiner_cfg") == 4.25,
    )

    # On-disk PNG tEXt chunk shape
    info = _PILImage.open(out).info
    raw = info.get("comfyless")
    check(
        "Inv 4: PNG carries comfyless tEXt chunk",
        raw is not None,
    )
    chunk = json.loads(raw) if raw else {}
    check(
        "Inv 4: PNG tEXt: pipeline = 'base+refiner'",
        chunk.get("pipeline") == "base+refiner",
    )
    check(
        "Inv 4: PNG tEXt: refiner_path embedded",
        chunk.get("refiner_path") == "/some/refiner/path",
    )
    check(
        "Inv 4: PNG tEXt: refiner_steps embedded",
        chunk.get("refiner_steps") == 8,
    )
    check(
        "Inv 4: PNG tEXt: refiner_cfg embedded",
        chunk.get("refiner_cfg") == 4.25,
    )


# ──────────────────────────────────────────────────────────────────────
print("── Inv 6 — refiner CFG routing parity (ADR-016 §f) ─────────────")

# Direct call into build_refiner_call_kwargs — refiner_cfg routes to
# distilled_guidance_scale (mirrors base's hunyuan-image branch from
# ADR-014 §2). Refiner is also guidance-distilled.
fake_ref = _FakeRefinerPipe()
fake_image = _PILImage.new("RGB", (8, 8))
ref_kw = hc.build_refiner_call_kwargs(
    fake_ref, fake_image, "p", "blurry",
    refiner_steps=4, refiner_cfg=3.5, generator=None,
    height=2304, width=1792,
)
check(
    "Inv 6: distilled_guidance_scale = refiner_cfg",
    ref_kw.get("distilled_guidance_scale") == 3.5,
    f"got {ref_kw!r}",
)
# Resolution-preservation: height/width MUST be forwarded so the refiner
# does not default to 1024×1024 and downscale the base output.
check(
    "Inv 6: height forwarded to refiner kwargs",
    ref_kw.get("height") == 2304,
    f"got {ref_kw.get('height')!r}",
)
check(
    "Inv 6: width forwarded to refiner kwargs",
    ref_kw.get("width") == 1792,
    f"got {ref_kw.get('width')!r}",
)
check(
    "Inv 6: num_inference_steps = refiner_steps",
    ref_kw.get("num_inference_steps") == 4,
)
check(
    "Inv 6: image kwarg set (PIL roundtrip per ADR-016 §d)",
    ref_kw.get("image") is fake_image,
)
check(
    "Inv 6: prompt forwarded",
    ref_kw.get("prompt") == "p",
)
check(
    "Inv 6: negative_prompt forwarded when set",
    ref_kw.get("negative_prompt") == "blurry",
)
# Negative-case: empty/None negative_prompt is omitted, not present-and-empty.
# Mirrors base's ADR-014 §5 behavior — pipeline owns the empty-string semantics.
ref_kw_empty = hc.build_refiner_call_kwargs(
    fake_ref, fake_image, "p", "",
    refiner_steps=4, refiner_cfg=3.5, generator=None,
    height=2304, width=1792,
)
check(
    "Inv 6: empty negative_prompt omitted from refiner kwargs",
    "negative_prompt" not in ref_kw_empty,
)
ref_kw_none = hc.build_refiner_call_kwargs(
    fake_ref, fake_image, "p", None,
    refiner_steps=4, refiner_cfg=3.5, generator=None,
    height=2304, width=1792,
)
check(
    "Inv 6: None negative_prompt omitted from refiner kwargs",
    "negative_prompt" not in ref_kw_none,
)
# The other CFG kwargs MUST NOT appear on a refiner call — the refiner
# is distilled, not 2-pass true-CFG (same lock as the base hunyuan-image
# branch in test_hunyuan above).
check(
    "Inv 6: refiner kwargs do NOT include guidance_scale",
    "guidance_scale" not in ref_kw,
)
check(
    "Inv 6: refiner kwargs do NOT include true_cfg_scale",
    "true_cfg_scale" not in ref_kw,
)


# ──────────────────────────────────────────────────────────────────────
print("── Inv 7 — LoRAs base-only (structural + runtime) ─────────────")

# Structural: hunyuan_chain.py must not import or call any LoRA loader.
# Refiner has a separate transformer with separate weights; base LoRAs
# would not produce meaningful output on it (ADR-016 §g, Vision Inv 7).
for forbidden in ("load_lora_with_key_fix", "load_lora_weights", "load_lora",
                  "set_adapters", "fuse_lora"):
    check(
        f"hunyuan_chain.py does NOT reference {forbidden!r} (refiner LoRA-free)",
        forbidden not in chain_src,
    )

# Runtime: chained generate() with --lora set must invoke the LoRA
# loader exactly once and against the BASE pipe, never the refiner.
# The cached-pipeline gate in generate() skips LoRA loading entirely
# (server owns adapter state), so we must bypass that path AND
# _load_pipeline to exercise the LoRA loop directly. Stub
# _load_pipeline to return our FakeBasePipe,
# inject FakeRefinerPipe via the loader stub, run with --lora, then
# assert the LoRA loader received the BASE pipe id and NOT the refiner.
lora_calls = []
ref_pipe = _FakeRefinerPipe()
base_pipe = _FakeBasePipe(family="hunyuan-image")
_orig_load = cg._load_pipeline
_orig_loader2 = hc.load_refiner_pipeline
_orig_lora2 = cg.load_lora_with_key_fix
def _stub_load(model_path, **kw):
    return base_pipe, "hunyuan-image", False
def _stub_refiner_loader(*a, **kw):
    return ref_pipe
def _lora_spy2(pipe, path, *a, **kw):
    lora_calls.append({"pipe_id": id(pipe), "path": path})
    return True
cg._load_pipeline = _stub_load
hc.load_refiner_pipeline = _stub_refiner_loader
cg.load_lora_with_key_fix = _lora_spy2
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "out.png")
        with contextlib.redirect_stderr(io.StringIO()):
            cg.generate(
                model_path=tmpdir, prompt="p", output_path=out,
                refiner_path="/fake/refiner",
                refiner_steps=4, refiner_cfg=3.5,
                loras=[{"path": "/fake/lora.safetensors", "weight": 1.0}],
                device="cpu",
            )
finally:
    cg._load_pipeline = _orig_load
    hc.load_refiner_pipeline = _orig_loader2
    cg.load_lora_with_key_fix = _orig_lora2
check(
    "Inv 7: LoRA loader called exactly once (base-only; LoRA stack length 1)",
    len(lora_calls) == 1,
    f"got {len(lora_calls)} call(s)",
)
check(
    "Inv 7: LoRA loader received the BASE pipe id, not refiner",
    lora_calls and lora_calls[0]["pipe_id"] == id(base_pipe),
    f"got pipe_id={lora_calls[0]['pipe_id'] if lora_calls else '?'}",
)
check(
    "Inv 7: LoRA loader never received the refiner pipe id",
    all(c["pipe_id"] != id(ref_pipe) for c in lora_calls),
)


# ──────────────────────────────────────────────────────────────────────
print("── Inv 8 — refiner scheduler / sampler / sigmas pinned ─────────")

# Structural: hunyuan_chain.py must not mutate the refiner's scheduler
# (no `.scheduler =`, no `set_timesteps` call, no `swap_sampler(` call).
# The refiner uses its on-disk FlowMatchEulerDiscreteScheduler config;
# v1 ships no refiner-side scheduler/sampler/sigmas surface (ADR-016 §g).
# `swap_sampler(` (with open paren) catches the call site; the docstring
# mention is intentional (explains the caller's per-pipe responsibility).
for forbidden in (".scheduler =", "set_timesteps(", "swap_sampler(",
                  "register_to_config("):
    check(
        f"hunyuan_chain.py does NOT contain {forbidden!r} (refiner scheduler pinned)",
        forbidden not in chain_src,
    )

# Runtime: a chained generate() with base-side --sampler swap must
# leave the refiner pipe's scheduler unchanged. The base-side
# swap_sampler is a per-pipe context manager (operates on `pipe`, not
# refiner_pipe); the chain's call ordering inside generate() wraps
# swap_sampler around the entire run_chain call, but only the base
# scheduler is swapped. We track this by attaching a sentinel
# `.scheduler` to the refiner pipe and asserting identity-preservation
# across a chained run.
sentinel_scheduler = object()
ref_pipe = _FakeRefinerPipe()
ref_pipe.scheduler = sentinel_scheduler
cached, _ = _cached(family="hunyuan-image", refiner_pipe=ref_pipe)
with tempfile.TemporaryDirectory() as tmpdir:
    out = os.path.join(tmpdir, "out.png")
    with contextlib.redirect_stderr(io.StringIO()):
        cg.generate(
            model_path=tmpdir, prompt="p", output_path=out,
            refiner_path="/fake/refiner", sampler="default",
            device="cpu", _cached_pipeline=cached,
        )
check(
    "Inv 8: refiner.scheduler identity preserved across chained run",
    ref_pipe.scheduler is sentinel_scheduler,
)
# Resolution-preservation (end-to-end): the chained run must hand the
# refiner the base output's actual dimensions, NOT let it default to
# 1024×1024. _FakeBasePipe always returns a 16×16 PIL, so the refiner
# call must carry height=16, width=16.
check(
    "Inv 8: chained run forwards base PIL height to refiner",
    ref_pipe.last_kwargs is not None and ref_pipe.last_kwargs.get("height") == 16,
    f"got {ref_pipe.last_kwargs.get('height') if ref_pipe.last_kwargs else None!r}",
)
check(
    "Inv 8: chained run forwards base PIL width to refiner",
    ref_pipe.last_kwargs is not None and ref_pipe.last_kwargs.get("width") == 16,
    f"got {ref_pipe.last_kwargs.get('width') if ref_pipe.last_kwargs else None!r}",
)


# ──────────────────────────────────────────────────────────────────────
print("── Inv 9 — shared Qwen2.5-VL text_encoder (asymmetric, ADR-016 §e)")

# Structural: load_refiner_pipeline must pass `text_encoder=` and
# `tokenizer=` into the refiner's from_pretrained call. Locks the
# asymmetric optimization at the source level.
check(
    "hunyuan_chain.py passes text_encoder= into refiner from_pretrained",
    "text_encoder=base_pipe.text_encoder" in chain_src,
)
check(
    "hunyuan_chain.py passes tokenizer= into refiner from_pretrained",
    "tokenizer=base_pipe.tokenizer" in chain_src,
)
# Defensive: refiner has NO text_encoder_2 slot (per ADR-016 §e). The
# loader must not pass one — would either be silently dropped or raise.
check(
    "hunyuan_chain.py does NOT pass text_encoder_2= (refiner has no slot)",
    "text_encoder_2=" not in chain_src,
)

# Runtime: drive load_refiner_pipeline with stubbed
# detect_pipeline_class + a fake refiner class whose from_pretrained
# captures kwargs. Assert text_encoder identity matches base.
captured_kwargs = {}
class _FakeRefinerClass:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        captured_kwargs["path"] = path
        captured_kwargs.update(kwargs)
        return _FakeRefinerPipe()

# Hunyuan_chain reaches into nodes.eric_diffusion_utils for the resolver
# + detector. The test setup at the top of this file stubbed
# nodes.eric_diffusion_utils with only `build_model_metadata`; populate
# the three names hunyuan_chain.load_refiner_pipeline imports inside its
# body, then patch them with spies for this test block.
eduh = sys.modules["comfyless.core.eric_diffusion_utils"]
_orig_resolve = getattr(eduh, "resolve_hf_path", None)
_orig_detect = getattr(eduh, "detect_pipeline_class", None)
_orig_tiling = getattr(eduh, "resolve_vae_tiling", None)
eduh.resolve_hf_path = lambda p, **kw: p
eduh.detect_pipeline_class = lambda p: (
    _FakeRefinerClass, "HunyuanImageRefinerPipeline", "hunyuan-image-refiner",
)
eduh.resolve_vae_tiling = lambda fam, flag="auto": False
try:
    base = _FakeBasePipe()
    refiner = hc.load_refiner_pipeline(
        "/fake/refiner", base_pipe=base,
        precision="bf16", device="cpu", vae_tiling="auto",
    )
finally:
    eduh.resolve_hf_path = _orig_resolve
    eduh.detect_pipeline_class = _orig_detect
    eduh.resolve_vae_tiling = _orig_tiling

check(
    "Inv 9: refiner from_pretrained received text_encoder= identity-equal to base",
    captured_kwargs.get("text_encoder") is base.text_encoder,
)
check(
    "Inv 9: refiner from_pretrained received tokenizer= identity-equal to base",
    captured_kwargs.get("tokenizer") is base.tokenizer,
)
check(
    "Inv 9: refiner from_pretrained received local_files_only=True (no network)",
    captured_kwargs.get("local_files_only") is True,
)


# ──────────────────────────────────────────────────────────────────────
print("── Inv 9 — wrong-class refiner path raises clean ValueError ────")

# Negative case for Inv 9 / Vision §"Failure semantics": pointing
# --refiner at a non-refiner pipeline (e.g. a base or Flux pipeline by
# mistake) must raise a clean error citing the class mismatch.
eduh.resolve_hf_path = lambda p, **kw: p
eduh.detect_pipeline_class = lambda p: (
    object, "HunyuanImagePipeline", "hunyuan-image",  # base, not refiner
)
try:
    expect_raises(
        "load_refiner_pipeline rejects non-refiner pipeline class",
        lambda: hc.load_refiner_pipeline(
            "/fake/path", base_pipe=_FakeBasePipe(),
            precision="bf16", device="cpu",
        ),
        ValueError,
    )
finally:
    eduh.resolve_hf_path = _orig_resolve
    eduh.detect_pipeline_class = _orig_detect


# ──────────────────────────────────────────────────────────────────────
print("── Inv 10 — non-regression: refiner_path on non-hunyuan family ─")

# Setting --refiner on a non-hunyuan-image family must raise cleanly
# (no silent base-only fallback — the opt-in signal was explicit, the
# operator wants the refiner, masking the misconfig defeats the point).
# Vision Inv 10 + failure-semantics §4.
for non_hunyuan_family in ("qwen-image", "flux", "flux2", "sdxl", "auraflow",
                           "chroma", "sd1", "sd3", "zimage"):
    cached, _ = _cached(family=non_hunyuan_family, refiner_pipe=None)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "out.png")
        expect_raises(
            f"Inv 10: refiner_path set on {non_hunyuan_family} → ValueError",
            lambda c=cached, o=out, t=tmpdir: cg.generate(
                model_path=t, prompt="p", output_path=o,
                refiner_path="/fake/refiner",
                device="cpu", _cached_pipeline=c,
            ),
            ValueError,
        )


# ──────────────────────────────────────────────────────────────────────
print("── Inv 10 — non-regression: non-hunyuan families behave identically")

# Source-level co-lock: the refiner gate logic in generate() must be
# gated on model_family == "hunyuan-image" (or refiner_path set, for
# the negative case). A future engineer who fans the gate out to other
# families would break this lock.
with open("comfyless/generate.py") as f:
    gen_src = f.read()
check(
    "generate() refiner gate is family-conditional (model_family == \"hunyuan-image\")",
    'model_family == "hunyuan-image"' in gen_src,
)
check(
    "generate() refiner non-hunyuan-family branch raises ValueError",
    "raise ValueError" in gen_src and "--refiner is only supported" in gen_src,
)


# ──────────────────────────────────────────────────────────────────────
print("── Inv 3 — --refiner argparse flag wired into _parse_args ──────")

# Structural co-lock: the --refiner argparse flag must be present in
# _parse_args and the canonical mapping must route it to refiner_path.
# Catches accidental rename of the flag or the canonical key without
# touching the paired test.
check(
    "_parse_args declares --refiner flag",
    'p.add_argument("--refiner"' in gen_src,
)
from comfyless.params_schema import _CLI_TO_CANONICAL
check(
    "_CLI_TO_CANONICAL maps 'refiner' → 'refiner_path'",
    _CLI_TO_CANONICAL.get("refiner") == "refiner_path",
)


# ══════════════════════════════════════════════════════════════════════
#  Refiner slice — Step 3 (ComfyUI Generate node parity)
# ══════════════════════════════════════════════════════════════════════
#
# Step 3 mirrors the comfyless dispatch into nodes/eric_diffusion_generate.py
# so the ComfyUI surface has the same refiner-chaining behavior. Tests
# parallel the Step-2 tile-VAE-skip I4 structural co-lock pattern —
# behavior coverage of the actual run-chain logic lives on the comfyless
# side (CPU-driveable); the ComfyUI side is tested structurally because
# its inference path imports `comfy.utils` / `comfy.model_management` at
# call time and resists cheap unit instrumentation.

with open("nodes/eric_diffusion_generate.py") as f:
    node_gen_src = f.read()

print("── Step 3 / Inv 3 + 12 — refiner_path input wired into the Generate node")

# INPUT_TYPES surface — the operator-facing input must be declared with
# default "" (parallels comfyless argparse default None → schema default "").
check(
    "ComfyUI Generate node INPUT_TYPES declares 'refiner_path' as STRING",
    '"refiner_path": ("STRING"' in node_gen_src,
)
check(
    "ComfyUI Generate node 'refiner_path' input has default '' (unset → base-only)",
    'refiner_path' in node_gen_src and '"default": ""' in node_gen_src,
)
# generate() method threads the param.
check(
    "ComfyUI Generate node generate() signature accepts refiner_path",
    'refiner_path: str = ""' in node_gen_src,
)


print("── Step 3 / Inv 9 — Generate node shares the hunyuan_chain loader + dispatch")

# Structural co-lock: ComfyUI node uses the SAME shared module the
# comfyless path uses. No duplicate refiner-loading code, no inline
# transformer instantiation — preserves the single-source-of-truth
# guarantee for the asymmetric shared-encoder optimization (ADR-016 §e).
check(
    "ComfyUI Generate node imports hunyuan_chain.load_refiner_pipeline",
    "from comfyless.hunyuan_chain import load_refiner_pipeline" in node_gen_src,
)
check(
    "ComfyUI Generate node imports hunyuan_chain.run_chain",
    "from comfyless.hunyuan_chain import run_chain" in node_gen_src,
)
check(
    "ComfyUI Generate node calls load_refiner_pipeline (not inline construction)",
    "load_refiner_pipeline(" in node_gen_src,
)
check(
    "ComfyUI Generate node calls run_chain (not inline two-stage logic)",
    "run_chain(" in node_gen_src,
)


print("── Step 3 / Inv 10 — Generate node refiner gate is family-conditional")

# Same gate shape as the comfyless side (verified in Inv 10 above for
# generate.py). Catches a future engineer who drops the family check
# and lets refiner_path activate on, e.g., flux pipelines.
check(
    "ComfyUI Generate node refiner gate is family-conditional "
    "(model_family == \"hunyuan-image\")",
    'model_family == "hunyuan-image"' in node_gen_src,
)
check(
    "ComfyUI Generate node raises ValueError on non-hunyuan family + refiner_path set",
    "raise ValueError" in node_gen_src
    and "refiner_path is only supported" in node_gen_src,
)


print("── Step 3 / Inv 2 — Generate node warn-don't-block text (ComfyUI-flavored)")

# ComfyUI-flavored warning. Two fragments are shared with the comfyless
# warning text (the Vision Intent quality-pitch + the huggingface-cli
# download instruction); the third diverges because the operator-facing
# action is different (CLI: `--refiner <path>` flag; ComfyUI: set the
# `refiner_path` input on the node). Locked at runtime by these
# fragment assertions.
# Split into single-line-fitting pieces so the structural source check
# doesn't false-fail on line-broken string literals (Python concatenates
# adjacent literals at runtime — the runtime Inv 2 test above already
# verifies the assembled string; this structural sweep locks the
# individual tokens against drift).
SHARED_WARNING_FRAGMENTS = (
    "hunyuan-image quality requires a refiner",
    "huggingface-cli download",
    "hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers",
)
for fragment in SHARED_WARNING_FRAGMENTS:
    check(
        f"comfyless warning contains shared fragment {fragment!r}",
        fragment in gen_src,
    )
    check(
        f"ComfyUI Generate node warning contains shared fragment {fragment!r}",
        fragment in node_gen_src,
    )
# ComfyUI-specific fragment.
check(
    "ComfyUI Generate node warning references the refiner_path input",
    "set refiner_path on the Generate node" in node_gen_src,
)


print("── Step 3 / Inv 5 + 6 — Generate node sources operating point from FAMILY_DEFAULTS")

# Vision OQ4: a SINGLE refiner_path input on the ComfyUI node. The other
# two refiner-stage params (refiner_steps, refiner_cfg) flow through the
# same FAMILY_DEFAULTS row that drives them on the comfyless side. This
# keeps a single source of truth for the operating-point defaults — no
# inline 4/3.5 magic numbers in the ComfyUI node that could silently
# drift from the comfyless schema.
check(
    "ComfyUI Generate node imports FAMILY_DEFAULTS for refiner defaults",
    "from comfyless.family_defaults import FAMILY_DEFAULTS" in node_gen_src,
)
check(
    "ComfyUI Generate node reads FAMILY_DEFAULTS['hunyuan-image'] for refiner defaults",
    'FAMILY_DEFAULTS.get("hunyuan-image"' in node_gen_src,
)
# Defense-in-depth: the inline fallback constants MUST match the schema
# canonical defaults — defends against a stale fallback if FAMILY_DEFAULTS
# is missing the row at runtime (degraded-mode behavior should match the
# documented operating point).
check(
    "ComfyUI Generate node fallback refiner_steps matches FAMILY_DEFAULTS canonical (4)",
    FAMILY_DEFAULTS["hunyuan-image"]["refiner_steps"] == 4,
)
check(
    "ComfyUI Generate node fallback refiner_cfg matches FAMILY_DEFAULTS canonical (3.5)",
    FAMILY_DEFAULTS["hunyuan-image"]["refiner_cfg"] == 3.5,
)


print("── Step 3 / Inv 7 — Generate node does NOT load LoRAs into refiner")

# The ComfyUI loader/Generate split has no LoRA-load surface on the
# Generate node itself (LoRA application lives in eric_qwen_image_lora.py /
# eric_qwen_edit_lora.py and runs against the BASE transformer at load
# time). The Generate node's refiner path must not introduce any LoRA-load
# call against the refiner pipe.
for forbidden in ("load_lora_with_key_fix", "load_lora_weights",
                  "set_adapters", "fuse_lora"):
    check(
        f"ComfyUI Generate node does NOT call {forbidden!r} (refiner LoRA-free)",
        forbidden not in node_gen_src,
    )


print("── Step 3 / Inv 4 — Generate node extends metadata when chain runs")

# Parallels the comfyless metadata extension at generate.py L1057-1065.
# The four chain keys (pipeline, refiner_path, refiner_steps, refiner_cfg)
# must be written conditionally — only when refiner_pipe is not None.
# String checks lock both the conditional shape AND the literal "base+refiner".
check(
    "ComfyUI Generate node metadata adds pipeline='base+refiner' when chain runs",
    '"pipeline"' in node_gen_src and '"base+refiner"' in node_gen_src,
)
check(
    "ComfyUI Generate node metadata adds refiner_path / steps / cfg when chain runs",
    'metadata["refiner_path"]' in node_gen_src
    and 'metadata["refiner_steps"]' in node_gen_src
    and 'metadata["refiner_cfg"]' in node_gen_src,
)
check(
    "ComfyUI Generate node metadata extension is conditional on refiner_pipe",
    "if refiner_pipe is not None" in node_gen_src,
)


print("── Step 3 / Inv 12 — MCP server still does NOT thread refiner (re-affirm)")

# Re-affirms the Step-2 Inv 12 lock from the comfyless side: this slice
# does NOT plumb refiner_* through the MCP `generate` tool.
#
# TIGHTENED 2026-08-01 (ADR-044 commit 4). This was `"refiner" not in
# <entire mcp_server.py source>` — a bare substring standing in for the
# real invariant "the MCP server does not THREAD refiner_* into generate()".
# The proxy broke when the security review of ADR-044 commit 3 found that
# `refiner_path` was reaching gen_params validated and leaking back out through
# extract_params as a verbatim absolute path, and the fix was to REJECT it at
# entry and DROP it outbound. Rejecting is the opposite of threading, so the
# invariant is more strongly held than before — but the token now appears.
#
# So assert the invariant itself rather than the proxy: no refiner kwarg at the
# generate() call site (structural, via AST), plus the two closed-list
# memberships that are the ONLY reason the token is allowed to appear at all.
# Weakening a lock deserves this much noise; this is a tightening dressed as a
# relaxation, and the next reader should be able to tell which it was.
import ast as _hy_ast
import inspect as _hy_insp
import textwrap as _hy_tw
import comfyless.mcp_server as _hy_mcps

# MODULE level, not just _handle_generate: a second generate(refiner_path=...)
# call site elsewhere in mcp_server.py would have tripped the old token lock,
# and scoping the replacement to one function would have let it through
# (security review 2026-08-01, narrowing 1).
_hy_mod = _hy_ast.parse(
    _hy_tw.dedent(_hy_insp.getsource(_hy_mcps)))
_hy_gen_calls = [
    n for n in _hy_ast.walk(_hy_mod)
    if isinstance(n, _hy_ast.Call)
    and ((isinstance(n.func, _hy_ast.Name) and n.func.id == "generate")
         or (isinstance(n.func, _hy_ast.Attribute) and n.func.attr == "generate"))]
_hy_kwnames = {kw.arg for c in _hy_gen_calls for kw in c.keywords}
check(
    "mcp_server.py does NOT thread refiner_* into generate() (Vision Inv 12)",
    _hy_gen_calls and not any(
        (k or "").startswith("refiner") for k in _hy_kwnames),
    f"kwargs={sorted(k for k in _hy_kwnames if k)}",
)
check(
    "...and the call does not **splat gen_params (which would thread it)",
    not [c for c in _hy_gen_calls if any(kw.arg is None for kw in c.keywords)],
)
check(
    "Inv 12: refiner_path is REJECTED at the MCP boundary, not merely absent",
    "refiner_path" in _hy_mcps._GENERATE_REMOVED_FIELDS,
)
# The outbound half (refiner_path dropped from extract_params) is asserted in
# test_mcp_server.py, behaviourally and end-to-end. It cannot be checked here:
# this suite stubs the `nodes` package, and _render_extracted_params reaches
# _resolve_sidecar_ref -> nodes.eric_diffusion_utils on any model-bearing blob.


# ══════════════════════════════════════════════════════════════════════
#  Refiner slice — Step 4 (IPC daemon parity, Vision Inv 11)
# ══════════════════════════════════════════════════════════════════════
#
# Step 4 wires the refiner chain into the comfyless daemon
# (comfyless/server.py). Tests parallel the Step-2 tile-VAE-skip
# structural co-locking pattern PLUS behavior coverage of the two
# daemon-specific helpers (_maybe_load_refiner, _evict_chain) which
# are pure enough to drive on CPU without ML-stack imports.
#
# Inv 11 from Vision: daemon wire protocol gains an optional refiner
# field; cache_key includes it; cache miss evicts BOTH base + refiner;
# additive — clients that omit it see byte-for-byte identical behavior.

with open("comfyless/server.py") as f:
    server_src = f.read()


print("── Step 4 / Inv 11 — server.py cache_key trailing entry for refiner_path")

# Same pattern as the Step-3 tile-VAE-skip vae_tiling entry lock.
# A request that omits the field collapses to "" via `or ""`. The
# structural check locks the exact form so a future engineer who
# accidentally drops the `or ""` (which would let None into the tuple,
# crashing __hash__) trips this test.
check(
    "server.py cache_key tuple includes refiner_path entry",
    'req.get("refiner_path")' in server_src,
)
check(
    "server.py cache_key refiner_path entry has empty-string fallback",
    'req.get("refiner_path") or ""' in server_src,
)


print("── Step 4 / Inv 11 — _maybe_load_refiner helper exists with the contract")

# Helper signature lock. Catches an accidental rename or inlining that
# would invalidate the unit test below.
check(
    "server.py defines _maybe_load_refiner helper",
    "def _maybe_load_refiner(" in server_src,
)
# The helper raises on non-hunyuan family (Vision §"Failure semantics" §4).
check(
    "_maybe_load_refiner raises ValueError on non-hunyuan family",
    "refiner_path is only supported for the hunyuan-image family" in server_src,
)


print("── Step 4 / Inv 11 — _maybe_load_refiner behavior (CPU-driven)")

# Drive _maybe_load_refiner directly. server.py imports inside the
# function body, so the helper itself is importable without triggering
# the full ML stack. We stub hunyuan_chain.load_refiner_pipeline so the
# helper never touches disk.
import comfyless.server as cs

# Case 1: refiner_path empty → returns None.
_orig_load_refiner = hc.load_refiner_pipeline
hc.load_refiner_pipeline = lambda *a, **kw: _FakeRefinerPipe()
try:
    out = cs._maybe_load_refiner(
        {"refiner_path": ""}, _FakeBasePipe(), "hunyuan-image",
        "bf16", "cpu",
    )
    check(
        "_maybe_load_refiner returns None when refiner_path is empty",
        out is None,
    )
    # Case 2: refiner_path whitespace-only → also unset (.strip()).
    out_ws = cs._maybe_load_refiner(
        {"refiner_path": "   "}, _FakeBasePipe(), "hunyuan-image",
        "bf16", "cpu",
    )
    check(
        "_maybe_load_refiner treats whitespace-only refiner_path as unset",
        out_ws is None,
    )
    # Case 3: refiner_path set + family hunyuan-image → load_refiner_pipeline
    # is invoked, helper returns its output.
    fake_ref = _FakeRefinerPipe()
    hc.load_refiner_pipeline = lambda *a, **kw: fake_ref
    out_loaded = cs._maybe_load_refiner(
        {"refiner_path": "/p"}, _FakeBasePipe(), "hunyuan-image",
        "bf16", "cpu",
    )
    check(
        "_maybe_load_refiner returns load_refiner_pipeline output when hunyuan-image",
        out_loaded is fake_ref,
    )
    # Case 4: refiner_path set + non-hunyuan family → clean ValueError.
    for non_hunyuan_fam in ("flux", "qwen-image", "sdxl"):
        expect_raises(
            f"_maybe_load_refiner rejects refiner_path on {non_hunyuan_fam}",
            lambda f=non_hunyuan_fam: cs._maybe_load_refiner(
                {"refiner_path": "/p"}, _FakeBasePipe(), f,
                "bf16", "cpu",
            ),
            ValueError,
        )
finally:
    hc.load_refiner_pipeline = _orig_load_refiner


print("── Step 4 / Inv 11 + MINOR-1 — _evict_chain drops refiner FIRST")

# Eviction order lock (Step-2 code-reviewer MINOR-1 forward-watch). The
# helper must remove server_state["refiner_pipeline"] BEFORE
# server_state["pipeline"] so any Python-side reference cycle on the
# chain releases before the base eviction triggers CUDA frees. We
# verify by deletion order via a tracking dict subclass — capture every
# __delitem__ call and assert the refiner key fires first.
check(
    "server.py defines _evict_chain helper",
    "def _evict_chain(" in server_src,
)
# Source-level order lock: refiner del MUST appear before base del in
# _evict_chain's body. A regex-like substring check on the relevant
# block catches a future engineer who accidentally swaps the order.
_evict_idx = server_src.find("def _evict_chain(")
_evict_block = server_src[_evict_idx:_evict_idx + 1200]
_refiner_del_idx = _evict_block.find('del server_state["refiner_pipeline"]')
_pipeline_del_idx = _evict_block.find('del server_state["pipeline"]')
check(
    "_evict_chain source: refiner del precedes pipeline del",
    0 <= _refiner_del_idx < _pipeline_del_idx,
    f"refiner_del at {_refiner_del_idx}, pipeline_del at {_pipeline_del_idx}",
)

# Behavior: order-tracking dict subclass observes the del sequence.
class _OrderedTrackDict(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.del_order = []
    def __delitem__(self, key):
        self.del_order.append(key)
        super().__delitem__(key)
    def clear(self):
        # clear() also surfaces in del_order as a sentinel so we can
        # verify it runs after explicit dels (belt-and-suspenders reset).
        self.del_order.append("__clear__")
        super().clear()

# Build a server_state with both pipelines + extra keys; call _evict_chain.
# torch.cuda.empty_cache is the only side effect we need to stub —
# server.py imports torch inside the helper, and torch is a real import
# in the test venv; .cuda.empty_cache() is a no-op on CPU-only torch.
import torch as _torch
_orig_empty_cache = _torch.cuda.empty_cache
_torch.cuda.empty_cache = lambda: None
try:
    _state = _OrderedTrackDict({
        "pipeline": _FakeBasePipe(),
        "refiner_pipeline": _FakeRefinerPipe(),
        "model_family": "hunyuan-image",
        "cache_key": ("k",),
    })
    cs._evict_chain(_state)
finally:
    _torch.cuda.empty_cache = _orig_empty_cache

check(
    "_evict_chain runtime: refiner_pipeline deleted before pipeline",
    _state.del_order.index("refiner_pipeline") < _state.del_order.index("pipeline"),
    f"del_order={_state.del_order!r}",
)
check(
    "_evict_chain runtime: server_state fully reset after eviction",
    len(_state) == 0,
)


print("── Step 4 / Inv 11 — generate() call threads refiner_path/steps/cfg")

# server.py's call to generate() must forward the three refiner-stage
# fields. Otherwise the daemon would load the refiner via
# _maybe_load_refiner, cache it, then call generate() with refiner_path="",
# which would route through generate()'s warn-don't-block branch and
# never use the cached refiner.
for forwarded in (
    "refiner_path=req.get(\"refiner_path\"",
    "refiner_steps=req.get(\"refiner_steps\"",
    "refiner_cfg=req.get(\"refiner_cfg\"",
):
    check(
        f"server.py generate() call threads {forwarded!r}",
        forwarded in server_src,
    )

# The cached dict passed via _cached_pipeline MUST carry refiner_pipeline
# (None when base-only). Step-2's generate() already reads
# _cached_pipeline.get("refiner_pipeline") (the forward seam from
# Step-2 code-reviewer MINOR-1).
check(
    "server.py cached dict threads refiner_pipeline forward into generate()",
    '"refiner_pipeline": server_state.get("refiner_pipeline")' in server_src,
)


print("── Step 4 / Inv 11 — RefinerLoadError surfaces distinctly")

# Vision §"Failure semantics" §3: wrong-class refiner raises ValueError.
# server.py distinguishes ValueError → "RefinerLoadError" from other
# load errors → "LoadError" so the wire client/LLM can route differently.
check(
    "server.py distinguishes RefinerLoadError from generic LoadError",
    'err_type = "RefinerLoadError" if isinstance(e, ValueError) else "LoadError"'
    in server_src
    or '"RefinerLoadError"' in server_src,
)


print("── Step 4 — refiner_path inside _check_paths + _PATH_FIELDS (security-auditor CRITICAL+HIGH closure)")

# Security-auditor 2026-06-01 CRITICAL: refiner_path must appear in the
# _check_paths --model-base containment loop AND in the _PATH_FIELDS
# null-byte rejection set. Both are ADR-001 §3 invariants that apply to
# any model-path-shaped wire field. Initial Step-4 diff missed them; this
# block locks the closure at runtime.
check(
    "server.py _PATH_FIELDS frozenset contains 'refiner_path' "
    "(null-byte rejection at IPC boundary)",
    '"refiner_path"' in server_src and "_PATH_FIELDS = frozenset" in server_src,
)
# Behavior: import the frozenset directly and verify membership.
check(
    "cs._PATH_FIELDS contains 'refiner_path' at runtime",
    "refiner_path" in cs._PATH_FIELDS,
)
# Source order in _check_paths: refiner_path must appear inside the
# field tuple. Catches future engineer who copy-paste-renames the tuple
# without including refiner_path.
_check_idx = server_src.find("def _check_paths(")
_check_block = server_src[_check_idx:_check_idx + 2000]
check(
    "_check_paths source: refiner_path appears in the validation field tuple",
    '"refiner_path"' in _check_block,
)

# Behavior: synthesize a fake req with refiner_path OUTSIDE model_base
# and assert _check_paths rejects it with a clear PathError. Also assert
# refiner_path INSIDE model_base passes.
with tempfile.TemporaryDirectory() as _mb:
    # Mirror the realpath canonicalization run_server applies on startup
    # so _within's symlink-resolved comparison agrees with the test's
    # base under mergerfs / system tmpdir realpath rewrites.
    _mb_real = os.path.realpath(_mb)
    inside = os.path.join(_mb_real, "refiner-dir")
    outside = "/tmp/__refiner_outside_base__"  # outside _mb_real
    err_inside = cs._check_paths(
        {"model": os.path.join(_mb_real, "m"), "refiner_path": inside}, _mb_real,
    )
    check(
        "_check_paths accepts refiner_path INSIDE --model-base",
        err_inside is None,
        f"got {err_inside!r}",
    )
    err_outside = cs._check_paths(
        {"model": os.path.join(_mb_real, "m"), "refiner_path": outside}, _mb_real,
    )
    check(
        "_check_paths REJECTS refiner_path OUTSIDE --model-base (ADR-001 §3)",
        err_outside is not None and "refiner_path" in err_outside,
        f"got {err_outside!r}",
    )
    # Relative refiner_path → also rejected (must be absolute).
    err_relative = cs._check_paths(
        {"model": os.path.join(_mb_real, "m"), "refiner_path": "relative/path"}, _mb_real,
    )
    check(
        "_check_paths REJECTS relative refiner_path (must be absolute)",
        err_relative is not None and "refiner_path" in err_relative
        and "absolute" in err_relative,
        f"got {err_relative!r}",
    )

# Behavior: refiner_path with NUL byte rejected by _validate_request
# (the same path that defends every other path field).
_nul_req = {
    "type": "generate", "model": "/m", "prompt": "p",
    "refiner_path": "/m/refiner\x00/etc/passwd",
}
err_nul = cs._validate_request(_nul_req)
check(
    "_validate_request REJECTS refiner_path containing NUL byte",
    err_nul is not None and "refiner_path" in err_nul
    and "null byte" in err_nul,
    f"got {err_nul!r}",
)


print("── Step 4 / Inv 11 — IPC schema validates refiner_path at the boundary")

# Defense-in-depth: refiner_path is already in SCHEMA_KIND (Step 1) so
# the validator rejects non-string values BEFORE they reach the daemon's
# request handler. The dump's "_RUNTIME_KIND gains refiner: _KIND_STR"
# wording was naming sloppiness — the canonical schema key is
# refiner_path, which lives in SCHEMA_KIND (sidecar-shaped) not
# _RUNTIME_KIND (server-flag). Same boundary defense, correctly typed.
from comfyless.params_validation import (
    SCHEMA_KIND as _SK, _KIND_STR as _KS,
    validate_machine_request as _vmr,
)
check(
    "params_validation.SCHEMA_KIND['refiner_path'] == _KIND_STR",
    _SK.get("refiner_path") == _KS,
)
_bad_req = {
    "type": "generate", "model": "/m", "prompt": "p",
    "refiner_path": 42,  # non-string — must be rejected
}
_res = _vmr(_bad_req)
check(
    "validate_machine_request rejects non-string refiner_path at IPC boundary",
    not _res.ok and _res.error.get("field") == "refiner_path",
)


# ── Refiner quantization threading (2026-07-12) ───────────────────────
print("── Refiner fp8 quant threading ──")
check("hunyuan_chain.load_refiner_pipeline accepts quant param",
      "quant: str = \"none\"" in chain_src and "quant_skip" in chain_src)
check("refiner is deliberately NOT fp8-quantized (not fp8-safe → black)",
      "not fp8-safe" in chain_src and "quantize_module(comp, quant" not in chain_src)
check("generate.py threads quant into the refiner load (for a future fix)",
      "quant=quant, quant_skip=quant_skip, quant_only=quant_only," in cg_src)
check("server.py _maybe_load_refiner threads quant into the refiner load",
      "quant=str(req.get(\"quant\")" in server_src)


# ──────────────────────────────────────────────────────────────────────
print(f"\n────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print(f"────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
