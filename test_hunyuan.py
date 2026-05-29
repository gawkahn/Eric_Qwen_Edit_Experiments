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
    ("nodes.eric_diffusion_samplers",
     {"sampler_choices": lambda *a, **k: ["default"], "swap_sampler": None}),
    ("nodes.eric_diffusion_utils",   {"build_model_metadata": lambda *a, **k: {}}),
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
# pre-existing behavior on 8×/16× VAEs). Sweeps the full family roster the
# slice's Vision §"Invariants 2" enumerates, plus the Hunyuan-Image refiner
# (intentionally NOT in the default-off set in v1; the refiner slice owns
# the call about extending the no-tile default there).
AUTO_TILED_FAMILIES = [
    "qwen-edit", "qwen-image",
    "flux", "flux2", "flux2klein",
    "chroma", "auraflow",
    "sd1", "sd3", "sdxl",
    "zimage",
    "stablecascade",
    "hunyuan-image-refiner",
]
for fam in AUTO_TILED_FAMILIES:
    check(
        f"{fam} + auto → tiling True (preserves current behavior)",
        utils_mod.resolve_vae_tiling(fam, "auto") is True,
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
    and "from .eric_diffusion_utils import" in nodes_loader_src,
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
# Two _load_pipeline call sites exist in _handle_generate (initial load and
# LoRA-removal-failure reload path); both must thread vae_tiling.
check(
    "server.py threads vae_tiling to BOTH _load_pipeline call sites",
    server_src.count("vae_tiling=req.get(\"vae_tiling\") or \"auto\"") == 2,
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


# ──────────────────────────────────────────────────────────────────────
print(f"\n────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print(f"────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
