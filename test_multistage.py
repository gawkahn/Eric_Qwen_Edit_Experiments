#!/usr/bin/env python3
"""Test harness for multistage sigma math and node wiring.

Runs without ComfyUI, GPU, or loaded models.  Validates:
- Sigma schedule shapes (linear, balanced, karras)
- Denoise truncation
- Monotonicity
- Seed propagation modes
- Dimension alignment
- Stage-skip logic
"""

import sys
import math
import numpy as np

# ── Import functions directly, bypassing ComfyUI-dependent __init__.py ──
sys.path.insert(0, ".")

# Mock folder_paths so ComfyUI-dependent modules can import without error
import types
folder_paths_mock = types.ModuleType("folder_paths")
folder_paths_mock.get_folder_paths = lambda *a, **kw: []
folder_paths_mock.get_full_path = lambda *a, **kw: None
sys.modules["folder_paths"] = folder_paths_mock

# Mock comfy modules
for mod_name in ("comfy", "comfy.utils", "comfy.model_management"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Now we can import the actual functions
import importlib
_ms_mod = importlib.import_module("nodes.eric_qwen_image_multistage")
build_sigma_schedule = _ms_mod.build_sigma_schedule

# Load scheduler module directly to avoid the ComfyUI __init__.py chain
import importlib.util
_sched_spec = importlib.util.spec_from_file_location(
    "_eric_diffusion_scheduler", "nodes/eric_diffusion_scheduler.py"
)
_sched_mod = importlib.util.module_from_spec(_sched_spec)
_sched_spec.loader.exec_module(_sched_mod)
scheduler_choices = _sched_mod.scheduler_choices
is_flow_match = _sched_mod.is_flow_match
_build_scheduler = _sched_mod._build_scheduler
swap_scheduler = _sched_mod.swap_scheduler

_gen_mod = importlib.import_module("nodes.eric_diffusion_generate")
compute_dimensions = _gen_mod.compute_dimensions
DIMENSION_ALIGNMENT = _gen_mod.DIMENSION_ALIGNMENT


passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


# ═══════════════════════════════════════════════════════════════════════
#  Sigma schedule tests
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Sigma schedules ===")

# --- Basic shape ---
for sched in ("linear", "balanced", "karras"):
    sigmas = build_sigma_schedule(30, 1.0, schedule=sched)
    check(f"{sched}_length_full", len(sigmas) == 30,
          f"expected 30, got {len(sigmas)}")
    check(f"{sched}_starts_at_1", abs(sigmas[0] - 1.0) < 1e-6,
          f"first sigma={sigmas[0]}")
    check(f"{sched}_ends_near_min", sigmas[-1] > 0 and sigmas[-1] < 0.1,
          f"last sigma={sigmas[-1]}")

# --- Monotonic (strictly decreasing) ---
for sched in ("linear", "balanced", "karras"):
    sigmas = build_sigma_schedule(50, 1.0, schedule=sched)
    diffs = [sigmas[i] - sigmas[i + 1] for i in range(len(sigmas) - 1)]
    check(f"{sched}_monotonic", all(d > 0 for d in diffs),
          f"non-decreasing step found")

# --- Denoise truncation ---
for sched in ("linear", "balanced", "karras"):
    full = build_sigma_schedule(30, 1.0, schedule=sched)
    partial = build_sigma_schedule(30, 0.6, schedule=sched)
    expected_keep = max(1, round(30 * 0.6))
    check(f"{sched}_denoise_length", len(partial) == expected_keep,
          f"expected {expected_keep}, got {len(partial)}")
    check(f"{sched}_denoise_starts_lower", partial[0] < full[0],
          f"partial start {partial[0]} >= full start {full[0]}")

# --- All schedules start at the SAME sigma for a given denoise ---
starts = {}
for sched in ("linear", "balanced", "karras"):
    sigmas = build_sigma_schedule(30, 0.7, schedule=sched)
    starts[sched] = sigmas[0]
check("denoise_consistent_start",
      abs(starts["linear"] - starts["balanced"]) < 1e-6
      and abs(starts["linear"] - starts["karras"]) < 1e-6,
      f"starts differ: {starts}")

# --- Karras concentrates more steps at low sigma than linear ---
lin = build_sigma_schedule(30, 1.0, schedule="linear")
kar = build_sigma_schedule(30, 1.0, schedule="karras")
mid = 0.5
lin_below_mid = sum(1 for s in lin if s < mid) / len(lin)
kar_below_mid = sum(1 for s in kar if s < mid) / len(kar)
check("karras_more_low_sigma", kar_below_mid > lin_below_mid,
      f"karras {kar_below_mid:.0%} vs linear {lin_below_mid:.0%} below {mid}")

# --- Balanced is between linear and karras ---
bal = build_sigma_schedule(30, 1.0, schedule="balanced")
bal_below_mid = sum(1 for s in bal if s < mid) / len(bal)
check("balanced_between",
      lin_below_mid <= bal_below_mid <= kar_below_mid,
      f"lin={lin_below_mid:.0%} bal={bal_below_mid:.0%} kar={kar_below_mid:.0%}")


# ═══════════════════════════════════════════════════════════════════════
#  Seed propagation tests
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Seed propagation ===")


def make_seeds(mode: str, base_seed: int) -> list:
    """Simulate the seed logic that will be in the generate node."""
    if mode == "same_all_stages":
        return [base_seed, base_seed, base_seed]
    elif mode == "offset_per_stage":
        return [base_seed, base_seed + 1, base_seed + 2]
    elif mode == "random_per_stage":
        import random
        rng = random.Random(base_seed)
        return [base_seed, rng.randint(0, 2**63), rng.randint(0, 2**63)]
    else:
        raise ValueError(f"Unknown seed mode: {mode}")


seeds_same = make_seeds("same_all_stages", 42)
check("same_all_equal", seeds_same[0] == seeds_same[1] == seeds_same[2])

seeds_offset = make_seeds("offset_per_stage", 42)
check("offset_sequential",
      seeds_offset == [42, 43, 44],
      f"got {seeds_offset}")

seeds_rand = make_seeds("random_per_stage", 42)
check("random_different",
      len(set(seeds_rand)) == 3,
      f"got {seeds_rand}")

# Deterministic: same base seed → same random seeds
seeds_rand2 = make_seeds("random_per_stage", 42)
check("random_deterministic", seeds_rand == seeds_rand2)


# ═══════════════════════════════════════════════════════════════════════
#  Dimension alignment tests
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Dimension alignment ===")

for ratio_name, (wr, hr) in [("16:9", (16, 9)), ("1:1", (1, 1)),
                               ("9:16", (9, 16)), ("3:2", (3, 2))]:
    for mp in (0.5, 1.0, 2.0, 4.0):
        w, h = compute_dimensions(wr, hr, mp)
        check(f"align_{ratio_name}_{mp}mp",
              w % DIMENSION_ALIGNMENT == 0 and h % DIMENSION_ALIGNMENT == 0,
              f"{w}x{h} not aligned to {DIMENSION_ALIGNMENT}")
        actual_mp = w * h / 1e6
        check(f"mp_{ratio_name}_{mp}mp",
              abs(actual_mp - mp) / mp < 0.15,
              f"target {mp} MP, got {actual_mp:.2f} MP")


# ═══════════════════════════════════════════════════════════════════════
#  Stage-skip logic tests
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Stage-skip logic ===")


def count_stages(upscale_s2: float, upscale_s3: float) -> int:
    """Replicate the stage-skip logic from the generate node."""
    do_s2 = upscale_s2 > 0
    do_s3 = do_s2 and upscale_s3 > 0
    return 3 if do_s3 else (2 if do_s2 else 1)


check("skip_all_upscale", count_stages(0.0, 2.0) == 1)
check("skip_s3_only", count_stages(2.0, 0.0) == 2)
check("all_stages", count_stages(2.0, 2.0) == 3)
check("s3_ignored_when_s2_zero", count_stages(0.0, 4.0) == 1)


# ═══════════════════════════════════════════════════════════════════════
#  Scheduler registry + swap tests
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Scheduler registry ===")

choices = scheduler_choices()
check("has_default", "default" in choices)
check("has_flow_match_euler", "flow_match_euler" in choices)
check("has_flow_match_heun", "flow_match_heun" in choices)
# Classic schedulers removed from registry — see _SCHEDULER_REGISTRY docstring.
check("classics_removed",
      not any(n in choices for n in ("dpmpp_2m", "deis", "unipc", "euler")))

# Build a real FlowMatchEuler scheduler (matches what GEN_PIPELINE models use)
import diffusers
fme_config = dict(
    num_train_timesteps=1000,
    shift=3.0,
    use_dynamic_shifting=True,
    base_shift=0.5,
    max_shift=1.15,
    base_image_seq_len=256,
    max_image_seq_len=4096,
)
fme = diffusers.FlowMatchEulerDiscreteScheduler(**fme_config)

check("fme_is_flow_match", is_flow_match(fme))

# Verify every registered non-default scheduler builds without error
print("\n=== Scheduler build (FlowMatchEuler config) ===")
for name in choices:
    if name == "default":
        continue
    try:
        new_sched, cls = _build_scheduler(name, dict(fme.config))
        check(f"build_{name}", new_sched is not None,
              f"returned None for {name}")
    except Exception as e:
        check(f"build_{name}", False, f"raised: {e}")

# Verify is_flow_match correctly classifies each
print("\n=== is_flow_match classification ===")
expected_flow_match = {"flow_match_euler", "flow_match_heun"}
for name in choices:
    if name == "default":
        continue
    new_sched, _ = _build_scheduler(name, dict(fme.config))
    got = is_flow_match(new_sched)
    want = name in expected_flow_match
    check(f"flow_match_{name}", got == want,
          f"expected {want}, got {got}")


# ═══════════════════════════════════════════════════════════════════════
#  swap_scheduler context manager
# ═══════════════════════════════════════════════════════════════════════

print("\n=== swap_scheduler context manager ===")


class _PipeStub:
    def __init__(self, scheduler):
        self.scheduler = scheduler


stub = _PipeStub(fme)
original_id = id(stub.scheduler)

# Default should not swap
with swap_scheduler(stub, "default"):
    check("default_noop", id(stub.scheduler) == original_id)

# flow_match_heun should swap and restore
with swap_scheduler(stub, "flow_match_heun"):
    check("heun_swapped",
          type(stub.scheduler).__name__ == "FlowMatchHeunDiscreteScheduler")
check("heun_restored_after", id(stub.scheduler) == original_id)

# Exception inside context must still restore
try:
    with swap_scheduler(stub, "flow_match_heun"):
        raise RuntimeError("simulated failure")
except RuntimeError:
    pass
check("restored_after_exception", id(stub.scheduler) == original_id)

# Regression: ValueError raised inside the with block must propagate cleanly
# (not be caught by the build-error except, which would cause a double-yield
# and trigger "generator didn't stop after throw()").
caught_value_error = False
generator_error = False
try:
    with swap_scheduler(stub, "flow_match_heun"):
        raise ValueError("simulated pipe() failure inside with block")
except ValueError:
    caught_value_error = True
except RuntimeError as e:
    if "didn't stop after throw" in str(e):
        generator_error = True
check("value_error_propagates", caught_value_error,
      "ValueError from inside `with` should reach the caller cleanly")
check("no_double_yield_bug", not generator_error,
      "Double-yield in context manager — see swap_scheduler structure")
check("restored_after_value_error", id(stub.scheduler) == original_id)

# Unknown scheduler name should NOT raise — just log and use original
with swap_scheduler(stub, "nonexistent_xyz"):
    check("unknown_name_fallback", id(stub.scheduler) == original_id)


# ═══════════════════════════════════════════════════════════════════════
#  UltraGen wiring tests
# ═══════════════════════════════════════════════════════════════════════
#
# We can't run the full generate() method without a real loaded pipeline,
# but we can verify the node class is well-formed: INPUT_TYPES matches the
# generate() signature, defaults are sensible, and the helper functions
# (CFG routing, VAE compatibility, sigma math) work in isolation.

print("\n=== UltraGen wiring ===")

import inspect
ultragen_mod = importlib.import_module("nodes.eric_diffusion_ultragen")
UltraGen = ultragen_mod.EricDiffusionUltraGen

# Class attributes
check("ultragen_category", UltraGen.CATEGORY == "Eric Diffusion")
check("ultragen_function", UltraGen.FUNCTION == "generate")
check("ultragen_return_image", UltraGen.RETURN_TYPES == ("IMAGE", "GEN_METADATA"))

# INPUT_TYPES has both required pipeline and prompt
input_types = UltraGen.INPUT_TYPES()
required = input_types.get("required", {})
check("ultragen_pipeline_input", "pipeline" in required)
check("ultragen_prompt_input", "prompt" in required)

# Check pipeline input type
check("ultragen_pipeline_type",
      required.get("pipeline", (None,))[0] == "GEN_PIPELINE")

# Critical optional inputs that the generate() method consumes
optional = input_types.get("optional", {})
critical_optionals = (
    "negative_prompt", "aspect_ratio", "seed", "seed_mode",
    "max_sequence_length",
    "s1_mp", "s1_steps", "s1_cfg",
    "upscale_to_stage2", "s2_steps", "s2_cfg", "s2_denoise", "s2_sigma_schedule",
    "upscale_to_stage3", "s3_steps", "s3_cfg", "s3_denoise", "s3_sigma_schedule",
    "upscale_vae", "upscale_vae_mode",
)
for opt_name in critical_optionals:
    check(f"ultragen_has_{opt_name}", opt_name in optional)

# generate() signature must accept all the optional inputs (otherwise ComfyUI
# will pass them as kwargs and Python will TypeError).
sig = inspect.signature(UltraGen.generate)
sig_params = set(sig.parameters.keys())
for name in critical_optionals:
    check(f"ultragen_sig_accepts_{name}", name in sig_params)
check("ultragen_sig_accepts_pipeline", "pipeline" in sig_params)
check("ultragen_sig_accepts_prompt", "prompt" in sig_params)

# Sigma schedule choices match what the dropdown exposes
sigma_choices = optional.get("s2_sigma_schedule", (None,))[0]
check("ultragen_s2_sigma_choices",
      sigma_choices == ["linear", "balanced", "karras"],
      f"got {sigma_choices}")

seed_mode_choices = optional.get("seed_mode", (None,))[0]
check("ultragen_seed_mode_choices",
      seed_mode_choices == ["same_all_stages", "offset_per_stage", "random_per_stage"])

upscale_mode_choices = optional.get("upscale_vae_mode", (None,))[0]
check("ultragen_upscale_mode_choices",
      upscale_mode_choices == ["disabled", "inter_stage", "final_decode", "both"])


# ═══════════════════════════════════════════════════════════════════════
#  UltraGen helper functions
# ═══════════════════════════════════════════════════════════════════════

print("\n=== UltraGen helpers ===")

# _vae_supports_upscale: needs an object with .vae.config exposing the
# Qwen-specific keys.
class _MockConfig:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class _MockVAE:
    def __init__(self, config):
        self.config = config


class _MockPipe:
    def __init__(self, vae):
        self.vae = vae


# Qwen-compatible VAE: has z_dim, latents_mean, latents_std
qwen_vae_cfg = _MockConfig(z_dim=16, latents_mean=[0.0]*16, latents_std=[1.0]*16)
qwen_pipe = _MockPipe(_MockVAE(qwen_vae_cfg))
check("vae_supports_qwen_compatible",
      ultragen_mod._vae_supports_upscale(qwen_pipe))

# Flux VAE: scaling_factor + shift_factor, no z_dim
flux_vae_cfg = _MockConfig(scaling_factor=0.3611, shift_factor=0.1159)
flux_pipe = _MockPipe(_MockVAE(flux_vae_cfg))
check("vae_does_not_support_flux",
      not ultragen_mod._vae_supports_upscale(flux_pipe))

# Pipeline with no VAE attribute
class _NoVaePipe:
    pass
check("vae_handles_no_vae_attr",
      not ultragen_mod._vae_supports_upscale(_NoVaePipe()))

# Pipeline with vae=None
check("vae_handles_none_vae",
      not ultragen_mod._vae_supports_upscale(_MockPipe(None)))


# _cfg_kwargs: routes per model family
print("\n=== UltraGen CFG routing ===")


class _StubPipe:
    """Minimal pipe with just enough to satisfy inspect.signature(pipe.__call__)."""
    def __call__(self, prompt=None, negative_prompt=None, height=None, width=None,
                 num_inference_steps=None, true_cfg_scale=None, guidance_scale=None,
                 generator=None, callback_on_step_end=None, output_type=None,
                 sigmas=None, latents=None, max_sequence_length=None):
        pass


stub_pipe = _StubPipe()
cfg = ultragen_mod._cfg_kwargs(stub_pipe, "qwen-image", False, 4.0, "ugly", 1024)
check("cfg_qwen_uses_true_cfg", "true_cfg_scale" in cfg and cfg["true_cfg_scale"] == 4.0)
check("cfg_qwen_includes_neg", cfg.get("negative_prompt") == "ugly")

cfg = ultragen_mod._cfg_kwargs(stub_pipe, "qwen-image", False, 4.0, None, 1024)
check("cfg_qwen_omits_neg_when_none", "negative_prompt" not in cfg)

cfg = ultragen_mod._cfg_kwargs(stub_pipe, "flux", True, 3.5, "ugly", 512)
check("cfg_flux_uses_guidance", "guidance_scale" in cfg and cfg["guidance_scale"] == 3.5)
check("cfg_flux_omits_neg", "negative_prompt" not in cfg)
check("cfg_flux_includes_max_seq", cfg.get("max_sequence_length") == 512)

cfg = ultragen_mod._cfg_kwargs(stub_pipe, "flux2", True, 3.5, "ugly", 512)
check("cfg_flux2_uses_guidance", "guidance_scale" in cfg)

# Unknown family: introspection fallback
cfg = ultragen_mod._cfg_kwargs(stub_pipe, "exotic_model", False, 4.0, "ugly", 512)
check("cfg_unknown_routed_via_introspection",
      isinstance(cfg, dict))


# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 50}")
print(f"  {passed} passed, {failed} failed")
print(f"{'=' * 50}")
sys.exit(1 if failed else 0)
