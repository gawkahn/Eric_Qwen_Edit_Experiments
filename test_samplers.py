#!/usr/bin/env python3
"""Test harness for eric_diffusion_samplers custom schedulers.

Verifies that the multistep schedulers:
  - Can be instantiated from a real FlowMatchEulerDiscreteScheduler config
  - Match the diffusers base step() output on the first step (Euler fallback)
  - Produce the expected AB2 / AB3 updates on subsequent steps on synthetic
    velocity data (no loaded model required)
  - Correctly reset their buffer state on set_timesteps
  - swap_sampler context manager restores the original scheduler cleanly
  - Handles ValueError inside the with-block without a double-yield bug

Runs without ComfyUI, GPU, or loaded diffusion models.
"""

import sys
import types
import importlib.util

import numpy as np
import torch
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


def check_close(name, a, b, atol=1e-6, rtol=1e-5):
    global passed, failed
    if torch.is_tensor(a):
        a = a.cpu().float()
    if torch.is_tensor(b):
        b = b.cpu().float()
    if torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=atol, rtol=rtol):
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        diff = (torch.as_tensor(a) - torch.as_tensor(b)).abs().max().item()
        print(f"  FAIL  {name}  max_diff={diff}")


# ── Import the samplers module directly (bypass ComfyUI __init__) ──────────
sys.path.insert(0, ".")

folder_paths_mock = types.ModuleType("folder_paths")
folder_paths_mock.get_folder_paths = lambda *a, **kw: []
folder_paths_mock.get_full_path = lambda *a, **kw: None
sys.modules["folder_paths"] = folder_paths_mock
for m in ("comfy", "comfy.utils", "comfy.model_management"):
    if m not in sys.modules:
        sys.modules[m] = types.ModuleType(m)

spec = importlib.util.spec_from_file_location(
    "eric_diffusion_samplers", "nodes/eric_diffusion_samplers.py"
)
samplers_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(samplers_mod)


# ── Build a reference FlowMatchEulerDiscreteScheduler ──────────────────────
#
# Config matches what the Flux pipeline uses at runtime.

FME_CONFIG = dict(
    num_train_timesteps=1000,
    shift=3.0,
    use_dynamic_shifting=True,
    base_shift=0.5,
    max_shift=1.15,
    base_image_seq_len=256,
    max_image_seq_len=4096,
)

reference = diffusers.FlowMatchEulerDiscreteScheduler(**FME_CONFIG)
reference.set_timesteps(num_inference_steps=20, device="cpu", mu=0.8)


print("\n=== Registry & choice list ===")

choices = samplers_mod.sampler_choices()
check("has_default", "default" in choices)
check("has_multistep2", "multistep2" in choices)
check("has_multistep3", "multistep3" in choices)


print("\n=== Instantiation from config ===")

# Build a multistep2 scheduler from the reference config
ms2 = samplers_mod._build_sampler_scheduler("multistep2", reference)
check("ms2_built", ms2 is not None)
check("ms2_correct_class",
      type(ms2).__name__ == "FlowMultistep2Scheduler",
      f"got {type(ms2).__name__}")
check("ms2_inherits_euler",
      isinstance(ms2, diffusers.FlowMatchEulerDiscreteScheduler))

ms3 = samplers_mod._build_sampler_scheduler("multistep3", reference)
check("ms3_built", ms3 is not None)
check("ms3_correct_class",
      type(ms3).__name__ == "FlowMultistep3Scheduler")

# Default returns None (no swap)
default_result = samplers_mod._build_sampler_scheduler("default", reference)
check("default_returns_none", default_result is None)

# Unknown raises
try:
    samplers_mod._build_sampler_scheduler("bogus_xyz", reference)
    check("unknown_raises", False, "did not raise")
except ValueError:
    check("unknown_raises", True)


print("\n=== set_timesteps signature introspection ===")
#
# Diffusers' retrieve_timesteps() uses inspect.signature() to check if the
# scheduler accepts custom `sigmas` and `mu`.  If set_timesteps is defined
# as `*args, **kwargs`, introspection sees zero named parameters and
# retrieve_timesteps rejects the scheduler with:
#   "The current scheduler class's set_timesteps does not support custom
#    sigmas schedules."
# So the method MUST have explicit named parameters that match the parent.

import inspect as _inspect

for name in ("multistep2", "multistep3"):
    sched = samplers_mod._build_sampler_scheduler(name, reference)
    sig = _inspect.signature(sched.set_timesteps)
    param_names = set(sig.parameters.keys())
    check(f"{name}_sig_has_sigmas", "sigmas" in param_names,
          f"set_timesteps signature missing 'sigmas': {param_names}")
    check(f"{name}_sig_has_mu", "mu" in param_names,
          f"set_timesteps signature missing 'mu': {param_names}")
    check(f"{name}_sig_has_num_inference_steps",
          "num_inference_steps" in param_names)
    check(f"{name}_sig_has_device", "device" in param_names)
    check(f"{name}_sig_has_timesteps", "timesteps" in param_names)


print("\n=== set_timesteps populates state from config ===")

ms2.set_timesteps(num_inference_steps=20, device="cpu", mu=0.8)
check("ms2_has_timesteps",
      hasattr(ms2, "timesteps") and len(ms2.timesteps) == 20,
      f"got {getattr(ms2, 'timesteps', None)}")
check("ms2_has_sigmas",
      hasattr(ms2, "sigmas") and len(ms2.sigmas) == 21,  # +1 terminal
      f"got len={len(getattr(ms2, 'sigmas', []))}")
check("ms2_buffer_reset",
      ms2._prev_model_output is None and ms2._prev_dt is None)


print("\n=== Multistep2 first step = Euler (matches reference) ===")

# The first call to step() should behave as plain Euler because no
# previous velocity exists.  On synthetic data we can verify bitwise match.

# Reset reference to step 0
ref = diffusers.FlowMatchEulerDiscreteScheduler(**FME_CONFIG)
ref.set_timesteps(num_inference_steps=20, device="cpu", mu=0.8)
ms2 = samplers_mod._build_sampler_scheduler("multistep2", ref)
ms2.set_timesteps(num_inference_steps=20, device="cpu", mu=0.8)

# Pretend latents and velocity
torch.manual_seed(42)
latents = torch.randn(1, 4096, 64)  # Flux-packed shape
v0 = torch.randn_like(latents)

ref_out = ref.step(v0, ref.timesteps[0], latents, return_dict=False)[0]
ms2_out = ms2.step(v0, ms2.timesteps[0], latents, return_dict=False)[0]
check_close("ms2_first_step_matches_euler", ms2_out, ref_out, atol=1e-5)


print("\n=== Multistep2 second step uses buffered velocity ===")

# After the first step the buffer is populated.  The second step should
# use Adams-Bashforth 2:
#   v_eff = (1 + r/2) * v_n - (r/2) * v_{n-1}
#   x_{n+1} = x_n + h_n * v_eff
#
# We verify by computing the expected output directly and comparing.

v1 = torch.randn_like(latents)
sigma_1 = ms2.sigmas[1]
sigma_2 = ms2.sigmas[2]
sigma_0 = ms2.sigmas[0]
h_cur = (sigma_2 - sigma_1).item()
h_prev = (sigma_1 - sigma_0).item()
r = h_cur / h_prev

expected_v_eff = (1.0 + r / 2.0) * v1 - (r / 2.0) * v0
expected_next = ms2_out + h_cur * expected_v_eff

ms2_out2 = ms2.step(v1, ms2.timesteps[1], ms2_out, return_dict=False)[0]
check_close("ms2_second_step_AB2", ms2_out2, expected_next, atol=1e-5)

# AB2 should differ from plain Euler on step 2 (unless v0 == v1)
euler_out2 = ms2_out + h_cur * v1
differ = (ms2_out2 - euler_out2).abs().max().item()
check("ms2_differs_from_euler_on_step2", differ > 1e-6,
      f"AB2 and Euler produced identical output (diff={differ})")


print("\n=== Multistep3 first three steps ===")

ref3 = diffusers.FlowMatchEulerDiscreteScheduler(**FME_CONFIG)
ref3.set_timesteps(num_inference_steps=20, device="cpu", mu=0.8)
ms3 = samplers_mod._build_sampler_scheduler("multistep3", ref3)
ms3.set_timesteps(num_inference_steps=20, device="cpu", mu=0.8)

x0 = torch.randn(1, 4096, 64)
v_a = torch.randn_like(x0)
v_b = torch.randn_like(x0)
v_c = torch.randn_like(x0)

# Step 0: Euler
x1 = ms3.step(v_a, ms3.timesteps[0], x0, return_dict=False)[0]
h0 = (ms3.sigmas[1] - ms3.sigmas[0]).item()
expected_x1 = x0 + h0 * v_a
check_close("ms3_step0_euler", x1, expected_x1, atol=1e-5)

# Step 1: AB2
x2 = ms3.step(v_b, ms3.timesteps[1], x1, return_dict=False)[0]
h1 = (ms3.sigmas[2] - ms3.sigmas[1]).item()
r1 = h1 / h0
expected_v_eff1 = (1.0 + r1 / 2.0) * v_b - (r1 / 2.0) * v_a
expected_x2 = x1 + h1 * expected_v_eff1
check_close("ms3_step1_AB2", x2, expected_x2, atol=1e-5)

# Step 2: AB3 (or AB2 fallback for non-uniform)
x3 = ms3.step(v_c, ms3.timesteps[2], x2, return_dict=False)[0]
check("ms3_step2_produces_output",
      x3.shape == x2.shape and not torch.allclose(x3, x2))


print("\n=== Buffer reset on new set_timesteps ===")

# After a second call to set_timesteps, the buffer should be None so the
# next step() falls back to Euler.
ms2.set_timesteps(num_inference_steps=10, device="cpu", mu=0.5)
check("ms2_buffer_cleared", ms2._prev_model_output is None)

latents_new = torch.randn(1, 1024, 64)
v_new = torch.randn_like(latents_new)
ref2 = diffusers.FlowMatchEulerDiscreteScheduler(**FME_CONFIG)
ref2.set_timesteps(num_inference_steps=10, device="cpu", mu=0.5)

ms2_out_new = ms2.step(v_new, ms2.timesteps[0], latents_new, return_dict=False)[0]
ref2_out_new = ref2.step(v_new, ref2.timesteps[0], latents_new, return_dict=False)[0]
check_close("ms2_first_step_after_reset_matches_euler",
            ms2_out_new, ref2_out_new, atol=1e-5)


print("\n=== swap_sampler context manager ===")


class _StubPipe:
    def __init__(self, scheduler):
        self.scheduler = scheduler


ref_sched = diffusers.FlowMatchEulerDiscreteScheduler(**FME_CONFIG)
ref_sched.set_timesteps(num_inference_steps=20, device="cpu", mu=0.8)
stub = _StubPipe(ref_sched)
original_id = id(stub.scheduler)

# default = no swap
with samplers_mod.swap_sampler(stub, "default"):
    check("swap_default_noop", id(stub.scheduler) == original_id)

# multistep2 swaps and restores
with samplers_mod.swap_sampler(stub, "multistep2"):
    check("swap_ms2_installed",
          type(stub.scheduler).__name__ == "FlowMultistep2Scheduler")
check("swap_ms2_restored", id(stub.scheduler) == original_id)

# multistep3 swaps and restores
with samplers_mod.swap_sampler(stub, "multistep3"):
    check("swap_ms3_installed",
          type(stub.scheduler).__name__ == "FlowMultistep3Scheduler")
check("swap_ms3_restored", id(stub.scheduler) == original_id)

# Exception inside with still restores
try:
    with samplers_mod.swap_sampler(stub, "multistep2"):
        raise RuntimeError("simulated failure")
except RuntimeError:
    pass
check("swap_restored_after_exception", id(stub.scheduler) == original_id)

# Regression: ValueError raised inside with must propagate, NOT trigger
# the double-yield bug (caught an earlier bug in swap_scheduler).
caught_value_error = False
double_yield_bug = False
try:
    with samplers_mod.swap_sampler(stub, "multistep2"):
        raise ValueError("simulated pipe() failure")
except ValueError:
    caught_value_error = True
except RuntimeError as e:
    if "didn't stop after throw" in str(e):
        double_yield_bug = True
check("swap_value_error_propagates", caught_value_error)
check("swap_no_double_yield", not double_yield_bug)
check("swap_restored_after_value_error", id(stub.scheduler) == original_id)

# Unknown name falls back with warning
with samplers_mod.swap_sampler(stub, "bogus_xyz"):
    check("swap_unknown_falls_back", id(stub.scheduler) == original_id)


# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 50}")
print(f"  {passed} passed, {failed} failed")
print(f"{'=' * 50}")
sys.exit(1 if failed else 0)
