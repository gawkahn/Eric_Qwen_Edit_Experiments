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
    "eric_diffusion_samplers", "comfyless/core/eric_diffusion_samplers.py"
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

# Iterate the FULL registry (minus "default"), not a hardcoded pair: the
# ADR-028 schedule gate evaluates the PRE-swap default scheduler as a proxy for
# the swapped sampler's sigma-acceptance, so EVERY registry sampler must accept
# `sigmas` in set_timesteps or a future entry would pass the gate then crash in
# retrieve_timesteps. This test forces that invariant on any new sampler.
for name in [n for n in samplers_mod.sampler_choices() if n != "default"]:
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
#  RES exponential-multistep samplers: res_2m / res_3m (ADR-029)
# ═══════════════════════════════════════════════════════════════════════
print("\n── RES samplers (res_2m / res_3m) ─────────────────────────────")
import math as _math  # noqa: E402


def _phi_closed(j, z):
    """Reference φ_j(z) = (e^z - Σ_{k<j} z^k/k!)/z^j (closed form)."""
    rem = sum(z ** k / _math.factorial(k) for k in range(j))
    return (_math.exp(z) - rem) / z ** j


# 1) φ-function port matches the RES4LYF closed form (to float precision; our
#    Taylor branch near z=0 is actually more accurate than the cancelling
#    closed form, so a slightly loose tol is correct here).
_phi_err = max(abs(samplers_mod._phi(j, z) - _phi_closed(j, z))
               for z in (-2.0, -1.0, -0.5, -0.1, -0.02) for j in (1, 2))
check("res:_phi matches RES4LYF closed form", _phi_err < 1e-8,
      f"max err {_phi_err}")
# Taylor branch (small z) stays finite and near the analytic limit 1/(j)!,
# where the naive closed form would divide ~0/0.
check("res:_phi stable at tiny z (no blowup)",
      abs(samplers_mod._phi(1, -1e-6) - 1.0) < 1e-4
      and abs(samplers_mod._phi(2, -1e-6) - 0.5) < 1e-4)

# 2) res_2m / res_3m build from a real FlowMatchEuler config and their
#    set_timesteps accepts sigmas (the drop-in / ADR-028 proxy invariant is
#    also pinned by the registry-wide loop above).
for _name in ("res_2m", "res_3m"):
    _sc = samplers_mod._build_sampler_scheduler(
        _name, diffusers.FlowMatchEulerDiscreteScheduler())
    check(f"{_name}_builds", type(_sc).__name__.startswith("FlowRES"))
    check(f"{_name}_in_registry", _name in samplers_mod.sampler_choices())

# 3) res_2m step matches the CORRECT exponential update — form (A):
#    e^{-h}·x + h·(b1·D_n + b2·D_{n-1}) with RAW denoised, as in ComfyUI core
#    res_multistep and RES4LYF's main sampler loop. The reference is that
#    independently-known-correct form, NOT the implementation's own expression.
_base = diffusers.FlowMatchEulerDiscreteScheduler()
_sigs = [1.0, 0.7, 0.45, 0.25, 0.1, 0.02]   # monotone descending, ends > 0
_sc = samplers_mod._build_sampler_scheduler("res_2m", _base)
_sc.set_timesteps(sigmas=_sigs, device="cpu")
_x = torch.randn(1, 4, 8, 8)
_v0 = torch.randn(1, 4, 8, 8)
_x1 = _sc.step(_v0, _sc.timesteps[0], _x, return_dict=False)[0]  # Euler, buffers D_prev
_v1 = torch.randn(1, 4, 8, 8)
_si = _sc.step_index
_sg = float(_sc.sigmas[_si]); _sgn = float(_sc.sigmas[_si + 1]); _sgp = float(_sc.sigmas[_si - 1])
_D_cur = _x1 - _sg * _v1
_D_prev = _x - float(_sc.sigmas[0]) * _v0
_h = -_math.log(_sgn / _sg); _hp = -_math.log(_sg / _sgp); _c2 = -_hp / _h
_b2 = _phi_closed(2, -_h) / _c2; _b1 = _phi_closed(1, -_h) - _b2
_ref = _math.exp(-_h) * _x1 + _h * (_b1 * _D_cur + _b2 * _D_prev)   # raw denoised
_got = _sc.step(_v1, _sc.timesteps[_si], _x1, return_dict=False)[0]
check("res_2m step matches the exponential update (raw-denoised form A)",
      (_got - _ref).abs().max().item() < 1e-5,
      f"max diff {(_got - _ref).abs().max().item()}")

# 3b) res_3m order-3 branch (γ / b3 path): after two bootstrap steps, step 2 must
#     match a form-(A) res_3m reference — otherwise a wrong γ is invisible.
_sc3 = samplers_mod._build_sampler_scheduler("res_3m", _base)
_sc3.set_timesteps(sigmas=_sigs, device="cpu")
_xa = torch.randn(1, 4, 8, 8)
_va = torch.randn(1, 4, 8, 8); _Da = _xa - float(_sc3.sigmas[0]) * _va
_xb = _sc3.step(_va, _sc3.timesteps[0], _xa, return_dict=False)[0]   # step0 Euler
_vb = torch.randn(1, 4, 8, 8); _Db = _xb - float(_sc3.sigmas[1]) * _vb
_xc = _sc3.step(_vb, _sc3.timesteps[1], _xb, return_dict=False)[0]   # step1 res_2m
_vc = torch.randn(1, 4, 8, 8)
_i2 = _sc3.step_index  # 2
_s2 = float(_sc3.sigmas[_i2]); _s2n = float(_sc3.sigmas[_i2 + 1])
_s2p = float(_sc3.sigmas[_i2 - 1]); _s2p2 = float(_sc3.sigmas[_i2 - 2])
_Dc = _xc - _s2 * _vc
_h2 = -_math.log(_s2n / _s2)
_c2_ = -(-_math.log(_s2 / _s2p)) / _h2
_c3_ = -(-_math.log(_s2 / _s2p2)) / _h2
_g = (3 * _c3_ ** 3 - 2 * _c3_) / (_c2_ * (2 - 3 * _c2_))
_B3 = _phi_closed(2, -_h2) / (_g * _c2_ + _c3_); _B2 = _g * _B3
_B1 = _phi_closed(1, -_h2) - _B2 - _B3
_ref3 = _math.exp(-_h2) * _xc + _h2 * (_B1 * _Dc + _B2 * _Db + _B3 * _Da)
_got3 = _sc3.step(_vc, _sc3.timesteps[_i2], _xc, return_dict=False)[0]
check("res_3m order-3 step matches the res_3m exponential update",
      (_got3 - _ref3).abs().max().item() < 1e-5,
      f"max diff {(_got3 - _ref3).abs().max().item()}")

# 4) DEFINITIVE correctness check — exactness on a constant nonlinearity. A model
#    whose denoised ≡ x0* (velocity v=(x-x0*)/σ) has the analytic solution
#    x_i - x0* = (σ_i/σ_0)(x_init - x0*) at EVERY step; exponential (and Euler)
#    methods are exact here. The previously-shipped double-counted form drifted
#    toward x0*/2 and would FAIL this (the old σ→0-terminal convergence check
#    masked it via the final Euler step).
for _name in ("res_2m", "res_3m"):
    _sc = samplers_mod._build_sampler_scheduler(_name, _base)
    _sc.set_timesteps(sigmas=_sigs, device="cpu")
    _x0star = torch.randn(1, 4, 8, 8)
    _xinit = torch.randn(1, 4, 8, 8)
    _x = _xinit.clone()
    _s0 = float(_sc.sigmas[0])
    _worst = 0.0
    for _i, _ts in enumerate(_sc.timesteps):
        _sg = float(_sc.sigmas[_i])
        _v = (_x - _x0star) / _sg
        _x = _sc.step(_v, _ts, _x, return_dict=False)[0]
        _exact = _x0star + (float(_sc.sigmas[_i + 1]) / _s0) * (_xinit - _x0star)
        _worst = max(_worst, (_x - _exact).abs().max().item())
    check(f"{_name}_exact_on_constant_denoised", _worst < 1e-4,
          f"worst per-step deviation {_worst}")


# ═══════════════════════════════════════════════════════════════════════
#  comfyless --schedule gate (ADR-028): _sigma_schedule_gate
# ═══════════════════════════════════════════════════════════════════════
print("\n── comfyless _sigma_schedule_gate (ADR-028) ──────────────────")
import comfyless.generate as _cg  # noqa: E402


class _GatePipe:
    """Fake pipe whose __call__ accepts a sigmas= kwarg (like every modern
    diffusers pipeline)."""
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, prompt=None, num_inference_steps=None, sigmas=None):
        pass


class _NoSigmasPipe:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, prompt=None):
        pass


_flow = _GatePipe(diffusers.FlowMatchEulerDiscreteScheduler())
_classic = _GatePipe(diffusers.EulerDiscreteScheduler())

# linear / unset are silent no-ops (the pipeline default needs no reshaping).
check("schedule 'linear' is a silent no-op",
      _cg._sigma_schedule_gate(_flow, "linear", "flux", 20) == (None, None))
check("schedule '' (unset) is a silent no-op",
      _cg._sigma_schedule_gate(_flow, "", "flux", 20) == (None, None))

# Happy path: flow-match scheduler → real sigmas, no warning.
_sig, _w = _cg._sigma_schedule_gate(_flow, "karras", "flux", 20)
check("flow-match: karras yields sigmas, no warning",
      _w is None and _sig is not None and len(_sig) == 20)
check("karras sigmas descend from 1.0 toward sigma_min",
      _sig[0] == 1.0 and _sig[0] > _sig[-1])
_sig_b, _ = _cg._sigma_schedule_gate(_flow, "balanced", "flux", 20)
check("balanced and karras produce different spacings", _sig_b != _sig)

# --schedule is ORTHOGONAL to --sampler (ADR-028 amendment 2026-07-13): the gate
# takes no sampler argument (no default-only restriction). The composition is
# correct because a real multistep scheduler subclasses FlowMatchEuler and stores
# externally-supplied sigmas verbatim — so karras spacing flows into the AB2
# integrator. This round-trip is the load-bearing evidence.
_ms_sched = samplers_mod._build_sampler_scheduler(
    "multistep2", diffusers.FlowMatchEulerDiscreteScheduler())
_ms_sched.set_timesteps(sigmas=[1.0, 0.6, 0.3, 0.1], device="cpu")
check("multistep scheduler consumes external sigmas (spacing ⟂ integrator)",
      [round(float(s), 3) for s in _ms_sched.sigmas.tolist()][:4]
      == [1.0, 0.6, 0.3, 0.1])

# Gate rejections all warn-and-ignore (sigmas None, reason string set).
_s, _w = _cg._sigma_schedule_gate(_classic, "karras", "sdxl", 20)
check("classic (non-flow-match) scheduler → schedule ignored with a reason",
      _s is None and _w is not None and "non-flow-match" in _w)
_heun = _GatePipe(diffusers.FlowMatchHeunDiscreteScheduler())
_s, _w = _cg._sigma_schedule_gate(_heun, "karras", "flux", 20)
check("FlowMatchHeun → schedule ignored (set_timesteps lacks sigmas)",
      _s is None and _w is not None and "set_timesteps" in _w)
_s, _w = _cg._sigma_schedule_gate(
    _NoSigmasPipe(diffusers.FlowMatchEulerDiscreteScheduler()), "karras", "flux", 20)
check("pipeline without a sigmas= kwarg → schedule ignored",
      _s is None and _w is not None and "does not accept" in _w)
check("scheduler=None → warn-and-ignore, no crash",
      _cg._sigma_schedule_gate(_GatePipe(None), "karras", "x", 20)[0] is None)

# The wiring seam _apply_sigma_schedule is what generate() actually calls — it
# injects call_kwargs["sigmas"] on the happy path and returns the warning
# otherwise (the wiring gap this slice closed).
_ck = {"prompt": "x", "num_inference_steps": 20}
_w = _cg._apply_sigma_schedule(_ck, _flow, "karras", "flux", 20)
check("_apply_sigma_schedule injects sigmas into call_kwargs on the happy path",
      _w is None and "sigmas" in _ck and len(_ck["sigmas"]) == 20)
_ck2 = {"prompt": "x"}
_w2 = _cg._apply_sigma_schedule(_ck2, _classic, "karras", "sdxl", 20)
check("_apply_sigma_schedule returns a warning + injects nothing on reject",
      _w2 is not None and "sigmas" not in _ck2)
_ck3 = {"prompt": "x"}
_w3 = _cg._apply_sigma_schedule(_ck3, _flow, "linear", "flux", 20)
check("_apply_sigma_schedule is a silent no-op for linear",
      _w3 is None and "sigmas" not in _ck3)

# The set_timesteps root-cause gate: FlowMatchHeun's real set_timesteps has no
# sigmas parameter (the property the gate now checks directly).
import inspect as _inspect  # noqa: E402
check("FlowMatchHeun.set_timesteps genuinely lacks a sigmas param (gate basis)",
      "sigmas" not in _inspect.signature(
          diffusers.FlowMatchHeunDiscreteScheduler.set_timesteps).parameters)
check("FlowMatchEuler.set_timesteps has a sigmas param",
      "sigmas" in _inspect.signature(
          diffusers.FlowMatchEulerDiscreteScheduler.set_timesteps).parameters)


# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 50}")
print(f"  {passed} passed, {failed} failed")
print(f"{'=' * 50}")
sys.exit(1 if failed else 0)
