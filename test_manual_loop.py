#!/usr/bin/env python3
"""Test harness for the manual denoising loop samplers.

Verifies that each sampler:
  - Runs without error on synthetic velocity data
  - Produces output of the correct shape
  - Matches the expected update rule (bitwise where applicable)
  - Converges to the correct ODE solution on a known analytic problem

Runs without ComfyUI, GPU, or loaded diffusion models — the "denoiser"
is a simple synthetic function whose true integral we know.

Analytic correctness test
-------------------------
We integrate the ODE  dx/dsigma = v(x, sigma)  with v(x, sigma) = -x
from x(1) = 1 over sigma in [1, 0].  The ODE dx/dsigma = -x has general
solution x(sigma) = C * exp(-sigma), and with x(1) = 1 we get C = e, so
the true value at sigma=0 is x(0) = e ≈ 2.71828.

The sampler update x_next = x + dsigma * (-x) is an explicit Euler
step on THIS ODE.  As sigma decreases (dsigma < 0), x grows toward e.
Different orders of sampler should converge to e with different error
rates (Euler O(h), Heun O(h²), RK3 O(h³)).

Different orders of sampler should converge to this value with
different error rates.  We run each sampler with the same step count
and check both absolute error and the expected ordering (RK3 < Heun < Euler).
"""

import sys
import math
import types
import importlib.util

import numpy as np
import torch


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


def check_close(name, a, b, atol=1e-6):
    global passed, failed
    if torch.is_tensor(a):
        a = a.cpu().float()
    if torch.is_tensor(b):
        b = b.cpu().float()
    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)
    if torch.allclose(a_t, b_t, atol=atol, rtol=0):
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        diff = (a_t - b_t).abs().max().item()
        print(f"  FAIL  {name}  max_diff={diff:.6e}")


# ── Import the manual loop module directly ───────────────────────────
sys.path.insert(0, ".")
sys.modules["folder_paths"] = types.ModuleType("folder_paths")
sys.modules["folder_paths"].get_folder_paths = lambda *a, **kw: []
sys.modules["folder_paths"].get_full_path = lambda *a, **kw: None
for m in ("comfy", "comfy.utils", "comfy.model_management"):
    if m not in sys.modules:
        sys.modules[m] = types.ModuleType(m)

spec = importlib.util.spec_from_file_location(
    "eric_diffusion_manual_loop", "nodes/eric_diffusion_manual_loop.py"
)
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


# ═══════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Registry ===")

names = ml.sampler_names()
for expected in ("flow_euler", "flow_heun", "flow_rk3",
                 "flow_multistep2", "flow_multistep3"):
    check(f"has_{expected}", expected in names)

check("cost_euler", ml.sampler_cost("flow_euler") == 1)
check("cost_heun", ml.sampler_cost("flow_heun") == 2)
check("cost_rk3", ml.sampler_cost("flow_rk3") == 3)
check("cost_multistep2", ml.sampler_cost("flow_multistep2") == 1)


# ═══════════════════════════════════════════════════════════════════════
#  Shape and basic correctness
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Basic shapes ===")

# Flux-like packed latent shape: [B, seq, C*4]
x0 = torch.randn(1, 4096, 64, dtype=torch.float32)
sigmas = torch.linspace(1.0, 0.0, 21)  # 20 steps + terminal zero

for name in names:
    fn = ml.get_sampler(name)
    out = fn(lambda x, s: torch.zeros_like(x), x0, sigmas)
    check(f"{name}_preserves_shape", out.shape == x0.shape,
          f"expected {x0.shape}, got {out.shape}")
    # With v=0 the output should equal the input exactly
    check_close(f"{name}_zero_velocity_noop", out, x0, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  Flow Euler matches the textbook update rule
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Flow Euler matches textbook update ===")

def const_velocity(x, sigma):
    return torch.ones_like(x)

x0 = torch.zeros(1, 4, 2, dtype=torch.float32)
sigmas = torch.tensor([1.0, 0.5, 0.0])
out = ml.flow_euler(const_velocity, x0, sigmas)

# Manual computation:
# step 0: v=1, dt = 0.5 - 1.0 = -0.5, x = 0 + (-0.5)*1 = -0.5
# step 1: v=1, dt = 0.0 - 0.5 = -0.5, x = -0.5 + (-0.5)*1 = -1.0
expected = torch.full_like(x0, -1.0)
check_close("euler_constant_v_two_steps", out, expected, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  Heun first step with two substeps on constant velocity
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Heun constant velocity ===")

x0 = torch.zeros(1, 4, 2, dtype=torch.float32)
sigmas = torch.tensor([1.0, 0.5, 0.0])
out = ml.flow_heun(const_velocity, x0, sigmas)

# Heun with constant v: (v + v) / 2 == v, so indistinguishable from Euler
# for THIS input.  But the code path is different — this confirms no
# catastrophic bug.
expected = torch.full_like(x0, -1.0)
check_close("heun_constant_v_matches_euler_output", out, expected, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  Analytic ODE convergence: dx/dt = -x, exact solution x(t) = exp(-(1-t))*x0
# ═══════════════════════════════════════════════════════════════════════
#
# Integration from t=1 (noise) to t=0 (clean). True x(0) = exp(-1)*x_0.
# We use sigma-as-time: sigmas descend from 1 to 0.
#
# Note: here "velocity" is dx/dt = -x. Euler update: x_{n+1} = x_n + dt*v_n
#       where dt = sigma_next - sigma_current (negative).

print("\n=== Analytic ODE convergence: dx/dt = -x ===")


def linear_decay_velocity(x, sigma):
    return -x


# Integrate from sigma=1 to sigma=0.1 (NOT exactly 0) so we measure the
# sampler's intrinsic convergence rate without the final-step Euler
# fallback introducing O(h) error.  In real Flux use, the loop's final
# step evaluates at a small positive sigma and updates to 0 — the
# fallback branch is only hit when sigma_next is literally zero.
SIGMA_START = 1.0
SIGMA_END = 0.1
# True solution: x(sigma) = C * exp(-sigma) with x(1) = 1 → C = e, so
# x(SIGMA_END) = e * exp(-SIGMA_END) = e^(1 - SIGMA_END)
true_solution = math.exp(SIGMA_START - SIGMA_END)


def run_on_analytic(sampler_name, num_steps):
    x0 = torch.tensor([1.0], dtype=torch.float64)
    sigmas = torch.linspace(SIGMA_START, SIGMA_END, num_steps + 1, dtype=torch.float64)
    fn = ml.get_sampler(sampler_name)
    result = fn(linear_decay_velocity, x0, sigmas)
    return result.item()

# Compare errors at a moderate step count
NUM_STEPS = 20

for name in ["flow_euler", "flow_heun", "flow_rk3",
             "flow_multistep2", "flow_multistep3"]:
    result = run_on_analytic(name, NUM_STEPS)
    error = abs(result - true_solution)
    print(f"    {name:20s}  result={result:.6f}  error={error:.3e}")
    check(f"{name}_converges_to_analytic", error < 0.1,
          f"error {error:.3e} too large")

# ── Heun should be STRICTLY more accurate than Euler ────────────────
print("\n=== Heun > Euler accuracy on this ODE ===")
euler_err = abs(run_on_analytic("flow_euler", NUM_STEPS) - true_solution)
heun_err = abs(run_on_analytic("flow_heun", NUM_STEPS) - true_solution)
check("heun_beats_euler", heun_err < euler_err,
      f"heun {heun_err:.3e} not < euler {euler_err:.3e}")

# ── RK3 should beat Heun at higher step counts ──────────────────────
print("\n=== RK3 > Heun accuracy on this ODE ===")
heun_err_hi = abs(run_on_analytic("flow_heun", 40) - true_solution)
rk3_err_hi = abs(run_on_analytic("flow_rk3", 40) - true_solution)
check("rk3_beats_heun_at_40_steps", rk3_err_hi < heun_err_hi,
      f"rk3 {rk3_err_hi:.3e} not < heun {heun_err_hi:.3e}")

# ── Multistep3 converges (buffer-based, not strict order comparison) ──
print("\n=== Multistep3 converges reasonably ===")
ms3_err = abs(run_on_analytic("flow_multistep3", NUM_STEPS) - true_solution)
check("multistep3_reasonable_error", ms3_err < 0.05,
      f"error {ms3_err:.3e} too large")


# ═══════════════════════════════════════════════════════════════════════
#  Convergence rate check — halving step size should quarter Heun error,
#  eighth RK3 error
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Convergence rates ===")

def convergence_ratio(name, steps_coarse, steps_fine):
    err_c = abs(run_on_analytic(name, steps_coarse) - true_solution)
    err_f = abs(run_on_analytic(name, steps_fine) - true_solution)
    return err_c / err_f if err_f > 1e-15 else float("inf")

euler_ratio = convergence_ratio("flow_euler", 10, 20)
heun_ratio = convergence_ratio("flow_heun", 10, 20)
rk3_ratio = convergence_ratio("flow_rk3", 10, 20)

print(f"    Euler 10→20 step error ratio:  {euler_ratio:.2f}  (expect ~2, 1st order)")
print(f"    Heun  10→20 step error ratio:  {heun_ratio:.2f}  (expect ~4, 2nd order)")
print(f"    RK3   10→20 step error ratio:  {rk3_ratio:.2f}  (expect ~8, 3rd order)")

check("euler_first_order", 1.5 < euler_ratio < 3.0,
      f"euler ratio {euler_ratio:.2f} outside [1.5, 3.0]")
check("heun_second_order", 3.0 < heun_ratio < 6.0,
      f"heun ratio {heun_ratio:.2f} outside [3.0, 6.0]")
check("rk3_third_order", 5.0 < rk3_ratio < 12.0,
      f"rk3 ratio {rk3_ratio:.2f} outside [5.0, 12.0]")


# ═══════════════════════════════════════════════════════════════════════
#  Flux sigma schedule: mu shift math
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Flux sigma schedule construction ===")

# Typical Flux scheduler config
flux_config = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
    "base_shift": 0.5,
    "max_shift": 1.15,
}

# Small sequence length → small mu
sigmas_small = ml.build_flux_sigmas(20, 256, flux_config, "linear")
check("sigmas_length_plus_one", len(sigmas_small) == 21,
      f"got {len(sigmas_small)}")
check("sigmas_starts_at_one", abs(float(sigmas_small[0]) - 1.0) < 1e-4,
      f"first={float(sigmas_small[0])}")
check("sigmas_ends_at_zero", abs(float(sigmas_small[-1])) < 1e-6,
      f"last={float(sigmas_small[-1])}")
check("sigmas_descending", all(float(sigmas_small[i]) >= float(sigmas_small[i+1])
                                for i in range(len(sigmas_small) - 1)))

# Large sequence length (higher resolution) → more mu shift → sigmas
# concentrated more towards higher values
sigmas_large = ml.build_flux_sigmas(20, 4096, flux_config, "linear")
check("high_res_has_more_high_sigma",
      float(sigmas_large[5]) > float(sigmas_small[5]),
      f"large={float(sigmas_large[5])} small={float(sigmas_small[5])}")

# Karras schedule produces different distribution
sigmas_karras = ml.build_flux_sigmas(20, 256, flux_config, "karras")
check("karras_different_from_linear",
      not np.allclose(np.asarray(sigmas_karras), np.asarray(sigmas_small), atol=1e-4))

# Beta schedules (new)
print("\n=== Beta and polyexp sigma schedules ===")

# Verify registry
check("schedule_names_has_beta57", "beta57" in ml.SIGMA_SCHEDULE_NAMES)
check("schedule_names_has_linear", "linear" in ml.SIGMA_SCHEDULE_NAMES)
check("schedule_names_has_karras", "karras" in ml.SIGMA_SCHEDULE_NAMES)

# Each schedule produces a valid descending 21-element tensor
for sched in ml.SIGMA_SCHEDULE_NAMES:
    s = ml.build_flux_sigmas(20, 256, flux_config, sched)
    check(f"{sched}_length_21", len(s) == 21, f"got {len(s)}")
    check(f"{sched}_starts_at_one", abs(float(s[0]) - 1.0) < 1e-3,
          f"first={float(s[0])}")
    check(f"{sched}_ends_at_zero", abs(float(s[-1])) < 1e-6,
          f"last={float(s[-1])}")
    # Descending (non-strict because some schedules can plateau)
    diffs = [float(s[i]) - float(s[i + 1]) for i in range(len(s) - 1)]
    check(f"{sched}_non_increasing", all(d >= -1e-6 for d in diffs))

# Verify different schedules produce meaningfully different distributions
raw_linear = ml._build_raw_sigmas(20, "linear")
raw_beta57 = ml._build_raw_sigmas(20, "beta57")
raw_beta13 = ml._build_raw_sigmas(20, "beta13")
raw_beta31 = ml._build_raw_sigmas(20, "beta31")

check("beta57_differs_from_linear",
      not np.allclose(raw_linear, raw_beta57, atol=0.01))
check("beta13_concentrates_low",
      # beta13 (α=1, β=3) should have MORE values in the low sigma range
      # than linear
      sum(1 for v in raw_beta13 if v < 0.5) > sum(1 for v in raw_linear if v < 0.5),
      "beta13 should concentrate more at low sigma than linear")
check("beta31_concentrates_high",
      sum(1 for v in raw_beta31 if v > 0.5) > sum(1 for v in raw_linear if v > 0.5),
      "beta31 should concentrate more at high sigma than linear")

# Invalid beta schedule rejected cleanly
try:
    ml._build_raw_sigmas(20, "betaXZ")
    check("invalid_beta_rejected", False, "should have raised")
except ValueError:
    check("invalid_beta_rejected", True)
except Exception as e:
    check("invalid_beta_rejected", False, f"raised wrong type: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  Refinement helpers — sigma truncation + noise injection
# ═══════════════════════════════════════════════════════════════════════

print("\n=== truncate_sigmas_for_denoise ===")

# Build a simple full schedule with terminal zero
full = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

# denoise=1.0 → unchanged
out = ml.truncate_sigmas_for_denoise(full, 1.0)
check("trunc_full_unchanged", torch.equal(out, full))

# denoise=0.5 → keep last 50% of denoising steps + terminal zero
# n_steps = 5 (excluding terminal), keep_steps = round(5*0.5) = 2 or 3
out = ml.truncate_sigmas_for_denoise(full, 0.5)
# 5 sigmas (excluding terminal) * 0.5 = 2.5, round = 2 or 3 (Python rounds to even, so 2)
check("trunc_half_preserves_zero", abs(float(out[-1])) < 1e-9)
check("trunc_half_starts_lower", float(out[0]) < float(full[0]))
check("trunc_half_shorter", len(out) < len(full))

# denoise=0.0 → smallest reasonable result (at least 1 step)
out = ml.truncate_sigmas_for_denoise(full, 0.0)
check("trunc_zero_at_least_one_step", len(out) >= 2,  # 1 step + terminal
      f"got len={len(out)}")

# Schedule WITHOUT terminal zero
no_term = torch.tensor([1.0, 0.7, 0.4, 0.1])
out = ml.truncate_sigmas_for_denoise(no_term, 0.5)
check("trunc_no_terminal_handled", abs(float(out[-1]) - 0.1) < 1e-6,
      f"last={float(out[-1])}")


print("\n=== inject_flow_noise ===")

x_clean = torch.ones(1, 16, 16) * 5.0  # constant tensor for predictability

# sigma=0 → unchanged
out = ml.inject_flow_noise(x_clean, 0.0)
check("inject_zero_unchanged", torch.allclose(out, x_clean))

# sigma=1 → pure noise (no input contribution); same SHAPE as input
torch.manual_seed(42)
out = ml.inject_flow_noise(x_clean, 1.0)
check("inject_one_shape", out.shape == x_clean.shape)
check("inject_one_not_constant", out.std().item() > 0.5,
      f"std={out.std().item()}")

# sigma=0.5 → blend
torch.manual_seed(42)
out = ml.inject_flow_noise(x_clean, 0.5)
# Expected: (1-0.5)*5 + 0.5*noise = 2.5 + 0.5*noise
# Mean should be ~2.5
check("inject_half_mean_close_to_2.5",
      abs(out.mean().item() - 2.5) < 0.5,
      f"mean={out.mean().item()}")


print("\n=== upscale_flux_latents shape correctness ===")

# Create a fake packed latent tensor for 1024x1024 (latent 128x128, packed seq 64*64=4096)
# Channels = 64 (16 base × 4 from packing)
src_h, src_w = 1024, 1024
src_h_lat = 2 * (src_h // 16)  # = 128
src_w_lat = 2 * (src_w // 16)  # = 128
src_seq = (src_h_lat // 2) * (src_w_lat // 2)  # = 4096
C = 16  # base channels
packed = torch.randn(1, src_seq, C * 4)

# Upscale to 1536x1536 (latent 192x192, seq 96*96=9216)
dst_h, dst_w = 1536, 1536
dst_h_lat = 2 * (dst_h // 16)  # = 192
dst_w_lat = 2 * (dst_w // 16)  # = 192
expected_dst_seq = (dst_h_lat // 2) * (dst_w_lat // 2)

# Try the upscale; it requires comfy.utils which we mocked.  Provide a
# minimal bislerp implementation for the test.
import types
comfy_utils_mock = types.ModuleType("comfy.utils")
def _fake_bislerp(x, w, h):
    return torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
comfy_utils_mock.bislerp = _fake_bislerp
sys.modules["comfy.utils"] = comfy_utils_mock

upscaled = ml.upscale_flux_latents(packed, src_h, src_w, dst_h, dst_w)
check("upscale_correct_seq_length",
      upscaled.shape == (1, expected_dst_seq, C * 4),
      f"expected {(1, expected_dst_seq, C * 4)}, got {tuple(upscaled.shape)}")
check("upscale_preserves_batch_and_channels",
      upscaled.shape[0] == packed.shape[0] and upscaled.shape[2] == packed.shape[2])

# Same-resolution upscale should be near-identity (within bislerp interpolation noise)
same = ml.upscale_flux_latents(packed, src_h, src_w, src_h, src_w)
check("upscale_same_resolution_shape", same.shape == packed.shape)


# ═══════════════════════════════════════════════════════════════════════
#  Flux.2 support
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Flux.2 empirical mu formula ===")

# Compare two call sites: above and below seq_len=4300 threshold
mu_small = ml.compute_flux2_shift_mu(2048, num_steps=20)
mu_medium = ml.compute_flux2_shift_mu(4096, num_steps=20)
mu_large = ml.compute_flux2_shift_mu(8192, num_steps=20)

check("mu2_is_float", isinstance(mu_small, float))
check("mu2_small_reasonable", 1.0 < mu_small < 3.0,
      f"mu_small={mu_small}")

# Above 4300, num_steps is ignored
mu_large_10 = ml.compute_flux2_shift_mu(8192, num_steps=10)
mu_large_100 = ml.compute_flux2_shift_mu(8192, num_steps=100)
check("mu2_large_independent_of_num_steps",
      abs(mu_large_10 - mu_large_100) < 1e-6,
      f"10→{mu_large_10}, 100→{mu_large_100}")

# Below 4300, mu varies with num_steps
mu_small_10 = ml.compute_flux2_shift_mu(2048, num_steps=10)
mu_small_200 = ml.compute_flux2_shift_mu(2048, num_steps=200)
check("mu2_small_varies_with_num_steps",
      abs(mu_small_10 - mu_small_200) > 0.1,
      f"10→{mu_small_10}, 200→{mu_small_200}")

# At num_steps=10, formula uses a1, b1
# m_10 at seq_len=2048: 8.73809524e-05 * 2048 + 1.89833333 = 2.0773
# Via linear interp at num_steps=10: should be exactly m_10
import math as _math
m_10_expected = 8.73809524e-05 * 2048 + 1.89833333
# The linear interp at num_steps=10 gives: a*10 + b where (a, b) fit m_10 and m_200
# At num_steps=10: should return m_10 exactly (by construction)
check("mu2_num_steps_10_exact",
      abs(mu_small_10 - m_10_expected) < 1e-4,
      f"expected {m_10_expected}, got {mu_small_10}")


print("\n=== Flux.2 sigma schedule construction ===")

# Same interface as Flux.1 but different mu
sigmas2 = ml.build_flux2_sigmas(20, 2048, "linear")
check("flux2_sigmas_length", len(sigmas2) == 21,
      f"got {len(sigmas2)}")
check("flux2_sigmas_starts_at_one", abs(float(sigmas2[0]) - 1.0) < 1e-4)
check("flux2_sigmas_ends_at_zero", abs(float(sigmas2[-1])) < 1e-6)
check("flux2_sigmas_descending",
      all(float(sigmas2[i]) >= float(sigmas2[i+1]) for i in range(len(sigmas2) - 1)))

# Flux.2 at different seq_len — just verify schedule is well-formed at both.
# Note: unlike Flux.1's linear mu, Flux.2's empirical formula is NOT
# monotonic in seq_len (regime transition at 4300 means larger resolutions
# can actually get SMALLER mu in the high-seq regime).  Don't assume a
# direction, just verify the schedule remains valid.
sigmas2_large = ml.build_flux2_sigmas(20, 8192, "linear")
check("flux2_large_seq_length_valid", len(sigmas2_large) == 21)
check("flux2_large_seq_starts_at_one",
      abs(float(sigmas2_large[0]) - 1.0) < 1e-4)
check("flux2_large_seq_ends_at_zero",
      abs(float(sigmas2_large[-1])) < 1e-6)
check("flux2_large_seq_descending",
      all(float(sigmas2_large[i]) >= float(sigmas2_large[i+1])
          for i in range(len(sigmas2_large) - 1)))
# The two schedules should differ SOMEWHERE (not literally identical)
check("flux2_seq_len_affects_schedule",
      not np.allclose(np.asarray(sigmas2), np.asarray(sigmas2_large), atol=1e-4))

# Karras vs linear should differ
sigmas2_karras = ml.build_flux2_sigmas(20, 2048, "karras")
check("flux2_karras_different_from_linear",
      not np.allclose(np.asarray(sigmas2), np.asarray(sigmas2_karras), atol=1e-4))


print("\n=== Flux.2 latent upscale shape correctness ===")

# Flux.2 packing: (B, H*W, C) where H and W are half the latent dims
# For 1024x1024: latent = 128x128, H_patched = W_patched = 64, seq = 4096
src_h, src_w = 1024, 1024
vae_scale = 8
src_h_patched = 2 * (src_h // (vae_scale * 2)) // 2  # = 64
src_w_patched = 2 * (src_w // (vae_scale * 2)) // 2  # = 64
src_seq = src_h_patched * src_w_patched  # = 4096
C = 64  # (16 base channels × 4 from patchification)

packed = torch.randn(1, src_seq, C)

# Build latent_ids: [0, h_idx, w_idx] row-major
latent_ids = torch.zeros(src_seq, 3, dtype=torch.int64)
h_idx = torch.arange(src_h_patched).repeat_interleave(src_w_patched)
w_idx = torch.arange(src_w_patched).repeat(src_h_patched)
latent_ids[:, 1] = h_idx
latent_ids[:, 2] = w_idx

# Upscale to 1536x1536: 96x96 patched, seq = 9216
dst_h, dst_w = 1536, 1536
dst_h_patched = 2 * (dst_h // (vae_scale * 2)) // 2  # = 96
dst_w_patched = 2 * (dst_w // (vae_scale * 2)) // 2  # = 96
expected_dst_seq = dst_h_patched * dst_w_patched

# upscale_flux2_latents now returns ONLY the upscaled packed latents.
# The caller is responsible for getting fresh latent_ids from
# pipe.prepare_latents() at the target resolution — Flux.2's id layout
# has >3 columns and manual construction produces invalid ids.
new_packed = ml.upscale_flux2_latents(
    packed, latent_ids, src_h, src_w, dst_h, dst_w, vae_scale_factor=vae_scale,
)
check("flux2_upscale_seq_length",
      new_packed.shape == (1, expected_dst_seq, C),
      f"expected {(1, expected_dst_seq, C)}, got {tuple(new_packed.shape)}")

# Same-res upscale should be near-identity (only bilinear noise)
same_packed = ml.upscale_flux2_latents(
    packed, latent_ids, src_h, src_w, src_h, src_w, vae_scale_factor=vae_scale,
)
check("flux2_upscale_same_res_shape", same_packed.shape == packed.shape)


# ═══════════════════════════════════════════════════════════════════════
#  Stochastic sampling (eta parameter) — regression tests
# ═══════════════════════════════════════════════════════════════════════

print("\n=== Stochastic sampling (eta) ===")

# Build a trivial ODE scenario
torch.manual_seed(42)
x_init = torch.randn(1, 64, 32, dtype=torch.float64)
sigmas_stoch = torch.linspace(1.0, 0.0, 11, dtype=torch.float64)

# Use a more interesting synthetic denoiser — not just zero velocity, so
# we can tell apart deterministic and stochastic outputs
def synth_denoiser(x, sigma):
    return -x * 0.5  # mild inward velocity

# eta=0 should match the previous deterministic behavior exactly.
# Run each sampler with eta=0 and compare to the same call without eta arg
# (which defaults to 0).
for sampler_name in ["flow_euler", "flow_heun", "flow_rk3",
                     "flow_multistep2", "flow_multistep3"]:
    fn = ml.get_sampler(sampler_name)
    result_default = fn(synth_denoiser, x_init.clone(), sigmas_stoch)
    result_eta0 = fn(synth_denoiser, x_init.clone(), sigmas_stoch, eta=0.0)
    check_close(f"{sampler_name}_eta_zero_matches_default",
                result_default, result_eta0, atol=1e-10)

# eta>0 should produce DIFFERENT output from eta=0
# (we pass a fresh torch.Generator so the fresh noise is reproducible)
for sampler_name in ["flow_euler", "flow_heun", "flow_multistep2"]:
    fn = ml.get_sampler(sampler_name)
    result_det = fn(synth_denoiser, x_init.clone(), sigmas_stoch, eta=0.0)
    g = torch.Generator().manual_seed(1234)
    result_stoch = fn(
        synth_denoiser, x_init.clone(), sigmas_stoch, eta=0.5, generator=g,
    )
    diff = (result_det - result_stoch).abs().max().item()
    check(f"{sampler_name}_eta_half_differs_from_zero",
          diff > 1e-3,
          f"deterministic and stochastic outputs match (diff={diff})")

# Same generator seed produces REPRODUCIBLE stochastic output
fn = ml.get_sampler("flow_euler")
g1 = torch.Generator().manual_seed(1234)
result_a = fn(synth_denoiser, x_init.clone(), sigmas_stoch, eta=0.5, generator=g1)
g2 = torch.Generator().manual_seed(1234)
result_b = fn(synth_denoiser, x_init.clone(), sigmas_stoch, eta=0.5, generator=g2)
check_close("stochastic_reproducible_same_seed", result_a, result_b, atol=1e-10)

# Different seeds produce DIFFERENT output
g3 = torch.Generator().manual_seed(5678)
result_c = fn(synth_denoiser, x_init.clone(), sigmas_stoch, eta=0.5, generator=g3)
diff = (result_a - result_c).abs().max().item()
check("stochastic_different_seeds_differ", diff > 1e-3,
      f"different seeds produced same output (diff={diff})")


# ═══════════════════════════════════════════════════════════════════════
#  Qwen-Image manual loop — pure-math checks
# ═══════════════════════════════════════════════════════════════════════
#
# These tests exercise the Qwen-specific pieces WITHOUT loading a Qwen
# model.  The goal is to verify:
#
#   1. Qwen's latent packing math is actually byte-identical to Flux's —
#      if this ever changes upstream, upscale_flux_latents would silently
#      corrupt Qwen multistage refinement and we'd see it in output, not
#      in an error.  The round-trip test catches it.
#
#   2. The norm-preserving CFG rescale produces outputs whose per-token
#      feature-vector norm matches the conditional pred's norm — that's
#      the whole point of the rescale, and getting it wrong means Qwen
#      looks like ordinary CFG (drifts at high scales).
#
#   3. The img_shapes structure matches what QwenImagePipeline builds so
#      the transformer positional encoding lines up.
#
# We can't test generate_qwen end-to-end without a real Qwen model, but
# these three pieces are where the Qwen-specific bugs would live.

print("\n── Qwen-Image pure-math checks ──")


# ── Test 1: Qwen packing = Flux packing ────────────────────────────────
# Simulate what Qwen's prepare_latents would produce (5D with trivial
# temporal dim) and confirm upscale_flux_latents handles it correctly.
#
# Qwen's _pack_latents and _unpack_latents (from diffusers source):
#   pack:   view(B, C, H//2, 2, W//2, 2) → permute(0,2,4,1,3,5) →
#           reshape(B, (H//2)*(W//2), C*4)
#   unpack: view(B, H//2, W//2, C//4, 2, 2) → permute(0,3,1,4,2,5) →
#           reshape(B, C//4, 1, H, W)     ← note the trivial temporal dim
#
# Flux's versions do exactly the same math except unpack returns 4D
# (no temporal dim).  So packed representations should be bitwise equal.

B, C_unpacked, H_lat, W_lat = 1, 16, 8, 8
# Fresh random unpacked latents (4D for comparison)
rng = torch.Generator().manual_seed(1337)
spatial_4d = torch.randn(
    B, C_unpacked, H_lat, W_lat, generator=rng,
)

# Flux pack (this is what upscale_flux_latents / generate_flux use):
flux_packed = spatial_4d.view(
    B, C_unpacked, H_lat // 2, 2, W_lat // 2, 2,
).permute(0, 2, 4, 1, 3, 5).reshape(
    B, (H_lat // 2) * (W_lat // 2), C_unpacked * 4,
)

# Qwen pack on the SAME data (just with a trivial temporal dim inserted).
# The static method definition is the exact same 6-step view → permute →
# reshape.  Qwen's prepare_latents wraps a 5D (B, 1, C, H, W) tensor
# because its VAE is 5D; but the view() call treats the tensor as flat
# memory so the extra 1-dim is inert.
qwen_5d = spatial_4d.unsqueeze(1)  # (B, 1, C, H, W)
qwen_packed = qwen_5d.view(
    B, C_unpacked, H_lat // 2, 2, W_lat // 2, 2,
).permute(0, 2, 4, 1, 3, 5).reshape(
    B, (H_lat // 2) * (W_lat // 2), C_unpacked * 4,
)

check_close("qwen_packing_matches_flux", flux_packed, qwen_packed, atol=0.0)

# Round-trip through upscale_flux_latents with src_h == dst_h (no scale):
# output should be bitwise equal to input because bislerp on an identity
# resize is a no-op.
#
# We use pixel dimensions (H_lat * vae_scale_factor) because that's what
# the helper expects.  Qwen's vae_scale_factor is 8.
vae_scale = 8
px_h, px_w = H_lat * vae_scale, W_lat * vae_scale

# Stub comfy.utils with a minimal bislerp that matches the ComfyUI
# semantics well enough for the identity case: (B,C,H,W), bislerp to
# (W_new, H_new) returns (B, C, H_new, W_new).  For identity sizes it's
# a no-op passthrough.  We don't need real bislerp here — we're testing
# the pack/unpack math, not the interpolation.
comfy_utils_mock = types.ModuleType("comfy.utils")

def _mock_bislerp(x, w_new, h_new):
    if x.shape[-1] == w_new and x.shape[-2] == h_new:
        return x
    # Simple nearest for non-identity; we don't test non-identity here.
    return torch.nn.functional.interpolate(x, size=(h_new, w_new), mode="nearest")

comfy_utils_mock.bislerp = _mock_bislerp
sys.modules["comfy.utils"] = comfy_utils_mock

upscaled_packed = ml.upscale_flux_latents(
    qwen_packed, src_h=px_h, src_w=px_w,
    dst_h=px_h, dst_w=px_w, vae_scale_factor=vae_scale,
)
check_close(
    "qwen_packed_identity_roundtrip",
    upscaled_packed, qwen_packed, atol=1e-6,
)


# ── Test 2: Norm-preserving CFG rescale ────────────────────────────────
# Replicate the formula from generate_qwen's denoiser in isolation and
# verify the per-token norm invariant: for every feature vector row,
# ||noise_pred[b, seq, :]|| should equal ||v_cond[b, seq, :]|| after
# the rescale, for any scale value > 1.

def qwen_cfg_rescale(v_cond, v_neg, scale):
    """Standalone copy of the rescale block inside generate_qwen.denoiser."""
    v_cond_f = v_cond.float()
    v_neg_f = v_neg.float()
    comb = v_neg_f + scale * (v_cond_f - v_neg_f)
    cond_norm = torch.norm(v_cond_f, dim=-1, keepdim=True)
    comb_norm = torch.clamp(torch.norm(comb, dim=-1, keepdim=True), min=1e-8)
    return (comb * (cond_norm / comb_norm)).to(dtype=v_cond.dtype)


rng = torch.Generator().manual_seed(42)
v_cond = torch.randn(1, 64, 128, generator=rng)  # (B, seq, feat)
v_neg = torch.randn(1, 64, 128, generator=rng)

for scale in (2.0, 4.0, 7.5):
    result = qwen_cfg_rescale(v_cond, v_neg, scale)
    result_norms = torch.norm(result, dim=-1)
    cond_norms = torch.norm(v_cond, dim=-1)
    check_close(
        f"qwen_cfg_rescale_preserves_norm_scale{scale}",
        result_norms, cond_norms, atol=1e-5,
    )

# Direction sanity: at scale=2 the rescaled vector's direction should
# match `comb`'s direction (just with a different magnitude).  Compare
# unit vectors.
scale = 2.0
comb = v_neg.float() + scale * (v_cond.float() - v_neg.float())
result = qwen_cfg_rescale(v_cond, v_neg, scale)
comb_unit = comb / torch.clamp(torch.norm(comb, dim=-1, keepdim=True), min=1e-8)
result_unit = result.float() / torch.clamp(
    torch.norm(result.float(), dim=-1, keepdim=True), min=1e-8,
)
check_close(
    "qwen_cfg_rescale_preserves_direction",
    comb_unit, result_unit, atol=1e-5,
)

# When cond and neg are identical, rescale should return cond unchanged
# (comb == cond, norm ratio == 1).  Any non-trivial scale.
v_same = torch.randn(1, 32, 64, generator=rng)
result_same = qwen_cfg_rescale(v_same, v_same, scale=4.0)
check_close(
    "qwen_cfg_rescale_identity_when_cond_eq_neg",
    result_same, v_same, atol=1e-5,
)


# ── Test 3: img_shapes structure matches QwenImagePipeline ─────────────
# The pipeline builds:
#   img_shapes = [[(1, h // vae_scale_factor // 2, w // vae_scale_factor // 2)]] * batch
# Outer list = batch (length B), inner list = one per image (length 1
# for t2i), tuple = (frame_count, H/16, W/16) where the 16 comes from
# vae_scale_factor (8) * packing factor (2).

for height, width in [(1024, 1024), (1536, 1024), (832, 1216)]:
    vae_sf = 8
    expected_tuple = (1, height // vae_sf // 2, width // vae_sf // 2)
    B = 1
    img_shapes = [[expected_tuple]] * B
    check(
        f"qwen_img_shapes_structure_{height}x{width}",
        (
            len(img_shapes) == B
            and len(img_shapes[0]) == 1
            and img_shapes[0][0] == expected_tuple
        ),
        f"got {img_shapes}",
    )


# ── Test 4: generate_qwen is importable and has expected signature ─────
check(
    "generate_qwen_importable",
    hasattr(ml, "generate_qwen") and callable(ml.generate_qwen),
)
check(
    "decode_qwen_latents_importable",
    hasattr(ml, "decode_qwen_latents") and callable(ml.decode_qwen_latents),
)

import inspect as _inspect
qwen_sig = _inspect.signature(ml.generate_qwen)
required_params = {
    "pipe", "prompt", "negative_prompt", "width", "height",
    "num_inference_steps", "guidance_scale", "sampler_name",
    "sigma_schedule", "generator", "initial_latents", "denoise", "eta",
}
actual_params = set(qwen_sig.parameters.keys())
missing = required_params - actual_params
check(
    "generate_qwen_has_required_params",
    not missing,
    f"missing: {missing}",
)


# ═══════════════════════════════════════════════════════════════════════
#  encode_image_to_packed_latents — i2i encode helper
# ═══════════════════════════════════════════════════════════════════════
#
# These tests exercise the helper without loading real Flux/Chroma/Qwen
# models.  We use:
#   - Direct math checks on the two tiny utility helpers
#     (_comfy_image_to_vae_input, _resize_image_if_needed).
#   - A MockPipe / MockVAE stub for the encode dispatch so we can verify
#     shape plumbing and normalization math without a model.

print("\n── encode_image_to_packed_latents (i2i encode helper) ──")


# ── Test: helper + utils importable ───────────────────────────────────
check(
    "encode_image_to_packed_latents_importable",
    hasattr(ml, "encode_image_to_packed_latents")
    and callable(ml.encode_image_to_packed_latents),
)
check(
    "_comfy_image_to_vae_input_importable",
    hasattr(ml, "_comfy_image_to_vae_input"),
)
check(
    "_resize_image_if_needed_importable",
    hasattr(ml, "_resize_image_if_needed"),
)


# ── Test: _comfy_image_to_vae_input math ──────────────────────────────
# ComfyUI: (B, H, W, 3) in [0, 1]   →   VAE: (B, 3, H, W) in [-1, 1]
img_01 = torch.tensor([
    [[[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]]]  # (1, 1, 2, 3)
], dtype=torch.float32)
assert img_01.shape == (1, 1, 2, 3), f"setup wrong: {img_01.shape}"

vae_input = ml._comfy_image_to_vae_input(img_01)
check(
    "comfy_to_vae_input_shape",
    tuple(vae_input.shape) == (1, 3, 1, 2),
    f"got {tuple(vae_input.shape)}",
)
# Channel 0 should be [0.0, 1.0] scaled to [-1, 1] → [-1, 1]
# Channel 1 should be [0.5, 0.5] scaled to [-1, 1] → [0, 0]
# Channel 2 should be [1.0, 0.0] scaled to [-1, 1] → [1, -1]
expected_ch0 = torch.tensor([[[-1.0, 1.0]]])
expected_ch1 = torch.tensor([[[0.0, 0.0]]])
expected_ch2 = torch.tensor([[[1.0, -1.0]]])
check_close("comfy_to_vae_ch0", vae_input[0, 0], expected_ch0[0], atol=1e-6)
check_close("comfy_to_vae_ch1", vae_input[0, 1], expected_ch1[0], atol=1e-6)
check_close("comfy_to_vae_ch2", vae_input[0, 2], expected_ch2[0], atol=1e-6)

# Bad shape rejected
try:
    ml._comfy_image_to_vae_input(torch.zeros(1, 3, 16, 16))  # wrong layout
    check("comfy_to_vae_rejects_channel_first", False, "should have raised")
except ValueError:
    check("comfy_to_vae_rejects_channel_first", True)


# ── Test: _resize_image_if_needed — no-op + resize paths ──────────────
img = torch.rand(1, 32, 48, 3)

# No-op: same dims returns same tensor
out = ml._resize_image_if_needed(img, 32, 48)
check(
    "resize_noop_returns_same_tensor",
    out is img,
    "no-op should return identity, not a copy",
)

# Actual resize: different dims changes shape to target
out = ml._resize_image_if_needed(img, 64, 96)
check(
    "resize_produces_target_shape",
    tuple(out.shape) == (1, 64, 96, 3),
    f"got {tuple(out.shape)}",
)
check(
    "resize_output_in_0_1",
    out.min().item() >= 0.0 and out.max().item() <= 1.0,
    f"range [{out.min():.4f}, {out.max():.4f}]",
)


# ── Test: encode helper dispatch for unsupported families ─────────────
class _MockParams:
    def __init__(self, dtype=torch.float32, device="cpu"):
        self._dtype = dtype
        self._device = torch.device(device)
    def __iter__(self):
        return iter([self])
    def __next__(self):
        return self
    @property
    def dtype(self):
        return self._dtype
    @property
    def device(self):
        return self._device


class _MockVAE:
    def __init__(self, kind="flux"):
        self.kind = kind
        self._params = _MockParams(dtype=torch.float32)
        class _C:
            pass
        self.config = _C()
        if kind == "qwen":
            self.config.z_dim = 16
            self.config.latents_mean = [0.0] * 16
            self.config.latents_std = [1.0] * 16
        else:
            self.config.scaling_factor = 0.3611
            self.config.shift_factor = 0.1159

    def parameters(self):
        return iter([self._params])

    def encode(self, x):
        # Return a stub with a .latent_dist that has .mode()
        # Flux/Chroma: 4D (B, 16, H/8, W/8)
        # Qwen: 5D (B, 16, 1, H/8, W/8)
        if self.kind == "qwen":
            # x is 5D (B, 3, 1, H, W)
            B, _, _, H, W = x.shape
            raw = torch.zeros(B, 16, 1, H // 8, W // 8, dtype=x.dtype, device=x.device)
        else:
            # x is 4D (B, 3, H, W)
            B, _, H, W = x.shape
            raw = torch.zeros(B, 16, H // 8, W // 8, dtype=x.dtype, device=x.device)
        class _Dist:
            def __init__(self, r):
                self._r = r
            def mode(self):
                return self._r
        class _Out:
            def __init__(self, r):
                self.latent_dist = _Dist(r)
        return _Out(raw)


class _MockTransformer:
    def __init__(self, dtype=torch.float32):
        self._params = _MockParams(dtype=dtype)
    def parameters(self):
        return iter([self._params])


class _MockPipe:
    def __init__(self, kind="flux"):
        self.vae = _MockVAE(kind)
        self.transformer = _MockTransformer()
        self.vae_scale_factor = 8
        self._execution_device = torch.device("cpu")


# Flux.2 deferred — should raise NotImplementedError
try:
    ml.encode_image_to_packed_latents(
        _MockPipe(), torch.rand(1, 256, 256, 3), 256, 256, "flux2",
    )
    check("encode_flux2_raises_not_implemented", False, "should have raised")
except NotImplementedError:
    check("encode_flux2_raises_not_implemented", True)

# Unknown family
try:
    ml.encode_image_to_packed_latents(
        _MockPipe(), torch.rand(1, 256, 256, 3), 256, 256, "made_up_family",
    )
    check("encode_unknown_family_raises", False, "should have raised")
except ValueError:
    check("encode_unknown_family_raises", True)


# ── Test: Flux/Chroma encode dispatch produces expected packed shape ───
# For a (256, 256) target and z_dim=16 mock:
#   H_lat = W_lat = 32 (256/8)
#   Packed shape: (B, (32/2)*(32/2), 16*4) = (1, 256, 64)
img_256 = torch.rand(1, 256, 256, 3)
packed_flux = ml.encode_image_to_packed_latents(
    _MockPipe("flux"), img_256, 256, 256, "flux",
)
check(
    "encode_flux_packed_shape",
    tuple(packed_flux.shape) == (1, 256, 64),
    f"got {tuple(packed_flux.shape)}",
)

# Same target via chroma family alias
packed_chroma = ml.encode_image_to_packed_latents(
    _MockPipe("flux"), img_256, 256, 256, "chroma",
)
check(
    "encode_chroma_packed_shape",
    tuple(packed_chroma.shape) == (1, 256, 64),
    f"got {tuple(packed_chroma.shape)}",
)


# ── Test: Qwen encode dispatch produces expected packed shape ───────────
# Qwen transposes (1,2) before pack so the pack still lands at (1, 256, 64)
# for a (256, 256) target with z_dim=16.
packed_qwen = ml.encode_image_to_packed_latents(
    _MockPipe("qwen"), img_256, 256, 256, "qwen-image",
)
check(
    "encode_qwen_packed_shape",
    tuple(packed_qwen.shape) == (1, 256, 64),
    f"got {tuple(packed_qwen.shape)}",
)


# ── Test: encode preserves aspect ratio via resize ─────────────────────
# Non-square target should produce non-square packed shape
# (384, 256) → H_lat=48, W_lat=32 → (1, 24*16, 64) = (1, 384, 64)
img_nonsq = torch.rand(1, 128, 128, 3)
packed_nonsq = ml.encode_image_to_packed_latents(
    _MockPipe("flux"), img_nonsq, 256, 384, "flux",  # width=256, height=384
)
# height=384 → H_lat=48, width=256 → W_lat=32
# seq = (48/2) * (32/2) = 24 * 16 = 384
check(
    "encode_nonsquare_packed_shape",
    tuple(packed_nonsq.shape) == (1, 384, 64),
    f"got {tuple(packed_nonsq.shape)}",
)


# ═══════════════════════════════════════════════════════════════════════
#  generate_qwen_edit — Qwen Edit manual loop (Phase 1)
# ═══════════════════════════════════════════════════════════════════════
#
# These tests exercise the Qwen Edit helpers without loading a real
# Qwen-Image-Edit model.  Coverage:
#
#   1. _calculate_qwen_edit_dimensions math — 32-aligned aspect-preserving
#   2. _resize_ref_for_qwen_edit shape correctness
#   3. _qwen_edit_comfy_to_pil_list conversion
#   4. generate_qwen_edit error paths (empty/None refs, bad flags,
#      missing pipe attributes) via a minimal mock pipe
#   5. Signature + import smoke
#
# A full end-to-end dispatch test would require mocking the entire
# Qwen Edit pipeline (transformer forward, VAE encode/decode, text
# encoder, processor, scheduler) which is brittle and gives less
# signal than a real ComfyUI smoke test.  We stop at validation
# boundaries and let the user smoke the real dispatch.

print("\n── Qwen Edit manual loop (Phase 1) ──")


# ── Test: helpers importable ──────────────────────────────────────────
check(
    "generate_qwen_edit_importable",
    hasattr(ml, "generate_qwen_edit") and callable(ml.generate_qwen_edit),
)
check(
    "_calculate_qwen_edit_dimensions_importable",
    hasattr(ml, "_calculate_qwen_edit_dimensions"),
)
check(
    "_resize_ref_for_qwen_edit_importable",
    hasattr(ml, "_resize_ref_for_qwen_edit"),
)
check(
    "_qwen_edit_comfy_to_pil_list_importable",
    hasattr(ml, "_qwen_edit_comfy_to_pil_list"),
)


# ── Test: generate_qwen_edit signature ────────────────────────────────
import inspect as _inspect_edit
qwen_edit_sig = _inspect_edit.signature(ml.generate_qwen_edit)
expected_params = {
    "pipe", "prompt", "negative_prompt", "reference_images",
    "vl_flags", "ref_flags", "output_width", "output_height",
    "num_inference_steps", "guidance_scale", "sampler_name",
    "sigma_schedule", "generator", "max_sequence_length",
    "progress_cb", "eta",
}
actual_params = set(qwen_edit_sig.parameters.keys())
missing = expected_params - actual_params
check(
    "generate_qwen_edit_has_all_params",
    not missing,
    f"missing: {missing}",
)


# ── Test: _calculate_qwen_edit_dimensions math ─────────────────────────
# Ported from QwenImageEditPlusPipeline.calculate_dimensions:
#   width = sqrt(target_area * ratio)
#   height = width / ratio
#   both rounded to nearest multiple of 32

# 1:1 square, 1024² area → 1024×1024
w, h = ml._calculate_qwen_edit_dimensions(1024 * 1024, 1.0)
check("qwen_edit_dims_1x1_1024", (w, h) == (1024, 1024), f"got {w}×{h}")

# 3:2 landscape, 1024² area → ~1248×832 (matches stock pipeline math)
w, h = ml._calculate_qwen_edit_dimensions(1024 * 1024, 1.5)
check("qwen_edit_dims_3x2_1024", (w, h) == (1248, 832), f"got {w}×{h}")

# 2:3 portrait, 1024² area → ~832×1248 (inverse of above)
w, h = ml._calculate_qwen_edit_dimensions(1024 * 1024, 2/3)
# sqrt(1024*1024 * 2/3) = 836, /32 round = 832; 832/(2/3) = 1248
check("qwen_edit_dims_2x3_1024", (w, h) == (832, 1248), f"got {w}×{h}")

# 16:9, 1024² area
w, h = ml._calculate_qwen_edit_dimensions(1024 * 1024, 16/9)
# sqrt(1024*1024 * 16/9) ≈ 1365 → 1376; 1376 / (16/9) ≈ 774 → 768
# Note: round() is banker's rounding, small off-by-32 possible
check(
    "qwen_edit_dims_16x9_32_aligned",
    w % 32 == 0 and h % 32 == 0 and abs(w/h - 16/9) < 0.08,
    f"got {w}×{h}, ratio {w/h:.3f}",
)

# Condition size (384²) at 1:1
w, h = ml._calculate_qwen_edit_dimensions(384 * 384, 1.0)
check("qwen_edit_dims_condition_1x1", (w, h) == (384, 384), f"got {w}×{h}")

# Output is ALWAYS 32-aligned regardless of input
for ratio in (0.5, 0.75, 1.0, 1.33, 1.5, 1.77, 2.0, 3.0):
    w, h = ml._calculate_qwen_edit_dimensions(1024 * 1024, ratio)
    check(
        f"qwen_edit_dims_ratio_{ratio}_32_aligned",
        w % 32 == 0 and h % 32 == 0,
        f"got {w}×{h}",
    )


# ── Test: _resize_ref_for_qwen_edit shape output ───────────────────────
# Input 512×512, target VAE area (1024²) at 1:1 → 1024×1024 output
img_sq = torch.rand(1, 512, 512, 3)
out = ml._resize_ref_for_qwen_edit(img_sq, 1024 * 1024)
check(
    "resize_qwen_edit_sq_to_vae",
    tuple(out.shape) == (1, 1024, 1024, 3),
    f"got {tuple(out.shape)}",
)

# No-op: already at target size
out = ml._resize_ref_for_qwen_edit(img_sq, 512 * 512)
check(
    "resize_qwen_edit_noop_same_size",
    out is img_sq,
    "should return identity when size matches",
)

# 16:9 input, condition area (384²) → aspect-preserving output
img_wide = torch.rand(1, 576, 1024, 3)  # 16:9
out = ml._resize_ref_for_qwen_edit(img_wide, 384 * 384)
_, oh, ow, _ = out.shape
check(
    "resize_qwen_edit_condition_16x9_aspect",
    oh % 32 == 0 and ow % 32 == 0 and abs((ow/oh) - (1024/576)) < 0.1,
    f"got {ow}×{oh}, aspect {ow/oh:.3f}",
)


# ── Test: _qwen_edit_comfy_to_pil_list conversion ──────────────────────
# Single image input produces single-item PIL list
img_one = torch.rand(1, 64, 64, 3)
pil_list = ml._qwen_edit_comfy_to_pil_list(img_one)
check("pil_list_single_len", len(pil_list) == 1)
check("pil_list_single_size", pil_list[0].size == (64, 64))
check("pil_list_single_mode", pil_list[0].mode == "RGB")

# Batched image input produces multi-item PIL list
img_batch = torch.rand(3, 128, 96, 3)
pil_list = ml._qwen_edit_comfy_to_pil_list(img_batch)
check("pil_list_batch_len", len(pil_list) == 3)
check(
    "pil_list_batch_size",
    all(p.size == (96, 128) for p in pil_list),
    f"sizes: {[p.size for p in pil_list]}",
)

# Bad shape rejected
try:
    ml._qwen_edit_comfy_to_pil_list(torch.zeros(1, 3, 64, 64))  # channel-first
    check("pil_list_rejects_channel_first", False, "should have raised")
except ValueError:
    check("pil_list_rejects_channel_first", True)


# ── Test: generate_qwen_edit error paths with minimal mock pipe ────────
class _QwenEditMockVAE:
    def __init__(self):
        self._params = _MockParams(dtype=torch.float32)
        class _C:
            z_dim = 16
            latents_mean = [0.0] * 16
            latents_std = [1.0] * 16
        self.config = _C()
    def parameters(self):
        return iter([self._params])


class _QwenEditMockTransformer:
    def __init__(self):
        self._params = _MockParams(dtype=torch.float32)
        class _C:
            guidance_embeds = False
            in_channels = 64
        self.config = _C()
    def parameters(self):
        return iter([self._params])


class _QwenEditMockPipe:
    """Minimal mock that satisfies hasattr checks but isn't runnable end-
    to-end.  Used only to exercise input validation."""
    def __init__(self):
        self.vae = _QwenEditMockVAE()
        self.transformer = _QwenEditMockTransformer()
        self.vae_scale_factor = 8
        self._execution_device = torch.device("cpu")
        self.processor = object()  # just needs to exist

    def encode_prompt(self, *args, **kwargs):
        raise NotImplementedError("mock — validation tests never reach here")

    def prepare_latents(self, *args, **kwargs):
        raise NotImplementedError("mock — validation tests never reach here")


# Empty list
try:
    ml.generate_qwen_edit(
        pipe=_QwenEditMockPipe(),
        prompt="x",
        negative_prompt=None,
        reference_images=[],
    )
    check("qwen_edit_empty_list_raises", False, "should have raised")
except ValueError as e:
    check(
        "qwen_edit_empty_list_raises",
        "at least one" in str(e).lower(),
        f"wrong error: {e}",
    )

# None reference_images
try:
    ml.generate_qwen_edit(
        pipe=_QwenEditMockPipe(),
        prompt="x",
        negative_prompt=None,
        reference_images=None,
    )
    check("qwen_edit_none_refs_raises", False, "should have raised")
except ValueError as e:
    check(
        "qwen_edit_none_refs_raises",
        "at least one" in str(e).lower(),
        f"wrong error: {e}",
    )

# Mismatched vl_flags length
try:
    ml.generate_qwen_edit(
        pipe=_QwenEditMockPipe(),
        prompt="x",
        negative_prompt=None,
        reference_images=[torch.zeros(1, 16, 16, 3)],
        vl_flags=[True, True],  # length 2 vs refs length 1
    )
    check("qwen_edit_vl_flags_len_mismatch_raises", False, "should have raised")
except ValueError as e:
    check(
        "qwen_edit_vl_flags_len_mismatch_raises",
        "vl_flags length" in str(e),
        f"wrong error: {e}",
    )

# Mismatched ref_flags length
try:
    ml.generate_qwen_edit(
        pipe=_QwenEditMockPipe(),
        prompt="x",
        negative_prompt=None,
        reference_images=[torch.zeros(1, 16, 16, 3)],
        ref_flags=[True, True, True],  # length 3 vs refs length 1
    )
    check("qwen_edit_ref_flags_len_mismatch_raises", False, "should have raised")
except ValueError as e:
    check(
        "qwen_edit_ref_flags_len_mismatch_raises",
        "ref_flags length" in str(e),
        f"wrong error: {e}",
    )

# All flags False
try:
    ml.generate_qwen_edit(
        pipe=_QwenEditMockPipe(),
        prompt="x",
        negative_prompt=None,
        reference_images=[torch.zeros(1, 16, 16, 3), torch.zeros(1, 16, 16, 3)],
        vl_flags=[False, False],
        ref_flags=[False, False],
    )
    check("qwen_edit_all_flags_false_raises", False, "should have raised")
except ValueError as e:
    check(
        "qwen_edit_all_flags_false_raises",
        "both VL and Ref flags" in str(e) or "flags" in str(e).lower(),
        f"wrong error: {e}",
    )

# Single tensor auto-wrapped to list (ergonomic signature)
# This one would proceed past validation into the real pipeline code
# and fail at the mock's encode_prompt NotImplementedError — that's OK,
# we just verify the auto-wrap accepted the call past the "empty list"
# validation.
try:
    ml.generate_qwen_edit(
        pipe=_QwenEditMockPipe(),
        prompt="x",
        negative_prompt=None,
        reference_images=torch.zeros(1, 16, 16, 3),  # bare tensor
    )
    check("qwen_edit_bare_tensor_wrapped", False, "should have hit mock NotImplementedError")
except NotImplementedError:
    check("qwen_edit_bare_tensor_wrapped", True)
except ValueError as e:
    if "at least one" in str(e).lower():
        check("qwen_edit_bare_tensor_wrapped", False, "bare tensor not auto-wrapped")
    else:
        check("qwen_edit_bare_tensor_wrapped", False, f"unexpected error: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: generate_qwen_edit refinement path
# ═══════════════════════════════════════════════════════════════════════
#
# These tests verify the initial_latents + denoise kwargs added to
# generate_qwen_edit in Phase 2 Slice 1.  The refinement path is what
# lets Advanced Edit Multistage call generate_qwen_edit at S2/S3 with
# upscaled latents from the previous stage.
#
# Full dispatch can't be tested without a real Qwen Edit model (same
# constraint as Phase 1), but we can verify:
#   - Signature includes the new kwargs with correct defaults
#   - Shape mismatch on initial_latents raises ValueError
#   - Default (None, 1.0) preserves Phase 1 single-stage behavior

print("\n── Phase 2: generate_qwen_edit refinement path ──")


# ── Test: signature has initial_latents + denoise ─────────────────────
edit_sig = _inspect_edit.signature(ml.generate_qwen_edit)
edit_params = edit_sig.parameters
check(
    "generate_qwen_edit_has_initial_latents",
    "initial_latents" in edit_params,
)
check(
    "generate_qwen_edit_has_denoise",
    "denoise" in edit_params,
)
check(
    "generate_qwen_edit_initial_latents_default_None",
    edit_params["initial_latents"].default is None,
)
check(
    "generate_qwen_edit_denoise_default_1",
    edit_params["denoise"].default == 1.0,
)


# ── Test: generate_qwen_edit with mismatched initial_latents shape ────
# Path: pipe validates, then validates refs, then calls prepare_latents
# (our mock raises NotImplementedError there) — BEFORE reaching the
# shape check.  We can't hit the shape check without a working mock
# prepare_latents.  Instead we verify the shape check code EXISTS by
# looking at the source.  Coarse but useful as a regression net.
import inspect as _inspect_src
src = _inspect_src.getsource(ml.generate_qwen_edit)
check(
    "generate_qwen_edit_has_shape_check",
    "does not match expected shape" in src
    and "initial_latents" in src,
    "refinement-path shape check not found in source",
)
check(
    "generate_qwen_edit_has_denoise_truncation",
    "truncate_sigmas_for_denoise" in src
    and "inject_flow_noise" in src,
    "refinement-path noise injection not found in source",
)


# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 50}")
print(f"  {passed} passed, {failed} failed")
print(f"{'=' * 50}")
sys.exit(1 if failed else 0)
