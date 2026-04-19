# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Manual Denoising Loop

Implements text-to-image generation for Flux-family diffusers pipelines
(Flux, Flux2, Chroma) by manually running the denoising loop instead of
calling ``pipe(**kwargs)``.

Why this exists
---------------
Diffusers pipelines' ``__call__`` hard-codes a specific Euler update rule
and tightly couples the scheduler's ``step()`` method to the iteration.
Running alternative samplers (Heun, Runge-Kutta, multistep) requires
either:
  - Subclassing the scheduler to override ``step()`` — works only for
    multistep methods that fit one model call per step
    (see eric_diffusion_samplers.py), OR
  - Running the denoising loop manually — required for single-step
    higher-order methods (Heun, RK3, etc.) which do multiple model
    evaluations per step.

This module takes the manual-loop path.  It replicates the pipeline's
prompt encoding, latent preparation, and VAE decode, but substitutes the
inner denoising loop with a pluggable sampler function.

Supported model families
------------------------
Flux, Flux2, Chroma, Qwen-Image.  Flux/Flux2/Chroma share most of the
infrastructure.  Qwen-Image has its own ``generate_qwen`` /
``decode_qwen_latents`` because its text encoder (Qwen2.5-VL) returns
(embeds, mask) tuples, its transformer takes ``encoder_hidden_states_mask``
+ ``img_shapes`` instead of txt_ids/img_ids, and its VAE is 5D (has a
trivial temporal dimension). Its CFG uses a norm-preserving rescale on
top of the standard classical CFG formula.  Qwen latent packing happens
to be byte-identical to Flux's so ``upscale_flux_latents`` is reused for
Qwen multistage refinement.  Qwen-Image-Edit (image-conditioning) is
out of scope — that's a separate pipeline with its own manual loop work.

Author: Eric Hiss (GitHub: EricRollei)
"""

import inspect
import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════
#  Sampler algorithms — manual denoising loops
# ═══════════════════════════════════════════════════════════════════════
#
# Each sampler is a function with the signature:
#     sampler(denoiser, x, sigmas, progress_cb) -> torch.Tensor
#
# Where:
#   - denoiser : callable (x, sigma) -> velocity tensor.
#                For Flux-family flow matching, the transformer's output
#                IS the velocity, so the denoiser is a thin wrapper around
#                a single forward pass.
#   - x        : initial latent tensor (packed, shape [B, seq, C*4])
#   - sigmas   : descending sigma schedule (list or 1-D tensor),
#                length = num_denoising_steps + 1, last value typically
#                near-zero.
#   - progress_cb : optional callback invoked after every model call
#                   (not every denoising step — so Heun invokes it twice
#                   per step).  Signature: progress_cb(step_fractional).
#
# The sampler returns the final latent after all steps.


# ── Stochastic sampling helper ──────────────────────────────────────────
#
# Flow-matching stochastic sampling (what RES4LYF calls "eta") injects
# fresh noise at every denoising step, scaled by the eta parameter.
# Mathematically: at each step, we replace part of the deterministic
# Euler update with a fresh gaussian noise contribution.  This helps the
# model escape attractors in its velocity field and can produce more
# detailed output, especially at lower step counts.
#
# The update rule (from the stochastic flow matching formulation used in
# FlowMatchEulerDiscreteScheduler's stochastic_sampling mode):
#
#   Deterministic (eta=0):  x_{n+1} = x_n + dt * v
#   Stochastic (eta>0):     x0 = x_n - sigma * v
#                           sigma_next_eff = sigma_{n+1} * sqrt(1 - eta^2)
#                           noise = eta * sigma_{n+1} * gaussian
#                           x_{n+1} = (1 - sigma_next_eff) * x0 + sigma_next_eff * v_extra + noise
#
# Practically simpler form (equivalent, used below):
#
#   x0 = x_n - sigma * v                    # implicit denoised estimate
#   x_{n+1} = (1 - sigma_next) * x0
#           + sigma_next * sqrt(1 - eta^2) * noise_deterministic
#           + sigma_next * eta * noise_fresh
#
# Where noise_deterministic is derived from the current state to keep
# the trajectory consistent, and noise_fresh is freshly sampled.


def _stochastic_step(
    x: torch.Tensor,
    v: torch.Tensor,
    sigma: float,
    sigma_next: float,
    eta: float,
    generator=None,
) -> torch.Tensor:
    """Perform one stochastic Euler step with noise injection.

    At eta=0 this is equivalent to the deterministic Euler update
    ``x + (sigma_next - sigma) * v`` (within floating-point tolerance).
    At eta>0 we inject fresh noise scaled by eta.

    **Precision note**: the stochastic path upcasts to float32 for the
    intermediate math and downcasts back to the input dtype on return.
    This mirrors diffusers' FlowMatchEulerDiscreteScheduler.step() which
    does the same thing.  Reason: bf16 (the native dtype for Flux/Chroma
    transformers) only has ~8 mantissa bits; the multi-step arithmetic
    in the stochastic path accumulates rounding errors that correlate
    with tensor position and produce visible banding artifacts in the
    decoded output.  Upcasting for ~6 operations per step costs
    negligible memory but eliminates the banding.

    The deterministic eta=0 path stays in the original dtype because its
    single fused ``x + dt * v`` operation doesn't accumulate error the
    same way — and we want to preserve bitwise equivalence with the
    pre-eta deterministic baseline.
    """
    if eta <= 0.0 or abs(float(sigma_next)) < 1e-6:
        # Deterministic Euler (also: final step always deterministic
        # since we can't inject noise at sigma=0).  Single fused op,
        # no precision issue, keep native dtype.
        return x + (sigma_next - sigma) * v

    # ── Stochastic path: upcast for precision ─────────────────────────
    original_dtype = x.dtype
    x_f = x.to(torch.float32)
    v_f = v.to(torch.float32)

    # Estimate x0 from current state using flow-matching: x = (1-σ)x0 + σε
    # so x0 = x - σ * v  (where v is the predicted velocity = ε - x0)
    sigma_f = float(sigma)
    sigma_next_f = float(sigma_next)
    x0 = x_f - sigma_f * v_f

    # Determine stochastic and deterministic noise contributions
    eta_sq = min(eta * eta, 1.0)
    det_scale = (1.0 - eta_sq) ** 0.5

    # Deterministic noise component — equivalent to `x + (1-σ)*v`
    # (just the current state's implicit noise term, rearranged).
    noise_det = (x_f - (1.0 - sigma_f) * x0) / max(sigma_f, 1e-6)

    # Fresh stochastic noise — use diffusers' randn_tensor helper which
    # handles generator/device mismatches cleanly.  Direct torch.randn on
    # CUDA with packed (B, seq, C) shapes can produce subtle per-row
    # pattern artifacts in the generated noise that manifest as
    # horizontal banding in the decoded output.  randn_tensor matches
    # the noise-generation path that pipe.prepare_latents uses, which
    # produces clean spatial noise.
    from diffusers.utils.torch_utils import randn_tensor
    noise_fresh = randn_tensor(
        x_f.shape,
        generator=generator,
        device=x_f.device,
        dtype=torch.float32,
    )

    # New state: clean component + scaled noise at target sigma
    result = (
        (1.0 - sigma_next_f) * x0
        + sigma_next_f * det_scale * noise_det
        + sigma_next_f * eta * noise_fresh
    )

    return result.to(original_dtype)


def flow_euler(
    denoiser: Callable,
    x: torch.Tensor,
    sigmas,
    progress_cb=None,
    eta: float = 0.0,
    generator=None,
) -> torch.Tensor:
    """Standard 1st-order Euler integration of the flow-matching ODE.

    Update rule (eta=0, deterministic):
        v_n   = denoiser(x_n, sigma_n)
        x_{n+1} = x_n + (sigma_{n+1} - sigma_n) * v_n

    With ``eta > 0``, uses stochastic Euler (see _stochastic_step).
    eta=0 matches ``FlowMatchEulerDiscreteScheduler.step()`` exactly.
    eta=0.5 is a common RES4LYF default.
    eta=1.0 is pure noise injection (typically too aggressive).
    """
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        v = denoiser(x, sigma)
        x = _stochastic_step(x, v, sigma, sigma_next, eta, generator)
        if progress_cb is not None:
            progress_cb(i + 1)
    return x


def flow_heun(
    denoiser: Callable,
    x: torch.Tensor,
    sigmas,
    progress_cb=None,
    eta: float = 0.0,
    generator=None,
) -> torch.Tensor:
    """2nd-order Heun's method (predictor-corrector) for flow matching.

    Heun = Euler predictor + trapezoidal corrector:
        v_n       = denoiser(x_n, sigma_n)            # predictor eval
        x_pred    = x_n + (sigma_{n+1} - sigma_n) * v_n
        v_pred    = denoiser(x_pred, sigma_{n+1})     # corrector eval
        x_{n+1}   = x_n + (sigma_{n+1} - sigma_n) * (v_n + v_pred) / 2

    Cost: 2 model evaluations per denoising step (2× slower than Euler).
    Accuracy: local truncation error O(h³) vs O(h²) for Euler.

    With ``eta > 0``, the integration step becomes stochastic.  The
    predictor and corrector evaluations remain deterministic (so the
    2nd-order accuracy is preserved), but the final update is computed
    via ``_stochastic_step`` using the averaged velocity ``(v + v_pred) / 2``.
    """
    n_steps = len(sigmas) - 1
    for i in range(n_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Predictor
        v = denoiser(x, sigma)
        x_pred = x + (sigma_next - sigma) * v
        if progress_cb is not None:
            progress_cb(i + 0.5)

        # Corrector — skip only if sigma_next is effectively zero
        if abs(float(sigma_next)) < 1e-6:
            x = x_pred
        else:
            v_pred = denoiser(x_pred, sigma_next)
            v_avg = (v + v_pred) / 2.0
            x = _stochastic_step(x, v_avg, sigma, sigma_next, eta, generator)

        if progress_cb is not None:
            progress_cb(i + 1)
    return x


def flow_rk3(
    denoiser: Callable,
    x: torch.Tensor,
    sigmas,
    progress_cb=None,
    eta: float = 0.0,
    generator=None,
) -> torch.Tensor:
    """Classical 3rd-order Runge-Kutta (Kutta's method) for flow matching.

    Three-stage explicit RK3 applied to the flow-matching ODE:
        k1 = v(x, t)
        k2 = v(x + (h/2)*k1, t + h/2)
        k3 = v(x - h*k1 + 2*h*k2, t + h)
        x_{n+1} = x_n + (h/6) * (k1 + 4*k2 + k3)

    Cost: 3 model evaluations per denoising step (3× Euler).
    Accuracy: local truncation error O(h⁴).

    With ``eta > 0``, the k1/k2/k3 evaluations remain deterministic but
    the final integration uses ``_stochastic_step`` with the RK3-weighted
    velocity ``(k1 + 4*k2 + k3) / 6``.

    Falls back to plain Euler ONLY when sigma_next is effectively zero
    (to avoid evaluating the model at sigma=0).  Schedules that end at a
    small positive sigma get the full RK3 update on every step.
    """
    n_steps = len(sigmas) - 1
    for i in range(n_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        h = sigma_next - sigma

        if abs(float(sigma_next)) < 1e-6:
            v = denoiser(x, sigma)
            x = x + h * v
            if progress_cb is not None:
                progress_cb(i + 1)
            continue

        sigma_mid = sigma + h / 2.0

        k1 = denoiser(x, sigma)
        if progress_cb is not None:
            progress_cb(i + 1 / 3.0)

        k2 = denoiser(x + (h / 2.0) * k1, sigma_mid)
        if progress_cb is not None:
            progress_cb(i + 2 / 3.0)

        k3 = denoiser(x - h * k1 + 2 * h * k2, sigma_next)
        if progress_cb is not None:
            progress_cb(i + 1)

        v_rk3 = (k1 + 4 * k2 + k3) / 6.0
        x = _stochastic_step(x, v_rk3, sigma, sigma_next, eta, generator)
    return x


def flow_multistep2(
    denoiser: Callable,
    x: torch.Tensor,
    sigmas,
    progress_cb=None,
    eta: float = 0.0,
    generator=None,
) -> torch.Tensor:
    """2nd-order Adams-Bashforth multistep.

    Update:
        r = h_n / h_{n-1}
        v_eff = (1 + r/2) * v_n - (r/2) * v_{n-1}
        x_{n+1} = x_n + h_n * v_eff
    First step falls back to Euler (no previous velocity).

    Cost: 1 model evaluation per step (same as Euler).

    With ``eta > 0``, final integration uses ``_stochastic_step`` with
    the effective multistep velocity.
    """
    prev_v = None
    prev_dt = None
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        v = denoiser(x, sigma)
        if prev_v is None:
            v_eff = v
        else:
            r = dt / prev_dt if prev_dt != 0 else 1.0
            v_eff = (1.0 + r / 2.0) * v - (r / 2.0) * prev_v

        x = _stochastic_step(x, v_eff, sigma, sigma_next, eta, generator)

        prev_v = v
        prev_dt = dt
        if progress_cb is not None:
            progress_cb(i + 1)
    return x


def flow_multistep3(
    denoiser: Callable,
    x: torch.Tensor,
    sigmas,
    progress_cb=None,
    eta: float = 0.0,
    generator=None,
) -> torch.Tensor:
    """3rd-order Adams-Bashforth multistep for flow matching.

    Steps 0-1 use lower-order methods (Euler, AB2); subsequent steps use
    the uniform AB3 coefficients (23/12, -16/12, 5/12) when step sizes are
    roughly uniform, falling back to AB2 otherwise.

    Cost: 1 model evaluation per step.
    With ``eta > 0``, final integration uses ``_stochastic_step``.
    """
    prev_v1 = None
    prev_v2 = None
    prev_dt1 = None
    prev_dt2 = None
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        v = denoiser(x, sigma)

        if prev_v1 is None:
            v_eff = v
        elif prev_v2 is None:
            r = dt / prev_dt1 if prev_dt1 != 0 else 1.0
            v_eff = (1.0 + r / 2.0) * v - (r / 2.0) * prev_v1
        else:
            ratio = abs(prev_dt1 / prev_dt2 - 1.0) + abs(dt / prev_dt1 - 1.0)
            if ratio < 0.1:
                v_eff = (23.0 / 12.0) * v - (16.0 / 12.0) * prev_v1 + (5.0 / 12.0) * prev_v2
            else:
                r = dt / prev_dt1 if prev_dt1 != 0 else 1.0
                v_eff = (1.0 + r / 2.0) * v - (r / 2.0) * prev_v1

        x = _stochastic_step(x, v_eff, sigma, sigma_next, eta, generator)

        prev_v2 = prev_v1
        prev_dt2 = prev_dt1
        prev_v1 = v
        prev_dt1 = dt
        if progress_cb is not None:
            progress_cb(i + 1)
    return x


# Registry: sampler name → (function, description, cost_multiplier)
# cost_multiplier = how many model calls per step (Euler = 1, Heun = 2, RK3 = 3)

SAMPLERS = {
    "flow_euler":     (flow_euler,     "1st-order Euler (baseline, matches default)", 1),
    "flow_heun":      (flow_heun,      "2nd-order Heun predictor-corrector (2× cost, sharper)", 2),
    "flow_rk3":       (flow_rk3,       "3rd-order Runge-Kutta (3× cost, very smooth)", 3),
    "flow_multistep2": (flow_multistep2, "2nd-order Adams-Bashforth (same cost as Euler)", 1),
    "flow_multistep3": (flow_multistep3, "3rd-order Adams-Bashforth (same cost as Euler)", 1),
}


def sampler_names() -> list:
    return list(SAMPLERS.keys())


def get_sampler(name: str):
    if name not in SAMPLERS:
        raise ValueError(
            f"Unknown sampler {name!r}. Valid: {sampler_names()}"
        )
    return SAMPLERS[name][0]


def sampler_cost(name: str) -> int:
    """Return the per-step model evaluation cost (1, 2, or 3)."""
    if name not in SAMPLERS:
        return 1
    return SAMPLERS[name][2]


# ═══════════════════════════════════════════════════════════════════════
#  Flux-family manual denoising pipeline
# ═══════════════════════════════════════════════════════════════════════

def _compute_flux_shift_mu(seq_len: int, scheduler_config: dict) -> float:
    """Replicate calculate_shift() from diffusers' Flux pipeline.

    The mu value determines how the flow-matching sigma schedule is shifted
    based on the latent sequence length (higher resolution = more shift).
    """
    base_seq_len = scheduler_config.get("base_image_seq_len", 256)
    max_seq_len  = scheduler_config.get("max_image_seq_len", 4096)
    base_shift   = scheduler_config.get("base_shift", 0.5)
    max_shift    = scheduler_config.get("max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return m * seq_len + b


def _apply_mu_shift(sigmas: torch.Tensor, mu: float) -> torch.Tensor:
    """Apply exponential time shift to a raw sigma schedule.

    Matches FlowMatchEulerDiscreteScheduler.time_shift() with
    time_shift_type='exponential' (Flux/Chroma default).

    Formula: shifted = exp(mu) / (exp(mu) + 1/raw - 1)

    Preserves the endpoints (sigma=0 stays 0, sigma=1 stays 1).
    """
    exp_mu = math.exp(mu)
    sigmas_np = np.asarray(sigmas, dtype=np.float64)
    # Guard against division by zero at sigmas near 1 or 0
    eps = 1e-12
    safe = np.clip(sigmas_np, eps, 1.0 - eps)
    shifted = exp_mu / (exp_mu + 1.0 / safe - 1.0)
    # Preserve exact endpoints
    shifted[sigmas_np <= 0] = 0.0
    shifted[sigmas_np >= 1] = 1.0
    return torch.tensor(shifted, dtype=sigmas.dtype if torch.is_tensor(sigmas) else torch.float32)


def truncate_sigmas_for_denoise(sigmas: torch.Tensor, denoise: float) -> torch.Tensor:
    """Truncate a full sigma schedule to the lower (more-denoised) portion.

    For ``denoise == 1.0``, returns the schedule unchanged (full denoise from
    pure noise).  For ``denoise < 1.0``, returns the LAST ``ceil(N * denoise)``
    sigmas — i.e. starts at a partially-noised state and runs only the last
    portion of the schedule.

    The terminal zero (if present) is preserved.

    Example: full schedule [1.0, 0.8, 0.6, 0.4, 0.2, 0.0], denoise=0.5
             → [0.4, 0.2, 0.0]  (3 sigmas = 2 denoising steps from 0.4 → 0)
    """
    if denoise >= 1.0:
        return sigmas

    # Detect terminal zero so we can preserve it after truncation
    has_terminal_zero = abs(float(sigmas[-1])) < 1e-9
    work = sigmas[:-1] if has_terminal_zero else sigmas

    n_steps = len(work)
    keep_steps = max(1, int(round(n_steps * denoise)))
    truncated = work[n_steps - keep_steps:]

    if has_terminal_zero:
        terminal = torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)
        truncated = torch.cat([truncated, terminal])
    return truncated


def inject_flow_noise(
    latents: torch.Tensor, sigma: float, generator=None,
) -> torch.Tensor:
    """Add flow-matching noise to latents at the given sigma.

    Flow-matching noise blend:
        x_t = (1 - sigma) * x_0 + sigma * noise

    For ``sigma == 0`` returns latents unchanged.  For ``sigma == 1``
    returns pure noise.  In between, returns a partially-noised mix.

    **Precision note**: like ``_stochastic_step``, this upcasts to
    float32 for the blend and downcasts at the end.  bf16 arithmetic on
    multi-term blends accumulates rounding errors that can manifest as
    banding artifacts in decoded output (see notes in _stochastic_step).
    """
    from diffusers.utils.torch_utils import randn_tensor

    if sigma <= 0:
        return latents
    if sigma >= 1:
        noise_full = randn_tensor(
            latents.shape, generator=generator,
            device=latents.device, dtype=torch.float32,
        )
        return noise_full.to(latents.dtype)

    original_dtype = latents.dtype
    latents_f = latents.to(torch.float32)
    # Use randn_tensor (not torch.randn) to avoid per-row noise pattern
    # artifacts on CUDA with packed (B, seq, C) shapes — see notes in
    # _stochastic_step.
    noise = randn_tensor(
        latents_f.shape, generator=generator,
        device=latents_f.device, dtype=torch.float32,
    )
    result = (1.0 - sigma) * latents_f + sigma * noise
    return result.to(original_dtype)


def _build_raw_sigmas(num_steps: int, schedule: str) -> np.ndarray:
    """Build a raw (pre-mu-shift) sigma schedule.

    Returns a descending array of length ``num_steps`` going from
    ``sigma_max = 1.0`` down to ``sigma_min = 1/num_steps``.  The curve
    shape between those endpoints depends on the schedule name.

    Supported schedules:
      linear        — uniform spacing
      balanced      — Karras ρ=3 (moderate low-sigma concentration)
      karras        — Karras ρ=7 (heavy low-sigma concentration)
      betaXY        — beta distribution with α=X, β=Y (single digit each)
                      e.g. beta57 (α=5, β=7), beta13 (low-concentration),
                      beta31 (high-concentration)

    Beta schedules use the inverse CDF (percent point function) of the
    beta distribution mapped to ``[sigma_min, sigma_max]``.  The α/β
    parameters control where steps concentrate:
      α low, β high  → concentrate at low sigma (fine detail focus)
      α high, β low  → concentrate at high sigma (composition focus)
      α == β         → symmetric around mid sigma
      α == β == 1    → uniform (equivalent to linear)

    ``beta57`` (α=5, β=7) slightly concentrates toward the mid-low sigma
    region — an empirically useful middle-ground RES4LYF popularized.
    """
    sigma_max = 1.0
    sigma_min = 1.0 / num_steps

    if schedule == "linear":
        return np.linspace(sigma_max, sigma_min, num_steps)

    if schedule == "balanced":
        rho = 3.0
        t = np.linspace(0.0, 1.0, num_steps)
        return (sigma_max ** (1.0 / rho)
                + t * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho

    if schedule == "karras":
        rho = 7.0
        t = np.linspace(0.0, 1.0, num_steps)
        return (sigma_max ** (1.0 / rho)
                + t * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho

    if schedule.startswith("beta") and len(schedule) == 6:
        # betaXY → (α = X/10, β = Y/10)
        #
        # This matches RES4LYF / ComfyUI's convention (both in
        # comfy.samplers.beta_scheduler and RES4LYF's calls to it).
        # They use α and β as DECIMAL values less than 1, e.g.
        # "beta57" means α=0.5, β=0.7.  Both parameters below 1 give
        # a U-shaped beta distribution that concentrates at BOTH ends
        # (high sigma AND low sigma) with rapid passage through the
        # middle — the "composition + detail, skip the mushy middle"
        # pattern that works well on flow-matching models.
        try:
            alpha = int(schedule[4]) / 10.0
            beta_param = int(schedule[5]) / 10.0
            if alpha <= 0 or beta_param <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"Invalid beta schedule {schedule!r}.  Use betaXY where X, Y "
                f"are digits 1-9 (interpreted as α=X/10, β=Y/10).  "
                f"Example: beta57 = α=0.5, β=0.7 (RES4LYF default)."
            )
        try:
            from scipy.stats import beta as _beta
        except ImportError:
            raise ImportError(
                "scipy is required for beta sigma schedules.  "
                "Install: pip install scipy"
            )
        # Follow ComfyUI's beta_scheduler algorithm:
        #   ts = 1 - linspace(0, 1, N, endpoint=False)  # descending (0, 1]
        #   raw_beta = beta.ppf(ts, α, β)               # descending in (0, 1)
        # The PPF output directly gives us sigma positions — no
        # normalization needed because beta.ppf(ts, α, β) naturally
        # spans (0, 1) when ts spans (0, 1].
        t = 1.0 - np.linspace(0.0, 1.0, num_steps, endpoint=False)
        raw_beta = _beta.ppf(t, alpha, beta_param)
        # Map to [sigma_min, sigma_max].  Clamp to avoid PPF numerical
        # edge cases producing values slightly outside [0, 1].
        raw_beta = np.clip(raw_beta, 0.0, 1.0)
        raw = sigma_min + raw_beta * (sigma_max - sigma_min)
        # Ensure strict descent (beta.ppf is monotonic but numerical
        # precision can produce tiny non-monotonicities near endpoints)
        for i in range(1, len(raw)):
            if raw[i] > raw[i - 1]:
                raw[i] = raw[i - 1]
        return raw

    raise ValueError(f"Unknown schedule: {schedule!r}")


# Names of all sigma schedules exposed to node dropdowns.
SIGMA_SCHEDULE_NAMES = [
    "linear",
    "balanced",
    "karras",
    "beta57",   # α=5, β=7 — RES4LYF-inspired mild low-mid concentration
    "beta75",   # α=7, β=5 — mild high-mid concentration
    "beta33",   # α=3, β=3 — symmetric, moderate mid concentration
    "beta13",   # α=1, β=3 — heavy low-sigma concentration
    "beta31",   # α=3, β=1 — heavy high-sigma concentration
]


def build_flux_sigmas(
    num_steps: int,
    image_seq_len: int,
    scheduler_config: dict,
    schedule: str = "linear",
) -> torch.Tensor:
    """Build a flow-matching sigma schedule with Flux mu-shifting applied.

    Args:
        num_steps        : number of denoising steps (not the returned length)
        image_seq_len    : packed latent sequence length (used to compute mu)
        scheduler_config : dict-like (or FrozenDict) from pipe.scheduler.config
        schedule         : see ``_build_raw_sigmas`` for supported options.

    Returns:
        torch.Tensor of length ``num_steps + 1`` with the terminal 0 appended,
        already mu-shifted.  Descending.
    """
    raw = _build_raw_sigmas(num_steps, schedule)
    mu = _compute_flux_shift_mu(image_seq_len, scheduler_config)
    shifted = _apply_mu_shift(torch.tensor(raw, dtype=torch.float32), mu)
    final = torch.cat([shifted, torch.zeros(1, dtype=shifted.dtype)])
    return final


def _ensure_text_encoders_on_device(pipe, device) -> None:
    """Move text encoders to the target device if they're not already there.

    **Critical: skip any module with accelerate hooks.**  When the
    pipeline was loaded with ``device_map="balanced"`` or any
    accelerate-based dispatch, the text encoders have ``_hf_hook``
    attributes and are actively managed by accelerate.  Manually calling
    ``.to(device)`` on a hooked module partially moves it, leaving the
    module in a split state where some tensors are on one device and
    others are on another.  Accelerate specifically warns about this
    ("You shouldn't move a model that is dispatched using accelerate
    hooks") and the result is device-mismatch errors during the next
    forward pass.

    This helper is only useful for the edge case where a text encoder
    was pre-loaded via the component loader's override path and NOT
    wrapped with accelerate hooks — e.g. single-GPU mode where we passed
    a CPU-loaded text encoder to ``from_pretrained`` and it stayed on
    CPU.  In that case there's no hook to manage device placement and
    our manual ``.to()`` is the only way to get it to the execution
    device.

    For normal multi-GPU balanced dispatch, this function is a no-op
    because every text encoder has ``_hf_hook`` set and gets skipped.
    """
    for te_name in ("text_encoder", "text_encoder_2"):
        te = getattr(pipe, te_name, None)
        if te is None:
            continue
        # NEVER touch accelerate-hooked modules — the hooks handle
        # device placement and manual .to() corrupts the state.
        if hasattr(te, "_hf_hook"):
            continue
        try:
            first_param = next(te.parameters(), None)
            if first_param is None:
                continue
            if first_param.device != device:
                te.to(device)
        except Exception as e:
            print(f"[EricDiffusion] Note: could not move "
                  f"{te_name} to {device}: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  Flux pipeline manual generation
# ═══════════════════════════════════════════════════════════════════════

def generate_flux(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    sampler_name: str,
    sigma_schedule: str,
    generator: Optional[torch.Generator],
    max_sequence_length: int = 512,
    progress_cb: Optional[Callable] = None,
    initial_latents: Optional[torch.Tensor] = None,
    denoise: float = 1.0,
    eta: float = 0.0,
) -> torch.Tensor:
    """Generate an image from a Flux/Flux2/Chroma pipeline via manual loop.

    Replicates ``FluxPipeline.__call__`` with the denoising loop substituted
    by one of the samplers in ``SAMPLERS``.  Returns the final latents in
    packed format — caller is responsible for VAE decoding.

    Args:
        initial_latents : if provided, use these as the starting state instead
                          of pure noise.  Required shape: packed [B, seq, C*4]
                          matching the (height, width) target.  Used for
                          multi-stage refinement: pass the upscaled latents
                          from the previous stage and set ``denoise < 1.0``.
        denoise         : fraction of the sigma schedule to use.  ``1.0`` runs
                          the full schedule from pure noise.  ``0.85`` runs the
                          last 85% of the schedule starting from a partially-
                          noised state.  Only meaningful with initial_latents.
    """
    # ── Input validation ────────────────────────────────────────────────
    if not hasattr(pipe, "transformer") or not hasattr(pipe, "vae"):
        raise ValueError("Pipeline must have .transformer and .vae attributes")
    if not hasattr(pipe, "encode_prompt"):
        raise ValueError(
            "Pipeline does not expose encode_prompt() — manual loop only "
            "supports Flux-family pipelines for now."
        )

    device = getattr(pipe, "_execution_device", None) or next(
        pipe.transformer.parameters()
    ).device
    dtype = pipe.transformer.dtype

    # Defensively ensure text encoders are on the execution device —
    # handles balanced-mode edge cases and component-loader overrides
    # that leave text encoders on CPU.
    _ensure_text_encoders_on_device(pipe, device)

    # ── Step 1: Encode prompts ─────────────────────────────────────────
    # FluxPipeline.encode_prompt returns (prompt_embeds, pooled_prompt_embeds,
    # text_ids) where text_ids is a positional ID tensor.
    sig = inspect.signature(pipe.encode_prompt)
    enc_kwargs = dict(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
    )
    if "max_sequence_length" in sig.parameters:
        enc_kwargs["max_sequence_length"] = max_sequence_length

    encode_result = pipe.encode_prompt(**enc_kwargs)
    if len(encode_result) == 3:
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_result
    elif len(encode_result) == 2:
        prompt_embeds, pooled_prompt_embeds = encode_result
        text_ids = None
    else:
        raise ValueError(f"encode_prompt returned {len(encode_result)} values, expected 2 or 3")

    # ── CFG mode selection ─────────────────────────────────────────────
    # Flux-family models split into TWO disjoint guidance mechanisms:
    #
    # 1. Distilled guidance (guidance_embeds=True): Flux dev, Flux2, etc.
    #    The model takes a `guidance` embedding as input and the
    #    distillation training has internalized the CFG behavior.
    #    Single forward pass per step. NEVER also do classical CFG —
    #    that would compound (double-CFG bug: oversaturated, ignores
    #    prompt details, "tunnel vision" focus on the main subject).
    #
    # 2. Classical CFG (guidance_embeds=False): Chroma, Flux schnell, etc.
    #    No guidance embedding. Two forward passes per step (positive +
    #    negative), combined as v_neg + scale * (v - v_neg). Requires
    #    a negative prompt to do anything.
    #
    # Diffusers' FluxPipeline.__call__ uses exactly this dispatch.
    uses_distilled_guidance = bool(
        getattr(pipe.transformer.config, "guidance_embeds", False)
    )

    neg_prompt_embeds = None
    neg_pooled_prompt_embeds = None
    do_cfg = (
        not uses_distilled_guidance
        and guidance_scale > 1.0
        and bool(negative_prompt)
    )
    if do_cfg:
        neg_result = pipe.encode_prompt(
            **{**enc_kwargs, "prompt": negative_prompt}
        )
        if len(neg_result) >= 2:
            neg_prompt_embeds = neg_result[0]
            neg_pooled_prompt_embeds = neg_result[1]

    # ── Step 2: Prepare initial latents ────────────────────────────────
    # Always call prepare_latents with latents=None so we get the
    # latent_image_ids tensor for THIS resolution.  If the caller provided
    # initial_latents, we'll swap them in afterward.
    num_channels_latents = pipe.transformer.config.in_channels // 4

    prepare_kwargs = dict(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    fresh_latents, latent_image_ids = pipe.prepare_latents(**prepare_kwargs)

    if initial_latents is None:
        latents = fresh_latents
    else:
        # Refinement path: use the caller's latents (already packed)
        # but still take the latent_image_ids for this resolution from the
        # prepare_latents call above.
        latents = initial_latents.to(device=device, dtype=fresh_latents.dtype)
        if latents.shape != fresh_latents.shape:
            raise ValueError(
                f"initial_latents shape {tuple(latents.shape)} does not match "
                f"expected shape for {height}x{width}: {tuple(fresh_latents.shape)}.\n"
                f"Did you forget to upscale the latents to the new resolution?"
            )

    # ── Step 3: Build the sigma schedule ───────────────────────────────
    seq_len = latents.shape[1]  # packed sequence length
    sigmas = build_flux_sigmas(
        num_steps=num_inference_steps,
        image_seq_len=seq_len,
        scheduler_config=dict(pipe.scheduler.config),
        schedule=sigma_schedule,
    )
    sigmas = sigmas.to(device=device)

    # ── Refinement: truncate schedule + inject noise at starting sigma ──
    if initial_latents is not None and denoise < 1.0:
        sigmas = truncate_sigmas_for_denoise(sigmas, denoise)
        starting_sigma = float(sigmas[0])
        latents = inject_flow_noise(latents, starting_sigma, generator=generator)
    elif initial_latents is not None and denoise >= 1.0:
        # Full re-denoise from noise — replace latents entirely with random
        latents = torch.randn(
            latents.shape, generator=generator,
            device=latents.device, dtype=latents.dtype,
        )

    # ── Step 4: Build the denoiser callable ────────────────────────────
    # The Flux transformer takes:
    #   hidden_states  (= latents)
    #   timestep       (= sigma * 1000, as per the pipeline convention)
    #   guidance       (guidance embedding scalar)
    #   pooled_projections
    #   encoder_hidden_states
    #   txt_ids
    #   img_ids
    # and returns velocity for that step.

    guidance_embed = None
    if uses_distilled_guidance:
        # Distilled-guidance models (Flux dev, Flux2): pass guidance_scale
        # as a transformer input embedding.  This is the model's *only*
        # CFG mechanism — classical CFG is disabled above.
        guidance_embed = torch.full(
            [1], guidance_scale, device=device, dtype=dtype,
        )

    def _call_transformer(x, sigma, embeds, pooled):
        # sigma is a scalar or 0-d tensor; convert to per-batch timestep
        if not torch.is_tensor(sigma):
            sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        else:
            sigma_t = sigma.to(device=device, dtype=dtype)
        timestep = sigma_t.expand(x.shape[0]).to(dtype=dtype)

        call_kwargs = dict(
            hidden_states=x,
            timestep=timestep / 1.0,  # sigma already in [0, 1] range
            pooled_projections=pooled,
            encoder_hidden_states=embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )
        if guidance_embed is not None:
            call_kwargs["guidance"] = guidance_embed

        out = pipe.transformer(**call_kwargs)
        return out[0] if isinstance(out, tuple) else out

    def denoiser(x, sigma):
        v = _call_transformer(x, sigma, prompt_embeds, pooled_prompt_embeds)
        if do_cfg:
            v_neg = _call_transformer(
                x, sigma, neg_prompt_embeds, neg_pooled_prompt_embeds,
            )
            v = v_neg + guidance_scale * (v - v_neg)
        return v

    # ── Step 5: Run the sampler ────────────────────────────────────────
    sampler_fn = get_sampler(sampler_name)

    # Wrap the pipeline's progress callback so it fires once per full step
    # (not once per sub-evaluation in Heun/RK3)
    last_step_reported = [0]
    def _progress_wrapper(step_fractional):
        step_int = int(step_fractional)
        if step_int > last_step_reported[0]:
            last_step_reported[0] = step_int
            if progress_cb is not None:
                progress_cb(step_int)

    with torch.no_grad():
        final_latents = sampler_fn(denoiser, latents, sigmas, _progress_wrapper, eta=eta, generator=generator)

    return final_latents


def upscale_flux_latents(
    packed_latents: torch.Tensor,
    src_h: int, src_w: int,
    dst_h: int, dst_w: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Upscale Flux-format packed latents from one resolution to another.

    Flux packs latents as ``(B, num_patches, C*4)`` where ``num_patches =
    (H_lat/2) * (W_lat/2)``.  To upscale we:
      1. Unpack to spatial format ``(B, C, H_lat, W_lat)``
      2. Run bislerp interpolation (norm-preserving, vector-aware)
      3. Re-pack to ``(B, new_num_patches, C*4)``

    Bislerp is what ComfyUI uses internally for latent upscaling — it
    blends interpolated values along the latent vector direction rather
    than treating each channel as independent, which preserves the
    norm/direction relationships that VAE decoders are sensitive to.

    Args:
        packed_latents : input ``(B, src_seq, C*4)``
        src_h, src_w   : pixel dimensions of the source resolution
        dst_h, dst_w   : pixel dimensions of the target resolution
        vae_scale_factor : VAE compression (default 8 for Flux)

    Returns:
        Packed latents at the new resolution: ``(B, dst_seq, C*4)``
    """
    import comfy.utils as comfy_utils

    # Compute source latent shape (matches FluxPipeline._unpack_latents math)
    src_h_lat = 2 * (int(src_h) // (vae_scale_factor * 2))
    src_w_lat = 2 * (int(src_w) // (vae_scale_factor * 2))
    dst_h_lat = 2 * (int(dst_h) // (vae_scale_factor * 2))
    dst_w_lat = 2 * (int(dst_w) // (vae_scale_factor * 2))

    B, src_seq, C4 = packed_latents.shape
    C = C4 // 4

    # Unpack: (B, src_seq, C*4) → (B, src_h_lat, src_w_lat // 2, C, 2, 2)
    spatial = packed_latents.view(
        B, src_h_lat // 2, src_w_lat // 2, C, 2, 2
    )
    spatial = spatial.permute(0, 3, 1, 4, 2, 5)
    spatial = spatial.reshape(B, C, src_h_lat, src_w_lat)

    # Bislerp upscale in latent space (preserves vector norms)
    upscaled = comfy_utils.bislerp(spatial, dst_w_lat, dst_h_lat)
    # bislerp returns (B, C, dst_h_lat, dst_w_lat)

    # Re-pack: (B, C, dst_h_lat, dst_w_lat) → (B, dst_seq, C*4)
    repacked = upscaled.view(
        B, C, dst_h_lat // 2, 2, dst_w_lat // 2, 2
    )
    repacked = repacked.permute(0, 2, 4, 1, 3, 5)
    repacked = repacked.reshape(
        B, (dst_h_lat // 2) * (dst_w_lat // 2), C * 4
    )

    return repacked


# ═══════════════════════════════════════════════════════════════════════
#  Chroma support — separate path because encode_prompt and transformer
#  signatures differ from Flux
# ═══════════════════════════════════════════════════════════════════════
#
# Chroma is architecturally a Flux derivative but the pipeline interface
# differs in ways that prevent `generate_flux` from handling it:
#
#   1. `encode_prompt` returns 6 values (not 3):
#        (prompt_embeds, text_ids, prompt_attention_mask,
#         neg_prompt_embeds, neg_text_ids, neg_attention_mask)
#      Positive and negative are encoded in ONE call, not two separate calls.
#
#   2. Chroma transformer forward has:
#      - NO pooled_projections (no CLIP pooled embedding)
#      - NO guidance parameter (guidance_embeds=False always)
#      - HAS attention_mask (passed alongside encoder_hidden_states)
#
#   3. Always classical CFG (guidance_embeds=False).  Two forward passes
#      per denoising step.
#
# Chroma DOES share Flux's mu/sigma math (`build_flux_sigmas`) and VAE
# normalization (`decode_flux_latents`), so those are reused as-is.
# Chroma also shares Flux's latent packing so `upscale_flux_latents`
# works unchanged for multistage refinement.


def generate_chroma(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    sampler_name: str,
    sigma_schedule: str,
    generator: Optional[torch.Generator],
    max_sequence_length: int = 512,
    progress_cb: Optional[Callable] = None,
    initial_latents: Optional[torch.Tensor] = None,
    denoise: float = 1.0,
    eta: float = 0.0,
) -> torch.Tensor:
    """Manual denoising loop for Chroma pipelines.

    Returns packed latents (same shape convention as Flux) so the standard
    `decode_flux_latents` and `upscale_flux_latents` work unchanged.
    """
    if not hasattr(pipe, "transformer") or not hasattr(pipe, "vae"):
        raise ValueError("Pipeline must have .transformer and .vae attributes")
    if not hasattr(pipe, "encode_prompt"):
        raise ValueError("Chroma pipeline does not expose encode_prompt()")

    device = getattr(pipe, "_execution_device", None) or next(
        pipe.transformer.parameters()
    ).device
    dtype = pipe.transformer.dtype

    # Defensively ensure text encoders are on the execution device.
    _ensure_text_encoders_on_device(pipe, device)

    # ── Step 1: Encode prompts (one call, returns 6 values) ───────────
    # Chroma's encode_prompt handles both positive and negative in a
    # single call and expects do_classifier_free_guidance=True to return
    # valid negative embeddings.  We pass an empty string if no negative
    # prompt is provided — Chroma generates an "unconditional" embedding
    # for that case.
    effective_neg = negative_prompt if negative_prompt else ""
    do_cfg = guidance_scale > 1.0

    sig = inspect.signature(pipe.encode_prompt)
    enc_kwargs = dict(
        prompt=prompt,
        negative_prompt=effective_neg,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
    )
    if "max_sequence_length" in sig.parameters:
        enc_kwargs["max_sequence_length"] = max_sequence_length

    encode_result = pipe.encode_prompt(**enc_kwargs)
    if len(encode_result) != 6:
        raise ValueError(
            f"Expected Chroma encode_prompt to return 6 values, got "
            f"{len(encode_result)}."
        )
    (
        prompt_embeds,
        text_ids,
        prompt_attention_mask,
        neg_prompt_embeds,
        neg_text_ids,
        neg_attention_mask,
    ) = encode_result

    # ── Step 2: Prepare initial latents ────────────────────────────────
    num_channels_latents = pipe.transformer.config.in_channels // 4
    prepare_kwargs = dict(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    fresh_latents, latent_image_ids = pipe.prepare_latents(**prepare_kwargs)

    if initial_latents is None:
        latents = fresh_latents
    else:
        latents = initial_latents.to(device=device, dtype=fresh_latents.dtype)
        if latents.shape != fresh_latents.shape:
            raise ValueError(
                f"initial_latents shape {tuple(latents.shape)} does not match "
                f"expected shape for {height}x{width}: {tuple(fresh_latents.shape)}.\n"
                f"Did you forget to upscale the latents to the new resolution?"
            )

    # ── Step 3: Build the sigma schedule (uses Flux mu math) ───────────
    image_seq_len = latents.shape[1]
    sigmas = build_flux_sigmas(
        num_steps=num_inference_steps,
        image_seq_len=image_seq_len,
        scheduler_config=dict(pipe.scheduler.config),
        schedule=sigma_schedule,
    )
    sigmas = sigmas.to(device=device)

    # Refinement: truncate + inject noise at starting sigma
    if initial_latents is not None and denoise < 1.0:
        sigmas = truncate_sigmas_for_denoise(sigmas, denoise)
        starting_sigma = float(sigmas[0])
        latents = inject_flow_noise(latents, starting_sigma, generator=generator)
    elif initial_latents is not None and denoise >= 1.0:
        latents = torch.randn(
            latents.shape, generator=generator,
            device=latents.device, dtype=latents.dtype,
        )

    # ── Extend attention masks to cover text + image tokens ────────────
    # Chroma's transformer attends over the concatenated sequence
    # [text_tokens, image_tokens].  The raw text-only masks from
    # encode_prompt only cover the text portion.  We must extend them
    # with ones for every image token position, or the SDPA broadcast
    # in the attention layers fails with a size mismatch against the
    # full combined sequence length.
    #
    # This mirrors what ChromaPipeline.__call__ does around line 200:
    #     attention_mask = self._prepare_attention_mask(
    #         batch_size, image_seq_len, dtype, attention_mask=prompt_attention_mask)
    prompt_attention_mask_full = pipe._prepare_attention_mask(
        batch_size=latents.shape[0],
        sequence_length=image_seq_len,
        dtype=prompt_embeds.dtype,
        attention_mask=prompt_attention_mask,
    )
    if do_cfg:
        neg_attention_mask_full = pipe._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=prompt_embeds.dtype,
            attention_mask=neg_attention_mask,
        )
    else:
        neg_attention_mask_full = None

    # ── Step 4: Build the denoiser callable ────────────────────────────
    # Chroma transformer forward signature:
    #   hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids,
    #   attention_mask, joint_attention_kwargs, ... , return_dict
    # NO pooled_projections, NO guidance.  attention_mask MUST be the
    # extended (text + image tokens) version, not the raw text mask.

    def _call_transformer(x, sigma, embeds, attn_mask, txt_ids_local):
        if not torch.is_tensor(sigma):
            sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        else:
            sigma_t = sigma.to(device=device, dtype=dtype)
        timestep = sigma_t.expand(x.shape[0]).to(dtype=dtype)

        out = pipe.transformer(
            hidden_states=x,
            encoder_hidden_states=embeds,
            timestep=timestep,
            img_ids=latent_image_ids,
            txt_ids=txt_ids_local,
            attention_mask=attn_mask,
            return_dict=False,
        )
        return out[0] if isinstance(out, tuple) else out

    def denoiser(x, sigma):
        v = _call_transformer(
            x, sigma, prompt_embeds, prompt_attention_mask_full, text_ids,
        )
        if do_cfg:
            v_neg = _call_transformer(
                x, sigma, neg_prompt_embeds, neg_attention_mask_full, neg_text_ids,
            )
            v = v_neg + guidance_scale * (v - v_neg)
        return v

    # ── Step 5: Run the sampler ────────────────────────────────────────
    sampler_fn = get_sampler(sampler_name)
    last_step_reported = [0]

    def _progress_wrapper(step_fractional):
        step_int = int(step_fractional)
        if step_int > last_step_reported[0]:
            last_step_reported[0] = step_int
            if progress_cb is not None:
                progress_cb(step_int)

    with torch.no_grad():
        final_latents = sampler_fn(denoiser, latents, sigmas, _progress_wrapper, eta=eta, generator=generator)

    return final_latents


# ═══════════════════════════════════════════════════════════════════════
#  Qwen-Image support — stock diffusers QwenImagePipeline (t2i only)
# ═══════════════════════════════════════════════════════════════════════
#
# Qwen-Image is architecturally a close cousin of Flux but the pipeline
# surface differs in several ways that keep it out of generate_flux /
# generate_chroma:
#
#   1. Text encoder is Qwen2.5-VL (multimodal LLM).  encode_prompt
#      returns a 2-tuple (prompt_embeds, prompt_embeds_mask).  No
#      pooled projection, no positional text ids.  Positive and negative
#      require TWO separate calls to encode_prompt (unlike Chroma which
#      does both in one call).
#
#   2. prepare_latents returns a single tensor (not the (latents, ids)
#      tuple Flux uses).  Position info is passed to the transformer
#      via ``img_shapes`` instead of latent_image_ids.
#
#   3. Transformer forward signature:
#        hidden_states, timestep, guidance, encoder_hidden_states,
#        encoder_hidden_states_mask, img_shapes, attention_kwargs
#      No pooled_projections, no txt_ids/img_ids, ``guidance`` is only
#      meaningful if transformer.config.guidance_embeds=True (which is
#      False on the public Qwen-Image checkpoints — guidance is dead
#      code, classical CFG is the real mechanism).
#
#   4. CFG uses **norm-preserving rescaling** on top of the standard
#      classical CFG formula:
#          comb = neg + scale * (cond - neg)
#          noise_pred = comb * (||cond|| / ||comb||)
#      where the norm is taken along the feature dim.  This matches
#      the model's distribution better than raw CFG at high scales.
#      The public recommendation is 50 steps at true_cfg_scale=4.0.
#
#   5. VAE is 5D (video-style).  _unpack_latents returns
#      ``(B, C, 1, H_lat, W_lat)`` — note the trivial temporal dim —
#      and decode emits a 5D tensor that needs a ``[:, :, 0]`` slice
#      to drop the temporal axis before it looks like a normal image.
#      Also requires explicit latents_mean/latents_std denormalization
#      (not the single scaling_factor/shift_factor the Flux VAE uses).
#
# Latent packing math is byte-identical to Flux: same
# ``(B, C, H//2, 2, W//2, 2) → (B, seq, C*4)`` pattern, same
# vae_scale_factor of 8, so ``upscale_flux_latents`` is reused unchanged
# for Qwen multistage refinement.  Only decode has to be Qwen-specific.


def generate_qwen(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    sampler_name: str,
    sigma_schedule: str,
    generator: Optional[torch.Generator],
    max_sequence_length: int = 512,
    progress_cb: Optional[Callable] = None,
    initial_latents: Optional[torch.Tensor] = None,
    denoise: float = 1.0,
    eta: float = 0.0,
) -> torch.Tensor:
    """Manual denoising loop for Qwen-Image (text-to-image) pipelines.

    Returns packed latents (same shape convention as Flux) so caller
    should pass them to ``decode_qwen_latents`` — NOT ``decode_flux_latents``,
    because Qwen's VAE is 5D and needs different denormalization.

    For multistage refinement, ``upscale_flux_latents`` works as-is on
    Qwen packed latents because the packing math is identical.

    The ``guidance_scale`` argument is Qwen's ``true_cfg_scale`` — i.e.
    classical two-pass CFG.  Values above 1.0 trigger the double forward
    pass; the public recommendation is 50 steps at scale 4.0.
    """
    if not hasattr(pipe, "transformer") or not hasattr(pipe, "vae"):
        raise ValueError("Pipeline must have .transformer and .vae attributes")
    if not hasattr(pipe, "encode_prompt"):
        raise ValueError("Qwen pipeline does not expose encode_prompt()")

    device = getattr(pipe, "_execution_device", None) or next(
        pipe.transformer.parameters()
    ).device
    dtype = pipe.transformer.dtype

    # Defensively ensure text encoders are on the execution device.
    _ensure_text_encoders_on_device(pipe, device)

    # ── Step 1: Encode prompts (two separate calls) ────────────────────
    # QwenImagePipeline.encode_prompt returns (prompt_embeds, prompt_embeds_mask).
    # Positive and negative are two independent calls; there's no
    # do_classifier_free_guidance flag that encodes both.
    #
    # CRITICAL: don't pass device= to encode_prompt.  The custom
    # pipelines use model_inputs.attention_mask downstream of the text
    # encoder in _extract_masked_hidden, and if we pass a device that
    # diverges from self._execution_device, model_inputs ends up on
    # that device while the text encoder's outputs get moved to
    # _execution_device by _fix_text_encoder_device's post-hook —
    # cross-device indexing crash.  Letting encode_prompt default to
    # _execution_device keeps everything consistent.  See also
    # generate_qwen_edit for the same fix.
    sig = inspect.signature(pipe.encode_prompt)
    enc_kwargs = dict(
        prompt=prompt,
        num_images_per_prompt=1,
    )
    if "max_sequence_length" in sig.parameters:
        enc_kwargs["max_sequence_length"] = max_sequence_length

    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(**enc_kwargs)

    # Move embeddings to the transformer's device for the sampler loop.
    prompt_embeds = prompt_embeds.to(device=device)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(device=device)

    # CFG only fires if we have a negative prompt AND scale > 1.
    # Same pattern as FluxPipeline for distilled-guidance models, but
    # Qwen's guidance_embeds is False so there's no distilled fallback —
    # scale=1 means no guidance at all, scale>1 without a negative is a
    # no-op that the pipeline warns about.
    do_cfg = guidance_scale > 1.0 and bool(negative_prompt)
    neg_prompt_embeds = None
    neg_prompt_embeds_mask = None
    if do_cfg:
        neg_prompt_embeds, neg_prompt_embeds_mask = pipe.encode_prompt(
            **{**enc_kwargs, "prompt": negative_prompt}
        )
        neg_prompt_embeds = neg_prompt_embeds.to(device=device)
        if neg_prompt_embeds_mask is not None:
            neg_prompt_embeds_mask = neg_prompt_embeds_mask.to(device=device)

    # ── Step 2: Prepare initial latents ────────────────────────────────
    # Qwen's prepare_latents returns a single packed tensor, not a tuple.
    num_channels_latents = pipe.transformer.config.in_channels // 4
    fresh_latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    if initial_latents is None:
        latents = fresh_latents
    else:
        latents = initial_latents.to(device=device, dtype=fresh_latents.dtype)
        if latents.shape != fresh_latents.shape:
            raise ValueError(
                f"initial_latents shape {tuple(latents.shape)} does not match "
                f"expected shape for {height}x{width}: {tuple(fresh_latents.shape)}.\n"
                f"Did you forget to upscale the latents to the new resolution?"
            )

    # ── Step 3: Build the sigma schedule ───────────────────────────────
    # Qwen uses the same calculate_shift formula as Flux (base_seq_len=256,
    # max_seq_len=4096, base_shift=0.5, max_shift=1.15) with parameters
    # read from the scheduler config — so build_flux_sigmas produces the
    # exact same schedule the stock pipeline would.
    image_seq_len = latents.shape[1]
    sigmas = build_flux_sigmas(
        num_steps=num_inference_steps,
        image_seq_len=image_seq_len,
        scheduler_config=dict(pipe.scheduler.config),
        schedule=sigma_schedule,
    )
    sigmas = sigmas.to(device=device)

    # Refinement: truncate schedule + inject noise at starting sigma
    if initial_latents is not None and denoise < 1.0:
        sigmas = truncate_sigmas_for_denoise(sigmas, denoise)
        starting_sigma = float(sigmas[0])
        latents = inject_flow_noise(latents, starting_sigma, generator=generator)
    elif initial_latents is not None and denoise >= 1.0:
        latents = torch.randn(
            latents.shape, generator=generator,
            device=latents.device, dtype=latents.dtype,
        )

    # ── Step 4: Build img_shapes for the transformer ────────────────────
    # The Qwen transformer uses img_shapes (list of list of tuples) for
    # positional information.  Outer list = batch, inner list = one entry
    # per image (always 1 for t2i), tuple = (frame_count, h/16, w/16).
    vae_scale_factor = pipe.vae_scale_factor
    img_shapes = [
        [(1, height // vae_scale_factor // 2, width // vae_scale_factor // 2)]
    ] * latents.shape[0]

    # ── Step 5: Build the denoiser callable ────────────────────────────
    # Qwen transformer forward signature:
    #   hidden_states, timestep, guidance, encoder_hidden_states_mask,
    #   encoder_hidden_states, img_shapes, attention_kwargs, return_dict
    # No pooled_projections, no txt_ids/img_ids.  guidance=None because
    # transformer.config.guidance_embeds is False on Qwen-Image-2512
    # (classical CFG is the actual guidance mechanism).
    guidance_embed = None
    if bool(getattr(pipe.transformer.config, "guidance_embeds", False)):
        # Hypothetical future Qwen variant with distilled guidance —
        # keep the code path defensive.  The public checkpoints do NOT
        # trigger this.
        guidance_embed = torch.full(
            [1], guidance_scale, device=device, dtype=torch.float32,
        ).expand(latents.shape[0])

    def _call_transformer(x, sigma, embeds, mask):
        if not torch.is_tensor(sigma):
            sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        else:
            sigma_t = sigma.to(device=device, dtype=dtype)
        # Qwen's FlowMatchEulerDiscrete stores timesteps as sigma * 1000
        # and the pipeline passes (timestep / 1000) to the transformer.
        # Our sigmas are already in [0, 1], so we pass them directly —
        # equivalent to the pipeline's (timestep / 1000) when the scheduler
        # is in default config.
        timestep = sigma_t.expand(x.shape[0]).to(dtype=dtype)

        out = pipe.transformer(
            hidden_states=x,
            timestep=timestep,
            guidance=guidance_embed,
            encoder_hidden_states_mask=mask,
            encoder_hidden_states=embeds,
            img_shapes=img_shapes,
            attention_kwargs={},
            return_dict=False,
        )
        return out[0] if isinstance(out, tuple) else out

    def denoiser(x, sigma):
        v_cond = _call_transformer(x, sigma, prompt_embeds, prompt_embeds_mask)
        if not do_cfg:
            return v_cond

        v_neg = _call_transformer(x, sigma, neg_prompt_embeds, neg_prompt_embeds_mask)

        # Norm-preserving CFG rescale (Qwen-specific).
        # Work in fp32 so the norm-ratio division stays stable — bf16 loses
        # too much precision on the tail of the schedule when velocities
        # shrink and norm ratios approach 1.0.
        v_cond_f = v_cond.float()
        v_neg_f = v_neg.float()
        comb = v_neg_f + guidance_scale * (v_cond_f - v_neg_f)
        cond_norm = torch.norm(v_cond_f, dim=-1, keepdim=True)
        comb_norm = torch.norm(comb, dim=-1, keepdim=True)
        # Guard against division by zero on the last step when comb ≈ 0.
        comb_norm = torch.clamp(comb_norm, min=1e-8)
        rescaled = comb * (cond_norm / comb_norm)
        return rescaled.to(dtype=v_cond.dtype)

    # ── Step 6: Run the sampler ────────────────────────────────────────
    sampler_fn = get_sampler(sampler_name)
    last_step_reported = [0]

    def _progress_wrapper(step_fractional):
        step_int = int(step_fractional)
        if step_int > last_step_reported[0]:
            last_step_reported[0] = step_int
            if progress_cb is not None:
                progress_cb(step_int)

    with torch.no_grad():
        final_latents = sampler_fn(
            denoiser, latents, sigmas, _progress_wrapper,
            eta=eta, generator=generator,
        )

    return final_latents


# ── VAE tiling helper ────────────────────────────────────────────────────────
#
# Standard diffusers VAE decoders OOM above ~4 MP without tiling.  The
# upscale VAE path already handles this for Qwen when connected; this helper
# covers every other decode site (Flux, Flux.2, Qwen without upscale VAE).
#
# Threshold matches the empirical breakpoint noted in the project backlog.

_TILING_THRESHOLD_PIXELS = 4_000_000  # 4 MP


def _maybe_enable_vae_tiling(vae, height: int, width: int) -> bool:
    """Enable VAE tiling when output exceeds 4 MP. Returns True if tiling was enabled."""
    if height * width <= _TILING_THRESHOLD_PIXELS:
        return False
    try:
        vae.enable_tiling(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_stride_height=192,
            tile_sample_stride_width=192,
        )
        print(
            f"[EricDiffusion] VAE tiling enabled for decode "
            f"({width}×{height} = {width * height / 1e6:.1f} MP > 4 MP threshold)"
        )
        return True
    except Exception:
        return False


def decode_qwen_latents(
    pipe, latents: torch.Tensor, height: int, width: int,
) -> torch.Tensor:
    """Decode final packed Qwen latents via the pipeline's 5D VAE.

    Replicates the post-loop decode step from QwenImagePipeline.__call__:
      1. Unpack latents via pipe._unpack_latents → (B, C, 1, H_lat, W_lat)
      2. Apply latents_mean / latents_std denormalization (per-channel)
      3. Run VAE decode, slice off the trivial temporal dim
      4. Post-process to [0, 1] float tensor

    Returns a ComfyUI-format image tensor [1, H, W, 3].
    """
    tiling_enabled = _maybe_enable_vae_tiling(pipe.vae, height, width)
    try:
        with torch.no_grad():
            latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
            latents = latents.to(pipe.vae.dtype)

            # Qwen's VAE config stores per-channel mean/std (length = z_dim)
            # as 1D lists. Reshape to (1, z_dim, 1, 1, 1) so broadcast hits
            # the channel axis of the 5D latent tensor.
            z_dim = pipe.vae.config.z_dim
            latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
                1, z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                1, z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            # VAE decode returns 5D; [:, :, 0] drops the trivial temporal dim
            # to get (B, 3, H, W).
            image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = pipe.image_processor.postprocess(image, output_type="pt")
    finally:
        if tiling_enabled:
            pipe.vae.disable_tiling()
    # image is [1, 3, H, W] in [0, 1]; convert to ComfyUI format [1, H, W, 3]
    return image.permute(0, 2, 3, 1).contiguous().float().cpu()


# ═══════════════════════════════════════════════════════════════════════
#  Flux.2 support — separate path because of multiple interface changes
# ═══════════════════════════════════════════════════════════════════════
#
# Flux.2 differs from Flux/Chroma in several deep ways:
#
#   1. Text encoder is Mistral3, not CLIP+T5.  encode_prompt returns
#      (prompt_embeds, text_ids) — no pooled_prompt_embeds.
#   2. Transformer forward does NOT accept pooled_projections.
#   3. Latent packing is (B, H*W, C) instead of (B, H*W/4, C*4).
#      Positions are encoded via latent_ids (scatter-based unpack).
#   4. VAE decode uses batch-norm denormalization + _unpatchify_latents,
#      not the simple scaling_factor/shift_factor path.
#   5. Mu-shift uses an empirical formula that depends on both seq_len
#      AND num_steps, not the linear calculate_shift.
#   6. Always distilled guidance (guidance_embeds=True hardcoded).
#
# Rather than cluttering the Flux path with conditionals, Flux.2 gets
# its own generate_flux2() / decode_flux2_latents() / upscale_flux2_latents()
# functions that share only the sampler infrastructure.


def compute_flux2_shift_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical mu formula from Flux2Pipeline.

    Flux.2 computes mu by linearly interpolating between two regimes fit
    at num_steps=10 and num_steps=200, with seq_len as the primary
    variable.  Above seq_len=4300 the 200-step formula is used directly
    (high-resolution images saturate the shift).

    This is completely different from Flux.1's calculate_shift() which
    was linear in seq_len only.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    # Values at num_steps=200 (a2/b2) and num_steps=10 (a1/b1)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    # Linear interpolation in num_steps: at 10 gives m_10, at 200 gives m_200
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def build_flux2_sigmas(
    num_steps: int,
    image_seq_len: int,
    schedule: str = "linear",
) -> torch.Tensor:
    """Build a flow-matching sigma schedule with Flux.2's empirical mu shift.

    Same as build_flux_sigmas() but uses compute_flux2_shift_mu() instead
    of the linear Flux calculate_shift().  Supports all schedule names
    from ``_build_raw_sigmas``.
    """
    raw = _build_raw_sigmas(num_steps, schedule)
    mu = compute_flux2_shift_mu(image_seq_len, num_steps)
    shifted = _apply_mu_shift(torch.tensor(raw, dtype=torch.float32), mu)
    final = torch.cat([shifted, torch.zeros(1, dtype=shifted.dtype)])
    return final


def generate_flux2(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],  # unused (Flux2 always distilled)
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    sampler_name: str,
    sigma_schedule: str,
    generator: Optional[torch.Generator],
    max_sequence_length: int = 512,
    progress_cb: Optional[Callable] = None,
    initial_latents: Optional[torch.Tensor] = None,
    initial_latent_ids: Optional[torch.Tensor] = None,
    denoise: float = 1.0,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Manual denoising loop for Flux.2 pipelines.

    Returns (packed_latents, latent_ids) — BOTH are needed downstream
    because Flux.2's decode uses position IDs to scatter tokens back into
    spatial form.  The caller must pass the same latent_ids to
    decode_flux2_latents() or upscale_flux2_latents().

    For refinement stages, pass initial_latents + initial_latent_ids
    from the previous stage (already upscaled to target resolution via
    upscale_flux2_latents).
    """
    if not hasattr(pipe, "transformer") or not hasattr(pipe, "vae"):
        raise ValueError("Pipeline must have .transformer and .vae attributes")
    if not hasattr(pipe, "encode_prompt"):
        raise ValueError("Flux.2 pipeline does not expose encode_prompt()")

    device = getattr(pipe, "_execution_device", None) or next(
        pipe.transformer.parameters()
    ).device
    dtype = pipe.transformer.dtype

    # Defensively ensure text encoders are on the execution device.
    _ensure_text_encoders_on_device(pipe, device)

    # ── Step 1: Encode prompts (Mistral3, returns 2-tuple) ─────────────
    sig = inspect.signature(pipe.encode_prompt)
    enc_kwargs = dict(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
    )
    if "max_sequence_length" in sig.parameters:
        enc_kwargs["max_sequence_length"] = max_sequence_length

    encode_result = pipe.encode_prompt(**enc_kwargs)
    if len(encode_result) != 2:
        raise ValueError(
            f"Expected Flux.2 encode_prompt to return 2 values, got "
            f"{len(encode_result)}.  Pipeline may be Flux.1 rather than Flux.2."
        )
    prompt_embeds, text_ids = encode_result

    # Flux.2 always uses distilled guidance — classical CFG not supported
    # by the model.  negative_prompt is silently ignored.

    # ── Step 2: Prepare latents + latent_ids ───────────────────────────
    num_channels_latents = pipe.transformer.config.in_channels // 4
    prepare_kwargs = dict(
        batch_size=1,
        num_latents_channels=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    fresh_latents, fresh_latent_ids = pipe.prepare_latents(**prepare_kwargs)

    if initial_latents is None:
        latents = fresh_latents
    else:
        # Refinement path: caller provides upscaled latents.
        latents = initial_latents.to(device=device, dtype=fresh_latents.dtype)
        if latents.shape != fresh_latents.shape:
            raise ValueError(
                f"initial_latents shape {tuple(latents.shape)} does not match "
                f"expected shape for {height}x{width}: {tuple(fresh_latents.shape)}.\n"
                f"Did you run upscale_flux2_latents() to the new resolution?"
            )

    # ALWAYS use pipe.prepare_latents's fresh_latent_ids — the pipe knows
    # Flux.2's exact positional-id layout (4 columns with some internal
    # convention the pos_embed expects), and any manual reconstruction is
    # guaranteed to be wrong across diffusers updates.  The caller's
    # initial_latent_ids (if any) is ignored.
    latent_ids = fresh_latent_ids

    # ── Step 3: Build Flux.2 sigma schedule ────────────────────────────
    seq_len = latents.shape[1]
    sigmas = build_flux2_sigmas(
        num_steps=num_inference_steps,
        image_seq_len=seq_len,
        schedule=sigma_schedule,
    )
    sigmas = sigmas.to(device=device)

    # Refinement: truncate + inject noise at starting sigma
    if initial_latents is not None and denoise < 1.0:
        sigmas = truncate_sigmas_for_denoise(sigmas, denoise)
        starting_sigma = float(sigmas[0])
        latents = inject_flow_noise(latents, starting_sigma, generator=generator)
    elif initial_latents is not None and denoise >= 1.0:
        latents = torch.randn(
            latents.shape, generator=generator,
            device=latents.device, dtype=latents.dtype,
        )

    # ── Step 4: Build the denoiser callable ────────────────────────────
    # Flux.2 transformer signature (NO pooled_projections):
    #   hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids,
    #   guidance, joint_attention_kwargs, return_dict
    guidance_embed = torch.full(
        [1], guidance_scale, device=device, dtype=dtype,
    )  # Flux.2 always uses distilled guidance

    def _call_transformer(x, sigma):
        if not torch.is_tensor(sigma):
            sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
        else:
            sigma_t = sigma.to(device=device, dtype=dtype)
        timestep = sigma_t.expand(x.shape[0]).to(dtype=dtype)

        out = pipe.transformer(
            hidden_states=x,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=guidance_embed,
            return_dict=False,
        )
        return out[0] if isinstance(out, tuple) else out

    def denoiser(x, sigma):
        # Single forward pass — Flux.2 never does classical CFG
        return _call_transformer(x, sigma)

    # ── Step 5: Run the sampler ────────────────────────────────────────
    sampler_fn = get_sampler(sampler_name)
    last_step_reported = [0]

    def _progress_wrapper(step_fractional):
        step_int = int(step_fractional)
        if step_int > last_step_reported[0]:
            last_step_reported[0] = step_int
            if progress_cb is not None:
                progress_cb(step_int)

    with torch.no_grad():
        final_latents = sampler_fn(denoiser, latents, sigmas, _progress_wrapper, eta=eta, generator=generator)

    return final_latents, latent_ids


def decode_flux2_latents(
    pipe, latents: torch.Tensor, latent_ids: torch.Tensor,
) -> torch.Tensor:
    """Decode Flux.2 packed latents using the Flux.2-specific path.

    Unlike Flux.1's simple unpack + scale/shift, Flux.2 needs:
      1. _unpack_latents_with_ids     — scatter tokens via position ids
      2. Batch-norm denormalization   — apply vae.bn running_mean/var
      3. _unpatchify_latents          — reverse the 2x2 patching
      4. vae.decode                   — produce pixel-space image

    The result is a ComfyUI image tensor [1, H, W, 3] in [0, 1].
    """
    with torch.no_grad():
        # Unpack with position ids: (B, seq, C*4) → (B, C*4, H_patched, W_patched)
        latents_spatial = pipe._unpack_latents_with_ids(latents, latent_ids)

        # Batch-norm denormalization
        vae = pipe.vae
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents_spatial.device, latents_spatial.dtype,
        )
        bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents_spatial.device, latents_spatial.dtype)
        latents_spatial = latents_spatial * bn_std + bn_mean

        # Unpatchify: (B, C*4, H_patched, W_patched) → (B, C, H_lat, W_lat)
        latents_unpatched = pipe._unpatchify_latents(latents_spatial)

        # Derive pixel dimensions from latent spatial shape for tiling check.
        # vae_scale_factor is 8 for Flux.2; latents_unpatched is (B, C, H_lat, W_lat).
        vae_scale = getattr(pipe, "vae_scale_factor", 8)
        _, _, h_lat, w_lat = latents_unpatched.shape
        pix_h, pix_w = h_lat * vae_scale, w_lat * vae_scale

        tiling_enabled = _maybe_enable_vae_tiling(vae, pix_h, pix_w)
        try:
            # VAE decode → pixel image
            image = vae.decode(latents_unpatched, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pt")
        finally:
            if tiling_enabled:
                vae.disable_tiling()

    # (1, 3, H, W) → (1, H, W, 3)
    return image.permute(0, 2, 3, 1).contiguous().float().cpu()


def upscale_flux2_latents(
    packed_latents: torch.Tensor,
    latent_ids: torch.Tensor,
    src_h: int, src_w: int,
    dst_h: int, dst_w: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Upscale Flux.2 packed latents between stages.

    Returns just the upscaled packed latents.  Latent positional IDs for
    the target resolution MUST be obtained separately from the pipeline
    (``pipe.prepare_latents(latents=None)``) — Flux.2's id layout is more
    complex than Flux.1's and the pipe is the authoritative source.

    Rationale: Flux.2's latent_ids have more than 3 columns (the pos_embed
    module iterates over 4 positional dimensions) with a specific internal
    convention.  Manually constructing row-major `[0, h, w]` ids like I
    tried initially gives an IndexError deep in the transformer's rotary
    embedding code.  The safest approach is to let the pipeline generate
    ids at the target resolution and trust them.

    This function does:
      1. Unpack packed_latents to patched spatial form using the source
         latent_ids (which we know are valid since they came from the pipe)
      2. Bislerp upscale in spatial form
      3. Re-pack as (B, seq, C) in row-major order

    Re-packing in row-major order is OK because the pipe's ids at the
    target resolution are also in row-major order — they will index the
    re-packed tokens correctly.
    """
    import comfy.utils as comfy_utils

    B, src_seq, C = packed_latents.shape
    dst_h_patched = 2 * (int(dst_h) // (vae_scale_factor * 2)) // 2
    dst_w_patched = 2 * (int(dst_w) // (vae_scale_factor * 2)) // 2

    # Unpack each batch element using the source ids (scatter-by-position)
    spatial_list = []
    for b in range(B):
        data = packed_latents[b]          # (src_seq, C)
        pos = latent_ids if latent_ids.dim() == 2 else latent_ids[b]
        # Flux.2 ids have more than 3 columns; we only need columns 1-2
        # for h/w positions (column 0 is batch/frame, higher columns are
        # additional positional dims we don't need to read).
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h = int(h_ids.max().item()) + 1
        w = int(w_ids.max().item()) + 1
        flat = h_ids * w + w_ids
        out = torch.zeros((h * w, C), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat.unsqueeze(1).expand(-1, C), data)
        out = out.view(h, w, C).permute(2, 0, 1)  # (C, H, W)
        spatial_list.append(out)
    spatial = torch.stack(spatial_list, dim=0)  # (B, C, src_H, src_W)

    # Bislerp upscale to target patched spatial dims
    upscaled = comfy_utils.bislerp(spatial, dst_w_patched, dst_h_patched)

    # Re-pack in row-major order: (B, C, H, W) → (B, H*W, C)
    B_up, C_up, H_up, W_up = upscaled.shape
    new_packed = upscaled.reshape(B_up, C_up, H_up * W_up).permute(0, 2, 1)
    return new_packed


def decode_flux_latents(pipe, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Decode final packed latents via the pipeline's VAE.

    Replicates the post-loop decode step from FluxPipeline.__call__:
      1. Unpack latents via pipe._unpack_latents
      2. Apply shift + scaling factor denormalization
      3. Run VAE decode
      4. Post-process to [0, 1] float tensor

    Returns a ComfyUI-format image tensor [1, H, W, 3].
    """
    tiling_enabled = _maybe_enable_vae_tiling(pipe.vae, height, width)
    try:
        with torch.no_grad():
            latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            image = pipe.vae.decode(latents, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pt")
    finally:
        if tiling_enabled:
            pipe.vae.disable_tiling()
    # image is [1, 3, H, W] in [0, 1]; convert to ComfyUI format [1, H, W, 3]
    return image.permute(0, 2, 3, 1).contiguous().float().cpu()


# ═══════════════════════════════════════════════════════════════════════
#  Image-to-image encode helper — used by the advanced generate/multistage
#  nodes to turn a user-supplied reference image into packed latents that
#  can be passed as ``initial_latents`` to generate_flux / generate_chroma
#  / generate_qwen.  Flux.2 is deferred (no stock img2img pipeline to
#  mirror, would require hand-rolled patchify + BN inverse encode).
# ═══════════════════════════════════════════════════════════════════════
#
# All three supported families land on the SAME packed shape:
#     (B, (H_lat/2) * (W_lat/2), num_channels_latents * 4)
# which is what their respective generate_* functions expect as
# initial_latents.  The difference is in the encode + normalization
# upstream of the pack step:
#
#   Flux / Chroma:
#       4D VAE → raw [B, C, H_lat, W_lat]
#       normalize: (raw - shift_factor) * scaling_factor
#
#   Qwen:
#       5D VAE → raw [B, z_dim, 1, H_lat, W_lat]
#       normalize: (raw - latents_mean) / latents_std  (per-channel)
#       transpose(1, 2) to [B, 1, z_dim, H_lat, W_lat]
#
# After normalization both families use the same _pack_latents math —
# view(B, C, H/2, 2, W/2, 2) → permute(0,2,4,1,3,5) → reshape(B, seq, C*4)
# — which is byte-identical to the upscale_flux_latents packing.


def _comfy_image_to_vae_input(image: torch.Tensor) -> torch.Tensor:
    """Convert a ComfyUI IMAGE tensor to VAE input format.

    ComfyUI format: (B, H, W, 3) in [0, 1], float32, on CPU.
    VAE input:      (B, 3, H, W) in [-1, 1].

    Does NOT add a temporal dim — callers that need 5D (Qwen) must
    unsqueeze(2) themselves after this call.
    """
    if image.dim() != 4 or image.shape[-1] != 3:
        raise ValueError(
            f"Expected ComfyUI IMAGE tensor (B, H, W, 3), got shape {tuple(image.shape)}"
        )
    # Permute to channel-first, scale [0,1] → [-1,1]
    x = image.permute(0, 3, 1, 2).contiguous()
    x = x * 2.0 - 1.0
    return x


def _resize_image_if_needed(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Resize a ComfyUI IMAGE tensor to (target_h, target_w) if it doesn't
    already match.  Uses bicubic interpolation with antialiasing.

    Input/output shape: (B, H, W, 3) in [0, 1].

    A non-resize pass-through returns the original tensor unchanged so
    the no-op case is free.
    """
    _, h, w, _ = image.shape
    if h == target_h and w == target_w:
        return image
    # Permute to NCHW for F.interpolate, then back.
    x = image.permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(
        x, size=(target_h, target_w),
        mode="bicubic", align_corners=False, antialias=True,
    )
    x = x.clamp(0.0, 1.0)  # bicubic can overshoot
    x = x.permute(0, 2, 3, 1).contiguous()
    print(
        f"[EricDiffusion] Reference image resized {w}×{h} → "
        f"{target_w}×{target_h} (bicubic) to match target dimensions"
    )
    return x


def encode_image_to_packed_latents(
    pipe,
    image: torch.Tensor,
    width: int,
    height: int,
    model_family: str,
) -> torch.Tensor:
    """Encode a reference image to packed latents matching what the family's
    generate_* function expects as ``initial_latents``.

    Args:
        pipe         : GEN_PIPELINE (any supported family).
        image        : ComfyUI IMAGE tensor (B, H, W, 3) in [0, 1].  If
                       its resolution doesn't match (width, height), it
                       will be bicubic-resized with a log note.
        width, height: Target pixel dimensions (after aspect/MP resolution).
                       Must be divisible by vae_scale_factor * 2 (16 for
                       vae_scale_factor=8) or they'll be silently floored
                       to the nearest valid size, matching what the stock
                       prepare_latents does.
        model_family : Short family string from the loader ("flux",
                       "chroma", "qwen-image", etc.).

    Returns:
        Packed latents tensor of shape
            (B, (H_lat/2) * (W_lat/2), num_channels_latents * 4)
        on the pipeline's execution device, in the transformer's dtype.

    Raises:
        NotImplementedError: for Flux.2 (no stock img2img pipeline to
                             mirror — deferred to a later slice).
        ValueError: for unknown families.
    """
    family = (model_family or "").lower()

    flux_families = (
        "flux", "chroma", "fluxpipeline", "chromapipeline",
    )
    qwen_families = (
        "qwen-image", "qwenimage", "qwenimagepipeline",
    )
    flux2_families = (
        "flux2", "flux2pipeline",
    )

    if family in flux2_families:
        raise NotImplementedError(
            "Flux.2 i2i is not yet supported in the manual loop — stock "
            "diffusers has no Flux2Img2ImgPipeline to mirror and the "
            "encode direction requires a hand-rolled patchify + "
            "batch-norm inverse.  Deferred to a later slice."
        )

    if family not in flux_families + qwen_families:
        raise ValueError(
            f"encode_image_to_packed_latents: unknown model_family "
            f"{model_family!r}.  Supported: Flux, Chroma, Qwen-Image."
        )

    # ── Resize to target dimensions if needed ──────────────────────────
    # Apply the same (height, width) flooring that prepare_latents uses so
    # the encoded latent shape matches what the sampler expects.
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    target_h = 2 * (int(height) // (vae_scale * 2)) * vae_scale
    target_w = 2 * (int(width) // (vae_scale * 2)) * vae_scale
    image = _resize_image_if_needed(image, target_h, target_w)

    # ── Determine device + dtype from the pipeline's transformer ───────
    # Use next(parameters()).dtype rather than .dtype property so the
    # helper works against both real diffusers modules (which expose
    # .dtype via ModelMixin) and test mocks that only implement
    # .parameters().
    device = getattr(pipe, "_execution_device", None) or next(
        pipe.transformer.parameters()
    ).device
    transformer_dtype = next(pipe.transformer.parameters()).dtype
    vae_dtype = next(pipe.vae.parameters()).dtype

    # ── Convert ComfyUI image → VAE input ──────────────────────────────
    vae_input = _comfy_image_to_vae_input(image).to(device=device, dtype=vae_dtype)

    # ── Family-specific encode + normalize + pack ──────────────────────
    with torch.no_grad():
        if family in qwen_families:
            # Qwen: 5D VAE, per-channel latents_mean/std, transpose before pack.
            vae_input_5d = vae_input.unsqueeze(2)  # (B, 3, 1, H, W)
            raw = pipe.vae.encode(vae_input_5d).latent_dist.mode()
            # raw shape: (B, z_dim, 1, H_lat, W_lat)

            z_dim = pipe.vae.config.z_dim
            latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
                1, z_dim, 1, 1, 1
            ).to(device=device, dtype=raw.dtype)
            latents_std_inv = (1.0 / torch.tensor(pipe.vae.config.latents_std)).view(
                1, z_dim, 1, 1, 1
            ).to(device=device, dtype=raw.dtype)
            normed = (raw - latents_mean) * latents_std_inv
            # normed shape: (B, z_dim, 1, H_lat, W_lat)

            # Transpose channel and temporal dims to match what Qwen's
            # _pack_latents expects: (B, 1, z_dim, H_lat, W_lat).
            # .contiguous() is required because transpose() leaves the
            # tensor non-contiguous and .view() would fail.
            normed = normed.transpose(1, 2).contiguous()
            # normed shape: (B, 1, z_dim, H_lat, W_lat)

            B = normed.shape[0]
            H_lat = normed.shape[3]
            W_lat = normed.shape[4]
            # Pack: view → permute → reshape.  Total elements invariant
            # under the trivial temporal dim so the view() works whether
            # we pass 5D or squeeze(2) first.
            packed = normed.view(B, z_dim, H_lat // 2, 2, W_lat // 2, 2)
            packed = packed.permute(0, 2, 4, 1, 3, 5)
            packed = packed.reshape(B, (H_lat // 2) * (W_lat // 2), z_dim * 4)

        else:
            # Flux / Chroma: 4D VAE, global scaling_factor + shift_factor.
            raw = pipe.vae.encode(vae_input).latent_dist.mode()
            # raw shape: (B, C, H_lat, W_lat)

            scaling_factor = pipe.vae.config.scaling_factor
            shift_factor = pipe.vae.config.shift_factor
            normed = (raw - shift_factor) * scaling_factor

            B, C, H_lat, W_lat = normed.shape
            packed = normed.view(B, C, H_lat // 2, 2, W_lat // 2, 2)
            packed = packed.permute(0, 2, 4, 1, 3, 5)
            packed = packed.reshape(B, (H_lat // 2) * (W_lat // 2), C * 4)

    # Return in the transformer's dtype so downstream operations don't
    # need to cast.  generate_flux/chroma/qwen all cast to their own
    # transformer_dtype internally anyway, but doing it here saves a copy.
    return packed.to(dtype=transformer_dtype)


# ═══════════════════════════════════════════════════════════════════════
#  Qwen-Image-Edit support — multi-reference compositional editing
# ═══════════════════════════════════════════════════════════════════════
#
# Edit is a fundamentally different beast from Generate, not a variation
# of it.  Key differences from generate_qwen:
#
#   1. Text encoder is Qwen2.5-VL run through a VL **processor** (not a
#      tokenizer).  The processor accepts BOTH text tokens AND raw image
#      pixels, fuses them into a single prompt embedding at the token
#      level.  Reference images are "woven into" the text embedding
#      through "Picture 1: <|vision_start|>..." tokens, not merely
#      concatenated as additional conditioning.
#
#   2. Reference images exist on TWO independent paths:
#        a. Semantic (VL) path — the image feeds the text encoder for
#           compositional prompt understanding ("the outfit from Picture
#           2").  Resized to CONDITION_IMAGE_SIZE = 384×384 area.
#        b. Pixel latent path — the image is VAE-encoded and its packed
#           latents are concatenated into the transformer's
#           ``hidden_states`` along the sequence axis.  Resized to
#           VAE_IMAGE_SIZE = 1024×1024 area.
#      The two paths are independent: a reference can be VL-only (used
#      for semantic context but not as pixel conditioning), ref-only
#      (used as pixel anchor but not mentioned to the text encoder),
#      or both.  Per-image vl/ref flags select the path.
#
#   3. Transformer input is [noise_latents, ref1_latents, ref2_latents,
#      ...] concatenated along dim=1.  All image latents share the same
#      packing format and the same channel count.  img_shapes becomes
#      a multi-entry list, one tuple per "image" in the concatenated
#      sequence (the noise being generated counts as the first entry).
#
#   4. After the transformer forward pass, slice noise_pred back to the
#      noise portion before applying CFG or the sampler step:
#        noise_pred = noise_pred[:, : noise_latents.size(1)]
#      The transformer returns velocity for the FULL concatenated
#      sequence including the reference portion; we only care about
#      the noise portion because the ref latents are constant context.
#
#   5. Output dimensions default to calculate_dimensions(1024*1024, r)
#      where r is the aspect ratio of the LAST reference image — Qwen
#      Edit preserves the "shape" of the last ref by convention.  Users
#      can override with explicit width/height.
#
#   6. CFG: same norm-preserving rescale as generate_qwen.  Two forward
#      passes per step when negative_prompt is set, fp32 rescale.
#
#   7. Scheduler + sigmas: same flow-matching mu-shift as base Qwen.
#      build_flux_sigmas works unchanged.
#
#   8. Decode: same decode_qwen_latents — the VAE is unchanged.  But
#      only the noise portion of the latents needs decoding; the ref
#      portion was context, not part of the output.
#
# The Qwen Edit CONDITION_IMAGE_SIZE and VAE_IMAGE_SIZE constants match
# what the stock QwenImageEditPlusPipeline uses (384² and 1024²).
# Different values would change VL token counts and pixel latent
# resolution, which would change model behavior in ways we don't want
# without explicit opt-in.


_QWEN_EDIT_CONDITION_IMAGE_SIZE = 384 * 384
_QWEN_EDIT_VAE_IMAGE_SIZE = 1024 * 1024


def _calculate_qwen_edit_dimensions(
    target_area: int, aspect_ratio: float,
) -> Tuple[int, int]:
    """Derive (width, height) preserving aspect ratio within target area,
    aligned to multiples of 32.  Matches stock QwenImageEditPlusPipeline's
    calculate_dimensions exactly.
    """
    width = math.sqrt(target_area * aspect_ratio)
    height = width / aspect_ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def _resize_ref_for_qwen_edit(
    image: torch.Tensor, target_area: int,
) -> torch.Tensor:
    """Resize a ComfyUI reference image to the given target area,
    preserving its aspect ratio.  Returns a new ComfyUI-format tensor
    (B, H, W, 3) in [0, 1].

    Used to produce BOTH the condition_image (384² area) and
    vae_image (1024² area) versions of each reference, matching the
    stock QwenImageEditPlusPipeline preprocessing.
    """
    _, h, w, _ = image.shape
    aspect = w / h
    target_w, target_h = _calculate_qwen_edit_dimensions(target_area, aspect)
    if target_h == h and target_w == w:
        return image
    x = image.permute(0, 3, 1, 2)
    x = torch.nn.functional.interpolate(
        x, size=(target_h, target_w),
        mode="bicubic", align_corners=False, antialias=True,
    )
    x = x.clamp(0.0, 1.0)
    return x.permute(0, 2, 3, 1).contiguous()


def _encode_qwen_ref_image_to_packed_latents(
    pipe, image: torch.Tensor, device, vae_dtype,
) -> Tuple[torch.Tensor, int, int]:
    """Encode a single Qwen Edit reference image through the pipeline's
    5D VAE and pack it to the format the transformer expects as part
    of ``hidden_states``.

    Args:
        image   : ComfyUI IMAGE tensor (B, H, W, 3) in [0, 1].  Already
                  resized to a vae-friendly size (VAE_IMAGE_SIZE area,
                  aspect-preserving, 32-aligned).
        device  : torch device for the returned packed latents.  The
                  VAE input is ALWAYS moved to the VAE's actual current
                  device before ``pipe.vae.encode`` — this defensive
                  behavior avoids device mismatch crashes when upstream
                  helpers (e.g. upscale_between_stages) have moved
                  pipe.vae to a different GPU than the caller expected.
                  The result is then moved to the passed-in ``device``
                  so subsequent concat operations with noise latents
                  work correctly.
        vae_dtype : torch dtype for the VAE input.

    Returns:
        (packed_latents, H_lat, W_lat) where packed_latents is
        (B, (H_lat/2) * (W_lat/2), z_dim * 4) ready to be concatenated
        along dim=1 with the noise latents on ``device``.
    """
    # ComfyUI (B,H,W,3) in [0,1] → VAE input (B,3,H,W) in [-1,1]
    vae_input = image.permute(0, 3, 1, 2).contiguous()
    vae_input = vae_input * 2.0 - 1.0

    # Defensive: move the input to the VAE's ACTUAL device at call time,
    # not the caller's assumed device.  Guards against upstream helpers
    # leaving pipe.vae on a different GPU than the caller thinks.
    try:
        vae_device = next(pipe.vae.parameters()).device
    except (StopIteration, AttributeError):
        vae_device = device
    vae_input = vae_input.to(device=vae_device, dtype=vae_dtype)
    # Qwen VAE is 5D — add trivial temporal dim
    vae_input_5d = vae_input.unsqueeze(2)

    raw = pipe.vae.encode(vae_input_5d).latent_dist.mode()
    # raw shape: (B, z_dim, 1, H_lat, W_lat) on vae_device

    # Normalization constants go on vae_device to match raw — mixing
    # devices here would crash if the caller's device != vae's device.
    z_dim = pipe.vae.config.z_dim
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
        1, z_dim, 1, 1, 1
    ).to(device=vae_device, dtype=raw.dtype)
    latents_std = torch.tensor(pipe.vae.config.latents_std).view(
        1, z_dim, 1, 1, 1
    ).to(device=vae_device, dtype=raw.dtype)
    # Qwen edit pipeline uses (raw - mean) / std directly (not *1/std)
    normed = (raw - latents_mean) / latents_std
    # normed shape: (B, z_dim, 1, H_lat, W_lat) on vae_device

    # Transpose to (B, 1, z_dim, H_lat, W_lat) so _pack_latents math lines up
    normed = normed.transpose(1, 2).contiguous()

    B = normed.shape[0]
    H_lat = normed.shape[3]
    W_lat = normed.shape[4]
    # Pack: view → permute → reshape (same math as Flux packing)
    packed = normed.view(B, z_dim, H_lat // 2, 2, W_lat // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    packed = packed.reshape(B, (H_lat // 2) * (W_lat // 2), z_dim * 4)

    # Move the packed result to the caller's expected device — this is
    # critical for downstream torch.cat([noise, refs], dim=1) calls
    # where noise is on the transformer's device.  If vae_device and
    # device differ (e.g. due to offload or cross-GPU state), this
    # move reconciles them here instead of crashing in the concat.
    if vae_device != device:
        packed = packed.to(device=device)

    return packed, H_lat, W_lat


def generate_qwen_edit(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    reference_images: list,
    vl_flags: Optional[list] = None,
    ref_flags: Optional[list] = None,
    output_width: Optional[int] = None,
    output_height: Optional[int] = None,
    num_inference_steps: int = 25,
    guidance_scale: float = 4.0,
    sampler_name: str = "flow_heun",
    sigma_schedule: str = "linear",
    generator: Optional[torch.Generator] = None,
    max_sequence_length: int = 1024,
    progress_cb: Optional[Callable] = None,
    eta: float = 0.0,
    initial_latents: Optional[torch.Tensor] = None,
    denoise: float = 1.0,
) -> Tuple[torch.Tensor, int, int]:
    """Manual denoising loop for Qwen-Image-Edit with 1-N reference images.

    Each reference image can be fed through two independent paths:
      - **VL path** (semantic): image goes to the text encoder as part
        of the fused prompt embedding.  Enables compositional prompts
        like "the outfit from Picture 2".
      - **Ref path** (pixel): image is VAE-encoded and its packed latents
        concatenate into the transformer's hidden_states so the model
        can attend to reference pixels directly.

    Both paths default to ON for every reference.  Per-image ``vl_flags``
    and ``ref_flags`` lists (one bool per reference) let the caller
    disable either path per slot for advanced use cases.

    Args:
        reference_images: Non-empty list of ComfyUI IMAGE tensors, each
                          (B, H, W, 3) in [0, 1].  Order is preserved —
                          the Nth entry becomes "Picture N" in the VL
                          processor's prompt template.
        vl_flags:         Optional list of bools (length == len(reference_images)).
                          True = this image feeds the text encoder.
                          Default: all True.
        ref_flags:        Optional list of bools (length == len(reference_images)).
                          True = this image's latents concatenate into
                          transformer hidden_states.  Default: all True.
        output_width, output_height: Optional override for output dimensions.
                          If unset, derived from the LAST reference image's
                          aspect ratio at ~1MP area (stock pipeline convention).
        initial_latents:  Optional pre-existing packed noise latents to use
                          as the starting state instead of fresh noise.  Used
                          by multistage refinement: caller passes upscaled
                          latents from the previous stage and sets
                          ``denoise < 1.0``.  When provided, must match the
                          shape that ``prepare_latents`` would produce for
                          the resolved (output_width, output_height).
        denoise:          Fraction of the sigma schedule to run.  ``1.0``
                          runs the full schedule from pure noise.  ``0.85``
                          runs the last 85% of the schedule starting from
                          a partially-noised state (requires
                          ``initial_latents``).  Ignored when
                          ``initial_latents is None``.

    Returns:
        (packed_noise_latents, output_height, output_width) — the noise
        portion of the final latents (reference portion is handled
        internally and not returned), plus the resolved output dimensions
        so the caller can pass them to decode_qwen_latents.

    Raises:
        ValueError: no reference images, flag list length mismatch,
                    all refs have both flags False (effectively no-op),
                    or initial_latents shape doesn't match target.
    """
    if not hasattr(pipe, "transformer") or not hasattr(pipe, "vae"):
        raise ValueError("Pipeline must have .transformer and .vae attributes")
    if not hasattr(pipe, "encode_prompt"):
        raise ValueError("Qwen Edit pipeline does not expose encode_prompt()")
    if not hasattr(pipe, "processor"):
        raise ValueError(
            "Qwen Edit pipeline missing VL processor — this function "
            "requires a QwenImageEditPlusPipeline (or subclass) with "
            "a Qwen2VLProcessor attached."
        )

    # ── Normalize reference_images and flags ──────────────────────────
    if reference_images is None:
        raise ValueError(
            "generate_qwen_edit requires at least one reference_image.  "
            "For pure text-to-image use generate_qwen instead."
        )
    # Accept a single tensor for backward-compat and ergonomic calls —
    # wrap it in a 1-list.
    if isinstance(reference_images, torch.Tensor):
        reference_images = [reference_images]
    if len(reference_images) == 0:
        raise ValueError(
            "generate_qwen_edit requires at least one reference image."
        )
    n_refs = len(reference_images)

    if vl_flags is None:
        vl_flags = [True] * n_refs
    if ref_flags is None:
        ref_flags = [True] * n_refs
    if len(vl_flags) != n_refs:
        raise ValueError(
            f"vl_flags length ({len(vl_flags)}) must match number of "
            f"reference images ({n_refs})"
        )
    if len(ref_flags) != n_refs:
        raise ValueError(
            f"ref_flags length ({len(ref_flags)}) must match number of "
            f"reference images ({n_refs})"
        )
    if not any(vl_flags) and not any(ref_flags):
        raise ValueError(
            "All reference images have both VL and Ref flags set False — "
            "nothing would be conditioned on them.  Enable at least one "
            "flag on at least one image, or remove the references "
            "entirely and use generate_qwen instead."
        )

    # Prefer the transformer's actual parameter device over
    # ``pipe._execution_device`` — the latter can return an unexpected
    # device in cross-GPU pipeline states (e.g. after an upscale VAE
    # helper has moved pipe.vae onto a different GPU) or may return
    # CPU under sequential_offload even while the transformer is
    # actively on CUDA.  The transformer's .parameters()[0].device
    # is the single source of truth for where sampling must happen.
    try:
        device = next(pipe.transformer.parameters()).device
    except StopIteration:
        device = getattr(pipe, "_execution_device", torch.device("cpu"))
    transformer_dtype = next(pipe.transformer.parameters()).dtype
    vae_dtype = next(pipe.vae.parameters()).dtype

    # Defensively ensure text encoders are on the execution device.
    _ensure_text_encoders_on_device(pipe, device)

    # ── Step 1: Derive output dimensions from LAST reference ──────────
    # Matches stock QwenImageEditPlusPipeline behavior — output preserves
    # the aspect ratio of image[-1] at ~1MP, 32-aligned.  Flags don't
    # affect dimension derivation.  Caller can override with explicit
    # output_width/output_height.
    _, last_h, last_w, _ = reference_images[-1].shape
    last_aspect = last_w / last_h
    if output_width is None or output_height is None:
        output_width, output_height = _calculate_qwen_edit_dimensions(
            _QWEN_EDIT_VAE_IMAGE_SIZE, last_aspect,
        )
    # Force alignment to vae_scale_factor * 2 regardless (transformer
    # requires multiples of 16 for vae_scale=8)
    vae_scale = getattr(pipe, "vae_scale_factor", 8)
    multiple_of = vae_scale * 2
    output_width = (output_width // multiple_of) * multiple_of
    output_height = (output_height // multiple_of) * multiple_of

    n_vl_active = sum(1 for f in vl_flags if f)
    n_ref_active = sum(1 for f in ref_flags if f)
    print(
        f"[EricDiffusion] Qwen Edit: {n_refs} reference(s) "
        f"(VL={n_vl_active}, Ref={n_ref_active}), output "
        f"{output_width}×{output_height} "
        f"({output_width * output_height / 1e6:.2f} MP)"
    )

    # ── Step 2: Encode prompt with VL-flagged references ─────────────
    # Build a PIL list from VL-flagged images only (resized to condition
    # area 384²).  Empty list → pass image=None to encode_prompt, which
    # falls back to text-only encoding.  Non-empty → images go through
    # the VL processor fused with the prompt.
    vl_pil_list = []
    for i, img in enumerate(reference_images):
        if not vl_flags[i]:
            continue
        cond_img = _resize_ref_for_qwen_edit(
            img, _QWEN_EDIT_CONDITION_IMAGE_SIZE,
        )
        vl_pil_list.extend(_qwen_edit_comfy_to_pil_list(cond_img))

    # CRITICAL: don't pass device= to encode_prompt.  Letting it default
    # to self._execution_device means model_inputs.attention_mask lands
    # on the same device that _fix_text_encoder_device's post-hook
    # targets, so the downstream _extract_masked_hidden call (which
    # does hidden_states[bool_mask]) doesn't crash with cross-device
    # indexing.
    #
    # If we pass device=transformer_device here, model_inputs goes onto
    # transformer_device (e.g. cuda:1), the text encoder pre-hook moves
    # a COPY onto text_encoder_device (e.g. cuda:0), outputs land on
    # _execution_device via the post-hook (e.g. cuda:0), but the
    # caller's model_inputs.attention_mask reference is still on
    # cuda:1 → cross-device indexing crash.
    #
    # After encode_prompt returns we manually move the embeddings to
    # the transformer's device for the sampler loop.
    enc_kwargs = dict(
        prompt=prompt,
        image=vl_pil_list if vl_pil_list else None,
        num_images_per_prompt=1,
    )
    sig = inspect.signature(pipe.encode_prompt)
    if "max_sequence_length" in sig.parameters:
        enc_kwargs["max_sequence_length"] = max_sequence_length

    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(**enc_kwargs)

    # Move embeddings to the transformer's device for the sampler loop.
    # They'll be concatenated with noise latents which live on the
    # transformer's device.
    prompt_embeds = prompt_embeds.to(device=device)
    if prompt_embeds_mask is not None:
        prompt_embeds_mask = prompt_embeds_mask.to(device=device)

    do_cfg = guidance_scale > 1.0 and bool(negative_prompt)
    neg_prompt_embeds = None
    neg_prompt_embeds_mask = None
    if do_cfg:
        neg_enc_kwargs = {**enc_kwargs, "prompt": negative_prompt}
        neg_prompt_embeds, neg_prompt_embeds_mask = pipe.encode_prompt(
            **neg_enc_kwargs
        )
        neg_prompt_embeds = neg_prompt_embeds.to(device=device)
        if neg_prompt_embeds_mask is not None:
            neg_prompt_embeds_mask = neg_prompt_embeds_mask.to(device=device)

    # ── Step 3: Encode Ref-flagged references to packed latents ───────
    # Each ref-flagged image goes through the Qwen 5D VAE independently,
    # gets packed, and is appended to the list.  Latent-space H/W per
    # ref are tracked for img_shapes.
    ref_packed_list = []
    ref_shape_entries = []  # one (1, h//2, w//2) tuple per ref-flagged image
    with torch.no_grad():
        for i, img in enumerate(reference_images):
            if not ref_flags[i]:
                continue
            vae_img = _resize_ref_for_qwen_edit(
                img, _QWEN_EDIT_VAE_IMAGE_SIZE,
            )
            packed, h_lat, w_lat = _encode_qwen_ref_image_to_packed_latents(
                pipe, vae_img, device, vae_dtype,
            )
            ref_packed_list.append(packed.to(dtype=transformer_dtype))
            ref_shape_entries.append((1, h_lat // 2, w_lat // 2))

    # Concatenate all ref-flagged packed latents along sequence axis.
    # If no ref-flagged images (all VL-only), concatenated_refs is None
    # and the denoiser skips the concat step.
    if ref_packed_list:
        concatenated_refs = torch.cat(ref_packed_list, dim=1)
    else:
        concatenated_refs = None

    # ── Step 4: Prepare noise latents for the output ──────────────────
    # Always call prepare_latents with latents=None first to get a
    # reference shape for this resolution — we use it to shape-check
    # any caller-supplied initial_latents before swapping them in.
    #
    # IMPORTANT: call prepare_latents with POSITIONAL args.  The stock
    # ``QwenImageEditPlusPipeline.prepare_latents`` uses kwarg name
    # ``num_channels_latents`` but the project's custom subclass
    # ``QwenEditPipeline`` (pipelines/pipeline_qwen_edit.py) overrides
    # the method with kwarg name ``num_channels`` — kwarg calls fail on
    # whichever class isn't the one you wrote the kwarg name for.
    # Positional order is identical on both (images, batch_size, chans,
    # height, width, dtype, device, generator, latents=None) so this
    # call works on either.
    num_channels_latents = pipe.transformer.config.in_channels // 4
    fresh_latents, _image_latents_unused = pipe.prepare_latents(
        None,                   # images — we handle refs ourselves
        1,                      # batch_size (Qwen Edit is hardcoded to 1 anyway)
        num_channels_latents,   # num_channels / num_channels_latents
        output_height,
        output_width,
        prompt_embeds.dtype,
        device,
        generator,
        None,                   # latents — None = generate fresh noise
    )
    # fresh_latents: (1, noise_seq, z_dim * 4) where
    #   noise_seq = (H_out/16) * (W_out/16)

    if initial_latents is None:
        noise_latents = fresh_latents
    else:
        # Refinement path: use caller's latents as the starting state
        # (typically upscaled from a previous stage's output).  Must
        # match the shape prepare_latents would have produced.
        noise_latents = initial_latents.to(
            device=device, dtype=fresh_latents.dtype,
        )
        if noise_latents.shape != fresh_latents.shape:
            raise ValueError(
                f"initial_latents shape {tuple(noise_latents.shape)} "
                f"does not match expected shape for "
                f"{output_height}x{output_width}: "
                f"{tuple(fresh_latents.shape)}.\n"
                f"Did you forget to upscale the latents to the new "
                f"resolution between stages?"
            )

    # ── Step 5: Build img_shapes for transformer positional encoding ──
    # Entry per "image" in the concatenated sequence:
    #   [noise_entry, ref_entry_1, ref_entry_2, ...]
    # Ref entries are only included for images with ref_flag=True —
    # ref_shape_entries was built to exactly match the ref_packed_list
    # order so the sequence alignment is guaranteed.
    noise_h_over_16 = output_height // vae_scale // 2
    noise_w_over_16 = output_width // vae_scale // 2
    img_shapes_inner = [(1, noise_h_over_16, noise_w_over_16)]
    img_shapes_inner.extend(ref_shape_entries)
    img_shapes = [img_shapes_inner]  # outer wrap for batch_size=1

    # ── Step 6: Build the sigma schedule (same as generate_qwen) ──────
    sigmas = build_flux_sigmas(
        num_steps=num_inference_steps,
        image_seq_len=noise_latents.shape[1],
        scheduler_config=dict(pipe.scheduler.config),
        schedule=sigma_schedule,
    )
    sigmas = sigmas.to(device=device)

    # ── Refinement: truncate schedule + inject noise at starting sigma ──
    # When initial_latents is provided with denoise < 1.0, we run only
    # the tail of the sigma schedule and re-noise the starting latents
    # to the starting sigma so the sampler picks up mid-trajectory.
    # denoise >= 1.0 with initial_latents replaces them with fresh noise
    # (effectively "start over from random noise but keep the refs").
    if initial_latents is not None and denoise < 1.0:
        sigmas = truncate_sigmas_for_denoise(sigmas, denoise)
        starting_sigma = float(sigmas[0])
        noise_latents = inject_flow_noise(
            noise_latents, starting_sigma, generator=generator,
        )
    elif initial_latents is not None and denoise >= 1.0:
        noise_latents = torch.randn(
            noise_latents.shape, generator=generator,
            device=noise_latents.device, dtype=noise_latents.dtype,
        )

    # ── Step 7: Build the denoiser callable ────────────────────────────
    # Concatenates [noise, ref_1, ref_2, ...] before the transformer call
    # and slices noise_pred back to the noise portion before returning.
    # All ref latents are captured by closure — constant across steps.
    guidance_embed = None
    if bool(getattr(pipe.transformer.config, "guidance_embeds", False)):
        guidance_embed = torch.full(
            [1], guidance_scale, device=device, dtype=torch.float32,
        ).expand(noise_latents.shape[0])

    noise_seq_len = noise_latents.shape[1]

    def _call_transformer(x, sigma, embeds, mask):
        if not torch.is_tensor(sigma):
            sigma_t = torch.tensor(sigma, device=device, dtype=transformer_dtype)
        else:
            sigma_t = sigma.to(device=device, dtype=transformer_dtype)
        timestep = sigma_t.expand(x.shape[0]).to(dtype=transformer_dtype)

        if concatenated_refs is not None:
            latent_model_input = torch.cat([x, concatenated_refs], dim=1)
        else:
            latent_model_input = x

        out = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            guidance=guidance_embed,
            encoder_hidden_states_mask=mask,
            encoder_hidden_states=embeds,
            img_shapes=img_shapes,
            attention_kwargs={},
            return_dict=False,
        )
        noise_pred_full = out[0] if isinstance(out, tuple) else out
        # Slice back to the noise portion — the transformer returned
        # velocity for the full [noise, refs...] sequence but we only
        # propagate the noise part through the sampler.
        return noise_pred_full[:, :noise_seq_len]

    def denoiser(x, sigma):
        v_cond = _call_transformer(x, sigma, prompt_embeds, prompt_embeds_mask)
        if not do_cfg:
            return v_cond

        v_neg = _call_transformer(x, sigma, neg_prompt_embeds, neg_prompt_embeds_mask)

        # Norm-preserving CFG rescale in fp32 (same as generate_qwen)
        v_cond_f = v_cond.float()
        v_neg_f = v_neg.float()
        comb = v_neg_f + guidance_scale * (v_cond_f - v_neg_f)
        cond_norm = torch.norm(v_cond_f, dim=-1, keepdim=True)
        comb_norm = torch.clamp(torch.norm(comb, dim=-1, keepdim=True), min=1e-8)
        rescaled = comb * (cond_norm / comb_norm)
        return rescaled.to(dtype=v_cond.dtype)

    # ── Step 8: Run the sampler ────────────────────────────────────────
    sampler_fn = get_sampler(sampler_name)
    last_step_reported = [0]

    def _progress_wrapper(step_fractional):
        step_int = int(step_fractional)
        if step_int > last_step_reported[0]:
            last_step_reported[0] = step_int
            if progress_cb is not None:
                progress_cb(step_int)

    with torch.no_grad():
        final_noise_latents = sampler_fn(
            denoiser, noise_latents, sigmas, _progress_wrapper,
            eta=eta, generator=generator,
        )

    return final_noise_latents, output_height, output_width


def _qwen_edit_comfy_to_pil_list(image: torch.Tensor) -> list:
    """Convert a ComfyUI IMAGE tensor (B, H, W, 3) in [0, 1] to a list
    of PIL.Image objects, which is what the Qwen2VLProcessor natively
    accepts in its ``images=`` argument.

    The VL processor can technically consume tensors too, but the
    stock QwenImageEditPlusPipeline routes through
    image_processor.resize → PIL → processor, and matching that path
    exactly avoids subtle normalization drift between our manual loop
    and the stock pipeline.
    """
    from PIL import Image as _PILImage

    if image.dim() != 4 or image.shape[-1] != 3:
        raise ValueError(
            f"Expected (B, H, W, 3) tensor, got {tuple(image.shape)}"
        )

    pil_list = []
    for i in range(image.shape[0]):
        # (H, W, 3) in [0, 1] → (H, W, 3) in uint8
        frame = (image[i].clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()
        pil_list.append(_PILImage.fromarray(frame, mode="RGB"))
    return pil_list
