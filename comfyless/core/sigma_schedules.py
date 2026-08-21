"""Sigma schedules for flow-matching partial denoise.

A schedule decides how a fixed budget of steps is *spaced* across the active
noise range. Every schedule here spans the same interval — from
``sigma_start`` (set by ``denoise``) down to ``sigma_min`` — so switching
schedules changes where the steps cluster, never the noise level a run starts
from. That shared-endpoint property is load bearing: it is what lets a sidecar
replay swap schedules without also changing how much noise is being removed.

Normalised flow-match space is used throughout: ``sigma_max`` is 1.0 and
``sigma_min`` is ``1 / num_steps``.

The five schedules
------------------
``linear``
    Even spacing. The default, and the fallback for anything unrecognised.
``balanced``
    Karras warping at rho=3. Mild bias toward low sigma, so composition gives
    up some budget to detail and texture.
``karras``
    Karras warping at rho=7, the EDM-optimal exponent. Strong bias toward low
    sigma, which suits a final detail pass.
``beta57``
    Inverse beta CDF warping at alpha=0.5, beta=0.7. Clusters steps at BOTH
    ends of the range rather than one.
``bong_tangent``
    A two-stage arctan S-curve that lingers around the mid-sigma pivots.

Provenance: this module is a from-scratch implementation of the schedule
contract that `nodes/eric_qwen_image_multistage.py` established, written for
the comfyless runtime so it carries no third-party lineage into the extracted
package (ADR-045 slice 3). The node pack keeps its own copy, untouched. The
underlying maths is published: the Karras warp is eq. 5 of Karras et al.,
"Elucidating the Design Space of Diffusion-Based Generative Models"
(NeurIPS 2022); ``beta57`` and ``bong_tangent`` reproduce the geometry of
ComfyUI's ``beta_scheduler`` and ClownsharkBatwing/RES4LYF's
``bong_tangent_scheduler`` respectively, both from their formulas.

Output equivalence with the node-pack original is pinned exhaustively by
`test_sigma_schedules.py` against a 13,000-case golden captured before the
split (`tests/golden/sigma_schedules.json.gz`). Any divergence is a
behaviour change that would silently alter replays of existing sidecars, so
that suite is the gate on editing anything here.
"""

from __future__ import annotations

import operator

import numpy as np

#: Schedule names accepted by :func:`build_sigma_schedule`, in doc order.
SIGMA_SCHEDULES = ("linear", "balanced", "karras", "beta57", "bong_tangent")

#: Karras warp exponents. rho=7 is the EDM-optimal value; rho=3 is the
#: gentler variant exposed as "balanced".
_KARRAS_RHO = {"balanced": 3.0, "karras": 7.0}

#: Inverse-beta-CDF shape parameters behind the "beta57" name.
_BETA57_ALPHA = 0.5
_BETA57_BETA = 0.7

#: bong_tangent geometry: the arctan slope reference, and the fraction of the
#: internal step budget that the first stage gets.
_BONG_SLOPE = 0.2
_BONG_SLOPE_REF_STEPS = 40.0
_BONG_STAGE_SPLIT = 0.6


def _positions(count: int) -> np.ndarray:
    """``count`` normalised positions spanning [0, 1] inclusive."""
    return np.linspace(0.0, 1.0, count)


def _karras_warp(sigma_hi: float, sigma_lo: float, count: int, rho: float) -> np.ndarray:
    """Karras/EDM spacing between two sigmas.

    Interpolates linearly in ``sigma ** (1 / rho)`` and undoes the exponent,
    which packs samples toward ``sigma_lo`` more tightly as ``rho`` grows.
    """
    inv_rho = 1.0 / rho
    hi_warped = sigma_hi ** inv_rho
    lo_warped = sigma_lo ** inv_rho
    return (hi_warped + _positions(count) * (lo_warped - hi_warped)) ** rho


def _beta_warp(sigma_hi: float, sigma_lo: float, count: int) -> np.ndarray:
    """Inverse-beta-CDF spacing, which clusters toward both endpoints."""
    from scipy.stats import beta as beta_dist

    # ppf runs 1 -> 0 as the position runs 0 -> 1, so this already descends.
    fraction = beta_dist.ppf(1.0 - _positions(count), _BETA57_ALPHA, _BETA57_BETA)
    return sigma_lo + fraction * (sigma_hi - sigma_lo)


def _bong_arc(count: int, pivot: float, sigma_hi: float, sigma_lo: float,
              slope: float) -> np.ndarray:
    """One arctan stage, rescaled to land exactly on [sigma_lo, sigma_hi].

    The raw arctan is normalised by its own first and last value rather than
    analytically, so the stage hits both endpoints regardless of where the
    pivot sits. A degenerate (flat) arc falls back to a unit span so the
    rescale cannot divide by zero.
    """
    raw = ((2.0 / np.pi) * np.arctan(-slope * (np.arange(count) - pivot)) + 1.0) / 2.0
    first, last = raw[0], raw[-1]
    span = first - last if abs(first - last) > 1e-9 else 1.0
    return (raw - last) / span * (sigma_hi - sigma_lo) + sigma_lo


def _bong_tangent(sigma_hi: float, sigma_lo: float, count: int) -> np.ndarray:
    """Two arctan stages joined at the midpoint sigma.

    RES4LYF runs ``count + 2`` steps internally and returns ``count + 1``
    sigmas, so an internal budget of ``count + 1`` yields exactly ``count``
    here. The stages split 60/40; stage 1 pivots at the split and stage 2 at
    its own origin. Stage 1 drops its final sample so the shared midpoint is
    emitted once, by stage 2.

    At ``count <= 2`` the split degenerates — one stage would be empty after
    that drop — so this falls back to even spacing, which also keeps the
    shared-start-sigma invariant that a truncated two-stage curve would break.
    """
    if count <= 2:
        return np.linspace(sigma_hi, sigma_lo, count)

    internal = count + 1
    slope = _BONG_SLOPE / (internal / _BONG_SLOPE_REF_STEPS)
    sigma_mid = (sigma_hi + sigma_lo) / 2.0
    split = int(internal * _BONG_STAGE_SPLIT)

    head = _bong_arc(split, split, sigma_hi, sigma_mid, slope)[:-1]
    tail = _bong_arc(internal - split, 0, sigma_mid, sigma_lo, slope)
    return np.concatenate([head, tail])


def _resolve_start_sigma(num_steps: int, denoise: float, sigma_min: float,
                         sigma_max: float) -> tuple[int, float]:
    """Return ``(step_count, sigma_start)`` for a denoise fraction.

    A partial denoise keeps the LOWER end of the range. The starting sigma is
    read off the even schedule at the truncation point so that every schedule
    begins at the same noise level for a given ``denoise`` — without this,
    curved schedules start somewhere quite different after truncation, which
    shows up as ghosting between stages.
    """
    if denoise >= 1.0:
        return num_steps, sigma_max
    step_count = max(1, int(round(num_steps * denoise)))
    even = np.linspace(sigma_max, sigma_min, num_steps)
    return step_count, float(even[num_steps - step_count])


def build_sigma_schedule(num_steps: int, denoise: float,
                         schedule: str = "linear",
                         power: float = 1.0) -> list:
    """Build a descending sigma schedule for flow-matching partial denoise.

    Args:
        num_steps: Total step budget. Also sets ``sigma_min = 1 / num_steps``.
        denoise:   Fraction of the range to cover. ``>= 1.0`` uses all of it;
                   below that, only the lower portion, over
                   ``round(num_steps * denoise)`` steps.
        schedule:  One of :data:`SIGMA_SCHEDULES`. Anything else is treated
                   as ``"linear"`` rather than raising, so an unknown name
                   from a stale sidecar still replays.
        power:     Accepted and ignored. Kept because it is part of the
                   established call signature and appears in sidecars.

    Returns:
        Descending sigmas, length ``round(num_steps * denoise)`` (or
        ``num_steps`` at full denoise).
    """
    del power  # accepted for signature compatibility; never applied

    sigma_max = 1.0
    sigma_min = 1.0 / num_steps
    count, sigma_start = _resolve_start_sigma(num_steps, denoise, sigma_min, sigma_max)
    # Reject a non-integer step count. Four of the five schedules would reject
    # it anyway on the way to np.linspace, but bong_tangent reaches np.arange
    # instead, which accepts floats and would quietly return a schedule while
    # its siblings raised. The guard sits HERE rather than at entry so the
    # order in which degenerate inputs are rejected is unchanged: a partial
    # denoise already raises inside _resolve_start_sigma, and a NaN denoise
    # still raises there first.
    count = operator.index(count)

    if schedule in _KARRAS_RHO:
        sigmas = _karras_warp(sigma_start, sigma_min, count, _KARRAS_RHO[schedule])
    elif schedule == "beta57":
        sigmas = _beta_warp(sigma_start, sigma_min, count)
    elif schedule == "bong_tangent":
        sigmas = _bong_tangent(sigma_start, sigma_min, count)
    else:
        sigmas = np.linspace(sigma_start, sigma_min, count)

    # Warped curves can overshoot their endpoints by a rounding step.
    return np.clip(sigmas, sigma_min, sigma_start).tolist()
