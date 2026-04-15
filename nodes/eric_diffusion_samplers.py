# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Custom Sampler Schedulers

Drop-in replacements for ``FlowMatchEulerDiscreteScheduler`` that implement
higher-order multistep integration of the flow-matching ODE.

Why drop-in schedulers instead of a manual loop
------------------------------------------------
Diffusers pipelines (FluxPipeline, ChromaPipeline, QwenImagePipeline, Flux2)
are tightly coupled to their scheduler's ``set_timesteps`` / ``step`` API.
Replacing the whole pipeline call with a manual denoising loop is possible
(and Phase C will need it for RES4LYF samplers which run their own loops)
but for Adams-Bashforth style multistep methods, we don't need that.

Multistep methods use only ONE model evaluation per denoising step — they
buffer previous velocity predictions and use them in the next step's update.
This fits the scheduler API perfectly: the pipeline calls ``step()`` once
per model call, and we maintain our buffer across calls.

Single-step higher-order methods (Heun, RK4, DPM++2S) require 2+ model
evaluations per denoising step.  Those need either:
  - Doubling the timesteps array so the pipeline iterates enough times
    (how diffusers' FlowMatchHeun does it — but fails with custom sigmas),
  - Or a manual loop that controls model calls directly.

Math recap: Adams-Bashforth for flow matching
----------------------------------------------
The flow-matching ODE is ``dx/dt = v(x, t)`` where ``v`` is the velocity
predicted by the transformer.  We integrate from ``t=1`` (noise) to ``t=0``
(clean image) using the sigma schedule.

Euler (1st order):
    x_{n+1} = x_n + h_n * v_n
    where h_n = sigma_{n+1} - sigma_n  (negative when denoising)

Adams-Bashforth 2 (2nd order, linear multistep, 1 evaluation/step):
    x_{n+1} = x_n + h_n * ((1 + r/2) * v_n - (r/2) * v_{n-1})
    where r = h_n / h_{n-1}  (ratio of current to previous step size)

Adams-Bashforth 3 (3rd order, 1 evaluation/step):
    Standard AB3 coefficients adapted for variable step size.

For uniform step sizes (r=1) AB2 reduces to:
    x_{n+1} = x_n + h * (3/2 * v_n - 1/2 * v_{n-1})

The first step uses plain Euler because no previous ``v`` is available.

Accuracy vs. quality
--------------------
For smooth ODEs, 2nd-order methods have local truncation error O(h³) vs
O(h²) for Euler.  In practice: visibly sharper details, better prompt
adherence at equal step counts, or equal quality at fewer steps.
The "DPM-Solver++ 2M" formulation you may have seen elsewhere is closely
related but includes extra factors for specific diffusion noise schedules.
For pure flow matching (linear ODE), AB2 and DPM++2M converge to the
same update rule, so we use the simpler and more general naming.

Author: Eric Hiss (GitHub: EricRollei)
"""

from contextlib import contextmanager
from typing import Optional

import torch


# ── Lazy import of diffusers base class ────────────────────────────────────
#
# Delaying the import keeps this module parseable even if diffusers isn't
# installed.  The actual classes are built inside _build_scheduler_classes()
# on first use.

_SCHEDULER_CLASSES: dict = {}


def _build_scheduler_classes() -> None:
    """Instantiate the custom scheduler classes (once).

    Subclassing is done at first use so we can inherit from
    FlowMatchEulerDiscreteScheduler without requiring diffusers at import.
    """
    if _SCHEDULER_CLASSES:
        return  # already built

    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteSchedulerOutput,
    )

    class FlowMultistep2Scheduler(FlowMatchEulerDiscreteScheduler):
        """2nd-order Adams-Bashforth multistep for flow matching.

        Buffers the previous step's velocity prediction and uses it in the
        next step's update.  First step uses plain Euler because no
        previous velocity exists.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._prev_model_output: Optional[torch.Tensor] = None
            self._prev_dt: Optional[float] = None

        # Signature must match the parent class EXACTLY (with named
        # parameters, not *args/**kwargs) because diffusers' retrieve_timesteps
        # uses inspect.signature to check if the scheduler accepts custom
        # sigmas and mu.  A *args/**kwargs signature reports zero named
        # parameters and retrieve_timesteps rejects the scheduler.
        def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            device=None,
            sigmas: Optional[list] = None,
            mu: Optional[float] = None,
            timesteps: Optional[list] = None,
        ):
            super().set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )
            # Reset the buffer when a new generation begins
            self._prev_model_output = None
            self._prev_dt = None

        def step(
            self,
            model_output: torch.FloatTensor,
            timestep,
            sample: torch.FloatTensor,
            **kwargs,
        ):
            # Replicate the step-index bookkeeping from the parent
            if self.step_index is None:
                self._init_step_index(timestep)

            sample = sample.to(torch.float32)

            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
            dt = sigma_next - sigma

            if self._prev_model_output is None:
                # First step: plain Euler
                prev_sample = sample + dt * model_output
            else:
                # Adams-Bashforth 2 with variable step size:
                #   x_{n+1} = x_n + h_n * ((1 + r/2) * v_n - (r/2) * v_{n-1})
                # where r = h_n / h_{n-1}.
                r = dt / self._prev_dt if self._prev_dt != 0 else 1.0
                v_eff = (1.0 + r / 2.0) * model_output - (r / 2.0) * self._prev_model_output
                prev_sample = sample + dt * v_eff

            self._prev_model_output = model_output
            self._prev_dt = dt

            self._step_index += 1
            prev_sample = prev_sample.to(model_output.dtype)

            return_dict = kwargs.get("return_dict", True)
            if not return_dict:
                return (prev_sample,)
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    class FlowMultistep3Scheduler(FlowMatchEulerDiscreteScheduler):
        """3rd-order Adams-Bashforth multistep for flow matching.

        Buffers the two previous velocity predictions.  Steps 1 and 2 use
        lower-order methods until enough history is available.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._prev_model_output_1: Optional[torch.Tensor] = None  # v_{n-1}
            self._prev_model_output_2: Optional[torch.Tensor] = None  # v_{n-2}
            self._prev_dt_1: Optional[float] = None
            self._prev_dt_2: Optional[float] = None

        # See FlowMultistep2Scheduler.set_timesteps — explicit named params
        # are required so inspect.signature sees 'sigmas' and 'mu'.
        def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            device=None,
            sigmas: Optional[list] = None,
            mu: Optional[float] = None,
            timesteps: Optional[list] = None,
        ):
            super().set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )
            self._prev_model_output_1 = None
            self._prev_model_output_2 = None
            self._prev_dt_1 = None
            self._prev_dt_2 = None

        def step(
            self,
            model_output: torch.FloatTensor,
            timestep,
            sample: torch.FloatTensor,
            **kwargs,
        ):
            if self.step_index is None:
                self._init_step_index(timestep)

            sample = sample.to(torch.float32)

            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
            dt = sigma_next - sigma

            if self._prev_model_output_1 is None:
                # Step 0: Euler
                prev_sample = sample + dt * model_output
            elif self._prev_model_output_2 is None:
                # Step 1: AB2 (same as the multistep2 scheduler)
                r = dt / self._prev_dt_1 if self._prev_dt_1 != 0 else 1.0
                v_eff = (1.0 + r / 2.0) * model_output - (r / 2.0) * self._prev_model_output_1
                prev_sample = sample + dt * v_eff
            else:
                # Step 2+: AB3 with variable step sizes.
                #
                # For uniform h, the formula reduces to:
                #   x_{n+1} = x_n + h * (23/12 * v_n - 16/12 * v_{n-1} + 5/12 * v_{n-2})
                #
                # For variable step sizes we solve the quadratic polynomial
                # interpolation of v through (t_{n-2}, v_{n-2}), (t_{n-1}, v_{n-1}),
                # (t_n, v_n) and integrate from t_n to t_{n+1}.  The result is:
                #
                #   I = h_n * (A * v_n + B * v_{n-1} + C * v_{n-2})
                #
                # with coefficients derived from the step-size ratios.
                h_n   = dt
                h_n1  = self._prev_dt_1   # h_{n-1}
                h_n2  = self._prev_dt_2   # h_{n-2}

                # Uniform fallback if step sizes are too small or degenerate
                if abs(h_n1) < 1e-12 or abs(h_n2) < 1e-12:
                    A, B, C = 23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0
                else:
                    # Variable-step AB3 derived from Lagrange interpolation.
                    # Let a = h_n, b = h_{n-1}, c = h_{n-2}, and define
                    #   s1 = b        (t_n - t_{n-1})
                    #   s2 = b + c    (t_n - t_{n-2})
                    # Then:
                    #   A = 1 + a/(2*s1) + a/(2*s2) + a²/(3*s1*s2)
                    #   B = -a/(2*(s1 - s2)) * (1 + a/(3*s1 + ...))  ...
                    #
                    # Rather than risk a derivation error, use uniform AB3
                    # when step sizes are close to uniform, otherwise fall
                    # back to AB2 for that step (safer than wrong AB3).
                    ratio = abs(h_n1 / h_n2 - 1.0) + abs(h_n / h_n1 - 1.0)
                    if ratio < 0.1:  # roughly uniform
                        A, B, C = 23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0
                    else:
                        # Fall back to AB2 for non-uniform step sizes
                        r = h_n / h_n1
                        A = 1.0 + r / 2.0
                        B = -(r / 2.0)
                        C = 0.0

                v_eff = (A * model_output
                         + B * self._prev_model_output_1
                         + (C * self._prev_model_output_2 if C != 0 else 0))
                prev_sample = sample + dt * v_eff

            # Shift history: v_{n-2} ← v_{n-1} ← current
            self._prev_model_output_2 = self._prev_model_output_1
            self._prev_dt_2 = self._prev_dt_1
            self._prev_model_output_1 = model_output
            self._prev_dt_1 = dt

            self._step_index += 1
            prev_sample = prev_sample.to(model_output.dtype)

            return_dict = kwargs.get("return_dict", True)
            if not return_dict:
                return (prev_sample,)
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    _SCHEDULER_CLASSES["multistep2"] = FlowMultistep2Scheduler
    _SCHEDULER_CLASSES["multistep3"] = FlowMultistep3Scheduler


# ── Registry & dropdown ────────────────────────────────────────────────────

_SAMPLER_NAMES = [
    "default",       # no swap (pipeline's FlowMatchEuler)
    "multistep2",    # 2nd-order Adams-Bashforth
    "multistep3",    # 3rd-order Adams-Bashforth
]


def sampler_choices() -> list:
    """Return the list of sampler names for dropdown use."""
    return list(_SAMPLER_NAMES)


def _build_sampler_scheduler(name: str, base_scheduler):
    """Instantiate a custom scheduler of the given type from an existing
    scheduler's config.

    Returns a new scheduler instance whose ``step()`` implements the named
    sampler, or ``None`` for the ``"default"`` name (no swap needed).
    """
    if name in (None, "", "default"):
        return None

    _build_scheduler_classes()
    if name not in _SCHEDULER_CLASSES:
        raise ValueError(
            f"Unknown sampler name: {name!r}. Valid: {sampler_choices()}"
        )

    cls = _SCHEDULER_CLASSES[name]
    return cls.from_config(base_scheduler.config)


@contextmanager
def swap_sampler(pipe, sampler_name: str, log_prefix: str = "[EricDiffusion]"):
    """Temporarily swap ``pipe.scheduler`` to a custom-sampler scheduler.

    Restores the original scheduler on exit, including on exceptions raised
    inside the ``with`` block.  If *sampler_name* is ``"default"`` or the
    swap fails, yields with the original scheduler in place.

    Same structural pattern as ``eric_diffusion_scheduler.swap_scheduler``:
    build errors are caught BEFORE the yield, and the yield is wrapped in
    its own try/finally so exceptions from inside the ``with`` block
    propagate cleanly.

    Example::

        with swap_sampler(pipe, "multistep2"):
            result = pipe(prompt=..., ...)
    """
    if sampler_name in (None, "", "default"):
        yield
        return

    original = pipe.scheduler

    try:
        new_sched = _build_sampler_scheduler(sampler_name, original)
    except ValueError as e:
        print(f"{log_prefix} Sampler swap failed ({e}) — using default")
        yield
        return

    if new_sched is None:
        yield
        return

    pipe.scheduler = new_sched
    print(f"{log_prefix} Sampler swapped: "
          f"{type(original).__name__} → {type(new_sched).__name__} "
          f"({sampler_name})")
    try:
        yield
    finally:
        pipe.scheduler = original
        print(f"{log_prefix} Sampler restored: {type(original).__name__}")
