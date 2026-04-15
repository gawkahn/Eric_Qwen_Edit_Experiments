# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Diffusion Scheduler Registry & Swap Helper

CURRENTLY UNUSED IN PRODUCTION — kept for Phase C (manual denoising loop).

Phase A status (2026-04-12)
---------------------------
This module was built to expose alternate schedulers via a dropdown on the
generate / multistage nodes.  In testing it turned out that **only
FlowMatchEulerDiscreteScheduler** actually works via a simple
``pipe.scheduler = ...`` swap on diffusers Flux/Chroma/Qwen pipelines:

- **FlowMatchHeunDiscreteScheduler** rejects the ``sigmas=`` argument that
  the Flux pipeline always passes (``set_timesteps`` doesn't accept it).
- **DPMSolverMultistepScheduler / DEIS / UniPC** in flow-sigma mode produce
  a mosaic of noise — their ``step()`` math assumes epsilon prediction,
  while flow-matching models output velocity, so updates are near-zero.
- **EulerDiscreteScheduler / HeunDiscreteScheduler** classic — same issue.

The clean solution is Phase C: a manual denoising loop where we own the
sigma generation, model output interpretation, and update rule.  At that
point this registry, ``swap_scheduler``, and ``is_flow_match`` will be
useful again — but with the helper bypassing ``pipe()`` entirely.

Until then, this module is unimported by any node UI.  The tests still
exercise it to keep it from rotting.

Author: Eric Hiss (GitHub: EricRollei)
"""

from contextlib import contextmanager
from typing import Optional


# ── Scheduler registry ──────────────────────────────────────────────────────
#
# Maps user-facing dropdown name → (diffusers class name, config override dict).
# Classes are imported lazily so a missing diffusers version doesn't break
# the whole module.
#
# The "default" entry is a no-op (keeps the pipeline's current scheduler).

_SCHEDULER_REGISTRY: dict = {
    "default": (None, {}),

    # ── Native flow-matching schedulers ─────────────────────────────────
    # These are the only schedulers that work via a simple scheduler swap
    # for Flux/Chroma/Qwen/Flux2 pipelines.  The classic schedulers
    # (DPMSolver, DEIS, UniPC, Euler, Heun) *can* be instantiated with
    # use_flow_sigmas=True, but their step() update rule still assumes
    # epsilon/noise prediction.  Flow-matching models output velocity,
    # so the classic solvers produce near-zero effective updates and
    # decode to a mosaic of noise.  Real support for those requires a
    # manual denoising loop (Phase C / RES4LYF integration work).
    "flow_match_euler": ("FlowMatchEulerDiscreteScheduler", {}),
    "flow_match_heun":  ("FlowMatchHeunDiscreteScheduler", {}),
}


def scheduler_choices() -> list:
    """Return the list of scheduler names for dropdown use."""
    return list(_SCHEDULER_REGISTRY.keys())


def is_flow_match(scheduler) -> bool:
    """Return True if *scheduler* is a native flow-matching scheduler.

    Native flow-match schedulers (FlowMatchEuler, FlowMatchHeun) apply mu-based
    dynamic shifting automatically.  Classic schedulers (DPMSolver, DEIS, UniPC,
    Euler, Heun) do not, even when run in ``use_flow_sigmas=True`` mode.

    Detection uses class name rather than config keys because diffusers'
    ConfigMixin stores all init kwargs in ``.config`` even the ignored ones,
    which would cause false positives on a config-key check.
    """
    if scheduler is None:
        return False
    class_name = type(scheduler).__name__
    return class_name.startswith("FlowMatch")


def _build_scheduler(name: str, base_config: dict):
    """Instantiate a scheduler by registry name from an existing config.

    Returns (new_scheduler, class_name) or raises ValueError if the name
    is unknown or the class isn't available in the installed diffusers.
    """
    if name not in _SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler name: {name!r}. "
                         f"Valid: {scheduler_choices()}")

    class_name, overrides = _SCHEDULER_REGISTRY[name]
    if class_name is None:  # "default" — caller should skip the swap
        return None, None

    import diffusers
    cls = getattr(diffusers, class_name, None)
    if cls is None:
        raise ValueError(
            f"Scheduler class '{class_name}' not available in installed "
            f"diffusers. Try: pip install -U diffusers"
        )

    # from_config copies matching keys from base_config and accepts overrides
    # via kwargs.  Unknown keys in base_config are silently ignored by
    # diffusers, which is what we want when converting between scheduler
    # families.
    merged = {**dict(base_config), **overrides}
    try:
        new_sched = cls.from_config(merged)
    except Exception as e:
        raise ValueError(
            f"Failed to build {class_name} from config: {e}"
        )
    return new_sched, class_name


@contextmanager
def swap_scheduler(pipe, scheduler_name: str, log_prefix: str = "[EricDiffusion]"):
    """Temporarily swap ``pipe.scheduler`` to the named scheduler.

    Restores the original scheduler on exit, including on exceptions raised
    inside the ``with`` block.  If *scheduler_name* is ``"default"`` or the
    swap fails (unknown name, missing class, config incompatibility), yields
    with the original scheduler in place — the caller should check the log
    if they expected a specific scheduler.

    Critical structural note: build errors are handled OUTSIDE the yield so
    that exceptions raised inside the ``with`` block propagate cleanly.
    Mixing them in a single try/except causes "generator didn't stop after
    throw()" because the generator would yield twice.

    Example::

        with swap_scheduler(pipe, "flow_match_heun"):
            result = pipe(prompt=..., sigmas=..., ...)
    """
    if scheduler_name in (None, "", "default"):
        yield
        return

    original = pipe.scheduler

    # ── Build phase: errors here mean "fall back to original" ──────────
    try:
        new_sched, class_name = _build_scheduler(
            scheduler_name, dict(original.config)
        )
    except ValueError as e:
        print(f"{log_prefix} Scheduler swap failed ({e}) — using original: "
              f"{type(original).__name__}")
        yield
        return

    if new_sched is None:  # "default" entry
        yield
        return

    # ── Run phase: any exception inside the with block must propagate ─
    pipe.scheduler = new_sched
    print(f"{log_prefix} Scheduler swapped: "
          f"{type(original).__name__} → {class_name}")
    try:
        yield
    finally:
        pipe.scheduler = original
        print(f"{log_prefix} Scheduler restored: {type(original).__name__}")
