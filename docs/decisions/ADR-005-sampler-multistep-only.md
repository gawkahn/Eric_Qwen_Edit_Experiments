# ADR-005: Custom Samplers — Multistep (Adams-Bashforth) Only

**Date:** 2026-04-21 (written retroactively; decision made ~2026-04-12)
**Status:** accepted

---

## Context

Higher-order samplers offer better quality per step than first-order Euler. Two
categories exist for flow-matching models:

**Multistep methods** (Adams-Bashforth AB2, AB3): buffer previous step velocity
predictions and use them in the next step's update. One model evaluation per step.
These fit the diffusers scheduler API (`set_timesteps` / `step`) exactly — the
pipeline calls `step()` once per model call and the scheduler maintains its own
history buffer across calls.

**Single-step higher-order methods** (Heun, RK3, RK4, DPM++2S): require 2+ model
evaluations per denoising step. The scheduler API provides exactly one model call
per `step()` invocation with no mechanism to request additional evaluations mid-step.

diffusers' own `FlowMatchHeunDiscreteScheduler` works around this by doubling the
timesteps array so the pipeline iterates enough times, but this approach breaks with
custom sigma schedules (which this project uses for `balanced` and `karras` schedules).

## Decision

Implement only Adams-Bashforth multistep schedulers (AB2, AB3) as drop-in
`FlowMatchEulerDiscreteScheduler` subclasses. These achieve 2nd- and 3rd-order
accuracy within the one-call-per-step contract and provide measurable quality
improvement — sharper details, better prompt adherence — with no pipeline changes.

AB2 uses linear multistep extrapolation; AB3 uses a quadratic fit with fallback to
AB2 for non-uniform step sizes (to avoid a numerically unstable variable-step AB3
derivation).

Heun, RK3, RK4, and DPM++2S are explicitly deferred pending a full manual denoising
loop (Phase C / RES4LYF territory). They are not "missing" — the constraint is
architectural.

## Alternatives Rejected

**Doubling timesteps array (FlowMatchHeun approach)** — breaks with custom sigma
schedules; also creates confusing step count semantics (user requests 28 steps,
gets 56 model calls).

**Full manual denoising loop now** — significant scope increase that blocks shipping
the multistep improvement; AB2/AB3 cover the near-term quality gain without it.

**Accept Euler quality to stay simple** — AB2 is not complex to implement and the
quality improvement is empirically visible at equal step counts.

## Deferred / Out of Scope

**Heun, RK3, RK4, DPM++2S** — require a manual denoising loop that controls model
calls directly. Tracked in TECH_DEBT.md. Trigger: Phase C manual loop implementation.

**SDXL/SD1 sampler compatibility** — these use DDPM-style schedulers lacking
`init_noise_sigma`; applying flow-match samplers to them crashes. The generate node
guards against this and falls back to `"default"` with a warning (`66329db`, `1fe0050`).

## Changelog

- ~2026-04-12: Initial AB2 + AB3 implementation
- ~2026-04-12: SDXL/SD1 sampler guard added (`66329db`, `1fe0050`)
- 2026-04-21: ADR written retroactively; Heun/RK3 deferral formally recorded in
  TECH_DEBT.md with this ADR as the reference

## AI-Disclosure

ADR authored by Claude Sonnet 4.6, 2026-04-21. Architecture and implementation by
Eric Hiss; rationale drawn from code docstrings in `eric_diffusion_samplers.py`.
Reviewed by Grant Kahn.
