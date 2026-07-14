# ADR-029: RES exponential-multistep samplers (res_2m / res_3m)

**Date:** 2026-07-13
**Status:** accepted

---

## Context

The custom-sampler set (`nodes/eric_diffusion_samplers.py`) held `multistep2` /
`multistep3` — 2nd/3rd-order **Adams-Bashforth** (polynomial) multistep methods,
implemented as drop-in `FlowMatchEulerDiscreteScheduler` subclasses that override
only `step()` (one model eval per step, buffering previous outputs). `--schedule`
was extended with RES4LYF's `beta57` / `bong_tangent` in ADR-028. This ADR adds the
matching **samplers** the user wanted: RES4LYF's `res_2m` / `res_3m`.

RES ("Refined Exponential Solver", ClownsharkBatwing/RES4LYF) methods are
**exponential** integrators, not polynomial: they use a log-sigma step size
`h = -ln(σ_next/σ)` and φ-function coefficients (`φ_j(z) = (e^z − Σ_{k<j} z^k/k!)/z^j`).
Investigation of the RES4LYF source established that the **multistep** variants
(`…m`) are one-eval-per-step (they reuse the buffered previous *denoised*
prediction as the second stage), so they are structurally drop-in-compatible —
unlike the **single-step** variants (`res_2s`, and the guide-based ClownSampler
methods), which need 2+ evals/step or the full RES manual sampling loop (the
"Phase C" work) and remain out of scope.

## Decision

Add `res_2m` and `res_3m` as drop-in `FlowMatchEulerDiscreteScheduler` subclasses
(`FlowRES2MScheduler` / `FlowRES3MScheduler`), mirroring the existing multistep
schedulers. The update is the RES exponential-multistep step, in the form used by
RES4LYF's main sampler loop (`beta/rk_sampler_beta.py`: `x_0 + h·Σ b_i·eps_i`,
eps anchored to x) and, equivalently, ComfyUI-core `res_multistep`
(`e^{-h}·x + h·Σ b_i·denoised_i`, raw denoised) — reimplemented from the formulas,
not copied:

```
denoised = x - σ·v                         # flow-matching x0 prediction
h = -ln(σ_next/σ) ,  c2 = -h_prev/h  [, c3 = -h_prev2/h]
res_2m:  b2 = φ2(-h)/c2 ,  b1 = φ1(-h) - b2
res_3m:  γ = (3c3³-2c3)/(c2(2-3c2)) ,  b3 = φ2(-h)/(γc2+c3) ,  b2 = γb3 ,  b1 = φ1(-h)-b2-b3
x_{n+1} = e^{-h}·x + h·Σ b_i·denoised_i     # raw-denoised form (A)
```

The two forms are algebraically identical because `h·Σb = h·φ1(-h) = 1 - e^{-h}`;
we implement form (A). **A first cut mistakenly combined both** — the `e^{-h}·x`
prefactor *and* eps-anchoring (`e^{-h}·x + h·Σb·(denoised - x)`) — which
double-counts the decay (state coefficient `2e^{-h}-1`, negative for h>ln2) and is
order-zero. Caught in code review before commit; see the Validation section and
Changelog.

The φ-functions are ported from RES4LYF `phi_functions.py`, computed in float64 via
the always-stable entire-function Taylor series `Σ z^m/(m+j)!` near z=0 (where the
closed form suffers catastrophic cancellation) and the closed form elsewhere.

**Bootstrapping** (multistep needs history): step 0 → flow Euler; `res_3m` step 1 →
`res_2m`; then full order. The **final step** (σ_next→0, where `h`→∞) falls back to
flow Euler — which at σ_next=0 already yields the x0 prediction. This matches
RES4LYF's initial-sampler / order-ramp behavior.

`res_2m`/`res_3m` join `_SAMPLER_NAMES`, so the comfyless `--sampler` choices and
the node `sampler_choices()` gain them automatically. They compose with
`--schedule` (ADR-028: sampler = integration order, schedule = sigma spacing,
orthogonal), and are excluded on classic (SDXL/SD1) schedulers by the existing
flow-match sampler guard.

## Validation

Numerically verified in `test_samplers.py` (does not need a GPU):
1. **φ port** matches the RES4LYF closed form to float precision (the Taylor
   branch is actually *more* accurate than the cancelling closed form near z=0).
2. **Step-form match:** `res_2m` (and the `res_3m` γ/b3 branch) reproduce the
   raw-denoised form (A) reference — `e^{-h}·x + h·Σ b_i·denoised_i` — to < 1e-5.
   The reference is derived from the independently-authored correct form (ComfyUI
   core `res_multistep`), NOT from the implementation's own expression.
3. **Exactness on a constant nonlinearity (definitive):** a fixed-x0 "model"
   (velocity `v=(x-x0*)/σ`, so `denoised≡x0*`) has the analytic solution
   `x_i - x0* = (σ_i/σ_0)(x_init - x0*)` at *every* step; exponential (and Euler)
   methods are exact here, so each step must match to ~1e-4. This is the check that
   catches the double-counting bug — the earlier "converges as σ→0" test was
   vacuous (the terminal σ=0 Euler step masks any prior drift) and the earlier
   "exact-step" test was circular (reference re-derived the same buggy form).

**Image-quality parity with RES4LYF on real models is NOT unit-testable and is the
user's acceptance gate — to be confirmed on a hot GPU run.** The numerical core
(formula, φ, exact-step, per-step exactness) is proven; visual parity is the
remaining, still-pending verification.

## Alternatives Rejected

- **Port the full RES4LYF RK framework** (`rk_sampler_beta.py` + coefficients +
  guides + SDE + implicit solvers, thousands of lines). Rejected — enormous surface
  for two multistep methods; the self-contained `calculate_res_2m/3m_step`
  functions are the exact update we need.
- **Re-derive the coefficients from first principles.** Rejected — the RES4LYF
  formulas are the authoritative reference; re-deriving invites drift from the
  behavior the user expects.
- **Approximate with AB-style coefficients** (the ADR-028 "option B"). Rejected —
  the user chose the faithful port (option A); AB≠RES.

## Deferred / Out of Scope

- **Single-step RES methods** (`res_2s`, `res_2s_stable`, RKMK, the guide-based
  ClownSampler methods). Trigger: they need 2+ evals/step or the RES manual loop
  ("Phase C") — a much larger integration.
- **`--iterate` axes for `--sampler` / `--schedule`.** Deferred to a dedicated
  follow-up now that the full sampler/schedule set exists (per the ADR-028 note).
- **SDE / eta noise, guides, and the RES `denoised` output.** Not part of the
  deterministic drop-in.

## Changelog

- 2026-07-13 — Initial. Accepted. `res_2m`/`res_3m` added as drop-in
  `FlowMatchEuler` subclasses; φ-functions + coefficients ported from RES4LYF,
  update in ComfyUI-core `res_multistep` form (A). `test_samplers.py` +20.
  Image-quality parity is a hot-run acceptance gate.
  **code-reviewer (Fable) caught a CRITICAL before commit:** the first cut combined
  the `e^{-h}·x` prefactor with eps-anchoring (double-counting the decay → order-0,
  converges to x0/2), and the two tests meant to prove fidelity were circular
  (reference = same buggy form) and vacuous (terminal-Euler step masked the drift).
  Fixed to form (A); replaced the tests with a form-(A) reference derived from
  ComfyUI core, a res_3m order-3 step-match, and a **per-step-exact** constant-
  nonlinearity check (the definitive test the buggy form fails); added
  duplicate-sigma (h==0) guards. Re-reviewed by code-reviewer (Fable).

**AI-Disclosure:** Claude (Fable 5) authored from a design conversation with Grant,
porting the numerical method from the RES4LYF source; Grant reviewed.
