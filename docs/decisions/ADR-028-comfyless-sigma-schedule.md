# ADR-028: Wire `--schedule` (sigma spacing) into comfyless generation

**Date:** 2026-07-13
**Status:** accepted

---

## Context

`comfyless.generate` has advertised `--schedule {linear,balanced,karras}` since the
CLI was scaffolded, but the value was **accepted and recorded in the sidecar and
never applied** — the docstring called it "reserved for future manual-loop use."
The `sampler` axis (`default`/`multistep2`/`multistep3`) *is* wired (via
`nodes.eric_diffusion_samplers.swap_sampler`); `schedule` was the dangling half.

Meanwhile the real sigma-schedule machinery already exists and is battle-tested in
the ComfyUI node path: `nodes/eric_qwen_image_multistage.build_sigma_schedule(num_steps,
denoise, schedule)` returns a descending, normalized sigma list whose **spacing**
differs by schedule (linear = uniform; balanced = Karras ρ=3; karras = Karras ρ=7,
concentrating steps at low sigma for fine detail). The UltraGen / multistage nodes
apply it by passing `sigmas=` to the pipeline `__call__` alongside
`num_inference_steps`. Every modern diffusers pipeline we drive (Flux, Flux.2,
Qwen-Image, Chroma, Krea, Z-Image, SDXL) accepts a `sigmas` kwarg.

The mismatch — a CLI flag whose `choices` list implies it works when it silently
does nothing — is the footgun this ADR closes.

## Decision

Wire `--schedule` into `comfyless.generate()` by reusing `build_sigma_schedule`
(no new sigma math), gated to the cases where custom flow-match sigmas are
**correct**, with a loud warn-and-ignore everywhere else (the project's
"warn, don't block on user-initiated footguns" rule).

For full-denoise txt2img (comfyless's only mode), compute
`sigmas = build_sigma_schedule(steps, denoise=1.0, schedule)` and inject
`call_kwargs["sigmas"]` before the pipe call, **only when all hold**:

1. `schedule != "linear"` — linear is the pipeline default; injecting it is a
   no-op, so skip and keep the native path.
2. `is_flow_match(pipe.scheduler)` is True — classic schedulers (DDPM/Euler/DPM/
   DEIS/UniPC on SDXL/SD1) either ignore flow sigmas or produce artifacts.
   Detected via `nodes.eric_diffusion_scheduler.is_flow_match` (class-name based).
3. `"sigmas" in sched.set_timesteps`'s signature — the direct "accepts sigmas"
   property (subsumes the `FlowMatchHeun` special case — its `set_timesteps` has
   no `sigmas` param — and is future-proof against a new flow-match scheduler
   that lacks the kwarg).
4. `"sigmas" in inspect.signature(pipe.__call__).parameters` — defensive; every
   current family passes, but a future pipeline that doesn't must warn-and-skip,
   not crash.

`--sampler` is **orthogonal** and imposes no gate: it sets the integration rule
(Euler vs Adams-Bashforth multistep) while `--schedule` sets the sigma spacing.
The multistep schedulers subclass `FlowMatchEulerDiscreteScheduler` and consume
externally-supplied sigmas verbatim (their `set_timesteps` forwards them to the
parent; `step()` overrides only the integrator), so a spacing composes with any
integrator. The gate evaluates the pre-swap default scheduler, which is a valid
proxy since every swapped-in sampler is a `FlowMatchEuler` subclass that also
accepts sigmas.

When any gate fails and `schedule != "linear"`, print a `WARNING: --schedule <x>
ignored — <reason>; using the pipeline default` to stderr and generate normally.

Because the gate lives inside `generate()`, both the in-process CLI path and the
per-GPU daemon (which calls `generate()`) get it identically; the sidecar already
records `schedule`, so `--params` replay is unchanged.

## Alternatives Rejected

- **Apply the custom sigmas on every family/scheduler.** Simplest, but produces
  visibly wrong output on classic schedulers (the exact bug
  `eric_diffusion_scheduler.py` warns about) and a hard crash on FlowMatchHeun.
  The gate is the point.
- **Leave it reserved and just drop `karras`/`balanced` from the CLI `choices`.**
  Honest, but throws away a feature whose engine already ships; the user wants
  the schedule control the node path has.
- **New sigma implementation in comfyless.** Rejected — `build_sigma_schedule` is
  the reviewed, tested source of truth; duplicating it invites drift.

*(The original ADR also rejected combining `--schedule` with a multistep
`--sampler`, on the mistaken premise that the multistep scheduler owns its own
sigmas. That premise was false and the restriction was lifted — see the
2026-07-13 amendment in the Changelog.)*

## Deferred / Out of Scope

- Partial-denoise / img2img schedules (comfyless is full-denoise txt2img only).
- The `power` parameter of `build_sigma_schedule` (reserved upstream too).

## Changelog

- 2026-07-13 — Initial. Accepted. Wires `--schedule` into `comfyless.generate()`
  via `build_sigma_schedule` behind a four-part flow-match gate; warn-and-ignore
  elsewhere.
- 2026-07-13 (post-review) — code-reviewer (Fable) verified correctness across
  the NAG, Hunyuan-refiner, daemon, and `num_inference_steps`+`sigmas` paths.
  Findings folded before commit:
  - **MEDIUM (N1 boundary):** a warn-and-ignored `--schedule` now records its
    reason in the sidecar `schedule_warnings` and is surfaced client-side by
    `_delegate_to_server` — the daemon's stderr never reaches the client, so
    stderr alone would silently reinstate the recorded-but-not-applied footgun
    over the wire (mirrors `nag_warnings` / `lora_warnings`).
  - **LOW (root cause):** the FlowMatchHeun exclusion is replaced by the direct
    property `"sigmas" in sched.set_timesteps` signature (subsumes Heun,
    future-proof against a new flow-match scheduler lacking sigmas support);
    `is_flow_match` is kept as the correct-interpretation gate.
  - **LOW (testability):** the injection is refactored into the `_apply_sigma_schedule`
    seam and unit-tested (the wiring gap this slice closed is now covered).
  - **DOC (Z-Image exception):** condition 1 says "linear is the pipeline
    default." True for most families, but Z-Image's stock default is a
    model-tuned curve (`get_default_z_image_sigmas`), NOT a uniform linspace. So
    `--schedule linear` on zimage keeps that tuned curve (linear = "use the
    pipeline default," not "force uniform"), and `balanced`/`karras` REPLACE the
    tuned default rather than reshape a linspace. Recorded so it isn't
    re-litigated; behavior is user-opted and warn-don't-block-consistent.
- 2026-07-13 (amendment — multistep restriction lifted) — the original gate
  condition "`effective_sampler == "default"`" was **removed**. Its premise —
  that a multistep `--sampler` generates and owns its own sigmas that would
  collide with `--schedule` — was **wrong**: the multistep schedulers subclass
  `FlowMatchEulerDiscreteScheduler`, override only `step()` (the integrator), and
  their `set_timesteps` accepts external sigmas verbatim (verified: a karras list
  round-trips into `self.sigmas`). Spacing (`--schedule`) and integration order
  (`--sampler`) are orthogonal and now compose — e.g. `--sampler multistep2
  --schedule karras` runs AB2 integration over karras spacing. `test_samplers.py`
  gains a composition test; the gate keeps only the flow-match + sigmas-accepting
  checks. Reviewed by code-reviewer (Fable).

- 2026-07-13 (RES4LYF schedules added) — `beta57` and `bong_tangent` added to
  `SCHEDULE_NAMES` and to the engine `build_sigma_schedule`
  (nodes/eric_qwen_image_multistage.py), so the comfyless `--schedule` path exposes
  them. (The UltraGen/multistage **node dropdowns** are still hardcoded
  linear/balanced/karras — the engine gained the schedules but the ComfyUI node UI
  did not; extending those dropdowns is a deferred follow-up.) Formulas
  **reimplemented from** ClownsharkBatwing/RES4LYF sigmas.py (not copied):
  `beta57` = ComfyUI `beta_scheduler(alpha=0.5, beta=0.7)` — inverse-beta-CDF warp
  of the normalized position (needs scipy, already a pinned dep); this one is
  formula-identical (a test recomputes `beta.ppf(1-t, 0.5, 0.7)` and asserts
  equality). `bong_tangent` = the two-stage arctan S-curve of
  `get_bong_tangent_sigmas` / `bong_tangent_scheduler` (pivot 0.6, slope 0.2),
  **approximated** in the normalized flow-match range: RES4LYF's 60/40 stage split
  and pivot placement are kept, but adapted to build_sigma_schedule's exact-`keep`
  contract (steps_internal = keep+1 replaces RES4LYF's `+2`/`[:-1]` ComfyUI
  bookkeeping; keep<=2 degenerates and falls back to linear so the shared-start
  invariant holds universally). An unknown schedule still falls back to linear.
  `test_multistage.py` +32 (independent beta.ppf recompute, two-stage-midpoint
  check, and small-keep edge cases keep=1..8). Reviewed by code-reviewer (Fable);
  folded the keep==1/2 start-sigma bug and the node-dropdown/faithfulness wording.
  NOTE: `--schedule`/`--sampler` are not yet `--iterate` axes — deferred to a
  follow-up once the res_2m/res_3m samplers land, then wired together.

**AI-Disclosure:** Claude (Fable 5) authored from a design conversation with Grant; Grant reviewed.
