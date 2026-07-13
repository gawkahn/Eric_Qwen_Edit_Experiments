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
2. `effective_sampler == "default"` — a non-default `--sampler` swaps in a
   multistep scheduler that generates and owns its *own* sigmas
   (`eric_diffusion_samplers` passes `sigmas=` internally); two sigma sources
   would collide. `--sampler` (non-default) + `--schedule` (non-linear) →
   schedule ignored with a warning.
3. `is_flow_match(pipe.scheduler)` is True **and** it is not a
   `FlowMatchHeun*` scheduler. Classic schedulers (DDPM/Euler/DPM/DEIS/UniPC on
   SDXL/SD1) either ignore flow sigmas or produce artifacts; `FlowMatchHeun`
   explicitly rejects the `sigmas=` argument. Detected via
   `nodes.eric_diffusion_scheduler.is_flow_match` (class-name based).
4. `"sigmas" in inspect.signature(pipe.__call__).parameters` — defensive; every
   current family passes, but a future pipeline that doesn't must warn-and-skip,
   not crash.

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
- **Also thread it through the multistep `--sampler` path.** Combining a custom
  spacing with a multistep sampler's own sigma generation is a separate, larger
  design; deferred. v1 makes them mutually exclusive with a warning.
- **New sigma implementation in comfyless.** Rejected — `build_sigma_schedule` is
  the reviewed, tested source of truth; duplicating it invites drift.

## Deferred / Out of Scope

- `--schedule` combined with a non-default `--sampler` (multistep). Trigger: a
  concrete need for Karras spacing under a multistep sampler.
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

**AI-Disclosure:** Claude (Fable 5) authored from a design conversation with Grant; Grant reviewed.
