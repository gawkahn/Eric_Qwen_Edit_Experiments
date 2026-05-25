# Vision Slice — Skip `enable_tiling()` for the Hunyuan-Image VAE

**Backlog ref:** `Image_gen/Backlog.md` → Immediate (queued 2026-05-24
alongside the refiner slice; "immediate next" per Grant).

**Suggested branch:** ride the refiner slice's branch, or its own
`hunyuan-tile-vae-skip` worktree — slice is small enough that the
overhead of a separate worktree is debatable.

**Risk level:** **L1** (one-line behavior change per file × 2 files;
gated on `model_family == "hunyuan-image"`; no Red Zone surface; the
"warn, don't block" memory applies if anything).

## Posture

> **Posture:** Boundary: loader behavior (one conditional call) +
> one new CLI flag. Risk factors: small but real impact on the
> `hunyuan-image` runtime VAE memory profile — disabling tiling for
> this family on small-VRAM cards could OOM during decode if the
> 32× VAE actually does need tiling at some output size. Defense:
> explicit `--vae-tiling` flag with a family-aware `auto` default;
> large-VRAM users (the design target) get clean output by default,
> small-VRAM users (24/48 GB) can force tiling on with `--vae-tiling
> on`. Not a security-truth surface.

## Why this slice exists (context)

Both loader paths call `pipeline.vae.enable_tiling()` unconditionally
for every loaded pipeline:

- `nodes/eric_diffusion_loader.py:179-180`
- `comfyless/generate.py:784-785`

Project CLAUDE.md ("Important Constraints" §) says: "`pipeline.vae.enable_tiling()`
is always called on generation pipelines — required for >2 MP decode
without OOM." That reasoning was made for the 8× and 16× compression
VAEs of the families originally supported (SDXL = 8×, Flux = 16×). At
those compression ratios, a 2 MP image's VAE latent is 256×256
(SDXL) or 128×128 (Flux), and decoding without tiling can hit OOM on
modest cards.

Hunyuan-Image 2.1's VAE is **32× compression** (Tencent README §Architecture).
A 2K 2048×2048 generation produces a 64×64 latent — *smaller* than what
the SDXL VAE handles unti­led on a 24 GB card. Tiling for Hunyuan is:
- Unnecessary for memory (32× compression already does the heavy lifting
  upstream).
- Potentially harmful for quality: tile-seam artifacts compound the
  banding diagnosed in the 2026-05-24 ADR-014 amendment. Web-Claude's
  quality diagnosis explicitly flagged tiling as a possible contributor.

## Intent

Make the `enable_tiling()` call **per-family-conditional and
operator-controllable** in both loader paths via a new explicit
`--vae-tiling on|off|auto` flag (default `auto`). Under `auto`:
hunyuan-image gets tiling **off**; every other family gets tiling
**on** (preserves current behavior). `--vae-tiling on` and
`--vae-tiling off` are explicit overrides.

## Invariants (must always be true)

1. **Hunyuan-image default-off under `auto`.** For a pipeline loaded
   with `model_family == "hunyuan-image"` and `--vae-tiling auto`
   (the default), `pipeline.vae.use_tiling` (or the equivalent
   diffusers introspection) is False.
2. **All other families default-on under `auto`.** Every other
   family path (qwen-*, flux*, sdxl, sd*, chroma, auraflow, zimage,
   stablecascade) with `--vae-tiling auto` continues to have
   `enable_tiling()` called as before. Locked at runtime by a
   per-family non-regression test in `test_hunyuan.py` (using the
   same pipeline-class stub pattern from Step 3 of the base slice).
3. **Explicit overrides honored.** `--vae-tiling on` forces tiling on
   regardless of family. `--vae-tiling off` forces tiling off
   regardless of family. The flag is the single, locally-reasoned
   surface for VAE tiling — no implicit coupling with
   `--sequential-offload` or other memory-pressure flags.
4. **ComfyUI node-side parity.** The `Eric Diffusion Load Model`
   node gains a corresponding optional `vae_tiling` input (string:
   `"auto" | "on" | "off"`, default `"auto"`); the resulting
   pipeline state matches the CLI semantics exactly.

## Failure semantics

- **Hunyuan VAE OOM during decode** (unlikely on ≥40 GB cards, possible
  on 24 GB cards): operator sees a CUDA OOM with a stderr hint pointing
  at `--vae-tiling on` as the explicit re-enable. Recoverable by
  re-running with the flag.
- **Invalid `--vae-tiling` value:** argparse-level rejection (`choices`
  list); non-zero exit with usage message. No silent fallback.
- **Pre-existing `enable_tiling()` semantics elsewhere:** untouched.

## Out of scope

- Tile-size tuning for any family (the `enable_tiling()` defaults are
  per-pipeline-class; we're not changing them).
- Per-family memory-management refactor (the existing
  `--sequential-offload` / `--attention-slicing` / `--offload-vae`
  flag surface stays — and stays decoupled from `--vae-tiling`).
- Capability flags in GEN_PIPELINE for "needs-tiling" — too much
  scope; the family-aware `auto` default is enough for v1.
- Validating against the artifact reduction empirically — that's an
  A/B test described in the proof hooks (smoke + visual comparison),
  not part of the slice's logical contract.
- Applying the same `--vae-tiling auto` semantics to a future Hunyuan
  refiner pipeline. The refiner slice (see
  `slice-hunyuan-image-2-1-refiner.md`) inherits this slice's
  family-conditional logic and extends it to the refiner pipeline's
  load path — handled there, not here.

## Proof hooks

**Positive cases** (unit, CPU):

- **Inv 1 — hunyuan auto-off.** Load the existing `HunyuanImagePipeline`
  fixture (already used in Step 2 detect tests) with `vae_tiling="auto"`;
  assert `pipeline.vae.use_tiling` is False. Symmetric assertion for
  both code paths: via `EricDiffusionLoader.load_pipeline()` AND via
  comfyless' `_load_pipeline()`.
- **Inv 2 — non-hunyuan auto-on.** For each existing family
  (qwen-image, qwen-edit, flux*, sdxl, sd*, chroma, auraflow, zimage,
  stablecascade), load via fixture with `vae_tiling="auto"`; assert
  `vae.use_tiling` is True (current behavior). Lock against a future
  edit that mis-orders the conditional and accidentally skips tiling
  for everyone.
- **Inv 3 — explicit overrides.** Two sub-cases:
  - Load `HunyuanImagePipeline` with `vae_tiling="on"`; assert
    `vae.use_tiling` is True (force-on wins over family default).
  - Load a non-hunyuan family (e.g. qwen-image) with
    `vae_tiling="off"`; assert `vae.use_tiling` is False (force-off
    wins over family default).
- **Inv 4 — ComfyUI node parity.** Drive `EricDiffusionLoader` with
  the `vae_tiling` input set to each of `"auto" | "on" | "off"`;
  assert the resulting `vae.use_tiling` matches the CLI semantics
  for the loaded family.

**Negative case:**

- Argparse rejection of `--vae-tiling garbage` (or equivalent invalid
  value); subprocess exits non-zero with usage message on stderr.

**Regression hook:** full 10-suite gate + `test_hunyuan.py` extensions
must continue to pass with 0 failures.

**Live A/B smoke** (separate, outside unit gate, post-host-NVML-fix):

```bash
./.venv/bin/python3 -m comfyless.generate --model /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/HunyuanImage-2.1-Diffusers --prompt "a quiet alpine lake at dawn, photorealistic" --output /tmp/hunyuan-2k-no-tiling.png
```

Side-by-side comparison vs the existing artifacted `/tmp/hunyuan-smoke.png`
(1K, tiling on — the original baseline) AND vs the 2K-with-tiling smoke
recorded in the refiner slice's commit body. Pass criterion: tile-seam
striations in the sky are visibly reduced or absent.

## §12 artifacts required before code

- No new ADR strictly required — this is a small mechanical fix that
  amends ADR-014's family-specific behavior. **However**, it should
  show up as a follow-on Changelog entry on ADR-014 (the same way the
  2026-05-24 amendment did), with a one-paragraph note pointing at the
  Tencent VAE's 32× compression ratio + the web-Claude diagnosis.
- Alternative: small standalone ADR if the reviewer thinks the
  per-family loader-behavior pattern warrants its own anchor (would
  generalize to "any future family that needs different default
  loader behavior").
- Recommended: ADR-014 Changelog entry, not a new ADR.

## Reviewer plan

- `code-reviewer` (Opus, `model: "opus"` at invocation). Small diff;
  reviewer pass on the change plan + diff is enough.
- No `security-auditor`.

## Open questions to settle in the change plan

1. **Diffusers introspection key.** What attribute does
   `AutoencoderKLHunyuanImage` expose to indicate tiling state?
   Probably `vae.use_tiling` (the common pattern) but verify; the
   tests need a reliable accessor.
2. **Flag default sentinel name.** Confirm `auto` as the family-aware
   default sentinel (vs e.g. `default` or empty-string). Lean
   `auto` — matches familiar CLI convention (`--color auto`, etc.).

## Status

- Drafted 2026-05-24, queued as the third immediate-next slice
  alongside the refiner Vision and the just-shipped 2K-defaults
  amendment.
- Awaiting Grant's review + approval. Can ship independently of, or
  in parallel with, the refiner slice — results are stackable (refiner
  fixes structural artifacts, tile-VAE-skip fixes tile-seam artifacts;
  both addressing different contributors to the original Step 5
  artifacted output).
