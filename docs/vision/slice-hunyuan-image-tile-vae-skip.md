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

> **Posture:** Boundary: loader behavior (one conditional call). Risk
> factors: small but real impact on the `hunyuan-image` runtime VAE
> memory profile — disabling tiling for this family on small-VRAM
> cards could OOM during decode if the 32× VAE actually does need
> tiling at some output size. Defense: per-family skip rather than
> universal disable; large-VRAM users (the design target) get clean
> output, small-VRAM users (24/48 GB) can still opt back into tiling
> via existing flags if needed. Not a security-truth surface.

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

Make the `enable_tiling()` call **per-family-conditional** in both loader
paths. For `hunyuan-image`, skip the call by default. All other families
behave identically to today (unconditional `enable_tiling()`).

## Invariants (must always be true)

1. **Hunyuan-image skip.** For a pipeline loaded with
   `model_family == "hunyuan-image"`, `pipeline.vae.tile_enabled` (or
   the equivalent diffusers introspection) is False unless the user
   explicitly opted in.
2. **All other families unchanged.** Every other family path
   (qwen-*, flux*, sdxl, sd*, chroma, auraflow, zimage, stablecascade)
   continues to have `enable_tiling()` called as before. Locked at
   runtime by a per-family non-regression test in `test_hunyuan.py`
   (using the same pipeline-class stub pattern from Step 3).
3. **Opt-in escape hatch.** If a future user encounters Hunyuan tiling
   needs (e.g. a 24 GB card trying to fit base + refiner), they can
   re-enable via an existing CLI flag OR a new explicit one — *decide
   in the change plan*. Recommended: piggyback on the existing
   `--attention-slicing` / `--sequential-offload` flags (they're
   memory-pressure indicators) by treating `--sequential-offload` as
   the signal to keep VAE tiling on for hunyuan-image. Avoids adding a
   new CLI flag for a corner case.
4. **No new CLI surface widening** in v1 unless invariant 3's
   piggyback isn't viable; the slice resolves which during the change
   plan.

## Failure semantics

- **Hunyuan VAE OOM during decode** (unlikely on ≥40 GB cards, possible
  on 24 GB cards if user disabled offload): operator sees a CUDA OOM
  with a stderr hint pointing at the `--sequential-offload` flag (which
  per invariant 3 also keeps tiling on for hunyuan-image). Recoverable
  by re-running with the flag.
- **Pre-existing `enable_tiling()` semantics elsewhere:** untouched.

## Out of scope

- Tile-size tuning for any family (the `enable_tiling()` defaults are
  per-pipeline-class; we're not changing them).
- Per-family memory-management refactor (the existing
  `--sequential-offload` / `--attention-slicing` / `--offload-vae`
  flag surface stays).
- Capability flags in GEN_PIPELINE for "needs-tiling" — too much
  scope; the per-family conditional is enough for v1.
- Validating against the artifact reduction empirically — that's an
  A/B test described in the proof hooks (smoke + visual comparison),
  not part of the slice's logical contract.

## Proof hooks

**Positive cases** (unit, CPU):

- `test_hunyuan.py` extension: load the existing `HunyuanImagePipeline`
  fixture path (already used in the Step 2 detect tests), call the
  loader entry-point, assert the resulting pipeline's
  `vae.use_tiling` (or `vae._tiling_enabled` — whatever the diffusers
  introspection key is) is False.
- Symmetric assertion for both code paths: load via
  `EricDiffusionLoader.load_pipeline()` AND via comfyless'
  `_load_pipeline()`; both leave tiling off for hunyuan-image.

**Non-regression sweep:**

- For each existing family (qwen-image, qwen-edit, flux*, sdxl, sd*,
  chroma, auraflow, zimage), load via a fixture and assert
  `vae.use_tiling` is True (current behavior). Lock against a future
  edit that mis-orders the conditional and accidentally skips tiling
  for everyone.

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
2. **Piggyback vs new CLI flag (invariant 3).** Confirm
   `--sequential-offload` as the re-enable signal, or add an explicit
   `--vae-tiling` boolean. Lean toward piggyback; "warn, don't block"
   memory applies to small-card users.
3. **Should the skip extend to the future Hunyuan refiner pipeline
   too?** Almost certainly yes (same 32× VAE class shape, same
   reasoning). Handle when the refiner slice lands; for now the
   Hunyuan-image skip is the immediate need.

## Status

- Drafted 2026-05-24, queued as the third immediate-next slice
  alongside the refiner Vision and the just-shipped 2K-defaults
  amendment.
- Awaiting Grant's review + approval. Can ship independently of, or
  in parallel with, the refiner slice — results are stackable (refiner
  fixes structural artifacts, tile-VAE-skip fixes tile-seam artifacts;
  both addressing different contributors to the original Step 5
  artifacted output).
