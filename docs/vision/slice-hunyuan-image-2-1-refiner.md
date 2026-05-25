# Vision Slice — Hunyuan-Image 2.1 base + refiner chaining

**Backlog ref:** `Image_gen/Backlog.md` → Immediate (queued 2026-05-24 as
the direct follow-on to slice `hunyuan-support` — "we don't have a usable
product without it" per Grant).

**Suggested branch:** `hunyuan-refiner` (parallel-session worktree pattern,
same as the 2.1-base slice).

**Risk level:** **L2** (loader + generate-path edits; new dispatch logic;
no Red Zone surface; no PII / auth / billing; *if* the reprompt-model
integration is folded into this slice it becomes **L3** because
`trust_remote_code=True` is a first-time-in-codebase security posture
change — see §"Out of scope" for the recommended split).

## Posture

> **Posture:** Boundary: domain rules (new dispatch path for the Hunyuan
> base+refiner stage pair) + loader machinery (the refiner is a separate
> diffusers pipeline that loads alongside the base and shares text encoders
> at runtime). Risk factors: broad impact on the `hunyuan-image` family's
> behavior (every prior bare-run now runs two stages instead of one — a
> deliberate quality improvement, but a runtime-cost increase the user
> sees as a 2× generation time); near security-truth surface (touches the
> same `resolve_hf_path` + auto-detect codepath covered by the 2026-04-23
> security review, though no behavioral change to the resolver itself);
> *no Red Zone touch* in the recommended scope (reprompt model is
> deliberately out of scope per §"Out of scope" §1).

## Why this slice exists (context)

Slice `hunyuan-support` shipped `HunyuanImagePipeline` as the
`hunyuan-image` family with auto-detection, distilled-guidance CFG
routing, and 2K-native dimension defaults. The 2026-05-24 amendment
(`3638daa`) addressed the 1K-OOD artifact root cause via family-defaults
dim overrides. But the Step 5 live smoke + external diagnosis identified
**a second artifact source the family-defaults amendment can't reach**:
base alone produces visible artifacts (sky banding, foil-textured sails,
hull warping) even at the documented 2K operating point, because Tencent
designed the model as a two-stage pipeline where the refiner explicitly
"further enhances image quality and clarity, while minimizing artifacts"
(Tencent README §Architecture).

ADR-014 §3 originally framed the refiner as "SDXL-style optional polish,
edit-pipeline home." Empirical evidence retracted that framing (ADR-014
2026-05-24 Changelog amendment): refiner is functionally Cascade-coupled
to the base — both stages required for clean output, even though the
data exchanged is images (structurally edit-shape) rather than latents
(structurally Cascade-shape). The architectural home is therefore a
**comfyless dispatch fork** analogous to `comfyless/cascade.py` per
ADR-010, *not* the edit-pipeline surface.

## Intent

Add **base + refiner chaining** as the default execution path for the
`hunyuan-image` family in comfyless. Loading `hunyuan-image` discovers
or accepts an adjacent `HunyuanImageRefinerPipeline` and chains it
automatically after the base; the user sees one `generate` call
producing one PNG, but two stages run under the hood. Failures of the
refiner pipeline fall back to a loud-warning base-only output so the
user is never silently blocked from getting *some* output. Adds the
corresponding ComfyUI node-side support so the unified `Eric Diffusion
Generate` flow also chains.

## Invariants (must always be true)

1. **Auto-discovery.** When `comfyless.generate --model <hunyuan-base-dir>`
   is invoked with no `--refiner` flag, the slice attempts to discover a
   sibling `*-Refiner-Diffusers/` directory (and/or an explicit
   convention-named path). On discovery, the refiner is loaded and the
   two-stage chain runs. On non-discovery, the slice runs base-only AND
   emits a loud stderr warning telling the operator (a) the refiner was
   not found, (b) where it looked, (c) that output quality will be
   degraded, and (d) the exact `huggingface-cli download` command for
   `hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers`.
2. **Explicit override.** `--refiner <path-or-skip>` overrides the
   discovery. `--refiner skip` runs base-only without the warning (power
   user opt-out). `--refiner <path>` uses the named path explicitly.
3. **Output identity.** A chained generation emits exactly one output PNG
   at `--output` (refiner's output, not base's). The PNG `comfyless`
   tEXt chunk carries explicit two-stage metadata: `pipeline:
   "base+refiner"`, base/refiner model_path values (both basenames per
   slice-1 invariant 12 when run from MCP; full paths from CLI / daemon
   per N29 regression guard), `refiner_steps`, `refiner_cfg`. Base-only
   generations carry the existing single-stage metadata shape (the new
   keys are absent, not present-and-empty).
4. **Family-defaults overlay extension.** `FAMILY_DEFAULTS["hunyuan-image"]`
   gains `refiner_steps: 4`, `refiner_cfg: 3.5` (per
   `HunyuanImageRefinerPipeline.__call__` signature defaults). These
   flow through the same ADR-009 precedence ladder (explicit-CLI >
   sidecar > family default > schema default). New canonical schema
   keys (`refiner_steps`, `refiner_cfg`) are added to `COMFYLESS_SCHEMA`
   and validated by the existing `test_params_schema` sweep.
5. **CFG routing parity.** The refiner's `_build_call_kwargs` branch
   matches the base's shape (cfg_scale → distilled_guidance_scale,
   negative_prompt forwarded if set), with the refiner's own
   `refiner_cfg` schema key feeding `distilled_guidance_scale` for the
   refiner call (analogous to how `true_cfg_scale` overrides cfg_scale
   for qwen-image).
6. **No silent regressions on other families.** The new dispatch fork
   only activates when `model_family == "hunyuan-image"`; every other
   family path (qwen-*, flux*, sdxl, sd*, chroma, auraflow, zimage,
   stablecascade) behaves identically pre- and post-slice. Locked at
   runtime by the regression sweep in `test_hunyuan.py` (which already
   spans 11 existing families).
7. **No new MCP exposure.** This slice does NOT plumb `--refiner` /
   `refiner_*` params through the MCP tool surface in this commit batch;
   that's a separate slice with its own security review per ADR-011 §3d
   ordering. The MCP `generate` tool continues to call into
   `comfyless.generate.generate()` with its existing argument shape;
   the new refiner kwargs default to "discover or warn" as in the CLI.
8. **Memory ceiling.** A bare run with all three models cached
   (reprompt would be 14 GB, base ~58 GB, refiner +20-25 GB delta over
   shared encoders → peak ~80 GB *without* reprompt; this slice
   excludes reprompt so peak is ~80 GB) fits within a single RTX PRO
   6000 (102 GB) without `--sequential-offload` or balanced device_map.
   24/48 GB cards still need `--sequential-offload` or balanced mode;
   the slice does not change the existing offload-flag surface, so
   smaller-card support continues to work via the existing flags.

## Failure semantics

- **Refiner discovery miss + no `--refiner skip`:** loud stderr warning
  + base-only run + non-zero exit code? *Reject — base-only with warning
  + zero exit* is the right behavior (the operator gets *an* image; the
  warning makes the regression explicit). This matches the
  "warn-don't-block on user-initiated footguns" memory.
- **Refiner load error (corrupt weights, missing component):** loud
  stderr error citing which load step failed; base-only fallback with
  loud warning; zero exit. Same posture.
- **Refiner inference error mid-generation (OOM, timeout):** propagate
  the error, but BEFORE propagation, emit a stderr line naming the
  stage so an LLM/MCP caller can distinguish "base failed" from
  "refiner failed." Non-zero exit.
- **`--refiner skip` on a `hunyuan-image` run:** silent base-only run.
  Equivalent to other families' behavior; documented opt-out for power
  users who don't want refiner.
- **Refiner family-defaults missing:** same as base — `_apply_family_defaults`
  short-circuits gracefully, schema defaults (`refiner_steps` / `refiner_cfg`)
  carry the run.
- **Cross-family auto-discovery confusion:** if a user points the loader
  at a non-Hunyuan model that happens to have a `*-Refiner-Diffusers/`
  sibling (e.g. some hypothetical future Flux refiner), the refiner code
  path remains gated on `model_family == "hunyuan-image"` so the sibling
  is ignored for non-Hunyuan families. Locked by invariant 6 + an
  invariant-6 negative test.

## Out of scope (explicit exclusions)

1. **Tencent reprompt model integration.** The bundled
   `tencent/HunyuanImage-2.1/reprompt/` directory contains a
   `HunYuanDenseV1ForCausalLM` (~7B params, ~14 GB bf16, custom
   tokenizer, `trust_remote_code=True` required). Per Grant's
   2026-05-24 direction the trust_remote_code stance has a viable
   "review-and-pin" posture from the abandoned Hunyuan-3 project that
   makes it usable here — **but adopting `trust_remote_code=True` as a
   first-class codebase capability is its own ADR and security review
   per global §5/§12** and would push this slice from L2 to L3 with a
   `security-auditor` invocation required on each code-touching commit.
   Splitting it out keeps the refiner slice clean L2 and lets the
   reprompt slice carry its own security artifact trail. Recommended
   order: refiner ships first → reprompt second (operator can hand-write
   a longer structured prompt in the interim, or use the existing
   `Eric Qwen Prompt Rewriter` API-LLM node pattern with a
   Hunyuan-flavored system prompt).
2. **VAE-tiling skip for hunyuan-image.** Adjacent quality fix
   (web-Claude diagnosis identified `pipeline.vae.enable_tiling()` —
   called unconditionally at `nodes/eric_diffusion_loader.py:179-180`
   and `comfyless/generate.py:784-785` — as potentially compounding the
   sky-banding via tile seams; for Hunyuan's 32× VAE on ≥100 GB GPUs
   tiling is also unnecessary for memory). Its own small Vision
   (`slice-hunyuan-image-tile-vae-skip.md`); could ship before, after,
   or in parallel with this refiner slice; results are stackable.
3. **MCP tool refiner exposure.** Wiring `refiner_*` params through the
   MCP `generate` tool's request schema needs its own security review
   per ADR-011 §3d; out of this slice's scope. The MCP generate path
   continues to call into the same `comfyless.generate.generate()`
   function and inherits the auto-discover behavior automatically.
4. **ComfyUI advanced multistage / ultragen integration.** This slice
   adds chained refiner support to the unified `Eric Diffusion Generate`
   node (basic generate); the advanced multistage / ultragen nodes
   (which have their own per-stage cfg/steps/sampler/denoise machinery)
   are out of scope. If a future slice wants the refiner pass to
   participate in multistage workflows, that's its own integration ADR.
5. **ControlNet variants of the refiner.** Not in scope.
6. **HunyuanImage-3.0** continues to live with Eric's
   `Comfy_HunyuanImage3` nodes per Backlog 2026-05-17. The pattern
   established here (comfyless dispatch fork for two-stage Hunyuan
   pipelines) may inform a future ai-stack-project slice that wraps
   Eric's nodes, but is out of scope here.
7. **`comfyless/server.py` daemon integration.** The IPC daemon path
   doesn't see the new refiner params yet — it continues to forward
   what the existing wire protocol carries, and the daemon's
   in-process `generate()` call inherits the auto-discover behavior.
   Explicit refiner-aware wire fields are a follow-up slice with its
   own ADR-001 amendment.

## Proof hooks

All `test_hunyuan.py` extensions run on CPU using fixture
`model_index.json` files for both base and refiner pipelines, with the
refiner pipeline class either real (if diffusers ships it cleanly
without instantiation cost) or stubbed (preferred — keep the unit
gate CPU-only).

**Positive cases** (one per invariant):

- **Inv 1 — auto-discovery.** Two fixture dirs (base + sibling
  `*-Refiner-Diffusers/`) → `_resolve_refiner_path` returns the sibling
  path. Base only → returns None + emits the warning. Three sub-cases
  (sibling present / sibling absent / explicit `--refiner skip`).
- **Inv 2 — explicit override.** `--refiner /some/path` wins over
  auto-discovery; `--refiner skip` opts out without warning;
  `--refiner /nonexistent/path` raises clean ValueError (fail-closed).
- **Inv 3 — output identity.** Mock PIL image through the two-stage
  pipeline; assert exactly one PNG written; assert metadata chunk
  carries `pipeline: "base+refiner"` + the new keys.
- **Inv 4 — defaults overlay extension.** Same shape as the existing
  Inv 3 of `test_hunyuan.py` but for `refiner_steps` / `refiner_cfg`.
- **Inv 5 — CFG routing parity.** `_build_call_kwargs` extended branch
  for the refiner; assert `distilled_guidance_scale` shape mirrors base.
- **Inv 6 — non-regression.** Re-run the existing 11-family sweep;
  assert refiner code path is gated on family == "hunyuan-image".
- **Inv 7 — MCP path unchanged.** `test_mcp_server` continues to pass
  with the same call-site shape (the MCP `generate` tool inherits the
  new auto-discover behavior without any new request-schema fields).
- **Inv 8 — memory ceiling claim.** This invariant is empirically
  validated by the live smoke (single-GPU base+refiner run completes
  without OOM); no CPU unit test.

**Negative cases:**

- Inv 1 negative — sibling dir exists but is NOT a refiner pipeline (wrong
  `_class_name`): clean error, fall through to base-only with warning.
- Inv 2 negative — `--refiner /path/to/non-hunyuan-pipeline` (e.g.
  pointing at Flux): clean error citing incompatible pipeline.
- Inv 3 negative — refiner inference error (synthetic exception
  injection): error propagated; stderr names the stage; non-zero exit.
- Inv 6 negative — load a non-Hunyuan model with a sibling
  `*-Refiner-Diffusers/` dir; assert refiner code path NOT entered.

**Regression hook** — full 10-suite gate + `test_hunyuan.py` extensions
must continue to pass with 0 failures. Plus `test_mcp_server` re-validation
(invariant 7).

**Live GPU smoke** — empirical proof of quality improvement:

```bash
./.venv/bin/python3 -m comfyless.generate --model /home/gawkahn/projects/ai-lab/ai-base/models/hf-local/HunyuanImage-2.1-Diffusers --prompt "a quiet alpine lake at dawn, photorealistic" --output /tmp/hunyuan-base-plus-refiner-smoke.png
```

Pass criterion: file written, PNG metadata carries `pipeline:
"base+refiner"`, side-by-side comparison vs the
`/tmp/hunyuan-smoke.png` from the original slice's Step 5 (or vs a
fresh base-only run via `--refiner skip`) shows visibly reduced
artifacts (sky banding, sail wrinkles, hull warping).

## §12 artifacts required before code

- **`ADR-016-hunyuan-image-base-refiner-chain.md`** (next ADR number;
  verify ADR-015 is the most recent at slice start). Documents:
  (a) the dispatch shape — auto-discover-and-chain in `comfyless/`
  (Cascade-pattern analog to `comfyless/cascade.py`) vs explicit
  two-step pipeline; (b) where the refiner pipeline lives in the
  loader machinery (separate cache slot? share the GEN_PIPELINE cache
  with a refiner-aware field?); (c) the `--refiner` flag semantics
  (auto / explicit / skip); (d) the discovery convention (sibling
  `*-Refiner-Diffusers/` only, or also a checked HF-cache fallback);
  (e) defaults values + their sourcing; (f) shared-text-encoders
  optimization (refiner pipeline loads should share `text_encoder` /
  `text_encoder_2` instances with the base pipeline when possible);
  (g) the metadata-chunk schema extension (new keys, what they mean);
  (h) confirmation that this slice does NOT make `trust_remote_code`
  changes (reprompt is a separate slice) — keeps reviewer plan at L2;
  (i) explicit reference to ADR-014's 2026-05-24 §3 amendment that
  motivated this slice.
- **Security review:** NOT required at L2 (no Red Zone surface
  touch, no `resolve_hf_path` / `_run_json_mode` / `comfyless/server.py`
  edit, no IPC change, no caller-supplied-path widening beyond the
  existing `--model` surface — `--refiner` adds a sibling path
  semantics gated by the same `resolve_hf_path` resolver). **If during
  implementation any of those surfaces unexpectedly need to be touched
  (e.g. the refiner needs a new caller-supplied path widening for its
  separate text encoder), STOP and re-evaluate per Vision §Reviewer-plan.**

## Reviewer plan

- **`code-reviewer` (Opus, `model: "opus"` at invocation per global
  §5A and the broken-frontmatter workaround)** — run after each
  non-trivial slice step, before commit. Non-negotiable.
- **`security-auditor`** — not invoked for this slice (matches the
  original Hunyuan-base slice's plan). If the reprompt-model
  integration is folded in at any point, this changes to
  `security-auditor` on every code-touching commit per ADR-013 §8
  trailing-note's spirit applied to security-posture changes (not just
  pin movement) — and the slice should be split first.
- **ADR-013 §8 trailing-note check:** this slice's success depends on
  the `hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers`
  package being installable against the existing `diffusers==0.37.1`
  pin. **If the refiner pipeline requires a diffusers bump, the
  trailing-note triggers and `security-auditor` layers onto every
  code-touching commit.** Verify at slice start via `from diffusers
  import HunyuanImageRefinerPipeline` on the existing pin (we already
  confirmed `HunyuanImageRefinerPipeline` is exported by
  `diffusers/__init__.py` in the original slice's ADR-014 §3 audit —
  high confidence no bump needed, but verify before assuming).

## Open questions to settle in the ADR

1. **Sibling-discovery convention.** Single fixed convention
   (`<base-dir>-Refiner-Diffusers/` next to `<base-dir>/`)? Or
   multi-strategy (sibling, HF-cache lookup for
   `hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers`, env
   var override)? Recommend: sibling-only as the v1, with explicit
   `--refiner` as the escape hatch for everything else. Keeps the
   loader simple.
2. **Shared text-encoder optimization.** Both pipelines need Qwen2.5-VL
   + T5 text encoders. Naively loading both pipelines independently
   doubles encoder VRAM. Cleanest fix: load the base pipeline first,
   construct the refiner pipeline by injecting the base's text
   encoders + tokenizers + scheduler at `from_pretrained(...,
   text_encoder=base.text_encoder, text_encoder_2=base.text_encoder_2,
   tokenizer=base.tokenizer, tokenizer_2=base.tokenizer_2)`. Saves
   ~24 GB VRAM. Verify the refiner pipeline class accepts these as
   constructor args.
3. **Refiner stage's input shape.** `HunyuanImageRefinerPipeline.__call__`
   takes `image: PipelineImageInput | None`. Is that a PIL image or a
   pre-decoded latent? If PIL, we VAE-encode the base output and feed
   it to the refiner — but the refiner uses a DIFFERENT VAE class
   (`AutoencoderKLHunyuanImageRefiner`). Need to confirm:
   re-encode-via-refiner-VAE? or skip the VAE roundtrip entirely (some
   refiners accept latent tensors directly via an `image_latents`
   kwarg)? Inspect the refiner pipeline's `__call__` body during ADR.
4. **PNG metadata schema versioning.** Adding new keys (`pipeline`,
   `refiner_steps`, `refiner_cfg`, refiner model_path) to the existing
   `comfyless` tEXt chunk. Backward compatibility: existing
   sidecar-replay (`--params <prior-png>`) on a pre-refiner image:
   missing keys default to base-only behavior (good). On a
   refiner-aware image replayed by a pre-refiner comfyless build: the
   build ignores unknown keys (good). Confirm both behaviors in tests.
5. **ComfyUI generate-node integration.** Add a `refiner_path` /
   `refiner_skip` input to the unified `Eric Diffusion Generate` node?
   Or rely on auto-discover only on the ComfyUI side? Cleanest: add a
   single optional `refiner_path` string input (default: empty =
   auto-discover; sentinel "skip" = base-only; any other value =
   explicit path).
6. **Smoke command in the Vision proof hooks** assumes auto-discover
   finds the sibling. If the smoke is run from a fresh host where the
   refiner isn't yet downloaded, the warning fires + base-only runs.
   Document the download command as part of the Vision's smoke setup
   so operators don't get surprised.

## Status

- Drafted 2026-05-24 as the immediate-next slice after the
  `hunyuan-support` 2026-05-24 amendment (`3638daa`).
- Awaiting Grant's review + approval before `/change-slice` →
  ADR-016 draft.
- Next action after approval: download the
  `hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers` weights
  (~20-25 GB delta over base; user-side step) and verify the diffusers
  pipeline class is importable on the existing pin.
