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

> **Posture:** Boundary: domain rules (new opt-in dispatch path for the
> Hunyuan base+refiner stage pair) + loader machinery (the refiner is a
> separate diffusers pipeline that loads alongside the base and shares
> the Qwen2.5-VL text encoder at runtime) + IPC daemon wire-protocol
> extension (`comfyless/server.py` gains an optional `refiner` field for
> cached two-stage runs). Risk factors: meaningful impact on the
> `hunyuan-image` family's behavior when `--refiner` is set (two-stage
> run, ~2× generation time, ~80 GB peak VRAM with shared encoders); near
> security-truth surface (touches `comfyless/server.py` — already a §12
> surface — and reuses the existing `resolve_hf_path` resolver for the
> new path). **No path-derivation auto-discovery** (deliberately rejected
> to avoid widening the path attack surface; see Intent + Invariants 1).
> *No Red Zone touch* (reprompt model is deliberately out of scope per
> §"Out of scope" §1).

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

Add **opt-in base + refiner chaining** for the `hunyuan-image` family
in comfyless via an explicit `--refiner <path>` flag (and a matching
ComfyUI node input). When `--refiner <path>` is set, the named refiner
pipeline loads alongside the base and runs as a second stage; the user
sees one `generate` call producing one PNG with two-stage metadata.
When `--refiner` is unset on a `hunyuan-image` run, the slice emits a
**loud stderr warning** ("hunyuan-image quality requires a refiner;
pass `--refiner <path>`; download with `huggingface-cli download
hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers`") and runs
base-only with zero exit code (matches the project's "warn, don't
block on user-initiated footguns" memory). **No filesystem search for
sibling refiner directories** — operator must point at the refiner
path explicitly. Adds IPC daemon support (cache-aware two-stage runs)
so the operational value of `--serve` scales with the doubled model
load.

## Invariants (must always be true)

1. **Opt-in only — no filesystem search.** The slice MUST NOT derive
   or stat sibling/parent/glob paths from `--model` to find a refiner.
   The only way to enable the refiner stage is to set `--refiner <path>`
   explicitly. Rationale: path-derivation on a caller-supplied input
   widens the security surface (TOCTOU, containment escape, symlink
   traversal interactions with the base path) — the same class of
   concern `lora_audit.py` had to defend against. Locked at runtime
   by a negative test asserting no `os.listdir` / `Path.glob` /
   `Path.iterdir` calls are made against `--model`'s parent during a
   bare `hunyuan-image` invocation.
2. **Warn-don't-block on hunyuan-image without `--refiner`.** When
   `model_family == "hunyuan-image"` and `--refiner` is unset, the
   slice emits a loud stderr warning (text per Intent) and runs
   base-only with zero exit code. Operator gets *an* image; the warning
   makes the quality regression explicit. Matches the
   `feedback_warn_dont_block` memory.
3. **`--refiner <path>` is the single enable.** No `--refiner skip`
   sentinel, no `--refiner auto`, no envvar override. The flag is
   either unset (base-only + warning per invariant 2) or set to a
   resolvable path (refiner stage runs). Path resolution goes through
   the same `resolve_hf_path` machinery that handles `--model` — no
   new resolver code.
4. **Output identity.** A chained generation emits exactly one output
   PNG at `--output` (refiner's output, not base's). The PNG
   `comfyless` tEXt chunk carries explicit two-stage metadata:
   `pipeline: "base+refiner"`, refiner model_path value (basename per
   slice-1 invariant 12 when run from MCP; full path from CLI / daemon
   per N29 regression guard), plus the effective `refiner_steps` and
   `refiner_cfg` values for the run. Base-only generations carry the
   existing single-stage metadata shape unchanged (the new keys are
   absent, not present-and-empty).
5. **Family-defaults overlay extension.** `FAMILY_DEFAULTS["hunyuan-image"]`
   gains `refiner_steps: 4`, `refiner_cfg: 3.5` (per the **Tencent
   refiner README** — diffusers signature default for cfg is 3.25 but
   the README is authoritative, same lesson as the 2K-mandatory
   amendment). These flow through the same ADR-009 precedence ladder
   (explicit-CLI > sidecar > family default > schema default). New
   canonical schema keys (`refiner_steps`, `refiner_cfg`) are added to
   `COMFYLESS_SCHEMA` and validated by the existing `test_params_schema`
   sweep. Both keys are no-ops when `--refiner` is unset.
6. **CFG routing parity.** A new refiner-side call-kwargs branch
   matches the base's shape (cfg_scale → distilled_guidance_scale,
   negative_prompt forwarded if set), with the refiner's own
   `refiner_cfg` schema key feeding `distilled_guidance_scale` for the
   refiner call (analogous to how `true_cfg_scale` overrides cfg_scale
   for qwen-image).
7. **LoRAs apply to base only.** The existing `--lora` machinery loads
   into the base pipeline's transformer. The refiner has a separate
   transformer with separate weights — base LoRAs would not produce
   meaningful output on it. The slice MUST NOT call any LoRA loader
   against the refiner pipeline. Locked at runtime by a negative test
   asserting the refiner pipeline's transformer has no PEFT adapter
   attached after a chained run with base LoRAs set. v1 ships no
   refiner-side LoRA surface; a future slice can add `--refiner-lora`
   if a use case emerges.
8. **Scheduler / sampler / sigmas pinned per-pipeline.** The refiner
   uses its own loaded scheduler config from disk
   (`FlowMatchEulerDiscreteScheduler` instance from the refiner
   checkpoint). The slice MUST NOT mutate the refiner's scheduler or
   apply base-side `--sampler` / `--sigmas` swaps to it. v1 ships no
   `--refiner-sampler` / `--refiner-sigmas` flags.
9. **Shared text encoder (memory optimization, asymmetric).** The
   refiner pipeline class only declares `text_encoder` (Qwen2.5-VL) —
   it has no T5/`text_encoder_2` slot. Construction MUST inject the
   base's loaded `text_encoder` and `tokenizer` into the refiner's
   `from_pretrained(...)` call to avoid double-loading the ~14 GB VL
   encoder. The base's T5/ByT5 stack is not relevant to the refiner
   and is not shared. Locked at runtime by an assertion that
   `id(base.text_encoder) == id(refiner.text_encoder)` after load.
10. **No silent regressions on other families.** The new dispatch fork
    only activates when `model_family == "hunyuan-image"` AND
    `--refiner` is set; every other family path (qwen-*, flux*, sdxl,
    sd*, chroma, auraflow, zimage, stablecascade) behaves identically
    pre- and post-slice. Locked at runtime by the regression sweep in
    `test_hunyuan.py` (which already spans 11 existing families).
11. **IPC daemon support — wire-protocol extension.** `comfyless/server.py`
    gains an optional `refiner` field in its request payload. When
    present and non-empty, the daemon forwards it to the in-process
    `generate()` call. The pipeline cache key is extended to include
    the refiner path (or `None`) so a base+refiner request does not
    collide with a base-only request for the same `--model`. Daemon
    behavior for clients that omit the field is byte-for-byte
    identical to today (additive field, not breaking).
12. **No new MCP exposure.** This slice does NOT plumb `--refiner` /
    `refiner_*` params through the MCP tool surface in this commit
    batch; that's a separate slice with its own security review per
    ADR-011 §3d ordering. The MCP `generate` tool continues to call
    into `comfyless.generate.generate()` with its existing argument
    shape; `refiner` defaults to unset (base-only + warning) for MCP
    callers.
13. **Memory ceiling.** A chained run with shared Qwen2.5-VL encoder
    (~14 GB shared instead of ~28 GB doubled) plus base transformer
    + base T5 + refiner transformer + refiner VAE fits within a
    single RTX PRO 6000 (102 GB) at ~80 GB peak without
    `--sequential-offload` or balanced device_map. 24/48 GB cards
    still need `--sequential-offload` or balanced mode; the slice
    does not change the existing offload-flag surface, so smaller-card
    support continues to work via the existing flags.

## Failure semantics

- **`hunyuan-image` run with `--refiner` unset:** loud stderr warning
  (text per Intent) + base-only run + zero exit code. Operator gets
  *an* image; the warning makes the quality regression explicit.
  Matches the `feedback_warn_dont_block` memory.
- **`--refiner <path>` points at a nonexistent or unresolvable path:**
  fail fast — same shape as `--model` resolution failure (clean
  ValueError citing the unresolved path; non-zero exit). The opt-in
  signal was explicit; the user wants the refiner, so a silent
  fallback would mask the misconfiguration.
- **`--refiner <path>` points at a non-`HunyuanImageRefinerPipeline`
  pipeline** (wrong `_class_name`, e.g. a base pipeline by mistake):
  clean error citing incompatible pipeline class; non-zero exit.
  Refiner is opt-in and class-checked; no silent fallback.
- **`--refiner` set on a non-hunyuan family** (e.g.
  `--model <flux-dir> --refiner <hunyuan-refiner-dir>`): clean error
  citing that refiner chaining is only supported for
  `model_family == "hunyuan-image"`; non-zero exit. Locked by
  invariant 10.
- **Refiner load error mid-load** (corrupt weights, missing
  component): clean error citing which load step failed; non-zero
  exit. No base-only fallback — the operator opted in, so a silent
  fallback would mask the breakage.
- **Refiner inference error mid-generation** (OOM, CUDA error, etc.):
  propagate the error, but BEFORE propagation emit a stderr line
  naming the stage (`refiner` not `base`) so an LLM/MCP caller can
  distinguish "base failed" from "refiner failed." Non-zero exit.
- **Refiner family-defaults missing or partial:**
  `_apply_family_defaults` short-circuits gracefully; schema defaults
  (`refiner_steps`, `refiner_cfg`) carry the run.
- **Daemon-mode cache collision risk:** the wire-protocol `refiner`
  field is included in the cache key (invariant 11); a missing /
  empty field is treated as `None`, distinct from any non-empty
  path. A request that omits the field on a hot daemon previously
  serving base+refiner does not get the cached two-stage pipeline
  back — it triggers a fresh base-only load.

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
7. **Refiner-side LoRA / sampler / sigmas / scheduler-swap surface.**
   No `--refiner-lora`, `--refiner-sampler`, `--refiner-sigmas`, or
   refiner-side scheduler-swap flags in v1 (invariants 7 and 8). The
   refiner runs with its own loaded scheduler and no adapters. Add
   only when a concrete use case lands.

## Proof hooks

All `test_hunyuan.py` extensions run on CPU using fixture
`model_index.json` files for both base and refiner pipelines, with the
refiner pipeline class either real (if diffusers ships it cleanly
without instantiation cost) or stubbed (preferred — keep the unit
gate CPU-only).

**Positive cases** (one per invariant):

- **Inv 1 — no filesystem search.** Monkeypatch `os.listdir`,
  `Path.glob`, `Path.iterdir`, and `Path.exists` with spies; run a
  bare `comfyless.generate --model <hunyuan-base-dir>` (no
  `--refiner`); assert the spies are not called against the base
  dir's parent or against any sibling-derived path. Locks the
  "no path-derivation" invariant at runtime.
- **Inv 2 — warn-don't-block.** Bare `hunyuan-image` run with
  `--refiner` unset writes a single PNG with single-stage metadata
  AND emits the documented stderr warning AND returns zero exit code.
- **Inv 3 — opt-in via path.** `--refiner <fixture-refiner-dir>`
  loads and chains. `--refiner` flag absent (already covered by
  Inv 2). No third sub-case — there is no `skip` sentinel to test.
- **Inv 4 — output identity.** Mock PIL image through the two-stage
  pipeline; assert exactly one PNG written; assert metadata chunk
  carries `pipeline: "base+refiner"`, the refiner model_path
  (basename or full per N29), and the effective `refiner_steps` /
  `refiner_cfg` values. Base-only run carries no `pipeline` key.
- **Inv 5 — defaults overlay extension.** Same shape as the existing
  Inv 3 of `test_hunyuan.py` but for `refiner_steps` / `refiner_cfg`.
  Schema-key collision check against existing `COMFYLESS_SCHEMA`.
- **Inv 6 — CFG routing parity.** Refiner-side `_build_call_kwargs`
  branch; assert `distilled_guidance_scale` shape mirrors base.
- **Inv 7 — LoRAs not applied to refiner.** Chained run with
  `--lora <fixture-lora>`; assert base pipeline's transformer has a
  PEFT adapter attached AND refiner pipeline's transformer does not.
- **Inv 8 — scheduler/sigmas pinned.** Chained run with base-side
  `--sampler` swap (if the family supports one); assert refiner's
  `scheduler` instance is the one loaded from disk, untouched by
  base-side mutations.
- **Inv 9 — shared text encoder.** After construction,
  `id(base.text_encoder) == id(refiner.text_encoder)`; refiner has
  no `text_encoder_2` slot.
- **Inv 10 — non-regression on other families.** Re-run the existing
  11-family sweep; assert refiner code path is gated on
  `family == "hunyuan-image" AND --refiner set`. Plus an extra case:
  `--model <non-hunyuan>` with `--refiner <path>` set — should
  raise (locked by §"Failure semantics").
- **Inv 11 — daemon wire protocol.** Send an IPC request with the
  new `refiner` field set; assert it reaches `generate()` and the
  cache key reflects both `(model, refiner)`. Then send a follow-up
  request omitting the field; assert it hits a fresh load (different
  cache key) rather than reusing the two-stage pipeline.
- **Inv 12 — MCP path unchanged.** `test_mcp_server` continues to
  pass with the same call-site shape (the MCP `generate` tool sees
  `refiner` default to unset; no new request-schema fields exposed).
- **Inv 13 — memory ceiling claim.** Empirically validated by the
  live smoke (single-GPU base+refiner run completes without OOM);
  no CPU unit test.

**Negative cases:**

- `--refiner /nonexistent/path`: clean ValueError; non-zero exit
  (no silent fallback). Counterpart to invariant 3.
- `--refiner /path/to/non-hunyuan-pipeline` (e.g. Flux): clean error
  citing incompatible pipeline class; non-zero exit.
- Refiner inference error (synthetic exception injection): error
  propagated; stderr names `refiner` stage; non-zero exit.
- `--model <non-hunyuan> --refiner <hunyuan-refiner>`: clean error;
  non-zero exit.
- Base LoRA applied → refiner transformer adapter count = 0
  (invariant 7 negative).
- IPC `refiner` field set to garbage type (non-string): wire-format
  rejection; non-zero protocol error.

**Regression hook** — full 10-suite gate + `test_hunyuan.py`
extensions + `test_server_robustness` + `test_mcp_server` must
continue to pass with 0 failures.

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
  (a) the dispatch shape — opt-in refiner stage in `comfyless/`,
  driven by `--refiner <path>`, no path derivation, no auto-discovery;
  (b) where the refiner pipeline lives in the loader machinery
  (separate cache slot keyed on `(model, refiner)`); (c) the
  `--refiner <path>` flag semantics (single enable, no `skip`/`auto`
  sentinels; opt-in only) and the warning-on-unset behavior;
  (d) defaults values + their sourcing (Tencent refiner README
  cfg=3.5 / steps=4); (e) asymmetric shared-text-encoder optimization
  (Qwen2.5-VL shared, T5/ByT5 not present on refiner); (f) the
  metadata-chunk schema extension (`pipeline: "base+refiner"`,
  refiner_path, refiner_steps, refiner_cfg keys); (g) refiner-side
  LoRA / scheduler / sampler / sigmas pinning posture (none in v1);
  (h) IPC daemon wire-protocol extension (additive `refiner` field;
  cache key composition); (i) confirmation that this slice does NOT
  make `trust_remote_code` changes (reprompt is a separate slice) —
  keeps Red Zone framing out; (j) explicit reference to ADR-014's
  2026-05-24 §3 amendment that motivated this slice.
- **Security review REQUIRED** for the IPC daemon touch. Per project
  CLAUDE.md "Review bar" §, any change to `comfyless/server.py` runs
  `security-auditor`. Output saved to
  `docs/security/review-hunyuan-refiner-server-<YYYY-MM-DD>.md` and
  referenced from ADR-016 Changelog + the server.py-touching commit
  body. Scope of the review: the new `refiner` wire field
  (deserialization shape, validation, cache-key correctness, absence
  of new path-derivation), plus a fresh look at the existing IPC
  surface that has not yet had a §12 review (per CLAUDE.md "Debt:
  No ADR or security review exists for `comfyless/server.py` (IPC)
  … when either surface is next modified, write the missing review
  before touching the code").

## Reviewer plan

- **`code-reviewer` (Opus, `model: "opus"` at invocation per global
  §5A and the broken-frontmatter workaround per memory
  `feedback_agent_model_pin_broken`)** — run after each non-trivial
  slice step, before commit. Non-negotiable.
- **`security-auditor` (Opus, `model: "opus"` at invocation)** —
  REQUIRED on every commit that touches `comfyless/server.py`
  (invariant 11). Per project CLAUDE.md "Review bar" §. Output saved
  to `docs/security/review-hunyuan-refiner-server-<YYYY-MM-DD>.md`
  and referenced from ADR-016 Changelog + the touching commit body.
  Also serves to close the pre-existing IPC §12 review debt.
- **ADR-013 §8 trailing-note check:** this slice's success depends on
  the `hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers`
  package being installable against the existing `diffusers==0.37.1`
  pin. **If the refiner pipeline requires a diffusers bump, the
  trailing-note triggers and `security-auditor` layers onto every
  code-touching commit (not just server.py-touching).** Verify at
  slice start via `from diffusers import HunyuanImageRefinerPipeline`
  on the existing pin (already confirmed exported by the diffusers
  Vision-side check; ADR-014 §3 audit corroborates — high confidence
  no bump needed, but verify before assuming).

## Open questions to settle in the ADR

1. **Shared text-encoder injection mechanics.** Refiner declares only
   `text_encoder` (Qwen2.5-VL) — no T5/`text_encoder_2`. Cleanest
   approach: load base first, construct refiner via
   `from_pretrained(refiner_path, text_encoder=base.text_encoder,
   tokenizer=base.tokenizer, torch_dtype=…)`. Verify the diffusers
   `HunyuanImageRefinerPipeline.from_pretrained` accepts these as
   override kwargs (standard diffusers pattern, high confidence but
   needs explicit check during ADR drafting).
2. **Refiner input shape.** `HunyuanImageRefinerPipeline.__call__`
   takes `image: PipelineImageInput | None`. Refiner uses a DIFFERENT
   VAE class (`AutoencoderKLHunyuanImageRefiner`), so base's latents
   are NOT valid input. v1 plan: PIL roundtrip (base decodes via its
   VAE → PIL → refiner's `image` param → refiner re-encodes via its
   own VAE). Investigation deferred to ADR-016: does the refiner
   pipeline expose a direct latent path (`latents=` or `image_latents=`
   kwarg) that bypasses its own encode entirely? Probably not since
   the VAEs differ, but inspect `__call__` body for completeness.
3. **PNG metadata schema additions.** New keys in the `comfyless`
   tEXt chunk: `pipeline: "base+refiner"`, refiner model_path,
   `refiner_steps`, `refiner_cfg`. Backward compatibility: existing
   sidecar-replay (`--params <prior-png>`) on a pre-refiner image:
   missing keys → base-only behavior (good). Refiner-aware image
   replayed by a pre-refiner comfyless build: unknown keys ignored
   (good). Confirm both behaviors in tests. ADR documents the schema.
4. **ComfyUI generate-node integration.** Add a single optional
   `refiner_path` string input to `Eric Diffusion Generate`
   (default empty → base-only + warning; any non-empty value → opt-in
   refiner stage). Mirrors the CLI exactly. No `refiner_skip` /
   sentinel inputs; an empty string is the unset state.
5. **Smoke command prerequisites.** The proof-hooks smoke assumes
   the refiner directory is downloaded. The Vision smoke setup
   documents the `huggingface-cli download
   hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers` step
   so operators don't get surprised.
6. **IPC cache eviction policy on refiner change.** When a daemon
   already holds a hot `(model, None)` pipeline and receives a
   request with `refiner` set, does it (a) evict the base-only entry
   and load the two-stage pipeline, or (b) keep both cached
   (memory-permitting)? Recommend (a) — single-slot cache is the
   existing daemon shape; document explicitly in ADR-016 so operators
   know switching modes incurs reload cost.

## Status

- Drafted 2026-05-24 as the immediate-next slice after the
  `hunyuan-support` 2026-05-24 amendment (`3638daa`).
- Revised 2026-05-25 per Grant's review pass: removed auto-discovery
  (security-surface widening); removed `--refiner skip` (redundant
  with absence); refiner cfg/steps remain flag-addressable since the
  pipeline accepts them independently; explicit LoRA / scheduler /
  sampler / sigmas pinning posture documented (no v1 refiner-side
  flags); IPC daemon support folded into scope (security-auditor
  added); asymmetric shared-encoder optimization sharpened (Qwen2.5-VL
  only, refiner has no T5 slot); refiner VAE class confirmed distinct
  (PIL roundtrip is v1 plan, latent bypass investigation deferred to
  ADR-016).
- Awaiting Grant's re-review + approval before `/change-slice` →
  ADR-016 draft.
- Next action after approval: download the
  `hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers` weights
  (~20-25 GB delta over base; user-side step) and verify the diffusers
  pipeline class is importable on the existing pin.
