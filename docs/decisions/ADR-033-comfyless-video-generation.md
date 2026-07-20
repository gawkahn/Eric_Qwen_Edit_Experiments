# ADR-033: comfyless Long-Form Video Generation (keyframe-anchored segment chaining)

**Date:** 2026-07-19
**Status:** accepted

---

## Context

End goal (Backlog "Video generation program", 2026-07-19): a scene description
goes to an LLM, which breaks it into per-segment prompts and keyframes; the
pipeline renders ~5 s video segments and chains them into 3–5+ minute videos.
Grant's prior art: extensive WAN 2.2 experience in ComfyUI, where the most
successful long-video method was sequential node groups, each seeding from the
previous segment's last frame.

Two evidence inputs precede this ADR (per §12 order: research → ADR → code):

**Deep research (2026-07-19, 25/25 claims verified 3–0).** All serious
open-weight candidates are natively supported in the exact diffusers version
this repo pins (0.39.0): Wan 2.2 (incl. MoE two-expert loading and
`last_image` on `WanImageToVideoPipeline`), LTX-2 (incl. arbitrary-index
keyframe conditioning and joint audio), HunyuanVideo-1.5 (first-frame only —
disqualified for chaining), SkyReels-V2, FramePack. LongCat-Video (bespoke
torchrun repo) claims minutes-long continuation but third-party reviews report
identity drift over long runs. diffusers 0.39.0 also ships
`LTXI2VLongMultiPromptPipeline` — sliding-window long video with per-window
prompts, overlap blending, and AdaIN latent-statistics color correction — which
serves as the reference design for drift mitigation.

**Hot-test measurements (2026-07-19, `video-smoke/RESULTS.md`; RTX PRO 6000
Blackwell 96 GB, one GPU per run, stock torch-2.11 SDPA — no flash-attn needed
on sm_120).** 720p ≈ 1280×704, 5 s segments; keyframe distances are mean-abs
pixel diff where ~4 is the VAE round-trip floor:

| Config | steps | time | last frame vs target keyframe |
|---|---|---|---|
| Wan 2.2 A14B I2V bf16 | 40 | 1890 s | 42 (unconditioned, as expected) |
| Wan 2.2 A14B + `last_image` | 40 | 1891 s | **6.0** |
| **Wan 2.2 + Lightning 4-step LoRA + `last_image`** | 4 | **132 s** | **5.8** |
| LTX-2 19B dev I2V (+audio) | 40 | 291 s | n/a (no last-frame param on I2V path) |

Decisive findings:
- **`last_image` conditioning works on Wan 2.2 A14B despite being untrained
  there** — the segment's final frame converges to the target keyframe at
  near-VAE-floor distance. No dedicated FLF2V checkpoint and no VACE needed.
- **Lightning distillation (Seko-V1 rank-64 LoRAs, one per MoE expert,
  `guidance_scale=1.0` both stages) retains both quality and convergence** at
  1/14th the cost. 2.2 min/segment is the practical operating point.
- Wan held exposure/color flat across all runs; LTX-2 drifted visibly darker
  within one segment. LTX-2 remains the only audio-native open model.
- Keyframe pair for the tests was authored with the repo's own image models
  (Qwen-Image-2512 same-seed prompt variation; Qwen-Edit-2511 for edits) —
  validating the keyframe-authoring premise and surfacing its failure modes
  (compound edit instructions partially execute; painterly drift on photoreal
  scenes).

## Decision

**1. Base segment engine: Wan2.2-I2V-A14B (diffusers layout, promoted to
hf-local 2026-07-19) + Wan2.2-Lightning 4-step Seko-V1 LoRAs + `last_image`
keyframe anchoring.** Defaults: 720p-class dims (multiple of 16), 81 frames @
16 fps, 4 steps, cfg 1.0 on both experts. A quality tier (bf16, 40 steps,
cfg 3.5) is retained as an option — same convergence, 14× cost.

**2. Chaining architecture: keyframe-anchored independent segments.** The plan
layer produces K+1 keyframes for K segments; segment *i* renders
(keyframe*i* → keyframe*i+1*) with per-segment prompt. Because every segment is
pinned at both ends (boundary error ≈ VAE floor), segments are **independent
and parallelizable across GPUs** — a 3-min video ≈ 36 segments ≈ 40 min wall on
two GPUs. This inverts the ComfyUI sequential method: drift cannot accumulate
across segments because no segment's output conditions another segment.
Stitching drops the duplicated boundary frame at each join; a post-stitch
color-match/AdaIN correction pass (numpy, LTXI2VLong pattern) is specified as
measure-then-apply — Wan showed no drift in testing, so v1 measures per-join
deltas and only corrects above a threshold.

**3. The segment plan is a JSON artifact (`plan.json`) and is the contract
between the human/CLI layer (now) and the LLM planner (slice 6, later).**
Schema (v1): ordered segments, each `{prompt, keyframe_start, keyframe_end,
seed, frames, steps?}` plus global defaults. CLI-first: the user hand-writes or
scripts the plan; the future planner emits the same shape. Machine-boundary
validation applies at ingestion (ADR-012 patterns).

**4. Code shape: `comfyless/video.py`, a separate dispatch module in the
`cascade.py` mold** — NOT wedged into the GEN_PIPELINE image family system.
Video has a different output kind (frame sequence + encode + audio mux later),
different params (frames/fps/keyframes), different memory profile. It reuses
the shared conventions: catalog-name model resolution (ADR-015 opaque
handles), sidecar metadata, `--params`/`--override` replay, path validation.
Daemon integration and MCP exposure are explicitly later slices.

**5. Video encode: `av` (PyAV), exact-pinned, comfyless-only dependency.**
In-process encode (frames stream to encoder; no temp PNG trees, no subprocess),
wheel-bundled FFmpeg libs pinned + hash-locked by uv — fits §11 supply-chain
policy where a system ffmpeg (unpinned, PATH-dependent, absent on this system)
does not. CPU x264 encodes a segment in ~1–2 s; NVENC is unnecessary at this
volume. **Deliberate divergence from the 17-pin pyproject/requirements
lockstep:** `av` joins `pyproject.toml` (uv path) but NOT the node-pack
`requirements.txt` — ComfyUI-side code never imports it. License note for
ADR-031: PyAV wheels bundle libx264 (GPL); binding is local-use only and the
distributed node pack does not gain the dep, so no distribution-license change.
The dep add lands in the first implementation slice with the usual approval.

**6. Weights placement:** eval weights promoted from `/mnt/nvme-2tb/hf-eval`
to `hf-local/{Wan2.2-I2V-A14B-Diffusers, LTX-2, Wan2.2-Lightning}` (plain
dirs, 2026-07-19). Video models resolve through the same hf-local/catalog
conventions as image models. LTX-2 is retained solely as the future audio-stage
candidate.

**7. Keyframe authoring (separate slice, depends on the comfyless edit-support
slice):** keyframes come from (a) same-seed prompt-variation on a base image
model — cheap default, (b) single-operation edit chains (qwen-edit / Klein)
when a specific frame must be preserved, judge-gated via the refine loop
(ADR-027), (c) user-supplied paths (Gimp etc.) — always accepted.

## Alternatives Rejected

- **LTX-2 as chaining engine** — within-segment exposure drift; no last-frame
  parameter on its I2V path (its `LTX2ConditionPipeline` does keyframes, but
  the drift disqualifies the backbone). Retained for audio.
- **Wan2.1-FLF2V-14B-720P dedicated checkpoint** — measured slower per step
  than Wan 2.2 (53–57 s vs 47 s) and redundant once `last_image` proved to work
  on 2.2 weights.
- **Wan VACE** — strictly more machinery (mask conventions, reference clips)
  than v1 needs; revisit if identity control across distant segments demands it.
- **LongCat-Video continuation** — bespoke torchrun inference repo (integration
  cost), vendor no-drift claim contradicted by third-party long-run reviews,
  hours-long generation.
- **Sequential last-frame seeding (the ComfyUI method)** — unbounded drift
  accumulation and inherently serial; keyframe anchoring bounds error per
  segment and unlocks GPU parallelism.
- **Adopting `LTXI2VLongMultiPromptPipeline` wholesale** — right mechanics,
  wrong backbone (LTX). Its overlap/AdaIN patterns are adopted as the
  correction reference instead.
- **System ffmpeg / imageio-ffmpeg** — unpinned system binary contradicts §11
  and the security model; imageio-ffmpeg is a pip-delivered opaque binary that
  still subprocesses.
- **lightx2v full int8 distilled checkpoints** — ComfyUI single-file format +
  int8; the LoRA route composes with the diffusers pipeline we already load and
  keeps the base weights canonical.

## Deferred / Out of Scope

- **LLM planner + MCP exposure (slice 6)** — Red Zone from first commit (LLM
  output drives generation params + file paths): its own spec, ADR amendment,
  and `security-auditor` review before code, per the refine.py precedent.
- Audio stage (LTX-2 or separate model); v1 videos are silent.
- Daemon (`--serve`) video routing; v1 is foreground CLI.
- ComfyUI video nodes.
- Refine-loop judging of video segments (motion-quality judge is unexplored).
- 1080p / upscale tier, frame interpolation to higher fps.
- fp8/nvfp4 quantization of the video transformers (fp8 expected to work per
  ADR-019 patterns; test when VRAM pressure demands).
- VACE-based identity/motion control.
- Wan 2.5+ (open-weight status unresolved as of research date).

## Proof hooks

- `video-smoke/RESULTS.md` + `video-smoke/video_smoke.py` — reproducible
  baseline runs behind every number in Context.
- Implementation slices carry `test_video.py` (plan-schema validation, dispatch,
  boundary-frame handling, encode round-trip) — negative cases per §5/§12.

## Changelog

- 2026-07-19: Proposed, on the strength of the deep-research report + hot-test
  matrix (both same day).
- 2026-07-19: Accepted by Grant same day. Eval cache `/mnt/nvme-2tb/hf-eval`
  deleted after promotion verified (incl. the redundant Wan2.1-FLF2V download).
- 2026-07-19 (slice V1): default model/LoRA resolution in `comfyless/video.py`
  is `--model-base` flag, falling back to the `COMFYLESS_MODEL_BASE` env var —
  a new convention introduced by this slice (generate.py takes the flag only).
  `--offload` added (CPU-offload, peak ~30–40 GB) after the live smoke hit a
  shared-GPU OOM. Sidecar `comfyless-video/1` carries `crf` and absolute
  model/LoRA paths per code review.

**AI-Disclosure:** Claude (Fable 5) authored; Grant reviewed.
