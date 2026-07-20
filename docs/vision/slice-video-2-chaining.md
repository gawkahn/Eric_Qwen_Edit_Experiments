# Vision — Video slice 2: plan.json chaining + stitch (`comfyless/video.py`)

**Date:** 2026-07-20 · **ADR:** ADR-033 §2–3 (accepted) · **Risk:** L2

> **Posture:** Boundary: entrypoint (plan.json ingestion — machine-facing
> format, future LLM emitter) + local file writes. Risk factors: multi-process
> dispatch, machine-boundary input; no auth/PII/network. CLI-local trust model
> same as slice V1 / `generate.py --params`.

## Intent

`python -m comfyless.video --plan plan.json --output movie.mp4` renders K
keyframe-anchored Wan 2.2 segments (optionally sharded across GPUs), keeps the
per-segment mp4s for judging/resume, and stitches one master mp4 with
boundary-frame dedup and measure-then-apply color correction at joins.

## Decisions settled here (were open after slice V1)

1. **Stitch container strategy — per-segment mp4s + one master re-encode.**
   Segments land as `<stem>_segments/seg_000.mp4` … each with its own
   `comfyless-video/1` sidecar (unchanged schema). The master is produced by
   decoding all segments → dedup → color-correct → single re-encode to
   `--output`, plus a master sidecar (`comfyless-video-plan/1` result echo).
   Per-segment files are the judging/resume unit; the master is the deliverable.
2. **Parallel dispatch — per-device worker subprocesses.** `--devices
   cuda:0,cuda:1` makes the parent shard segment indices round-robin and spawn
   one child CLI process per device (`--plan … --only-segments i,j --device X
   --no-stitch`); each child loads the pipeline **once** and renders its shard
   sequentially; the parent waits for all children, then stitches. One device →
   plain in-process sequential loop, no subprocess. Rejected: two pipelines in
   one process (2× ~40–80 GB + offload/threading fragility), one subprocess per
   segment (pays the model load per segment).
3. **Schema hardening.** `fps`, dims, and lightning/base mode are
   **global-only** (stitch requires uniform fps/dims; mixing tiers changes the
   look mid-video). Per-segment overrides: `seed`, `frames`, `steps`, `cfg`,
   `negative_prompt`. `prompt`, `keyframe_start`, `keyframe_end` are required
   per segment — in plan mode every segment is pinned at both ends (ADR-033 §2);
   the single-segment CLI keeps `--keyframe-end` optional.
4. **Cuts are legal.** Segment *i+1*'s `keyframe_start` ≠ segment *i*'s
   `keyframe_end` is a deliberate scene cut: loud notice, no dedup, no color
   correction across that join. Continuity is detected by comparing the two
   keyframe paths (realpath equality).

## plan.json v1 (`comfyless-video-plan/1`)

```json
{
  "schema": "comfyless-video-plan/1",
  "defaults": {"frames": 81, "fps": 16, "steps": null, "cfg": null,
                "negative_prompt": null, "seed": -1},
  "lightning": true,
  "segments": [
    {"prompt": "…", "keyframe_start": "kf/a.png", "keyframe_end": "kf/b.png",
     "seed": 7, "steps": 4, "cfg": 1.0}
  ]
}
```

Ingestion is ADR-012 machine-boundary style, all before any torch import:
byte cap on the file (1 MB), unknown keys rejected at every level, type +
range checks (frames via the V1 validator, steps 1–100, cfg 0–20, fps 1–60,
prompt ≤ 5000 chars, ≤ 200 segments, ≥ 1 segment), keyframe paths must exist
and decode. Relative keyframe paths resolve against the plan file's directory.

## Invariants (must always be true)

- Invalid plan → exit 2 with the offending field named; **nothing written, no
  GPU touched** (validation completes before the torch import).
- The master mp4 exists **only if every segment rendered**. Its frame count
  = Σ segment frames − (number of continuous joins).
- Boundary-frame dedup and color correction happen **only at continuous
  joins**; cut joins keep both frames untouched.
- Color correction is measure-then-apply: per-join per-channel mean/std deltas
  are always measured and recorded in the master sidecar; frames are modified
  only when the delta exceeds the threshold (`--color-correct-threshold`,
  default 2.0 on the 0–255 scale; `--color-correct off` disables applying,
  never the measuring).
- Each device-worker loads the pipeline exactly once; each segment renders
  exactly once per run (with `--resume`, segments whose mp4 + sidecar already
  exist are skipped with a notice).
- Slice V1 single-segment CLI behavior is unchanged (existing flags, sidecar
  schema, exit codes) — the V1 test block stays green untouched.
- `local_files_only=True` everywhere; `av` stays out of `requirements.txt`.

## Failure semantics

Fail-closed, resume-friendly: any segment failure (in-process or a child's
nonzero exit) → parent names the failed segment indices, exits nonzero, and
does **not** stitch; completed per-segment mp4s remain on disk for `--resume`.
`--stitch-only` re-runs dedup/correction/encode from existing segment files.
A child that dies without output is a failure of its whole shard.

## Out of scope

LLM planner + MCP exposure (slice 6 — Red Zone, own spec + security-auditor
first), audio, daemon routing, keyframe authoring, ComfyUI nodes, video
judging, interpolation/upscale tiers, cross-segment identity control (VACE).

## Negative-case tests (minimum)

- Unknown key (top level, defaults, segment) → rejected, key named.
- Missing required segment field / empty segments / > 200 segments → rejected.
- Oversize plan file (> 1 MB) → rejected before JSON parse.
- Bad frames/steps/cfg/fps values → rejected with valid range named.
- Missing keyframe path in a middle segment → exit 2 before any render.
- Cut join: no dedup, no correction, notice emitted.
- Below-threshold color delta: measured + recorded, frames byte-identical.
- Failed segment (stub) → no master written, exit ≠ 0.

## Proof hooks

- `./.venv/bin/python3 test_video.py` — V1 block untouched + new plan
  validation, shard assignment, dedup, AdaIN math on synthetic frames, stitch
  round-trip via `av`, resume skip, master-sidecar round-trip, and the
  negative cases above.
- `python -m py_compile comfyless/video.py`
- Live smoke (Grant): 2–3 segment plan from the `video-smoke/` keyframes,
  `--devices cuda:0,cuda:1`, confirm parallel wall-clock ≈ slowest shard and
  join quality by eye.

## Edit scope (hard)

`comfyless/video.py`, `test_video.py`, this Vision doc (+ vault mirror),
ADR-033 Changelog append. Nothing else.

**Lens:** team-portable — schema versioning, worker-subprocess dispatch, and
fail-closed ingestion all follow existing repo conventions (ADR-012, cascade
mold); nothing solo-specific.
