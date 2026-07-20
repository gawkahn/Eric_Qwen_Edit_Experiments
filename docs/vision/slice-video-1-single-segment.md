# Vision — Video slice 1: single-segment CLI (`comfyless/video.py`)

**Date:** 2026-07-19 · **ADR:** ADR-033 (accepted) · **Risk:** L2 (new module,
no Red Zone surface; caller-supplied local paths, CLI-local trust model same as
`generate.py`)

## What must be true when done

1. `python -m comfyless.video --keyframe-start A.png --keyframe-end B.png
   --prompt "..." --output seg.mp4` renders one Wan 2.2 segment anchored at
   both ends and writes a playable H.264 mp4 + a JSON sidecar with the full
   replayable param set (`schema: "comfyless-video/1"`).
2. Lightning is the default mode (Seko-V1 LoRAs from hf-local, 4 steps,
   cfg 1.0 both experts); `--no-lightning` selects the quality tier (40 steps,
   cfg 3.5). Explicit `--steps`/`--cfg` override either default
   (ADR-009 sentinel pattern).
3. Dimensions derive from the start keyframe, aligned down to /16; an end
   keyframe with different dims is resized to match with a loud warning
   (warn-don't-block). `num_frames` must satisfy (n−1) % 4 == 0 — violations
   are rejected with the nearest valid values named.
4. All model loads are `local_files_only=True`; defaults point at
   `hf-local/Wan2.2-I2V-A14B-Diffusers` and `hf-local/Wan2.2-Lightning`.
5. The GEN_PIPELINE image family system, its caches, and `generate.py` dispatch
   are untouched (separate module per ADR-033 §4).

## What must never happen

- Silent network fetch (no implicit HF downloads).
- Encode dep (`av`) leaking into the node-pack `requirements.txt` (ADR-033 §5).
- A segment silently rendered without the requested end anchor (if
  `--keyframe-end` was given and cannot be used, that is an error, not a skip).

## Negative-case tests (minimum)

- Missing / non-image keyframe path → loud error, exit ≠ 0.
- Bad `--frames` (e.g. 80) → rejected, message names 77/81.
- End-keyframe dim mismatch → resized + warning captured.
- Sidecar round-trip: written sidecar re-parses and contains model paths,
  LoRA config, seed, dims, frames, fps, steps, cfg, prompts.
- Encode round-trip: synthetic frames → mp4 → decode via av → frame count and
  dims match.

## Edit scope (hard)

`comfyless/video.py` (new), `test_video.py` (new), `pyproject.toml` + `uv.lock`
(av exact pin — approved via ADR-033 acceptance), `CLAUDE.md` (one-line
lockstep-exception note), this Vision doc. Nothing else.

## Proof hooks

`py_compile` on the new module; `test_video.py` (no-GPU) green via
`./.venv/bin/python3`; live smoke: re-render the RESULTS.md dog segment through
the CLI and confirm timing/convergence parity with the harness run.

## Out of scope (later slices)

plan.json multi-segment chaining + stitch (slice V2), keyframe authoring,
daemon/MCP, audio, ComfyUI nodes.

**Lens:** team-portable — module boundary, schema versioning, and dep policy
all follow existing repo conventions; nothing solo-specific.
