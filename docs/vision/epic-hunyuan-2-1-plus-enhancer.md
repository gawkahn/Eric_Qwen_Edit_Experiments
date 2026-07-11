# Vision (epic): Hunyuan-Image 2.1 onto main + comfyless Prompt Enhancer

**Date:** 2026-07-11
**Status:** proposed
**Risk:** L2 — touches runtime-core files (`comfyless/generate.py`, `server.py`,
`nodes/eric_diffusion_*`), IPC daemon parity, and introduces the first
`trust_remote_code` in the repo.

Governing ADRs: **ADR-025** (Hunyuan CFG routing, renumbered from branch 014),
**ADR-016** (Hunyuan base+refiner chain), **ADR-026** (prompt enhancer).

---

## What must be true when done

1. The `hunyuan-image` family (base + optional refiner) generates from the
   comfyless CLI on main, against the on-disk `HunyuanImage-2.1-Diffusers` /
   `HunyuanImage-2.1-Refiner-Diffusers` weights.
2. comfyless has a CLI prompt-enhancement path — inline (`--enhance-prompt`) and
   offline (`comfyless enhance` list→list) — with backend/recipe decoupling per
   ADR-026, including the local Hunyuan `reprompt` backend for a faithful 2.1 test.
3. No regression to the 167 commits of runtime-core work that landed on main since
   the branch forked (fp8/quant, NAG, ADR-020 daemon-per-GPU). Full suite green.

## Findings that shape the approach (investigation 2026-07-11)

- **Re-apply, not merge.** `hunyuan-support` is 167 commits behind main; a raw
  `git merge` conflicts in exactly the runtime-core hot files main rewrote. The
  branch's intent is re-derived onto current main using the branch ADRs, Vision,
  and `test_hunyuan.py` as the spec. The self-contained `comfyless/hunyuan_chain.py`
  ports near-verbatim; the family wiring is re-derived into main's current files.
- **ADR-016 is a free gap on main** — it was reserved on the branch but never
  merged, so the refiner ADR keeps 016. Only the CFG-routing ADR collides
  (branch 014 vs main's lora-audit 014) → renumbered to **ADR-025**.
- **The prompt enhancer was never built** (branch has zero enhancer code; ADRs
  never mention it). It is new construction — see ADR-026.
- **NAG is NOT needed for Hunyuan 2.1.** The diffusers `HunyuanImagePipeline`
  ships an *enabled* `AdaptiveProjectedMixGuidance` guider (`guidance_scale 3.5`)
  and full `negative_prompt_embeds` params — the negative prompt is already
  consumed through a real guidance path, unlike Flux/Flux2 where NAG earns its
  keep. No NAG-Hunyuan slice.
- **CFG characterization must be re-verified against diffusers 0.39.0.** ADR-014
  was written against 0.36-dev and framed Hunyuan as "1× forward pass, negatives
  ignored." The 0.39.0 pipeline's enabled APG guider contradicts that (implies
  real 2-pass guidance that uses negatives). ADR-025 must not carry the stale
  claim onto main unverified.

## Decomposition

### Part 1 — base+refiner re-apply
- **Slice 1** — Port `comfyless/hunyuan_chain.py` + `test_hunyuan.py` onto main;
  land ADR-025 (renumbered) + ADR-016 (kept); resolve doc conflicts. Green tests.
- **Slice 2** — Re-derive family wiring into main's current files: `hunyuan-image`
  family string, `distilled_guidance_scale` CFG branch, `family_defaults` row,
  `--vae-tiling`/`--refiner` flags. **Includes the CFG re-verification** against
  0.39.0 (perturb-negative empirical check; confirm 1-pass vs 2-pass; update
  ADR-025 if the "distilled/negatives-ignored" framing no longer holds).
  → `code-reviewer` (Opus).
- **Slice 3** — Daemon + MCP parity for `refiner_path` (`_PATH_FIELDS`,
  `_check_paths`, cache-key in `comfyless/server.py`). → `security-auditor` (Opus,
  IPC surface).

### Part 2 — prompt enhancer (ADR-026)
- **Slice 4** — ADR-026 (done) + `security-auditor` on the `trust_remote_code`
  decision.
- **Slice 5** — enhancement core + `openai-endpoint` backend + backend/recipe
  registries + inline `--enhance-prompt`/`--enhance-recipe` + three generic
  recipes per family (`-generic`, `-preserve-subject`, `-vary-setting`), seeded
  from the node's `SYSTEM_PROMPT_EN`. → `code-reviewer` (Opus).
- **Slice 5b** — offline `comfyless enhance` list→list + `--variations` +
  provenance sidecar. → `code-reviewer` (Opus).
- **Slice 5c** *(deferred)* — regional `keep`/`enhance`/`vary` span markers.
- **Slice 6** — `hunyuan-reprompt` local backend (local 14 GB `hunyuan_v1_dense`,
  `trust_remote_code`, hash-pinned reviewed tokenizer). → `security-auditor` (Opus).

### Part 3 — close-out
- **Slice 7** — live CLI smoke on the on-disk weights (Grant), then merge to main.

## Invariants

- **I1** — Existing family behaviour (qwen-image / flux / flux2 / zimage /
  cascade / krea) is byte-unchanged by the Hunyuan family addition. Proof: full
  suite + a pre/post pixel-MSE spot check on one non-Hunyuan family.
- **I2** — `distilled_guidance_scale` (or whatever slice-2 verification concludes)
  is routed only for `hunyuan-image`; no other family's CFG kwarg changes.
- **I3** — `refiner_path`, like every path field, is validated at the IPC boundary
  (`_check_paths`) and never crosses the MCP boundary as an absolute path.
- **I4** — Enhancement runs once at generation time; `--params` replay uses the
  stored enhanced prompt and never re-calls the LLM (ADR-026 §7).
- **I5** — Inline enhancement is memoized per unique input prompt within a run:
  a fixed prompt across a LoRA/transformer sweep enhances once (clean A/B).
- **I6** — `trust_remote_code=True` is used only for the vendored, `local_files_only`,
  reviewed + hash-pinned reprompt tokenizer (ADR-026 §8); never re-fetches.
- **I7** — The offline transform's output is a flat JSON list directly consumable
  by one `--iterate prompt` command; the provenance sidecar is ignored by iterate.

## Failure semantics

- Wrong `--refiner` class → hard reject at load (no silent fallback), per ADR-016.
- Enhancer backend unreachable / endpoint error → loud failure with the backend
  name; generation does not silently proceed on the un-enhanced prompt without a
  warning.
- Malformed recipe / unknown backend name → fail-closed with the offending name.
- Missing/altered reprompt tokenizer hash → refuse to load the local backend.

## Out of scope

- NAG for Hunyuan (redundant — APG guider). HunyuanImageRefinerPipeline beyond the
  chain already specced. Edit/inpaint/ControlNet Hunyuan variants. Regional
  enhancement markers (5c). MCP surfacing of enhancement. Recipe authoring beyond
  the three v1 generics.

## Proof hooks

- `python3 test_hunyuan.py` green on main (ported from branch, 2022 lines).
- Full 17-suite regression, 0 failures, against `./.venv/bin/python3`.
- New enhancer tests: core memoization, backend registry fail-closed, recipe
  selection + family default, offline list→list round-trip + provenance,
  sidecar/replay-no-recall, trust_remote_code hash-pin refusal.
- Live CLI smoke: one base gen + one base+refiner gen on the on-disk weights;
  perturb-negative check for the CFG re-verification; one `--enhance-prompt` run;
  one `comfyless enhance … --variations 3` → `--iterate prompt` round-trip.

## Numbering decisions

- Enhancer = **ADR-026** (new). Hunyuan CFG-routing 014 → **ADR-025** (collision
  with main's lora-audit ADR-014). Refiner keeps **ADR-016** (free gap it reserved).

**AI-Disclosure:** Claude (Opus 4.8) authored; Grant reviewed.
