# Security Review — ADR-027 slice 4 (seed-image entry)

**AI-Disclosure:** `security-auditor` (Claude Fable 5, pinned per §5A) authored;
Grant reviewed. Findings folded into `comfyless/refine.py` before commit.

**Date:** 2026-07-15
**Surface:** `comfyless/refine.py::build_config_from_seed` (+ `resolve_lora_ops`
path_was_discarded, `_stat_within_bytes`, arg parser, `main()` wiring).
**Threat model:** §12 machine-driven-params surface (a caller-supplied image's
embedded metadata seeds generation params). NOT §5 Red Zone (single-user local,
no auth/PII/billing, no new network egress).
**Verdict:** **APPROVE with conditions** — no CRITICAL/HIGH. Four MEDIUMs, all
folded before commit (see ADR-027 Changelog 2026-07-15 for the fold-in record).

---

## Summary

Slice 4 adds seed-image entry: `build_config_from_seed` reads a caller-supplied
image's embedded comfyless chunk (or an explicit `--params` sidecar) and seeds
the working config with full schema authority. The two untrusted channels are
(1) the LLM judge output (unchanged from slice 3, still gated by the F1 closed
allowlist) and (2) the seed image's embedded metadata (F4, new).

Verified: the F5 gate runs FIRST (`load_seed_image_capped`, before any metadata
parse); extraction drops unknown keys via `_validate_params`; the seed-LoRA chain
(`.safetensors`-strip → `resolve_reference` basename-strip → NUL/forbidden-char
gate → NFC catalog lookup → `expected_kind="lora"` → existence → union-`_within`
realpath containment) is airtight — `..`, backslash-only, NUL, no-extension,
wrong-case-extension, double-extension, trailing-slash, and name-collision inputs
all fail closed or re-bind to a catalog entry under operator roots; a foreign
directory is never honored as a load path. HF resolution is fail-closed on both
planes. `WorkingConfig.base` never reaches `build_judge_user_text`/`verdict_record`
(both `_assert_no_paths`-gated), the LLM cannot mutate `base` (`apply_overrides`
copies it), and resolver notices stay stderr-only (constraint (b)).

## Findings (all folded)

**[MEDIUM] F4 echo omitted `upscale_vae_path`** (`_SEED_ECHO_PATH_FIELDS`) — a
schema key that survives `_validate_params`, rides in `base`, and is loaded as
model weights on the daemon wire. → added to the echo tuple with a keep-in-sync
note vs `server._PATH_FIELDS`; `upscale_vae_subfolder` traversal already confined
in `_load_upscale_vae` (ADR-030 review).

**[MEDIUM] Seed-embedded prompt enters judge/planner context, unbounded** — the
seed's `prompt` becomes the judge target and re-enters context every iteration
(second injection channel, F8 class; blast radius bounded by F1/F2 + human review
of `winners/`, but not length-capped). → **Ruling:** the target prompt is the ONE
necessary exemption to the slice-3 constraint; now capped at
`OVERRIDE_PROMPT_MAX_CHARS`, symmetric with the planner prompt. Recorded in the
ADR Changelog.

**[MEDIUM] `--params` file read escaped the F5 byte cap** — unbounded `json.load`
(local DoS). → `_stat_within_bytes(args.params, SEED_IMAGE_MAX_BYTES)` before read.

**[MEDIUM] Cold in-process path honors seed component paths outside every root** —
on the daemon path `_check_paths` rejects out-of-roots paths; on the cold path
they load directly. Same accepted `generate --params` trust model (the LLM
verifiably cannot reach `base`), so not treated as a break — but the F4 echo is
the sole compensating control and printed nothing about containment. → the echo
now flags each path OUTSIDE the roots ("loads on the cold path only") via
`_within`. A fail-closed cold-path containment gate is a trust-model change
deferred to Grant (TECH_DEBT 2026-07-15).

**[INFO] Seed resource values honored without bounds** (lora list length,
weights, dims) — deliberate user-authority decision; all fail closed (crash, not
escape). Recorded, no action.

**[INFO] Non-string path fields skip the F4 echo** (`_validate_params` keeps
type-mismatched values; the echo filters `isinstance(str)`) — later crashes as a
traceback rather than being surfaced. Cosmetic; no action required.

## Answers to the review questions

1. Load path outside roots? LLM channel: no. Seed channel: no on the daemon path;
   yes on the COLD path for `model`/component paths — same accepted trust model;
   echo strengthened (MEDIUM-4). HF egress fail-closed both paths.
2. Seed-LoRA chain airtight? Yes — traced every crafted variant; all fail closed
   or re-bind within roots.
3. Seed content in judge context? Paths/notices/`base`: no (`_assert_no_paths`,
   stderr-only notices). The prompt: yes, necessarily — capped (MEDIUM-2).
4. F5 at entry? Yes — `load_seed_image_capped` is the first statement. Gap: the
   `--params` file (fixed, MEDIUM-3).
5. Numeric/DoS gaps? `--params` uncapped (fixed); seed prompt length (fixed);
   list/weights/dims (INFO, user authority, fail-closed).

## Code-reviewer (Fable) — companion functional review

needs-changes, all folded: weight `0.0`→`1.0` coercion bug (real); cold-path
ADR-030 `upscale_vae_path`/`upscale_vae_subfolder` omission (daemon/cold
divergence); `_extract` catch tuple + non-dict guard; malformed-lora-entry
notices; seed-mode CLI-flags-ignored note; abspath HF-repo caveat documented;
test strengthening (weight-0, bare-name, relative-abspath, whitespace prompt,
string weight, over-cap prompt, `--params` byte cap). Core design (merge
semantics, model precedence, `.safetensors`-strip↔`path_was_discarded` interplay,
closed resolver plane) confirmed sound.
