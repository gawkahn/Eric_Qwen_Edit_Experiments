# ADR-009: Per-Family Default Params (cfg / steps / sampler / schedule)

**Date:** 2026-04-25
**Status:** accepted

---

## Context

`comfyless.generate` currently has one set of schema defaults
(`COMFYLESS_SCHEMA` in `comfyless/generate.py`): `cfg_scale=3.5`,
`steps=28`, `sampler="default"`, `schedule="linear"`. These were tuned
for Flux/Flux.2 — the families the tool was first built around — and
work fine there.

A 1000-prompt × multi-model word-salad sweep on 2026-04-24 surfaced the
problem: those same defaults starve other families.

- **Pony / Illustrious** (SDXL fine-tunes): at `cfg=3.5` they "give up"
  on gibberish prompts and emit blotchy junk pools of color. Both
  families want `cfg ≈ 7-8` — well-known in the SDXL community.
- **Stabilityai base SDXL / SD3.5**: under-driven at `cfg=3.5`. SAI's
  own model cards recommend `cfg=7` for SDXL.
- **Qwen-Image-2512**: official recommendation is `true_cfg_scale=4.0`
  with `steps=50` (documented in `CLAUDE.md`). Today the user has to
  pass these on the command line every time.
- **Flux / Chroma**: schema defaults are correct.

The cross-family iteration pattern — `--iterate model models.json` with
N prompts — produces noise that **looks like** "model X is worse than
Y" when the real story is "model X was misconfigured." This silently
corrupts comparisons.

This is also a **prerequisite for the Auto-Refinement Loop** (Backlog →
Ideas): an LLM judge cannot meaningfully compare candidate images
across families if each candidate was generated outside its sweet spot.

## Decision

### Add a family-default overlay

Introduce `comfyless/family_defaults.py` containing a single dict:

```python
FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Keys: model_family string returned by infer_model_family().
    # Values: PARTIAL param dicts — only keys this family has an opinion on.
    # Anything not listed here falls through to COMFYLESS_SCHEMA defaults.
    "qwen-image": {"true_cfg_scale": 4.0, "steps": 50},  # source: official model card
    "sdxl":       {"cfg_scale": 7.0,  "steps": 28},      # source: SAI model card
    ...
}
```

Family detection uses the existing `detect_pipeline_class` /
`infer_model_family` machinery — no new detection logic. Family strings
are exactly the set already produced by `nodes/eric_diffusion_utils.py`:
`qwen-image`, `qwen-edit`, `flux2klein`, `flux2`, `chroma`, `flux`,
`auraflow`, `sd3`, `sdxl`, `sd1`, `zimage`. Unknown families pass
through with no overlay.

### Precedence ladder (final)

```
schema_default  <  family_default  <  sidecar  <  --override  <  explicit --flag  <  --iterate axis
```

- **Schema default** — the existing `COMFYLESS_SCHEMA` baseline.
- **Family default** — the new layer; only for keys the family
  declares.
- **Sidecar** (`--params <file>` or `--params <png>`) — user-saved
  param snapshots win over family defaults.
- **`--override key=value`** — explicit per-invocation patch.
- **Explicit CLI flag** — `--cfg`, `--steps`, etc. when not at sentinel
  `None`.
- **`--iterate axis`** — per-iteration patch wins over everything.

### Implementation: `explicit_keys` tracking

The crux is distinguishing "user said 3.5" from "schema seeded 3.5."
Both live in the merged param dict as `cfg_scale=3.5`; without a
sentinel, the family overlay can't tell which one to clobber.

Solution: build `explicit_keys: set[str]` during the merge in
`_run_cli_mode` containing every canonical key that came from sidecar,
override, or a non-None CLI flag. Family defaults are then applied
**only to keys not in `explicit_keys`** and **not in iterated axes**.

### Where the overlay runs

Inside `_run_one`, immediately after the per-iteration
`resolve_hf_path` block. We call `detect_pipeline_class(p_cur["model"])`
to extract the family string, then write family values into `p_cur`
for non-explicit / non-iterated keys.

This placement matters:

- **Per-iteration** so `--iterate model models.json` can apply
  different defaults across a single sweep (the whole point).
- **After** `resolve_hf_path` so HF repo IDs are already on disk and
  `model_index.json` is readable.
- **Before** `_load_pipeline` so the values flow through to
  `generate()` unchanged — `generate()` itself stays family-agnostic.

The extra `detect_pipeline_class` call duplicates work already done
inside `_load_pipeline`, but it only reads `model_index.json` — cheap
and idempotent.

### Editability is a hard constraint

The first round of values WILL be wrong. Empirical sweeps will refine
them. The dict is therefore designed for one-edit changes:

- One file, `comfyless/family_defaults.py`. Nothing else to touch when
  adding or adjusting a family.
- Single dict, alphabetical by family.
- One inline comment per family naming the source of the value
  (official model card, empirical sweep, community consensus).
- Partial dicts — a family's entry only lists keys it has an opinion
  on. Adding a new opinion = one new key. Removing one = delete the
  key, schema default takes over.

## Alternatives Rejected

- **Drop schema defaults to `None` and have `generate()` fill them in
  per-family.** Rejected: changes `generate()`'s signature contract,
  breaks every caller that introspects the schema for defaults, and
  scatters family knowledge across the codebase. Localizing the layer
  in `_run_cli_mode` keeps the blast radius small.

- **Apply the overlay inside `generate()` itself.** Rejected: would
  require plumbing `explicit_keys` and `iterated_axes` through the
  function signature. That couples generation to CLI semantics. Keep
  `generate()` family-agnostic.

- **Per-fine-tune patterns (`*pony*`, `*illustrious*`).** Rejected for
  round one. Pony and Illustrious are SDXL fine-tunes; `model_family`
  resolves all three to `"sdxl"`, and `cfg=7` works for all three. If
  empirical sweeps later show fine-tune-specific divergence, the
  natural channel is per-prompt `--params` overlays (which already
  win over family defaults), not a sub-family layer. Per-prompt
  sensitivity also reflects the user's intuition — "their sensitivity
  is going to be more per-prompt, what we're doing is setting
  reasonable starting points."

- **Calibration-first (sweep before code).** Rejected. The starting
  values are stubs from official model cards / community consensus;
  empirical refinement is a follow-up slice driven by the
  cross-transformers sweep enabled by ADR-008's `--limit` flag. Code
  first so the sweep itself benefits from per-family defaults.

- **YAML / JSON config file instead of Python dict.** Rejected as
  premature. Editing a Python dict with comments is faster than
  editing a YAML file and re-running schema validation, and we have
  no need yet for runtime override of these values. Revisit if
  external tools want to override family defaults without forking
  the package.

## Deferred / Out of Scope

- **Empirical calibration sweep** — separate slice, runs after this
  lands and benefits from `--iterate model --limit N`. Per-family
  cfg/steps refinement is the actual scientific work; this ADR just
  unblocks it.
- **Per-fine-tune patterns** — see Alternatives Rejected. Re-open if
  empirical evidence shows model-name-pattern overlays add value
  beyond what `--params` sidecars provide.
- **Schedule-by-family** — `schedule` is in `FAMILY_DEFAULTS` shape
  but most families today share `"linear"`. Family-specific schedule
  values can be added without code change as evidence emerges.
- **Sampler-by-family** — same posture. SDXL/SD1 already trigger a
  warning when `sampler != "default"` because their schedulers don't
  support sampler swap; family default for both stays `"default"`.
- **Default propagation in `--json` mode** — the `--json` bridge
  receives a fully-formed param dict from the caller, so the caller
  is responsible for whatever defaults it wants. We do NOT inject
  family defaults into the `--json` path in v1. If a future LLM
  agent caller wants family defaults, it can read this dict via a
  small helper export.

## Changelog

- **2026-07-24** — **CFG-knob aliasing fix** (both appliers:
  `generate._apply_family_defaults` + `refine._overlay_family_defaults`).
  `--cfg` and `--true-cfg` are two spellings of one knob:
  `build_call_kwargs` routes an explicit `cfg_scale` onto
  `true_cfg_scale` for non-guidance-embeds families, but only when
  `true_cfg_scale` is None — so a family default filling
  `true_cfg_scale` (qwen-image/qwen-edit: 4.0) silently DEFEATED an
  operator's explicit `--cfg` (observed live: `--cfg 1` on qwen-edit +
  a Lightning 8-step LoRA ran double-pass true-CFG 4.0 — CFG burn on a
  distilled setup, plausibly the artifact source in the first edit-mode
  refine smokes). The precedence ladder is unchanged; the fix extends
  "explicit" across the alias pair one-directionally: an explicit or
  iterated `cfg_scale` suppresses the `true_cfg_scale` family default
  (loud log line), while an explicit `true_cfg_scale` does NOT suppress
  a `cfg_scale` default (krea-class families default `cfg_scale`;
  `true_cfg` is inert there, so symmetric suppression would only break
  their defaults). Pinned in test_params_schema (explicit + iterated) and
  test_refine (parity + defaults-still-apply negative).
  **Review fold (both Fable, no fallback, same day):** code-reviewer
  APPROVED after verifying all five consumer paths (CLI in-process,
  daemon delegation — server.py applies no family defaults, --iterate,
  MCP, refine entries; seed-image entry never calls the overlay).
  Folds: end-to-end qwen routing pin at the incident junction
  (cfg 1.0 + suppressed default → true_cfg 1.0; explicit --true-cfg
  outranks), no-both-knobs FAMILY_DEFAULTS structural guard, refine
  suppression log silenced when --true-cfg is also explicit, log
  wording "explicit/iterated". security-auditor LOW folded: the
  generate-side explicit test is now value-aware (a replayed sidecar
  `"cfg_scale": null` no longer suppresses — the family default keeps
  masking the degenerate pair); INFO verdicts: MCP gains no authority
  it lacked (it could already set true_cfg_scale directly, and the
  machine boundary rejects null cfg_scale), refine's seed path cannot
  reach the suppression, logs injection-clean. Bonus fix noted by the
  reviewer: `--iterate cfg_scale` sweeps on qwen families were
  previously inert (every iteration ran true-CFG 4.0) — now they work.
  Review: `docs/security/review-adr-009-cfg-aliasing-2026-07-24.md`.

- **2026-04-25** — proposed and accepted (this document).
- **2026-04-25** — clarification (reviewer fold-in): the family overlay
  applies to BOTH the in-process path and the daemon delegation path
  in `_run_one`, since the overlay runs before the
  `_delegate_to_server` call. Only `--json` mode skips the overlay
  (caller responsibility, as documented above). No code change; this
  is a documentation hedge against future refactors that might split
  the paths.
- **2026-06-25** — Krea-2 support slice
  (`docs/vision/slice-krea2-support.md`). Two extensions, no change to the
  precedence ladder:
  1. **One pipeline class may now map to two families via model
     metadata.** `Krea2Pipeline` maps to `"krea"`, and the distilled
     variant — identified by `is_distilled: true` in `model_index.json` —
     to `"krea-turbo"`, so Krea-2-Raw (52 steps / cfg 3.5) and
     Krea-2-Turbo (8 steps / cfg 0.0, CFG disabled) get distinct
     defaults despite sharing a class. `infer_model_family` gains an
     optional `is_distilled` arg (default False → single-arg form
     unchanged for all existing callers); `detect_pipeline_class` and
     `comfyless/catalog.py:scan_model_family` read the flag and pass it.
     Both krea families route through the existing `guidance_scale`
     branch in `_build_call_kwargs` (flux-like single-pass CFG).
  2. **The MCP caller now applies the overlay.** When this ADR was
     written the MCP surface did not exist (it arrived with ADR-011); its
     `generate` handler applied hardcoded fallbacks (28 steps / cfg 3.5)
     instead of `FAMILY_DEFAULTS`, so an agent omitting params got
     wrong-for-family values. Per the "caller responsibility" model of
     this ADR, the MCP handler is now a caller that applies the overlay
     (fill canonical keys absent from the agent payload; explicit agent
     values win). This affects ALL families, not just Krea (e.g.
     qwen-image now gets 50 steps via MCP). The **daemon**
     (`comfyless/server.py`) is unchanged and remains caller-responsible:
     its only client is the CLI, which already applies the overlay in
     `_run_one` before delegating. Runtime generation is gated on a
     diffusers release shipping `Krea2Pipeline` (see `TECH_DEBT.md` →
     Dependencies); classification and defaults work on the current pin.

- **2026-07-06** — Z-Image base/Turbo split (no code to the precedence
  ladder; a third variant-detection signal). Z-Image ships **two** models
  under one bare `ZImagePipeline` class — `Z-Image-base` and the
  step-distilled `Z-Image-Turbo` — but, **unlike Krea-2, carries no
  `is_distilled` marker** in `model_index.json`. The only structural delta
  is scheduler `shift` (base 6.0 / Turbo 3.0), a tuning value, not a
  reliable discriminator. So the Turbo variant is detected by **`"turbo"`
  in the model dir/repo path** — the signal everyone actually uses (HF
  `Tongyi/Z-Image-Turbo`). `infer_model_family` gains an optional
  `name_hint` arg (default `""` → existing 1-/2-arg callers unchanged);
  `detect_pipeline_class` and `comfyless/catalog.py:scan_model_family` pass
  the model path. The heuristic is **scoped to the `zimage` family** so a
  stray `"turbo"` in any other family's path is a no-op. New family
  `"zimage-turbo"` (8 steps / cfg 1.0) is added to `FAMILY_DEFAULTS` and to
  the `zimage` `guidance_scale` branch in `_build_call_kwargs` (it MUST be
  listed there — the introspection fallback would emit `true_cfg_scale`,
  which `ZImagePipeline.__call__` rejects, and drop CFG entirely). base
  keeps 30 steps / cfg 4.0. **Empirical basis:** a batch of Turbo-trained
  LoRAs rendered pure noise under base params (30/4.0) and cleanly at 8/1.0
  in gen-validation (2026-07-06); resolves the same-day TECH_DEBT entry.
  **Alternative rejected:** scheduler-`shift` heuristic — it is a tuning
  value that upstream can change, whereas the repo/dir name is stable and
  human-authored. **Deferred:** per-*model* default overrides (an operator
  manifest) would generalize beyond name-heuristics if more no-marker
  distills appear; not built until a second case exists.
- **2026-07-22** — FLUX.2 Klein distilled/base split (defect fix, Grant's
  live-review catch). The single `flux2klein` row (24 steps / cfg 3.5,
  claimed "BFL Klein model card") matched **neither** Klein checkpoint's
  README: the step-distilled flagship `FLUX.2-klein-9B` documents
  `guidance_scale=1.0, num_inference_steps=4`; the non-distilled
  `FLUX.2-klein-base-9B` documents `guidance_scale=4.0, steps=50`. Both are
  bare `Flux2KleinPipeline`, but the flagship carries `is_distilled: true`
  in `model_index.json` — the same reliable marker as Krea-2, so this is
  the Krea pattern, not the Z-Image name-hint fallback. **Naming
  orientation is inverted from Krea, matching BFL's own:** the *marked
  distilled* checkpoint keeps the plain `"flux2klein"` (now 4 steps / cfg
  1.0, added to `DISTILLED_FAMILIES` — a 4-step budget destroys
  non-distilled weights); the *unmarked* checkpoint maps to new
  `"flux2klein-base"` (50 / 4.0). `flux2klein-base` added to
  `_build_call_kwargs`'s flux branch, `_NAG_CFG_OWNS_NEGATIVE` /
  `_NAG_MODULES` (nag_flux2, same rows as flux2klein), `_REF_FAMILY_KINDS`
  (ADR-036: refs work identically on both), and the catalog
  `_FAMILY_HINT_RULES` (`klein-base` hints, placed before the plain
  `klein` rules). Note `Flux2KleinPipeline` runs REAL CFG at cfg>1 (unlike
  flux/flux2 guidance embeds) — cfg 1.0 on the distill = CFG off, single
  pass. **Residual risk:** a distilled Klein repack that strips
  `is_distilled` gets base defaults (slow/overcooked, recognizable) — the
  safer failure direction than 4 steps on base weights (pure noise, the
  zimage-turbo lesson). Catalog DBs built before this split record the
  base checkpoint as `flux2klein`; rebuild via `catalog_cli build`. A
  second catalog consequence (code review 2026-07-22): LoRA sidecars
  hinting plain "klein" resolve to `flux2klein` (the distilled family)
  and so will not surface for `flux2klein-base` models under
  exact-family matching, even though Klein LoRAs are most plausibly
  base-trained — acceptable for now; revisit if Klein LoRA
  recommendations misfire. The sweep also covers the ComfyUI node layer
  (`eric_diffusion_{generate,multistage,ultragen,advanced_generate,
  advanced_multistage,flux2_edit}.py` family tuples), which shares
  `infer_model_family` via `detect_pipeline_class`.

## AI-Disclosure

Claude (Opus 4.7) authored; Grant reviewed. 2026-07-06 zimage-turbo
amendment: Claude (Fable 5) authored; Grant reviewed. 2026-07-22 Klein
split: Claude (Fable 5) authored; Grant reviewed.
