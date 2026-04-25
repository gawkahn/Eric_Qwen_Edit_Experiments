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

- **2026-04-25** — proposed and accepted (this document).

## AI-Disclosure

Claude (Opus 4.7) authored; Grant reviewed.
