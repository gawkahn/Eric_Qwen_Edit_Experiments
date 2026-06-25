# Slice Vision — Krea-2 (Raw / Turbo) support in comfyless

**Date:** 2026-06-25
**ADR:** Extends [ADR-009](../decisions/ADR-009-per-family-default-params.md) (per-family default params; Status: accepted) — this slice adds the `krea` / `krea-turbo` families and records that one pipeline class may map to two families via model metadata. Touches the [ADR-011](../decisions/ADR-011-comfyless-mcp-server.md) MCP `generate` surface (caller-side family-default application) and the [ADR-015](../decisions/ADR-015-mcp-catalog-reference-resolution.md) catalog scan (family classification).
**Status:** APPROVED by Grant 2026-06-25 (decisions: code-first / no nightly-or-git diffusers pin / no vendoring; Raw↔Turbo distinguished by the model's own `is_distilled` flag → two families; MCP caller applies `FAMILY_DEFAULTS` for CLI parity; daemon stays caller-responsible per ADR-009).
**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored; Grant reviewed and approved.

---

## Posture

> **Posture:** Boundary: integration (new model family flowing through the existing comfyless CLI / daemon / MCP generate paths). Risk factors: **no security truth** (no Red Zone surface; no new external-input handling — defaults are applied to already-validated param dicts). Broad-ish impact: the MCP family-default change affects *all* families, not just Krea. Risk level **L1**.

## Slice

Add first-class support for the two downloaded Krea-2 models —
`Krea-2-Raw` and `Krea-2-Turbo` — across the comfyless generate surfaces.
Both declare `_class_name: "Krea2Pipeline"` and use single-pass
`guidance_scale` (flux-like CFG), but differ in recommended defaults and
in the `is_distilled` flag in their `model_index.json` (Raw `false`,
Turbo `true`).

1. **Family detection** maps `Krea2Pipeline` → `krea`, and the
   distilled variant → `krea-turbo`, using the model's own
   `is_distilled` flag (not the directory name).
2. **Family defaults** give each variant its model-card sweet spot:
   Raw = 52 steps / cfg 3.5; Turbo = 8 steps / cfg 0.0 (CFG disabled).
3. **CFG routing** sends both families through the `guidance_scale`
   branch (negative prompt ignored, single forward pass).
4. **MCP parity** — the MCP `generate` caller now applies
   `FAMILY_DEFAULTS` (it previously used hardcoded fallbacks), so an
   agent that omits `steps`/`cfg_scale` gets the correct per-family
   defaults — matching the CLI client.

## Runtime gating (decided, not fixed here)

`Krea2Pipeline` exists **only on diffusers `main`** — no PyPI release
ships it (0.38.0 verified without it). The other Krea deps are already
satisfied by current pins (`transformers 5.5.3` has `Qwen3VLModel`;
`diffusers 0.37.1` has `AutoencoderKLQwenImage`). Per the approved
decision, this slice does **not** bump diffusers (no nightly/git pin, no
vendoring). Generation therefore raises the existing clear "upgrade
diffusers" `ValueError` at load time until a future tagged diffusers
release lands. All non-runtime support (catalog scan / `list_models`
classification, family defaults, CFG routing) works on the current pin
because the catalog scan classifies by reading `model_index.json` and
does not require the diffusers class to be importable. See `TECH_DEBT.md`
→ Dependencies (Krea runtime blocker) and the deferred dep slice.

## Four signals

- **Who** — the **CLI user** (`comfyless.generate`), the **daemon**
  (`comfyless/server.py`) via its CLI client, and the **LLM agent**
  (`comfyless/mcp_server.py`). The operator configures `--model-base` so
  the Krea dirs are discoverable.
- **Data** — *read*: each model's `model_index.json` (`_class_name`,
  `is_distilled`) at scan time and load time; the static
  `FAMILY_DEFAULTS` dict. *Written*: family-default values into the param
  dict for non-explicit keys (ADR-009 precedence ladder).
- **Boundary** — `nodes/eric_diffusion_utils.py`
  (`infer_model_family`, `detect_pipeline_class`); `comfyless/catalog.py`
  (`scan_model_family`); `comfyless/family_defaults.py`;
  `comfyless/generate.py` (`_build_call_kwargs`);
  `comfyless/mcp_server.py` (generate handler + tool docstring). The
  daemon (`server.py`) needs no family-default change (ADR-009: defaults
  are caller-responsibility; the CLI client already applies them) and
  inherits CFG routing for free.
- **Failure** — a model whose `model_index.json` is missing/unreadable
  classifies as no-family and passes through unchanged (existing
  behavior). On the current diffusers pin, `generate` fails closed at
  load with the existing "upgrade diffusers" error — never a silent
  wrong-pipeline load.

## Risk level

**L1.** Additive: new family rows + one new substring pattern + adding
two strings to an existing CFG-routing branch. The one cross-cutting
change is the MCP family-default application, which alters MCP behavior
for *all* families (they now get their declared defaults instead of
hardcoded 28/3.5). Runs `code-reviewer` (Opus) per slice before commit.
No `security-auditor` trigger: no Red Zone surface, and no new external
input handling (the MCP change applies defaults to an already-validated
payload). If the daemon's input handling were to change, re-evaluate —
it is not changing here.

## Intent

Make Krea-2-Raw and Krea-2-Turbo usable across the comfyless generate
surfaces with their correct per-variant defaults and CFG routing, derived
from the models' own metadata, without bumping diffusers — so the moment
a diffusers release ships `Krea2Pipeline`, generation works with zero
further code change.

## Invariants (must always be true)

1. **`infer_model_family("Krea2Pipeline")` → `"krea"`**; with
   `is_distilled=True` → `"krea-turbo"`. The single-arg call form is
   byte-compatible for every existing caller (new param defaults False).
2. **Family is derived from metadata, not directory name.** Renaming or
   relocating the model directory does not change its family.
3. **`FAMILY_DEFAULTS["krea"] = {cfg_scale: 3.5, steps: 52}` and
   `["krea-turbo"] = {cfg_scale: 0.0, steps: 8}`**, applied only to keys
   not explicit and not iterated (ADR-009 precedence preserved; explicit
   caller values still win — negative case).
4. **Both krea families route through `guidance_scale`** in
   `_build_call_kwargs` (not `true_cfg_scale`); `cfg_scale=0.0` is passed
   through unchanged for Turbo.
5. **Catalog scan classifies the two dirs** as `krea` / `krea-turbo` on
   the current diffusers pin (no class import required).
6. **MCP `generate` applies `FAMILY_DEFAULTS`** for canonical keys absent
   from the agent payload; keys the agent set explicitly are untouched.
7. **No behavior change for existing families** beyond the MCP path now
   honoring their already-declared `FAMILY_DEFAULTS` (intended).
8. **No diffusers/transformers dependency change** in this slice.

## Proof

- Unit suites stay at 0 failures via `./.venv/bin/python3`:
  `test_manual_loop.py`, `test_params_schema.py`, `test_mcp_server.py`,
  `test_server_robustness.py`.
- New cases: family detection (both variants); `FAMILY_DEFAULTS` overlay
  + explicit-key negative case; `_build_call_kwargs` routes krea to
  `guidance_scale` incl. `cfg=0.0`; catalog scan reports the right family
  for each Krea dir; MCP generate applies family defaults and a
  daemon generate request routes krea CFG correctly.

## Out of scope / deferred

- **Diffusers dependency bump** to enable actual generation — separate,
  ADR'd slice gated on a tagged release exporting `Krea2Pipeline`.
- **Turbo `mu`/timestep-shift (1.15)** — no `COMFYLESS_SCHEMA` knob;
  diffusers' default dynamic shift is used. TECH_DEBT entry.
- **Custom sampler swaps on krea** — untested (flux-like flow-match);
  default sampler only.
- **Daemon-side family-default injection** — explicitly NOT added;
  ADR-009 keeps defaults caller-responsible and the CLI client supplies
  them.
