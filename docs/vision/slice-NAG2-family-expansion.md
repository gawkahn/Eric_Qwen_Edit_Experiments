# Vision — Slice NAG2: family expansion (flux, flux2, flux2klein, zimage)

**Status:** implemented 2026-07-09 (see ADR-024 changelog; code-reviewer
blocking finding F1 folded); live per-family A/B by Grant pending.
Design: ADR-024. Risk: L2.
**AI-Disclosure:** Claude (Fable 5) authored; Grant directed the expansion.

## What must be true when done

1. `--negative-prompt ... --nag-scale 4` demonstrably influences output on
   flux, flux2, flux2klein, and (at cfg 0) zimage/zimage-turbo — same UX as
   krea, no new flags.
2. Family gate is a single table (`_nag_gate(family, nag_scale, cfg_scale)`):
   flux-family always NAG-eligible; zimage/krea families cfg-gated
   (cfg>0 → loud "CFG owns the negative" skip).
3. Every prior ADR-023 invariant holds per family: dormancy byte-identical
   at scale<=1; unconditional finally-restore (cached pipelines never leak);
   backend-pin copy; module-calls-only (quant-safe); loud skips via
   `nag_warnings` across daemon/MCP boundaries.
4. Existing krea NAG behavior is bit-unchanged (regression: test_nag suite
   green without edits to krea legs beyond the _nag_gate signature).

## What must never happen

- N1..N6 from slice-NAG carry over per architecture.
- N7: a Flux/Flux2 prefix filter that silently skips single-stream blocks
  (both `transformer_blocks.` and `single_transformer_blocks.` get NAG'd).
- N8: Z-Image refiner blocks (noise/context/siglip) NAG'd — joint
  `transformer.layers` ONLY.
- N9: Z-Image restore path that depends on AttentionMixin (it doesn't
  exist there) — hand-swap capture/restore must be self-contained.
- N10: NAG merge after Flux.2's fused to_out (the MLP is folded in — merge
  must hit the attention slice before the cat).
- N11: image-slice anchor wrong side (Z-Image is image-FIRST `[:, :x_len]`;
  flux-family is text-first `[:, txt:]`).

## How we prove it (CPU-only unless noted)

- Per-arch tiny-transformer tests mirroring test_nag.py's krea sections:
  processor dormancy (scale<=1 == stock, exact), lane re-sync through a
  full forward, image-slice-only rewrite, selection filter (Z-Image:
  refiners untouched; Flux/Flux2: both prefixes swapped), backend-pin copy,
  restore-to-original-instances.
- Gate-table tests: all 7 families × {scale off/on} × {cfg 0 / >0}.
- Routing guards per mirror (dormant delegate, CFG-interplay, ref-image
  warn+stock for Flux.2) via the monkeypatched-stock-call pattern.
- Live (Grant): same-seed A/B per family with a quirk-targeting negative.

## Change boundary / edit scope (declared)

- NEW `pipelines/nag_common.py` (nag_merge moves here; nag_krea2 re-exports).
- NEW `pipelines/nag_flux.py`, `pipelines/nag_flux2.py`,
  `pipelines/nag_zimage.py`.
- `pipelines/nag_krea2.py`: import nag_merge from nag_common (no behavior
  change).
- `comfyless/generate.py`: `_nag_gate` table rewrite (+cfg param), family→
  module dispatch for `nag_pipe_call`, fold the inline krea cfg check into
  the gate. Also fix the wrong `family_defaults.py` zimage-turbo comment
  (cfg 1.0 does NOT collapse CFG — pipeline gates at >0).
- Tests: extend `test_nag.py` (per-arch sections + gate table), adjust
  the two `_nag_gate` call-shape legs in `test_params_schema.py`.
- Docs: ADR-024 changelog close-out; manual pages lose the "only Krea"
  verbiage (Comfyless_Manual / Comfyless_Models / Comfyless_MCP + the two
  in-code help strings + MCP tool schema description).

## Out of scope

Chroma (CFG-routing fix is the right tool there); Flux.2 ref-image NAG;
Z-Image Omni; compute sharing; ComfyUI nodes; `--nag-negative`.
