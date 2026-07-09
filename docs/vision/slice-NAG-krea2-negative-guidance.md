# Vision — Slice NAG: Normalized Attention Guidance for Krea-2 (comfyless)

**Status:** ready to implement (fresh session). Design: ADR-023. Risk: L2.
**AI-Disclosure:** Claude (Fable 5) authored; Grant approved design 2026-07-08.

## What must be true when done

1. `comfyless generate --model <krea-2-turbo> --negative "..." --nag-scale 4`
   produces an image where the negative prompt demonstrably influences output
   (A/B same-seed pair differs; quirk suppression judged by Grant).
2. NAG params (`nag_scale`, `nag_tau`, `nag_alpha`, `nag_end`) are recorded in
   the sidecar and replayed by `--params` (SCHEMA_KIND, quant precedent);
   explicit CLI wins via None-sentinels.
3. NAG works under `--quant fp8` (torchao Float8Tensor projections) AND at high
   resolution (cuDNN backend preserved — no SDPA math fallback).
4. `nag_scale` unset/<=1 → byte-identical behavior to today (NAG fully dormant;
   stock processors untouched).
5. Non-krea family + `--nag-scale` → loud warning naming the family, NAG skipped
   (warn-don't-block; no silent no-op).

## What must never happen

- N1: NAG silently inactive when requested on krea (the moody/LoRA lesson —
  every skip is loud).
- N2: text_fusion attention gets NAG'd (only `transformer_blocks.*`).
- N3: processor touches `.weight` of projections (breaks under quant).
- N4: swapped processors lose the cuDNN pin (S² OOM at high res).
- N5: negative branch drifts into an independent trajectory (lane re-sync after
  every block is part of the method).
- N6: stock processors not restored after the nag_end window / after generation
  (a cached pipeline must not leak NAG state into the next non-NAG request —
  the MCP/daemon pipeline cache makes this a REAL hazard, not theoretical).

## How we prove it

- Unit (CPU, no GPU): NAG formula on synthetic tensors (φ/τ/α math exact vs the
  reference equations incl. τ-clip inactive/active branches); processor-selection
  filter (transformer_blocks in, text_fusion out); schema round-trip for the four
  params (params-schema auto-growth + sentinel pins, quant-test template);
  family gating (krea accepts, others warn+skip); dormancy (scale<=1 installs
  nothing — assert attn_processors unchanged).
- Live (Grant): same-seed A/B on a Turbo checkpoint with a quirk-targeting
  negative; then the same under `--quant fp8`; then 2144² for the cuDNN check.
  Exit-code/warning check for the non-krea family case.

## Change boundary / edit scope (declared)

- NEW `pipelines/nag_krea2.py` (or `nodes/` — implementer's call, one module):
  `NAGKrea2AttnProcessor` + `Krea2NAGPipeline` (thin subclass) + an
  `apply/remove` helper pair.
- `comfyless/generate.py`: family-gated activation + pipeline-class selection for
  krea when NAG requested; argparse (None-sentinels); generate() signature;
  wire request builder; metadata emission.
- `comfyless/params_validation.py` + `comfyless/params_schema.py`: 4 new
  SCHEMA_KIND keys + defaults (nag_scale 0.0, nag_tau 2.5, nag_alpha 0.25,
  nag_end 1.0).
- `comfyless/server.py`: cache-key inclusion (NAG params change output — a
  cached pipeline serving a different nag config must miss; check whether they
  belong in `_request_cache_key` like quant or are per-request safe because the
  processor is installed per-call — DECIDE and test either way).
- `comfyless/mcp_server.py`: tool schema params + forwarding.
- Tests: extend `test_params_schema.py` (schema legs) + a new
  `test_nag.py` (formula + selection + gating; CPU-only) — or fold into an
  existing suite if small. CLAUDE.md counts.
- Docs: ADR-023 changelog close-out; TECH_DEBT for deferred items (compute
  sharing, other families, ComfyUI nodes).

## Implementation crib (from research — verified 2026-07-08)

- Porting template: official repo `nag/attention_flux_nag.py` L109-176 (single-
  stream branch), formula at `nag/attention_nag.py` L103-110, pipeline pattern
  `nag/pipeline_flux_nag.py` (embeds cat L284-285; window restore L375;
  `_set_nag_attn_processor` L31-46). Re-fetch from
  github.com/ChenDarYen/Normalized-Attention-Guidance (session scratchpad copies
  are gone).
- Local anchor points: `Krea2AttnProcessor.__call__` (transformer_krea2.py:58-88;
  output before out-proj at :86); joint concat :505; slice-back :517; GQA flag
  :82; mask build :494-499; `set_attn_processor` via AttentionMixin :330.
  comfyless: cuDNN pin :747/:766/:992; `_build_call_kwargs` krea branch :618-630;
  negative_prompt normalization :1203; `pipe(**call_kwargs)` :1249. Param-hop
  table: ADR-023 / the quant commit ad12b18.
- Defaults: φ=4 (try 5), τ=2.5, α=0.25, end=1.0 (8-step). L1 norm per paper.
- Cost expectation to verify live: ~1.9-2× wall full-window; ~1.5× at end 0.6.

## Out of scope

Compute-sharing optimization; VSF benchmark; other model families; ComfyUI
nodes; separate --nag-negative; L2-norm variant; per-layer windows.
