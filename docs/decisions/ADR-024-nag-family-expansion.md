# ADR-024: NAG family expansion — Flux.1, Flux.2, Flux.2-Klein, Z-Image

**Date:** 2026-07-09
**Status:** accepted (Grant directed the expansion 2026-07-09: "do the NAG expansion")
**AI-Disclosure:** Claude (Fable 5) authored from three parallel wiring-map research passes over installed diffusers 0.39.0; load-bearing lines re-verified by the main session.
**Relates to:** ADR-023 (the Krea-2 NAG port this generalizes; formula, UX, and sidecar decisions carry over unchanged), ADR-009 (family gating), ADR-019 (quant coexistence).

---

## Context

ADR-023 shipped NAG (Normalized Attention Guidance, arXiv:2505.21179) for
krea/krea-turbo and deferred "other families." Verified live 2026-07-09.
The remaining families where negative prompts are dead in comfyless:

- **flux / flux2**: guidance-distilled — `guidance_scale` is an *embedding*,
  not CFG. comfyless's flux branch never forwards `negative_prompt`; the
  stock `Flux2Pipeline` doesn't even accept one. Negatives are dead at
  EVERY cfg value.
- **flux2klein**: same transformer as flux2; the Klein pipeline supports
  real CFG only when `is_distilled=false` and `guidance_scale > 1` — the
  distilled Klein checkpoint (the one worth running) is CFG-dead like Turbo.
- **zimage / zimage-turbo**: `ZImagePipeline` runs classic CFG whenever
  `guidance_scale > 0` (verified `pipeline_z_image.py:282-283` — note the
  `family_defaults.py` comment claiming cfg 1.0 "collapses to a single
  pass" is mechanically wrong; CFG is live at 1.0). Negatives are dead only
  in the cfg-0 regime — which is the documented Turbo recommendation
  (8 steps, guidance 0.0).

### Architecture facts that shape the ports (verified in installed diffusers 0.39.0)

| | Flux.1 | Flux.2 | Z-Image |
|---|---|---|---|
| Blocks | 19 dual-stream + 38 single-stream | 8 dual + 48 single (parallel/fused) | 2+2 refiners (single-modality) + 30 joint |
| Joint-seq order | **text-first** | **text-first** | **image-first** (`[x, cap]`, basic mode) |
| NAG image slice | dual: 2nd split output; single: `[:, txt:]` | dual: 2nd split output; single: `[:, txt:]` **before the fused to_out** | front `[:, :x_len]`, `x_len` = image tokens padded to ×32 |
| GQA | none (24 heads MHA) | none (48 heads MHA) | none (30 heads; `n_kv_heads` config exists but is ignored) |
| Gate | in block, outside processor | in block, outside processor | in block (tanh), outside processor |
| Rotary | `apply_rotary_emb(..., sequence_dim=1)` on (B,S,H,D) | same | complex `freqs_cis` applied inside processor |
| Processor install | `AttentionMixin.set_attn_processor`, keys `transformer_blocks.*` + `single_transformer_blocks.*` (ALL get NAG) | same two prefixes, **two processor classes** (dual vs parallel) | **no AttentionMixin** — hand-swap `transformer.layers[i].attention.processor`; refiners excluded |
| Text length | fixed ≤512 (T5) | **fixed 512** both encoders (Mistral3 base / Qwen3 Klein) | **variable/ragged** — per-sample lists, `pad_sequence` + bool mask handle unequal pos/neg lengths natively |
| Backend pins | `_attention_backend`/`_parallel_config` on every processor class (H1 applies) | same | same |
| Stock negative path | `true_cfg_scale` exists (unused by comfyless) | **none** — NAG mirror adds the negative encode | CFG at cfg>0 |
| Kwarg filtering | forward filters by processor signature → NAG state on the instance | same | same (and the pipeline never forwards attention kwargs at all) |

Flux.2's single-stream blocks are ViT-22B-style parallel blocks: QKV is
fused with the MLP-in projection and the attention out-projection is fused
with the MLP-out (`to_qkv_mlp_proj` / single `to_out`). The NAG merge must
happen on the attention output slice BEFORE the `cat([attn, mlp])` +
fused `to_out`.

The reference repo's Flux file uses **L2** norms; the paper and our Krea
port use **L1**. We standardize on **L1 everywhere** (deliberate deviation
from the reference Flux file, consistent with ADR-023).

## Decision

1. **Families added:** `flux`, `flux2`, `flux2klein`, `zimage`,
   `zimage-turbo` (joining `krea`, `krea-turbo`).
2. **Gate semantics (generalizes ADR-023's cfg guard):** NAG activates when
   `nag_scale > 1` AND the family is supported AND classic CFG is not the
   negative's owner for that config:
   - `flux`, `flux2`, `flux2klein`: **always eligible** (guidance embeds ≠
     CFG; comfyless never routes negatives to these families). Klein's
     non-distilled+cfg>1 real-CFG case is guarded in the mirror
     (warn + stock), matching the Krea pattern.
   - `zimage`, `zimage-turbo`: eligible at `cfg_scale <= 0` (CFG consumes
     the negative at cfg>0 — warn + skip, mirroring krea).
   - `krea`, `krea-turbo`: unchanged (eligible at cfg<=0).
   - `_nag_gate(model_family, nag_scale, cfg_scale)` gains the cfg param
     and owns the whole table; the krea-specific inline cfg check in
     `generate()` folds into it.
3. **Shared formula:** `nag_merge` moves to `pipelines/nag_common.py`;
   `nag_krea2` re-exports it (no test churn). L1, per-token, τ clip, α
   blend — byte-identical math across all families.
4. **Per-architecture modules**, each following the proven Krea-2 shape
   (per-call processor install, batch-2 `[positive|negative]` lanes, lane
   re-sync on image tokens, `nag_end` window, unconditional finally-restore,
   unbound `nag_pipe_call` for cached pipelines):
   - `pipelines/nag_flux.py` — one `NAGFluxAttnProcessor` (dual branch
     derives text length from `encoder_hidden_states`; single branch uses
     injected `text_seq_len`) + `NAGFluxPipeline` mirror. **Latents tiled
     to 2B** — the reference's B-latent tiling trick + Trunc-norm
     monkeypatch is rejected (installed single blocks use
     `AdaLayerNormZeroSingle`, which the patch doesn't cover). Reuses the
     module-level `_get_qkv_projections` helper.
   - `pipelines/nag_flux2.py` — `NAGFlux2AttnProcessor` (dual) +
     `NAGFlux2ParallelSelfAttnProcessor` (single; injected `text_seq_len`;
     merge before the fused out-proj) + `NAGFlux2Pipeline` (adds the
     negative encode the stock pipeline lacks) + `NAGFlux2KleinPipeline`.
   - `pipelines/nag_zimage.py` — `NAGZSingleStreamAttnProcessor` (front
     slice via injected `x_len`) + `NAGZImagePipeline` (list-based batch-2:
     `pos_list + neg_list`, ragged text allowed) + custom
     `apply/remove` helpers that hand-swap `transformer.layers[i]
     .attention.processor` (no AttentionMixin) and never touch the
     refiner stacks.
5. **Dispatch:** `generate.py` maps family → module `nag_pipe_call` with
   lazy per-family imports (a missing pipeline class in an older diffusers
   degrades to the loud-skip warning, not a crash).
6. **No schema/wire changes:** the ADR-023 quadruple, sidecar replay,
   cache-key exclusion, and `nag_warnings` channel already cover every
   family.

## Binding implementation hazards

- **H1 (backend pins)** applies to all three arches — copy
  `_attention_backend`/`_parallel_config` off every replaced processor.
- **H2 analog differs per arch:** Flux/Flux2 NAG **all** attention modules
  (both prefixes — a Krea-style single-prefix filter would silently skip
  38/48 single blocks); Z-Image NAGs ONLY `transformer.layers` (refiners
  are single-modality and must stay stock).
- **H5 (module calls only)** holds everywhere; torchao quant coexists.
- **H6 revised:** no in-processor gates outside Krea — NAG operates on the
  raw attention output; block-level gates are lane-identical because temb
  is (image tokens identical by re-sync induction).
- **HZ-1 (Z-Image restore):** without AttentionMixin there is no
  count-checked `set_attn_processor` — the custom remove helper must
  restore by walking the SAME `transformer.layers` list; capture originals
  before any swap; finally-restore unconditional.
- **HZ-2 (Z-Image x_len):** the processor's front slice needs the padded
  image-token count (`ceil(image_seq / SEQ_MULTI_OF) * SEQ_MULTI_OF`,
  SEQ_MULTI_OF=32) — computed in the mirror from the latent grid and
  injected; identical across lanes because both lanes share the latent
  resolution.
- **HF2-1 (Flux.2 ref images):** the kontext-style `image_latents` path is
  NOT NAG-supported in v1 — mirror warns + delegates to stock when
  reference images are in play.
- **HZ-3 (velocity sign):** the Z-Image mirror must keep the stock
  `noise_pred = -noise_pred` sign flip and lane-0 slice order exactly.

## Cost

Unchanged from ADR-023: NAG'd steps run batch-2 ≈ 1.9-2.0× wall
full-window; `nag_end 0.5-0.75` proportionally less. Flux.2 at 48+8 blocks
and Z-Image at 30 blocks carry the same relative overhead.

## Review bar

`code-reviewer` (Fable) required — three new attention-surgery surfaces.
`security-auditor` NOT triggered: no new params, no wire/schema change, no
caller-supplied content parsing; the only `generate.py` change is the gate
table + dispatch map (same exemption ADR-023's reviewer confirmed).

## Alternatives Rejected

- Reference-style B-latent tiling + Trunc-norm monkeypatch for Flux
  (fragile against `AdaLayerNormZeroSingle`; monkeypatching stock diffusers
  classes at runtime is exactly the global-state leak our per-call
  install/restore discipline exists to avoid).
- One mega-processor handling all arches (signatures, rotary conventions,
  and slice anchors differ too much; per-arch modules keep each mirror
  auditable against its stock pipeline).
- Chroma in this slice: de-distilled, real CFG works — the right fix there
  is routing negatives to CFG, not NAG. Deferred.
- L2 norms for Flux to match the reference file: rejected for cross-family
  consistency with the paper + ADR-023.

## Deferred / Out of Scope

Chroma (CFG routing fix instead); qwen-image distill LoRAs; Flux.2
reference-image (kontext) NAG; Z-Image Omni mode; ComfyUI nodes; compute
sharing (unchanged from ADR-023); separate `--nag-negative`.

## Changelog

- 2026-07-09 (verification) — VERIFIED live by Grant on flux, flux2, and
  zimage: effects present (not guaranteed per-image — expected for
  attention-space guidance), zimage confirmed to need cfg 0 (the gate's
  loud skip at cfg>0 is correct — Z-Image's real CFG owns the negative
  there), flux families confirmed working at their default guidance
  (guidance embeds ≠ CFG; no gate, as designed). `--quant fp8` + NAG
  coexist (H5 held) and fit Flux.2-dev on one GPU. Flux effect strength
  noted as softer than Krea — consistent with HF1-1 (negative flows via
  T5 tokens only; positive pooled shared) plus the strong distillation
  prior at guidance 3.5; tuning order: scale 5-6 → alpha 0.3-0.4 → lower
  guidance. flux2klein live check still pending.
- 2026-07-09 (later) — IMPLEMENTED. `pipelines/nag_common.py` (shared
  formula + tail/front/full lane-merge helpers), `pipelines/nag_flux.py`,
  `pipelines/nag_flux2.py` (two processor variants + shared denoise loop +
  base/Klein mirrors), `pipelines/nag_zimage.py` (hand-swap install,
  front-slice merge, list-based batch-2); gate table + importlib dispatch
  in `generate.py`; `family_defaults.py` zimage-turbo comment corrected.
  NEW HAZARD FOUND BY TESTS (HF1-1): Flux.1's temb includes the pooled
  CLIP text projection — per-lane pooled embeds diverge the image lanes
  and break NAG's same-queries requirement; the mirror tiles the POSITIVE
  pooled to both lanes (the reference repo's TruncAdaLayerNorm hack solves
  the same problem); pinned with a divergence negative-control test.
  code-reviewer (Fable) 2026-07-09: 1 blocking finding folded (Z-Image
  partial-swap restore — apply now appends into a caller-owned list
  BEFORE each swap, so a mid-apply failure restores the swapped prefix;
  contract test added) + 6 advisories folded (stale comments, deliberate
  cache_context omission documented, callback KeyError parity, zimage
  gate leg). security-auditor exemption per Review bar confirmed by the
  reviewer. Tests: test_nag.py 101 (+51), schema +10; all 17 suites
  2537/2537. Live per-family A/B by Grant pending.
- 2026-07-09 — Initial. Accepted. Research basis: three parallel wiring-map
  agents over installed diffusers 0.39.0 (transformer_flux.py,
  transformer_flux2.py + both Flux.2 pipelines, transformer_z_image.py +
  pipeline_z_image.py), load-bearing lines re-verified in the main session
  (CFG thresholds, processor bodies, concat orders, AttentionMixin
  availability).
