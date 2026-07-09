# ADR-023: NAG — Normalized Attention Guidance for distilled (cfg≤1) models

**Date:** 2026-07-08
**Status:** accepted (design approved by Grant 2026-07-08; implementation pending in a fresh session — see `docs/vision/slice-NAG-krea2-negative-guidance.md`)
**AI-Disclosure:** Claude (Fable 5) authored from two research passes (web method-verification + local wiring map); Grant reviewed and made the UX/scope calls.
**Relates to:** ADR-019 (the `--quant fp8` path NAG must coexist with), ADR-009 (family defaults), the 2026-07-08 quant-sidecar precedent (SCHEMA_KIND params).

---

## Context

Krea-2-Turbo is guidance-distilled: it runs at `cfg_scale 0.0`, 8 steps, and the
Krea2 pipeline **silently ignores `negative_prompt` when `guidance_scale <= 0`**
(`pipeline_krea2.py` L426-427, L582-594 — negative embeds are only computed under
classic CFG). The Turbo checkpoints have consistent quirks Grant wants to suppress,
and there is currently NO negative-guidance mechanism available on them.

**NAG (Normalized Attention Guidance)** — arXiv:2505.21179, Chen et al., NeurIPS
2025 — is a training-free negative-guidance method that operates in **attention-
output space** rather than score space, which is why it works where CFG is dead
(few-step / guidance-distilled models). Author-maintained reference code exists for
diffusers (github.com/ChenDarYen/Normalized-Attention-Guidance) and ComfyUI
(github.com/ChenDarYen/ComfyUI-NAG). **Neither supports Krea-2** (verified
2026-07-08: ComfyUI-NAG covers Flux/Kontext/Wan/HunyuanVideo/Chroma/HiDream/SD3.5/
SDXL/SD; diffusers core has no NAG at all — the `guiders` family is CFG-shaped).
We would be first on Krea-2.

### The method (paper Eqs. 7-10; verified against reference code)

At every attention layer where text conditioning participates, with Z⁺ the
attention output under the positive context and Z⁻ under the negative context
(same queries):

```
Z̃    = Z⁺ + φ·(Z⁺ − Z⁻)              # extrapolation; φ = nag_scale
R[i] = ‖Z̃[i]‖₁ / ‖Z⁺[i]‖₁            # per-token L1 norm ratio (dim=-1, keepdim)
Ẑ[i] = Z̃[i] · min(R[i], τ) / R[i]    # clip norm growth at τ = nag_tau
Z    = α·Ẑ + (1−α)·Z⁺                # blend; α = nag_alpha
```

Reference: `nag/attention_nag.py` L103-110 in the official repo (code form
`Z⁺·φ − Z⁻·(φ−1)`, identical). Note: the paper + cross-attn processor use **p=1**;
the author's Flux joint-attn file uses p=2. **We start with L1 per the paper.**

Defaults (paper Table 5; Flux few-step row is our analog): **φ=4-5 (Schnell demo
uses 5), τ=2.5, α=0.25**, full-window application for few-step models
(`nag_end=1.0`); few-step models tolerate higher τ/α and full windows.

### Krea-2 architecture facts that shape the port (verified in installed diffusers)

- **Single-stream joint self-attention** (`transformer_krea2.py`): text is fused
  (Qwen3-VL stack → `Krea2TextFusion` → `txt_in`) then **concatenated text-first
  with image tokens** into ONE sequence (L505), through 28 identical self-attention
  blocks; image tokens sliced back at L517. There is NO cross-attention. This is
  exactly the Flux *single-stream* case — `nag/attention_flux_nag.py` L109-176 in
  the official repo is the porting template. NAG applies to the **image-token slice**
  of the joint attention output; text tokens stay un-guided.
- **Processor hook exists:** `Krea2Attention(nn.Module, AttentionModuleMixin)` with
  `_default_processor_cls = Krea2AttnProcessor` (L94-95); the model carries
  `AttentionMixin` (L330) → `transformer.set_attn_processor(dict)` works. NAG state
  lives ON the processor instance — `attention_kwargs` cannot carry it (the
  processor signature has no `**kwargs`; forward filters kwargs, L128-135).
- **Lane re-sync is part of the method:** after each NAG application the guided
  image tokens are written back into BOTH batch lanes, so the lanes differ only in
  their text tokens — the negative branch is not an independent trajectory.
- **Fixed text length is a gift:** prompts pad to `max_sequence_length=512` in a
  `[prefix|prompt|PAD|suffix]` template (`pipeline_krea2.py` L229-241), so positive
  and negative embeds always share `text_seq_len`; batch-doubling needs no
  re-padding and reuses the single unbatched `position_ids`.
- **GQA 48 q / 12 kv heads** (head_dim 128) in main blocks; text_fusion blocks are
  20/20. Bool key-padding mask `(B,1,1,text+image)` built at L494-499.

## Decision

1. **Method: NAG, ported from the official Flux single-stream processor.**
   Alternatives noted for the record: VSF (arXiv:2508.10931 — cheaper, younger,
   benchmark later if NAG's overhead bothers us), Orthogonal Negative Guidance
   (arXiv:2605.29390 — concept-removal oriented, no confirmed code), NASA
   (arXiv:2412.02687 — NAG's un-normalized predecessor, artifact-prone). PAG/SEG/SLG
   are quality boosters, not negative prompting, and are CFG-family on distilled
   models.
2. **UX (Grant's call): reuse `--negative`.** `--nag-scale > 1` activates NAG and
   consumes the EXISTING negative prompt — "negative prompts now work on turbo."
   No separate `--nag-negative` in v1 (add later only if CFG-negative and
   NAG-negative ever need to differ on a model where both work).
3. **Scope v1: krea family only** (`krea`, `krea-turbo`), comfyless path only
   (ComfyUI nodes later if wanted). Naive batch-2 implementation — no image-token
   compute sharing (the paper's +87%-overhead optimization is explicitly deferred;
   correctness first).
4. **Params are sidecar-replayable** (SCHEMA_KIND, the quant precedent — NAG
   changes output content): `nag_scale` (default 0.0 = off; >1 activates),
   `nag_tau` (2.5), `nag_alpha` (0.25), `nag_end` (1.0, fraction of steps).
   Family-gated: non-krea families reject-or-warn loudly (no silent no-op).
5. **Implementation shape:** `NAGKrea2AttnProcessor` + a thin `Krea2Pipeline`
   subclass (encode negative via existing `encode_prompt`, cat embeds+masks on
   batch dim, tile latents/timestep, install processors on `transformer_blocks.*`
   ONLY, slice lane 0 of output, hot-restore processors when the `nag_end` window
   closes — reference pipeline pattern `pipeline_flux_nag.py` L284-285, L375,
   `_set_nag_attn_processor` L31-46).

## Binding implementation hazards (from the wiring map — each is a review item)

- **H1 — cuDNN pin ordering:** `set_attention_backend("_native_cudnn")`
  (comfyless/generate.py `_pin_krea_attention_backend`, ~:747/:766, applied at
  :992) writes `_attention_backend` onto EXISTING processor instances. A NAG
  processor installed after the pin starts with `None` → SDPA auto-select → math
  backend → the S² OOM the pin exists to prevent. The NAG processor must inherit/
  copy the backend attr (or the install order must guarantee re-pinning).
- **H2 — text_fusion exclusion:** `set_attn_processor` matches the text_fusion
  `Krea2Attention` instances too; only replace processors whose names start with
  `transformer_blocks.` (build from `attn_processors.keys()` and filter).
- **H3 — GQA:** the processor must reuse `dispatch_attention_fn` with
  `enable_gqa` exactly as stock (48/12, head_dim 128); never hand-build attention.
- **H4 — mask geometry:** batch-2 needs the two text masks catted on the batch
  dim; image-mask half stays all-ones; shape stays `(2B,1,1,text+image)`.
- **H5 — torchao under `--quant fp8`:** the processor must CALL the projection
  modules (`attn.to_q(...)` etc.) and never touch `.weight` or hand-matmul —
  quantized Linears break otherwise. Done right, NAG works under quant for free.
- **H6 — output-gate placement:** apply NAG to the image slice after the sigmoid
  gate multiply and before `to_out[0]` (mirrors the Flux "before out-projection"
  placement; lanes' image-token gates are identical because of lane re-sync).

## Cost (measured expectations, 8-step Turbo @1024², ~4608 tokens)

Active NAG steps run the transformer at batch 2: full-window ≈ **1.9-2.0× wall**;
`nag_end 0.5-0.75` ≈ 1.5-1.75×. VRAM +1-3 GB transient. One extra text-encoder
pass per generation (negligible). This is the price of having ANY negative
guidance on a distilled model — CFG would cost the same and doesn't work.

## Review bar

`code-reviewer` (Fable) required — non-trivial numerics + attention surgery.
`security-auditor` NOT triggered: no new caller-supplied content parsing, no IPC/
trust-boundary change (new params are typed through the established SCHEMA_KIND
machine-boundary validation), no code loading. The reviewer may escalate if the
implementation drifts from this scope.

## Alternatives Rejected

- Waiting for upstream: diffusers has no NAG and ComfyUI-NAG has no Krea-2;
  no signal either is coming.
- VSF as v1: less battle-tested; benchmark later against working NAG.
- Separate `--nag-negative` param: rejected for v1 (Grant; fewer knobs).
- Image-token compute sharing in v1: rejected — optimization after correctness.

## Deferred / Out of Scope

- ComfyUI node surface for NAG; other families (Flux/Qwen distilled variants
  could reuse the machinery later); compute-sharing optimization; per-block layer
  windows (paper applies to all layers uniformly); L2-norm variant.

## Changelog

- 2026-07-08 — Initial. Accepted (UX + session plan decided by Grant). Research
  basis: arXiv:2505.21179 + official repo (ChenDarYen/Normalized-Attention-
  Guidance: `nag/attention_flux_nag.py` single-stream path is the porting
  template; `nag/attention_nag.py` L103-110 the formula reference;
  `nag/pipeline_flux_nag.py` the pipeline pattern) + ComfyUI-NAG coverage check +
  installed-diffusers wiring map (`transformer_krea2.py`, `pipeline_krea2.py` —
  line refs above). Implementation to follow in a fresh session per
  `docs/vision/slice-NAG-krea2-negative-guidance.md`.
