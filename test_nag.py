#!/usr/bin/env python3
"""Tests for NAG (Normalized Attention Guidance) on Krea-2 — ADR-023.

CPU-only, no GPU, no model weights: exercises the NAG formula against the
reference equations (paper arXiv:2505.21179 Eqs. 7-10 / official repo
attention_nag.py L103-110), the processor-selection filter (hazards H1/H2),
processor-level dormancy and lane re-sync on a tiny Krea2Transformer2DModel,
and the pipeline-level routing guards (dormant delegate, CFG interplay,
unbound nag_pipe_call).

Sidecar/schema legs live in test_params_schema.py; daemon cache-key and
forwarding legs live in test_server_robustness.py.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from pipelines import nag_krea2 as nag
from diffusers.models.transformers.transformer_krea2 import (
    Krea2Attention,
    Krea2AttnProcessor,
    Krea2Transformer2DModel,
)
from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


torch.manual_seed(0)


# ──────────────────────────────────────────────────────────────────────
print("── nag_merge formula vs reference equations ───────────────────")

# Reference (attention_nag.py L103-110, L1 per paper):
#   z_g   = z_pos * s - z_neg * (s - 1)
#   ratio = |z_g|_1 / |z_pos|_1          (per token, dim=-1, keepdim)
#   z_g   = z_g * min(ratio, tau) / ratio
#   out   = z_g * alpha + z_pos * (1 - alpha)
def _reference_nag(z_pos, z_neg, s, tau, alpha):
    z_g = z_pos * s - z_neg * (s - 1.0)
    norm_pos = torch.norm(z_pos, p=1, dim=-1, keepdim=True)
    norm_g = torch.norm(z_g, p=1, dim=-1, keepdim=True)
    ratio = norm_g / norm_pos
    z_g = z_g * torch.minimum(ratio, torch.full_like(ratio, tau)) / ratio
    return z_g * alpha + z_pos * (1.0 - alpha)


_zp = torch.randn(2, 6, 16)

# tau-INACTIVE branch: z_neg nearly equal to z_pos keeps the norm ratio
# ~1 << tau, so the clip must be a no-op and the result must equal the
# UNCLIPPED extrapolation blend exactly.
_zn_close = _zp + 0.01 * torch.randn_like(_zp)
_out = nag.nag_merge(_zp, _zn_close, 4.0, 2.5, 0.25)
_unclipped = (_zp * 4.0 - _zn_close * 3.0) * 0.25 + _zp * 0.75
check("tau-inactive branch equals unclipped extrapolation blend",
      torch.allclose(_out, _unclipped, atol=1e-6),
      f"max diff {(_out - _unclipped).abs().max().item()}")
check("tau-inactive branch matches reference equations",
      torch.allclose(_out, _reference_nag(_zp, _zn_close, 4.0, 2.5, 0.25),
                     atol=1e-6))

# tau-ACTIVE branch: z_neg = -z_pos makes z_g = (2s-1)*z_pos, so the norm
# ratio is exactly 2s-1 = 7 > tau. The clipped z_g must be tau/(2s-1) of
# the raw extrapolation.
_zn_opp = -_zp
_out = nag.nag_merge(_zp, _zn_opp, 4.0, 2.5, 0.25)
_zg_raw = 7.0 * _zp
_expected = (_zg_raw * (2.5 / 7.0)) * 0.25 + _zp * 0.75
check("tau-active branch clips norm growth at tau",
      torch.allclose(_out, _expected, atol=1e-5),
      f"max diff {(_out - _expected).abs().max().item()}")
check("tau-active branch matches reference equations",
      torch.allclose(_out, _reference_nag(_zp, _zn_opp, 4.0, 2.5, 0.25),
                     atol=1e-6))

# Random-tensor equivalence across a parameter sweep (both branches mixed
# per token) — the implementation IS the reference, element-exact.
for _s, _tau, _alpha in ((2.0, 1.5, 0.5), (5.0, 2.5, 0.25), (8.0, 3.5, 1.0)):
    _zn = torch.randn_like(_zp) * 3.0
    _got = nag.nag_merge(_zp, _zn, _s, _tau, _alpha)
    _ref = _reference_nag(_zp, _zn, _s, _tau, _alpha)
    check(f"reference-exact at scale={_s}, tau={_tau}, alpha={_alpha}",
          torch.allclose(_got, _ref, atol=1e-6),
          f"max diff {(_got - _ref).abs().max().item()}")

# scale=1 is the mathematical no-op: z_g == z_pos, ratio == 1, out == z_pos.
_out = nag.nag_merge(_zp, torch.randn_like(_zp), 1.0, 2.5, 0.25)
check("scale=1 returns z_pos exactly (mathematical no-op)",
      torch.allclose(_out, _zp, atol=1e-6))

# The clip is PER TOKEN (dim=-1): a token whose growth exceeds tau is
# clipped while a same-batch token below tau is not.
_zp2 = torch.ones(1, 2, 4)
_zn2 = torch.stack([torch.ones(4) * 0.99, -torch.ones(4)], dim=0)[None]
_out2 = nag.nag_merge(_zp2, _zn2, 4.0, 2.5, 0.25)
_ref2 = _reference_nag(_zp2, _zn2, 4.0, 2.5, 0.25)
check("clip decision is per-token (mixed branches in one call)",
      torch.allclose(_out2, _ref2, atol=1e-6))
_ratio_tok1 = (_zp2[0, 1] * 4.0 - _zn2[0, 1] * 3.0).abs().sum() / _zp2[0, 1].abs().sum()
check("per-token setup actually spans both branches (sanity)",
      _ratio_tok1.item() > 2.5)


# ──────────────────────────────────────────────────────────────────────
print("\n── processor selection filter (H1/H2) on tiny transformer ────")

torch.manual_seed(1)
_tiny = Krea2Transformer2DModel(
    in_channels=16,
    num_layers=2,
    attention_head_dim=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=64,
    timestep_embed_dim=16,
    text_hidden_dim=32,
    num_text_layers=2,
    text_num_attention_heads=2,
    text_num_key_value_heads=2,
    text_intermediate_size=48,
    num_layerwise_text_blocks=1,
    num_refiner_text_blocks=1,
    axes_dims_rope=(4, 2, 2),
).eval()

_before = dict(_tiny.attn_processors)
_main_keys = [k for k in _before if k.startswith("transformer_blocks.")]
_fusion_keys = [k for k in _before if not k.startswith("transformer_blocks.")]
check("tiny model has both main-block and text_fusion processors",
      len(_main_keys) == 2 and len(_fusion_keys) >= 2,
      f"main={len(_main_keys)}, fusion={len(_fusion_keys)}")

# H1: mimic comfyless's cuDNN pin BEFORE the NAG install — the pin writes
# _attention_backend onto the EXISTING processor instances.
for _proc in _before.values():
    _proc._attention_backend = "_native_cudnn"

_origin = nag.apply_nag_processors(
    _tiny, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=4)
_after = _tiny.attn_processors

check("all transformer_blocks.* processors replaced with NAG (H2)",
      all(isinstance(_after[k], nag.NAGKrea2AttnProcessor)
          for k in _main_keys))
check("text_fusion processors untouched — same instances (H2)",
      all(_after[k] is _before[k] for k in _fusion_keys))
check("NAG processors inherit the cuDNN backend pin (H1)",
      all(_after[k]._attention_backend == "_native_cudnn"
          for k in _main_keys))
check("NAG processors carry the install-time NAG state",
      all(_after[k].nag_scale == 4.0 and _after[k].nag_tau == 2.5
          and _after[k].nag_alpha == 0.25 and _after[k].text_seq_len == 4
          for k in _main_keys))
check("apply returns the original processor mapping",
      set(_origin) == set(_before)
      and all(_origin[k] is _before[k] for k in _before))

nag.remove_nag_processors(_tiny, _origin)
_restored = _tiny.attn_processors
check("remove restores the ORIGINAL processor instances (N6)",
      all(_restored[k] is _before[k] for k in _before))
check("origin dict survives restore (re-restorable)",
      set(_origin) == set(_before))
nag.remove_nag_processors(_tiny, _origin)  # idempotent — must not raise
check("restore is repeatable with the same origin dict", True)

# Undo the simulated cuDNN pin — the CPU-only forwards below need the
# default (auto-select) backend.
for _proc in _before.values():
    _proc._attention_backend = None


# ──────────────────────────────────────────────────────────────────────
print("\n── processor dormancy + lane re-sync (module level) ───────────")

torch.manual_seed(2)
_attn = Krea2Attention(hidden_size=32, num_heads=4, num_kv_heads=2).eval()
_TXT = 4
_x_img = torch.randn(1, 6, 32)
_x_txt_pos = torch.randn(1, _TXT, 32)
_x_txt_neg = torch.randn(1, _TXT, 32)
# Joint [text | image] lanes: identical image tokens, different text.
_lane_pos = torch.cat([_x_txt_pos, _x_img], dim=1)
_lane_neg = torch.cat([_x_txt_neg, _x_img], dim=1)
_joint = torch.cat([_lane_pos, _lane_neg], dim=0)

with torch.no_grad():
    _attn.set_processor(Krea2AttnProcessor())
    _stock_out = _attn(_joint)

    # Dormant NAG processor (scale<=1) is op-for-op the stock processor.
    _attn.set_processor(nag.NAGKrea2AttnProcessor(
        nag_scale=1.0, text_seq_len=_TXT))
    _dormant_out = _attn(_joint)
check("dormant NAG processor (scale=1) equals stock output exactly",
      torch.equal(_dormant_out, _stock_out),
      f"max diff {(_dormant_out - _stock_out).abs().max().item()}")

with torch.no_grad():
    _attn.set_processor(nag.NAGKrea2AttnProcessor(
        nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=_TXT))
    _nag_out = _attn(_joint)
check("active NAG changes the positive lane's image tokens",
      not torch.allclose(_nag_out[0, _TXT:], _stock_out[0, _TXT:]))
check("lane re-sync: both lanes carry identical guided image tokens (N5)",
      torch.equal(_nag_out[0, _TXT:], _nag_out[1, _TXT:]))
# Text tokens must match what the stock processor produces for the SAME
# joint input — NAG rewrites the image slice only (invariant N2 analogue
# at the token level).
check("text tokens equal stock output (image-slice-only rewrite)",
      torch.equal(_nag_out[:, :_TXT], _stock_out[:, :_TXT]))

with torch.no_grad():
    # Negative control: identical lanes -> z_neg == z_pos -> NAG collapses
    # to the identity and the output equals stock.
    _joint_same = torch.cat([_lane_pos, _lane_pos], dim=0)
    _attn.set_processor(Krea2AttnProcessor())
    _stock_same = _attn(_joint_same)
    _attn.set_processor(nag.NAGKrea2AttnProcessor(
        nag_scale=4.0, text_seq_len=_TXT))
    _nag_same = _attn(_joint_same)
check("identical lanes -> NAG is a no-op (NEGATIVE control)",
      torch.allclose(_nag_same, _stock_same, atol=1e-5),
      f"max diff {(_nag_same - _stock_same).abs().max().item()}")

with torch.no_grad():
    # Guards: odd batch or unknown text length -> guidance silently
    # disabled at the processor (the pipeline never produces these during
    # the NAG window; the guard keeps stray shapes safe).
    _attn.set_processor(nag.NAGKrea2AttnProcessor(
        nag_scale=4.0, text_seq_len=None))
    _no_txt = _attn(_joint)
check("text_seq_len=None disables guidance (equals stock)",
      torch.equal(_no_txt, _stock_out))
with torch.no_grad():
    _attn.set_processor(Krea2AttnProcessor())
    _stock_odd = _attn(_joint[:1])
    _attn.set_processor(nag.NAGKrea2AttnProcessor(
        nag_scale=4.0, text_seq_len=_TXT))
    _nag_odd = _attn(_joint[:1])
check("odd batch disables guidance (equals stock)",
      torch.equal(_nag_odd, _stock_odd))


# ──────────────────────────────────────────────────────────────────────
print("\n── end-to-end tiny-transformer forward (H4 mask geometry) ─────")

torch.manual_seed(3)
_B, _TXTS, _GH, _GW = 1, 4, 2, 3
_IMG = _GH * _GW
_hidden = torch.randn(_B, _IMG, 16)
_enc_pos = torch.randn(_B, _TXTS, 2, 32)
_enc_neg = torch.randn(_B, _TXTS, 2, 32)
_mask_pos = torch.tensor([[True, True, True, True]])
_mask_neg = torch.tensor([[True, True, False, False]])
_pos_ids = Krea2Pipeline.prepare_position_ids(_TXTS, _GH, _GW, torch.device("cpu"))
_t = torch.tensor([0.5])

with torch.no_grad():
    _stock_lane0 = _tiny(
        hidden_states=_hidden,
        encoder_hidden_states=_enc_pos,
        timestep=_t,
        position_ids=_pos_ids,
        encoder_attention_mask=_mask_pos,
        return_dict=False,
    )[0]

    _origin2 = nag.apply_nag_processors(
        _tiny, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=_TXTS)
    try:
        _out2b = _tiny(
            hidden_states=torch.cat([_hidden, _hidden], dim=0),
            encoder_hidden_states=torch.cat([_enc_pos, _enc_neg], dim=0),
            timestep=torch.cat([_t, _t], dim=0),
            position_ids=_pos_ids,
            encoder_attention_mask=torch.cat([_mask_pos, _mask_neg], dim=0),
            return_dict=False,
        )[0]
    finally:
        nag.remove_nag_processors(_tiny, _origin2)

check("batch-2 forward with per-lane masks runs (H4 geometry)",
      _out2b.shape == (2, _IMG, 16), f"shape {tuple(_out2b.shape)}")
check("velocity predictions identical across lanes (re-sync through "
      "all blocks + final layer)",
      torch.allclose(_out2b[0], _out2b[1], atol=1e-5),
      f"max diff {(_out2b[0] - _out2b[1]).abs().max().item()}")
check("NAG'd prediction differs from stock single-lane prediction "
      "(guidance has effect)",
      not torch.allclose(_out2b[0], _stock_lane0[0], atol=1e-4))

with torch.no_grad():
    _post = _tiny(
        hidden_states=_hidden,
        encoder_hidden_states=_enc_pos,
        timestep=_t,
        position_ids=_pos_ids,
        encoder_attention_mask=_mask_pos,
        return_dict=False,
    )[0]
check("stock behavior restored after remove (no NAG leak, N6)",
      torch.equal(_post, _stock_lane0))


# ──────────────────────────────────────────────────────────────────────
print("\n── pipeline routing guards (dormant / CFG interplay) ──────────")

_recorded: dict = {}
_orig_call = Krea2Pipeline.__call__


def _fake_stock_call(self, **kw):
    _recorded.clear()
    _recorded.update(kw)
    return "STOCK"


Krea2Pipeline.__call__ = _fake_stock_call
try:
    _p = object.__new__(nag.Krea2NAGPipeline)

    # Dormant (scale<=1): delegates to the stock __call__ with the stock
    # kwargs — no nag_* keys leak through (invariant 4).
    _r = nag.Krea2NAGPipeline.__call__(_p, prompt="x", nag_scale=0.0,
                                       guidance_scale=0.0)
    check("dormant NAG delegates to stock __call__", _r == "STOCK")
    check("dormant delegate forwards the prompt",
          _recorded.get("prompt") == "x")
    check("dormant delegate leaks no nag_* kwargs",
          not any(k.startswith("nag_") for k in _recorded))

    # CFG interplay: classic CFG active (guidance_scale>0) + NAG requested
    # -> loud skip to stock CFG (never silent, never combined in v1).
    _r = nag.Krea2NAGPipeline.__call__(_p, prompt="x", nag_scale=5.0,
                                       guidance_scale=3.5,
                                       negative_prompt="bad")
    check("NAG + classic CFG routes to stock CFG (v1 scope)", _r == "STOCK")
    check("CFG route keeps the negative prompt for CFG",
          _recorded.get("negative_prompt") == "bad")

    # nag_pipe_call runs the NAG __call__ UNBOUND on a stock pipeline
    # instance (the daemon/MCP cached-pipeline path).
    _sp = object.__new__(Krea2Pipeline)
    _r = nag.nag_pipe_call(_sp, prompt="y", nag_scale=0.5)
    check("nag_pipe_call works unbound on a stock Krea2Pipeline", _r == "STOCK")
    check("nag_pipe_call dormant path forwards kwargs",
          _recorded.get("prompt") == "y")
finally:
    Krea2Pipeline.__call__ = _orig_call


# ──────────────────────────────────────────────────────────────────────
print("\n── source pins (window / restore / integration) ───────────────")

import inspect

_src = inspect.getsource(nag.Krea2NAGPipeline.__call__)
check("nag_end window predicate present (step-index fraction)",
      "i >= nag_end * self._num_timesteps" in _src)
check("processors restored in a finally (N6 pin)",
      "finally:" in _src and "remove_nag_processors" in
      _src.split("finally:", 1)[1])
check("lane-0 slice of the batch-2 prediction",
      "noise_pred[: latents.shape[0]]" in _src)

# Window predicate semantics (documented behavior of the inline check).
_window = lambda end, n: [i for i in range(n) if not (i >= end * n)]
check("nag_end=1.0 keeps NAG on for all steps",
      _window(1.0, 8) == list(range(8)))
check("nag_end=0.5 keeps NAG on for the first half",
      _window(0.5, 8) == [0, 1, 2, 3])
check("nag_end=0.0 never applies NAG",
      _window(0.0, 8) == [])

_gen_src = Path("comfyless/generate.py").read_text()
check("generate() routes NAG through nag_pipe_call under swap_sampler",
      "nag_pipe_call(pipe, **call_kwargs)" in _gen_src)
check("generate() gates NAG by family via _nag_gate (cfg-aware, ADR-024)",
      "_nag_gate(model_family, nag_scale, cfg_scale)" in _gen_src)
check("generate() dispatches NAG modules per family (ADR-024)",
      "_NAG_MODULES[model_family]" in _gen_src)

# Review-fold pins (code-reviewer 2026-07-08): N1 loudness must CROSS the
# daemon/MCP boundary — stderr inside generate() is a server log there.
# (ADR-024: the CFG-interplay rule folded INTO _nag_gate's family table.)
check("CFG-interplay rule lives in the gate table (client-visible skip)",
      "cfg_owns and cfg_scale > 0" in _gen_src)
check("NAG skips ride metadata as nag_warnings (N1 boundary channel)",
      'metadata["nag_warnings"] = nag_warnings' in _gen_src)
check("daemon client surfaces wire nag_warnings on stderr",
      'metadata.get("nag_warnings")' in _gen_src)
_mcp_src = Path("comfyless/mcp_server.py").read_text()
check("MCP surfaces nag_warnings as agent notices",
      "WARNING: NAG" in _mcp_src
      and 'metadata.get("nag_warnings")' in _mcp_src)
check("finally-restore is UNCONDITIONAL (partial-swap-proof N6)",
      "if nag_applied:" not in _src.split("finally:", 1)[1])


# ══════════════════════════════════════════════════════════════════════
# ADR-024 family expansion — Flux.1 / Flux.2 / Z-Image
# ══════════════════════════════════════════════════════════════════════

from pipelines import nag_common
from pipelines import nag_flux
from pipelines import nag_flux2
from pipelines import nag_zimage

print("\n── nag_common: shared formula (ADR-024) ───────────────────────")
check("nag_krea2.nag_merge IS nag_common.nag_merge (single source)",
      nag.nag_merge is nag_common.nag_merge)
_zp3 = torch.randn(2, 5, 8)
_zn3 = torch.randn(2, 5, 8)
_joint3 = torch.cat([torch.randn(2, 3, 8), torch.zeros(2, 5, 8)], dim=1)
_joint3[:1, 3:] = _zp3[:1]
_joint3[1:, 3:] = _zn3[:1]
_tail = nag_common.nag_lane_merge_tail(_joint3.clone(), 3, 4.0, 2.5, 0.25)
_ref3 = _reference_nag(_zp3[:1], _zn3[:1], 4.0, 2.5, 0.25)
check("lane_merge_tail applies the formula to the tail image slice",
      torch.allclose(_tail[0, 3:], _ref3[0], atol=1e-6))
check("lane_merge_tail re-syncs both lanes", torch.equal(_tail[0, 3:], _tail[1, 3:]))
_jf = torch.cat([torch.zeros(2, 4, 8), torch.randn(2, 3, 8)], dim=1)
_jf[:1, :4] = _zp3[:1, :4]
_jf[1:, :4] = _zn3[:1, :4]
_front = nag_common.nag_lane_merge_front(_jf.clone(), 4, 4.0, 2.5, 0.25)
check("lane_merge_front applies the formula to the FRONT image slice (N11)",
      torch.allclose(_front[0, :4],
                     _reference_nag(_zp3[:1, :4], _zn3[:1, :4], 4.0, 2.5, 0.25)[0],
                     atol=1e-6))
check("lane_merge_front re-syncs both lanes", torch.equal(_front[0, :4], _front[1, :4]))
check("lane_merge_front leaves text tokens per-lane",
      not torch.allclose(_front[0, 4:], _front[1, 4:]))


print("\n── Flux.1: selection / dormancy / re-sync ─────────────────────")

from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

torch.manual_seed(10)
_ftiny = FluxTransformer2DModel(
    patch_size=1,
    in_channels=8,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=4,
    joint_attention_dim=16,
    pooled_projection_dim=16,
    guidance_embeds=False,
    axes_dims_rope=(2, 4, 2),
).eval()

_fbefore = dict(_ftiny.attn_processors)
check("tiny Flux has BOTH dual and single prefixes",
      any(k.startswith("transformer_blocks.") for k in _fbefore)
      and any(k.startswith("single_transformer_blocks.") for k in _fbefore))
for _p in _fbefore.values():
    _p._attention_backend = "_native_cudnn"
_forigin = nag_flux.apply_nag_flux_processors(
    _ftiny, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=3)
_fafter = _ftiny.attn_processors
check("ALL Flux attention processors replaced (N7 — no prefix skipped)",
      all(isinstance(p, nag_flux.NAGFluxAttnProcessor) for p in _fafter.values()))
check("Flux NAG processors inherit backend pins (H1)",
      all(p._attention_backend == "_native_cudnn" for p in _fafter.values()))
nag_flux.remove_nag_flux_processors(_ftiny, _forigin)
check("Flux restore returns the ORIGINAL instances (N6)",
      all(_ftiny.attn_processors[k] is _fbefore[k] for k in _fbefore))
for _p in _fbefore.values():
    _p._attention_backend = None

_B, _TXT_F, _IMG_F = 1, 3, 4
_fx = torch.randn(_B, _IMG_F, 8)
_ftxt_pos = torch.randn(_B, _TXT_F, 16)
_ftxt_neg = torch.randn(_B, _TXT_F, 16)
_fpool_pos = torch.randn(_B, 16)
_fpool_neg = torch.randn(_B, 16)
_fimg_ids = torch.zeros(_IMG_F, 3)
_fimg_ids[:, 1] = torch.arange(_IMG_F)
_ftxt_ids = torch.zeros(_TXT_F, 3)
_ft = torch.tensor([0.5])


def _flux_fwd(model, x, txt, pool, tvec):
    return model(
        hidden_states=x,
        encoder_hidden_states=txt,
        pooled_projections=pool,
        timestep=tvec,
        img_ids=_fimg_ids,
        txt_ids=_ftxt_ids,
        guidance=None,
        return_dict=False,
    )[0]


with torch.no_grad():
    _f_stock1 = _flux_fwd(_ftiny, _fx, _ftxt_pos, _fpool_pos, _ft)
    _f_stock2 = _flux_fwd(
        _ftiny,
        torch.cat([_fx, _fx]), torch.cat([_ftxt_pos, _ftxt_neg]),
        torch.cat([_fpool_pos, _fpool_neg]), torch.cat([_ft, _ft]),
    )

    _forigin = nag_flux.apply_nag_flux_processors(
        _ftiny, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=_TXT_F)
    try:
        _f_dorm = _flux_fwd(
            _ftiny,
            torch.cat([_fx, _fx]), torch.cat([_ftxt_pos, _ftxt_neg]),
            torch.cat([_fpool_pos, _fpool_neg]), torch.cat([_ft, _ft]),
        )
    finally:
        nag_flux.remove_nag_flux_processors(_ftiny, _forigin)
check("Flux dormant NAG (scale=1) equals stock batch-2 exactly",
      torch.equal(_f_dorm, _f_stock2),
      f"max diff {(_f_dorm - _f_stock2).abs().max().item()}")

with torch.no_grad():
    _forigin = nag_flux.apply_nag_flux_processors(
        _ftiny, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=_TXT_F)
    try:
        # HF1-1: the pipeline feeds the POSITIVE pooled embeds to BOTH
        # lanes — Flux.1's temb includes the pooled text projection, so
        # per-lane pooled would modulate each lane differently and break
        # lane identity. Mirror that here.
        _f_nag = _flux_fwd(
            _ftiny,
            torch.cat([_fx, _fx]), torch.cat([_ftxt_pos, _ftxt_neg]),
            torch.cat([_fpool_pos, _fpool_pos]), torch.cat([_ft, _ft]),
        )
        # NEGATIVE control pinning HF1-1: per-lane pooled DIVERGES the
        # lanes (this is exactly the failure the pipeline design avoids).
        _f_nag_bad = _flux_fwd(
            _ftiny,
            torch.cat([_fx, _fx]), torch.cat([_ftxt_pos, _ftxt_neg]),
            torch.cat([_fpool_pos, _fpool_neg]), torch.cat([_ft, _ft]),
        )
    finally:
        nag_flux.remove_nag_flux_processors(_ftiny, _forigin)
check("Flux lane re-sync: velocity predictions identical across lanes "
      "(positive pooled tiled to both — HF1-1)",
      torch.allclose(_f_nag[0], _f_nag[1], atol=1e-5),
      f"max diff {(_f_nag[0] - _f_nag[1]).abs().max().item()}")
check("HF1-1 NEGATIVE control: per-lane pooled embeds diverge the lanes",
      not torch.allclose(_f_nag_bad[0], _f_nag_bad[1], atol=1e-5))
check("pipeline mirror tiles the POSITIVE pooled for both lanes (HF1-1 pin)",
      "torch.cat(\n            [pooled_prompt_embeds, pooled_prompt_embeds], dim=0\n        )"
      in inspect.getsource(nag_flux.NAGFluxPipeline.__call__))
check("Flux NAG'd prediction differs from stock (guidance has effect)",
      not torch.allclose(_f_nag[:1], _f_stock1, atol=1e-4))
with torch.no_grad():
    _f_post = _flux_fwd(_ftiny, _fx, _ftxt_pos, _fpool_pos, _ft)
check("Flux stock behavior restored after remove (N6)",
      torch.equal(_f_post, _f_stock1))


print("\n── Flux.2: variant selection / dormancy / re-sync ─────────────")

from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

torch.manual_seed(11)
_f2tiny = Flux2Transformer2DModel(
    patch_size=1,
    in_channels=8,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=4,
    joint_attention_dim=16,
    timestep_guidance_channels=8,
    mlp_ratio=1.0,
    axes_dims_rope=(2, 2, 2, 2),
    guidance_embeds=True,
).eval()

_f2before = dict(_f2tiny.attn_processors)
_f2origin = nag_flux2.apply_nag_flux2_processors(
    _f2tiny, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=3)
_f2after = _f2tiny.attn_processors
check("Flux2 dual blocks get the dual NAG variant",
      all(isinstance(_f2after[k], nag_flux2.NAGFlux2AttnProcessor)
          for k in _f2after if k.startswith("transformer_blocks.")))
check("Flux2 single blocks get the PARALLEL NAG variant (N7)",
      all(isinstance(_f2after[k], nag_flux2.NAGFlux2ParallelSelfAttnProcessor)
          for k in _f2after if k.startswith("single_transformer_blocks.")))
nag_flux2.remove_nag_flux2_processors(_f2tiny, _f2origin)
check("Flux2 restore returns the ORIGINAL instances (N6)",
      all(_f2tiny.attn_processors[k] is _f2before[k] for k in _f2before))

_f2x = torch.randn(1, 4, 8)
_f2txt_pos = torch.randn(1, 3, 16)
_f2txt_neg = torch.randn(1, 3, 16)
_f2img_ids = torch.zeros(4, 4)
_f2img_ids[:, 1] = torch.arange(4)
_f2txt_ids = torch.zeros(3, 4)
_f2t = torch.tensor([0.5])
_f2g = torch.tensor([4.0])


def _flux2_fwd(model, x, txt, tvec, gvec):
    out = model(
        hidden_states=x,
        timestep=tvec,
        guidance=gvec,
        encoder_hidden_states=txt,
        txt_ids=_f2txt_ids,
        img_ids=_f2img_ids,
        return_dict=False,
    )[0]
    return out[:, : x.shape[1]]


with torch.no_grad():
    _f2_stock1 = _flux2_fwd(_f2tiny, _f2x, _f2txt_pos, _f2t, _f2g)
    _f2_stock2 = _flux2_fwd(
        _f2tiny, torch.cat([_f2x, _f2x]), torch.cat([_f2txt_pos, _f2txt_neg]),
        torch.cat([_f2t, _f2t]), _f2g.expand(2),
    )
    _f2origin = nag_flux2.apply_nag_flux2_processors(
        _f2tiny, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=3)
    try:
        _f2_dorm = _flux2_fwd(
            _f2tiny, torch.cat([_f2x, _f2x]), torch.cat([_f2txt_pos, _f2txt_neg]),
            torch.cat([_f2t, _f2t]), _f2g.expand(2),
        )
    finally:
        nag_flux2.remove_nag_flux2_processors(_f2tiny, _f2origin)
check("Flux2 dormant NAG (scale=1) equals stock batch-2 exactly",
      torch.equal(_f2_dorm, _f2_stock2),
      f"max diff {(_f2_dorm - _f2_stock2).abs().max().item()}")

with torch.no_grad():
    _f2origin = nag_flux2.apply_nag_flux2_processors(
        _f2tiny, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, text_seq_len=3)
    try:
        _f2_nag = _flux2_fwd(
            _f2tiny, torch.cat([_f2x, _f2x]), torch.cat([_f2txt_pos, _f2txt_neg]),
            torch.cat([_f2t, _f2t]), _f2g.expand(2),
        )
    finally:
        nag_flux2.remove_nag_flux2_processors(_f2tiny, _f2origin)
check("Flux2 lane re-sync: velocity predictions identical across lanes",
      torch.allclose(_f2_nag[0], _f2_nag[1], atol=1e-5),
      f"max diff {(_f2_nag[0] - _f2_nag[1]).abs().max().item()}")
check("Flux2 NAG'd prediction differs from stock (guidance has effect)",
      not torch.allclose(_f2_nag[:1], _f2_stock1, atol=1e-4))
with torch.no_grad():
    _f2_post = _flux2_fwd(_f2tiny, _f2x, _f2txt_pos, _f2t, _f2g)
check("Flux2 stock behavior restored after remove (N6)",
      torch.equal(_f2_post, _f2_stock1))


print("\n── Z-Image: hand-swap / refiner exclusion / ragged lanes ─────")

from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

torch.manual_seed(12)
_ztiny = ZImageTransformer2DModel(
    in_channels=4,
    dim=32,
    n_layers=2,
    n_refiner_layers=1,
    n_heads=4,
    n_kv_heads=4,
    cap_feat_dim=16,
    axes_dims=[4, 2, 2],
    axes_lens=[64, 32, 32],
).eval()

_zmain_before = [blk.attention.processor for blk in _ztiny.layers]
_zref_before = [blk.attention.processor for blk in _ztiny.noise_refiner] + \
               [blk.attention.processor for blk in _ztiny.context_refiner]
_zorigin = nag_zimage.apply_nag_zimage_processors(
    _ztiny, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, image_seq_len=32)
check("Z-Image: all JOINT layers get NAG processors",
      all(isinstance(blk.attention.processor,
                     nag_zimage.NAGZSingleStreamAttnProcessor)
          for blk in _ztiny.layers))
check("Z-Image: refiner blocks untouched (N8)",
      [blk.attention.processor for blk in _ztiny.noise_refiner]
      + [blk.attention.processor for blk in _ztiny.context_refiner]
      == _zref_before)
nag_zimage.remove_nag_zimage_processors(_zorigin)
check("Z-Image restore returns the ORIGINAL instances (N6/N9 hand-swap)",
      [blk.attention.processor for blk in _ztiny.layers] == _zmain_before)

# Reviewer F1 (2026-07-09): a mid-apply failure must still leave the
# swapped PREFIX restorable — apply appends each pair into the
# caller-owned list BEFORE its swap. Simulate a transformer whose layers
# iterable dies after the first block.
import types as _types


class _ExplodingLayers:
    def __init__(self, real_first):
        self._first = real_first

    def __iter__(self):
        yield self._first
        raise RuntimeError("mid-apply failure")


_fake_tr = _types.SimpleNamespace(layers=_ExplodingLayers(_ztiny.layers[0]))
_partial: list = []
_raised = False
try:
    nag_zimage.apply_nag_zimage_processors(
        _fake_tr, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25,
        image_seq_len=32, origin_out=_partial)
except RuntimeError:
    _raised = True
check("F1: mid-apply exception propagates", _raised)
check("F1: swapped prefix captured in the caller-owned list",
      len(_partial) == 1
      and isinstance(_ztiny.layers[0].attention.processor,
                     nag_zimage.NAGZSingleStreamAttnProcessor))
nag_zimage.remove_nag_zimage_processors(_partial)
check("F1: prefix restore returns the original processor (no leak)",
      _ztiny.layers[0].attention.processor is _zmain_before[0])

# Batch-2 with RAGGED captions: positive 5 tokens, negative 9 tokens — the
# variable-length path is Z-Image's defining difference.
_zx = torch.randn(4, 1, 4, 4)  # (C=4, F=1, H=4, W=4) -> 4 image tokens, padded to 32
_zcap_pos = torch.randn(5, 16)
_zcap_neg = torch.randn(9, 16)
_zt1 = torch.tensor([0.5])
_zt2 = torch.tensor([0.5, 0.5])


def _z_fwd(model, xs, ts, caps):
    return model(list(xs), ts, list(caps), return_dict=False)[0]


with torch.no_grad():
    _z_stock1 = _z_fwd(_ztiny, [_zx], _zt1, [_zcap_pos])[0]
    _z_stock2 = _z_fwd(_ztiny, [_zx, _zx], _zt2, [_zcap_pos, _zcap_neg])

    _zorigin = nag_zimage.apply_nag_zimage_processors(
        _ztiny, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.25, image_seq_len=32)
    try:
        _z_dorm = _z_fwd(_ztiny, [_zx, _zx], _zt2, [_zcap_pos, _zcap_neg])
    finally:
        nag_zimage.remove_nag_zimage_processors(_zorigin)
check("Z-Image dormant NAG (scale=1) equals stock batch-2 exactly",
      all(torch.equal(a, b) for a, b in zip(_z_dorm, _z_stock2)))

with torch.no_grad():
    _zorigin = nag_zimage.apply_nag_zimage_processors(
        _ztiny, nag_scale=4.0, nag_tau=2.5, nag_alpha=0.25, image_seq_len=32)
    try:
        _z_nag = _z_fwd(_ztiny, [_zx, _zx], _zt2, [_zcap_pos, _zcap_neg])
    finally:
        nag_zimage.remove_nag_zimage_processors(_zorigin)
check("Z-Image lane re-sync holds with RAGGED pos/neg captions",
      torch.allclose(_z_nag[0], _z_nag[1], atol=1e-5),
      f"max diff {(_z_nag[0] - _z_nag[1]).abs().max().item()}")
check("Z-Image NAG'd prediction differs from stock (guidance has effect)",
      not torch.allclose(_z_nag[0], _z_stock1, atol=1e-4))
with torch.no_grad():
    _z_post = _z_fwd(_ztiny, [_zx], _zt1, [_zcap_pos])[0]
check("Z-Image stock behavior restored after remove (N6)",
      torch.equal(_z_post, _z_stock1))


print("\n── ADR-024 pipeline routing guards ────────────────────────────")

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.configuration_utils import FrozenDict

_rec2: dict = {}


def _fake2(self, **kw):
    _rec2.clear()
    _rec2.update(kw)
    return "STOCK"


for _stock_cls, _nag_cls, _extra in (
    (FluxPipeline, nag_flux.NAGFluxPipeline, {}),
    (Flux2Pipeline, nag_flux2.NAGFlux2Pipeline, {}),
    (ZImagePipeline, nag_zimage.NAGZImagePipeline, {"guidance_scale": 0.0}),
):
    _orig2 = _stock_cls.__call__
    _stock_cls.__call__ = _fake2
    try:
        _pp = object.__new__(_nag_cls)
        _r2 = _nag_cls.__call__(_pp, prompt="x", nag_scale=0.0, **_extra)
        check(f"{_nag_cls.__name__}: dormant delegates to stock", _r2 == "STOCK")
        check(f"{_nag_cls.__name__}: dormant leaks no nag_* kwargs",
              not any(k.startswith("nag_") for k in _rec2))
    finally:
        _stock_cls.__call__ = _orig2

# Interplay guards per family.
_orig2 = FluxPipeline.__call__
FluxPipeline.__call__ = _fake2
try:
    _pp = object.__new__(nag_flux.NAGFluxPipeline)
    check("Flux: NAG + true-CFG routes to stock",
          nag_flux.NAGFluxPipeline.__call__(
              _pp, prompt="x", nag_scale=5.0, true_cfg_scale=4.0,
              negative_prompt="bad") == "STOCK")
    check("Flux: NAG + IP-adapter routes to stock",
          nag_flux.NAGFluxPipeline.__call__(
              _pp, prompt="x", nag_scale=5.0,
              ip_adapter_image=object()) == "STOCK")
finally:
    FluxPipeline.__call__ = _orig2

_orig2 = Flux2Pipeline.__call__
Flux2Pipeline.__call__ = _fake2
try:
    _pp = object.__new__(nag_flux2.NAGFlux2Pipeline)
    check("Flux2: NAG + reference image routes to stock (HF2-1)",
          nag_flux2.NAGFlux2Pipeline.__call__(
              _pp, prompt="x", nag_scale=5.0, image=object()) == "STOCK")
finally:
    Flux2Pipeline.__call__ = _orig2

_orig2 = Flux2KleinPipeline.__call__
Flux2KleinPipeline.__call__ = _fake2
try:
    _pp = object.__new__(nag_flux2.NAGFlux2KleinPipeline)
    _pp._internal_dict = FrozenDict({"is_distilled": False})
    check("Klein: NAG + real CFG (non-distilled, cfg>1) routes to stock",
          nag_flux2.NAGFlux2KleinPipeline.__call__(
              _pp, prompt="x", nag_scale=5.0, guidance_scale=4.0) == "STOCK")
    check("Klein: dormant delegates to stock",
          nag_flux2.NAGFlux2KleinPipeline.__call__(
              _pp, prompt="x", nag_scale=0.0) == "STOCK")
finally:
    Flux2KleinPipeline.__call__ = _orig2

_orig2 = ZImagePipeline.__call__
ZImagePipeline.__call__ = _fake2
try:
    _pp = object.__new__(nag_zimage.NAGZImagePipeline)
    check("Z-Image: NAG at cfg>0 routes to stock (CFG owns the negative)",
          nag_zimage.NAGZImagePipeline.__call__(
              _pp, prompt="x", nag_scale=5.0, guidance_scale=5.0) == "STOCK")
finally:
    ZImagePipeline.__call__ = _orig2

# nag_pipe_call dispatch: flux2 module picks Klein vs base by instance class.
_origA, _origB = Flux2Pipeline.__call__, Flux2KleinPipeline.__call__
Flux2Pipeline.__call__ = _fake2
Flux2KleinPipeline.__call__ = _fake2
try:
    _base = object.__new__(Flux2Pipeline)
    _klein = object.__new__(Flux2KleinPipeline)
    _klein._internal_dict = FrozenDict({"is_distilled": True})
    check("flux2 nag_pipe_call routes base instances",
          nag_flux2.nag_pipe_call(_base, prompt="x", nag_scale=0.5) == "STOCK")
    check("flux2 nag_pipe_call routes Klein instances",
          nag_flux2.nag_pipe_call(_klein, prompt="x", nag_scale=0.5) == "STOCK")
finally:
    Flux2Pipeline.__call__, Flux2KleinPipeline.__call__ = _origA, _origB

# Source pins: every ADR-024 mirror restores in an unconditional finally.
for _mod, _restore in (
    (nag_flux.NAGFluxPipeline.__call__, "remove_nag_flux_processors"),
    (nag_zimage.NAGZImagePipeline.__call__, "remove_nag_zimage_processors"),
    (nag_flux2._nag_denoise_flux2, "remove_nag_flux2_processors"),
):
    _s = inspect.getsource(_mod)
    _tail_src = _s.split("finally:", 1)
    check(f"{_restore}: finally-restore present and unconditional",
          len(_tail_src) == 2 and _restore in _tail_src[1]
          and "if nag_applied:" not in _tail_src[1])


print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
