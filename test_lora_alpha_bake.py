#!/usr/bin/env python3
"""_bake_lora_alpha_scales — fold alpha/rank into LoRA weights (noise fix).

Root cause it guards (diffusers 0.39 QwenImage LoRA converter): when a kohya
LoRA's lora_down/up keys are renamed to lora_A/lora_B BEFORE the pipeline load,
the converter's "already in diffusers format" branch copies the weights unscaled
and DISCARDS the .alpha keys — dropping the alpha/rank scale, so an alpha != rank
LoRA applies far too strong (rank 64 / alpha 16 → 4×) → noise. _bake_lora_alpha_scales
folds alpha/rank into the weights up front (matching the converter's get_alpha_scales
numerics) so every downstream path applies the correct magnitude.

Pure CPU, no GPU. Run: ./.venv/bin/python3 test_lora_alpha_bake.py  (expect 0 failures)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import comfyless  # noqa: F401 — installs folder_paths/comfy stubs

import torch

from nodes.eric_qwen_edit_lora import _bake_lora_alpha_scales

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def _module(base, rank, out_dim, in_dim, alpha, *, ab=False):
    """One kohya (or A/B) LoRA module state-dict fragment."""
    torch.manual_seed(abs(hash(base)) % (2**31))
    down = torch.randn(rank, in_dim)   # lora_down / lora_A: (rank, in)
    up = torch.randn(out_dim, rank)    # lora_up   / lora_B: (out, rank)
    dn, un = ("lora_A", "lora_B") if ab else ("lora_down", "lora_up")
    return {
        f"{base}.{dn}.weight": down,
        f"{base}.{un}.weight": up,
        f"{base}.alpha": torch.tensor(float(alpha)),
    }, down, up


def _delta(sd, base):
    dn = sd.get(f"{base}.lora_down.weight", sd.get(f"{base}.lora_A.weight"))
    up = sd.get(f"{base}.lora_up.weight", sd.get(f"{base}.lora_B.weight"))
    return up.float() @ dn.float()


# ── alpha != rank: baked delta must equal original * (alpha/rank) ────────────
BASE = "transformer_blocks.0.attn.to_q"
sd, down0, up0 = _module(BASE, rank=64, out_dim=128, in_dim=96, alpha=16)
orig_delta = up0.float() @ down0.float()
baked = _bake_lora_alpha_scales(dict(sd))
check("alpha!=rank: .alpha key dropped", f"{BASE}.alpha" not in baked)
check("alpha!=rank: down/up keys retained",
      f"{BASE}.lora_down.weight" in baked and f"{BASE}.lora_up.weight" in baked)
baked_delta = _delta(baked, BASE)
expected = orig_delta * (16.0 / 64.0)   # scale = alpha/rank = 0.25
check("alpha!=rank: baked delta == original * alpha/rank (0.25x)",
      torch.allclose(baked_delta, expected, atol=1e-5),
      detail=f"max|Δ|={float((baked_delta-expected).abs().max()):.2e}")
# The unscaled (buggy) magnitude would be 4x too strong — prove we are NOT that.
check("alpha!=rank: baked delta is NOT the unscaled (4x) magnitude",
      not torch.allclose(baked_delta, orig_delta, atol=1e-3))

# ── alpha > rank (scale > 1) ─────────────────────────────────────────────────
B2 = "transformer_blocks.1.attn.to_k"
sd2, d2, u2 = _module(B2, rank=16, out_dim=64, in_dim=64, alpha=32)
baked2 = _bake_lora_alpha_scales(dict(sd2))
check("alpha>rank: baked delta == original * 2.0",
      torch.allclose(_delta(baked2, B2), (u2.float() @ d2.float()) * 2.0, atol=1e-5))

# ── alpha == rank: numeric no-op on the product, alpha still dropped ─────────
B3 = "transformer_blocks.2.attn.to_v"
sd3, d3, u3 = _module(B3, rank=32, out_dim=32, in_dim=32, alpha=32)
baked3 = _bake_lora_alpha_scales(dict(sd3))
check("alpha==rank: delta unchanged (scale 1.0)",
      torch.allclose(_delta(baked3, B3), u3.float() @ d3.float(), atol=1e-5))
check("alpha==rank: .alpha key still dropped", f"{B3}.alpha" not in baked3)

# ── lora_A/lora_B naming + alpha is also baked (ai-toolkit A/B + alpha) ──────
B4 = "transformer_blocks.3.attn.to_q"
sd4, d4, u4 = _module(B4, rank=64, out_dim=48, in_dim=48, alpha=8, ab=True)
baked4 = _bake_lora_alpha_scales(dict(sd4))
check("A/B naming: baked delta == original * alpha/rank (0.125x)",
      torch.allclose(_delta(baked4, B4), (u4.float() @ d4.float()) * (8.0/64.0), atol=1e-5))
check("A/B naming: .alpha dropped", f"{B4}.alpha" not in baked4)

# ── already-diffusers LoRA (no .alpha) passes through UNCHANGED ──────────────
no_alpha = {
    "transformer_blocks.0.attn.to_q.lora_A.weight": torch.randn(16, 32),
    "transformer_blocks.0.attn.to_q.lora_B.weight": torch.randn(32, 16),
}
out = _bake_lora_alpha_scales(dict(no_alpha))
check("no .alpha: dict returned unchanged (same object contents)",
      set(out.keys()) == set(no_alpha.keys())
      and all(torch.equal(out[k], no_alpha[k]) for k in no_alpha))

# ── orphan .alpha (no paired weights) is left in place, not crashed ──────────
orphan = {"some.module.alpha": torch.tensor(4.0)}
out_o = _bake_lora_alpha_scales(dict(orphan))
check("orphan .alpha: no paired weights → left as-is, no crash",
      "some.module.alpha" in out_o)

# ── corrupt-file guards (code review): malformed .alpha must not crash/hang ──
B5 = "transformer_blocks.4.attn.to_q"
sd5, d5, u5 = _module(B5, rank=32, out_dim=32, in_dim=32, alpha=8)
# multi-element alpha → .item() would raise; must be skipped (weights untouched)
sd5[f"{B5}.alpha"] = torch.tensor([8.0, 8.0])
b5 = _bake_lora_alpha_scales(dict(sd5))
check("multi-element .alpha: skipped, no crash, weights unscaled",
      torch.allclose(_delta(b5, B5), u5.float() @ d5.float(), atol=1e-5))

B6 = "transformer_blocks.5.attn.to_q"
sd6, d6, u6 = _module(B6, rank=32, out_dim=32, in_dim=32, alpha=-4)
# negative alpha → diffusers' balancing loop would spin forever; must be skipped
b6 = _bake_lora_alpha_scales(dict(sd6))
check("negative .alpha: skipped (no infinite loop), weights unscaled",
      torch.allclose(_delta(b6, B6), u6.float() @ d6.float(), atol=1e-5))

# ── product invariant: scale_down * scale_up == alpha/rank across many cases ─
inv_ok = True
for rank, alpha in [(64, 16), (16, 64), (8, 8), (128, 4), (32, 96), (4, 1)]:
    bb = f"m.r{rank}.a{alpha}"
    s, dd, uu = _module(bb, rank=rank, out_dim=32, in_dim=32, alpha=alpha)
    bk = _bake_lora_alpha_scales(dict(s))
    got = _delta(bk, bb)
    exp = (uu.float() @ dd.float()) * (alpha / rank)
    if not torch.allclose(got, exp, atol=1e-4):
        inv_ok = False
        break
check("product invariant: baked delta == orig * alpha/rank for all rank/alpha",
      inv_ok)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
