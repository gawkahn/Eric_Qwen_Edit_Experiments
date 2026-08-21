#!/usr/bin/env python3
"""Test harness for comfyless.core bislerp — latent-space spherical resize.

`bislerp` replaced a call into ComfyUI's `comfy.utils.bislerp` (ADR-045
slice 2). That import was a hard dependency the comfyless shim never
stubbed, so `upscale_flux_latents` raised AttributeError outside ComfyUI,
and it would have pulled GPL-3.0 code into this package. The replacement
is ours, its behaviour matched to ComfyUI's by black-box probing.

Invariants, each with a negative case:

  1. Interpolation math  — direction is SLERPed (not lerp-then-normalise)
                           and magnitude is LERPed. Negative: the t=0.1
                           great-circle point must NOT equal the
                           normalised linear blend.
  2. Sample coordinates  — align_corners=False centre alignment with edge
                           clamping. Negative: align_corners=True spacing
                           is rejected.
  3. Degenerate pairs    — parallel keeps v0 verbatim; antiparallel falls
                           back to a plain lerp; a zero vector contributes
                           no direction. Negative: none of these may emit
                           NaN/Inf.
  4. Pass order          — width before height. Negative: the transposed
                           order must give a DIFFERENT result (proving the
                           passes are genuinely non-commutative, so the
                           order is a real choice and not decoration).
  5. Shape / dtype       — output is (B, C, height, width) in the input
                           dtype. Negative: half precision must not leak
                           out as float32.
  6. ComfyUI agreement   — where a real ComfyUI is importable, agree to
                           ~2e-06 on tie-free coordinates (every realistic
                           upscale ratio), and DIVERGE at integer ties,
                           where ComfyUI emits a one-row spike and we do
                           not. Skipped when ComfyUI is absent.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from comfyless.core.eric_diffusion_manual_loop import (
    bislerp,
    _bislerp_blend,
    _bislerp_taps,
)

passed = 0
failed = 0
skipped = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def check_close(name, got, want, atol=1e-6):
    got_t = torch.as_tensor(got, dtype=torch.float64)
    want_t = torch.as_tensor(want, dtype=torch.float64)
    d = (got_t - want_t).abs().max().item()
    check(name, d <= atol, f"max|diff|={d:.3e} > {atol:.1e}")


def skip(name, why):
    global skipped
    skipped += 1
    print(f"  SKIP  {name}  ({why})")


def rowvec(vs):
    """list of C-vectors along W -> (1, C, 1, W)"""
    t = torch.tensor(vs, dtype=torch.float32)
    return t.t().unsqueeze(0).unsqueeze(2)


# ── 1. Interpolation math ─────────────────────────────────────────────
print("\n=== 1. slerp direction + lerp magnitude ===")

# Orthogonal unit vectors: the blend must trace the 90-degree great circle,
# so the angle is exactly t * 90 degrees and the norm stays 1.
out = bislerp(rowvec([[1.0, 0.0], [0.0, 1.0]]), 5, 1)[0, :, 0, :].t()
angles = [math.degrees(math.atan2(v[1].item(), v[0].item())) for v in out]
check_close("orthogonal pair traces the great circle", angles, [0.0, 9.0, 45.0, 81.0, 90.0], atol=1e-4)
check_close("unit inputs keep unit norm", [v.norm().item() for v in out], [1.0] * 5, atol=1e-6)

# NEGATIVE: lerp-then-normalise gives a different point. Index 1 of a
# 2->5 resize samples t=0.1, so the linear blend to beat is [0.9, 0.1].
q = bislerp(rowvec([[1.0, 0.0], [0.0, 1.0]]), 5, 1)[0, :, 0, 1]
lin = torch.tensor([0.9, 0.1])
lin = lin / lin.norm()
check("NEGATIVE: not lerp-then-normalise", (q - lin).abs().max().item() > 1e-3,
      f"slerp={q.tolist()} vs normalised-lerp={lin.tolist()}")

# Magnitude lerps linearly while the direction rotates.
out = bislerp(rowvec([[1.0, 0.0], [0.0, 2.0]]), 5, 1)[0, :, 0, :].t()
check_close("magnitude lerps linearly", [v.norm().item() for v in out],
            [1.0, 1.1, 1.5, 1.9, 2.0], atol=1e-5)


# ── 2. Sample coordinates ─────────────────────────────────────────────
print("\n=== 2. align_corners=False coordinates ===")

idx_lo, idx_hi, frac = _bislerp_taps(2, 5, torch.device("cpu"))
coords = (idx_lo.to(torch.float64) + frac.to(torch.float64)).tolist()
check_close("2->5 coords are centre-aligned and edge-clamped", coords,
            [0.0, 0.1, 0.5, 0.9, 1.0], atol=1e-6)
check("taps stay in range", int(idx_lo.min()) >= 0 and int(idx_hi.max()) <= 1)

# NEGATIVE: align_corners=True would space 2->5 as 0,.25,.5,.75,1.
check("NEGATIVE: not align_corners=True spacing",
      abs(coords[1] - 0.25) > 1e-3, f"coords={coords}")

idx_lo, _, frac = _bislerp_taps(1, 4, torch.device("cpu"))
check("degenerate src=1 collapses to index 0",
      int(idx_lo.max()) == 0 and float(frac.abs().max()) == 0.0)


# ── 3. Degenerate vector pairs ────────────────────────────────────────
print("\n=== 3. parallel / antiparallel / zero ===")

# Parallel (dot == 1): v0 is kept verbatim, magnitude included.
out = bislerp(rowvec([[1.0, 0.0], [3.0, 0.0]]), 5, 1)[0, 0, 0, :]
check_close("parallel keeps v0 verbatim", out.tolist(), [1.0, 1.0, 1.0, 1.0, 3.0])

# Antiparallel (dot == -1): no meaningful rotation, so plain vector lerp.
out = bislerp(rowvec([[1.0, 0.0], [-2.0, 0.0]]), 5, 1)[0, 0, 0, :]
check_close("antiparallel falls back to vector lerp", out.tolist(),
            [1.0, 0.7, -0.5, -1.7, -2.0], atol=1e-5)

# Zero vector has no direction: it contributes nothing to the blend, but
# the magnitude still lerps, giving |out| = t * sin(t * 90deg).
out = bislerp(rowvec([[0.0, 0.0], [0.0, 1.0]]), 5, 1)[0, :, 0, :].t()
want = [t * math.sin(math.radians(t * 90.0)) for t in (0.0, 0.1, 0.5, 0.9, 1.0)]
check_close("zero vector contributes no direction", [v.norm().item() for v in out], want, atol=1e-6)

# NEGATIVE: none of the degenerate paths may emit NaN/Inf. Note this
# disciplines the PARALLEL mask specifically -- drop it and acos(1) == 0
# gives an exact 0/0. It does NOT discipline the antiparallel mask, whose
# float32 sin(acos(-1)) is -8.7e-08, so dropping that mask yields huge but
# finite values; the antiparallel branch is pinned by the value check above.
finite = True
for pair in ([[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]],
             [[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [5.0, -3.0]]):
    finite &= bool(torch.isfinite(bislerp(rowvec(pair), 7, 1)).all())
check("NEGATIVE: degenerate pairs stay finite", finite)

# A vector can be nonzero yet have an exactly-zero float32 norm: components
# around 1e-23 square to less than the smallest subnormal. Then v/m is
# +/-inf, not nan -- and nan_to_num's DEFAULTS map those to +/-dtype-max,
# which multiply out to +inf and -inf in the dot product. The resulting nan
# dot satisfies neither mask, so it reaches a live output row.
tiny = bislerp(rowvec([[1e-23, -1e-23], [1.0, 1.0]]), 5, 1)
check("NEGATIVE: subnormal-norm vector does not leak NaN",
      bool(torch.isfinite(tiny).all()), f"got {tiny.flatten().tolist()}")
# Having no length, it must follow the same law as the zero vector above:
# magnitude lerps from 0 to |v1| while the surviving direction contributes
# sin(t * 90deg). Here |v1| = sqrt(2), not 1.
check_close("zero-length vector follows the zero-vector law",
            [v.norm().item() for v in tiny[0, :, 0, :].t()],
            [math.sqrt(2.0) * t * math.sin(math.radians(t * 90.0))
             for t in (0.0, 0.1, 0.5, 0.9, 1.0)],
            atol=1e-6)

t = torch.full((3, 1), 0.5)
blend = _bislerp_blend(torch.zeros(3, 4), torch.zeros(3, 4), t)
check("all-zero blend is exactly zero", bool((blend == 0).all()))


# ── 4. Pass order ─────────────────────────────────────────────────────
print("\n=== 4. width pass precedes height pass ===")

torch.manual_seed(0)
x = torch.randn(1, 4, 3, 5)
wh = bislerp(x, 9, 7)
# Doing it in two explicit steps, width first, must reproduce the one-shot
# result exactly.
check_close("bislerp(x,W,H) == height(width(x))",
            wh, bislerp(bislerp(x, 9, 3), 9, 7), atol=1e-6)
# NEGATIVE: the transposed order genuinely differs, so the order is a
# real decision rather than an arbitrary one.
hw = bislerp(bislerp(x, 5, 7), 9, 7)
check("NEGATIVE: height-first gives a different result",
      (wh - hw).abs().max().item() > 1e-4,
      f"max|diff|={(wh - hw).abs().max().item():.3e}")


# ── 5. Shape / dtype ──────────────────────────────────────────────────
print("\n=== 5. shape and dtype ===")

check("output shape is (B, C, height, width)",
      tuple(bislerp(torch.randn(2, 16, 8, 6), 11, 13).shape) == (2, 16, 13, 11))
check_close("same-size resize is the identity", bislerp(x, 5, 3), x, atol=1e-6)

dtypes_ok = True
for dt in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
    dtypes_ok &= bislerp(torch.randn(1, 8, 4, 4).to(dt), 8, 8).dtype == dt
check("dtype is preserved", dtypes_ok)
# NEGATIVE: half precision must not silently widen to float32.
check("NEGATIVE: fp16 does not leak out as fp32",
      bislerp(torch.randn(1, 8, 4, 4).half(), 8, 8).dtype != torch.float32)
check("half precision stays finite",
      bool(torch.isfinite(bislerp(torch.randn(1, 8, 5, 5).half(), 9, 9)).all()))


# ── 6. Agreement with a real ComfyUI, where available ─────────────────
print("\n=== 6. ComfyUI cross-check (optional) ===")


def _find_comfy_bislerp():
    for root in ("/home/gawkahn/projects/ai-lab/ai-stack-data/comfy-dev/run/ComfyUI",
                 "/home/gawkahn/projects/ai-lab/ai-stack-data/comfy1/run/ComfyUI",
                 "/home/gawkahn/projects/ai-lab/ai-stack-data/comfy0/run/ComfyUI"):
        if not Path(root, "comfy", "utils.py").is_file():
            continue
        saved_path = list(sys.path)
        # comfyless installs no-op `comfy`/`folder_paths` stubs at import
        # time, and its `comfy.utils` stub has no bislerp. Evict them or
        # the real module is never reached.
        saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules)
                      if k == "comfy" or k.startswith("comfy.")}
        sys.path.insert(0, root)
        try:
            from comfy.utils import bislerp as ref  # noqa: PLC0415
            return ref
        except Exception:
            sys.path[:] = saved_path
            sys.modules.update(saved_mods)
    return None


def _has_tie(src, dst):
    """True when a sample coordinate lands on an exact integer, where
    ComfyUI's two tap arrays can disagree."""
    for i in range(dst):
        c = (i + 0.5) * src / dst - 0.5
        if 0 < c < src - 1 and abs(c - round(c)) < 1e-6:
            return True
    return False


REF = _find_comfy_bislerp()
if REF is None:
    skip("ComfyUI agreement", "no importable ComfyUI on this host")
    skip("ComfyUI tie divergence", "no importable ComfyUI on this host")
else:
    # Realistic latent upscales: all tie-free, so we must match to float32
    # rounding. These are the ratios upscale_flux_latents actually sees.
    torch.manual_seed(2)
    worst = 0.0
    checked = 0
    for C in (16, 64):
        for (H, W, dh, dw) in [(64, 64, 128, 128), (80, 48, 160, 96),
                               (96, 96, 144, 144), (52, 76, 104, 152),
                               (32, 32, 48, 48)]:
            if _has_tie(H, dh) or _has_tie(W, dw):
                continue
            src = torch.randn(1, C, H, W)
            worst = max(worst, (REF(src, dw, dh) - bislerp(src, dw, dh)).abs().max().item())
            checked += 1
    check(f"matches ComfyUI on {checked} tie-free latent upscales",
          worst <= 2e-6, f"worst max|diff|={worst:.3e}")

    # DELIBERATE DIVERGENCE. Rows [a, a, b] resized 3 -> 19: coordinate 9
    # is exactly 1.5*(3/19)*19 == 1.0, a tie. ComfyUI's tap arrays disagree
    # by 2 there and it emits row `b` as a spike between two `a` rows; we
    # interpolate smoothly and emit `a`.
    torch.manual_seed(5)
    a, b = torch.randn(4), torch.randn(4)
    stack = torch.stack([a, a, b]).t().reshape(1, 4, 3, 1)
    ref_row = REF(stack, 1, 19)[0, :, 9, 0]
    our_row = bislerp(stack, 1, 19)[0, :, 9, 0]
    check("ComfyUI spikes at an integer-tie coordinate",
          (ref_row - b).abs().max().item() < 1e-5,
          "expected upstream to emit row b at the tie")
    check("we interpolate smoothly through the tie instead",
          (our_row - a).abs().max().item() < 1e-5,
          f"row9={our_row.tolist()} expected a={a.tolist()}")


# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed, {skipped} skipped")
print("─" * 50)
sys.exit(1 if failed else 0)
