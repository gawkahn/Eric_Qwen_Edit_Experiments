#!/usr/bin/env python3
"""Test harness for comfyless.core.upscale_vae_decode — ADR-030 2x Wan decode.

The decode moved out of `nodes/eric_qwen_upscale_vae.py` into a comfyless-owned
implementation (ADR-045 slice 3b). The node pack keeps its original, untouched.

The real upscale VAE is a multi-GB decoder, so equivalence is proven against a
DETERMINISTIC STUB standing in for it: everything around the opaque
`vae.decode` call — latent unpacking, the per-channel normalisation, the
pixel_shuffle, the range mapping, the tiling decision, device selection and
the transformer offload/restore — is exercised on CPU with no weights. The
frozen hashes below were captured FROM THE NODE-PACK ORIGINAL, so they encode
its behaviour rather than this module's.

Invariants, each with a negative case:

  1. Unpack math        — packed (B, seq, C*4) regroups to (B, C, 1, H, W).
                          Negative: it must be a pure regroup, so a round trip
                          through the documented repack returns the input
                          EXACTLY, and the frame axis must be present.
  2. Decode equivalence — frozen output hashes over a shape/batch/scale-factor
                          matrix. Negative: hashes are over exact float32
                          bytes, so any last-bit drift in the normalisation
                          order fails (the divide-by-reciprocal detail).
  3. Tiling decision    — tiled iff a latent side exceeds 128, with the exact
                          tile geometry. Negative: at or below the threshold
                          `enable_tiling` must NOT be called and `use_tiling`
                          is set False instead.
  4. Range mapping      — output is float32 in [0, 1], channels-last.
                          Negative: values outside [-1, 1] from the decoder
                          must be CLAMPED, not wrapped or rescaled.
  5. Offload/restore    — the transformer returns to its original device on
                          every exit path. Negative: it must be restored even
                          when the decode RAISES, and a decode failure must
                          still propagate rather than be swallowed. The
                          offload half only runs on CUDA: the guard is
                          `device.type != "cpu"`, so a CPU-only run proves the
                          restore but never enters the offload branch at all.
                          Those assertions skip rather than pass vacuously.
"""

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from comfyless.core.upscale_vae_decode import (
    decode_latents_with_upscale_vae,
    decode_latents_with_upscale_vae_safe,
    unpack_qwen_latents,
)

passed = 0
failed = 0
Z_DIM = 16


skipped = 0


def skip(name, why):
    global skipped
    skipped += 1
    print(f"  SKIP  {name}  ({why})")


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


# ── Deterministic stand-ins for the real models ───────────────────────
class StubUpscaleVAE(nn.Module):
    """12-channel decoder at 8x the latent size, with no weights to load.

    Deliberately emits values outside [-1, 1] so the clamp is load bearing.
    """

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.w = nn.Parameter(torch.arange(4, dtype=dtype).reshape(1, 4, 1, 1) / 3.0)
        self.calls = []
        self.use_tiling = None

    def enable_tiling(self, **kw):
        self.calls.append(("enable_tiling", kw))

    def decode(self, x, return_dict=False):
        self.calls.append(("decode", tuple(x.shape)))
        b, c, _f, h, w = x.shape
        up = F.interpolate(x.reshape(b, c, h, w), scale_factor=8, mode="nearest")
        stacked = torch.stack([up[:, i % c] * (1.0 + 0.1 * i) for i in range(12)], dim=1)
        return (stacked.mul(1.4).unsqueeze(2),)


class _StubCfg:
    z_dim = Z_DIM
    latents_mean = [0.01 * i - 0.08 for i in range(Z_DIM)]
    latents_std = [0.5 + 0.03 * i for i in range(Z_DIM)]


class StubPipeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.config = _StubCfg()


class StubPipe:
    def __init__(self):
        self.transformer = nn.Linear(4, 4)
        self.vae = StubPipeVAE()


CPU = torch.device("cpu")


def packed(h, w, vsf=8, b=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    h_lat = 2 * (h // (vsf * 2))
    w_lat = 2 * (w // (vsf * 2))
    return torch.randn(b, (h_lat // 2) * (w_lat // 2), Z_DIM * 4, generator=g)


def digest(t):
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


# ── 1. Unpack math ────────────────────────────────────────────────────
print("\n=== 1. latent unpacking ===")

lat = packed(256, 128)
spatial = unpack_qwen_latents(lat, 256, 128, 8)
check("unpack yields (B, C, 1, H_lat, W_lat)",
      tuple(spatial.shape) == (1, Z_DIM, 1, 32, 16), f"got {tuple(spatial.shape)}")
check("the singleton frame axis is present (Wan is a video VAE)",
      spatial.shape[2] == 1)

# NEGATIVE: a pure regroup must round-trip exactly. Repack by inverting the
# documented permutation; anything that interpolated or reordered would fail.
b, c, _f, h_lat, w_lat = spatial.shape
back = (spatial.reshape(b, c, h_lat // 2, 2, w_lat // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, (h_lat // 2) * (w_lat // 2), c * 4))
check("NEGATIVE: unpack round-trips bit-exactly (no interpolation)",
      torch.equal(back, lat))

shapes_ok = True
for hh, ww in ((64, 64), (1024, 1024), (96, 160), (2048, 1024)):
    s = unpack_qwen_latents(packed(hh, ww), hh, ww, 8)
    shapes_ok &= tuple(s.shape) == (1, Z_DIM, 1, 2 * (hh // 16), 2 * (ww // 16))
check("unpack shape law holds across sizes", shapes_ok)


# ── 2. Decode equivalence against the node-pack original ──────────────
print("\n=== 2. frozen equivalence with the node-pack original ===")

# key -> (sha256 of float32 bytes, shape, tiled?)
GOLDEN = {
    (64, 64, 8, 1, 0):     ("ecb36dfb76c3f07a97a3b9c20adbc0381691cb168722b9e88edc7679b36f3440", (1, 128, 128, 3), False),
    (256, 256, 8, 1, 0):   ("fc85dd6a029e81dc1eec0514b3c54d7cbb57343a8ca5d45c3538e12f81139c3c", (1, 512, 512, 3), False),
    (512, 512, 8, 2, 1):   ("671994eaebcfa60ff5216f6dcd58dbb3ebadb3a9f71cc4fe7ac9385c0566d4ce", (2, 1024, 1024, 3), False),
    (1024, 1024, 8, 1, 0): ("456363a01a2aba2e29ab3d74bf19614598bac7d2ccf4a47b49642f85c163b2ef", (1, 2048, 2048, 3), False),
    (2048, 1024, 8, 1, 3): ("467dd3bf963fd553850724f28eb4f779215039a32d2119ac7c190370fa967dcf", (1, 4096, 2048, 3), True),
    (96, 160, 8, 1, 2):    ("e138944bd302813faf8d3d0d54bcddc4f6503fd59432820c5c376f7fd877d496", (1, 192, 320, 3), False),
    (512, 512, 4, 1, 0):   ("456363a01a2aba2e29ab3d74bf19614598bac7d2ccf4a47b49642f85c163b2ef", (1, 2048, 2048, 3), False),
    (512, 512, 16, 1, 0):  ("fc85dd6a029e81dc1eec0514b3c54d7cbb57343a8ca5d45c3538e12f81139c3c", (1, 512, 512, 3), False),
}

for (h, w, vsf, bsz, seed), (want_sha, want_shape, want_tiled) in GOLDEN.items():
    vae = StubUpscaleVAE()
    out = decode_latents_with_upscale_vae(
        packed(h, w, vsf, bsz, seed), vae, StubPipeVAE(), h, w, vsf, device=CPU)
    tag = f"{h}x{w} vsf={vsf} b={bsz}"
    check(f"{tag}: output matches the original byte for byte",
          digest(out) == want_sha, f"{digest(out)[:16]} != {want_sha[:16]}")
    check(f"{tag}: shape {want_shape}", tuple(out.shape) == want_shape,
          f"got {tuple(out.shape)}")
    check(f"{tag}: tiling {'on' if want_tiled else 'off'}",
          any(c[0] == "enable_tiling" for c in vae.calls) == want_tiled)

# dtype of the upscale VAE must not leak into the result
for dt in (torch.float16, torch.bfloat16, torch.float32):
    out = decode_latents_with_upscale_vae(
        packed(64, 64), StubUpscaleVAE(dt), StubPipeVAE(), 64, 64, 8, device=CPU)
    check(f"output is float32 regardless of vae dtype {dt}", out.dtype == torch.float32)


# ── 3. Tiling decision ────────────────────────────────────────────────
print("\n=== 3. tiling threshold and geometry ===")

# latent side = 2 * (px // 16); 1024px -> 128 (at threshold), 1040px -> 130
vae = StubUpscaleVAE()
decode_latents_with_upscale_vae(packed(2048, 2048), vae, StubPipeVAE(), 2048, 2048, 8, device=CPU)
tiling = [kw for name, kw in vae.calls if name == "enable_tiling"]
check("tiled decode enables tiling exactly once", len(tiling) == 1)
check("tile geometry matches the original",
      tiling and tiling[0] == {"tile_sample_min_height": 256, "tile_sample_min_width": 256,
                               "tile_sample_stride_height": 192, "tile_sample_stride_width": 192},
      f"got {tiling}")

# NEGATIVE: at the threshold (latent side == 128) tiling must stay OFF, and
# the flag is set False rather than left alone.
vae = StubUpscaleVAE()
decode_latents_with_upscale_vae(packed(1024, 1024), vae, StubPipeVAE(), 1024, 1024, 8, device=CPU)
check("NEGATIVE: latent side == 128 does not tile",
      not any(c[0] == "enable_tiling" for c in vae.calls))
check("NEGATIVE: untiled path sets use_tiling False", vae.use_tiling is False)

# one side over threshold is enough
vae = StubUpscaleVAE()
decode_latents_with_upscale_vae(packed(2048, 512), vae, StubPipeVAE(), 2048, 512, 8, device=CPU)
check("a single oversized side triggers tiling",
      any(c[0] == "enable_tiling" for c in vae.calls))


# ── 4. Range mapping ──────────────────────────────────────────────────
print("\n=== 4. output range and layout ===")

out = decode_latents_with_upscale_vae(packed(128, 128), StubUpscaleVAE(), StubPipeVAE(), 128, 128, 8, device=CPU)
check("output is channels-last with 3 channels", out.shape[-1] == 3)
check("output lies in [0, 1]",
      float(out.min()) >= 0.0 and float(out.max()) <= 1.0,
      f"range [{float(out.min())}, {float(out.max())}]")
check("output is 2x the requested pixel size",
      tuple(out.shape[1:3]) == (256, 256))


class OutOfRangeVAE(StubUpscaleVAE):
    def decode(self, x, return_dict=False):
        b, c, _f, h, w = x.shape
        big = F.interpolate(x.reshape(b, c, h, w), scale_factor=8, mode="nearest")
        return (torch.stack([big[:, i % c] for i in range(12)], dim=1).mul(50.0).unsqueeze(2),)


# NEGATIVE: a decoder that overshoots [-1, 1] must be clamped. If the range
# map wrapped or rescaled instead, saturated pixels would not sit exactly on
# the endpoints.
out = decode_latents_with_upscale_vae(packed(64, 64), OutOfRangeVAE(), StubPipeVAE(), 64, 64, 8, device=CPU)
check("NEGATIVE: out-of-range decoder output is clamped, not wrapped",
      float(out.min()) == 0.0 and float(out.max()) == 1.0,
      f"range [{float(out.min())}, {float(out.max())}]")
# A CLAMP pins the overshoot exactly onto 0.0 and 1.0 while leaving the
# in-range middle untouched. A rescale would compress everything inward and
# hit neither endpoint exactly; a binarisation would leave nothing in between.
saturated = ((out == 0.0) | (out == 1.0)).float().mean().item()
interior = ((out > 0.0) & (out < 1.0)).any().item()
check("NEGATIVE: clamp pins the overshoot onto the exact endpoints",
      saturated > 0.5, f"only {saturated:.1%} saturated")
check("NEGATIVE: clamp is not a binarisation — mid values survive", interior)


# ── 5. Transformer offload / restore ──────────────────────────────────
print("\n=== 5. transformer offload and guaranteed restore ===")

pipe = StubPipe()
before = next(pipe.transformer.parameters()).device
out = decode_latents_with_upscale_vae_safe(packed(128, 128), StubUpscaleVAE(), pipe, 128, 128, 8, device=CPU)
check("safe wrapper returns the decoded image",
      tuple(out.shape) == (1, 256, 256, 3))
check("transformer device unchanged after a clean decode",
      next(pipe.transformer.parameters()).device == before)


class BoomVAE(StubUpscaleVAE):
    def decode(self, x, return_dict=False):
        raise RuntimeError("simulated OOM")


# NEGATIVE: the restore must survive a raising decode, AND the error must
# still propagate — a finally block that swallowed it would hide an OOM.
pipe = StubPipe()
before = next(pipe.transformer.parameters()).device
raised = None
try:
    decode_latents_with_upscale_vae_safe(packed(128, 128), BoomVAE(), pipe, 128, 128, 8, device=CPU)
except Exception as exc:
    raised = type(exc)
check("NEGATIVE: a failing decode still raises", raised is RuntimeError, f"raised {raised}")
check("NEGATIVE: transformer restored even when the decode raises",
      next(pipe.transformer.parameters()).device == before)


class NoTransformer:
    def __init__(self):
        self.vae = StubPipeVAE()
        self.transformer = None


# NEGATIVE: a pipeline with no readable transformer must still decode rather
# than crash in the offload bookkeeping.
out = decode_latents_with_upscale_vae_safe(
    packed(64, 64), StubUpscaleVAE(), NoTransformer(), 64, 64, 8, device=CPU)
check("NEGATIVE: missing transformer degrades to a plain decode",
      tuple(out.shape) == (1, 128, 128, 3))

# The upscale VAE is handed back to CPU on every exit path, including failure.
vae = StubUpscaleVAE()
decode_latents_with_upscale_vae(packed(64, 64), vae, StubPipeVAE(), 64, 64, 8, device=CPU)
check("upscale VAE ends on CPU after a clean decode",
      next(vae.parameters()).device.type == "cpu")
vae = BoomVAE()
try:
    decode_latents_with_upscale_vae(packed(64, 64), vae, StubPipeVAE(), 64, 64, 8, device=CPU)
except RuntimeError:
    pass
check("upscale VAE ends on CPU even when the decode raises",
      next(vae.parameters()).device.type == "cpu")


# ── 5b. The offload branch itself (CUDA only) ─────────────────────────
print("\n=== 5b. offload branch on a real device ===")

if not torch.cuda.is_available():
    skip("transformer is offloaded during the decode", "no CUDA device")
    skip("transformer is restored after a CUDA decode", "no CUDA device")
    skip("transformer is restored after a raising CUDA decode", "no CUDA device")
else:
    seen = {}

    class WatchingVAE(StubUpscaleVAE):
        """Records where the transformer sits at the moment of decode."""

        def __init__(self, pipe, tag, boom=False):
            super().__init__()
            self.pipe, self.tag, self.boom = pipe, tag, boom

        def decode(self, x, return_dict=False):
            seen[self.tag] = next(self.pipe.transformer.parameters()).device.type
            if self.boom:
                raise RuntimeError("simulated OOM")
            return super().decode(x, return_dict)

    dev = torch.device("cuda:0")
    pipe = StubPipe()
    pipe.transformer = pipe.transformer.to(dev)
    pipe.vae = pipe.vae.to(dev)
    before = next(pipe.transformer.parameters()).device
    decode_latents_with_upscale_vae_safe(
        packed(128, 128), WatchingVAE(pipe, "ok").to(dev), pipe, 128, 128, 8)
    check("transformer is offloaded during the decode", seen.get("ok") == "cpu",
          f"transformer was on {seen.get('ok')} during decode")
    check("transformer is restored after a CUDA decode",
          next(pipe.transformer.parameters()).device == before)

    pipe = StubPipe()
    pipe.transformer = pipe.transformer.to(dev)
    pipe.vae = pipe.vae.to(dev)
    before = next(pipe.transformer.parameters()).device
    try:
        decode_latents_with_upscale_vae_safe(
            packed(128, 128), WatchingVAE(pipe, "boom", boom=True).to(dev),
            pipe, 128, 128, 8)
    except RuntimeError:
        pass
    check("transformer is restored after a raising CUDA decode",
          next(pipe.transformer.parameters()).device == before)


# ──────────────────────────────────────────────────────────────────────
print("\n" + "─" * 50)
print(f"  {passed} passed, {failed} failed, {skipped} skipped")
print("─" * 50)
sys.exit(1 if failed else 0)
