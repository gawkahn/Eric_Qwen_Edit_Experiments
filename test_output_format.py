"""Tests for comfyless output-format resolution + save (ADR-034 slice 1).

Covers the pure resolver (D2 inference / contradiction, D3 quality mapping +
range rejection) and the format-aware save helper — with the PNG path proven
byte-identical to the prior unconditional-PNG behavior (the slice's primary
regression guard).

Run: ./.venv/bin/python3 test_output_format.py   (0 failures expected)
"""
import json
import os
import sys
import tempfile

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfyless.output_format import (
    OutputFormat,
    resolve_output_format,
    quality_fraction_to_int,
    JPEG_QUALITY_MAX,
    DEFAULT_QUALITY_FRACTION,
)
from comfyless.generate import _save_with_metadata, _resolve_savepath

_failures = []


def check(cond, label):
    if cond:
        print(f"  ok: {label}")
    else:
        print(f"  FAIL: {label}")
        _failures.append(label)


def expect_raises(fn, label, exc=ValueError):
    try:
        fn()
    except exc:
        print(f"  ok: {label}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL: {label} (raised {type(e).__name__}, wanted {exc.__name__})")
        _failures.append(label)
        return
    print(f"  FAIL: {label} (did not raise)")
    _failures.append(label)


# ── D3: quality mapping ──────────────────────────────────────────────────
print("D3 quality mapping")
check(quality_fraction_to_int(0.7) == 70, "0.7 -> 70 (ADR example)")
check(quality_fraction_to_int(1.0) == JPEG_QUALITY_MAX == 95, "1.0 -> 95 ceiling")
check(quality_fraction_to_int(0.5) == 50, "0.5 -> 50")
check(quality_fraction_to_int(0.90) == 90, "0.90 -> 90 (below ceiling, exact)")
check(quality_fraction_to_int(0.95) == 95, "0.95 -> 95 (at ceiling)")
check(quality_fraction_to_int(0.99) == 95, "0.99 -> 95 (clamped to ceiling)")
check(quality_fraction_to_int(0.001) == 1, "0.001 -> clamped up to 1, never 0")
check(quality_fraction_to_int(DEFAULT_QUALITY_FRACTION) == 70, "default fraction -> 70")

print("D3 quality range rejection (unconditional)")
for bad in (1.5, 0.0, -0.1, 2, True, False, "0.7", None):
    expect_raises(lambda b=bad: quality_fraction_to_int(b), f"reject quality {bad!r}")


# ── D2: format resolution / inference ────────────────────────────────────
print("D2 inference (flag absent -> extension infers, else png)")
check(resolve_output_format(None, None, "/tmp/x.png").name == "png", "infer png from .png")
check(resolve_output_format(None, None, "/tmp/x.jpg").name == "jpeg", "infer jpeg from .jpg")
check(resolve_output_format(None, None, "/tmp/x.jpeg").name == "jpeg", "infer jpeg from .jpeg")
check(resolve_output_format(None, None, "/tmp/x.JPG").name == "jpeg", "infer jpeg case-insensitive")
check(resolve_output_format(None, None, "/tmp/x.webp").name == "png", "unknown ext -> png default")
check(resolve_output_format(None, None, None).name == "png", "no path -> png default")
check(resolve_output_format(None, None, "/tmp/noext").name == "png", "no extension -> png default")

print("D2 explicit flag wins")
check(resolve_output_format("jpeg", None, None).name == "jpeg", "explicit jpeg, no path")
check(resolve_output_format("jpg", None, None).name == "jpeg", "jpg alias -> jpeg")
check(resolve_output_format("jpg", None, None).extension == ".jpg", "jpg alias ext is .jpg")
check(resolve_output_format("jpg", None, "/tmp/x.jpg").name == "jpeg", "jpg alias agrees with .jpg path")
expect_raises(lambda: resolve_output_format("jpg", None, "/tmp/x.png"), "jpg alias vs .png path -> error")
check(resolve_output_format("jpeg", None, "/tmp/x.jpg").extension == ".jpg", "jpeg ext is .jpg")
check(resolve_output_format("png", None, "/tmp/x.png").extension == ".png", "png ext is .png")
check(resolve_output_format("jpeg", None, "/tmp/x.jpg").pil_format == "JPEG", "jpeg pil_format")

print("D2 contradiction is an error, not a rewrite")
expect_raises(lambda: resolve_output_format("png", None, "/tmp/x.jpg"), "png flag vs .jpg path")
expect_raises(lambda: resolve_output_format("jpeg", None, "/tmp/x.png"), "jpeg flag vs .png path")
expect_raises(lambda: resolve_output_format("gif", None, None), "unknown format value rejected")

print("embeds_text_chunk property")
check(resolve_output_format("png", None, None).embeds_text_chunk is True, "png embeds tEXt")
check(resolve_output_format("jpeg", None, None).embeds_text_chunk is False, "jpeg no tEXt")


# ── Save helper: PNG byte-identical regression ───────────────────────────
print("PNG save is byte-identical to the prior unconditional-PNG behavior")
_img = Image.new("RGB", (16, 12), (10, 20, 30))
_meta = {"seed": 42, "model": "/models/x", "prompt": "a test"}

with tempfile.TemporaryDirectory() as d:
    # Reference: exactly what the old code did.
    ref = os.path.join(d, "ref.png")
    info = PngInfo()
    info.add_text("comfyless", json.dumps(_meta, default=str))
    _img.save(ref, pnginfo=info)
    ref_bytes = open(ref, "rb").read()

    # output_format=None -> must be byte-identical.
    a = os.path.join(d, "a.png")
    _save_with_metadata(_img, a, _meta, output_format=None)
    check(open(a, "rb").read() == ref_bytes, "output_format=None byte-identical to old PNG")

    # explicit png OutputFormat -> also byte-identical.
    b = os.path.join(d, "b.png")
    _save_with_metadata(_img, b, _meta, output_format=resolve_output_format("png", None, None))
    check(open(b, "rb").read() == ref_bytes, "explicit png byte-identical to old PNG")

    # tEXt chunk round-trips.
    reread = Image.open(a)
    check(reread.text.get("comfyless") == json.dumps(_meta, default=str),
          "comfyless tEXt chunk round-trips")


# ── Save helper: JPEG path ───────────────────────────────────────────────
print("JPEG save: decodable, no tEXt, quality honored, RGBA flattened")
with tempfile.TemporaryDirectory() as d:
    jf = os.path.join(d, "out.jpg")
    _save_with_metadata(_img, jf, _meta, output_format=resolve_output_format("jpeg", None, None))
    reread = Image.open(jf)
    check(reread.format == "JPEG", "jpeg file decodes as JPEG")
    check(not getattr(reread, "text", {}), "jpeg carries no tEXt chunk")

    # Quality honored: low quality is smaller than high quality (content-varying,
    # but a flat 512px gradient reliably compresses smaller at q=10 than q=95).
    grad = Image.new("RGB", (256, 256))
    grad.putdata([((x + y) % 256, x % 256, y % 256) for y in range(256) for x in range(256)])
    lo = os.path.join(d, "lo.jpg")
    hi = os.path.join(d, "hi.jpg")
    _save_with_metadata(grad, lo, _meta, output_format=OutputFormat("jpeg", ".jpg", "JPEG", 10))
    _save_with_metadata(grad, hi, _meta, output_format=OutputFormat("jpeg", ".jpg", "JPEG", 95))
    check(os.path.getsize(lo) < os.path.getsize(hi), "lower quality -> smaller file")

    # RGBA image flattened to RGB (JPEG has no alpha) instead of raising.
    rgba = Image.new("RGBA", (8, 8), (1, 2, 3, 128))
    af = os.path.join(d, "rgba.jpg")
    _save_with_metadata(rgba, af, _meta, output_format=resolve_output_format("jpeg", None, None))
    check(Image.open(af).mode == "RGB", "RGBA flattened to RGB for jpeg")


# ── _resolve_savepath honors the extension ───────────────────────────────
print("_resolve_savepath extension")
with tempfile.TemporaryDirectory() as d:
    tmpl = os.path.join(d, "img")
    p_png = _resolve_savepath(tmpl, "m", 1, 10, 4.0, "default")
    check(p_png.endswith("0001.png"), "default extension is .png (unchanged)")
    p_jpg = _resolve_savepath(tmpl + "j", "m", 1, 10, 4.0, "default", extension=".jpg")
    check(p_jpg.endswith("0001.jpg"), "extension='.jpg' yields .jpg counter file")


# ── Unsupported-path flags are rejected loudly, not ignored (review N-2) ──
# ADR-034 D2 loudness: the --json bridge and cascade dispatch don't handle
# --output-format yet, so they must reject rather than silently emit PNG.
# Source-inspected (the repo's pattern for dispatch guards, cf.
# test_server_robustness delegation guard) — behavioral invocation would pull
# the full torch import graph.
print("unsupported-path flags rejected loudly")
import comfyless.generate as _gen  # noqa: E402
_src = open(_gen.__file__).read()
check("--json rejects --output-format/--quality (OutputFormatNotSupported)",
      "OutputFormatNotSupported" in _src
      and "args.output_format is not None or args.quality is not None" in _src)
check("cascade dispatch rejects --output-format/--quality",
      "not supported for " in _src and "Stable Cascade yet (ADR-034 slice 4)" in _src)


# ── Sidecar provenance is recorded but never replayed (ADR-034) ──────────
print("output-format sidecar provenance is replay-filtered")
from comfyless.generate import _load_sidecar, _SKIP_SIDECAR_KEYS  # noqa: E402
check("output_format + quality in _SKIP_SIDECAR_KEYS",
      {"output_format", "quality"} <= _SKIP_SIDECAR_KEYS)
# The recorded fraction is carried on the resolved OutputFormat.
check("OutputFormat carries the quality fraction",
      resolve_output_format("jpeg", 0.9, None).quality_fraction == 0.9)
check("default fraction carried when --quality omitted",
      resolve_output_format("jpeg", None, None).quality_fraction == DEFAULT_QUALITY_FRACTION)
with tempfile.TemporaryDirectory() as d:
    sc = os.path.join(d, "s.json")
    json.dump({"model": "/m/x", "prompt": "p", "seed": 5,
               "output_format": "jpeg", "quality": 0.9}, open(sc, "w"))
    loaded = _load_sidecar(sc)
    check("replay drops output_format (not a generation input)", "output_format" not in loaded)
    check("replay drops quality (not a generation input)", "quality" not in loaded)
    check("replay keeps real params", loaded.get("seed") == 5)


print()
if _failures:
    print(f"{len(_failures)} FAILURE(S):")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print("All output-format tests passed.")
