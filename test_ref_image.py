#!/usr/bin/env python3
"""Tests for comfyless/ref_image.py — ADR-035 slice 2 ingestion helper.

CPU-only, no GPU, no model weights, no network. The negative cases are the
point: an arbitrary user file must not be able to (a) blow past the byte cap,
(b) blow past the pixel cap via forged huge dimensions BEFORE decode, (c)
dispatch outside the PNG/JPEG/WEBP allowlist into PIL's plugin zoo, or (d) leak
its own bytes through an error message (6g). Plus the positive contract:
single-read SHA-256 determinism, first-frame-only decode, RGB normalization.

See docs/vision/slice-ref-image-cli.md and ADR-035 decision 6.
"""

import hashlib
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image

from comfyless.ref_image import (
    REF_IMAGE_FORMATS,
    REF_IMAGE_MAX_BYTES,
    REF_IMAGE_MAX_PIXELS,
    LoadedRefImage,
    RefImageError,
    load_ref_image_capped,
)

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


def expect_error(name, fn):
    """Assert fn() raises RefImageError; return the exception for further asserts."""
    global passed, failed
    try:
        fn()
    except RefImageError as e:
        passed += 1
        print(f"  PASS  {name}")
        return e
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"  FAIL  {name}  raised {type(e).__name__}, want RefImageError: {e}")
        return None
    else:
        failed += 1
        print(f"  FAIL  {name}  did not raise")
        return None


# ── Fixtures ─────────────────────────────────────────────────────────────────
def _write(tmp, name, data: bytes) -> str:
    p = os.path.join(tmp, name)
    with open(p, "wb") as f:
        f.write(data)
    return p


def _save_image(tmp, name, img, **save_kw) -> str:
    p = os.path.join(tmp, name)
    img.save(p, **save_kw)
    return p


def _forge_png(w: int, h: int) -> bytes:
    """A minimal, structurally-valid PNG (signature + IHDR + IEND) declaring
    w×h in its IHDR. Tiny on disk regardless of dimensions — the decompression-
    bomb shape: small file, huge declared pixel count. Image.open reports .size
    from IHDR without decoding any pixel data."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit truecolor RGB
    ihdr = (struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data
            + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF))
    iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    return sig + ihdr + iend


tmp = tempfile.mkdtemp(prefix="ref_image_test_")

# ── Positive contract: valid PNG/JPEG/WEBP round-trips ───────────────────────
png_path = _save_image(tmp, "ok.png", Image.new("RGB", (16, 24), (10, 20, 30)), format="PNG")
res = load_ref_image_capped(png_path)
check("PNG: returns LoadedRefImage", isinstance(res, LoadedRefImage))
check("PNG: image mode is RGB", res.image.mode == "RGB")
check("PNG: image size preserved", res.image.size == (16, 24), detail=str(res.image.size))
check("PNG: path preserved", res.path == png_path)
with open(png_path, "rb") as f:
    png_bytes = f.read()
check("PNG: size_bytes == file length", res.size_bytes == len(png_bytes),
      detail=f"{res.size_bytes} vs {len(png_bytes)}")
check("PNG: sha256 over exact file bytes (6d)",
      res.sha256 == hashlib.sha256(png_bytes).hexdigest())

jpg_path = _save_image(tmp, "ok.jpg", Image.new("RGB", (8, 8), (200, 100, 50)), format="JPEG")
check("JPEG: decodes to RGB", load_ref_image_capped(jpg_path).image.mode == "RGB")

webp_path = _save_image(tmp, "ok.webp", Image.new("RGB", (8, 8), (0, 128, 255)),
                        format="WEBP", lossless=True)
check("WEBP: decodes to RGB", load_ref_image_capped(webp_path).image.mode == "RGB")

# Palette / grayscale sources normalize to RGB.
pal_path = _save_image(tmp, "pal.png", Image.new("P", (8, 8)), format="PNG")
check("palette PNG normalized to RGB", load_ref_image_capped(pal_path).image.mode == "RGB")
gray_path = _save_image(tmp, "gray.png", Image.new("L", (8, 8), 128), format="PNG")
check("grayscale PNG normalized to RGB", load_ref_image_capped(gray_path).image.mode == "RGB")

# ── SHA-256 single-read determinism ──────────────────────────────────────────
check("sha256 deterministic across two loads",
      load_ref_image_capped(png_path).sha256 == load_ref_image_capped(png_path).sha256)
png2 = _save_image(tmp, "ok2.png", Image.new("RGB", (16, 24), (10, 20, 31)), format="PNG")  # +1 in blue
check("sha256 differs when one byte differs",
      load_ref_image_capped(png_path).sha256 != load_ref_image_capped(png2).sha256)

# ── Byte cap (before any decode) ─────────────────────────────────────────────
SECRET = b"TOPSECRETMARKER_should_never_appear_in_an_error_string"
over_path = _write(tmp, "over.bin", SECRET + b"\x00" * 100)
e = expect_error("oversize file rejected by byte cap",
                 lambda: load_ref_image_capped(over_path, max_bytes=10))
check("byte-cap error names the path",
      e is not None and over_path in str(e))
check("byte-cap error does not echo file bytes (6g)",
      e is not None and "TOPSECRETMARKER" not in str(e), detail=str(e))

# Boundary: exactly max_bytes passes the cap gate (fails later as a non-image,
# but NOT with a byte-cap message).
exact = _write(tmp, "exact.bin", b"A" * 10)
e = expect_error("exactly-max_bytes file passes byte gate (fails as non-image)",
                 lambda: load_ref_image_capped(exact, max_bytes=10))
check("boundary: not rejected by byte cap", e is not None and "byte cap" not in str(e),
      detail=str(e))
# One byte over the cap IS a byte-cap rejection.
overby1 = _write(tmp, "over1.bin", b"A" * 11)
e = expect_error("max_bytes+1 rejected by byte cap",
                 lambda: load_ref_image_capped(overby1, max_bytes=10))
check("boundary: rejected with byte-cap reason", e is not None and "byte cap" in str(e),
      detail=str(e))

# ── Pixel cap (decompression bomb) enforced BEFORE full decode ───────────────
# 100 Mpx: above OUR 67 Mpx cap, below Pillow's own ~179 Mpx DecompressionBomb
# limit — so Image.open succeeds reading the header and OUR pixel-cap branch is
# what rejects it (a larger forge would trip Pillow's coarser backstop first,
# also fail-closed but via the open-branch).
bomb = _write(tmp, "bomb.png", _forge_png(10000, 10000))  # 100 Mpx, ~45 bytes on disk
check("forged bomb PNG really is tiny on disk", os.path.getsize(bomb) < 200)
e = expect_error("decompression bomb rejected by our pixel cap",
                 lambda: load_ref_image_capped(bomb))
check("pixel-cap error names dimensions and path",
      e is not None and "10000x10000" in str(e) and bomb in str(e), detail=str(e))
# A more extreme forge is still rejected (Pillow's own backstop, rewrapped).
huge = _write(tmp, "huge.png", _forge_png(30000, 30000))  # 900 Mpx
expect_error("extreme forge still rejected (Pillow backstop, rewrapped)",
             lambda: load_ref_image_capped(huge))
# Same code path via a lowered cap on a real (small) image, proving the gate
# fires on the lazy header before convert().
e = expect_error("pixel cap fires on small real image with lowered cap",
                 lambda: load_ref_image_capped(png_path, max_pixels=100))  # 16*24=384 > 100
check("lowered-cap error reports px count", e is not None and "384 px" in str(e), detail=str(e))

# ── Format allowlist ─────────────────────────────────────────────────────────
gif_path = _save_image(tmp, "x.gif", Image.new("RGB", (8, 8), (1, 2, 3)), format="GIF")
e = expect_error("GIF rejected by format allowlist", lambda: load_ref_image_capped(gif_path))
check("format error names path and allowlist",
      e is not None and gif_path in str(e) and "PNG" in str(e), detail=str(e))
bmp_path = _save_image(tmp, "x.bmp", Image.new("RGB", (8, 8), (1, 2, 3)), format="BMP")
expect_error("BMP rejected by format allowlist", lambda: load_ref_image_capped(bmp_path))
tiff_path = _save_image(tmp, "x.tiff", Image.new("RGB", (8, 8), (1, 2, 3)), format="TIFF")
expect_error("TIFF rejected by format allowlist", lambda: load_ref_image_capped(tiff_path))

# ── Sidecar JSON fed as a reference: fails without leaking its contents (6g) ──
JSON_SECRET = "api_key_hunter2_and_a_prompt_you_should_not_see"
json_path = _write(tmp, "meta.json", ('{"secret": "%s"}' % JSON_SECRET).encode())
e = expect_error("sidecar JSON rejected (not a decodable image)",
                 lambda: load_ref_image_capped(json_path))
check("JSON error names the path", e is not None and json_path in str(e))
check("JSON error does not leak file contents (6g)",
      e is not None and JSON_SECRET not in str(e), detail=str(e))

# ── 6g: convert-branch (broken-body) error leaks neither bytes nor PIL text ──
# A valid PNG header + corrupt IDAT reaches the .convert() rewrap. We embed an
# ASCII marker in the corrupt body and assert the RefImageError carries only the
# path + exception class — never str(e) (which, on some malformed-chunk paths,
# interpolates raw file bytes via Pillow's repr(cid), 6g).
def _broken_body_png(marker: bytes) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    def _chunk(t, d):
        return (struct.pack(">I", len(d)) + t + d
                + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF))
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", b"\xff\xff\xff" + marker)  # not a valid zlib stream
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend

CONVERT_MARKER = b"CONVERTSECRETMARKER"
broken = _write(tmp, "broken.png", _broken_body_png(CONVERT_MARKER))
e = expect_error("broken-body PNG rejected at decode", lambda: load_ref_image_capped(broken))
check("convert-branch error names path + class, no str(e)",
      e is not None and broken in str(e) and "OSError" in str(e), detail=str(e))
check("convert-branch error does not leak file bytes (6g)",
      e is not None and "CONVERTSECRETMARKER" not in str(e), detail=str(e))

# ── Missing / unreadable path ────────────────────────────────────────────────
missing = os.path.join(tmp, "does_not_exist.png")
e = expect_error("missing file raises RefImageError", lambda: load_ref_image_capped(missing))
check("missing-file error names the path", e is not None and missing in str(e))

# ── First-frame-only decode (multi-frame WEBP) ───────────────────────────────
frame0 = Image.new("RGB", (8, 8), (255, 0, 0))   # red
frame1 = Image.new("RGB", (8, 8), (0, 0, 255))   # blue
anim = os.path.join(tmp, "anim.webp")
frame0.save(anim, format="WEBP", save_all=True, append_images=[frame1],
            duration=100, lossless=True)
anim_res = load_ref_image_capped(anim)
px = anim_res.image.getpixel((0, 0))
check("multi-frame WEBP decodes FRAME 0 only (red, not blue)",
      px[0] > 200 and px[2] < 60, detail=f"pixel={px}")
check("multi-frame WEBP size is a single frame", anim_res.image.size == (8, 8),
      detail=str(anim_res.image.size))

# ── Module constants sanity ──────────────────────────────────────────────────
check("byte cap mirrors seed 64 MB", REF_IMAGE_MAX_BYTES == 64 * 1024 ** 2)
check("pixel cap mirrors seed ~67 MP", REF_IMAGE_MAX_PIXELS == 64 * 1024 ** 2)
check("format allowlist is exactly PNG/JPEG/WEBP",
      tuple(REF_IMAGE_FORMATS) == ("PNG", "JPEG", "WEBP"))

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
