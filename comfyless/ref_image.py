"""Reference-image ingestion for the comfyless edit surface (ADR-035 slice 2).

The security core that every reference image passes through before its pixels
reach a model. One entry point — `load_ref_image_capped` — enforces, in order:
a bounded single read (byte cap), a format allowlist, a decompression-bomb
pixel cap checked on the lazy header *before* full decode, first-frame-only
semantics, and a SHA-256 over the exact bytes read. See ADR-035 decision 6
(6c/6d/6g) and `docs/vision/slice-ref-image-cli.md`.

Deliberately standalone. Two boundaries live elsewhere on purpose:

- **Containment** — which roots a path may be read from (ADR-035 6a / decision
  7 `ref_image_roots`) is the CALLER's job and is intentionally NOT enforced
  here. A typed-at-CLI path carries no containment gate (decision 6b), so this
  helper must not impose `ref_image_roots` on every path. It decides whether the
  *content* is safe to decode, never whether the *path* is allowed.
- **Reference-count cap** (= 8, decision 6f) is a list-level concern enforced at
  the parse (slice 1) and daemon-wire (slice 4) sites, not per image here.
"""
from __future__ import annotations

import hashlib
import io
import os
import stat as _stat
from dataclasses import dataclass

#: Byte cap on a single reference-image read. Mirrors refine's
#: SEED_IMAGE_MAX_BYTES (`comfyless/refine.py`) — same 64 MB ceiling, defined
#: locally so this module carries no dependency on refine.
REF_IMAGE_MAX_BYTES = 64 * 1024 ** 2
#: Decompressed-pixel cap (~67 MP), enforced on the lazy header BEFORE full
#: decode. Mirrors refine's SEED_IMAGE_MAX_PIXELS. The byte cap does not bound
#: pixel count (a small file can declare huge dimensions), and PIL's own
#: MAX_IMAGE_PIXELS is a mutable process-global we do not rely on.
REF_IMAGE_MAX_PIXELS = 64 * 1024 ** 2
#: Decode is pinned to this allowlist (ADR-035 6c) so `Image.open` never
#: dispatches across PIL's full plugin zoo — the EPS plugin shells out to
#: Ghostscript and the rarely-exercised C decoders carry most of Pillow's CVE
#: history. Keyframe authoring needs nothing beyond these three. (A JPEG file
#: carrying MPO markers dispatches through the "JPEG" entry to Pillow's
#: MpoImageFile — still a libjpeg-decoded, first-frame-only read here, not an
#: allowlist escape into a foreign codec.)
REF_IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")


class RefImageError(Exception):
    """A reference image failed ingestion (byte cap, format, pixel cap, decode).

    Messages name the offending path and a reason and NEVER echo file bytes
    (ADR-035 6g) — a co-located sidecar JSON fed as a reference must fail
    without leaking its contents, denying an attacker-*readable*-read oracle."""


@dataclass(frozen=True)
class LoadedRefImage:
    """A successful ingestion: an RGB PIL image plus provenance.

    `sha256` is computed over the exact bytes read (ADR-035 6d), so it describes
    the same bytes that were decoded — no check-then-use window. Decision 7's
    sidecar recording persists this value."""
    image: object       # PIL.Image.Image in mode "RGB"
    sha256: str
    path: str
    size_bytes: int


def load_ref_image_capped(
    path: str,
    max_bytes: int = REF_IMAGE_MAX_BYTES,
    max_pixels: int = REF_IMAGE_MAX_PIXELS,
    formats=REF_IMAGE_FORMATS,
) -> LoadedRefImage:
    """Ingest one reference image, fail-closed at the first violation.

    Order (ADR-035 6c/6d/6g): bounded single read → byte cap → SHA-256 over
    those bytes → format-allowlisted lazy open → pixel cap on the header BEFORE
    full decode → first-frame RGB decode. Raises `RefImageError` naming the
    path and reason on any violation; never echoes file bytes. Containment is
    the caller's concern, not this helper's (6b)."""
    from PIL import Image

    # Non-regular-file guard (ADR-035 slice 4; closes slice-2 LOW-2). Open
    # NON-BLOCKING and reject anything that is not S_ISREG *before* reading a
    # byte. A FIFO / device / socket path would otherwise block open() or read()
    # forever; at the daemon decode site that wedges the single-threaded accept
    # loop — a one-request DoS. O_NONBLOCK makes the open return immediately even
    # for a writer-less FIFO, and the fstat then rejects it; for a regular file
    # O_NONBLOCK has no effect on the read. os.open follows symlinks (no
    # O_NOFOLLOW) deliberately — a symlinked keyframe is legitimate and the guard
    # cares about the final target's TYPE, which fstat reports.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        raise RefImageError(f"cannot read reference image {path!r}: {e}") from e
    # Single bounded read (6d): read at most max_bytes + 1 so an oversize file
    # is detected without ever pulling its full length into memory. This read is
    # authoritative — there is no prior os.stat, hence no check-then-use window
    # between a size check and the read the caps and hash actually describe.
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode):
            raise RefImageError(
                f"reference image {path!r} is not a regular file "
                f"(mode {_stat.S_IFMT(st.st_mode):#o})")
        with os.fdopen(fd, "rb") as fp:
            fd = -1  # fdopen owns the fd now; the with-block closes it
            data = fp.read(max_bytes + 1)
    except OSError as e:
        raise RefImageError(f"cannot read reference image {path!r}: {e}") from e
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > max_bytes:
        raise RefImageError(
            f"reference image {path!r} exceeds byte cap {max_bytes}")

    # Hash the exact bytes read (6d) — the same bytes decoded below.
    sha256 = hashlib.sha256(data).hexdigest()

    # Format-allowlisted lazy open (6c): reads the header, not the pixels, and
    # never dispatches outside the allowlist.
    try:
        img = Image.open(io.BytesIO(data), formats=list(formats))
        w, h = img.size
    except Exception as e:  # noqa: BLE001 — PIL raises a zoo of errors on bad input
        # Surface only the exception CLASS, never str(e) (6g): PIL's PNG plugin
        # interpolates raw file bytes into some messages — e.g. PngImagePlugin
        # `broken PNG file (chunk {repr(cid)})` echoes a 4-byte chunk id read
        # from the file. The class name plus the allowed-formats hint is the
        # actionable part and structurally cannot carry file content.
        raise RefImageError(
            f"cannot open reference image {path!r} "
            f"(allowed formats: {', '.join(formats)}): {type(e).__name__}") from e

    # Pixel cap on the lazy header BEFORE full decode (decompression-bomb guard).
    if w * h > max_pixels:
        raise RefImageError(
            f"reference image {path!r} is {w}x{h} ({w * h} px), "
            f"exceeds pixel cap {max_pixels}")

    # First-frame-only RGB decode (6c): Image.open lands on frame 0 of a
    # multi-frame container; we convert without seeking, so per-frame pixel
    # accounting cannot be gamed by a many-frame animation.
    try:
        rgb = img.convert("RGB")
    except Exception as e:  # noqa: BLE001
        # Class only, never str(e) (6g) — the .load() path can raise a
        # byte-echoing SyntaxError from a malformed trailing chunk id.
        raise RefImageError(
            f"cannot decode reference image {path!r}: {type(e).__name__}") from e

    return LoadedRefImage(
        image=rgb, sha256=sha256, path=path, size_bytes=len(data))
