"""Centralized output-format resolution for comfyless still images (ADR-034).

Single source of truth for the on-disk file extension and the PIL ``save()``
arguments per supported format, so no call site composes an extension or a
save() kwarg by hand (ADR-034 D1). Slice 1 wires the CLI in-process path; the
daemon, MCP, cascade, and refine paths route through the same helper in later
slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# name -> (on-disk extension, PIL format string)
_FORMATS = {
    "png":  (".png", "PNG"),
    "jpeg": (".jpg", "JPEG"),
}

# Accepted --output-format spellings that normalize onto a canonical name.
_FORMAT_ALIASES = {"jpg": "jpeg"}

# Extensions that infer a format when --output-format is absent (ADR-034 D2).
_EXT_TO_NAME = {
    ".png":  "png",
    ".jpg":  "jpeg",
    ".jpeg": "jpeg",
}

# PIL's useful JPEG quality ceiling; above this, file size grows with little
# visible gain (ADR-034 D3), which is why the 1.0 fraction maps here, not 100.
JPEG_QUALITY_MAX = 95

# Effective --quality when the flag is omitted (ADR-034 D3 default 0.7).
DEFAULT_QUALITY_FRACTION = 0.7


@dataclass(frozen=True)
class OutputFormat:
    """Resolved output format: extension + PIL save parameters."""

    name: str          # "png" | "jpeg"
    extension: str     # ".png" | ".jpg"
    pil_format: str    # "PNG" | "JPEG"
    quality: int       # PIL 1..95 (meaningful for jpeg; unused for png)
    # The effective 0.0-1.0 fraction (the --quality knob; default when omitted).
    # Recorded as jpeg sidecar provenance — the value a user would re-set, which
    # the output file cannot reveal. Defaulted so positional construction in
    # tests/callers predating this field keeps working.
    quality_fraction: float = DEFAULT_QUALITY_FRACTION

    @property
    def embeds_text_chunk(self) -> bool:
        """Only PNG carries the comfyless ``tEXt`` metadata chunk (ADR-034 §2).

        JPEG provenance lives in the JSON sidecar alone.
        """
        return self.name == "png"


def quality_fraction_to_int(q: float) -> int:
    """Map a ``0.0 < q <= 1.0`` fraction to a PIL JPEG quality (ADR-034 D3).

    The fraction maps linearly onto 1..100 (``0.7 -> 70``) and is then clamped
    to PIL's useful ceiling of 95 (``1.0 -> 95``): above 95, file size grows
    with little visible gain, which is why 1.0 maps to 95 rather than 100.

    Validation is unconditional — a malformed value is rejected even for PNG,
    where the value is otherwise ignored — so the caller can surface a single
    directed CLI error. ``bool`` is rejected explicitly (it is an ``int``
    subclass and ``True``/``False`` are never a meaningful quality).
    """
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise ValueError(f"--quality must be a number in (0.0, 1.0]; got {q!r}")
    if not (0.0 < q <= 1.0):
        raise ValueError(f"--quality must be in (0.0, 1.0]; got {q}")
    return max(1, min(JPEG_QUALITY_MAX, round(q * 100)))


def resolve_output_format(
    format_flag: Optional[str],
    quality: Optional[float],
    output_path: Optional[str],
) -> OutputFormat:
    """Resolve the effective output format (ADR-034 D2/D3).

    - An explicit ``format_flag`` wins.
    - When it is absent, the ``output_path`` extension infers the format
      (``.jpg`` / ``.jpeg`` -> jpeg, ``.png`` -> png); otherwise png.
    - An explicit flag that contradicts the extension is an error, not a
      silent rewrite (ADR-034 D2) — the daemon/MCP paths generate filenames
      the caller does not control, where a rewrite would be invisible. Pass
      ``output_path=None`` for savepath templates and the default sentinel,
      whose extension must not be treated as caller-authored.

    ``quality`` is validated whenever supplied (even for png). It is applied
    only for jpeg; supplying it with png is a no-op the caller may warn about.
    """
    inferred = None
    if output_path:
        low = output_path.lower()
        for ext, nm in _EXT_TO_NAME.items():
            if low.endswith(ext):
                inferred = nm
                break

    if format_flag is None:
        name = inferred or "png"
    else:
        name = _FORMAT_ALIASES.get(format_flag, format_flag)
        if name not in _FORMATS:
            accepted = sorted(set(_FORMATS) | set(_FORMAT_ALIASES))
            raise ValueError(
                f"--output-format must be one of {accepted}; got {format_flag!r}"
            )
        if inferred is not None and inferred != name:
            raise ValueError(
                f"--output-format {name} contradicts the output path extension "
                f"of '{output_path}' (which infers {inferred}). "
                f"Remove one so they agree."
            )

    # Validate quality unconditionally when supplied; apply only for jpeg.
    frac = DEFAULT_QUALITY_FRACTION if quality is None else quality
    qi = quality_fraction_to_int(frac)

    ext, pil_fmt = _FORMATS[name]
    return OutputFormat(name=name, extension=ext, pil_format=pil_fmt,
                        quality=qi, quality_fraction=float(frac))
