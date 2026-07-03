#!/usr/bin/env python3
# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""Loading-prognosis audit for single-file model checkpoints.

Walks one or more directory trees of .safetensors files and reports, per
file, which loader path applies and whether it is expected to work —
WITHOUT loading any weights (header-only, seconds for hundreds of files).

Verdict classes (most→least problematic):
  BNB-NF4    bitsandbytes NF4 — unsupported (ADR-019 dropped NF4)
  NVFP4      nvfp4 block layout (.weight_scale_2) — deferred (ADR-019)
  SVDQ       nunchaku SVDQuant (.qweight/.wscales/.smooth) — unsupported,
             needs nunchaku kernels
  CQ-<fmt>   comfy_quant descriptor with a non-float8 format — unsupported
  CQ-FP8     comfy_quant descriptor, float8 — slice C-d territory
             (annotated C-a-equal vs weight-only by input_scale presence)
  AIO        full-pipeline bundle (multiple component prefixes) —
             pipeline-level from_single_file fallback applies, YMMV
  SCALED     scaled fp8 (C-a/C-b) — handled natively since ADR-019 slice C
  PLAINFP8   plain fp8 cast, no scales — standard path upcasts (always worked)
  HI-PREC    bf16/fp16/fp32 — standard path ([prefix] tag = dominant
             model.diffusion_model.-style prefix, handled by prefix-strip)

Usage:
  python3 audit_single_files.py DIR [DIR ...]
  python3 audit_single_files.py --json DIR ...   # machine-readable, one
                                                 # object per file (for the
                                                 # catalog project)

Grouped human output by default; --json emits {"path", "verdict", "detail",
"family", "n_fp8"} records. Exit code 0 always (it's a report, not a gate).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys

_FP8_DTYPES = ("F8_E4M3", "F8_E5M2")

#: Component prefixes that indicate a full-pipeline (AIO) bundle when two or
#: more distinct groups appear in one file.
_AIO_GROUPS = {
    "vae": ("first_stage_model.", "vae."),
    "te":  ("cond_stage_model.", "text_encoders.", "text_encoder."),
    "dit": ("model.diffusion_model.", "model.model."),
}

#: (family label, key substrings — first hit wins)
_FAMILY_HINTS = [
    ("BFL(flux/flux2/klein/chroma)", ("double_blocks.", "img_mlp")),
    ("wan", (".cross_attn.",)),
    ("SGM-unet(sdxl/sd1)", ("input_blocks.",)),
    ("LLM-TE", ("model.layers.",)),
    ("qwen/diffusers-DiT", ("add_k_proj",)),
    ("diffusers-DiT", ("transformer_blocks.",)),
]


def _header(path: str):
    """Read the safetensors header dict (bounded). Returns (hdr, hdr_len)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if n > 100_000_000:  # safetensors' own cap; refuse bombs
            raise ValueError(f"header too large ({n} bytes)")
        return json.loads(fh.read(n)), n


def _read_tiny_tensor(path: str, hdr_len: int, info: dict, cap: int = 256) -> bytes:
    """Read the raw bytes of a small tensor (comfy_quant descriptors)."""
    lo, hi = info["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(8 + hdr_len + lo)
        # max(0, ...): a hostile/corrupt header with hi < lo would make
        # read(negative) slurp to EOF — gigabytes (reviewer finding 1).
        return fh.read(max(0, min(hi - lo, cap)))


def _guess_family(keys) -> str:
    sample = " ".join(keys[:300])
    for label, pats in _FAMILY_HINTS:
        if any(p in sample for p in pats):
            return label
    return "?"


def audit_file(path: str) -> dict:
    """Header-only prognosis for one .safetensors file."""
    rec = {"path": path, "verdict": "?", "detail": "", "family": "?", "n_fp8": 0}
    try:
        hdr, hlen = _header(path)
    except Exception as e:  # noqa: BLE001 — report, don't crash the walk
        rec.update(verdict="UNREADABLE", detail=f"{type(e).__name__}: {e}")
        return rec

    keys = [k for k in hdr if k != "__metadata__"]
    dt = {k: v.get("dtype") for k, v in hdr.items()
          if isinstance(v, dict) and k != "__metadata__"}
    fp8_keys = [k for k in keys if dt.get(k) in _FP8_DTYPES]
    rec["n_fp8"] = len(fp8_keys)
    rec["family"] = _guess_family(keys)

    # ── Quant-format signatures, most specific first ─────────────────────
    # Mirrors the loader's own bnb detection in _diagnose_slot_mismatch
    # (all five markers — reviewer finding 2: a narrower set here would
    # report HI-PREC for a file the loader will reject). .SCB is bnb Int8,
    # so the verdict is BNB, not NF4-specific.
    if any(".quant_state." in k or ".absmax" in k or ".bitsandbytes" in k
           or k.endswith(".SCB") or k.endswith(".weight_format")
           for k in keys):
        rec.update(verdict="BNB",
                   detail="bitsandbytes NF4/Int8 — unsupported "
                          "(ADR-019 dropped bnb)")
        return rec

    if any(k.endswith((".qweight", ".wscales")) for k in keys):
        rec.update(verdict="SVDQ",
                   detail="nunchaku SVDQuant — needs nunchaku kernels, unsupported")
        return rec

    if any(k.endswith(".weight_scale_2") for k in keys):
        rec.update(verdict="NVFP4",
                   detail="nvfp4 block layout — deferred (ADR-019)")
        return rec

    cq = [k for k in keys if k.endswith(".comfy_quant")]
    if cq:
        try:
            raw = _read_tiny_tensor(path, hlen, hdr[cq[0]])
            fmt = json.loads(raw.decode("utf-8", "replace")).get("format", "?")
        except Exception:  # noqa: BLE001
            fmt = "unparseable"
        if "float8" in str(fmt):
            has_in = any(k.endswith((".input_scale", ".scale_input"))
                         for k in keys)
            mode = "cq-a (both scales)" if has_in \
                else "cq-w (weight-only)"
            rec.update(verdict="CQ-FP8",
                       detail=f"comfy_quant float8, {mode} — native loader "
                              f"since slice C-d")
        else:
            rec.update(verdict=f"CQ-{str(fmt)[:12]}",
                       detail=f"comfy_quant format {fmt!r} — unsupported")
        return rec

    has_scales = any(k.endswith((".weight_scale", ".scale_weight"))
                     for k in keys)
    if fp8_keys and has_scales:
        rec.update(verdict="SCALED",
                   detail="scaled fp8 (C-a/C-b) — native loader since slice C")
        return rec
    if fp8_keys:
        rec.update(verdict="PLAINFP8",
                   detail="fp8 cast, no scales — standard path upcasts")
        return rec

    # ── Structure signatures ──────────────────────────────────────────────
    groups = set()
    for k in keys[:4000]:
        for g, prefixes in _AIO_GROUPS.items():
            if k.startswith(prefixes):
                groups.add(g)
    if len(groups) >= 2:
        rec.update(verdict="AIO",
                   detail=f"full-pipeline bundle ({'+'.join(sorted(groups))}) "
                          f"— pipeline-level fallback, YMMV")
        return rec

    prefixed = any(k.startswith("model.diffusion_model.") for k in keys[:200])
    rec.update(verdict="HI-PREC",
               detail="bf16/fp16 — standard path"
                      + (" [prefix-strip applies]" if prefixed else ""))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Header-only loading prognosis for single-file checkpoints")
    ap.add_argument("roots", nargs="+", help="directory tree(s) to walk")
    ap.add_argument("--json", action="store_true",
                    help="one JSON object per line (for machine consumers)")
    args = ap.parse_args()

    records = []
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"warning: not a directory, skipped: {root}", file=sys.stderr)
            continue
        for dirpath, _, files in os.walk(root):
            for fn in sorted(files):
                if fn.lower().endswith(".safetensors"):
                    records.append(audit_file(os.path.join(dirpath, fn)))

    if args.json:
        for r in records:
            print(json.dumps(r))
        return 0

    by_verdict = collections.defaultdict(list)
    for r in records:
        by_verdict[r["verdict"]].append(r)
    order = ["BNB", "SVDQ", "NVFP4", "AIO", "UNREADABLE"]
    order += sorted(v for v in by_verdict
                    if v.startswith("CQ-") and v != "CQ-FP8")
    order += ["CQ-FP8", "SCALED", "PLAINFP8", "HI-PREC"]
    for v in order:
        if v not in by_verdict:
            continue
        print(f"\n═══ {v} — {by_verdict[v][0]['detail'].split('—')[-1].strip()} "
              f"({len(by_verdict[v])}) ═══")
        for r in by_verdict[v]:
            print(f"  {r['path']}   [{r['family']}]")
    print(f"\n{len(records)} files audited.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
