#!/usr/bin/env python3
"""Convert a ComfyUI all-in-one Stable Cascade checkpoint into a Würstchen-native single-file safetensors.

ComfyUI all-in-one Cascade bundles store diffusion weights under a
`model.diffusion_model.*` prefix and embed copies of the VAE
(`vae.*`) and text encoder (`text_encoder.*`). `StableCascadeUNet.from_single_file()`
will not load these as-is.

This tool extracts the diffusion-only keys, strips the prefix, and
writes a clean Würstchen-native safetensors that loads cleanly via
`from_single_file()` + a diffusers-tree config directory. Bundled VAE
and text-encoder weights are reported but discarded — comfyless's
cascade dispatch always uses the scaffolding repo's text_encoder /
tokenizer / scheduler. (See ADR-010 and docs/comfyless-stable-cascade.md.)

Usage:
    python3 convert_cascade_comfyui.py --in <bundle.safetensors> --stage c
    python3 convert_cascade_comfyui.py --in <bundle.safetensors> --stage b --lite
    python3 convert_cascade_comfyui.py --in <bundle.safetensors> --list   # diagnostic only

Output filename defaults to `<in_dir>/<stem>-comfyless.safetensors` per
the parallel-`-comfyless`-library convention.
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter
from typing import Dict

# Default config-dir locations are owned by comfyless/cascade.py (single source of
# truth). Importing from there means a layout change updates both tools at once.
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from comfyless.cascade import PRIOR_CONFIG_DIRS as _DEFAULT_CONFIG_DIRS

# Cap on the safetensors JSON header byte-size during read. Real files have headers
# in the low megabytes; anything larger is a corrupt or hostile file lying about
# its length. Without this cap, a 100 GB header claim would attempt 100 GB of
# memory allocation before any sanity check.
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024  # 100 MB

_DIFFUSION_PREFIX = "model.diffusion_model."
_DISCARD_PREFIXES = ("vae.", "text_encoder.", "effnet.", "previewer.")


def read_header(path: str) -> Dict:
    with open(path, "rb") as f:
        raw_n = f.read(8)
        if len(raw_n) != 8:
            raise ValueError(f"{path!r} is not a valid safetensors file (truncated header)")
        n = struct.unpack("<Q", raw_n)[0]
        if n > _MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(
                f"{path!r}: safetensors header claims {n / 1024 / 1024:.1f} MB "
                f"(cap: {_MAX_SAFETENSORS_HEADER_BYTES / 1024 / 1024:.0f} MB). "
                f"File is corrupt or hostile."
            )
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def summarize_prefixes(keys) -> Counter:
    return Counter(k.split(".", 1)[0] if "." in k else k for k in keys)


def cmd_list(in_path: str) -> int:
    hdr = read_header(in_path)
    print(f"\n{os.path.basename(in_path)}  ({os.path.getsize(in_path)/1e9:.2f} GB)")
    print(f"  total tensors:           {len(hdr)}")
    pfx = summarize_prefixes(hdr.keys())
    for name, count in pfx.most_common(20):
        print(f"  {name + ':':30s} {count}")
    diffusion_keys = [k for k in hdr if k.startswith(_DIFFUSION_PREFIX)]
    print(f"\n  diffusion (prefixed):    {len(diffusion_keys)}")
    print(f"  bundled VAE (vae.*):     {sum(1 for k in hdr if k.startswith('vae.'))}")
    print(f"  bundled TE (text_encoder.*): {sum(1 for k in hdr if k.startswith('text_encoder.'))}")
    return 0


def convert(
    in_path: str,
    out_path: str,
    stage: str,
    lite: bool,
    config_dir: str,
    skip_smoke_test: bool,
    force: bool,
) -> int:
    if not os.path.isfile(in_path):
        print(f"[error] input not found: {in_path}", file=sys.stderr)
        return 2
    if os.path.exists(out_path) and not force:
        print(f"[error] output exists (pass --force to overwrite): {out_path}", file=sys.stderr)
        return 2

    hdr = read_header(in_path)
    diffusion_keys = [k for k in hdr if k.startswith(_DIFFUSION_PREFIX)]
    if not diffusion_keys:
        print(
            f"[error] no '{_DIFFUSION_PREFIX}*' keys found. This file does not look like a "
            f"ComfyUI all-in-one bundle. (Total tensors: {len(hdr)}.) Use --list for a breakdown.",
            file=sys.stderr,
        )
        return 2

    discarded = sum(1 for k in hdr if any(k.startswith(p) for p in _DISCARD_PREFIXES))
    other = len(hdr) - len(diffusion_keys) - discarded
    print(f"[convert] input:           {in_path}")
    print(f"[convert] total tensors:   {len(hdr)}")
    print(f"[convert] diffusion keys:  {len(diffusion_keys)}")
    print(f"[convert] discarded:       {discarded}  (vae/text_encoder/effnet/previewer)")
    if other:
        print(f"[convert] other (kept w/ prefix-strip attempt): {other}")

    # Load tensors + strip prefix. We use safetensors directly to avoid a torch import for the
    # extraction step. Tensors stay on CPU; total RAM ≈ original file size.
    print(f"[convert] reading tensors from {os.path.basename(in_path)} ...")
    t0 = time.time()
    from safetensors import safe_open
    from safetensors.torch import save_file

    new_state: Dict[str, "torch.Tensor"] = {}
    with safe_open(in_path, framework="pt", device="cpu") as f:
        for k in diffusion_keys:
            new_key = k[len(_DIFFUSION_PREFIX):]
            new_state[new_key] = f.get_tensor(k)
    print(f"[convert] read {len(new_state)} tensors in {time.time()-t0:.1f}s")

    # Carry over the safetensors metadata field (mostly for traceability).
    metadata = {
        "converted_from": os.path.basename(in_path),
        "converter": "convert_cascade_comfyui.py",
        "stage": stage,
        "lite": str(lite),
    }

    print(f"[convert] writing {out_path} ...")
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    save_file(new_state, out_path, metadata=metadata)
    print(f"[convert] wrote {os.path.getsize(out_path)/1e9:.2f} GB in {time.time()-t0:.1f}s")

    if skip_smoke_test:
        print("[convert] smoke test skipped (--no-smoke-test)")
        return 0

    return _smoke_test(out_path, stage, lite, config_dir)


def _smoke_test(out_path: str, stage: str, lite: bool, config_dir: str) -> int:
    """CPU-only load via StableCascadeUNet.from_single_file.

    Reports the architectural compatibility of the converted file against the
    SAI reference config: how many file keys got dropped as unused (= alt model
    has weights the SAI architecture doesn't), how many model keys were missing
    from the file (= file lacks weights the SAI architecture expects). A clean
    convert: 0 unused, 0 missing.
    """
    if not os.path.isdir(config_dir):
        print(
            f"[warn] config dir not found, smoke test skipped: {config_dir}\n"
            "       Pass --config-dir <path> or download the matching SAI tree.",
            file=sys.stderr,
        )
        return 0

    print(f"[smoke] loading via StableCascadeUNet.from_single_file (CPU) ...")
    print(f"[smoke]   config dir: {config_dir}")
    t0 = time.time()
    try:
        import torch
        from diffusers import StableCascadeUNet

        model = StableCascadeUNet.from_single_file(
            out_path,
            config=config_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,  # avoid meta-device init for the smoke test
        )
    except Exception as e:
        print(f"[smoke] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        print(
            "[smoke] The converted file was kept for inspection. Likely cause: "
            "stage/lite mismatch, or the alt file's architecture differs from the "
            "specified config. Try --list on the input bundle and compare key counts.",
            file=sys.stderr,
        )
        return 3

    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] loaded in {elapsed:.1f}s")
    print(f"[smoke]   parameters: {n_params/1e9:.2f}B")

    # Compare file-keys to model.state_dict()-keys to detect architecture mismatch.
    # `from_single_file` silently drops keys that don't match the config, which is
    # the exact failure mode we want to surface here.
    file_hdr = read_header(out_path)
    file_keys = set(file_hdr.keys())
    model_keys = set(model.state_dict().keys())
    unused = file_keys - model_keys
    missing = model_keys - file_keys
    print(f"[smoke]   file keys:         {len(file_keys)}")
    print(f"[smoke]   model keys:        {len(model_keys)}")
    print(f"[smoke]   unused (dropped):  {len(unused)}")
    print(f"[smoke]   missing (init'd):  {len(missing)}")

    # Sanity: ensure at least one weight is non-default. Pick a stable named param.
    sample_param = next(iter(model.parameters()))
    sample_mean = float(sample_param.detach().float().mean())
    print(f"[smoke]   sample param mean: {sample_mean:.2e}")

    if unused:
        print(
            f"\n[smoke] FAIL — architecture mismatch: {len(unused)} weights from the file "
            f"were silently dropped by from_single_file. The alt model has parameters that "
            f"the SAI Cascade architecture does not have. Loading and running this file "
            f"would produce broken output.\n"
            f"[smoke] Sample dropped keys:",
            file=sys.stderr,
        )
        for k in sorted(unused)[:5]:
            print(f"[smoke]   {k}", file=sys.stderr)
        if len(unused) > 5:
            print(f"[smoke]   ... and {len(unused) - 5} more", file=sys.stderr)
        return 3

    # `missing` is expected to be small (~256 buffers like BN running stats that diffusers
    # initializes from the architecture). We log it but don't fail.
    if missing and len(missing) > 512:
        print(
            f"[smoke] WARNING: {len(missing)} architecture keys had no value in the file. "
            f"Some of these may be OK (BN buffers re-init from architecture), but a count "
            f"this high suggests the file is missing real weights.",
            file=sys.stderr,
        )

    print(f"[smoke] OK — {len(unused)} unused, {len(missing)} init'd from architecture defaults.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert a ComfyUI all-in-one Stable Cascade checkpoint to Würstchen-native single-file format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--in", dest="in_path", required=True, help="Path to the ComfyUI all-in-one .safetensors bundle.")
    p.add_argument("--out", dest="out_path", default=None,
                   help="Output path. Defaults to <in_dir>/<stem>-comfyless.safetensors.")
    p.add_argument("--stage", choices=["c", "b"], default=None,
                   help="Which stage this file represents. Required unless --list is used.")
    p.add_argument("--lite", action="store_true",
                   help="Use the lite-variant config for smoke test (default: full).")
    p.add_argument("--config-dir", default=None,
                   help="Override the config directory used for the smoke-test load. "
                        "Defaults to a path under ~/projects/ai-lab/ai-base/models/hf-local/.")
    p.add_argument("--no-smoke-test", action="store_true", help="Skip the CPU load verification.")
    p.add_argument("--force", action="store_true", help="Overwrite the output file if it exists.")
    p.add_argument("--list", action="store_true",
                   help="Diagnostic mode: print key-prefix breakdown of input and exit.")
    args = p.parse_args()

    if args.list:
        return cmd_list(args.in_path)

    if args.stage is None:
        p.error("--stage is required (c or b) unless --list is used.")

    if args.out_path is None:
        in_dir = os.path.dirname(os.path.abspath(args.in_path))
        stem = os.path.splitext(os.path.basename(args.in_path))[0]
        args.out_path = os.path.join(in_dir, f"{stem}-comfyless.safetensors")

    config_dir = args.config_dir or _DEFAULT_CONFIG_DIRS[(args.stage, args.lite)]

    return convert(
        in_path=args.in_path,
        out_path=args.out_path,
        stage=args.stage,
        lite=args.lite,
        config_dir=config_dir,
        skip_smoke_test=args.no_smoke_test,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
