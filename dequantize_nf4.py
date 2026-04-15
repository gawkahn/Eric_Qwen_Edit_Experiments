#!/usr/bin/env python3
"""Dequantize a bitsandbytes NF4 Flux transformer checkpoint.

Loads a CivitAI-style single-file NF4 quantized Flux transformer via
bitsandbytes, dequantizes every 4-bit parameter back to full precision,
and saves the result as a single .safetensors file with diffusers-format
keys.  The output can then be dropped into the Component Loader's
`transformer_path` field — our existing single-file loading fallback
(direct state-dict load) will pick it up and match 100% of keys.

Usage
-----
    python3 dequantize_nf4.py \\
        --input /path/to/flux_nf4_checkpoint.safetensors \\
        --base-model /path/to/FLUX.1-dev \\
        --output /path/to/flux_my_checkpoint_dequantized.safetensors \\
        [--dtype bf16]

The --base-model directory must contain a `transformer/config.json` (the
standard diffusers layout).  This provides the architecture config for
the dequantized model.  A copy of FLUX.1-dev works for any standard Flux
transformer fine-tune.

Memory
------
Loads the model twice briefly (quantized + unquantized).  For full Flux
that's roughly:
  NF4 storage : ~3 GB
  FP16/BF16   : ~12 GB
  Peak RAM    : ~18 GB during conversion

Disk size
---------
Output is ~4x the NF4 input size because you're trading compression for
framework compatibility.  Keep the NF4 original around if you care about
disk space.

Author: Eric Hiss (GitHub: EricRollei)
Utility script — not part of the ComfyUI node set.
"""

import argparse
import os
import shutil
import sys
import tempfile

import torch
from safetensors.torch import save_file


def _detect_params4bit_class():
    """Find bitsandbytes' Params4bit class across version differences.

    bnb's class hierarchy has shifted over time.  This tries several
    import paths and returns the first one that works.
    """
    candidates = [
        ("bitsandbytes.nn", "Params4bit"),
        ("bitsandbytes.nn.modules", "Params4bit"),
    ]
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    raise RuntimeError(
        "Could not locate bitsandbytes Params4bit class.  "
        "Install a recent version:  pip install -U bitsandbytes"
    )


def _dequantize_param(param, target_dtype):
    """Dequantize a single Params4bit parameter back to full precision."""
    import bitsandbytes as bnb

    dequantized = bnb.functional.dequantize_4bit(
        param.data, param.quant_state,
    )
    return dequantized.to(dtype=target_dtype)


def dequantize_nf4(input_path, base_model_path, output_path, target_dtype=torch.bfloat16):
    """Main dequantization routine.

    Args:
        input_path      : path to the NF4 .safetensors checkpoint
        base_model_path : path to a diffusers Flux model dir (for config)
        output_path     : destination .safetensors file
        target_dtype    : torch.bfloat16 or torch.float16
    """
    # ── Sanity checks ───────────────────────────────────────────────
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    transformer_config_dir = os.path.join(base_model_path, "transformer")
    if not os.path.isdir(transformer_config_dir):
        raise FileNotFoundError(
            f"Base model does not have transformer/ subdir: {transformer_config_dir}\n"
            f"Point --base-model at a full Flux diffusers directory."
        )

    config_json = os.path.join(transformer_config_dir, "config.json")
    if not os.path.isfile(config_json):
        raise FileNotFoundError(
            f"Base transformer config.json not found: {config_json}"
        )

    # ── Import heavy deps ───────────────────────────────────────────
    try:
        from diffusers import FluxTransformer2DModel
        from diffusers import BitsAndBytesConfig
    except ImportError as e:
        raise RuntimeError(
            f"diffusers import failed: {e}\n"
            f"This script must be run with a Python that has diffusers "
            f"and bitsandbytes installed (use the ComfyUI venv)."
        )

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"bitsandbytes import failed: {e}\n"
            f"Install with:  pip install bitsandbytes"
        )

    Params4bit = _detect_params4bit_class()

    # ── Step 1: build a temporary diffusers directory around the NF4 file ──
    # diffusers' `from_pretrained` expects a directory with config.json
    # alongside the weights.  We create a temp dir that satisfies this.
    print(f"[dequantize] Creating temporary diffusers directory …")
    with tempfile.TemporaryDirectory(prefix="dequant_nf4_") as temp_dir:
        shutil.copy(config_json, os.path.join(temp_dir, "config.json"))
        temp_weights = os.path.join(temp_dir, "diffusion_pytorch_model.safetensors")
        # Symlink is faster and avoids copying a multi-GB file, but fall
        # back to copy if symlinks aren't supported on this filesystem.
        try:
            os.symlink(os.path.abspath(input_path), temp_weights)
        except (OSError, NotImplementedError):
            shutil.copy(input_path, temp_weights)

        # ── Step 2: load the NF4 checkpoint via bitsandbytes ────────
        print(f"[dequantize] Loading NF4 checkpoint (this may take a minute) …")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        quantized = FluxTransformer2DModel.from_pretrained(
            temp_dir,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            local_files_only=True,
        )

    # ── Step 3: build a fresh non-quantized model from the same config ──
    print(f"[dequantize] Building non-quantized sibling model …")
    unquantized = FluxTransformer2DModel.from_config(quantized.config)
    unquantized = unquantized.to(target_dtype)

    # ── Step 4: iterate parameters, dequantize as needed, copy over ──
    print(f"[dequantize] Dequantizing parameters …")
    unquant_params = dict(unquantized.named_parameters())
    quantized_count = 0
    passthrough_count = 0
    mismatched = []

    for name, q_param in quantized.named_parameters():
        target = unquant_params.get(name)
        if target is None:
            mismatched.append(name)
            continue

        if isinstance(q_param, Params4bit):
            dequant = _dequantize_param(q_param, target_dtype)
            quantized_count += 1
        else:
            dequant = q_param.data.to(target_dtype)
            passthrough_count += 1

        if dequant.shape != target.shape:
            # Some quantized weights are stored in transposed/flattened form;
            # try to reshape before giving up.
            try:
                dequant = dequant.reshape(target.shape)
            except RuntimeError:
                raise RuntimeError(
                    f"Shape mismatch for '{name}': "
                    f"dequantized {tuple(dequant.shape)} vs "
                    f"target {tuple(target.shape)}"
                )

        target.data.copy_(dequant)

    # Also copy buffers (non-parameter tensors like running_mean/var)
    for name, buf in quantized.named_buffers():
        if name in dict(unquantized.named_buffers()):
            dict(unquantized.named_buffers())[name].data.copy_(
                buf.data.to(target_dtype)
            )

    print(f"[dequantize] Dequantized {quantized_count} quantized params, "
          f"pass-through {passthrough_count} non-quantized params")

    if mismatched:
        print(f"[dequantize] WARNING: {len(mismatched)} parameter(s) in "
              f"the quantized model have no counterpart in the unquantized "
              f"model (sample: {mismatched[:3]})")

    # ── Step 5: save ────────────────────────────────────────────────
    print(f"[dequantize] Saving to {output_path} …")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # state_dict includes both parameters and buffers by default
    state_dict = unquantized.state_dict()
    # Ensure everything is on CPU and in the target dtype
    state_dict = {
        k: v.detach().cpu().to(target_dtype) if v.is_floating_point()
           else v.detach().cpu()
        for k, v in state_dict.items()
    }

    save_file(state_dict, output_path)

    # Also drop a config.json next to the output so users who want a
    # full diffusers directory can just add diffusion_pytorch_model.safetensors
    config_output = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        os.path.basename(output_path).replace(".safetensors", ".config.json")
    )
    shutil.copy(config_json, config_output)

    input_size = os.path.getsize(input_path) / 1e9
    output_size = os.path.getsize(output_path) / 1e9
    print()
    print(f"[dequantize] Done.")
    print(f"  Input  (NF4) : {input_size:.2f} GB")
    print(f"  Output ({target_dtype}): {output_size:.2f} GB")
    print(f"  Saved: {output_path}")
    print(f"  Config: {config_output}")
    print()
    print(f"Drop {output_path} into the Component Loader's transformer_path")
    print(f"field.  It will load via the direct state-dict fallback path "
          f"with 100% key match.")


def main():
    parser = argparse.ArgumentParser(
        description="Dequantize a bitsandbytes NF4 Flux transformer checkpoint.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the NF4 .safetensors checkpoint"
    )
    parser.add_argument(
        "--base-model", required=True,
        help="Path to a diffusers Flux model directory (for transformer/config.json)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for the dequantized .safetensors file"
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
        help="Target dtype for the dequantized weights (default: bf16)"
    )
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype]

    try:
        dequantize_nf4(
            args.input, args.base_model, args.output, target_dtype,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
