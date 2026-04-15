# ────────────────────────────────────────────────────────────────────
#  Eric Qwen — Wan2.1 / Qwen-compatible 2× Upscale VAE
# ────────────────────────────────────────────────────────────────────
#  Loader and decode utility for spacepxl/Wan2.1-VAE-upscale2x.
#
#  The Wan2.1 and Qwen-Image VAEs are architecturally identical and
#  share the same latent space.  This upscale VAE is a decoder-only
#  finetune that outputs 12 channels instead of 3.  After decode,
#  pixel_shuffle(12→3, 2×) produces a 2× upscaled image — effectively
#  a free 2× super-resolution during VAE decode.
#
#  Model weights: spacepxl/Wan2.1-VAE-upscale2x (Apache-2.0)
#    https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x
#  Reference impl: spacepxl/ComfyUI-VAE-Utils (MIT)
#    https://github.com/spacepxl/ComfyUI-VAE-Utils
#
#  This file is an independent implementation (no code from the above
#  repos) that loads the model via HuggingFace Diffusers.
#
#  License (this file): CC BY-NC 4.0 / Commercial dual — Eric Hiss, 2026
# ────────────────────────────────────────────────────────────────────

from __future__ import annotations

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
#  Loader Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenUpscaleVAELoader:
    """Load the Wan2.1 2× upscale VAE (decoder-only finetune).

    Compatible with both Wan2.1 and Qwen-Image latent spaces.
    The model is kept on CPU until decode is requested.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "spacepxl/Wan2.1-VAE-upscale2x",
                    "tooltip": (
                        "HuggingFace model ID or local path.\n"
                        "Default: spacepxl/Wan2.1-VAE-upscale2x"
                    ),
                }),
            },
            "optional": {
                "subfolder": ("STRING", {
                    "default": "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1",
                    "tooltip": (
                        "Subfolder within the repo containing\n"
                        "config.json + model weights.\n"
                        "Leave blank if model_path already\n"
                        "points to the correct directory."
                    ),
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision. bfloat16 recommended.",
                }),
            },
        }

    RETURN_TYPES = ("UPSCALE_VAE",)
    RETURN_NAMES = ("upscale_vae",)
    FUNCTION = "load_vae"
    CATEGORY = "EricQwen/loaders"

    def load_vae(self, model_path: str,
                 subfolder: str = "diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1",
                 dtype: str = "bfloat16"):
        from diffusers import AutoencoderKLWan

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dt = dtype_map.get(dtype, torch.bfloat16)

        kwargs = {"torch_dtype": dt}
        if subfolder and subfolder.strip():
            kwargs["subfolder"] = subfolder.strip()

        print(f"[EricQwen] Loading upscale VAE from {model_path} ...")
        vae = AutoencoderKLWan.from_pretrained(model_path, **kwargs)
        vae.eval()
        print(f"[EricQwen] Upscale VAE loaded (dtype={dt}). "
              f"Decoder out_channels={vae.config.out_channels}")
        return (vae,)


# ═══════════════════════════════════════════════════════════════════════
#  Decode helper (used by UltraGen integration and standalone)
# ═══════════════════════════════════════════════════════════════════════

def decode_latents_with_upscale_vae(
    packed_latents: torch.Tensor,
    upscale_vae,
    pipe_vae,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    device=None,
) -> torch.Tensor:
    """Decode packed Qwen/Wan latents using the 2× upscale VAE.

    Replicates the exact same latent-normalization that the diffusers
    QwenImagePipeline applies before VAE decode, then feeds the result
    through the upscale VAE's 12-channel decoder followed by
    ``pixel_shuffle(2)`` for 2× spatial resolution.

    Args:
        packed_latents : Raw packed latents ``[B, seq, C*4]`` from
                         ``pipe(output_type="latent")``.
        upscale_vae    : The loaded ``AutoencoderKLWan`` upscale model.
        pipe_vae       : The pipeline's **original** VAE (needed for
                         ``latents_mean`` / ``latents_std`` config).
        height, width  : Pixel dimensions of the stage that produced
                         these latents (before upscale).
        vae_scale_factor : Spatial compression (default 8).
        device         : Optional explicit device.  If None, uses the
                         CURRENT device of ``pipe_vae`` (defensive
                         default that avoids hardcoding ``cuda:0`` and
                         creating cross-GPU pipeline state when the
                         pipeline was loaded on cuda:1 or elsewhere).

    Returns:
        ComfyUI IMAGE tensor ``[B, 2*H, 2*W, 3]`` in float32, [0, 1].
    """
    from .eric_qwen_image_multistage import _unpack_latents

    # Pick device: explicit > pipe_vae's current device > cuda fallback.
    # Critical fix: never hardcode "cuda" (which aliases to cuda:0) —
    # that causes cross-GPU pipeline state when the user loaded on
    # cuda:1 and breaks the next VAE operation with a device mismatch.
    if device is None:
        try:
            device = next(pipe_vae.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = next(upscale_vae.parameters()).dtype

    # 1) Move upscale VAE to GPU
    upscale_vae = upscale_vae.to(device)

    # Enable tiled decode for large latents to avoid OOM
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    # Tile if latent spatial dims exceed 128 (~1024px per side before 2× upscale)
    if h_lat > 128 or w_lat > 128:
        upscale_vae.enable_tiling(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_sample_stride_height=192,
            tile_sample_stride_width=192,
        )
        print(f"[EricQwen] Tiled VAE decode enabled (latent {h_lat}×{w_lat})")
    else:
        upscale_vae.use_tiling = False

    try:
        # 2) Unpack from [B, seq, C*4] to [B, C, 1, H_lat, W_lat]
        spatial = _unpack_latents(packed_latents, height, width,
                                  vae_scale_factor)
        spatial = spatial.to(device=device, dtype=dtype)

        # 3) Apply latent normalization (same as QwenImagePipeline decode path)
        #    latents = latents / latents_std + latents_mean
        z_dim = pipe_vae.config.z_dim
        latents_mean = (
            torch.tensor(pipe_vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(pipe_vae.config.latents_std)
            .view(1, z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )
        spatial = spatial / latents_std + latents_mean

        # 4) Decode with upscale VAE
        with torch.no_grad():
            decoded = upscale_vae.decode(spatial, return_dict=False)[0]
        # decoded shape: [B, 12, 1, H, W]

        # 5) Squeeze frame dim, pixel_shuffle 12ch → 3ch at 2× resolution
        decoded = decoded.squeeze(2)  # [B, 12, H, W]
        image = F.pixel_shuffle(decoded, upscale_factor=2)  # [B, 3, 2H, 2W]

        # 6) Normalize [-1, 1] → [0, 1]
        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0.0, 1.0)

        # 7) Convert to ComfyUI IMAGE format: [B, H, W, C]
        image = image.permute(0, 2, 3, 1).cpu().float()

        return image

    finally:
        # 8) Offload upscale VAE back to CPU
        upscale_vae.to("cpu")
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
#  Safe wrapper: transformer offload + upscale VAE decode + guaranteed restore
# ═══════════════════════════════════════════════════════════════════════

def decode_latents_with_upscale_vae_safe(
    packed_latents: torch.Tensor,
    upscale_vae,
    pipe,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    device=None,
    log_prefix: str = "[EricQwen]",
) -> torch.Tensor:
    """Upscale VAE decode with guaranteed transformer offload/restore.

    Wraps ``decode_latents_with_upscale_vae`` with the VRAM-freeing
    transformer offload that high-resolution decodes need, AND
    **guarantees the transformer gets restored to its original device
    on every exit path** (normal return, exception, interrupt).

    Why this helper exists: the naive pattern

        pipe.transformer = pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
        tensor = decode_latents_with_upscale_vae(...)
        return tensor

    leaves the transformer on CPU after the decode returns.  diffusers
    pipelines do NOT auto-move components back to GPU on the next
    ``pipe(...)`` call — they run wherever they were last placed.  The
    next run then either crashes with a device mismatch or silently
    runs Qwen/Flux 20B inference on CPU at roughly 1/100th the normal
    speed (GPU idle, CPU pegged, generation advancing at ~1%/minute).

    This helper captures the transformer's device BEFORE the offload,
    does the decode, and restores the device in a ``finally`` block so
    the next run starts in the correct state regardless of how this
    one exited.

    Args:
        packed_latents   : Latents to decode, shape ``[B, seq, C*4]``.
        upscale_vae      : The Wan2.1 upscale VAE (AutoencoderKLWan).
        pipe             : The full pipeline object (needed to access
                           both ``pipe.transformer`` and ``pipe.vae``).
                           Note: this differs from the lower-level
                           ``decode_latents_with_upscale_vae`` which
                           takes ``pipe_vae`` directly.
        height, width    : Pixel dimensions of the pre-upscale stage.
        vae_scale_factor : Spatial compression (default 8).
        device           : Optional explicit target device for the
                           decode.  If None, uses pipe_vae's current
                           device.
        log_prefix       : Log line prefix for the offload/restore
                           messages.  Defaults to "[EricQwen]".

    Returns:
        ComfyUI IMAGE tensor ``[B, 2H, 2W, 3]`` in float32, [0, 1].
    """
    # Capture transformer's current device BEFORE offloading.  This is
    # what we restore to in the finally block.
    try:
        transformer_device = next(pipe.transformer.parameters()).device
    except (StopIteration, AttributeError):
        transformer_device = None

    try:
        # Offload transformer to CPU to free VRAM for the decode.
        # High-resolution Wan2.1 decode is memory-intensive and the
        # transformer is the biggest component on the GPU at this
        # point.  Skip if we couldn't read its device.
        if transformer_device is not None and transformer_device.type != "cpu":
            try:
                pipe.transformer = pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
            except Exception as e:
                print(
                    f"{log_prefix} Transformer offload failed "
                    f"(continuing with decode anyway): {e}"
                )

        print(f"{log_prefix} Upscale VAE decode (2×) ...")
        tensor = decode_latents_with_upscale_vae(
            packed_latents, upscale_vae, pipe.vae,
            height, width, vae_scale_factor,
            device=device,
        )
        return tensor

    finally:
        # CRITICAL: restore transformer to its original device on ALL
        # exit paths.  Without this, the next pipe() call runs
        # inference on CPU (silent) or crashes on device mismatch.
        if transformer_device is not None:
            try:
                current = next(pipe.transformer.parameters()).device
                if current != transformer_device:
                    pipe.transformer = pipe.transformer.to(transformer_device)
                    print(
                        f"{log_prefix} Transformer restored to "
                        f"{transformer_device} (was {current} after "
                        f"upscale VAE decode)"
                    )
            except Exception as e:
                print(
                    f"{log_prefix} WARNING: failed to restore transformer "
                    f"to {transformer_device}: {e}. Next run may be very "
                    f"slow or crash — restart ComfyUI if generation is "
                    f"stuck."
                )


# ═══════════════════════════════════════════════════════════════════════
#  Inter-stage helper: decode 2× with upscale VAE, re-encode to latents
# ═══════════════════════════════════════════════════════════════════════

def upscale_between_stages(
    packed_latents: torch.Tensor,
    upscale_vae,
    pipe_vae,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    device=None,
) -> tuple:
    """Decode latents at 2× via the upscale VAE, then re-encode with the
    standard Qwen VAE to produce packed latents at 2× spatial resolution.

    This replaces the bislerp inter-stage upscale with a higher-quality
    decode→2×→re-encode round trip so the next stage operates on a
    sharper, higher-resolution canvas.

    Normalization math (must match diffusers QwenImagePipeline):
      Decode: ``spatial = packed_norm / latents_std + latents_mean``
              → ``vae.decode(spatial)`` → pixels [-1, 1]
      Encode: ``vae.encode(pixels)`` → ``posterior.mode()`` → raw
              → ``packed_norm = (raw - latents_mean) * latents_std``

    Args:
        packed_latents : ``[B, seq, C*4]`` from ``output_type="latent"``.
        upscale_vae    : ``AutoencoderKLWan`` upscale model.
        pipe_vae       : Pipeline's standard ``AutoencoderKLQwenImage``.
        height, width  : Pixel dimensions of the producing stage.
        vae_scale_factor : Spatial compression (default 8).

    Returns:
        ``(packed_latents, new_h, new_w)`` — packed latents in pipeline-
        normalized format and the 2× pixel dimensions.
    """
    from .eric_qwen_image_multistage import _unpack_latents, _pack_latents

    # Pick device: explicit > pipe_vae's current device > cuda fallback.
    # Critical: NEVER hardcode "cuda" here (aliases to cuda:0), because
    # that moves pipe.vae from wherever it actually is (e.g. cuda:1)
    # onto cuda:0, leaving pipe.vae and pipe.transformer on different
    # GPUs.  The next VAE operation then fails with a device mismatch.
    if device is None:
        try:
            device = next(pipe_vae.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    up_dtype = next(upscale_vae.parameters()).dtype
    vae_dtype = next(pipe_vae.parameters()).dtype

    # Detect whether pipe_vae is accelerate-dispatched (balanced mode).
    # When dispatched, calling pipe_vae.to(device) emits a warning and
    # leaves the VAE in a half-moved state where the outermost module's
    # device attribute changed but the per-layer parameters stayed
    # where accelerate originally placed them.  The next encode call
    # then crashes because accelerate's per-layer hooks try to route
    # inputs from one device to weights on another.  Skip the move
    # entirely when dispatched and trust accelerate's hooks to handle
    # device routing during encode.
    pipe_vae_dispatched = hasattr(pipe_vae, "_hf_hook")
    if not pipe_vae_dispatched:
        for submodule in pipe_vae.modules():
            if hasattr(submodule, "_hf_hook"):
                pipe_vae_dispatched = True
                break

    z_dim = pipe_vae.config.z_dim

    # Pre-compute normalization vectors
    mean_t = torch.tensor(pipe_vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
    std_t = torch.tensor(pipe_vae.config.latents_std).view(1, z_dim, 1, 1, 1)

    # ── Step 1: Decode with upscale VAE → 2× pixels ──────────────────
    upscale_vae = upscale_vae.to(device)
    try:
        spatial = _unpack_latents(packed_latents, height, width,
                                  vae_scale_factor)
        spatial = spatial.to(device=device, dtype=up_dtype)

        # Decode normalization: vae_input = packed * config_std + config_mean
        #   (matches QwenImagePipeline: latents / (1/std) + mean = latents * std + mean)
        std_dec = std_t.to(device=device, dtype=up_dtype)
        mean_dec = mean_t.to(device=device, dtype=up_dtype)
        spatial = spatial * std_dec + mean_dec

        with torch.no_grad():
            decoded = upscale_vae.decode(spatial, return_dict=False)[0]
        # decoded: [B, 12, 1, H_lat, W_lat]

        decoded = decoded.squeeze(2)                          # [B, 12, H, W]
        pixels_2x = F.pixel_shuffle(decoded, upscale_factor=2)  # [B, 3, 2H, 2W]
        # pixels in [-1, 1] — keep raw for re-encoding
        del decoded, spatial
    finally:
        upscale_vae.to("cpu")
        torch.cuda.empty_cache()

    new_h = pixels_2x.shape[2]
    new_w = pixels_2x.shape[3]

    # ── Step 2: Re-encode with standard Qwen VAE → latents at 2× ─────
    # Only call pipe_vae.to(device) when it's NOT accelerate-dispatched.
    # Dispatched VAEs are managed by accelerate's per-layer hooks, and
    # calling .to() on them creates a half-moved state that crashes the
    # next encode call.
    if not pipe_vae_dispatched:
        pipe_vae = pipe_vae.to(device)
    try:
        # VAE encode expects [B, C, num_frames, H, W].  We move pixels
        # to the SAME device as the VAE's first parameter — for an
        # accelerate-dispatched VAE that's where accelerate placed the
        # first layer, and accelerate's hooks will route inputs through
        # subsequent layers automatically.
        try:
            target_device_for_input = next(pipe_vae.parameters()).device
        except (StopIteration, AttributeError):
            target_device_for_input = device
        pixels_5d = pixels_2x.unsqueeze(2).to(
            device=target_device_for_input, dtype=vae_dtype,
        )
        del pixels_2x

        with torch.no_grad():
            posterior = pipe_vae.encode(pixels_5d).latent_dist
            raw_latents = posterior.mode()
            # raw_latents: [B, z_dim, 1, H_lat_new, W_lat_new]
        del pixels_5d

        # Inverse normalization: packed = (raw - mean) / std
        #   (inverse of decode: vae_input = packed * std + mean)
        # Place normalization constants on raw_latents' actual device,
        # not the function's `device` variable — accelerate may have
        # moved raw_latents to a different device than where pixels_5d
        # started, and mixing devices in the arithmetic crashes.
        raw_device = raw_latents.device
        mean_enc = mean_t.to(device=raw_device, dtype=vae_dtype)
        std_enc = std_t.to(device=raw_device, dtype=vae_dtype)
        norm_latents = (raw_latents - mean_enc) / std_enc

        # Pack to pipeline format [B, seq, C*4]
        new_packed = _pack_latents(norm_latents)
    finally:
        # Don't offload pipe_vae here — the pipeline manages it
        pass

    print(f"[UltraGen] Inter-stage VAE upscale: {height}×{width} → "
          f"{new_h}×{new_w} (latent {norm_latents.shape[3]}×{norm_latents.shape[4]})")

    return new_packed, new_h, new_w
