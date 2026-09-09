"""2x upscale-VAE decode for Qwen/Wan latents (ADR-030).

spacepxl's `Wan2.1-VAE-upscale2x` is a decoder-only finetune of the Wan 2.1
VAE that emits **12** channels instead of 3. Those 12 channels are a
`pixel_shuffle(2)` block, so unshuffling them yields an RGB image at twice the
latent-implied resolution: generate at 1024, decode at 2048, without a second
diffusion pass. It shares Qwen's latent space, which is why this path is gated
to the Qwen family.

The decode is memory-hungry at high resolution, so the public entry point
offloads the transformer first and — this is the part that matters —
guarantees it is put back. diffusers pipelines do not re-place components on
the next call: a transformer left on CPU means the following run either dies
on a device mismatch or silently runs a 20B model on CPU at roughly 1/100th
speed, which looks like a hang rather than an error.

Provenance: written for the comfyless runtime so the extracted package carries
no third-party lineage on this path (ADR-045 slice 3b). The node pack keeps
its own copy, untouched. `test_upscale_vae_decode.py` pins equivalence against
frozen goldens captured from that original.

Note on the latent normalisation below: it is written as
``spatial / (1 / std) + mean`` rather than the algebraically identical
``spatial * std + mean``. That is deliberate. The two differ in the last bit
in floating point, the result is hashed by the pixel-baseline harness, and
matching the original exactly is the whole point of this module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Decode is tiled once either latent side exceeds this, which corresponds to
#: roughly 1024px before the 2x upscale.
_TILE_THRESHOLD_LAT = 128

#: Tile geometry for the tiled path, in SAMPLE (pixel) space.
_TILE_MIN = 256
_TILE_STRIDE = 192


def unpack_qwen_latents(latents: torch.Tensor, height: int, width: int,
                        vae_scale_factor: int) -> torch.Tensor:
    """Flow-packed ``(B, seq, C*4)`` -> spatial ``(B, C, 1, H_lat, W_lat)``.

    Each packed token holds a 2x2 spatial block of latent channels, so this
    is a regroup-and-transpose rather than any kind of interpolation. The
    singleton frame axis is what the Wan VAE expects, since it is natively a
    video model.
    """
    batch, _tokens, packed_channels = latents.shape
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    channels = packed_channels // 4

    blocks = latents.view(batch, h_lat // 2, w_lat // 2, channels, 2, 2)
    # (B, h/2, w/2, C, 2, 2) -> (B, C, h/2, 2, w/2, 2) so the 2x2 block axes
    # sit next to the spatial axes they belong to before the flatten.
    blocks = blocks.permute(0, 3, 1, 4, 2, 5)
    return blocks.reshape(batch, channels, 1, h_lat, w_lat)


def _resolve_decode_device(pipe_vae, device):
    """Explicit device wins; otherwise follow the pipeline's own VAE.

    Never defaults to a bare ``"cuda"``: that aliases to cuda:0 and would
    drag pipeline state across GPUs for anyone who loaded on cuda:1.
    """
    if device is not None:
        return device
    try:
        return next(pipe_vae.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_latents_with_upscale_vae(
    packed_latents: torch.Tensor,
    upscale_vae,
    pipe_vae,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    device=None,
) -> torch.Tensor:
    """Decode packed Qwen latents through the 2x upscale VAE.

    Args:
        packed_latents: ``(B, seq, C*4)`` straight from
            ``pipe(output_type="latent")``.
        upscale_vae: the loaded ``AutoencoderKLWan`` upscale decoder.
        pipe_vae: the pipeline's OWN VAE — read only for its
            ``latents_mean`` / ``latents_std`` config, which the upscale
            decoder does not carry.
        height, width: pixel size of the stage that produced the latents,
            BEFORE the 2x upscale.
        vae_scale_factor: spatial compression of the latent space.
        device: optional override for where the decode runs.

    Returns:
        ``(B, 2*height, 2*width, 3)`` float32 in [0, 1], channels-last.
    """
    device = _resolve_decode_device(pipe_vae, device)
    dtype = next(upscale_vae.parameters()).dtype
    upscale_vae = upscale_vae.to(device)

    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    if h_lat > _TILE_THRESHOLD_LAT or w_lat > _TILE_THRESHOLD_LAT:
        upscale_vae.enable_tiling(
            tile_sample_min_height=_TILE_MIN,
            tile_sample_min_width=_TILE_MIN,
            tile_sample_stride_height=_TILE_STRIDE,
            tile_sample_stride_width=_TILE_STRIDE,
        )
        print(f"[EricQwen] Tiled VAE decode enabled (latent {h_lat}×{w_lat})")
    else:
        upscale_vae.use_tiling = False

    try:
        spatial = unpack_qwen_latents(packed_latents, height, width, vae_scale_factor)
        spatial = spatial.to(device=device, dtype=dtype)

        # Undo the per-channel scaling the pipeline applied at encode time.
        # See the module docstring for why this is a divide by the reciprocal.
        z_dim = pipe_vae.config.z_dim
        latents_mean = (
            torch.tensor(pipe_vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )
        inv_latents_std = (
            1.0
            / torch.tensor(pipe_vae.config.latents_std)
            .view(1, z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )
        spatial = spatial / inv_latents_std + latents_mean

        with torch.no_grad():
            decoded = upscale_vae.decode(spatial, return_dict=False)[0]

        # (B, 12, 1, H, W) -> drop the frame axis -> unshuffle 12ch into RGB
        # at 2x the spatial size.
        decoded = decoded.squeeze(2)
        image = F.pixel_shuffle(decoded, upscale_factor=2)

        image = (image + 1.0) / 2.0
        image = torch.clamp(image, 0.0, 1.0)
        return image.permute(0, 2, 3, 1).cpu().float()

    finally:
        # Always give the VRAM back, including on the OOM this decode is the
        # most likely operation in the run to cause.
        upscale_vae.to("cpu")
        torch.cuda.empty_cache()


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
    """:func:`decode_latents_with_upscale_vae` with transformer offload/restore.

    Takes the whole ``pipe`` rather than just its VAE, because it needs
    ``pipe.transformer`` to offload and ``pipe.vae`` for the latent stats.

    The transformer's device is captured BEFORE the offload and restored in a
    ``finally``, so every exit path — return, exception, KeyboardInterrupt —
    leaves the pipeline where the next run expects it. A failure to restore is
    warned about loudly rather than raised, since by then the caller already
    has its image and losing it to a cleanup error would be worse.
    """
    try:
        transformer_device = next(pipe.transformer.parameters()).device
    except (StopIteration, AttributeError):
        transformer_device = None

    try:
        if transformer_device is not None and transformer_device.type != "cpu":
            try:
                pipe.transformer = pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
            except Exception as exc:
                print(f"{log_prefix} Transformer offload failed "
                      f"(continuing with decode anyway): {exc}")

        print(f"{log_prefix} Upscale VAE decode (2×) ...")
        return decode_latents_with_upscale_vae(
            packed_latents, upscale_vae, pipe.vae,
            height, width, vae_scale_factor,
            device=device,
        )

    finally:
        if transformer_device is not None:
            try:
                current = next(pipe.transformer.parameters()).device
                if current != transformer_device:
                    pipe.transformer = pipe.transformer.to(transformer_device)
                    print(f"{log_prefix} Transformer restored to "
                          f"{transformer_device} (was {current} after "
                          f"upscale VAE decode)")
            except Exception as exc:
                print(f"{log_prefix} WARNING: failed to restore transformer "
                      f"to {transformer_device}: {exc}. Next run may be very "
                      f"slow or crash — restart ComfyUI if generation is "
                      f"stuck.")
