"""NAG (Normalized Attention Guidance) for Z-Image — ADR-024.

Z-Image inverts several Krea-2/Flux assumptions:

- The joint sequence is **image-FIRST** (`[x, cap]` in basic mode) — the
  NAG merge slices the FRONT `[:, :image_seq_len]` (hazard N11), where
  `image_seq_len` is the image-token count padded up to SEQ_MULTI_OF=32
  (HZ-2); identical across lanes because both lanes share the latent
  resolution.
- Text is genuinely **variable-length**: embeds are Python lists of
  per-sample tensors and the transformer pads ragged batches internally
  (`pad_sequence` + boolean attn mask), so unequal positive/negative
  prompt lengths need no special handling.
- The transformer has **no AttentionMixin** (HZ-1/N9) — processors are
  hand-swapped on `transformer.layers[i].attention` (the 30 joint
  blocks ONLY; the noise/context/siglip refiner stacks are
  single-modality and stay stock — N8).
- The pipeline's CFG convention is `guidance_scale > 0` (verified
  pipeline_z_image.py:282) — at cfg>0 classic CFG consumes the negative,
  so comfyless gates NAG to the cfg<=0 regime (the Turbo recommendation).
- The velocity is sign-flipped before the scheduler step (HZ-3) — the
  mirror preserves `noise_pred = -noise_pred` and the float32 latents
  discipline exactly.
"""

from __future__ import annotations

import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.pipelines.z_image.pipeline_z_image import (
    ZImagePipeline,
    calculate_shift,
    get_default_z_image_sigmas,
    retrieve_timesteps,
)
from diffusers.utils import logging

from pipelines.nag_common import nag_lane_merge_front


logger = logging.get_logger(__name__)

# Image tokens are padded up to a multiple of this inside the transformer
# (transformer_z_image.SEQ_MULTI_OF); the front-slice length must match.
_SEQ_MULTI_OF = 32


class NAGZSingleStreamAttnProcessor:
    """ZSingleStreamAttnProcessor with NAG on the front image slice.

    Mirrors the stock processor exactly (module projections, per-head
    RMSNorms, complex-RoPE via `freqs_cis`, mask promotion,
    `dispatch_attention_fn`) and merges lanes on `[:, :image_seq_len]`
    before `to_out[0]`.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        nag_scale: float = 1.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        image_seq_len: int | None = None,
    ) -> None:
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.image_seq_len = image_seq_len

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def apply_rotary_emb(x_in: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs = freqs.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        batch_size = hidden_states.shape[0]
        if (
            self.nag_scale > 1.0
            and self.image_seq_len is not None
            and batch_size % 2 == 0
            and batch_size >= 2
        ):
            # Image-FIRST joint sequence: merge the front slice (N11).
            hidden_states = nag_lane_merge_front(
                hidden_states, self.image_seq_len,
                self.nag_scale, self.nag_tau, self.nag_alpha,
            )

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


def apply_nag_zimage_processors(
    transformer,
    *,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
    image_seq_len: int,
    origin_out: list | None = None,
) -> list:
    """Install NAG processors on the 30 JOINT blocks (`transformer.layers`)
    by hand — Z-Image has no AttentionMixin (HZ-1). The refiner stacks
    (noise/context/siglip) are single-modality and stay stock (N8).

    Each (attention_module, original_processor) pair is appended to
    `origin_out` (or a fresh list) BEFORE its swap, so a caller-owned list
    tracks every swapped layer even if this loop dies mid-apply — the
    finally-restore then unwinds exactly the swapped prefix (reviewer F1,
    2026-07-09: an internal list returned at the end loses the prefix on a
    mid-apply exception, leaking NAG processors into a cached pipeline).
    Backend/parallel pins copied (H1). Returns the list.
    """
    origin = origin_out if origin_out is not None else []
    for blk in transformer.layers:
        attn = blk.attention
        old_proc = attn.processor
        nag_proc = NAGZSingleStreamAttnProcessor(
            nag_scale=nag_scale,
            nag_tau=nag_tau,
            nag_alpha=nag_alpha,
            image_seq_len=image_seq_len,
        )
        nag_proc._attention_backend = getattr(old_proc, "_attention_backend", None)
        nag_proc._parallel_config = getattr(old_proc, "_parallel_config", None)
        origin.append((attn, old_proc))
        attn.processor = nag_proc
    return origin


def remove_nag_zimage_processors(origin: list) -> None:
    """Restore the (module, processor) pairs captured at apply time.
    Idempotent; safe on a partially-applied swap PROVIDED the caller holds
    the list `apply_nag_zimage_processors` appended into (pass
    `origin_out` and keep a reference before the try)."""
    for attn, old_proc in origin:
        attn.processor = old_proc


class NAGZImagePipeline(ZImagePipeline):
    """ZImagePipeline with NAG negative guidance.

    NAG targets the cfg-0 regime (the Turbo recommendation): at
    `guidance_scale > 0` classic CFG already consumes the negative and
    NAG steps aside loudly. Invocable UNBOUND on a stock ZImagePipeline
    via `nag_pipe_call`.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list | None = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt=None,
        num_images_per_prompt: int | None = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list = ["latents"],
        max_sequence_length: int = 512,
        nag_scale: float = 0.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        nag_end: float = 1.0,
    ):
        stock_kwargs = dict(
            prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps, sigmas=sigmas,
            guidance_scale=guidance_scale,
            cfg_normalization=cfg_normalization, cfg_truncation=cfg_truncation,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt, generator=generator,
            latents=latents, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type, return_dict=return_dict,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        if nag_scale <= 1.0:
            return ZImagePipeline.__call__(self, **stock_kwargs)
        if guidance_scale > 0:
            # ZImagePipeline runs classic CFG at any guidance_scale > 0
            # (verified 0.39.0) — the negative already works there.
            logger.warning(
                "NAG requested (nag_scale=%s) but classic CFG is active "
                "(guidance_scale=%s > 0) — CFG already consumes the "
                "negative prompt on Z-Image. Skipping NAG; run "
                "guidance_scale=0 (the Turbo recommendation) to use NAG.",
                nag_scale, guidance_scale,
            )
            return ZImagePipeline.__call__(self, **stock_kwargs)

        # ── NAG-active mirror of ZImagePipeline.__call__ (0.39.0) ──
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # Encode BOTH lanes: pass do_classifier_free_guidance=True so the
        # stock encoder produces the negative list even though our own
        # guidance is 0 (stock at cfg 0 skips the negative entirely).
        if prompt_embeds is not None and prompt is None:
            if negative_prompt_embeds is None:
                raise ValueError(
                    "NAG with `prompt_embeds` requires `negative_prompt_embeds`."
                )
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=True,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            negative_prompt_embeds = [
                npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)
            ]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        # The transformer pads image tokens up to a SEQ_MULTI_OF multiple;
        # the processor's front slice must cover the padded length (HZ-2).
        padded_image_seq_len = image_seq_len + (-image_seq_len) % _SEQ_MULTI_OF

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        if sigmas is None:
            sigmas = get_default_z_image_sigmas(num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        self.scheduler.set_begin_index(0)

        # Caller-owned capture list: apply appends each pair BEFORE its
        # swap, so a mid-apply failure still restores the swapped prefix
        # in the finally (N6/HZ-1; reviewer F1 2026-07-09).
        origin_procs: list = []
        nag_applied = False
        try:
            apply_nag_zimage_processors(
                self.transformer,
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
                image_seq_len=padded_image_seq_len,
                origin_out=origin_procs,
            )
            nag_applied = True

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    if nag_applied and i >= nag_end * self._num_timesteps:
                        remove_nag_zimage_processors(origin_procs)
                        nag_applied = False

                    timestep = t.expand(latents.shape[0])
                    timestep = (1000 - timestep) / 1000

                    latents_typed = latents.to(self.transformer.dtype)
                    if nag_applied:
                        # Batch-2 [positive | negative] lanes, mirroring the
                        # stock CFG batch build (list-based API).
                        latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                        embeds_in = prompt_embeds + negative_prompt_embeds
                        timestep_in = timestep.repeat(2)
                    else:
                        latent_model_input = latents_typed
                        embeds_in = prompt_embeds
                        timestep_in = timestep

                    latent_model_input = latent_model_input.unsqueeze(2)
                    latent_model_input_list = list(latent_model_input.unbind(dim=0))

                    model_out_list = self.transformer(
                        latent_model_input_list, timestep_in, embeds_in, return_dict=False
                    )[0]

                    if nag_applied:
                        # Lane re-sync makes both lanes' image predictions
                        # identical; keep the positive lanes.
                        model_out_list = model_out_list[:actual_batch_size]

                    noise_pred = torch.stack([o.float() for o in model_out_list], dim=0)
                    noise_pred = noise_pred.squeeze(2)
                    # HZ-3: stock velocity sign flip.
                    noise_pred = -noise_pred

                    latents = self.scheduler.step(
                        noise_pred.to(torch.float32), t, latents, return_dict=False
                    )[0]
                    assert latents.dtype == torch.float32

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds", negative_prompt_embeds
                        )

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
        finally:
            # N6: unconditional restore (idempotent on the pair list).
            remove_nag_zimage_processors(origin_procs)

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput

        return ZImagePipelineOutput(images=image)


def nag_pipe_call(pipe, **call_kwargs):
    """Run `NAGZImagePipeline.__call__` unbound on a (possibly stock)
    ZImagePipeline instance — the daemon/MCP cached-pipeline path."""
    return NAGZImagePipeline.__call__(pipe, **call_kwargs)
