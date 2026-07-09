"""NAG (Normalized Attention Guidance) for Flux.2 and Flux.2-Klein — ADR-024.

Flux.2 is dual-architecture: `Flux2TransformerBlock` dual-stream blocks
(separate text/image streams, text-first joint attention, split back inside
the processor) plus `Flux2SingleTransformerBlock` PARALLEL blocks (ViT-22B
style: QKV fused with the MLP-in projection, and the attention
out-projection FUSED with the MLP-out into one `to_out`). Two NAG processor
variants are therefore required, selected by module-path prefix; a
Krea-style single-prefix filter would silently skip the single-stream
stack (hazard N7).

Merge points (ADR-024):
- dual: post-split image tensor, pre-`to_out` (mirrors nag_flux).
- parallel: the `[:, text_seq_len:]` tail of the attention output BEFORE
  the `cat([attn, mlp])` + fused `to_out` (N10 — after the fusion the MLP
  path is folded in and the seam is gone). `text_seq_len` is injected on
  the processor instance (the parallel call path has no
  `encoder_hidden_states` argument).

The base `Flux2Pipeline` has NO negative-prompt path at all (pure
guidance-distilled); the NAG mirror adds the negative encode itself,
following the Klein pipeline's pattern. Both encoders (Mistral3 base /
Qwen3 Klein) pad text to a fixed `max_sequence_length`, so the batch-2
lanes always share text length, rope, and ids (the transformer collapses
3-D id tensors to `[0]` anyway). Reference-image (kontext-style) inputs
are NOT NAG-supported in v1 — warn + stock (HF2-1).
"""

from __future__ import annotations

import numpy as np
import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import _get_qkv_projections
from diffusers.pipelines.flux2.pipeline_flux2 import (
    Flux2Pipeline,
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.utils import logging

from pipelines.nag_common import nag_lane_merge_full, nag_lane_merge_tail


logger = logging.get_logger(__name__)

_DUAL_PREFIX = "transformer_blocks."
_SINGLE_PREFIX = "single_transformer_blocks."


class NAGFlux2AttnProcessor:
    """Flux2AttnProcessor (dual-stream) with NAG on the post-split image
    tensor. Mirrors the stock processor exactly otherwise."""

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        nag_scale: float = 1.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
    ) -> None:
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        batch_size = hidden_states.shape[0]
        guidance_ready = (
            self.nag_scale > 1.0 and batch_size % 2 == 0 and batch_size >= 2
        )

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            if guidance_ready:
                hidden_states = nag_lane_merge_full(
                    hidden_states, self.nag_scale, self.nag_tau, self.nag_alpha
                )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class NAGFlux2ParallelSelfAttnProcessor:
    """Flux2ParallelSelfAttnProcessor (single-stream / parallel) with NAG
    on the `[:, text_seq_len:]` tail of the attention output, applied
    BEFORE the fused attn+MLP out-projection (N10)."""

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        nag_scale: float = 1.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        text_seq_len: int | None = None,
    ) -> None:
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.text_seq_len = text_seq_len

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        batch_size = hidden_states.shape[0]
        if (
            self.nag_scale > 1.0
            and self.text_seq_len is not None
            and batch_size % 2 == 0
            and batch_size >= 2
        ):
            hidden_states = nag_lane_merge_tail(
                hidden_states, self.text_seq_len,
                self.nag_scale, self.nag_tau, self.nag_alpha,
            )

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


def apply_nag_flux2_processors(
    transformer,
    *,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
    text_seq_len: int,
) -> dict:
    """Install NAG processors on BOTH Flux.2 block families, selecting the
    variant by module-path prefix (dual vs parallel — N7). Returns the
    original mapping; backend/parallel pins are copied per instance (H1).
    """
    origin = dict(transformer.attn_processors)
    replacement = {}
    for name, proc in origin.items():
        if name.startswith(_DUAL_PREFIX):
            nag_proc = NAGFlux2AttnProcessor(
                nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha,
            )
        elif name.startswith(_SINGLE_PREFIX):
            nag_proc = NAGFlux2ParallelSelfAttnProcessor(
                nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha,
                text_seq_len=text_seq_len,
            )
        else:
            replacement[name] = proc
            continue
        nag_proc._attention_backend = getattr(proc, "_attention_backend", None)
        nag_proc._parallel_config = getattr(proc, "_parallel_config", None)
        replacement[name] = nag_proc
    transformer.set_attn_processor(replacement)
    return origin


def remove_nag_flux2_processors(transformer, origin: dict) -> None:
    """Restore the processors captured by `apply_nag_flux2_processors`."""
    transformer.set_attn_processor(dict(origin))


def _nag_denoise_flux2(
    pipe,
    *,
    latents,
    prompt_embeds,
    negative_prompt_embeds,
    text_ids,
    latent_ids,
    timesteps,
    num_inference_steps,
    num_warmup_steps,
    guidance_scale,
    use_guidance_embed: bool,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
    nag_end: float,
    callback_on_step_end,
    callback_on_step_end_tensor_inputs,
    device,
):
    """Shared batch-2 NAG denoising loop for Flux2 base and Klein mirrors.

    Klein passes `use_guidance_embed=False` (its transformer call uses
    `guidance=None`); the base pipeline always embeds guidance. Returns
    the final latents. Installs/restores processors with the N6
    unconditional finally.
    """
    nag_prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)

    origin_attn_procs = dict(pipe.transformer.attn_processors)
    nag_applied = False
    try:
        apply_nag_flux2_processors(
            pipe.transformer,
            nag_scale=nag_scale,
            nag_tau=nag_tau,
            nag_alpha=nag_alpha,
            text_seq_len=prompt_embeds.shape[1],
        )
        nag_applied = True

        if hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index(0)
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                if nag_applied and i >= nag_end * len(timesteps):
                    remove_nag_flux2_processors(pipe.transformer, origin_attn_procs)
                    nag_applied = False

                pipe._current_timestep = t
                if nag_applied:
                    latent_in = torch.cat([latents, latents], dim=0)
                    embeds_in = nag_prompt_embeds
                else:
                    latent_in = latents
                    embeds_in = prompt_embeds
                timestep = t.expand(latent_in.shape[0]).to(latents.dtype)
                latent_model_input = latent_in.to(pipe.transformer.dtype)

                if use_guidance_embed:
                    guidance = torch.full(
                        [1], guidance_scale, device=device, dtype=torch.float32
                    ).expand(latent_in.shape[0])
                else:
                    guidance = None

                # ids may be 3-D (batched); the transformer collapses them
                # to [0], so both lanes share rope without tiling.
                # (Klein's stock loop wraps this in cache_context("cond");
                # deliberately omitted — comfyless enables no step-cache
                # helpers, and a batch-2 NAG call must not be cache-tagged.)
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=embeds_in,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=pipe.attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1) :]
                if nag_applied:
                    noise_pred = noise_pred[: latents.shape[0]]

                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0
                ):
                    progress_bar.update()
    finally:
        # N6: unconditional restore.
        remove_nag_flux2_processors(pipe.transformer, origin_attn_procs)

    return latents


class NAGFlux2Pipeline(Flux2Pipeline):
    """Flux2Pipeline with NAG. The stock pipeline has no negative path at
    all (guidance-distilled) — NAG is the ONLY way a negative prompt does
    anything here. `negative_prompt` is a NEW parameter on this mirror."""

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        prompt=None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list | None = None,
        guidance_scale: float | None = 4.0,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple = (10, 20, 30),
        caption_upsample_temperature: float = None,
        negative_prompt=None,
        nag_scale: float = 0.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        nag_end: float = 1.0,
    ):
        stock_kwargs = dict(
            image=image, prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps, sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt, generator=generator,
            latents=latents, prompt_embeds=prompt_embeds,
            output_type=output_type, return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
            caption_upsample_temperature=caption_upsample_temperature,
        )

        if nag_scale <= 1.0:
            return Flux2Pipeline.__call__(self, **stock_kwargs)
        if image is not None:
            # HF2-1: the kontext-style reference-image path adds image
            # latents to the joint sequence — out of NAG v1 scope.
            logger.warning(
                "NAG requested but reference image(s) supplied — NAG does "
                "not support the reference-image path in v1. Skipping NAG.",
            )
            return Flux2Pipeline.__call__(self, **stock_kwargs)

        # ── NAG-active mirror of Flux2Pipeline.__call__ (0.39.0) ──
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if caption_upsample_temperature:
            prompt = self.upsample_prompt(
                prompt, images=image, temperature=caption_upsample_temperature, device=device
            )
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        # NAG negative lane — the stock pipeline never encodes one.
        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        negative_prompt_embeds, _ = self.encode_prompt(
            prompt=negative_prompt,
            prompt_embeds=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        latents = _nag_denoise_flux2(
            self,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            text_ids=text_ids,
            latent_ids=latent_ids,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            guidance_scale=guidance_scale,
            use_guidance_embed=True,
            nag_scale=nag_scale,
            nag_tau=nag_tau,
            nag_alpha=nag_alpha,
            nag_end=nag_end,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            device=device,
        )

        self._current_timestep = None

        if output_type == "latent":
            image_out = latents
        else:
            latents = self._unpack_latents_with_ids(latents, latent_ids)
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(latents.device, latents.dtype)
            latents = latents * latents_bn_std + latents_bn_mean
            latents = self._unpatchify_latents(latents)
            image_out = self.vae.decode(latents, return_dict=False)[0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)

        from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

        return Flux2PipelineOutput(images=image_out)


class NAGFlux2KleinPipeline(Flux2KleinPipeline):
    """Flux2KleinPipeline with NAG. The distilled Klein checkpoint
    (`is_distilled=true`) is CFG-dead — the NAG target. When real CFG
    would run (non-distilled + guidance > 1), NAG steps aside loudly.
    `negative_prompt` (string) is a NEW parameter on this mirror — the
    stock Klein only takes `negative_prompt_embeds`."""

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        prompt=None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list | None = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple = (9, 18, 27),
        negative_prompt=None,
        nag_scale: float = 0.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        nag_end: float = 1.0,
    ):
        stock_kwargs = dict(
            image=image, prompt=prompt, height=height, width=width,
            num_inference_steps=num_inference_steps, sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt, generator=generator,
            latents=latents, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type, return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if nag_scale <= 1.0:
            return Flux2KleinPipeline.__call__(self, **stock_kwargs)
        if guidance_scale > 1 and not self.config.is_distilled:
            logger.warning(
                "NAG requested (nag_scale=%s) but classic CFG is active on "
                "this non-distilled Klein (guidance_scale=%s > 1) — CFG "
                "already consumes the negative. Skipping NAG.",
                nag_scale, guidance_scale,
            )
            return Flux2KleinPipeline.__call__(self, **stock_kwargs)
        if image is not None:
            logger.warning(
                "NAG requested but reference image(s) supplied — NAG does "
                "not support the reference-image path in v1. Skipping NAG.",
            )
            return Flux2KleinPipeline.__call__(self, **stock_kwargs)

        # ── NAG-active mirror of Flux2KleinPipeline.__call__ (0.39.0) ──
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )
        if negative_prompt is None and negative_prompt_embeds is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        negative_prompt_embeds, _ = self.encode_prompt(
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        latents = _nag_denoise_flux2(
            self,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            text_ids=text_ids,
            latent_ids=latent_ids,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            guidance_scale=guidance_scale,
            use_guidance_embed=False,  # Klein passes guidance=None
            nag_scale=nag_scale,
            nag_tau=nag_tau,
            nag_alpha=nag_alpha,
            nag_end=nag_end,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            device=device,
        )

        self._current_timestep = None

        # Klein tail (0.39.0): latents are unpacked BEFORE the output_type
        # branch, with pre-computed latent height/width.
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latents = self._unpack_latents_with_ids(
            latents, latent_ids, latent_height // 2, latent_width // 2
        )
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        if output_type == "latent":
            image_out = latents
        else:
            image_out = self.vae.decode(latents, return_dict=False)[0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out,)

        from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

        return Flux2PipelineOutput(images=image_out)


def nag_pipe_call(pipe, **call_kwargs):
    """Unbound NAG call for a cached Flux2 / Flux2Klein pipeline —
    dispatches on the instance's class."""
    if isinstance(pipe, Flux2KleinPipeline) or pipe.__class__.__name__ == "Flux2KleinPipeline":
        return NAGFlux2KleinPipeline.__call__(pipe, **call_kwargs)
    return NAGFlux2Pipeline.__call__(pipe, **call_kwargs)
