"""NAG (Normalized Attention Guidance) for Flux.1 — ADR-024.

Port of the ADR-023 Krea-2 pattern to `FluxPipeline` (installed diffusers
0.39.0 API — the refactored `FluxAttention(AttentionModuleMixin)` style,
NOT the old `Attention` class the official reference was written against).

Flux.1 runs 19 dual-stream blocks (separate text/image streams, joint
attention, text-first concat, split back inside the processor) followed by
38 single-stream blocks (block concatenates `[text | image]` and re-splits
after; the processor sees one joint tensor). NAG applies to the IMAGE
tokens in BOTH block types (hazard N7: a single-prefix filter would
silently skip the 38 single blocks):

- dual: the attention output is split into (image, text) inside the
  processor — merge lanes on the isolated image tensor, pre-`to_out`.
- single: merge on `[:, text_seq_len:]` of the joint output (text-first),
  with `text_seq_len` injected on the processor instance (the single-block
  call path never passes `encoder_hidden_states`).

Block-level AdaLN gates sit OUTSIDE the processors and are lane-identical
(same temb both lanes), so lane re-sync keeps image tokens identical
through every block — ADR-024 H6-revised.

Latents are tiled to batch-2 (the reference repo's B-latent tiling trick +
Trunc-norm monkeypatch is rejected — ADR-024 Alternatives Rejected).
"""

from __future__ import annotations

import numpy as np
import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.utils import logging

from comfyless.core.pipelines.nag_common import nag_lane_merge_full, nag_lane_merge_tail


logger = logging.get_logger(__name__)


class NAGFluxAttnProcessor:
    """FluxAttnProcessor with NAG on the image-token slice.

    Mirrors the stock processor exactly (`_get_qkv_projections`, unflatten
    heads, RMSNorms, text-first added-KV concat, rotary with
    `sequence_dim=1`, `dispatch_attention_fn`) and merges lanes on the
    image tokens: dual-stream via the post-split image tensor, single-
    stream via the injected `text_seq_len` tail slice.
    """

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
                # Dual-stream: `hidden_states` is now the pure image slice.
                hidden_states = nag_lane_merge_full(
                    hidden_states, self.nag_scale, self.nag_tau, self.nag_alpha
                )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states
        else:
            if guidance_ready and self.text_seq_len is not None:
                # Single-stream: joint [text | image]; image is the tail.
                hidden_states = nag_lane_merge_tail(
                    hidden_states, self.text_seq_len,
                    self.nag_scale, self.nag_tau, self.nag_alpha,
                )
            return hidden_states


def apply_nag_flux_processors(
    transformer,
    *,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
    text_seq_len: int,
) -> dict:
    """Install NAG processors on EVERY Flux attention module — both the
    `transformer_blocks.*` (dual) and `single_transformer_blocks.*`
    (single) prefixes carry image tokens (ADR-024 N7). Returns the
    original processor mapping for `remove_nag_flux_processors`. Each NAG
    processor copies the replaced instance's backend/parallel pins (H1).
    """
    origin = dict(transformer.attn_processors)
    replacement = {}
    for name, proc in origin.items():
        nag_proc = NAGFluxAttnProcessor(
            nag_scale=nag_scale,
            nag_tau=nag_tau,
            nag_alpha=nag_alpha,
            text_seq_len=text_seq_len,
        )
        nag_proc._attention_backend = getattr(proc, "_attention_backend", None)
        nag_proc._parallel_config = getattr(proc, "_parallel_config", None)
        replacement[name] = nag_proc
    transformer.set_attn_processor(replacement)
    return origin


def remove_nag_flux_processors(transformer, origin: dict) -> None:
    """Restore the processors captured by `apply_nag_flux_processors`
    (pass a copy — `set_attn_processor` pops from the dict it's handed)."""
    transformer.set_attn_processor(dict(origin))


class NAGFluxPipeline(FluxPipeline):
    """FluxPipeline with NAG negative guidance (`nag_scale > 1` activates).

    `nag_scale <= 1` delegates to the stock `__call__` untouched. With
    true-CFG active (`true_cfg_scale > 1` + a negative) or IP-adapter
    inputs, NAG is skipped with a loud warning (v1 scope). Invocable
    UNBOUND on a stock FluxPipeline via `nag_pipe_call`.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        prompt_2=None,
        negative_prompt=None,
        negative_prompt_2=None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        negative_ip_adapter_image=None,
        negative_ip_adapter_image_embeds=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
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
            prompt=prompt, prompt_2=prompt_2,
            negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
            true_cfg_scale=true_cfg_scale, height=height, width=width,
            num_inference_steps=num_inference_steps, sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt, generator=generator,
            latents=latents, prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            negative_ip_adapter_image=negative_ip_adapter_image,
            negative_ip_adapter_image_embeds=negative_ip_adapter_image_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            output_type=output_type, return_dict=return_dict,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        # Dormant: byte-identical stock behavior.
        if nag_scale <= 1.0:
            return FluxPipeline.__call__(self, **stock_kwargs)

        # true-CFG already consumes the negative (2x transformer passes);
        # combining it with NAG is out of scope — loud skip (N1).
        has_neg = negative_prompt is not None or (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
        )
        if true_cfg_scale > 1 and has_neg:
            logger.warning(
                "NAG requested (nag_scale=%s) but true CFG is active "
                "(true_cfg_scale=%s > 1 with a negative) — CFG already "
                "consumes the negative prompt. Skipping NAG.",
                nag_scale, true_cfg_scale,
            )
            return FluxPipeline.__call__(self, **stock_kwargs)
        if any(x is not None for x in (
                ip_adapter_image, ip_adapter_image_embeds,
                negative_ip_adapter_image, negative_ip_adapter_image_embeds)):
            logger.warning(
                "NAG requested but IP-adapter inputs are present — the NAG "
                "batch-2 lane layout does not support IP-adapter in v1. "
                "Skipping NAG.",
            )
            return FluxPipeline.__call__(self, **stock_kwargs)

        # ── NAG-active path: mirrors FluxPipeline.__call__ (diffusers
        # 0.39.0) with batch-2 [positive | negative] lanes. ──
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, prompt_2, height, width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None else None
        )

        # Positive lane.
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        # Negative lane (NAG negative; empty-prompt default).
        if negative_prompt is None and negative_prompt_embeds is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        negative_prompt_embeds, negative_pooled_prompt_embeds, _ = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        # T5 embeds are padded to max_sequence_length, so both lanes share
        # text_seq_len and ONE unbatched text_ids serves both.
        #
        # HF1-1 (found by the lane re-sync test): Flux.1's temb is
        # time_text_embed(timestep, POOLED text projections) — per-lane
        # pooled embeds would give each lane different AdaLN modulation and
        # the image tokens would diverge after every block, breaking NAG's
        # same-queries requirement (the reference repo's TruncAdaLayerNorm
        # hack modulates the image stream with the positive temb for the
        # same reason). The negative CONTEXT enters through the T5 token
        # sequence; the pooled conditioning stays positive in BOTH lanes.
        nag_prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        nag_pooled_embeds = torch.cat(
            [pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )
        del negative_pooled_prompt_embeds  # deliberately unused (HF1-1)

        origin_attn_procs = dict(self.transformer.attn_processors)
        nag_applied = False

        try:
            apply_nag_flux_processors(
                self.transformer,
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
                text_seq_len=prompt_embeds.shape[1],
            )
            nag_applied = True

            self.scheduler.set_begin_index(0)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    if nag_applied and i >= nag_end * self._num_timesteps:
                        remove_nag_flux_processors(self.transformer, origin_attn_procs)
                        nag_applied = False

                    self._current_timestep = t
                    if nag_applied:
                        latent_in = torch.cat([latents, latents], dim=0)
                        embeds_in = nag_prompt_embeds
                        pooled_in = nag_pooled_embeds
                    else:
                        latent_in = latents
                        embeds_in = prompt_embeds
                        pooled_in = pooled_prompt_embeds
                    timestep = t.expand(latent_in.shape[0]).to(latents.dtype)

                    if self.transformer.config.guidance_embeds:
                        guidance = torch.full(
                            [1], guidance_scale, device=device, dtype=torch.float32
                        ).expand(latent_in.shape[0])
                    else:
                        guidance = None

                    # Stock wraps this in transformer.cache_context("cond");
                    # deliberately omitted — comfyless enables no diffusers
                    # step-cache helpers, and tagging a batch-2 NAG call as
                    # "cond" would poison such a cache if one ever appeared.
                    noise_pred = self.transformer(
                        hidden_states=latent_in,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_in,
                        encoder_hidden_states=embeds_in,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    if nag_applied:
                        # Lane re-sync makes both lanes' predictions
                        # identical; keep lane 0.
                        noise_pred = noise_pred[: latents.shape[0]]

                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
        finally:
            # N6: unconditional restore — a cached pipeline must never leak
            # NAG processors, even on error or partial swap.
            remove_nag_flux_processors(self.transformer, origin_attn_procs)

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

        return FluxPipelineOutput(images=image)


def nag_pipe_call(pipe, **call_kwargs):
    """Run `NAGFluxPipeline.__call__` unbound on a (possibly stock)
    FluxPipeline instance — the daemon/MCP cached-pipeline path."""
    return NAGFluxPipeline.__call__(pipe, **call_kwargs)
