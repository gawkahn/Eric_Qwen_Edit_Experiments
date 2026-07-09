"""NAG (Normalized Attention Guidance) for Krea-2 — ADR-023.

Training-free negative guidance for guidance-distilled (cfg<=0) Krea-2
checkpoints, ported from the official Flux single-stream reference
(github.com/ChenDarYen/Normalized-Attention-Guidance,
`nag/attention_flux_nag.py` + `nag/pipeline_flux_nag.py`; formula
`nag/attention_nag.py` L103-110, paper arXiv:2505.21179 Eqs. 7-10).

Krea-2 runs a single joint `[text | image]` sequence through 28 identical
self-attention blocks (no cross-attention), which is exactly the Flux
single-stream case. NAG runs the transformer at batch 2 (positive lane,
negative lane), extrapolates the attention output on the IMAGE-TOKEN slice
only, L1-normalizes the growth (tau clip), blends (alpha), and writes the
guided image tokens back into BOTH lanes so the negative branch never
drifts into an independent trajectory (lane re-sync — part of the method,
not an optimization).

Scale convention: `nag_scale` follows the reference CODE form
`Z_g = Z+ * scale - Z- * (scale - 1)` (== paper's `Z+ + phi*(Z+ - Z-)`
with phi = scale - 1). scale <= 1 is a mathematical no-op and NAG stays
fully dormant (no processors touched).

Implementation hazards tracked from ADR-023:
  H1 — NAG processors copy `_attention_backend` / `_parallel_config` from
       the processor they replace, so comfyless's cuDNN pin survives the
       swap (SDPA math-fallback OOM at high res otherwise).
  H2 — only `transformer_blocks.*` processors are replaced; the
       text_fusion Krea2Attention instances keep their stock processors.
  H3 — attention runs through `dispatch_attention_fn` with `enable_gqa`
       exactly as the stock Krea2AttnProcessor (48 q / 12 kv heads).
  H5 — projections are invoked as modules (`attn.to_q(...)`), never via
       `.weight`, so torchao-quantized Linears (--quant fp8) work as-is.
  H6 — NAG applies after the sigmoid output-gate multiply and before
       `to_out[0]` (the Flux "before out-projection" placement; the
       lanes' image-token gates are identical because of lane re-sync).
"""

from __future__ import annotations

import numpy as np
import torch

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.krea2.pipeline_krea2 import (
    Krea2Pipeline,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.utils import logging


logger = logging.get_logger(__name__)

# Only processors under this module prefix get NAG'd (hazard H2): the
# text_fusion stage also hosts Krea2Attention instances, but NAG is defined
# on the joint [text|image] sequence of the main blocks only.
_NAG_BLOCK_PREFIX = "transformer_blocks."


# The formula moved to nag_common with the ADR-024 family expansion;
# re-exported here so existing importers (tests) keep working unchanged.
from pipelines.nag_common import nag_merge  # noqa: F401


class NAGKrea2AttnProcessor:
    """Krea2AttnProcessor with NAG on the image-token slice.

    Mirrors the stock processor exactly (projections, q/k norms, rotary,
    dispatch_attention_fn with GQA, sigmoid gate) and then — when guidance
    is active — applies `nag_merge` to the image tokens using the batch's
    positive lane (first half) vs negative lane (second half), writing the
    guided tokens back into both lanes (lane re-sync).

    NAG state lives on the processor instance: Krea2Attention.forward
    filters kwargs against the processor signature, so `attention_kwargs`
    cannot carry it.
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
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)

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
            enable_gqa=attn.num_heads != attn.num_kv_heads,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)

        batch_size = hidden_states.shape[0]
        apply_guidance = (
            self.nag_scale > 1.0
            and self.text_seq_len is not None
            and batch_size % 2 == 0
            and batch_size >= 2
        )
        if apply_guidance:
            origin_batch = batch_size // 2
            txt = self.text_seq_len
            image_positive = hidden_states[:origin_batch, txt:]
            image_negative = hidden_states[origin_batch:, txt:]
            image_guided = nag_merge(
                image_positive, image_negative,
                self.nag_scale, self.nag_tau, self.nag_alpha,
            )
            # Lane re-sync: both lanes carry the guided image tokens so the
            # lanes differ only in their text tokens (invariant N5).
            hidden_states[:origin_batch, txt:] = image_guided
            hidden_states[origin_batch:, txt:] = image_guided

        return attn.to_out[0](hidden_states)


def apply_nag_processors(
    transformer,
    *,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
    text_seq_len: int,
) -> dict:
    """Install NAG processors on the main transformer blocks.

    Returns the ORIGINAL `attn_processors` dict for `remove_nag_processors`.
    Only names under `transformer_blocks.` are replaced (hazard H2); each
    NAG processor copies the replaced instance's `_attention_backend` /
    `_parallel_config` so the cuDNN pin survives the swap (hazard H1).
    """
    origin = dict(transformer.attn_processors)
    replacement = {}
    for name, proc in origin.items():
        if name.startswith(_NAG_BLOCK_PREFIX):
            nag_proc = NAGKrea2AttnProcessor(
                nag_scale=nag_scale,
                nag_tau=nag_tau,
                nag_alpha=nag_alpha,
                text_seq_len=text_seq_len,
            )
            nag_proc._attention_backend = getattr(proc, "_attention_backend", None)
            nag_proc._parallel_config = getattr(proc, "_parallel_config", None)
            replacement[name] = nag_proc
        else:
            replacement[name] = proc
    transformer.set_attn_processor(replacement)
    return origin


def remove_nag_processors(transformer, origin: dict) -> None:
    """Restore the processors captured by `apply_nag_processors`.

    Idempotent for a given `origin`. `set_attn_processor` pops from the
    dict it is handed, so pass a copy — `origin` stays reusable.
    """
    transformer.set_attn_processor(dict(origin))


class Krea2NAGPipeline(Krea2Pipeline):
    """Krea2Pipeline with NAG negative guidance (`nag_scale > 1` activates).

    `nag_scale <= 1` (the default) delegates to the stock `__call__`
    untouched — no processors are installed, behavior is byte-identical
    (Vision invariant 4). With classic CFG active (`guidance_scale > 0`)
    NAG is skipped with a loud warning: CFG already consumes the negative
    prompt on those checkpoints.

    The `__call__` body can also be invoked UNBOUND on a stock
    Krea2Pipeline instance (see `nag_pipe_call`) — it adds no instance
    state beyond transient attributes, which is what lets comfyless's
    daemon/MCP pipeline caches stay class-agnostic.
    """

    @property
    def do_normalized_attention_guidance(self) -> bool:
        return getattr(self, "_nag_scale", 0.0) > 1.0

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        attention_kwargs=None,
        max_sequence_length: int = 512,
        nag_scale: float = 0.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.25,
        nag_end: float = 1.0,
    ):
        stock_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            output_type=output_type,
            return_dict=return_dict,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
        )

        # Dormant: byte-identical stock behavior, stock processors untouched.
        if nag_scale <= 1.0:
            return Krea2Pipeline.__call__(self, **stock_kwargs)

        # Classic CFG already consumes the negative prompt when
        # guidance_scale > 0 — running NAG on top is out of scope for v1.
        # Loud skip, never silent (invariant N1).
        if guidance_scale > 0:
            logger.warning(
                "NAG requested (nag_scale=%s) but classic CFG is active "
                "(guidance_scale=%s > 0) — CFG already consumes the negative "
                "prompt on this checkpoint. Skipping NAG and running stock "
                "CFG. NAG targets distilled cfg<=0 checkpoints (Turbo).",
                nag_scale, guidance_scale,
            )
            return Krea2Pipeline.__call__(self, **stock_kwargs)

        # ── NAG-active path: mirrors Krea2Pipeline.__call__ (diffusers
        # 0.39.0) with a batch-2 [positive | negative] lane layout. ──
        multiple = self.vae_scale_factor * self.patch_size
        if height % multiple != 0 or width % multiple != 0:
            rounded_height = ((height + multiple - 1) // multiple) * multiple
            rounded_width = ((width + multiple - 1) // multiple) * multiple
            logger.warning(
                f"`height` and `width` must be multiples of {multiple}; rounding up "
                f"from {height}x{width} to {rounded_height}x{rounded_width}."
            )
            height, width = rounded_height, rounded_width

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._nag_scale = nag_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Positive lane.
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )
        # Negative lane (the NAG negative — Grant's UX call: --negative is
        # reused; an absent negative degrades to the empty prompt exactly
        # like the stock CFG branch).
        if negative_prompt is None and negative_prompt_embeds is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            prompt=negative_prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )

        num_channels_latents = self.transformer.config.in_channels // (self.patch_size**2)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        grid_height = height // (self.vae_scale_factor * self.patch_size)
        grid_width = width // (self.vae_scale_factor * self.patch_size)
        # Fixed-length text template: positive and negative embeds share
        # text_seq_len, so ONE unbatched position_ids serves both lanes.
        position_ids = self.prepare_position_ids(
            prompt_embeds.shape[1], grid_height, grid_width, device
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        if self.config.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 6400),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Batch-2 lane layout: [positive | negative] on the batch dim.
        nag_prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
        nag_prompt_embeds_mask = torch.cat(
            [prompt_embeds_mask, negative_prompt_embeds_mask], dim=0
        )

        # Capture the stock processors BEFORE any swap so the finally below
        # can restore even from a partially-applied set_attn_processor.
        origin_attn_procs = dict(self.transformer.attn_processors)
        nag_applied = False

        try:
            apply_nag_processors(
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

                    # nag_end window: NAG runs on the first
                    # `nag_end`-fraction of steps; past it, hot-restore the
                    # stock processors and drop back to single-lane batch.
                    if nag_applied and i >= nag_end * self._num_timesteps:
                        remove_nag_processors(self.transformer, origin_attn_procs)
                        nag_applied = False

                    self._current_timestep = t
                    if nag_applied:
                        latent_in = torch.cat([latents, latents], dim=0)
                        embeds_in = nag_prompt_embeds
                        mask_in = nag_prompt_embeds_mask
                    else:
                        latent_in = latents
                        embeds_in = prompt_embeds
                        mask_in = prompt_embeds_mask
                    timestep = (
                        (t / self.scheduler.config.num_train_timesteps)
                        .expand(latent_in.shape[0])
                        .to(latents.dtype)
                    )

                    noise_pred = self.transformer(
                        hidden_states=latent_in,
                        encoder_hidden_states=embeds_in,
                        timestep=timestep,
                        position_ids=position_ids,
                        encoder_attention_mask=mask_in,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                    if nag_applied:
                        # Lane re-sync makes both lanes' image tokens (and
                        # thus velocity predictions) identical; keep lane 0.
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
            # Invariant N6: a cached pipeline must NEVER leak NAG processors
            # into the next request — restore unconditionally, even on error
            # or a partially-applied swap (re-restoring stock over stock is
            # an idempotent no-op).
            remove_nag_processors(self.transformer, origin_attn_procs)
            self._nag_scale = 0.0

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        from diffusers.pipelines.krea2.pipeline_output import Krea2PipelineOutput

        return Krea2PipelineOutput(images=image)


def nag_pipe_call(pipe, **call_kwargs):
    """Run `Krea2NAGPipeline.__call__` unbound on a (possibly stock)
    Krea2Pipeline instance.

    The daemon and MCP server cache ONE pipeline per config; NAG must not
    change the cached object's class or shape (its params stay out of the
    cache key — the processors are installed per-call and restored in a
    `finally`). The unbound call preserves the instance's offload hooks,
    device state, and swapped scheduler exactly.
    """
    return Krea2NAGPipeline.__call__(pipe, **call_kwargs)
