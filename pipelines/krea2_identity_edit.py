"""Krea-2 identity-preserving instruction edit (ADR-043, epic
``docs/vision/epic-krea2-identity-edit.md``).

Two conditioning paths run together, both at the *pipeline* layer — diffusers'
native ``Krea2Transformer2DModel`` already accepts caller-built ``position_ids``
and an arbitrary-length packed ``hidden_states``, so nothing here patches the
transformer:

1. **In-context source prepend.** Each source image is VAE-encoded at the target
   resolution and prepended as clean tokens carrying RoPE frame ``1..N``; the
   target carries frame 0. The transformer's own output slice drops the text
   lane, and we drop the source lanes.
2. **Grounded instruction encode.** The instruction is encoded through the
   Qwen3-VL tower with the source image(s) in a vision block, so edit semantics
   are grounded on the actual pixels rather than the words alone.

``ref_boost`` adds ``log(ref_boost)`` to the attention logits where target
queries attend to source keys. It rides a swapped attention processor (the
``nag_krea2.py`` mold) rather than a reimplemented forward, because the native
forward builds its ``attention_mask`` internally from ``encoder_attention_mask``
and offers no seam for a float bias.

**Known limits, measured 2026-07-31 (epic "Live validation"):** ``ref_boost``
cannot separate identity from a spatially-adjacent edit — it instructs the model
to copy harder from the very region being changed. Hair edits work near 1.25 and
are suppressed outright at the model card's default of 4.0. Tuned values ride
the catalog rather than a schema default (epic D11).

Two sources are ``[scene, identity]`` in that fixed order (epic D8). Two-source
is *compositing*, not attribute transfer: it cannot copy a haircut from image 1,
because the slot carrying the reference also carries identity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import PIL.Image

from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline
from diffusers.pipelines.krea2.pipeline_output import Krea2PipelineOutput

__all__ = [
    "MAX_SOURCES",
    "DEFAULT_REF_BOOST",
    "DEFAULT_GROUNDING_PX",
    "build_vl_image_processor",
    "token_id_consistency_warnings",
    "edit_position_ids",
    "Krea2IdentityEditAttnProcessor",
    "apply_identity_processors",
    "remove_identity_processors",
    "Krea2IdentityEditPipeline",
    "identity_edit_pipe_call",
]

#: Fixed slot order for two-source edits: frame 1 = scene, frame 2 = identity.
MAX_SOURCES = 2

#: Model-card defaults. NOTE: 4.0 suits edits spatially separate from the
#: identity (clothing); face-adjacent edits measured best near 1.25 (epic D4).
DEFAULT_REF_BOOST = 4.0
DEFAULT_GROUNDING_PX = 768

_BLOCK_PREFIX = "transformer_blocks."

# Qwen3-VL preprocessing constants. Only these are code-owned; every
# shape-critical value is read from the live encoder's config (D10).
_VL_IMAGE_MEAN = [0.5, 0.5, 0.5]
_VL_IMAGE_STD = [0.5, 0.5, 0.5]
_VL_RESAMPLE = 3  # PIL.Image.BICUBIC

_GROUNDED_SYSTEM = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
_GROUNDED_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_IMAGE_PAD = "<|image_pad|>"
_VISION_BLOCK = f"<|vision_start|>{_IMAGE_PAD}<|vision_end|>"
#: Vision control tokens stripped from the USER's instruction before it is
#: pasted into the template. The processor expands one placeholder per supplied
#: image and raises once the grids run out, so an instruction that happens to
#: contain one would crash on a count mismatch it did not cause.
_VISION_CONTROL_TOKENS = (_IMAGE_PAD, "<|vision_start|>", "<|vision_end|>",
                          "<|video_pad|>")


# ---------------------------------------------------------------------------
# D10 — the VL image processor, built from the LIVE text encoder
# ---------------------------------------------------------------------------
def build_vl_image_processor(text_encoder):
    """Construct the Qwen3-VL image processor from ``text_encoder``'s own config.

    Krea-2 ships no ``preprocessor_config.json``, so there is nothing to load.
    Every shape-critical value is read from the **loaded encoder instance**, not
    from the checkpoint directory: ``--te1`` lets a user substitute the text
    encoder, and a processor built from the checkpoint path would silently
    describe an encoder that is not in the slot (ADR-043 constraint 1).
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    vision = getattr(getattr(text_encoder, "config", None), "vision_config", None)
    if vision is None:
        raise ValueError(
            "krea2-identity-edit: the loaded text encoder exposes no `vision_config`; "
            "the grounded encode needs a Qwen3-VL-class encoder with a vision tower. "
            "If you passed --te1, that checkpoint is text-only."
        )
    missing = [
        name for name in ("patch_size", "temporal_patch_size", "spatial_merge_size")
        if getattr(vision, name, None) is None
    ]
    if missing:
        raise ValueError(
            "krea2-identity-edit: the loaded text encoder's vision_config is missing "
            f"{', '.join(missing)} — cannot build a matching image processor."
        )

    return Qwen2VLImageProcessor(
        do_resize=True,
        resample=_VL_RESAMPLE,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=list(_VL_IMAGE_MEAN),
        image_std=list(_VL_IMAGE_STD),
        do_convert_rgb=True,
        patch_size=vision.patch_size,
        temporal_patch_size=vision.temporal_patch_size,
        merge_size=vision.spatial_merge_size,
    )


def build_vl_processor(text_encoder, tokenizer):
    """Compose the FULL ``Qwen3VLProcessor`` for the grounded encode.

    Stock Krea-2 text2img drives this encoder **text-only**. The identity edit
    drives the same encoder **multimodally** — the source pixels go through its
    vision tower — and that mode has an input contract the text-only path never
    touches: image placeholders expanded to one token per merged vision patch,
    and an ``mm_token_type_ids`` modality mask for multimodal RoPE, both of
    which the encoder hard-asserts on.

    Building only the IMAGE processor and tokenizing separately (the first cut,
    2026-07-31) silently made THIS repo the owner of that text-side contract,
    which is versioned with transformers rather than with our checkpoint — so
    each transformers bump could add another required field, discoverable only
    on a GPU. Two of them bit in a row on the first live run. Handing the
    pieces to HF's own processor instead keeps that half where it belongs.

    ADR-043 constraint 1 is unchanged and is the reason this is COMPOSED rather
    than loaded: the geometry still comes from the live encoder's
    ``vision_config`` via :func:`build_vl_image_processor`, and the tokenizer is
    the loaded pipeline's own. Nothing reads a checkpoint directory, so a
    ``--te1`` substitution cannot leave the processor describing an encoder that
    is not in the slot. ``AutoProcessor.from_pretrained`` is not an option
    regardless: Krea-2 ships no ``preprocessor_config.json``.

    The video processor is stock and unused — ``Qwen3VLProcessor`` requires the
    slot to be filled, and no video path exists here.
    """
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor)

    return Qwen3VLProcessor(
        image_processor=build_vl_image_processor(text_encoder),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )


def strip_vision_control_tokens(instruction: str) -> str:
    """Remove vision control tokens from USER text before templating.

    The processor expands one placeholder per supplied image and raises when the
    grids run out, so an instruction containing ``<|image_pad|>`` would fail on
    a count mismatch the user did not cause and could not diagnose. Repo-owned
    template text is added AFTER this, so the real placeholders are unaffected.
    """
    for token in _VISION_CONTROL_TOKENS:
        instruction = instruction.replace(token, "")
    return instruction


def token_id_consistency_warnings(tokenizer, text_encoder_config) -> List[str]:
    """Warn (never block) when tokenizer and encoder disagree on vision tokens.

    A ``--te1`` override can pair an encoder with the checkpoint's tokenizer.
    Mismatched ids produce a legible failure, and swapping encoders is a user
    choice, so this warns and proceeds (house rule: warn, don't block).
    """
    warnings: List[str] = []
    pairs = (
        ("<|image_pad|>", "image_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
    )
    for token, attr in pairs:
        cfg_id = getattr(text_encoder_config, attr, None)
        if cfg_id is None:
            continue
        try:
            tok_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:  # pragma: no cover - tokenizer without the vocab
            continue
        if tok_id is None or tok_id != cfg_id:
            warnings.append(
                f"krea2-identity-edit: tokenizer maps {token} to {tok_id!r} but the "
                f"text encoder config declares {attr}={cfg_id!r}. The grounded encode "
                "will misalign the vision block. Proceeding (did you pass --te1 with a "
                "mismatched encoder?)."
            )
    return warnings


# ---------------------------------------------------------------------------
# Position ids: [text | source(frame 1..N) | target(frame 0)]
# ---------------------------------------------------------------------------
def edit_position_ids(
    text_seq_len: int, grid_height: int, grid_width: int, n_src: int, device
) -> torch.Tensor:
    """Rotary coords of shape ``(text + n_src*grid + grid, 3)``.

    Text sits at the origin; source block ``i`` carries frame ``i + 1``; the
    target carries frame 0. Generic over ``n_src`` deliberately — the layout is
    the same for one or two sources, and the ORDER is semantic (epic D8).
    """
    text_ids = torch.zeros(text_seq_len, 3, device=device)

    def _img_ids(frame: int) -> torch.Tensor:
        ids = torch.zeros(grid_height, grid_width, 3, device=device)
        ids[..., 0] = frame
        ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
        ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
        return ids.reshape(grid_height * grid_width, 3)

    blocks = [text_ids]
    blocks += [_img_ids(i + 1) for i in range(n_src)]
    blocks += [_img_ids(0)]
    return torch.cat(blocks, dim=0)


# ---------------------------------------------------------------------------
# ref_boost — additive attention-logit bias, carried by a swapped processor
# ---------------------------------------------------------------------------
class Krea2IdentityEditAttnProcessor:
    """Stock Krea-2 attention plus a ``log(ref_boost)`` bias on source keys.

    Target queries get ``log(ref_boost)`` added on the source-block keys, which
    is equivalent to scaling the source's post-softmax weight before
    renormalization. ``ref_boost == 1.0`` is a no-op and is never installed.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, *, text_len: int, src_len: int, tgt_len: int, ref_boost: float):
        self.text_len = int(text_len)
        self.src_len = int(src_len)
        self.tgt_len = int(tgt_len)
        self.ref_boost = float(ref_boost)
        self._bias_cache: Optional[torch.Tensor] = None

    def _bias(self, device, dtype) -> torch.Tensor:
        """``(1, 1, L, L)`` additive bias; built once per call, reused per block."""
        if (
            self._bias_cache is not None
            and self._bias_cache.device == device
            and self._bias_cache.dtype == dtype
        ):
            return self._bias_cache
        total = self.text_len + self.src_len + self.tgt_len
        bias = torch.zeros(1, 1, total, total, device=device, dtype=dtype)
        rows0 = self.text_len + self.src_len  # first target row
        bias[:, :, rows0:, self.text_len:rows0] = math.log(max(self.ref_boost, 1e-4))
        self._bias_cache = bias
        return bias

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            # apply_rotary_emb is annotated as possibly returning a tuple (the
            # use_real=False path); the Krea rotary embeds always take the
            # tensor path, exactly as the stock Krea2AttnProcessor assumes.
            query = cast(torch.Tensor, apply_rotary_emb(query, image_rotary_emb, sequence_dim=1))
            key = cast(torch.Tensor, apply_rotary_emb(key, image_rotary_emb, sequence_dim=1))

        attn_mask = self._merge_mask(attention_mask, query.dtype, query.device)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            enable_gqa=attn.num_heads != attn.num_kv_heads,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)

    def _merge_mask(self, attention_mask, dtype, device) -> torch.Tensor:
        """Combine the caller's key-padding mask with our float bias.

        The transformer hands down either ``None`` or a broadcast ``(B, 1, 1, L)``
        mask. A row-dependent bias needs ``(*, *, L, L)``, so a bool mask is
        converted to additive form first; the result is always float.
        """
        bias = self._bias(device, dtype)
        if attention_mask is None:
            return bias
        if attention_mask.dtype == torch.bool:
            additive = torch.zeros_like(attention_mask, dtype=dtype)
            additive.masked_fill_(~attention_mask, torch.finfo(dtype).min)
            return additive + bias
        return attention_mask.to(dtype) + bias


def apply_identity_processors(
    transformer, *, text_len: int, src_len: int, tgt_len: int, ref_boost: float
) -> dict:
    """Install the ref_boost processors; return the pre-swap originals.

    Only ``transformer_blocks.`` entries are replaced, and each replacement
    copies the original's ``_attention_backend`` / ``_parallel_config`` so the
    Krea cuDNN pin survives the swap (ADR-023 hazard H1).

    The returned dict is a convenience for direct callers and tests. ``__call__``
    does NOT rely on it: it captures ``attn_processors`` before calling this, so
    that a failure part-way through ``set_attn_processor`` still has an origin to
    restore from (ADR-044 commit 1). A caller that takes the return value instead
    has no restore path on exactly the failure that needs one.
    """
    origin = dict(transformer.attn_processors)
    replacement = {}
    for name, proc in origin.items():
        if name.startswith(_BLOCK_PREFIX):
            new = Krea2IdentityEditAttnProcessor(
                text_len=text_len, src_len=src_len, tgt_len=tgt_len, ref_boost=ref_boost
            )
            new._attention_backend = getattr(proc, "_attention_backend", None)
            new._parallel_config = getattr(proc, "_parallel_config", None)
            replacement[name] = new
        else:
            replacement[name] = proc
    transformer.set_attn_processor(replacement)
    return origin


def remove_identity_processors(transformer, origin: dict) -> None:
    """Restore processors captured by :func:`apply_identity_processors`.

    ``set_attn_processor`` pops from the dict it is handed, so pass a copy —
    ``origin`` stays reusable.
    """
    transformer.set_attn_processor(dict(origin))


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class Krea2IdentityEditPipeline(Krea2Pipeline):
    """Krea2Pipeline with identity-preserving instruction edit.

    With no ``image`` the stock ``__call__`` runs untouched, so a krea text2img
    run is byte-identical to today. The ``__call__`` body may also be invoked
    UNBOUND on a stock ``Krea2Pipeline`` instance (see
    :func:`identity_edit_pipe_call`), which is what lets the daemon's pipeline
    cache stay class-agnostic.

    **Durable instance state — one deliberate exception.** This used to claim
    "adds no durable instance state," and that was false (ADR-044 security
    review, Finding 6): :meth:`_identity_vl_processor` memoizes onto
    ``self._krea2_identity_vl_processor`` / ``_krea2_identity_vl_encoder_id``,
    and under the unbound call ``self`` is the daemon's CACHED
    ``Krea2Pipeline`` — so those attributes outlive the request. They are inert
    (stock code never reads them, and the memo is keyed on
    ``(id(text_encoder), id(tokenizer))``, both pinned by the daemon's cache
    key), but ADR-044's cache-key argument leans on knowing exactly what
    survives a call, so the exception is named rather than denied. Attention
    processors are the state that must NOT survive — see ``__call__`` step 4.

    **Consequence — every method defined HERE must be called class-qualified**
    (``Krea2IdentityEditPipeline._foo(self, ...)``), never ``self._foo(...)``.
    Under the unbound call ``self`` IS a stock ``Krea2Pipeline``, so a
    ``self.``-dispatched subclass method raises ``AttributeError`` through
    diffusers' ``ConfigMixin.__getattr__``. Stock attributes (``self.vae``,
    ``self.prepare_latents``, ``self._execution_device``) and plain assignment
    (``self._guidance_scale = ...``) are fine — only subclass-defined *methods*
    need qualifying. Found on the first GPU run, 2026-07-31: CPU tests had only
    ever exercised the bound path, where ``self.`` resolves fine.
    ``test_krea2_identity.py`` now AST-guards this.
    """

    # -- grounded encode ----------------------------------------------------
    def _identity_vl_processor(self):
        """Build (and memoize on the instance) the D10 VL processor."""
        cached = getattr(self, "_krea2_identity_vl_processor", None)
        ref = getattr(self, "_krea2_identity_vl_encoder_id", None)
        # Keyed on BOTH components it is composed from: --te1 swaps the encoder
        # (geometry) and a tokenizer swap moves the vision token IDS, either of
        # which invalidates the processor.
        current = (id(self.text_encoder), id(self.tokenizer))
        if cached is not None and ref == current:
            return cached
        processor = build_vl_processor(self.text_encoder, self.tokenizer)
        self._krea2_identity_vl_processor = processor
        self._krea2_identity_vl_encoder_id = current
        return processor

    @staticmethod
    def _cap_longest_side(image: PIL.Image.Image, grounding_px: int) -> PIL.Image.Image:
        image = image.convert("RGB")
        if grounding_px and max(image.size) > grounding_px:
            scale = grounding_px / max(image.size)
            image = image.resize(
                (max(16, round(image.size[0] * scale)), max(16, round(image.size[1] * scale))),
                PIL.Image.Resampling.LANCZOS,
            )
        return image

    def _grounded_encode(
        self, instruction: str, sources: Sequence[PIL.Image.Image], grounding_px: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ``instruction`` grounded on one or two source images.

        Returns ``(prompt_embeds, prompt_embeds_mask)`` shaped like the stock
        Krea-2 text conditioning: ``(1, seq, num_text_layers, dim)`` and
        ``(1, seq)`` bool.
        """
        device = self._execution_device
        prefix_idx = self.prompt_template_encode_start_idx
        # Class-qualified, not self.-dispatched: see the __call__ note — `self`
        # here is a STOCK Krea2Pipeline, which has no subclass methods.
        processor = Krea2IdentityEditPipeline._identity_vl_processor(self)

        images = [Krea2IdentityEditPipeline._cap_longest_side(img, grounding_px)
                  for img in sources]
        if len(images) == 1:
            body = _VISION_BLOCK
        else:
            # The multi-image convention labels each slot so the VLM can refer
            # to them; order is semantic (frame 1 = scene, frame 2 = identity).
            body = "".join(f"Picture {i + 1}: {_VISION_BLOCK}" for i in range(len(images)))

        text = (_GROUNDED_SYSTEM + body
                + strip_vision_control_tokens(instruction or "")
                + _GROUNDED_SUFFIX)

        # ONE call with text AND images: the processor expands each placeholder
        # to one token per merged vision patch and returns the mm_token_type_ids
        # modality mask alongside input_ids. Both are hard requirements of the
        # encoder's multimodal path, and both are HF's to own — see
        # build_vl_processor.
        # NOTE: ProcessorMixin.__call__ funnels its kwargs through a TypedDict
        # it does not re-declare, so a type checker reads `return_tensors` as
        # unknown. It is the documented HF idiom and is exercised live.
        inputs = processor(
            text=[text], images=images,
            return_tensors="pt")  # pyright: ignore[reportCallIssue]
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device).bool()

        encoder_kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            value = inputs.get(key)
            if value is None:
                continue
            value = value.to(device)
            if torch.is_floating_point(value):
                value = value.to(self.text_encoder.dtype)
            encoder_kwargs[key] = value

        outputs = self.text_encoder(**encoder_kwargs)
        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in self.text_encoder_select_layers], dim=2
        )

        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states.to(self.transformer.dtype), attention_mask

    # -- source latents -----------------------------------------------------
    def _encode_source_latents(
        self, sources: Sequence[PIL.Image.Image], height: int, width: int
    ) -> torch.Tensor:
        """VAE-encode each source at the TARGET resolution and pack, in order.

        Training pairs are same-size, so every source is preprocessed to the
        target grid; that also keeps one ``grid_h x grid_w`` for the whole
        position-id layout.
        """
        device = self._execution_device
        packed_blocks = []
        for source in sources:
            pixels = self.image_processor.preprocess(
                source.convert("RGB"), height=height, width=width
            )
            pixels = pixels.unsqueeze(2).to(device=device, dtype=self.vae.dtype)

            latent = self.vae.encode(pixels).latent_dist.mode()
            mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latent = (latent - mean) / std
            latent = latent[:, :, 0]

            batch, channels, lat_h, lat_w = latent.shape
            packed_blocks.append(
                self._pack_latents(latent, batch, channels, lat_h, lat_w).to(
                    self.transformer.dtype
                )
            )
        return torch.cat(packed_blocks, dim=1)

    # -- entry --------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str], None] = None,
        image: Union[PIL.Image.Image, Sequence[PIL.Image.Image], None] = None,
        negative_prompt: Union[str, List[str], None] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 8,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        generator=None,
        latents: Optional[torch.Tensor] = None,
        ref_boost: float = DEFAULT_REF_BOOST,
        grounding_px: int = DEFAULT_GROUNDING_PX,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        **kwargs,
    ):
        if image is None:
            # No reference: stock text2img, byte-identical to today.
            size_kwargs: Dict[str, Any] = {}
            if height is not None:
                size_kwargs["height"] = height
            if width is not None:
                size_kwargs["width"] = width
            return Krea2Pipeline.__call__(
                self,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **size_kwargs,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                max_sequence_length=max_sequence_length,
                **kwargs,
            )

        sources = Krea2IdentityEditPipeline._normalize_sources(image)
        device = self._execution_device
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._guidance_scale = guidance_scale
        self._current_timestep = None

        for message in token_id_consistency_warnings(self.tokenizer, self.text_encoder.config):
            print(f"[comfyless] {message}")

        auto_height, auto_width = Krea2IdentityEditPipeline._target_size_for(
            self, sources[0])
        out_height = int(height) if height is not None else auto_height
        out_width = int(width) if width is not None else auto_width
        multiple = self.vae_scale_factor * self.patch_size
        if out_height % multiple or out_width % multiple:
            raise ValueError(
                f"krea2-identity-edit: height and width must be multiples of {multiple}; "
                f"got {out_height}x{out_width}."
            )

        # 1. Grounded prompt encode (both conditioning paths are co-active).
        prompt_embeds, prompt_embeds_mask = Krea2IdentityEditPipeline._grounded_encode(
            self,
            prompt if isinstance(prompt, str) else (prompt[0] if prompt else ""),
            sources,
            int(grounding_px),
        )

        # 2. Target latents + the source blocks that precede them.
        num_channels_latents = self.transformer.config.in_channels // (self.patch_size**2)
        target_latents = self.prepare_latents(
            num_images_per_prompt,
            num_channels_latents,
            out_height,
            out_width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        source_packed = Krea2IdentityEditPipeline._encode_source_latents(
            self, sources, out_height, out_width)

        grid_height = out_height // multiple
        grid_width = out_width // multiple
        position_ids = edit_position_ids(
            prompt_embeds.shape[1], grid_height, grid_width, len(sources), device
        )

        text_len = int(prompt_embeds.shape[1])
        src_len = int(source_packed.shape[1])
        tgt_len = int(target_latents.shape[1])

        # 3. Timesteps. `mu` keys off the TARGET token count, not the prepended
        # sequence — the source block is conditioning, not something we denoise.
        from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift, retrieve_timesteps

        sigmas_list: List[float] = (
            [float(s) for s in np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)]
            if sigmas is None
            else [float(s) for s in sigmas]
        )
        if bool(getattr(self.config, "is_distilled", False)):
            mu = 1.15
        else:
            mu = calculate_shift(
                tgt_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 6400),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        timesteps, _resolved_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas_list, mu=mu
        )
        step_list = list(cast(Sequence[Any], timesteps))
        self._num_timesteps = len(step_list)

        # 4. Denoise with the source prepended. ref_boost == 1.0 keeps the stock
        # processors and the maskless fast path.
        #
        # Capture the stock processors BEFORE any swap, and apply INSIDE the try,
        # so the finally below restores even from a partially-applied
        # set_attn_processor — the nag_krea2.py mold (ADR-023), which this module
        # claimed to follow and did not until ADR-044 commit 1. The install used
        # to sit outside the try with `origin` assigned only from the return
        # value, so a mid-apply failure left the identity processors installed
        # with nothing to restore them.
        #
        # This is load-bearing ONLY because the pipeline object outlives the
        # call: under `identity_edit_pipe_call` this body runs unbound on the
        # daemon's CACHED Krea2Pipeline. Residue there is not a lost run, it is a
        # silently wrong one — stale processors carry frozen text_len/src_len/
        # tgt_len, so a follow-up at a DIFFERENT resolution crashes loudly but one
        # at the SAME resolution (the --iterate sweep case) just gets a wrong
        # attention bias. See ADR-044 and its security review, Finding 2.
        origin = dict(self.transformer.attn_processors) \
            if float(ref_boost) != 1.0 else None
        try:
            if origin is not None:
                apply_identity_processors(
                    self.transformer,
                    text_len=text_len,
                    src_len=src_len,
                    tgt_len=tgt_len,
                    ref_boost=float(ref_boost),
                )
            self.scheduler.set_begin_index(0)
            for t in step_list:
                if self.interrupt:
                    continue
                self._current_timestep = t
                timestep = (
                    (t / self.scheduler.config.num_train_timesteps)
                    .expand(target_latents.shape[0])
                    .to(target_latents.dtype)
                )
                combined = torch.cat([source_packed, target_latents], dim=1)
                model_out = self.transformer(
                    hidden_states=combined,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    position_ids=position_ids,
                    encoder_attention_mask=prompt_embeds_mask,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                # The transformer drops the text lane; drop the source lanes.
                noise_pred = model_out[:, -tgt_len:]

                target_latents = self.scheduler.step(
                    noise_pred, t, target_latents, return_dict=False
                )[0]
        finally:
            if origin is not None:
                remove_identity_processors(self.transformer, origin)
            self._current_timestep = None

        # 5. Decode.
        if output_type == "latent":
            image_out = target_latents
        else:
            unpacked = self._unpack_latents(target_latents, out_height, out_width).to(
                self.vae.dtype
            )
            mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(unpacked.device, unpacked.dtype)
            )
            inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(unpacked.device, unpacked.dtype)
            unpacked = unpacked / inv_std + mean
            decoded = self.vae.decode(unpacked, return_dict=False)[0][:, :, 0]
            image_out = self.image_processor.postprocess(
                decoded, output_type=output_type or "pil"
            )

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image_out,)
        return Krea2PipelineOutput(images=cast(Any, image_out))

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _normalize_sources(image) -> List[PIL.Image.Image]:
        """Coerce the ``image`` argument to an ordered list of PIL sources.

        Order is semantic: index 0 = scene (frame 1), index 1 = identity
        (frame 2). A third source is a hard error — silently dropping one reads
        to the user as a model failure (ADR-043 constraint 2).
        """
        if isinstance(image, PIL.Image.Image):
            sources = [image]
        elif isinstance(image, (list, tuple)):
            sources = list(image)
        else:
            sources = [image]
        if not sources:
            raise ValueError("krea2-identity-edit: no source image supplied.")
        if len(sources) > MAX_SOURCES:
            raise ValueError(
                f"krea2-identity-edit accepts at most {MAX_SOURCES} reference images "
                f"(#1 = scene, #2 = identity); got {len(sources)}."
            )
        out = []
        for src in sources:
            if not isinstance(src, PIL.Image.Image):
                src = PIL.Image.fromarray(np.asarray(src))
            out.append(src.convert("RGB"))
        return out

    def _target_size_for(self, source: PIL.Image.Image, max_megapixels: float = 1.0):
        """Match the output aspect to the source and snap to the token grid."""
        multiple = self.vae_scale_factor * self.patch_size
        width, height = source.size
        megapixels = (width * height) / 1e6
        if megapixels > max_megapixels:
            scale = (max_megapixels / megapixels) ** 0.5
            width, height = round(width * scale), round(height * scale)
        width = max(multiple, (width // multiple) * multiple)
        height = max(multiple, (height // multiple) * multiple)
        return height, width


def identity_edit_pipe_call(pipe, **call_kwargs):
    """Run :meth:`Krea2IdentityEditPipeline.__call__` unbound on ``pipe``.

    The daemon and MCP server cache ONE pipeline per config; the identity edit
    must not change the cached object's class or shape (its params stay out of
    the cache key — processors are installed per call and restored in a
    ``finally``). The unbound call preserves the instance's offload hooks,
    device state, and swapped scheduler exactly.
    """
    return Krea2IdentityEditPipeline.__call__(pipe, **call_kwargs)
