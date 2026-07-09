"""Shared NAG (Normalized Attention Guidance) math — ADR-023 / ADR-024.

`nag_merge` is the single formula every per-architecture NAG module uses
(paper arXiv:2505.21179 Eqs. 7-10, reference `nag/attention_nag.py`
L103-110). L1 norms per the paper — a deliberate, cross-family-consistent
deviation from the reference repo's Flux file, which uses L2 (ADR-024
Alternatives Rejected).

Scale convention (code form): `Z_g = Z+ * s - Z- * (s - 1)`; `s <= 1` is a
mathematical no-op, which is why `nag_scale > 1` activates everywhere.
"""

from __future__ import annotations

import torch


def nag_merge(
    z_positive: torch.Tensor,
    z_negative: torch.Tensor,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
) -> torch.Tensor:
    """The NAG formula (paper Eqs. 7-10; reference attention_nag.py L103-110).

    Extrapolate `Z+ * scale - Z- * (scale-1)`, clip the per-token L1-norm
    growth ratio at `nag_tau`, then blend with `nag_alpha`. Norms are per
    token over the feature dim (dim=-1, keepdim), L1 per the paper.
    """
    z_guidance = z_positive * nag_scale - z_negative * (nag_scale - 1.0)
    norm_positive = torch.norm(
        z_positive, p=1, dim=-1, keepdim=True
    ).expand(*z_positive.shape)
    norm_guidance = torch.norm(
        z_guidance, p=1, dim=-1, keepdim=True
    ).expand(*z_guidance.shape)

    ratio = norm_guidance / norm_positive
    z_guidance = z_guidance * torch.minimum(
        ratio, ratio.new_ones(1) * nag_tau
    ) / ratio

    return z_guidance * nag_alpha + z_positive * (1.0 - nag_alpha)


def nag_lane_merge_tail(
    hidden_states: torch.Tensor,
    text_seq_len: int,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
) -> torch.Tensor:
    """NAG + lane re-sync for a text-FIRST joint sequence (flux-family,
    krea): image tokens are `[:, text_seq_len:]`; batch is `[pos | neg]`
    lanes. Mutates and returns `hidden_states` (callers own the tensor).
    """
    origin_batch = hidden_states.shape[0] // 2
    image_positive = hidden_states[:origin_batch, text_seq_len:]
    image_negative = hidden_states[origin_batch:, text_seq_len:]
    guided = nag_merge(image_positive, image_negative,
                       nag_scale, nag_tau, nag_alpha)
    hidden_states[:origin_batch, text_seq_len:] = guided
    hidden_states[origin_batch:, text_seq_len:] = guided
    return hidden_states


def nag_lane_merge_front(
    hidden_states: torch.Tensor,
    image_seq_len: int,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
) -> torch.Tensor:
    """NAG + lane re-sync for an image-FIRST joint sequence (Z-Image basic
    mode `[x, cap]`): image tokens are `[:, :image_seq_len]`.
    """
    origin_batch = hidden_states.shape[0] // 2
    image_positive = hidden_states[:origin_batch, :image_seq_len]
    image_negative = hidden_states[origin_batch:, :image_seq_len]
    guided = nag_merge(image_positive, image_negative,
                       nag_scale, nag_tau, nag_alpha)
    hidden_states[:origin_batch, :image_seq_len] = guided
    hidden_states[origin_batch:, :image_seq_len] = guided
    return hidden_states


def nag_lane_merge_full(
    image_states: torch.Tensor,
    nag_scale: float,
    nag_tau: float,
    nag_alpha: float,
) -> torch.Tensor:
    """NAG + lane re-sync when the tensor is ALREADY the pure image slice
    (flux-family dual-stream blocks split text off before this runs).
    """
    origin_batch = image_states.shape[0] // 2
    guided = nag_merge(image_states[:origin_batch], image_states[origin_batch:],
                       nag_scale, nag_tau, nag_alpha)
    image_states[:origin_batch] = guided
    image_states[origin_batch:] = guided
    return image_states
