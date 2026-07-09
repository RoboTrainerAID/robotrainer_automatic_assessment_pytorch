"""
Masking math for time-series models.

The input layout itself (how X is structured, how to unpack it) lives in
framework/data/schema.py — import split_inputs/group_shapes from there.
This module only contains the mask arithmetic.

IMPORTANT: lengths/masks must always be derived from the mask tensors,
never from the values. After standardization the padded positions are
re-zeroed, but inferring validity from values (the old `|x| > 1e-8`
sentinel) is fragile and was broken by scaling.
"""

import torch


def timestep_mask(x_mask: torch.Tensor) -> torch.Tensor:
    """
    Collapse the per-channel mask to a per-timestep validity mask.

    Args:
        x_mask: (B, P, C, T) bool/float — True where the bin holds data.

    Returns:
        (B, P, T) float mask — 1.0 where ANY channel has data at that bin.
    """
    return x_mask.any(dim=2).float()


def masked_mean_over_time(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean over the time dimension counting only valid timesteps.

    Args:
        feat: (N, T, D) features per timestep.
        mask: (N, T) float/bool validity per timestep.

    Returns:
        (N, D) masked mean. Sequences with no valid step return zeros.
    """
    m = mask.float().unsqueeze(-1)                # (N, T, 1)
    denom = m.sum(dim=1).clamp(min=1.0)           # (N, 1)
    return (feat * m).sum(dim=1) / denom


def mask_lengths(mask: torch.Tensor) -> torch.Tensor:
    """
    Sequence lengths for pack_padded_sequence: index of last valid
    timestep + 1. Sequences without any valid step get length 1
    (their pooled output is zeroed by the mask anyway).

    Args:
        mask: (N, T) bool/float validity per timestep.

    Returns:
        (N,) int64 lengths on CPU (as required by pack_padded_sequence).
    """
    m = mask.bool()
    T = m.size(1)
    any_valid = m.any(dim=1)
    # position of last True: T - argmax(reversed)
    last = T - m.flip(1).float().argmax(dim=1)
    lengths = torch.where(any_valid, last, torch.ones_like(last))
    return lengths.long().cpu()


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax that assigns zero weight to invalid positions.

    Rows without any valid position return all-zero weights (so a
    weighted sum over them yields zeros, consistent with
    masked_mean_over_time).

    Args:
        logits: (..., T) attention logits.
        mask: same shape, bool/float validity.

    Returns:
        weights with the same shape; sums to 1 over `dim` for rows that
        have at least one valid position, 0 otherwise.
    """
    m = mask.bool()
    w = torch.softmax(logits.masked_fill(~m, -1e9), dim=dim)
    return w * m.float()


def masked_mean_std_over_time(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Statistics pooling: masked mean AND masked std over time, concatenated.

    Args:
        feat: (N, T, D) features per timestep.
        mask: (N, T) float/bool validity per timestep.

    Returns:
        (N, 2*D) = [mean, std]. Sequences with no valid step return
        (near-)zeros (std is clamped at sqrt(1e-6) for gradient safety).
    """
    m = mask.float().unsqueeze(-1)                       # (N, T, 1)
    denom = m.sum(dim=1).clamp(min=1.0)                  # (N, 1)
    mean = (feat * m).sum(dim=1) / denom                 # (N, D)
    var = (((feat - mean.unsqueeze(1)) * m) ** 2).sum(dim=1) / denom
    std = var.clamp(min=1e-6).sqrt()
    return torch.cat([mean, std], dim=-1)


def masked_avg_pool1d(x: torch.Tensor, mask: torch.Tensor, stride: int):
    """
    Temporal downsampling that respects the validity mask: each output
    bin is the mean of the VALID inputs in its window (padding never
    dilutes the average), and the new mask marks windows that contained
    at least one valid input.

    Args:
        x: (N, D, T) features (channels-first, as used by Conv1d).
        mask: (N, T) bool/float validity per timestep.
        stride: window size == stride (non-overlapping windows).

    Returns:
        (x_ds, mask_ds): (N, D, ceil(T/stride)) and (N, ceil(T/stride)) float mask.
    """
    if stride <= 1:
        return x, mask.float()
    m = mask.float().unsqueeze(1)                        # (N, 1, T)
    # Identical pooling for numerator and denominator -> any implicit
    # zero-padding of the last (partial) window cancels in the ratio.
    num = torch.nn.functional.avg_pool1d(x * m, kernel_size=stride, stride=stride, ceil_mode=True)
    den = torch.nn.functional.avg_pool1d(m, kernel_size=stride, stride=stride, ceil_mode=True)
    x_ds = num / den.clamp(min=1e-8)
    mask_ds = (den.squeeze(1) > 0).float()
    return x_ds, mask_ds
