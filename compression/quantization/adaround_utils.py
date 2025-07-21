"""
 Some functions in this file is copied from https://github.com/yhhhli/BRECQ
 and modified for this project's needs.
"""
from typing import Union, Sequence

import torch

from compression.quantization.helpers import ste_floor, ste_clip
from compression.quantization.quantization import two_pow
from constants import DEVICE


def get_soft_targets(
    alpha: torch.Tensor,
    gamma: float,
    zeta: float,
    bitwidth: Union[int, Sequence[int], torch.Tensor]
) -> torch.Tensor:
    """
    Compute AdaRound soft-targets per weight.

    Parameters
    ----------
    alpha    : torch.Tensor
        Trainable parameters (same shape as the weight tensor).
    gamma    : float
        Lower stretch factor for multi-bit channels.
    zeta     : float
        Upper stretch factor for multi-bit channels.
    bitwidth : int | Sequence[int] | torch.Tensor
        • int  → single bit-width for all channels
        • 1-D list / tuple / tensor → per-channel bit-widths
          (length must equal alpha.shape[0]).

    Returns
    -------
    torch.Tensor
        Soft targets in (0, 1) with the same shape as `alpha`.
    """
    # Base gate in (0,1)
    s = torch.sigmoid(alpha)

    # Fast path: single bit-width for the whole tensor
    if isinstance(bitwidth, (int, float)):
        if bitwidth == 1:                          # binary
            return s
        return torch.clamp(s * (zeta - gamma) + gamma, 0.0, 1.0)

    # Per-channel bit-widths ────────────────────────────────────────────────
    # Convert to tensor on the correct device / dtype
    bw = torch.as_tensor(bitwidth, device=alpha.device)

    if bw.ndim != 1 or bw.numel() != alpha.shape[0]:
        raise ValueError(f"bitwidth must be 1-D and match alpha's channel dimension, {bitwidth.shape}")

    # Create broadcastable mask: True for binary channels
    mask = (bw == 1).view(-1, *([1] * (alpha.ndim - 1)))  # shape → (C,1,1,…)

    # Stretch for multi-bit, keep s for binary, broadcast automatically
    stretched = torch.clamp(s * (zeta - gamma) + gamma, 0.0, 1.0)
    return torch.where(mask, s, stretched)


# def modified_soft_quantization(w, alpha, n_bits, delta, gamma, zeta, zero_point, bitwidths, training=False, eps=1e-8,
#                                gradient_factor=1.0):
#     w = w.detach()
#     w_floor = ste_floor(w / (delta + eps), gradient_factor=gradient_factor)
#
#     if training:
#         w_int = w_floor + get_soft_targets(alpha, gamma, zeta, bitwidths)
#     else:
#         w_int = w_floor + (alpha >= 0).float()
#
#     n_levels = two_pow(n_bits).to(DEVICE)
#
#     # Compute max values for clamping:
#     if isinstance(n_levels, torch.Tensor) and n_levels.dim() > 0:
#         # per‐channel max: shape [C,1] for broadcasting across W
#         max_vals = n_levels - 1
#         max_vals = max_vals.view(max_vals.shape[0], *([1] * (w_int.ndim - 1)))
#         min_vals = torch.zeros_like(max_vals, dtype=w_int.dtype, device=w_int.device)
#     else:
#         # scalar max
#         max_vals = n_levels - 1
#         min_vals = 0
#     w_quant = ste_clip(w_int + zero_point, min_vals, max_vals)
#     w_float_q = (w_quant - zero_point) * delta
#
#     return w_float_q


def modified_soft_quantization(
    w: torch.Tensor,
    alpha: torch.Tensor,
    n_bits: Union[int, torch.Tensor],
    delta: torch.Tensor,
    gamma: float,
    zeta: float,
    zero_point: torch.Tensor,
    bitwidths: Union[int, float, Sequence[int], Sequence[float], torch.Tensor],
    training: bool = False,
    eps: float = 1e-8,
    gradient_factor: float = 1.0
) -> torch.Tensor:
    """
    AdaRound-style weight quantisation that handles per-channel 1-bit layers.

    * 1-bit → signed binary grid  {−Δ, +Δ}
    * ≥2 bit → regular uniform grid with learnable offset

    Parameters
    ----------
    w, alpha      : full-precision weights and AdaRound parameters (same shape).
    n_bits        : scalar or per-channel tensor – used for clamping ≥2-bit chans.
    delta         : per-channel scale (broadcastable to `w`).
    gamma, zeta   : AdaRound stretch factors.
    zero_point    : per-channel or scalar zero-point (ignored for 1-bit chans).
    bitwidths     : int | list | 1-D tensor (length = out-channels).
    training      : True → soft targets, False → hard rounding.
    """

    w = w.detach()
    DEVICE = w.device  # use the weight's device throughout
    # ───────────────────────────────────────────────────────────────────
    # 1.  Make `bitwidths` a tensor  [C]
    # ───────────────────────────────────────────────────────────────────
    if isinstance(bitwidths, (int, float)):
        bw_tensor = torch.full((w.shape[0],), bitwidths, device=DEVICE)
    elif isinstance(bitwidths, (list, tuple)):
        bw_tensor = torch.tensor(bitwidths, device=DEVICE)
    else:                                   # already a tensor
        bw_tensor = bitwidths.to(DEVICE)

    if bw_tensor.ndim != 1 or bw_tensor.numel() != w.shape[0]:
        raise ValueError("len(bitwidths) must equal the first dim (channels) of w")

    # mask with shape broadcastable to w : True ↦ binary channels
    bin_mask = (bw_tensor == 1).view(-1, *([1] * (w.ndim - 1)))

    # ───────────────────────────────────────────────────────────────────
    # 2.  MULTI-BIT branch  (re-use original path)
    # ───────────────────────────────────────────────────────────────────
    w_floor = ste_floor(w / (delta + eps), gradient_factor=gradient_factor)

    if training:
        soft = get_soft_targets(alpha, gamma, zeta, bw_tensor)
        w_int_multi = w_floor + soft
    else:
        w_int_multi = w_floor + (alpha >= 0).float()

    n_levels = two_pow(n_bits).to(DEVICE)    # scalar or per-channel tensor

    if torch.is_tensor(n_levels) and n_levels.dim() > 0:
        max_vals = (n_levels - 1).view(-1, *([1] * (w.ndim - 1)))
        min_vals = torch.zeros_like(max_vals, dtype=w.dtype)
    else:
        max_vals, min_vals = n_levels - 1, 0

    w_q_multi  = ste_clip(w_int_multi + zero_point, min_vals, max_vals)
    w_deq_mul  = (w_q_multi - zero_point) * delta

    # ───────────────────────────────────────────────────────────────────
    # 3.  BINARY branch   {−Δ, +Δ}
    # ───────────────────────────────────────────────────────────────────
    if training:
        s            = get_soft_targets(alpha, gamma, zeta, bw_tensor)  # (0,1)
        w_deq_bin    = delta * (2 * s - 1)                              # soft
    else:
        hard         = (alpha >= 0).float()
        w_deq_bin    = delta * (2 * hard - 1)                           # hard

    # ───────────────────────────────────────────────────────────────────
    # 4.  Combine per channel
    # ───────────────────────────────────────────────────────────────────
    w_out = torch.where(bin_mask, w_deq_bin, w_deq_mul)
    return w_out