from collections import Counter

from typing import List, Dict, Union

import torch

from compression.configs.layer_comrpression_config import LayerCompressionConfig
from constants import LAYER_COMPRESSION_CONFIG, SIZE, MSE


# def record_config(cost_tensor, deltas_tensor, zps_tensor, channel_idx, bit2idx, current_bits, in_channels):
#     idxs = torch.tensor([bit2idx[b] for b in current_bits], dtype=torch.long)
#     total_size = in_channels * float(sum(current_bits))
#     total_mse = float(cost_tensor[channel_idx, idxs].sum().item())
#     layer_cc = LayerCompressionConfig(bit_width_quantization=current_bits.copy())
#     layer_cc.set_weights_params(
#         deltas_tensor[channel_idx, idxs],
#         zps_tensor[channel_idx, idxs]
#     )
#     return {
#         LAYER_COMPRESSION_CONFIG: layer_cc,
#         SIZE: total_size,
#         MSE: total_mse,
#     }
#
#
# def generate_missing_uniform_configs(N, results, cost_tensor, deltas_tensor, zps_tensor, bits, channel_idx, bit2idx, in_channels):
#     missing_uniform_configs = []
#
#     # ensure uniform-bit candidates are present
#     existing_bits = {
#         cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization[0]
#         for cfg in results
#         if len(set(cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization)) == 1
#     }
#
#     for b in bits:
#         if b not in existing_bits:
#             idx = bit2idx[b]
#             total_size = in_channels * (b * N)
#             total_mse  = float(cost_tensor[channel_idx, idx].sum().item())
#             layer_cc = LayerCompressionConfig(bit_width_quantization=[b]*N)
#             layer_cc.set_weights_params(
#                 deltas_tensor[:, idx],
#                 zps_tensor   [:, idx]
#             )
#             missing_uniform_configs.append({
#                 LAYER_COMPRESSION_CONFIG: layer_cc,
#                 SIZE:  total_size,
#                 MSE:   total_mse,
#             })
#
#     return missing_uniform_configs


def record_config(
        cost_desc: torch.Tensor,
        deltas_desc: torch.Tensor,
        zps_desc: torch.Tensor,
        bit2idx: Dict[int, int],
        current_bits: List[int],
        in_channels: int,
) -> Dict[str, Union["LayerCompressionConfig", float]]:
    """
    Pack a single layer-compression candidate.

    Parameters
    ----------
    cost_desc, deltas_desc, zps_desc
        Tensors already **re-ordered** to `bits_desc` (shape [N, K]).
    current_bits
        List[int] of length N with the chosen bit-width per channel.
    """
    idxs = torch.tensor([bit2idx[b] for b in current_bits], dtype=torch.long)
    total_size = in_channels * float(sum(current_bits))
    total_mse = float(cost_desc[torch.arange(cost_desc.size(0)), idxs].sum().item())

    layer_cc = LayerCompressionConfig(bit_width_quantization=current_bits.copy())
    layer_cc.set_weights_params(
        deltas_desc[torch.arange(cost_desc.size(0)), idxs],
        zps_desc[torch.arange(cost_desc.size(0)), idxs],
    )
    return {
        LAYER_COMPRESSION_CONFIG: layer_cc,
        SIZE: total_size,
        MSE: total_mse,
    }


# --------------------------------------------------------------------------- #
# Helper: ensure uniform-bit configs exist
# --------------------------------------------------------------------------- #
def generate_missing_uniform_configs(
        N: int,
        existing: List[Dict[str, Union["LayerCompressionConfig", float]]],
        cost_desc: torch.Tensor,
        deltas_desc: torch.Tensor,
        zps_desc: torch.Tensor,
        bits_desc: List[int],
        bit2idx: Dict[int, int],
        in_channels: int,
) -> List[Dict[str, Union["LayerCompressionConfig", float]]]:
    uniform_present = {
        cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization[0]
        for cfg in existing
        if len(set(cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization)) == 1
    }
    missing = []
    for b in bits_desc:
        if b in uniform_present:
            continue
        idx = bit2idx[b]
        total_size = in_channels * float(b * N)
        total_mse = float(cost_desc[:, idx].sum().item())

        layer_cc = LayerCompressionConfig(bit_width_quantization=[b] * N)
        layer_cc.set_weights_params(deltas_desc[:, idx], zps_desc[:, idx])

        missing.append({
            LAYER_COMPRESSION_CONFIG: layer_cc,
            SIZE: total_size,
            MSE: total_mse,
        })
    return missing


def is_simd_valid(bitwidths: List[int], simd: int) -> bool:
    """
    Return True iff the multiset of bitwidths can be partitioned into
    groups of size `simd`, plus at most one “remainder” group of size N % simd.
    """
    N = len(bitwidths)
    rem = N % simd

    counts = Counter(bitwidths)
    # how many channels of each bitwidth do we have mod simd?
    bad = [cnt % simd for cnt in counts.values() if cnt % simd != 0]

    if rem == 0:
        # no remainder allowed, so all counts must be exactly divisible
        return len(bad) == 0
    else:
        # exactly one bitwidth may have the remainder count
        return len(bad) == 1 and bad[0] == rem


# def lambda_frontier_candidates(
#         cost_tensor: torch.Tensor,
#         bitwidths: List[int],
#         deltas: List[torch.Tensor],
#         zero_points: List[torch.Tensor],
#         in_channels: int,
#         max_candidates: Optional[int] = None,
#         add_uniform_candidates: bool = True,
# ) -> List[Dict[str, Union["LayerCompressionConfig", float]]]:
#     """
#     General λ-frontier candidate generator for *arbitrary* bit-width sets.
#
#     The routine starts every channel at the largest available bit-width and
#     iteratively selects the cheapest next transition (in λ = ΔMSE/Δbits)
#     until the channel reaches the smallest bit-width.  This naturally
#     supports multi-bit skips whenever those have a lower λ than all
#     sequences of finer-grained steps.
#
#     Parameters
#     ----------
#     cost_tensor
#         Distortion per channel / bit-width  (shape [N, |M|]).
#     bitwidths
#         Iterable of available bit-widths (order irrelevant).
#     deltas, zero_points
#         Quant-params in the *same* order as `bitwidths`.
#     in_channels
#         Multiplier for the total‐size calculation inside `record_config`.
#     max_candidates
#         Optional upper bound on the number of returned configs.
#     add_uniform_candidates
#         If True, guarantees that the uniform configs {b,b,…} for every
#         b ∈ bitwidths are present.
#
#     Returns
#     -------
#     List[Dict]  – one entry per candidate (see original signature).
#     """
#     # ── 1.  Sort bit-widths from *largest* → *smallest* ───────────────────────────
#     bits_desc, deltas_desc, zps_desc = zip(
#         *sorted(zip(bitwidths, deltas, zero_points), key=lambda x: -x[0])
#     )
#     bits_desc = list(bits_desc)                    # e.g. [8,4,2]
#     K = len(bits_desc)                             # number of levels
#     bit2idx = {b: i for i, b in enumerate(bits_desc)}
#
#     N = cost_tensor.shape[0]
#     rows = torch.arange(N, device=DEVICE)
#
#     # MSE matrix in the *descending* bit-order:  [N, K]
#     # Map each bit-width to *its original column* in cost_tensor
#     bit2col = {b: i for i, b in enumerate(bitwidths)}  # <- original order!
#
#     # Re-order the MSE matrix to descending bits
#     mse = cost_tensor[:, [bit2col[b] for b in bits_desc]]
#
#     # ── 2.  Build a per-channel λ-schedule (greedy, allows skips) ────────────────
#     max_steps = K - 1                              # each channel needs ≤ K-1 drops
#     lam_steps   = torch.full((max_steps, N), float("inf"), device=DEVICE)
#     to_bit_step = torch.full((max_steps, N), bits_desc[0],  # dummy init
#                              dtype=torch.int16, device=DEVICE)
#
#     # Current level index (0 == highest bit) for every channel
#     cur_idx2 = torch.zeros(N, dtype=torch.long, device=DEVICE)
#
#     bits_tensor2 = torch.tensor(bits_desc, dtype=torch.float32, device=DEVICE)
#
#     for step2 in range(max_steps):
#         # ΔMSE / Δbits for *all* possible downward jumps from the current level
#         cur_mse2  = mse[rows, cur_idx2]                              # [N]
#         cur_bits2 = bits_tensor2[cur_idx2]                            # [N]
#
#         # Broadcast to [N,K]: each element is Δmse / Δbits or inf (invalid)
#         diff_mse2  = mse2 - cur_mse2.unsqueeze(1)                     # [N,K]
#         diff_bits2 = cur_bits2.unsqueeze(1) - bits_tensor            # [N,K]
#         lam_cand2  = torch.where(diff_bits2 > 0,
#                                 diff_mse2 / diff_bits2,
#                                 torch.full_like(diff_mse2, float("inf")))
#
#         # Best next transition per channel
#         lam_min2, lo_idx2 = lam_cand2.min(dim=1)                      # [N]
#         lam_steps2[step]   = lam_min2
#         to_bit_step2[step] = bits_tensor[lo_idx].to(torch.int16)
#
#         cur_idx2 = lo_idx2                                          # descend
#
#         # Early exit: everyone reached the lowest level
#         if (cur_idx2 == K - 1).all():
#             lam_steps2   = lam_steps2[:step + 1]
#             to_bit_step2 = to_bit_step2[:step + 1]
#             max_steps2   = step2 + 1
#             break
#
#     # ── 3.  Unique λ grid (plus the baseline “all-high” config) ─────────────────
#     lam_unique = torch.unique(lam_steps2[lam_steps.isfinite()])
#     lam_unique, _ = lam_unique.sort()
#
#     if (max_candidates is not None) and (lam_unique.numel() + 1 > max_candidates):
#         idxs = torch.linspace(0, lam_unique.numel() - 1,
#                               steps=max_candidates - 1,
#                               dtype=torch.long, device=lam_unique.device)
#         lam_unique = lam_unique[idxs]
#
#     L = lam_unique.numel()                         # λ grid size (≠ steps!)
#
#     # ── 4.  Build candidate bit-maps in parallel ────────────────────────────────
#     cfg_bits = torch.full((L + 1, N), bits_desc[0],
#                           dtype=torch.int16, device=DEVICE)        # row-0: all-high
#
#     for s in range(max_steps):                    # later steps overwrite earlier
#         mask = lam_unique[:, None] >= lam_steps[s].unsqueeze(0)     # [L,N]
#         cfg_bits[1:] = torch.where(mask,
#                                    to_bit_step[s].unsqueeze(0).expand_as(mask),
#                                    cfg_bits[1:])
#
#     # ── 5.  Convert to framework dictionaries ───────────────────────────────────
#     deltas_tensor = torch.stack(list(deltas_desc), dim=1)           # [N,K]
#     zps_tensor    = torch.stack(list(zps_desc),   dim=1)
#
#     results: List[Dict[str, Union["LayerCompressionConfig", float]]] = [
#         record_config(
#             cost_tensor,
#             deltas_tensor,
#             zps_tensor,
#             rows,
#             bit2idx,
#             cfg_bits[i].tolist(),
#             in_channels,
#         )
#         for i in range(L + 1)
#     ]
#     # largest → smallest (size) order
#     results.reverse()
#     if add_uniform_candidates:
#         missing_uniform_configs = generate_missing_uniform_configs(
#             N, results,
#             cost_tensor, deltas_tensor, zps_tensor,
#             bits_desc, rows, bit2idx, in_channels
#         )
#         for m_cfg in missing_uniform_configs:
#             bisect.insort_left(results, m_cfg, key=lambda d: d[SIZE])
#
#
#     return results