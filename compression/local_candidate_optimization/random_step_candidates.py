from random import random

from typing import Union, Dict, List

import torch

from compression.local_candidate_optimization.search_utils import record_config, generate_missing_uniform_configs


def random_mixed_precision_candidates(
    cost_tensor: torch.Tensor,
    bitwidths: List[int],
    deltas: List[torch.Tensor],
    zero_points: List[torch.Tensor],
    in_channels: int,
    seed: int = None
) -> List[Dict[str, Union['LayerCompressionConfig', float]]]:
    """
    Randomized descent from all-high to all-low precision.
    At each step, randomly drop one channel's bitwidth by one level.

    Args:
        cost_tensor (Tensor[N, M]): per-channel cost for each bitwidth.
        bitwidths (List[int]): M available bitwidths (will be sorted).
        deltas (List[Tensor[N]]): per-bitwidth list of channel deltas.
        zero_points (List[Tensor[N]]): per-bitwidth list of channel zero_points.
        in_channels (int): multiplier for total size.
        seed (int, optional): random seed for reproducibility.

    Returns:
        List of dicts with keys:
          - LAYER_COMPRESSION_CONFIG: LayerCompressionConfig instance
          - SIZE: float total size
          - MSE: float total mse
    """
    N, M = cost_tensor.shape
    # sort bitwidths together with deltas and zero_points
    sorted_items = sorted(zip(bitwidths, deltas, zero_points), key=lambda x: x[0])
    bits, deltas_sorted, zps_sorted = map(list, zip(*sorted_items))
    bit2idx = {b: i for i, b in enumerate(bits)}

    # stack for easy indexing
    deltas_tensor = torch.stack(deltas_sorted, dim=1)  # [N, M]
    zps_tensor    = torch.stack(zps_sorted, dim=1)    # [N, M]

    # start all-high precision
    current_bits = [bits[-1]] * N
    results: List[Dict[str, Union['LayerCompressionConfig', float]]] = []
    channel_idx = torch.arange(N)

    # record initial config
    results.append(
        record_config(cost_tensor, deltas_tensor, zps_tensor, channel_idx, bit2idx, current_bits, in_channels))

    # random descent until all-low precision
    while True:
        # list all possible single-channel drops
        moves = []
        for i, b in enumerate(current_bits):
            j = bit2idx[b]
            if j > 0:
                moves.append((i, bits[j-1]))
        if not moves:
            break

        # pick a random drop
        channel, new_b = random.choice(moves)
        current_bits[channel] = new_b

        # record config
        results.append(
            record_config(cost_tensor, deltas_tensor, zps_tensor, channel_idx, bit2idx, current_bits, in_channels))

    # ensure uniform-bit candidates are present
    results.extend(
        generate_missing_uniform_configs(N, results, cost_tensor, deltas_tensor, zps_tensor, bits, bit2idx, in_channels))
    results.reverse()
    return results