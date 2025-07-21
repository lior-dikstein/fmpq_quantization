from __future__ import annotations
import itertools
import numpy as np

from typing import List, Tuple

import torch

from compression.local_candidate_optimization.greedy_candidate_search import greedy_mixed_precision_candidates, \
    lambda_frontier_candidates
from constants import LAYER_COMPRESSION_CONFIG, SIZE, MSE, DEVICE

# ------------------------------------------------------------------------- #
# Core helpers                                                              #
# ------------------------------------------------------------------------- #
def generate_cost_tensors(N: int, bits=(2, 4, 8), seed: int = 42):
    """
    Create per-channel size & MSE cost tensors.

    Args:
        N:    number of channels.
        bits: allowed bit-widths.
        seed: RNG seed.

    Returns:
        size_cost  – (N,|bits|) size per channel/bit.
        mse_cost   – (N,|bits|) mse  per channel/bit.
    """
    rng = np.random.default_rng(seed)
    size_cost = np.tile(bits, (N, 1)).astype(float)

    base = 1.0 / np.asarray(bits, float)           # inverse relation
    noise = rng.uniform(0.0, 0.05, (N, M))
    mse_cost = base + noise
    return size_cost, mse_cost


def total_costs(size_cost, mse_cost, assign):
    """Total size & mse for one assignment vector."""
    rows = np.arange(size_cost.shape[0])
    return (
        float(size_cost[rows, assign].sum()),
        float(mse_cost[rows, assign].sum()),
    )


def enumerate_all_candidates(cost_tensor: torch.Tensor, size_tensor: torch.Tensor, bitwidths: List[int], ) -> List[
    Tuple[List[int], float, float]]:
    """
    Exhaustively enumerate every bit-width assignment and return
    (bits, total_distortion, total_size) for each one.
    """
    N, M = cost_tensor.shape
    assignments = []
    for combo in itertools.product(range(M), repeat=N):  # every index vector
        bits = [bitwidths[j] for j in combo]
        d = float(cost_tensor[range(N), combo].sum())
        s = float(size_tensor[range(N), combo].sum())
        assignments.append((bits, d, s))
    return assignments

def pareto_mask(size_arr: np.ndarray, mse_arr: np.ndarray) -> np.ndarray:
    """
    Strict Pareto front (min-size, min-mse).
    Keeps only the point with the lowest MSE for every distinct size,
    then discards any point dominated by a cheaper one.
    """
    best_per_size = {}
    best_idx      = {}
    for idx, (s, e) in enumerate(zip(size_arr, mse_arr)):
        if (s not in best_per_size) or (e < best_per_size[s]):
            best_per_size[s] = e
            best_idx[s]      = idx

    keep = np.zeros_like(size_arr, bool)
    best_mse_so_far = np.inf
    for s in sorted(best_idx):                       # ascending size
        i = best_idx[s]
        if mse_arr[i] < best_mse_so_far:
            keep[i] = True
            best_mse_so_far = mse_arr[i]
    return keep


def print_real_pareto(
    combos: np.ndarray,
    mask: np.ndarray,
    bitwidths: List[int],
    total_size: np.ndarray,
    total_mse: np.ndarray,
) -> None:
    """Dump the true Pareto set, sorted by total size ascending."""
    print("\nReal Pareto set (bits, size, mse):")
    order = np.argsort(total_size[mask])
    for idx in np.where(mask)[0][order]:
        bits = [bitwidths[j] for j in combos[idx]]
        print(f"  {bits}  | {total_size[idx]:.2f}  | {total_mse[idx]:.2f}")
    print("-" * 60)


def validate_results(pareto):
    for p in pareto:
        assert torch.sum(torch.abs(
            torch.tensor(p[LAYER_COMPRESSION_CONFIG].bit_width_quantization).to(DEVICE).view(-1, 1) - p[
                LAYER_COMPRESSION_CONFIG].delta.to(DEVICE))) == 0
        assert torch.sum(torch.abs(
            torch.tensor(p[LAYER_COMPRESSION_CONFIG].bit_width_quantization).to(DEVICE).view(-1, 1) - p[
                LAYER_COMPRESSION_CONFIG].zero_point.to(DEVICE))) == 0
    print('Pareto results validated')


def _to_key(size: float, mse: float, ndigits: int = 9) -> Tuple[float, float]:
    """Round pair for stable set membership."""
    return (round(size, ndigits), round(mse, ndigits))


def _candidate_bits(c) -> List[int]:
    """Extract bit-width list from algorithm candidate object."""
    return list(c[LAYER_COMPRESSION_CONFIG].bit_width_quantization)


# def analyse_algo(
#     name: str,
#     cand_list,
#     bitwidths: List[int],
#     size_cost: np.ndarray,
#     mse_cost: np.ndarray,
#     pareto_set: set[Tuple[float, float]],
# ) -> None:
#     """Print counts for a single algorithm."""
#     if cand_list is None:
#         return
#
#     N = size_cost.shape[0]
#     rows = range(N)
#
#     in_pareto = 0
#     for c in cand_list:
#         bits = _candidate_bits(c)
#         idxs = [bitwidths.index(int(b)) for b in bits]
#
#         size = sum(size_cost[i, idx] for i, idx in zip(rows, idxs))
#         mse = sum(mse_cost[i, idx] for i, idx in zip(rows, idxs))
#
#         if _to_key(size, mse) in pareto_set:
#             in_pareto += 1
#
#     total = len(cand_list)
#     print(
#         f"{name:<25} | generated: {total:>5d} | "
#         f"on Pareto: {in_pareto:>5d} | "
#         f"off Pareto: {total - in_pareto:>5d}"
#     )


def analyse_algo(
    name: str,
    cand_list,
    bitwidths: List[int],
    size_cost: np.ndarray,
    mse_cost: np.ndarray,
    pareto_set: set[Tuple[float, float]],
) -> None:
    """
    Print every candidate produced by an algorithm, flagging Pareto status,
    and summarise counts.
    """
    if cand_list is None:
        return

    N = size_cost.shape[0]
    rows = range(N)
    total = pareto_hits = 0

    print(f"\n{name} candidates (bits, size, mse, status):")
    for c in cand_list:
        bits = [int(b) for b in c[LAYER_COMPRESSION_CONFIG].bit_width_quantization]
        idxs = [bitwidths.index(b) for b in bits]
        size = sum(size_cost[i, j] for i, j in zip(rows, idxs))
        mse  = sum(mse_cost[i, j]  for i, j in zip(rows, idxs))
        key  = (_to_key(size, mse))
        status = "Pareto" if key in pareto_set else "¬Pareto"
        if status == "Pareto":
            pareto_hits += 1
        total += 1
        print(f"  {bits}  | {size:.2f}  | {mse:.2f}  | {status}")

    print(
        f"{name:<25} | generated: {total:5d} "
        f"| on Pareto: {pareto_hits:5d} | off Pareto: {total - pareto_hits:5d}"
    )


def print_all_candidates(
    combos: np.ndarray,
    bitwidths: List[int],
    total_size: np.ndarray,
    total_mse: np.ndarray,
    pareto_set: set[Tuple[float, float]],
) -> None:
    """
    List every assignment produced by exhaustive enumeration, sorted
    by total size → mse, and label each as Pareto / ¬Pareto.

    Parameters
    ----------
    combos      : (K, N) index array returned by itertools.product
    bitwidths   : list of available bit-widths (e.g. [2, 4, 8])
    total_size  : (K,) vector of summed sizes
    total_mse   : (K,) vector of summed MSEs
    pareto_set  : {(size, mse)} keys of true Pareto points
    """
    print("\nAll candidates (bits, size, mse, status):")
    order = np.lexsort([total_mse, total_size])      # size↑ then mse↑
    pareto_hits = 0
    for idx in order:
        bits   = [bitwidths[j] for j in combos[idx]]
        s, e   = total_size[idx], total_mse[idx]
        status = "Pareto" if (_to_key(s, e) in pareto_set) else "¬Pareto"
        if status == "Pareto":
            pareto_hits += 1
        print(f"  {bits}  | {s:.2f}  | {e:.2f}  | {status}")
    print(f"\nTotal: {len(order)}  |  Pareto: {pareto_hits}  |  Non-Pareto: {len(order)-pareto_hits}")
    print("-" * 60)


def print_delta_ratios_8xx(
    cost_tensor: torch.Tensor,
    bitwidths: List[int],
) -> None:
    """
    For each channel prints: 8→4 and 8→2  (ΔMSE, Δsize, ΔMSE/Δsize).

    Example output
    --------------
    Ch 0: 8→4 ΔMSE= 9.50  Δsize=4  ratio=2.38 | 8→2 ΔMSE=47.00  Δsize=6  ratio=7.83
    """
    idx8 = bitwidths.index(8)
    idx4 = bitwidths.index(4)
    idx2 = bitwidths.index(2)

    print("\nΔMSE / Δsize per channel (two-step vs one-step):")
    for ch in range(cost_tensor.shape[0]):
        dm_84 = float(cost_tensor[ch, idx4] - cost_tensor[ch, idx8])   # 8→4
        ds_84 = 8 - 4
        r_84  = dm_84 / ds_84

        dm_42 = float(cost_tensor[ch, idx2] - cost_tensor[ch, idx4])   # 8→2
        ds_42 = 4 - 2
        r_42  = dm_42 / ds_42


        dm_82 = float(cost_tensor[ch, idx2] - cost_tensor[ch, idx8])   # 8→2
        ds_82 = 8 - 2
        r_82  = dm_82 / ds_82

        print(
            f"Ch {ch}: "
            f"8→4 ΔMSE={dm_84:6.2f}  Δsize={ds_84}  ratio={r_84:6.2f} | "
            f"4→2 ΔMSE={dm_42:6.2f}  Δsize={ds_42}  ratio={r_42:6.2f} | "
            f"8→2 ΔMSE={dm_82:6.2f}  Δsize={ds_82}  ratio={r_82:6.2f}"
        )
    print("-" * 60)


if __name__ == "__main__":
    from helpers.timers import SegmentTimer

    timer = SegmentTimer()
    torch.manual_seed(0)
    N, bitwidths = 1500,[2, 4, 8]
    M = len(bitwidths)
    max_bit = max(bitwidths)

    # Generate distortion so that lower bit ⇒ higher MSE ------------------- #
    size_cost, mse_cost = generate_cost_tensors(N, bitwidths)

    # mse_cost = np.array([
    #     [50, 20, 1],
    #     [22, 8, 1],
    #     [48, 45, 1],
    # ])

    # print_delta_ratios_8xx(torch.tensor(mse_cost), bitwidths)

    # Exhaustive enumeration (ground truth)
    # combos = np.array(list(itertools.product(range(len(bitwidths)), repeat=N)), dtype=int)
    # rows = np.repeat(np.arange(N)[None, :], combos.shape[0], axis=0)
    # total_size = size_cost[rows, combos].sum(axis=1)
    # total_mse = mse_cost[rows, combos].sum(axis=1)
    # mask = pareto_mask(total_size, total_mse)
    # pareto_set = {_to_key(total_size[i], total_mse[i]) for i in np.where(mask)[0]}
    # print_real_pareto(combos, mask, bitwidths, total_size, total_mse)

    print(f'{N} channels, {M} bit options')
    # print(f"Total candidates (exhaustive): {total_size.size}")
    # print(f"Real Pareto points           : {sum(mask)}\n")

    # print_all_candidates(combos, bitwidths, total_size, total_mse, pareto_set)


    base_distortion = torch.rand(N, 1).to(DEVICE)          # per-channel baseline
    scale = torch.tensor([max_bit / b for b in bitwidths]).to(DEVICE).float()  # 8→1, 4→2, 2→4

    cost = base_distortion * scale              # shape (N, M)    size = torch.tensor(bitwidths, dtype=torch.float32).expand(N, M)
    deltas = [b* torch.ones_like(base_distortion).to(DEVICE) for b in bitwidths]
    zero_points = [b * torch.ones_like(base_distortion).to(DEVICE) for b in bitwidths]
    size = torch.tensor(bitwidths, dtype=torch.float32).to(DEVICE).expand(N, M)

    # # Len pareto 1200, 400 channels, 3 bit options - 27.7273s
    # boa = BOAStarQuantizer(mse_cost, bitwidths, deltas, zero_points, out_channels=1, max_solutions=None)
    # pareto = boa.search()
    # timer.segment('BOAStarQuantizer')
    # validate_results(pareto)
    #
    # eboa = EpsilonBOAStarQuantizer(mse_cost, bitwidths, deltas, zero_points, epsilon=0.005, out_channels=1, max_solutions=None)
    # pareto2 = eboa.search()
    # timer.segment('EpsilonBOAStarQuantizer')
    # validate_results(pareto2)

    greedy = greedy_mixed_precision_candidates(mse_cost, bitwidths, deltas, zero_points, out_channels=1)
    timer.segment('greedy_mixed_precision_candidates')
    print(f'Greedy 1 {len(greedy)}')
    validate_results(greedy)
    # analyse_algo("Greedy candidates", greedy, bitwidths, size_cost, mse_cost, pareto_set)

    # analyse_algo("Greedy2 candidates", greedy2, bitwidths, size_cost, mse_cost, pareto_set)


    greedy3 = lambda_frontier_candidates(torch.tensor(mse_cost).to(DEVICE), bitwidths, deltas, zero_points, out_channels=1)
    timer.segment('greedy_mixed_precision_candidates3')
    validate_results(greedy3)
    print(f'Greedy 3 {len(greedy3)}')

    # analyse_algo("Greedy3 candidates", greedy3, bitwidths, size_cost, mse_cost, pareto_set)



    # Exhaustive enumeration ------------------------------------------------ #
    all_solutions = enumerate_all_candidates(cost, size, bitwidths)
