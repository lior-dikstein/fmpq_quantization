from collections import defaultdict

import torch
from typing import Dict, List, Optional, Union, Tuple

from compression.local_candidate_optimization.greedy_candidate_search import lambda_frontier_candidates, \
    lambda_frontier_candidates_grouped
from compression.local_candidate_optimization.search_utils import record_config, is_simd_valid
from constants import DEVICE, LAYER_COMPRESSION_CONFIG, SIZE, MSE

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS (reuse the same names used elsewhere in your code-base)
# ────────────────────────────────────────────────────────────────────────────────
DISTORTION_KEY_CANDIDATES  = ("D", "distortion", "mse") # allowed distortion keys
SIZE_KEY_CANDIDATES        = ("S", "size", "bits")      # allowed size-of-weights keys


# ────────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def _random_simd_groups(n: int, simd: int, g: torch.Generator) -> List[torch.Tensor]:
    """
    Split `n` indices into random groups of length `simd` (remainder allowed).
    """
    perm = torch.randperm(n, generator=g, device=DEVICE)
    return [perm[i : i + simd] for i in range(0, n, simd)]


def _extract_ds(cfg: Dict[str, Union[float, int]]) -> Tuple[float, int]:
    """
    Robustly extract (distortion, size) from the candidate dictionary.
    """
    return cfg[MSE], cfg[SIZE], cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization


def _pareto_frontier(candidates: List[Dict[str, Union[float, int]]]
                     ) -> List[Dict[str, Union[float, int]]]:
    """
    Keep **one** config per *total-size* (S) – the one with the minimal
    distortion (D).  Return them in descending-size order.
    """
    size2best = {}  # S → cfg with minimal D
    for cfg in candidates:
        d, s, bw = _extract_ds(cfg)
        if (s not in size2best) or (d < _extract_ds(size2best[s])[0]):
            size2best[s] = cfg

    # descending-size order (⩓ to match the rest of the pipeline)
    return [size2best[s] for s in sorted(size2best.keys(), reverse=True)]


def _local_refine(
    bits: torch.Tensor,                     # [N]  int16 (on DEVICE)
    cost: torch.Tensor,                     # [N, |M|] (on DEVICE)
    bit2idx: Dict[int, int],
    gen: torch.Generator,
    max_steps: int = 2500,
) -> torch.Tensor:
    """
    Greedy first-improve pair swaps.  O(max_steps).
    *SIMD counts stay unchanged* because we only swap bit-labels.
    """
    N          = bits.numel()
    bit_tensor = torch.tensor(sorted(bit2idx), dtype=torch.int16, device=bits.device)
    idxs       = torch.arange(N, device=bits.device)

    for _ in range(max_steps):
        # pick two *different bit* channels
        i, j = torch.randint(0, N, (2,), generator=gen, device=bits.device)
        if bits[i] == bits[j]:
            continue

        bi, bj = bits[i].item(), bits[j].item()
        ci_old = cost[i, bit2idx[bi]]
        cj_old = cost[j, bit2idx[bj]]
        ci_new = cost[i, bit2idx[bj]]
        cj_new = cost[j, bit2idx[bi]]

        if ci_new + cj_new < ci_old + cj_old:        # strict improvement
            bits[i], bits[j] = bits[j], bits[i]      # swap in-place
    return bits


# ────────────────────────────────────────────────────────────────────────────────
# Smarter prune – v2 (no “one-bit-up”, keep TOP_K_PER_SIZE per size)
# ────────────────────────────────────────────────────────────────────────────────
def _smarter_prune(
        candidates: List[Dict[str, Union["LayerCompressionConfig", float]]],
        top_k_per_size: int,
) -> List[Dict[str, Union["LayerCompressionConfig", float]]]:
    """
    Bucket by size → keep TOP_K_PER_SIZE best-MSE per bucket → sort size↓, MSE↑.
    """
    buckets: Dict[int, List[Tuple[float, Dict]]] = defaultdict(list)

    # 1. gather + trim buckets
    for cfg in candidates:
        d, s, bit_widths = _extract_ds(cfg)
        b    = buckets[s]
        b.append((d, cfg))
        b.sort(key=lambda x: x[0])
        if len(b) > top_k_per_size:
            b.pop()

    # 2. flatten & sort
    pool = [cfg for lst in buckets.values() for _, cfg in lst]
    pool.sort(key=lambda c: (_extract_ds(c)[1], _extract_ds(c)[0]), reverse=True)
    return pool


# ────────────────────────────────────────────────────────────────────────────────
# MAIN ROUTINE
# ────────────────────────────────────────────────────────────────────────────────
def lambda_frontier_candidates_grouped_random(
        cost_tensor: torch.Tensor,                # [N, |M|]
        bitwidths:   List[int],
        deltas:      List[torch.Tensor],
        zero_points: List[torch.Tensor],
        in_channels: int,
        simd: int = 1,
        max_candidates: Optional[int] = None,
        iterations: int = 500,
        add_uniform_candidates: bool = True,
        random_state: Optional[int] = None,
        entropy: Optional[int] = None,
        hessian: Optional[int] = None,
) -> List[Dict[str, Union["LayerCompressionConfig", float]]]:
    """
    λ-frontier candidate search with *random* SIMD groupings and Pareto pruning.

    For `iterations` rounds:
      1. Randomly partition the N channels into groups of length `simd`
         (final group may be shorter when N % simd ≠ 0).
      2. Run `lambda_frontier_candidates` on the grouped cost matrix.
      3. Expand each group-level config back to per-channel bit-widths and
         collect the resulting candidates.

    After all rounds, compute the (D, S) Pareto frontier of the aggregated
    candidate pool and return this set (largest-size → smallest-size order).

    Parameters
    ----------
    cost_tensor
        Distortion/MSE per channel and bit-width.
    bitwidths, deltas, zero_points, in_channels
        Same semantics as in the base routine.
    simd
        SIMD group size.  When simd == 1, this degenerates to the plain search.
    iterations
        Number of random regrouping rounds to perform.
    max_candidates, add_uniform_candidates
        Passed through to the inner search.
    random_state
        Optional seed for reproducible random groupings.

    Returns
    -------
    List[Dict] – Pareto-optimal candidates in size-descending order.
    """

    # Fallback to the standard (ungrouped) search.
    N = cost_tensor.size(0)

    base_candidates =  lambda_frontier_candidates(
        cost_tensor=cost_tensor,
        bitwidths=bitwidths,
        deltas=deltas,
        zero_points=zero_points,
        in_channels=in_channels,
        max_candidates=max_candidates,
        add_uniform_candidates=add_uniform_candidates,
    )
    if simd == 1:
        return base_candidates

    all_candidates = [
        cfg for cfg in base_candidates
        if is_simd_valid(cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization, simd)
    ]

    if hessian is not None:
        all_candidates.extend(lambda_frontier_candidates_grouped(cost_tensor, hessian, bitwidths, deltas, zero_points, in_channels, simd, max_candidates))
    if entropy is not None:
        all_candidates.extend(lambda_frontier_candidates_grouped(cost_tensor, entropy, bitwidths, deltas, zero_points, in_channels, simd, max_candidates))

    bits_desc, deltas_desc, zps_desc = zip(
        *sorted(zip(bitwidths, deltas, zero_points), key=lambda x: -x[0])
    )
    bits_desc = list(bits_desc)
    bit2idx = {b: i for i, b in enumerate(bits_desc)}
    bit2col = {b: i for i, b in enumerate(bitwidths)}

    cost_desc = cost_tensor[:, [bit2col[b] for b in bits_desc]].to(DEVICE)
    deltas_desc = torch.stack(deltas_desc, dim=1).to(DEVICE)
    zps_desc = torch.stack(zps_desc, dim=1).to(DEVICE)

    rows = torch.arange(cost_tensor.size(0), device=DEVICE)
    gen = torch.Generator(device=DEVICE)
    if random_state is not None:
        gen.manual_seed(random_state)


    for _ in range(iterations):
        # ── 1. random grouping ────────────────────────────────────────────────
        groups = _random_simd_groups(N, simd, gen)         # list[Tensor]

        # Aggregate cost per group.
        group_cost = torch.stack(
            [cost_tensor[idx].sum(dim=0) for idx in groups], dim=0
        )                                                  # [G, |M|]

        # ── 2. λ-frontier on groups ───────────────────────────────────────────
        group_results = lambda_frontier_candidates(
            cost_tensor=group_cost,
            bitwidths=bitwidths,
            deltas=deltas,
            zero_points=zero_points,
            in_channels=in_channels,
            max_candidates=max_candidates,
            add_uniform_candidates=add_uniform_candidates,
        )


        for cfg in group_results:
            group_bits = cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization  # len == G
            bits_per_channel = -1*torch.ones(N, dtype=torch.int16, device=DEVICE)
            for g_idx, ch_idx in enumerate(groups):
                bits_per_channel[ch_idx] = group_bits[g_idx]

            all_candidates.append(
                record_config(
                    cost_desc,
                    deltas_desc,
                    zps_desc,
                    bit2idx,
                    bits_per_channel.tolist(),
                    in_channels,
                )
            )

        all_candidates_sorted = sorted(all_candidates, key=lambda x: x[SIZE])
        # ── 4. Pareto pruning ────────────────────────────────────────────────────
        all_candidates = _smarter_prune(all_candidates_sorted, simd)
        # all_candidates = _pareto_frontier(all_candidates)
    return list(reversed(all_candidates))
