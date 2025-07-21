from bisect import insort_left

import bisect

import math
import numpy as np
from k_means_constrained import KMeansConstrained
from typing import List, Union, Dict, Optional
import torch
from compression.configs.layer_comrpression_config import LayerCompressionConfig
from compression.local_candidate_optimization.search_utils import record_config, generate_missing_uniform_configs
from constants import DEVICE, LAYER_COMPRESSION_CONFIG, SIZE
from sklearn.cluster import KMeans


def greedy_mixed_precision_candidates(
    cost_tensor: torch.Tensor,
    bitwidths: List[int],
    deltas: List[torch.Tensor],
    zero_points: List[torch.Tensor],
    in_channels: int,
    max_candidates: Optional[int] = None,
    add_uniform_candidates: bool = False,
) -> List[Dict[str, Union['LayerCompressionConfig', float]]]:
    """
    Greedy marginal‐cost descent with upper bound on recorded configs.
    Records LayerCompressionConfig, size, and mse for up to max_candidates.
    """
    N, M = cost_tensor.shape

    # sort bitwidths, deltas, zero_points together
    sorted_items = sorted(zip(bitwidths, deltas, zero_points), key=lambda x: x[0])
    bits, deltas_sorted, zps_sorted = map(list, zip(*sorted_items))
    bit2idx = {b: i for i, b in enumerate(bits)}

    # stack for indexing
    deltas_tensor = torch.stack(deltas_sorted, dim=1)  # [N, M]
    zps_tensor    = torch.stack(zps_sorted, dim=1)    # [N, M]

    # total descent steps (excluding initial)
    total_moves = N * (len(bits) - 1)
    total_candidates = total_moves + 1

    # decide whether to record every step or sample
    if max_candidates is None or total_candidates <= max_candidates:
        record_all = True
    else:
        record_all = False
        # precompute which drop_counts to record at
        positions = {
            int(round(i * total_moves / (max_candidates - 1)))
            for i in range(max_candidates)
        }

    current_bits = [bits[-1]] * N
    results = []
    drop_count = 0
    channel_idx = torch.arange(N)

    # record initial config
    results.append(
        record_config(cost_tensor, deltas_tensor, zps_tensor, channel_idx, bit2idx, current_bits, in_channels))

    while True:
        # collect possible drops
        moves = []
        for i, b in enumerate(current_bits):
            j = bit2idx[b]
            if j > 0:
                j_low = j - 1
                b_low = bits[j_low]
                dc = float(cost_tensor[i, j_low] - cost_tensor[i, j])
                db = b - b_low
                moves.append((dc/db, i, b_low))
        if not moves:
            break

        # apply the cheapest drop
        _, channel, new_b = min(moves, key=lambda x: x[0])
        current_bits[channel] = new_b
        drop_count += 1

        # record if sampling or if we're under the limit
        if record_all or drop_count in positions:
            results.append(
                record_config(cost_tensor, deltas_tensor, zps_tensor, channel_idx, bit2idx, current_bits, in_channels))

    if add_uniform_candidates:
        # ensure uniform-bit candidates are present
        results.extend(
            generate_missing_uniform_configs(N, results, cost_tensor, deltas_tensor, zps_tensor, bits, channel_idx, bit2idx, in_channels))
    results.reverse()
    return results



def _build_groups_kmeans(
    cost_tensor: torch.Tensor,
    simd: int,
    random_state: int = 0,
) -> List[torch.Tensor]:
    """
    Partition channels with k-means on the MSE matrix (N × |M|).

    * k = ceil(N / simd) clusters.
    * Any cluster larger than `simd` is split into fixed-size chunks.
    """
    N, _ = cost_tensor.shape
    k = math.ceil(N / simd)

    # sklearn works on CPU & float32
    labels = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init="auto",
    ).fit_predict(cost_tensor.cpu().float().numpy())

    # build Torch index tensors per cluster, then enforce the ≤ simd rule
    groups: List[torch.Tensor] = []
    for c in range(k):
        idx = torch.nonzero(torch.tensor(labels) == c, as_tuple=False).squeeze(1)
        for i in range(0, idx.numel(), simd):
            groups.append(idx[i : i + simd].to(cost_tensor.device))
    return groups


def _kmeans(x: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    N = x.size(0)
    device = x.device
    centres = x[torch.randint(0, N, (1,), device=device)]
    for _ in range(1, k):
        dist2 = torch.cdist(x, centres).min(dim=1).values**2
        probs  = dist2 / dist2.sum()
        centres = torch.cat([centres, x[torch.multinomial(probs, 1)]], dim=0)

    for _ in range(iters):
        labels = torch.cdist(x, centres).argmin(dim=1)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centres[j] = x[mask].mean(dim=0)
    return labels                    # [N]

# ---------------------------------------------------------
# Build *fixed-size* SIMD groups
# ---------------------------------------------------------
def _build_fixed_simd_groups(
    cost_tensor: torch.Tensor,
    simd: int
) -> List[torch.Tensor]:
    """
    1. k-means on cost curves    → rough similarity order
    2. concatenate all channel indices ordered by cluster id
    3. cut the list into blocks of exactly `simd` channels
       (last block may be < simd → remainder group)
    """
    N, _ = cost_tensor.shape
    k = math.ceil(N / simd)                         # target clusters
    labels = _kmeans(cost_tensor.float(), k)        # [N]

    # concatenation preserving cluster order
    ordered_idx = torch.cat(
        [(labels == lab).nonzero(as_tuple=False).squeeze(1)
         for lab in labels.unique(sorted=True)],
        dim=0,
    )                                               # [N]

    groups = [
        ordered_idx[i : i + simd]
        for i in range(0, N, simd)
    ]                                               # last one may be shorter
    return groups



def lambda_frontier_candidates(
    cost_tensor: torch.Tensor,
    bitwidths: List[int],
    deltas: List[torch.Tensor],
    zero_points: List[torch.Tensor],
    in_channels: int,
    max_candidates: Optional[int] = None,
    add_uniform_candidates: bool = True,
) -> List[Dict[str, Union["LayerCompressionConfig", float]]]:
    """
    Generate mixed-precision candidates via λ-frontier search
    (largest→smallest bit traversal with greedy λ = ΔMSE/Δbits steps).

    All tensors can reside on CPU or CUDA — device is taken from `cost_tensor`.
    The SIMD divisibility constraint (if `simd` is not None) is enforced
    on the returned set.
    """
    # ---- 0.Basic validations -------------------------------------------------
    assert len(set(bitwidths)) == len(bitwidths), "duplicate bit-widths"
    N = cost_tensor.size(0)

    # ---- 1.Sort bit-widths from high→low & build re-ordered views ------------
    bits_desc, deltas_desc, zps_desc = zip(
        *sorted(zip(bitwidths, deltas, zero_points), key=lambda x: -x[0])
    )
    bits_desc = list(bits_desc)                        # e.g. [8,4,2]
    K = len(bits_desc)
    bit2idx = {b: i for i, b in enumerate(bits_desc)}  # indices in *desc* order
    bit2col = {b: i for i, b in enumerate(bitwidths)}  # original columns

    cost_desc   = cost_tensor[:, [bit2col[b] for b in bits_desc]].to(DEVICE)
    deltas_desc = torch.stack(deltas_desc, dim=1).to(DEVICE)
    zps_desc    = torch.stack(zps_desc,   dim=1).to(DEVICE)

    # Consistency checks
    # assert cost_desc.shape == deltas_desc.shape[0:2] == zps_desc.shape[0:2] == (N, K)

    rows = torch.arange(N, device=DEVICE)
    bits_tensor = torch.tensor(bits_desc, dtype=torch.float32, device=DEVICE)

    # ---- 2.Per-channel greedy λ schedule ------------------------------------
    max_steps = K - 1
    lam_steps   = torch.full((max_steps, N), float("inf"), device=DEVICE)
    to_bit_step = torch.full((max_steps, N), bits_desc[0],
                             dtype=torch.int16, device=DEVICE)

    cur_idx = torch.zeros(N, dtype=torch.long, device=DEVICE)
    for step in range(max_steps):
        cur_mse  = cost_desc[rows, cur_idx]
        cur_bits = bits_tensor[cur_idx]

        diff_mse  = cost_desc - cur_mse.unsqueeze(1)
        diff_bits = cur_bits.unsqueeze(1) - bits_tensor
        lam_cand  = torch.where(diff_bits > 0,
                                diff_mse / diff_bits,
                                torch.full_like(diff_mse, float("inf")))

        lam_min, lo_idx = lam_cand.min(dim=1)
        lam_steps[step]   = lam_min
        to_bit_step[step] = bits_tensor[lo_idx].to(torch.int16)
        cur_idx = lo_idx

        if (cur_idx == K - 1).all():
            lam_steps   = lam_steps[: step + 1]
            to_bit_step = to_bit_step[: step + 1]
            break

    # ---- 3.Unique λ grid -----------------------------------------------------
    lam_unique = torch.unique(lam_steps[lam_steps.isfinite()]).sort().values
    if max_candidates and lam_unique.numel() + 1 > max_candidates:
        idxs = torch.linspace(0, lam_unique.numel() - 1,
                              steps=max_candidates - 1).round().long().unique()
        lam_unique = lam_unique[idxs]

    L = lam_unique.numel()

    # ---- 4.Build candidate bitmaps ------------------------------------------
    cfg_bits = torch.full((L + 1, N), bits_desc[0],
                          dtype=torch.int16, device=DEVICE)  # row-0: all-high

    for s in range(lam_steps.size(0)):                       # later steps overwrite
        mask = lam_unique[:, None] >= lam_steps[s].unsqueeze(0)
        cfg_bits[1:] = torch.where(mask,
                                   to_bit_step[s].unsqueeze(0).expand_as(mask),
                                   cfg_bits[1:])

    # ---- 5.Pack configs ------------------------------------------------------
    results: List[Dict[str, Union["LayerCompressionConfig", float]]] = [
        record_config(
            cost_desc,
            deltas_desc,
            zps_desc,
            bit2idx,
            cfg_bits[i].tolist(),
            in_channels,
        )
        for i in range(L + 1)
    ]
    # largest → smallest (size) order
    results.reverse()
    if add_uniform_candidates:
        missing = generate_missing_uniform_configs(
            N, results, cost_desc, deltas_desc, zps_desc,
            bits_desc, bit2idx, in_channels,
        )
        for cfg in missing:
            insort_left(results, cfg, key=lambda d: d[SIZE])

    return results

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
#     cur_idx = torch.zeros(N, dtype=torch.long, device=DEVICE)
#
#     bits_tensor = torch.tensor(bits_desc, dtype=torch.float32, device=DEVICE)
#
#     for step in range(max_steps):
#         # ΔMSE / Δbits for *all* possible downward jumps from the current level
#         cur_mse  = mse[rows, cur_idx]                              # [N]
#         cur_bits = bits_tensor[cur_idx]                            # [N]
#
#         # Broadcast to [N,K]: each element is Δmse / Δbits or inf (invalid)
#         diff_mse  = mse - cur_mse.unsqueeze(1)                     # [N,K]
#         diff_bits = cur_bits.unsqueeze(1) - bits_tensor            # [N,K]
#         lam_cand  = torch.where(diff_bits > 0,
#                                 diff_mse / diff_bits,
#                                 torch.full_like(diff_mse, float("inf")))
#
#         # Best next transition per channel
#         lam_min, lo_idx = lam_cand.min(dim=1)                      # [N]
#         lam_steps[step]   = lam_min
#         to_bit_step[step] = bits_tensor[lo_idx].to(torch.int16)
#
#         cur_idx = lo_idx                                           # descend
#
#         # Early exit: everyone reached the lowest level
#         if (cur_idx == K - 1).all():
#             lam_steps   = lam_steps[:step + 1]
#             to_bit_step = to_bit_step[:step + 1]
#             max_steps   = step + 1
#             break
#
#     # ── 3.  Unique λ grid (plus the baseline “all-high” config) ─────────────────
#     lam_unique = torch.unique(lam_steps[lam_steps.isfinite()])
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


def _build_groups_by_grouping_param(grouping_param: torch.Tensor, simd: int) -> List[torch.Tensor]:
    """
    Return a list of 1-D LongTensors – each is the index set of one group.
    Groups are contiguous slices of the grouping_param-descending order.
    """
    N = grouping_param.numel()
    sorted_idx = torch.argsort(grouping_param, descending=True)
    return [
        sorted_idx[i : i + simd]
        for i in range(0, N, simd)
    ]


def kmeans_fixed_simd_groups(cost_tensor: torch.Tensor,
                             simd: int,
                             random_state: int = 0):
    """
    Return a list of channel-index tensors.
    All groups have length == simd, except a single remainder (N % simd).
    """
    N, _ = cost_tensor.shape
    k          = math.ceil(N / simd)
    remainder  = N % simd                      # 0 ↔ perfect packing
    size_min   = 1 if remainder else simd      # allow small final group
    size_max   = simd

    # k-means with capacity constraints (CPU, float32)
    km = KMeansConstrained(
        n_clusters=k,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state,
    )
    labels = km.fit_predict(cost_tensor.cpu().float().numpy())

    # turn each cluster label into an index tensor
    groups = [
        torch.as_tensor(np.where(labels == c)[0], dtype=torch.long,
                        device=cost_tensor.device)
        for c in range(k)
        if (labels == c).any()
    ]

    # sanity checks
    assert sum(g.numel() for g in groups) == N
    for g in groups[:-1]:
        assert g.numel() == simd             # all but maybe the last are full

    return groups


def lambda_frontier_candidates_grouped(
        cost_tensor: torch.Tensor,  # [N, |M|]
        grouping_param: torch.Tensor,  # [N]
        bitwidths: List[int],
        deltas: List[torch.Tensor],
        zero_points: List[torch.Tensor],
        in_channels: int,
        simd: int = 1,
        max_candidates: Optional[int] = None,
        add_uniform_candidates: bool = True,
        use_k_means: bool = False,
        random_state: int = 0,
) -> List[Dict[str, Union["LayerCompressionConfig", float]]]:
    """
    λ-frontier search with *SIMD grouping*.

    When `simd > 1` the N channels are first partitioned into
    `⌈N / simd⌉` groups of (at most) `simd` consecutive channels in the
    wieght-descending ordering.  A single bit-width is then assigned to
    every member of a group during the search.

    Parameters
    ----------
    cost_tensor
        Distortion/MSE per channel and bit-width.
    grouping_param
        Per-channel grouping_param used solely for grouping.
    bitwidths, deltas, zero_points, in_channels
        Same as before.
    simd
        Group size.  `simd == 1` ⇒ behaves exactly like the un-grouped search.
    max_candidates, add_uniform_candidates
        Passed through unchanged.

    Returns
    -------
    List[Dict] – identical schema to the original function.
    """
    if simd == 1:
        # No grouping – defer to the plain implementation.
        return lambda_frontier_candidates(
            cost_tensor=cost_tensor,
            bitwidths=bitwidths,
            deltas=deltas,
            zero_points=zero_points,
            in_channels=in_channels,
            max_candidates=max_candidates,
            add_uniform_candidates=add_uniform_candidates,
        )

    if use_k_means:
        groups = kmeans_fixed_simd_groups(cost_tensor, simd, random_state=random_state)  # list[Tensor]
    else:
        groups = _build_groups_by_grouping_param(grouping_param.to(DEVICE), simd)   # list[Tensor]
    G = len(groups)

    # Aggregate distortion per group (sum over member channels)
    group_cost = torch.stack(               # [G, |M|]
        [cost_tensor[idx].sum(dim=0) for idx in groups],
        dim=0
    )

    # ── 2.  run λ-frontier search on the *groups* ───────────────────────────
    group_results = lambda_frontier_candidates(
        cost_tensor=group_cost,                       # shape [G, |M|]
        bitwidths=bitwidths,
        deltas=deltas,                           # still per *channel* but we only need them later
        zero_points=zero_points,
        in_channels=in_channels,
        max_candidates=max_candidates,
        add_uniform_candidates=add_uniform_candidates,
    )

    expanded_results: List[Dict[str, Union["LayerCompressionConfig", float]]] = []
    for cfg in group_results:
        group_bits: List[int] = cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization     # length == G

        # Broadcast group bit to its member channels
        bits_per_channel = torch.empty(len(grouping_param), dtype=torch.int16).to(DEVICE)
        for g_idx, ch_idx in enumerate(groups):
            bits_per_channel[ch_idx] = group_bits[g_idx]

        # Re-run record_config on the *per-channel* assignment
        expanded_results.append(
            record_config(
                cost_tensor,
                torch.stack(list(deltas), dim=1),
                torch.stack(list(zero_points), dim=1),
                {b: i for i, b in enumerate(bitwidths)},
                bits_per_channel.tolist(),
                in_channels,
            )
        )

    # Preserve the size-descending order already set by the inner routine
    return expanded_results