import math
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch

from compression.local_candidate_optimization.greedy_candidate_search import lambda_frontier_candidates
from compression.local_candidate_optimization.search_utils import record_config
from constants import LAYER_COMPRESSION_CONFIG


# NOTE: the following helpers are assumed to exist in the surrounding code-base.
#   • lambda_frontier_candidates  – the *original* SIMD-agnostic generator
#   • record_config               – packs a {LayerCompressionConfig, size, mse…} dict
#   • generate_missing_uniform_configs (optional)
#   • DEVICE                      – torch.device (cpu / cuda)
#
# If they live in other modules, just update the imports accordingly.

# ──────────────────────────────────────────────────────────────────────────────
# Helper:  pick per-bit target counts that respect the SIMD groups
# ──────────────────────────────────────────────────────────────────────────────
def _nearest_simd_multiple(cnt: int, simd: int) -> int:
    """
    Round ``cnt`` to the nearest multiple of ``simd``.
    Ties are rounded **down** (e.g. 48 → 32 when simd=32).
    """
    down = (cnt // simd) * simd
    up = down + simd
    return down if (cnt - down) <= (up - cnt) else up


def _simd_target_histogram(
    hist: Dict[int, int],
    simd: int,
) -> Tuple[Dict[int, int], int]:
    """
    For every bit-width in *hist*, round the count to the nearest multiple
    of ``simd``.  After this pass – where **all buckets are multiples of
    simd** – the global sum will, in general, deviate from N by ``diff``.
    This ``diff`` ( ‐simd < diff < simd ) is returned so the caller can put
    the inevitable remainder into **one** bucket of its choice.
    """
    tgt = {b: _nearest_simd_multiple(c, simd) for b, c in hist.items()}
    diff = sum(hist.values()) - sum(tgt.values())   # ∈ (-simd, simd)
    return tgt, diff


# ──────────────────────────────────────────────────────────────────────────────
# Helper:  greedy, Δ-cost-minimal channel relocation
# ──────────────────────────────────────────────────────────────────────────────
def _greedy_balance(
    bits_tensor: torch.Tensor,
    target_cnt: Dict[int, int],
    cost_tensor: torch.Tensor,
    bit2idx: Dict[int, int],
) -> torch.Tensor:
    """
    Re-assign channels so the per-bit histogram equals `target_cnt`
    (all multiples of `simd`, Σ == N) while adding as little
    distortion as possible.  Guaranteed to terminate even if
    `cost_tensor` contains inf / nan.
    """
    bits_list = list(bit2idx.keys())
    cost_cpu  = cost_tensor.cpu()

    def channel_lists() -> Dict[int, List[int]]:
        return {
            b: (bits_tensor == b).nonzero(as_tuple=False).flatten().tolist()
            for b in bits_list
        }

    chan_per_bit = channel_lists()

    while True:
        # exact diff – prevents drift
        diff_cnt = {b: target_cnt[b] - len(v) for b, v in chan_per_bit.items()}
        if all(v == 0 for v in diff_cnt.values()):
            break  # balanced 🎉

        surpluses = [b for b in bits_list if diff_cnt[b] < 0 and chan_per_bit[b]]
        deficits  = [b for b in bits_list if diff_cnt[b] > 0]

        # ── 1) try the optimal finite-Δcost move ────────────────────────────
        best_move, best_delta = None, float("inf")
        for src_b in surpluses:
            src_idx = bit2idx[src_b]
            chans   = chan_per_bit[src_b]

            for dst_b in deficits:
                dst_idx = bit2idx[dst_b]
                src_c   = cost_cpu[chans, src_idx]
                dst_c   = cost_cpu[chans, dst_idx]
                valid   = torch.isfinite(src_c) & torch.isfinite(dst_c)
                if not valid.any():
                    continue

                d_cost = dst_c[valid] - src_c[valid]
                pos    = int(d_cost.argmin())
                ch     = chans[int(torch.nonzero(valid).flatten()[pos])]

                if d_cost[pos] < best_delta:
                    best_delta = float(d_cost[pos])
                    best_move  = (ch, src_b, dst_b)

        # ── 2) fallback: no surpluses or no finite move ─────────────────────
        if best_move is None:
            # choose *any* donor with at least one channel
            donor_b = next(b for b in bits_list if chan_per_bit[b])
            # choose any deficit, or if none exist (all positive diff gone)
            # borrow arbitrarily to push counts toward target
            recv_b  = deficits[0] if deficits else (
                next(b for b in bits_list if b != donor_b)
            )
            best_move = (chan_per_bit[donor_b][0], donor_b, recv_b)

        # ── 3) apply move and refresh bookkeeping ───────────────────────────
        ch, src_b, dst_b = best_move
        bits_tensor[ch]          = dst_b
        chan_per_bit[src_b].remove(ch)
        chan_per_bit[dst_b].append(ch)

    return bits_tensor

import networkx as nx                                  # add this import

def _min_cost_flow_balance(
    bits_tensor: torch.Tensor,                         # [N]  current bits
    target_cnt: Dict[int, int],                        # per-bit bucket targets
    cost_tensor: torch.Tensor,                         # [N, |M|]  MSE / KL …
    bit2idx: Dict[int, int],
) -> torch.Tensor:
    """
    Re-assign exactly the surplus channels so the final histogram equals
    `target_cnt`, while minimising Σ Δcost.  Uses a bipartite min-cost flow
    solved by NetworkX (`network_simplex` backend).

    • Left part  = every *surplus* channel (supply = 1).
    • Right part = one node per *deficit* bucket (demand = #missing channels).
    • Edge (channel → bucket) weight = Δcost = cost_dst − cost_src.

    Complexity  O(n log n) with modern simplex; fine for N ≤ 10 k.
    """
    DEVICE = bits_tensor.device
    bits_list = list(bit2idx.keys())

    # ---------- 1.  compute surplus / deficit channels -----------------------
    cur_hist = Counter(bits_tensor.tolist())
    diff = {b: target_cnt[b] - cur_hist.get(b, 0) for b in bits_list}

    surplus_channels: List[Tuple[int, int]] = []      # (ch_idx, src_bit)
    for b, d in diff.items():
        if d < 0:                                     # surplus
            candidates = (bits_tensor == b).nonzero(as_tuple=False).flatten()
            surplus_channels.extend(
                [(int(ch.item()), b) for ch in candidates[: -d]]  # only |d| move
            )

    deficit_buckets = {b: d for b, d in diff.items() if d > 0}     # need +d

    # trivial case – already balanced
    if not surplus_channels:
        return bits_tensor

    # ---------- 2.  build bipartite flow graph -------------------------------
    G = nx.DiGraph()

    # channel (supply) nodes
    for ch_idx, src_b in surplus_channels:
        G.add_node(("ch", ch_idx), demand=-1)

    # bucket (demand) nodes
    for b, need in deficit_buckets.items():
        G.add_node(("bucket", b), demand=need)

    # edges: surplus channel → every deficit bucket
    cost_cpu = cost_tensor.cpu()
    for ch_idx, src_b in surplus_channels:
        src_cost = float(cost_cpu[ch_idx, bit2idx[src_b]])
        for dst_b, need in deficit_buckets.items():
            dst_cost = float(cost_cpu[ch_idx, bit2idx[dst_b]])
            delta = dst_cost - src_cost
            # cap Δcost at +inf if either side is ±inf / nan
            if math.isfinite(delta):
                G.add_edge(
                    ("ch", ch_idx),
                    ("bucket", dst_b),
                    weight=delta,
                    capacity=1,
                )

    # ---------- 3.  run min-cost flow ---------------------------------------
    nx.network_simplex(G)       # fills in edge attribute 'flow'

    # ---------- 4.  apply the chosen moves -----------------------------------
    for (u, v, data) in G.edges(data=True):
        if data.get("flow", 0) != 1:
            continue
        (_, ch_idx) = u
        (_, dst_b)  = v
        bits_tensor[ch_idx] = dst_b

    return bits_tensor


import math, torch
from scipy.optimize import linear_sum_assignment   # SciPy ≥ 1.4

import math, torch
from collections import Counter
from typing import Dict, List, Sequence
from scipy.optimize import linear_sum_assignment     # SciPy ≥ 1.4

# ──────────────────────────────────────────────────────────────────────────────
# Helper:  optimal SIMD histogram fix-up via Hungarian assignment
# ──────────────────────────────────────────────────────────────────────────────
def _hungarian_balance(
    bits_tensor: torch.Tensor,                        # [N]  current bits
    target_cnt: Dict[int, int],                       # desired per-bit counts
    cost_tensor: torch.Tensor,                        # [N, |M|]  distortion
    bit2idx: Dict[int, int],
) -> torch.Tensor:
    """
    Move the minimal-cost subset of channels so the final histogram equals
    `target_cnt` (all buckets multiples of simd, Σ == N).  The move set is
    solved optimally with Hungarian (linear_sum_assignment).

    Runs in <<1 ms because the move count ≤ (simd – 1).
    """
    DEVICE   = bits_tensor.device
    bits_cpu = bits_tensor.cpu()
    cost_cpu = cost_tensor.cpu()
    bits_list = list(bit2idx.keys())

    # ---------- 1.  diff between current and target -------------------------
    cur_hist = Counter(bits_cpu.tolist())
    diff = {b: target_cnt[b] - cur_hist.get(b, 0) for b in bits_list}

    # ---------- 1a.  *robust fix* – ensure totals match ---------------------
    surplus = -sum(v for v in diff.values() if v < 0)   # positive number
    deficit =  sum(v for v in diff.values() if v > 0)
    if surplus != deficit:
        gap = surplus - deficit        # ±k, |k| < simd by construction

        # Put the gap into the bucket whose average cost impact is smallest
        def avg_penalty(b: int) -> float:
            idxs = (bits_cpu == b).nonzero(as_tuple=False).flatten()
            if idxs.numel() == 0:
                return math.inf
            return float(cost_cpu[idxs, bit2idx[b]].mean())

        cheapest_b = min(bits_list, key=avg_penalty)
        diff[cheapest_b] -= gap        # now Σ diff == 0 again
    # -----------------------------------------------------------------------

    donors: List[tuple[int, int]] = []      # (channel, src_bit)
    slots:  List[int] = []                  # dst_bit, one per deficit slot

    for b, d in diff.items():
        if d < 0:  # surplus
            donor_idx = (bits_cpu == b).nonzero(as_tuple=False).flatten()[: -d]
            donors.extend([(int(ch), b) for ch in donor_idx])
        elif d > 0:  # need channels
            slots.extend([b] * d)

    if not donors:          # already balanced
        return bits_tensor

    assert len(donors) == len(slots), "internal invariants violated"

    n_moves = len(donors)
    C = torch.empty((n_moves, n_moves), dtype=torch.float64)

    for i, (ch, src_b) in enumerate(donors):
        src_cost = cost_cpu[ch, bit2idx[src_b]]
        for j, dst_b in enumerate(slots):
            dst_cost = cost_cpu[ch, bit2idx[dst_b]]
            C[i, j] = dst_cost - src_cost

    row_ind, col_ind = linear_sum_assignment(C.numpy())

    for r, c in zip(row_ind, col_ind):
        ch, _ = donors[r]
        dst_b  = slots[c]
        bits_tensor[ch] = dst_b

    return bits_tensor

# ──────────────────────────────────────────────────────────────────────────────
# Public API:  λ-frontier → SIMD-aware candidates  (Strategy 1)
# ──────────────────────────────────────────────────────────────────────────────
def lambda_frontier_candidates_simd(
    cost_tensor: torch.Tensor,
    bitwidths: List[int],
    deltas: List[torch.Tensor],
    zero_points: List[torch.Tensor],
    in_channels: int,
    simd: int = 1,
    max_candidates: Optional[int] = None,
    add_uniform_candidates: bool = False,
) -> List[Dict[str, Union["LayerCompressionConfig", float]]]:
    """
    Generate mixed-precision candidates that **respect a fixed SIMD group
    size**.  Strategy 1 takes the high-quality per-channel bit-allocations
    produced by :func:`lambda_frontier_candidates` (simd == 1) and massages
    each one so that, except for the inevitable global remainder
    ``N mod simd``, every bit-width appears a multiple of *simd* times.

    The channel moves are chosen greedily, always picking the smallest
    increase in distortion (ΔMSE) first.

    Parameters
    ----------
    cost_tensor
        Distortion per channel / bit-width (shape ``[N, |M|]``).
    bitwidths
        Sequence of available bit-widths (order irrelevant).
    deltas, zero_points
        Per-bit tensors in the *same* order as *bitwidths*.
    in_channels
        Multiplier used inside ``record_config`` for total-size estimates.
    simd
        SIMD group size to enforce.  ``simd == 1`` is a no-op that simply
        forwards to the original generator.
    max_candidates
        Upper bound on the number of λ-frontier points to start from.
    add_uniform_candidates
        Forwarded to the inner λ-frontier call.

    Returns
    -------
    List[Dict]
        A list of candidate dictionaries identical in structure to those
        returned by :func:`lambda_frontier_candidates`, but now guaranteed to
        satisfy the SIMD constraint.
    """
    if simd == 1:
        # Fast path – nothing to do
        return lambda_frontier_candidates(
            cost_tensor=cost_tensor,
            bitwidths=bitwidths,
            deltas=deltas,
            zero_points=zero_points,
            in_channels=in_channels,
            max_candidates=max_candidates,
            add_uniform_candidates=add_uniform_candidates,
        )

    # ── 1.  High-quality starting point (λ-frontier, SIMD-agnostic) ────────────
    base_cands = lambda_frontier_candidates(
        cost_tensor,
        bitwidths,
        deltas,
        zero_points,
        in_channels,
        max_candidates=max_candidates,
        add_uniform_candidates=add_uniform_candidates,
    )

    # ── 2.  Common constants reused for every adjusted candidate ───────────────
    device = cost_tensor.device
    N = cost_tensor.shape[0]
    bits_desc = sorted(bitwidths, reverse=True)       # like in the original impl.
    bit2idx = {b: i for i, b in enumerate(bitwidths)}

    rows = torch.arange(N, device=device)

    deltas_tensor = torch.stack(list(deltas), dim=1)      # [N, |M|]
    zps_tensor    = torch.stack(list(zero_points), dim=1)

    adjusted_results: List[Dict[str, Union["LayerCompressionConfig", float]]] = []

    # ── 3.  Process every λ-frontier point independently ───────────────────────
    for cand in base_cands:
        # Extract the per-channel bit allocation from the candidate dict
        bits_vec = cand[LAYER_COMPRESSION_CONFIG].bit_width_quantization


        bits_tensor = torch.as_tensor(bits_vec, dtype=torch.int16, device=device)

        # Histogram → nearest SIMD multiples (all buckets rounded)
        hist = Counter(bits_tensor.tolist())
        target_cnt, diff = _simd_target_histogram(hist, simd)
        for b in bitwidths:
            if b not in target_cnt.keys():
                target_cnt[b] = 0

        # Put the inevitable remainder (|diff| < simd) into the cheapest bucket
        if diff != 0:
            # Evaluate *average* Δ-cost of adding/removing one channel
            avg_penalty: Dict[int, float] = {}
            for b in bitwidths:
                b_idx = bit2idx[b]
                channels_b = (bits_tensor == b).nonzero(as_tuple=False).flatten()
                if len(channels_b) == 0:
                    avg_penalty[b] = math.inf
                    continue
                # Δ-cost for bumping **this** bucket by ±1 channel is
                # approximated as the mean |∂MSE/∂channel| inside the bucket
                mse_b = cost_tensor[channels_b, b_idx]
                avg_penalty[b] = float(mse_b.mean().item())

            # Choose the bucket with the lowest penalty
            tweak_b = min(avg_penalty, key=avg_penalty.get)
            target_cnt[tweak_b] += diff     # diff may be ±(N mod simd)

        # ── Greedy balancing (smallest Δ-cost moves first) ──────────────────
        # bits_tensor = _greedy_balance(bits_tensor, target_cnt, cost_tensor, bit2idx)
        # bits_tensor = _min_cost_flow_balance(bits_tensor, target_cnt, cost_tensor, bit2idx)
        bits_tensor = _hungarian_balance(bits_tensor, target_cnt, cost_tensor, bit2idx)

        # ── Wrap-up: turn the balanced bit-vector into a config dict ────────
        adjusted_results.append(
            record_config(
                cost_tensor,
                deltas_tensor,
                zps_tensor,
                rows,
                bit2idx,
                bits_tensor.tolist(),
                in_channels,
            )
        )

    # Keep the same size-descending order as the original λ-frontier output
    adjusted_results.sort(key=lambda d: d["size"])
    return adjusted_results
