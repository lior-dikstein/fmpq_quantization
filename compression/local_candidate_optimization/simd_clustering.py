# --------------------------------------------------------------------------- #
#  Strategy-2: cost-curve clustering with fixed SIMD blocks                   #
# --------------------------------------------------------------------------- #
import math
import torch
from typing import Any, Dict, List, Optional, Sequence, Union

from compression.local_candidate_optimization.greedy_candidate_search import lambda_frontier_candidates
from compression.local_candidate_optimization.search_utils import record_config
from constants import LAYER_COMPRESSION_CONFIG


# ---------------------------------------------------------
# Lightweight k-means  (same as before, very small helper)
# ---------------------------------------------------------
                                # len == ceil(N/simd)

# ---------------------------------------------------------
# Main Strategy-2 function
# ---------------------------------------------------------
def lambda_frontier_candidates_strategy2(
    cost_tensor: torch.Tensor,
    bitwidths: Sequence[int],
    deltas: Sequence[torch.Tensor],
    zero_points: Sequence[torch.Tensor],
    in_channels: int,
    simd: int = 32,
    max_candidates: Optional[int] = None,
    add_uniform_candidates: bool = False,
) -> List[Dict[str, Union["LayerCompressionConfig", float]]]:
    """
    Strategy 2 – k-means cost-curve clustering with *fixed* SIMD blocks.
    Guarantees every returned candidate satisfies the SIMD rule.
    """
    if simd == 1:
        # simply defer to the baseline search
        return lambda_frontier_candidates(
            cost_tensor=cost_tensor, 
            bitwidths=bitwidths, 
            deltas=deltas, 
            zero_points=zero_points, 
            in_channels=in_channels,
            max_candidates=max_candidates,
            add_uniform_candidates=add_uniform_candidates,
        )

    DEVICE = cost_tensor.device
    N      = cost_tensor.size(0)
    rows   = torch.arange(N, device=DEVICE)
    bit2idx = {b: i for i, b in enumerate(bitwidths)}
    deltas_tensor = torch.stack(list(deltas), dim=1)
    zps_tensor    = torch.stack(list(zero_points), dim=1)

    # ---------- 1. grouping ---------------------------------------------------
    groups = _build_fixed_simd_groups(cost_tensor, simd)        # list[LongT]
    G = len(groups)

    # ---------- 2. aggregate costs per group ---------------------------------
    group_cost = torch.stack(
        [cost_tensor[idx].sum(dim=0) for idx in groups], dim=0   # [G, |M|]
    )

    # ---------- 3. λ-frontier on groups --------------------------------------
    group_candidates = lambda_frontier_candidates(
        group_cost, bitwidths, deltas, zero_points, in_channels,
        max_candidates=max_candidates,
        add_uniform_candidates=add_uniform_candidates,
    )

    # ---------- 4. expand & verify SIMD legality -----------------------------
    expanded: List[Dict[str, Union["LayerCompressionConfig", float]]] = []
    for cfg in group_candidates:
        grp_bits: List[int] = cfg[LAYER_COMPRESSION_CONFIG].bit_width_quantization

        per_chan = torch.empty(N, dtype=torch.int16, device=DEVICE)
        for g, idx in enumerate(groups):
            per_chan[idx] = grp_bits[g]

        # SIMD legality check
        hist = {b: int((per_chan == b).sum()) for b in bitwidths}
        non_multiple = [b for b, cnt in hist.items() if cnt % simd != 0]
        if len(non_multiple) > 1:
            raise RuntimeError(
                f"Strategy-2 produced illegal candidate: "
                f"buckets with remainders = {non_multiple}"
            )

        expanded.append(
            record_config(
                cost_tensor,
                deltas_tensor,
                zps_tensor,
                rows,
                bit2idx,
                per_chan.cpu().tolist(),
                in_channels,
            )
        )

    return expanded
