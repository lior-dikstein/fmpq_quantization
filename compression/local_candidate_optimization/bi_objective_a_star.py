from __future__ import annotations

import numpy as np
import time

import itertools

import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import torch

from compression.configs.layer_comrpression_config import LayerCompressionConfig
from constants import LAYER_COMPRESSION_CONFIG, SIZE, MSE, DEVICE


# ─────────────────────────────────────────────────────────────────────────────
# Internal priority-queue entry (compact, sortable by (f1, f2))
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(order=True, slots=True)
class _Node:
    f1: float
    f2: float
    idx: int                      # number of channels already decided
    g1: float = field(compare=False)
    g2: float = field(compare=False)
    parent: int = field(compare=False)
    col: int = field(compare=False)     # column index (bit-width) chosen at this step

# ─────────────────────────────────────────────────────────────────────────────
# Bi-objective A* for mixed-precision candidate generation
# ─────────────────────────────────────────────────────────────────────────────
class BOAStarQuantizer:
    """
    Bi-objective A* search that enumerates the Pareto-optimal bit-width
    assignments for N weight channels under two criteria:

    • g1 — total distortion (for example, sum of per-channel MSEs).
    • g2 — *extra* memory bits beyond the minimal bit-width baseline
            (bit_width − min_bit) summed across channels.

    Bi-objective A* for mixed-precision bit selection (distortion + extra bits).

    Parameters
    ----------
    cost_tensor        : (N, M)  distortion per channel × bit-width.
    bitwidths          : list[int]  available bit-widths (length M).
    delta_list         : list[tensor]  length M; each tensor shape (N,) or (N,1).
    zero_point_list    : list[tensor]  same layout as delta_list.
    in_channels       : int  multiplier for memory cost (default 1).
    max_solutions      : int|None  optional cap on number of Pareto points.
    """

    def __init__(
        self,
        cost_tensor: torch.Tensor,
        bitwidths: List[int],
        delta_list: List[torch.Tensor],
        zp_list: List[torch.Tensor],
        *,
        in_channels: int = 1,
        max_solutions: int | None = None,
    ) -> None:

        self.cost = cost_tensor  # (N, M)
        self.bits = bitwidths
        self.in_channels = in_channels
        self.delta_list = delta_list
        self.zp_list = zp_list

        self.N, self.M = cost_tensor.shape
        self.max_sols = max_solutions

        # extra-bits objective -------------------------------------------- #
        min_bit = min(bitwidths)
        extra_bits_row = (
                                 torch.tensor(bitwidths, dtype=torch.float32) - min_bit
                         ) * in_channels
        self.size_inc = extra_bits_row.expand(self.N, -1)  # (N, M)

        # distortion heuristic (tight) ------------------------------------ #
        min_cost = self.cost.min(dim=1).values
        self.h1 = torch.cumsum(min_cost.flip(0), 0).flip(0).tolist() + [0.0]

        # size heuristic (always zero) ------------------------------------ #
        self.h2 = [0.0] * (self.N + 1)

    def _reconstruct_solutions(self, solutions, closed):
        # reconstruct solutions (unchanged) ---------------------------- #
        results: List[dict] = []
        for sol in solutions:
            cols: List[int] = []
            node = sol
            while node.parent != -1:
                cols.append(node.col)
                node = closed[node.parent]
            cols.reverse()

            bits = [self.bits[c] for c in cols]
            delta_vec = torch.empty(self.N, dtype=self.delta_list[0].dtype).to(DEVICE)
            zp_vec = torch.empty(self.N, dtype=self.zp_list[0].dtype).to(DEVICE)
            total_mse = 0.0
            for ch in range(self.N):
                col = cols[ch]
                delta_vec[ch] = self.delta_list[col][ch]
                zp_vec[ch] = self.zp_list[col][ch]
                total_mse += self.cost[ch, col].item()

            total_size = self.in_channels * sum(bits)
            layer_cc = LayerCompressionConfig(bit_width_quantization=bits)
            layer_cc.set_weights_params(
                delta_vec.view(*delta_vec.shape, *[1] * (self.delta_list[0].dim() - delta_vec.dim())).to(DEVICE),
                zp_vec.view(*zp_vec.shape, *[1] * (self.zp_list[0].dim() - zp_vec.dim())).to(DEVICE),
            )
            results.append({
                LAYER_COMPRESSION_CONFIG: layer_cc,
                SIZE: total_size,
                MSE: float(total_mse),
            })

        results.sort(key=lambda x: sum(x[LAYER_COMPRESSION_CONFIG].bit_width_quantization))
        return results


    def search(self) -> List[dict]:
        """
        Returns
        -------
        List[dict]  — each dict has keys
            • bits        : List[int]
            • delta       : List[float]
            • zero_point  : List[float]
            • distortion  : float  (g1)
            • extra_bits  : float  (g2)
        ordered by ascending extra_bits.
        """
        open_pq: List[_Node] = []
        closed: List[_Node | None] = []

        g2_min_by_depth = [float("inf")] * (self.N + 1)
        best_goal_g2 = float("inf")
        solutions: List[_Node] = []

        # start ------------------------------------------------------------ #
        heapq.heappush(
            open_pq,
            _Node(
                f1=self.h1[0],
                f2=0.0,
                idx=0,
                g1=0.0,
                g2=0.0,
                parent=-1,
                col=-1,
            ),
        )

        # search loop ------------------------------------------------------ #
        while open_pq and (self.max_sols is None or len(solutions) < self.max_sols):
            n = heapq.heappop(open_pq)

            if n.g2 >= g2_min_by_depth[n.idx] or n.f2 >= best_goal_g2:
                continue
            g2_min_by_depth[n.idx] = n.g2

            if n.idx == self.N:  # goal
                best_goal_g2 = n.g2
                closed.append(n)
                solutions.append(n)
                continue

            cur_idx = len(closed)
            closed.append(n)

            i = n.idx  # expand channel i
            for j in range(self.M):
                g1_new = n.g1 + float(self.cost[i, j])
                g2_new = n.g2 + float(self.size_inc[i, j])
                idx_new = i + 1

                child = _Node(
                    f1=g1_new + self.h1[idx_new],
                    f2=g2_new + self.h2[idx_new],
                    idx=idx_new,
                    g1=g1_new,
                    g2=g2_new,
                    parent=cur_idx,
                    col=j,
                )

                if child.g2 >= g2_min_by_depth[idx_new] or child.f2 >= best_goal_g2:
                    continue
                heapq.heappush(open_pq, child)

        return self._reconstruct_solutions(solutions, closed)


class EpsilonBOAStarQuantizer(BOAStarQuantizer):
    """
       ε-dominance approximate variant of `BOAStarQuantizer`.

       Differences from the exact search
       ---------------------------------
       • Keeps a global Pareto-frontier 𝔽 of already expanded labels.
       • A node (g₁,g₂) is *ε-dominated* if ∃(f₁,f₂)∈𝔽 such that
           f₁ ≤ (1+ε)·g₁  and  f₂ ≤ (1+ε)·g₂ .
         ε-dominated nodes are never inserted/expanded.
       • When a non-dominated node is popped, it is inserted into 𝔽
         after removing any labels that it ε-dominates in turn.

       Parameters
       ----------
       * Same as `BOAStarQuantizer` *
       epsilon        : float   relative tolerance (ε > 0). Default 0.01 (1 %).
       max_frontier   : int|None  optional cap on |𝔽| (keeps best‐g₂ labels).
       """

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
            self,
            cost_tensor: torch.Tensor,
            bitwidths: List[int],
            delta_list: List[torch.Tensor],
            zp_list: List[torch.Tensor],
            *,
            in_channels: int = 1,
            max_solutions: Optional[int] = None,
            epsilon: float = 0.005,
            max_frontier: Optional[int] = None,
    ):
        super().__init__(
            cost_tensor,
            bitwidths,
            delta_list,
            zp_list,
            in_channels=in_channels,
            max_solutions=max_solutions,
        )
        self.eps = max(float(epsilon), 0.0)
        self.max_frontier = max_frontier
        # one frontier per depth
        self._frontier: List[List[Tuple[float, float]]] = [[] for _ in range(self.N + 1)]

        # ------------------------------------------------------------------ #
        # ε-dominance helpers                                                #
        # ------------------------------------------------------------------ #

    def _eps_dominated(self, depth: int, g1: float, g2: float) -> bool:
        """True iff (g1,g2) ε-dominated by some label at the same depth."""
        tol1, tol2 = g1 * (1.0 + self.eps), g2 * (1.0 + self.eps)
        return any(f1 <= tol1 and f2 <= tol2 for f1, f2 in self._frontier[depth])

    def _add_to_frontier(self, depth: int, g1: float, g2: float) -> None:
        """Insert label at depth and prune labels it ε-dominates."""
        tol1, tol2 = (1.0 + self.eps) * g1, (1.0 + self.eps) * g2
        kept = [
            (f1, f2)
            for f1, f2 in self._frontier[depth]
            if not (g1 <= (1.0 + self.eps) * f1 and g2 <= (1.0 + self.eps) * f2)
        ]
        kept.append((g1, g2))
        if self.max_frontier is not None and len(kept) > self.max_frontier:
            kept.sort(key=lambda t: t[1])  # keep smallest-g₂ labels
            kept = kept[: self.max_frontier]
        self._frontier[depth] = kept

        # ------------------------------------------------------------------ #
        # public search                                                      #
        # ------------------------------------------------------------------ #

    def search(self) -> List[dict]:
        open_pq: List[_Node] = []
        closed: List[Optional[_Node]] = []

        g2_min_by_depth = [float("inf")] * (self.N + 1)
        best_goal_g2 = float("inf")
        solutions: List[_Node] = []

        # initial label -------------------------------------------------- #
        heapq.heappush(
            open_pq,
            _Node(
                f1=self.h1[0],
                f2=0.0,
                idx=0,
                g1=0.0,
                g2=0.0,
                parent=-1,
                col=-1,
            ),
        )

        # main A* loop --------------------------------------------------- #
        while open_pq and (self.max_sols is None or len(solutions) < self.max_sols):
            n = heapq.heappop(open_pq)

            if self._eps_dominated(n.idx, n.g1, n.g2):
                continue
            if n.g2 >= g2_min_by_depth[n.idx] or n.f2 >= best_goal_g2:
                continue
            g2_min_by_depth[n.idx] = n.g2
            self._add_to_frontier(n.idx, n.g1, n.g2)

            if n.idx == self.N:  # full assignment reached
                best_goal_g2 = n.g2
                closed.append(n)
                solutions.append(n)
                continue

            cur_idx = len(closed)
            closed.append(n)
            i = n.idx  # expand channel i

            for j in range(self.M):
                g1_new = n.g1 + float(self.cost[i, j])
                g2_new = n.g2 + float(self.size_inc[i, j])
                new_depth = i + 1

                if self._eps_dominated(new_depth, g1_new, g2_new):
                    continue
                child = _Node(
                    f1=g1_new + self.h1[new_depth],
                    f2=g2_new,
                    idx=new_depth,
                    g1=g1_new,
                    g2=g2_new,
                    parent=cur_idx,
                    col=j,
                )
                if child.g2 >= g2_min_by_depth[new_depth] or child.f2 >= best_goal_g2:
                    continue
                heapq.heappush(open_pq, child)

        return self._reconstruct_solutions(solutions, closed)


# ------------------------------------------------------------------------- #
# Core helpers                                                              #
# ------------------------------------------------------------------------- #
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


def validate_results(pareto):
    print(f'Len pareto {len(pareto)}, {N} channels, {M} bit options')
    for p in pareto:
        assert torch.sum(torch.abs(
            torch.tensor(p[LAYER_COMPRESSION_CONFIG].bit_width_quantization).to(DEVICE).view(-1, 1) - p[
                LAYER_COMPRESSION_CONFIG].delta.to(DEVICE))) == 0
        assert torch.sum(torch.abs(
            torch.tensor(p[LAYER_COMPRESSION_CONFIG].bit_width_quantization).to(DEVICE).view(-1, 1) - p[
                LAYER_COMPRESSION_CONFIG].zero_point.to(DEVICE))) == 0
    print('Pareto results validated')


if __name__ == "__main__":
    from helpers.timers import SegmentTimer

    timer = SegmentTimer()
    torch.manual_seed(0)
    N, bitwidths = 400, [2, 4, 8]
    M = len(bitwidths)
    max_bit = max(bitwidths)

    # Generate distortion so that lower bit ⇒ higher MSE ------------------- #
    base_distortion = torch.rand(N, 1).to(DEVICE)          # per-channel baseline
    scale = torch.tensor([max_bit / b for b in bitwidths]).to(DEVICE).float()  # 8→1, 4→2, 2→4

    cost = base_distortion * scale              # shape (N, M)    size = torch.tensor(bitwidths, dtype=torch.float32).expand(N, M)
    deltas = [b* torch.ones_like(base_distortion).to(DEVICE) for b in bitwidths]
    zero_points = [b * torch.ones_like(base_distortion).to(DEVICE) for b in bitwidths]
    size = torch.tensor(bitwidths, dtype=torch.float32).to(DEVICE).expand(N, M)

    # Len pareto 1200, 400 channels, 3 bit options - 27.7273s
    # boa = BOAStarQuantizer(cost, bitwidths, deltas, zero_points, in_channels=1, max_solutions=None)
    # pareto = boa.search()
    # timer.segment('BOAStarQuantizer')
    # validate_results(pareto)

    eboa = EpsilonBOAStarQuantizer(cost, bitwidths, deltas, zero_points, epsilon=0.005, in_channels=1, max_solutions=None)
    pareto2 = eboa.search()
    timer.segment('EpsilonBOAStarQuantizer')
    validate_results(pareto2)


    # Exhaustive enumeration ------------------------------------------------ #
    # all_solutions = enumerate_all_candidates(cost, size, bitwidths)
    # mask = [s[0] in pareto_points for s in all_solutions]

    # # Display --------------------------------------------------------------- #
    # print("Bits\t\tCost\t\tSize\tPareto")
    # for (bits, d, s), is_p in sorted(zip(all_solutions, mask),
    #                                  key=lambda t: t[0][2]):  # sort by size
    #     print(f"{bits}\t{d:.4f}\t{s:.0f}\t{is_p}")

    # for bits, d, s in pareto:
    #     print(bits, d, s)