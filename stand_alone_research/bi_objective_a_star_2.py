from __future__ import annotations

import time

import itertools

import heapq
from dataclasses import dataclass, field
from typing import List, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Internal priority-queue entry (compact, sortable by (f1, f2))
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(order=True, slots=True)
class _Node:
    f1: float                    # estimated total distortion
    f2: float                    # estimated extra-bits size
    idx: int                     # number of channels already decided
    g1: float = field(compare=False)   # accumulated distortion
    g2: float = field(compare=False)   # accumulated extra-bits
    parent: int = field(compare=False) # index in closed list
    bit: int = field(compare=False)    # bit chosen at this step


# ─────────────────────────────────────────────────────────────────────────────
# Bi-objective A* for mixed-precision candidate generation
# ─────────────────────────────────────────────────────────────────────────────
class BOAStarQuantizer:
    """
    Bi-objective A* search that enumerates the Pareto-optimal bit-width
    assignments for N weight channels under two criteria:

    • g1 — total distortion (sum of per-channel MSEs).
    • g2 — *extra* memory bits beyond the minimal bit-width baseline
            (bit_width − min_bit) summed across channels.

    The algorithm is optimal and uses constant-time dominance tests, with
    a tight (per-row minimum) heuristic for distortion and a trivial (zero)
    heuristic for size.
    """

    # --------------------------------------------------------------------- #
    # Construction                                                          #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        cost_tensor: torch.Tensor,
        size_tensor: torch.Tensor,
        bitwidths: List[int],
        *,
        out_channels: int = 1,
        max_solutions: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        cost_tensor   : (N, M) tensor.  cost_tensor[i, j] = distortion for
                        channel *i* at bit-width bitwidths[j].
        size_tensor   : (N, M) tensor.  size_tensor[i, j] = memory cost
                        (bits / bytes) for channel *i* at bit-width bitwidths[j].
        bitwidths     : list of available bit-widths, same order as columns.
        out_channels  : multiplier for size (e.g. groups); default 1.
        max_solutions : optional cap on the number of Pareto points returned.
        """
        self.bits = bitwidths
        self.cost = cost_tensor
        # extra bits relative to the cheapest bit-width per channel
        self.size_inc = (
            size_tensor - size_tensor.min(dim=1, keepdim=True).values
        ) * out_channels

        self.N, self.M = self.cost.shape
        self.max_sols = max_solutions

        # distortion heuristic: per-row minimum suffix sum (tight & admissible)
        min_cost = self.cost.min(dim=1).values
        self.h1 = torch.cumsum(min_cost.flip(0), dim=0).flip(0).tolist() + [0.0]

        # size heuristic: always zero (extra bits can be 0 for remaining channels)
        self.h2 = [0.0] * (self.N + 1)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def search(self) -> List[Tuple[List[int], float, float]]:
        """
        Executes BOA* and returns a list of Pareto-optimal solutions.

        Returns
        -------
        List of tuples (bits, distortion, extra_bits), ordered by ascending
        extra_bits.
        """
        open_pq: List[_Node] = []
        closed: List[_Node | None] = []

        g2_min_by_depth = [float("inf")] * (self.N + 1)  # BOA* table
        best_goal_g2 = float("inf")                      # best size so far
        solutions: List[_Node] = []

        # start node (no channels processed yet)
        start = _Node(
            f1=self.h1[0],
            f2=0.0,
            idx=0,
            g1=0.0,
            g2=0.0,
            parent=-1,
            bit=-1,
        )
        heapq.heappush(open_pq, start)

        # main search loop
        while open_pq and (self.max_sols is None or len(solutions) < self.max_sols):
            n = heapq.heappop(open_pq)

            # constant-time dominance pruning
            if n.g2 >= g2_min_by_depth[n.idx] or n.f2 >= best_goal_g2:
                continue
            g2_min_by_depth[n.idx] = n.g2

            # goal state: all channels assigned
            if n.idx == self.N:
                best_goal_g2 = n.g2
                closed.append(n)
                solutions.append(n)
                continue

            cur_idx = len(closed)
            closed.append(n)

            i = n.idx  # expand next channel
            for j in range(self.M):
                g1_new = n.g1 + float(self.cost[i, j])
                g2_new = n.g2 + float(self.size_inc[i, j])
                idx_new = i + 1

                child = _Node(
                    f1=g1_new + self.h1[idx_new],
                    f2=g2_new,               # h2 == 0 for all idx
                    idx=idx_new,
                    g1=g1_new,
                    g2=g2_new,
                    parent=cur_idx,
                    bit=self.bits[j],
                )

                if child.g2 >= g2_min_by_depth[idx_new] or child.f2 >= best_goal_g2:
                    continue
                heapq.heappush(open_pq, child)

        # reconstruct bit-vectors
        results: List[Tuple[List[int], float, float]] = []
        for sol in solutions:
            bits: List[int] = []
            node = sol
            while node.parent != -1:
                bits.append(node.bit)
                node = closed[node.parent]
            results.append((bits[::-1], sol.g1, sol.g2))

        results.sort(key=lambda t: t[2])  # stable order by extra_bits
        return results



# ------------------------------------------------------------------------- #
# Core helpers                                                              #
# ------------------------------------------------------------------------- #
def enumerate_all_candidates(
    cost_tensor: torch.Tensor,
    size_tensor: torch.Tensor,
    bitwidths: List[int],
) -> List[Tuple[List[int], float, float]]:
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



# ------------------------------------------------------------------------- #
# Example validation with “logical” costs                                   #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    # torch.manual_seed(0)  # reproducibility
    #
    # # Problem setup -------------------------------------------------------- #
    # N, bitwidths = 3, [2, 4, 8]  # channels & available bit-widths
    # M = len(bitwidths)
    # max_bit = max(bitwidths)
    #
    # # Generate distortion so that lower bit ⇒ higher MSE ------------------- #
    # base_distortion = torch.rand(N, 1)          # per-channel baseline
    # scale = torch.tensor([max_bit / b for b in bitwidths]).float()  # 8→1, 4→2, 2→4
    # cost = base_distortion * scale              # shape (N, M)
    #
    # # Memory cost is proportional to bit-width ----------------------------- #
    # size = torch.tensor(bitwidths, dtype=torch.float32).expand(N, M)
    #
    # # Exhaustive enumeration ------------------------------------------------ #
    # all_solutions = enumerate_all_candidates(cost, size, bitwidths)
    # mask = pareto_front(all_solutions)
    #
    # # Display --------------------------------------------------------------- #
    # print("Bits\t\tCost\t\tSize\tPareto")
    # for (bits, d, s), is_p in sorted(zip(all_solutions, mask),
    #                                  key=lambda t: t[0][2]):  # sort by size
    #     print(f"{bits}\t{d:.4f}\t{s:.0f}\t{is_p}")
    t_start = time.time()
    torch.manual_seed(0)
    N, bitwidths = 1024, [2, 4, 8]
    M = len(bitwidths)
    max_bit = max(bitwidths)

    # Generate distortion so that lower bit ⇒ higher MSE ------------------- #
    base_distortion = torch.rand(N, 1)          # per-channel baseline
    scale = torch.tensor([max_bit / b for b in bitwidths]).float()  # 8→1, 4→2, 2→4
    cost = base_distortion * scale              # shape (N, M)    size = torch.tensor(bitwidths, dtype=torch.float32).expand(N, M)
    size = torch.tensor(bitwidths, dtype=torch.float32).expand(N, M)

    boa = BOAStarQuantizer(cost, size, bitwidths, out_channels=1, max_solutions=None)
    pareto = boa.search()
    pareto_points = [p[0] for p in pareto]
    # Exhaustive enumeration ------------------------------------------------ #
    # all_solutions = enumerate_all_candidates(cost, size, bitwidths)
    # mask = [s[0] in pareto_points for s in all_solutions]

    print(len(pareto_points))
    print(f'Time for algorithm on {N} channels and {M} bit options: {int(time.time()-t_start)}')
    # # Display --------------------------------------------------------------- #
    # print("Bits\t\tCost\t\tSize\tPareto")
    # for (bits, d, s), is_p in sorted(zip(all_solutions, mask),
    #                                  key=lambda t: t[0][2]):  # sort by size
    #     print(f"{bits}\t{d:.4f}\t{s:.0f}\t{is_p}")

    # for bits, d, s in pareto:
    #     print(bits, d, s)