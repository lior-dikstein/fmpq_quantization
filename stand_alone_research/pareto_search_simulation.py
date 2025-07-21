import itertools
import numpy as np
import plotly.express as px

# ───────────────────────────────────────────────────────────────────────────────
# Cost-tensor generation
# ───────────────────────────────────────────────────────────────────────────────
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
    noise = rng.uniform(0.0, 0.05, (N, len(bits)))
    mse_cost = base + noise
    return size_cost, mse_cost


# ───────────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────────
def total_costs(size_cost, mse_cost, assign):
    """Total size & mse for one assignment vector."""
    rows = np.arange(size_cost.shape[0])
    return (
        float(size_cost[rows, assign].sum()),
        float(mse_cost[rows, assign].sum()),
    )


def pareto_mask(size_arr, mse_arr):
    """Boolean mask of Pareto-optimal points (minimising both)."""
    order = np.argsort(size_arr)
    best_mse = np.inf
    mask = np.zeros_like(size_arr, bool)
    for idx in order:
        if mse_arr[idx] < best_mse:
            mask[idx] = True
            best_mse = mse_arr[idx]
    return mask


# ───────────────────────────────────────────────────────────────────────────────
# Greedy search: ΔMSE / Δsize criterion
# ───────────────────────────────────────────────────────────────────────────────
def greedy_candidates(size_cost, mse_cost, bits=(2, 4, 8)):
    """
    Generate candidates by greedily flipping the channel whose
    ΔMSE / Δsize ratio (C) is minimal at each step.

    Returns:
        List of assignment vectors, from all-max to all-min.
    """
    N, M = size_cost.shape
    idx_max = len(bits) - 1
    assign = np.full(N, idx_max, dtype=int)
    candidates = [assign.copy()]

    while np.any(assign):
        cur_size, cur_mse = total_costs(size_cost, mse_cost, assign)

        best_C, best_ch = None, None
        for ch in range(N):
            if assign[ch] == 0:
                continue
            new_assign = assign.copy()
            new_assign[ch] -= 1
            new_size, new_mse = total_costs(size_cost, mse_cost, new_assign)

            d_mse = new_mse - cur_mse        # >0
            d_size = cur_size - new_size     # >0
            C = d_mse / d_size
            if best_C is None or C < best_C:
                best_C, best_ch = C, ch

        assign[best_ch] -= 1
        candidates.append(assign.copy())

    return np.array(candidates)


# ───────────────────────────────────────────────────────────────────────────────
# Main demo
# ───────────────────────────────────────────────────────────────────────────────
N = 8
BITS = (2, 4, 8)

size_cost, mse_cost = generate_cost_tensors(N, BITS)

# Exhaustive enumeration (ground truth)
combos = np.array(list(itertools.product(range(len(BITS)), repeat=N)), dtype=int)
rows = np.repeat(np.arange(N)[None, :], combos.shape[0], axis=0)
total_size = size_cost[rows, combos].sum(axis=1)
total_mse = mse_cost[rows, combos].sum(axis=1)

pareto = pareto_mask(total_size, total_mse)

print(f"Total combinations: {total_size.size}")
print(f"Pareto-optimal points: {pareto.sum()}")

# Interactive scatter
fig = px.scatter(
    x=total_size,
    y=total_mse,
    color=np.where(pareto, "Pareto", "Non-Pareto"),
    labels={"x": "Total Size (bits)", "y": "Total MSE"},
    title=f"Size vs MSE – N={N}",
    opacity=0.35,
)
fig.update_traces(marker=dict(size=5))
fig.show()

# Greedy sequence (for user inspection)
greedy_seq = greedy_candidates(size_cost, mse_cost, BITS)
print("\nGreedy path (first 5 assignments shown):")
for i, a in enumerate(greedy_seq[:5]):
    print(f"{i:2d}: {BITS}-indices -> {a}")

