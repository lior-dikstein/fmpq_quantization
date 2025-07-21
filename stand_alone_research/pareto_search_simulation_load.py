import itertools
import numpy as np
import plotly.express as px
import torch

from compression.local_candidate_optimization.greedy_candidate_search import lambda_frontier_candidates
from compression.local_candidate_optimization.simd_random_group_search import lambda_frontier_candidates_grouped_random
from compression.local_candidate_optimization.search_utils import is_simd_valid
from constants import LAYER_COMPRESSION_CONFIG, SIZE, MSE, DEVICE
from debug.save_and_load_variables import load_debug_state
from helpers.timers import SegmentTimer
from stand_alone_research.pareto_search_2 import validate_results

def compare_results(res1, res2):
    if len(res1) != len(res2):
        return False
    for r1, r2 in zip(res1, res2):
        if r1[LAYER_COMPRESSION_CONFIG].bit_width_quantization != r1[
            LAYER_COMPRESSION_CONFIG].bit_width_quantization \
            and r1[SIZE] != r2[SIZE] \
            and r1[MSE] != r2[MSE]:
            return False
    return True


timer = SegmentTimer()
load_debug_state('deit_t_ch2')


# greedy = greedy_mixed_precision_candidates(mse_array, bitwidths, deltas, zero_points, out_channels, max_candidates)
# timer.segment('greedy_mixed_precision_candidates')
# print(f'Greedy 1 {len(greedy)}')


# analyse_algo("Greedy2 candidates", greedy2, bitwidths, size_cost, mse_cost, pareto_set)
deltas = [b* torch.ones_like(deltas[0]).to(DEVICE) for b in bitwidths]
zero_points = [b * torch.ones_like(zero_points[0]).to(DEVICE) for b in bitwidths]
simd = 32
greedy3 = lambda_frontier_candidates(mse_array, bitwidths, deltas, zero_points, out_channels, max_candidates, add_uniform_candidates=True)
timer.segment('greedy_mixed_precision_candidates3')
validate_results(greedy3)
print(f'Greedy 3 {len(greedy3)}')
valid_configs = [
        cfg for cfg in greedy3
        if is_simd_valid(cfg['layer_compression_config'].bit_width_quantization, simd)
    ]
print(f'Greedy 3 SIMD {len(valid_configs)}')

# entropy = torch.rand_like(mse_array[:,0])
#
# greedy4 = lambda_frontier_candidates_grouped(mse_array, entropy, bitwidths, deltas, zero_points, out_channels, simd, max_candidates, use_k_means=True)
# timer.segment('greedy_mixed_precision_candidates4')
# validate_results(greedy4)
# print(f'Greedy 4 {len(greedy4)}')
#
# valid_configs4 = [
#         cfg for cfg in greedy4
#         if is_simd_valid(cfg['layer_compression_config'].bit_width_quantization, simd)
#     ]
# print(f'Greedy 4 SIMD {len(valid_configs4)}')
#
# greedy5 = lambda_frontier_candidates_simd(
#     mse_array,
#     bitwidths,
#     simd=simd,
#     deltas=deltas,
#     zero_points=zero_points,
#     out_channels=out_channels,
# )
# timer.segment('greedy_mixed_precision_candidates5')
#
# validate_results(greedy5)
# print(f'Greedy 5 {len(greedy5)}')
# valid_configs5 = [
#         cfg for cfg in greedy5
#         if is_simd_valid(cfg['layer_compression_config'].bit_width_quantization, simd)
#     ]
# print(f'Greedy 5 SIMD {len(valid_configs5)}')

greedy6 = lambda_frontier_candidates_grouped_random(
    mse_array,
    bitwidths,
    simd=simd,
    deltas=deltas,
    zero_points=zero_points,
    in_channels=out_channels,
)
timer.segment('greedy_mixed_precision_candidates6')

validate_results(greedy6)
print(f'Greedy 6 {len(greedy6)}')
valid_configs6 = [
        cfg for cfg in greedy6
        if is_simd_valid(cfg['layer_compression_config'].bit_width_quantization, simd)
    ]
print(f'Greedy 6 SIMD {len(valid_configs6)}')



# greedy6 = lambda_frontier_candidates_strategy2(mse_array,
#     bitwidths,
#     simd=simd,
#     deltas=deltas,
#     zero_points=zero_points,
#     out_channels=out_channels)
# timer.segment('greedy_mixed_precision_candidates6')
#
# validate_results(greedy6)
# print(f'Greedy 6 {len(greedy6)}')
# valid_configs6 = [
#         cfg for cfg in greedy6
#         if is_simd_valid(cfg['layer_compression_config'].bit_width_quantization, simd)
#     ]
# print(f'Greedy 6 SIMD {len(valid_configs6)}')




# print(f'1 vs 2: {compare_results(greedy, greedy2)}')
# print(f'1 vs 3: {compare_results(greedy, greedy3)}')
# print(f'3 vs 4: {compare_results(greedy3, greedy4)}')
# analyse_algo("Greedy3 candidates", greedy3, bitwidths, size_cost, mse_cost, pareto_set)



