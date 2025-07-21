import time

import torch

from compression.configs.compression_config import CandidateSearchAlg
from compression.local_candidate_optimization.bi_objective_a_star import BOAStarQuantizer, EpsilonBOAStarQuantizer
from compression.local_candidate_optimization.greedy_candidate_search import greedy_mixed_precision_candidates, \
    lambda_frontier_candidates, lambda_frontier_candidates_grouped
from compression.local_candidate_optimization.random_step_candidates import random_mixed_precision_candidates
from compression.local_candidate_optimization.search_utils import is_simd_valid
from compression.local_candidate_optimization.simd_random_group_search import lambda_frontier_candidates_grouped_random


def run_pareto2(candidate_search_alg, in_list_to_pareto, entropy, hessian_per_channel, in_channels, max_candidates, simd=32):
    mse_array = torch.stack([p[0] for p in in_list_to_pareto], dim=1)
    bitwidths = [p[1].bit_width_quantization for p in in_list_to_pareto]
    deltas = [p[1].delta for p in in_list_to_pareto]
    zero_points = [p[1].zero_point for p in in_list_to_pareto]
    start_search_time = time.time()
    if candidate_search_alg == CandidateSearchAlg.BOA_STAR:
        boa = BOAStarQuantizer(mse_array, bitwidths, deltas, zero_points, in_channels=in_channels, max_solutions=None)
        candidates = boa.search()
    if candidate_search_alg == CandidateSearchAlg.EPS_BOA_STAR:
        boa = EpsilonBOAStarQuantizer(mse_array, bitwidths, deltas, zero_points, in_channels=in_channels, max_solutions=None)
        candidates = boa.search()
    elif candidate_search_alg == CandidateSearchAlg.GREEDY:
        candidates = greedy_mixed_precision_candidates(mse_array, bitwidths, deltas, zero_points, in_channels, max_candidates)
    elif candidate_search_alg == CandidateSearchAlg.LAMBDA:
        candidates = lambda_frontier_candidates(
            cost_tensor=mse_array, bitwidths=bitwidths, deltas=deltas, zero_points=zero_points,
            in_channels=in_channels, max_candidates=max_candidates)
    elif candidate_search_alg == CandidateSearchAlg.LAMBDA_GROUP_BY_ENTROPY:
        candidates = lambda_frontier_candidates_grouped(
            cost_tensor=mse_array, grouping_param=entropy, bitwidths=bitwidths, deltas=deltas,
            zero_points=zero_points, in_channels=in_channels, simd=simd, max_candidates=max_candidates)
    elif candidate_search_alg == CandidateSearchAlg.LAMBDA_GROUP_BY_HESSIAN:
        candidates = lambda_frontier_candidates_grouped(
            cost_tensor=mse_array, grouping_param=hessian_per_channel, bitwidths=bitwidths, deltas=deltas,
            zero_points=zero_points, in_channels=in_channels, simd=simd, max_candidates=max_candidates)
    elif candidate_search_alg == CandidateSearchAlg.LAMBDA_GROUP_BY_MSE:
        candidates = lambda_frontier_candidates_grouped(
            cost_tensor=mse_array, grouping_param=hessian_per_channel, bitwidths=bitwidths, deltas=deltas,
            zero_points=zero_points, in_channels=in_channels, simd=simd, max_candidates=max_candidates, use_k_means=True)
    elif candidate_search_alg == CandidateSearchAlg.LAMBDA_GROUP_RANDOM_PARETO:
        candidates = lambda_frontier_candidates_grouped_random(
            cost_tensor=mse_array, bitwidths=bitwidths, deltas=deltas, zero_points=zero_points,
            in_channels=in_channels, simd=simd, max_candidates=max_candidates,
            entropy=entropy, hessian=hessian_per_channel)
    elif candidate_search_alg == CandidateSearchAlg.RANDOM_STEP:
        candidates = random_mixed_precision_candidates(mse_array, bitwidths, deltas, zero_points, in_channels, max_candidates)
    # print(f'all candidates: {len(candidates)}, all candidates alternative: {len(candidates2)}')
    valid_configs = [
        cfg for cfg in candidates
        if is_simd_valid(cfg['layer_compression_config'].bit_width_quantization, simd)
    ]
    print(f'all candidates: {len(candidates)}, simd valid candidates {len(valid_configs)}, search time: {int(time.time() - start_search_time)}')
    return valid_configs

