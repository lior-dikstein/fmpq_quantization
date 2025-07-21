from typing import Dict, Optional

import torch
from ortools.constraint_solver.pywrapcp import IntVar, Solver
from ortools.linear_solver import pywraplp
from tqdm import tqdm

from compression.configs.compression_config import MPCost
from compression.error_interpolation import compute_batch_score_per_config, \
    normalize_per_batch_score, ScorePoint, compute_channel_score_per_config, compute_sqnr_interpolation_score, \
    compute_wmse_score_per_config
from compression.ordering import generate_pareto_cost, generate_mp_cost
from constants import SOLVER_TIME_LIMIT
from helpers.utils import is_compressed_layer
from compression.quantization.distance_metrics import sqnr_distance, logits_kl_divergence, channel_entropy


def init_lp_vars(solver: Solver, layer_to_metrics_mapping: Dict[str, Dict[int, float]]) -> Dict[str, Dict[int, IntVar]]:
    layer_to_indicator_vars_mapping = dict()

    for i, (layer, nbits_to_metric) in enumerate(layer_to_metrics_mapping.items()):
        layer_to_indicator_vars_mapping[layer] = dict()

        for nbits in nbits_to_metric.keys():
            var = solver.IntVar(0, 1, f"layer_{i}_{layer}_{nbits}")
            layer_to_indicator_vars_mapping[layer][nbits] = var

    return layer_to_indicator_vars_mapping


def formalize_problem(solver,
                      layer_to_indicator_vars_mapping: Dict[str, Dict[int, IntVar]],
                      layer_to_metrics_mapping: Dict[str, Dict[int, float]],
                      layer_to_size_mapping: Optional[Dict[str, Dict[int, float]]],
                      target_weights_memory: Optional[float]):
    layers = layer_to_metrics_mapping.keys()

    # Objective (minimize acc loss)
    objective = solver.Objective()

    # Objective (minimize loss)
    for layer in layers:
        for nbits, indicator in layer_to_indicator_vars_mapping[layer].items():
            objective.SetCoefficient(indicator, layer_to_metrics_mapping[layer][nbits])
    objective.SetMinimization()

    # Constraint of only one indicator == 1 for each layer
    for layer in layers:
        constraint = solver.Constraint(1, 1)  # Enforcing that sum == 1
        for v in layer_to_indicator_vars_mapping[layer].values():
            constraint.SetCoefficient(v, 1)

    # Constraint for model size
    if target_weights_memory is not None:
        weights_memory_constraint = solver.Constraint(-solver.infinity(), target_weights_memory)
    else:
        weights_memory_constraint = None

    for layer in layers:
        for nbits_idx, indicator in layer_to_indicator_vars_mapping[layer].items():
            if target_weights_memory:
                weights_memory_constraint.SetCoefficient(indicator, layer_to_size_mapping[layer][nbits_idx])

    return solver


def _build_index2name_mapping(qm):
    index = 0
    index2name = {}
    ##############################################################################
    # Scan model and disable compression
    ##############################################################################
    for n, m in tqdm(qm.named_modules(), desc='Apply quantization to modules'):
        if is_compressed_layer(m):
            index2name.update({index: (n, m)})
            index += 1

    return index2name


def _compute_quant_float_sizes(compressed_layers_index2name):
    size_map = {layer_index: {} for layer_index in compressed_layers_index2name}
    float_size = 0

    for layer_index, (n, m) in compressed_layers_index2name.items():
        float_size += m.compute_float_size()
        for config_index, compression_config in enumerate(m.pareto_config):
            size_map[layer_index][config_index] = m.compute_size(cfg=compression_config)

    return size_map, float_size


from typing import List, Tuple

from typing import List, Tuple
import numpy as np


def select_points_for_metric(
        all_points: List[Tuple[float, float]],
        budget: int
) -> List[int]:
    """
    Select indices of points for exact metric computation given a budget,
    ensuring selected points are well spaced in the (size, score) space
    for effective interpolation.

    Uses farthest-point sampling on normalized size and score.
    """
    n = len(all_points)
    # budget = max(budget, n//20)
    if budget >= n or budget <= 0:
        return list(range(n))

    # Extract and normalize size and score to [0,1]
    scores = np.array([pt[0] for pt in all_points], dtype=float)
    sizes = np.array([pt[1] for pt in all_points], dtype=float)
    sz_norm = (sizes - sizes.min()) / (np.ptp(sizes) or 1)
    sc_norm = (scores - scores.min()) / (np.ptp(scores) or 1)
    pts = np.stack((sz_norm, sc_norm), axis=1)

    # Farthest-point sampling
    selected = [0]  # start with largest-size point (index 0 since sorted by size desc)
    # compute initial distances to first selected point
    distances = np.linalg.norm(pts - pts[selected[0]], axis=1)

    for _ in range(1, budget):
        # pick point with maximum distance to current set
        next_idx = int(np.argmax(distances))
        selected.append(next_idx)
        # update distances: keep min distance to any selected point
        dist_new = np.linalg.norm(pts - pts[next_idx], axis=1)
        distances = np.minimum(distances, dist_new)

    return sorted(selected)


def build_ips_distance_mapping(compressed_layers_index2name, cc, representative_data_loader,
                               model_manager, output_ref, qm):
    distance_map = {layer_index: {} for layer_index in compressed_layers_index2name}

    # compute global error scores
    for layer_index, (n, m) in tqdm(compressed_layers_index2name.items(),
                                    "Computing per-layer global error for Mixed Precision..."):
        base_points_sqnr = select_points_for_metric(m.pareto, cc.num_inter_points)

        ## Quant only scores
        quant_only_cfgs = {ScorePoint(sorted_lr_idx=None, pareto_idx=i): cfg for i, cfg in
                           enumerate(m.pareto_config) if i in base_points_sqnr}

        if cc.mp_per_channel_cost == MPCost.SQNR:
            score_fn = sqnr_distance
            layer_base_points_scores = compute_batch_score_per_config(quant_only_cfgs, score_fn, representative_data_loader, model_manager, output_ref, m, qm)
            layer_base_points_scores = normalize_per_batch_score(layer_base_points_scores, m.weight.device)
            layer_base_points_scores = list(layer_base_points_scores.values())
        elif cc.mp_per_channel_cost == MPCost.KL:
            score_fn = logits_kl_divergence
            layer_base_points_scores = compute_batch_score_per_config(quant_only_cfgs, score_fn, representative_data_loader, model_manager, output_ref, m, qm)
            layer_base_points_scores = [torch.mean(torch.tensor(scores)).item() for k, scores in
                                        layer_base_points_scores.items()]
        elif cc.mp_per_channel_cost in [MPCost.EWQ, MPCost.HMSE_SUM, MPCost.HMSE_MEAN, MPCost.MSE]:
            if cc.mp_per_channel_cost == MPCost.EWQ:
                weight = channel_entropy(m.weight)
            elif cc.mp_per_channel_cost in [MPCost.HMSE_SUM, MPCost.HMSE_MEAN]:
                weight = m.w_hessian
                reduction = 'sum' if cc.mp_per_channel_cost == MPCost.HMSE_SUM else 'mean'
            elif cc.mp_per_channel_cost == MPCost.MSE:
                weight = torch.ones_like(m.w_hessian, device=m.w_hessian.device)
            layer_base_points_scores = compute_wmse_score_per_config(quant_only_cfgs, m, weight, reduction=reduction)
            layer_base_points_scores = [torch.mean(torch.tensor(scores)).item() for k, scores in
                                        layer_base_points_scores.items()]
        layer_inter_scores = compute_sqnr_interpolation_score(m.pareto, base_points_sqnr, layer_base_points_scores)

        distance_map[layer_index].update({k: score for k, score in enumerate(layer_inter_scores)})

    return distance_map


def run_solver(qmodel, cc, representative_data_loader, model_manager, output_ref):
    with torch.no_grad():
        index2nm = _build_index2name_mapping(qmodel)
        layer_to_size_mapping, float_size = _compute_quant_float_sizes(index2nm)

        print(f"Model Float Size [bytes]: {float_size}")

        layer_to_metrics_mapping = build_ips_distance_mapping(index2nm, cc, representative_data_loader, model_manager,
                                                              output_ref, qmodel)

        print(layer_to_metrics_mapping)

    def opt_func(target_compression_rate):
        solver = pywraplp.Solver.CreateSolver('CBC')  # Use 'CBC' as solver
        layer_to_indicator_vars_mapping = init_lp_vars(solver, layer_to_metrics_mapping)
        target_weights_memory = target_compression_rate * float_size
        lp_problem = formalize_problem(solver,
                                       layer_to_indicator_vars_mapping=layer_to_indicator_vars_mapping,
                                       layer_to_metrics_mapping=layer_to_metrics_mapping,
                                       layer_to_size_mapping=layer_to_size_mapping,
                                       target_weights_memory=target_weights_memory)

        # Solve the problem with a time limit (if applicable)
        lp_problem.SetTimeLimit(SOLVER_TIME_LIMIT * 1000)  # OR-Tools time limit is in milliseconds
        status = lp_problem.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print("Solution found!")
        elif status == pywraplp.Solver.FEASIBLE:
            print("Feasible solution found within time limit.")
        else:
            raise Exception("No solution found.")

        # Retrieve the solution values for the indicators
        indicators_values_per_layer = [
            layer_to_indicator_vars_mapping[layer_idx]
            for layer_idx, (name, m) in index2nm.items()
        ]

        # Ensure exactly one indicator variable is set to 1 per layer
        for layer_indicators in indicators_values_per_layer:
            assert sum(indicator.solution_value() for indicator in layer_indicators.values()) == 1, \
                "ILP solution should include exactly one candidate with indicator value 1 for each layer."

        # Collect compression results for each layer
        compression_results = {
            name: m.pareto_config[
                next(b_idx for b_idx, ind in indicators_values_per_layer[layer_index].items() if
                     ind.solution_value() == 1)
            ]
            for layer_index, (name, m) in index2nm.items()
        }

        res = []
        for k, v in compression_results.items():
            res.append((k, v.bit_width_quantization,))
        print(res)
        return compression_results

    return opt_func


def build_ips_distance_mapping2(compression_wrapper_module, n_channels, cc, hessian_data):

    quant_only_cfgs = {ScorePoint(sorted_lr_idx=None, pareto_idx=i): cfg for i, cfg in
                       enumerate(compression_wrapper_module.pareto_config)}
    cost = generate_mp_cost(hessian_data, compression_wrapper_module.weight, compression_wrapper_module.weight_channel_dim, cc.mp_per_channel_cost)
    layer_base_points_scores = compute_channel_score_per_config(quant_only_cfgs, compression_wrapper_module, cost)
    distance_map = {ch_index: {k.pareto_idx: score[ch_index].item() for k, score in layer_base_points_scores.items()} for ch_index in range(n_channels)}

    return distance_map

def run_solver_per_channel(compression_wrapper_module, cc, hessian_data=None):
    with torch.no_grad():
        n_channels = compression_wrapper_module.weight.shape[0]
        float_size = compression_wrapper_module.compute_float_size()
        channel_to_size_mapping: Dict[int, Dict[int, float]] = {
            ch_index: {
                i: compression_wrapper_module.compute_size(cfg) / n_channels
                for i, cfg in enumerate(compression_wrapper_module.pareto_config)
                if cfg.bit_width_quantization in cc.weight_per_channel_bit_list
            }
            for ch_index in range(n_channels)
        }
        channel_to_metrics_mapping = build_ips_distance_mapping2(compression_wrapper_module, n_channels, cc, compression_wrapper_module.w_hessian)
        print()

    def opt_func(target_compression_rate):
        solver = pywraplp.Solver.CreateSolver('CBC')  # Use 'CBC' as solver
        layer_to_indicator_vars_mapping = init_lp_vars(solver, channel_to_metrics_mapping)
        target_weights_memory = target_compression_rate * float_size
        lp_problem = formalize_problem(solver,
                                       layer_to_indicator_vars_mapping=layer_to_indicator_vars_mapping,
                                       layer_to_metrics_mapping=channel_to_metrics_mapping,
                                       layer_to_size_mapping=channel_to_size_mapping,
                                       target_weights_memory=target_weights_memory)

        # Solve the problem with a time limit (if applicable)
        lp_problem.SetTimeLimit(SOLVER_TIME_LIMIT * 1000)  # OR-Tools time limit is in milliseconds
        status = lp_problem.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print("Solution found!")
        elif status == pywraplp.Solver.FEASIBLE:
            print("Feasible solution found within time limit.")
        else:
            raise Exception("No solution found.")

        # Retrieve the solution values for the indicators
        indicators_values_per_layer = [
            layer_to_indicator_vars_mapping[ch_idx]
            for ch_idx in range(n_channels)
        ]

        # Ensure exactly one indicator variable is set to 1 per layer
        for layer_indicators in indicators_values_per_layer:
            assert sum(indicator.solution_value() for indicator in layer_indicators.values()) == 1, \
                "ILP solution should include exactly one candidate with indicator value 1 for each layer."

        # Collect compression results for each layer
        compression_results = {
            ch_index: compression_wrapper_module.pareto_config[
                next(b_idx for b_idx, ind in indicators_values_per_layer[ch_index].items() if
                     ind.solution_value() == 1)
            ]
            for ch_index in range(n_channels)
        }

        res = []
        for k, v in compression_results.items():
            res.append((k, v.bit_width_quantization,))
        print(res)
        return compression_results

    return opt_func