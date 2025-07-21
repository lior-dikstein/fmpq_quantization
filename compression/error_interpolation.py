from itertools import product
from typing import NamedTuple, Union, List, Tuple

import numpy as np
import torch

from compression.configs.compression_config import ParetoCost
from compression.ordering import generate_mp_cost
from compression.quantization.distance_metrics import weighted_mse_quant_loss, channel_entropy

"""
sorted_lr_idx: the index of the config option in a sorted per-ab-bitwidth options list
pareto_idx: the index of the config in the module's pareto_config list
"""
ScorePoint = NamedTuple('ScorePoint', [('sorted_lr_idx', Union[int, None]), ('pareto_idx', int)])

def compute_channel_score_per_config(base_compression_options,
                                     compression_wrapper_module,
                                     cost):
    """
    base_compression_options is ScorePoint --> config
    """

    channel_scores_per_cfg = {}

    with torch.no_grad():
        compression_wrapper_module.enable_compression()
        for k, compression_config in base_compression_options.items():
            compression_wrapper_module.set_compression_config(compression_config)  # Set compression config
            score = cost(compression_wrapper_module.compress_weight())
            channel_scores_per_cfg[k] = score
        compression_wrapper_module.disable_compression()

    return channel_scores_per_cfg


def compute_batch_score_per_config(base_compression_options, score_fn, representative_data_loader, model_manager, output_ref,
                                   compression_wrapper_module, qm):
    """
    base_compression_options is ScorePoint --> config
    """

    batch_scores_per_cfg = {k: [] for k in base_compression_options}

    with torch.no_grad():
        compression_wrapper_module.enable_compression()
        for batch_idx, batch in enumerate(representative_data_loader):
            data = model_manager.data_to_device(batch)
            batch_output_ref = output_ref[batch_idx].to(model_manager.device)
            for k, compression_config in base_compression_options.items():
                compression_wrapper_module.set_compression_config(compression_config)  # Set compression config
                score = score_fn(qm, data, batch_output_ref, model_manager)
                batch_scores_per_cfg[k].append(score)
        compression_wrapper_module.disable_compression()

    return batch_scores_per_cfg


def normalize_per_batch_score(batch_scores_per_cfg, device):
    """
    batch_scores_per_cfg is ScorePoint --> List of per batch scores
    """
    return {
        k: torch.mean(torch.tensor([s[0] for s in scores], device=device)).item()
                  / torch.mean(torch.tensor([s[1] for s in scores], device=device)).item()
        for k, scores in batch_scores_per_cfg.items()}


def compute_sqnr_for_base_candidates(inter_points, representative_data_loader, model_manager, output_ref,
                                     m, qm):
    batch_scores_per_cfg = compute_batch_score_per_config(inter_points, representative_data_loader,
                                                          model_manager, output_ref, m, qm)

    sqnr_per_base_config = normalize_per_batch_score(batch_scores_per_cfg, m.weight.device)

    return sqnr_per_base_config


def compute_sqnr_interpolation_score2(
    all_points: List[Tuple[float, float]],
    selected_indices: List[int],
    selected_metrics: List[float],
    power: float = 2.0,
    eps: float = 1e-6
) -> List[float]:
    """
    Interpolate metric values from selected points to all points using inverse-distance weighting.

    Args:
        all_points: list of (size, score) tuples.
        selected_indices: indices in all_points with known metric values.
        selected_metrics: metric values corresponding to selected_indices.
        power: exponent for distance weighting (defaults to 2).
        eps: small constant to avoid division by zero.

    Returns:
        List of interpolated metric values for every point in all_points.
    """
    # extract and normalize size and score to [0,1]
    sizes = np.array([pt[1] for pt in all_points], dtype=float)
    scores = np.array([pt[0] for pt in all_points], dtype=float)
    sz_norm = (sizes - sizes.min()) / (np.ptp(sizes) or 1)
    sc_norm = (scores - scores.min()) / (np.ptp(scores) or 1)
    pts = np.stack((sz_norm, sc_norm), axis=1)  # shape (n,2)

    # selected point coordinates and values
    sel_pts = pts[selected_indices]            # shape (m,2)
    sel_vals = np.array(selected_metrics, dtype=float)  # shape (m,)

    # compute distances from every point to each selected point
    diff = pts[:, None, :] - sel_pts[None, :, :]  # shape (n,m,2)
    dist = np.linalg.norm(diff, axis=2)           # shape (n,m)

    # inverse-distance weights
    weights = 1.0 / ((dist + eps) ** power)       # shape (n,m)

    # weighted average interpolation
    interp = (weights * sel_vals[None, :]).sum(axis=1) / weights.sum(axis=1)

    # assign exact values for selected points
    interp[selected_indices] = sel_vals

    return interp.tolist()


def compute_sqnr_interpolation_score(
    all_points: List[Tuple[float, float]],
    selected_indices: List[int],
    selected_metrics: List[float]
) -> List[float]:
    """
    Linearly interpolate metric values based on size for all points.

    Args:
        all_points: list of (size, score) tuples, sorted by size descending.
        selected_indices: indices with known metric values.
        selected_metrics: metric values for selected points.

    Returns:
        List of interpolated metric values for all points.
    """
    # extract sizes
    sizes = np.array([pt[1] for pt in all_points], dtype=float)

    # sort selected points by index to preserve order
    sel_idx_sorted, sel_vals_sorted = zip(*sorted(zip(selected_indices, selected_metrics)))
    sel_sizes = sizes[list(sel_idx_sorted)]
    sel_vals = np.array(sel_vals_sorted)

    # perform piecewise linear interpolation (endpoints are extrapolated as constant)
    interp = np.interp(sizes, sel_sizes, sel_vals)

    return list(interp)


def compute_wmse_score_per_config(base_compression_options, compression_wrapper_module, mse_weight, reduction='mean'):
    """
    base_compression_options is ScorePoint --> config
    """

    scores_per_cfg = {k: [] for k in base_compression_options}
    float_weight = compression_wrapper_module.reshape_per_channel()
    with torch.no_grad():
        compression_wrapper_module.enable_compression()
        cost = generate_mp_cost(mse_weight.reshape(float_weight.shape[0], -1), float_weight, 0,
                                 ParetoCost.HMSEPerOutChannel)

        for k, compression_config in base_compression_options.items():
            compression_wrapper_module.set_compression_config(compression_config)  # Set compression config
            score = cost(compression_wrapper_module.compress_weight().reshape([float_weight.shape[0], -1]))
            if reduction == 'mean':
                ss = score.mean().item()
            elif reduction == 'sum':
                ss = score.sum().item()
            else:
                raise Exception()
            scores_per_cfg[k].append(ss)
        compression_wrapper_module.disable_compression()

    return scores_per_cfg