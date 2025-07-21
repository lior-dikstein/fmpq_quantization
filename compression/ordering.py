import torch.linalg
from tqdm import tqdm

from compression.configs.compression_config import CompressionConfig, ParetoCost, MPCost
from compression.ordering_v2 import run_pareto2
from compression.quantization.distance_metrics import weighted_mse_quant_loss, channel_entropy
from compression.quantization.quantization import uniform_quantization
from utils.candidates_utils import get_quantization_only_candidates


def run_pareto(in_list_to_pareto):
    _pareto = torch.tensor([p[:2] for p in in_list_to_pareto])
    mse_array = _pareto[:, 0]
    size_array = _pareto[:, 1]

    N = mse_array.size(0)
    mse_array_exp = mse_array.unsqueeze(0).expand(N, N)
    size_array_exp = size_array.unsqueeze(0).expand(N, N)

    # Check for domination conditions
    mse_array_dominated = (mse_array_exp <= mse_array_exp.T)
    size_array_dominated = (size_array_exp <= size_array_exp.T)

    # Exclude self-comparisons
    self_mask = torch.eye(N, dtype=torch.bool, device=mse_array.device)

    # A point is dominated if there exists another point satisfying both conditions
    dominated = torch.any((mse_array_dominated & size_array_dominated) & ~self_mask, dim=1)

    # Select undominated points
    undominated_mask = ~dominated

    res = [in_list_to_pareto[i] for i in range(len(undominated_mask)) if undominated_mask[i]]
    return res


def generate_pareto_cost(weighting, in_w, cost_type: ParetoCost):
    def _cost(w_tilde):

        error = in_w - w_tilde
        if cost_type == ParetoCost.MSE or cost_type == ParetoCost.EWQ:
            return torch.mean(error ** 2)
        elif cost_type == ParetoCost.HMSEPerOutChannel:
            if len(weighting.shape) == 4:
                w_per_out_channel = torch.mean(weighting, dim=[-1, -2]).max(dim=-1, keepdim=True)[0]
            else:
                w_per_out_channel = torch.max(weighting, dim=-1, keepdim=True)[0]
            return torch.mean(w_per_out_channel * error ** 2)  # Ordering score
        else:
            raise NotImplemented
    return _cost


def generate_point_ordering(in_weights, compression_options, base_size, n_in, n_out, in_a, in_b,
                            in_cc: CompressionConfig, hessian_for_pareto, weight_channel_dim, max_candidates, simd):

    local_pareto_scores = []
    local_pareto_scores2 = []
    cost = generate_pareto_cost(hessian_for_pareto, in_weights, in_cc.pareto_cost)
    cost2 = generate_mp_cost(hessian_for_pareto.reshape(in_weights.shape[0], -1), in_weights, weight_channel_dim, in_cc.pareto_cost)


    # Compute MSE cost for quantization only candidates
    quantization_only_candidates = get_quantization_only_candidates(in_cc.weight_bit_list, compression_options)
    for _, lcc in quantization_only_candidates.items():
        w_q = uniform_quantization(in_weights,
                                   lcc.delta.reshape((lcc.delta.shape[0], -1)),
                                   lcc.zero_point.reshape((lcc.zero_point.shape[0], -1)),
                                   lcc.bit_width_quantization)
        cost_q = cost(w_q).item()
        cost2_q = cost2(w_q)
        size_q = base_size * lcc.bit_width_quantization
        local_pareto_scores.append([cost_q, size_q, lcc])
        local_pareto_scores2.append([cost2_q, lcc])

    # Compute Pareto front
    pareto = run_pareto(local_pareto_scores)
    entropy = channel_entropy(in_weights)
    hessian_per_channel = hessian_for_pareto.mean(dim=list(range(len(hessian_for_pareto.shape)))[1:])
    pareto2 = run_pareto2(in_cc.candidate_search_alg, local_pareto_scores2, entropy, hessian_per_channel, n_in, max_candidates, simd)
    deltas = [p[1].delta for p in local_pareto_scores2]
    zero_points = [p[1].zero_point for p in local_pareto_scores2]
    x = [p for p in local_pareto_scores if p not in pareto]
    pareto += x
    mse = torch.stack([l[0] for l in local_pareto_scores2], 0)
    return pareto, pareto2, mse, deltas, zero_points



def generate_mp_cost(weighting, in_w, channel_dim, cost_type: ParetoCost):
    def _cost(w_tilde):

        if cost_type == ParetoCost.MSE:
            return channelwise_mse(w_tilde, in_w, channel_dim)
        elif cost_type == ParetoCost.HMSEPerOutChannel:
            return channelwise_mse(w_tilde, in_w, channel_dim, weighting)
        elif cost_type == ParetoCost.EWQ:
            return channelwise_entropy_weighted_quant_loss(in_w, w_tilde, channel_dim=channel_dim, reduction="none")
        else:
            raise NotImplemented
    return _cost


def channelwise_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        channel_dim: int = 1,
        weight: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes MSE between two tensors across all dimensions except the specified channel dimension,
    with optional weighting.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (batch_size, num_channels, ...).
        target (torch.Tensor): Ground truth tensor with the same shape as pred.
        channel_dim (int): Dimension index corresponding to the channels. Default is 1.
        weight (torch.Tensor, optional): Weight tensor of same shape as pred. Defaults to None.

    Returns:
        torch.Tensor: Tensor of shape (num_channels,) containing MSE for each channel.
    """
    # Ensure the input tensors have the same shape
    assert pred.shape == target.shape, "Input tensors must have the same shape."
    if weight is not None:
        assert weight.shape == pred.shape, "Weight tensor must have the same shape as input tensors."

    # Move channel dimension to first dimension if not already
    if channel_dim != 0:
        pred = pred.transpose(0, channel_dim)
        target = target.transpose(0, channel_dim)
        if weight is not None:
            weight = weight.transpose(0, channel_dim)

    # Flatten all dimensions except channels
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    if weight is not None:
        weight_flat = weight.reshape(weight.shape[0], -1)

    # Compute weighted MSE for each channel
    diff = pred_flat - target_flat
    if weight is not None:
        mse = torch.sum(weight_flat * (diff ** 2), dim=1) / torch.sum(weight_flat, dim=1).clamp(min=1e-8)
    else:
        mse = torch.mean(diff ** 2, dim=1)

    return mse


def channelwise_entropy_weighted_quant_loss(
    fp_weight: torch.Tensor,
    q_weight: torch.Tensor,
    num_bins: int = 256,
    reduction: str = "mean",
    channel_dim: int = 0
) -> torch.Tensor:
    """
    Entropy-Weighted Quantisation Loss (EWQ), per-channel along arbitrary dimension.

    EWQ = Σ_c H_c · ||ΔW_c||²,
    where ΔW = fp_weight − q_weight and H_c is the Shannon entropy of channel c.

    Parameters
    ----------
    fp_weight : torch.Tensor
        Full-precision weights, shape (..., C, ...).
    q_weight  : torch.Tensor
        Quantised weights of identical shape.
    num_bins  : int
        Histogram bins used to estimate channel entropy.
    reduction : {"sum", "mean", "none"}
        Reduction over channels.
    channel_dim : int
        Dimension index corresponding to channels.

    Returns
    -------
    torch.Tensor
        Scalar if reduction != "none", else per-channel loss of shape (C,).
    """
    if fp_weight.shape != q_weight.shape:
        raise ValueError("fp_weight and q_weight must have the same shape.")

    # Move channel dimension to front
    if channel_dim != 0:
        fp = fp_weight.transpose(0, channel_dim)
        q  = q_weight.transpose(0, channel_dim)
    else:
        fp, q = fp_weight, q_weight

    # Flatten all dims except channel
    delta = (fp - q).reshape(fp.shape[0], -1)              # (C, N)
    mse_per_ch = (delta ** 2).sum(dim=1)                   # (C,)

    # Compute per-channel entropy on full-precision weights
    entropy = _channel_entropy(fp, num_bins=num_bins)      # (C,)

    # Weighted per-channel loss
    ewq = entropy * mse_per_ch                              # (C,)

    # Reduction
    if reduction == "sum":
        return ewq.sum()
    elif reduction == "mean":
        return ewq.mean()
    elif reduction == "none":
        return ewq
    else:
        raise ValueError("reduction must be one of {'sum','mean','none'}.")