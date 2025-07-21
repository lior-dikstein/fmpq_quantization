import torch
import torch.nn.functional as F

def logits_kl_divergence(
        in_model,
        data,
        output_ref: torch.Tensor,
        model_manager,
        reduction: str = "batchmean"
) -> torch.Tensor:
    """
    Compute the Kullback–Leibler divergence KL(P‖Q) between the soft-max
    distributions induced by two sets of logits.

    Args
    ----
    quant_logits : torch.Tensor
        Raw (unnormalised) logits from the *quantised* or perturbed model,
        shape (B, C, …), where B is the batch size and C the number of classes.
    ref_logits   : torch.Tensor
        Raw logits from the *full-precision* reference model with identical
        shape to `quant_logits`.
    reduction    : str, default "batchmean"
        Reduction over the batch. Options follow `torch.nn.KLDivLoss`
        (“none”, “batchmean”, “sum”, “mean”).

    Returns
    -------
    torch.Tensor
        Scalar KL divergence if `reduction` ≠ "none", else the per-sample KL
        values with shape (B,).

    Notes
    -----
    * KL(P‖Q) is computed, where P = softmax(ref_logits) and
      Q = softmax(quant_logits).
    * Uses log-softmax for numerical stability and delegates reduction to
      `torch.nn.functional.kl_div`.
    """
    output = model_manager.forward(in_model, data)
    quant_logits = output['logits'] if isinstance(output, dict) else output
    ref_logits = output_ref['logits'] if isinstance(output, dict) else output_ref
    ref_logits = ref_logits.to(model_manager.device)

    # Convert logits to probability/log-probability distributions
    log_q = F.log_softmax(quant_logits, dim=1)
    p     = F.softmax(ref_logits,   dim=1)

    # KL divergence KL(P‖Q)
    kl = F.kl_div(log_q, p, reduction=reduction)

    return kl.item()

def sqnr_distance(in_model, data, output_ref, model_manager):
    output = model_manager.forward(in_model, data)
    out_ref = output_ref['logits'] if isinstance(output, dict) else output_ref
    out_ref = out_ref.to(model_manager.device)

    if isinstance(output, dict):
        delta = torch.mean(((output['logits'] - out_ref)/out_ref.max()) ** 2)
        norm = torch.mean((out_ref/out_ref.max()) ** 2)
    else:
        delta = torch.mean((output - out_ref) ** 2)
        norm = torch.mean(out_ref ** 2)

    return delta.item(), norm.item()



def channel_entropy(
    w: torch.Tensor,
    num_bins: int = 256,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Shannon entropy (bits) of each output‐channel weight distribution.

    Parameters
    ----------
    w : torch.Tensor
        Float-precision weights, shape (C, *), where the first dim is the
        output channel.
    num_bins : int, default 256
        Histogram bins.
    eps : float, default 1e-12
        Numerical stability.

    Returns
    -------
    torch.Tensor
        Entropy per channel, shape (C,).
    """
    C = w.shape[0]
    ent = torch.empty(C, device=w.device, dtype=w.dtype)

    # Common bin edges per layer (min/max over all channels)
    layer_min, layer_max = w.min().item(), w.max().item()
    bin_edges = torch.linspace(layer_min, layer_max, num_bins + 1, device=w.device)

    for c in range(C):
        # Histogram counts → probability mass
        hist = torch.histc(w[c].view(-1), bins=num_bins, min=layer_min, max=layer_max)
        p = hist / (hist.sum() + eps)
        ent[c] = -(p * (p + eps).log2()).sum()

    return ent


def weighted_mse_quant_loss(
    fp_weight: torch.Tensor,
    q_weight: torch.Tensor,
    mse_weight: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Weighted Mean Squared Error (MSE) Quantization Loss.

    The loss is computed as:
    WMSE = Σ_c w_c · ||ΔW_c||² ,
    where ΔW = fp_weight − q_weight and
    w_c is the provided weight for channel c.

    Parameters
    ----------
    fp_weight : torch.Tensor
        Full-precision weights, shape (C, *).

    q_weight : torch.Tensor
        Quantized weights of identical shape.

    mse_weight : torch.Tensor
        Per-channel weighting factors, shape (C,).

    reduction : {"sum", "mean", "none"}, default "mean"
        Specifies the reduction to apply over channels:
        - "sum": Sum of all channel-wise losses.
        - "mean": Mean of all channel-wise losses.
        - "none": No reduction, returns per-channel losses.

    Returns
    -------
    torch.Tensor
        Weighted MSE loss as a scalar if reduction ≠ "none", else per-channel loss (C,).

    Notes
    -----
    * Assumes first dimension is the output channel (Conv/Linear).
    """
    if fp_weight.shape != q_weight.shape:
        raise ValueError("fp_weight and q_weight must have the same shape.")

    delta = (fp_weight - q_weight).view(fp_weight.shape[0], -1)  # (C, N)
    mse = delta ** 2
    mse_per_ch = (mse ** 2).sum(dim=1)                         # (C,)
    if mse_weight.shape == mse_per_ch.shape:
        wmse = mse_weight * mse_per_ch                              # (C,)
    elif fp_weight.shape == mse_weight.shape:
        mse_weight = mse_weight.view(fp_weight.shape[0], -1)  # (C, N)
        wmse = mse_weight * mse
    else:
        raise ValueError("mse and mse_weight must have the same shape.")

    if reduction == "sum":
        return wmse.sum()
    elif reduction == "mean":
        return wmse.mean()
    elif reduction == "none":
        return wmse
    else:
        raise ValueError("reduction must be one of {'sum','mean','none'}.")