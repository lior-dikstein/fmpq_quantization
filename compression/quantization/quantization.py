"""
 Some functions in this file is copied from https://github.com/zkkli/RepQ-ViT or https://github.com/zysxmu/ERQ
 and modified for this project's needs.
"""
import numpy as np
import torch.linalg

from constants import DEVICE


def two_pow(x):
    """
    Compute 2**x where x can be int, float, list, tuple, or torch.Tensor.
    """
    if isinstance(x, (int, float)):
        x = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, (list, tuple)):
        x = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        pass  # already a tensor
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

    return torch.pow(2, x)


def ste_round(x: torch.Tensor, gradient_factor=1.0) -> torch.Tensor:
    """
    Return the rounded values of a tensor.
    """
    return (torch.round(x) - x * gradient_factor).detach() + x * gradient_factor


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


def uniform_quantize_min_max(x, max, min, n_bits):
    delta = (max - min) / (2 ** n_bits - 1)
    zero_point = (- min / delta).round()
    # we assume weight quantization is always signed
    x_int = torch.round(x / delta)
    x_quant = torch.clamp(x_int + zero_point, 0, 2 ** n_bits - 1)
    x_float_q = (x_quant - zero_point) * delta
    return x_float_q


def uniform_quantization(x, delta, zero_point, n_bits):
    if (isinstance(n_bits, int) and n_bits == 1) or (
            isinstance(n_bits, torch.Tensor) and torch.all(n_bits == 1)
    ):
        # ----- binary branch -----
        # STE – keep gradient ≈ identity.
        x_int = ste_round(torch.sign(x))  # {-1, +1}
        # Optional: allow zeros when x==0 to avoid NaN gradients
        x_int[x == 0] = 0
        return x_int * delta  # {-Δ, 0, +Δ}

    n_levels = two_pow(n_bits).to(DEVICE)
    x_int = torch.round(x / delta) + zero_point

    # Compute max values for clamping:
    if isinstance(n_levels, torch.Tensor) and n_levels.dim() > 0:
        # per‐channel max: shape [C,1] for broadcasting across W
        max_vals = n_levels - 1
        max_vals = max_vals.view(max_vals.shape[0], *([1] * (x_int.ndim - 1)))
        min_vals = torch.zeros_like(max_vals, dtype=x_int.dtype, device=x_int.device)
    else:
        # scalar max
        max_vals = n_levels - 1
        min_vals = 0

    # Clamp each channel between 0 and its max
    x_quant = torch.clamp(x_int, min=min_vals, max=max_vals)
    x_dequant = (x_quant - zero_point) * delta
    return x_dequant


def search_weights_scale_perc(x: torch.Tensor, n_bits, hessian=None, channel_wise: bool = False, x_ref=None,
                              x_complement=None,
                              new_mode=False):
    delta, zero_point = None, None
    if channel_wise:
        x_clone = x.clone().detach()
        n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]

        use_n_bits_per_channel = False
        if isinstance(n_bits, (list, tuple)):
            if len(n_bits) == 1:
                n_bits = n_bits[0]
            else:
                assert len(n_bits) == n_channels
                use_n_bits_per_channel = True

        if len(x.shape) == 4:
            x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
        elif len(x.shape) == 2:
            x_max = x_clone.abs().max(dim=-1)[0]
        elif len(x.shape) == 3:
            x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
        else:
            raise NotImplementedError

        delta = x_max.clone()
        zero_point = x_max.clone()
        # determine the scale and zero point channel-by-channel
        for c in range(n_channels):
            n_bits_channel = n_bits[c] if use_n_bits_per_channel else n_bits
            if hessian is not None:
                if len(x.shape) == 3:
                    c_hessian = hessian[:, :, c]
                else:
                    c_hessian = hessian[c]
            else:
                c_hessian = None

            x_ref_c = None
            if len(x.shape) == 3:
                if x_ref is not None:
                    x_ref_c = x_ref[:, :, c]
                delta[c], zero_point[c] = search_weights_scale_perc(x_clone[:, :, c], n_bits=n_bits_channel,
                                                                    channel_wise=False, x_ref=x_ref_c,
                                                                    x_complement=x_complement,
                                                                    hessian=c_hessian, new_mode=new_mode)
            else:
                if x_ref is not None:
                    x_ref_c = x_ref[c]
                delta[c], zero_point[c] = search_weights_scale_perc(x_clone[c], n_bits=n_bits_channel,
                                                                    channel_wise=False, x_ref=x_ref_c,
                                                                    x_complement=x_complement,
                                                                    hessian=c_hessian, new_mode=new_mode)
        if len(x.shape) == 4:
            delta = delta.view(-1, 1, 1, 1)
            zero_point = zero_point.view(-1, 1, 1, 1)
        elif len(x.shape) == 2:
            delta = delta.view(-1, 1)
            zero_point = zero_point.view(-1, 1)
        elif len(x.shape) == 3:
            delta = delta.view(1, 1, -1)
            zero_point = zero_point.view(1, 1, -1)
        else:
            raise NotImplementedError
    else:
        x_clone = x.clone().detach()
        if x_ref is not None:
            x_ref = x_ref.clone().detach()
            x_complement = x_complement.clone().detach()

        best_score = 1e+10
        pct_dict = {8: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    7: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    6: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    5: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    4: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    3: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    2: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1],
                    1: [0.97, 0.98, 0.99, 0.995, 0.9995, 0.9997, 0.9999, 0.99995, 0.99999, 1]}

        for pct in pct_dict.get(n_bits):
            try:
                new_max = torch.quantile(x_clone.reshape(-1), pct)
                new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
            except:
                new_max = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
                new_min = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = uniform_quantize_min_max(x_clone, new_max, new_min, n_bits)

            if x_ref is None:
                if hessian is None:
                    score = lp_loss(x_clone, x_q, p=2, reduction='all')
                else:
                    score = (torch.sqrt(hessian) * ((x_clone - x_q).abs())).pow(2).mean()
            else:
                if hessian is None:
                    score = lp_loss(x_ref, x_q @ x_complement, p=2, reduction='all')
                else:
                    score = (torch.sqrt(hessian) * ((x_ref - x_q @ x_complement).abs())).pow(2).mean()
            if score < best_score:
                best_score = score
                delta = (new_max - new_min) / (2 ** n_bits - 1)
                zero_point = (- new_min / delta).round()
    return delta, zero_point


if __name__ == '__main__':
    x = torch.randn(64, 3, 55, 55).to(DEVICE)
    n_bits = 1
    delta, zero_point = search_weights_scale_perc(x, n_bits)
    xq = uniform_quantization(x, delta, zero_point, n_bits)
    delta = x.amax(dim=tuple(range(1, x.ndim)))-x.amin(dim=tuple(range(1, x.ndim)))
    delta = delta.view(delta.shape[0], *([1] * (x.ndim - 1))).to(DEVICE)
    zero_point = torch.zeros_like(delta).to(DEVICE)
    # Define options
    options = torch.tensor([2, 4, 8])

    # Number of elements you want
    size = 64  # for example

    # Sample random indices and map to options
    n_bits = options[torch.randint(0, len(options), (size,))]
    xq = uniform_quantization(x, delta, zero_point, n_bits.to(DEVICE))