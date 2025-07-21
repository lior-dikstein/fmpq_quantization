from tqdm import tqdm
from typing import Dict

import numpy as np
import torch

from compression.configs.layer_comrpression_config import LayerCompressionConfig
from constants import DEVICE
from debug.save_and_load_variables import load_debug_state
from helpers.timers import SegmentTimer
from helpers.utils import is_compressed_layer, torch_tensor_to_numpy


def _sol_to_config(u_sol, deltas, zero_points, k_set):
    configs = {}
    for n, d, z, s in zip(deltas.keys(), deltas.values(), zero_points.values(), u_sol):
        d = torch.stack(list(d), dim=1)
        z = torch.stack(list(z), dim=1)
        s_not_nan = s[~np.isnan(s)]
        idxs = s_not_nan.astype(int)
        bits_config = np.array(k_set)[idxs]
        layer_cc = LayerCompressionConfig(bit_width_quantization=list(bits_config))
        layer_cc.set_weights_params(
            d[torch.arange(idxs.shape[0]), idxs],
            z[torch.arange(idxs.shape[0]), idxs])
        configs[n] = layer_cc
    return configs

def run_continuous_solver(qmodel, cc):
    w_dict, e_dict, size_dict, deltas, zero_points = {}, {}, {}, {}, {}
    float_size = 0

    for n, m in tqdm(qmodel.named_modules()):
        if is_compressed_layer(m):
            float_size += m.compute_float_size()
            w_dict[n] = m.hessian_per_channel
            e_dict[n] = m.mse.transpose(1,0)
            size_dict[n] = m.n_in
            deltas[n] = m.deltas
            zero_points[n] = m.zero_points
    return continuous_solver(w_dict=w_dict, e_dict=e_dict, size=size_dict,k_set=cc.weight_bit_list, deltas=deltas, zero_points=zero_points, float_size=float_size)


def continuous_solver(w_dict: Dict[str, torch.Tensor],
                      e_dict: Dict[str, torch.Tensor],
                      size: Dict[str, int],
                      deltas: Dict[str, int],
                      zero_points: Dict[str, int],
                      k_set: list,
                      float_size: int):
    """

    w_dict: A dictionary mapping from layer name to a vector of size R^{C_{\ell}} each element weights the specific output channel.
    e_dict: A dictionary mapping from layer name to a vector of size R^{C_{\ell}\times |K|} each element represents the output channel for a specifc bit-width.
    size: A dictionary mapping from layer name to a int represent the number of input channel.
    k_set: A list of bit-width order such that it machs e_dict order.

    """
    layer_name_list = list(size.keys())
    c_out = np.asarray([e_dict[n].shape[0] for n in layer_name_list])
    c_max = np.max(c_out)
    c_in = np.asarray([size[n] for n in layer_name_list])
    max_memory = np.sum(c_in * c_out) * np.max(k_set)
    min_memory = np.sum(c_in * c_out) * np.min(k_set)

    D = np.zeros([len(layer_name_list), c_max, len(k_set)])
    for i, n in enumerate(layer_name_list):
        aa = torch_tensor_to_numpy(w_dict[n].unsqueeze(1) * e_dict[n])
        D[i, :c_out[i], :] = aa
        D[i, c_out[i]:, :] = np.nan

    S = np.zeros([len(layer_name_list), c_max, len(k_set)])
    for i, n in enumerate(layer_name_list):
        S[i, :c_out[i], :] = size[n] * np.ones((1, c_out[i], 1)) * np.asarray(k_set).reshape(1, 1, -1)
        S[i, c_out[i]:, :] = np.nan

    def solve_for_lambda(in_lambda):
        o = D + in_lambda * S
        u = np.argmin(o, axis=-1).astype("float")
        u[np.isnan(np.sum(D, axis=-1))] = np.nan
        return u

    def find_lambda_max():
        lambda_max = 0
        for iba in range(len(k_set)):
            for ibb in range(len(k_set)):
                if iba == ibb:
                    continue
                _lambda = np.nanmax((D[:, :, ibb] - D[:, :, iba]) / (S[:, :, iba] - S[:, :, ibb]))
                lambda_max = np.maximum(_lambda, lambda_max)
        return lambda_max

    def constraint(u):
        u_int = np.copy(u.astype("int"))
        u_int[np.isnan(u)] = np.argmax(k_set)
        _b = np.asarray(k_set)[u_int].astype("float")
        _b[np.isnan(u)] = np.nan
        channel_size = S[:, :, -1] / k_set[-1] * _b
        return np.nansum(channel_size)

    def error(u):
        u_int = np.copy(u.astype("int"))
        u_int[np.isnan(u)] = np.argmax(k_set)

        return np.nansum(np.take_along_axis(D, np.expand_dims(u_int, axis=-1), axis=-1))

    def func(target_compression_rate: float, n_iteration: int = 200):
        target = target_compression_rate * float_size
        if target > max_memory or target < min_memory:
            raise Exception(f"Invalid target compression. Targer memory: {target}, max memory {max_memory}, min memory {min_memory}")
        lambda_max = find_lambda_max()
        lambda_min = 0
        for iter in range(n_iteration):
            _lambda = (lambda_max + lambda_min) / 2
            u = solve_for_lambda(_lambda)
            c = constraint(u)
            if c < target:
                lambda_max = _lambda
            else:
                lambda_min = _lambda
        u_sol = solve_for_lambda(lambda_max)
        u_bound = solve_for_lambda(lambda_min)
        configs = _sol_to_config(u_sol, deltas, zero_points, k_set)
        return configs, error(u_sol),error(u_sol)-error(u_bound)

    return func, min_memory, max_memory


if __name__ == '__main__':
    timer = SegmentTimer()
    load_debug_state('solver_resnet18')
    optimization_function2, _, _ = run_continuous_solver(qmodel, cc)
    compression_results2 = optimization_function2(3 / 32)
    timer.segment('solver')
