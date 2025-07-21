import operator
from copy import deepcopy
from functools import reduce

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from compression.configs.compression_config import CompressionConfig, ABSPLIT, SVDScores, \
    ThresholdMethod, ParetoCost
from compression.configs.layer_comrpression_config import LayerCompressionConfig
from compression.module.compression_options import CompressionOptions
from compression.ordering import generate_point_ordering
from compression.quantization.adaround_utils import modified_soft_quantization
from compression.quantization.quantization import uniform_quantization, search_weights_scale_perc
from constants import LAYER_COMPRESSION_CONFIG, SIZE, MSE
from utils.memory_utils import get_gpu_memory_map

BYTE_SCALE = 8


class CompressionWrapper(nn.Module):
    def __init__(self, in_module, in_compression_config: CompressionConfig, node_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.add_module("base_module", in_module.to(in_module.weight.device))
        except Exception as e:
            get_gpu_memory_map()
            raise torch.OutOfMemoryError(e)

        try:
            self.add_module("pre_base_module", deepcopy(in_module))
        except Exception as e:
            get_gpu_memory_map()
            raise torch.OutOfMemoryError(e)
        self.pre_base_module.bias = None
        self.is_matmul = isinstance(in_module, nn.Linear)
        self.weight_channel_dim = 0
        self.num_channels = in_module.out_features if self.is_matmul else in_module.out_channels
        self.node_name = node_name
        self.is_conv = False
        grouped_conv = False
        if isinstance(in_module, nn.Conv2d):
            self.is_conv = True
            grouped_conv = in_module.groups > 1
            self.is_matmul = (in_module.kernel_size[0] == 1 and in_module.kernel_size[1] == 1)
        self.grouped_conv = grouped_conv

        self.register_parameter('weight', nn.Parameter(in_module.weight.clone().detach()))

        self.output_channels = self.weight.shape[0]
        self.max_bit_width = max(in_compression_config.weight_bit_list)

        self.n_out = self.weight.shape[0]
        self.n_in = self.weight.shape[1]

        self.pareto_config = []
        self.pareto = []
        self.res_q = None
        self.res_lrq = None

        self.compression_active = False

        self.w_hessian = torch.ones_like(self.weight)
        self.f_score = None
        self.compression_options = None

        # AdaRound parameters
        self.reconstructed = False
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.qparam_train = False

        self.train_weight_scale = False
        self.inited = False
        self.channel_wise = True
        self.store_input = None
        self.bias = in_module.bias
        if self.is_conv:
            self.stride = in_module.stride
            self.padding = in_module.padding
            self.dilation = in_module.dilation
            self.groups = in_module.groups
        else:
            self.out_features = in_module.out_features

    def init_layer_reconstruction(self):
        self.register_buffer('original_weights', self.base_module.weight.clone().detach())
        _alpha = self._init_alpha(self.original_weights.data.detach(),
                                  self.base_n_bits, self.delta.detach())
        self.register_parameter('alpha', nn.Parameter(_alpha))
        self.reconstructed = True

    def _init_alpha(self, w: torch.Tensor, b: int, th: torch.Tensor, eps=1e-8):
        delta = th
        w_floor = torch.floor(w / (delta + eps))
        rest = (w / (delta + eps)) - w_floor  # rest of rounding [0, 1)
        alpha = -self._safe_log((self.zeta - self.gamma) / (rest - self.gamma) - 1, 1e-16)  # => sigmoid(alpha) = rest
        return alpha

    def init_layer_compression(self, in_compression_config: CompressionConfig, output_ref,
                               representative_data_loader=None, qm=None, model_manager=None,
                               hessian_mp=False, debug=False,
                               config_to_set=None):

        if in_compression_config.threshold_method == ThresholdMethod.HMSE:
            hessian = self.w_hessian.to(self.get_weights_device())
        else:
            hessian = None

        if config_to_set is None:
            in_bit_widths = in_compression_config.weight_bit_list
            self.compression_options = CompressionOptions(self.n_out, self.n_in, in_compression_config, self.weight.device)
            with torch.no_grad():
                in_weights = self.weight
                base_size = self.n_in * self.n_out

                self._init_weights_quantization_for_candidates(in_weights, hessian=hessian, debug=debug)

                hessian = self.w_hessian.to(self.get_weights_device())
                pareto, pareto2, mse, deltas, zero_points = generate_point_ordering(self.reshape_per_channel(),
                                                          self.compression_options,
                                                          base_size, self.n_in, self.n_out,
                                                          in_a=None, in_b=None,
                                                          in_cc=in_compression_config,
                                                          hessian_for_pareto=hessian,
                                                          weight_channel_dim=self.weight_channel_dim,
                                                          max_candidates=in_compression_config.max_candidates,
                                                          simd=in_compression_config.simd)
                print(len(pareto2))
            self.mse = mse
            self.deltas = deltas
            self.zero_points = zero_points
            if in_compression_config.weights_mp_per_ch:
                self.pareto_config = [p[LAYER_COMPRESSION_CONFIG] for p in pareto2]
                self.pareto = [[p[MSE], p[SIZE]] for p in pareto2]
            else:
                self.pareto_config = [p[2] for p in pareto]
                self.pareto = [p[:2] for p in pareto]

        else:
            # given config to set
            if not hessian_mp:
                hessian = torch.ones_like(self.weight)
            if len(config_to_set) == 1:
                bit_width = config_to_set[0]
                # assert bit_width in in_compression_config.weight_bit_list
                delta, zero_point = search_weights_scale_perc(x=self.weight, n_bits=bit_width,
                                                              channel_wise=self.channel_wise,
                                                              hessian=hessian)
                co = LayerCompressionConfig(bit_width_quantization=bit_width)
                co.set_weights_params(delta, zero_point)

            else:
                raise Exception("Unexpected config to set")

            self.compression_options = CompressionOptions(self.n_out, self.n_in, in_compression_config,
                                                          self.weight.device,
                                                          compression_options=[co])

    def _init_weights_quantization_for_candidates(self, in_weights, hessian, debug=False):
        for co in tqdm(self.compression_options.get_compression_options_list(),
                       "Running param search for compression candidates"):
            if debug:
                shape = (in_weights.size(0),) + (1,) * (in_weights.ndim - 1)
                delta = torch.ones(shape, dtype=in_weights.dtype, device=in_weights.device)
                zero_point= torch.ones(shape, dtype=in_weights.dtype, device=in_weights.device)
            else:
                delta, zero_point = search_weights_scale_perc(x=in_weights, n_bits=co.bit_width_quantization,
                                                              channel_wise=self.channel_wise, hessian=hessian)
            co.set_weights_params(delta, zero_point)

    def set_compression_config(self, compression_config: LayerCompressionConfig):
        dtype = self.weight.dtype
        if compression_config.delta is None:
            if hasattr(self, 'delta'):
                del self.delta
        else:
            self.register_parameter('delta', nn.Parameter(compression_config.delta.type(dtype)))

        if compression_config.zero_point is None:
            if hasattr(self, 'zero_point'):
                del self.zero_point
        else:
            self.register_parameter('zero_point', nn.Parameter(compression_config.zero_point.type(dtype)))

        self.base_n_bits = compression_config.bit_width_quantization

    def compress_weight(self):
        if self.reconstructed:
            gradient_factor = self._lsq_g_factor(self.original_weights,
                                                 self.base_n_bits) if self.train_weight_scale else 1.0
            w = modified_soft_quantization(self.original_weights, self.alpha, self.base_n_bits,
                                           self.delta, self.gamma, self.zeta, zero_point=self.zero_point,
                                           training=self.training,
                                           gradient_factor=gradient_factor,
                                           bitwidths=self.base_n_bits)
        else:
            w = uniform_quantization(self.weight, self.delta, self.zero_point, self.base_n_bits)

        return w.type(self.weight.dtype)

    def get_trainable_params(self):
        params = []
        if self.reconstructed:
            params.append(self.alpha)
        else:
            params.append(self.weight)
        bias_params = [self.base_module.bias] if self.base_module.bias is not None else None
        scale_params = [self.delta]

        return params, bias_params, scale_params

    def get_weights_device(self):
        return self.weight.device

    def reshape_per_channel(self):
        return self.weight.reshape([self.output_channels, -1])

    @staticmethod
    def _safe_log(x, eps):
        return torch.log(torch.max(x, torch.Tensor([eps]).to(x.device)))

    def set_train_weight_scale(self):
        self.train_weight_scale = True

    def reset_layer_reconstruction(self):
        if hasattr(self, 'alpha'):
            del self.alpha
        self.weight.data = self.original_weights.data
        self.reconstructed = False

    @property
    def hessian_per_channel(self):
        return self.w_hessian.to(self.get_weights_device()).reshape(
            [self.output_channels, -1]).sum(dim=-1)

    def add_weights_hessian_information(self, w_hessian):
        self.w_hessian = w_hessian

    @staticmethod
    def _lsq_g_factor(w, b):
        return 1 / float(torch.sqrt(torch.numel(w) * (2 ** (b - 1) - 1)))

    def enable_compression(self):
        self.compression_active = True

    def disable_compression(self):
        self.compression_active = False

    @property
    def n_compression_options(self):
        return len(self.pareto_config)

    def compute_float_size(self):
        return self.weight.nbytes

    def compute_size(self, cfg):
        num_elements_per_channel = lambda x: reduce(operator.mul, [s for i, s in enumerate(x.shape) if i != self.weight_channel_dim], 1)
        bits_per_channel = sum(cfg.bit_width_quantization) if isinstance(cfg.bit_width_quantization, (
            list, tuple)) else cfg.bit_width_quantization * self.weight.shape[self.weight_channel_dim]
        return bits_per_channel * num_elements_per_channel(self.weight) / BYTE_SCALE


    def _get_bops_input_scale(self):
        if isinstance(self.base_module, nn.Linear):
            if len(self.input_shape) > 1:
                scale = np.prod(self.input_shape[:-1])
            else:
                scale = 1
        elif isinstance(self.base_module, nn.Conv2d):
            if len(self.input_shape) == 3:  # Assume batch is remove
                scale = ((self.input_shape[1] + self.padding[0]) / self.stride[0]) * (
                        (self.input_shape[2] + self.padding[1]) / self.stride[1])
            else:
                raise Exception()
        else:
            raise Exception("Unexpected op for bops computation")

        return scale

    def compute_float_bops(self):
        scale = self._get_bops_input_scale()
        float_bits = 8 * self.weight.dtype.itemsize
        return scale * float_bits * self.weight.numel() * float_bits

    def compute_bops(self, cfg, act_nbits):
        scale = self._get_bops_input_scale()

        assert cfg.bit_width_quantization is not None
        return scale * cfg.bit_width_quantization * self.weight.numel() * act_nbits

        return (cost_a + cost_b).item()

    def forward(self, x):
        if self.compression_active:
            w = self.compress_weight()
            if self.is_conv:
                return torch.nn.functional.conv2d(x, w, self.base_module.bias, self.base_module.stride,
                                                  self.base_module.padding, self.base_module.dilation,
                                                  self.base_module.groups)
            else:
                return torch.nn.functional.linear(x, w, self.base_module.bias)
        else:
            self.base_module.weight.data = self.weight.detach()
            return self.base_module(x)
