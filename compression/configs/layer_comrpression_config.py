from dataclasses import dataclass

import torch

from compression.quantization.quantization import uniform_quantization


@dataclass
class LayerCompressionConfig:
    bit_width_quantization: int = None
    threshold_quantization: torch.Tensor = None
    delta: torch.Tensor = None
    zero_point: torch.Tensor = None
    scale: torch.Tensor = None
    scale_inverse: torch.Tensor = None

    def set_weights_params(self, delta, zero_point):
        self.delta, self.zero_point = delta, zero_point

    def size(self, n_in, n_out):
        return n_in * n_out * self.bit_width_quantization