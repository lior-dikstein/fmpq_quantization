import math
from functools import partial
from itertools import product

import numpy as np
import torch

from compression.configs.layer_comrpression_config import LayerCompressionConfig


class CompressionOptions:
    def __init__(self, in_n_out, in_n_in, in_compression_config, device, compression_options=None):
        self.n_out = in_n_out
        self.n_in = in_n_in
        self.base_size = self.n_in * self.n_out
        self.compression_config = in_compression_config

        self.compression_options_list = [LayerCompressionConfig(bit_width_quantization=nbits) for
                                         nbits in
                                         in_compression_config.weight_bit_list] if compression_options is None else \
            compression_options


    def get_compression_options_list(self):
        return self.compression_options_list

    def get_quantization_only_compression(self, bit_width):
        return [c for c in self.compression_options_list if
                c.bit_width_quantization == bit_width][0]

    def _build_compression_options(self):
        in_bit_widths = self.compression_config.weight_bit_list
        bit_for_lr = [b for b in in_bit_widths]
        if self.compression_config.two_bit_quant_only:
            bit_for_lr.remove(2)
        if self.compression_config.three_bit_quant_only:
            bit_for_lr.remove(3)
        ##########################################################

    @staticmethod
    def _simd_fn(in_d, simd):
        return math.ceil(in_d / simd) * simd
