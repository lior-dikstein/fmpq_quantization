import numpy as np
import torch

from compression.module.compression_wrapper import CompressionWrapper
from compression.quantization.activation_quantization import ActivationQuantizer


def is_compressed_layer(module, name=None):
    return module is not None and isinstance(module, CompressionWrapper)


def is_quantized_activation(module):
    return module is not None and isinstance(module, ActivationQuantizer)


def torch_tensor_to_numpy(tensor: torch.Tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, list):
        return [torch_tensor_to_numpy(t) for t in tensor]
    elif isinstance(tensor, tuple):
        return tuple([torch_tensor_to_numpy(t) for t in tensor])
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().contiguous().numpy()
    else:
        raise Exception(f'Conversion of type {type(tensor)} to {type(np.ndarray)} is not supported')
