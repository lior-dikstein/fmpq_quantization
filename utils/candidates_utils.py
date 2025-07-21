from itertools import product
from typing import NamedTuple, Optional


class CandidateKey(NamedTuple):
    w_nbits: Optional[int] = None

def get_quantization_only_candidates(weight_bit_list, compression_options):
    return {CandidateKey(w_nbits=n): compression_options.get_quantization_only_compression(n) for n in weight_bit_list}
