from dataclasses import dataclass
from typing import List
from enum import Enum


class ABSPLIT(Enum):
    US_V = "US_V"
    U_SV = "U_SV"


class ThresholdMethod(Enum):
    MSE = 'MSE'
    HMSE = 'HMSE'


class SVDScores(Enum):
    ID = 'ID'
    LFH = 'LFH'


class SolverType(Enum):
    ILP = 'ILP'
    CONTINUOUS = "CONTINUOUS"

class CandidateSearchAlg(Enum):
    GREEDY = 'GREEDY'
    LAMBDA = 'LAMBDA'
    LAMBDA_GROUP_BY_MSE = 'LAMBDA_GROUP_BY_MSE'
    LAMBDA_GROUP_BY_ENTROPY = 'LAMBDA_GROUP_BY_ENTROPY'
    LAMBDA_GROUP_BY_HESSIAN = 'LAMBDA_GROUP_BY_HESSIAN'
    LAMBDA_GROUP_RANDOM_PARETO = 'LAMBDA_GROUP_RANDOM_PARETO'
    BOA_STAR = 'BOA_STAR'
    EPS_BOA_STAR = 'EPS_BOA_STAR'
    RANDOM_STEP = 'RANDOM_STEP'


class ParetoCost(Enum):
    MSE = 'MSE'
    EWQ = 'EWQ'
    HMSEPerOutChannel = 'HMSEPerOutChannel'


class MPCost(Enum):
    MSE = 'MSE'
    HMSE_SUM = 'HMSE_SUM'
    HMSE_MEAN = 'HMSE_MEAN'
    HMSE_CONTINUOUS = 'HMSE_CONTINUOUS'
    SQNR = 'SQNR'
    KL = 'KL'
    EWQ = 'EWQ'


@dataclass
class CompressionConfig:
    weight_bit_list: List
    weight_per_channel_bit_list: List
    candidate_search_alg: CandidateSearchAlg
    pareto_cost: ParetoCost
    mp_per_channel_cost: MPCost
    threshold_method: ThresholdMethod
    optimize_scale: bool = True
    simd: int = 1
    max_candidates: int = 10000
    simd: int = 1
    num_inter_points: int = 2
    activation_n_bits: int = 8
    activation_mp: bool = False
    weights_mp_per_ch: bool = False
    disable_softmax_log_scale: bool = False
    disable_ln_reparam: bool = False

    ## TODO: remove before publish
    two_bit_quant_only: bool = False
    three_bit_quant_only: bool = False
