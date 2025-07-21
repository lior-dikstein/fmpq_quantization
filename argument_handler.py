import argparse
import os

from compression.configs.compression_config import (ThresholdMethod, ABSPLIT, ParetoCost, SVDScores, MPCost,
                                                    CandidateSearchAlg, SolverType)

from constants import TRAIN_DIR, VAL_DIR


def argument_handler():
    #################################
    ######### Run Arguments #########
    #################################

    # Settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', '-m', type=str, default="vit_s",
                        help='The name of the model to run')
    parser.add_argument('--model_type', type=str, default='vision', choices=['vision'])

    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR)
    parser.add_argument('--val_dir', type=str, default=VAL_DIR)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=1024)

    parser.add_argument('--eval_float_accuracy', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    # Quantization
    parser.add_argument('--weight_n_bits', type=int, default=8,
                        help="For Quantization this is the avg weight bit.")
    parser.add_argument('--bit_options', nargs='+', type=float, default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument('--bit_options_per_channel', nargs='+', type=float, default=[2, 4, 8])

    # Weight Quantization
    parser.add_argument('--threshold_method', type=str, default='HMSE',
                        choices=[i.name for i in ThresholdMethod])

    # Activation Quantization
    parser.add_argument('--disable_mp', action='store_true', default=False)
    parser.add_argument('--weights_mp_per_ch', action='store_true', default=False)
    parser.add_argument('--disable_activation_quantization', action='store_true', default=False)
    parser.add_argument('--activation_n_bits', type=int, default=8)
    parser.add_argument('--act_num_samples', type=int, default=32)
    parser.add_argument('--pareto_max_candidates', type=int, default=10000)
    parser.add_argument('--activation_mp', action='store_true', default=False)
    parser.add_argument('--disable_ln_reparam', action='store_true', default=False)
    parser.add_argument('--disable_softmax_log_scale', action='store_true', default=False)
    parser.add_argument('--disable_ridge_regression', action='store_true', default=False)
    parser.add_argument('--ridge_regression_num_samples', type=int, default=32)

    # Hessians
    parser.add_argument('--h_n_iters', type=int, default=100)
    parser.add_argument('--h_w_num_samples', type=int, default=32)

    # MRaP Search
    parser.add_argument('--candidate_search_alg', type=str, default='LAMBDA',
                        choices=[i.name for i in CandidateSearchAlg])
    parser.add_argument('--solver_type', type=str, default='ILP',
                        choices=[i.name for i in SolverType])
    parser.add_argument('--mp_per_channel_cost', type=str, default='SQNR',
                        choices=[i.name for i in MPCost])
    parser.add_argument('--pareto_cost', type=str, default='HMSEPerOutChannel',
                        choices=[i.name for i in ParetoCost])
    parser.add_argument('--num_inter_points', type=int, default=5)
    parser.add_argument('--simd', type=int, default=1)

    parser.add_argument('--disable_finetune', action='store_true', default=False)
    parser.add_argument('--finetune_iters', type=int, default=20000)
    parser.add_argument('--finetune_batch_size', type=int, default=32)
    parser.add_argument('--finetune_lr', type=float, default=0.3)
    parser.add_argument('--reg_factor', type=float, default=0.3)

    ## TODO: remove before publish
    parser.add_argument('--project_name', type=str, default='low_rank_base')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--exp_notes', type=str, default=None)
    parser.add_argument('--wandb_notes', type=str, default=None)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--group_str', type=str, default='')
    parser.add_argument('--log_folder', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results_tmp/log'))
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--two_bit_quant_only', action='store_true', default=False)
    parser.add_argument('--three_bit_quant_only', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--collect_stats', action='store_true', default=False)
    #########################################################################################################

    args = parser.parse_args()

    args.solver_type = SolverType(args.solver_type)
    args.candidate_search_alg = CandidateSearchAlg(args.candidate_search_alg)
    args.mp_per_channel_cost = MPCost(args.mp_per_channel_cost)
    args.pareto_cost = ParetoCost(args.pareto_cost)
    args.threshold_method = ThresholdMethod(args.threshold_method)

    if args.debug:
        print('************************************')
        print('Overriding parameters for debug mode')
        print('************************************')
        args.num_samples = 10
        args.train_dir = args.val_dir
        args.num_workers = 0
        args.act_num_samples = 4
        args.h_w_num_samples = 2
        args.h_n_iters = 2
        args.batch_size = 2
        args.finetune_iters = 2
        args.finetune_batch_size = 2

    if args.collect_stats:
        args.wandb = False

    return args
