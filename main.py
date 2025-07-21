import numpy as np

import torch
from tqdm import tqdm

from argument_handler import argument_handler
from compression.configs.compression_config import CompressionConfig, SolverType
from compression.module.prepare_module import prepare_module
from compression.quantization.activation_quantization import activation_quantization_param_search
from compression.quantization.activation_quantization import insert_activation_quantization, \
    ln_reparameterization, hook_fp_act, replace_W
from compression.quantization.finetune import FineTuning
from compression.solvers.continuous_solver import run_continuous_solver
from compression.solvers.ilp_solver import run_solver
from compression.weighted_svd.weights_lfh import set_model_hessian_scores
from constants import LINEAR_QUANTIZE_OPERATORS
from debug.save_and_load_variables import save_debug_state
from helpers.set_seed import set_seed
from helpers.timers import SegmentTimer
from helpers.utils import is_compressed_layer, is_quantized_activation
from model_managers.base_model import BaseModel, ModelManagerArgs
from model_managers.vision_model import VisionModel

## TODO: remove before publish
import wandb
from helpers.wandb_util import wandb_and_log_init
import os


def get_model_manager(**kwargs) -> BaseModel:
    model_type_map = {
        "vision": VisionModel
    }

    model_manager_args = ModelManagerArgs(
        model_name=kwargs["model_name"],
        batch_size=kwargs["batch_size"],
        val_batch_size=kwargs["val_batch_size"],
        num_samples=kwargs["num_samples"],
        train_dir=kwargs["train_dir"],
        val_dir=kwargs["val_dir"]
    )
    model_type = kwargs['model_type']
    model_manager = model_type_map[model_type]
    return model_manager(model_manager_args)


def register_input_shapes_hook(module, name, handles):
    def hook(module, inputs, outputs):
        if is_compressed_layer(module) and not hasattr(module, 'input_shape'):
            module.input_shape = inputs[0].shape[1:]  # Ignore batch size

    return module.register_forward_hook(hook)


def compute_float_references(compressed_model, representative_dataset):
    handles = []
    for name, layer in compressed_model.named_modules():
        if is_compressed_layer(layer):
            handle = register_input_shapes_hook(layer, name, handles)
            handles.append(handle)

    output_ref = {}  # batch_idx --> layer --> output tensor
    for batch_idx, batch in enumerate(representative_dataset):
        data = model_manager.data_to_device(batch)
        batch_output_ref = model_manager.forward(compressed_model, data)
        output_ref[batch_idx] = {k: v.cpu() for k, v in batch_output_ref.items()} \
            if isinstance(batch_output_ref, dict) else batch_output_ref.cpu()

    for h in handles:
        h.remove()

    return output_ref


def run_evaluation(model_manager, compressed_model, val_data_loader, disable_activation_quantization):
    """Evaluate the model, toggling activation quantization if enabled."""
    # Gather all quantized-activation modules
    quant_modules = []
    if not disable_activation_quantization:
        quant_modules = [
            m for _, m in compressed_model.named_modules()
            if is_quantized_activation(m)
        ]
        # Enable activation quantization
        for m in quant_modules:
            m.set_activation_quantization(True)

    # Perform evaluation
    acc = model_manager.evaluate(compressed_model, val_data_loader)

    # Revert activation quantization to original state
    for m in quant_modules:
        m.set_activation_quantization(False)

    return acc


if __name__ == '__main__':
    """
    1. Run with activation quantization
    2. fix HMSE to MSE in cost
    3. Run with MSE instead of SINR in MP metric
    """
    #####################
    ######## Init #######
    #####################
    timer = SegmentTimer()
    args = argument_handler()
    set_seed(args.seed)

    if args.wandb:
        print("Initializing Weights & Biases connection...")
        act_bits_str = 'A32' if args.disable_activation_quantization else f'A{args.activation_n_bits}'
        w_bits_str = f'W{args.weight_n_bits}'
        threshold_method_str = args.threshold_method.value
        group_name = f'{args.exp}_{args.model_name}_{w_bits_str}{act_bits_str}_iterP{args.num_inter_points}_{threshold_method_str}_{args.group_str}'
        run_name = f'{group_name}_s{args.seed}'
        print('Run: ', run_name)
        log_folder = wandb_and_log_init(args, group_name, run_name)

    model_manager = get_model_manager(**vars(args))
    float_model = model_manager.float_model
    float_model.eval()
    val_data_loader = model_manager.get_validation_data_loader(num_workers=args.num_workers)
    float_accuracy = model_manager.float_accuracy
    if args.eval_float_accuracy:
        float_accuracy = model_manager.evaluate(float_model, val_data_loader)
        model_manager.set_float_accuracy(float_accuracy)

    cc = CompressionConfig(weight_bit_list=args.bit_options,
                           weight_per_channel_bit_list=args.bit_options_per_channel,
                           threshold_method=args.threshold_method,
                           num_inter_points=args.num_inter_points,
                           candidate_search_alg=args.candidate_search_alg,
                           mp_per_channel_cost=args.mp_per_channel_cost,
                           pareto_cost=args.pareto_cost,
                           max_candidates=args.pareto_max_candidates,
                           simd=args.simd,
                           activation_n_bits=args.activation_n_bits,
                           activation_mp=args.activation_mp,
                           weights_mp_per_ch=args.weights_mp_per_ch,
                           disable_softmax_log_scale=args.disable_softmax_log_scale,
                           disable_ln_reparam=args.disable_ln_reparam,
                           two_bit_quant_only=args.two_bit_quant_only,  ## TODO: remove before publish
                           three_bit_quant_only=args.three_bit_quant_only,  ## TODO: remove before publish
                           )

    #######################
    ##### Load Dataset ####
    #######################
    representative_dataset = model_manager.get_representative_dataset(args.num_samples, False, False, num_workers=args.num_workers)

    weight_n_bits = args.weight_n_bits

    compressed_model, float_model = prepare_module(float_model, model_manager, cc)

    # Compute float output reference
    with torch.no_grad():
        output_ref = compute_float_references(compressed_model, representative_dataset)

    ######################
    #### Init Hessian ####
    ######################
    h_num_samples = args.h_w_num_samples
    batch = next(iter(representative_dataset))
    if isinstance(batch, dict) or type(batch).__name__ == 'BatchEncoding':
        h_images = {k: v[:h_num_samples] for k, v in batch.items()}
    else:
        h_images = batch[0][:h_num_samples]
    h_n_iter = args.h_n_iters

    ## TODO: remove args.disable_low_rank before publish
    set_model_hessian_scores(compressed_model, h_images, n_iter=h_n_iter, quant_only=True)
    timer.segment("compute hessians")
    ##################################
    #### Init weights compression ####
    ##################################
    layer_counter = 0
    for n, m in tqdm(compressed_model.named_modules()):
        if is_compressed_layer(m):
            m.init_layer_compression(in_compression_config=cc,
                                     output_ref=output_ref,
                                     representative_data_loader=representative_dataset,
                                     qm=compressed_model,
                                     debug=args.debug if layer_counter > 5 else False,
                                     model_manager=model_manager)
            layer_counter += 1
    timer.segment("init layer compression")    ###############################

    if args.collect_stats:
        stats_dict = {} # key -> layer, value -> np.array size 2 x num_candidates (2 for MSE, size)
        for n, m in tqdm(compressed_model.named_modules()):
            if is_compressed_layer(m):
                stats_dict[n] = np.array(m.pareto).transpose(1,0)
                h_mean = m.w_hessian.mean(dim=list(range(len(m.w_hessian.shape)))[1:])
                stats_dict[n] ={'mse': m.mse,
                                'hessian_sum_per_channel': m.hessian_per_channel,
                                'hessian_mean_per_channel': h_mean}
        save_debug_state(args.model_name,
                         base_dir='/Vols/vol_design/tools/swat/users/liord/03_forPeople/forHai/fmp_2025_2',
                         var_names=['stats_dict'])
        exit(1)
    #### Prepare validation DS ####
    ###############################
    val_data_loader = model_manager.get_validation_data_loader(num_workers=args.num_workers)

    ########################
    #### Prepare Solver ####
    ########################
    compressed_model = compressed_model.to(model_manager.device)

    finetune = None
    if not args.disable_finetune:
        model_manager.batch_size = args.finetune_batch_size
        finetune_repdatset = model_manager.get_representative_dataset(args.num_samples, True, True, num_workers=args.num_workers)
        model_manager.batch_size = args.batch_size
        finetune = FineTuning(finetune_repdatset, model_manager, iters=args.finetune_iters,
                              batch_size=args.finetune_batch_size,
                              lr=args.finetune_lr, reg_factor=args.reg_factor, wandb_en=args.wandb)

    # check model weights dtype for size rate calculation
    float_model_n_bits = [m.weight.dtype.itemsize for m in compressed_model.modules() if hasattr(m, 'weight')]
    if len(set(float_model_n_bits)) > 1:
        raise Exception("mixed float precision")
    float_model_n_bits = 8 * float_model_n_bits[0]

    ####################
    #### Run Solver ####
    ####################
    if not args.disable_mp:
        if args.solver_type == SolverType.ILP:
            optimization_function = run_solver(compressed_model, cc, representative_dataset,
                                               model_manager, output_ref)
            compression_results = optimization_function(weight_n_bits / float_model_n_bits)
        elif args.solver_type == SolverType.CONTINUOUS:
            optimization_function, _, _ = run_continuous_solver(compressed_model, cc)
            compression_results, error, u_bound = optimization_function(weight_n_bits / float_model_n_bits)
            print(f'Continuous solver error {error}, bound {u_bound}')
    comp_layers = [(n, m) for n, m in compressed_model.named_modules() if is_compressed_layer(m)]
    sol = {}
    for idx, (n, m) in enumerate(comp_layers):
        layer_bit_width = args.weight_n_bits if args.disable_mp else compression_results[n].bit_width_quantization
        layer_bit_width_per_ch = layer_bit_width
        sol[(idx, n)] = (layer_bit_width_per_ch, )

    # Recreating the compressed model
    compressed_model, _ = prepare_module(float_model, model_manager, cc)
    timer.segment("mp solver")
    #################################
    #### Activation Quantization ####
    #################################
    if not args.disable_activation_quantization:
        compressed_model = insert_activation_quantization(model=compressed_model,
                                                          input_activations_quant=LINEAR_QUANTIZE_OPERATORS,
                                                          compression_config=cc)

        ##############################
        # Activation threshold search
        ##############################
        calib_samples = []
        rep_dataset = iter(representative_dataset)
        act_num_samples = min(args.act_num_samples, args.num_samples)
        num_batches = 1 if act_num_samples == args.batch_size else act_num_samples // args.batch_size + 1
        num_batches = min(num_batches, len(representative_dataset))
        for _ in range(num_batches):
            batch_samples = model_manager.data_to_device(next(rep_dataset))
            calib_samples.append(batch_samples)
        calib_samples = torch.cat(calib_samples, dim=0)

        calib_samples_rr = []
        rep_dataset = iter(representative_dataset)
        ridge_regression_num_samples = min(args.ridge_regression_num_samples, args.num_samples)
        num_batches = 1 if ridge_regression_num_samples == args.batch_size else (
                ridge_regression_num_samples // args.batch_size + 1)
        num_batches = min(num_batches, len(representative_dataset))
        for _ in range(num_batches):
            batch_samples = model_manager.data_to_device(next(rep_dataset))
            calib_samples_rr.append(batch_samples)
        calib_samples_rr = torch.cat(calib_samples_rr, dim=0)

        for n, m in compressed_model.named_modules():
            if is_quantized_activation(m):
                m.set_activation_quantization(True)

        compressed_model = activation_quantization_param_search(quant_model=compressed_model,
                                                                calib_samples=calib_samples[:act_num_samples],
                                                                model_manager=model_manager,
                                                                compression_config=cc)
        for n, m in compressed_model.named_modules():
            if is_quantized_activation(m):
                m.set_activation_quantization(False)

        ##############################
        # LayerNorm Reparametrization
        ##############################
        if not args.disable_ln_reparam:
            ln_reparameterization(compressed_model, cc=cc)

        ###################
        # Ridge Regression
        ##################
        if not args.disable_ridge_regression:
            for n, m in compressed_model.named_modules():
                if is_quantized_activation(m):
                    m.set_activation_quantization(False)
            fp_folder_path = hook_fp_act(compressed_model, calib_samples_rr[:args.ridge_regression_num_samples],
                                         args)

            for n, m in compressed_model.named_modules():
                if is_quantized_activation(m):
                    m.set_activation_quantization(True)

            replace_W(compressed_model, fp_folder_path)

            for n, m in compressed_model.named_modules():
                if is_quantized_activation(m):
                    m.set_activation_quantization(False)
    timer.segment("activation quantization")
    ###########################
    #### Recompute Hessian ####
    ###########################
    h_num_samples = args.h_w_num_samples
    samples, _ = next(iter(representative_dataset))
    h_images = samples[:h_num_samples]
    h_n_iter = args.h_n_iters

    ## TODO: remove quant_only before publish
    set_model_hessian_scores(compressed_model, h_images, n_iter=h_n_iter, quant_only=True)
    timer.segment("recompute hessians")
    ########################################
    ### Recalibrate compressed model params
    ########################################
    comp_layers = [(n, m) for n, m in compressed_model.named_modules() if is_compressed_layer(m)]
    for idx, (n, m) in tqdm(enumerate(comp_layers)):
        sol_config = sol[(idx, n)]
        m.init_layer_compression(in_compression_config=cc,
                                 output_ref=output_ref,
                                 representative_data_loader=representative_dataset,
                                 qm=compressed_model,
                                 model_manager=model_manager,
                                 debug=args.debug,
                                 config_to_set=sol_config)
    timer.segment("recalibrate compressed model params")
    ##############################
    #### Set Compressed Model ####
    ##############################
    for n, m in compressed_model.named_modules():
        if is_compressed_layer(m):
            assert len(m.compression_options.compression_options_list) == 1
            m.set_compression_config(m.compression_options.compression_options_list[0])
            m.enable_compression()

    if not args.debug:
        acc_before_finetune = run_evaluation(model_manager, compressed_model, val_data_loader, args.disable_activation_quantization)
    timer.segment("acc_before_finetune")
    if not args.disable_finetune:
        assert finetune is not None, "Finetune function not initialized."
        finetune(compressed_model, float_model)

        acc = run_evaluation(model_manager, compressed_model, val_data_loader, args.disable_activation_quantization)
    else:
        acc = acc_before_finetune
    timer.segment("finetune")

    print("float accuracy:", float_accuracy)
    print(f"compressed accuracy before fine tuning: avg. bits = {weight_n_bits}, acc = {acc_before_finetune}")
    print(f"compressed accuracy: avg. bits = {weight_n_bits}, acc = {acc}")
    # Print results to console
    timer.print_segments()

    #######################################################################
    # TODO: remove before publish
    if args.wandb:
        timer.log_to_wandb()
        if isinstance(acc, dict):
            run_logs = {"average_bit_width": weight_n_bits}
            float_results = {"float_" + k: v for k, v in float_accuracy.items()}
            compressed_results = {"compressed_" + k: v for k, v in acc.items()}
            print(compressed_results)
            run_logs.update(float_results)
            run_logs.update(compressed_results)
            wandb.log(run_logs)
            accuracy_key = "accuracy" if "accuracy" in acc else "accuracy_m"  # accuracy_m is used in mnli glue task
            if accuracy_key in acc:
                wandb.log({f"{weight_n_bits}_bits": acc[accuracy_key]})
        else:
            wandb.log({"compressed_accuracy": acc,
                       "acc_before_finetune": acc_before_finetune,
                       "float_accuracy": float_accuracy,
                       "average_bit_width": weight_n_bits})
            wandb.log({f"{weight_n_bits}_bits": acc})

        # Clear large wandb files: run-<id>.wandb and logs/debug-internal.log
        run_path = os.path.sep.join(wandb.run.dir.split(os.path.sep)[:-1])
        internal_log_file = os.path.join(run_path, 'logs', 'debug-internal.log')
        wandb_files = [os.path.join(run_path, f) for f in os.listdir(run_path) if f.endswith('wandb')]
        wandb.finish()

        # Delete the debug-internal.log file
        if os.path.isfile(internal_log_file):
            os.remove(internal_log_file)
        for f in wandb_files:
            os.remove(f)
    #######################################################################
