import subprocess

def send_job_to_grid(gpu_queue,
                     exec_file,
                     exp,
                     project_name,
                     model_name,
                     log_folder,
                     seed,
                     exp_notes=None,
                     simd=1,
                     weight_n_bits=8,
                     activation_n_bits=8,
                     act_num_samples=32,
                     bit_options_per_channel=[2, 3, 4, 5, 6, 7, 8],
                     bit_options=[2, 3, 4, 5, 6, 7, 8],
                     solver_type='ILP',
                     candidate_search_alg='LAMBDA',
                     threshold_method='HMSE',
                     mp_per_channel_cost='SQNR',
                     wandb=True,
                     wandb_notes='',
                     batch_size=128,
                     val_batch_size=200,
                     num_samples=1024,
                     num_workers=8,
                     h_n_iters=100,
                     h_w_num_samples=32,
                     pareto_cost='HMSEPerOutChannel',
                     num_inter_points=5,
                     finetune_iters=20000,
                     finetune_batch_size=32,
                     finetune_lr=0.3,
                     reg_factor=0.3,
                     eval_float_accuracy=False,
                     disable_activation_quantization=False,
                     activation_mp=False,
                     disable_ln_reparam=False,
                     collect_stats=False,
                     disable_ridge_regression=False,
                     disable_mp=False,
                     weights_mp_per_ch=False,
                     disable_finetune=False,
                     two_bit_quant_only=False,
                     three_bit_quant_only=False,
                     ):
    group_str_full = ''

    if gpu_queue is None:
        cmd_list = ["python", exec_file]
    else:
        cmd_list = ["nc", "run", "-C", gpu_queue]
        cmd_list += ["python", exec_file]

    ##################################################################
    # Mandatory parameters
    ##################################################################
    cmd_list += ["--exp", exp,
                 "--project_name", project_name,
                 "--seed", str(seed),
                 "--model_name", model_name,
                 "--log_folder", log_folder,
                 "--weight_n_bits", str(weight_n_bits),
                 "--activation_n_bits", str(activation_n_bits),
                 "--act_num_samples", str(act_num_samples),
                 "--batch_size", str(batch_size),
                 "--val_batch_size", str(val_batch_size),
                 "--num_samples", str(num_samples),
                 "--num_workers", str(num_workers),
                 "--candidate_search_alg", str(candidate_search_alg),
                 "--solver_type", str(solver_type),
                 "--threshold_method", str(threshold_method),
                 "--mp_per_channel_cost", str(mp_per_channel_cost),
                 "--h_n_iters", str(h_n_iters),
                 "--simd", str(simd),
                 "--h_w_num_samples", str(h_w_num_samples),
                 "--pareto_cost", str(pareto_cost),
                 "--num_inter_points", str(num_inter_points),
                 "--finetune_iters", str(finetune_iters),
                 "--finetune_batch_size", str(finetune_batch_size),
                 "--finetune_lr", str(finetune_lr),
                 "--reg_factor", str(reg_factor),
                 ]

    # Add --bit_options followed by each value as a separate arg
    cmd_list += ["--bit_options"] + [str(b) for b in bit_options]
    # Add --bit_options followed by each value as a separate arg
    cmd_list += ["--bit_options_per_channel"] + [str(b) for b in bit_options_per_channel]

    ############################
    # General parameters
    ############################
    if wandb:
        cmd_list += ["--wandb"]
    if exp_notes is not None:
        cmd_list += ['--exp_notes', exp_notes]
    if wandb_notes is not None:
        cmd_list += ['--wandb_notes', wandb_notes]

    ############################
    # Store True parameters
    ############################
    if collect_stats:
        cmd_list += ["--collect_stats"]
    if eval_float_accuracy:
        cmd_list += ["--eval_float_accuracy"]
    if disable_activation_quantization:
        cmd_list += ["--disable_activation_quantization"]
    if weights_mp_per_ch:
        cmd_list += ["--weights_mp_per_ch"]
        group_str_full += '_wPerChMp'
    if disable_mp:
        cmd_list += ["--disable_mp"]
        group_str_full += '_noMp'
    if activation_mp:
        cmd_list += ["--activation_mp"]
        group_str_full += '_ActMp'
    if disable_ln_reparam:
        cmd_list += ["--disable_ln_reparam"]
        group_str_full += '_DisLn'
    if disable_ridge_regression:
        cmd_list += ["--disable_ridge_regression"]
        group_str_full += '_DisRR'
    if disable_finetune:
        cmd_list += ["--disable_finetune"]
        group_str_full += '_noFT'
    if two_bit_quant_only:
        cmd_list += ["--two_bit_quant_only"]
        group_str_full += '_2bitQonly'
    if three_bit_quant_only:
        cmd_list += ["--three_bit_quant_only"]
        group_str_full += '_3bitQonly'

    if len(group_str_full) > 0:
        cmd_list += ['--group_str', group_str_full]

    print(f"command: {cmd_list}")
    proc = subprocess.Popen(cmd_list)
    rc = proc.wait()
    if rc != 0:
        print(f"cmd failed: {cmd_list}")
        print(f"return code {rc}")
        exit(1)
