from constants import GPUALGALL, GPU, GPU6000, GPUALG3090, GPUALG, GPUALG2, VIT_SMALL, VIT_BASE, DEIT_TINY, DEIT_SMALL, \
    DEIT_BASE, SWIN_TINY, SWIN_SMALL, SWIN_BASE, RESNET18, RESNET50, MBV2
from copy_and_run import default_copy_and_run
from lab.experiments.running_configs import SetupConfigs, DataConfigs

from lab.send_job_to_grid import send_job_to_grid

just_count_the_runs = False
# just_count_the_runs = True
exec_file = None
if not just_count_the_runs:
    exec_file = default_copy_and_run(dest='/Vols/vol_design/tools/swat/users/liord/02_copy_and_run')
log_folder = "/Vols/vol_design/tools/swat/users/liord/wandb_logs"

PROJECT_NAME = "MpPerCh_May2025"
gpu_queue = GPUALGALL
setup_configs = [
    # SetupConfigs(model_name=SWIN_TINY, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=RESNET18, disable_ridge_regression=True, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=RESNET50, disable_ridge_regression=True, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=MBV2, disable_ridge_regression=True, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=SWIN_SMALL, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=SWIN_BASE, gpu_queue=gpu_queue),
    SetupConfigs(model_name=DEIT_TINY, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=DEIT_SMALL, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=DEIT_BASE, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=VIT_SMALL, gpu_queue=gpu_queue),
    # SetupConfigs(model_name=VIT_BASE, gpu_queue=gpu_queue),

]

quant_configs = [
    # DataConfigs(
    #     disable_finetune=True,
    #     threshold_method='MSE',
    #     pareto_cost='MSE',
    #     wandb_notes='mp'
    # ),
    # DataConfigs(
    #     disable_finetune=True,
    #     wandb_notes='mp-hmse'
    # ),
    # DataConfigs(
    #     threshold_method='MSE',
    #     pareto_cost='MSE',
    #     wandb_notes='mp-fine'
    # ),
    # DataConfigs(
    #     threshold_method='MSE',
    #     pareto_cost='MSE',
    #     disable_mp=True,
    #     wandb_notes='fixed-fine'
    # ),
    # DataConfigs(
    #     wandb_notes='mp-hmse-fine'
    # ),
    # DataConfigs(
    #     disable_mp=True,
    #     disable_finetune=True,
    #     wandb_notes='fixed'
    # ),
    # DataConfigs(
    #     weights_mp_per_ch=True,
    #     disable_finetune=True,
    #     wandb_notes='mp-ch-hmse'
    # ),
    # DataConfigs(
    #     weights_mp_per_ch=True,
    #     threshold_method='MSE',
    #     pareto_cost='MSE',
    #     disable_finetune=True,
    #     wandb_notes='mp-ch'
    # ),
    DataConfigs(
        weights_mp_per_ch=True,
        wandb_notes='mp-ch-hmse'
    ),
    DataConfigs(
        weights_mp_per_ch=True,
        threshold_method='MSE',
        pareto_cost='MSE',
        wandb_notes='mp-ch'
    ),
    DataConfigs(
        weights_mp_per_ch=True,
        threshold_method='MSE',
        pareto_cost='EWQ',
        wandb_notes='mp-ch-EWQ'
    ),
    # DataConfigs(
    #     weights_mp_per_ch=True,
    #     threshold_method='MSE',
    #     pareto_cost='MSE',
    #     simd=16,
    #     wandb_notes='mp-ch-simd16'
    # ),
    # DataConfigs(
    #     weights_mp_per_ch=True,
    #     threshold_method='MSE',
    #     pareto_cost='MSE',
    #     simd=32,
    #     wandb_notes='mp-ch-simd32'
    # ),
    # DataConfigs(
    #     weights_mp_per_ch=True,
    #     simd=16,
    #     wandb_notes='mp-ch-simd16-hmse'
    # ),
    # DataConfigs(
    #     weights_mp_per_ch=True,
    #     simd=32,
    #     wandb_notes='mp-ch-simd32-hmse'
    # ),
]

global_configs = DataConfigs(
    exp='046',
    exp_notes='KL, 2-4-8, LAMBDA no uniform candidates',
    disable_activation_quantization=False,
    candidate_search_alg='LAMBDA',
    bit_options=[2, 4, 8],
    # bit_options=[2, 3, 4, 5, 6, 7, 8],
    mp_per_channel_cost='KL',
    # bit_options=[4],
    num_inter_points=200,
    # bit_options_per_channel=[2, 3, 4, 5, 6, 7, 8]
)



num_runs_counter = 0
# num_seeds = [0, 1, 2, 3, 4]
num_seeds = [0]
group_counter = 0
global_params = global_configs.get_run_params()

for weight_n_bits in [3, 4]:
    for activation_n_bits in [4]:
        for img_ind, quant_config in enumerate(quant_configs):
            quant_params = quant_config.get_run_params()
            for set_ind, setup_config in enumerate(setup_configs):
                setup_params = setup_config.get_run_params()
                for k in setup_params.keys():
                    if k in quant_params.keys():
                        quant_params.pop(k)
                for random_seed in num_seeds:
                    num_runs_counter += 1
                    if not just_count_the_runs:
                        send_job_to_grid(
                            weight_n_bits=weight_n_bits,
                            activation_n_bits=activation_n_bits,
                            seed=random_seed,
                            exec_file=exec_file,
                            log_folder=log_folder,
                            project_name=PROJECT_NAME,
                            **setup_params,
                            **quant_params,
                            **global_params
                        )


print('____________________________')
print(f'Total number of runs: {num_runs_counter}')