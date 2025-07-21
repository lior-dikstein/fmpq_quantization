import os
import wandb

from constants import UNIQUE_ID


def wandb_and_log_init(args, group_name, run_name):  # , group_name, run_name):

    run_folder = os.path.join(args.log_folder, "results", args.project_name, args.model_name)
    run_folder = os.path.join(run_folder, UNIQUE_ID)
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    if args.wandb:
        wandb.init(project=args.project_name,
                   group=group_name, dir=args.log_folder,
                   name=run_name + '_' + UNIQUE_ID,
                   notes=args.wandb_notes)
        wandb.config.update(args)

    return run_folder