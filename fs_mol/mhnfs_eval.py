import os
import sys
import warnings
import argparse
import logging
import json
import time
from functools import partial
from dataclasses import asdict

import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
import torch
import hydra
from collections import OrderedDict

FS_MOL_CHECKOUT_PATH = os.path.abspath(os.getcwd())
os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

from fs_mol.configs import *
from fs_mol.utils.mhnfs_eval import test_model_fn
from fs_mol.data import DataFold
from fs_mol.utils.test_utils import add_eval_cli_args, set_up_test_run, eval_model
from fs_mol.external_repositories.mhnfs.mhnfs.models import MHNfs


warnings.filterwarnings("ignore")

BASE_SUPPORT_SET_SIZE = [16, 32, 64, 128]

logging.basicConfig(
    format=f"""{time.strftime("%d_%b_%H_%M", time.localtime())}:::%(levelname)s:%(message)s""",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_command_line():
    """
    The parse_command_line function parses the command line arguments and returns them as an object.

    Args:

    Returns:
        A namespace object containing the arguments passed to the program
    """
    parser = argparse.ArgumentParser(
        description="Launches Simpleshot on a given fold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_eval_cli_args(parser)

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the run on wandb",
    )

    parser.add_argument(
        "--fold",
        type=str,
        choices=["TRAIN", "VALIDATION", "TEST"],
        default="TEST",
        help="Fold to launch the test on",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="FS-Mol",
        help="Name of the wandb project",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=320,
        help="Maximum batch size to allow when running through inference on model.",
    )

    parser.add_argument(
        "--out-file",
        type=str,
        default="results_simpleshot.csv",
        help="Path to the output file",
    )

    args = parser.parse_args()
    return args


def provide_clean_checkpoint(path_checkpoint):
    checkpoint = torch.load(path_checkpoint)
    cleaned_state_dict = OrderedDict(
        [
            (k.replace("hopfield_chemTrainSpace", "contextModule"), v)
            for k, v in checkpoint["state_dict"].items()
        ]
    )
    cleaned_state_dict = OrderedDict(
        [
            (k.replace("transformer", "crossAttentionModule", 1), v)
            for k, v in cleaned_state_dict.items()
        ]
    )
    return cleaned_state_dict


def launch_evaluation(
    name,
    fsmol_dataset,
    save_dir,
    args,
    cfg,
    log_loss=True,
):
    wandb.init(
        project=args.wandb_project,
        name=name,
    )

    wandb.config.update({"support_set_size": args.train_sizes})
    logger.info(f"Launching inference on fold {args.fold}")

    datafold = DataFold[args.fold]
    p_bar = tqdm(
        total=len(args.train_sizes)
        * len(fsmol_dataset._fold_to_data_paths[datafold])
        * args.num_runs,
        desc="Progression : ",
        leave=True,
    )
    for i in BASE_SUPPORT_SET_SIZE:
        wandb.define_metric(f"delta_aucpr_{i}", hidden=True)
        wandb.define_metric(f"best_delta_aucpr_{i}", hidden=True)

    df_preds = pd.DataFrame(
        columns=["y_true", "y_pred_pos", "task_name", "support_size"]
    )

    model = MHNfs(cfg)
    model = model.to("cuda")
    model._update_context_set_embedding()
    model.eval()
    checkpoint = provide_clean_checkpoint(
        "backbone_pretrained_models/model_weights/mhnfs/epoch=94-step=19855.ckpt"
    )
    model.load_state_dict(checkpoint)

    # model = model.load_from_checkpoint(path_checkpoint)

    results = eval_model(
        test_model_fn=partial(
            test_model_fn,
            p_bar=p_bar,
            cfg=cfg,
            model=model,
        ),
        dataset=fsmol_dataset,
        fold=datafold,
        train_set_sample_sizes=args.train_sizes,
        num_samples=args.num_runs,
        seed=args.seed,
        out_dir=args.save_dir,
    )

    df_task_result = pd.DataFrame()
    for i in range(
        len(
            args.train_sizes,
        )
    ):
        df_task_result_i = pd.DataFrame(
            {k: asdict(v_list[i]) for k, v_list in results.items() if len(v_list) > i}
        ).T
        df_task_result = pd.concat([df_task_result, df_task_result_i])
    df_task_result["delta_aucpr"] = (
        df_task_result["avg_precision"] - df_task_result["fraction_pos_test"]
    )
    for size in args.train_sizes:
        wandb.log(
            {
                f"delta_aucpr_{size}": df_task_result[
                    df_task_result.num_train == size
                ].delta_aucpr.mean()
            }
        )
    finalize_evaluation(df_task_result, save_dir=save_dir)


def finalize_evaluation(
    full_results: pd.DataFrame,
    save_dir: str,
):
    res_grouped = full_results.groupby("num_train")["delta_aucpr"].mean().reset_index()
    res_table = wandb.Table(dataframe=res_grouped)
    wandb.log(
        {
            "Results": wandb.plot.line(
                res_table,
                "num_train",
                "delta_aucpr",
                title="Results",
            )
        }
    )

    full_results["support_set_size"] = full_results.num_train
    if not (save_dir == "" or save_dir is None):
        full_results.to_csv(save_dir + "/full_results.csv")


@hydra.main(config_path="external_repositories/mhnfs/configs", config_name="cfg")
def main(cfg):
    os.chdir(FS_MOL_CHECKOUT_PATH)
    args = parse_command_line()

    wandb.login()

    _, dataset = set_up_test_run("TIM", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    name = (
        args.name
        if not args.name == ""
        else f"""MHNFS_{time.strftime("%d_%b_%H_%M", time.localtime())}"""
    )

    launch_evaluation(name, dataset, args.save_dir, args, cfg)
    wandb.finish()


if __name__ == "__main__":
    main()
