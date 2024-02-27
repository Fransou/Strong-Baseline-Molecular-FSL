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

FS_MOL_CHECKOUT_PATH = os.path.abspath(os.getcwd())
os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

from fs_mol.configs import *
from fs_mol.utils.fh_eval import test_model_fn
from fs_mol.data import DataFold
from fs_mol.utils.test_utils import eval_model
from fs_mol.utils.test_utils import add_eval_cli_args, set_up_test_run

warnings.filterwarnings("ignore")

BASE_SUPPORT_SET_SIZE = [16]

logging.basicConfig(
    format=f"""{time.strftime("%d_%b_%H_%M", time.localtime())}:::%(levelname)s:%(message)s""",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def update_wandb_config(config, name):
    """
    The update_wandb_config function takes in a config object and the name of the config object.
    It then updates wandb's configuration with that information.

    Args:
        config: Update the wandb config with the parameters of a model
        name: Identify the config in wandb

    """
    wandb.config.update({f"{name}": asdict(config)})


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
        "--model-config",
        type=str,
        default="fs_mol/configs/FH_configs/config.json",
        help="Path to the model config file",
    )

    parser.add_argument(
        "--fold",
        type=str,
        choices=["TRAIN", "VALIDATION", "TEST"],
        default="VALIDATION",
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


def finalize_evaluation(
    full_results: pd.DataFrame,
    save_dir: str,
):
    """
    The finalize_evaluation function is used to log the results of the evaluation.

    Args:
        full_results: pd.DataFrame: Store the results of each iteration
        out_file: str: Specify the name of the output file
    """
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


def launch_evaluation(
    name,
    fsmol_dataset,
    save_dir,
    model_config,
    args,
    log_loss=True,
):
    wandb.init(
        project=args.wandb_project,
        name=name,
    )

    wandb.config.update({"support_set_size": args.train_sizes})
    update_wandb_config(model_config, "model_config")
    logger.info(f"Launching inference on fold {args.fold}")
    logger.info(f"Model config: {model_config}")

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
    results = eval_model(
        test_model_fn=partial(
            test_model_fn,
            model_config=model_config,
            p_bar=p_bar,
        ),
        dataset=fsmol_dataset,
        fold=datafold,
        train_set_sample_sizes=args.train_sizes,
        num_samples=args.num_runs,
        seed=args.seed,
        out_dir=args.save_dir,
        threshold_activity=args.threshold_activity,
    )
    df_task_result = pd.DataFrame()
    for i in range(len(args.train_sizes)):
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


def main():
    args = parse_command_line()

    wandb.login()

    _, dataset = set_up_test_run("TIM", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    name = (
        args.name
        if not args.name == ""
        else f"""FH_{time.strftime("%d_%b_%H_%M", time.localtime())}"""
    )
    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    model_config = FhConfig(**model_config)

    launch_evaluation(
        name,
        dataset,
        args.save_dir,
        model_config,
        args,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
