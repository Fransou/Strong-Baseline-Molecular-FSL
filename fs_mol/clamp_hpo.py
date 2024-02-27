"""
Launches Simpleshot on a given fold, to measure the performance of the model.
"""
import os
import sys
import argparse
import logging
import time
import yaml
from dataclasses import asdict, fields
import json

FS_MOL_CHECKOUT_PATH = os.path.abspath(os.getcwd())
os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

import wandb

from fs_mol.qprobe_eval import launch_evaluation
from fs_mol.configs import *
from fs_mol.utils.test_utils import add_eval_cli_args, set_up_test_run


logging.basicConfig(
    format=f"""{time.strftime("%d_%b_%H_%M", time.localtime())}:::%(levelname)s:%(message)s""",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)


def parse_command_line():
    """Parse command line arguments."""
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
        "--wandb-project",
        type=str,
        default="FS-Mol",
        help="Name of the wandb project",
    )

    parser.add_argument(
        "--fold",
        type=str,
        choices=["TRAIN", "VALIDATION", "TEST"],
        default="VALIDATION",
        help="Fold to launch the test on",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=320,
        help="Maximum batch size to allow when running through inference on model.",
    )

    parser.add_argument(
        "--out_file",
        type=str,
        default="results_simpleshot.csv",
        help="Path to the output file",
    )
    parser.add_argument("--log-loss", action="store_true")
    parser.add_argument("--no-log-loss", dest="log_loss", action="store_false")
    parser.set_defaults(log_loss=True)

    parser.add_argument("--pca-plot", action="store_true")
    parser.add_argument("--no-pca-plot", dest="pca_plot", action="store_false")
    parser.set_defaults(pca_plot=False)

    args = parser.parse_args()
    return args


def generate_model_config(dict_wandb, run_keys):
    """Generate model config from wandb config"""

    model_config = {}
    loss_config = {}
    laplacian_config = {}
    optimizer_config = {}
    lmbds = []
    with open("fs_mol/configs/QP/expe.json", "r") as f:
        model_config_test = json.load(f)
    model_config["backbone_config"] = model_config_test["backbone_config"]
    for k, v in dict_wandb.items():
        if k in run_keys:
            if k.startswith("loss_config"):
                loss_config[k.split("__")[1]] = v
            elif k.startswith("optimizer_config"):
                optimizer_config[k.split("__")[1]] = v
            else:
                model_config[k] = v
    model_config["loss_config"] = loss_config
    model_config["optimizer_config"] = optimizer_config
    return ClampConfig(**model_config)


def main():
    args = parse_command_line()
    with open("fs_mol/hpo_config/hpo_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    out_dir, dataset = set_up_test_run("TIM", args, torch=True)

    wandb.login()
    run = wandb.init(config=config)
    if args.name == "":
        name = "HPO_"
    else:
        name = args.name

    run_keys = []
    for k in wandb.config.keys():
        if not k in ["program", "parameters", "metric", "name", "command", "method"]:
            run_keys.append(k)
    model_config = generate_model_config(dict(wandb.config), run_keys)
    launch_evaluation(
        name,
        dataset,
        None,
        model_config,
        args,
        log_loss=True,
    )


main()
