import os
import sys
import warnings

import argparse
import logging
import json
import time
from dataclasses import asdict
from dpu_utils.utils import RichPath
import torch

import wandb

FS_MOL_CHECKOUT_PATH = os.path.abspath(os.getcwd())
FS_MOL_DATASET_PATH = os.path.join(FS_MOL_CHECKOUT_PATH, "datasets")
os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

from fs_mol.configs import FhConfig

from fs_mol.utils.fh_pretraining_utils import (
    FHPretrainer,
)
from fs_mol.data import FSMolDataset, DataFold


warnings.filterwarnings("ignore")

BASE_SUPPORT_SET_SIZE = [16, 32, 64, 128]

logging.basicConfig(
    format=f"""{time.strftime("%d_%b_%H_%M", time.localtime())}:::%(levelname)s:%(message)s""",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def update_wandb_config(config, name):
    """Update wandb config"""

    wandb.config.update({f"{name}": asdict(config)})


def parse_command_line():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trains a GNN on cross-entropy loss on the training set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of the run on wandb",
    )

    parser.add_argument(
        "--model_config",
        type=str,
        default="fs_mol/configs/FH_configs/config.json",
        help="Path to the model config file",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Path to the output file",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_command_line()
    name, out_dir = (
        args.name,
        args.out_dir,
    )
    name = (
        name if not name == "" else f"""FH_{time.strftime("%d_%b_%H_%M", time.localtime())}"""
    )

    wandb.login()
    wandb.init(project="fsmol-pretraining", name=name)

    with open(args.model_config, "r") as f:
        model_config = json.load(f)
    model_config = FhConfig(**model_config)

    print(model_config)

    fsmol_dataset = FSMolDataset.from_directory(
        directory=RichPath.create(FS_MOL_DATASET_PATH + "/fs-mol"),
        task_list_file=RichPath.create(FS_MOL_DATASET_PATH + "/fsmol-0.1.json"),
    )
    num_tasks = fsmol_dataset.get_num_fold_tasks(DataFold.TRAIN)
    model_config.n_tasks = num_tasks

    out_dir = os.path.join(out_dir, name)
    model_config.out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    update_wandb_config(model_config, "model_config")
    trainer = FHPretrainer(model_config, device, fsmol_dataset)
    trainer(fsmol_dataset)
