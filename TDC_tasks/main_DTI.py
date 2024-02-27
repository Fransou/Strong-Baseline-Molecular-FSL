# TODO add FH, GNN-MAML, PAR?
import os
import sys
import pandas as pd
from dataclasses import dataclass

FS_MOL_CHECKOUT_PATH = "./"
os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

import torch
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
import argparse

import json
import torch
from fs_mol.configs import QProbeConfig
from sklearn.metrics import average_precision_score, roc_auc_score
import time

from TDC_tasks.utils_DTI import dataset_loader, split_support_query, get_dataset_from_file
from TDC_tasks.estimators import (
    AdktEvaluator,
    PrototypicalNetworkEvaluator,
    SimpleBaselineEvaluator,
    SimSearchEvaluator,
    LinearProbeEvaluator,
    # MultitaskEvaluator,
    # MamlEvaluator,
    # ClampEvaluator,
)
#from TDC_tasks.estimators.clamp_estimators import prepro_smiles


MODELS = {
    "protonet": PrototypicalNetworkEvaluator(),
    # "clamp": ClampEvaluator(),
    "QP": SimpleBaselineEvaluator(),
    "simsearch": SimSearchEvaluator(),
    "adkt": AdktEvaluator(),
    # "multitask": MultitaskEvaluator(),
    # "maml": MamlEvaluator(),
    "linear_probe": LinearProbeEvaluator(),
}


def run_eval_fn_on_task_sample(evaluator, task_sample, y_support, y_query, fp_dict=None):
    if False:#isinstance(evaluator, ClampEvaluator):
        y = evaluator(
            torch.stack([fp_dict[s.smiles] for s in task_sample.train_samples]),
            torch.stack([fp_dict[s.smiles] for s in task_sample.test_samples]),
            y_support,
            y_query,
        )
    else:
        y = evaluator(task_sample, y_support, y_query)
    return dict(
        auprc=average_precision_score(y_query, y),
        auroc=roc_auc_score(y_query, y),
        auprc_neg=average_precision_score(1 - y_query, 1 - y),
        auroc_neg=roc_auc_score(1 - y_query, 1 - y),
    )


def add_metrics_to_results(results, metrics):
    for k, v in metrics.items():
        results[k].append(v)
    return results


def run_on_task_sample(results, task_sample, task_name, y_support, y_query, random_seed, fp_dict=None):
    for model_name, evaluator in MODELS.items():
        t0 = time.time()
        run_result = run_eval_fn_on_task_sample(evaluator, task_sample, y_support, y_query, fp_dict=fp_dict)
        t1 = time.time()
        results["task_name"].append(task_name)
        results["task_size"].append(y_support.shape[0])
        results["random_seed"].append(random_seed)
        results["model"].append(model_name)
        results["runtime"].append(t1 - t0)
        results = add_metrics_to_results(results, run_result)
    return results


def main(task_sizes, random_seeds, task_names):
    results = {
        "task_name": [],
        "task_size": [],
        "random_seed": [],
        "model": [],
        "auprc": [],
        "auroc": [],
        "auprc_neg": [],
        "auroc_neg": [],
        "runtime": [],
    }
    for task_name in task_names:
        df = get_dataset_from_file(f"TDC_tasks/data/DTI/{task_name}.csv", task_name)
        p_bar = tqdm(total=len(task_sizes) * len(random_seeds) * len(MODELS.keys()) * df.Target_ID.nunique())
        task_loader = dataset_loader(df, "fs_mol/preprocessing/utils/helper_files")
        for i_task, task in enumerate(iter(task_loader)):
            fp_dict = None
            if "clamp" in MODELS.keys():
                smiles = [s.smiles for s in task.samples]
                fp = prepro_smiles(smiles)
                fp_dict = {smiles[i]: fp[i] for i in range(len(smiles))}
            prop_pos = np.mean([int(sample.bool_label) for sample in task.samples])
            for random_seed in random_seeds:
                for task_size in task_sizes:
                    task_sample, y_support, y_query = split_support_query(
                        task, task_size, prop_pos_support=prop_pos, random_seed=random_seed
                    )
                    if task_sample is None:
                        continue
                    results = run_on_task_sample(
                        results, task_sample, task_name, y_support, y_query, random_seed, fp_dict=fp_dict
                    )
                    p_bar.update(len(MODELS.keys()))
        results_df = pd.DataFrame(results)
        for model_name in MODELS.keys():
            for task_name in results_df.task_name.unique():
                results_df[(results_df["model"] == model_name) & (results_df["task_name"] == task_name)].to_csv(
                    f"TDC_tasks/results/DTI_prop/{model_name}_{task_name}_results.csv"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--random_seeds", nargs="+", type=int, default=[i for i in range(10)])
    parser.add_argument("--task_names", nargs="+", type=str, default=["BindingDB_Ki", "BindingDB_IC50"])
    parser.add_argument("--models", nargs="+", type=str, default=MODELS.keys())
    args = parser.parse_args()
    print(args)
    random_seeds = args.random_seeds
    task_names = args.task_names
    task_sizes = args.task_sizes
    MODELS = {k: MODELS[k] for k in args.models}
    main(task_sizes, random_seeds, task_names)
