import os
import sys
import pandas as pd
import argparse

FS_MOL_CHECKOUT_PATH = "./"
os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

import torch

import numpy as np
from tqdm import tqdm


from TDC_tasks.utils_HTS import open_dataset, split_support_query
from TDC_tasks.estimators import (
    PrototypicalNetworkEvaluator,
    SimpleBaselineEvaluator,
    SimSearchEvaluator,
    AdktEvaluator,
    MultitaskEvaluator,
    LinearProbeEvaluator,
    ClampEvaluator,
)
from TDC_tasks.estimators.clamp_estimators import prepro_smiles
import time


MODELS = {
    "protonet": PrototypicalNetworkEvaluator,
    "QP": SimpleBaselineEvaluator,
    "simsearch": SimSearchEvaluator,
    "adkt": AdktEvaluator,
    "linear_probe": LinearProbeEvaluator,
    "clamp": ClampEvaluator,
}


def get_hitrate_from_predictions(y_pred, y_query):
    sorted_predictions = np.argsort(y_pred)[::-1]
    sorted_y_query = y_query[sorted_predictions]
    hit_rates_prot = np.cumsum(sorted_y_query) / np.arange(1, len(sorted_y_query) + 1)
    return hit_rates_prot


def run_eval_fn_on_task_sample(eval_fn, task_sample, y_support, y_query):
    y = eval_fn(task_sample, y_support, y_query)
    hitrate = get_hitrate_from_predictions(y, y_query)
    return hitrate


def add_hitrate_to_results(results, hitrate, run_metadata):
    step_size = 50 if len(hitrate) > 5000 else 10
    for n_mols in range(step_size, len(hitrate) - step_size, step_size):
        results["hitrate"].append(hitrate[n_mols - 1])
        results["n_mols"].append(n_mols)
        for k, v in run_metadata.items():
            results[k].append(v)


def main(task_sizes, random_seeds, task_names, task_props, p_bar):
    for task_name in task_names:
        if os.path.exists(f"TDC_tasks/data/HTS/{task_name}_preprocessed.csv"):
            task = open_dataset(
                f"TDC_tasks/data/HTS/{task_name}_preprocessed.csv",
                "fs_mol/preprocessing/utils/helper_files",
            )
        else:
            task = open_dataset(
                f"TDC_tasks/data/HTS/{task_name}.csv",
                "fs_mol/preprocessing/utils/helper_files",
            )
        smiles = [s.smiles for s in task.samples]
        if "clamp" in MODELS.keys():
            fp = prepro_smiles(smiles)
            fp_dict = {smiles[i]: fp[i] for i in range(len(smiles))}
        for random_seed in random_seeds:
            for task_prop in task_props:
                for task_size in task_sizes:
                    task_sample, y_support, y_query = split_support_query(
                        task, task_size, task_prop, random_seed=random_seed
                    )
                    for model_name, estimator in MODELS.items():
                        t0 = time.time()
                        if model_name == "clamp":
                            y = estimator(
                                torch.stack(
                                    [
                                        fp_dict[s.smiles]
                                        for s in task_sample.train_samples
                                    ]
                                ),
                                torch.stack(
                                    [
                                        fp_dict[s.smiles]
                                        for s in task_sample.test_samples
                                    ]
                                ),
                                y_support,
                                y_query,
                            )
                        else:
                            y = estimator(task_sample, y_support, y_query)
                        t1 = time.time()
                        hitrate = get_hitrate_from_predictions(y, y_query)

                        run_metadata = {
                            "task_name": task_name,
                            "task_size": task_size,
                            "task_prop": task_prop,
                            "random_seed": random_seed,
                            "model": model_name,
                            "query_prop": np.mean(y_query),
                            "runtime": t1 - t0,
                        }

                        add_hitrate_to_results(results, hitrate, run_metadata)
                        p_bar.update(1)
                    del task_sample, y_support, y_query
        del task

        df = pd.DataFrame(results)
        for model_name in MODELS.keys():
            for task_name in task_names:
                df[(df["model"] == model_name) & (df["task_name"] == task_name)].to_csv(
                    f"TDC_tasks/results/HTS/{model_name}_{task_name}_results.csv"
                )

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random_seeds", nargs="+", type=int, default=[i for i in range(20)]
    )
    parser.add_argument(
        "--task_names",
        nargs="+",
        type=str,
        default=[
            "cav3_t-type_calcium_channels_butkiewicz",
            # "m1_muscarinic_receptor_antagonists_butkiewicz",
            # "m1_muscarinic_receptor_agonists_butkiewicz",
            # "orexin1_receptor_butkiewicz",
            # "SARSCoV2_3CLPro_Diamond",
            # "HIV",
            # "SARSCoV2_Vitro_Touret",
            # "ALDH1",
            # "ESR1_ant",
            # "GBA",
            # "MAPK1",
            # "MTORC1",
            # "OPRK1",
            # "PKM2",
            # "PPARG",
            # "TP53",
            # "VDR"
        ],
    )
    parser.add_argument("--task_sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--task_props", nargs="+", type=float, default=[0.05, 0.1])
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=MODELS.keys(),
    )
    args = parser.parse_args()
    random_seeds = args.random_seeds
    task_names = args.task_names
    task_sizes = args.task_sizes
    task_props = args.task_props
    MODELS = {model_name: MODELS[model_name]() for model_name in args.models}

    results = {
        "task_name": [],
        "task_size": [],
        "task_prop": [],
        "random_seed": [],
        "hitrate": [],
        "model": [],
        "n_mols": [],
        "query_prop": [],
        "runtime": [],
    }
    p_bar = tqdm(
        total=len(task_sizes)
        * len(random_seeds)
        * len(MODELS.keys())
        * len(task_names)
        * len(task_props)
    )
    results = main(task_sizes, random_seeds, task_names, task_props, p_bar)
    for model_name in MODELS.keys():
        for task_name in task_names:
            results[
                (results["model"] == model_name) & (results["task_name"] == task_name)
            ].to_csv(f"TDC_tasks/results/HTS/{model_name}_{task_name}_results.csv")
