import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_DTI import get_dataset_from_file, bounds_n_mols_task_name
import seaborn as sns


def get_all_datasets_stats(task_names=["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50"]):
    full_df = pd.DataFrame()
    for task_name in task_names:
        filename = f"TDC_tasks/data/DTI/{task_name}.csv"
        df = get_dataset_from_file(filename, task_name)
        df["task_name"] = task_name
        full_df = pd.concat([full_df, df])
    return full_df


def get_dataset_target_stats(task_names=["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50"], plot=False):
    full_df = get_all_datasets_stats(task_names)
    full_df = full_df.groupby(["task_name", "Target_ID"]).mean().reset_index()
    full_df["imb_diff"] = np.abs(1 / 2 - full_df.Y_bin)
    full_df["imb_H"] = -(full_df.Y_bin * np.log(full_df.Y_bin) + (1 - full_df.Y_bin) * np.log(1 - full_df.Y_bin))
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for task_name in task_names:
            sns.histplot(
                full_df[full_df.task_name == task_name],
                x="imb_diff",
                label=task_name,
                binwidth=0.1,
                stat="probability",
                common_norm=False,
                alpha=0.3,
                ax=axes[0],
                binrange=(0, 0.4),
            )
            axes[0].legend()
            axes[0].set_xlabel("Imabalance measured as: $0.5 - \max(prop_{+}, prop_{-})$")

            axes[1].set_xlabel("Class balance measured as: $H([prop_{+}, prop_-]])$")
            sns.histplot(
                full_df[full_df.task_name == task_name],
                x="imb_H",
                label=task_name,
                binwidth=0.1,
                stat="probability",
                common_norm=False,
                alpha=0.3,
                ax=axes[1],
                binrange=(0.3, 0.7),
            )

            # Threshold distrib
            sns.histplot(
                full_df[full_df.task_name == task_name],
                x="threshold",
                label=task_name,
                binwidth=1,
                stat="probability",
                common_norm=False,
                alpha=0.3,
                ax=axes[2],
                binrange=(0, 9),
            )
            axes[2].set_xlabel("Threshold used for binarization")
    return full_df


def get_kinases_stats(task_names=["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50"]):
    full_df = pd.DataFrame()
    for task_name in task_names:
        filename = f"TDC_tasks/data/DTI/{task_name}.csv"
        df = pd.read_csv(filename)
        df["task_name"] = task_name
        full_df = pd.concat([full_df, df])
    full_df_or = full_df.copy()
    full_df.task_name = full_df.task_name.apply(lambda x: x.split("_")[0])
    iskinase = (
        full_df[["Target", "task_name"]]
        .drop_duplicates()[["Target"]]
        .groupby(["Target"])
        .value_counts()
        .reset_index()
        .rename(columns={0: "counts"})
    )
    iskinase["iskinase"] = iskinase.counts > 1
    iskinase = iskinase.drop(columns=["counts"])
    full_df_or = full_df_or[["Target", "Target_ID", "task_name"]].drop_duplicates()
    full_df_or = full_df_or.merge(iskinase, on=["Target"])[["Target_ID", "task_name", "iskinase"]]
    return full_df_or
