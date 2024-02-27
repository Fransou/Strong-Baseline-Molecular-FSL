import sys
import os
import pandas as pd
import numpy as np

# This should be the location of the checkout of the FS-Mol repository:
FS_MOL_CHECKOUT_PATH = "/home/philippe/fsl/git_repos/few_shot_drug"
FS_MOL_DATASET_PATH = "/home/philippe/fsl/git_repos/few_shot_drug/datasets"

os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)
from dpu_utils.utils import RichPath
from fs_mol.data import FSMolDataset, DataFold
import matplotlib.pyplot as plt
from tqdm import trange


full_df = pd.DataFrame()
TASK_NAMES = ["DAVIS", "BindingDB_Kd", "BindingDB_Ki", "BindingDB_IC50", "KIBA"]

dataset = FSMolDataset.from_directory(
    directory=RichPath.create(FS_MOL_DATASET_PATH + "/fs-mol"),
    task_list_file=RichPath.create(FS_MOL_DATASET_PATH + "/fsmol-0.1.json"),
)
task_iterable = dataset.get_task_reading_iterable(DataFold.TRAIN)

for task in TASK_NAMES:
    filename = f"TDC_tasks/data/DTI/{task}_preprocessed.csv"
    df = pd.read_csv(filename)
    df["task"] = task
    full_df = pd.concat([full_df, df])


df_fs = {
    "Drug": [],
    "fs_mol_target": [],
}

task_iterable = iter(task_iterable)

for i_task in trange(4938):
    try:
        task = next(task_iterable)
    except StopIteration:
        break
    except Exception as e:
        print(e)
        continue
    for s in task.samples:
        smiles = s.smiles
        df_fs["Drug"].append(smiles)
        df_fs["fs_mol_target"].append(i_task)

df_common = full_df.merge(
    pd.DataFrame(df_fs),
    left_on=[
        "Drug",
    ],
    right_on=[
        "Drug",
    ],
    how="inner",
)
df_not_common = full_df[~full_df.Drug.isin(df_common.Drug)]
df_not_common["fs_mol_target"] = -1

df_intersect = pd.concat([df_common, df_not_common])
df_intersect["is_in_fs"] = df_intersect["fs_mol_target"] != -1

intersect_ratio_greedy = df_intersect.groupby(["task", "Target_ID"]).is_in_fs.mean()

true_intersect_ratio = intersect_ratio_greedy[intersect_ratio_greedy > 0.3].reset_index()

full_df_inters = df_intersect.merge(
    true_intersect_ratio, left_on=["task", "Target_ID"], right_on=["task", "Target_ID"], how="inner"
)
full_df_count = (
    full_df_inters.groupby(["task", "Target_ID"]).Drug_ID.count().rename("count")
)
full_df_count = 1 / full_df_count


common_tasks = (
    full_df_inters[full_df_inters.fs_mol_target != -1]
    .join(full_df_count, on=["task", "Target_ID"], how="inner")
    .groupby(["task", "Target_ID", "fs_mol_target"])["count"]
    .sum()
)
common_tasks.drop(columns=["fs_mol_target"]).reset_index().groupby(["task", "Target_ID"])["count"].max().hist(bins=100)
plt.show()

common_tasks.to_csv("TDC_tasks/data/DTI/common_tasks.csv")
