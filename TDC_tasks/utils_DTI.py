import os
import sys
import pandas as pd
from fs_mol.data.fsmol_task import MoleculeDatapoint, GraphData
from fs_mol.preprocessing.featurisers.molgraph_utils import molecule_to_graph
from fs_mol.data import FSMolTask, FSMolBatch, FSMolBatcher, FSMolTaskSample
from rdkit import DataStructs
from rdkit.Chem import (
    Mol,
    MolFromSmiles,
    rdFingerprintGenerator,
    CanonSmiles,
    Descriptors,
)
from dpu_utils.utils import RichPath
import numpy as np
from tqdm import tqdm

tqdm.pandas()


bounds_n_mols_task_name = {
    "DAVIS": (60, 5000),
    "BindingDB_Kd": (60, 5000),
    "BindingDB_Ki": (60, 5000),
    "BindingDB_IC50": (60, 5000),
    "KIBA": (60, 5000),
}

thresholds_task_name = {
    "DAVIS": -5.0,
    "BindingDB_Kd": -5.0,
    "BindingDB_Ki": -6.0,
    "BindingDB_IC50": -6.0,
    "KIBA": -5.0,
}

df_common_to_fs_mol = pd.read_csv("TDC_tasks/data/DTI/common_tasks.csv")


def get_feature_extractors_from_metadata(metadata_path, metadata_filename="metadata.pkl.gz"):
    metapath = RichPath.create(metadata_path)
    path = metapath.join(metadata_filename)
    metadata = path.read_by_file_suffix()
    return metadata["feature_extractors"]


def get_dataset_from_file(file, task_name):
    df_commom_loc = df_common_to_fs_mol[df_common_to_fs_mol.task == task_name]
    if not (file[:-4] + "_preprocessed.csv").split("/")[-1] in os.listdir("TDC_tasks/data/DTI/"):
        df = pd.read_csv(file)
        df = df[~df.Target_ID.isin(df_commom_loc.Target_ID)]
        df = df[df.Drug.progress_apply(isvalid_for_prepro)]
        df = (
            df.groupby(["Target_ID", "Drug_ID", "Drug"])
            .Y.mean()[df.groupby(["Target_ID", "Drug_ID", "Drug"]).Y.nunique() == 1]
            .reset_index()
        )
        df.to_csv(file[:-4] + "_preprocessed.csv")
    else:
        df = pd.read_csv(file[:-4] + "_preprocessed.csv")
    df.Y = np.log10(df.Y + 1e-10) -9

    # Get the threshodls (medina clipped in [0,9]) for each target
    df_task_name_joined = df.groupby("Target_ID").Y.agg(["median", "min", "max", "count"]).reset_index()
    df_task_name_joined["threshold"] = np.clip(df_task_name_joined["median"], -9, thresholds_task_name[task_name])

    # Dropping values equal to the threshold
    df = df.join(df_task_name_joined[["threshold", "Target_ID"]].set_index("Target_ID"), on="Target_ID")
    df = df.drop(df[df.Y == df.threshold].index)
    df["Y_bin"] = df.Y < df.threshold
    df["Y_bin"] = df["Y_bin"].astype(int)

    # Dropping targets with not enough molecules
    df_count_target = df.Target_ID.value_counts()
    min_size, max_size = bounds_n_mols_task_name[task_name]
    df = df[df.Target_ID.isin(df_count_target[(df_count_target > min_size) & (df_count_target < max_size)].index)]

    # Dropping targets with an imbalance over 0.9/0.1
    df_target = df.groupby("Target_ID").Y_bin.mean()
    df = df[df.Target_ID.isin(df_target[(df_target > 0.1) & (df_target < 0.6)].index)]

    return df


def isvalid_for_prepro(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetDegree() > 6:
            return False
    return True


def dataset_loader(df, metadata_path, keep_prop=False):
    # get pre-defined atom_feature_extractors from metadata provided in FS-Mol
    atom_feature_extractors = get_feature_extractors_from_metadata(metadata_path)
    for i_target, target_id in enumerate(df["Target_ID"].unique()):
        dataset = []
        for i, row in df[df.Target_ID == target_id].iterrows():
            label = row["Y_bin"]
            smiles = row["Drug"]
            rdkit_mol = MolFromSmiles(smiles)
            fingerprints_vect = rdFingerprintGenerator.GetCountFPs([rdkit_mol], fpType=rdFingerprintGenerator.MorganFP)[
                0
            ]
            fingerprint = np.zeros((0,), np.float32)  # Generate target pointer to fill
            DataStructs.ConvertToNumpyArray(fingerprints_vect, fingerprint)
            graph_dict = molecule_to_graph(rdkit_mol, atom_feature_extractors)
            adjacency_lists = []
            for adj_list in graph_dict["adjacency_lists"]:
                if adj_list:
                    adjacency_lists.append(np.array(adj_list, dtype=np.int64))
                else:
                    adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int64))
            graph = GraphData(
                node_features=np.array(graph_dict["node_features"], dtype=np.float32),
                adjacency_lists=adjacency_lists,
                edge_features=[],
            )

            mol = MoleculeDatapoint(
                task_name="antibiotics",
                smiles=smiles,
                graph=graph,
                fingerprint=fingerprint,
                descriptors=0,
                pxc50=label,
                numeric_label=label,
                task_threshold=0,
                bool_label=label == 1,
            )
            dataset.append(mol)
            del mol
        yield FSMolTask(name=f"{i_target}", samples=dataset)
        del dataset


def split_support_query(task, n_support, prop_pos_support=0.5, random_seed=42):
    n_support_pos = int(n_support * prop_pos_support)
    n_support_neg = n_support - n_support_pos

    if n_support_neg < 0 or n_support_pos < 0:
        raise ValueError("n_support_neg and n_support_pos must be positive")

    positive_samples = [sample for sample in task.samples if sample.bool_label]
    negative_samples = [sample for sample in task.samples if not sample.bool_label]
    # shuffle samples
    positive_samples = np.random.RandomState(seed=random_seed).permutation(positive_samples)
    negative_samples = np.random.RandomState(seed=random_seed).permutation(negative_samples)
    if len(positive_samples) <= n_support_pos or len(negative_samples) <= n_support_neg:
        return None, None, None
    support = np.concatenate([positive_samples[:n_support_pos], negative_samples[:n_support_neg]])
    query = np.concatenate([positive_samples[n_support_pos:], negative_samples[n_support_neg:]])
    y_support = np.concatenate([np.ones(n_support_pos), np.zeros(n_support_neg)])
    y_query = np.concatenate(
        [np.ones(len(positive_samples) - n_support_pos), np.zeros(len(negative_samples) - n_support_neg)]
    )

    return FSMolTaskSample("", support, query, query), y_support, y_query
