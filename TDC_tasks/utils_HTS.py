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


def get_feature_extractors_from_metadata(metadata_path, metadata_filename="metadata.pkl.gz"):
    metapath = RichPath.create(metadata_path)
    path = metapath.join(metadata_filename)
    metadata = path.read_by_file_suffix()
    return metadata["feature_extractors"]


def isvalid_for_prepro(smiles):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetDegree() > 6:
            return False
    return True


def open_dataset(file, metadata_path, max_len=30000, just_df=False):
    if "preprocessed" in file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
        df = df[df.Drug.progress_apply(isvalid_for_prepro)]
        if df.shape[0] > max_len:
            positive_samples = df[df["Y"] == 1]
            negative_samples = df[df["Y"] == 0]
            prop = positive_samples.shape[0] / df.shape[0]
            if positive_samples.shape[0] / max_len > 0.07:
                n_pos = max(160, int(max_len * prop))  # Useful only if the dataset is smaller than max_len
                n_neg = max_len - n_pos
                df = pd.concat(
                    [
                        positive_samples.sample(n=n_pos, random_state=0),
                        negative_samples.sample(n=n_neg, random_state=0),
                    ]
                )
            else:
                df = pd.concat(
                    [positive_samples, negative_samples.sample(n=int(max_len - positive_samples.shape[0]), random_state=0)]
                )
        df.to_csv(file.replace(".csv", "_preprocessed.csv"), index=False)
    if just_df:
        return df
    dataset = []

    # get pre-defined atom_feature_extractors from metadata provided in FS-Mol
    atom_feature_extractors = get_feature_extractors_from_metadata(metadata_path)
    for i, row in df.iterrows():
        label = row["Y"]
        smiles = row["Drug"]

        rdkit_mol = MolFromSmiles(smiles)
        if rdkit_mol is None:
            continue

        fingerprints_vect = rdFingerprintGenerator.GetCountFPs([rdkit_mol], fpType=rdFingerprintGenerator.MorganFP)[0]
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
            task_name="hts",
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
    return FSMolTask(name=file, samples=dataset)


def split_support_query(task, n_support, prop_pos_support=0.5, random_seed=42):
    n_support_pos = max(1, int(n_support * prop_pos_support))
    n_support_neg = n_support - n_support_pos
    positive_samples = [sample for sample in task.samples if sample.bool_label]
    negative_samples = [sample for sample in task.samples if not sample.bool_label]
    # shuffle samples
    positive_samples = np.random.RandomState(seed=random_seed).permutation(positive_samples)
    negative_samples = np.random.RandomState(seed=random_seed).permutation(negative_samples)

    if len(positive_samples) <= n_support_pos or len(negative_samples) <= n_support_neg:
        raise ValueError(
            f"Number of positive samples ({len(positive_samples)}) or negative samples ({len(negative_samples)}) too small"
        )

    support = np.concatenate([positive_samples[:n_support_pos], negative_samples[:n_support_neg]])
    query = np.concatenate([positive_samples[n_support_pos:], negative_samples[n_support_neg:]])
    y_support = np.concatenate([np.ones(n_support_pos), np.zeros(n_support_neg)])

    y_query = np.concatenate(
        [np.ones(len(positive_samples) - n_support_pos), np.zeros(len(negative_samples) - n_support_neg)]
    )

    return FSMolTaskSample("", support, query, query), y_support, y_query
