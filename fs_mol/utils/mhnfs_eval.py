from typing import Optional
from tqdm import tqdm
import numpy as np

import torch
from omegaconf import OmegaConf
from fs_mol.external_repositories.mhnfs.mhnfs.models import MHNfs

from fs_mol.configs import *
from fs_mol.data import FSMolTaskSample
from fs_mol.utils.metrics import compute_binary_task_metrics
from fs_mol.data.fsmol_batcher import (
    FSMolBatcher,
)
import pickle


@torch.no_grad()
def test_model_fn(
    task_sample: FSMolTaskSample,
    temp_out_folder: str,
    seed: int,
    p_bar: Optional[tqdm] = None,
    cfg: Optional[OmegaConf] = None,
    model: MHNfs = None,
):
    name_dict_mol_smiles_id_test = pickle.load(
        open(cfg.system.data.path + cfg.system.data.dir_test + cfg.system.data.name_dict_mol_smiles_id, "rb")
    )
    name_mol_inputs = np.load(
        open(cfg.system.data.path + cfg.system.data.dir_test + cfg.system.data.name_mol_inputs, "rb")
    )

    query_ids = [
        name_dict_mol_smiles_id_test[smiles] for smiles in [sample.smiles for sample in task_sample.test_samples]
    ]
    support_ids = [
        name_dict_mol_smiles_id_test[smiles] for smiles in [sample.smiles for sample in task_sample.train_samples]
    ]

    # recover the mol inputs as tensors

    support_inputs = torch.tensor(name_mol_inputs[support_ids], dtype=torch.float32)

    # We seaparte the support set into actives and inactives
    support_labels = [sample.bool_label for sample in task_sample.train_samples]
    support_actives_inputs = support_inputs[support_labels]
    support_inactives_inputs = support_inputs[~np.array(support_labels)]

    PAD_SIZE = max(support_actives_inputs.shape[0], support_inactives_inputs.shape[0])
    # We pad the support set to the same size : 256
    support_actives_inputs_padded = torch.zeros((PAD_SIZE, support_actives_inputs.shape[1]))
    support_actives_inputs_padded[: support_actives_inputs.shape[0], :] = support_actives_inputs
    support_inactives_inputs_padded = torch.zeros((PAD_SIZE, support_inactives_inputs.shape[1]))
    support_inactives_inputs_padded[: support_inactives_inputs.shape[0], :] = support_inactives_inputs

    batch_size = 2
    y_pred_query = []
    for i in range(0, len(query_ids), batch_size):
        query_inputs = torch.tensor(
            name_mol_inputs[query_ids[i : min(i + batch_size, len(query_ids))]], dtype=torch.float32
        ).unsqueeze(1)
        support_actives_inputs_padded_batch = support_actives_inputs_padded.unsqueeze(0).expand(
            query_inputs.shape[0], -1, -1
        )
        support_inactives_inputs_padded_batch = support_inactives_inputs_padded.unsqueeze(0).expand(
            query_inputs.shape[0], -1, -1
        )

        support_active_mask = torch.cat(
            [
                torch.cat([torch.tensor([True] * d), torch.tensor([False] * (PAD_SIZE - d))]).reshape(1, -1)
                for d in [support_actives_inputs.shape[0]]
            ],
            dim=0,
        ).expand(support_actives_inputs_padded_batch.shape[0], -1)
        support_inactive_mask = torch.cat(
            [
                torch.cat([torch.tensor([True] * d), torch.tensor([False] * (PAD_SIZE - d))]).reshape(1, -1)
                for d in [support_inactives_inputs.shape[0]]
            ],
            dim=0,
        ).expand(support_inactives_inputs_padded_batch.shape[0], -1)
        y_pred_query.append(
            model(
                query_inputs.to("cuda"),
                support_actives_inputs_padded_batch.to("cuda"),
                support_inactives_inputs_padded_batch.to("cuda"),
                torch.tensor([support_actives_inputs.shape[0]] * query_inputs.shape[0]).to("cuda"),
                torch.tensor([support_inactives_inputs.shape[0]] * query_inputs.shape[0]).to("cuda"),
                support_active_mask.to("cuda"),
                support_inactive_mask.to("cuda"),
            )
        )
    y_pred_query = torch.cat(y_pred_query, dim=0).detach().cpu().numpy()

    test_metrics = compute_binary_task_metrics(
        y_pred_query, np.array([sample.bool_label for sample in task_sample.test_samples])
    )
    if p_bar is not None:
        p_bar.update(1)

    return test_metrics
