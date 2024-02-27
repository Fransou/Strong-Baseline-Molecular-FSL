from typing import Optional

import torch
from torch import nn
from sklearn.metrics import average_precision_score
import datamol as dm
from tqdm import tqdm

from fs_mol.configs import *
from fs_mol.data import FSMolTaskSample
from fs_mol.utils.metrics import compute_binary_task_metrics


class SimSearchEval:
    def __init__(
        self,
        name: str,
        inference_task_sample: FSMolTaskSample,
        n_neighb: int = 3,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_name = name
        self.n_neighb = n_neighb

        self.y_support = torch.Tensor(
            [s.bool_label for s in inference_task_sample.train_samples]
        )
        self.y_query = torch.Tensor(
            [s.bool_label for s in inference_task_sample.test_samples]
        )
        self.support = [s.smiles for s in inference_task_sample.train_samples]
        self.query = [s.smiles for s in inference_task_sample.test_samples]

    def __call__(
        self,
    ):
        self.tanimoto_distance = dm.similarity.cdist(
            self.query,
            self.support,
        )
        logits = torch.Tensor(self.tanimoto_distance)
        y_pred = nn.Softmax(dim=1)(-self.n_neighb * logits) @ self.y_support
        avg_p_score = average_precision_score(
            self.y_query, y_pred.detach().cpu().numpy()
        )
        return y_pred.detach().cpu().numpy()


def test_model_fn(
    task_sample: FSMolTaskSample,
    temp_out_folder: str,
    seed: int,
    p_bar: Optional[tqdm] = None,
    n_neighb: int = 3,
):
    task_trainer = SimSearchEval(
        name=task_sample.name,
        inference_task_sample=task_sample,
        n_neighb=n_neighb,
    )
    y_pred_query = task_trainer()
    test_metrics = compute_binary_task_metrics(
        y_pred_query, task_trainer.y_query.detach().cpu().numpy()
    )
    if p_bar is not None:
        p_bar.update(1)

    return test_metrics
