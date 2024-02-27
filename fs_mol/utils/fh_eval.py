from typing import Optional

import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import datamol as dm

from fs_mol.models.frequent_hitter import FrequentHitter
from fs_mol.utils.torch_utils import torchify

from fs_mol.configs import *
from fs_mol.data import FSMolTaskSample
from fs_mol.utils.metrics import compute_binary_task_metrics
from fs_mol.data.fsmol_batcher import (
    FSMolBatcher,
)




class FrequentHitterEval:
    def __init__(
        self,
        config,
        name: str,
        inference_task_sample: FSMolTaskSample,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_name = name
        self.support = inference_task_sample.train_samples
        self.query = inference_task_sample.test_samples
        batcher = FSMolBatcher(max_num_graphs=1024)

        self.y_support = torch.Tensor([int(s.bool_label) for s in self.support])
        self.y_query = torch.Tensor([int(s.bool_label) for s in self.query])
        self.support = batcher.batch(self.support)
        self.query = batcher.batch(self.query)
        self.model = FrequentHitter(config=self.config)

    @torch.no_grad()
    def __call__(
        self,
    ):
        y_pred = []
        for batch, _ in self.query:
            batch = torchify(batch, device=self.device)
            y_pred.append(self.model(batch))
        y_pred = torch.cat(y_pred, dim=0)

        avg_p_score = average_precision_score(
            self.y_query, y_pred.detach().cpu().numpy()
        )

        return y_pred.detach().cpu().numpy()


def test_model_fn(
    task_sample: FSMolTaskSample,
    temp_out_folder: str,
    seed: int,
    model_config: QProbeConfig,
    p_bar: Optional[tqdm] = None,
):
    task_trainer = FrequentHitterEval(
        config=model_config,
        name=task_sample.name,
        inference_task_sample=task_sample,
    )
    y_pred_query = task_trainer()
    test_metrics = compute_binary_task_metrics(
        y_pred_query, task_trainer.y_query.detach().cpu().numpy()
    )
    if p_bar is not None:
        p_bar.update(1)

    return test_metrics
