from typing import Optional, Union

import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import datamol as dm

from fs_mol.models.modelfactory import GnnModelFactory, ClampModelFactory

from fs_mol.configs import *
from fs_mol.data import FSMolTaskSample
from fs_mol.utils.metrics import compute_binary_task_metrics

import wandb


class BaseClassTrainer:
    def __init__(
        self,
        config,
        name: str,
        inference_task_sample: FSMolTaskSample,
        df_log_loss: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        self.config = config
        self.task_name = name
        self.support = inference_task_sample.train_samples
        self.query = inference_task_sample.test_samples
        self.df_log_loss = df_log_loss

        self.y_support = torch.Tensor([int(s.bool_label) for s in self.support])
        self.y_query = torch.Tensor([int(s.bool_label) for s in self.query])

        self.model = None
        self._get_classfier(**kwargs)

    def _get_classfier(self):
        raise NotImplementedError

    def log_step_results(self, i, loss, y_pred, df_log_loss=None):
        """Log step results"""
        y_query_pred = y_pred[len(self.y_support) :].detach().cpu().numpy()
        avg_p_score = average_precision_score(self.y_query, y_query_pred)
        if (
            i % (1 + self.config.optimizer_config.epochs[len(self.y_support)] // 101)
            == 0
        ):
            if df_log_loss is not None:
                estimate_prior = y_query_pred.mean()
                df_log_loss.loc[df_log_loss.shape[0]] = [
                    i,
                    loss.item(),
                    avg_p_score,
                    torch.mean(self.y_query),
                    float(estimate_prior),
                    self.task_name,
                    len(self.y_support),
                    self.param_grad_norm,
                ]
        return avg_p_score

    def train_step(self, optimizer):
        """Train step"""
        if hasattr(self.model, "update_params"):
            self.model.update_params()
        else:
            optimizer.zero_grad()
            loss, y_pred = self.model.get_loss(return_loss_components=True)
            loss.backward()
            self.param_grad_norm = float(
                torch.norm(self.model.vector_param.grad, p=2).detach().cpu().numpy()
            )
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.optimizer_config.clip_grad_norm
            )
            optimizer.step()
        return loss, y_pred

    def __call__(
        self,
        df_log_loss=None,
        compute_task_stats=True,
    ):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.config.optimizer_config.learning_rate
        )
        n_epochs = self.config.optimizer_config.epochs[len(self.y_support)]
        for i in range(n_epochs):
            loss, y_pred = self.train_step(optimizer)
            if not self.config.test:
                avg_p_score = self.log_step_results(i, loss, y_pred, df_log_loss)
        if n_epochs == 0:
            y_pred = self.model.predict()
        if n_epochs == 0 or self.config.test:
            avg_p_score = average_precision_score(
                self.y_query, y_pred[len(self.y_support) :].detach().cpu().numpy()
            )

        df_results = pd.DataFrame(
            {
                "y_true": self.y_query.detach().cpu().numpy(),
                "y_pred_pos": y_pred[len(self.y_support) :].detach().cpu().numpy(),
                "task_name": self.task_name,
                "support_size": len(self.y_support),
            }
        )
        return y_pred[len(self.y_support) :].detach().cpu().numpy(), df_results


class GnnTaskTrainer(BaseClassTrainer):
    def __init__(
        self,
        config,
        name: str,
        inference_task_sample: FSMolTaskSample,
        df_log_loss: Optional[pd.DataFrame] = None,
    ):
        super().__init__(
            config,
            name,
            inference_task_sample,
            df_log_loss,
        )

    def _get_classfier(self):
        self.model = GnnModelFactory(
            model_config=self.config,
            support=self.support,
            query=self.query,
            y_support=self.y_support,
            y_query=self.y_query,
        )


class ClampTaskTrainer(BaseClassTrainer):
    def __init__(
        self,
        config: Union[ClampConfig, None],
        name: str,
        inference_task_sample: FSMolTaskSample,
        df_log_loss: Optional[pd.DataFrame] = None,
    ):
        smiles = {
            "support": [
                sample.smiles for sample in inference_task_sample.train_samples
            ],
            "query": [sample.smiles for sample in inference_task_sample.test_samples],
        }
        y_support = torch.Tensor(
            [int(s.bool_label) for s in inference_task_sample.train_samples]
        )
        y_query = torch.Tensor(
            [int(s.bool_label) for s in inference_task_sample.test_samples]
        )
        super().__init__(
            config,
            name=name,
            inference_task_sample=inference_task_sample,
            df_log_loss=df_log_loss,
            smiles=smiles,
        )

    def _get_classfier(self, smiles):
        self.model = ClampModelFactory(
            self.config,
            smiles=smiles,
            y_support=self.y_support,
            y_query=self.y_query,
            other_data={},
        )


def get_df_log_loss():
    """Get dataframe for logging loss"""
    return pd.DataFrame(
        columns=[
            "epoch",
            "loss",
            "avg_precision",
            "prop_positive",
            "estimated_prior",
            "chembl_id",
            "support_set_size",
            "param_grad_norm",
        ]
    )


def get_trainer_from_config(
    config,
    name: str,
    inference_task_sample: FSMolTaskSample,
    df_log_loss: Optional[pd.DataFrame] = None,
):
    """Get trainer from config"""

    if isinstance(config, ClampConfig):
        return ClampTaskTrainer(
            config=config,
            name=name,
            inference_task_sample=inference_task_sample,
            df_log_loss=df_log_loss,
        )

    if isinstance(config, (QProbeConfig, LinearProbeConfig)):
        return GnnTaskTrainer(
            config=config,
            name=name,
            inference_task_sample=inference_task_sample,
            df_log_loss=df_log_loss,
        )

    raise NotImplementedError


def test_model_fn(
    task_sample: FSMolTaskSample,
    temp_out_folder: str,
    seed: int,
    model_config: QProbeConfig,
    p_bar: Optional[tqdm] = None,
    df_loss_log: Optional[pd.DataFrame] = None,
    df_results: Optional[pd.DataFrame] = None,
):
    task_trainer = get_trainer_from_config(
        config=model_config,
        name=task_sample.name,
        inference_task_sample=task_sample,
        df_log_loss=df_loss_log,
    )
    y_pred_query, df_results_task = task_trainer(
        df_log_loss=df_loss_log,
    )
    test_metrics = compute_binary_task_metrics(
        y_pred_query, task_trainer.y_query.detach().cpu().numpy()
    )
    if p_bar is not None:
        p_bar.update(1)
    if not df_results is None:
        for i in range(df_results_task.shape[0]):
            df_results.loc[df_results.shape[0]] = df_results_task.iloc[i]

    return test_metrics
