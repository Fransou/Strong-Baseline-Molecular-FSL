import os
import subprocess
import json
from dataclasses import asdict
from functools import partial
import logging

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import wandb

from fs_mol.models.modelfactory import ModelFactory, PretrainingDatasetFactory
from fs_mol.data import DataFold
from fs_mol.utils.domain_util import df_domain
from fs_mol.configs import LinearProbeConfig, QProbeConfig

logger = logging.getLogger(__name__)


class BaseClassPreTrainer:
    """Base class for pretraining a model"""

    def __init__(self, config, device, fsmol_dataset):
        self.config = config
        self.device = device
        self.df_norm_param_list = []
        self.df_norm_param = None

        logger.info(f"Config : {self.config}")
        self.model = ModelFactory(model_config=self.config)
        self.model = self.model.to(self.device)
        self.model.train()

        self.train_dataset = PretrainingDatasetFactory(fsmol_dataset, config, device)

        self.n_iters = self.model.get_niters(self.train_dataset)

        def get_task_domain(name):
            row = df_domain[df_domain.chembl_id == name]
            if len(row) == 0:
                return 0
            return row.EC.values[0]

        self.id_task_domain = {
            i: get_task_domain(name)
            for i, name in enumerate(
                fsmol_dataset.get_task_names(data_fold=DataFold.TRAIN)
            )
        }
        self.running_best_metric = 0

    def log_validation(self, epoch, fsmol_dataset):
        """Log validation results and save model"""
        save_path = os.path.join(self.config.out_dir, f"model.pt")
        torch.save(self.model.backbone.state_dict(), save_path)

        evaluator_config = self.config.fsl_config
        evaluator_config.backbone = save_path

        with open(os.path.join(self.config.out_dir, "evaluator_config.json"), "w") as f:
            json.dump(asdict(evaluator_config), f)

        if isinstance(evaluator_config, LinearProbeConfig):
            eval_process = subprocess.Popen(
                [
                    "python",
                    "fs_mol/linearprobe_eval.py",
                    "--name",
                    f"eval_{epoch}_{wandb.run.name}",
                    "--save-dir",
                    self.config.out_dir,
                    "--fold",
                    "VALIDATION",
                    "--model-config",
                    os.path.join(self.config.out_dir, "evaluator_config.json"),
                    "--wandb-project",
                    "fsmol-pretraining",
                    "--train-sizes",
                    "[16,64]",
                ],
                stdout=subprocess.PIPE,
            )
        elif isinstance(evaluator_config, QProbeConfig):
            eval_process = subprocess.Popen(
                [
                    "python",
                    "fs_mol/qprobe_eval.py",
                    "--name",
                    f"eval_{epoch}_{wandb.run.name}",
                    "--save-dir",
                    self.config.out_dir,
                    "--fold",
                    "VALIDATION",
                    "--model-config",
                    os.path.join(self.config.out_dir, "evaluator_config.json"),
                    "--wandb-project",
                    "fsmol-pretraining",
                    "--train-sizes",
                    "[16,64]",
                ],
                stdout=subprocess.PIPE,
            )
        else:
            raise ValueError(f"Unknown evaluator config type: {type(evaluator_config)}")
        eval_output, _ = eval_process.communicate()
        self.gather_eval_results(epoch, save_path)

    def gather_eval_results(self, epoch, save_path):
        df = pd.read_csv(os.path.join(self.config.out_dir, "best_delta_aucpr.csv"))
        avg_delta_aucpr = df["avg_precision"].mean()

        wandb.log({"avg_delta_aucpr": avg_delta_aucpr, "epoch": epoch + 1})
        for support_size in df.support_set_size.unique():
            wandb.log(
                {
                    f"delta_aucpr_{support_size}": df[
                        df.support_set_size == support_size
                    ]["avg_precision"].iloc[0],
                    "epoch": epoch + 1,
                }
            )

        print(f"====     avg_delta_aucpr: {avg_delta_aucpr}     =====")
        if avg_delta_aucpr > self.running_best_metric:
            self.running_best_metric = avg_delta_aucpr
            torch.save(self.model.backbone.state_dict(), save_path + "_best.pt")

    def train_step(self, optimizer, epoch, i, batch):
        """Train step"""
        optimizer.zero_grad()
        inp, labels, sample_to_task_id, thresholds = batch
        logits = self.model(inp)
        labels = torch.tensor(labels, dtype=torch.float, device=self.device)
        sample_to_task_id = torch.tensor(sample_to_task_id).to(self.device)

        logits_pred = torch.gather(logits, 1, sample_to_task_id.view(-1, 1)).squeeze()
        loss = self.model.loss(logits_pred, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.optimizer_config.clip_grad_norm
        )
        if epoch % 5 == 0:
            df_p_norm = self.model.get_param_grad_norm()
            df_p_norm["epoch"] = epoch
            self.df_norm_param_list.append(df_p_norm)
        optimizer.step()

        logger.info(f"epoch {epoch} --- step {i} --- loss: {loss.item()}")
        wandb.log({"loss": loss.item(), "epoch": epoch + i / self.n_iters})
        wandb.log(
            {
                "entropy": (
                    logits_pred * torch.log(logits_pred)
                    + (1 - logits_pred) * torch.log(1 - logits_pred)
                )
                .mean()
                .item(),
                "epoch": epoch + i / self.n_iters,
            }
        )

    def __call__(
        self,
        fsmol_dataset,
        t_0=10,
    ):
        optimizer = torch.optim.Adam(
            self.model.parameters(), self.config.optimizer_config.learning_rate
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, t_0)

        for epoch in range(self.config.optimizer_config.epochs[16]):
            for i, batch in enumerate(self.train_dataset):
                self.train_step(optimizer, epoch, i, batch)
                scheduler.step(epoch + i / self.n_iters)
                wandb.log(
                    {
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch + i / self.n_iters,
                    }
                )

            if epoch % 5 == 0:
                df_p_norm_ep = pd.concat(self.df_norm_param_list)
                df_p_norm_ep = (
                    df_p_norm_ep.groupby(["param_name", "epoch"]).mean().reset_index()
                )
                if self.df_norm_param is None:
                    self.df_norm_param = df_p_norm_ep
                else:
                    self.df_norm_param = pd.concat([self.df_norm_param, df_p_norm_ep])
                self.df_norm_param_list = []
                wandb.log(
                    {"param_grad_norm": wandb.Table(dataframe=self.df_norm_param)}
                )

            if epoch % self.config.validate_every == 0:
                self.log_validation(epoch, fsmol_dataset)
