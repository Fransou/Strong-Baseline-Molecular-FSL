import os
import subprocess
import json
from dataclasses import asdict
from functools import partial
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import wandb

from fs_mol.models.frequent_hitter import FrequentHitter
from fs_mol.models.modelfactory import PretrainingDatasetFactory
from fs_mol.data import DataFold
from fs_mol.utils.domain_util import df_domain
from fs_mol.utils.torch_utils import torchify

logger = logging.getLogger(__name__)


class FHPretrainer:
    def __init__(self, config, device, fsmol_dataset):
        self.config = config
        self.device = device

        logger.info(f"Config : {self.config}")
        self.model = FrequentHitter(config=self.config)
        self.model = self.model.to(self.device)
        self.model.train()

        self.train_dataset = PretrainingDatasetFactory(fsmol_dataset, config, device)

        self.loss = torch.nn.BCELoss()
        self.n_iters = 0
        for _ in self.train_dataset:
            self.n_iters += 1

        self.running_best_metric = 0

    def log_validation(self, epoch, fsmol_dataset):
        """Log validation results and save model"""
        save_path = os.path.join(self.config.out_dir, f"model.pt")
        torch.save(self.model.backbone.state_dict(), save_path)

        evaluator_config = self.config
        evaluator_config.backbone = save_path
        with open(os.path.join(self.config.out_dir, "evaluator_config.json"), "w") as f:
            json.dump(asdict(evaluator_config), f)
        eval_process = subprocess.Popen(
            [
                "python",
                "fs_mol/fh_eval.py",
                "--name",
                f"eval_{epoch}_{wandb.run.name}",
                "--save-dir",
                self.config.out_dir,
                "--model-config",
                os.path.join(self.config.out_dir, "evaluator_config.json"),
                "--fold",
                "VALIDATION",
                "--wandb-project",
                "fsmol-pretraining",
                "--train-sizes",
                "[16]",
            ],
            stdout=subprocess.PIPE,
        )
        eval_output, _ = eval_process.communicate()
        self.gather_eval_results(epoch, save_path)

    def gather_eval_results(self, epoch, save_path):
        df = pd.read_csv(os.path.join(self.config.out_dir, "full_results.csv"))
        avg_delta_aucpr = df["delta_aucpr"].mean()

        wandb.log({"avg_delta_aucpr": avg_delta_aucpr, "epoch": epoch + 1})
        print(f"====     avg_delta_aucpr: {avg_delta_aucpr}     =====")
        if avg_delta_aucpr > self.running_best_metric:
            self.running_best_metric = avg_delta_aucpr
            torch.save(self.model.backbone.state_dict(), save_path + "_best.pt")

    def train_step(self, optimizer, epoch, i, batch):
        """Train step"""
        optimizer.zero_grad()
        inp, labels, sample_to_task_id, thresholds = batch
        inp = torchify(inp, device=self.device)
        logits = self.model(inp).squeeze(1)
        labels = torch.tensor(labels, dtype=torch.float, device=self.device)

        loss = self.loss(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.optimizer_config.clip_grad_norm
        )
        optimizer.step()

        logger.info(f"epoch {epoch} --- step {i} --- loss: {loss.item()}")
        if int(epoch + i / self.n_iters * 100) % 10 == 0:
            wandb.log({"loss": loss.item(), "epoch": epoch + i / self.n_iters})

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

            if epoch % self.config.validate_every == 0:
                self.log_validation(epoch, fsmol_dataset)
