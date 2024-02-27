import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import pandas as pd

import datamol as dm
import numpy as np
from torch_geometric.data import DataLoader

from fs_mol.utils.torch_utils import torchify
from fs_mol.models.backbone import Backbone
from fs_mol.data.fsmol_batcher import (
    FSMolBatcher,
)


class GnnBaseClass(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = 0
        self.config = config
        self.__init_backbone()

        if self.in_channels == 0:
            self.in_channels = config.backbone_config.fc_out_dim

    def __init_backbone(self):
        self.backbone = Backbone(config=self.config.backbone_config)
        if self.config.backbone is not None and self.config.backbone != "":
            self.backbone.from_pretrained(self.config.backbone)
        self.unfreeze_backbone()

    def unfreeze_backbone(self):
        if self.config.unfreeze_backbone == "train":
            for n, p in self.backbone.named_parameters():
                p.requires_grad = True
        elif self.config.unfreeze_backbone != "all":
            self.backbone.eval()
            for n, p in self.backbone.named_parameters():
                unfreeze = False
                for template in self.config.unfreeze_backbone:
                    if n.startswith(template):
                        unfreeze = True
                if unfreeze:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


class QProbe(GnnBaseClass):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_precision_matrices = None
        self.eigenvalue_cov = []

    def initialize_data_model_(self, benchmark, **kwargs):
        self.benchmark = benchmark
        if benchmark == "fs_mol":
            self.initialize_data_model_fs_mol(**kwargs)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

    def initialize_data_model_fs_mol(
        self,
        support,
        query,
        y_support: torch.Tensor,
        y_query: torch.Tensor,
    ):
        self.backbone = self.backbone.to(self.device)
        self.support = support
        self.query = query
        self.y_support = y_support.to(self.device)
        self.y_query = y_query.to(self.device)
        self.fp_support = None
        self.fp_query = None

        self.z = None
        self.z_support = None
        self.z_query = None

        batcher = FSMolBatcher(max_num_graphs=1024)
        self.support = batcher.batch(support)
        self.query = batcher.batch(query)

        self._get_embeddings()

        self._init_model()
        self._init_loss()

    @torch.no_grad()
    def _get_embeddings(self):
        z_query = []
        for i, (batch, _) in enumerate(self.query):
            if self.fp_query is not None:
                fp_query = self.fp_query[i * self.config.max_num_graphs : (i + 1) * self.config.max_num_graphs]
            else:
                fp_query = None
                batch = torchify(batch, device=self.device)
            z_query.append(self.forward(batch, fp_query))
        z_query = torch.cat(z_query, dim=0)

        z_support = []
        for i, (batch, _) in enumerate(self.support):
            if self.fp_support is not None:
                fp_support = self.fp_support[i * self.config.max_num_graphs : (i + 1) * self.config.max_num_graphs]
            else:
                fp_support = None
                batch = torchify(batch, device=self.device)
            z_support.append(self.forward(batch, fp_support))
        z_support = torch.cat(z_support, dim=0)

        z = torch.cat([z_support, z_query], dim=0).float()
        z = z / torch.norm(z, dim=1, keepdim=True)

        self.z = z
        self.z_support = z[: len(self.y_support)]
        self.z_query = z[len(self.y_support) :]


    def forward(self, argv, fingerprints=None):
        x = self.backbone(argv, fingerprints=fingerprints)
        return x

    def mahalanobis_distance(self, z, class_means, class_precision_matrices):
        repeated_difference = z.unsqueeze(0) - class_means.unsqueeze(1)
        first_half = torch.matmul(repeated_difference, class_precision_matrices)
        logits = torch.mul(first_half, repeated_difference)
        logits = logits.sum(dim=2).transpose(1, 0)
        return logits

    def compute_class_means_and_precisions(
        self, features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = []
        precisions = []
        for c in torch.unique(labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(features, 0, self._extract_class_indices(labels, c))
            # mean pooling examples to form class means
            means.append(prototypes[c.long()].unsqueeze(1))
            lambda_k_tau = class_features.size(0) / (class_features.size(0) + 1)
            lambda_k_tau = self.config.lmbd_maha
            Sigma = self._estimate_cov(class_features, means[-1])
            Sigma = (1-lambda_k_tau) * Sigma + lambda_k_tau * torch.eye(
                class_features.size(1), class_features.size(1), device=self.device
            )
            Sigma = torch.inverse(Sigma)
            precisions.append(Sigma)

        means = torch.stack(means)
        precisions = torch.stack(precisions)
        return means, precisions

    @staticmethod
    def _extract_class_indices(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    @staticmethod
    def _estimate_cov(
        examples: torch.Tensor,
        means: torch.Tensor,
        rowvar: bool = False,
        inplace: bool = False,
    ) -> torch.Tensor:
        """
        SCM: Function based on the suggested implementation of Modar Tensai
        and his answer as noted in:
        https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError("m has more than 2 dimensions")
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= means
        else:
            examples = examples - means
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    def _init_loss(self):
        def loss(p_support, y_support):
            loss_item = nn.BCELoss()(p_support, y_support.float().to(p_support.device))
            return loss_item

        self.loss = loss

    def _init_model(
        self,
    ):
        self.vector_param = nn.Parameter(
            torch.stack(
                [
                    torch.mean(self.z_support[self.y_support == label], dim=0)
                    / torch.mean(self.z_support[self.y_support == label], dim=0).norm()
                    for label in torch.sort(torch.unique(self.y_support)).values
                ]
            ),
            requires_grad=True,
        )

        # initialize covariance matrix with the empirical covariance matrix
        if self.config.opt_cov:
            precision = torch.cat(
                (
                    torch.eye(self.z_support.shape[1], self.z_support.shape[1]).unsqueeze(0),
                    torch.eye(self.z_support.shape[1], self.z_support.shape[1]).unsqueeze(0),
                ),
                dim=0,
            ).to(self.device)
            self.precision = nn.Parameter(precision, requires_grad=True)

    def predict(self, only_support=False, batch_size=256):
        if only_support:
            z_tot = self.z_support
        else:
            z_tot = self.z

        prototype = self.vector_param
        z_tot = z_tot / torch.norm(z_tot, dim=1, keepdim=True)

        preds = []
        for i in range(0, z_tot.shape[0], batch_size):
            z = z_tot[i : min(batch_size + i, z_tot.shape[0])]

            if self.config.opt_cov:
                class_precision_matrices = self.precision @ self.precision.transpose(2, 1)
                dist = self.mahalanobis_distance(z, prototype, class_precision_matrices)
            else:
                _, class_precision_matrices = self.compute_class_means_and_precisions(
                    self.z_support, self.y_support, prototype
                )
                dist = self.mahalanobis_distance(z, prototype, class_precision_matrices)
            p_proto = torch.softmax(-dist * self.config.temperature, dim=-1)[:, 1]
            preds.append(p_proto)
        return torch.cat(preds, dim=0)

    def get_loss(self, return_loss_components=False):
        if not return_loss_components:
            p_support = self.predict(only_support=True)
            loss = self.loss(p_support, self.y_support)
            return loss, p_support

        p_all = self.predict(only_support=False)
        p_support = p_all[: len(self.y_support)]
        p_query = p_all[len(self.y_support) :]
        self.y_pred = p_query
        loss = self.loss(p_support, self.y_support)
        return loss, p_all

class GNN_Multitask(GnnBaseClass):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.config = config
        self.n_tasks = config.n_tasks

        self.tail = nn.Linear(self.in_channels, self.n_tasks)

        self.loss = nn.BCEWithLogitsLoss()
        for n, p in self.named_parameters():
            print(f"Parameter {n} : requires grad : {p.requires_grad}, of shape {p.shape}")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def get_niters(self, dataset):
        n_iters = 0
        for _ in dataset:
            n_iters += 1
        return n_iters

    def forward(self, inp, fingerprints=None):
        if fingerprints is None:
            inp = torchify(inp, device=self.device)
        x = self.backbone(inp, fingerprints=fingerprints)

        x = nn.SELU()(x)
        x = self.tail(x)

        return x

    def get_loss(self, y_pred, y_true):
        loss = self.loss(y_pred, y_true)
        loss += self.config.l1_reg * torch.norm(self.tail, p=1)
        return loss

    def get_param_grad_norm(self):
        df = pd.DataFrame(columns=["param_name", "grad_norm"])
        for n, p in self.named_parameters():
            if not ("batch_norm" in n or "layer_norm" in n or "batchnorm" in n) and p.grad is not None:
                p_mean = p.grad.norm(dim=-1).mean().item()
                n = (
                    n.replace(".bias", "")
                    .replace(".weight", "")
                    .replace("backbone.", "")
                    .replace("molecule_model.", "")
                    .replace("graph_desc_model.", "GD")
                )
                for i_mlp in range(5):
                    n = (
                        n.replace(f".mlp.{i_mlp}", "")
                        .replace(f".edge_embedding{i_mlp}", "")
                        .replace(f"x_embedding{i_mlp}", "node_embedding")
                    )

                df.loc[df.shape[0]] = [n, p_mean + 1e-8]
        return df
