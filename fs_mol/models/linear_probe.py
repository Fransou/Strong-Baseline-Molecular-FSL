import torch
import torch.nn as nn
from typing import Optional, Dict

import datamol as dm
from torch_geometric.data import DataLoader

from fs_mol.utils.torch_utils import torchify
from fs_mol.models.backbone import Backbone
from fs_mol.data.fsmol_batcher import (
    FSMolBatcher,
)

class LinearProbe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = 0
        self.config = config
        self.__init_backbone()

        if self.in_channels == 0:
            self.in_channels = config.backbone_config.fc_out_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init_backbone(self):
        self.backbone = Backbone(config=self.config.backbone_config)
        if self.config.backbone is not None and self.config.backbone != "":
            self.backbone.from_pretrained(self.config.backbone)
        self.backbone.eval()

    def initialize_data_model_(self, benchmark, **kwargs):
        self.benchmark = benchmark
        self.initialize_data_model_fs_mol(**kwargs)

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
                fp_query = self.fp_query[
                    i
                    * self.config.max_num_graphs : (i + 1)
                    * self.config.max_num_graphs
                ]
            else:
                fp_query = None
                batch = torchify(batch, device=self.device)
            z_query.append(self.forward(batch, fp_query))
        z_query = torch.cat(z_query, dim=0)

        z_support = []
        for i, (batch, _) in enumerate(self.support):
            if self.fp_support is not None:
                fp_support = self.fp_support[
                    i
                    * self.config.max_num_graphs : (i + 1)
                    * self.config.max_num_graphs
                ]
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
        self.bias = nn.Parameter(
            torch.zeros(self.vector_param.shape[0], device=self.device),
            requires_grad=True,
        )

    def predict(self, only_support=False, batch_size=256):
        if only_support:
            z_tot = self.z_support
        else:
            z_tot = self.z
        prototype = self.vector_param / torch.norm(
            self.vector_param, dim=1, keepdim=True
        )
        z_tot = z_tot / torch.norm(z_tot, dim=1, keepdim=True)
        preds = []
        for i in range(0, z_tot.shape[0], batch_size):
            z = z_tot[i : min(batch_size + i, z_tot.shape[0])]
            dist = torch.cdist(z, prototype) ** 2 + self.bias.unsqueeze(0)
            p_proto = torch.softmax(-dist * self.config.temperature, dim=-1)[:, 1]
            preds.append(p_proto)
        return torch.cat(preds, dim=0)

    def get_loss(self, return_loss_components=False):
        if not return_loss_components:
            p_support = self.predict(only_support=True)
            loss = self.loss(p_support, self.y_support, None)
            return loss, p_support
        p_all = self.predict(only_support=False)
        p_support = p_all[: len(self.y_support)]
        p_query = p_all[len(self.y_support) :]
        self.y_pred = p_query
        loss = self.loss(p_support, self.y_support)

        return loss, p_all
