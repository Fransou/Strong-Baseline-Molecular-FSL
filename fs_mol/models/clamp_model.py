import torch
from typing import Literal, Dict

import numpy as np
from torch import nn

from fs_mol.configs import ClampConfig
from clamp.utils import load_model


class ClampBaseClass(nn.Module):
    """Base class for CLAMP models."""

    def __init__(self, config):
        super().__init__()
        self.in_channels = 768
        self.config = config
        self.__init_backbone()
        self.device = "cuda"

    def __init_backbone(self):
        clamp_model = load_model(
            self.config.model_path,
            4096,
            355,
        ).to("cuda")
        del clamp_model.assay_encoder
        self.backbone = clamp_model
        self.backbone.eval()

    def prepro_smiles(self, smi):
        from mhnreact.molutils import convert_smiles_to_fp

        fp_size = self.backbone.compound_encoder.linear_input.weight.shape[1]
        fp_inp = convert_smiles_to_fp(
            smi, which=self.backbone.hps["compound_mode"], fp_size=fp_size, njobs=1
        ).astype(np.float32)
        compound_features = torch.tensor(fp_inp).to(self.device)
        return compound_features

    def forward(self, inp, batch_size=256):
        for i in range(0, len(inp), batch_size):
            if isinstance(inp[0], str):
                x = self.prepro_smiles(inp[i : min(batch_size + i, len(inp))])
            else:
                x = inp[i : min(batch_size + i, len(inp))]
            if i == 0:
                out = self.backbone.compound_encoder(x)
            else:
                out = torch.cat([out, self.backbone.compound_encoder(x)], dim=0)
        return out


class ClampFSL(ClampBaseClass):
    """CLAMP model for FS learning."""

    def __init__(
        self,
        smiles: Dict[Literal["support", "query"], str],
        y_support: torch.Tensor,
        y_query: torch.Tensor,
        other_data: Dict[str, torch.Tensor],
        config: ClampConfig,
    ):
        super().__init__(config)
        self.config = config
        self.support = smiles["support"]
        self.query = smiles["query"]
        self.y_support = y_support
        self.y_query = y_query

        self.input_support = other_data.get("input_support", None)
        self.input_query = other_data.get("input_query", None)
        self._get_embeddings()
        self.__init_loss()
        self.__init_model()

    def __init_loss(self):
        def loss(p_support, y_support):
            loss_item = nn.BCELoss()(p_support, y_support.float().to(p_support.device))
            return loss_item

        self.loss = loss

    @torch.no_grad()
    def _get_embeddings(self):
        z_query = self.forward(self.query)
        z_support = self.forward(self.support)
        z = torch.cat([z_support, z_query], dim=0).float()

        z = z / torch.norm(z, dim=1, keepdim=True)

        self.z = z
        self.z_support = z[: len(self.y_support)]
        self.z_query = z[len(self.y_support) :]

    def __init_model(
        self,
    ):
        z_support = self.forward(self.support)
        self.vector_param = nn.Parameter(
            torch.stack(
                [
                    torch.mean(z_support[self.y_support == label], dim=0)
                    for label in np.unique(self.y_support)
                ]
            ),
            requires_grad=True,
        )

    def predict(self, only_support=False, batch_size=256):
        if only_support:
            z_tot = self.z_support
        else:
            z_tot = self.z
        preds = []
        z_tot = z_tot / torch.norm(z_tot, dim=1, keepdim=True)
        for i in range(0, z_tot.shape[0], batch_size):
            z = z_tot[i : min(batch_size + i, z_tot.shape[0])]

            prototype = self.vector_param / torch.norm(
                self.vector_param, dim=1, keepdim=True
            )
            dist = torch.cdist(z, prototype) ** 2
            p_proto = torch.softmax(-dist * self.config.temperature, dim=-1)[:, 1]
            preds.append(p_proto)
        return torch.cat(preds, dim=0)

    def get_loss(self, return_loss_components=False, only_support=False):
        """Compute the loss of the model."""
        p_all = self.predict(only_support)
        p_support, p_query = p_all[: len(self.y_support)], p_all[len(self.y_support) :]
        loss = self.loss(p_support, self.y_support)
        return loss, p_all
