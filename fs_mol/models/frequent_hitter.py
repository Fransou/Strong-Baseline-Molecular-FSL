import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.models as tgm
import torch_geometric.nn.pool as tgp
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Literal, Dict
from copy import deepcopy
import pandas as pd

import datamol as dm
import numpy as np
from torch_geometric.data import DataLoader

from fs_mol.utils.torch_utils import torchify
from fs_mol.configs import *
from fs_mol.models.backbone import Backbone
from fs_mol.data.fsmol_batcher import (
    FSMolBatcher,
)

class FrequentHitter(nn.Module):
    def __init__(self, config):
        config.backbone_config.fc_out_dim = 1
        super().__init__()
        self.in_channels = 0
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__init_backbone()

    def __init_backbone(self):
        self.backbone = Backbone(config=self.config.backbone_config)
        if self.config.backbone is not None and self.config.backbone != "":
            self.backbone.from_pretrained(self.config.backbone)
        self.backbone = self.backbone.to(self.device)

    def forward(
        self,
        argv,
    ):
        x = self.backbone(argv)
        return torch.sigmoid(x)
