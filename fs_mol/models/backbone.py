from dataclasses import dataclass
from typing import List
from typing_extensions import Literal
from collections import OrderedDict

import torch
from torch import nn


from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)


FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


@dataclass(frozen=True)
class BackboneConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"


class FFBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.3, residual=False, norm="batch"):
        super().__init__()
        self.residual = residual and dim_in == dim_out
        if norm == "none":
            norm_fn = nn.Identity
        if norm == "batch":
            norm_fn = nn.BatchNorm1d
        elif norm == "layer":
            norm_fn = nn.LayerNorm
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("dropout", nn.Dropout(dropout)),
                    ("linear", nn.Linear(dim_in, dim_out)),
                    ("relu", nn.ReLU()),
                    ("batchnorm", norm_fn(dim_out)),
                ]
            )
        )

    def forward(self, x):
        if self.residual:
            return x + self.fc(x)
        return self.fc(x)


class Backbone(nn.Module):
    def __init__(self, config=BackboneConfig()):
        super().__init__()
        self.config = config
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(config.graph_feature_extractor_config)

        self.use_fc = self.config.used_features.endswith("+fc")

        # Create MLP if needed:
        if self.use_fc:
            # Determine dimension:
            fc_in_dim = 0
            if "gnn" in self.config.used_features:
                fc_in_dim += self.config.graph_feature_extractor_config.readout_config.output_dim
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM

            layers = [
                (
                    "fc0",
                    FFBlock(
                        fc_in_dim, config.fc_hidden_dim, dropout=config.fc_dropout, norm=config.norm, residual=config.residual
                    ),
                )
            ]
            for i in range(config.fc_layers):
                layers.append(
                    (
                        f"fc{i+1}",
                        FFBlock(
                            config.fc_hidden_dim,
                            config.fc_hidden_dim,
                            dropout=config.fc_dropout,
                            norm=config.norm,
                            residual=config.residual,
                        ),
                    )
                )
            layers.append(
                (
                    "fc_out",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("dropout", nn.Dropout(config.fc_dropout)),
                                ("linear", nn.Linear(config.fc_hidden_dim, config.fc_out_dim)),
                            ]
                        )
                    ),
                )
            )
            self.fc = OrderedDict(layers)
            self.fc = nn.Sequential(self.fc)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch, fingerprints=None):
        features: List[torch.Tensor] = []

        if "gnn" in self.config.used_features:
            features.append(self.graph_feature_extractor(batch))
        if "ecfp" in self.config.used_features:
            if fingerprints is None:
                fingerprints = batch.fingerprints.float()
            features.append(fingerprints)

        features = torch.cat(features, dim=1)

        if self.use_fc:
            features = self.fc(features)
        return features

    def from_pretrained(self, model_file):
        state_dict = torch.load(model_file)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            print("WARNING : Could not load model weights. Trying to load only graph_feature_extractor weights.")
            state_dict = {k.replace("graph_feature_extractor.", ""):v for k,v in state_dict.items() if "graph_feature_extractor" in k}
            # load only graph_feature_extractor
            self.graph_feature_extractor.load_state_dict(state_dict)
        return
