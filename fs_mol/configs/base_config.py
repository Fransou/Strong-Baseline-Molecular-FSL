from dataclasses import dataclass, field
from typing import List, Literal, Dict, Union, Optional

from fs_mol.data.fsmol_dataset import NUM_NODE_FEATURES, NUM_EDGE_TYPES


def init_dataclasses(item, class_item):
    if item is None:
        return item
    if not isinstance(item, class_item):
        item = class_item(**item)
    return item


@dataclass(frozen=False)
class OptimizerConfig:
    learning_rate: float = 1e-3
    epochs: Union[Dict[int, int], int] = 100
    clip_grad_norm: float = 1.0

    def __post_init__(self):
        if isinstance(self.epochs, int):
            self.epochs = {i: self.epochs for i in [16, 32, 64, 128, 256]}
        self.epochs = {int(k): v for k, v in self.epochs.items()}


@dataclass(frozen=False)
class GraphReadoutConfig:
    readout_type: Literal[
        "sum",
        "min",
        "max",
        "mean",
        "weighted_sum",
        "weighted_mean",
        "combined",
    ] = "mean"
    use_all_states: bool = True
    num_heads: int = 12
    head_dim: int = 64
    output_dim: int = 512


@dataclass(frozen=False)
class GNNConfig:
    type: str = "PNA"
    num_edge_types: int = NUM_EDGE_TYPES
    hidden_dim: int = 128
    num_heads: int = 4
    per_head_dim: int = 64
    intermediate_dim: int = 1024
    message_function_depth: int = 1
    num_layers: int = 5
    dropout_rate: float = 0.3
    use_rezero_scaling: bool = True
    make_edges_bidirectional: bool = True


@dataclass(frozen=False)
class GraphFeatureExtractorConfig:
    initial_node_feature_dim: int = NUM_NODE_FEATURES
    gnn_config: GNNConfig = GNNConfig()
    readout_config: GraphReadoutConfig = GraphReadoutConfig()
    output_norm: Literal["off", "layer", "batch"] = "layer"

    def __post_init__(self):
        self.gnn_config = init_dataclasses(self.gnn_config, GNNConfig)
        self.readout_config = init_dataclasses(self.readout_config, GraphReadoutConfig)


@dataclass(frozen=False)
class BackboneConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = (
        GraphFeatureExtractorConfig()
    )
    used_features: Literal[
        "gnn",
        "ecfp",
        "pc-descs",
        "gnn+ecfp",
        "ecfp+fc",
        "pc-descs+fc",
        "gnn+ecfp+pc-descs+fc",
    ] = "gnn+ecfp+fc"
    kind: Literal["fsmol", "moleculenet"] = "fsmol"
    fc_hidden_dim: int = 1024
    fc_out_dim: int = 1024
    fc_dropout: float = 0.3
    fc_layers: int = 1
    residual: bool = False
    norm: Literal["batch", "layer"] = "batch"

    def __post_init__(self):
        self.graph_feature_extractor_config = init_dataclasses(
            self.graph_feature_extractor_config, GraphFeatureExtractorConfig
        )


@dataclass(frozen=False)
class BaseGnnConfig:
    backbone: Optional[str] = ""
    max_num_graphs: int = 6400
    test: bool = False
    unfreeze_backbone: Union[List[str], str] = field(default_factory=lambda: [])
    backbone_config: BackboneConfig = field(default_factory=BackboneConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
