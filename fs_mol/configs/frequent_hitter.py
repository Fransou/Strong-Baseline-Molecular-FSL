from dataclasses import dataclass, field
from typing import Optional

from fs_mol.configs.base_config import OptimizerConfig, BackboneConfig


@dataclass(frozen=False)
class FhConfig:
    backbone: Optional[str] = "backbone_pretrained_models/model_best_FH_512.pt"
    max_num_graphs: int = 128
    backbone_config: BackboneConfig = field(default_factory=BackboneConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    validate_every: int = 1

    def __post_init__(self):
        if not isinstance(self.optimizer_config, OptimizerConfig):
            self.optimizer_config = OptimizerConfig(**self.optimizer_config)
        if not self.backbone_config is None and not isinstance(
            self.backbone_config, BackboneConfig
        ):
            self.backbone_config = BackboneConfig(**self.backbone_config)
