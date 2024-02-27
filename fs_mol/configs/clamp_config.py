from dataclasses import dataclass, field

from fs_mol.configs.base_config import (
    OptimizerConfig,
    init_dataclasses,
)


@dataclass(frozen=False)
class ClampConfig:
    model_path: str = "backbone_pretrained_models/clamp"
    temperature: float = 1.0
    test: bool = True
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self):
        self.optimizer_config = init_dataclasses(self.optimizer_config, OptimizerConfig)
