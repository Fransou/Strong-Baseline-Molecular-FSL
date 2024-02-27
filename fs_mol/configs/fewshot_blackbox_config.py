from dataclasses import dataclass
import json
from typing import Union

from fs_mol.configs.base_config import (
    OptimizerConfig,
    BackboneConfig,
    BaseGnnConfig,
    init_dataclasses,
)


@dataclass(frozen=False)
class QProbeConfig(BaseGnnConfig):
    temperature: float = 1.0
    opt_cov: bool = False
    lmbd_maha: float = 0.2

    def __post_init__(self):
        self.optimizer_config = init_dataclasses(self.optimizer_config, OptimizerConfig)
        self.backbone_config = init_dataclasses(self.backbone_config, BackboneConfig)


@dataclass(frozen=False)
class GnnPreTrainingConfig(BaseGnnConfig):
    validate_every: int = 1
    temperature: float = 10.0
    n_tasks: int = 0
    out_dir = "outputs"
    fsl_config: Union[
        "str", QProbeConfig,
    ] = "fs_mol/configs/QP/model_test_config.json"

    def __post_init__(self):
        self.optimizer_config = init_dataclasses(self.optimizer_config, OptimizerConfig)
        if isinstance(self.fsl_config, str):
            self.fsl_config = QProbeConfig(
                **json.load(open(self.fsl_config))
            )
        elif isinstance(self.fsl_config, dict):
            self.fsl_config = init_dataclasses(
                self.fsl_config, QProbeConfig
            )
        self.backbone_config = init_dataclasses(self.backbone_config, BackboneConfig)
        self.fsl_config.backbone_config = self.backbone_config
