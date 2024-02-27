"""Classes that create models and datasets based on config."""
from typing import Union

from fs_mol.configs import (
    GnnPreTrainingConfig,
    GnnPreTrainingConfigLP,
    QProbeConfig,
    LinearProbeConfig,
    ClampConfig,
    FhConfig,
)

from fs_mol.data.gnn_data import MultitaskTaskSampleBatchIterableGnn
from fs_mol.data import DataFold


class ModelFactory:
    """Factory class for creating models based on config."""

    def __new__(cls, model_config, **kwargs):
        if isinstance(model_config, (ClampConfig,)):
            return ClampModelFactory(model_config=model_config, **kwargs)
        if isinstance(model_config, (GnnPreTrainingConfig, GnnPreTrainingConfigLP)):
            return GnnModelFactory(model_config=model_config, **kwargs)

        raise NotImplementedError


class PretrainingDatasetFactory:
    """Factory class for creating datasets based on config."""

    def __new__(
        cls,
        fsmol_dataset,
        config,
        device,
    ):
        train_task_name_to_id = {
            name: i
            for i, name in enumerate(
                fsmol_dataset.get_task_names(data_fold=DataFold.TRAIN)
            )
        }
        if isinstance(config, (GnnPreTrainingConfig, GnnPreTrainingConfigLP, FhConfig)):
            return MultitaskTaskSampleBatchIterableGnn(
                fsmol_dataset,
                data_fold=DataFold.TRAIN,
                task_name_to_id=train_task_name_to_id,
                max_num_graphs=config.max_num_graphs,
                device=device,
            )
        raise NotImplementedError


class ClampModelFactory:
    """Factory class to create CLAMP models."""

    def __new__(
        cls,
        model_config: Union[ClampConfig,],
        **kwargs,
    ):
        if isinstance(model_config, ClampConfig):
            from fs_mol.models.clamp_model import ClampFSL
            return ClampFSL(config=model_config, **kwargs)
        raise ValueError(f"Unknown CLAMP model config type: {type(model_config)}")


class GnnModelFactory:
    def __new__(
        cls,
        model_config: Union[
            QProbeConfig,
            GnnPreTrainingConfig,
            LinearProbeConfig,
            GnnPreTrainingConfigLP,
        ],
        benchmark: str = "fs_mol",
        **kwargs,
    ):
        if isinstance(model_config, QProbeConfig):
            from fs_mol.models.qprobe import QProbe

            model = QProbe(
                config=model_config,
            )
            model.initialize_data_model_(benchmark, **kwargs)

        elif isinstance(model_config, LinearProbeConfig):
            from fs_mol.models.linear_probe import LinearProbe

            model = LinearProbe(
                config=model_config,
            )
            model.initialize_data_model_(benchmark, **kwargs)

        elif isinstance(model_config, (GnnPreTrainingConfig, GnnPreTrainingConfigLP)):
            from fs_mol.models.qprobe import GNN_Multitask

            return GNN_Multitask(
                config=model_config,
                **kwargs,
            )
        #
        else:
            raise ValueError(f"Unknown model config type: {type(model_config)}")
        return model
