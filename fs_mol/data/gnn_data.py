"""Multitask GNN dataloader."""
from functools import partial
from typing import Tuple, Optional, Dict, Any, List
import dataclasses
from dpu_utils.utils import RichPath

import numpy as np
import torch
from rdkit.Chem import AllChem
from torch_geometric.data.batch import Batch

from fs_mol.utils.torch_utils import torchify
from fs_mol.data import (
    DataFold,
    FSMolDataset,
    FSMolTask,
    RandomTaskSampler,
    FSMolBatcher,
    MoleculeDatapoint,
    fsmol_batch_finalizer,
)
from fs_mol.data.multitask import FSMolMultitaskBatch


def multitask_batcher_init_fn(batch_data: Dict[str, Any]):
    """Initialize batch data."""
    batch_data["sample_to_task_id"] = []
    batch_data["smiles"] = []
    batch_data["descriptors"] = []
    batch_data["pxc50"] = []
    batch_data["threshold"] = []


def multitask_batcher_add_sample_fn(
    batch_data: Dict[str, Any],
    sample_id: int,
    sample: MoleculeDatapoint,
    task_name_to_id: Dict[str, int],
):
    """Add sample to batch."""
    batch_data["sample_to_task_id"].append(task_name_to_id[sample.task_name])
    batch_data["smiles"].append(sample.smiles)
    batch_data["pxc50"].append(sample.pxc50)
    batch_data["threshold"].append(sample.task_threshold)


def multitask_batcher_finalizer_fn(batch_data: Dict[str, Any]) -> Tuple[FSMolMultitaskBatch, np.ndarray]:
    """Finalize batch."""
    plain_batch = fsmol_batch_finalizer(batch_data)
    return (
        FSMolMultitaskBatch(
            sample_to_task_id=np.stack(batch_data["sample_to_task_id"], axis=0),
            **dataclasses.asdict(plain_batch),
        ),
        np.stack(batch_data["bool_labels"], axis=0),
        np.stack(batch_data["pxc50"], axis=0),
        batch_data["threshold"],
    )


def get_multitask_batcher(
    task_name_to_id: Dict[str, int],
    max_num_graphs: Optional[int] = None,
    max_num_nodes: Optional[int] = None,
    max_num_edges: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> FSMolBatcher[FSMolMultitaskBatch, np.ndarray]:
    """Get multitask batcher defined with the function above."""

    def finalizer(batch_data: Dict[str, Any]):
        finalized_batch = multitask_batcher_finalizer_fn(batch_data)
        if device is not None:
            finalized_batch = torchify(finalized_batch, device)

        return finalized_batch

    return FSMolBatcher(
        max_num_graphs=max_num_graphs,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        init_callback=multitask_batcher_init_fn,
        per_datapoint_callback=partial(multitask_batcher_add_sample_fn, task_name_to_id=task_name_to_id),
        finalizer_callback=finalizer,
    )


class MultitaskTaskSampleBatchIterableGnn:
    """Iterable for loading multitask samples for a GNN."""

    def __init__(
        self,
        dataset: FSMolDataset,
        data_fold: DataFold,
        task_name_to_id: Dict[str, int],
        device: torch.device,
        max_num_graphs: Optional[int] = None,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
        num_chunked_tasks: int = 8,
        repeat: bool = False,
        regression: bool = False,
    ):
        self._dataset = dataset
        self._data_fold = data_fold
        self._num_chunked_tasks = num_chunked_tasks
        self._repeat = repeat
        self._device = device
        self._regression = regression

        self._task_sampler = RandomTaskSampler(train_size_or_ratio=1024, valid_size_or_ratio=0, test_size_or_ratio=0)
        self._batcher = get_multitask_batcher(
            task_name_to_id=task_name_to_id,
            max_num_graphs=max_num_graphs,
            max_num_nodes=max_num_nodes,
            max_num_edges=max_num_edges,
        )

    def __iter__(self):
        def paths_to_mixed_samples(paths: List[RichPath], idx: int, threshold_activity:float, **kwargs):
            loaded_samples: List[MoleculeDatapoint] = []
            for i, path in enumerate(paths):
                task = FSMolTask.load_from_file(path)
                task_sample = self._task_sampler.sample(task, seed=idx + i)
                loaded_samples.extend(task_sample.train_samples)
            if self._data_fold == DataFold.TRAIN:
                np.random.shuffle(loaded_samples)

            for (
                features,
                labels,
                num_labels,
                threshold
            ) in self._batcher.batch(loaded_samples):
                if self._regression:
                    yield features, num_labels, features.sample_to_task_id, threshold
                else:
                    yield features, labels, features.sample_to_task_id, threshold

        return iter(
            self._dataset.get_task_reading_iterable(
                data_fold=self._data_fold,
                task_reader_fn=paths_to_mixed_samples,
                repeat=self._repeat,
                reader_chunk_size=16,
            )
        )
