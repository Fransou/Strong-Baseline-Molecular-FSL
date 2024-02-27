import numpy as np
import torch

from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.protonet import PrototypicalNetwork
from fs_mol.utils.protonet_utils import (
    PrototypicalNetworkTrainer,
    evaluate_protonet_model,
)
from fs_mol.data.protonet import task_sample_to_pn_task_sample, get_protonet_batcher
from fs_mol.utils.torch_utils import torchify

from line_profiler_pycharm import profile

class PrototypicalNetworkEvaluator:
    @profile
    def __init__(self, file="backbone_pretrained_models/PN-Support64_best_validation.pt", device="cuda", **kwargs):
        self.device = torch.device(device)
        self.batcher = get_protonet_batcher(max_num_graphs=1024)
        model_weights_file = resolve_starting_model_file(
            model_file=file,
            model_cls=PrototypicalNetworkTrainer,
            out_dir="",
            use_fresh_param_init=False,
            device=device,
        )
        self.model = PrototypicalNetworkTrainer.build_from_model_file(
            model_weights_file,
            device=device,
        )
        self.model.to(device)
        self.model.eval()

    @profile
    def __call__(self, task_sample, y_support, y_query, **kwargs):
        pn_task_sample = task_sample_to_pn_task_sample(task_sample, self.batcher)
        task_preds = []
        for batch_features, batch_label in zip(pn_task_sample.batches, pn_task_sample.batch_labels):
            batch_logits = self.model(torchify(batch_features, self.device))
            batch_preds = torch.nn.functional.softmax(batch_logits, dim=1).detach().cpu().numpy()
            task_preds.append(batch_preds[:, 1])
        y = np.concatenate(task_preds)
        return y
