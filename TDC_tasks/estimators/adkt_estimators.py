import numpy as np
import torch
from botorch.optim.fit import fit_gpytorch_scipy

from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.adaptive_dkt import ADKTModel
from fs_mol.utils.adaptive_dkt_utils import ADKTModelTrainer
from fs_mol.data.dkt import get_dkt_batcher, task_sample_to_dkt_task_sample
from fs_mol.utils.torch_utils import torchify


class AdktEvaluator:
    def __init__(self, file="backbone_pretrained_models/adkf-ift-classification.pt", device="cuda", **kwargs):
        self.batcher = get_dkt_batcher(max_num_graphs=1025)
        self.device = device
        model_weights_file = resolve_starting_model_file(
            model_file=file,
            model_cls=ADKTModel,
            out_dir="",
            use_fresh_param_init=False,
            device=device,
        )
        self.model = ADKTModelTrainer.build_from_model_file(
            model_weights_file,
            device=device,
        )
        self.model = self.model.to(device)


    def __call__(self, task_sample, y_support, y_query, **kwargs):
        task_preds = []
        dkt_task_sample = task_sample_to_dkt_task_sample(task_sample, self.batcher, filter_numeric_labels=True)
        for batch_features in dkt_task_sample.batches:
            batch_features = torchify(batch_features, self.device)
            self.model.train()
            _ = self.model(batch_features, train_loss=True)
            fit_gpytorch_scipy(self.model.mll)
            self.model.eval()
            with torch.no_grad():
                batch_logits = self.model(batch_features, train_loss=None)
                batch_preds = torch.sigmoid(batch_logits.mean).detach().cpu().numpy()
                task_preds.append(batch_preds)
        task_preds = np.concatenate(task_preds)
        del dkt_task_sample
        del batch_logits
        del batch_features
        return task_preds
