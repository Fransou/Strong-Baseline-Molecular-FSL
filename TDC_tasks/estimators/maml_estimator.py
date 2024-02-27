import numpy as np
import torch
from dataclasses import dataclass

from fs_mol.maml_test import load_model_for_eval
from fs_mol.utils.maml_utils import eval_model_by_finetuning_on_task
from fs_mol.utils.torch_utils import torchify


@dataclass(frozen=True)
class MamlArgs:
    trained_model: str = "backbone_pretrained_models/MAML-Support16_best_validation.pkl"
    model_params_override: dict = None
    use_fresh_param_init: bool = False


class MamlEvaluator:
    def __init__(self, file="backbone_pretrained_models/MAML-Support16_best_validation.pkl", device="cuda", **kwargs):
        args = MamlArgs(trained_model=file, **kwargs)

        self.model = load_model_for_eval(args)
        self.base_model_weights = {var.name: var.value() for var in self.model.trainable_variables}

    def __call__(self, task_sample, y_support, y_query, **kwargs):
        results = eval_model_by_finetuning_on_task(
            model=self.model,
            model_weights=self.base_model_weights,
            task_sample=task_sample,
            temp_out_folder="",
            max_num_nodes_in_batch=10000,
            metric_to_use="avg_precision",
            quiet=True,
            return_preds=True,
        )
        y_pred = np.concatenate([r["predictions"].numpy() for r in results], axis=0)
        return y_pred
