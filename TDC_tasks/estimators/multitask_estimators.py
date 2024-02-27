import torch
from typing import Optional
from functools import partial

from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.gnn_multitask import GNNMultitaskModel
from fs_mol.data.multitask import get_multitask_inference_batcher
from fs_mol.data import (
    FSMolBatcher,
    FSMolBatchIterable,
    FSMolTaskSample,
)


class MultitaskEvaluator:
    def __init__(self, file="backbone_pretrained_models/multitask_best_model.pt", device="cpu", **kwargs):
        self.device = device
        self.model_weights_file = resolve_starting_model_file(
            model_file=file,
            model_cls=GNNMultitaskModel,
            out_dir="",
            use_fresh_param_init=False,
            config_overrides={"num_tasks": 1},
            device=device,
        )


    def __call__(self, task_sample, y_support, y_query, epochs=1, **kwargs):
        model = GNNMultitaskModel.build_from_model_file(self.model_weights_file, quiet=True, device=self.device)
        load_model_weights(model, self.model_weights_file, load_task_specific_weights=False)
        model.to(self.device)
        (optimizer, lr_scheduler) = create_optimizer(
            model,
            lr=0.00005,
            task_specific_lr=0.0001,
            warmup_steps=2,
            task_specific_warmup_steps=2,
        )
        batcher = get_multitask_inference_batcher(max_num_graphs=513, device=self.device)
        train_data = FSMolBatchIterable(task_sample.train_samples, batcher, shuffle=True, seed=0)
        test_data = FSMolBatchIterable(task_sample.test_samples, batcher, shuffle=False, seed=0)
        model.train()
        for _ in range(epochs):
            for batch_features, batch_label in train_data:
                batch_logits = model(batch_features)
                batch_loss = torch.nn.BCEWithLogitsLoss()(
                    batch_logits.molecule_binary_label, batch_label.float().unsqueeze(1)
                )
                batch_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        model.eval()
        task_preds = []
        for batch_features, batch_label in test_data:
            batch_logits = model(batch_features)
            batch_preds = torch.nn.Sigmoid()(batch_logits.molecule_binary_label)
            task_preds.append(batch_preds[:, 0])
        y = torch.cat(task_preds)
        return y.detach().cpu().numpy()


def load_model_weights(
    model,
    path: str,
    load_task_specific_weights: bool,
    quiet: bool = False,
    device: Optional[torch.device] = None,
) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_model_state(checkpoint, load_task_specific_weights, quiet)


def linear_warmup(cur_step: int, warmup_steps: int = 0) -> float:
    if cur_step >= warmup_steps:
        return 1.0
    return cur_step / warmup_steps


def create_optimizer(
    model,
    lr: float = 0.001,
    task_specific_lr: float = 0.005,
    warmup_steps: int = 1000,
    task_specific_warmup_steps: int = 100,
):
    # Split parameters into shared and task-specific ones:
    shared_parameters, task_spec_parameters = [], []
    for param_name, param in model.named_parameters():
        if model.is_param_task_specific(param_name):
            task_spec_parameters.append(param)
        else:
            shared_parameters.append(param)

    opt = torch.optim.Adam(
        [
            {"params": task_spec_parameters, "lr": task_specific_lr},
            {"params": shared_parameters, "lr": lr},
        ],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt,
        lr_lambda=[
            partial(linear_warmup, warmup_steps=task_specific_warmup_steps),  # for task specific paramters
            partial(linear_warmup, warmup_steps=warmup_steps),  # for shared paramters
        ],
    )

    return opt, scheduler
