import numpy as np
import torch
import json
from fs_mol.models.linear_probe import LinearProbe
from fs_mol.configs import LinearProbeConfig

from fs_mol.utils.torch_utils import torchify

class LinearProbeEvaluator:
    def __init__(self, file="fs_mol/configs/QP/lp.json", device="cuda", **kwargs):
        self.device = device
        with open(file) as f:
            config = json.load(f)
        self.config = LinearProbeConfig(**config)
        self.model = LinearProbe(self.config)
        self.model.to(device)
        self.model.eval()
        self.model.backbone.eval()

    def __call__(self, task_sample, y_support, y_query, **kwargs):
        self.model.initialize_data_model_fs_mol(
            task_sample.train_samples,
            task_sample.test_samples,
            torch.tensor(y_support.astype(np.float32), device=self.device),
            torch.tensor(y_query.astype(np.float32), device=self.device),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-3)
        n_epochs = self.config.optimizer_config.epochs.get(y_support.shape[0], self.config.optimizer_config.epochs[128])
        self.model.backbone.eval()
        for i in range(n_epochs):
            optimizer.zero_grad()
            loss, p_all = self.model.get_loss()
            loss.backward()
            optimizer.step()
        y_pred = self.model.predict()[len(y_support):].detach().cpu().numpy()
        return y_pred

