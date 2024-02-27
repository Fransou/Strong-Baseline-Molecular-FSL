import numpy as np
import torch
import json
from fs_mol.models.clamp_model import ClampFSL, ClampConfig

from fs_mol.utils.torch_utils import torchify


FPSIZE = 4096
FP_MODE = 'morganc+rdkc'

class ClampEvaluator:
    def __init__(self, file="fs_mol/configs/Clamp/eval.json", device="cuda", **kwargs):
        self.device = device
        with open(file) as f:
            config = json.load(f)
        self.config = ClampConfig(**config)


    def __call__(self, fp_supp, fp_query, y_support, y_query, **kwargs):
        self.model = ClampFSL(
            smiles={
                "support": fp_supp.to(self.device),
                "query": fp_query.to(self.device),
            },
            y_support=torch.tensor(y_support),
            y_query=torch.tensor(y_query),
            other_data={},
            config=self.config
        )
        self.model.to(self.device)
        self.model.eval()
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



def prepro_smiles(smi):
    from mhnreact.molutils import convert_smiles_to_fp

    fp_size = FPSIZE
    fp_inp = convert_smiles_to_fp(smi, which=FP_MODE, fp_size=fp_size, njobs=1).astype(
        np.float32
    )
    compound_features = torch.tensor(fp_inp)
    return compound_features