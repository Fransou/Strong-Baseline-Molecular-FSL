"""
Performance evaluation of the MHNfs model on the FS-Mol test set 
"""

# ---------------------------------------------------------------------------------------
# Variables
# to do: path_checkpoint = "... set correct path ..."

# ---------------------------------------------------------------------------------------
# Libraries
import os
import torch
from collections import OrderedDict
import hydra
from pytorch_lightning import seed_everything
import sys

from models import MHNfs
from metrics import compute_auc_score, compute_dauprc_score

from dataloader import FSMolDataModule


# ---------------------------------------------------------------------------------------
# Functions


def provide_clean_checkpoint(path_checkpoint=path_checkpoint):
    checkpoint = torch.load(path_checkpoint)
    cleaned_state_dict = OrderedDict(
        [
            (k.replace("hopfield_chemTrainSpace", "contextModule"), v)
            for k, v in checkpoint["state_dict"].items()
        ]
    )
    cleaned_state_dict = OrderedDict(
        [
            (k.replace("transformer", "crossAttentionModule", 1), v)
            for k, v in cleaned_state_dict.items()
        ]
    )
    return cleaned_state_dict


def evaluate_performance(model, test_dataloader):
    all_preds = list()
    all_labels = list()
    all_target_ids = list()
    print("Make predictions ...")
    for batch in test_dataloader:
        # Prepare inputs
        query_embedding = batch["queryMolecule"]
        labels = batch["label"].squeeze().float()
        support_actives_embedding_padded = batch["supportSetActives"]
        support_inactives_embedding_padded = batch["supportSetInactives"]
        target_ids = batch["taskIdx"]

        support_actives_size = batch["supportSetActivesSize"]
        support_inactives_size = batch["supportSetActivesSize"]

        support_actives_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([True] * d), torch.tensor([False] * (12 - d))]
                ).reshape(1, -1)
                for d in support_actives_size
            ],
            dim=0,
        )
        support_inactives_mask = torch.cat(
            [
                torch.cat(
                    [torch.tensor([True] * d), torch.tensor([False] * (12 - d))]
                ).reshape(1, -1)
                for d in support_inactives_size
            ],
            dim=0,
        )

        # Make predictions
        predictions = (
            model(
                query_embedding.to('cuda'),
                support_actives_embedding_padded.to('cuda'),
                support_inactives_embedding_padded.to('cuda'),
                support_actives_size.to('cuda'),
                support_inactives_size.to('cuda'),
                support_actives_mask.to('cuda'),
                support_inactives_mask.to('cuda'),
            )
            .detach()
            .cpu()
            .float()
        )

        # Store batch outcome
        all_preds.append(predictions)
        all_labels.append(labels)
        all_target_ids.append(target_ids)

    predictions = torch.cat([p for p in all_preds], axis=0)
    labels = torch.cat([l for l in all_labels], axis=0)
    target_ids = torch.cat([t for t in all_target_ids], axis=0)

    # Compute metrics
    print("Compute AUC and  ΔAUC-PR ...")
    auc, _, _ = compute_auc_score(predictions, labels, target_ids)
    d_auc_pr, _, _ = compute_dauprc_score(predictions, labels, target_ids)
    return auc, d_auc_pr


@hydra.main(config_path="../configs/", config_name="cfg")
def run_evaluation_script(cfg):
    cfg.system.ressources.device = "cuda" #"cpu"

    # Set seed
    seed_everything(1234)

    # Load checkpoint
    #print("Load checkpoint ...")
    #state_dict = provide_clean_checkpoint()

    # Load model
    print("Load model ...")
    model = MHNfs(cfg)
    model = model.to('cuda')
    
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    
    #model = model.load_from_checkpoint(path_checkpoint)
    model.eval()
    model._update_context_set_embedding()

    # Load datamodule
    print("Load datamodule ...")
    dm = FSMolDataModule(cfg)
    dm.setup()
    dataloader = dm.test_dataloader()

    # Evaluation on testset
    auc, d_auc_pr = evaluate_performance(model, dataloader)
    print(f"Mean     AUC: {auc}")
    print(f"Mean ΔAUC-PR: {d_auc_pr}")


# ---------------------------------------------------------------------------------------
# Execute script
if __name__ == "__main__":
    run_evaluation_script()
