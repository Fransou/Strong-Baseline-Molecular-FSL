import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import hydra
from argparse import ArgumentParser

from dataloader import FSMolDataModule

from models import MHNfs


@hydra.main(config_path="../configs", config_name="cfg")
def train(cfg):
    """
    Training loop for model training on FS-Mol.

    - A FS-Mol data-module includes dataloader for training, validation and test.
    - MHNfs defines the model which is trained
    - a pytorch lightning trainer object takes the model and the data module in performs
      model training
    - For logging, we use wandb

    inputs:
    - cfg: hydra config file
    """

    seed_everything(cfg.training.seed)

    # Load data module
    dm = FSMolDataModule(cfg)

    # Load model
    model = MHNfs(cfg).to("cuda")
    model._update_context_set_embedding()

    # Prepare logger
    logger = pl_loggers.WandbLogger(
        save_dir="../logs/", name=cfg.experiment_name, project=cfg.project_name
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="dAUPRC_val", mode="max", save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Setup trainer
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=cfg.training.epochs,
        accumulate_grad_batches=5,
    )

    # Train
    trainer.fit(model, dm)


if __name__ == "__main__":
    train()
