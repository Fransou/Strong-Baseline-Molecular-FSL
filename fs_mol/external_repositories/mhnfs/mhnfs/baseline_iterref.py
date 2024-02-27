# Libraries
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import pytorch_lightning as pl

from modules import EncoderBlock, LayerNormalizingBlock


#---------------------------------------------------------------------------------------
# Mappings

#---------------------------------------------------------------------------------------
# Functions and modules
# - Loss
def BCE(predictions, labels):
    criterion = torch.nn.BCELoss()

    loss = criterion(predictions, labels)
    return loss
losses = {'BCE': BCE,}

# - IterRef Module
class IterRefEmbedding(nn.Module):
    def __init__(self, config: OmegaConf):
        super(IterRefEmbedding, self).__init__()

        self.config = config

        self.lstm_s = torch.nn.LSTM(
            input_size=config.model.associationSpace_dim * 2,
            hidden_size=config.model.associationSpace_dim,
            batch_first=True,
        )

        self.lstm_q = torch.nn.LSTM(
            input_size=config.model.associationSpace_dim * 2,
            hidden_size=config.model.associationSpace_dim,
            batch_first=True,
        )

    def forward(
        self,
        query_representation,
        support_set_actives_representation,
        support_set_inactives_representation,
    ):
        def cosine_distance(x, y):
            div_stabilizer = torch.tensor([1e-8]).to(x.device)

            x_norm = x.norm(p=2, dim=2, keepdim=True)
            x = x.div(x_norm.expand_as(x) + div_stabilizer)

            y_norm = y.norm(p=2, dim=2, keepdim=True)
            y = y.div(y_norm.expand_as(y) + div_stabilizer)

            sim = x @ torch.transpose(y, 1, 2)

            return sim

        # Initialization:
        # Initialize refinement delta values
        support_set_representation = torch.cat(
            [support_set_actives_representation, support_set_inactives_representation],
            1,
        )
        q_refine = torch.zeros_like(query_representation)
        s_refine = torch.zeros_like(support_set_representation)

        # Initialize temp set of attention mechanism
        z = support_set_representation

        # Initialize states for lstms
        h_s = torch.unsqueeze(
            torch.zeros_like(
                support_set_representation.reshape(
                    -1, self.config.model.associationSpace_dim
                )
            ),
            0,
        )
        c_s = torch.unsqueeze(
            torch.zeros_like(
                support_set_representation.reshape(
                    -1, self.config.model.associationSpace_dim
                )
            ),
            0,
        )
        h_q = torch.unsqueeze(
            torch.zeros_like(
                query_representation.reshape(-1, self.config.model.associationSpace_dim)
            ),
            0,
        )
        c_q = torch.unsqueeze(
            torch.zeros_like(
                query_representation.reshape(-1, self.config.model.associationSpace_dim)
            ),
            0,
        )

        for i in range(self.config.model.number_iteration_steps):
            # Attention mechanism
            # - Support set
            cosine_sim_s = cosine_distance(z + s_refine, support_set_representation)
            attention_values_s = torch.nn.Softmax(dim=2)(cosine_sim_s)
            linear_comb_s = attention_values_s @ support_set_representation

            # - Query
            cosine_sim_q = cosine_distance(query_representation + q_refine, z)
            attention_values_q = torch.nn.Softmax(dim=2)(cosine_sim_q)
            linear_comb_q = attention_values_q @ z

            # Concatenate and prepare variables for lstms
            s_lstm_in = torch.cat([s_refine, linear_comb_s], dim=2)
            q_lstm_in = torch.cat([q_refine, linear_comb_q], dim=2)

            # Feed inputs in lstm
            s_lstm_in = torch.unsqueeze(
                s_lstm_in.reshape(-1, self.config.model.associationSpace_dim * 2), 1
            )
            s_refine, (h_s, c_s) = self.lstm_s(s_lstm_in, (h_s, c_s))
            s_refine = s_refine.reshape(-1, 24, self.config.model.associationSpace_dim)

            q_lstm_in = torch.unsqueeze(
                q_lstm_in.reshape(-1, self.config.model.associationSpace_dim * 2), 1
            )
            q_refine, (h_q, c_q) = self.lstm_q(q_lstm_in, (h_q, c_q))
            q_refine = q_refine.reshape(-1, 1, self.config.model.associationSpace_dim)

            # Update temp set for attention mechnism
            z = linear_comb_s

        q_updated = query_representation + q_refine
        s_updated = support_set_representation + s_refine

        s_updated_actices = s_updated[:, :12, :]
        s_updated_inactices = s_updated[:, 12:, :]

        return q_updated, s_updated_actices, s_updated_inactices

# -distance metrics
def cosineSim(
    Q,
    S,
    supportSetSize,
    scaling,
    device="cpu",
    l2Norm=True,
):
    """
    Similarity search approach, based on
    - query-, support sets split for a multi task setting
    - metric: cosine similarity
    - support-set here only consists of active molecules
    - only pytorch supported
    :param Q: query-set, torch tensor, shape[numb_tasks,*,d]
    :param S: support-set, torch tensor, shape[numb_tasks,*,d]
    :return: Predictions for each query molecule in every task
    """
    # Support-set sizes
    # supportSet_sizes = list()
    # for task_idx in range(S.shape[0]):
    #    size = supportSetSize[task_idx]
    #    size = S[task_idx,:,:].shape[0]
    #    supportSet_sizes.append(size)
    # supportSet_sizes = torch.tensor(supportSet_sizes).float().reshape(-1,1).to(device=device)

    # L2 - Norm

    if l2Norm == True:
        Q_div = torch.unsqueeze(Q.pow(2).sum(dim=2).sqrt(), 2)
        Q_div[Q_div == 0] = 1  
        S_div = torch.unsqueeze(S.pow(2).sum(dim=2).sqrt(), 2)
        S_div[S_div == 0] = 1  

        Q = Q / Q_div
        S = S / S_div

    similarities = Q @ torch.transpose(S, 1, 2)

    # mask: remove padded support set artefacts
    mask = torch.zeros_like(similarities)
    for task_idx in range(S.shape[0]):
        realSize = supportSetSize[task_idx]
        if realSize > 0:
            mask[task_idx, :, :realSize] = torch.ones_like(mask[task_idx, :, :realSize])

    similarities = similarities * mask

    similaritySums = similarities.sum(
        dim=2
    )  # For every query molecule: Sum over support set molecules

    if scaling == "1/N":
        stabilizer = torch.tensor(1e-8).float()
        predictions = (
            1 / (2.0 * supportSetSize.reshape(-1, 1) + stabilizer) * similaritySums
        )
    if scaling == "1/sqrt(N)":
        stabilizer = torch.tensor(1e-8).float()
        predictions = (
            1
            / (2.0 * torch.sqrt(supportSetSize.reshape(-1, 1).float()) + stabilizer)
            * similaritySums
        )

    return predictions
distance_metrics = {"cosineSim": cosineSim,}

# -optimizer
def define_opimizer(config, parameters):
    if config.model.training.optimizer == 'AdamW':
        base_optimizer = torch.optim.AdamW
    elif config.model.training.optimizer == 'SGD':
        base_optimizer = torch.optim.SGD
    else:
        base_optimizer = torch.optim.Adam

    optimizer = base_optimizer(parameters, lr=config.model.training.lr, weight_decay=config.model.training.weightDecay)

    if config.model.training.lrScheduler.usage:
        lrs_1 = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                  start_factor=config.model.training.lr,
                                                  total_iters=5)
        lrs_2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.994)

        lrs = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                    schedulers=[lrs_1, lrs_2],
                                                    milestones=[40])

        lr_dict = {
            'scheduler': lrs,
            'monitor': 'loss_val'
        }

        return [optimizer], [lr_dict]
    else:
        return optimizer

#---------------------------------------------------------------------------------------
# IterRef based few-shot model
class IterRefNeuralSearch(pl.LightningModule):
    def __init__(self, config: OmegaConf):
        super(IterRefNeuralSearch, self).__init__()

        # Config
        self.config = config

        # Loss functions
        self.LossFunction = losses[config.model.training.loss]

        # Hyperparameter
        self.save_hyperparameters(config)

        # Encoder
        self.encoder = EncoderBlock(config)

        # Layernormalizing-block
        if self.config.model.layerNormBlock.usage == True:
            self.layerNormBlock = LayerNormalizingBlock(config)

        # IterRefEmbedding-block
        self.iterRefEmbeddingBlock = IterRefEmbedding(config)

        # Similarity Block
        self.similarity_function = distance_metrics[config.model.similarityBlock.type]

        # Output function
        self.sigmoid = torch.nn.Sigmoid()
        self.prediction_scaling = config.model.prediction_scaling

    def forward(
        self,
        queryMols,
        supportMolsActive,
        supportMolsInactive,
        supportSetActivesSize=0,
        supportSetInactivesSize=0,
    ):
        # Embeddings
        query_embedding = self.encoder(queryMols)
        supportActives_embedding = self.encoder(supportMolsActive)
        supportInactives_embedding = self.encoder(
            supportMolsInactive
        )  # Todo: add if clause below

        # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
        #    supportInactives_embedding = self.encoder(supportMolsInactive)

        # Layer normalization:
        if self.config.model.layerNormBlock.usage == True:
            # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
            (
                query_embedding,
                supportActives_embedding,
                supportInactives_embedding,
            ) = self.layerNormBlock(
                query_embedding, supportActives_embedding, supportInactives_embedding
            )
            # else:
            #    (query_embedding, supportActives_embedding,
            #     supportInactives_embedding) = self.layerNormBlock(query_embedding, supportActives_embedding,
            #                                                       None)

        # IterRef
        (
            query_embedding,
            supportActives_embedding,
            supportInactives_embedding,
        ) = self.iterRefEmbeddingBlock(
            query_embedding, supportActives_embedding, supportInactives_embedding
        )

        # Similarities:
        predictions_supportActives = self.similarity_function(
            query_embedding,
            supportActives_embedding,
            supportSetActivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )
        # if (self.config.targetSet.numberInactives != 0 or self.config.targetSet.ratioInactives != -1.):
        _predictions_supportInactives = self.similarity_function(
            query_embedding,
            supportInactives_embedding,
            supportSetInactivesSize,
            device=self.device,
            scaling=self.config.model.similarityBlock.scaling,
            l2Norm=self.config.model.similarityBlock.l2Norm,
        )
        # predictions_supportInactives = 1. - _predictions_supportInactives
        # predictions = 0.5 * (predictions_supportActives + predictions_supportInactives)
        predictions = predictions_supportActives - _predictions_supportInactives
        # else:
        #    predictions = predictions_supportActives

        predictions = self.sigmoid(self.prediction_scaling * predictions)

        return predictions

    def training_step(self, batch, batch_idx):
        queryMols = batch["queryMolecule"]
        labels = batch["label"]
        supportMolsActive = batch["supportSetActives"]
        supportMolsInactive = batch["supportSetInactives"]
        supportSetActivesSize = batch["supportSetActivesSize"]
        supportSetInactivesSize = batch["supportSetInactivesSize"]
        target_idx = batch["taskIdx"]

        predictions = self.forward(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
        )
        predictions = torch.squeeze(predictions)

        loss = self.LossFunction(predictions, labels.reshape(-1))

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        return output

    def validation_step(self, batch, batch_idx):
        queryMols = batch["queryMolecule"]
        labels = batch["label"].squeeze().float()
        supportMolsActive = batch["supportSetActives"]
        supportMolsInactive = batch["supportSetInactives"]
        supportSetActivesSize = batch["supportSetActivesSize"]
        supportSetInactivesSize = batch["supportSetActivesSize"]
        # number_querySet_actives = batch['number_querySet_actives']
        # number_querySet_inactives = batch['number_querySet_inactives']
        target_idx = batch["taskIdx"]

        predictions = self.forward(
            queryMols,
            supportMolsActive,
            supportMolsInactive,
            supportSetActivesSize,
            supportSetInactivesSize,
        ).float()

        loss = self.LossFunction(predictions.reshape(-1), labels)
        # loss = self.LossFunction(predictions[labels!= -1], labels[labels!= -1])

        output = {
            "loss": loss,
            "predictions": predictions,
            "labels": labels,
            "target_idx": target_idx,
        }
        #'number_querySet_actives':number_querySet_actives,
        #'number_querySet_inactives':number_querySet_inactives}
        return output

    def training_epoch_end(self, step_outputs):
        log_dict_training = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], axis=0)
        labels = torch.cat([x["labels"] for x in step_outputs], axis=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], axis=0)

        pred_max = torch.max(predictions)
        pred_min = torch.min(predictions)

        auc, _, _ = auc_score_train(predictions.cpu(), labels.cpu(), target_ids.cpu())
        deltaAUPRC, _, _ = deltaAUPRC_score_train(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )

        epoch_dict = {
            "loss_train": epoch_loss,
            "auc_train": auc,
            "dAUPRC_train": deltaAUPRC,
            "debug_pred_max_train": pred_max,
            "debug_pred_min_train": pred_min,
        }
        log_dict_training.update(epoch_dict)
        self.log_dict(log_dict_training, "training", on_epoch=True)

    def validation_epoch_end(self, step_outputs):
        log_dict_val = dict()

        # Predictions
        predictions = torch.cat([x["predictions"] for x in step_outputs], axis=0)
        labels = torch.cat([x["labels"] for x in step_outputs], axis=0)
        epoch_loss = torch.mean(torch.tensor([x["loss"] for x in step_outputs]))
        target_ids = torch.cat([x["target_idx"] for x in step_outputs], axis=0)

        auc, _, _ = auc_score_train(predictions.cpu(), labels.cpu(), target_ids.cpu())
        deltaAUPRC, _, _ = deltaAUPRC_score_train(
            predictions.cpu(), labels.cpu(), target_ids.cpu()
        )

        epoch_dict = {"loss_val": epoch_loss, "auc_val": auc, "dAUPRC_val": deltaAUPRC}
        log_dict_val.update(epoch_dict)
        self.log_dict(log_dict_val, "validation", on_epoch=True)

        # epoch_loss_seeds = torch.zeros(5)
        # auc_seeds = np.zeros(5)
        # dauprc_seeds = np.zeros(5)

        # Predictions
        # for dl_idx in range(5):
        #    predictions = torch.cat([x['predictions'] for x in step_outputs[dl_idx]], axis=0)
        #    labels = torch.cat([x['labels'] for x in step_outputs[dl_idx]], axis=0)
        #    epoch_loss = torch.sum(torch.tensor([x["loss"] for x in step_outputs[dl_idx]]))
        #    target_ids = torch.cat([x['target_idx'] for x in step_outputs[dl_idx]], axis=0)
        #    number_querySet_actives = torch.cat([x['number_querySet_actives'] for x in step_outputs[dl_idx]], axis=0)
        #    number_querySet_inactives = torch.cat([x['number_querySet_inactives'] for x in step_outputs[dl_idx]], axis=0)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(),
        #                             lr=self.config.model.training.lr,
        #                             weight_decay=self.config.model.training.weightDecay)
        # if self.config.model.training.lrScheduler.usage == True:
        #    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                      patience=self.config.model.training.lrScheduler.patience,
        #                                                      factor=self.config.model.training.lrScheduler.factor,
        #                                                      min_lr=self.config.model.training.lrScheduler.min_lr)
        #    lr_dict = {
        #        'scheduler': lr_scheduler,
        #        'monitor': 'loss_val'
        #    }
        #    return [optimizer], [lr_dict]
        # else:
        #    return optimizer
        return define_opimizer(self.config, self.parameters())
