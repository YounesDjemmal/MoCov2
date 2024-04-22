import copy
import torch
import torch.nn as nn
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
import pytorch_lightning as pl
from lightly.loss import NTXentLoss

class MocoV2Model(pl.LightningModule):
    def __init__(self, memory_bank_size=4096, max_epochs=100):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(8192, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=0.1, memory_bank_size=(memory_bank_size, 128)
        )
        self.max_epochs = max_epochs

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.999)  # increased momentum
        update_momentum(self.projection_head, self.projection_head_momentum, 0.999)  # increased momentum

        # get queries
        q1 = self.backbone(x_q).flatten(start_dim=1)
        q1 = self.projection_head(q1)
        q2 = self.backbone(x_k).flatten(start_dim=1)
        q2 = self.projection_head(q2)

        # get keys
        k1, shuffle = batch_shuffle(x_q)
        k1 = self.backbone_momentum(k1).flatten(start_dim=1)
        k1 = self.projection_head_momentum(k1)
        k1 = batch_unshuffle(k1, shuffle)

        k2, shuffle = batch_shuffle(x_k)
        k2 = self.backbone_momentum(k2).flatten(start_dim=1)
        k2 = self.projection_head_momentum(k2)
        k2 = batch_unshuffle(k2, shuffle)

        # symmetrized loss
        loss = self.criterion(q1, k2) + self.criterion(q2, k1)
        self.log("train_loss_ssl", loss)
        return loss
    
    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params.detach().cpu().numpy(), self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]