import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from typing import List, Union

class SequenceModelWrapper(pl.LightningModule):
    def __init__(
            self, 
            model: nn.Module,
            learning_rate: float = 0.1,
            momentum: float = 0.9,
            optimizer: str = "sgd",
            ntasgd: int = -1):
        super().__init__()

        self.model = model
        self.lr = learning_rate
        self.momentum = momentum

        if optimizer not in ["sgd", "adam"]:
            raise ValueError("Please provide either 'sgd' or 'adam' as optimizer")

        self.optimizer = optimizer
        self.ntasgd = ntasgd
        self.ntasgd_trigger = False
        self.logs = []

    def forward(
            self, 
            inputs: torch.Tensor, 
            lengths: List[int],
            hidden: Union[List[torch.Tensor], None] = None):
        return self.model(inputs, lengths, hidden)
        
    def training_step(self, batch, batch_idx):
        return self._compute_forward_and_loss(batch)

    def training_epoch_end(self, outputs):
        optimizer = self.optimizers()
        opt = optimizer.optimizer
        print(opt)
        _, _ = self._compute_epoch_level_metrics(outputs, "Train")
 
    def validation_step(self, batch, batch_idx):
        return self._compute_forward_and_loss(batch)

    def validation_epoch_end(self, outputs) -> None:       
        _, perplexity = self._compute_epoch_level_metrics(outputs, "Validation")
        self.logs.append(perplexity)
        if (self.ntasgd > -1  and not self.ntasgd_trigger and (self.current_epoch >= self.ntasgd and perplexity > min(self.logs[:-self.ntasgd]))):
            self.ntasgd_trigger = True
            # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/optimizer.html#LightningOptimizer
            # Note by the author: thank God Python does not have private attributes
            self.optimizers()._optimizer = torch.optim.ASGD(
                self.parameters(), 
                lr=self.lr, 
                t0=0, 
                lambd=0., 
                weight_decay=1e-6)

    def test_step(self, batch, batch_idx):
        return self._compute_forward_and_loss(batch)

    def test_epoch_end(self, outputs) -> None:       
        _, _ = self._compute_epoch_level_metrics(outputs, "Test")

    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=1e-6, 
                momentum=self.momentum,
                nesterov=True)
        elif self.optimizer == "adam":
            optimizer = optim.Adam(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=1e-6)
        return optimizer

    def _compute_forward_and_loss(self, batch):
        inputs, targets, lengths = batch
        outputs, _ = self(inputs, lengths)
        targets = targets.view(-1)
        loss = F.cross_entropy(outputs, targets, ignore_index=self.model.pad_value)
        return {"loss": loss}

    def _compute_epoch_level_metrics(self, outputs, stage_name: str) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        perplexity = math.exp(loss)
        self.logger.experiment.add_scalar(
            f"Loss/{stage_name}",
            loss,
            self.current_epoch)
         
        self.logger.experiment.add_scalar(
            f"Perplexity/{stage_name}",
            perplexity,
            self.current_epoch)
        return loss, perplexity