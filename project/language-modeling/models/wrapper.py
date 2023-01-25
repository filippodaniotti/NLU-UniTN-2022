import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import List, Union

class SequenceModel(pl.LightningModule):
    def __init__(
            self, 
            model: nn.Module,):
        super().__init__()
        self.model = model

    def forward(
            self, 
            inputs: torch.Tensor, 
            lengths: List[int],
            hidden: Union[List[torch.Tensor], None] = None):
        return self.model(inputs, lengths, hidden)
        
    def training_step(self, batch, batch_idx):
        return self.compute_forward_and_loss(batch)

    def training_epoch_end(self, outputs):
        self.compute_epoch_level_metrics(outputs, "Train")
 
    def validation_step(self, batch, batch_idx):
        return self.compute_forward_and_loss(batch)

    def validation_epoch_end(self, outputs) -> None:       
        self.compute_epoch_level_metrics(outputs, "Validation")

    def test_step(self, batch, batch_idx):
        return self.compute_forward_and_loss(batch)

    def test_epoch_end(self, outputs) -> None:       
        self.compute_epoch_level_metrics(outputs, "Test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=1e-6, momentum=0.9)
        return optimizer

    def compute_forward_and_loss(self, batch):
        inputs, targets, lengths = batch
        outputs, _ = self(inputs, lengths)
        targets = targets.view(-1)
        loss = F.cross_entropy(outputs, targets, ignore_index=self.model.pad_value)
        return {"loss": loss}

    def compute_epoch_level_metrics(self, outputs, stage_name: str) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            f"Loss/{stage_name}",
            loss,
            self.current_epoch)
         
        self.logger.experiment.add_scalar(
            f"Perplexity/{stage_name}",
            math.exp(loss),
            self.current_epoch)
