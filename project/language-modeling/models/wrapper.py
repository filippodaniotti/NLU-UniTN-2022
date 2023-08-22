import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

class SequenceModelWrapper(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            cost_function: nn.Module,
            learning_rate: float = 0.1,
            momentum: float = 0.9,
            optimizer: str = "sgd",
            ntasgd: int = -1,):
        super().__init__()

        self.model = model
        self.cost_fn = cost_function
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
            lengths: list[int],
            hidden: list[torch.Tensor] | None = None):
        return self.model(inputs, lengths, hidden)

    def training_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        inputs, targets, lengths = batch

        loss, ppl = self._forward_loss_ppl(inputs, targets, lengths)
        metrics = self._print_metrics(loss.item(), ppl.item(), "Train")
        return loss


    def validation_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        inputs, targets, lengths = batch
        loss, ppl = self._forward_loss_ppl(inputs, targets, lengths)
        metrics = self._print_metrics(loss.item(), ppl.item(), "Valid")
        self.logs.append(ppl)
        return metrics
    
    def on_validation_epoch_end(self) -> None:
        if (self.ntasgd > -1  and not self.ntasgd_trigger and (self.current_epoch >= self.ntasgd and self.logs[-1] > min(self.logs[:-self.ntasgd]))):
            self._switch_to_asgd()


    def test_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        loss, ppl = self._forward_loss_ppl(inputs, targets, lengths)
        metrics = self._print_metrics(loss.item(), ppl.item(), "Test")
        return metrics


    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=1e-6,
                momentum=self.momentum,
                nesterov=True)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, verbose=True)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "Loss/Valid",
                    "interval": "epoch",
                },
            }

    def _forward_loss_ppl(self, inputs, targets, lengths):
        outputs, hiddens = self(inputs, lengths)
        loss = self.cost_fn(outputs, targets.view(-1))
        return loss, torch.exp(loss)
    
    def _tbptt_forward_loss_ppl(self, inputs, targets, lengths):
        hiddens = None
        batch_loss = .0
        opt = self.optimizers()
        for split_idx, (inps, tars) in enumerate(zip(inputs, targets)):
            # compute lengths
            # detach hiddens
            outputs, hiddens = self(inps, lengths, hiddens)
            opt.zero_grad()
            loss = self.cost_fn(outputs, tars.view(-1))
            self.manual_backward(loss)
            # clip gradient
            opt.step()
            batch_loss += loss.item()
        batch_loss /= (split_idx + 1)
        return batch_loss, torch.exp(batch_loss)

    def _print_metrics(self, loss: float, ppl: float, stage: str):
        metrics = {f"Loss/{stage}": loss, f"Perplexity/{stage}": ppl}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False, batch_size=128)
        return metrics

    def _switch_to_asgd(self):
        self.print(f"Using NT-ASGD at epoch {self.current_epoch}")
        self.ntasgd_trigger = True
        # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/core/optimizer.html#LightningOptimizer
        # Note by the author: thank God Python does not have private attributes
        self.optimizers()._optimizer = torch.optim.ASGD(
            self.parameters(),
            lr=self.lr,
            t0=0,
            lambd=0.,
            weight_decay=1e-6)
        self.lr_schedulers().optimizer = self.optimizers()._optimizer

    def _get_tbptt_step(self):
        mu = self.tbptt_config["mu"]
        std = self.tbptt_config["std"]
        p = self.tbptt_config["p"]

        mu = mu if np.random.random() < p else mu/2
        tbptt_step = int(np.random.normal(mu, std))
        tbptt_step = max(5, tbptt_step)
        tbptt_step = min(tbptt_step, 82-10)

        return tbptt_step

    def _detach_hidden(self, hiddens):
        hiddens, cells = hiddens
        hiddens = [hidden.detach() for hidden in hiddens]
        cells = [cell.detach() for cell in cells]
        return hiddens, cells