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
            ntasgd: int = -1,
            tbptt: bool = False,
            tbptt_config: Dict[str, int] = {}):
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
        
        self.tbptt = tbptt
        if tbptt:
            # placeholder, computed according to Merity et al.
            self.truncated_bptt_steps = 5
            self.tbptt_config = tbptt_config

    def forward(
            self, 
            inputs: torch.Tensor, 
            lengths: List[int],
            hidden: Union[List[torch.Tensor], None] = None):
        return self.model(inputs, lengths, hidden)
        
    def training_step(self, batch, batch_idx, *args):
        if self.tbptt:
            hiddens = args[0]
        return self._compute_forward_and_loss(batch)

    def training_epoch_end(self, outputs):
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
        return optimizer

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        if self.tbptt:
            self.truncated_bptt_steps = self._get_tbptt_step()

    def tbptt_split_batch(self, batch: Any, split_size: int) -> List[Any]:
        splits = []
        return [batch]
    # def tbptt_split_batch(self, batch, split_size):
    #     splits = []
    #     for t in range(0, time_dims[0], split_size):
    #         batch_split = []
    #         for i, x in enumerate(batch):
    #             if isinstance(x, torch.Tensor):
    #                 split_x = x[:, t:t + split_size]
    #             elif isinstance(x, collections.abc.Sequence):
    #                 split_x = [None] * len(x)
    #                 for batch_idx in range(len(x)):
    #                   split_x[batch_idx] = x[batch_idx][t:t + split_size]
    #             batch_split.append(split_x)
    #         splits.append(batch_split)
    #     return splits

    def _compute_forward_and_loss(self, batch):
        inputs, targets, lengths = batch
        outputs, (hidden, _) = self(inputs, lengths)
        targets = targets.view(-1)
        loss = F.cross_entropy(outputs, targets, ignore_index=self.model.pad_value)
        return_dict = {"loss": loss}
        if self.tbptt:
            return_dict["hiddens"] = hidden
        return return_dict

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

    def _get_tbptt_step(self):
        mu = self.tbptt_config["mu"]
        std = self.tbptt_config["std"]
        p = self.tbptt_config["p"]

        mu = mu if np.random.random() < p else mu/2
        tbptt_step = int(np.random.normal(mu, std)) 
        tbptt_step = max(5, tbptt_step)
        tbptt_step = min(tbptt_step, 82-10)

        return tbptt_step