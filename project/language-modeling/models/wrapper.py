import math
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from models.lstm import BaselineLSTM
from models.merity import MerityLSTM
from data.data_module import Lang

from typing import Any

class SequenceModelWrapper(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            cost_function: nn.Module,
            learning_rate: float = 0.1,
            momentum: float = 0.9,
            optimizer: str = "sgd",
            ntasgd: int = -1,
            tbptt: bool = False,
            batch_size: int = 128):
        super().__init__()

        self.model = model
        self.cost_fn = cost_function

        self.lr = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

        if optimizer not in ["sgd", "adam"]:
            raise ValueError("Please provide either 'sgd' or 'adam' as optimizer")

        self.optimizer = optimizer
        self.ntasgd = ntasgd
        self.ntasgd_trigger = False
        self.validation_step_loss = []
        self.validation_epochs_loss = []

        self.tbptt = tbptt
        if tbptt:
            self.automatic_optimization = False

        self.results = []

    def forward(
            self,
            inputs: torch.Tensor,
            lengths: list[int],
            hidden: list[torch.Tensor] | None = None,
            split_idx: int | None = 0,):
        if self.tbptt:
            if split_idx is None:
                raise ValueError("Please provide a split index when using TBPTT")
            return self.model(inputs, lengths, hidden, split_idx)
        else:
            return self.model(inputs, lengths, hidden)

    def training_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        inputs, targets, lengths = batch
        forward_step = self.forward_wrapper if not self.tbptt else self.tbptt_forward_wrapper
        loss, _ = forward_step(inputs, targets, lengths)
        _ = self._print_metrics(loss.item(), "Train")
        return loss


    def validation_step(self, batch: tuple[torch.tensor, torch.tensor], batch_idx: int):
        inputs, targets, lengths = batch
        loss, _ = self.forward_wrapper(inputs, targets, lengths)
        _ = self._print_metrics(loss.item(), "Valid")
        self.validation_step_loss.append(loss)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        if self.ntasgd > -1 and not self.ntasgd_trigger and self.current_epoch > self.ntasgd:
            self.validation_epochs_loss.append(torch.stack(self.validation_step_loss).mean())
            self.validation_step_loss.clear()
            is_not_improving = self.validation_epochs_loss[-1] > min(self.validation_epochs_loss[:-self.ntasgd])
            if is_not_improving:
                self._switch_to_asgd()

    def test_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        loss, outputs = self.forward_wrapper(inputs, targets, lengths)
        metrics = self._print_metrics(loss.item(), "Test")
        self.results.append({
            "inputs": inputs.numpy().squeeze(),
            "targets": targets.numpy().squeeze(),
            "lengths": lengths,
            "outputs": outputs.numpy(),
            "loss": loss.numpy(),
        })
        return metrics

    @torch.no_grad()
    def generate(
            self, 
            prompt: str, 
            lang: Lang, 
            max_len: int = 30, 
            mode: str = "argmax",
            allow_unk: bool = False,
            temperature: float = 1.0) -> str:
        
        if mode not in ["multinomial", "argmax"]:
            raise ValueError("Please provide either 'multinomial' or 'argmax' as mode")        
            
        get_pred = lambda o, m: \
                        torch.argmax(o, dim=1)[-1] if m == "argmax"  \
                        else torch.multinomial(o, num_samples=1)
        
        prompt = prompt.lower().split(" ")
        text = [lang.words2ids[w] for w in prompt]
        hidden = self.model._init_hidden(1)
        pred = None

        while pred != lang.words2ids["<eos>"] and len(text) < max_len:
            inp = torch.tensor(text).unsqueeze(0)
            length = np.asarray([len(text)], dtype=np.int64)
            output, hidden = self(inp, length, hidden)
            output = output[-1, :].unsqueeze(0)
            softmax = torch.softmax(output / temperature, dim=1)
            pred = get_pred(softmax, mode).item()

            if not allow_unk:
                while pred == lang.words2ids["<unk>"]:
                    pred = get_pred(softmax, "multinomial").item()
            
            text.append(pred)

        return " ".join([lang.ids2words[w] for w in text])


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

    def forward_wrapper(self, inputs, targets, lengths):
        outputs, _ = self(inputs, lengths)
        loss = self.cost_fn(outputs, targets.view(-1))
        return loss, outputs
    
    def tbptt_forward_wrapper(self, inputs, targets, lengths):
        hiddens = self.model._init_hidden(inputs[0].shape[0])
        batch_loss = .0
        opt = self.optimizers()
        sch = self.lr_schedulers()
        for split_idx, (inps, tars) in enumerate(zip(inputs, targets)):
            # compute lengths depending on the current split
            lengths = torch.where(inps != 0, 1, 0).cpu()
            lengths = torch.sum(lengths, dim=1)
            if split_idx > 0:
                hiddens = self._detach_hidden(hiddens)
            outputs, hiddens = self(inps, lengths, hiddens, split_idx)
            opt.zero_grad()
            loss = self.cost_fn(outputs, tars.view(-1))
            self.manual_backward(loss)
            # clip gradients
            self.clip_gradients(opt, gradient_clip_val=0.25, gradient_clip_algorithm="norm")
            opt.step()
            try: 
                sch.step(self.trainer.callback_metrics["Loss/Valid"])
            except KeyError:
                pass
            batch_loss += loss.item()
        batch_loss /= (split_idx + 1)
        batch_loss = torch.tensor(batch_loss, device=inputs[0].device)
        return batch_loss, None

    def _print_metrics(self, loss: float, stage: str):
        metrics = {f"Loss/{stage}": loss, f"Perplexity/{stage}": math.exp(loss)}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False, batch_size=self.batch_size)
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

    def _detach_hidden(self, hiddens):
        hiddens, cells = hiddens
        hiddens = [hidden.detach() for hidden in hiddens]
        cells = [cell.detach() for cell in cells]
        return hiddens, cells

    @classmethod
    def dump_results(cls, results: dict[str, Any], filename: str):
        with open(filename, "wb") as f:
            pickle.dump(results, f)
    
    @classmethod
    def load_model(
            cls,
            checkpoint_path: str,
            map_location: str,
            model: nn.Module,
            cost_function: nn.Module,
            batch_size: int = 1,
        ) -> "SequenceModelWrapper":

        state_dict = torch.load(checkpoint_path, map_location=map_location)["state_dict"]

        if isinstance(model, MerityLSTM):
            print("")
            to_rmv = []
            for k in state_dict.keys():
                if k.startswith("model.old_lstm.weight_hh") and (not k.endswith('_raw')):
                    to_rmv.append(k)
            for layer in to_rmv:
                del state_dict[layer]

        wrapper = cls(model, cost_function, batch_size=batch_size)
        wrapper.load_state_dict(state_dict)
        return wrapper
