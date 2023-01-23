import torch
import math
from torch import nn
from torchmetrics import Perplexity
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import Callable, Union, List

class BaseLSTM(pl.LightningModule):
    def __init__(
            self, 
            num_classes: int, 
            embedding_dim: int, 
            hidden_dim: int, 
            num_layers: int = 1, 
            pad_value: int = 0,
            init_weights: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pad_value = pad_value

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(
            self, 
            input: torch.Tensor, 
            lengths: List[int],
            hidden: Union[List[torch.Tensor], None] = None):
        embedding = self.embedding(input)
        packed_input = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input, hidden)    
        output, output_lenghts = pad_packed_sequence(packed_output, batch_first=True)
        prediction = self.fc(output)
        prediction = prediction.reshape(-1, prediction.shape[2])
        return prediction, (hidden, cell)
        
    def training_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        outputs, _ = self(inputs, lengths)
        targets = targets.view(-1)
        loss = F.cross_entropy(outputs, targets, ignore_index=self.pad_value)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log_dict({"train_loss": loss, "train_ppl": math.exp(loss)}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        outputs, _ = self(inputs, lengths)
        targets = targets.view(-1)
        loss = F.cross_entropy(outputs, targets, ignore_index=self.pad_value)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log_dict({"val_loss": loss, "val_ppl": math.exp(loss)}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
