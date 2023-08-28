import numpy as np
import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(
            self,
            num_classes: int,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
            pad_value: int = 0,
            p_dropout: float | None = None,
            layer_norm: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pad_value = pad_value

        self.embedding = nn.Embedding(num_classes, embedding_dim, pad_value)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,)
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None
        self.norm = nn.LayerNorm(embedding_dim) if layer_norm else None

    def forward(
            self,
            inputs: torch.Tensor,
            lengths: np.ndarray[int],
            hidden: list[torch.Tensor] | None = None):
        embedding = self.embedding(inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed_inputs, hidden)
        outputs, output_lenghts = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        if self.norm is not None:
            outputs = self.norm(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        prediction = self.fc(outputs)
        prediction = prediction.reshape(-1, prediction.shape[2])
        return prediction, (hidden, cell)