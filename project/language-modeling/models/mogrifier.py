import numpy as np

import torch
from torch import nn
from torch import zeros

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

class MogLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r
   
    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i+1) % 2 == 0: 
                h = (2*torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2*torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct
    
    
class MogLSTMLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int,
            mogrify_steps: int):
        super(MogLSTMLayer, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        assert num_layers > 0, "num_layers must be greater than 0"
        self.layers = [MogLSTMCell(embedding_dim, hidden_dim, mogrify_steps)]
        for _ in range(num_layers - 1):
            self.layers.append(MogLSTMCell(hidden_dim, hidden_dim, mogrify_steps))

    def forward(self, inputs, hiddens):
        hiddens, cells = hiddens
        hn, cn = self.layers[0](inputs, (hiddens[0], cells[0]))
        for layer, hidden, cell in zip(self.layers[1:], hiddens[1:], cells[1:]):
            hn, cn = layer(hn, (hidden, cell))

        return hn, cn
    
    def _init_hiddens(self, batch_size, device):
        hidden = zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        cell = zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return hidden, cell
    

class MogrifierLSTM(nn.Module):
    def __init__(
            self, 
            num_classes: int, 
            embedding_dim: int, 
            hidden_dim: int, 
            num_layers: int = 2,
            mogrify_steps: int = 5, 
            tie_weights: bool = True, 
            p_dropout: float = 0.5,
            pad_value: int = 0):
        super(MogrifierLSTM, self).__init__()
        self.pad_value = pad_value
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.lstm = MogLSTMLayer(embedding_dim, hidden_dim, num_layers, mogrify_steps)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p_dropout)
        
        if tie_weights:
            self.fc.weight = self.embedding.weight
        
    def forward(self, inputs, lengths, hiddens):
        embedding = self.embedding(inputs)
        hiddens = hiddens if hiddens else self._init_hiddens(inputs.shape[0], inputs.device)

        outputs = []
        hidden_states = []
        for step in range(np.max(lengths)):
            x = self.dropout(embedding[:, step])
            hn, _ = self.lstm(x, hiddens)
            out = self.fc(self.dropout(hn))
            outputs.append(out.unsqueeze(1))
            hidden_states.append(hn.unsqueeze(1))
            

        hidden_states = torch.cat(hidden_states, dim = 1)   # (batch_size, max_len, hidden_dim)
        outputs = torch.cat(outputs, dim = 1)               # (batch_size, max_len, num_classes)
        outputs = outputs.reshape(-1, outputs.shape[-1])    # (batch_size * max_len, num_classes)
        return outputs, hidden_states 
    
    def _init_hiddens(self, batch_size, device):
        return self.lstm._init_hiddens(batch_size, device)