import torch
import torch.nn as nn

from typing import Union, List

class BaselineLSTM(nn.Module):
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

        self.embedding = nn.Embedding(num_classes, embedding_dim, pad_value)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

        if init_weights:
            self.init_weights()

    def forward(
            self, 
            inputs: torch.Tensor, 
            lengths: List[int],
            hidden: Union[List[torch.Tensor], None] = None):
        embedding = self.embedding(inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed_inputs, hidden)    
        outputs, output_lenghts = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        prediction = self.fc(outputs)
        prediction = prediction.reshape(-1, prediction.shape[2])
        return prediction, (hidden, cell)

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                    elif 'weight_hh' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                    elif 'bias' in name:
                        param.data.fill_(0)
            else:
                if type(m) in [nn.Linear]:
                    torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias != None:
                        m.bias.data.fill_(0.01)