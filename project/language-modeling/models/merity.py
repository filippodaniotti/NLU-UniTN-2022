import torch
import torch.nn as nn
from torch.autograd import Variable

from typing import Union, List

from lstm import BaselineLSTM


class LockedDropout(nn.Module):
    """
    https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class MerityLSTM(BaselineLSTM):
    def __init__(
            self, 
            num_classes: int, 
            embedding_dim: int, 
            hidden_dim: int, 
            num_layers: int = 1, 
            pad_value: int = 0,
            p_lockdrop: float = 0.5,
            init_weights: bool = True,
            tie_weights: bool = True):

        super().__init__(
            num_classes, 
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            pad_value,)
        
        self.in_locked_dropout = LockedDropout(p = p_lockdrop)
        self.out_locked_dropout = LockedDropout(p = p_lockdrop)
        
        if tie_weights:
            assert embedding_dim == hidden_dim, 'cannot tie, check dims'
            self.embedding.weight = self.fc.weight

        if init_weights:
            self.init_weights()


    def forward(
            self, 
            inputs: torch.Tensor, 
            lengths: List[int],
            hidden: Union[List[torch.Tensor], None] = None):
        embedding = self.embedding(inputs)
        embedding = self.in_locked_dropout(embedding)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.lstm(packed_inputs, hidden)    
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = self.out_locked_dropout(outputs)
        prediction = self.fc(outputs)
        prediction = prediction.reshape(-1, prediction.shape[2])
        return prediction, hidden
        

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
            elif type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
            elif type(m) in [nn.Embedding]:
                torch.nn.init.uniform_(m.weight, -0.1, 0.1)

