import torch
import torch.nn as nn

from typing import Union, List

from .lstm import BaselineLSTM
from .dropout import LockedDropout, WeightDropLSTM

class MerityLSTM(BaselineLSTM):
    def __init__(
            self, 
            num_classes: int, 
            embedding_dim: int, 
            hidden_dim: int, 
            num_layers: int = 1, 
            pad_value: int = 0,
            locked_dropout = False,
            p_lockdrop: float = 0.5,
            embedding_dropout = False,
            p_embdrop: float = 0.5,
            init_weights: bool = False,
            tie_weights: bool = False):
        super().__init__(
            num_classes, 
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            pad_value,)
        
        self.locked_dropout = locked_dropout
        if locked_dropout:
            self.in_locked_dropout = LockedDropout(p = p_lockdrop)
            self.out_locked_dropout = LockedDropout(p = p_lockdrop)
          
        self.embedding_dropout = embedding_dropout
        self.p_embdrop = p_embdrop

        if tie_weights:
            assert embedding_dim == hidden_dim, 'cannot tie, check dims'
            self.embedding.weight = self.fc.weight

        if init_weights:
            self.dumb()
            self._init_weights()

    def forward(
            self, 
            inputs: torch.Tensor, 
            lengths: List[int],
            hidden: Union[List[torch.Tensor], None] = None):
        # embedding dropout
        embedding = self._embedded_dropout(self.embedding, inputs, self.p_embdrop) if self.embedding_dropout else self.embedding(inputs) 
        # in locked dropout
        embedding = self.in_locked_dropout(embedding) if self.locked_dropout else embedding
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.lstm(packed_inputs, hidden)    
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # out locked dropout
        outputs = self.out_locked_dropout(outputs) if self.locked_dropout else embedding
        prediction = self.fc(outputs)
        prediction = prediction.reshape(-1, prediction.shape[2])
        return prediction, hidden
    def dumb(self):
        print("dio")

    def _init_weights(self):
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
      
    def embedded_dropout(self, embed, words, dropout=0.1, scale=None):
        if dropout:
            mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
            masked_embed_weight = mask * embed.weight
        else:
            masked_embed_weight = embed.weight
        if scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = torch.nn.functional.embedding(words, masked_embed_weight,
            padding_idx, embed.max_norm, embed.norm_type,
            embed.scale_grad_by_freq, embed.sparse
        )
        return X
