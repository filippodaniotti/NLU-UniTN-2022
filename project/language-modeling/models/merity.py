import torch
import torch.nn as nn

from .lstm import BaselineLSTM
from .dropout import LockedDropout, WeightDropLSTM, EmbeddingDropout

from torch import tensor

class MerityLSTM(BaselineLSTM):
    def __init__(
            self,
            num_classes: int,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
            pad_value: int = 0,
            locked_dropout = False,
            p_lockdrop: float = 0.4,
            embedding_dropout = False,
            p_embdrop: float = 0.1,
            weight_dropout: bool = False,
            p_lstmdrop: float = 0.3,
            p_hiddrop: float = 0.5,
            init_weights: bool = False,
            tie_weights: bool = False):
        super().__init__(
            num_classes,
            embedding_dim,
            hidden_dim,
            num_layers,
            pad_value,)

        if embedding_dropout > .0:
            self.embedding = EmbeddingDropout(num_classes, embedding_dim, pad_value, dropout=p_embdrop)

        if weight_dropout > .0:
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                bidirectional=False,
                dropout=p_lstmdrop)
            self.old_lstm = self.lstm
            weights = [f'weight_hh_l{i}' for i in range(num_layers)]
            self.lstm = WeightDropLSTM(self.old_lstm, weights, dropout=p_hiddrop)

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
            self.apply(self._init_weights)

    def forward(
            self,
            inputs: tensor,
            lengths: list[int],
            hiddens: tuple[tensor, tensor] | None = None,
            split_idx: int | None = 0,
        ):
        batch_size = inputs.shape[0]
        # embedding dropout
        embedding = self.embedding(inputs)
        # in locked dropout
        embedding = self.in_locked_dropout(embedding) if self.locked_dropout else embedding
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)

        if split_idx > 0:
            h_n, c_n = torch.stack(hiddens[0], dim=0), torch.stack(hiddens[1], dim=0)
            h_n = h_n[:, :inputs.shape[0], :]
            c_n = c_n[:, :inputs.shape[0], :]
            hiddens = [h_n, c_n]

        packed_outputs, hiddens = self.lstm(packed_inputs, hiddens)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # out locked dropout
        outputs = self.out_locked_dropout(outputs) if self.locked_dropout else outputs
        prediction = self.fc(outputs)
        prediction = prediction.reshape(-1, prediction.shape[2])
        return prediction, hiddens

    def _init_weights(self, mat):
        for m in mat.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(
                                param[idx*mul:(idx+1)*mul])
                    elif 'weight_hh' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(
                                param[idx*mul:(idx+1)*mul])
                    elif 'bias' in name:
                        param.data.fill_(0)
            else:
                if type(m) in [nn.Embedding]:
                    torch.nn.init.uniform_(m.weight, -0.1, 0.1)
                elif type(m) in [nn.Linear]:
                    if m.bias != None:
                        m.bias.data.fill_(0.1)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden, cell