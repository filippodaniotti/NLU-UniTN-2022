import torch
import torch.nn as nn

from .lstm import BaselineLSTM
from .dropout import LockedDropout, WeightDropLSTM, EmbeddingDropout

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
            weight_drop: bool = False,
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

        if embedding_dropout:
            self.embedding = EmbeddingDropout(num_classes, embedding_dim, pad_value, dropout=p_embdrop)

        if weight_drop:
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
            # self._init_weights()
            self.apply(self._init_weights)

    def forward(
            self,
            inputs: torch.Tensor,
            lengths: list[int],
            hiddens: tuple[torch.Tensor, torch.Tensor] | None = None):
        batch_size = inputs.shape[0]
        # embedding dropout
        embedding = self.embedding(inputs)
        # in locked dropout
        embedding = self.in_locked_dropout(embedding) if self.locked_dropout else embedding
        # try:
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        # if hiddens is not None:
        #     ht = hiddens[0][:, :batch_size, :].contiguous()
        #     ct = hiddens[1][:, :batch_size, :].contiguous()
        #     hidden = (ht, ct)
        packed_outputs, hidden = self.lstm(packed_inputs, hiddens)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # out locked dropout
        outputs = self.out_locked_dropout(outputs) if self.locked_dropout else outputs
        prediction = self.fc(outputs)
        prediction = prediction.reshape(-1, prediction.shape[2])
        return prediction, hidden

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
