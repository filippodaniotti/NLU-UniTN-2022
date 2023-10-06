import numpy as np
import torch.nn as nn
from torch import stack, zeros, tensor

from .lstm import BaselineLSTM
from .dropout import LockedDropout, WeightDropLSTM, EmbeddingDropout

class MerityLSTM(BaselineLSTM):
    """
    PyTorch module implementing the AWD-LSTM model proposed in Merity et al. 
    https://arxiv.org/abs/1708.02182

    Args:
        num_classes (int): The number of output classes.
        embedding_dim (int): The dimension of the word embeddings.
        hidden_dim (int): The dimension of the hidden LSTM layers.
        num_layers (int, optional): The number of LSTM layers (default is 1).
        pad_value (int, optional): The value of the padding token (default is 0).
        locked_dropout (bool, optional): Flag for locked dropout (default is False).
        p_lockdrop (float): Probability of dropout for locked dropout (default is 0.4).
        embedding_dropout (bool, optional): Flag for apply embedding dropout (default is False).
        p_embdrop (float): Probability of dropout for embedding dropout (default is 0.1).
        weight_dropout (bool, optional): Flag for apply weight dropout (default is False).
        p_lstmdrop (float): Probability of dropout for LSTM layers (default is 0.3).
        p_hiddrop (float): Probability of dropout for hidden-to-hidden matrices (default is 0.5).
        init_weights (bool, optional): Flag for weights initialization (default is False).
        tie_weights (bool, optional): Flag for weight tying on embedding and fully connected layers (default is False).

    Attributes:
        num_layers (int): The number of LSTM layers.
        hidden_dim (int): The dimension of the hidden LSTM layers.
        embedding_dim (int): The dimension of the input word embeddings.
        pad_value (int, optional): The value of the padding token.
        embedding (nn.Embedding | EmbeddingDropout): The embedding layer.
        lstm (nn.LSTM | WeightDropLSTM): The LSTM layers.
        locked_dropout (bool): Whether locked dropout is applied.
        in_locked_dropout (LockedDropout, optional): Input locked dropout layer (if locked_dropout is True).
        out_locked_dropout (LockedDropout, optional): Output locked dropout layer (if locked_dropout is True).

    Methods:
        forward(inputs, lengths, hiddens=None, split_idx=0):
            Forward pass of the model.

        _init_weights(mat):
            Initialize the model weights.

    """
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

        if embedding_dropout:
            self.embedding = EmbeddingDropout(num_classes, embedding_dim, pad_value, dropout=p_embdrop)

        if weight_dropout:
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

        if tie_weights:
            assert embedding_dim == hidden_dim, 'cannot tie, check dims'
            self.embedding.weight = self.fc.weight

        if init_weights:
            self.apply(self._init_weights)

    def forward(
            self,
            inputs: tensor,
            lengths: np.ndarray[int],
            hiddens: tuple[tensor, tensor] | None = None,
            split_idx: int | None = 0,
        ):
        # embedding dropout
        embedding = self.embedding(inputs)
        # in locked dropout
        embedding = self.in_locked_dropout(embedding) if self.locked_dropout else embedding
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)

        if split_idx is not None and split_idx > 0:
            h_n, c_n = stack(hiddens[0], dim=0), stack(hiddens[1], dim=0)
            h_n = h_n[:, :inputs.shape[0], :].contiguous()
            c_n = c_n[:, :inputs.shape[0], :].contiguous()
            hiddens = (h_n, c_n)

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
                            nn.init.xavier_uniform_(
                                param[idx*mul:(idx+1)*mul])
                    elif 'weight_hh' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            nn.init.xavier_uniform_(
                                param[idx*mul:(idx+1)*mul])
                    elif 'bias' in name:
                        param.data.fill_(0)
            else:
                if type(m) in [nn.Embedding, EmbeddingDropout]:
                    nn.init.uniform_(m.weight, -0.1, 0.1)
                elif type(m) in [nn.Linear]:
                    if m.bias != None:
                        m.bias.data.fill_(0.1)
