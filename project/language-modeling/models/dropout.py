import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WeightDropLSTM(nn.LSTM):
    """
    Wrapper around :class:`torch.nn.LSTM` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        self._weight_drop(weights, weight_dropout)


    def _weight_drop(self, weights, dropout):
        """
        Helper for `WeightDrop`.
        """

        for name_w in weights:
            w = getattr(self, name_w)
            del self._parameters[name_w]
            self.register_parameter(name_w + '_raw', nn.Parameter(w))

        original_self_forward = self.forward

        def forward(*args, **kwargs):
            for name_w in weights:
                raw_w = getattr(self, name_w + '_raw')
                w = nn.functional.dropout(raw_w, p=dropout, training=self.training)
                w = nn.Parameter(w)
                setattr(self, name_w, w)

            return original_self_forward(*args, **kwargs)

        setattr(self, 'forward', forward)

class LockedDropout(nn.Module):
    """
    https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    """
    def __init__(self, p):
        super().__init__()
        self.dropout = p

    def forward(self, x):
        if not self.training:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        mask = mask.expand_as(x)
        return mask * x
    
class EmbeddingDropout(nn.Embedding):
    """
    https://github.com/carpedm20/ENAS-pytorch/blob/0468b8c4ddcf540c9ed6f80c27289792ff9118c9/models/shared_rnn.py#L51
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 padding_idx=0,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 dropout=0.1,
                 scale=None):
        nn.Embedding.__init__(self,
                                    num_embeddings=num_embeddings,
                                    embedding_dim=embedding_dim,
                                    padding_idx=padding_idx,
                                    max_norm=max_norm,
                                    norm_type=norm_type,
                                    scale_grad_by_freq=scale_grad_by_freq,
                                    sparse=sparse)
        self.dropout = dropout
        assert (dropout >= 0.0) and (dropout < 1.0), ('Dropout must be >= 0.0 '
                                                      'and < 1.0')
        self.scale = scale

    def forward(self, inputs):  # pylint:disable=arguments-differ
        """Embeds `inputs` with the dropped out embedding weight matrix."""
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0

        if dropout:
            mask = self.weight.data.new(self.weight.size(0), 1)
            mask.bernoulli_(1 - dropout)
            mask = mask.expand_as(self.weight)
            mask = mask / (1 - dropout)
            masked_weight = self.weight * Variable(mask)
        else:
            masked_weight = self.weight
        if self.scale and self.scale != 1:
            masked_weight = masked_weight * self.scale

        return F.embedding(inputs,
                           masked_weight,
                           max_norm=self.max_norm,
                           norm_type=self.norm_type,
                           scale_grad_by_freq=self.scale_grad_by_freq,
                           sparse=self.sparse)