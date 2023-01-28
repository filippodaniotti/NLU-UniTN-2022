import torch.nn as nn
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
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x