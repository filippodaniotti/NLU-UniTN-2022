import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _weight_drop(module, weights, dropout):
    """
    Apply DropConnect to the specified weights of a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module to which weight dropout is applied.
        weights (list): A list of strings representing the names of the module's
            weight parameters to which dropout should be applied.
        dropout (float): The probability of dropping out each weight during training.

    Note:
        This function modifies the module in-place by adding '_raw' versions of
        the specified weight parameters and applying weight dropout during forward
        passes.

    Example:
        ```python
        # Usage example:
        lstm = nn.LSTM(input_size=10, hidden_size=20)
        weights_to_drop = ['weight_ih_l0', 'weight_hh_l0']
        _weight_drop(lstm, weights_to_drop, dropout=0.5)
        ```
    """
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', nn.Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            # FIX: convert weights back to nn.Paramter before assignment
            w = nn.Parameter(nn.functional.dropout(
                raw_w, p=dropout, training=module.training))
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)

class WeightDropLSTM(nn.Module):
    """
    Applies DropConnect to specified weights of an input module.

    Args:
        module (nn.Module): The input module to which weight dropout is applied.
        weights (list): A list of strings representing the names of the module's
            weight parameters to which dropout should be applied.
        dropout (float, optional): The probability of dropping out each weight during training.
            Default is 0.0 (no dropout).

    Note:
        This module wraps the input module, applying weight dropout to the specified
        weights as specified by the `_weight_drop` function.

    Example:
        ```python
        # Usage example:
        lstm = nn.LSTM(input_size=10, hidden_size=20)
        weights_to_drop = ['weight_ih_l0', 'weight_hh_l0']
        weight_dropped_lstm = WeightDropLSTM(lstm, weights_to_drop, dropout=0.5)
        ```
    """
    def __init__(self, module, weights, dropout=0.0):
        super().__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward

class LockedDropout(nn.Module):
    """
    https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    When using standard dropout on a RNN model, a new mask is generated 
    for each time step. Variational Dropout (LockedDropout) generates a 
    single mask and uses it repeatedly for all elements of the sequence 
    within a forward step, ensuring temporal consistency across time steps.

    Args:
        p (float): The probability of an element to be zeroed. Defaults to 0.5.
    
    Attributes:
        dropout (float): The probability of an element to be zeroed.
    
    Methods:
        forward(x: torch.Tensor): Performs forward propagation.

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
    Class for dropping out embeddings by zero'ing out parameters in the
    embedding matrix.

    This is equivalent to dropping out particular words, e.g., in the sentence
    'the quick brown fox jumps over the lazy dog', dropping out 'the' would
    lead to the sentence '### quick brown fox jumps over ### lazy dog' (in the
    embedding vector space).

    See 'A Theoretically Grounded Application of Dropout in Recurrent Neural
    Networks', (Gal and Ghahramani, 2016).

    Args:
        num_embeddings (int): The number of unique embeddings in the vocabulary.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (int, optional): If given, pads the output with zeros at
            the specified index. Default is 0.
        max_norm (float, optional): If given, the embeddings are normalized to
            have a maximum L2 norm of `max_norm`. Default is None.
        norm_type (float, optional): The type of norm to apply if `max_norm` is
            given (e.g., 2 for L2 norm). Default is 2.
        scale_grad_by_freq (bool, optional): If True, scales the gradients by
            the frequency of the embeddings during backward pass. Default is False.
        sparse (bool, optional): If True, uses a sparse embedding. Default is False.
        dropout (float, optional): The dropout probability. Should be a value
            between 0.0 and 1.0. Dropout is applied only during training. Default is 0.1.
        scale (float, optional): An optional scaling factor to apply to the
            embedding weights. Default is None.

    Attributes:
        dropout (float): The dropout probability.
        scale (float): The scaling factor to apply to the embedding weights.

    Methods:
        forward(inputs):
            Embeds the input indices with the dropout applied to the embedding
            weight matrix.
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