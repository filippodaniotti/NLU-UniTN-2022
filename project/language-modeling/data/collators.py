from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from typing import List, Tuple, Callable

def get_collator(pad_token: int = 0) -> Callable:
    def collate_fn(data: List[Tuple[Tensor, Tensor, int]]):
        inputs = []
        targets = []
        lengths = []

        for inp, tar, leng in data:
            inputs.append(inp)
            targets.append(tar)
            lengths.append(leng)

        inputs = pad_sequence(inputs, batch_first = True, padding_value=pad_token)
        targets = pad_sequence(targets, batch_first = True, padding_value=pad_token)

        return inputs, targets, lengths

    return collate_fn