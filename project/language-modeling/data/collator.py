import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict

from torch import tensor, flip, split

from typing import Any

def get_collator(
        pad_token: int = 0, 
        p_reverse: float | None = None) -> callable:
    
    def collate_fn(data: list[tuple[tensor, tensor, int]]):
        inputs = []
        targets = []
        lengths = []

        if p_reverse is not None:
            p = [p_reverse, 1-p_reverse]
            reverse_map = list(np.random.choice([True, False], len(data), p=p))
        else:
            reverse_map = list(np.fromiter([False]*len(data), dtype=bool))
        
        for (inp, tar, leng), to_reverse in zip(data, reverse_map):
            
            inp = torch.flip(inp, dims=(0,)) if to_reverse else inp
            tar = torch.flip(tar, dims=(0,)) if to_reverse else tar

            inputs.append(inp)
            targets.append(tar)
            lengths.append(leng)
            
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first = True, padding_value=pad_token)
        targets = nn.utils.rnn.pad_sequence(targets, batch_first = True, padding_value=pad_token)

        return inputs, targets, np.asarray(lengths)

    return collate_fn