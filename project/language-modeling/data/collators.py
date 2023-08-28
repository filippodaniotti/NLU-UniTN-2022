import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict

from torch import tensor, flip, split

def get_collator(
        pad_token: int = 0, 
        tbptt: bool = False,
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


        if tbptt:

            tbptt_lengths = []
            tbptt_inputs = defaultdict(list)
            tbptt_targets = defaultdict(list)

            sorted_idxs = np.argsort(lengths)[::-1]
            for word_idx in sorted_idxs:
                length, inp, tar = lengths[word_idx], inputs[word_idx], targets[word_idx]
                tbptt_lengths.append(length)

                split_inp = torch.split(inp, 20, dim=0)
                split_tar = torch.split(tar, 20, dim=0)
                
                for split_idx, (i, t) in enumerate(zip(split_inp, split_tar)):
                    tbptt_inputs[split_idx].append(i)
                    tbptt_targets[split_idx].append(t)

            lengths = tbptt_lengths
            inputs = [
                nn.utils.rnn.pad_sequence(inp, batch_first=True, padding_value=pad_token)
                for inp in tbptt_inputs.values()
            ]
            targets = [
                nn.utils.rnn.pad_sequence(tar, batch_first=True,
                            padding_value=pad_token)
                for tar in tbptt_targets.values()
            ]

        else:
            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first = True, padding_value=pad_token)
            targets = nn.utils.rnn.pad_sequence(targets, batch_first = True, padding_value=pad_token)

        return inputs, targets, np.asarray(lengths)

    return collate_fn