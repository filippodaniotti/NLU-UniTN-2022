import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import tensor
from typing import Any

class SequenceCollator:
    def __init__(
            self,
            pad_value: int = 0, 
            tbptt: bool = False,
            tbptt_config: dict[str, Any] | None = None,
            part_shuffle: bool = False,
        ) -> None:
        
        self.pad_value = pad_value
        self.tbptt = tbptt
        self.tbptt_config = tbptt_config
        self.part_shuffle = part_shuffle

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.collate_fn(*args, **kwargs)
    
    def collate_fn(self, data: list[tuple[tensor, tensor, int]]):
        inputs = []
        targets = []
        lengths = []

        for inp, tar, leng in data:

            inputs.append(inp)
            targets.append(tar)
            lengths.append(leng)

        if self.part_shuffle:
            inputs, targets = self.partial_shuffle(zip(inputs, targets), lengths)

        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first = True, padding_value=self.pad_value)
        targets = nn.utils.rnn.pad_sequence(targets, batch_first = True, padding_value=self.pad_value)

        if self.tbptt:
            inputs, targets = self.tbptt_split_batch(inputs, targets)

        return inputs, targets, np.asarray(lengths)

    def get_split_step(self):
        mu = self.tbptt_config["mu"]
        std = self.tbptt_config["std"]
        p = self.tbptt_config["p"]

        mu = mu if np.random.random() < p else mu/2
        split_step = max(10, int(np.random.normal(mu, std)))

        return split_step
    
    def tbptt_split_batch(self, inputs, targets):
        split_step = self.get_split_step()
        inputs = list(torch.split(inputs, split_step, dim=1))
        targets = list(torch.split(targets, split_step, dim=1))

        # pad last batch
        if inputs[-1].shape[1] < split_step:
            inputs[-1] = F.pad(inputs[-1], (0, split_step - inputs[-1].shape[1]))
            targets[-1] = F.pad(targets[-1], (0, split_step - inputs[-1].shape[1]))

        # remove empty sentences and batches
        get_length = lambda x: torch.sum(x.ne(self.pad_value)).item()
        for split_idx in range(len(inputs)):
            inputs[split_idx] = torch.stack([i for i in inputs[split_idx] if get_length(i) > 0])
            targets[split_idx] = torch.stack([t for t in targets[split_idx] if get_length(t) > 0])

        return inputs, targets
    
    def partial_shuffle(data, lengths):
        """
        Method from "Partially Shuffling the Training Data to Improve Language Models" by Ofir Press
        https://arxiv.org/abs/1903.04167
        Implementation adapted from https://github.com/ofirpress/PartialShuffle/blob/master/partial_shuffle.py
        """

        shifted_inputs = []
        shifted_targets = []
        for i, row in enumerate(data):
            inp, tar = row
            split = np.random.randint(lengths[i])
            shifted_inputs.append(
                torch.cat((inp[split:], inp[:split])) #partial shuffle of a single row
            )
            shifted_targets.append(
                torch.cat((tar[split:-1], torch.cat((inp[:split], tar[-1].unsqueeze(0))))) #ensure eos token is last item
            )
        return shifted_inputs, shifted_targets
