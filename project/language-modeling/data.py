from torch import tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from typing import List, Dict, Tuple, Callable

def get_collator(pad_token: int = 0) -> Callable:
    def collate_fn(data: List[Tuple[tensor, tensor, int]]):
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

class SentsDataset(Dataset):
    def __init__(self, sents: List[str], w2i: Dict[str, int]) -> None:
        inputs = []
        targets = []
        lengths = []

        for sent in sents:
            words = sent.split()
            words += ['<eos>']
            words = [w2i[w] for w in words]
            inputs.append(words[:-1])
            targets.append(words[1:])
            lengths.append(len(words[1:]))

        self.inputs = inputs
        self.targets = targets
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        inp = self.inputs[index] 
        target = self.targets[index]
        length = self.lengths[index]
        return tensor(inp), tensor(target), length
