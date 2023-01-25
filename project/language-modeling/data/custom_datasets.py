from torch import Tensor
from torch.utils.data import Dataset

from typing import List, Dict

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
        return Tensor(inp), Tensor(target), length
