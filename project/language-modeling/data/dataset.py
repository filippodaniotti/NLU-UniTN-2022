from torch import tensor
from torch.utils.data import Dataset


class SentsDataset(Dataset):
    def __init__(self, sents: list[str], w2i: dict[str, int]) -> None:
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        lengths: list[int] = []

        for sent in sents:
            words = sent.split() + ['<eos>']
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
