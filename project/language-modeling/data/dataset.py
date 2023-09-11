from torch import tensor
from torch.utils.data import Dataset


class SentsDataset(Dataset):
    def __init__(self, sents: list[str], w2i: dict[str, int]) -> None:
        self.sents: list[list[int]] = []

        for sent in sents:
            words = sent.split() + ['<eos>']
            words = [w2i[w] for w in words]
            self.sents.append(words)

    def __len__(self) -> int:
        return len(self.sents)

    def __getitem__(self, index: int):
        inp = self.sents[index][:-1]
        target = self.sents[index][1:]
        length = len(inp)
        return tensor(inp), tensor(target), length
