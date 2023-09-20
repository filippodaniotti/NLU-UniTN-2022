from torch import tensor
from torch.utils.data import Dataset

from .lang import Lang


class SentsDataset(Dataset):
    def __init__(self, sents: list[str], lang: Lang) -> None:
        self.sents: list[list[int]] = []

        for sent in sents:
            words = sent.split() + ['<eos>']
            words = [lang[w] for w in words]
            self.sents.append(words)

    def __len__(self) -> int:
        return len(self.sents)

    def __getitem__(self, index: int):
        inp = self.sents[index][:-1]
        target = self.sents[index][1:]
        length = len(inp)
        return tensor(inp), tensor(target), length
