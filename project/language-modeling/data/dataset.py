from torch import tensor
from torch.utils.data import Dataset

from .lang import Lang


class SentsDataset(Dataset):
    """
    Dataset class to serve each sentence as a data point. It takes a list 
    of sentences and a Lang object and prepares the data for 
    sequence-to-sequence tasks. Each sentence undergo:
    - tokenization 
    - mapping to integer IDs
    - appending an <eos> token at the end

    Args:
        sents (list[str]): A list of sentences to be used in the dataset.
        lang (Lang): The language object that provides token-to-ID mappings.

    Attributes:
        sents (list[list[int]]): A list of lists of integer sequences, where each sequence
            represents a sentence with <eos> token appended.
    
    Methods:
        __len__(): Get the number of sentences in the dataset.
        __getitem__(index: int): Get data point and serve it as:
                                - inp: the input sequence
                                - target: the target sequence
                                - length: the length of the sequence
    """
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
