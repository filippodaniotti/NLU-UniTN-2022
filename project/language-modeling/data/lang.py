from collections import Counter

class Lang():
    def __init__(
            self, 
            elements,
            pad_value: int = 0,
            eos_token: int = None, 
            cutoff: int = 0,
            parse_sents: bool = False):
        words = elements if not parse_sents else self.flatten_sentences(elements)            
        self.words2ids, self.ids2words = self.map_tokens(
            words, pad_value, eos_token, cutoff=cutoff)
        
    def __len__(self) -> int:
        return len(self.words2ids)
    
    def __getitem__(self, key: str | int) -> str | int:
        if isinstance(key, str):
            return self.words2ids[key]
        elif isinstance(key, int):
            return self.ids2words[key]
        else:
            raise TypeError("Key must be either str or int")
        
    def map_tokens(
            self, 
            words: list[str], 
            pad_value: int, 
            eos_token: int, 
            cutoff: int = None) -> tuple[dict[str, int], dict[int, str]]:
        """
        No need to tackle oov, there are none of them
        """
        w2id, id2w = {}, {}

        w2id['<pad>'] = pad_value
        id2w[w2id['<pad>']] = '<pad>'
        
        count = Counter(words)
        curr_len = 1
        for token, freq in count.items():
            if freq > cutoff:
                w2id[token] = curr_len
                id2w[curr_len] = token
                curr_len += 1

        if eos_token is None:
            w2id['<eos>'] = curr_len
            id2w[curr_len] = '<eos>'
        else:
            w2id['<eos>'] = eos_token
            w2id[eos_token] = "<eos>"

        return w2id, id2w

    def flatten_sentences(self, sents: list[str]) -> list[str]:
        words = []
        for sent in sents:
            words.extend(sent.split())

        return words
