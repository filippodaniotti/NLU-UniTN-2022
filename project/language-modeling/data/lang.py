from collections import Counter

from typing import List, Dict, Tuple

class Lang():
    def __init__(
            self, 
            elements,
            pad_value: int = 0,
            eos_token: int = None, 
            cutoff: int = 0,
            parse_sents: bool = False):
        words = elements if not parse_sents else self.get_words_from_sents(elements)            
        self.words2ids, self.ids2words = self.map_tokens(
            words, pad_value, eos_token, cutoff=cutoff)
        
    def map_tokens(
            self, 
            words: List[str], 
            pad_value: int, 
            eos_token: int, 
            cutoff: int = None) -> Tuple[Dict[str, int], Dict[int, str]]:
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
        else:
            w2id['<eos>'] = eos_token

        return w2id, id2w

    def get_words_from_sents(self, sents: List[str]) -> List[str]:
        words = []
        for sent in sents:
            words.extend(sent.split())

        return words
