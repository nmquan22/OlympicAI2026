"""Word-boundary BPE for learning. Train on the training split only."""
import json
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


def words(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def merge(symbols, pair):
    result = []
    pos = 0
    while pos < len(symbols):
        if tuple(symbols[pos:pos + 2]) == pair:
            result.append("".join(pair))
            pos += 2
        else:
            result.append(symbols[pos])
            pos += 1
    return tuple(result)


@dataclass
class BPE:
    vocab: dict
    merges: list

    @classmethod
    def train(cls, texts, num_merges=200):
        counts = Counter(tuple(word) + ("</w>",) for text in texts for word in words(text))
        symbols = sorted({token for word in counts for token in word})
        tokens = ["<pad>", "<unk>", "<bos>", "<eos>", *symbols]
        rules = []
        for _ in range(num_merges):
            pairs = Counter()
            for word, count in counts.items():
                for pair in pairwise(word):
                    pairs[pair] += count
            if not pairs:
                break
            best = min(pairs, key=lambda pair: (-pairs[pair], pair))
            updated = Counter()
            for word, count in counts.items():
                updated[merge(word, best)] += count
            counts = updated
            rules.append(best)
            tokens.append("".join(best))
        return cls({token: i for i, token in enumerate(dict.fromkeys(tokens))}, rules)

    def encode(self, text, max_length=128):
        if max_length < 2:
            raise ValueError("max_length must accommodate BOS and EOS")
        ids = []
        for word in words(text):
            pieces = tuple(word) + ("</w>",)
            for rule in self.merges:
                pieces = merge(pieces, rule)
            ids.extend(self.vocab.get(piece, 1) for piece in pieces)
        return [2, *ids[:max_length - 2], 3]

    def save(self, path):
        Path(path).write_text(json.dumps(self.__dict__, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["vocab"], [tuple(pair) for pair in data["merges"]])
