"""Caption preprocessing.

The raw ``description`` field is ONE string holding ~10 captions separated by
newlines. ``split_description`` turns it into a clean list of captions:

1. split on newlines
2. normalise unicode / whitespace, lowercase
3. drop empty lines and exact duplicates (keeping order)
4. normalise punctuation (keep only letters, digits, basic punctuation)

``tokenize`` produces word tokens used for statistics now and for the
captioning vocabulary in Milestone 2.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

_WS = re.compile(r"\s+")
_ALLOWED = re.compile(r"[^a-z0-9\s,.'\-]")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.])")
_NO_SPACE_AFTER_COMMA = re.compile(r",(?=[a-z])")


def clean_caption(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = _ALLOWED.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _NO_SPACE_AFTER_COMMA.sub(", ", text)
    text = _WS.sub(" ", text).strip()
    text = text.strip(" ,")
    return text


def split_description(description: str | None) -> list[str]:
    if not description:
        return []
    seen, out = set(), []
    for line in description.split("\n"):
        cap = clean_caption(line)
        if cap and cap not in seen:
            seen.add(cap)
            out.append(cap)
    return out


def raw_caption_lines(description: str | None) -> list[str]:
    """Lines before any cleaning (used by the EDA to quantify what cleaning changes)."""
    if not description:
        return []
    return description.split("\n")


def tokenize(caption: str) -> list[str]:
    return _TOKEN.findall(caption.lower())


def word_counts(captions: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for cap in captions:
        counter.update(tokenize(cap))
    return counter


class Vocabulary:
    """Word <-> index mapping with special tokens (used by the captioning task)."""

    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.itos: list[str] = [self.PAD, self.BOS, self.EOS, self.UNK]
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    def build(self, captions: Iterable[str]) -> "Vocabulary":
        counts = word_counts(captions)
        for word, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if c >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
        return self

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, caption: str, add_special: bool = True) -> list[int]:
        unk = self.stoi[self.UNK]
        ids = [self.stoi.get(t, unk) for t in tokenize(caption)]
        if add_special:
            ids = [self.stoi[self.BOS]] + ids + [self.stoi[self.EOS]]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        words = []
        for i in ids:
            w = self.itos[int(i)]
            if w == self.EOS:
                break
            if w in (self.PAD, self.BOS):
                continue
            words.append(w)
        return " ".join(words)
