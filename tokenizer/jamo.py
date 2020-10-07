from typing import List
from unicodedata import normalize

from tokenizer.base import BaseTokenizer


class JamoTokenizer(BaseTokenizer):
    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        return list("▁".join([normalize("NFKD", token) for token in text.strip().split(" ")]))

    def detokenize(self, tokens: List[str]) -> str:
        return normalize("NFKC", "".join(tokens)).replace("▁", " ")
