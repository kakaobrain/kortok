from typing import List

from mosestokenizer import MosesTokenizer

from tokenizer.base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    def __init__(self):
        self.tokenizer = MosesTokenizer()

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer(text.strip())

    def detokenize(self, tokens: List[str]) -> str:
        text = " ".join(tokens).strip()
        return text

    def close(self):
        self.tokenizer.close()
