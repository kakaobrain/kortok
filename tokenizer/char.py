from typing import List

from tokenizer.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        text = text.strip().replace(" ", "▁")
        return list(text)

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▁", " ").strip()
        return text
