from typing import List

import sentencepiece as spm

from tokenizer.base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_path: str, reverse: bool = False):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.reverse = reverse

    def tokenize(self, text: str) -> List[str]:
        if self.reverse:
            tokenized = self.sp.EncodeAsPieces(text[::-1].strip())
            tokenized = [s[::-1] for s in tokenized][::-1]
        else:
            tokenized = self.sp.EncodeAsPieces(text.strip())

        return tokenized

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▁", " ").strip()
        return text
