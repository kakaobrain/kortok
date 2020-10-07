from abc import abstractmethod
from typing import List


class BaseTokenizer:
    """Tokenizer meta class"""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError("Tokenizer::tokenize() is not implemented")
