"""단어들을 일관된 인터페이스로 읽어들이기 위한 모듈입니다."""
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional


class Vocab:
    def __init__(self, vocab_path: str, pad_token: str = "[PAD]", unk_token: str = "[UNK]"):
        """
        Vocab Constructor

        :param vocab_path: 로드할 vocab path
        :param pad_token: 사용할 PAD 토큰
        :param unk_token: 사용할 UNK 토큰
        """
        self.__vocab: Dict[str, int] = self._load_vocab_file(vocab_path)
        self.__inv_vocab: Dict[int, str] = {v: k for k, v in self.__vocab.items()}

        self.pad_token = pad_token
        self.pad_token_id = self.__vocab[self.pad_token]
        self.unk_token = unk_token
        self.unk_token_id = self.__vocab[self.unk_token]

    def __contains__(self, key: str) -> bool:
        return key in self.__vocab

    def __len__(self) -> int:
        return len(self.__vocab.keys())

    def get_vocab(self) -> List[str]:
        return list(self.__vocab.keys())

    def convert_token_to_id(self, token: str, default: Optional[int] = None) -> int:
        """
        토큰 하나를 인덱스로 바꾸는 함수

        :param token: 바꿀 토큰
        :param default: 만약 해당 토큰이 vocab에 없다면 반환할 default 값
        :raise KeyError: ``default`` 없이 unk 토큰이 넘어왔을 때
        """
        if default:
            return self.__vocab.get(token, default)
        return self.__vocab[token]

    def convert_id_to_token(self, idx: int, default: Optional[str] = None) -> str:
        """
        인덱스 하나를 토큰으로 바꾸는 함수

        :param idx: 바꿀 인덱스
        :param default: 만약 해당 인덱스가 vocab에 없다면 반환할 default 값
        :raise KeyError: ``default`` 없이 알 수 없는 인덱스가 넘어왔을 때
        """
        if default:
            return self.__inv_vocab.get(idx, default)
        return self.__inv_vocab[idx]

    def convert_tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        """
        토큰 여러개를 인덱스들로 바꾸는 함수

        :param tokens: 바꿀 토큰들
        """
        return [self.__vocab.get(item, self.unk_token_id) for item in tokens]

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        """
        인덱스 여러개를 토큰들로 바꾸는 함수

        :param idx: 바꿀 인덱스들
        """
        return [self.__inv_vocab[item] for item in ids]

    def dump(self, vocab_path: str):
        with open(vocab_path, "w") as f:
            f.write("\n".join(self.__vocab.keys()))

    @staticmethod
    def _load_vocab_file(vocab_path: str) -> Dict[str, int]:
        vocab: Dict[str, int] = OrderedDict()
        with open(vocab_path, "r") as f:
            for index, token in enumerate(f):
                token = token.strip().split("\t")[0]

                if token in vocab:
                    raise ValueError(f"Vocab에 중복된 토큰 {token}이 있습니다.")

                vocab[token] = index

        return vocab
