from functools import partial
from typing import List, Optional, Tuple, TypeVar

from transformers import BertConfig, BertForPreTraining, BertModel

from tokenizer import BaseTokenizer, Vocab

T = TypeVar("T", int, float)


def load_pretrained_bert(config: BertConfig, model_path: str):
    if model_path.endswith(".index"):
        bert_model = BertForPreTraining.from_pretrained(model_path, config=config, from_tf=True).bert
    elif model_path.endswith(".pth"):
        bert_model = BertModel.from_pretrained(model_path, config=config)
    else:
        raise ValueError(f"Wrong model path ({model_path})")
    return bert_model


def convert_single_to_feature(
    sentence: str, tokenizer: BaseTokenizer, vocab: Vocab, max_length: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    sentence 하나를 feature로 바꾸어주는 함수입니다.
    :param sentence: 변환할 sentence.
    :param pipeline: pnlp의 NLPPipeline으로, 전처리 및 tokenize pipeline. (tokenize는 필수로 포함되어야 함.)
    :param vocab: token을 indexing하기 위한 Vocab ( ``pnlp.Vocab`` )
    :param max_sequence_length: 넘지 말아야 할 최대 길이
    :return: 변환된 feature. (``Tuple[List[int], List[int], List[int], List[int]]``)
    """
    example = tokenizer.tokenize(sentence)
    example = example[: max_length - 2]
    example = ["[CLS]"] + example + ["[SEP]"]

    token_ids = vocab.convert_tokens_to_ids(example)
    attention_mask = [1] * len(token_ids)
    token_type_ids = [0] * len(token_ids)

    return (token_ids, attention_mask, token_type_ids)


def convert_pair_to_feature(
    sentence_a: str, sentence_b: str, tokenizer: BaseTokenizer, vocab: Vocab, max_length: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    sentence_a와 sentence_b를 feature로 바꾸어주는 함수입니다.
    :param sentences: sentence_a와 sentence_b로 이루어진 ``Tuple``
    :param pipeline: pnlp의 NLPPipeline으로, 전처리 및 tokenize pipeline. (tokenize는 필수로 포함되어야 함.)
    :param vocab: token을 indexing하기 위한 Vocab ( ``pnlp.Vocab`` )
    :param max_length: 넘지 말아야 할 최대 길이
    :param special_token: 사용할 special token.
    :return: 변환된 feature. (``Tuple[List[int], List[float], List[int], List[int]]``)
    """

    example_a = tokenizer.tokenize(sentence_a)
    example_b = tokenizer.tokenize(sentence_b)

    truncate_pair_example(example_a, example_b, max_length - 3)
    example = ["[CLS]"] + example_a + ["[SEP]"] + example_b + ["[SEP]"]

    token_ids = vocab.convert_tokens_to_ids(example)
    attention_mask = [1] * len(token_ids)
    token_type_ids = [0] * (len(example_a) + 2) + [1] * (len(example_b) + 1)

    assert len(token_ids) == len(attention_mask)
    assert len(token_ids) == len(token_type_ids)

    return (token_ids, attention_mask, token_type_ids)


def truncate_pair_example(tokens_a: List[str], tokens_b: List[str], max_length: int):
    """최대 길이를 넘지 않도록 example을 자르는 함수
    :param tokens_a: sequence of tokens from sentence A
    :param tokens_b: sequence of tokens from sentence A
    :param max_length: 넘지 말아야 할 최대 길이
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def pad_sequences(sequences: List[List[T]], padding_value: T, max_length: Optional[int] = None) -> List[List[T]]:
    """sequence들의 list에 대해 padding을 진행합니다.
    * ``max_length`` 가 주어지지 않으면 sequence 내의 최대 길이를 기준으로 padding을 진행합니다.
    * ``max_length`` 보다 긴 length를 가진 sequence가 있으면 ``ValueError`` 를 raise합니다.
    :param sequences: padding을 진행할 sequence들의 list.
    :param padding_value: padding을 진행할 value.
    :param max_length: padding을 진행할 기준 길이.(이 길이에 맞춰서 padding)
    :return padded_sequences: padding을 한 결과 sequence들의 list.
    """
    max_length = max_length or max(map(len, sequences))
    padding_fn = partial(pad_sequence, padding_value=padding_value, max_length=max_length)
    return list(map(padding_fn, sequences))


def pad_sequence(sequence: List[T], padding_value: T, max_length: int) -> List[T]:
    """sequence들의 list에 대해 padding을 진행합니다.
    * ``max_length`` 보다 긴 length를 가진 sequence가 있으면 ``ValueError`` 를 raise합니다.
    :param sequence: padding을 진행할 sequence.
    :param padding_value: padding을 진행할 value.
    :param max_length: padding을 진행할 기준 길이.(이 길이에 맞춰서 padding)
    :return padded_sequence: padding을 한 결과 sequence.
    """
    if len(sequence) > max_length:
        raise ValueError(f"Max Length {max_length} is smaller than Sequence length {len(sequence)}")
    return sequence + [padding_value] * (max_length - len(sequence))
