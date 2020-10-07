# How to build vocabulary

## Consonant and Vowel ([JamoTokenizer](../tokenizer/jamo.py))

**example:**

```bash
$ python scripts/build_jamo_vocab.py --vocab=200
>> start tokenization ...
>> complete tokenization for all files. (elapsed time: 00:03:55)
>> oov: 0/539283497 (0.00%)
>> done.
```

## Syllable ([CharTokenizer](../tokenizer/char.py))

**example:**

```bash
$ python scripts/build_char_vocab.py --vocab=2000
>> start tokenization ...
>> complete tokenization for all files. (elapsed time: 00:01:53)
>> oov: 950873/279458799 (0.34%)
>> done.
```

## Morpheme ([MeCabTokenizer](../tokenizer/mecab.py))

**example:**

```bash
$ python scripts/build_mecab.py --vocab_size=8000
>> start tokenization ...
>> complete tokenization for all files. (elapsed time: 00:01:35)
>> oov: 20227115/135323506 (14.95%)
>> done.
```

**result:**

```bash
$ head ./resources/mecab-8k/fairseq.vocab
>> ▃ 58262281
>> . 4573370
>> 의 3808904
>> 다 3594077
>> 이 3502365
>> 는 3441298
>> , 3201410
>> 에 2883200
>> 을 2693685
>> 하 2452804
```

## Subword ([SentencePieceTokenizer](../tokenizer/sentencepiece.py))

**example:**

```bash
$ python scripts/train_sentencepiece.py --vocab_size=8000
>> sentencepiece_trainer.cc(116) LOG(INFO) Running command: --input=./dataset/wiki/kowiki-200420.txt --model_prefix=./resources/sp-8k/tok --vocab_size=8000 --model_type=bpe --character_coverage=1.0 --normalization_rule_name=identity --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] --unk_surface=[UNK] --user_defined_symbols=[CLS],[SEP],[MASK]
>> sentencepiece_trainer.cc(49) LOG(INFO) Starts training with :
>> ...
```

**result:**

```bash
$ head ./resources/sp-8k/fairseq.vocab
>> ▁1 -0
>> ▁이 -1
>> 으로 -2
>> 에서 -3
>> ▁있 -4
>> ▁2 -5
>> ▁그 -6
>> ▁대 -7
>> ▁사 -8
>> 이다 -9
```

### English Subword for translation task

**example:**

```bash
python scripts/train_sentencepiece.py --vocab_size=32000 --tokenizer_type="en"
```

## Morpheme-aware Subword ([MeCabSentencePieceTokenizer](../tokenizer/mecab_sp.py))

#### 1) Create MeCab-tokenized corpus

```bash
python scripts/mecab_tokenization.py
```

#### 2) Train BPE on MeCab-tokenized corpus

**example:**

```bash
python scripts/train_sentencepiece.py --vocab_size=8000 --tokenizer_type="mecab_tokenized"
>> sentencepiece_trainer.cc(116) LOG(INFO) Running command: --input=./dataset/wiki/mecab_tokenized/mecab/kowiki-200420.txt --model_prefix=./resources/mecab_sp-8k/tok --vocab_size=8000 --model_type=bpe --character_coverage=1.0 --normalization_rule_name=identity --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] --unk_surface=[UNK] --user_defined_symbols=[CLS],[SEP],[MASK]
>> sentencepiece_trainer.cc(49) LOG(INFO) Starts training with :
>> ...
```

**result:**

```bash
$ head ./resources/mecab_sp-8k/fairseq.vocab
>> ▁▃ -0
>> ▁이 -1
>> ▁. -2
>> ▁에 -3
>> ▁다 -4
>> ▁의 -5
>> ▁는 -6
>> ▁, -7
>> ▁하 -8
>> ▁을 -9
```

## Word ([WordTokenizer](../tokenizer/word.py))

**example:**

```bash
python scripts/build_word_vocab.py --vocab=64000
>> start tokenization ...
>> complete tokenization for all files. (elapsed time: 00:00:52)
>> oov: 19946533/60729995 (32.84%)
>> done.
```
