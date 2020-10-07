# An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks

> **An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks** [[pdf](https://arxiv.org/abs/2010.02534)]<br>
> [Kyubyong Park](https://github.com/Kyubyong)\*, [Joohong Lee](https://github.com/roomylee)\*, [Seongbo Jang](https://github.com/sb-jang)\*, [Dawoon Jung](https://github.com/noowad93)\*<br>
> Accepted to AACL-IJCNLP 2020. (*indicates equal contribution)

> **Abstract**: Typically, tokenization is the very first step in most text processing works.
> As a token serves as an atomic unit that embeds the contextual information of text, how to define a token plays a decisive role in the performance of a model.<br>
> Even though Byte Pair Encoding (BPE) has been considered the*de facto* standard tokenization method due to its simplicity and universality, it still remains unclear whether BPE works best across all languages and tasks.
In this paper, we test several tokenization strategies in order to answer our primary research question, that is, "What is the best tokenization strategy for Korean NLP tasks?"<br>
> Experimental results demonstrate that a hybrid approach of morphological segmentation followed by BPE works best in Korean to/from English machine translation and natural language understanding tasks such as KorNLI, KorSTS, NSMC, and PAWS-X. As an exception, for KorQuAD, the Korean extension of SQuAD, BPE segmentation turns out to be the most effective.

## Installation

```bash
pip install -r requirements.txt
```

## Tokenization Strategies

There are 6 tokenization strategies for Korean. See [here](scripts/README.md) to prepare and use each strategy.

1. Consonant and Vowel
2. Syllable
3. Morpheme
4. Subword
5. Morpheme-aware Subword
6. Word

The corpus used for building vocabulary and training BPE models is as follows, which was extracted and refined via [attardi/wikiextractor](https://github.com/attardi/wikiextractor).

- Korean Wikipedia: <https://dumps.wikimedia.org/kowiki>
- English Wikipedia: <https://dumps.wikimedia.org/enwiki>

## Korean from/to English Translation

| Tokenization           | Vocab Size | ko-en (Dev) | ko-en (Test) | en-ko (Dev) | en-ko (Test) | OOV Rate | Avg. Length |
| ---------------------- | ---------- | ----------- | ------------ | ----------- | ------------ | -------- | ----------- |
| CV                     | 166        | 39.11       | 38.56        | 36.52       | 36.45        | 0.02     | 142.75      |
| Syllable               | 2K         | 39.3        | 38.75        | 38.64       | 38.45        | 0.06     | 69.20       |
| Morpheme               | 8K         | 31.59       | 31.24        | 32.44       | 32.19        | 7.51     | 49.19       |
|                        | 16K        | 34.38       | 33.8         | 35.74       | 35.52        | 4.67     | 49.19       |
|                        | 32K        | 36.19       | 35.74        | 36.51       | 36.12        | 2.72     | 49.19       |
|                        | 64K        | *37.88*     | *37.37*      | *37.51*     | *37.03*      | 1.4      | 49.19       |
| Subword                | 4K         | 39.18       | 38.75        | *38.31*     | *38.18*      | 0.07     | 48.02       |
|                        | 8K         | 39.16       | 38.75        | 38.09       | 37.94        | 0.08     | 38.44       |
|                        | 16K        | *39.22*     | *38.77*      | 37.64       | 37.34        | 0.1      | 33.69       |
|                        | 32K        | 39.05       | 38.69        | 37.11       | 36.98        | 0.11     | 30.21       |
|                        | 64K        | 37.02       | 36.46        | 35.77       | 35.64        | 0.12     | 27.50       |
| Morpheme-aware Subword | 4K         | 39.41       | 38.95        | 39.29       | 39.13        | 0.06     | 65.17       |
|                        | 8K         | 39.42       | 39.06        | 39.78       | 39.61        | 0.06     | 56.79       |
|                        | 16K        | 39.84       | 39.41        | 40.23       | 40.04        | 0.07     | 53.30       |
|                        | 32K        | **41.00**   | **40.34**    | **40.43**   | **40.41**    | 0.07     | 51.38       |
|                        | 64K        | 39.62       | 39.34        | 38.63       | 38.42        | 0.07     | 50.27       |
| Word                   | 64K        | 7.04        | 7.07         | 18.68       | 18.42        | 26.2     | 18.96       |

### Dataset

Recently, [Korean-English parallel corpus](https://www.aihub.or.kr/aidata/87) was publicly released by [AI Hub](https://www.aihub.or.kr/), which was gathered from various sources such as news, government web sites, legal documents, etc. We downloaded the news data which amount to 800K sentence pairs, and randomly split them into 784K (train), 8K (dev), and 8K (test).

### Training & Evaluation

We ran all the experiments using [pytorch/fairseq](https://github.com/pytorch/fairseq) (Ott et al., 2019), a PyTorch based deep learning library for sequence to sequence models.

#### 1. Preprocess

```bash
fairseq-preprocess \
--source-lang ko \
--target-lang en \
--trainpref ./dataset/translation/mecab_sp-8k/train \
--validpref ./dataset/translation/mecab_sp-8k/dev \
--testpref ./dataset/translation/mecab_sp-8k/test \
--destdir ./dataset/translation/mecab_sp-8k/preprocessed/ko-en \
--srcdict ./resources/en_sp-32k/fairseq.vocab \
--tgtdict ./resources/mecab_sp-8k/fairseq.vocab
```

#### 2. Training

We used Transformer (Vaswani et al., 2017), the state-of-the-art model for neural machine translation. We mostly followed the base model configuration: 6 blocks of 512-2048 units with 8 attention heads.

```bash
fairseq-train ./dataset/translation/mecab_sp-8k/preprocessed/ko-en \
--arch transformer \
--share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--dropout 0.3 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-epoch 50 \
--batch-size 128 \
--save-dir translation_ckpt/mecab_sp-8k/ko-en \
--disable-validation
```

#### 3. Evaluation

We report BLEU scores on both the dev and the test sets using [Moses](http://www.statmt.org/moses/) [multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) script. Following WAT 2019 (Nakazawa et al., 2019), Moses tokenizer and MeCab-ko are used for tokenizing the evaluation data.

```bash
fairseq-generate ./dataset/translation/mecab_sp-8k/preprocessed \
--path translation_ckpt/mecab_sp-8k/checkpoint_best.pt \
--batch-size 512 \
--beam 5 \
--remove-bpe sentencepiece
```

## Korean Natural Language Understanding

| Tokenization           | Vocab Size                                                                                                  | KorQuAD               | KorNLI    | KorSTS    | NSMC      | PAWS-X    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------- | --------- | --------- | --------- | --------- |
| CV                     | [166](https://www.dropbox.com/s/m840vmgwis9glzq/Dawoon%20Jung%20-%20jamo-200-pretrained-bert.tar?dl=0)      | 59.66 / 73.91         | 70.6      | 71.2      | 77.22     | 71.47     | 87.97     | 87.89     | 58        | 55.20     |
| Syllable               | [2K](https://www.dropbox.com/s/560c8lehfijn0mr/Dawoon%20Jung%20-%20char-2k-pretrained-bert.tar?dl=0)        | 69.10 / 83.29         | 73.98     | 73.47     | 82.7      | 75.86     | 88.94     | 89.07     | 68.65     | 67.20     |
| Morpheme               | [32K](https://www.dropbox.com/s/215tb4ublea3orp/Dawoon%20Jung%20-%20mecab-32k-pretrained-bert.tar?dl=0)     | 68.05 / 83.82         | 74.86     | 74.37     | 82.37     | 76.83     | 87.87     | 88.04     | 69.3      | 67.20     |
|                        | [64K](https://www.dropbox.com/s/81w8kw44bi7vxh9/Dawoon%20Jung%20-%20mecab-64k-pretrained-bert.tar?dl=0)     | *70.68* / *85.25*     | *75.06*   | *75.69*   | *83.21*   | *77.38*   | *88.72*   | *88.88*   | *73.40*   | *68.65*   |
| Subword                | [4K](https://www.dropbox.com/s/qm44kklrcy511fb/Dawoon%20Jung%20-%20bpe-4k-pretrained-bert.tar?dl=0)         | 71.48 / 83.11         | 74.38     | 74.03     | 83.37     | 76.8      | 89.08     | 89.3      | 72        | 69.60     |
|                        | [8K](https://www.dropbox.com/s/yhju79ovwdliwqx/Dawoon%20Jung%20-%20bpe-8k-pretrained-bert.tar?dl=0)         | 72.91 / 85.11         | 74.18     | 74.65     | 83.23     | 76.42     | 89.08     | 89.19     | 73.45     | 69.00     |
|                        | [16K](https://www.dropbox.com/s/llgynno8lavnvpn/Dawoon%20Jung%20-%20bpe-16k-pretrained-bert.tar?dl=0)       | 73.42 / 85.75         | 74.46     | *75.15*   | 83.3      | 76.41     | 88.89     | 88.88     | 73.4      | 70.70     |
|                        | [32K](https://www.dropbox.com/s/6n1dp2dhjneb5hd/Dawoon%20Jung%20-%20bpe-32k-pretrained-bert.tar?dl=0)       | **74.04** / 86.30     | *74.74*   | 74.29     | 83.02     | 77.01     | *89.39*   | *89.38*   | 74.05     | 70.95     |
|                        | [64K](https://www.dropbox.com/s/epx8g2r4d27zx4f/Dawoon%20Jung%20-%20bpe-64k-pretrained-bert.tar?dl=0)       | **74.04** / **86.66** | 73.73     | 74.55     | *83.52*   | *77.47*   | 88.8      | 89.19     | *75.85*   | *72.10*   |
| Morpheme-aware Subword | [4K](https://www.dropbox.com/s/f5twq5hoca56cz6/Dawoon%20Jung%20-%20mecab_bpe-4k-pretrained-bert.tar?dl=0)   | 67.53 / 81.93         | 73.53     | 73.45     | 83.34     | 76.03     | 88.93     | 89.32     | 69.75     | 67.45     |
|                        | [8K](https://www.dropbox.com/s/0zsac31zrdtveuv/Dawoon%20Jung%20-%20mecab_bpe-8k-pretrained-bert.tar?dl=0)   | 70.90 / 84.57         | 74.14     | 73.95     | 83.71     | 76.07     | 89.37     | 89.29     | 73.4      | 71.30     |
|                        | [16K](https://www.dropbox.com/s/nrg1z0nhe4ua0tr/Dawoon%20Jung%20-%20mecab_bpe-16k-pretrained-bert.tar?dl=0) | 69.47 / 83.36         | 75.02     | 74.99     | 83.22     | 76.59     | 89.33     | 89.41     | 75.05     | 71.70     |
|                        | [32K](https://www.dropbox.com/s/mczbb3kf7fzt9l3/Dawoon%20Jung%20-%20mecab_bpe-32k-pretrained-bert.tar?dl=0) | *72.65* / *86.35*     | 74.1      | 75.13     | 83.65     | **78.11** | 89.53     | 89.65     | 74.6      | 71.60     |
|                        | [64K](https://www.dropbox.com/s/jlcuccreq1rb6jm/Dawoon%20Jung%20-%20mecab_bpe-64k-pretrained-bert.tar?dl=0) | 69.48 / 83.73         | **76.39** | **76.61** | **84.29** | 76.78     | **89.82** | **89.66** | **76.15** | **74.00** |
| Word                   | [64K](https://www.dropbox.com/s/0ovofnol5j1g5ha/Dawoon%20Jung%20-%20word-64k-pretrained-bert.tar?dl=0)      | 1.54 / 8.86           | 64.06     | 65.83     | 69        | 60.41     | 70.1      | 70.58     | 58.25     | 55.30     |

### Pre-training

For  each  tokenization  strategy,  pre-training of BERT-Base model (Devlin et al., 2019)  was  performed with a Cloud TPU v3-8 for 1M steps using the official code of [google-research/bert](https://github.com/google-research/bert).

We set the training hyper-parameters of all models as follows:  `batch_size=1024`, `max_sequence_length=128`, `learning_rate=5e-5`, `warm_up_steps=10000`.

Because the Korean Wiki corpus (640 MB) is not enough in volume for the pre-training purpose, we additionally downloaded the recent [dump of Namuwiki](http://dump.thewiki.kr/) (5.5 GB) and extracted plain texts  using  [Namu  Wiki  Extractor](https://github.com/jonghwanhyeon/namu-wiki-extractor).

### Fine-tuning

After converting each pre-trained model in TensorFlow into PyTorch, we fine-tuned them using [huggingface/transformers](https://github.com/huggingface/transformers) (Wolf et al., 2019).

**example:**

```bash
python tasks/<TASK_NAME>/run_train.py --tokenizer <TOKENIZER_NAME>
```

## Citation

```plain
@article{park2020empirical,
  title={An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks},
  author={Park, Kyubyong and Lee, Joohong and Jang, Seongbo and Jung, Dawoon},
  journal={arXiv preprint arXiv:2010.02534},
  year={2020}
}
```
