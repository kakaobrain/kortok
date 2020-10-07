import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig

from tasks.bert_utils import load_pretrained_bert
from tasks.korsts.config import TrainConfig
from tasks.korsts.data_utils import load_data
from tasks.korsts.dataset import KorSTSDataset
from tasks.korsts.model import KorSTSModel
from tasks.korsts.trainer import Trainer
from tasks.logger import get_logger
from tokenizer import (
    CharTokenizer,
    JamoTokenizer,
    MeCabSentencePieceTokenizer,
    MeCabTokenizer,
    SentencePieceTokenizer,
    Vocab,
    WordTokenizer,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # config
    config = TrainConfig(**args)
    config = config._replace(
        log_dir=config.log_dir.format(config.tokenizer),
        summary_dir=config.summary_dir.format(config.tokenizer),
        # checkpoint_dir=config.checkpoint_dir.format(config.tokenizer),
    )
    set_seed(config.seed)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.summary_dir, exist_ok=True)
    # os.makedirs(config.checkpoint_dir, exist_ok=True)

    # logger
    logger = get_logger(log_path=os.path.join(config.log_dir, "logs.txt"))
    logger.info(config)

    # 기본적인 모듈들 생성 (vocab, tokenizer)
    tokenizer_dir = os.path.join(config.resource_dir, config.tokenizer)
    logger.info(f"get vocab and tokenizer from {tokenizer_dir}")
    vocab = Vocab(os.path.join(tokenizer_dir, "tok.vocab"))
    if config.tokenizer.startswith("mecab-"):
        tokenizer = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))
    elif config.tokenizer.startswith("sp-"):
        tokenizer = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
    elif config.tokenizer.startswith("mecab_sp-"):
        mecab = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))
        sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
        tokenizer = MeCabSentencePieceTokenizer(mecab, sp)
    elif config.tokenizer.startswith("char-"):
        tokenizer = CharTokenizer()
    elif config.tokenizer.startswith("word-"):
        tokenizer = WordTokenizer()
    elif config.tokenizer.startswith("jamo-"):
        tokenizer = JamoTokenizer()
    else:
        raise ValueError("Wrong tokenizer name.")

    # 모델에 넣을 데이터 준비
    # Train
    logger.info(f"read training data from {config.train_path}")
    train_sentence_as, train_sentence_bs, train_labels = load_data(config.train_path)
    # Dev
    logger.info(f"read dev data from {config.dev_path}")
    dev_sentence_as, dev_sentence_bs, dev_labels = load_data(config.dev_path)
    # Test
    logger.info(f"read test data from {config.test_path}")
    test_sentence_as, test_sentence_bs, test_labels = load_data(config.test_path)

    # 데이터로 dataloader 만들기
    # Train
    logger.info("create data loader using training data")
    train_dataset = KorSTSDataset(
        train_sentence_as, train_sentence_bs, train_labels, vocab, tokenizer, config.max_sequence_length
    )
    train_random_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, sampler=train_random_sampler, batch_size=config.batch_size)
    # Dev
    logger.info("create data loader using dev data")
    dev_dataset = KorSTSDataset(
        dev_sentence_as, dev_sentence_bs, dev_labels, vocab, tokenizer, config.max_sequence_length
    )
    dev_data_loader = DataLoader(dev_dataset, batch_size=1024)
    # Test
    logger.info("create data loader using test data")
    test_dataset = KorSTSDataset(
        test_sentence_as, test_sentence_bs, test_labels, vocab, tokenizer, config.max_sequence_length
    )
    test_data_loader = DataLoader(test_dataset, batch_size=1024)

    # Summary Writer 준비
    summary_writer = SummaryWriter(log_dir=config.summary_dir)

    # 모델을 준비하는 코드
    logger.info("initialize model and convert bert pretrained weight")
    bert_config = BertConfig.from_json_file(
        os.path.join(config.resource_dir, config.tokenizer, config.bert_config_file_name)
    )
    model = KorSTSModel(bert_config, config.dropout_prob)
    model.bert = load_pretrained_bert(
        bert_config, os.path.join(config.resource_dir, config.tokenizer, config.pretrained_bert_file_name)
    )

    trainer = Trainer(config, model, train_data_loader, dev_data_loader, test_data_loader, logger, summary_writer)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tokenizer", type=str)

    parser.add_argument("--resource_dir", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v}

    main(args)
