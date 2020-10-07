# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert)."""

import argparse
import glob
import json
import logging
import os
import random
import timeit

import numpy as np
import torch
from attrdict import AttrDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup, squad_convert_examples_to_features
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

from tasks.bert_utils import load_pretrained_bert
from tasks.korquad.evaluate_v1_0 import eval_during_train
from tasks.korquad.model import KorQuADModel
from tasks.korquad.tokenization import BertTokenizer
from tokenizer import (
    CharTokenizer,
    JamoTokenizer,
    MeCabSentencePieceTokenizer,
    MeCabTokenizer,
    SentencePieceTokenizer,
    Vocab,
    WordTokenizer,
)

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Train batch size per GPU = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 1
    steps_trained_in_current_epoch = 0

    logger.info("  Starting fine-tuning.")

    tr_loss = 0.0
    model.zero_grad()
    # Added here for reproductibility
    set_seed(args.seed)

    for epoch in tqdm(range(int(args.num_train_epochs)), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Batch")):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:
                        logger.info("***** Eval results *****")
                        results = evaluate(args, model, tokenizer, global_step=global_step)
                        for key in sorted(results.keys()):
                            logger.info(f"  {key} = {results[key]}")

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        logger.info(f"Epoch {epoch + 1} done")

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", global_step=None):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Eval"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info(f"  Evaluation done in total {evalTime} secs ({evalTime / len(dataset)} sec per example)")

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, f"predictions_{prefix}.json")
    output_nbest_file = os.path.join(args.output_dir, f"nbest_predictions_{prefix}.json")

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, f"null_odds_{prefix}.json")
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        False,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    # Write the result
    # Write the evaluation result on file
    output_dir = os.path.join(args.output_dir, "eval")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, f"eval_result_{global_step}.txt")

    logger.info("***** Official Eval results *****")
    with open(output_eval_file, "w", encoding="utf-8") as f:
        official_eval_results = eval_during_train(args)
        for key in sorted(official_eval_results.keys()):
            logger.info(f"  {key} = {official_eval_results[key]}")
            f.write(f" {key} = {official_eval_results[key]}\n")
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir, f"cached_{'dev' if evaluate else 'train'}_{args.tokenizer}_{args.max_seq_length}"
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        logger.info(f"Loading features from cached file {cached_features_file}")
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info(f"Creating features from dataset file at {input_dir}")

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        # For MeCab tokenizer, we remove '\n' in all texts in all examples
        for example in examples:
            example.question_text = example.question_text.replace("\n", "")
            example.context_text = example.context_text.replace("\n", "")
            if example.answer_text is not None:
                example.answer_text = example.answer_text.replace("\n", "")
            example.title = example.title.replace("\n", "")

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        logger.info(f"Saving features into cached file {cached_features_file}")
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def main(cli_args):
    # Read from config file and make args
    with open("./tasks/korquad/config.json", "r") as f:
        args = AttrDict(json.load(f))
    args.seed = cli_args.seed
    args.tokenizer = cli_args.tokenizer
    args.output_dir = args.output_dir.format(args.tokenizer)
    args.resource_dir = cli_args.resource_dir
    args.data_dir = cli_args.data_dir
    logger.info(f"Training/evaluation parameters {args}")

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    init_logger()
    set_seed(args.seed)

    logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)  # Reduce model loading logs

    # custom tokenizers
    tokenizer_dir = os.path.join(args.resource_dir, args.tokenizer)
    logger.info(f"get vocab and tokenizer from {tokenizer_dir}")
    if args.tokenizer.startswith("mecab-"):
        custom_tokenizer = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))
    elif args.tokenizer.startswith("sp-"):
        custom_tokenizer = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
    elif args.tokenizer.startswith("mecab_sp-"):
        mecab = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))
        sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
        custom_tokenizer = MeCabSentencePieceTokenizer(mecab, sp)
    elif args.tokenizer.startswith("char-"):
        custom_tokenizer = CharTokenizer()
    elif args.tokenizer.startswith("word-"):
        custom_tokenizer = WordTokenizer()
    elif args.tokenizer.startswith("jamo-"):
        custom_tokenizer = JamoTokenizer()
    else:
        raise ValueError("Wrong tokenizer name.")

    # Load pretrained model and tokenizer
    config = BertConfig.from_json_file(os.path.join(args.resource_dir, args.tokenizer, args.bert_config_file_name))
    tokenizer = BertTokenizer(os.path.join(tokenizer_dir, "tok.vocab"), custom_tokenizer)
    model = KorQuADModel(config)
    model.bert = load_pretrained_bert(
        config, os.path.join(args.resource_dir, args.tokenizer, args.pretrained_bert_file_name)
    )
    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(args.device)

    logger.info(f"Training/evaluation parameters {args}")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce model loading logs
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info(f"Evaluate the following checkpoints: {checkpoints}")

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1]
            model = KorQuADModel.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + (f"_{global_step}" if global_step else ""), v) for k, v in result.items())
            results.update(result)

        output_dir = os.path.join(args.output_dir, "eval")
        with open(os.path.join(output_dir, "eval_result.txt"), "w", encoding="utf-8") as f:
            official_eval_results = eval_during_train(args)
            for key in sorted(official_eval_results.keys()):
                logger.info(f"  {key} = {official_eval_results[key]}")
                f.write(f" {key} = {official_eval_results[key]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, required=True)

    parser.add_argument("--resource_dir", type=str, default="./resources")
    parser.add_argument("--data_dir", type=str, default="./dataset/nlu_tasks/korquad")

    cli_args = parser.parse_args()

    main(cli_args)
