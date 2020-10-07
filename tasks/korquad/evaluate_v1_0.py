from __future__ import print_function

import argparse
import json
import os
import re
import string
import sys
from collections import Counter

from konlpy.tag import Mecab

"""KorQuAD v1.0에 대한 공식 평가 스크립트 """
"""본 스크립트는 SQuAD v1.1 평가 스크립트 https://rajpurkar.github.io/SQuAD-explorer/ 를 바탕으로 작성됨."""


def postprocess(text):
    """post-processing"""
    # strip punctuations
    text = text.strip("""!"\#$&'()*+,\-./:;<=>?@\^_‘{|}~《》""")

    # pair parentheses
    if text.count("(") == text.count(")") + 1:
        text += ")"
    elif text.count("(") + 1 == text.count(")"):
        text = "(" + text

    # strip unanalyzed postpositions
    if len(text) > 3 and text[-3:] in ("부터는", "이라는", "에서는", "에게는"):
        text = text[:-3]
        return text
    elif len(text) > 2 and text[-2:] in ("으로", "과의", "에는", "와의", "하는", "되는", "이다"):
        text = text[:-2]
        return text
    elif len(text) > 1 and text[-1] in "의이가들로간에과와은는도만씩을를":
        text = text[:-1]
        return text

    m = Mecab()
    # we want to extract noun phrase of the last eojeol only.
    noun_phrase = " ".join(text.rsplit(" ", 1)[:-1])
    eojeols = m.pos(text, flatten=False)
    if eojeols:
        last_eojeol = eojeols[-1]

        for i, token in enumerate(last_eojeol[::-1]):
            _, pos = token
            if pos[0] in ("N", "S"):  # N: Nouns, S: 외국어, 한자, 숫자 etc.
                break
            elif pos.startswith("XSN"):
                break
        i = len(last_eojeol) - i
        last_eojeol = "".join(morph for morph, _ in last_eojeol[:i])

        noun_phrase += " " + last_eojeol

    return noun_phrase.strip()


def normalize_answer(s):
    def remove_(text):
        """ 불필요한 기호 제거 """
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: postprocess(x["text"]), qa["answers"]))
                prediction = postprocess(predictions[qa["id"]])
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {"official_exact_match": exact_match, "official_f1": f1}


def eval_during_train(args):
    expected_version = "KorQuAD_v1.0"

    dataset_file = os.path.join(args.data_dir, args.predict_file)
    prediction_file = os.path.join(args.output_dir, "predictions_.json")

    with open(dataset_file) as dataset_f:
        dataset_json = json.load(dataset_f)
        read_version = "_".join(dataset_json["version"].split("_")[:-1])
        if read_version != expected_version:
            print("Evaluation expects " + expected_version + ", but got dataset with " + read_version, file=sys.stderr)
        dataset = dataset_json["data"]
    with open(prediction_file) as prediction_f:
        predictions = json.load(prediction_f)

    return evaluate(dataset, predictions)


if __name__ == "__main__":
    expected_version = "KorQuAD_v1.0"
    parser = argparse.ArgumentParser(description="Evaluation for KorQuAD " + expected_version)
    parser.add_argument("dataset_file", help="Dataset file")
    parser.add_argument("prediction_file", help="Prediction File")
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        read_version = "_".join(dataset_json["version"].split("_")[:-1])
        if read_version != expected_version:
            print("Evaluation expects " + expected_version + ", but got dataset with " + read_version, file=sys.stderr)
        dataset = dataset_json["data"]
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
