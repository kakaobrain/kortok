import argparse
import json
import os

import sentencepiece as spm

INPUT_KO_CORPUS = "./dataset/wiki/kowiki-200420.txt"
INPUT_EN_CORPUS = "./dataset/wiki/sample_en-wiki-200420.txt"  # for English SentencePiece(BPE) Tokenizer
INPUT_MECAB_TOKENIZED_CORPUS = "./dataset/wiki/mecab_tokenized/mecab/kowiki-200420.txt"  # for MeCab-SentencePiece Tokenizer

OUTPUT_DIR = "./resources"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--character_coverage", type=str, default=1.0)
    parser.add_argument(
        "--normalization_rule_name",
        type=str,
        default="identity",
        choices=["nmt_nfkc", "nfkc", "nmt_nfkc_cf", "nfkc_cf", "identity"],
    )  # set "nmt_nfkc" for english training
    parser.add_argument("--pad_piece", type=str, default="[PAD]", help="index=0")
    parser.add_argument("--unk_piece", type=str, default="[UNK]", help="index=1")
    parser.add_argument("--bos_piece", type=str, default="[BOS]", help="index=2")
    parser.add_argument("--eos_piece", type=str, default="[EOS]", help="index=3")
    parser.add_argument("--unk_surface", type=str, default="[UNK]")
    parser.add_argument(
        "--special_symbols",
        type=str,
        default="[CLS],[SEP],[MASK]",
        help="Special tokens. You can pass a comma-separated list of special tokens.",
    )
    parser.add_argument(
        "--tokenizer_type", type=str, default="ko", choices=["ko", "en", "mecab_tokenized"]
    )  # ko: Korean Wiki Corpus, en: English Wiki Corpus, mecab_tokenized: Korean Wiki Corpus tokenized by MeCab
    args = vars(parser.parse_args())
    print(args)

    # set output dir
    if args["tokenizer_type"] == "ko":
        input_corpus = INPUT_KO_CORPUS
        output_dir = os.path.join(OUTPUT_DIR, f"sp-{int(args['vocab_size'])//1000}k")
    elif args["tokenizer_type"] == "en":
        input_corpus = INPUT_EN_CORPUS
        output_dir = os.path.join(OUTPUT_DIR, f"en_sp-{int(args['vocab_size'])//1000}k")
    elif args["tokenizer_type"] == "mecab_tokenized":
        input_corpus = INPUT_MECAB_TOKENIZED_CORPUS
        output_dir = os.path.join(OUTPUT_DIR, f"mecab_sp-{int(args['vocab_size'])//1000}k")
    else:
        raise ValueError
    os.makedirs(output_dir, exist_ok=True)

    # save arguments info
    output_info_path = os.path.join(output_dir, "build_info.json")
    with open(output_info_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)

    cmd = f"--input={input_corpus} "
    cmd += f"--model_prefix={os.path.join(output_dir, 'tok')} "
    cmd += f"--vocab_size={args['vocab_size']} "
    cmd += f"--model_type=bpe "
    cmd += f"--character_coverage={args['character_coverage']} "
    cmd += f"--normalization_rule_name={args['normalization_rule_name']} "
    cmd += f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
    cmd += f"--pad_piece={args['pad_piece']} "
    cmd += f"--unk_piece={args['unk_piece']} "
    cmd += f"--bos_piece={args['bos_piece']} "
    cmd += f"--eos_piece={args['eos_piece']} "
    cmd += f"--unk_surface={args['unk_surface']} "
    cmd += f"--user_defined_symbols={args['special_symbols']} "

    spm.SentencePieceTrainer.Train(cmd)

    # fairseq vocab
    with open(os.path.join(output_dir, "fairseq.vocab"), "w") as fout:
        with open(os.path.join(output_dir, "tok.vocab"), "r") as fin:
            start_idx = 4 + len(args["special_symbols"].split(","))  # pad, unk, bos, eos + special_symbols
            for line in fin.readlines()[start_idx:]:
                splitted = line.split("\t")
                fout.write(f"{' '.join(splitted)}")
