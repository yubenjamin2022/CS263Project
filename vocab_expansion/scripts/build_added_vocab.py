#!/usr/bin/env python3
"""Select added n-grams until vocabulary grows by a target percentage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer

from phrase_tokenizer import canonical_added_token


def read_candidate_file(path: str | Path) -> List[Tuple[float, int, str]]:
    rows: List[Tuple[float, int, str]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if len(parts) < 3:
                continue
            try:
                score = float(parts[0].strip())
                freq = int(parts[1].strip())
            except ValueError:
                continue
            phrase = parts[2].strip()
            if not phrase:
                continue
            rows.append((score, freq, phrase))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--candidate-file",
        default="AutoPhrase/models/BioASQ/all_ngrams_score_ge_0.8_by_freq.txt",
        help="TSV: score<TAB>freq<TAB>phrase in descending frequency order.",
    )
    parser.add_argument(
        "--growth-ratio",
        type=float,
        default=0.10,
        help="Vocabulary expansion ratio relative to the original vocab size.",
    )
    parser.add_argument(
        "--output-file",
        default="AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv",
        help="TSV output with columns: score, freq, phrase, token.",
    )
    parser.add_argument(
        "--joiner",
        default="_",
        help="Replacement used to convert multi-word phrases to single-token surface forms.",
    )
    parser.add_argument("--no-lowercase", action="store_true", help="Do not lowercase phrases.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lowercase = not args.no_lowercase

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=args.local_files_only
    )
    vocab = tokenizer.get_vocab()
    original_vocab_size = len(vocab)
    target_new_tokens = int(math.ceil(original_vocab_size * args.growth_ratio))

    candidates = read_candidate_file(args.candidate_file)

    selected: List[Tuple[float, int, str, str]] = []
    seen_new_tokens = set()
    for score, freq, phrase in candidates:
        new_token = canonical_added_token(phrase, joiner=args.joiner, lowercase=lowercase)
        if not new_token or new_token in vocab or new_token in seen_new_tokens:
            continue
        seen_new_tokens.add(new_token)
        selected.append((score, freq, phrase, new_token))
        if len(selected) >= target_new_tokens:
            break

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["score", "freq", "phrase", "token"])
        writer.writerows(selected)

    meta = {
        "model_name_or_path": args.model_name_or_path,
        "candidate_file": args.candidate_file,
        "growth_ratio": args.growth_ratio,
        "original_vocab_size": original_vocab_size,
        "target_new_tokens": target_new_tokens,
        "selected_new_tokens": len(selected),
        "joiner": args.joiner,
        "lowercase": lowercase,
        "output_file": str(out_path),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
