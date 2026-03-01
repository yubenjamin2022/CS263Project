#!/usr/bin/env python3
"""Analyze added phrase/token frequency in BioASQ test questions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import load_dataset


WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def normalize_words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    phrase_ids: List[int] = field(default_factory=list)


def build_phrase_trie(phrases: Sequence[str]) -> TrieNode:
    root = TrieNode()
    for idx, phrase in enumerate(phrases):
        words = normalize_words(phrase)
        if not words:
            continue
        node = root
        for w in words:
            node = node.children.setdefault(w, TrieNode())
        node.phrase_ids.append(idx)
    return root


def find_phrase_matches(words: List[str], root: TrieNode) -> List[int]:
    """Return all matched phrase ids (with repeats for repeated occurrences)."""
    out: List[int] = []
    n = len(words)
    for i in range(n):
        node = root
        j = i
        while j < n:
            nxt = node.children.get(words[j])
            if nxt is None:
                break
            node = nxt
            if node.phrase_ids:
                out.extend(node.phrase_ids)
            j += 1
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--added-ngrams-file",
        default="AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv",
        help="TSV from build_added_vocab.py.",
    )
    parser.add_argument("--dataset-name", default="rag-datasets/rag-mini-bioasq")
    parser.add_argument("--qa-config", default="question-answer-passages")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--output-tsv",
        default="outputs/added_token_question_frequency.tsv",
        help="Per-token/phrase frequency table.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/added_token_question_frequency_summary.json",
        help="Summary stats and top matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows: List[Tuple[float, int, str, str]] = []
    with open(args.added_ngrams_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for parts in reader:
            if not parts:
                continue
            if parts[0].strip().lower() == "score":
                continue
            if len(parts) >= 4:
                score = float(parts[0])
                freq = int(parts[1])
                phrase = parts[2].strip()
                token = parts[3].strip()
            elif len(parts) >= 3:
                score = float(parts[0])
                freq = int(parts[1])
                phrase = parts[2].strip()
                token = phrase.lower().replace(" ", "_")
            else:
                continue
            if phrase:
                rows.append((score, freq, phrase, token))

    if not rows:
        raise ValueError(f"No added phrases found in {args.added_ngrams_file}")

    scores = [r[0] for r in rows]
    corpus_freqs = [r[1] for r in rows]
    phrases = [r[2] for r in rows]
    tokens = [r[3] for r in rows]
    trie = build_phrase_trie(phrases)

    ds = load_dataset(args.dataset_name, args.qa_config, split=args.split)
    if args.max_questions is not None and args.max_questions > 0:
        ds = ds.select(range(min(args.max_questions, len(ds))))

    question_texts = []
    for row in ds:
        q = row.get("question", row.get("query", ""))
        question_texts.append(str(q))

    q_doc_freq = Counter()   # in how many questions phrase appears
    q_occ_freq = Counter()   # total occurrences across all questions

    for q in question_texts:
        words = normalize_words(q)
        hits = find_phrase_matches(words, trie)
        if not hits:
            continue
        q_occ_freq.update(hits)
        q_doc_freq.update(set(hits))

    out_rows = []
    for i in range(len(phrases)):
        out_rows.append(
            (
                q_doc_freq[i],
                q_occ_freq[i],
                scores[i],
                corpus_freqs[i],
                phrases[i],
                tokens[i],
            )
        )
    out_rows.sort(key=lambda x: (-x[0], -x[1], -x[2], x[4]))

    out_tsv = Path(args.output_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "questions_with_phrase",
                "total_occurrences_in_questions",
                "autophrase_score",
                "autophrase_corpus_freq",
                "phrase",
                "token",
            ]
        )
        writer.writerows(out_rows)

    seen = sum(1 for r in out_rows if r[0] > 0)
    summary = {
        "num_questions": len(question_texts),
        "num_added_phrases": len(phrases),
        "num_added_phrases_seen_in_questions": seen,
        "coverage_ratio": seen / float(len(phrases)),
        "total_phrase_occurrences_in_questions": int(sum(q_occ_freq.values())),
        "top20_by_question_freq": [
            {
                "questions_with_phrase": int(a),
                "total_occurrences_in_questions": int(b),
                "autophrase_score": float(c),
                "autophrase_corpus_freq": int(d),
                "phrase": e,
                "token": f,
            }
            for a, b, c, d, e, f in out_rows[:20]
        ],
        "output_tsv": str(out_tsv),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
