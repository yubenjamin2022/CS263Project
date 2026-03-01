#!/usr/bin/env python3
"""Find training samples whose positive passages contain added phrases.

This script is intended to verify coverage of newly added tokens/phrases
before fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset

from phrase_tokenizer import load_added_ngrams


WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _normalize_words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def _first_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v if v else None
    if isinstance(value, dict):
        for k in ("text", "contents", "content", "passage", "context", "body", "document"):
            if k in value:
                t = _first_text(value[k])
                if t:
                    return t
        return None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            t = _first_text(item)
            if t:
                return t
    return None


def _extract_query(example: Dict[str, Any]) -> Optional[str]:
    for k in ("query", "question", "title", "prompt", "instruction"):
        if k in example:
            t = _first_text(example[k])
            if t:
                return t
    return None


def _extract_positive_generic(example: Dict[str, Any]) -> Optional[str]:
    for k in ("positive", "pos", "context", "passage", "document"):
        if k in example:
            t = _first_text(example[k])
            if t:
                return t
    for k in (
        "positive_ctxs",
        "positive_contexts",
        "positives",
        "contexts",
        "documents",
        "passages",
        "ctxs",
    ):
        if k in example:
            t = _first_text(example[k])
            if t:
                return t
    return None


def _extract_msmarco_positive(example: Dict[str, Any]) -> Optional[str]:
    passages = example.get("passages")
    if not isinstance(passages, dict):
        return None
    texts = passages.get("passage_text")
    selected = passages.get("is_selected")
    if not isinstance(texts, Sequence):
        return None

    if isinstance(selected, Sequence):
        for text, sel in zip(texts, selected):
            try:
                chosen = int(sel) == 1
            except Exception:
                chosen = False
            if chosen:
                t = _first_text(text)
                if t:
                    return t

    for text in texts:
        t = _first_text(text)
        if t:
            return t
    return None


@dataclass
class _TrieNode:
    children: Dict[str, "_TrieNode"] = field(default_factory=dict)
    phrase_ids: List[int] = field(default_factory=list)


class PhraseMatcher:
    def __init__(self, phrases: Sequence[str]):
        self.root = _TrieNode()
        for idx, phrase in enumerate(phrases):
            words = _normalize_words(phrase)
            if not words:
                continue
            node = self.root
            for word in words:
                node = node.children.setdefault(word, _TrieNode())
            node.phrase_ids.append(idx)

    def find_phrase_ids(self, text: str) -> List[int]:
        words = _normalize_words(text)
        matches = set()
        n = len(words)
        for i in range(n):
            node = self.root
            j = i
            while j < n:
                nxt = node.children.get(words[j])
                if nxt is None:
                    break
                node = nxt
                if node.phrase_ids:
                    matches.update(node.phrase_ids)
                j += 1
        return list(matches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--added-ngrams-file",
        default="AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv",
        help="TSV from build_added_vocab.py with columns including phrase/token.",
    )
    parser.add_argument("--dataset-name", default="ms_marco")
    parser.add_argument("--dataset-config", default="v1.1")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--per-token-limit",
        type=int,
        default=2,
        help="Max number of matched samples to keep per added token.",
    )
    parser.add_argument(
        "--max-dataset-examples",
        type=int,
        default=300000,
        help="Stop scanning after this many dataset rows.",
    )
    parser.add_argument(
        "--max-added-tokens",
        type=int,
        default=None,
        help="Optional cap on added phrases for faster debugging.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use HF streaming mode for large datasets.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/msmarco_added_token_coverage.jsonl",
        help="Matched records, one line per phrase-sample pair.",
    )
    parser.add_argument(
        "--output-summary-tsv",
        default="data/msmarco_added_token_coverage_summary.tsv",
        help="Per-phrase coverage counts.",
    )
    return parser.parse_args()


def _iter_dataset(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {"split": args.split, "streaming": args.streaming}
    if args.dataset_config:
        ds = load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    else:
        ds = load_dataset(args.dataset_name, **kwargs)

    if not args.streaming and args.max_dataset_examples is not None and args.max_dataset_examples > 0:
        cap = min(args.max_dataset_examples, len(ds))
        ds = ds.select(range(cap))
    return ds


def main() -> None:
    args = parse_args()

    entries = load_added_ngrams(args.added_ngrams_file)
    if args.max_added_tokens is not None and args.max_added_tokens > 0:
        entries = entries[: args.max_added_tokens]
    phrases = [p for p, _ in entries]
    tokens = [t for _, t in entries]
    matcher = PhraseMatcher(phrases)

    counts: DefaultDict[int, int] = defaultdict(int)
    matches_out: List[Dict[str, Any]] = []

    total_rows = 0
    for row in _iter_dataset(args):
        if args.max_dataset_examples is not None and args.max_dataset_examples > 0 and total_rows >= args.max_dataset_examples:
            break
        total_rows += 1

        if args.dataset_name == "ms_marco":
            query = _first_text(row.get("query"))
            positive = _extract_msmarco_positive(row)
        else:
            query = _extract_query(row)
            positive = _extract_positive_generic(row)
        if not query or not positive:
            continue

        phrase_ids = matcher.find_phrase_ids(positive)
        if not phrase_ids:
            continue

        for pid in phrase_ids:
            if counts[pid] >= args.per_token_limit:
                continue
            counts[pid] += 1
            matches_out.append(
                {
                    "phrase_id": pid,
                    "phrase": phrases[pid],
                    "token": tokens[pid],
                    "query": query,
                    "positive": positive,
                    "dataset_row_index": total_rows - 1,
                }
            )

        # Early-stop if every phrase reached requested count.
        if len(counts) == len(phrases) and all(counts[i] >= args.per_token_limit for i in range(len(phrases))):
            break

        if total_rows % 10000 == 0:
            covered = sum(1 for i in range(len(phrases)) if counts[i] > 0)
            print(
                f"Scanned {total_rows} rows | covered {covered}/{len(phrases)} phrases | "
                f"saved pairs {len(matches_out)}"
            )

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in matches_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_summary = Path(args.output_summary_tsv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["phrase_id", "phrase", "token", "coverage_count"])
        for i, (phrase, token) in enumerate(zip(phrases, tokens)):
            writer.writerow([i, phrase, token, counts[i]])

    covered = sum(1 for i in range(len(phrases)) if counts[i] > 0)
    saturated = sum(1 for i in range(len(phrases)) if counts[i] >= args.per_token_limit)
    meta = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "rows_scanned": total_rows,
        "phrases_total": len(phrases),
        "phrases_with_at_least_1_match": covered,
        "phrases_with_full_quota": saturated,
        "per_token_limit": args.per_token_limit,
        "matched_records": len(matches_out),
        "output_jsonl": str(out_jsonl),
        "output_summary_tsv": str(out_summary),
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
