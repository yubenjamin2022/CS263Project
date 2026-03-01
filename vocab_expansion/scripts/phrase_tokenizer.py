#!/usr/bin/env python3
"""Greedy phrase-aware tokenizer wrapper for Hugging Face tokenizers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def canonical_added_token(phrase: str, joiner: str = "_", lowercase: bool = True) -> str:
    """Convert a phrase to a single-token surface form."""
    cleaned = " ".join(phrase.strip().split())
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned.replace(" ", joiner)


def load_added_ngrams(
    path: str | Path,
    joiner: str = "_",
    lowercase: bool = True,
) -> List[Tuple[str, str]]:
    """Load added n-grams from TSV/TXT.

    Supported formats:
    - TSV with header including `phrase` and optional `token`.
    - TSV rows: score, freq, phrase, token.
    - One phrase per line.
    """
    rows: List[Tuple[str, str]] = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Added n-gram file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        sample = f.readline()
        f.seek(0)

        # Heuristic for TSV-like rows.
        if "\t" in sample:
            reader = csv.reader(f, delimiter="\t")
            first = True
            for parts in reader:
                if not parts:
                    continue
                if first and parts[0].lower() in {"score", "phrase"}:
                    first = False
                    continue
                first = False
                phrase = ""
                token = ""
                if len(parts) >= 4:
                    phrase = parts[2].strip()
                    token = parts[3].strip()
                elif len(parts) == 2:
                    phrase = parts[0].strip()
                    token = parts[1].strip()
                elif len(parts) == 1:
                    phrase = parts[0].strip()
                elif len(parts) >= 3:
                    phrase = parts[2].strip()
                if not phrase:
                    continue
                if not token:
                    token = canonical_added_token(phrase, joiner=joiner, lowercase=lowercase)
                rows.append((phrase, token))
        else:
            for line in f:
                phrase = line.strip()
                if not phrase:
                    continue
                token = canonical_added_token(phrase, joiner=joiner, lowercase=lowercase)
                rows.append((phrase, token))

    # Preserve order; dedupe by token.
    seen = set()
    deduped: List[Tuple[str, str]] = []
    for phrase, token in rows:
        if token in seen:
            continue
        seen.add(token)
        deduped.append((phrase, token))
    return deduped


@dataclass
class _TrieNode:
    children: Dict[str, "_TrieNode"] = field(default_factory=dict)
    token: Optional[str] = None


class GreedyPhraseTokenizer:
    """Greedy longest-match phrase pre-tokenizer + base tokenizer wrapper."""

    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizerBase,
        added_ngrams: Sequence[Tuple[str, str]] | Sequence[str],
        joiner: str = "_",
        lowercase: bool = True,
    ) -> None:
        self.base_tokenizer = base_tokenizer
        self.joiner = joiner
        self.lowercase = lowercase
        self.root = _TrieNode()
        self.max_len = 1

        entries: List[Tuple[str, str]] = []
        for item in added_ngrams:
            if isinstance(item, str):
                phrase = item
                token = canonical_added_token(item, joiner=joiner, lowercase=lowercase)
            else:
                phrase, token = item
            entries.append((phrase, token))

        for phrase, token in entries:
            pieces = [p for p in phrase.split() if p]
            if not pieces:
                continue
            if self.lowercase:
                pieces = [p.lower() for p in pieces]
            self.max_len = max(self.max_len, len(pieces))
            node = self.root
            for piece in pieces:
                node = node.children.setdefault(piece, _TrieNode())
            node.token = token

    def preprocess_text(self, text: str) -> str:
        words = text.split()
        if not words:
            return text

        normalized = [w.lower() if self.lowercase else w for w in words]
        out: List[str] = []
        i = 0
        n = len(words)

        while i < n:
            node = self.root
            best_token = None
            best_len = 0
            for j in range(i, min(n, i + self.max_len)):
                piece = normalized[j]
                if piece not in node.children:
                    break
                node = node.children[piece]
                if node.token is not None:
                    best_token = node.token
                    best_len = j - i + 1
            if best_token is None:
                out.append(words[i])
                i += 1
            else:
                out.append(best_token)
                i += best_len

        return " ".join(out)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self.base_tokenizer.tokenize(self.preprocess_text(text), **kwargs)

    def encode(self, text: str, **kwargs):
        return self.base_tokenizer.encode(self.preprocess_text(text), **kwargs)

    def __call__(self, text=None, text_pair=None, **kwargs):
        def _map(x):
            if x is None:
                return None
            if isinstance(x, str):
                return self.preprocess_text(x)
            if isinstance(x, list):
                return [self.preprocess_text(t) if isinstance(t, str) else t for t in x]
            return x

        return self.base_tokenizer(_map(text), text_pair=_map(text_pair), **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy phrase-aware tokenization demo")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--added-ngrams-file", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--joiner", default="_")
    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lowercase = not args.no_lowercase
    entries = load_added_ngrams(args.added_ngrams_file, joiner=args.joiner, lowercase=lowercase)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    phrase_tokenizer = GreedyPhraseTokenizer(tokenizer, entries, joiner=args.joiner, lowercase=lowercase)
    rewritten = phrase_tokenizer.preprocess_text(args.text)
    pieces = phrase_tokenizer.tokenize(args.text)
    print(rewritten)
    print(" ".join(pieces))


if __name__ == "__main__":
    main()
