#!/usr/bin/env python3
"""Download/load model, add tokens, and initialize new embeddings by mean composition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer

from phrase_tokenizer import load_added_ngrams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model id or local checkpoint path.",
    )
    parser.add_argument(
        "--added-ngrams-file",
        default="AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv",
        help="Output from build_added_vocab.py (or any compatible phrase/token list).",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/all-MiniLM-L6-v2-bioasq-expanded",
        help="Where to save the expanded SentenceTransformer checkpoint.",
    )
    parser.add_argument("--joiner", default="_")
    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--device", default=None, help="cuda/cpu/mps. Default: SentenceTransformer auto.")
    parser.add_argument("--safe-serialization", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _load_phrase_token_map(path: str | Path, joiner: str, lowercase: bool) -> List[Tuple[str, str]]:
    # Keep phrase ordering stable from input file.
    return load_added_ngrams(path=path, joiner=joiner, lowercase=lowercase)


def main() -> None:
    args = parse_args()
    lowercase = not args.no_lowercase

    model_kwargs = {"local_files_only": args.local_files_only}
    if args.device is not None:
        model_kwargs["device"] = args.device
    model = SentenceTransformer(args.model_name_or_path, **model_kwargs)

    transformer_module = model._first_module()
    tokenizer = transformer_module.tokenizer
    auto_model = transformer_module.auto_model

    entries = _load_phrase_token_map(args.added_ngrams_file, joiner=args.joiner, lowercase=lowercase)

    base_vocab = tokenizer.get_vocab()
    embedding_layer = auto_model.get_input_embeddings()
    base_weights = embedding_layer.weight.data.detach().clone()
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        raise ValueError("Tokenizer has no unk_token_id; cannot initialize missing token decomposition.")

    token_to_phrase: Dict[str, str] = {}
    token_init_vectors: Dict[str, torch.Tensor] = {}
    tokens_to_add: List[str] = []

    for phrase, token in entries:
        if token in base_vocab or token in token_to_phrase:
            continue
        ids = tokenizer.encode(phrase, add_special_tokens=False)
        if not ids:
            ids = [unk_id]
        init_vec = base_weights[ids].mean(dim=0)
        token_to_phrase[token] = phrase
        token_init_vectors[token] = init_vec
        tokens_to_add.append(token)

    num_added = tokenizer.add_tokens(tokens_to_add, special_tokens=False)
    auto_model.resize_token_embeddings(len(tokenizer))
    new_embedding_layer = auto_model.get_input_embeddings()

    for token in tokens_to_add[:num_added]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        new_embedding_layer.weight.data[token_id].copy_(token_init_vectors[token])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir), safe_serialization=args.safe_serialization)

    stats_path = output_dir / "added_tokens_init.tsv"
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["token_id", "token", "phrase"])
        for token in tokens_to_add[:num_added]:
            writer.writerow([tokenizer.convert_tokens_to_ids(token), token, token_to_phrase[token]])

    meta = {
        "source_model": args.model_name_or_path,
        "added_ngrams_file": args.added_ngrams_file,
        "requested_additions": len(tokens_to_add),
        "actual_added_tokens": num_added,
        "new_vocab_size": len(tokenizer),
        "output_dir": str(output_dir),
        "added_tokens_init_tsv": str(stats_path),
    }
    (output_dir / "expansion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
