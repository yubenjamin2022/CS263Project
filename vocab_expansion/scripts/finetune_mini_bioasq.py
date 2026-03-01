#!/usr/bin/env python3
"""Fine-tune SentenceTransformer on rag-mini-bioasq retrieval pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

# SentenceTransformers v5 defaults to DataParallel when multiple GPUs are visible.
# Some systems hit CUDA peer-mapping limits with DP; default to one visible GPU.
if os.environ.get("ST_FORCE_SINGLE_GPU", "1") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from phrase_tokenizer import GreedyPhraseTokenizer, load_added_ngrams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        default="checkpoints/all-MiniLM-L6-v2-bioasq-expanded",
        help="Local checkpoint path (recommended) or model id.",
    )
    parser.add_argument(
        "--dataset-name",
        default="ms_marco",
        help="HF dataset id. Default uses MS MARCO passage ranking data.",
    )
    parser.add_argument(
        "--dataset-config",
        default="v1.1",
        help="Optional HF dataset config/subset name (e.g., v1.1 for ms_marco).",
    )
    parser.add_argument(
        "--split",
        default="train[:1%]",
        help="Dataset split (supports slicing). Default is a small subset.",
    )
    parser.add_argument(
        "--coverage-jsonl",
        default=None,
        help="Optional JSONL from find_added_token_coverage.py. If set, training pairs come from this file.",
    )
    parser.add_argument(
        "--added-ngrams-file",
        default="AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv",
        help="Optional phrase list for preprocessing. Use empty string to disable.",
    )
    parser.add_argument("--output-dir", default="checkpoints/all-MiniLM-L6-v2-bioasq-finetuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda/cpu/mps. Default: auto")
    parser.add_argument(
        "--freeze-transformer",
        action="store_true",
        help="Freeze transformer encoder weights while keeping input token embeddings trainable.",
    )
    parser.add_argument(
        "--train-added-embeddings-only",
        action="store_true",
        help="Only update embedding rows for added tokens; freeze all other parameters.",
    )
    parser.add_argument(
        "--added-token-ids-file",
        default="",
        help="Optional TSV of token IDs (e.g. checkpoints/.../added_tokens_init.tsv) for --train-added-embeddings-only.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _extract_positive(example: Dict[str, Any]) -> Optional[str]:
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

    # Prefer annotated positives.
    if isinstance(selected, Sequence):
        for text, sel in zip(texts, selected):
            if int(sel) == 1:
                t = _first_text(text)
                if t:
                    return t

    # Fallback to first non-empty passage text.
    for text in texts:
        t = _first_text(text)
        if t:
            return t
    return None


def load_pair_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    max_samples: Optional[int],
) -> List[Tuple[str, str]]:
    if dataset_config:
        data: Dataset = load_dataset(dataset_name, dataset_config, split=split)  # type: ignore[assignment]
    else:
        data = load_dataset(dataset_name, split=split)  # type: ignore[assignment]
    if max_samples is not None and max_samples > 0:
        data = data.select(range(min(max_samples, len(data))))

    pairs: List[Tuple[str, str]] = []
    for row in data:
        if dataset_name == "ms_marco":
            query = _first_text(row.get("query"))
            positive = _extract_msmarco_positive(row)
        else:
            query = _extract_query(row)
            positive = _extract_positive(row)
        if query and positive:
            pairs.append((query, positive))

    if not pairs:
        raise ValueError(
            "No (query, positive) pairs could be extracted from the dataset. "
            "Inspect dataset columns and update extraction rules."
        )
    return pairs


def load_pair_dataset_from_coverage_jsonl(path: str, max_samples: Optional[int]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = _first_text(row.get("query"))
            positive = _first_text(row.get("positive"))
            if query and positive:
                pairs.append((query, positive))
            if max_samples is not None and max_samples > 0 and len(pairs) >= max_samples:
                break
    if not pairs:
        raise ValueError(f"No valid pairs found in coverage file: {path}")
    return pairs


def load_added_token_ids_from_tsv(path: str) -> List[int]:
    ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "token_id":
                continue
            try:
                ids.append(int(row[0]))
            except ValueError:
                continue
    return sorted(set(ids))


def resolve_added_token_ids(
    model: SentenceTransformer,
    entries: List[Tuple[str, str]],
    added_token_ids_file: str,
) -> List[int]:
    if added_token_ids_file:
        token_ids = load_added_token_ids_from_tsv(added_token_ids_file)
        if token_ids:
            return token_ids

    tokenizer = model.tokenizer
    vocab = tokenizer.get_vocab()
    unk_id = tokenizer.unk_token_id
    token_ids: List[int] = []
    for _, token in entries:
        if token not in vocab:
            continue
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None:
            continue
        tid = int(tid)
        if unk_id is not None and tid == int(unk_id):
            continue
        token_ids.append(tid)
    return sorted(set(token_ids))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_kwargs = {"local_files_only": args.local_files_only}
    if args.device is not None:
        model_kwargs["device"] = args.device
    model = SentenceTransformer(args.model_name_or_path, **model_kwargs)

    phrase_tokenizer = None
    entries: List[Tuple[str, str]] = []
    if args.added_ngrams_file:
        entries = load_added_ngrams(args.added_ngrams_file)
        phrase_tokenizer = GreedyPhraseTokenizer(model.tokenizer, entries)

    first_module = model._first_module()
    auto_model = first_module.auto_model

    if args.train_added_embeddings_only:
        if not entries and not args.added_token_ids_file:
            raise ValueError(
                "--train-added-embeddings-only requires --added-ngrams-file or --added-token-ids-file."
            )
        token_ids = resolve_added_token_ids(model, entries, args.added_token_ids_file)
        if not token_ids:
            raise ValueError(
                "Could not resolve any added token IDs. "
                "Pass the expanded checkpoint and/or --added-token-ids-file."
            )

        for _, param in model.named_parameters():
            param.requires_grad = False
        emb_weight = auto_model.get_input_embeddings().weight
        emb_weight.requires_grad = True

        # Mask gradients so only selected token rows are updated.
        mask_cpu = torch.zeros(emb_weight.shape[0], dtype=torch.bool)
        mask_cpu[token_ids] = True
        mask_cache = {"mask": None}

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = mask_cache["mask"]
            if mask is None or mask.device != grad.device:
                mask = mask_cpu.to(grad.device)
                mask_cache["mask"] = mask
            return grad * mask.unsqueeze(1).to(dtype=grad.dtype)

        emb_weight.register_hook(_mask_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        emb_dim = int(emb_weight.shape[1])
        effective_trainable = len(token_ids) * emb_dim
        print(
            f"train-added-embeddings-only enabled: rows={len(token_ids)} "
            f"(vocab={emb_weight.shape[0]}, dim={emb_dim}), "
            f"effective params={effective_trainable}/{total} "
            f"({100.0 * effective_trainable / max(1, total):.2f}%), "
            f"tensor trainable params={trainable}/{total}."
        )
    elif args.freeze_transformer:
        for _, param in auto_model.named_parameters():
            param.requires_grad = False
        # Keep token embedding matrix trainable (includes added token rows).
        auto_model.get_input_embeddings().weight.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"freeze-transformer enabled: trainable params={trainable}/{total} "
            f"({100.0 * trainable / max(1, total):.2f}%)."
        )

    if args.coverage_jsonl:
        pairs = load_pair_dataset_from_coverage_jsonl(
            path=args.coverage_jsonl,
            max_samples=args.max_train_samples,
        )
    else:
        pairs = load_pair_dataset(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            max_samples=args.max_train_samples,
        )

    train_examples: List[InputExample] = []
    for query, positive in pairs:
        if phrase_tokenizer is not None:
            query = phrase_tokenizer.preprocess_text(query)
            positive = phrase_tokenizer.preprocess_text(positive)
        train_examples.append(InputExample(texts=[query, positive]))

    train_dataloader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    total_steps = max(1, len(train_dataloader) * args.epochs)
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))

    fit_weight_decay = 0.0 if args.train_added_embeddings_only else 0.01
    if args.train_added_embeddings_only:
        print("train-added-embeddings-only: setting weight_decay=0.0 to avoid drift on non-target rows.")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        optimizer_params={"lr": args.learning_rate},
        weight_decay=fit_weight_decay,
        show_progress_bar=True,
    )

    print(f"Saved fine-tuned model to: {args.output_dir}")


if __name__ == "__main__":
    main()
