#!/usr/bin/env python3
"""Evaluate retrieval on rag-mini-bioasq with optional phrase-aware preprocessing.

This mirrors the notebook flow in ../notebooks/exploration.ipynb:
- encode all passages
- encode all test questions
- retrieve by dense dot-product top-k
- report Precision/Recall/MRR/NDCG at k in {1,5,10,20} (configurable)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from phrase_tokenizer import GreedyPhraseTokenizer, load_added_ngrams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name-or-path",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base or expanded checkpoint path/model id.",
    )
    parser.add_argument("--dataset-name", default="rag-datasets/rag-mini-bioasq")
    parser.add_argument("--text-corpus-config", default="text-corpus")
    parser.add_argument("--qa-config", default="question-answer-passages")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k-values", default="1,5,10,20")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-passages", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda/cpu/mps. Default: auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--added-ngrams-file",
        default="",
        help="Optional TSV/TXT phrase list for greedy phrase merge preprocessing.",
    )
    parser.add_argument(
        "--no-phrase-merge",
        action="store_true",
        help="Disable phrase merge preprocessing even if --added-ngrams-file is provided.",
    )
    parser.add_argument("--output-json", default="", help="Optional path to save full metrics JSON.")
    return parser.parse_args()


def _extract_passage_id(row: Dict[str, Any]) -> int:
    for key in ("id", "pid", "passage_id"):
        if key in row:
            return int(row[key])
    raise KeyError(f"Could not find passage id field in row keys: {sorted(row.keys())}")


def _extract_passage_text(row: Dict[str, Any]) -> str:
    for key in ("passage", "text", "contents", "content", "body"):
        if key in row and row[key]:
            return str(row[key])
    raise KeyError(f"Could not find passage text field in row keys: {sorted(row.keys())}")


def _extract_question(row: Dict[str, Any]) -> str:
    for key in ("question", "query", "prompt"):
        if key in row and row[key]:
            return str(row[key])
    raise KeyError(f"Could not find question field in row keys: {sorted(row.keys())}")


def _parse_relevant_ids(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
        return [int(parsed)]
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return [int(x) for x in raw]
    return [int(raw)]


def compute_metrics(
    retrieved_passages: List[List[int]],
    relevant_by_query: List[List[int]],
    k: int,
) -> Dict[str, float]:
    precision_at_k: List[float] = []
    recall_at_k: List[float] = []
    mrr_at_k: List[float] = []
    ndcg_at_k: List[float] = []

    for i, relevant_list in enumerate(relevant_by_query):
        relevant = set(relevant_list)
        retrieved = retrieved_passages[i][:k]

        hits = len(set(retrieved) & relevant)
        precision_at_k.append(hits / float(k))
        recall_at_k.append(hits / float(len(relevant)) if relevant else 0.0)

        rr = 0.0
        for rank, pid in enumerate(retrieved, start=1):
            if pid in relevant:
                rr = 1.0 / float(rank)
                break
        mrr_at_k.append(rr)

        if relevant:
            idcg = sum(1.0 / math.log2(i + 2.0) for i in range(min(len(relevant), k)))
            dcg = 0.0
            for rank, pid in enumerate(retrieved, start=1):
                if pid in relevant:
                    dcg += 1.0 / math.log2(rank + 1.0)
            ndcg_at_k.append(dcg / idcg if idcg > 0 else 0.0)
        else:
            ndcg_at_k.append(0.0)

    n = float(len(retrieved_passages))
    return {
        "precision": sum(precision_at_k) / n,
        "recall": sum(recall_at_k) / n,
        "mrr": sum(mrr_at_k) / n,
        "ndcg": sum(ndcg_at_k) / n,
    }


def _maybe_preprocess(texts: List[str], phrase_tokenizer: GreedyPhraseTokenizer | None) -> List[str]:
    if phrase_tokenizer is None:
        return texts
    return [phrase_tokenizer.preprocess_text(t) for t in texts]


def main() -> None:
    args = parse_args()
    start_time = time.time()
    k_values = sorted({int(x.strip()) for x in args.k_values.split(",") if x.strip()})
    if not k_values:
        raise ValueError("No valid k values passed to --k-values")
    max_k = max(k_values)

    model_kwargs = {"local_files_only": args.local_files_only}
    if args.device is not None:
        model_kwargs["device"] = args.device
    model = SentenceTransformer(args.model_name_or_path, **model_kwargs)

    phrase_tokenizer = None
    phrase_merge_enabled = bool(args.added_ngrams_file) and not args.no_phrase_merge
    if phrase_merge_enabled:
        entries = load_added_ngrams(args.added_ngrams_file)
        phrase_tokenizer = GreedyPhraseTokenizer(model.tokenizer, entries)
        print(f"Phrase merge enabled with {len(entries)} phrases/tokens from {args.added_ngrams_file}.")
    else:
        print("Phrase merge disabled.")

    text_corpus = load_dataset(args.dataset_name, args.text_corpus_config)
    qap = load_dataset(args.dataset_name, args.qa_config)

    passage_split_name = "passages"
    qa_split_name = args.split

    passage_rows = text_corpus[passage_split_name]
    qa_rows = qap[qa_split_name]

    if args.max_passages is not None:
        passage_rows = passage_rows.select(range(min(args.max_passages, len(passage_rows))))
    if args.max_questions is not None:
        qa_rows = qa_rows.select(range(min(args.max_questions, len(qa_rows))))

    pid_to_passage: Dict[int, str] = {}
    print("Building passage dictionary...")
    for i in tqdm(range(len(passage_rows))):
        row = passage_rows[i]
        pid_to_passage[_extract_passage_id(row)] = _extract_passage_text(row)

    sorted_pids = sorted(pid_to_passage.keys())
    idx_to_pid = {idx: pid for idx, pid in enumerate(sorted_pids)}
    passages_list = [pid_to_passage[pid] for pid in sorted_pids]
    passages_list = _maybe_preprocess(passages_list, phrase_tokenizer)

    print(f"Encoding {len(passages_list)} passages...")
    passage_emb = model.encode(
        passages_list,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Encoding {len(qa_rows)} questions...")
    questions_list = [_extract_question(qa_rows[i]) for i in range(len(qa_rows))]
    questions_list = _maybe_preprocess(questions_list, phrase_tokenizer)
    question_emb = model.encode(
        questions_list,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    device = model.device
    question_t = torch.from_numpy(question_emb).to(device)
    passage_t = torch.from_numpy(passage_emb).to(device)

    print("Retrieving top-k passages...")
    retrieved_passages: List[List[int]] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(question_t), args.batch_size)):
            q_batch = question_t[i : i + args.batch_size]
            scores = q_batch @ passage_t.T
            _, topk_idx = torch.topk(scores, k=max_k, dim=1)
            for indices in topk_idx.cpu().tolist():
                retrieved_passages.append([idx_to_pid[idx] for idx in indices])

    relevant_by_query: List[List[int]] = []
    for i in range(len(qa_rows)):
        row = qa_rows[i]
        raw_relevant = row.get("relevant_passage_ids", row.get("relevant_passages"))
        relevant_by_query.append(_parse_relevant_ids(raw_relevant))

    metrics_by_k: Dict[str, Dict[str, float]] = {}
    print("\nMetrics:")
    for k in k_values:
        metrics = compute_metrics(retrieved_passages, relevant_by_query, k)
        metrics_by_k[str(k)] = metrics
        print(
            f"k={k:>2d} | "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"mrr={metrics['mrr']:.4f} "
            f"ndcg={metrics['ndcg']:.4f}"
        )

    result = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_name": args.dataset_name,
        "text_corpus_config": args.text_corpus_config,
        "qa_config": args.qa_config,
        "split": args.split,
        "num_passages": len(passages_list),
        "num_questions": len(questions_list),
        "k_values": k_values,
        "phrase_merge_enabled": phrase_merge_enabled,
        "added_ngrams_file": args.added_ngrams_file if phrase_merge_enabled else "",
        "metrics": metrics_by_k,
        "elapsed_seconds": time.time() - start_time,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
