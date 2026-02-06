#!/usr/bin/env python3
"""
Debug script to inspect passage ID formats and find the mismatch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datasets import load_dataset


def main():
    print("=" * 60)
    print("Debugging Passage ID Mismatch")
    print("=" * 60)
    
    # Load both configs
    print("\n[1] Loading corpus (text-corpus config)...")
    corpus_dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")
    corpus_split = corpus_dataset["passages"]
    
    print("\n[2] Loading questions (question-answer-passages config)...")
    qa_dataset = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages")
    qa_split = qa_dataset["test"]
    
    # Inspect corpus
    print("\n" + "=" * 60)
    print("CORPUS INSPECTION")
    print("=" * 60)
    print(f"Corpus columns: {corpus_split.column_names}")
    print(f"Number of passages: {len(corpus_split)}")
    
    print("\nFirst 5 corpus entries:")
    for i in range(min(5, len(corpus_split))):
        item = corpus_split[i]
        print(f"  {i}: {item}")
    
    # Get all corpus IDs
    corpus_ids = set()
    for item in corpus_split:
        # Try different possible field names
        if "id" in item:
            corpus_ids.add(item["id"])
        if "passage_id" in item:
            corpus_ids.add(item["passage_id"])
    
    print(f"\nSample corpus IDs: {list(corpus_ids)[:10]}")
    print(f"Corpus ID type: {type(list(corpus_ids)[0]) if corpus_ids else 'N/A'}")
    
    # Inspect questions
    print("\n" + "=" * 60)
    print("QUESTIONS INSPECTION")
    print("=" * 60)
    print(f"Questions columns: {qa_split.column_names}")
    print(f"Number of questions: {len(qa_split)}")
    
    print("\nFirst 3 question entries (RAW):")
    for i in range(min(3, len(qa_split))):
        item = qa_split[i]
        print(f"\n  Question {i}:")
        print(f"    question: {item['question'][:80]}...")
        print(f"    answer: {str(item['answer'])[:80] if item['answer'] else 'None'}...")
        
        # Show RAW relevant_passage_ids
        raw_ids = item['relevant_passage_ids']
        print(f"    relevant_passage_ids (raw): {repr(raw_ids)[:100]}...")
        print(f"    Type: {type(raw_ids)}")
        
        # If it's a string, show how to parse it
        if isinstance(raw_ids, str):
            parsed = [int(x.strip()) for x in raw_ids.split(",") if x.strip().isdigit()]
            print(f"    Parsed as integers: {parsed[:5]}...")
    
    # Get all relevant IDs from questions (properly parsed)
    relevant_ids = set()
    for item in qa_split:
        raw = item["relevant_passage_ids"]
        if isinstance(raw, str):
            # Parse comma-separated string
            for x in raw.split(","):
                x = x.strip()
                if x.isdigit():
                    relevant_ids.add(int(x))
        elif isinstance(raw, list):
            for x in raw:
                relevant_ids.add(int(x) if isinstance(x, str) else x)
    
    print(f"\nSample relevant passage IDs (parsed): {list(relevant_ids)[:10]}")
    
    # Check overlap
    print("\n" + "=" * 60)
    print("ID OVERLAP CHECK")
    print("=" * 60)
    
    overlap = corpus_ids & relevant_ids
    print(f"Corpus IDs: {len(corpus_ids)}")
    print(f"Relevant IDs in questions: {len(relevant_ids)}")
    print(f"Overlap: {len(overlap)}")
    
    if len(overlap) == 0:
        print("\n⚠️  NO OVERLAP! This explains the 0.0 scores.")
        print("\nLet's compare ID formats:")
        corpus_sample = list(corpus_ids)[:3]
        relevant_sample = list(relevant_ids)[:3]
        print(f"  Corpus IDs look like: {corpus_sample}")
        print(f"  Relevant IDs look like: {relevant_sample}")
        
        # Check if it's a type mismatch
        if corpus_ids and relevant_ids:
            corpus_type = type(list(corpus_ids)[0])
            relevant_type = type(list(relevant_ids)[0])
            print(f"\n  Corpus ID type: {corpus_type}")
            print(f"  Relevant ID type: {relevant_type}")
    else:
        print(f"\n✓ Found {len(overlap)} overlapping IDs!")
        print(f"  Sample overlapping IDs: {list(overlap)[:5]}")
        print(f"\n  This means {len(overlap)/len(relevant_ids)*100:.1f}% of relevant IDs exist in corpus")


if __name__ == "__main__":
    main()
