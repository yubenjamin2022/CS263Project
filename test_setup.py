#!/usr/bin/env python3
"""
Quick test script to verify the baseline setup works.

This runs a minimal test with 5 samples to verify:
1. Dataset loads correctly
2. Retrieval works
3. Evaluation computes metrics

Usage:
    python test_setup.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_data_loading():
    """Test that the dataset loads correctly."""
    print("\n[1/4] Testing data loading...")
    
    from data_loader import BioASQDataLoader
    
    loader = BioASQDataLoader(max_samples=5)
    questions, corpus = loader.load()
    
    assert len(questions) == 5, f"Expected 5 questions, got {len(questions)}"
    assert len(corpus) > 0, "Corpus is empty"
    assert "question" in questions[0], "Missing 'question' field"
    assert "answer" in questions[0], "Missing 'answer' field"
    assert "relevant_passage_ids" in questions[0], "Missing 'relevant_passage_ids' field"
    
    print(f"   ✓ Loaded {len(questions)} questions")
    print(f"   ✓ Loaded {len(corpus)} corpus passages")
    print(f"   ✓ Sample question: {questions[0]['question'][:80]}...")
    
    return loader


def test_retrieval(loader):
    """Test that retrieval works."""
    print("\n[2/4] Testing retrieval...")
    
    from retriever import DenseRetriever
    
    retriever = DenseRetriever()
    
    # Index a subset of corpus for speed
    corpus_texts, corpus_ids = loader.get_corpus_texts_and_ids()
    subset_size = min(500, len(corpus_texts))
    
    retriever.index_corpus(
        corpus_texts[:subset_size],
        corpus_ids[:subset_size],
        show_progress=False
    )
    
    # Test retrieval
    questions = loader.get_questions()
    retrieved_ids, scores = retriever.retrieve(
        questions[:2],
        top_k=5,
        show_progress=False
    )
    
    assert len(retrieved_ids) == 2, "Wrong number of results"
    assert len(retrieved_ids[0]) == 5, "Wrong number of retrieved passages"
    assert all(isinstance(s, float) for s in scores[0]), "Scores should be floats"
    
    print(f"   ✓ Indexed {subset_size} passages")
    print(f"   ✓ Retrieved {len(retrieved_ids[0])} passages per query")
    print(f"   ✓ Top score: {scores[0][0]:.4f}")
    
    return retriever, retrieved_ids


def test_evaluation(loader, retrieved_ids):
    """Test that evaluation works."""
    print("\n[3/4] Testing evaluation...")
    
    from evaluate import RetrievalEvaluator
    
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
    
    relevant_ids = loader.get_relevant_passage_ids()[:2]
    
    results = evaluator.evaluate(retrieved_ids, relevant_ids)
    
    assert "MRR" in results, "Missing MRR metric"
    assert "Recall@1" in results, "Missing Recall@1"
    assert "Precision@5" in results, "Missing Precision@5"
    
    print(f"   ✓ MRR: {results['MRR']:.4f}")
    print(f"   ✓ Recall@5: {results['Recall@5']:.4f}")
    print(f"   ✓ Precision@5: {results['Precision@5']:.4f}")
    
    return results


def test_imports():
    """Test that all imports work."""
    print("\n[4/4] Testing imports...")
    
    try:
        import torch
        print(f"   ✓ torch {torch.__version__}")
    except ImportError:
        print("   ✗ torch not installed")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("   ✓ sentence-transformers")
    except ImportError:
        print("   ✗ sentence-transformers not installed")
        return False
    
    try:
        from datasets import load_dataset
        print("   ✓ datasets")
    except ImportError:
        print("   ✗ datasets not installed")
        return False
    
    try:
        import numpy as np
        print(f"   ✓ numpy {np.__version__}")
    except ImportError:
        print("   ✗ numpy not installed")
        return False
    
    # Optional dependencies
    try:
        import faiss
        print(f"   ✓ faiss")
    except ImportError:
        print("   ⚠ faiss not installed (will use numpy fallback)")
    
    try:
        from bert_score import score
        print("   ✓ bert-score")
    except ImportError:
        print("   ⚠ bert-score not installed (generation eval will fail)")
    
    try:
        from sacrebleu.metrics import BLEU
        print("   ✓ sacrebleu")
    except ImportError:
        print("   ⚠ sacrebleu not installed (BLEU eval will fail)")
    
    return True


def main():
    print("=" * 60)
    print("BioASQ RAG Baseline - Setup Test".center(60))
    print("=" * 60)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Some required dependencies are missing.")
        print("   Run: pip install -r requirements.txt")
        return 1
    
    try:
        # Test data loading
        loader = test_data_loading()
        
        # Test retrieval
        retriever, retrieved_ids = test_retrieval(loader)
        
        # Test evaluation
        results = test_evaluation(loader, retrieved_ids)
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!".center(60))
        print("=" * 60)
        print("\nYou can now run the full baseline with:")
        print("  python run_baseline.py --max-samples 50 --retrieval-only")
        print("\nOr with generation:")
        print("  python run_baseline.py --max-samples 20")
        print("")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
