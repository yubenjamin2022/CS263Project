#!/usr/bin/env python3
"""
BioASQ RAG Baseline - Main Entry Point

Run the full RAG pipeline for medical question answering:
1. Load BioASQ dataset
2. Build dense retrieval index
3. Retrieve relevant passages for each question
4. Evaluate retrieval with Recall/Precision/MRR
5. Generate answers using a local LLM (Qwen2.5-0.5B-Instruct by default)
6. Evaluate generation with BERTScore/BLEU

Usage:
    python run_baseline.py                    # Run full pipeline
    python run_baseline.py --retrieval-only   # Only evaluate retrieval
    python run_baseline.py --top-k 10         # Retrieve 10 passages
    python run_baseline.py --max-samples 100  # Use 100 samples only
    
Available Models (--generation-model):
    - Qwen/Qwen2.5-0.5B-Instruct (default, ~1GB RAM)
    - Qwen/Qwen2.5-1.5B-Instruct (~3GB RAM)
    - Qwen/Qwen2.5-3B-Instruct (~6GB RAM)
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~2GB RAM)
"""

import argparse
import os
import sys
import yaml
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="BioASQ RAG Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # General arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Override embedding model from config"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of passages to retrieve"
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run and evaluate retrieval (skip generation)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--generation-model",
        type=str,
        default=None,
        help="Local LLM for generation (default: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation step (alias for --retrieval-only)"
    )
    parser.add_argument(
        "--num-passages",
        type=int,
        default=None,
        help="Number of passages to use for generation"
    )
    
    # Other arguments
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save predictions to file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
    
    # Get settings (CLI args override config)
    embedding_model = args.embedding_model or config.get("retrieval", {}).get(
        "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    generation_model = args.generation_model or config.get("generation", {}).get(
        "model_name", "Qwen/Qwen2.5-0.5B-Instruct"
    )
    top_k = args.top_k or config.get("retrieval", {}).get("top_k", 5)
    num_passages = args.num_passages or config.get("generation", {}).get(
        "num_passages_for_generation", 3
    )
    k_values = config.get("evaluation", {}).get("k_values", [1, 3, 5, 10])
    
    skip_generation = args.retrieval_only or args.skip_generation
    
    # Print settings
    print("\n" + "=" * 60)
    print("BioASQ RAG Baseline".center(60))
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Embedding model:  {embedding_model}")
    print(f"  Generation model: {generation_model}")
    print(f"  Top-k retrieval:  {top_k}")
    print(f"  Passages for gen: {num_passages}")
    print(f"  Max samples:      {args.max_samples or 'all'}")
    print(f"  Skip generation:  {skip_generation}")
    print("=" * 60 + "\n")
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        generation_model=generation_model,
        top_k=top_k,
        num_passages_for_generation=num_passages,
        k_values=k_values
    )
    
    # Load data
    logger.info("Loading dataset...")
    pipeline.load_data(
        split=args.split,
        max_samples=args.max_samples
    )
    
    # Index corpus
    logger.info("Building retrieval index...")
    pipeline.index_corpus(batch_size=args.batch_size)
    
    # Run pipeline
    logger.info("Running evaluation pipeline...")
    results = pipeline.run_full_pipeline(
        skip_generation=skip_generation,
        output_dir=args.output_dir,
        save_predictions=not args.no_save
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE".center(60))
    print("=" * 60)
    
    if "retrieval" in results:
        print("\nRetrieval Performance:")
        for metric, value in results["retrieval"].items():
            print(f"  {metric:<20} {value:.4f}")
    
    if "generation" in results:
        print("\nGeneration Performance:")
        for metric, value in results["generation"].items():
            print(f"  {metric:<20} {value:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
