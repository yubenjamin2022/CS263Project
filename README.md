# BioASQ RAG Baseline

A baseline implementation for Retrieval-Augmented Generation (RAG) on the BioASQ medical QA dataset.

## Overview

This repository implements a dense retrieval + generation pipeline for medical question answering:
1. **Dense Retrieval**: Uses sentence-transformers to encode queries and passages
2. **Retrieval Evaluation**: Measures Recall@k, Precision@k, and MRR
3. **Answer Generation**: Uses a small LLM (Qwen 1.5B) to generate answers from retrieved passages
4. **Generation Evaluation**: Measures BERTScore and BLEU against ground truth answers

## Project Structure

```
bioasq-rag-baseline/
├── src/
│   ├── data_loader.py      # Load and preprocess BioASQ dataset
│   ├── retriever.py        # Dense retrieval implementation
│   ├── generator.py        # LLM-based answer generation
│   ├── evaluate.py         # Evaluation metrics
│   └── pipeline.py         # End-to-end RAG pipeline
├── configs/
│   └── config.yaml         # Configuration file
├── run_baseline.py         # Main entry point
└── requirements.txt        # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the full pipeline
```bash
python run_baseline.py
```

### Run individual components
```bash
# Evaluate retrieval only
python run_baseline.py --retrieval-only

# Skip generation (useful for quick retrieval experiments)
python run_baseline.py --skip-generation

# Use a different number of retrieved passages
python run_baseline.py --top-k 10
```

## Configuration

Edit `configs/config.yaml` to modify:
- Embedding model for retrieval
- Generation model
- Number of passages to retrieve
- Evaluation settings

## Dataset

Uses the `rag-datasets/rag-mini-bioasq` dataset from Hugging Face, which contains:
- Medical questions
- Relevant passage IDs (ground truth for retrieval)
- Ground truth answers

## Metrics

### Retrieval Metrics
- **Recall@k**: Fraction of relevant passages retrieved in top-k
- **Precision@k**: Fraction of retrieved passages that are relevant
- **MRR**: Mean Reciprocal Rank of first relevant passage

### Generation Metrics
- **BERTScore**: Semantic similarity using BERT embeddings
- **BLEU**: N-gram overlap score

## Team

CS263 Team 11
- Ravisri Valluri
- Zhao Xu
- Benjamin Yu
