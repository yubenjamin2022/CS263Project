# Vocab Expansion Pipeline (BioASQ)

## Quick Start: `plus100` Full-Finetune Sequence

This is the exact command sequence used for the `+100` full-finetune run and final evaluation:

```bash
pip install tokenmonster
git clone https://github.com/PythonNut/superbpe
cd /home/ravisri/code/courses/CS263Project/vocab_expansion
conda activate python310
```

```bash
python scripts/build_added_vocab.py \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --candidate-file AutoPhrase/models/BioASQ/all_ngrams_score_ge_0.8_by_freq.txt \
  --growth-ratio 0.0033 \
  --output-file AutoPhrase/models/BioASQ/added_ngrams_100.tsv
```

```bash
python scripts/expand_model_embeddings.py \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_100.tsv \
  --output-dir checkpoints/all-MiniLM-L6-v2-bioasq-expanded-plus100
```

```bash
python scripts/find_added_token_coverage.py \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_100.tsv \
  --dataset-name ms_marco \
  --dataset-config v1.1 \
  --split train \
  --streaming \
  --per-token-limit 100 \
  --max-dataset-examples 2000000 \
  --output-jsonl data/msmarco_added_token_coverage_100.jsonl \
  --output-summary-tsv data/msmarco_added_token_coverage_p100_summary.tsv
```

```bash
WANDB_DISABLED=true TRANSFORMERS_NO_TF=1 ST_FORCE_SINGLE_GPU=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/finetune_mini_bioasq.py \
  --model-name-or-path checkpoints/all-MiniLM-L6-v2-bioasq-expanded-plus100 \
  --coverage-jsonl data/msmarco_added_token_coverage_100.jsonl \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_100.tsv \
  --added-token-ids-file checkpoints/all-MiniLM-L6-v2-bioasq-expanded-plus100/added_tokens_init.tsv \
  --output-dir checkpoints/all-MiniLM-L6-v2-bioasq-finetuned-plus100 \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 2e-5
```

```bash
python scripts/eval_retrieval_bioasq.py \
  --model-name-or-path checkpoints/all-MiniLM-L6-v2-bioasq-finetuned-plus100 \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_100.tsv \
  --output-json outputs/eval_plus100_merge.json
```

This folder contains an end-to-end pipeline for:

1. AutoPhrase phrase mining on `bioasq-corpus.txt`
2. Expanding `sentence-transformers/all-MiniLM-L6-v2` vocabulary
3. Expanding embedding matrix for added tokens
4. Fine-tuning
5. Retrieval evaluation on `rag-datasets/rag-mini-bioasq`

Run commands from this directory:

```bash
cd /home/ravisri/code/courses/CS263Project/vocab_expansion
```

## 0) Environment

Use your existing env (example):

```bash
conda activate python310
```

## 1) AutoPhrase Extraction

Train AutoPhrase model on BioASQ corpus:

```bash
cd AutoPhrase
RAW_TRAIN=../bioasq-corpus.txt MODEL=models/BioASQ THREAD=10 ./auto_phrase.sh
cd ..
```

AutoPhrase outputs phrase quality lists under:

- `AutoPhrase/models/BioASQ/AutoPhrase.txt`
- `AutoPhrase/models/BioASQ/AutoPhrase_multi-words.txt`
- `AutoPhrase/models/BioASQ/AutoPhrase_single-word.txt`

If needed, generate thresholded phrase list by score/frequency from existing `AutoPhrase_with_freq.txt`:

```bash
awk -F'\t' 'NF>=3 && ($1+0)>=0.8 {print $1"\t"$2"\t"$3}' \
  AutoPhrase/models/BioASQ/AutoPhrase_with_freq.txt \
  | sort -t$'\t' -k2,2nr -k1,1nr \
  > AutoPhrase/models/BioASQ/all_ngrams_score_ge_0.8_by_freq.txt
```

## 2) Build Added Vocab (10%)

Select new tokens from AutoPhrase candidates until vocab grows by 10%:

```bash
python scripts/build_added_vocab.py \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --candidate-file AutoPhrase/models/BioASQ/all_ngrams_score_ge_0.8_by_freq.txt \
  --growth-ratio 0.10 \
  --output-file AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv
```

## 3) Expand Model Embeddings

Adds tokens to tokenizer and expands embedding matrix. New token rows are initialized
as mean of original tokenization embeddings for each phrase.

```bash
python scripts/expand_model_embeddings.py \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv \
  --output-dir checkpoints/all-MiniLM-L6-v2-bioasq-expanded
```

## 4) Build Coverage-Oriented Training Pairs (Optional but Recommended)

Find query/positive pairs from MS MARCO where positives contain added phrases:

```bash
python scripts/find_added_token_coverage.py \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv \
  --dataset-name ms_marco \
  --dataset-config v1.1 \
  --split train \
  --streaming \
  --per-token-limit 10 \
  --max-dataset-examples 2000000 \
  --output-jsonl data/msmarco_added_token_coverage_p10.jsonl \
  --output-summary-tsv data/msmarco_added_token_coverage_p10_summary.tsv
```

## 5) Fine-Tune

### 5.1 Train added-token rows only

```bash
WANDB_DISABLED=true TRANSFORMERS_NO_TF=1 ST_FORCE_SINGLE_GPU=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/finetune_mini_bioasq.py \
  --model-name-or-path checkpoints/all-MiniLM-L6-v2-bioasq-expanded \
  --coverage-jsonl data/msmarco_added_token_coverage_p10.jsonl \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv \
  --added-token-ids-file checkpoints/all-MiniLM-L6-v2-bioasq-expanded/added_tokens_init.tsv \
  --train-added-embeddings-only \
  --output-dir checkpoints/all-MiniLM-L6-v2-bioasq-finetuned-added-only \
  --epochs 2 \
  --batch-size 64 \
  --learning-rate 2e-5
```

### 5.2 Alternative: freeze transformer, train full embedding matrix

```bash
WANDB_DISABLED=true TRANSFORMERS_NO_TF=1 ST_FORCE_SINGLE_GPU=1 CUDA_VISIBLE_DEVICES=0 \
python scripts/finetune_mini_bioasq.py \
  --model-name-or-path checkpoints/all-MiniLM-L6-v2-bioasq-expanded \
  --coverage-jsonl data/msmarco_added_token_coverage_p10.jsonl \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv \
  --freeze-transformer \
  --output-dir checkpoints/all-MiniLM-L6-v2-bioasq-finetuned-coverage \
  --epochs 2 \
  --batch-size 64 \
  --learning-rate 2e-5
```

## 6) Retrieval Evaluation

### 6.1 Base model

```bash
python scripts/eval_retrieval_bioasq.py \
  --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 \
  --output-json outputs/eval_base.json
```

### 6.2 Fine-tuned model (with phrase merge)

```bash
python scripts/eval_retrieval_bioasq.py \
  --model-name-or-path checkpoints/all-MiniLM-L6-v2-bioasq-finetuned-added-only \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv \
  --output-json outputs/eval_added_only_merge.json
```

## 7) Analyze Added Token Frequency in Test Questions

```bash
python scripts/analyze_added_token_question_freq.py \
  --added-ngrams-file AutoPhrase/models/BioASQ/added_ngrams_10pct.tsv \
  --dataset-name rag-datasets/rag-mini-bioasq \
  --qa-config question-answer-passages \
  --split test \
  --output-tsv outputs/added_token_question_frequency.tsv \
  --output-json outputs/added_token_question_frequency_summary.json
```
