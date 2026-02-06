"""
Evaluation metrics for retrieval and generation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluates retrieval performance with standard IR metrics."""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Initialize the evaluator.
        
        Args:
            k_values: List of k values for Recall@k and Precision@k
        """
        self.k_values = k_values
    
    def compute_recall_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
        k: int
    ) -> float:
        """
        Compute Recall@k.
        
        Recall@k = (# relevant docs in top-k) / (# total relevant docs)
        
        Args:
            retrieved_ids: List of retrieved passage ID lists
            relevant_ids: List of ground truth relevant passage ID lists
            k: Number of top results to consider
            
        Returns:
            Mean Recall@k across all queries
        """
        recalls = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if not relevant:
                continue
                
            top_k_retrieved = set(retrieved[:k])
            relevant_set = set(relevant)
            
            hits = len(top_k_retrieved & relevant_set)
            recall = hits / len(relevant_set)
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def compute_precision_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
        k: int
    ) -> float:
        """
        Compute Precision@k.
        
        Precision@k = (# relevant docs in top-k) / k
        
        Args:
            retrieved_ids: List of retrieved passage ID lists
            relevant_ids: List of ground truth relevant passage ID lists
            k: Number of top results to consider
            
        Returns:
            Mean Precision@k across all queries
        """
        precisions = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            top_k_retrieved = set(retrieved[:k])
            relevant_set = set(relevant)
            
            hits = len(top_k_retrieved & relevant_set)
            precision = hits / k
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def compute_mrr(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]]
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).
        
        MRR = mean(1 / rank of first relevant doc)
        
        Args:
            retrieved_ids: List of retrieved passage ID lists
            relevant_ids: List of ground truth relevant passage ID lists
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            
            reciprocal_ranks.append(rr)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def compute_hit_rate_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
        k: int
    ) -> float:
        """
        Compute Hit Rate@k (fraction of queries with at least one relevant doc in top-k).
        
        Args:
            retrieved_ids: List of retrieved passage ID lists
            relevant_ids: List of ground truth relevant passage ID lists
            k: Number of top results to consider
            
        Returns:
            Hit Rate@k
        """
        hits = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if not relevant:
                continue
                
            top_k_retrieved = set(retrieved[:k])
            relevant_set = set(relevant)
            
            has_hit = len(top_k_retrieved & relevant_set) > 0
            hits.append(1.0 if has_hit else 0.0)
        
        return np.mean(hits) if hits else 0.0
    
    def evaluate(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute all retrieval metrics.
        
        Args:
            retrieved_ids: List of retrieved passage ID lists
            relevant_ids: List of ground truth relevant passage ID lists
            
        Returns:
            Dictionary of metric names to values
        """
        results = {}
        
        # MRR
        results["MRR"] = self.compute_mrr(retrieved_ids, relevant_ids)
        
        # Recall@k and Precision@k for each k
        for k in self.k_values:
            results[f"Recall@{k}"] = self.compute_recall_at_k(
                retrieved_ids, relevant_ids, k
            )
            results[f"Precision@{k}"] = self.compute_precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            results[f"HitRate@{k}"] = self.compute_hit_rate_at_k(
                retrieved_ids, relevant_ids, k
            )
        
        return results


class GenerationEvaluator:
    """Evaluates generation quality using BERTScore and BLEU."""
    
    def __init__(
        self,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        device: Optional[str] = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            bertscore_model: Model to use for BERTScore
            device: Device for computation
        """
        self.bertscore_model = bertscore_model
        self.device = device
        
        self._bertscore_scorer = None
        self._bleu_scorer = None
    
    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str],
        max_length: int = 512  # Truncate to avoid overflow
    ) -> Dict[str, float]:
        """
        Compute BERTScore.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            max_length: Maximum character length (truncate longer texts)
            
        Returns:
            Dictionary with precision, recall, f1 scores
        """
        from bert_score import score
        
        logger.info("Computing BERTScore...")
        
        # Filter out failed predictions (error messages) and truncate long texts
        valid_pairs = []
        for pred, ref in zip(predictions, references):
            # Skip error messages
            if pred.startswith("[") or pred.startswith("Error"):
                continue
            
            # Truncate long texts to avoid overflow
            pred_truncated = pred[:max_length] if len(pred) > max_length else pred
            ref_truncated = ref[:max_length] if len(ref) > max_length else ref
            
            valid_pairs.append((pred_truncated, ref_truncated))
        
        if not valid_pairs:
            logger.warning("No valid predictions to evaluate!")
            return {
                "BERTScore_Precision": 0.0,
                "BERTScore_Recall": 0.0,
                "BERTScore_F1": 0.0,
                "BERTScore_ValidSamples": 0
            }
        
        valid_preds, valid_refs = zip(*valid_pairs)
        
        logger.info(f"Evaluating {len(valid_preds)}/{len(predictions)} valid predictions")
        
        P, R, F1 = score(
            list(valid_preds),
            list(valid_refs),
            model_type=self.bertscore_model,
            device=self.device,
            verbose=True
        )
        
        return {
            "BERTScore_Precision": P.mean().item(),
            "BERTScore_Recall": R.mean().item(),
            "BERTScore_F1": F1.mean().item(),
            "BERTScore_ValidSamples": len(valid_preds)
        }
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
        max_length: int = 512  # Truncate to avoid issues
    ) -> Dict[str, float]:
        """
        Compute BLEU score using sacrebleu.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            max_length: Maximum character length (truncate longer texts)
            
        Returns:
            Dictionary with BLEU scores
        """
        from sacrebleu.metrics import BLEU
        
        logger.info("Computing BLEU score...")
        
        # Filter out failed predictions and truncate long texts
        valid_pairs = []
        for pred, ref in zip(predictions, references):
            # Skip error messages
            if pred.startswith("[") or pred.startswith("Error"):
                continue
            
            # Truncate long texts
            pred_truncated = pred[:max_length] if len(pred) > max_length else pred
            ref_truncated = ref[:max_length] if len(ref) > max_length else ref
            
            valid_pairs.append((pred_truncated, ref_truncated))
        
        if not valid_pairs:
            logger.warning("No valid predictions to evaluate!")
            return {
                "BLEU": 0.0,
                "BLEU_1": 0.0,
                "BLEU_2": 0.0,
                "BLEU_3": 0.0,
                "BLEU_4": 0.0,
                "BLEU_ValidSamples": 0
            }
        
        valid_preds, valid_refs = zip(*valid_pairs)
        
        bleu = BLEU()
        
        # sacrebleu expects references as list of lists
        refs = [[ref] for ref in valid_refs]
        
        # Compute corpus-level BLEU
        result = bleu.corpus_score(list(valid_preds), list(zip(*refs)))
        
        return {
            "BLEU": result.score,
            "BLEU_1": result.precisions[0] if result.precisions else 0.0,
            "BLEU_2": result.precisions[1] if len(result.precisions) > 1 else 0.0,
            "BLEU_3": result.precisions[2] if len(result.precisions) > 2 else 0.0,
            "BLEU_4": result.precisions[3] if len(result.precisions) > 3 else 0.0,
            "BLEU_ValidSamples": len(valid_preds)
        }
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute all generation metrics.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            
        Returns:
            Dictionary of metric names to values
        """
        results = {}
        
        # BERTScore
        bertscore_results = self.compute_bertscore(predictions, references)
        results.update(bertscore_results)
        
        # BLEU
        bleu_results = self.compute_bleu(predictions, references)
        results.update(bleu_results)
        
        return results


def format_results(results: Dict[str, float], title: str = "Results") -> str:
    """
    Format results dictionary as a nice table string.
    
    Args:
        results: Dictionary of metric names to values
        title: Title for the table
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 50,
        title.center(50),
        "=" * 50,
    ]
    
    for metric, value in results.items():
        lines.append(f"  {metric:<30} {value:.4f}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def main():
    """Test the evaluators."""
    # Test retrieval evaluator
    retrieval_eval = RetrievalEvaluator(k_values=[1, 3, 5])
    
    # Sample data
    retrieved = [
        ["doc1", "doc2", "doc3", "doc4", "doc5"],
        ["doc6", "doc1", "doc7", "doc8", "doc9"],
        ["doc10", "doc11", "doc3", "doc12", "doc13"]
    ]
    relevant = [
        ["doc1", "doc3"],
        ["doc1", "doc6"],
        ["doc3", "doc14"]
    ]
    
    retrieval_results = retrieval_eval.evaluate(retrieved, relevant)
    print(format_results(retrieval_results, "Retrieval Metrics"))
    
    # Test generation evaluator
    gen_eval = GenerationEvaluator()
    
    predictions = [
        "Hemoglobin carries oxygen in the blood.",
        "The heart pumps blood through the body.",
    ]
    references = [
        "Hemoglobin is responsible for transporting oxygen in red blood cells.",
        "The heart is responsible for pumping blood throughout the circulatory system.",
    ]
    
    gen_results = gen_eval.evaluate(predictions, references)
    print(format_results(gen_results, "Generation Metrics"))


if __name__ == "__main__":
    main()
