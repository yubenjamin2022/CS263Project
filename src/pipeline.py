"""
End-to-end RAG pipeline combining retrieval and generation.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from data_loader import BioASQDataLoader
from retriever import DenseRetriever
from generator import AnswerGenerator
from evaluate import RetrievalEvaluator, GenerationEvaluator, format_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for BioASQ dataset."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generation_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        top_k: int = 5,
        num_passages_for_generation: int = 3,
        k_values: List[int] = [1, 3, 5, 10],
        device: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Model for dense retrieval
            generation_model: Local LLM for answer generation
            top_k: Number of passages to retrieve
            num_passages_for_generation: Number of passages to use for generation
            k_values: K values for retrieval evaluation
            device: Device to use for models
        """
        self.top_k = top_k
        self.num_passages_for_generation = num_passages_for_generation
        self.k_values = k_values
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        self.data_loader = None
        self.retriever = DenseRetriever(
            model_name=embedding_model,
            device=device
        )
        self.generator = None
        self.generation_model = generation_model
        self.device = device
        
        # Evaluators
        self.retrieval_evaluator = RetrievalEvaluator(k_values=k_values)
        self.generation_evaluator = None
        
        # Storage
        self.corpus = None
        self.questions_data = None
    
    def load_data(
        self,
        dataset_name: str = "rag-datasets/rag-mini-bioasq",
        split: str = "test",
        max_samples: Optional[int] = None
    ) -> None:
        """Load the dataset."""
        logger.info(f"Loading dataset: {dataset_name}")
        
        self.data_loader = BioASQDataLoader(
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples
        )
        
        self.questions_data, self.corpus = self.data_loader.load()
    
    def index_corpus(self, batch_size: int = 32) -> None:
        """Build the retrieval index."""
        if self.corpus is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        corpus_texts, corpus_ids = self.data_loader.get_corpus_texts_and_ids()
        self.retriever.index_corpus(corpus_texts, corpus_ids, batch_size=batch_size)
    
    def run_retrieval(
        self,
        batch_size: int = 32
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Run retrieval for all questions.
        
        Returns:
            Tuple of (retrieved_ids, scores)
        """
        questions = self.data_loader.get_questions()
        
        retrieved_ids, scores = self.retriever.retrieve(
            questions,
            top_k=self.top_k,
            batch_size=batch_size
        )
        
        return retrieved_ids, scores
    
    def evaluate_retrieval(
        self,
        retrieved_ids: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_ids: List of retrieved passage ID lists
            
        Returns:
            Dictionary of retrieval metrics
        """
        relevant_ids = self.data_loader.get_relevant_passage_ids()
        
        results = self.retrieval_evaluator.evaluate(retrieved_ids, relevant_ids)
        
        return results
    
    def run_generation(
        self,
        retrieved_ids: List[List[str]],
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate answers using retrieved passages.
        
        Args:
            retrieved_ids: List of retrieved passage ID lists
            show_progress: Whether to show progress bar
            
        Returns:
            List of generated answers
        """
        # Initialize generator lazily
        if self.generator is None:
            self.generator = AnswerGenerator(
                model_name=self.generation_model,
                device=self.device
            )
        
        questions = self.data_loader.get_questions()
        
        # Get passage texts for generation
        passages_list = []
        for pids in retrieved_ids:
            passages = []
            for pid in pids[:self.num_passages_for_generation]:
                text = self.corpus.get(pid, "")
                if text:
                    passages.append(text)
            passages_list.append(passages)
        
        # Generate answers
        answers = self.generator.generate_batch(
            questions,
            passages_list,
            show_progress=show_progress
        )
        
        return answers
    
    def evaluate_generation(
        self,
        predictions: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate generation quality.
        
        Args:
            predictions: List of generated answers
            
        Returns:
            Dictionary of generation metrics
        """
        # Initialize evaluator lazily
        if self.generation_evaluator is None:
            self.generation_evaluator = GenerationEvaluator(device=self.device)
        
        references = self.data_loader.get_ground_truth_answers()
        
        results = self.generation_evaluator.evaluate(predictions, references)
        
        return results
    
    def run_full_pipeline(
        self,
        skip_generation: bool = False,
        output_dir: str = "outputs",
        save_predictions: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Run the full RAG pipeline with evaluation.
        
        Args:
            skip_generation: Whether to skip generation step
            output_dir: Directory to save outputs
            save_predictions: Whether to save predictions to file
            
        Returns:
            Dictionary with retrieval and generation metrics
        """
        results = {}
        
        # Run retrieval
        logger.info("Running retrieval...")
        retrieved_ids, scores = self.run_retrieval()
        
        # Evaluate retrieval
        logger.info("Evaluating retrieval...")
        retrieval_metrics = self.evaluate_retrieval(retrieved_ids)
        results["retrieval"] = retrieval_metrics
        
        print("\n" + format_results(retrieval_metrics, "Retrieval Metrics"))
        
        if not skip_generation:
            # Run generation
            logger.info("Running generation...")
            predictions = self.run_generation(retrieved_ids)
            
            # Evaluate generation
            logger.info("Evaluating generation...")
            generation_metrics = self.evaluate_generation(predictions)
            results["generation"] = generation_metrics
            
            print("\n" + format_results(generation_metrics, "Generation Metrics"))
            
            # Save predictions
            if save_predictions:
                os.makedirs(output_dir, exist_ok=True)
                self._save_predictions(
                    retrieved_ids,
                    predictions,
                    output_dir
                )
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return results
    
    def _save_predictions(
        self,
        retrieved_ids: List[List[str]],
        predictions: List[str],
        output_dir: str
    ) -> None:
        """Save predictions to file."""
        questions = self.data_loader.get_questions()
        references = self.data_loader.get_ground_truth_answers()
        relevant_ids = self.data_loader.get_relevant_passage_ids()
        
        output_data = []
        for i, (q, pred, ref, ret, rel) in enumerate(zip(
            questions, predictions, references, retrieved_ids, relevant_ids
        )):
            output_data.append({
                "id": i,
                "question": q,
                "prediction": pred,
                "reference": ref,
                "retrieved_ids": ret[:self.num_passages_for_generation],
                "relevant_ids": rel,
                "retrieved_passages": [
                    self.corpus.get(pid, "")[:500]
                    for pid in ret[:self.num_passages_for_generation]
                ]
            })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"predictions_{timestamp}.json")
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Predictions saved to {output_path}")


def main():
    """Test the pipeline."""
    # Initialize pipeline
    pipeline = RAGPipeline(
        top_k=5,
        num_passages_for_generation=3
    )
    
    # Load data (small sample for testing)
    pipeline.load_data(max_samples=10)
    
    # Index corpus
    pipeline.index_corpus()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        skip_generation=True,  # Skip for quick test
        output_dir="outputs"
    )
    
    print("\nPipeline test complete!")


if __name__ == "__main__":
    main()
