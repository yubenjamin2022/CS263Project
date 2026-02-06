"""
Dense retrieval implementation using sentence-transformers and FAISS.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using numpy for similarity search.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retriever using sentence-transformers for encoding."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_faiss: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the dense retriever.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
            use_faiss: Whether to use FAISS for efficient similarity search
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Index storage
        self.corpus_embeddings = None
        self.corpus_ids = None
        self.faiss_index = None
        
    def index_corpus(
        self,
        corpus_texts: List[str],
        corpus_ids: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> None:
        """
        Build the retrieval index from corpus texts.
        
        Args:
            corpus_texts: List of passage texts
            corpus_ids: List of passage IDs (parallel to texts)
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        """
        logger.info(f"Indexing {len(corpus_texts)} passages...")
        
        self.corpus_ids = corpus_ids
        
        # Encode corpus in batches
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        logger.info(f"Corpus embeddings shape: {self.corpus_embeddings.shape}")
        
        # Build FAISS index if enabled
        if self.use_faiss:
            self._build_faiss_index()
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for efficient similarity search."""
        logger.info("Building FAISS index...")
        
        dim = self.corpus_embeddings.shape[1]
        
        # Use inner product (equivalent to cosine sim for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.corpus_embeddings.astype(np.float32))
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Retrieve top-k passages for each query.
        
        Args:
            queries: List of query strings
            top_k: Number of passages to retrieve per query
            batch_size: Batch size for encoding queries
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (retrieved_ids, scores)
            - retrieved_ids: List of lists of passage IDs
            - scores: List of lists of similarity scores
        """
        if self.corpus_embeddings is None:
            raise ValueError("Corpus not indexed. Call index_corpus() first.")
        
        logger.info(f"Retrieving passages for {len(queries)} queries...")
        
        # Encode queries
        query_embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        if self.use_faiss and self.faiss_index is not None:
            # Use FAISS for search
            scores, indices = self.faiss_index.search(
                query_embeddings.astype(np.float32),
                top_k
            )
        else:
            # Use numpy for search
            scores, indices = self._numpy_search(query_embeddings, top_k)
        
        # Convert indices to passage IDs
        retrieved_ids = []
        for query_indices in indices:
            query_ids = [self.corpus_ids[idx] for idx in query_indices]
            retrieved_ids.append(query_ids)
        
        return retrieved_ids, scores.tolist()
    
    def _numpy_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback numpy-based similarity search."""
        # Compute all similarities
        similarities = np.dot(query_embeddings, self.corpus_embeddings.T)
        
        # Get top-k indices and scores
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(similarities, top_indices, axis=1)
        
        return top_scores, top_indices
    
    def retrieve_single(
        self,
        query: str,
        top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-k passages for a single query.
        
        Args:
            query: Query string
            top_k: Number of passages to retrieve
            
        Returns:
            Tuple of (passage_ids, scores)
        """
        ids, scores = self.retrieve([query], top_k=top_k, show_progress=False)
        return ids[0], scores[0]
    
    def save_index(self, path: str) -> None:
        """Save the index to disk."""
        import pickle
        
        data = {
            "corpus_embeddings": self.corpus_embeddings,
            "corpus_ids": self.corpus_ids,
            "model_name": self.model_name
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """Load a saved index from disk."""
        import pickle
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.corpus_embeddings = data["corpus_embeddings"]
        self.corpus_ids = data["corpus_ids"]
        
        if self.use_faiss:
            self._build_faiss_index()
        
        logger.info(f"Index loaded from {path}")


def main():
    """Test the retriever with sample data."""
    from data_loader import BioASQDataLoader
    
    # Load data
    loader = BioASQDataLoader(max_samples=10)
    questions, corpus = loader.load()
    
    corpus_texts, corpus_ids = loader.get_corpus_texts_and_ids()
    
    # Initialize retriever
    retriever = DenseRetriever()
    
    # Index corpus (use subset for testing)
    retriever.index_corpus(corpus_texts[:1000], corpus_ids[:1000])
    
    # Test retrieval
    test_query = questions[0]["question"]
    retrieved_ids, scores = retriever.retrieve_single(test_query, top_k=5)
    
    print("\n" + "="*60)
    print("RETRIEVAL TEST")
    print("="*60)
    print(f"Query: {test_query}")
    print(f"\nGround truth passage IDs: {questions[0]['relevant_passage_ids'][:3]}")
    print(f"\nRetrieved passages:")
    for i, (pid, score) in enumerate(zip(retrieved_ids, scores)):
        print(f"  {i+1}. [{score:.4f}] {pid}")
        print(f"     {corpus.get(pid, 'N/A')[:100]}...")


if __name__ == "__main__":
    main()
