"""
Data loader for the BioASQ dataset.

Loads and preprocesses the rag-datasets/rag-mini-bioasq dataset from Hugging Face.

The dataset has two configs:
- 'text-corpus': Contains the passage corpus with passage_id and passage
- 'question-answer-passages': Contains questions with answers and relevant_passage_ids
"""

from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioASQDataLoader:
    """Loads and manages the BioASQ dataset for RAG evaluation."""
    
    def __init__(
        self,
        dataset_name: str = "rag-datasets/rag-mini-bioasq",
        split: str = "test",
        max_samples: Optional[int] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use for questions
            max_samples: Maximum number of samples to load (None for all)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        
        self.questions_data = None
        self.corpus = None
        self.corpus_id_to_idx = None
        
    def load(self) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Load the dataset and return questions and corpus.
        
        Returns:
            Tuple of (questions_list, corpus_dict)
            - questions_list: List of dicts with 'question', 'answer', 'relevant_passage_ids'
            - corpus_dict: Dict mapping passage_id to passage_text
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        # Load the corpus from 'text-corpus' config
        logger.info("Loading corpus from 'text-corpus' config...")
        corpus_dataset = load_dataset(self.dataset_name, "text-corpus")
        
        # Load questions from 'question-answer-passages' config
        logger.info("Loading questions from 'question-answer-passages' config...")
        qa_dataset = load_dataset(self.dataset_name, "question-answer-passages")
        
        # Get the questions split
        questions_split = qa_dataset[self.split]
        
        # Get the corpus split (text-corpus only has 'passages' split)
        corpus_split = corpus_dataset["passages"]
        
        # Build corpus dictionary
        logger.info("Building corpus dictionary...")
        self.corpus = {}
        self.corpus_id_to_idx = {}
        
        for idx, item in enumerate(corpus_split):
            # The corpus uses 'id' and 'passage' fields
            # Ensure passage_id is an integer for consistent matching
            passage_id = int(item["id"]) if isinstance(item["id"], str) else item["id"]
            passage_text = item["passage"]
            self.corpus[passage_id] = passage_text
            self.corpus_id_to_idx[passage_id] = idx
        
        logger.info(f"Loaded {len(self.corpus)} passages")
        
        # Process questions
        logger.info("Processing questions...")
        self.questions_data = []
        
        for item in questions_split:
            # Parse relevant_passage_ids - may be string or list
            relevant_ids = item["relevant_passage_ids"]
            
            # If it's a string, parse it (could be comma-separated or JSON-like)
            if isinstance(relevant_ids, str):
                # Try to parse as comma-separated integers
                relevant_ids = [
                    int(x.strip()) for x in relevant_ids.split(",") 
                    if x.strip().isdigit()
                ]
            elif isinstance(relevant_ids, list):
                # Convert to integers if they're strings
                relevant_ids = [
                    int(x) if isinstance(x, str) else x 
                    for x in relevant_ids
                ]
            
            question_entry = {
                "question": item["question"],
                "answer": item["answer"],
                "relevant_passage_ids": relevant_ids
            }
            self.questions_data.append(question_entry)
            
            if self.max_samples and len(self.questions_data) >= self.max_samples:
                break
        
        logger.info(f"Loaded {len(self.questions_data)} questions")
        
        return self.questions_data, self.corpus
    
    def get_corpus_texts_and_ids(self) -> Tuple[List[str], List[str]]:
        """
        Get corpus as parallel lists of texts and IDs.
        
        Returns:
            Tuple of (texts_list, ids_list)
        """
        if self.corpus is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        ids = list(self.corpus.keys())
        texts = [self.corpus[pid] for pid in ids]
        return texts, ids
    
    def get_questions(self) -> List[str]:
        """Get list of question texts."""
        if self.questions_data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return [q["question"] for q in self.questions_data]
    
    def get_ground_truth_answers(self) -> List[str]:
        """Get list of ground truth answers."""
        if self.questions_data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return [q["answer"] for q in self.questions_data]
    
    def get_relevant_passage_ids(self) -> List[List[str]]:
        """Get list of relevant passage IDs for each question."""
        if self.questions_data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return [q["relevant_passage_ids"] for q in self.questions_data]
    
    def get_passage_text(self, passage_id: str) -> Optional[str]:
        """Get the text of a specific passage by ID."""
        if self.corpus is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return self.corpus.get(passage_id)


def main():
    """Test the data loader."""
    loader = BioASQDataLoader(max_samples=5)
    questions, corpus = loader.load()
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total passages in corpus: {len(corpus)}")
    print(f"Total questions loaded: {len(questions)}")
    
    print("\n" + "="*60)
    print("SAMPLE QUESTION")
    print("="*60)
    sample = questions[0]
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Relevant passage IDs: {sample['relevant_passage_ids'][:3]}...")
    
    # Show a relevant passage
    if sample['relevant_passage_ids']:
        pid = sample['relevant_passage_ids'][0]
        passage_text = corpus.get(pid, "NOT FOUND")
        print(f"\nFirst relevant passage ({pid}):")
        print(f"{passage_text[:500] if passage_text != 'NOT FOUND' else passage_text}...")


if __name__ == "__main__":
    main()
