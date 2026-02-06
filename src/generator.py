"""
Answer generation using a local LLM (Qwen2.5-0.5B-Instruct by default).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers using retrieved passages and a local LLM."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize the answer generator.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "Qwen/Qwen2.5-0.5B-Instruct" (500M params, ~1GB RAM)
                - "Qwen/Qwen2.5-1.5B-Instruct" (1.5B params, ~3GB RAM)
                - "Qwen/Qwen2.5-3B-Instruct" (3B params, ~6GB RAM)
                - "microsoft/Phi-3-mini-4k-instruct" (3.8B params)
                - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B params)
            device: Device to use ('cuda', 'cpu', or None for auto)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            load_in_8bit: Load model in 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit quantization (requires bitsandbytes)
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading generation model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        model_kwargs = {}
        
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
            logger.info("Loading model in 4-bit quantization")
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
            logger.info("Loading model in 8-bit quantization")
        elif self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded successfully")
    
    def _format_prompt(
        self,
        question: str,
        passages: List[str],
        passage_ids: Optional[List[str]] = None
    ) -> str:
        """Format the prompt with question and retrieved passages."""
        # Build context from passages
        context_parts = []
        for i, passage in enumerate(passages):
            context_parts.append(f"[Passage {i+1}] {passage}")
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""You are a helpful medical assistant. Answer the question based only on the provided context passages. If the answer cannot be found in the passages, say "I cannot find the answer in the provided context."

Be concise and directly answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate(
        self,
        question: str,
        passages: List[str],
        passage_ids: Optional[List[str]] = None
    ) -> str:
        """
        Generate an answer for a single question.
        
        Args:
            question: The user's question
            passages: List of retrieved passage texts
            passage_ids: Optional passage IDs
            
        Returns:
            Generated answer string
        """
        prompt = self._format_prompt(question, passages, passage_ids)
        
        # Format for chat model
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p if self.do_sample else 1.0,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode, removing the input prompt
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer.strip()
    
    def generate_batch(
        self,
        questions: List[str],
        passages_list: List[List[str]],
        passage_ids_list: Optional[List[List[str]]] = None,
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate answers for multiple questions.
        
        Args:
            questions: List of questions
            passages_list: List of passage lists (one per question)
            passage_ids_list: Optional list of passage ID lists
            show_progress: Whether to show progress
            
        Returns:
            List of generated answers
        """
        from tqdm import tqdm
        
        answers = []
        items = list(zip(questions, passages_list))
        if passage_ids_list:
            items = [(q, p, pid) for (q, p), pid in zip(items, passage_ids_list)]
        
        if show_progress:
            items = tqdm(items, desc=f"Generating answers ({self.model_name.split('/')[-1]})")
        
        for item in items:
            if passage_ids_list:
                question, passages, pids = item
            else:
                question, passages = item
                pids = None
            
            try:
                answer = self.generate(question, passages, pids)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                answer = "[Generation failed]"
            
            answers.append(answer)
        
        return answers


def main():
    """Test the generator."""
    print("=" * 60)
    print("Testing Local LLM Generator")
    print("=" * 60)
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available, using CPU (will be slower)")
    
    # Initialize generator with small model
    print("\nLoading Qwen2.5-0.5B-Instruct...")
    generator = AnswerGenerator(
        model_name="Qwen/Qwen2.5-0.5B-Instruct"
    )
    
    # Test with sample data
    question = "What is the function of hemoglobin?"
    passages = [
        "Hemoglobin is a protein in red blood cells that carries oxygen from the lungs to the body's tissues and returns carbon dioxide from the tissues back to the lungs.",
        "The hemoglobin molecule is made up of four protein chains, two alpha chains and two beta chains, each with a ring-like heme group containing an iron atom.",
        "Hemoglobin levels are measured as part of a complete blood count (CBC) test."
    ]
    
    print(f"\nQuestion: {question}")
    print(f"Passages provided: {len(passages)}")
    
    answer = generator.generate(question, passages)
    
    print(f"\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()
