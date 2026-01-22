"""
Data Loader for SQuAD Dataset
Handles downloading and preprocessing the SQuAD 2.0 dataset
"""

from datasets import load_dataset
from typing import List, Dict, Tuple
import json


class SQuADLoader:
    """Load and preprocess SQuAD dataset for RAG"""

    def __init__(self, dataset_name: str = "squad_v2", split: str = "validation"):
        """
        Initialize the SQuAD loader

        Args:
            dataset_name: Name of the dataset ('squad' or 'squad_v2')
            split: Dataset split to use ('train' or 'validation')
        """
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None

    def load(self, max_samples: int = None) -> List[Dict]:
        """
        Load the SQuAD dataset

        Args:
            max_samples: Maximum number of samples to load (None for all)

        Returns:
            List of dictionaries containing context, question, and answer
        """
        print(f"Loading {self.dataset_name} dataset ({self.split} split)...")

        # Load dataset from Hugging Face
        self.dataset = load_dataset(self.dataset_name, split=self.split)

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        print(f"Loaded {len(self.dataset)} samples")
        return self.dataset

    def get_contexts(self) -> List[Dict[str, str]]:
        """
        Extract unique contexts from the dataset

        Returns:
            List of dictionaries with 'id' and 'text' keys
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Get unique contexts (many questions share the same context)
        unique_contexts = {}

        for idx, item in enumerate(self.dataset):
            context = item['context']
            if context not in unique_contexts:
                unique_contexts[context] = f"ctx_{len(unique_contexts)}"

        contexts = [
            {"id": ctx_id, "text": context}
            for context, ctx_id in unique_contexts.items()
        ]

        print(f"Extracted {len(contexts)} unique contexts")
        return contexts

    def get_qa_pairs(self) -> List[Dict]:
        """
        Extract question-answer pairs

        Returns:
            List of dictionaries with question, answer, and context info
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        qa_pairs = []

        for item in self.dataset:
            # SQuAD 2.0 includes unanswerable questions
            if item['answers']['text']:  # Only answerable questions
                qa_pair = {
                    'id': item['id'],
                    'question': item['question'],
                    'context': item['context'],
                    'answers': item['answers']['text'],  # List of acceptable answers
                    'answer_starts': item['answers']['answer_start']
                }
                qa_pairs.append(qa_pair)

        print(f"Extracted {len(qa_pairs)} answerable QA pairs")
        return qa_pairs

    def save_to_file(self, filepath: str):
        """Save the processed dataset to a JSON file"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        data = {
            'contexts': self.get_contexts(),
            'qa_pairs': self.get_qa_pairs()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Dataset saved to {filepath}")


def main():
    """Example usage"""
    loader = SQuADLoader(dataset_name="squad_v2", split="validation")
    loader.load(max_samples=100)  # Load first 100 samples for testing

    contexts = loader.get_contexts()
    qa_pairs = loader.get_qa_pairs()

    print(f"\nExample Context:")
    print(contexts[0]['text'][:200] + "...")

    print(f"\nExample QA Pair:")
    print(f"Q: {qa_pairs[0]['question']}")
    print(f"A: {qa_pairs[0]['answers'][0]}")


if __name__ == "__main__":
    main()
