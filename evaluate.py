from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import RAGPipeline
from data_loader import SQuADLoader
from vector_store import VectorStore
from config import RAGConfig, BALANCED_CONFIG
from config_utils import create_vector_store_from_config, create_rag_pipeline_from_config, create_data_loader_from_config
from typing import List, Dict
import json
from tqdm import tqdm
from collections import defaultdict
import re
import sys

# Ensure UTF-8 output in Windows terminals to avoid UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


class RAGEvaluator:
    """Evaluate RAG pipeline performance"""

    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize evaluator

        Args:
            rag_pipeline: RAGPipeline instance to evaluate
        """
        self.rag = rag_pipeline

    @staticmethod
    def normalize_answer(text: str) -> str:
        """
        Normalize answer text for comparison
        Removes punctuation, extra whitespace, and converts to lowercase
        """
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text.lower())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def exact_match(self, prediction: str, ground_truths: List[str]) -> bool:
        """
        Check if prediction matches any ground truth (after normalization)

        Args:
            prediction: Model's predicted answer
            ground_truths: List of acceptable answers

        Returns:
            True if exact match found
        """
        normalized_pred = self.normalize_answer(prediction)
        return any(
            normalized_pred == self.normalize_answer(gt)
            for gt in ground_truths
        )

    def f1_score(self, prediction: str, ground_truths: List[str]) -> float:
        """
        Calculate token-level F1 score

        Args:
            prediction: Model's predicted answer
            ground_truths: List of acceptable answers

        Returns:
            Maximum F1 score across all ground truths
        """
        def get_tokens(text):
            return self.normalize_answer(text).split()

        pred_tokens = get_tokens(prediction)

        if not pred_tokens:
            return 0.0

        max_f1 = 0.0

        for gt in ground_truths:
            gt_tokens = get_tokens(gt)

            if not gt_tokens:
                continue

            common = set(pred_tokens) & set(gt_tokens)
            num_common = len(common)

            if num_common == 0:
                continue

            precision = num_common / len(pred_tokens)
            recall = num_common / len(gt_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
            max_f1 = max(max_f1, f1)

        return max_f1

    def contains_answer(self, prediction: str, ground_truths: List[str]) -> bool:
        """
        Check if any ground truth is contained in the prediction

        Args:
            prediction: Model's predicted answer
            ground_truths: List of acceptable answers

        Returns:
            True if any ground truth is found in prediction
        """
        normalized_pred = self.normalize_answer(prediction)
        return any(
            self.normalize_answer(gt) in normalized_pred
            for gt in ground_truths
        )

    def evaluate_retrieval(
        self,
        qa_pairs: List[Dict],
        top_k: int = 3
    ) -> Dict:
        """
        Evaluate retrieval performance (without generation)

        Args:
            qa_pairs: List of question-answer pairs with context
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with retrieval metrics
        """
        print("Evaluating retrieval performance...")

        total = 0
        context_found = 0
        mrr_sum = 0  # Mean Reciprocal Rank

        for qa in tqdm(qa_pairs):
            question = qa['question']
            expected_context = self.normalize_answer(qa['context'])

            # Retrieve documents
            retrieved = self.rag.retrieve(question, top_k=top_k)

            # Check if the correct context is in retrieved results
            found_rank = None
            for rank, doc in enumerate(retrieved, 1):
                if self.normalize_answer(doc['text']) == expected_context:
                    context_found += 1
                    found_rank = rank
                    break
                # Also check if retrieved text is a chunk of the expected context
                elif (self.normalize_answer(doc['text']) in expected_context or
                      expected_context in self.normalize_answer(doc['text'])):
                    context_found += 1
                    found_rank = rank
                    break

            if found_rank:
                mrr_sum += 1 / found_rank

            total += 1

        metrics = {
            "total_questions": total,
            "context_recall": context_found / total if total > 0 else 0,
            "mean_reciprocal_rank": mrr_sum / total if total > 0 else 0
        }

        return metrics

    def evaluate_generation(
        self,
        qa_pairs: List[Dict],
        top_k: int = 3,
        max_samples: int = None
    ) -> Dict:
        """
        Evaluate end-to-end RAG performance

        Args:
            qa_pairs: List of question-answer pairs
            top_k: Number of contexts to retrieve
            max_samples: Maximum number of samples to evaluate (None for all)

        Returns:
            Dictionary with performance metrics
        """
        if max_samples:
            qa_pairs = qa_pairs[:max_samples]

        print(f"Evaluating generation on {len(qa_pairs)} questions...")

        results = []
        exact_matches = 0
        f1_scores = []
        contains_matches = 0

        for qa in tqdm(qa_pairs):
            question = qa['question']
            ground_truths = qa['answers']

            try:
                # Query the RAG pipeline
                response = self.rag.query(question, top_k=top_k)
                prediction = response['answer']

                # Calculate metrics
                em = self.exact_match(prediction, ground_truths)
                f1 = self.f1_score(prediction, ground_truths)
                contains = self.contains_answer(prediction, ground_truths)

                if em:
                    exact_matches += 1
                if contains:
                    contains_matches += 1

                f1_scores.append(f1)

                results.append({
                    "question": question,
                    "prediction": prediction,
                    "ground_truths": ground_truths,
                    "exact_match": em,
                    "f1_score": f1,
                    "contains_answer": contains
                })

            except Exception as e:
                print(f"\nError processing question: {question}")
                print(f"Error: {str(e)}")
                continue

        # Calculate aggregate metrics
        total = len(results)
        metrics = {
            "total_questions": total,
            "exact_match": exact_matches / total if total > 0 else 0,
            "f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            "contains_answer": contains_matches / total if total > 0 else 0
        }

        return {
            "metrics": metrics,
            "results": results
        }

    def save_results(self, results: Dict, filepath: str):
        """Save evaluation results to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Run full evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Number of contexts to retrieve (uses config default if not specified)")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for results")
    parser.add_argument("--config", type=str, choices=["default", "fast", "accurate", "balanced"],
                        default="balanced", help="Configuration profile to use")
    args = parser.parse_args()

    # Load configuration
    if args.config == "default":
        config = RAGConfig.default()
    elif args.config == "fast":
        from config import FAST_CONFIG
        config = FAST_CONFIG
    elif args.config == "accurate":
        from config import ACCURATE_CONFIG
        config = ACCURATE_CONFIG
    else:
        config = BALANCED_CONFIG

    # Override config values if specified
    if args.max_samples:
        config.data.max_samples = args.max_samples
    top_k = args.top_k if args.top_k else config.top_k

    # Load dataset
    print(f"Loading dataset with config: {args.config}")
    config.data.dataset_name = "squad_v2"  # Force correct dataset
    loader = create_data_loader_from_config(config)

    contexts = loader.get_contexts()
    qa_pairs = loader.get_qa_pairs()

    print(f"\nDataset loaded:")
    print(f"  - Unique contexts: {len(contexts)}")
    print(f"  - QA pairs: {len(qa_pairs)}")

    # Initialize vector store
    print("\nInitializing vector store...")
    vector_store = create_vector_store_from_config(config.vector_store)

    # Check if collection already exists with documents
    if vector_store.collection.count() == 0:
        print("Adding documents to vector store...")
        vector_store.add_documents(contexts)
    else:
        print(f"Using existing collection with {vector_store.collection.count()} documents")

    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    rag = create_rag_pipeline_from_config(config, vector_store)

    # Initialize evaluator
    evaluator = RAGEvaluator(rag)

    # Evaluate retrieval
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION")
    print("="*60)
    retrieval_metrics = evaluator.evaluate_retrieval(qa_pairs[:50], top_k=top_k)
    print("\nRetrieval Metrics:")
    for key, value in retrieval_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Evaluate generation (on smaller subset due to API costs)
    print("\n" + "="*60)
    print("GENERATION EVALUATION")
    print("="*60)
    eval_results = evaluator.evaluate_generation(
        qa_pairs,
        top_k=top_k,
        max_samples=min(20, len(qa_pairs))  # Limit for API costs
    )

    print("\nGeneration Metrics:")
    for key, value in eval_results['metrics'].items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Show some examples
    print("\n" + "="*60)
    print("EXAMPLE RESULTS")
    print("="*60)
    for i, result in enumerate(eval_results['results'][:3], 1):
        print(f"\nExample {i}:")
        print(f"Q: {result['question']}")
        print(f"Expected: {result['ground_truths'][0]}")
        print(f"Predicted: {result['prediction']}")
        print(f"F1 Score: {result['f1_score']:.4f}")

    # Save results
    evaluator.save_results(eval_results, args.output)


if __name__ == "__main__":
    main()
