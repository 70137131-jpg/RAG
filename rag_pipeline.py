"""
RAG (Retrieval-Augmented Generation) Pipeline
Combines retrieval from vector store with LLM generation
Enhanced with context compression and hallucination prevention
"""

from typing import List, Dict, Optional, Tuple
from vector_store import VectorStore
import os
from dotenv import load_dotenv
from openai import OpenAI
import re

# Try to import Google Generative AI (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class RAGPipeline:
    """Complete RAG pipeline for question answering"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_model: str = "gpt-oss-20b",  # Model name for OpenRouter or direct API
        temperature: float = 0.1,
        max_tokens: int = 300,
        max_context_length: int = 4000,
        enable_compression: bool = True,
        require_citations: bool = True
    ):
        """
        Initialize the RAG pipeline

        Args:
            vector_store: Initialized VectorStore instance
            llm_model: LLM model name (e.g., gpt-oss-20b)
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            max_context_length: Maximum character length for combined contexts
            enable_compression: Whether to compress contexts before generation
            require_citations: Whether to require citations in answers
        """
        load_dotenv()

        self.vector_store = vector_store
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_length = max_context_length
        self.enable_compression = enable_compression
        self.require_citations = require_citations

        # Check if using Gemini
        self.use_gemini = llm_model.startswith("gemini") or "gemini" in llm_model.lower()
        
        if self.use_gemini:
            # Initialize Gemini client
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "google-generativeai package is required for Gemini. Install it with: pip install google-generativeai"
                )
            
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Please set GOOGLE_API_KEY in your .env file for Gemini models."
                )
            
            genai.configure(api_key=google_api_key)
            self.gemini_model = genai.GenerativeModel(llm_model)
            self.client = None  # Not used for Gemini
        else:
            # Initialize OpenAI-compatible client (works with OpenRouter, OpenAI, DeepSeek, etc.)
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key not found. Please set OPENROUTER_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, or GOOGLE_API_KEY in your .env file."
                )

            # Check if using OpenRouter (API key starts with sk-or-v1-)
            if api_key.startswith("sk-or-v1-"):
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            else:
                # Use base URL from env or default to OpenAI
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
            self.gemini_model = None

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with metadata
        """
        return self.vector_store.search(query, top_k=top_k)

    def compress_context(self, context: str, query: str, max_sentences: int = 5) -> str:
        """
        Compress context by extracting most relevant sentences

        Args:
            context: Full context text
            query: User's question
            max_sentences: Maximum number of sentences to keep

        Returns:
            Compressed context with most relevant sentences
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) <= max_sentences:
            return context

        # Simple relevance scoring based on query word overlap
        query_words = set(query.lower().split())
        scored_sentences = []

        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            scored_sentences.append((overlap, sent))

        # Sort by relevance and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [sent for _, sent in scored_sentences[:max_sentences]]

        return ' '.join(top_sentences)

    def truncate_contexts(self, contexts: List[str], query: str) -> List[str]:
        """
        Truncate contexts to fit within max_context_length

        Args:
            contexts: List of context strings
            query: User's question for compression

        Returns:
            Truncated/compressed contexts
        """
        if not self.enable_compression:
            # Simple truncation without compression
            combined = "\n\n".join(contexts)
            if len(combined) <= self.max_context_length:
                return contexts

            # Truncate from the end
            truncated = []
            current_length = 0
            for ctx in contexts:
                if current_length + len(ctx) + 2 <= self.max_context_length:
                    truncated.append(ctx)
                    current_length += len(ctx) + 2
                else:
                    break
            return truncated if truncated else [contexts[0][:self.max_context_length]]

        # Compress each context
        compressed = [self.compress_context(ctx, query) for ctx in contexts]

        # Check if still too long
        combined = "\n\n".join(compressed)
        if len(combined) <= self.max_context_length:
            return compressed

        # Further truncation if needed
        truncated = []
        current_length = 0
        for ctx in compressed:
            if current_length + len(ctx) + 2 <= self.max_context_length:
                truncated.append(ctx)
                current_length += len(ctx) + 2
            else:
                # Try to fit a truncated version
                remaining = self.max_context_length - current_length - 2
                if remaining > 100:
                    truncated.append(ctx[:remaining] + "...")
                break

        return truncated if truncated else [compressed[0][:self.max_context_length]]

    def generate(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate answer using LLM with retrieved contexts
        Enhanced with compression and hallucination prevention

        Args:
            query: User's question
            contexts: Retrieved context passages
            system_prompt: Custom system prompt (optional)

        Returns:
            Generated answer with optional citations
        """
        # Truncate/compress contexts if needed
        processed_contexts = self.truncate_contexts(contexts, query)

        # Enhanced system prompt with stricter grounding
        if system_prompt is None:
            if self.require_citations:
                system_prompt = """You are a precise question-answering assistant. Follow these rules strictly:

1. ONLY use information explicitly stated in the provided contexts
2. Include citations by referencing [Context N] after each claim
3. If the answer requires information from multiple contexts, cite all relevant sources
4. If the contexts don't contain sufficient information, respond: "I cannot answer this question based on the provided contexts."
5. Do NOT use external knowledge or make assumptions
6. Keep answers concise and directly relevant to the question
7. Quote exact phrases from contexts when appropriate using "quotation marks" with [Context N] citation"""
            else:
                system_prompt = """You are a precise question-answering assistant. Follow these rules strictly:

1. ONLY use information explicitly stated in the provided contexts
2. If the contexts don't contain sufficient information, respond: "I cannot answer this question based on the provided contexts."
3. Do NOT use external knowledge, assumptions, or inferences not directly supported by the text
4. Keep answers concise and directly relevant to the question
5. Quote exact phrases from contexts when appropriate using "quotation marks"
6. If uncertain, express that uncertainty clearly"""

        # Combine contexts with clear numbering
        combined_context = "\n\n".join([
            f"[Context {i+1}]:\n{ctx}"
            for i, ctx in enumerate(processed_contexts)
        ])

        # Create the enhanced prompt template
        full_prompt = f"""{system_prompt}

===== CONTEXTS =====
{combined_context}

===== QUESTION =====
{query}

===== ANSWER =====
Based on the provided contexts:"""

        # Call LLM API (Gemini or OpenAI-compatible)
        try:
            if self.use_gemini:
                # Use Gemini API
                generation_config = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
                
                # Gemini uses a single prompt format
                prompt_text = f"{system_prompt}\n\n{full_prompt}"
                
                response = self.gemini_model.generate_content(
                    prompt_text,
                    generation_config=generation_config
                )
                
                answer = response.text.strip()
            else:
                # Use OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                answer = response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"Error generating response with LLM: {str(e)}")

        # Post-process to detect potential hallucinations
        if self.require_citations and "[Context" not in answer and "cannot answer" not in answer.lower():
            # If citations are required but not present, add a warning
            answer = f"{answer}\n\n[Note: This answer may not be fully grounded in the provided contexts]"

        return answer

    def verify_grounding(self, answer: str, contexts: List[str]) -> Tuple[bool, float]:
        """
        Verify if the answer is grounded in the provided contexts

        Args:
            answer: Generated answer
            contexts: Source contexts

        Returns:
            Tuple of (is_grounded, confidence_score)
        """
        # Remove citation markers for comparison
        clean_answer = re.sub(r'\[Context \d+\]', '', answer).lower()

        # Check if answer claims inability to answer
        if "cannot answer" in clean_answer:
            return True, 1.0

        # Extract meaningful words from answer (excluding common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been'}
        answer_words = set(re.findall(r'\b\w+\b', clean_answer)) - stop_words

        if not answer_words:
            return True, 0.5  # Empty answer

        # Check overlap with contexts
        context_text = ' '.join(contexts).lower()
        context_words = set(re.findall(r'\b\w+\b', context_text)) - stop_words

        overlap = len(answer_words & context_words)
        total = len(answer_words)

        confidence = overlap / total if total > 0 else 0.0

        # Consider well-grounded if >70% of answer words appear in contexts
        is_grounded = confidence > 0.7

        return is_grounded, confidence

    def query(
        self,
        question: str,
        top_k: int = 3,
        return_metadata: bool = False,
        verify_grounding: bool = False
    ) -> Dict:
        """
        Complete RAG query: retrieve + generate
        Enhanced with grounding verification

        Args:
            question: User's question
            top_k: Number of contexts to retrieve
            return_metadata: Whether to return retrieval metadata
            verify_grounding: Whether to verify answer grounding

        Returns:
            Dictionary with answer and optionally metadata
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k=top_k)

        # Step 2: Extract context texts
        contexts = [doc['text'] for doc in retrieved_docs]

        # Step 3: Generate answer
        answer = self.generate(question, contexts)

        # Step 4: Verify grounding if requested
        grounding_info = None
        if verify_grounding:
            is_grounded, confidence = self.verify_grounding(answer, contexts)
            grounding_info = {
                "is_grounded": is_grounded,
                "confidence": confidence,
                "warning": None if is_grounded else "Answer may contain hallucinated information"
            }

        # Prepare response
        response = {"answer": answer}

        if return_metadata:
            response["metadata"] = {
                "retrieved_docs": retrieved_docs,
                "num_contexts": len(contexts),
                "num_contexts_used": len(self.truncate_contexts(contexts, question)),
                "context_compressed": self.enable_compression,
                "citations_required": self.require_citations,
                "model": self.llm_model
            }
            if grounding_info:
                response["metadata"]["grounding"] = grounding_info

        return response

    def batch_query(
        self,
        questions: List[str],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Process multiple questions in batch

        Args:
            questions: List of questions
            top_k: Number of contexts per question

        Returns:
            List of answers with metadata
        """
        results = []

        for question in questions:
            result = self.query(question, top_k=top_k, return_metadata=True)
            results.append({
                "question": question,
                "answer": result["answer"],
                "metadata": result["metadata"]
            })

        return results


def main():
    """Example usage of RAG pipeline"""
    from data_loader import SQuADLoader

    # Load a small sample of SQuAD data
    print("Loading SQuAD dataset...")
    loader = SQuADLoader(dataset_name="squad_v2", split="validation")
    loader.load(max_samples=50)

    contexts = loader.get_contexts()
    qa_pairs = loader.get_qa_pairs()

    # Initialize vector store and add documents
    print("\nInitializing vector store...")
    vector_store = VectorStore(collection_name="squad_demo")
    vector_store.reset()
    vector_store.add_documents(contexts)

    # Initialize RAG pipeline
    print("\nInitializing RAG pipeline...")
    rag = RAGPipeline(vector_store=vector_store)

    # Test with a sample question
    print("\n" + "="*60)
    print("RAG DEMO")
    print("="*60)

    test_question = qa_pairs[0]['question']
    expected_answer = qa_pairs[0]['answers'][0]

    print(f"\nQuestion: {test_question}")
    print(f"\nExpected Answer: {expected_answer}")

    result = rag.query(test_question, top_k=2, return_metadata=True)

    print(f"\nRAG Answer: {result['answer']}")
    print(f"\nRetrieved {len(result['metadata']['retrieved_docs'])} contexts")

    # Show retrieved contexts
    print("\nRetrieved Contexts:")
    for i, doc in enumerate(result['metadata']['retrieved_docs'], 1):
        print(f"\n{i}. (Distance: {doc['distance']:.4f})")
        print(f"   {doc['text'][:150]}...")


if __name__ == "__main__":
    main()
