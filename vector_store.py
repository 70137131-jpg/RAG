"""
Vector Store using ChromaDB
Handles embedding and retrieval of documents
"""

# Disable TensorFlow in transformers (not needed for this project)
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import tiktoken

# Conditional import for CrossEncoder (only needed for reranking)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None


class VectorStore:
    """Vector database for storing and retrieving document embeddings"""

    def __init__(
        self,
        collection_name: str = "squad_contexts",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_token_chunking: bool = True,
        reranker_model: Optional[str] = None
    ):
        """
        Initialize the vector store

        Args:
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Name of the sentence-transformers model
            chunk_size: Maximum size per chunk (tokens if use_token_chunking, else chars)
            chunk_overlap: Overlap between chunks (tokens if use_token_chunking, else chars)
            use_token_chunking: If True, use token-aware chunking instead of character-based
            reranker_model: Optional cross-encoder model for reranking (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_token_chunking = use_token_chunking

        # Initialize tokenizer for token-aware chunking
        if use_token_chunking:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except (ValueError, KeyError, LookupError) as e:
                print(f"Warning: Failed to load tiktoken ({e}), falling back to character chunking")
                self.use_token_chunking = False
                self.tokenizer = None
        else:
            self.tokenizer = None

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize optional reranker
        self.reranker = None
        if reranker_model:
            print(f"Loading reranker model: {reranker_model}")
            try:
                self.reranker = CrossEncoder(reranker_model)
            except Exception as e:
                print(f"Warning: Failed to load reranker model: {e}")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} documents")
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection '{collection_name}'")

    def chunk_text_by_tokens(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks by token count (more accurate for LLMs)

        Args:
            text: Text to chunk
            chunk_size: Maximum tokens per chunk (uses self.chunk_size if None)
            overlap: Number of overlapping tokens (uses self.chunk_overlap if None)

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        if not self.tokenizer:
            # Fallback to character-based chunking
            return self.chunk_text_by_chars(text, chunk_size, overlap)

        # Encode text to tokens
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            # Get chunk of tokens
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position with overlap
            start += chunk_size - overlap

            # Break if we've reached the end
            if end >= len(tokens):
                break

        return chunks

    def chunk_text_by_chars(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks by character count (legacy method)

        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk (uses self.chunk_size if None)
            overlap: Number of overlapping characters (uses self.chunk_overlap if None)

        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0

        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space

            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                # Keep last few words for overlap
                overlap_words = int(len(current_chunk) * (overlap / chunk_size))
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_size = sum(len(w) + 1 for w in current_chunk)

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks (uses token-aware or char-based chunking)

        Args:
            text: Text to chunk
            chunk_size: Maximum size per chunk
            overlap: Overlap size

        Returns:
            List of text chunks
        """
        if self.use_token_chunking:
            return self.chunk_text_by_tokens(text, chunk_size, overlap)
        else:
            return self.chunk_text_by_chars(text, chunk_size, overlap)

    def add_documents(
        self,
        documents: List[Dict[str, str]],
        chunk_documents: bool = False,
        batch_size: int = 100
    ):
        """
        Add documents to the vector store

        Args:
            documents: List of dicts with 'id' and 'text' keys
            chunk_documents: Whether to split documents into chunks
            batch_size: Number of documents to process at once
        """
        print(f"Adding {len(documents)} documents to vector store...")

        all_texts = []
        all_ids = []
        all_metadatas = []

        for doc in tqdm(documents, desc="Processing documents"):
            doc_id = doc['id']
            doc_text = doc['text']

            if chunk_documents:
                chunks = self.chunk_text(doc_text)
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_ids.append(f"{doc_id}_chunk_{i}")
                    all_metadatas.append({
                        "source_id": doc_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
            else:
                all_texts.append(doc_text)
                all_ids.append(doc_id)
                all_metadatas.append({"source_id": doc_id})

        # Add documents in batches
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Adding to ChromaDB"):
            batch_texts = all_texts[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]

            # Generate embeddings
            embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadatas
            )

        print(f"Successfully added documents. Total count: {self.collection.count()}")

    def search(
        self,
        query: str,
        top_k: int = 3,
        rerank: bool = True,
        rerank_top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for relevant documents with optional reranking

        Args:
            query: Search query
            top_k: Number of final results to return
            rerank: Whether to use reranker if available
            rerank_top_n: Number of candidates to fetch before reranking (default: top_k * 3)

        Returns:
            List of dictionaries with document info and relevance scores
        """
        # If reranker is available and enabled, fetch more candidates for reranking
        fetch_k = top_k
        if rerank and self.reranker:
            rerank_top_n = rerank_top_n or (top_k * 3)
            fetch_k = min(rerank_top_n, self.collection.count())

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        # Apply reranking if enabled and model is available
        if rerank and self.reranker and len(formatted_results) > 0:
            # Create query-document pairs for reranking
            pairs = [[query, doc['text']] for doc in formatted_results]

            # Get reranker scores
            rerank_scores = self.reranker.predict(pairs)

            # Add rerank scores to results
            for doc, score in zip(formatted_results, rerank_scores):
                doc['rerank_score'] = float(score)

            # Sort by rerank score (higher is better for cross-encoder)
            formatted_results = sorted(
                formatted_results,
                key=lambda x: x['rerank_score'],
                reverse=True
            )

            # Keep only top_k after reranking
            formatted_results = formatted_results[:top_k]

        return formatted_results

    def reset(self):
        """Delete and recreate the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection '{self.collection_name}'")
        except ValueError:
            # Collection doesn't exist, nothing to delete
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection '{self.collection_name}'")

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "embedding_model": self.embedding_model_name,
            "persist_directory": self.persist_directory
        }


def main():
    """Example usage"""
    # Create sample documents
    documents = [
        {
            "id": "doc1",
            "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel."
        },
        {
            "id": "doc2",
            "text": "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states."
        },
        {
            "id": "doc3",
            "text": "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor."
        }
    ]

    # Initialize vector store
    vector_store = VectorStore(collection_name="test_collection")
    vector_store.reset()
    vector_store.add_documents(documents)

    # Search
    results = vector_store.search("What is in Paris?", top_k=2)

    print("\nSearch Results:")
    for result in results:
        print(f"\n- {result['id']} (distance: {result['distance']:.4f})")
        print(f"  {result['text'][:100]}...")


if __name__ == "__main__":
    main()
