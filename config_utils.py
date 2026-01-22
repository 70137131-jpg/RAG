"""
Utility functions to create components from configuration
"""

from config import RAGConfig, VectorStoreConfig, LLMConfig
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from data_loader import SQuADLoader


def create_vector_store_from_config(config: VectorStoreConfig, reranker_model: str = None) -> VectorStore:
    """
    Create a VectorStore instance from configuration

    Args:
        config: VectorStoreConfig instance
        reranker_model: Optional reranker model name

    Returns:
        Initialized VectorStore
    """
    return VectorStore(
        collection_name=config.collection_name,
        persist_directory=config.persist_directory,
        embedding_model=config.embedding_model,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        use_token_chunking=True,  # Use token-aware chunking by default
        reranker_model=reranker_model
    )


def create_rag_pipeline_from_config(config: RAGConfig, vector_store: VectorStore = None) -> RAGPipeline:
    """
    Create a RAGPipeline instance from configuration

    Args:
        config: RAGConfig instance
        vector_store: Optional pre-initialized VectorStore (creates new one if None)

    Returns:
        Initialized RAGPipeline
    """
    if vector_store is None:
        vector_store = create_vector_store_from_config(config.vector_store)

    return RAGPipeline(
        vector_store=vector_store,
        llm_model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )


def create_data_loader_from_config(config: RAGConfig) -> SQuADLoader:
    """
    Create a SQuADLoader instance from configuration

    Args:
        config: RAGConfig instance

    Returns:
        Initialized SQuADLoader
    """
    loader = SQuADLoader(
        dataset_name=config.data.dataset_name,
        split=config.data.split
    )
    loader.load(max_samples=config.data.max_samples)
    return loader
