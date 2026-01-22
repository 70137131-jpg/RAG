"""
Configuration classes for RAG system
Provides typed configuration with environment variable support
"""

from typing import Optional
from dataclasses import dataclass, field
import os


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    collection_name: str = "squad_contexts"
    persist_directory: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    reranker_model: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for LLM"""
    model: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: int = 300
    base_url: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data loading"""
    dataset_name: str = "squad_v2"
    split: str = "validation"
    max_samples: int = 100


@dataclass
class RAGConfig:
    """Main configuration for RAG system"""
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    top_k: int = 3

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables"""
        vs_config = VectorStoreConfig(
            collection_name=os.getenv("COLLECTION_NAME", "squad_contexts"),
            persist_directory=os.getenv("PERSIST_DIRECTORY", "./chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            reranker_model=os.getenv("RERANKER_MODEL", None)
        )

        llm_config = LLMConfig(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "300")),
            base_url=os.getenv("OPENAI_BASE_URL", None)
        )

        data_config = DataConfig(
            dataset_name=os.getenv("DATASET_NAME", "squad_v2"),
            split=os.getenv("DATASET_SPLIT", "validation"),
            max_samples=int(os.getenv("MAX_SAMPLES", "100"))
        )

        return cls(
            vector_store=vs_config,
            llm=llm_config,
            data=data_config,
            top_k=int(os.getenv("TOP_K", "3"))
        )

    @classmethod
    def default(cls) -> "RAGConfig":
        """Default configuration"""
        return cls()

    @classmethod
    def fast(cls) -> "RAGConfig":
        """Fast configuration with smaller model"""
        config = cls()
        config.vector_store.embedding_model = "all-MiniLM-L6-v2"
        config.vector_store.chunk_size = 300
        config.llm.model = "gemini-1.5-flash"
        config.llm.temperature = 0.1
        config.top_k = 2
        return config

    @classmethod
    def accurate(cls) -> "RAGConfig":
        """Accurate configuration with larger model"""
        config = cls()
        config.vector_store.embedding_model = "all-mpnet-base-v2"
        config.vector_store.chunk_size = 700
        config.llm.model = "gemini-1.5-flash"
        config.llm.temperature = 0.0
        config.llm.max_tokens = 500
        config.top_k = 5
        return config

    @classmethod
    def balanced(cls) -> "RAGConfig":
        """Balanced configuration"""
        config = cls()
        config.vector_store.embedding_model = "all-MiniLM-L6-v2"
        config.vector_store.chunk_size = 500
        config.llm.model = "gemini-1.5-flash"
        config.llm.temperature = 0.1
        config.llm.max_tokens = 300
        config.top_k = 3
        return config

    @classmethod
    def gemini(cls) -> "RAGConfig":
        """Gemini configuration"""
        config = cls()
        config.vector_store.embedding_model = "all-MiniLM-L6-v2"
        config.vector_store.chunk_size = 500
        config.llm.model = "gemini-1.5-flash"  # or "gemini-1.5-pro" for better quality
        config.llm.temperature = 0.1
        config.llm.max_tokens = 300
        config.top_k = 3
        return config


FAST_CONFIG = RAGConfig.fast()
BALANCED_CONFIG = RAGConfig.balanced()
ACCURATE_CONFIG = RAGConfig.accurate()
GEMINI_CONFIG = RAGConfig.gemini()