"""
Configuration classes for RAG system
Provides typed configuration with environment variable support
"""

from typing import Optional
from dataclasses import dataclass, field
import os


class ConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


def _env_str(name: str, default: str, *, allow_empty: bool = False) -> str:
    value = os.getenv(name, default).strip()
    if not value and not allow_empty:
        raise ConfigError(f"{name} must not be empty")
    return value


def _env_int(name: str, default: int, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer, got {raw_value!r}") from exc

    if min_value is not None and value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"{name} must be <= {max_value}, got {value}")
    return value


def _env_float(name: str, default: float, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        value = default
    else:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number, got {raw_value!r}") from exc

    if min_value is not None and value < min_value:
        raise ConfigError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"{name} must be <= {max_value}, got {value}")
    return value


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
    model: str = "gemini-2.5-flash"
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
        persist_directory = (
            os.getenv("PERSIST_DIRECTORY")
            or os.getenv("PERSIST_DIR")
            or "./chroma_db"
        ).strip()
        if not persist_directory:
            raise ConfigError("PERSIST_DIRECTORY/PERSIST_DIR must not be empty")

        chunk_size = _env_int("CHUNK_SIZE", 500, min_value=1)
        chunk_overlap = _env_int("CHUNK_OVERLAP", 50, min_value=0)
        if chunk_overlap >= chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP must be smaller than CHUNK_SIZE "
                f"(got overlap={chunk_overlap}, size={chunk_size})"
            )

        vs_config = VectorStoreConfig(
            collection_name=_env_str("COLLECTION_NAME", "squad_contexts"),
            persist_directory=persist_directory,
            embedding_model=_env_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            reranker_model=os.getenv("RERANKER_MODEL") or None
        )

        llm_config = LLMConfig(
            model=_env_str("LLM_MODEL", "gemini-2.5-flash"),
            temperature=_env_float("TEMPERATURE", 0.1, min_value=0.0, max_value=2.0),
            max_tokens=_env_int("MAX_TOKENS", 300, min_value=1),
            base_url=os.getenv("OPENAI_BASE_URL") or None
        )

        data_config = DataConfig(
            dataset_name=_env_str("DATASET_NAME", "squad_v2"),
            split=_env_str("DATASET_SPLIT", "validation"),
            max_samples=_env_int("MAX_SAMPLES", 100, min_value=1)
        )

        return cls(
            vector_store=vs_config,
            llm=llm_config,
            data=data_config,
            top_k=_env_int("TOP_K", 3, min_value=1, max_value=10)
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
        config.llm.model = "gemini-2.5-flash"
        config.llm.temperature = 0.1
        config.top_k = 2
        return config

    @classmethod
    def accurate(cls) -> "RAGConfig":
        """Accurate configuration with larger model"""
        config = cls()
        config.vector_store.embedding_model = "all-mpnet-base-v2"
        config.vector_store.chunk_size = 700
        config.llm.model = "gemini-2.5-flash"
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
        config.llm.model = "gemini-2.5-flash"
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
        config.llm.model = "gemini-2.5-flash"  # or "gemini-2.5-pro" for better quality
        config.llm.temperature = 0.1
        config.llm.max_tokens = 300
        config.top_k = 3
        return config


FAST_CONFIG = RAGConfig.fast()
BALANCED_CONFIG = RAGConfig.balanced()
ACCURATE_CONFIG = RAGConfig.accurate()
GEMINI_CONFIG = RAGConfig.gemini()
