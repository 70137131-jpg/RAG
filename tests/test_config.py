"""
Tests for runtime configuration parsing and validation.
"""

import unittest
from unittest.mock import patch

from config import ConfigError, RAGConfig


class TestRAGConfigFromEnv(unittest.TestCase):
    """Configuration should fail fast with clear errors for bad env values."""

    def test_defaults_from_empty_env(self):
        with patch.dict("os.environ", {}, clear=True):
            config = RAGConfig.from_env()

        self.assertEqual(config.vector_store.collection_name, "squad_contexts")
        self.assertEqual(config.vector_store.persist_directory, "./chroma_db")
        self.assertEqual(config.llm.model, "gemini-2.5-flash")
        self.assertEqual(config.top_k, 3)

    def test_valid_env_overrides(self):
        env = {
            "COLLECTION_NAME": "prod_docs",
            "PERSIST_DIRECTORY": "/data/chroma",
            "EMBEDDING_MODEL": "all-mpnet-base-v2",
            "CHUNK_SIZE": "800",
            "CHUNK_OVERLAP": "80",
            "LLM_MODEL": "gpt-4o-mini",
            "TEMPERATURE": "0.2",
            "MAX_TOKENS": "500",
            "TOP_K": "5",
            "DATASET_NAME": "squad_v2",
            "DATASET_SPLIT": "train",
            "MAX_SAMPLES": "250",
        }

        with patch.dict("os.environ", env, clear=True):
            config = RAGConfig.from_env()

        self.assertEqual(config.vector_store.collection_name, "prod_docs")
        self.assertEqual(config.vector_store.persist_directory, "/data/chroma")
        self.assertEqual(config.vector_store.chunk_size, 800)
        self.assertEqual(config.vector_store.chunk_overlap, 80)
        self.assertEqual(config.llm.temperature, 0.2)
        self.assertEqual(config.llm.max_tokens, 500)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.data.max_samples, 250)

    def test_invalid_integer_has_clear_error(self):
        with patch.dict("os.environ", {"TOP_K": "many"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "TOP_K must be an integer"):
                RAGConfig.from_env()

    def test_top_k_is_bounded(self):
        with patch.dict("os.environ", {"TOP_K": "25"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "TOP_K must be <= 10"):
                RAGConfig.from_env()

    def test_temperature_is_bounded(self):
        with patch.dict("os.environ", {"TEMPERATURE": "3"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "TEMPERATURE must be <= 2.0"):
                RAGConfig.from_env()

    def test_chunk_overlap_must_be_smaller_than_chunk_size(self):
        env = {"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}

        with patch.dict("os.environ", env, clear=True):
            with self.assertRaisesRegex(ConfigError, "CHUNK_OVERLAP must be smaller"):
                RAGConfig.from_env()

    def test_empty_required_string_is_rejected(self):
        with patch.dict("os.environ", {"LLM_MODEL": "   "}, clear=True):
            with self.assertRaisesRegex(ConfigError, "LLM_MODEL must not be empty"):
                RAGConfig.from_env()


if __name__ == "__main__":
    unittest.main()
