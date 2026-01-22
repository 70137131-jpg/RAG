"""
Tests for vector_store module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore"""

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.SentenceTransformer')
    def setUp(self, mock_sentence_transformer, mock_chroma_client):
        """Set up test fixtures"""
        # Mock ChromaDB client
        self.mock_collection = Mock()
        self.mock_collection.count = Mock(return_value=0)

        mock_client_instance = Mock()
        mock_client_instance.get_collection = Mock(side_effect=ValueError("Collection not found"))
        mock_client_instance.create_collection = Mock(return_value=self.mock_collection)
        mock_chroma_client.return_value = mock_client_instance

        # Mock embedding model - return numpy array (code calls .tolist())
        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        mock_sentence_transformer.return_value = mock_model

        # Create VectorStore instance
        self.vs = VectorStore(
            collection_name="test_collection",
            persist_directory="./test_db",
            use_token_chunking=False  # Use char-based for simplicity
        )

    def test_initialization(self):
        """Test VectorStore initialization"""
        self.assertEqual(self.vs.collection_name, "test_collection")
        self.assertEqual(self.vs.persist_directory, "./test_db")
        self.assertIsNotNone(self.vs.embedding_model)
        self.assertIsNotNone(self.vs.collection)

    def test_chunk_text_by_chars(self):
        """Test character-based text chunking"""
        text = "This is a test. " * 100  # Long text
        chunks = self.vs.chunk_text_by_chars(text, chunk_size=50, overlap=10)

        # Verify chunks
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertTrue(len(chunk) > 0)

    def test_search_basic(self):
        """Test basic search functionality"""
        # Mock collection query response
        self.mock_collection.query = Mock(return_value={
            'ids': [['doc1', 'doc2', 'doc3']],
            'documents': [['Text 1', 'Text 2', 'Text 3']],
            'metadatas': [[{'key': 'value'}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3]]
        })

        # Perform search
        results = self.vs.search("test query", top_k=3, rerank=False)

        # Verify results
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['id'], 'doc1')
        self.assertEqual(results[0]['text'], 'Text 1')
        self.assertEqual(results[0]['distance'], 0.1)

    def test_search_empty_collection(self):
        """Test search on empty collection"""
        self.mock_collection.query = Mock(return_value={
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        })

        results = self.vs.search("test query", top_k=3)
        self.assertEqual(len(results), 0)

    def test_get_stats(self):
        """Test getting vector store statistics"""
        self.mock_collection.count = Mock(return_value=100)

        stats = self.vs.get_stats()

        self.assertIn('collection_name', stats)
        self.assertIn('total_documents', stats)
        self.assertIn('embedding_model', stats)
        self.assertEqual(stats['total_documents'], 100)


class TestTokenChunking(unittest.TestCase):
    """Test token-aware chunking"""

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.SentenceTransformer')
    @patch('vector_store.tiktoken.get_encoding')
    def test_token_chunking(self, mock_tiktoken, mock_sentence_transformer, mock_chroma_client):
        """Test token-aware text chunking"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=list(range(100)))  # 100 tokens
        mock_tokenizer.decode = Mock(side_effect=lambda x: f"chunk_{len(x)}_tokens")
        mock_tiktoken.return_value = mock_tokenizer

        # Mock ChromaDB
        mock_collection = Mock()
        mock_collection.count = Mock(return_value=0)
        mock_client_instance = Mock()
        mock_client_instance.create_collection = Mock(return_value=mock_collection)
        mock_chroma_client.return_value = mock_client_instance

        # Mock embedding model
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # Create VectorStore with token chunking
        vs = VectorStore(
            collection_name="test",
            use_token_chunking=True,
            chunk_size=30,
            chunk_overlap=5
        )

        # Test chunking
        text = "A" * 1000
        chunks = vs.chunk_text_by_tokens(text, chunk_size=30, overlap=5)

        # Verify
        self.assertTrue(len(chunks) > 1)
        mock_tokenizer.encode.assert_called()
        mock_tokenizer.decode.assert_called()


if __name__ == '__main__':
    unittest.main()
