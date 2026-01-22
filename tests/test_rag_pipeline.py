"""
Tests for RAG pipeline
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    """Test cases for RAGPipeline"""

    @patch('rag_pipeline.OpenAI')
    @patch('rag_pipeline.load_dotenv')
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'sk-or-v1-test-api-key', 'MODEL': 'gpt-oss-20b'})
    def setUp(self, mock_load_dotenv, mock_openai):
        """Set up test fixtures"""
        # Mock vector store
        self.mock_vs = Mock()
        self.mock_vs.search = Mock(return_value=[
            {'id': 'doc1', 'text': 'Context 1', 'distance': 0.1},
            {'id': 'doc2', 'text': 'Context 2', 'distance': 0.2},
        ])

        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        # Include citation to avoid grounding note being added
        mock_message.content = "Mocked answer [Context 1]"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = Mock(return_value=mock_response)
        mock_openai.return_value = mock_client

        # Create RAGPipeline
        self.rag = RAGPipeline(
            vector_store=self.mock_vs,
            llm_model="gpt-oss-20b",
            temperature=0.1,
            max_tokens=300
        )

    def test_initialization(self):
        """Test RAGPipeline initialization"""
        self.assertEqual(self.rag.vector_store, self.mock_vs)
        self.assertEqual(self.rag.llm_model, "gpt-oss-20b")
        self.assertEqual(self.rag.temperature, 0.1)
        self.assertEqual(self.rag.max_tokens, 300)

    def test_retrieve(self):
        """Test document retrieval"""
        results = self.rag.retrieve("test query", top_k=2)

        # Verify vector store search was called
        self.mock_vs.search.assert_called_once_with("test query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['text'], 'Context 1')

    def test_generate(self):
        """Test answer generation"""
        contexts = ["Context 1", "Context 2"]
        answer = self.rag.generate("What is the answer?", contexts)

        # Verify answer contains the mocked answer (may have additional notes)
        self.assertIn("Mocked answer", answer)
        self.rag.client.chat.completions.create.assert_called_once()

    def test_query_basic(self):
        """Test end-to-end query"""
        result = self.rag.query("test question", top_k=2, return_metadata=False)

        # Verify structure
        self.assertIn('answer', result)
        self.assertIn("Mocked answer", result['answer'])

    def test_query_with_metadata(self):
        """Test query with metadata"""
        result = self.rag.query("test question", top_k=2, return_metadata=True)

        # Verify structure
        self.assertIn('answer', result)
        self.assertIn('metadata', result)
        self.assertIn('retrieved_docs', result['metadata'])
        self.assertIn('num_contexts', result['metadata'])
        self.assertEqual(result['metadata']['num_contexts'], 2)

    def test_batch_query(self):
        """Test batch query processing"""
        questions = ["Question 1", "Question 2", "Question 3"]
        results = self.rag.batch_query(questions, top_k=2)

        # Verify results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertEqual(result['question'], questions[i])
            self.assertIn('answer', result)
            self.assertIn('metadata', result)


class TestRAGPipelineMocked(unittest.TestCase):
    """Test RAG pipeline with fully mocked LLM"""

    def test_generate_with_custom_prompt(self):
        """Test generation with custom system prompt"""
        mock_vs = Mock()

        with patch('rag_pipeline.OpenAI') as mock_openai, \
             patch('rag_pipeline.load_dotenv'), \
             patch.dict('os.environ', {'OPENROUTER_API_KEY': 'sk-or-v1-test-key', 'MODEL': 'gpt-oss-20b'}):

            # Setup mock
            mock_client = Mock()
            mock_response = Mock()
            mock_choice = Mock()
            mock_message = Mock()
            mock_message.content = "Custom answer [Context 1]"
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create = Mock(return_value=mock_response)
            mock_openai.return_value = mock_client

            # Create pipeline
            rag = RAGPipeline(vector_store=mock_vs)

            # Generate with custom prompt
            custom_prompt = "You are a specialized assistant."
            contexts = ["context"]
            answer = rag.generate("test", contexts, system_prompt=custom_prompt)

            # Verify custom prompt was included in the user message (it's in the full_prompt)
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]['messages']
            user_message = messages[1]['content']
            self.assertIn(custom_prompt, user_message)
            self.assertIn("Custom answer", answer)


if __name__ == '__main__':
    unittest.main()
