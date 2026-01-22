"""
Tests for Flask web application
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json


class TestFlaskApp(unittest.TestCase):
    """Test cases for Flask application endpoints"""

    @patch('app.initialize_rag')
    def setUp(self, mock_init_rag):
        """Set up test fixtures"""
        # Import app here to avoid initialization issues
        from app import app as flask_app

        self.app = flask_app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

        # Mock RAG system
        self.mock_rag = Mock()
        self.mock_rag.query = Mock(return_value={
            'answer': 'Test answer',
            'metadata': {
                'retrieved_docs': [
                    {'id': 'doc1', 'text': 'Context 1', 'distance': 0.1},
                    {'id': 'doc2', 'text': 'Context 2', 'distance': 0.2},
                ],
                'num_contexts': 2,
                'model': 'test-model'
            }
        })
        self.mock_rag.vector_store = Mock()
        self.mock_rag.vector_store.collection = Mock()
        self.mock_rag.vector_store.collection.count = Mock(return_value=100)
        self.mock_rag.vector_store.get_stats = Mock(return_value={
            'total_documents': 100,
            'embedding_model': 'test-model',
            'collection_name': 'test-collection'
        })

        mock_init_rag.return_value = self.mock_rag

    def test_index_route(self):
        """Test index route"""
        with self.app.test_request_context():
            response = self.client.get('/')
            self.assertEqual(response.status_code, 200)

    @patch('app.config', None)
    @patch('app.initialize_rag')
    def test_query_endpoint(self, mock_init_rag):
        """Test query API endpoint"""
        mock_init_rag.return_value = self.mock_rag

        with self.app.test_request_context():
            response = self.client.post(
                '/api/query',
                data=json.dumps({'question': 'What is AI?'}),
                content_type='application/json'
            )

            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data['success'])
            self.assertIn('answer', data)
            self.assertIn('sources', data)

    def test_query_empty_question(self):
        """Test query endpoint with empty question"""
        with self.app.test_request_context():
            response = self.client.post(
                '/api/query',
                data=json.dumps({'question': ''}),
                content_type='application/json'
            )

            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertFalse(data['success'])

    @patch('app.initialize_rag')
    def test_health_endpoint(self, mock_init_rag):
        """Test health check endpoint"""
        mock_init_rag.return_value = self.mock_rag

        with self.app.test_request_context():
            response = self.client.get('/health')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data['status'], 'healthy')
            self.assertIn('documents_indexed', data)

    @patch('app.initialize_rag')
    def test_stats_endpoint(self, mock_init_rag):
        """Test stats API endpoint"""
        mock_init_rag.return_value = self.mock_rag

        with self.app.test_request_context():
            response = self.client.get('/api/stats')
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data['success'])
            self.assertIn('stats', data)
            self.assertIn('total_documents', data['stats'])

    @patch('app.initialize_rag')
    def test_metrics_endpoint(self, mock_init_rag):
        """Test Prometheus metrics endpoint"""
        mock_init_rag.return_value = self.mock_rag

        with self.app.test_request_context():
            response = self.client.get('/metrics')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content_type, 'text/plain; charset=utf-8')

            # Verify metrics format
            text = response.data.decode('utf-8')
            self.assertIn('rag_total_queries', text)
            self.assertIn('rag_documents_indexed', text)


if __name__ == '__main__':
    unittest.main()
