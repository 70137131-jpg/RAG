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
        from app import app as flask_app, rate_limit_state

        self.app = flask_app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        rate_limit_state.clear()

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

    @patch('app.MAX_QUESTION_LENGTH', 5)
    def test_query_rejects_long_question(self):
        """Test query endpoint rejects oversized questions"""
        with self.app.test_request_context():
            response = self.client.post(
                '/api/query',
                data=json.dumps({'question': 'What is AI?'}),
                content_type='application/json'
            )

            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertFalse(data['success'])
            self.assertIn('exceeds', data['error'])

    @patch('app.initialize_rag')
    def test_query_backend_error_is_sanitized(self, mock_init_rag):
        """Internal backend errors should not leak to API clients"""
        mock_init_rag.side_effect = RuntimeError("internal vector store path")

        with self.app.test_request_context():
            response = self.client.post(
                '/api/query',
                data=json.dumps({'question': 'What is AI?'}),
                content_type='application/json'
            )

            self.assertEqual(response.status_code, 500)
            data = json.loads(response.data)
            self.assertEqual(data['error'], 'Failed to process query')
            self.assertNotIn('internal vector store path', data['error'])

    @patch('app.RATE_LIMIT_REQUESTS', 1)
    @patch('app.RATE_LIMIT_WINDOW_SECONDS', 60)
    @patch('app.initialize_rag')
    def test_query_rate_limit(self, mock_init_rag):
        """Test simple per-client query rate limiting"""
        mock_init_rag.return_value = self.mock_rag

        with self.app.test_request_context():
            first = self.client.post(
                '/api/query',
                data=json.dumps({'question': 'What is AI?'}),
                content_type='application/json'
            )
            second = self.client.post(
                '/api/query',
                data=json.dumps({'question': 'What is AI?'}),
                content_type='application/json'
            )

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 429)
            self.assertIn('Retry-After', second.headers)

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

    @patch.dict('os.environ', {'ADMIN_TOKEN': 'secret'})
    @patch('app.initialize_rag')
    def test_stats_endpoint_requires_admin_token_when_configured(self, mock_init_rag):
        """Stats endpoint is public by default, protected when ADMIN_TOKEN is set"""
        mock_init_rag.return_value = self.mock_rag

        with self.app.test_request_context():
            unauthorized = self.client.get('/api/stats')
            authorized = self.client.get('/api/stats', headers={'X-Admin-Token': 'secret'})

            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(authorized.status_code, 200)

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

    @patch.dict('os.environ', {'ADMIN_TOKEN': 'secret'})
    @patch('app.initialize_rag')
    def test_metrics_endpoint_requires_admin_token_when_configured(self, mock_init_rag):
        """Metrics endpoint is protected when ADMIN_TOKEN is set"""
        mock_init_rag.return_value = self.mock_rag

        with self.app.test_request_context():
            unauthorized = self.client.get('/metrics')
            authorized = self.client.get('/metrics', headers={'Authorization': 'Bearer secret'})

            self.assertEqual(unauthorized.status_code, 401)
            self.assertEqual(authorized.status_code, 200)


if __name__ == '__main__':
    unittest.main()
