"""
Tests for data_loader module
"""

import unittest
from unittest.mock import Mock, patch
from data_loader import SQuADLoader


class TestSQuADLoader(unittest.TestCase):
    """Test cases for SQuADLoader"""

    def setUp(self):
        """Set up test fixtures"""
        self.loader = SQuADLoader(dataset_name="squad_v2", split="validation")

    def test_initialization(self):
        """Test loader initialization"""
        self.assertEqual(self.loader.dataset_name, "squad_v2")
        self.assertEqual(self.loader.split, "validation")
        self.assertIsNone(self.loader.dataset)

    @patch('data_loader.load_dataset')
    def test_load_with_max_samples(self, mock_load_dataset):
        """Test loading dataset with max_samples limit"""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.select = Mock(return_value=mock_dataset)
        mock_load_dataset.return_value = mock_dataset

        # Load with max_samples
        result = self.loader.load(max_samples=50)

        # Verify
        mock_load_dataset.assert_called_once_with("squad_v2", split="validation")
        mock_dataset.select.assert_called_once()
        self.assertIsNotNone(self.loader.dataset)

    @patch('data_loader.load_dataset')
    def test_get_contexts(self, mock_load_dataset):
        """Test extracting unique contexts"""
        # Mock dataset with duplicate contexts
        mock_data = [
            {'context': 'Context 1', 'question': 'Q1', 'answers': {'text': ['A1'], 'answer_start': [0]}},
            {'context': 'Context 1', 'question': 'Q2', 'answers': {'text': ['A2'], 'answer_start': [0]}},
            {'context': 'Context 2', 'question': 'Q3', 'answers': {'text': ['A3'], 'answer_start': [0]}},
        ]

        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        mock_dataset.__len__ = Mock(return_value=3)
        mock_load_dataset.return_value = mock_dataset

        # Load and get contexts
        self.loader.load()
        contexts = self.loader.get_contexts()

        # Should have 2 unique contexts
        self.assertEqual(len(contexts), 2)
        self.assertTrue(all('id' in ctx and 'text' in ctx for ctx in contexts))

    @patch('data_loader.load_dataset')
    def test_get_qa_pairs(self, mock_load_dataset):
        """Test extracting QA pairs"""
        # Mock dataset with answerable and unanswerable questions
        mock_data = [
            {
                'id': '1',
                'context': 'Context 1',
                'question': 'Q1',
                'answers': {'text': ['Answer 1'], 'answer_start': [0]}
            },
            {
                'id': '2',
                'context': 'Context 2',
                'question': 'Q2',
                'answers': {'text': [], 'answer_start': []}  # Unanswerable
            },
            {
                'id': '3',
                'context': 'Context 3',
                'question': 'Q3',
                'answers': {'text': ['Answer 3'], 'answer_start': [0]}
            },
        ]

        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_data))
        mock_dataset.__len__ = Mock(return_value=3)
        mock_load_dataset.return_value = mock_dataset

        # Load and get QA pairs
        self.loader.load()
        qa_pairs = self.loader.get_qa_pairs()

        # Should have 2 answerable QA pairs
        self.assertEqual(len(qa_pairs), 2)
        for qa in qa_pairs:
            self.assertIn('id', qa)
            self.assertIn('question', qa)
            self.assertIn('answers', qa)
            self.assertTrue(len(qa['answers']) > 0)


if __name__ == '__main__':
    unittest.main()
