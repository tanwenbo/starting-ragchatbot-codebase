import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os
import sys

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_embedding_model = "all-MiniLM-L6-v2"
        self.max_results = 5
        
        # Sample test data
        self.sample_lessons = [
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
            Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson1")
        ]
        
        self.sample_course = Course(
            title="Test Course: Introduction to Testing",
            course_link="https://example.com/course",
            instructor="Test Instructor",
            lessons=self.sample_lessons
        )
        
        self.sample_chunks = [
            CourseChunk(
                content="This is the introduction to our test course.",
                course_title=self.sample_course.title,
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="In this lesson, we'll get started with basic concepts.",
                course_title=self.sample_course.title,
                lesson_number=1,
                chunk_index=1
            )
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_vector_store_initialization(self, mock_embedding_func, mock_client):
        """Test VectorStore initialization"""
        # Mock ChromaDB components
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_collection = Mock()
        mock_client_instance.get_or_create_collection.return_value = mock_collection
        
        # Create VectorStore
        vector_store = VectorStore(
            chroma_path=self.temp_dir,
            embedding_model=self.test_embedding_model,
            max_results=self.max_results
        )
        
        # Assertions
        self.assertIsNotNone(vector_store)
        self.assertEqual(vector_store.max_results, self.max_results)
        mock_client.assert_called_once()
        mock_embedding_func.assert_called_once_with(model_name=self.test_embedding_model)
        # Should create two collections
        self.assertEqual(mock_client_instance.get_or_create_collection.call_count, 2)
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_metadata(self, mock_embedding_func, mock_client):
        """Test adding course metadata to vector store"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Create VectorStore
        vector_store = VectorStore(self.temp_dir, self.test_embedding_model)
        
        # Add course metadata
        vector_store.add_course_metadata(self.sample_course)
        
        # Verify catalog.add was called with correct parameters
        mock_catalog.add.assert_called_once()
        call_args = mock_catalog.add.call_args
        
        # Check the call arguments
        self.assertEqual(call_args[1]['documents'], [self.sample_course.title])
        self.assertEqual(call_args[1]['ids'], [self.sample_course.title])
        
        # Check metadata structure
        metadata = call_args[1]['metadatas'][0]
        self.assertEqual(metadata['title'], self.sample_course.title)
        self.assertEqual(metadata['instructor'], self.sample_course.instructor)
        self.assertEqual(metadata['course_link'], self.sample_course.course_link)
        self.assertEqual(metadata['lesson_count'], len(self.sample_course.lessons))
        self.assertIn('lessons_json', metadata)
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_content(self, mock_embedding_func, mock_client):
        """Test adding course content chunks to vector store"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Create VectorStore
        vector_store = VectorStore(self.temp_dir, self.test_embedding_model)
        
        # Add course content
        vector_store.add_course_content(self.sample_chunks)
        
        # Verify content.add was called
        mock_content.add.assert_called_once()
        call_args = mock_content.add.call_args
        
        # Check documents
        expected_documents = [chunk.content for chunk in self.sample_chunks]
        self.assertEqual(call_args[1]['documents'], expected_documents)
        
        # Check metadata
        expected_metadata = [{
            "course_title": chunk.course_title,
            "lesson_number": chunk.lesson_number,
            "chunk_index": chunk.chunk_index
        } for chunk in self.sample_chunks]
        self.assertEqual(call_args[1]['metadatas'], expected_metadata)
        
        # Check IDs format
        expected_ids = [f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}" 
                       for chunk in self.sample_chunks]
        self.assertEqual(call_args[1]['ids'], expected_ids)
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_success(self, mock_embedding_func, mock_client):
        """Test successful search operation"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock successful search results
        mock_search_results = {
            'documents': [["Test document content", "Another document"]],
            'metadatas': [[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ]],
            'distances': [[0.1, 0.2]]
        }
        mock_content.query.return_value = mock_search_results
        
        # Mock course name resolution
        mock_catalog.query.return_value = {
            'documents': [["Test Course"]],
            'metadatas': [[{"title": "Test Course: Introduction to Testing"}]]
        }
        
        # Create VectorStore
        vector_store = VectorStore(self.temp_dir, self.test_embedding_model)
        
        # Perform search
        results = vector_store.search("test query", course_name="Test Course")
        
        # Assertions
        self.assertIsInstance(results, SearchResults)
        self.assertIsNone(results.error)
        self.assertEqual(len(results.documents), 2)
        self.assertEqual(results.documents[0], "Test document content")
        self.assertEqual(len(results.metadata), 2)
        self.assertEqual(results.metadata[0]["course_title"], "Test Course")
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_no_course_found(self, mock_embedding_func, mock_client):
        """Test search when course name is not found"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock no course found in catalog
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        # Create VectorStore
        vector_store = VectorStore(self.temp_dir, self.test_embedding_model)
        
        # Perform search with non-existent course
        results = vector_store.search("test query", course_name="Nonexistent Course")
        
        # Assertions
        self.assertIsInstance(results, SearchResults)
        self.assertIsNotNone(results.error)
        self.assertIn("No course found matching", results.error)
        self.assertTrue(results.is_empty())
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_empty_results(self, mock_embedding_func, mock_client):
        """Test search with empty results"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock empty search results
        mock_content.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Create VectorStore
        vector_store = VectorStore(self.temp_dir, self.test_embedding_model)
        
        # Perform search
        results = vector_store.search("nonexistent query")
        
        # Assertions
        self.assertIsInstance(results, SearchResults)
        self.assertIsNone(results.error)
        self.assertTrue(results.is_empty())
        self.assertEqual(len(results.documents), 0)
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_exception(self, mock_embedding_func, mock_client):
        """Test search when ChromaDB raises an exception"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock exception during search
        mock_content.query.side_effect = Exception("ChromaDB connection failed")
        
        # Create VectorStore
        vector_store = VectorStore(self.temp_dir, self.test_embedding_model)
        
        # Perform search
        results = vector_store.search("test query")
        
        # Assertions
        self.assertIsInstance(results, SearchResults)
        self.assertIsNotNone(results.error)
        self.assertIn("Search error", results.error)
        self.assertIn("ChromaDB connection failed", results.error)
        self.assertTrue(results.is_empty())
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_existing_course_titles(self, mock_embedding_func, mock_client):
        """Test getting existing course titles"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock catalog.get() response
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        # Create VectorStore
        vector_store = VectorStore(self.temp_dir, self.test_embedding_model)
        
        # Get course titles
        titles = vector_store.get_existing_course_titles()
        
        # Assertions
        self.assertEqual(titles, ['Course 1', 'Course 2', 'Course 3'])
        mock_catalog.get.assert_called_once()
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_count(self, mock_embedding_func, mock_client):
        """Test getting course count"""
        # Setup mocks
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_client_instance.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        # Mock catalog.get() response
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        # Create VectorStore
        vector_store = VectorStore(self.temp_dir, self.test_embedding_model)
        
        # Get course count
        count = vector_store.get_course_count()
        
        # Assertions
        self.assertEqual(count, 3)
        mock_catalog.get.assert_called_once()


if __name__ == '__main__':
    unittest.main()