import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


class TestRAGSystem(unittest.TestCase):
    """Test cases for RAGSystem integration functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test config
        self.test_config = Config()
        self.test_config.ANTHROPIC_API_KEY = "test-api-key"
        self.test_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.test_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.test_config.CHUNK_SIZE = 800
        self.test_config.CHUNK_OVERLAP = 100
        self.test_config.MAX_RESULTS = 5
        self.test_config.MAX_HISTORY = 2
        
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        self.test_config.CHROMA_PATH = self.temp_dir
        
        # Sample test data
        self.sample_course = Course(
            title="Test Course: Introduction to Testing",
            course_link="https://example.com/course",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
                Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson1")
            ]
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_rag_system_initialization(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test RAGSystem initialization and component setup"""
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Check that all components were initialized
        mock_vector.assert_called_once_with(
            self.test_config.CHROMA_PATH,
            self.test_config.EMBEDDING_MODEL,
            self.test_config.MAX_RESULTS
        )
        mock_ai_gen.assert_called_once_with(
            self.test_config.ANTHROPIC_API_KEY,
            self.test_config.ANTHROPIC_MODEL
        )
        mock_doc_proc.assert_called_once_with(
            self.test_config.CHUNK_SIZE,
            self.test_config.CHUNK_OVERLAP
        )
        mock_session.assert_called_once_with(self.test_config.MAX_HISTORY)
        
        # Check that tools are registered
        self.assertIsNotNone(rag_system.tool_manager)
        self.assertIsNotNone(rag_system.search_tool)
        self.assertIsNotNone(rag_system.outline_tool)
        
        # Check tool registration
        tools = rag_system.tool_manager.tools
        self.assertIn("search_course_content", tools)
        self.assertIn("get_course_outline", tools)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_execution_flow(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test complete query execution flow"""
        # Setup mocks
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Test response"
        
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_conversation_history.return_value = "Previous conversation"
        
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Mock tool manager
        rag_system.tool_manager.get_last_sources = Mock(return_value=["source1", "source2"])
        rag_system.tool_manager.reset_sources = Mock()
        rag_system.tool_manager.get_tool_definitions = Mock(return_value=[{"name": "test_tool"}])
        
        # Execute query
        response, sources = rag_system.query("Test query", session_id="test_session")
        
        # Check AI generator was called correctly
        mock_ai_instance.generate_response.assert_called_once()
        call_args = mock_ai_instance.generate_response.call_args
        
        # Check query parameter
        expected_prompt = "Answer this question about course materials: Test query"
        self.assertEqual(call_args[1]['query'], expected_prompt)
        
        # Check conversation history
        self.assertEqual(call_args[1]['conversation_history'], "Previous conversation")
        
        # Check tools were provided
        self.assertEqual(call_args[1]['tools'], [{"name": "test_tool"}])
        
        # Check tool manager was provided
        self.assertEqual(call_args[1]['tool_manager'], rag_system.tool_manager)
        
        # Check response and sources
        self.assertEqual(response, "Test response")
        self.assertEqual(sources, ["source1", "source2"])
        
        # Check session management
        mock_session_instance.add_exchange.assert_called_once_with(
            "test_session", "Test query", "Test response"
        )
        
        # Check sources were reset
        rag_system.tool_manager.reset_sources.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_without_session(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test query execution without session ID"""
        # Setup mocks
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Test response"
        
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_conversation_history.return_value = None
        
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Mock tool manager
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        rag_system.tool_manager.get_tool_definitions = Mock(return_value=[])
        
        # Execute query without session
        response, sources = rag_system.query("Test query")
        
        # Check that history retrieval wasn't attempted
        mock_session_instance.get_conversation_history.assert_not_called()
        
        # Check that session wasn't updated
        mock_session_instance.add_exchange.assert_not_called()
        
        # Check AI generator was called with no history
        call_args = mock_ai_instance.generate_response.call_args
        self.assertIsNone(call_args[1]['conversation_history'])
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_add_course_document_success(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test successful course document addition"""
        # Setup mocks
        mock_doc_proc_instance = Mock()
        mock_doc_proc.return_value = mock_doc_proc_instance
        
        sample_chunks = [
            CourseChunk(content="Test content", course_title="Test Course", lesson_number=0, chunk_index=0)
        ]
        mock_doc_proc_instance.process_course_document.return_value = (self.sample_course, sample_chunks)
        
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Add course document
        course, chunk_count = rag_system.add_course_document("test_file.txt")
        
        # Check document processor was called
        mock_doc_proc_instance.process_course_document.assert_called_once_with("test_file.txt")
        
        # Check vector store operations
        mock_vector_instance.add_course_metadata.assert_called_once_with(self.sample_course)
        mock_vector_instance.add_course_content.assert_called_once_with(sample_chunks)
        
        # Check return values
        self.assertEqual(course, self.sample_course)
        self.assertEqual(chunk_count, 1)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_add_course_document_failure(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test course document addition failure"""
        # Setup mocks
        mock_doc_proc_instance = Mock()
        mock_doc_proc.return_value = mock_doc_proc_instance
        mock_doc_proc_instance.process_course_document.side_effect = Exception("Processing failed")
        
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Add course document (should handle exception)
        course, chunk_count = rag_system.add_course_document("bad_file.txt")
        
        # Check failure handling
        self.assertIsNone(course)
        self.assertEqual(chunk_count, 0)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_exists, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test successful course folder addition"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.txt", "README.md"]  # Only .txt files should be processed
        
        mock_doc_proc_instance = Mock()
        mock_doc_proc.return_value = mock_doc_proc_instance
        
        sample_chunks = [
            CourseChunk(content="Test content", course_title="Test Course", lesson_number=0, chunk_index=0)
        ]
        mock_doc_proc_instance.process_course_document.return_value = (self.sample_course, sample_chunks)
        
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        mock_vector_instance.get_existing_course_titles.return_value = []  # No existing courses
        
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Add course folder
        total_courses, total_chunks = rag_system.add_course_folder("test_folder")
        
        # Check that only .txt files were processed (2 files)
        self.assertEqual(mock_doc_proc_instance.process_course_document.call_count, 2)
        
        # Check return values (2 courses, each with 1 chunk)
        self.assertEqual(total_courses, 2)
        self.assertEqual(total_chunks, 2)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    def test_add_course_folder_nonexistent(self, mock_exists, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test course folder addition with nonexistent folder"""
        # Setup mocks
        mock_exists.return_value = False
        
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Add nonexistent course folder
        total_courses, total_chunks = rag_system.add_course_folder("nonexistent_folder")
        
        # Check failure handling
        self.assertEqual(total_courses, 0)
        self.assertEqual(total_chunks, 0)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test course analytics retrieval"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        mock_vector_instance.get_course_count.return_value = 3
        mock_vector_instance.get_existing_course_titles.return_value = ["Course 1", "Course 2", "Course 3"]
        
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Get analytics
        analytics = rag_system.get_course_analytics()
        
        # Check analytics structure
        self.assertIsInstance(analytics, dict)
        self.assertEqual(analytics["total_courses"], 3)
        self.assertEqual(analytics["course_titles"], ["Course 1", "Course 2", "Course 3"])
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_tool_registration_completeness(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test that all required tools are properly registered"""
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Check tool manager exists
        self.assertIsNotNone(rag_system.tool_manager)
        
        # Check tools are registered
        tools = rag_system.tool_manager.tools
        self.assertIn("search_course_content", tools)
        self.assertIn("get_course_outline", tools)
        
        # Check tool definitions can be retrieved
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        self.assertEqual(len(tool_definitions), 2)
        
        # Check tool names in definitions
        tool_names = [defn["name"] for defn in tool_definitions]
        self.assertIn("search_course_content", tool_names)
        self.assertIn("get_course_outline", tool_names)
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_tool_execution_integration(self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector):
        """Test that tools can be executed through the tool manager"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        
        # Mock search results
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 0}],
            distances=[0.1]
        )
        mock_vector_instance.search.return_value = search_results
        
        # Create RAGSystem
        rag_system = RAGSystem(self.test_config)
        
        # Execute search tool
        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="test query"
        )
        
        # Check tool executed successfully
        self.assertIsInstance(result, str)
        self.assertIn("Test content", result)
        
        # Check vector store was called
        mock_vector_instance.search.assert_called_once()


if __name__ == '__main__':
    unittest.main()