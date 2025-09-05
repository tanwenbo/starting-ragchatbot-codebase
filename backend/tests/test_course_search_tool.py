import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool
from vector_store import VectorStore, SearchResults
from models import Course, Lesson


class TestCourseSearchTool(unittest.TestCase):
    """Test cases for CourseSearchTool functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock vector store
        self.mock_vector_store = Mock(spec=VectorStore)
        
        # Create CourseSearchTool instance
        self.search_tool = CourseSearchTool(self.mock_vector_store)
        
        # Sample test data
        self.sample_course_title = "Test Course: Introduction to Testing"
        self.sample_results = SearchResults(
            documents=[
                "This is the introduction to our test course.",
                "In this lesson, we'll get started with basic concepts."
            ],
            metadata=[
                {
                    "course_title": self.sample_course_title,
                    "lesson_number": 0,
                    "chunk_index": 0
                },
                {
                    "course_title": self.sample_course_title,
                    "lesson_number": 1,
                    "chunk_index": 1
                }
            ],
            distances=[0.1, 0.2]
        )
    
    def test_get_tool_definition(self):
        """Test tool definition structure"""
        definition = self.search_tool.get_tool_definition()
        
        # Check basic structure
        self.assertIsInstance(definition, dict)
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)
        
        # Check input schema
        schema = definition["input_schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertEqual(schema["required"], ["query"])
        
        # Check properties
        properties = schema["properties"]
        self.assertIn("query", properties)
        self.assertIn("course_name", properties)
        self.assertIn("lesson_number", properties)
        
        # Check query property
        self.assertEqual(properties["query"]["type"], "string")
        
        # Check course_name property
        self.assertEqual(properties["course_name"]["type"], "string")
        
        # Check lesson_number property
        self.assertEqual(properties["lesson_number"]["type"], "integer")
    
    def test_execute_successful_search(self):
        """Test successful search execution"""
        # Setup mock
        self.mock_vector_store.search.return_value = self.sample_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson0"
        
        # Execute search
        result = self.search_tool.execute("test query")
        
        # Assertions
        self.assertIsInstance(result, str)
        self.assertIn("Test Course: Introduction to Testing", result)
        self.assertIn("This is the introduction to our test course", result)
        self.assertIn("In this lesson, we'll get started", result)
        
        # Check that vector store search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Check that sources were tracked
        self.assertIsInstance(self.search_tool.last_sources, list)
        self.assertEqual(len(self.search_tool.last_sources), 2)
    
    def test_execute_with_course_filter(self):
        """Test search execution with course name filter"""
        # Setup mock
        self.mock_vector_store.search.return_value = self.sample_results
        
        # Execute search with course filter
        result = self.search_tool.execute("test query", course_name="Test Course")
        
        # Check that vector store search was called with course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=None
        )
        
        # Check result is formatted properly
        self.assertIsInstance(result, str)
        self.assertIn("Test Course: Introduction to Testing", result)
    
    def test_execute_with_lesson_filter(self):
        """Test search execution with lesson number filter"""
        # Setup mock
        self.mock_vector_store.search.return_value = self.sample_results
        
        # Execute search with lesson filter
        result = self.search_tool.execute("test query", lesson_number=1)
        
        # Check that vector store search was called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )
        
        # Check result is formatted properly
        self.assertIsInstance(result, str)
    
    def test_execute_with_both_filters(self):
        """Test search execution with both course and lesson filters"""
        # Setup mock
        self.mock_vector_store.search.return_value = self.sample_results
        
        # Execute search with both filters
        result = self.search_tool.execute(
            "test query", 
            course_name="Test Course",
            lesson_number=1
        )
        
        # Check that vector store search was called with both filters
        self.mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=1
        )
    
    def test_execute_with_error(self):
        """Test search execution when vector store returns error"""
        # Setup mock with error
        error_results = SearchResults.empty("Test error message")
        self.mock_vector_store.search.return_value = error_results
        
        # Execute search
        result = self.search_tool.execute("test query")
        
        # Check error is returned
        self.assertEqual(result, "Test error message")
        
        # Check no sources were tracked
        self.assertEqual(len(self.search_tool.last_sources), 0)
    
    def test_execute_with_empty_results(self):
        """Test search execution with empty results"""
        # Setup mock with empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = empty_results
        
        # Execute search
        result = self.search_tool.execute("nonexistent query")
        
        # Check no results message
        self.assertEqual(result, "No relevant content found.")
        
        # Check no sources were tracked
        self.assertEqual(len(self.search_tool.last_sources), 0)
    
    def test_execute_empty_results_with_filters(self):
        """Test search execution with empty results and filters"""
        # Setup mock with empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = empty_results
        
        # Execute search with filters
        result = self.search_tool.execute(
            "nonexistent query",
            course_name="Test Course",
            lesson_number=1
        )
        
        # Check no results message includes filter info
        expected = "No relevant content found in course 'Test Course' in lesson 1."
        self.assertEqual(result, expected)
    
    def test_format_results_with_links(self):
        """Test result formatting with lesson links"""
        # Setup mock to return lesson links
        self.mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson0",
            "https://example.com/lesson1"
        ]
        
        # Format results
        formatted = self.search_tool._format_results(self.sample_results)
        
        # Check formatting
        self.assertIsInstance(formatted, str)
        self.assertIn("[[Test Course: Introduction to Testing - Lesson 0]]", formatted)
        self.assertIn("https://example.com/lesson0", formatted)
        self.assertIn("[[Test Course: Introduction to Testing - Lesson 1]]", formatted)
        self.assertIn("https://example.com/lesson1", formatted)
        
        # Check that sources include links
        expected_sources = [
            "[Test Course: Introduction to Testing - Lesson 0](https://example.com/lesson0)",
            "[Test Course: Introduction to Testing - Lesson 1](https://example.com/lesson1)"
        ]
        self.assertEqual(self.search_tool.last_sources, expected_sources)
    
    def test_format_results_without_links(self):
        """Test result formatting without lesson links"""
        # Setup mock to return no lesson links
        self.mock_vector_store.get_lesson_link.return_value = None
        
        # Format results
        formatted = self.search_tool._format_results(self.sample_results)
        
        # Check formatting without links
        self.assertIsInstance(formatted, str)
        self.assertIn("[Test Course: Introduction to Testing - Lesson 0]", formatted)
        self.assertIn("[Test Course: Introduction to Testing - Lesson 1]", formatted)
        
        # Check that sources don't include links
        expected_sources = [
            "Test Course: Introduction to Testing - Lesson 0",
            "Test Course: Introduction to Testing - Lesson 1"
        ]
        self.assertEqual(self.search_tool.last_sources, expected_sources)
    
    def test_format_results_with_unknown_course(self):
        """Test result formatting with unknown course title"""
        # Create results with unknown course
        unknown_results = SearchResults(
            documents=["Test content"],
            metadata=[{
                "course_title": "unknown",
                "lesson_number": None,
                "chunk_index": 0
            }],
            distances=[0.1]
        )
        
        # Format results
        formatted = self.search_tool._format_results(unknown_results)
        
        # Check formatting
        self.assertIn("[unknown]", formatted)
        
        # Check sources
        self.assertEqual(self.search_tool.last_sources, ["unknown"])
    
    def test_source_tracking_reset(self):
        """Test that sources are properly tracked and can be reset"""
        # Setup mock
        self.mock_vector_store.search.return_value = self.sample_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson0"
        
        # Execute search
        self.search_tool.execute("test query")
        
        # Check sources were tracked
        self.assertTrue(len(self.search_tool.last_sources) > 0)
        
        # Reset sources
        self.search_tool.last_sources = []
        
        # Check sources were reset
        self.assertEqual(len(self.search_tool.last_sources), 0)
    
    def test_multiple_document_formatting(self):
        """Test formatting with multiple documents"""
        # Setup mock
        self.mock_vector_store.get_lesson_link.return_value = None
        
        # Format results
        formatted = self.search_tool._format_results(self.sample_results)
        
        # Check that multiple documents are separated correctly
        parts = formatted.split("\n\n")
        self.assertEqual(len(parts), 2)
        
        # Each part should have header and content
        for part in parts:
            self.assertIn("[Test Course: Introduction to Testing", part)
            self.assertIn("Lesson", part)


if __name__ == '__main__':
    unittest.main()