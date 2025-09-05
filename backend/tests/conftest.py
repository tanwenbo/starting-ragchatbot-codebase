import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
import sys

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from config import Config


@pytest.fixture
def sample_lessons():
    """Sample lesson data for testing"""
    return [
        Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
        Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
    ]


@pytest.fixture
def sample_course(sample_lessons):
    """Sample course data for testing"""
    return Course(
        title="Test Course: Introduction to Testing",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=sample_lessons
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is the introduction to our test course. We'll learn about testing methodologies.",
            course_title=sample_course.title,
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="In this lesson, we'll get started with basic concepts and terminology.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced topics include mocking, fixtures, and integration testing strategies.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "This is the introduction to our test course. We'll learn about testing methodologies.",
            "In this lesson, we'll get started with basic concepts and terminology."
        ],
        metadata=[
            {
                "course_title": "Test Course: Introduction to Testing",
                "lesson_number": 0,
                "chunk_index": 0
            },
            {
                "course_title": "Test Course: Introduction to Testing", 
                "lesson_number": 1,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = ":memory:"  # Use in-memory for tests
    return config


@pytest.fixture
def temp_chroma_path():
    """Temporary ChromaDB path for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    with patch('anthropic.Anthropic') as mock_client:
        # Mock successful response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        
        # Mock tool use response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test query"}
        mock_tool_response.content = [mock_tool_content]
        
        mock_client_instance = Mock()
        mock_client_instance.messages.create.side_effect = [mock_response]
        mock_client.return_value = mock_client_instance
        
        yield mock_client_instance


@pytest.fixture
def mock_vector_store(sample_course, sample_course_chunks, sample_search_results):
    """Mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Mock search method
    mock_store.search.return_value = sample_search_results
    
    # Mock course resolution
    mock_store._resolve_course_name.return_value = sample_course.title
    
    # Mock metadata methods
    mock_store.get_all_courses_metadata.return_value = [
        {
            "title": sample_course.title,
            "instructor": sample_course.instructor,
            "course_link": sample_course.course_link,
            "lessons": [
                {
                    "lesson_number": lesson.lesson_number,
                    "lesson_title": lesson.title,
                    "lesson_link": lesson.lesson_link
                }
                for lesson in sample_course.lessons
            ]
        }
    ]
    
    # Mock link methods
    mock_store.get_course_link.return_value = sample_course.course_link
    mock_store.get_lesson_link.return_value = sample_course.lessons[0].lesson_link
    
    # Mock course count and titles
    mock_store.get_course_count.return_value = 1
    mock_store.get_existing_course_titles.return_value = [sample_course.title]
    
    return mock_store


@pytest.fixture
def mock_empty_search_results():
    """Mock empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def mock_error_search_results():
    """Mock error search results for testing"""
    return SearchResults.empty("Test error message")


@pytest.fixture
def sample_course_document():
    """Sample course document content for testing"""
    return """Course Title: Test Course: Introduction to Testing
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0
This is the introduction to our test course. We'll learn about testing methodologies.
Testing is crucial for software development and ensures code reliability.

Lesson 1: Getting Started
Lesson Link: https://example.com/lesson1
In this lesson, we'll get started with basic concepts and terminology.
Unit tests focus on individual components while integration tests check interactions.

Lesson 2: Advanced Topics
Lesson Link: https://example.com/lesson2
Advanced topics include mocking, fixtures, and integration testing strategies.
These techniques help create robust and maintainable test suites.
"""


@pytest.fixture
def enable_test_mode():
    """Set environment variables for testing"""
    original_env = os.environ.copy()
    os.environ['TESTING'] = 'true'
    os.environ['ANTHROPIC_API_KEY'] = 'test-api-key'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)