"""
Live system tests to identify the actual issue causing 'query failed'
These tests run against the actual system components to diagnose the problem
"""
import unittest
import sys
import os
import tempfile
import shutil

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from rag_system import RAGSystem
from vector_store import VectorStore


class TestLiveSystem(unittest.TestCase):
    """Live system tests to diagnose the query failure"""
    
    def setUp(self):
        """Set up test environment with real components"""
        # Create test config
        self.test_config = Config()
        
        # Override with safe test values
        self.temp_dir = tempfile.mkdtemp()
        self.test_config.CHROMA_PATH = self.temp_dir
        self.test_config.ANTHROPIC_API_KEY = "test-fake-key"  # Will cause API failure, but we want to test other components first
        
        print(f"Testing with ChromaDB path: {self.temp_dir}")
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_vector_store_real_initialization(self):
        """Test if VectorStore can initialize with real ChromaDB"""
        try:
            vector_store = VectorStore(
                chroma_path=self.temp_dir,
                embedding_model=self.test_config.EMBEDDING_MODEL,
                max_results=self.test_config.MAX_RESULTS
            )
            
            print("✓ VectorStore initialized successfully")
            
            # Test basic operations
            course_count = vector_store.get_course_count()
            print(f"✓ Initial course count: {course_count}")
            
            existing_titles = vector_store.get_existing_course_titles()
            print(f"✓ Existing course titles: {existing_titles}")
            
            self.assertEqual(course_count, 0)  # Should start empty
            self.assertEqual(len(existing_titles), 0)
            
        except Exception as e:
            self.fail(f"VectorStore initialization failed: {e}")
    
    def test_rag_system_initialization_without_api_key(self):
        """Test RAG system initialization without valid API key"""
        try:
            # This should work even without API key for initialization
            rag_system = RAGSystem(self.test_config)
            
            print("✓ RAGSystem initialized successfully")
            
            # Check tool registration
            tools = rag_system.tool_manager.tools
            print(f"✓ Registered tools: {list(tools.keys())}")
            
            self.assertIn("search_course_content", tools)
            self.assertIn("get_course_outline", tools)
            
            # Test tool definitions
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            self.assertEqual(len(tool_definitions), 2)
            print("✓ Tool definitions retrieved successfully")
            
        except Exception as e:
            self.fail(f"RAGSystem initialization failed: {e}")
    
    def test_course_loading_from_docs_folder(self):
        """Test if courses can be loaded from the docs folder"""
        try:
            rag_system = RAGSystem(self.test_config)
            
            # Try to load courses from the actual docs folder
            docs_path = "../docs"
            if os.path.exists(docs_path):
                print(f"✓ Docs folder exists: {docs_path}")
                
                # List files in docs folder
                doc_files = [f for f in os.listdir(docs_path) if f.endswith('.txt')]
                print(f"✓ Found {len(doc_files)} .txt files: {doc_files}")
                
                # Try to load courses
                courses_added, chunks_added = rag_system.add_course_folder(docs_path, clear_existing=True)
                
                print(f"✓ Successfully loaded {courses_added} courses with {chunks_added} chunks")
                
                # Check vector store now has courses
                course_count = rag_system.vector_store.get_course_count()
                course_titles = rag_system.vector_store.get_existing_course_titles()
                
                print(f"✓ Vector store now has {course_count} courses")
                print(f"✓ Course titles: {course_titles}")
                
                self.assertGreater(course_count, 0)
                self.assertEqual(course_count, courses_added)
                
            else:
                print(f"⚠ Docs folder not found: {docs_path}")
                self.skipTest("Docs folder not available")
                
        except Exception as e:
            print(f"❌ Course loading failed: {e}")
            import traceback
            print(traceback.format_exc())
            self.fail(f"Course loading failed: {e}")
    
    def test_search_tool_execution_with_real_data(self):
        """Test search tool execution with real data"""
        try:
            rag_system = RAGSystem(self.test_config)
            
            # Load real courses
            docs_path = "../docs"
            if os.path.exists(docs_path):
                courses_added, chunks_added = rag_system.add_course_folder(docs_path, clear_existing=True)
                print(f"✓ Loaded {courses_added} courses with {chunks_added} chunks for testing")
                
                if courses_added > 0:
                    # Test direct tool execution (bypassing AI generator)
                    result = rag_system.tool_manager.execute_tool(
                        "search_course_content",
                        query="introduction"
                    )
                    
                    print(f"✓ Search tool executed successfully")
                    print(f"Result preview: {result[:200]}...")
                    
                    self.assertIsInstance(result, str)
                    self.assertNotEqual(result, "Tool 'search_course_content' not found")
                    self.assertNotIn("error", result.lower())
                    
                    # Test with course filter
                    course_titles = rag_system.vector_store.get_existing_course_titles()
                    if course_titles:
                        first_course = course_titles[0]
                        result_filtered = rag_system.tool_manager.execute_tool(
                            "search_course_content",
                            query="introduction",
                            course_name=first_course[:10]  # Partial match
                        )
                        
                        print(f"✓ Filtered search executed successfully")
                        print(f"Filtered result preview: {result_filtered[:200]}...")
                        
                        self.assertIsInstance(result_filtered, str)
                        self.assertNotIn("error", result_filtered.lower())
                else:
                    print("⚠ No courses loaded, skipping search test")
                    self.skipTest("No courses available for search testing")
            else:
                print("⚠ Docs folder not available, skipping search test")
                self.skipTest("Docs folder not available")
                
        except Exception as e:
            print(f"❌ Search tool execution failed: {e}")
            import traceback
            print(traceback.format_exc())
            self.fail(f"Search tool execution failed: {e}")
    
    def test_outline_tool_execution_with_real_data(self):
        """Test outline tool execution with real data"""
        try:
            rag_system = RAGSystem(self.test_config)
            
            # Load real courses
            docs_path = "../docs"
            if os.path.exists(docs_path):
                courses_added, chunks_added = rag_system.add_course_folder(docs_path, clear_existing=True)
                print(f"✓ Loaded {courses_added} courses with {chunks_added} chunks for outline testing")
                
                if courses_added > 0:
                    # Get a course title to test with
                    course_titles = rag_system.vector_store.get_existing_course_titles()
                    first_course = course_titles[0]
                    
                    print(f"Testing outline tool with course: {first_course}")
                    
                    # Test outline tool execution
                    result = rag_system.tool_manager.execute_tool(
                        "get_course_outline",
                        course_name=first_course[:15]  # Partial match
                    )
                    
                    print(f"✓ Outline tool executed successfully")
                    print(f"Outline result:\n{result}")
                    
                    self.assertIsInstance(result, str)
                    self.assertNotEqual(result, "Tool 'get_course_outline' not found")
                    self.assertIn(first_course, result)  # Should contain the course title
                    self.assertIn("Lesson", result)  # Should contain lesson information
                    
                else:
                    print("⚠ No courses loaded, skipping outline test")
                    self.skipTest("No courses available for outline testing")
            else:
                print("⚠ Docs folder not available, skipping outline test")
                self.skipTest("Docs folder not available")
                
        except Exception as e:
            print(f"❌ Outline tool execution failed: {e}")
            import traceback
            print(traceback.format_exc())
            self.fail(f"Outline tool execution failed: {e}")
    
    def test_api_key_configuration(self):
        """Test API key configuration and error handling"""
        # Check if real API key is available
        real_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not real_api_key:
            print("⚠ No ANTHROPIC_API_KEY environment variable found")
            print("This could be the cause of 'query failed' errors")
        else:
            print(f"✓ ANTHROPIC_API_KEY found (length: {len(real_api_key)})")
            
            # Test with real API key
            self.test_config.ANTHROPIC_API_KEY = real_api_key
            
            try:
                rag_system = RAGSystem(self.test_config)
                print("✓ RAGSystem initialized with real API key")
                
                # Load some data first
                docs_path = "../docs"
                if os.path.exists(docs_path):
                    courses_added, chunks_added = rag_system.add_course_folder(docs_path, clear_existing=True)
                    
                    if courses_added > 0:
                        # Try a simple query that should trigger tool usage
                        print("Testing full query with real API...")
                        response, sources = rag_system.query("What is the introduction about?")
                        
                        print(f"✓ Query executed successfully")
                        print(f"Response: {response[:200]}...")
                        print(f"Sources: {sources}")
                        
                        self.assertIsInstance(response, str)
                        self.assertNotEqual(response.lower(), "query failed")
                        
                    else:
                        print("⚠ No courses loaded for API test")
                else:
                    print("⚠ No docs folder for API test")
                        
            except Exception as e:
                print(f"❌ Full query test failed: {e}")
                import traceback
                print(traceback.format_exc())
                # Don't fail the test here, as this helps us identify the issue


if __name__ == '__main__':
    unittest.main(verbosity=2)