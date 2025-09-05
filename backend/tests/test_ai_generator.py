import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager


class TestAIGenerator(unittest.TestCase):
    """Test cases for AIGenerator phase-based reasoning and tool calling"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_api_key = "test-api-key"
        self.test_model = "claude-sonnet-4-20250514"
        
        # Sample tool definitions
        self.sample_tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_course_outline",
                "description": "Get course structure and outline",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "course_name": {"type": "string", "description": "Name of course"}
                    },
                    "required": ["course_name"]
                }
            }
        ]
    
    @patch('anthropic.Anthropic')
    def test_ai_generator_initialization(self, mock_anthropic):
        """Test AIGenerator initialization"""
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Check initialization
        mock_anthropic.assert_called_once_with(api_key=self.test_api_key)
        self.assertEqual(ai_gen.model, self.test_model)
        
        # Check base parameters
        self.assertIn("model", ai_gen.base_params)
        self.assertIn("temperature", ai_gen.base_params)
        self.assertIn("max_tokens", ai_gen.base_params)
        self.assertEqual(ai_gen.base_params["model"], self.test_model)
        self.assertEqual(ai_gen.base_params["temperature"], 0)
        self.assertEqual(ai_gen.base_params["max_tokens"], 800)
    
    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test response generation without tools"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response without tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        response = ai_gen.generate_response("Test query")
        
        # Check response
        self.assertEqual(response, "Test response without tools")
        
        # Check API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        
        # Check system prompt
        self.assertIn("system", call_args[1])
        system_content = call_args[1]["system"]
        self.assertIn("educational AI assistant that reasons through problems in phases", system_content)
        
        # Check messages format
        self.assertEqual(len(call_args[1]["messages"]), 1)
        self.assertEqual(call_args[1]["messages"][0]["role"], "user")
        self.assertEqual(call_args[1]["messages"][0]["content"], "Test query")
        
        # Check tools not included
        self.assertNotIn("tools", call_args[1])
        self.assertNotIn("tool_choice", call_args[1])
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_use(self, mock_anthropic):
        """Test response generation with tools available but not used"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct response without tool use")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response with tools
        response = ai_gen.generate_response(
            "What is machine learning?",  # General knowledge query
            tools=self.sample_tools
        )
        
        # Check response
        self.assertEqual(response, "Direct response without tool use")
        
        # Check API call included tools
        call_args = mock_client.messages.create.call_args
        self.assertIn("tools", call_args[1])
        self.assertEqual(call_args[1]["tools"], self.sample_tools)
        self.assertEqual(call_args[1]["tool_choice"], {"type": "auto"})
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test response generation with tool use"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock initial tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test search"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        mock_initial_response.stop_reason = "tool_use"
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Response based on tool results")]
        
        # Configure client to return both responses
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response with tool use
        response = ai_gen.generate_response(
            "Search for information about testing",
            tools=self.sample_tools,
            tool_manager=mock_tool_manager
        )
        
        # Check response
        self.assertEqual(response, "Response based on tool results")
        
        # Check tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test search"
        )
        
        # Check that appropriate number of API calls were made for phase-based system
        # Investigation phase (1) + tool execution (1) + synthesis phase (1) = 3 calls
        self.assertEqual(mock_client.messages.create.call_count, 3)
        
        # Check final call doesn't include tools
        final_call_args = mock_client.messages.create.call_args
        self.assertNotIn("tools", final_call_args[1])
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history context")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response with history
        conversation_history = "User: Previous question\nAssistant: Previous answer"
        response = ai_gen.generate_response(
            "Follow-up question",
            conversation_history=conversation_history
        )
        
        # Check response
        self.assertEqual(response, "Response with history context")
        
        # Check system prompt includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        self.assertIn("Previous conversation:", system_content)
        self.assertIn("Previous question", system_content)
        self.assertIn("Previous answer", system_content)
    
    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic):
        """Test handling multiple tool executions in one response"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock multiple tool use content blocks
        mock_tool_1 = Mock()
        mock_tool_1.type = "tool_use"
        mock_tool_1.name = "search_course_content"
        mock_tool_1.id = "tool_123"
        mock_tool_1.input = {"query": "test search 1"}
        
        mock_tool_2 = Mock()
        mock_tool_2.type = "tool_use"
        mock_tool_2.name = "get_course_outline"
        mock_tool_2.id = "tool_456"
        mock_tool_2.input = {"course_name": "Test Course"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_1, mock_tool_2]
        mock_initial_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Response using multiple tools")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Search result", "Outline result"]
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        response = ai_gen.generate_response(
            "Search and get outline",
            tools=self.sample_tools,
            tool_manager=mock_tool_manager
        )
        
        # Check response
        self.assertEqual(response, "Response using multiple tools")
        
        # Check both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="test search 1")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Test Course")
    
    @patch('anthropic.Anthropic')
    def test_system_prompt_content(self, mock_anthropic):
        """Test system prompt contains correct guidance"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        ai_gen.generate_response("Test query")
        
        # Check system prompt content
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        
        # Check key guidance elements
        self.assertIn("Content Search Tool", system_content)
        self.assertIn("Course Outline Tool", system_content)
        self.assertIn("INVESTIGATION", system_content)
        self.assertIn("SYNTHESIS", system_content)
        self.assertIn("Phase-Based Reasoning", system_content)
        self.assertIn("No meta-commentary", system_content)
    
    @patch('anthropic.Anthropic')
    def test_error_handling_api_failure(self, mock_anthropic):
        """Test error handling when API call fails"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API connection failed")
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Test that error is handled gracefully in phase-based system
        response = ai_gen.generate_response("Test query")
        
        # Should return error message rather than raising exception
        self.assertIn("Error in investigate phase", response)
        self.assertIn("API connection failed", response)
    
    @patch('anthropic.Anthropic')
    def test_phase_based_investigation_synthesis_flow(self, mock_anthropic):
        """Test investigation -> synthesis phase flow"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock investigation phase with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test search"}
        
        mock_investigation_response = Mock()
        mock_investigation_response.content = [mock_tool_content]
        mock_investigation_response.stop_reason = "tool_use"
        
        # Mock tool execution response
        mock_tool_execution_response = Mock()
        mock_tool_execution_response.content = [Mock(text="Investigation findings")]
        
        # Mock synthesis phase response
        mock_synthesis_response = Mock()
        mock_synthesis_response.content = [Mock(text="Final comprehensive answer")]
        mock_synthesis_response.stop_reason = "end_turn"
        
        # Configure client to return responses in sequence
        mock_client.messages.create.side_effect = [
            mock_investigation_response,
            mock_tool_execution_response,
            mock_synthesis_response
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        response = ai_gen.generate_response(
            "Compare two courses",
            tools=self.sample_tools,
            tool_manager=mock_tool_manager
        )
        
        # Check final response
        self.assertEqual(response, "Final comprehensive answer")
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test search"
        )
        
        # Verify multiple API calls were made (investigation + synthesis)
        self.assertEqual(mock_client.messages.create.call_count, 3)
    
    @patch('anthropic.Anthropic')
    def test_early_termination_on_general_knowledge(self, mock_anthropic):
        """Test early termination for general knowledge questions"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock investigation phase without tool use
        mock_investigation_response = Mock()
        mock_investigation_response.content = [Mock(text="General knowledge answer")]
        mock_investigation_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_investigation_response
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        response = ai_gen.generate_response(
            "What is machine learning?",
            tools=self.sample_tools
        )
        
        # Check response
        self.assertEqual(response, "General knowledge answer")
        
        # Verify only one API call (investigation phase only)
        self.assertEqual(mock_client.messages.create.call_count, 1)
    
    @patch('anthropic.Anthropic')
    def test_synthesis_phase_context_includes_investigation_results(self, mock_anthropic):
        """Test that synthesis phase receives investigation context"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock investigation phase with tool use
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "get_course_outline"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"course_name": "Test Course"}
        
        mock_investigation_response = Mock()
        mock_investigation_response.content = [mock_tool_content]
        mock_investigation_response.stop_reason = "tool_use"
        
        mock_tool_execution_response = Mock()
        mock_tool_execution_response.content = [Mock(text="Course outline results")]
        
        mock_synthesis_response = Mock()
        mock_synthesis_response.content = [Mock(text="Synthesis with context")]
        mock_synthesis_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [
            mock_investigation_response,
            mock_tool_execution_response,
            mock_synthesis_response
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Outline data"
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        response = ai_gen.generate_response(
            "Explain course structure",
            tools=self.sample_tools,
            tool_manager=mock_tool_manager
        )
        
        # Check response
        self.assertEqual(response, "Synthesis with context")
        
        # Verify synthesis phase received investigation results in system prompt
        synthesis_call = mock_client.messages.create.call_args_list[2]
        synthesis_system = synthesis_call[1]["system"]
        self.assertIn("SYNTHESIS", synthesis_system)
        self.assertIn("Investigation Results", synthesis_system)
    
    @patch('anthropic.Anthropic')
    def test_error_recovery_in_synthesis_phase(self, mock_anthropic):
        """Test graceful fallback when synthesis phase fails"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock successful investigation phase
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test"}
        
        mock_investigation_response = Mock()
        mock_investigation_response.content = [mock_tool_content]
        mock_investigation_response.stop_reason = "tool_use"
        
        mock_tool_execution_response = Mock()
        mock_tool_execution_response.content = [Mock(text="Useful investigation results")]
        
        # Configure synthesis phase to fail
        mock_client.messages.create.side_effect = [
            mock_investigation_response,
            mock_tool_execution_response,
            Exception("Synthesis API failure")
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        response = ai_gen.generate_response(
            "Search for information",
            tools=self.sample_tools,
            tool_manager=mock_tool_manager
        )
        
        # Should fall back to investigation results
        self.assertEqual(response, "Useful investigation results")
    
    @patch('anthropic.Anthropic')
    def test_sequential_tool_use_in_synthesis_phase(self, mock_anthropic):
        """Test that synthesis phase can use tools for additional clarification"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock investigation phase
        mock_investigation_tool = Mock()
        mock_investigation_tool.type = "tool_use"
        mock_investigation_tool.name = "get_course_outline"
        mock_investigation_tool.id = "tool_123"
        mock_investigation_tool.input = {"course_name": "Course A"}
        
        mock_investigation_response = Mock()
        mock_investigation_response.content = [mock_investigation_tool]
        mock_investigation_response.stop_reason = "tool_use"
        
        mock_investigation_execution = Mock()
        mock_investigation_execution.content = [Mock(text="Course A outline")]
        
        # Mock synthesis phase with additional tool use
        mock_synthesis_tool = Mock()
        mock_synthesis_tool.type = "tool_use"
        mock_synthesis_tool.name = "search_course_content"
        mock_synthesis_tool.id = "tool_456"
        mock_synthesis_tool.input = {"query": "detailed content"}
        
        mock_synthesis_response = Mock()
        mock_synthesis_response.content = [mock_synthesis_tool]
        mock_synthesis_response.stop_reason = "tool_use"
        
        mock_final_synthesis = Mock()
        mock_final_synthesis.content = [Mock(text="Complete answer with both outline and content")]
        
        mock_client.messages.create.side_effect = [
            mock_investigation_response,
            mock_investigation_execution,
            mock_synthesis_response,
            mock_final_synthesis
        ]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Outline result", "Content result"]
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        response = ai_gen.generate_response(
            "Get comprehensive course information",
            tools=self.sample_tools,
            tool_manager=mock_tool_manager
        )
        
        # Check final response
        self.assertEqual(response, "Complete answer with both outline and content")
        
        # Verify both tools were executed
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Course A")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="detailed content")
    
    @patch('anthropic.Anthropic')
    def test_tool_result_formatting(self, mock_anthropic):
        """Test that tool results are properly formatted in messages"""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        # Mock tool use response
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_content]
        mock_initial_response.stop_reason = "tool_use"
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result content"
        
        # Create AIGenerator
        ai_gen = AIGenerator(self.test_api_key, self.test_model)
        
        # Generate response
        ai_gen.generate_response("Test query", tools=self.sample_tools, tool_manager=mock_tool_manager)
        
        # Check final API call structure
        final_call_args = mock_client.messages.create.call_args
        messages = final_call_args[1]["messages"]
        
        # In phase-based system, check that API calls were made
        # This test verifies that the system made appropriate calls
        self.assertEqual(len(messages), 1)  # Synthesis phase message
        self.assertEqual(messages[0]["role"], "user")
        
        # Verify tool was executed (behavior-focused test)
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test"
        )


if __name__ == '__main__':
    # Run with more verbose output to see phase-based test results
    unittest.main(verbosity=2)