import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Base system prompt for conversational reasoning phases
    BASE_SYSTEM_PROMPT = """You are an educational AI assistant that reasons through problems in phases to provide comprehensive answers.

**Phase-Based Reasoning:**
- INVESTIGATION: Gather relevant information using available tools when you need specific course content
- SYNTHESIS: Create comprehensive answers using all available context

**Tool Usage Philosophy:**
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, or course overviews
- Tools are available throughout your reasoning process
- Use judgment about when additional information would be helpful
- Focus on answering the user's actual question efficiently

**Response Guidelines:**
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use tools to gather accurate information
- **Multi-part queries**: Investigate systematically, then synthesize findings
- **No meta-commentary**: Provide direct answers without explaining your process

All responses must be:
1. **Brief and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
"""
    
    # Phase configurations for different reasoning stages
    PHASE_CONFIGS = {
        "investigate": {
            "system_suffix": "\n\n**Current Phase: INVESTIGATION**\nYour goal is to gather relevant information. Use tools to search for content when needed. If you find useful information, you can choose to investigate further or provide a complete answer.",
            "max_tokens": 600
        },
        "synthesize": {
            "system_suffix": "\n\n**Current Phase: SYNTHESIS**\nYou have access to previous investigation results. Provide a comprehensive answer using all available context. You may use tools once more if additional clarification is needed.",
            "max_tokens": 800
        }
    }
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_phases: int = 2) -> str:
        """
        Generate AI response using phase-based conversational reasoning.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_phases: Maximum reasoning phases (default 2)
            
        Returns:
            Generated response as string
        """
        
        # Phase 1: Investigation
        investigation_result = self._execute_phase(
            "investigate", query, conversation_history, tools, tool_manager
        )
        
        # Check if we should terminate early
        if self._should_terminate_after_investigation(investigation_result):
            return self._extract_final_response(investigation_result)
        
        # Phase 2: Synthesis with investigation context
        synthesis_result = self._execute_synthesis_phase(
            query, conversation_history, investigation_result, tools, tool_manager
        )
        
        return self._extract_final_response(synthesis_result)
    
    def _execute_phase(self, phase_name: str, query: str, conversation_history: Optional[str],
                      tools: Optional[List], tool_manager) -> Dict[str, Any]:
        """
        Execute a single reasoning phase.
        
        Args:
            phase_name: Name of the phase ("investigate" or "synthesize")
            query: Original user query
            conversation_history: Previous conversation context
            tools: Available tools
            tool_manager: Manager to execute tools
            
        Returns:
            Dict containing response and execution metadata
        """
        # Build phase-specific system content
        phase_config = self.PHASE_CONFIGS[phase_name]
        system_content = self._build_phase_system_content(
            phase_name, conversation_history
        )
        
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
            "max_tokens": phase_config["max_tokens"]
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        try:
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # Handle tool execution if needed
            if response.stop_reason == "tool_use" and tool_manager:
                final_response = self._handle_tool_execution(response, api_params, tool_manager)
                return {
                    "response": final_response,
                    "used_tools": True,
                    "phase": phase_name,
                    "success": True
                }
            
            # Return direct response
            return {
                "response": response.content[0].text,
                "used_tools": False,
                "phase": phase_name,
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Error in {phase_name} phase: {str(e)}",
                "used_tools": False,
                "phase": phase_name,
                "success": False,
                "error": str(e)
            }
    
    def _execute_synthesis_phase(self, original_query: str, conversation_history: Optional[str],
                                investigation_result: Dict[str, Any], tools: Optional[List],
                                tool_manager) -> Dict[str, Any]:
        """
        Execute synthesis phase with investigation context.
        
        Args:
            original_query: Original user query
            conversation_history: Previous conversation context
            investigation_result: Results from investigation phase
            tools: Available tools
            tool_manager: Manager to execute tools
            
        Returns:
            Dict containing response and execution metadata
        """
        # Build synthesis context with investigation results
        synthesis_query = self._build_synthesis_query(original_query, investigation_result)
        system_content = self._build_phase_system_content(
            "synthesize", conversation_history, investigation_result
        )
        
        # Prepare API call parameters
        phase_config = self.PHASE_CONFIGS["synthesize"]
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": synthesis_query}],
            "system": system_content,
            "max_tokens": phase_config["max_tokens"]
        }
        
        # Add tools for potential additional clarification
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        try:
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # Handle tool execution if needed (limited to one additional call)
            if response.stop_reason == "tool_use" and tool_manager:
                final_response = self._handle_tool_execution(response, api_params, tool_manager)
                return {
                    "response": final_response,
                    "used_tools": True,
                    "phase": "synthesize",
                    "success": True
                }
            
            # Return direct response
            return {
                "response": response.content[0].text,
                "used_tools": False,
                "phase": "synthesize",
                "success": True
            }
            
        except Exception as e:
            # Fallback: Use investigation results if synthesis fails
            if investigation_result.get("success") and investigation_result.get("response"):
                return {
                    "response": investigation_result["response"],
                    "used_tools": investigation_result.get("used_tools", False),
                    "phase": "synthesis_fallback",
                    "success": True,
                    "fallback_reason": f"Synthesis failed: {str(e)}"
                }
            
            return {
                "response": f"Unable to complete synthesis: {str(e)}",
                "used_tools": False,
                "phase": "synthesize",
                "success": False,
                "error": str(e)
            }
    
    def _build_phase_system_content(self, phase_name: str, conversation_history: Optional[str],
                                   investigation_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Build system content for specific phase.
        
        Args:
            phase_name: Name of the phase
            conversation_history: Previous conversation context
            investigation_result: Results from investigation phase (for synthesis)
            
        Returns:
            Complete system prompt for the phase
        """
        # Start with base prompt
        system_content = self.BASE_SYSTEM_PROMPT
        
        # Add phase-specific suffix
        phase_config = self.PHASE_CONFIGS[phase_name]
        system_content += phase_config["system_suffix"]
        
        # Add conversation history if available
        if conversation_history:
            system_content += f"\n\nPrevious conversation:\n{conversation_history}"
        
        # Add investigation results for synthesis phase
        if phase_name == "synthesize" and investigation_result:
            if investigation_result.get("success") and investigation_result.get("response"):
                system_content += f"\n\n**Investigation Results:**\n{investigation_result['response']}"
        
        return system_content
    
    def _build_synthesis_query(self, original_query: str, investigation_result: Dict[str, Any]) -> str:
        """
        Build query for synthesis phase incorporating investigation results.
        
        Args:
            original_query: Original user query
            investigation_result: Results from investigation phase
            
        Returns:
            Enhanced query for synthesis phase
        """
        if investigation_result.get("success") and investigation_result.get("response"):
            return f"Original question: {original_query}\n\nBased on the investigation, please provide a comprehensive answer."
        else:
            return original_query
    
    def _should_terminate_after_investigation(self, investigation_result: Dict[str, Any]) -> bool:
        """
        Determine if we should terminate after investigation phase.
        
        Args:
            investigation_result: Results from investigation phase
            
        Returns:
            True if should terminate, False to continue to synthesis
        """
        # Terminate if investigation failed
        if not investigation_result.get("success"):
            return True
        
        # Terminate if no tools were used (likely a general knowledge question)
        if not investigation_result.get("used_tools"):
            return True
        
        # Continue to synthesis for tool-based responses
        return False
    
    def _extract_final_response(self, phase_result: Dict[str, Any]) -> str:
        """
        Extract the final response text from phase result.
        
        Args:
            phase_result: Result from phase execution
            
        Returns:
            Final response text
        """
        return phase_result.get("response", "Unable to generate response")
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    # Legacy method kept for backward compatibility
    def _legacy_generate_response(self, query: str,
                                 conversation_history: Optional[str] = None,
                                 tools: Optional[List] = None,
                                 tool_manager=None) -> str:
        """
        Legacy single-round response generation for backward compatibility.
        """
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.BASE_SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.BASE_SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text