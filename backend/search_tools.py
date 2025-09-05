from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Get lesson link if available
            lesson_link = None
            if lesson_num is not None and course_title != 'unknown':
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
            
            # Build context header with clickable link
            if lesson_link:
                # Make the entire header clickable, opening in new tab
                header = f"[[{course_title} - Lesson {lesson_num}]]({lesson_link})"
            else:
                # Fallback to plain text if no link available
                header_text = f"{course_title}"
                if lesson_num is not None:
                    header_text += f" - Lesson {lesson_num}"
                header = f"[{header_text}]"
            
            # Track source for the UI with clickable links
            if lesson_link:
                # Use clickable format for sources too
                source = f"[{course_title} - Lesson {lesson_num}]({lesson_link})"
            else:
                # Fallback to plain text if no link available
                source = course_title
                if lesson_num is not None:
                    source += f" - Lesson {lesson_num}"
            sources.append(source)
            
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for getting course outlines with title, link, and lesson structure"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get course structure including title, link, and complete lesson list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }
    
    def execute(self, course_name: str) -> str:
        """
        Execute the outline tool to get course structure.
        
        Args:
            course_name: Course name to get outline for
            
        Returns:
            Formatted course outline or error message
        """
        # Step 1: Get all course metadata first
        all_courses = self.store.get_all_courses_metadata()
        if not all_courses:
            return "No courses found in the system"
        
        # Step 2: Find best matching course using direct text matching
        best_match = None
        course_name_lower = course_name.lower()
        
        # Try exact title match first
        for course in all_courses:
            if course.get('title', '').lower() == course_name_lower:
                best_match = course
                break
        
        # If no exact match, try partial matching
        if not best_match:
            for course in all_courses:
                title = course.get('title', '').lower()
                if course_name_lower in title or title.startswith(course_name_lower):
                    best_match = course
                    break
        
        # If still no match, use vector store's fuzzy matching as fallback
        if not best_match:
            resolved_title = self.store._resolve_course_name(course_name)
            if resolved_title:
                for course in all_courses:
                    if course.get('title') == resolved_title:
                        best_match = course
                        break
        
        if not best_match:
            available_courses = [course.get('title', 'Unknown') for course in all_courses]
            return f"No course found matching '{course_name}'. Available courses: {', '.join(available_courses)}"
        
        # Step 3: Format the outline
        return self._format_outline(best_match)
    
    def _format_outline(self, course_metadata: Dict[str, Any]) -> str:
        """Format course outline with title, link, and lessons"""
        title = course_metadata.get('title', 'Unknown Course')
        course_link = course_metadata.get('course_link')
        instructor = course_metadata.get('instructor', 'Unknown')
        lessons = course_metadata.get('lessons', [])
        
        # Build course header with link if available
        if course_link:
            header = f"# [{title}]({course_link})"
        else:
            header = f"# {title}"
        
        header += f"\n**Instructor:** {instructor}\n"
        
        # Add lessons list
        if lessons:
            header += f"\n**Lessons ({len(lessons)} total):**\n"
            for lesson in lessons:
                lesson_num = lesson.get('lesson_number', '?')
                lesson_title = lesson.get('lesson_title', 'Untitled')
                lesson_link = lesson.get('lesson_link')
                
                if lesson_link:
                    header += f"- Lesson {lesson_num}: [{lesson_title}]({lesson_link})\n"
                else:
                    header += f"- Lesson {lesson_num}: {lesson_title}\n"
        else:
            header += "\n**No lessons found**\n"
        
        return header


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []