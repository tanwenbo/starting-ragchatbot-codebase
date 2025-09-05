#!/usr/bin/env python3
"""
RAG System Diagnostic Tool

This script diagnoses common issues that cause 'query failed' errors.
Run this to identify problems with your RAG system setup.

Usage: python diagnose_system.py
"""

import os
import sys
from config import Config
from rag_system import RAGSystem


def print_header(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_check(description, status, details=""):
    """Print a check result"""
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {description}")
    if details:
        print(f"   {details}")


def diagnose_environment():
    """Diagnose environment and configuration issues"""
    print_header("ENVIRONMENT DIAGNOSIS")
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print_check(f"ANTHROPIC_API_KEY found", True, f"Length: {len(api_key)} characters")
    else:
        print_check("ANTHROPIC_API_KEY missing", False, 
                   "Set your API key: export ANTHROPIC_API_KEY='your-key-here'")
    
    # Check .env file
    env_file_exists = os.path.exists(".env") or os.path.exists("../.env")
    print_check(f".env file exists", env_file_exists, 
               ".env file should contain ANTHROPIC_API_KEY=your-key-here")
    
    # Check config
    config = Config()
    print_check(f"Config loaded", True, f"Model: {config.ANTHROPIC_MODEL}")
    print_check(f"ChromaDB path", os.path.exists(config.CHROMA_PATH), 
               f"Path: {config.CHROMA_PATH}")
    
    # Check docs folder
    docs_exists = os.path.exists("../docs")
    if docs_exists:
        doc_files = [f for f in os.listdir("../docs") if f.endswith('.txt')]
        print_check(f"Course documents found", len(doc_files) > 0, 
                   f"Found {len(doc_files)} .txt files in ../docs")
    else:
        print_check("Course documents folder", False, "Missing ../docs folder")
    
    return api_key is not None


def diagnose_system_initialization():
    """Diagnose system initialization"""
    print_header("SYSTEM INITIALIZATION DIAGNOSIS")
    
    try:
        config = Config()
        rag_system = RAGSystem(config)
        print_check("RAG system initialized", True)
        
        # Check tools
        tools = rag_system.tool_manager.tools
        print_check(f"Search tool registered", "search_course_content" in tools)
        print_check(f"Outline tool registered", "get_course_outline" in tools)
        
        # Check vector store
        course_count = rag_system.vector_store.get_course_count()
        print_check(f"Courses loaded", course_count > 0, f"Found {course_count} courses")
        
        if course_count == 0:
            print("   🔧 Try loading courses:")
            print("      cd backend && python -c \"from rag_system import RAGSystem; from config import config; r=RAGSystem(config); print(r.add_course_folder('../docs', clear_existing=True))\"")
        
        return True, rag_system
        
    except Exception as e:
        print_check("RAG system initialization", False, f"Error: {str(e)}")
        return False, None


def diagnose_tool_execution(rag_system):
    """Diagnose tool execution"""
    print_header("TOOL EXECUTION DIAGNOSIS")
    
    if not rag_system:
        print_check("Tool execution test", False, "RAG system not available")
        return False
    
    try:
        # Test search tool
        search_result = rag_system.tool_manager.execute_tool(
            "search_course_content", 
            query="introduction"
        )
        search_success = not search_result.startswith("Tool") and not search_result.startswith("No relevant")
        print_check("Search tool execution", search_success)
        if not search_success:
            print(f"   Result: {search_result[:100]}...")
        
        # Test outline tool
        course_titles = rag_system.vector_store.get_existing_course_titles()
        if course_titles:
            outline_result = rag_system.tool_manager.execute_tool(
                "get_course_outline",
                course_name=course_titles[0][:10]
            )
            outline_success = "# " in outline_result and "Lesson" in outline_result
            print_check("Outline tool execution", outline_success)
            if not outline_success:
                print(f"   Result: {outline_result[:100]}...")
        else:
            print_check("Outline tool execution", False, "No courses available to test")
        
        return True
        
    except Exception as e:
        print_check("Tool execution", False, f"Error: {str(e)}")
        return False


def diagnose_full_query(rag_system, has_api_key):
    """Diagnose full query execution"""
    print_header("FULL QUERY DIAGNOSIS")
    
    if not rag_system:
        print_check("Full query test", False, "RAG system not available")
        return False
        
    if not has_api_key:
        print_check("Full query test", False, "API key required for full query test")
        return False
    
    try:
        response, sources = rag_system.query("What is the introduction about?")
        
        # Check response
        response_valid = response and response != "query failed" and len(response) > 10
        print_check("Query response received", response_valid)
        if response_valid:
            print(f"   Response preview: {response[:100]}...")
        else:
            print(f"   Response: {response}")
        
        # Check sources
        sources_valid = sources and len(sources) > 0
        print_check("Sources returned", sources_valid, f"Found {len(sources)} sources")
        
        return response_valid and sources_valid
        
    except Exception as e:
        print_check("Full query execution", False, f"Error: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False


def provide_recommendations(diagnoses):
    """Provide recommendations based on diagnosis results"""
    print_header("RECOMMENDATIONS")
    
    env_ok, init_ok, tools_ok, query_ok = diagnoses
    
    if not env_ok:
        print("🔧 ENVIRONMENT FIXES NEEDED:")
        print("   1. Set ANTHROPIC_API_KEY environment variable")
        print("   2. Create .env file with: ANTHROPIC_API_KEY=your-key-here")
        print("   3. Ensure ../docs folder exists with .txt course files")
        
    if not init_ok:
        print("🔧 INITIALIZATION FIXES NEEDED:")
        print("   1. Check ChromaDB installation: uv add chromadb")
        print("   2. Check sentence-transformers: uv add sentence-transformers")
        print("   3. Clear and rebuild vector store if corrupted")
        
    if not tools_ok:
        print("🔧 TOOL EXECUTION FIXES NEEDED:")
        print("   1. Reload course documents into vector store")
        print("   2. Check ChromaDB permissions and disk space")
        print("   3. Verify embedding model can be downloaded")
        
    if not query_ok:
        print("🔧 QUERY EXECUTION FIXES NEEDED:")
        print("   1. Verify API key is valid and has credits")
        print("   2. Check network connectivity to Anthropic API")
        print("   3. Review system prompt and tool definitions")
        
    if all(diagnoses):
        print("🎉 SYSTEM IS HEALTHY!")
        print("   All components are working correctly.")
        print("   If you're still seeing 'query failed', check:")
        print("   - Frontend JavaScript console for errors")
        print("   - Browser network tab for failed requests")
        print("   - FastAPI server logs for error details")


def main():
    """Main diagnostic function"""
    print("RAG SYSTEM DIAGNOSTIC TOOL")
    print("This tool will help identify why queries are failing.\n")
    
    # Run diagnostics
    env_ok = diagnose_environment()
    init_ok, rag_system = diagnose_system_initialization()
    tools_ok = diagnose_tool_execution(rag_system)
    query_ok = diagnose_full_query(rag_system, env_ok)
    
    # Provide recommendations
    provide_recommendations((env_ok, init_ok, tools_ok, query_ok))
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()