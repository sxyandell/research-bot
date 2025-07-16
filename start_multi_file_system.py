#!/usr/bin/env python3
"""
Quick start script for Multi-File QTL System

This script provides a simple way to:
1. Initialize the multi-file QTL system
2. Start the interactive chatbot
3. Handle any setup issues
"""

import os
import sys

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import pandas
        import numpy
        import chromadb
        import duckdb
        from sentence_transformers import SentenceTransformer
        print("✅ All required dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nTo install missing dependencies, run:")
        print("pip install pandas numpy chromadb duckdb sentence-transformers")
        return False

def check_optional_dependencies():
    """Check optional dependencies for enhanced functionality."""
    optional_deps = []
    
    try:
        import google.generativeai
        optional_deps.append("Google Gemini")
    except ImportError:
        pass
    
    try:
        import openai
        optional_deps.append("OpenAI GPT")
    except ImportError:
        pass
    
    if optional_deps:
        print(f"✅ Optional LLM support: {', '.join(optional_deps)}")
    else:
        print("⚠️  No LLM APIs available - limited SQL generation")
        print("Set GOOGLE_API_KEY or OPENAI_API_KEY for enhanced functionality")

def main():
    """Main startup function."""
    print("🧬 Multi-File QTL Analysis System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    check_optional_dependencies()
    
    print("\nStarting system...")
    
    try:
        # Import and run the chatbot
        from multi_file_qtl_chatbot import MultiFileQTLChatbot
        
        print("Initializing chatbot...")
        chatbot = MultiFileQTLChatbot()
        
        print("🚀 Starting interactive chat...")
        chatbot.run_chat()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure multi_file_qtl_system.py and multi_file_qtl_chatbot.py are available")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("\nTry running the test script first:")
        print("python test_multi_file_system.py")

if __name__ == "__main__":
    main() 