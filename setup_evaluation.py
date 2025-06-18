#!/usr/bin/env python3
"""
Setup script for Ragas evaluation of QTL RAG system.
Handles dependency installation and configuration.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies for Ragas evaluation."""
    print("📦 Installing dependencies for Ragas evaluation...")
    
    # Dependencies specifically for Ragas evaluation
    ragas_deps = [
        "ragas>=0.1.0",
        "datasets>=2.14.0", 
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "openai>=1.0.0",
        "langchain-community>=0.0.10"
    ]
    
    for dep in ragas_deps:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True

def check_api_keys():
    """Check if required API keys are configured."""
    print("\n🔑 Checking API key configuration...")
    
    config_file = Path("config.env")
    if not config_file.exists():
        print("❌ config.env file not found")
        return False
    
    # Read config file
    with open(config_file, 'r') as f:
        content = f.read()
    
    has_openai = "OPENAI_API_KEY=" in content
    has_google = "GOOGLE_API_KEY=" in content
    
    print(f"OpenAI API Key: {'✅ Found' if has_openai else '❌ Missing'}")
    print(f"Google API Key: {'✅ Found' if has_google else '❌ Missing'}")
    
    if not has_openai:
        print("\n⚠️  OpenAI API Key required for Ragas evaluation")
        print("Please add OPENAI_API_KEY=your_key_here to config.env")
        return False
    
    return True

def create_sample_config():
    """Create a sample config file if it doesn't exist."""
    config_file = Path("config.env")
    
    if config_file.exists():
        print("✅ config.env already exists")
        return
    
    sample_config = """# API Keys for RAG System
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Other API keys
ANTHROPIC_API_KEY=your_anthropic_key_here

# Evaluation Settings
RAGAS_EVALUATION_MODEL=gpt-4-turbo-preview
RAGAS_EMBEDDING_MODEL=text-embedding-ada-002
"""
    
    with open(config_file, 'w') as f:
        f.write(sample_config)
    
    print("📝 Created sample config.env file")
    print("Please edit config.env and add your actual API keys")

def test_basic_imports():
    """Test if all required packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    test_imports = [
        ("ragas", "Ragas framework"),
        ("langchain", "LangChain"),
        ("openai", "OpenAI"),
        ("datasets", "Datasets"),
        ("chromadb", "ChromaDB"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    all_good = True
    for package, description in test_imports:
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            all_good = False
    
    return all_good

def verify_rag_system():
    """Verify that the existing RAG system components are available."""
    print("\n🔍 Verifying RAG system components...")
    
    required_files = [
        ("vectordb.py", "Vector database setup"),
        ("enhanced_rag.py", "Enhanced RAG system"),
        ("enhanced_vectordb_chunks.json", "Enhanced vector chunks"),
        ("chroma_db", "ChromaDB directory")
    ]
    
    all_present = True
    for file_path, description in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - {file_path} not found")
            all_present = False
    
    return all_present

def main():
    """Run the complete setup process."""
    print("🧬 QTL RAG Evaluation Setup")
    print("=" * 40)
    
    # Step 1: Create sample config if needed
    create_sample_config()
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Step 3: Test imports
    if not test_basic_imports():
        print("❌ Some required packages are not available")
        return False
    
    # Step 4: Check API keys
    if not check_api_keys():
        print("❌ API key configuration incomplete")
        return False
    
    # Step 5: Verify RAG system
    if not verify_rag_system():
        print("❌ RAG system components not found")
        print("Please ensure you have run the RAG system setup first")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run basic test: python test_ragas_evaluation.py")
    print("2. Run full evaluation: python rag_evaluation.py")
    print("3. Check results in ragas_evaluation_results.json")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please address the issues above.")
        sys.exit(1) 