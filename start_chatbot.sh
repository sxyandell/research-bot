#!/bin/bash

# Genetic QTL Research Chatbot Startup Script
echo "🧬 Starting Genetic QTL Research Chatbot..."
echo "================================================"

# Check if required files exist
if [ ! -f "config.env" ]; then
    echo "❌ Error: config.env file not found!"
    echo "Please create config.env with your GOOGLE_API_KEY"
    exit 1
fi

if [ ! -f "qtl_chunks_top_qtls_only.json" ] && [ ! -f "enhanced_rag_chunks.json" ]; then
    echo "❌ Error: No QTL chunks found!"
    echo "Please run chunking.py first to generate QTL chunks"
    echo "Expected: qtl_chunks_top_qtls_only.json or enhanced_rag_chunks.json"
    exit 1
fi

if [ ! -d "chroma_db" ]; then
    echo "⚠️  Warning: ChromaDB not found. Running vectordb.py to create it..."
    python3 vectordb.py
fi

# Set default port if not specified
PORT=${1:-51174}
HOST=${2:-0.0.0.0}

echo "🔧 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   URL: http://$HOST:$PORT"
echo "================================================"

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port $PORT is already in use!"
    echo "   Try a different port: ./start_chatbot.sh 5001"
    exit 1
fi

# Start the chatbot
echo "🚀 Launching chatbot..."
echo "   Press Ctrl+C to stop"
echo "================================================"

# Run with error handling
python3 web_chatbot.py || {
    echo "❌ Error: Failed to start chatbot!"
    echo "Check that all dependencies are installed:"
    echo "   pip install flask google-generativeai chromadb python-dotenv"
    exit 1
} 