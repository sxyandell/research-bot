import textwrap
import chromadb
import numpy as np
import pandas as pd
import json

from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai.types as types

import os
import google.generativeai as genai

from dotenv import load_dotenv, dotenv_values 

# Add sentence transformers import
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers not available. Install with: pip install sentence-transformers")

# Load environment variables from config.env
load_dotenv('config.env')

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# List available models first
# print("Available models:")
# for model in genai.list_models():
#     print(f"- {model.name}")


# Initialize the client
client = genai.GenerativeModel('gemini-1.0-pro')

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model='embedding-001',
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.append(result['embedding'])
            
        return embeddings
    
    def name(self) -> str:
        """Return the name of the embedding function."""
        return "google_embedding_function"

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for local embeddings")
        print("🔄 Loading local embedding model (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Local embedding model loaded")
    
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input).tolist()
        return embeddings
    
    def name(self) -> str:
        return "local_embedding_function"

def load_qtl_chunks(file_path='qtl_chunks_top_qtls_only.json'):
    """Load QTL chunks from JSON file"""
    with open(file_path, 'r') as f:
        chunks = json.load(f)
    return [chunk['content'] for chunk in chunks]  # Extract just the content for embedding

def create_chroma_db(documents, name, use_local_embeddings=False):

    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Choose embedding function
    if use_local_embeddings:
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for local embeddings")
        embedding_function = LocalEmbeddingFunction()
        collection_name = f"{name}_local"
        print(f"🔄 Using local embeddings for collection: {collection_name}")
    else:
        embedding_function = GoogleEmbeddingFunction()
        collection_name = name
        print(f"🔄 Using Google embeddings for collection: {collection_name}")
    
    # Delete collection if it exists (for clean restart)
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create collection with chosen embedding function
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    
    # Add documents
    print(f"🔄 Processing {len(documents)} documents...")
    collection.add(
        documents=documents,
        ids=[str(i) for i in range(len(documents))]
    )
    
    print(f"✅ Created persistent collection '{collection_name}' with {len(documents)} documents")
    return collection

if __name__ == "__main__":
    import sys
    
    # Check command line arguments for local embeddings
    use_local = "--local" in sys.argv
    
    if use_local:
        print("🌟 Creating ChromaDB with LOCAL embeddings (no API required)")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("❌ sentence-transformers not available. Install with:")
            print("   pip install sentence-transformers")
            sys.exit(1)
    else:
        print("🌟 Creating ChromaDB with GOOGLE embeddings")
    
    # Load chunks from JSON file
    documents = load_qtl_chunks()
    
    # Set up the DB
    collection = create_chroma_db(documents, "qtl_database", use_local_embeddings=use_local)
    
    # Test the collection
    print(f"\n🧪 Testing collection...")
    test_results = collection.query(
        query_texts=["What are the highest LOD scores?"],
        n_results=2,
        include=['documents', 'distances']
    )
    
    print(f"✅ Test successful! Found {len(test_results['documents'][0])} relevant documents")
    
    if use_local:
        print("\n🎉 Local embedding database ready!")
        print("💡 Use: chatbot = QTLChatbot(use_local_embeddings=True)")
    else:
        print("\n🎉 Google embedding database ready!")
        print("💡 Use: chatbot = QTLChatbot(use_local_embeddings=False)")

# Get all documents with their embeddings
all_results = collection.get(
    include=['documents', 'embeddings']
)


print("\nFirst 5 Document Embeddings:")
for i, (doc, embedding) in enumerate(zip(all_results['documents'], all_results['embeddings'])):
    if i >= 5:  # Only show first 5
        break

print("\nDocument Embeddings:")
for i, (doc, embedding) in enumerate(zip(all_results['documents'], all_results['embeddings'])):
    print(f"\nDocument {i+1}:")
    print(f"Content: {doc[:100]}...")  # Show first 100 chars
    print(f"Embedding (first 5 dimensions): {embedding[:5]}")  # Show first 5 dimensions to keep output readable

# Example query with embeddings

# query_text = "What is the top QTL with highest LOD score and what do they tell us?"
# query_results = collection.query(
#     query_texts=[query_text],
#     n_results=2,
#     include=['documents', 'embeddings']
# )

# print("\nQuery Results:")
# print(f"Query: {query_text}")
# for i, (doc, embedding) in enumerate(zip(query_results['documents'][0], query_results['embeddings'][0])):
#     if i >= 5:  # Only show first 5
#         break
#     print(f"\nResult {i+1}:")
#     print(f"Content: {doc[:200]}...")
#     print(f"Embedding (first 10 dimensions): {embedding[:10]}")

# print("Sample documents:")
# results = collection.peek()
# print(results)

