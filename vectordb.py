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

# Load environment variables from config.env
load_dotenv('config.env')

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=AIzaSyBoPSlZo-AtwUBq-aEhsSwCIanCiSVS-YE)

# List available models first
print("Available models:")
for model in genai.list_models():
    print(f"- {model.name}")

# Initialize with the correct model name
client = genai.GenerativeModel('gemini-1.0-pro')

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            embedding = genai.embed_content(
                model='embedding-001',
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(embedding['embedding'])
            
        return embeddings

def load_qtl_chunks(file_path='qtl_chunks_top_qtls_only.json'):
    """Load QTL chunks from JSON file"""
    with open(file_path, 'r') as f:
        chunks = json.load(f)
    return [chunk['content'] for chunk in chunks]  # Extract just the content for embedding

def create_chroma_db(documents, name):
    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete collection if it exists (for clean restart)
    try:
        chroma_client.delete_collection(name=name)
        print(f"Deleted existing collection: {name}")
    except:
        pass
    
    # Create collection with Google's embedding function
    collection = chroma_client.create_collection(
        name=name,
        embedding_function=GoogleEmbeddingFunction()
    )
    
    # Add documents
    collection.add(
        documents=documents,
        ids=[str(i) for i in range(len(documents))]
    )
    
    print(f"✅ Created persistent collection '{name}' with {len(documents)} documents")
    return collection

# Load chunks from JSON file
documents = load_qtl_chunks()

# Set up the DB
collection = create_chroma_db(documents, "qtl_database")

# Get all documents with their embeddings
all_results = collection.get(
    include=['documents', 'embeddings']
)

print("\nDocument Embeddings:")
for i, (doc, embedding) in enumerate(zip(all_results['documents'], all_results['embeddings'])):
    print(f"\nDocument {i+1}:")
    print(f"Content: {doc[:100]}...")  # Show first 100 chars
    print(f"Embedding (first 5 dimensions): {embedding[:5]}")  # Show first 5 dimensions to keep output readable

# Example query with embeddings
query_text = "What are the top 5 QTLs with highest LOD scores and what do they tell us?"
query_results = collection.query(
    query_texts=[query_text],
    n_results=2,
    include=['documents', 'embeddings']
)

print("\nQuery Results:")
print(f"Query: {query_text}")
for i, (doc, embedding) in enumerate(zip(query_results['documents'][0], query_results['embeddings'][0])):
    print(f"\nResult {i+1}:")
    print(f"Content: {doc[:200]}...")
    print(f"Embedding (first 10 dimensions): {embedding[:10]}")

# print("Sample documents:")
# results = collection.peek()
# print(results)
#test