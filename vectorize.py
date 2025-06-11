import json
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
import os

class QTLVectorizer:
    def __init__(self, embedding_method: str = "sentence_transformers"):
        """
        Initialize vectorizer with choice of embedding method.
        
        Args:
            embedding_method: "openai" or "sentence_transformers"
        """
        self.embedding_method = embedding_method
        self.model = None
        self.embeddings = []
        self.chunks = []
        self.chunk_metadata = []
        
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the appropriate embedding model."""
        if self.embedding_method == "openai":
            # Will be loaded when needed (requires API key)
            print("📡 Using OpenAI embeddings (requires API key)")
            
        elif self.embedding_method == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                print("🤖 Loading SentenceTransformer model...")
                # Use a good general-purpose model that works well for scientific text
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Model loaded successfully")
            except ImportError:
                print("❌ sentence-transformers not installed. Install with:")
                print("pip install sentence-transformers")
                raise
        else:
            raise ValueError("embedding_method must be 'openai' or 'sentence_transformers'")
    
    def load_chunks(self, chunks_file: str):
        """Load chunks from JSON file."""
        print(f"📁 Loading chunks from {chunks_file}")
        with open(chunks_file, 'r') as f:
            self.chunks = json.load(f)
        
        print(f"✅ Loaded {len(self.chunks)} chunks")
        
        # Extract metadata for easier searching later
        self.chunk_metadata = []
        for chunk in self.chunks:
            metadata = {
                'id': chunk['id'],
                'type': chunk['type'],
                'metadata': chunk['metadata']
            }
            self.chunk_metadata.append(metadata)
    
    def embed_chunks_openai(self, api_key: str, model: str = "text-embedding-3-small"):
        """Embed chunks using OpenAI API."""
        import openai
        
        openai.api_key = api_key
        
        print(f"🔤 Creating embeddings using OpenAI {model}")
        embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            print(f"Processing chunk {i+1}/{len(self.chunks)}", end='\r')
            
            try:
                response = openai.embeddings.create(
                    input=chunk['content'],
                    model=model
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"\n❌ Error embedding chunk {i}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
        
        self.embeddings = np.array(embeddings)
        print(f"\n✅ Created {len(embeddings)} embeddings")
    
    def embed_chunks_sentence_transformers(self):
        """Embed chunks using SentenceTransformers (free, local)."""
        if self.model is None:
            raise ValueError("SentenceTransformer model not loaded")
        
        print("🔤 Creating embeddings using SentenceTransformers")
        
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in self.chunks]
        
        # Create embeddings in batches for efficiency
        print("Processing embeddings...")
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Created {len(self.embeddings)} embeddings with dimension {self.embeddings.shape[1]}")
    
    def embed_chunks(self, openai_api_key: Optional[str] = None):
        """Embed all chunks using the specified method."""
        if not self.chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")
        
        if self.embedding_method == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            self.embed_chunks_openai(openai_api_key)
        else:
            self.embed_chunks_sentence_transformers()
    
    def save_vectors(self, output_file: str):
        """Save embeddings and metadata to file."""
        data = {
            'embeddings': self.embeddings,
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata,
            'embedding_method': self.embedding_method,
            'embedding_dimension': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Saved vectors to {output_file}")
        print(f"   - {len(self.embeddings)} embeddings")
        print(f"   - Dimension: {self.embeddings.shape[1]}")
        print(f"   - File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    def load_vectors(self, vector_file: str):
        """Load previously saved vectors."""
        print(f"📁 Loading vectors from {vector_file}")
        
        with open(vector_file, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.chunks = data['chunks']
        self.chunk_metadata = data['chunk_metadata']
        self.embedding_method = data['embedding_method']
        
        print(f"✅ Loaded {len(self.embeddings)} vectors")
        print(f"   - Dimension: {data['embedding_dimension']}")
        print(f"   - Method: {self.embedding_method}")
    
    def search(self, query: str, top_k: int = 5, openai_api_key: Optional[str] = None):
        """
        Search for similar chunks given a query.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            openai_api_key: Required if using OpenAI embeddings
        
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call embed_chunks() or load_vectors() first.")
        
        # Embed the query
        if self.embedding_method == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required for query embedding")
            
            import openai
            openai.api_key = openai_api_key
            
            response = openai.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = np.array(response.data[0].embedding)
            
        else:
            if self.model is None:
                self._load_embedding_model()
            query_embedding = self.model.encode([query])
            query_embedding = query_embedding[0]  # Remove batch dimension
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'similarity': similarities[idx],
                'rank': len(results) + 1
            })
        
        return results
    
    def print_search_results(self, results: List[Dict], query: str):
        """Pretty print search results."""
        print(f"\n🔍 Search results for: '{query}'")
        print("=" * 80)
        
        for result in results:
            chunk = result['chunk']
            similarity = result['similarity']
            rank = result['rank']
            
            print(f"\n{rank}. Similarity: {similarity:.3f}")
            print(f"   Chunk ID: {chunk['id']}")
            print(f"   Type: {chunk['type']}")
            
            # Show key metadata
            metadata = chunk['metadata']
            if 'lod_range' in metadata:
                print(f"   LOD range: {metadata['lod_range'][0]:.2f} - {metadata['lod_range'][1]:.2f}")
            if 'qtl_count' in metadata:
                print(f"   QTL count: {metadata['qtl_count']}")
            if 'genes' in metadata and len(metadata['genes']) <= 5:
                print(f"   Genes: {', '.join(metadata['genes'])}")
            
            # Show content preview
            content_preview = chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content']
            print(f"   Content: {content_preview}")
            print("-" * 40)


# Example usage
if __name__ == "__main__":
    # Initialize vectorizer
    print("🚀 QTL Data Vectorizer")
    print("\nChoose embedding method:")
    print("1. SentenceTransformers (free, local)")
    print("2. OpenAI embeddings (requires API key, better quality)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        api_key = input("Enter OpenAI API key: ").strip()
        vectorizer = QTLVectorizer("openai")
    else:
        vectorizer = QTLVectorizer("sentence_transformers")
        api_key = None
    
    # Load and embed chunks
    vectorizer.load_chunks("qtl_chunks_top_qtls_only.json")
    vectorizer.embed_chunks(openai_api_key=api_key)
    
    # Save vectors
    output_file = "qtl_vectors.pkl"
    vectorizer.save_vectors(output_file)
    
    # Test search
    print("\n" + "="*50)
    print("🔍 Testing search functionality")
    
    test_queries = [
        "high LOD score genes",
        "chromosome 1 QTLs",
        "liver metabolism genes",
        "cis-acting QTLs"
    ]
    
    for query in test_queries:
        results = vectorizer.search(query, top_k=3, openai_api_key=api_key)
        vectorizer.print_search_results(results, query)
        print("\n" + "="*80) 