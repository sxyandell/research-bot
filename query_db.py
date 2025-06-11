import chromadb
import google.generativeai as genai
import os
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            embedding = genai.embed_content(
                model='embedding-001',
                content=text,
                task_type="retrieval_query"  # Different task type for queries
            )
            embeddings.append(embedding['embedding'])
            
        return embeddings

def query_database(query_text, n_results=3):
    """Query the persistent QTL database"""
    # Connect to persistent database
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get the collection
    try:
        collection = chroma_client.get_collection(
            name="qtl_database",
            embedding_function=GoogleEmbeddingFunction()
        )
        print(f"✅ Connected to database with {collection.count()} documents")
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None
    
    # Perform the query
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=['documents', 'distances']
    )
    
    return results

def format_results(results, query):
    """Format and display query results"""
    print(f"\n🔍 Query: '{query}'")
    print("=" * 80)
    
    if not results or not results['documents'][0]:
        print("No results found.")
        return
    
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"\n{i+1}. Similarity Score: {1-distance:.3f}")
        print(f"   Content Preview: {doc[:200]}...")
        print("-" * 40)

if __name__ == "__main__":
    print("🧬 QTL Database Query System")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What are the genes with highest LOD scores?",
        "Tell me about chromosome 3 QTLs", 
        "Which QTLs are cis-acting?",
        "genes with LOD score above 500"
    ]
    
    for query in test_queries:
        results = query_database(query, n_results=2)
        if results:
            format_results(results, query)
            print("\n" + "="*80)
    
    # Interactive mode
    print("\n🎯 Interactive Query Mode")
    print("Type your questions (or 'quit' to exit):")
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        results = query_database(query, n_results=3)
        if results:
            format_results(results, query)
    
    print("\nThanks for using the QTL Database Query System! 👋") 