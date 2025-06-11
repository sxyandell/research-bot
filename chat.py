import google.generativeai as genai
from dotenv import load_dotenv
import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Load environment variables
load_dotenv('config.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            embedding = genai.embed_content(
                model='embedding-001',  # Match the model in vectordb.py
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(embedding['embedding'])
            
        return embeddings

def chat_with_qtl_data():
    """Chat interface for querying QTL data using existing ChromaDB collection"""
    print("🧬 QTL Research Assistant")
    print("=" * 50)
    
    # Connect to ChromaDB and list available collections
    try:
        client = chromadb.PersistentClient(path="./chroma_db")  # Use persistent client
        collections = client.list_collections()
        if not collections:
            print("\n❌ No collections found in ChromaDB")
            print("Please run 'python vectordb.py' first to create the database.")
            return
            
        print("\nAvailable collections:")
        for i, coll in enumerate(collections, 1):
            print(f"{i}. {coll.name}")
            
        # Use the first collection by default
        collection = collections[0]
        # Set the embedding function
        collection.embedding_function = GoogleEmbeddingFunction()
        print(f"\n✅ Using collection: {collection.name}")
        
        chat_model = genai.GenerativeModel('gemini-1.0-pro')
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return
    
    print("\nAsk questions about the QTL data (type 'quit' to exit)")
    print("\nExample questions:")
    print("- What are the top QTLs with highest LOD scores?")
    print("- Tell me about significant QTLs on specific chromosomes")
    print("- What genes show interesting patterns?")
    print("-" * 50)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        try:
            print("\n🔍 Analyzing...")
            
            # Get relevant chunks
            results = collection.query(
                query_texts=[query],
                n_results=3,
                include=['documents']
            )
            
            # Format context
            context = "\nRelevant QTL Information:\n"
            for i, doc in enumerate(results['documents'][0], 1):
                context += f"\nChunk {i}:\n{doc}\n"
            
            # Create prompt and get response
            prompt = f"""You are a genetics expert analyzing QTL (Quantitative Trait Loci) data.
            Focus on providing insights about the QTLs, their LOD scores, and genetic significance.
            Base your answers only on the provided context. If information is not in the context, say so.
            
            Context: {context}
            
            Question: {query}"""
            
            response = chat_model.generate_content(prompt)
            
            print("\n📊 Answer:")
            print("-" * 50)
            print(response.text)
            print("-" * 50)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    
    print("\nThanks for using the QTL Research Assistant! 👋")

if __name__ == "__main__":
    chat_with_qtl_data() 