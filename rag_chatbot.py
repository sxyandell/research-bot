import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
import textwrap
from chromadb import Documents, EmbeddingFunction, Embeddings

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            embedding = genai.embed_content(
                model='models/embedding-001',  # This is the correct embedding model name
                content=text,
                task_type="retrieval_document",
                title="QTL data"
            )
            embeddings.append(embedding['embedding'])
            
        return embeddings

class QTLChatbot:
    def __init__(self):
        """Initialize the QTL chatbot with ChromaDB and Gemini."""
        # Load environment variables
        load_dotenv('config.env')
        
        # Configure Google API
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Print available models
        # print("Available models:")
        # for m in genai.list_models():
        #     print(f"- {m.name}")
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')  # Using Flash model instead of Pro
        
        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection with the correct embedding function
        try:
            self.collection = self.chroma_client.get_collection(
                name="qtl_database",
                embedding_function=GoogleEmbeddingFunction()
            )
        except ValueError:
            print("Creating new collection...")
            self.collection = self.chroma_client.create_collection(
                name="qtl_database",
                embedding_function=GoogleEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"}
            )
        
        # Chat history
        self.chat_history = []
    
    def _format_context(self, results: Dict[str, Any]) -> str:
        """Format retrieved chunks into a context string."""
        context_parts = []
        
        if not results['documents'][0]:  # If no results found
            return "No relevant QTL data found for this query."
        
        context_parts.append("\nRetrieved QTL Information:")
        context_parts.append("=" * 40)
        
        for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
            context_parts.append(f"\nQTL Entry #{i} (Relevance Score: {1-score:.3f})")
            context_parts.append("-" * 40)
            context_parts.append(doc)
            context_parts.append("\n" + "-" * 40)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM using the query and context."""
        return f"""
Context:
{context}

Question: {query}

Answer the question using ONLY the information provided above. Be specific and cite the data where relevant."""
    
    def query(self, user_input: str, n_results: int = 2) -> str:
        """Process a user query and return a response."""
        try:
            # Get relevant chunks from ChromaDB
            results = self.collection.query(
                query_texts=[user_input],
                n_results=n_results,
                include=['documents', 'distances']
            )
            
            # Format context from retrieved chunks
            context = self._format_context(results)
            
            # Create prompt
            prompt = self._create_prompt(user_input, context)
            
            # Debug printing
            print("\n" + "="*80)
            print("DEBUG - Retrieved Context:")
            print(context)
            print("\nDEBUG - Full Prompt to AI:")
            print(prompt)
            print("="*80 + "\n")
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Update chat history
            self.chat_history.append({
                "user": user_input,
                "assistant": response.text,
                "context": context
            })
            
            return response.text
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    def chat(self):
        """Interactive chat loop."""
        print("QTL Research Assistant ready! (Type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            response = self.query(user_input)
            print("\nAssistant:", textwrap.fill(response, width=80))

if __name__ == "__main__":
    # Initialize and start chatbot
    chatbot = QTLChatbot()
    chatbot.chat() 

#test