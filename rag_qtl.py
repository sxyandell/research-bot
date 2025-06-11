from vectorize import QTLVectorizer
import openai
from typing import List, Dict, Any
import textwrap

class QTLAssistant:
    def __init__(self, vectors_file: str = "qtl_vectors.pkl", openai_api_key: str = None):
        """Initialize the QTL assistant with pre-computed vectors."""
        self.vectorizer = QTLVectorizer()
        self.vectorizer.load_vectors(vectors_file)
        self.openai_api_key = openai_api_key
        
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        context_parts = []
        
        for i, result in enumerate(chunks, 1):
            chunk = result['chunk']
            similarity = result['similarity']
            
            # Add chunk content with metadata
            context_parts.append(f"[Chunk {i} - Similarity: {similarity:.3f}]")
            
            # Add key metadata
            metadata = chunk['metadata']
            if 'lod_range' in metadata:
                context_parts.append(f"LOD range: {metadata['lod_range'][0]:.2f} - {metadata['lod_range'][1]:.2f}")
            if 'qtl_count' in metadata:
                context_parts.append(f"Number of QTLs: {metadata['qtl_count']}")
            if 'genes' in metadata:
                if isinstance(metadata['genes'], list) and len(metadata['genes']) > 0:
                    # Convert all elements to strings before joining
                    genes = [str(g) for g in metadata['genes'][:5]]
                    context_parts.append(f"Genes: {', '.join(genes)}")
            
            # Add the actual content
            context_parts.append(chunk['content'])
            context_parts.append("-" * 40)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using GPT based on the retrieved context."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for answer generation")
        
        system_prompt = """You are a genetics expert analyzing QTL (Quantitative Trait Loci) data. 
        Answer questions based ONLY on the provided context. Be specific and include numbers/statistics when available.
        If the context doesn't contain enough information to answer the question, say so.
        Format your responses in a clear, scientific manner."""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",  # or gpt-3.5-turbo if preferred
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, query: str, top_k: int = 3) -> str:
        """Main method to answer questions about QTL data."""
        # 1. Retrieve relevant chunks
        results = self.vectorizer.search(
            query=query,
            top_k=top_k,
            openai_api_key=self.openai_api_key
        )
        
        # 2. Format context from chunks
        context = self.format_context(results)
        
        # 3. Generate answer
        answer = self.generate_answer(query, context)
        
        return answer

def print_wrapped(text: str, width: int = 80):
    """Print text with nice wrapping."""
    print("\n".join(textwrap.wrap(text, width=width)))

# Example usage
if __name__ == "__main__":
    print("🧬 QTL Research Assistant")
    print("=" * 50)
    
    # Get API key
    api_key = input("Enter your OpenAI API key: ").strip()
    
    # Initialize assistant
    assistant = QTLAssistant(
        vectors_file="qtl_vectors.pkl",
        openai_api_key=api_key
    )
    
    print("\n✅ Assistant ready! Ask questions about your QTL data.")
    print("Type 'quit' to exit.")
    print("-" * 50)
    
    # Example questions to show
    print("\nExample questions you can ask:")
    examples = [
        "What are the genes with the highest LOD scores?",
        "Tell me about QTLs on chromosome 11",
        "Which QTLs are cis-acting?",
        "What's the relationship between LOD scores and gene types?",
        "Summarize the most significant findings in the data"
    ]
    for ex in examples:
        print(f"- {ex}")
    
    # Interactive loop
    while True:
        print("\n" + "=" * 50)
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print("\n🔍 Searching and analyzing...")
        answer = assistant.answer_question(query)
        
        print("\n📝 Answer:")
        print("-" * 50)
        print_wrapped(answer)
    
    print("\nThanks for using the QTL Research Assistant! 👋") 