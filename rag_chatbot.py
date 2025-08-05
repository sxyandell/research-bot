import chromadb
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
import os
from typing import List, Dict, Any
import textwrap
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests

# Add OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

# Add at the top after other imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers not available. Install with: pip install sentence-transformers")

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Ensure input is a list
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model="embedding-001",
                content=text,
                task_type="SEMANTIC_SIMILARITY"
            )
            embeddings.append(result['embedding'])
            
        return embeddings

class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for local embeddings")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
    
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input).tolist()
        return embeddings

class QTLChatbot:
    def __init__(self, use_openai_backup=True, use_local_embeddings=False, ollama_url="http://127.0.0.1:11434/api/generate", ollama_model="llama3.2:latest"):
        """Initialize the QTL chatbot with ChromaDB and Ollama."""
        # Load environment variables
        load_dotenv('config.env')
        
        # Choose embedding function
        self.use_local_embeddings = use_local_embeddings
        if use_local_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_function = LocalEmbeddingFunction()
            print("✅ Using local embeddings (sentence-transformers)")
        else:
            # Configure Google API for embeddings
            GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            genai.configure(api_key=GOOGLE_API_KEY)
            self.embedding_function = GoogleEmbeddingFunction()
            print("✅ Using Google embeddings")

        # Configure OpenAI as backup
        self.use_openai_backup = use_openai_backup and OPENAI_AVAILABLE
        if self.use_openai_backup:
            OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
            if OPENAI_API_KEY and OPENAI_API_KEY != 'your_openai_api_key_here':
                openai.api_key = OPENAI_API_KEY
                print("✅ OpenAI backup configured")
            else:
                print("⚠️ OPENAI_API_KEY not found or is placeholder - no backup available")
                self.use_openai_backup = False
        
        # Ollama configuration
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        print(f"✅ Ollama configured: {self.ollama_url} (model: {self.ollama_model})")
        
        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection with the chosen embedding function
        try:
            self.collection = self.chroma_client.get_collection(
                name="qtl_database_local" if use_local_embeddings else "qtl_database",
                embedding_function=self.embedding_function
            )
            print(f"✅ Connected to ChromaDB with {self.collection.count()} documents")
        except ValueError:
            print("Collection not found. You may need to recreate it with the chosen embedding method.")
            raise

        # Chat history
        self.chat_history = []
    
    def _format_context(self, results: Dict[str, Any]) -> str:
        """Format retrieved chunks into a context string with enhanced biological information."""
        context_parts = []
        
        if not results['documents'][0]:  # If no results found
            return "No relevant QTL data found for this query."
        
        context_parts.append("\nRetrieved QTL Information:")
        context_parts.append("=" * 40)
        
        # Group by gene types for better biological context
        entries_by_type = {}
        for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
            # Try to extract gene type from document
            gene_type = "Unknown"
            if "Gene Type:" in doc:
                try:
                    gene_type = doc.split("Gene Type: ")[1].split(" |")[0].split("\n")[0]
                except:
                    pass
            
            if gene_type not in entries_by_type:
                entries_by_type[gene_type] = []
            entries_by_type[gene_type].append((doc, score, i))
        
        # Add summary of gene types found
        context_parts.append(f"\n📊 Gene Types Found: {', '.join(entries_by_type.keys())}")
        context_parts.append("-" * 40)
        
        for gene_type, entries in entries_by_type.items():
            if len(entries) > 1:
                context_parts.append(f"\n🧬 {gene_type} genes ({len(entries)} found):")
            
            for doc, score, i in entries:
                context_parts.append(f"\nQTL Entry #{i} (Relevance: {1-score:.3f}) - {gene_type}")
                context_parts.append("-" * 40)
                context_parts.append(doc)
                context_parts.append("\n" + "-" * 40)
        
        return "\n".join(context_parts)
    
    def _create_enhanced_prompt(self, query: str, context: str) -> str:
        """Create an enhanced prompt with better biological context."""
        return f"""You are an expert research assistant specializing in QTL (Quantitative Trait Loci) analysis and mouse genetics. 
Your expertise includes:
- Understanding gene regulation (cis vs trans-acting QTLs)
- Interpreting statistical significance (LOD scores, p-values)
- Recognizing gene types (protein_coding, lncRNA, pseudogenes, etc.)
- Understanding chromosomal organization and gene clustering

BIOLOGICAL CONTEXT GUIDELINES:
1. **QTL Types**: 
   - cis-acting: Gene regulation occurs locally (same chromosome region)
   - trans-acting: Gene regulation occurs distally (different chromosome/distant region)

2. **Gene Types Significance**:
   - protein_coding: Direct functional genes
   - lncRNA: Regulatory long non-coding RNAs
   - processed_pseudogene: Evolutionary remnants, may have regulatory roles
   - transcribed_*: Actively transcribed regulatory elements

3. **Statistical Significance**:
   - LOD > 10: Highly significant
   - LOD 5-10: Moderately significant  
   - LOD < 5: Suggestive

4. **Genomic Analysis**:
   - Look for clustering patterns
   - Consider chromosomal context
   - Identify potential regulatory networks

ANSWER REQUIREMENTS:
- Always cite specific LOD scores, positions, and statistical measures
- Explain biological significance of cis vs trans effects
- Highlight patterns in gene types or chromosomal clustering
- Provide confidence intervals when discussing locations
- If multiple genes are involved, discuss potential biological pathways
- Use scientific terminology appropriately

Retrieved Data:
{context}

User Question: {query}

Provide a comprehensive, scientifically accurate answer using ONLY the provided data. Include biological interpretation where relevant."""

    def _call_ollama(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the response text."""
        try:
            print("[INFO] Using Ollama for text generation.")
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return "[Ollama] " + data.get("response", "[No response from Ollama]")
        except Exception as e:
            return f"❌ Ollama error: {str(e)}"

    def _generate_with_openai_backup(self, prompt: str) -> str:
        """Try Ollama first, fallback to OpenAI if available."""
        try:
            # Try Ollama first
            response_text = self._call_ollama(prompt)
            if response_text and not response_text.startswith("❌"):
                return response_text
            else:
                print("⚠️ Ollama failed, trying OpenAI backup...")
                if self.use_openai_backup:
                    try:
                        print("[INFO] Using OpenAI for text generation.")
                        response = openai.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are an expert QTL research assistant."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=1000,
                            temperature=0.1
                        )
                        return "🔄 [OpenAI Backup] " + response.choices[0].message.content
                    except Exception as openai_error:
                        return f"❌ Both Ollama and OpenAI failed. Ollama: {response_text[:100]}... OpenAI: {str(openai_error)[:100]}..."
                else:
                    return f"❌ Ollama failed and no OpenAI backup configured. Error: {response_text[:200]}..."
        except Exception as e:
            return f"❌ An error occurred: {str(e)}"

    def process_query(self, user_input: str, n_results: int = 10) -> str:
        """Process a user query with enhanced biological analysis."""
        try:
            # Get query embedding using the chosen embedding function
            query_embedding = self.embedding_function([user_input])[0]
            
            # Get relevant chunks from ChromaDB with more results for better context
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'distances']
            )
            
            # Format context from retrieved chunks
            context = self._format_context(results)
            
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(user_input, context)
            
            # Generate response with backup (Ollama first, then OpenAI)
            response_text = self._generate_with_openai_backup(prompt)
            
            # Update chat history
            self.chat_history.append({
                "user": user_input,
                "assistant": response_text,
                "context": context,
                "n_results": len(results['documents'][0]) if results['documents'] else 0
            })
            
            return response_text
            
        except Exception as e:
            return f"❌ An error occurred during query processing: {str(e)}"
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the QTL database."""
        try:
            count = self.collection.count()
            # Sample a few documents to analyze content
            sample = self.collection.peek(limit=50)
            
            gene_types = {}
            chromosomes = set()
            
            for doc in sample['documents']:
                if "Gene Type:" in doc:
                    try:
                        gene_type = doc.split("Gene Type: ")[1].split(" |")[0].split("\n")[0]
                        gene_types[gene_type] = gene_types.get(gene_type, 0) + 1
                    except:
                        pass
                
                if "Chromosome" in doc:
                    try:
                        chrom = doc.split("Chromosome ")[1].split(",")[0].split(" |")[0]
                        chromosomes.add(chrom)
                    except:
                        pass
            
            return {
                "total_documents": count,
                "gene_types_sample": gene_types,
                "chromosomes_sample": sorted(list(chromosomes)),
                "collection_name": self.collection.name
            }
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self):
        """Interactive chat loop with enhanced features."""
        # Show database stats
        stats = self.get_collection_stats()
        print(f"🧬 QTL Research Assistant ready!")
        print(f"📊 Database: {stats.get('total_documents', 'Unknown')} QTL entries loaded")
        if 'gene_types_sample' in stats:
            print(f"🔬 Gene types available: {', '.join(stats['gene_types_sample'].keys())}")
        print(f"💡 Type 'help' for example questions, 'stats' for database info, or 'quit' to exit")
        print("-" * 80)
        
        while True:
            user_input = input("\n🧬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye! Happy researching!")
                break
            elif user_input.lower() == 'help':
                print("""
🔬 Example Questions:
• "What are the top QTLs with highest LOD scores?"
• "Tell me about cis-acting QTLs on chromosome 1"
• "Which protein-coding genes have significant QTLs?"
• "What QTLs are associated with [specific gene name]?"
• "Show me trans-acting QTLs with LOD > 400"
• "What's the confidence interval for [gene name]?"
• "Compare cis vs trans QTLs in the dataset"
                """)
                continue
            elif user_input.lower() == 'stats':
                stats = self.get_collection_stats()
                print(f"\n📊 Database Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            print("\n🔍 Searching QTL database...")
            response = self.process_query(user_input)
            print(f"\n🤖 Assistant:\n{textwrap.fill(response, width=90, subsequent_indent='    ')}")

if __name__ == "__main__":
    # Initialize and start chatbot
    try:
        print("🧬 Initializing QTL Research Assistant...")
        print("💡 Using Google embeddings (free) + OpenAI text generation")
        
        # Check if OpenAI key is configured
        load_dotenv('config.env')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        if not OPENAI_API_KEY or OPENAI_API_KEY == 'your_openai_api_key_here':
            print("⚠️ OpenAI API key not configured. Options:")
            print("   1. Add OPENAI_API_KEY to config.env")
            print("   2. Use local embeddings: python3 rag_chatbot.py --local")
            print("   3. Wait for Google quota to reset")
            
            # Ask user what they want to do
            choice = input("\nChoose option (1/2/3) or press Enter to continue with local embeddings: ").strip()
            
            if choice == "1":
                print("Please add your OpenAI API key to config.env and restart")
                exit(1)
            elif choice == "2" or choice == "":
                print("🔄 Switching to local embeddings mode...")
                chatbot = QTLChatbot(use_local_embeddings=True, use_openai_backup=True)
            else:
                print("Waiting for Google quota reset...")
                chatbot = QTLChatbot(use_local_embeddings=False, use_openai_backup=False)
        else:
            # Use Google embeddings + OpenAI text generation (optimal)
            chatbot = QTLChatbot(use_local_embeddings=False, use_openai_backup=True)
        
        chatbot.chat()
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        print("💡 Make sure your API keys are set in config.env and ChromaDB is set up")

#test