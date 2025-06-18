import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from chromadb import Documents, EmbeddingFunction, Embeddings

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
            result = genai.embed_content(
                model="embedding-001",
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            embeddings.append(result['embedding'])
            
        return embeddings

class EnhancedQTLChatbot:
    """Enhanced QTL chatbot with computational capabilities."""
    
    def __init__(self):
        # Initialize Gemini
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma_client.get_collection(
                name="qtl_database",
                embedding_function=GoogleEmbeddingFunction()
            )
            print(f"✅ Connected to vector database")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            self.collection = None
        
        # Load QTL data for computations
        self.qtl_data = self._load_qtl_data()
    
    def _load_qtl_data(self) -> pd.DataFrame:
        """Load QTL data from enhanced chunks."""
        try:
            with open('enhanced_rag_chunks.json', 'r') as f:
                data = json.load(f)
                chunks = data.get('enhanced_chunks', [])
            
            all_records = []
            for chunk in chunks:
                if chunk.get('type') == 'top_qtls' and 'raw_data' in chunk:
                    all_records.extend(chunk['raw_data'])
            
            if all_records:
                df = pd.DataFrame(all_records)
                print(f"✅ Loaded {len(df)} QTL records for computation")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def _is_computational_query(self, query: str) -> bool:
        """Check if query requires computation."""
        computational_terms = [
            'average', 'mean', 'sum', 'total', 'count', 'number',
            'maximum', 'minimum', 'highest', 'lowest', 'statistics',
            'distribution', 'range', 'how many'
        ]
        return any(term in query.lower() for term in computational_terms)
    
    def _perform_computation(self, query: str) -> str:
        """Perform computation based on query."""
        if self.qtl_data.empty:
            return "❌ No data available for computation"
        
        query_lower = query.lower()
        
        # Average LOD score
        if 'average' in query_lower and 'lod' in query_lower:
            if 'cis' in query_lower:
                cis_data = self.qtl_data[self.qtl_data['cis'] == True]
                avg_lod = cis_data['qtl_lod'].mean()
                return f"Average LOD score for cis-acting QTLs: {avg_lod:.2f}\nBased on {len(cis_data)} cis-acting QTLs"
            elif 'trans' in query_lower:
                trans_data = self.qtl_data[self.qtl_data['cis'] == False]
                avg_lod = trans_data['qtl_lod'].mean()
                return f"Average LOD score for trans-acting QTLs: {avg_lod:.2f}\nBased on {len(trans_data)} trans-acting QTLs"
            else:
                avg_lod = self.qtl_data['qtl_lod'].mean()
                return f"Average LOD score: {avg_lod:.2f}\nBased on {len(self.qtl_data)} QTLs"
        
        # Count queries
        elif 'how many' in query_lower or 'count' in query_lower:
            if 'chromosome' in query_lower:
                chr_counts = self.qtl_data['qtl_chr'].value_counts().sort_index()
                result = "QTLs per chromosome:\n"
                for chr_num, count in chr_counts.items():
                    result += f"Chromosome {chr_num}: {count} QTLs\n"
                return result
            elif 'cis' in query_lower:
                cis_count = self.qtl_data['cis'].sum()
                return f"Number of cis-acting QTLs: {cis_count}\nOut of {len(self.qtl_data)} total QTLs"
            elif 'trans' in query_lower:
                trans_count = (self.qtl_data['cis'] == False).sum()
                return f"Number of trans-acting QTLs: {trans_count}\nOut of {len(self.qtl_data)} total QTLs"
            else:
                return f"Total number of QTLs: {len(self.qtl_data)}"
        
        # Maximum/minimum
        elif 'maximum' in query_lower or 'highest' in query_lower:
            max_idx = self.qtl_data['qtl_lod'].idxmax()
            max_qtl = self.qtl_data.loc[max_idx]
            return f"Highest LOD score: {max_qtl['qtl_lod']:.2f}\nGene: {max_qtl['gene_symbol']}\nChromosome: {max_qtl['qtl_chr']}\nPosition: {max_qtl['qtl_pos']:.2f} Mb"
        
        elif 'minimum' in query_lower or 'lowest' in query_lower:
            min_idx = self.qtl_data['qtl_lod'].idxmin()
            min_qtl = self.qtl_data.loc[min_idx]
            return f"Lowest LOD score: {min_qtl['qtl_lod']:.2f}\nGene: {min_qtl['gene_symbol']}\nChromosome: {min_qtl['qtl_chr']}\nPosition: {min_qtl['qtl_pos']:.2f} Mb"
        
        # Statistics
        elif 'statistics' in query_lower or 'stats' in query_lower:
            stats = self.qtl_data['qtl_lod'].describe()
            return f"LOD Score Statistics:\nCount: {stats['count']:.0f}\nMean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}\nMin: {stats['min']:.2f}\nMax: {stats['max']:.2f}\nMedian: {stats['50%']:.2f}"
        
        # Sum
        elif 'sum' in query_lower and 'lod' in query_lower:
            total_lod = self.qtl_data['qtl_lod'].sum()
            return f"Sum of all LOD scores: {total_lod:.2f}\nBased on {len(self.qtl_data)} QTLs"
        
        else:
            return "❌ Computational query not recognized"
    
    def process_query(self, query: str) -> str:
        """Process query with computation and biological context."""
        try:
            # Check if computational
            if self._is_computational_query(query):
                computation_result = self._perform_computation(query)
                
                # Get biological context
                if self.collection:
                    embedding_fn = GoogleEmbeddingFunction()
                    query_embedding = embedding_fn([query])[0]
                    
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=3,
                        include=['documents', 'distances']
                    )
                    
                    context = "\n".join(results['documents'][0][:2])
                    
                    prompt = f"""You are a QTL genetics expert. The user asked: "{query}"

COMPUTATIONAL RESULT:
{computation_result}

BIOLOGICAL CONTEXT:
{context}

Provide a response that explains both the computational result and its biological significance."""
                    
                    response = self.model.generate_content(prompt)
                    return f"📊 COMPUTATION:\n{computation_result}\n\n🧬 BIOLOGICAL SIGNIFICANCE:\n{response.text}"
                else:
                    return f"📊 COMPUTATION:\n{computation_result}"
            
            else:
                # Regular biological query
                if self.collection:
                    embedding_fn = GoogleEmbeddingFunction()
                    query_embedding = embedding_fn([query])[0]
                    
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=5,
                        include=['documents', 'distances']
                    )
                    
                    context = "\n\n".join(results['documents'][0])
                    
                    prompt = f"""You are a QTL genetics expert. Answer based on the context provided.

Context: {context}

Question: {query}

Answer:"""
                    
                    response = self.model.generate_content(prompt)
                    return response.text
                else:
                    return "❌ Database not available"
                    
        except Exception as e:
            return f"❌ Error: {str(e)}"

def main():
    print("🧬 Enhanced QTL Research Assistant")
    print("✨ Now with computational capabilities!")
    print("-" * 50)
    
    chatbot = EnhancedQTLChatbot()
    
    print("\n💡 Try these computational queries:")
    print("• What is the average LOD score?")
    print("• What is the average LOD score for cis QTLs?") 
    print("• How many QTLs are on each chromosome?")
    print("• What is the highest LOD score and which gene?")
    print("• Show me statistics for the LOD scores")
    
    while True:
        query = input("\n🔬 Your question: ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        if query:
            response = chatbot.process_query(query)
            print(f"\n🤖 Response:\n{response}")

if __name__ == "__main__":
    main() 