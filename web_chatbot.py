from flask import Flask, render_template, request, jsonify, session
import os
import json
import re
import statistics
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import uuid

# Load environment variables
load_dotenv('config.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
app.secret_key = os.urandom(24)

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
        
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            result = genai.embed_content(
                model='embedding-001',
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            embeddings.append(result['embedding'])
        return embeddings

class GeneticQTLChatbot:
    def __init__(self):
        # Load QTL data for computational queries
        with open('enhanced_rag_chunks.json', 'r') as f:
            self.rag_data = json.load(f)
        
        # Initialize ChromaDB for semantic search
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma_client.get_collection(
                name="qtl_database",
                embedding_function=GoogleEmbeddingFunction()
            )
            print("✅ Connected to existing ChromaDB collection")
        except:
            print("❌ ChromaDB collection not found. Please run vectordb.py first.")
            self.collection = None
        
        # Initialize Gemini model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Precompute QTL statistics for fast computation
        self.qtl_stats = self._precompute_stats()
    
    def _precompute_stats(self):
        """Precompute statistics from QTL data for fast retrieval"""
        all_qtls = []
        for chunk in self.rag_data['enhanced_chunks']:
            if 'raw_data' in chunk and chunk['raw_data']:
                if isinstance(chunk['raw_data'], list):
                    all_qtls.extend(chunk['raw_data'])
                else:
                    all_qtls.append(chunk['raw_data'])
        
        if not all_qtls:
            return {}
        
        lod_scores = [q.get('qtl_lod', 0) for q in all_qtls if q.get('qtl_lod')]
        chromosomes = [q.get('qtl_chr') for q in all_qtls if q.get('qtl_chr')]
        genes = [q.get('gene_symbol') for q in all_qtls if q.get('gene_symbol')]
        cis_trans = [q.get('cis') for q in all_qtls if q.get('cis') is not None]
        
        return {
            'total_qtls': len(all_qtls),
            'lod_scores': lod_scores,
            'avg_lod': statistics.mean(lod_scores) if lod_scores else 0,
            'max_lod': max(lod_scores) if lod_scores else 0,
            'min_lod': min(lod_scores) if lod_scores else 0,
            'chromosomes': list(set(chromosomes)),
            'genes': list(set(genes)),
            'cis_count': sum(1 for x in cis_trans if x),
            'trans_count': sum(1 for x in cis_trans if not x),
            'all_qtls': all_qtls
        }
    
    def detect_computational_query(self, query):
        """Detect if query requires computation"""
        computational_patterns = [
            r'\b(average|mean|avg)\b.*\b(lod|score)\b',
            r'\bhow many\b.*\b(qtl|gene|chromosome)\b',
            r'\b(count|number of)\b.*\b(qtl|gene|chromosome)\b',
            r'\b(highest|lowest|maximum|minimum|max|min)\b.*\b(lod|score)\b',
            r'\b(calculate|compute|sum)\b',
            r'\b(statistics|stats)\b',
            r'\b(total|overall)\b.*\b(qtl|gene)\b',
            r'\b(cis|trans)\b.*\b(acting|regulation)\b',
            r'\bchromosome\s+\d+\b'
        ]
        
        query_lower = query.lower()
        return any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in computational_patterns)
    
    def perform_computation(self, query):
        """Perform computational analysis on QTL data"""
        query_lower = query.lower()
        stats = self.qtl_stats
        
        results = []
        
        # Average LOD score queries
        if re.search(r'\b(average|mean|avg)\b.*\b(lod|score)\b', query_lower):
            results.append(f"🧬 **Average LOD Score**: {stats['avg_lod']:.2f}")
            results.append(f"📊 **Range**: {stats['min_lod']:.2f} - {stats['max_lod']:.2f}")
        
        # Counting queries
        if re.search(r'\b(how many|count|number of)\b.*\b(qtl|gene|chromosome)\b', query_lower):
            results.append(f"🔢 **Total QTLs**: {stats['total_qtls']}")
            results.append(f"🧬 **Unique Genes**: {len(stats['genes'])}")
            results.append(f"🧭 **Chromosomes**: {len(stats['chromosomes'])}")
        
        # Cis/trans regulation
        if re.search(r'\b(cis|trans)\b.*\b(acting|regulation)\b', query_lower):
            cis_pct = (stats['cis_count'] / stats['total_qtls'] * 100) if stats['total_qtls'] > 0 else 0
            trans_pct = (stats['trans_count'] / stats['total_qtls'] * 100) if stats['total_qtls'] > 0 else 0
            results.append(f"🎯 **Cis-acting QTLs**: {stats['cis_count']} ({cis_pct:.1f}%)")
            results.append(f"🔄 **Trans-acting QTLs**: {stats['trans_count']} ({trans_pct:.1f}%)")
        
        # Highest/lowest LOD scores
        if re.search(r'\b(highest|lowest|maximum|minimum|max|min)\b.*\b(lod|score)\b', query_lower):
            # Find QTL with highest LOD
            max_qtl = max(stats['all_qtls'], key=lambda x: x.get('qtl_lod', 0))
            min_qtl = min(stats['all_qtls'], key=lambda x: x.get('qtl_lod', 0))
            
            if 'highest' in query_lower or 'maximum' in query_lower or 'max' in query_lower:
                results.append(f"🏆 **Highest LOD Score**: {max_qtl.get('qtl_lod', 0):.2f}")
                results.append(f"🧬 **Gene**: {max_qtl.get('gene_symbol', 'Unknown')}")
                results.append(f"📍 **Location**: Chr {max_qtl.get('qtl_chr', '?')}, {max_qtl.get('qtl_pos', 0):.2f} Mb")
            
            if 'lowest' in query_lower or 'minimum' in query_lower or 'min' in query_lower:
                results.append(f"📉 **Lowest LOD Score**: {min_qtl.get('qtl_lod', 0):.2f}")
                results.append(f"🧬 **Gene**: {min_qtl.get('gene_symbol', 'Unknown')}")
                results.append(f"📍 **Location**: Chr {min_qtl.get('qtl_chr', '?')}, {min_qtl.get('qtl_pos', 0):.2f} Mb")
        
        # Chromosome-specific queries
        chr_match = re.search(r'\bchromosome\s+(\d+)\b', query_lower)
        if chr_match:
            target_chr = chr_match.group(1)
            chr_qtls = [q for q in stats['all_qtls'] if str(q.get('qtl_chr', '')).strip() == target_chr]
            if chr_qtls:
                chr_lods = [q.get('qtl_lod', 0) for q in chr_qtls]
                results.append(f"🧭 **Chromosome {target_chr}**: {len(chr_qtls)} QTLs")
                results.append(f"📊 **Average LOD**: {statistics.mean(chr_lods):.2f}")
                results.append(f"🎯 **Range**: {min(chr_lods):.2f} - {max(chr_lods):.2f}")
        
        return "\n".join(results) if results else None
    
    def get_rag_context(self, query, max_results=3):
        """Get relevant context from vector database"""
        if not self.collection:
            return "Vector database not available."
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=['documents']
            )
            
            if results['documents'] and results['documents'][0]:
                context_docs = results['documents'][0]
                return "\n\n---\n\n".join(context_docs)
            
        except Exception as e:
            print(f"RAG query error: {e}")
        
        return "No relevant context found."
    
    def generate_response(self, user_query):
        """Generate comprehensive response combining computation and RAG"""
        # Check if it's a computational query
        computational_result = None
        if self.detect_computational_query(user_query):
            computational_result = self.perform_computation(user_query)
        
        # Get relevant biological context
        rag_context = self.get_rag_context(user_query)
        
        # Prepare prompt for Gemini
        prompt_parts = [
            "You are a genetics research assistant specializing in QTL (Quantitative Trait Loci) analysis.",
            "You help researchers understand genetic associations in liver tissue.",
            f"User question: {user_query}",
        ]
        
        if computational_result:
            prompt_parts.extend([
                "\n**COMPUTATIONAL RESULTS:**",
                computational_result,
                "\nPlease interpret these computational results in biological context."
            ])
        
        prompt_parts.extend([
            "\n**RELEVANT QTL DATA:**",
            rag_context,
            "\nProvide a comprehensive answer that combines the computational results (if any) with biological interpretation.",
            "Use emojis and formatting to make the response engaging and clear.",
            "Focus on the biological significance and research implications."
        ])
        
        try:
            response = self.model.generate_content("\n".join(prompt_parts))
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize chatbot
chatbot = GeneticQTLChatbot()

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Generate response
        response = chatbot.generate_response(user_message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error processing message: {str(e)}'
        }), 500

@app.route('/stats')
def get_stats():
    """API endpoint to get QTL dataset statistics"""
    return jsonify(chatbot.qtl_stats)

if __name__ == '__main__':
    print("🧬 Starting Genetic QTL Research Chatbot...")
    print(f"📊 Loaded {chatbot.qtl_stats.get('total_qtls', 0)} QTLs")
    print(f"🧬 Covering {len(chatbot.qtl_stats.get('genes', []))} genes")
    print(f"🧭 Across {len(chatbot.qtl_stats.get('chromosomes', []))} chromosomes")
    print("🚀 Server starting on http://localhost:51174")
    print(f"🌐 Access from external: http://128.104.116.141:51174")
    print(f"🌐 Alternative access: http://attie.diabetes.wisc.edu:51174")
    
    app.run(host='0.0.0.0', port=51174, debug=False) 