from flask import Flask, render_template, request, jsonify, session
import os
import uuid
from dotenv import load_dotenv

# Import the new, powerful system
from hybrid_qtl_system import HybridQTLSystem

# Load environment variables
load_dotenv('config.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Initialize the Hybrid QTL System ---
print("Connecting to the Hybrid QTL System (this may take a moment)...")

# Define paths (can be configured via environment variables in the future)
db_path = "./hybrid_chroma_db"
csv_path = "/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv"

# Check for required Google API key
if not GOOGLE_API_KEY:
    print("\n❌ Error: GOOGLE_API_KEY not found in config.env")
    exit()

# Instantiate the main system
system = HybridQTLSystem(
    csv_file_path=csv_path,
    chroma_db_path=db_path
)

# Setup all necessary components of the system
system.setup_embedding_models(google_api_key=GOOGLE_API_KEY)
system.setup_vector_store(use_google_embeddings=False) # Use local for speed
system.setup_gwas_database()

print("✅ System ready.")
if hasattr(system, 'gwas_client') and system.gwas_client:
    print("🧬 GWAS integration ready for human-mouse cross-species analysis.")

# --- End System Initialization ---

def format_gwas_results_for_web(results: dict) -> str:
    """Formats comprehensive GWAS-QTL analysis results into a markdown string for the web UI."""
    trait_class = results.get('trait_class', 'Unknown')
    lines = [f"### 🧬 GWAS-QTL Analysis Results: {trait_class.upper()} Traits"]

    if 'error' in results:
        lines.append(f"**❌ Analysis failed:** {results['error']}")
        return "\n".join(lines)

    if 'gwas_genes' in results:
        gwas_count = results['gwas_genes']['human_gene_count']
        ortholog_count = results['gwas_genes']['mouse_ortholog_count']
        lines.append(f"\n**📊 Step 1: GWAS Gene Identification**")
        lines.append(f"- Found {gwas_count} human genes for `{trait_class}` traits in the GWAS Catalog.")
        lines.append(f"- Converted to {ortholog_count} unique mouse orthologs for analysis.")

    if 'cis_eqtl_genes' in results:
        cis_count = results['cis_eqtl_genes']['count']
        lines.append(f"\n**🎯 Step 2: Cis-eQTL Overlap**")
        lines.append(f"- Found **{cis_count}** mouse orthologs with cis-eQTLs in the liver study.")

    if 'trans_eqtl_genes' in results:
        trans_count = results['trans_eqtl_genes']['count']
        lines.append(f"\n**🌐 Step 3: Trans-eQTL Overlap**")
        lines.append(f"- Found **{trans_count}** mouse orthologs with trans-eQTLs in the liver study.")

    if 'potential_hub_genes' in results:
        hub_count = results['potential_hub_genes']['count']
        hub_genes = results['potential_hub_genes']['genes']
        lines.append(f"\n**⭐ Step 4: Potential Hub Genes**")
        lines.append(f"- Identified **{hub_count}** potential hub genes with both cis- and trans-eQTLs.")
        if hub_genes:
            lines.append(f"- **Hub Genes**: `{', '.join(hub_genes[:10])}`" + ("..." if len(hub_genes) > 10 else ""))

    if 'overlap_analysis' in results:
        overlap = results['overlap_analysis']
        lines.append("\n**📈 Summary Statistics:**")
        lines.append(f"- GWAS genes with any QTL: **{overlap.get('gwas_with_any_qtl', 0)}**")
        lines.append(f"- GWAS genes with only cis-QTL: **{overlap.get('gwas_with_cis_only', 0)}**")
        lines.append(f"- GWAS genes with only trans-QTL: **{overlap.get('gwas_with_trans_only', 0)}**")
        lines.append(f"- GWAS genes without QTL in study: **{overlap.get('gwas_without_qtl', 0)}**")

    return "\n\n".join(lines)

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
        # --- NEW: Handle special GWAS commands ---
        if user_message.lower().startswith('gwas:'):
            trait_class = user_message.lower().split(':')[1].strip()
            
            if trait_class == 'all':
                trait_classes = ['glycemic', 'lipid', 'hepatic']
                all_results_text = []
                for tc in trait_classes:
                    results = system.comprehensive_gwas_qtl_analysis(tc)
                    all_results_text.append(format_gwas_results_for_web(results))
                
                final_response = "\n\n<hr>\n\n".join(all_results_text)
                
            elif trait_class in ['glycemic', 'lipid', 'hepatic']:
                results = system.comprehensive_gwas_qtl_analysis(trait_class)
                final_response = format_gwas_results_for_web(results)
                
            else:
                final_response = f"❌ **Unknown trait class:** `{trait_class}`. Please use 'glycemic', 'lipid', 'hepatic', or 'all'."
            
            return jsonify({'response': final_response})
        # --- End special command handling ---
        
        # Use the new system's 'ask' method for all other queries
        results = system.ask(user_message)
        
        # Extract the AI response from the results dictionary
        ai_response = results.get('ai_response', "Sorry, I encountered an issue and couldn't generate a response.")
        
        return jsonify({
            'response': ai_response
        })
    
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Hybrid QTL Research Web Chatbot...")
    print("   Please wait for the system to fully initialize.")
    print("🚀 Server starting on http://localhost:51174")
    
    # Note: app.run should be managed by a production server like Gunicorn
    app.run(host='0.0.0.0', port=51174, debug=False) 