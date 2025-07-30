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
        # All queries, including GWAS, now go through the main system's 'ask' method.
        # The intelligent router in the backend will handle dispatching to the correct tool.
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