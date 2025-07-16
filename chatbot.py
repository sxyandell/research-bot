from hybrid_qtl_system import HybridQTLSystem
import pprint
import textwrap
import pandas as pd
import os
from dotenv import load_dotenv
import argparse
import shutil


def display_ai_results(results):
    """Formats and displays the AI-generated response and its sources."""
    
    ai_response = results.get('ai_response', "No AI response was generated.")
    
    print("\n" + "="*80)
    print("🤖 Assistant Response".center(80))
    print("="*80)
    print(textwrap.fill(ai_response, width=80))
    print("-" * 80)
    
    # Optionally display the sources used
    if results.get('results'):
        print("💡 This answer was generated based on the following information:")
        
        intent = results.get('detected_intent')
        if intent == 'semantic':
            for i, doc in enumerate(results['results'], 1):
                doc_type = doc['metadata'].get('type', 'N/A')
                doc_id = doc.get('id', 'N/A')
                print(f"  - [{i}] Document Type: {doc_type}, ID: {doc_id}")
        
        elif intent == 'analytical':
            sql = results.get('sql_query', 'N/A')
            print(f"  - Analytical query: {sql}")
    print("="*80)


def chatbot_loop(system: HybridQTLSystem):
    """Starts an interactive loop to chat with the QTL system."""
    print("\n" + "="*50)
    print(" Hybrid QTL Chatbot is Ready! ".center(50, "="))
    print("="*50)
    print("Ask me anything about your QTL data.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            query = input("\nYour question > ")
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not query:
                continue

            # Use the new 'ask' method to get a synthesized response
            results = system.ask(query)
            display_ai_results(results)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid QTL Chatbot")
    parser.add_argument(
        '--rebuild-db',
        action='store_true',
        help="Force a full rebuild of the vector database."
    )
    args = parser.parse_args()

    # Define paths
    db_path = "./hybrid_chroma_db"
    csv_path = "/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv"

    if args.rebuild_db:
        print("🗑️ Rebuilding database: Deleting old vector store...")
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"✅ Deleted {db_path}")

    # This will connect to the existing database without rebuilding it.
    print("Connecting to the Hybrid QTL System (this may take a moment)...")
    
    # Load environment variables from config.env
    load_dotenv('config.env')
    
    # Securely get the Google API key from environment
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        print("\n❌ Error: GOOGLE_API_KEY not found.")
        print("Please ensure you have a 'config.env' file in the same directory as chatbot.py,")
        print("and that it contains the line: GOOGLE_API_KEY='your_actual_api_key'")
        exit()

    system = HybridQTLSystem(
        csv_path,
        chroma_db_path=db_path
    )
    
    # Setup models with the provided key
    system.setup_embedding_models(google_api_key=google_api_key)
    
    # Setup the vector store (will build only if it doesn't exist)
    system.setup_vector_store(use_google_embeddings=False) # Still use local for speed
    
    print("✅ System ready.")
    
    chatbot_loop(system)