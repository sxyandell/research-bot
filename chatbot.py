from hybrid_qtl_system import HybridQTLSystem
import pprint
import textwrap
import pandas as pd
import os
from dotenv import load_dotenv
import argparse
import shutil
import duckdb


def display_ensembl_results(results):
    """Display Ensembl API results in a formatted way."""
    
    print("\n" + "="*80)
    print("🔬 Ensembl API Results".center(80))
    print("="*80)
    
    if 'ensemble_data' in results:
        ensemble_data = results['ensemble_data']
        
        # Display gene information
        if 'ensemble_info' in ensemble_data:
            info = ensemble_data['ensemble_info']
            print(f"📊 Gene Information:")
            if 'display_name' in info:
                print(f"   Gene Name: {info['display_name']}")
            if 'description' in info:
                print(f"   Description: {info['description']}")
            if 'biotype' in info:
                print(f"   Biotype: {info['biotype']}")
            if 'version' in info:
                print(f"   Version: {info['version']}")
        
        # Display variants
        if 'variants' in ensemble_data:
            variants = ensemble_data['variants']
            print(f"\n🧬 Variants ({len(variants)} found):")
            for i, variant in enumerate(variants[:5], 1):  # Show first 5
                print(f"   [{i}] {variant.get('id', 'N/A')} - {variant.get('consequence_type', 'N/A')}")
            if len(variants) > 5:
                print(f"   ... and {len(variants) - 5} more variants")
        
        # Display orthologs
        if 'orthologs' in ensemble_data:
            orthologs = ensemble_data['orthologs']
            print(f"\n🔄 Orthologs ({len(orthologs)} found):")
            for i, ortholog in enumerate(orthologs[:3], 1):  # Show first 3
                if 'target' in ortholog:
                    target = ortholog['target']
                    print(f"   [{i}] {target.get('display_name', 'N/A')} ({target.get('species', 'N/A')})")
            if len(orthologs) > 3:
                print(f"   ... and {len(orthologs) - 3} more orthologs")
    
    # Display cross-species information
    if 'human_orthologs' in results:
        human_data = results['human_orthologs']
        if 'ortholog_count' in human_data and human_data['ortholog_count'] > 0:
            print(f"\n🌍 Cross-Species Analysis:")
            print(f"   Human Orthologs: {human_data['ortholog_count']} found")
            
            if 'human_gene_details' in human_data:
                human_gene = human_data['human_gene_details']
                print(f"   Primary Human Ortholog: {human_gene.get('display_name', 'N/A')}")
                if 'description' in human_gene:
                    print(f"   Human Gene Description: {human_gene['description']}")
    
    print("="*80)


def display_ai_results(results):
    """Formats and displays the AI-generated response and its sources."""
    
    ai_response = results.get('ai_response', "No AI response was generated.")
    
    print("\n" + "="*80)
    print("🤖 Assistant Response".center(80))
    print("="*80)
    print(textwrap.fill(ai_response, width=80))
    print("-" * 80)
    
    # Display Ensembl API results if available
    if 'ensemble_data' in results or 'human_orthologs' in results:
        display_ensembl_results(results)
    
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
    
    # Show data sources used
    if 'data_sources' in results:
        print(f"📚 Data Sources: {', '.join(results['data_sources'])}")
    
    print("="*80)


def display_gwas_analysis_results(results):
    """Display comprehensive GWAS-QTL analysis results in a formatted way."""
    
    trait_class = results.get('trait_class', 'Unknown')
    
    print("\n" + "="*80)
    print(f"🧬 GWAS-QTL Analysis Results: {trait_class.upper()} Traits".center(80))
    print("="*80)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return
    
    # Display GWAS genes
    if 'gwas_genes' in results:
        gwas_count = results['gwas_genes']['human_gene_count']
        ortholog_count = results['gwas_genes']['mouse_ortholog_count']
        
        print(f"📊 Step 1: Found {gwas_count} human genes for {trait_class} traits in GWAS.")
        print(f"   - Converted to {ortholog_count} unique mouse orthologs for analysis.")
        
        if ortholog_count > 0:
            example_genes = results['gwas_genes']['genes'][:10]
            print(f"   Mouse ortholog examples: {', '.join(example_genes)}")
            if ortholog_count > 10:
                print(f"   ... and {ortholog_count - 10} more.")
    
    # Display cis-eQTL results
    if 'cis_eqtl_genes' in results:
        cis_count = results['cis_eqtl_genes']['count']
        cis_peaks = results['cis_eqtl_genes']['qtl_peaks']
        print(f"\n🎯 Step 2: Found {cis_count} GWAS genes with cis-eQTL in DO liver study")
        print(f"   Total cis-QTL peaks: {cis_peaks}")
        
        if cis_count > 0:
            cis_genes = results['cis_eqtl_genes']['genes'][:5]
            print(f"   Top genes: {', '.join(cis_genes)}")
    
    # Display trans-eQTL results  
    if 'trans_eqtl_genes' in results:
        trans_count = results['trans_eqtl_genes']['count']
        trans_peaks = results['trans_eqtl_genes']['qtl_peaks']
        print(f"\n🌐 Step 3: Found {trans_count} GWAS genes with trans-eQTL in DO liver study")
        print(f"   Total trans-QTL peaks: {trans_peaks}")
        
        if trans_count > 0:
            trans_genes = results['trans_eqtl_genes']['genes'][:5]
            print(f"   Top genes: {', '.join(trans_genes)}")
    
    # Display potential hub genes
    if 'potential_hub_genes' in results:
        hub_count = results['potential_hub_genes']['count']
        print(f"\n⭐ Step 4: Found {hub_count} potential hub genes (both cis and trans QTLs)")
        
        if hub_count > 0:
            hub_genes = results['potential_hub_genes']['genes']
            print(f"   Hub genes: {', '.join(hub_genes)}")
    
    # Display overlap analysis
    if 'overlap_analysis' in results:
        overlap = results['overlap_analysis']
        print(f"\n📈 Summary Statistics:")
        print(f"   • GWAS genes with any QTL: {overlap.get('gwas_with_any_qtl', 0)}")
        print(f"   • GWAS genes with cis-QTL only: {overlap.get('gwas_with_cis_only', 0)}")
        print(f"   • GWAS genes with trans-QTL only: {overlap.get('gwas_with_trans_only', 0)}")
        print(f"   • GWAS genes with both: {overlap.get('gwas_with_both', 0)}")
        print(f"   • GWAS genes without QTL: {overlap.get('gwas_without_qtl', 0)}")
    
    print("="*80)


def show_gwas_help():
    """Display help for GWAS analysis commands."""
    print("\n" + "="*60)
    print("🧬 GWAS-QTL Analysis Commands")
    print("="*60)
    print("Available trait classes:")
    print("  • glycemic  - Diabetes, glucose, insulin resistance")
    print("  • lipid     - Cholesterol, triglycerides, lipoproteins") 
    print("  • hepatic   - Liver function, fatty liver, hepatic enzymes")
    print()
    print("Commands:")
    print("  gwas:glycemic   - Analyze glycemic trait genes")
    print("  gwas:lipid      - Analyze lipid trait genes") 
    print("  gwas:hepatic    - Analyze hepatic trait genes")
    print("  gwas:all        - Analyze all three trait classes")
    print("  help:gwas       - Show this help")
    print("="*60)


def chatbot_loop(system: HybridQTLSystem):
    """Starts an interactive loop to chat with the QTL system."""
    print("\n" + "="*50)
    print(" Hybrid QTL Chatbot is Ready! ".center(50, "="))
    print("="*50)
    print("Ask me anything about your QTL data.")
    print("Type 'help:gwas' for GWAS analysis commands.")
    print("Type 'test:ensembl' to test Ensembl API connection.")
    print("Type 'test:genes' to test gene matching between orthologs and QTL database.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            query = input("\nYour question > ")
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not query:
                continue

            # Handle special GWAS commands
            if query.lower() == 'help:gwas':
                show_gwas_help()
                continue
            
            elif query.lower() == 'test:ensembl':
                print("\n🔬 Testing Ensembl API connection...")
                if system.ensemble_client:
                    # Test basic connectivity first
                    try:
                        # Test with a simple gene lookup
                        test_gene = "Apoe"
                        gene_info = system.ensemble_client.get_gene_info(test_gene)
                        if gene_info:
                            print("✅ Ensembl API connection successful!")
                            print(f"📊 Found gene info for {test_gene}")
                        else:
                            print("⚠️ Ensembl API connected but no gene data found")
                    except Exception as e:
                        print(f"❌ Ensembl API connection failed: {e}")
                else:
                    print("❌ Ensembl API client not available!")
                continue
            
            elif query.lower() == 'test:genes':
                print("\n🔍 Testing gene matching between orthologs and QTL database...")
                test_results = system.test_gene_matching()
                print("✅ Gene matching test complete. Check logs for details.")
                continue
            
            elif query.lower().startswith('gwas:'):
                trait_class = query.lower().split(':')[1].strip()
                
                if trait_class == 'all':
                    # Run analysis for all trait classes
                    trait_classes = ['glycemic', 'lipid', 'hepatic']
                    for tc in trait_classes:
                        print(f"\n🔍 Running analysis for {tc} traits...")
                        try:
                            results = system.comprehensive_gwas_qtl_analysis(tc)
                            display_gwas_analysis_results(results)
                            
                            # Optionally export results
                            export_choice = input(f"\nExport {tc} results to CSV? (y/n): ").lower()
                            if export_choice == 'y':
                                system.export_results_to_csv(results)
                                print(f"✅ Results exported for {tc}")
                        except Exception as e:
                            print(f"❌ Error analyzing {tc} traits: {e}")
                
                elif trait_class in ['glycemic', 'lipid', 'hepatic']:
                    print(f"\n🔍 Running comprehensive GWAS-QTL analysis for {trait_class} traits...")
                    print("This may take a few minutes to query GWAS data...")
                    
                    try:
                        results = system.comprehensive_gwas_qtl_analysis(trait_class)
                        display_gwas_analysis_results(results)
                        
                        # Ask if user wants to export results
                        export_choice = input("\nExport results to CSV files? (y/n): ").lower()
                        if export_choice == 'y':
                            system.export_results_to_csv(results)
                            print("✅ Results exported to ./gwas_qtl_results/")
                            
                    except Exception as e:
                        print(f"❌ Error running GWAS analysis: {e}")
                        print("Make sure you have internet connection for GWAS data access.")
                
                else:
                    print(f"❌ Unknown trait class: {trait_class}")
                    print("Available: glycemic, lipid, hepatic, all")
                
                continue

            # Regular chatbot query
            results = system.ask(query)
            display_ai_results(results)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid QTL Chatbot with GWAS Integration")
    parser.add_argument(
        '--rebuild-db',
        action='store_true',
        help="Force a full rebuild of the vector database."
    )
    parser.add_argument(
        '--gwas-analysis',
        choices=['glycemic', 'lipid', 'hepatic', 'all'],
        help="Run GWAS analysis for specified trait class and exit."
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
    
    # NEW: Initialize the GWAS data handler (will download file on first run)
    system.setup_gwas_database()
    
    print("✅ System ready.")
    
    # Check if GWAS integration is available
    if hasattr(system, 'gwas_client') and system.gwas_client:
        print("🧬 GWAS integration ready for human-mouse cross-species analysis.")
    else:
        print("⚠️  GWAS integration not available. Check gwas_integration.py")
    
    # Handle command-line GWAS analysis
    if args.gwas_analysis:
        if args.gwas_analysis == 'all':
            trait_classes = ['glycemic', 'lipid', 'hepatic']
            for trait_class in trait_classes:
                print(f"\n🔍 Running analysis for {trait_class} traits...")
                results = system.comprehensive_gwas_qtl_analysis(trait_class)
                display_gwas_analysis_results(results)
                system.export_results_to_csv(results)
        else:
            print(f"\n🔍 Running analysis for {args.gwas_analysis} traits...")
            results = system.comprehensive_gwas_qtl_analysis(args.gwas_analysis)
            display_gwas_analysis_results(results)
            system.export_results_to_csv(results)
        
        print("\n✅ Analysis complete. Results exported to ./gwas_qtl_results/")
        exit()
    
    chatbot_loop(system)