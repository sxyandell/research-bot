#!/usr/bin/env python3
"""
Test script for the hybrid QTL system to validate it works with TRUE/FALSE cis values
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv('config.env')

def test_basic_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    
    try:
        from hybrid_qtl_system import HybridQTLSystem
        print("✅ HybridQTLSystem imported successfully")
        
        from hybrid_rag_chatbot import HybridRAGChatbot
        print("✅ HybridRAGChatbot imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_hybrid_system_basic():
    """Test basic functionality of the hybrid system."""
    print("\n🧪 Testing hybrid system setup...")
    
    try:
        # Initialize with local embeddings only (no API keys needed)
        from hybrid_qtl_system import HybridQTLSystem
        
        system = HybridQTLSystem("/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv")
        print(f"✅ System initialized with {len(system.raw_data)} QTL records")
        
        # Setup embedding models (local only)
        system.setup_embedding_models()
        print("✅ Embedding models setup complete")
        
        # Test a simple SQL query
        test_query = """
        SELECT cis, COUNT(*) as count 
        FROM qtl_peaks 
        GROUP BY cis 
        LIMIT 5
        """
        result = system.analytical_query(test_query)
        print(f"✅ SQL query successful, {len(result)} rows returned")
        print(f"   Cis/Trans distribution preview:")
        for _, row in result.iterrows():
            print(f"   - {row['cis']}: {row['count']} QTLs")
        
        # Test gene details lookup
        gene_details = system.get_gene_details('Actb')  # Common gene
        if gene_details['qtl_count'] > 0:
            print(f"✅ Gene lookup successful: Actb has {gene_details['qtl_count']} QTLs")
        else:
            print("ℹ️ Gene 'Actb' not found, trying another...")
            # Get first gene from data
            first_gene = system.raw_data['gene_symbol'].dropna().iloc[0]
            gene_details = system.get_gene_details(first_gene)
            print(f"✅ Gene lookup successful: {first_gene} has {gene_details['qtl_count']} QTLs")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid system test failed: {e}")
        return False

def test_summary_generation():
    """Test summary document generation."""
    print("\n🧪 Testing summary document generation...")
    
    try:
        from hybrid_qtl_system import HybridQTLSystem
        
        system = HybridQTLSystem("/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv")
        
        # Generate summaries
        summaries = system.generate_summary_documents()
        print(f"✅ Generated {len(summaries)} summary documents")
        
        # Check different types
        types_found = set(doc['type'] for doc in summaries)
        print(f"   Summary types: {', '.join(types_found)}")
        
        # Show sample gene summary
        gene_summaries = [doc for doc in summaries if doc['type'] == 'gene_summary']
        if gene_summaries:
            sample = gene_summaries[0]
            print(f"   Sample gene summary for: {sample['metadata']['gene_symbol']}")
            print(f"   - QTL count: {sample['metadata']['qtl_count']}")
            print(f"   - Max LOD: {sample['metadata']['max_lod']:.2f}")
            print(f"   - Cis QTLs: {sample['metadata']['cis_count']}")
            print(f"   - Trans QTLs: {sample['metadata']['trans_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Summary generation test failed: {e}")
        return False

def test_chatbot_basic():
    """Test basic chatbot functionality."""
    print("\n🧪 Testing hybrid RAG chatbot...")
    
    try:
        from hybrid_rag_chatbot import HybridRAGChatbot
        
        # Initialize with just API keys from environment (they can be None)
        google_api_key = os.getenv('GOOGLE_API_KEY')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        chatbot = HybridRAGChatbot(
            csv_file_path="/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv",
            google_api_key=google_api_key,
            openai_api_key=openai_api_key
        )
        print("✅ Chatbot initialized successfully")
        print(f"   Layer 1: {len(chatbot.qtl_system.summary_docs)} summary documents")
        print(f"   Layer 2: {len(chatbot.qtl_system.raw_data)} raw QTL records")
        
        # Test intent detection
        test_queries = [
            ("What are QTLs?", "semantic"),
            ("What are the top 10 genes?", "analytical"),
            ("How many QTLs are on chromosome 1?", "analytical"),
            ("Tell me about metabolic pathways", "semantic")
        ]
        
        print("\n   Testing intent detection:")
        for query, expected in test_queries:
            detected = chatbot.detect_query_intent(query)
            status = "✅" if detected == expected else "⚠️"
            print(f"   {status} '{query}' -> {detected} (expected: {expected})")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatbot test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Hybrid QTL System Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_hybrid_system_basic, 
        test_summary_generation,
        test_chatbot_basic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Hybrid system is ready to use.")
        print("\n💡 Next steps:")
        print("   - Run: python hybrid_qtl_system.py")
        print("   - Or run: python hybrid_rag_chatbot.py")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 