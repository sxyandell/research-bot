#!/usr/bin/env python3
"""
Test script for the Ensembl API tool functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_qtl_system import HybridQTLSystem

def test_ensembl_api():
    """Test the Ensembl API tool functionality."""
    
    print("🧪 Testing Ensembl API Tool Functionality")
    print("=" * 50)
    
    # Initialize system (you'll need to provide a valid CSV file path)
    csv_file = input("Enter path to your QTL CSV file (or press Enter to skip CSV loading): ").strip()
    
    if not csv_file:
        print("⚠️ Skipping CSV loading - testing only Ensembl API functionality")
        # Create a minimal system without CSV loading
        system = HybridQTLSystem.__new__(HybridQTLSystem)
        system.ensembl_client = None
        system.ensembl_client = system.__class__.__init__.__globals__['EnsemblAPIClient']()
    else:
        try:
            system = HybridQTLSystem(csv_file)
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return
    
    print(f"\n✅ System initialized")
    
    # Test 1: Basic Ensembl client availability
    print(f"\n🔬 Test 1: Ensembl Client Availability")
    if system.ensembl_client:
        print("✅ Ensembl client is available")
    else:
        print("❌ Ensembl client not available")
        return
    
    # Test 2: Basic API connectivity
    print(f"\n🔬 Test 2: Basic API Connectivity")
    try:
        if system.test_ensembl_connection():
            print("✅ Ensembl API connectivity successful")
        else:
            print("❌ Ensembl API connectivity failed")
            return
    except AttributeError:
        print("⚠️ test_ensembl_connection method not found, testing basic connectivity manually...")
        # Test basic connectivity manually
        try:
            import requests
            response = requests.get("https://rest.ensembl.org/info/species", 
                                 headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                print("✅ Basic Ensembl API connectivity successful")
            else:
                print(f"❌ Basic API connectivity failed: {response.status_code}")
                return
        except Exception as e:
            print(f"❌ Basic API connectivity failed: {e}")
            return
    
    # Test 3: Individual API methods
    print(f"\n🔬 Test 3: Individual API Methods")
    
    test_genes = ["Apoe", "Gnai3", "Actb"]
    
    for gene in test_genes:
        print(f"\n--- Testing {gene} ---")
        
        # Test gene info
        try:
            gene_info = system.ensembl_client.get_gene_info(gene)
            if gene_info:
                print(f"✅ Gene info: Found {len(gene_info)} data fields")
            else:
                print(f"⚠️ Gene info: No data returned")
        except Exception as e:
            print(f"❌ Gene info: {e}")
        
        # Test variants
        try:
            variants = system.ensembl_client.get_variants(gene)
            if variants:
                print(f"✅ Variants: Found {len(variants)} variants")
            else:
                print(f"⚠️ Variants: No variants found")
        except Exception as e:
            print(f"❌ Variants: {e}")
        
        # Test orthologs
        try:
            orthologs = system.ensembl_client.get_orthologs(gene, "homo_sapiens")
            if orthologs:
                print(f"✅ Orthologs: Found {len(orthologs)} human orthologs")
            else:
                print(f"⚠️ Orthologs: No human orthologs found")
        except Exception as e:
            print(f"❌ Orthologs: {e}")
    
    # Test 4: New Ensembl API tool
    print(f"\n🔬 Test 4: New Ensembl API Tool")
    try:
        test_results = system.test_ensembl_api_tool()
        print("✅ Ensembl API tool test completed")
        
        print("\n📊 Tool Test Results:")
        for test_name, result in test_results.items():
            if 'error' in result:
                print(f"  ❌ {test_name}: {result['error']}")
            else:
                print(f"  ✅ {test_name}: Success")
                if 'result' in result and result['result']:
                    if isinstance(result['result'], dict):
                        print(f"    Data fields: {list(result['result'].keys())[:5]}")
                    elif isinstance(result['result'], list):
                        print(f"    Result count: {len(result['result'])}")
                        
    except Exception as e:
        print(f"❌ Ensembl API tool test failed: {e}")
    
    # Test 5: Tool routing (if CSV data is available)
    if hasattr(system, 'raw_data') and system.raw_data is not None:
        print(f"\n🔬 Test 5: Tool Routing with LLM")
        print("Testing if the LLM can route queries to the Ensembl API tool...")
        
        test_queries = [
            "What are the variants for gene Apoe?",
            "Show me gene info for Gnai3 from Ensembl",
            "What orthologs exist for gene Actb?",
            "Get gene function data for Gapdh"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                result = system.intelligent_router(query)
                print(f"Tool chosen: {result.get('method', 'Unknown')}")
                print(f"Arguments: {result.get('arguments', {})}")
            except Exception as e:
                print(f"❌ Routing failed: {e}")
    
    print(f"\n🎉 Ensembl API testing complete!")
    print("💡 The Ensembl API tool is now available for use with the intelligent router.")

if __name__ == "__main__":
    test_ensembl_api() 