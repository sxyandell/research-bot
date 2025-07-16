#!/usr/bin/env python3
"""
Test script for Multi-File QTL System

Validates that the system can:
1. Discover and load all 40 QTL data files
2. Create unified database and vector store
3. Handle semantic and analytical queries
4. Perform cross-trait analysis
"""

import pandas as pd
import time
from multi_file_qtl_system import MultiFileQTLSystem

def test_file_discovery():
    """Test file discovery functionality."""
    print("🔍 Testing file discovery...")
    
    system = MultiFileQTLSystem()
    files_by_trait = system.discover_files()
    
    print(f"Discovered trait types: {list(files_by_trait.keys())}")
    
    total_files = sum(len(files) for files in files_by_trait.values())
    print(f"Total files found: {total_files}")
    
    # Expected trait types
    expected_traits = [
        'clinical_traits', 'liver_genes', 'liver_isoforms', 
        'liver_lipids', 'liver_splice_juncs', 'plasma_metabolites'
    ]
    
    for trait in expected_traits:
        if trait in files_by_trait:
            print(f"✅ {trait}: {len(files_by_trait[trait])} files")
        else:
            print(f"❌ Missing trait type: {trait}")
    
    return files_by_trait

def test_data_loading():
    """Test data loading and database setup."""
    print("\n📊 Testing data loading...")
    
    system = MultiFileQTLSystem()
    start_time = time.time()
    
    try:
        system.load_all_data()
        
        # Check loaded data
        total_records = sum(len(df) for df in system.trait_data.values())
        print(f"Total records loaded: {total_records:,}")
        
        # Test SQL database
        test_query = "SELECT COUNT(*) as total FROM qtl_data"
        result = system.sql_query(test_query)
        db_records = result.iloc[0, 0] if not result.empty else 0
        
        print(f"Records in SQL database: {db_records:,}")
        
        if total_records == db_records:
            print("✅ Data loading successful")
        else:
            print("❌ Mismatch between loaded data and database")
        
        # Print trait summary
        print("\nTrait summary:")
        for trait, df in system.trait_data.items():
            print(f"  {trait}: {len(df):,} records")
        
        load_time = time.time() - start_time
        print(f"Loading time: {load_time:.1f} seconds")
        
        return system
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_vector_store(system):
    """Test vector store creation."""
    print("\n🔮 Testing vector store setup...")
    
    start_time = time.time()
    
    try:
        # Generate summaries
        summaries = system.generate_summaries()
        print(f"Generated {len(summaries)} summary documents")
        
        # Setup vector store
        system.setup_vector_store(summaries)
        
        # Test semantic search
        test_query = "What are QTLs?"
        results = system.semantic_search(test_query, n_results=3)
        
        if results:
            print("✅ Vector store working")
            print(f"Sample search result: {results[0]['content'][:100]}...")
        else:
            print("❌ Vector store search failed")
        
        setup_time = time.time() - start_time
        print(f"Vector store setup time: {setup_time:.1f} seconds")
        
    except Exception as e:
        print(f"❌ Error setting up vector store: {e}")

def test_analytical_queries(system):
    """Test analytical SQL queries."""
    print("\n📈 Testing analytical queries...")
    
    test_queries = [
        "SELECT trait_type, COUNT(*) as count FROM qtl_data GROUP BY trait_type ORDER BY count DESC",
        "SELECT COUNT(DISTINCT gene_symbol) as unique_genes FROM qtl_data",
        "SELECT MAX(qtl_lod) as max_lod FROM qtl_data",
        "SELECT COUNT(*) as cis_qtls FROM qtl_data WHERE cis = 'TRUE'",
        "SELECT qtl_chr, COUNT(*) as count FROM qtl_data GROUP BY qtl_chr ORDER BY qtl_chr LIMIT 5"
    ]
    
    for i, query in enumerate(test_queries, 1):
        try:
            result = system.sql_query(query)
            if not result.empty:
                print(f"✅ Query {i}: {len(result)} rows returned")
            else:
                print(f"❌ Query {i}: No results")
        except Exception as e:
            print(f"❌ Query {i} failed: {e}")

def test_semantic_queries(system):
    """Test semantic search queries."""
    print("\n🔍 Testing semantic queries...")
    
    test_queries = [
        "What are quantitative trait loci?",
        "Explain liver gene expression",
        "Tell me about plasma metabolites",
        "How do cis-acting QTLs work?",
        "Describe genetic regulation"
    ]
    
    for i, query in enumerate(test_queries, 1):
        try:
            results = system.semantic_search(query, n_results=2)
            if results:
                print(f"✅ Semantic query {i}: {len(results)} results")
            else:
                print(f"❌ Semantic query {i}: No results")
        except Exception as e:
            print(f"❌ Semantic query {i} failed: {e}")

def test_trait_filtering(system):
    """Test trait-specific filtering."""
    print("\n🎯 Testing trait filtering...")
    
    trait_filters = [
        'liver_genes',
        'liver_lipids', 
        'clinical_traits',
        'plasma_metabolites'
    ]
    
    for trait in trait_filters:
        try:
            results = system.semantic_search(
                "high LOD score QTLs", 
                n_results=2, 
                trait_filter=trait
            )
            if results:
                print(f"✅ {trait} filter: {len(results)} results")
            else:
                print(f"❌ {trait} filter: No results")
        except Exception as e:
            print(f"❌ {trait} filter failed: {e}")

def test_cross_trait_queries(system):
    """Test cross-trait analysis capabilities."""
    print("\n🔀 Testing cross-trait analysis...")
    
    # Compare trait types
    try:
        comparison_query = """
        SELECT 
            trait_type,
            COUNT(*) as total_qtls,
            COUNT(DISTINCT gene_symbol) as unique_genes,
            AVG(qtl_lod) as avg_lod,
            MAX(qtl_lod) as max_lod
        FROM qtl_data 
        GROUP BY trait_type 
        ORDER BY total_qtls DESC
        """
        
        result = system.sql_query(comparison_query)
        if not result.empty:
            print("✅ Cross-trait comparison successful")
            print(result.to_string(index=False))
        else:
            print("❌ Cross-trait comparison failed")
            
    except Exception as e:
        print(f"❌ Cross-trait analysis error: {e}")

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🧬 Multi-File QTL System - Comprehensive Test")
    print("=" * 60)
    
    start_time = time.time()
    
    # Test 1: File discovery
    files_by_trait = test_file_discovery()
    
    # Test 2: Data loading
    system = test_data_loading()
    
    if not system:
        print("❌ Cannot continue tests - data loading failed")
        return
    
    # Test 3: Vector store
    test_vector_store(system)
    
    # Test 4: Analytical queries
    test_analytical_queries(system)
    
    # Test 5: Semantic queries
    test_semantic_queries(system)
    
    # Test 6: Trait filtering
    test_trait_filtering(system)
    
    # Test 7: Cross-trait analysis
    test_cross_trait_queries(system)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n⏱️  Total test time: {total_time:.1f} seconds")
    print("🎉 Comprehensive test completed!")
    
    return system

if __name__ == "__main__":
    system = run_comprehensive_test()
    
    if system:
        print("\n" + "=" * 60)
        print("🚀 System ready for interactive use!")
        print("Run: python multi_file_qtl_chatbot.py") 