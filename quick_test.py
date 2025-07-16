#!/usr/bin/env python3
"""
Quick test of multi-file QTL system
"""

import pandas as pd
from hybrid_qtl_system import HybridQTLSystem
import logging

logging.basicConfig(level=logging.INFO)

def test_basic_queries():
    """Test basic SQL queries on the unified dataset."""
    
    print("🧪 Testing Multi-File QTL System with unified dataset...")
    
    # Initialize with unified dataset
    system = HybridQTLSystem("./unified_qtl_data.csv")
    
    print(f"✅ Loaded {len(system.raw_data):,} QTL records")
    print(f"📊 Unique genes: {system.raw_data['gene_symbol'].nunique():,}")
    print(f"🏷️ Trait types: {system.raw_data['trait_type'].unique()}")
    
    # Test queries
    test_queries = [
        # Basic stats
        "SELECT COUNT(*) as total_qtls FROM qtl_peaks",
        
        # Highest LOD score
        """
        SELECT gene_symbol, trait_type, MAX(qtl_lod) as max_lod, COUNT(*) as qtl_count
        FROM qtl_peaks 
        WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
        GROUP BY gene_symbol, trait_type
        ORDER BY max_lod DESC 
        LIMIT 1
        """,
        
        # Count by trait type
        """
        SELECT trait_type, COUNT(*) as qtl_count, COUNT(DISTINCT gene_symbol) as unique_genes
        FROM qtl_peaks 
        WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
        GROUP BY trait_type 
        ORDER BY qtl_count DESC
        """,
        
        # Top 5 genes by LOD
        """
        SELECT gene_symbol, trait_type, MAX(qtl_lod) as max_lod
        FROM qtl_peaks 
        WHERE gene_symbol IS NOT NULL AND gene_symbol != 'nan'
        GROUP BY gene_symbol, trait_type
        ORDER BY max_lod DESC 
        LIMIT 5
        """
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}:")
        try:
            result = system.analytical_query(query)
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_basic_queries() 