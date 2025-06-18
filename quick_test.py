#!/usr/bin/env python3
"""Quick test of computational capabilities"""

import json
import pandas as pd

def load_qtl_data():
    """Load QTL data from enhanced chunks."""
    try:
        with open('enhanced_rag_chunks.json', 'r') as f:
            data = json.load(f)
            chunks = data.get('enhanced_chunks', [])
        
        all_records = []
        for chunk in chunks:
            if chunk.get('type') == 'top_qtls' and 'raw_data' in chunk:
                all_records.extend(chunk['raw_data'])
        
        return pd.DataFrame(all_records)
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def perform_computations():
    """Perform sample computations on QTL data."""
    print("🧪 Testing QTL Data Computations")
    print("=" * 40)
    
    qtl_data = load_qtl_data()
    
    if qtl_data.empty:
        print("❌ No data available")
        return
    
    print(f"✅ Loaded {len(qtl_data)} QTL records\n")
    
    # Basic statistics
    print("📊 BASIC STATISTICS:")
    print(f"Average LOD score: {qtl_data['qtl_lod'].mean():.2f}")
    print(f"Maximum LOD score: {qtl_data['qtl_lod'].max():.2f}")
    print(f"Minimum LOD score: {qtl_data['qtl_lod'].min():.2f}")
    print(f"Standard deviation: {qtl_data['qtl_lod'].std():.2f}")
    
    # Cis vs Trans
    print(f"\n🔗 CIS vs TRANS:")
    cis_count = qtl_data['cis'].sum()
    trans_count = len(qtl_data) - cis_count
    print(f"Cis-acting QTLs: {cis_count} ({100*cis_count/len(qtl_data):.1f}%)")
    print(f"Trans-acting QTLs: {trans_count} ({100*trans_count/len(qtl_data):.1f}%)")
    
    # Average by type
    cis_avg = qtl_data[qtl_data['cis'] == True]['qtl_lod'].mean()
    trans_avg = qtl_data[qtl_data['cis'] == False]['qtl_lod'].mean()
    print(f"Average LOD for cis QTLs: {cis_avg:.2f}")
    print(f"Average LOD for trans QTLs: {trans_avg:.2f}")
    
    # Chromosome distribution
    print(f"\n🧬 CHROMOSOME DISTRIBUTION:")
    chr_counts = qtl_data['qtl_chr'].value_counts().sort_index()
    print("Top 5 chromosomes by QTL count:")
    for chr_num, count in chr_counts.head().items():
        print(f"  Chromosome {chr_num}: {count} QTLs")
    
    # Top genes
    print(f"\n🧬 TOP GENES BY LOD SCORE:")
    top_genes = qtl_data.nlargest(5, 'qtl_lod')[['gene_symbol', 'qtl_lod', 'qtl_chr', 'cis']]
    for idx, row in top_genes.iterrows():
        regulation = "cis" if row['cis'] else "trans"
        print(f"  {row['gene_symbol']}: LOD {row['qtl_lod']:.1f} (Chr {row['qtl_chr']}, {regulation})")
    
    print(f"\n✅ Computational analysis complete!")
    print(f"💡 Your enhanced chatbot can now answer questions like:")
    print(f"   • 'What is the average LOD score?' → {qtl_data['qtl_lod'].mean():.2f}")
    print(f"   • 'How many cis QTLs?' → {cis_count}")
    print(f"   • 'Which gene has the highest LOD?' → {qtl_data.loc[qtl_data['qtl_lod'].idxmax(), 'gene_symbol']}")

if __name__ == "__main__":
    perform_computations() 