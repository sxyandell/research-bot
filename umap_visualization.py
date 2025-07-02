import numpy as np
import umap
import matplotlib.pyplot as plt
import chromadb
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('config.env')

def get_embeddings_from_chromadb():
    """Get embeddings from the existing ChromaDB collection"""
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get the existing collection
    collection = chroma_client.get_collection(name="qtl_database")
    
    # Get all documents with their embeddings
    all_results = collection.get(
        include=['documents', 'embeddings', 'metadatas']
    )
    
    return all_results

def visualize_embeddings_umap():
    """Visualize the embeddings using UMAP"""
    
    # Get embeddings from ChromaDB
    print("Loading embeddings from ChromaDB...")
    results = get_embeddings_from_chromadb()
    
    embeddings = results['embeddings']
    documents = results['documents']
    metadatas = results['metadatas']
    
    print(f"Loaded {len(embeddings)} document embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    # Convert embeddings to numpy array
    X = np.array(embeddings)
    
    # Reduce to 2D using UMAP
    print("Applying UMAP dimensionality reduction...")
    umap_model = umap.UMAP(n_neighbors=min(15, len(X)-1), min_dist=0.1, metric='cosine', random_state=42)
    X_embedded = umap_model.fit_transform(X)
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Color points by chromosome
    chromosomes = [meta.get('qtl_chr', 'Unknown') if meta else 'Unknown' for meta in metadatas]
    unique_chromosomes = list(set(chromosomes))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_chromosomes)))
    
    # Plot points colored by chromosome
    for i, chrom in enumerate(unique_chromosomes):
        mask = [c == chrom for c in chromosomes]
        plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                   c=[colors[i]], label=f'Chr {chrom}', s=100, alpha=0.7)
    
    # Add labels for each point
    for i, (x, y) in enumerate(X_embedded):
        # Get gene symbol from metadata
        gene_symbol = metadatas[i].get('gene_symbol', f'Doc_{i}') if metadatas[i] else f'Doc_{i}'
        plt.annotate(gene_symbol, (x, y), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.8)
    
    plt.title("UMAP Visualization of QTL Document Embeddings\nColored by Chromosome", 
              fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.legend(title="Chromosome", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('qtl_embeddings_umap.png', dpi=300, bbox_inches='tight')
    print("UMAP visualization saved as 'qtl_embeddings_umap.png'")
    
    # Show the plot
    plt.show()
    
    # Print summary information
    print("\n" + "="*50)
    print("UMAP EMBEDDING VISUALIZATION SUMMARY")
    print("="*50)
    print(f"Total documents: {len(documents)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"UMAP n_neighbors used: {min(15, len(X)-1)}")
    print(f"UMAP min_dist: 0.1")
    print(f"UMAP metric: cosine")
    
    # Show document information
    print("\nDocument Information:")
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        gene_symbol = meta.get('gene_symbol', f'Doc_{i}') if meta else f'Doc_{i}'
        chromosome = meta.get('qtl_chr', 'Unknown') if meta else 'Unknown'
        lod_score = meta.get('qtl_lod', 'Unknown') if meta else 'Unknown'
        print(f"{i+1:2d}. {gene_symbol:15s} | Chr {chromosome:2s} | LOD: {lod_score:8.2f}")

def create_umap_analysis():
    """Create additional UMAP-based visualizations"""
    
    results = get_embeddings_from_chromadb()
    embeddings = results['embeddings']
    metadatas = results['metadatas']
    
    X = np.array(embeddings)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('QTL Document Embeddings - UMAP Analysis', fontsize=16, fontweight='bold')
    
    # 1. UMAP visualization colored by LOD score
    umap_model = umap.UMAP(n_neighbors=min(15, len(X)-1), min_dist=0.1, metric='cosine', random_state=42)
    X_embedded = umap_model.fit_transform(X)
    
    lod_scores = [meta.get('qtl_lod', 0) if meta else 0 for meta in metadatas]
    scatter1 = axes[0, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                                 c=lod_scores, cmap='viridis', s=100, alpha=0.7)
    axes[0, 0].set_title('UMAP: Colored by LOD Score')
    axes[0, 0].set_xlabel('UMAP Component 1')
    axes[0, 0].set_ylabel('UMAP Component 2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='LOD Score')
    
    # 2. UMAP visualization colored by chromosome
    chromosomes = [meta.get('qtl_chr', 'Unknown') if meta else 'Unknown' for meta in metadatas]
    unique_chromosomes = list(set(chromosomes))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_chromosomes)))
    
    for i, chrom in enumerate(unique_chromosomes):
        mask = [c == chrom for c in chromosomes]
        axes[0, 1].scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                          c=[colors[i]], label=f'Chr {chrom}', s=100, alpha=0.7)
    axes[0, 1].set_title('UMAP: Colored by Chromosome')
    axes[0, 1].set_xlabel('UMAP Component 1')
    axes[0, 1].set_ylabel('UMAP Component 2')
    axes[0, 1].legend(title="Chromosome", fontsize=8)
    
    # 3. LOD Score distribution
    axes[1, 0].hist(lod_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Distribution of LOD Scores')
    axes[1, 0].set_xlabel('LOD Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Chromosome distribution
    chrom_counts = {}
    for chrom in chromosomes:
        chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
    
    chrom_names = list(chrom_counts.keys())
    chrom_values = list(chrom_counts.values())
    
    bars = axes[1, 1].bar(chrom_names, chrom_values, alpha=0.7, color='lightcoral')
    axes[1, 1].set_title('Distribution by Chromosome')
    axes[1, 1].set_xlabel('Chromosome')
    axes[1, 1].set_ylabel('Number of QTLs')
    
    # Add value labels on bars
    for bar, value in zip(bars, chrom_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('qtl_embeddings_umap_analysis.png', dpi=300, bbox_inches='tight')
    print("UMAP analysis saved as 'qtl_embeddings_umap_analysis.png'")
    plt.show()

def compare_umap_parameters():
    """Compare different UMAP parameters"""
    
    results = get_embeddings_from_chromadb()
    embeddings = results['embeddings']
    metadatas = results['metadatas']
    
    X = np.array(embeddings)
    
    # Different UMAP configurations
    configs = [
        {'n_neighbors': 5, 'min_dist': 0.1, 'name': 'Low neighbors (5)'},
        {'n_neighbors': 15, 'min_dist': 0.1, 'name': 'Default (15)'},
        {'n_neighbors': 30, 'min_dist': 0.1, 'name': 'High neighbors (30)'},
        {'n_neighbors': 15, 'min_dist': 0.01, 'name': 'Low min_dist (0.01)'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('UMAP Parameter Comparison', fontsize=16, fontweight='bold')
    
    for idx, config in enumerate(configs):
        row = idx // 2
        col = idx % 2
        
        # Apply UMAP with current config
        umap_model = umap.UMAP(
            n_neighbors=min(config['n_neighbors'], len(X)-1), 
            min_dist=config['min_dist'], 
            metric='cosine', 
            random_state=42
        )
        X_embedded = umap_model.fit_transform(X)
        
        # Color by chromosome
        chromosomes = [meta.get('qtl_chr', 'Unknown') if meta else 'Unknown' for meta in metadatas]
        unique_chromosomes = list(set(chromosomes))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_chromosomes)))
        
        for i, chrom in enumerate(unique_chromosomes):
            mask = [c == chrom for c in chromosomes]
            axes[row, col].scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                                 c=[colors[i]], label=f'Chr {chrom}', s=80, alpha=0.7)
        
        axes[row, col].set_title(config['name'])
        axes[row, col].set_xlabel('UMAP Component 1')
        axes[row, col].set_ylabel('UMAP Component 2')
        if idx == 0:  # Only show legend on first plot
            axes[row, col].legend(title="Chromosome", fontsize=6)
    
    plt.tight_layout()
    plt.savefig('qtl_embeddings_umap_comparison.png', dpi=300, bbox_inches='tight')
    print("UMAP parameter comparison saved as 'qtl_embeddings_umap_comparison.png'")
    plt.show()

if __name__ == "__main__":
    print("QTL Document Embeddings - UMAP Visualization")
    print("="*50)
    
    # Check if ChromaDB collection exists
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="qtl_database")
        print("✅ Found existing ChromaDB collection")
        
        # Create UMAP visualizations
        visualize_embeddings_umap()
        create_umap_analysis()
        compare_umap_parameters()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please make sure to run vectordb.py first to create the ChromaDB collection.") 