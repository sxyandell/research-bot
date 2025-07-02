import pandas as pd
import json
from typing import List, Dict, Any

class QTLDataChunker:
    def __init__(self, csv_file_path: str):
        """Initialize with the path to the QTL peaks CSV file."""
        self.csv_file = csv_file_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load the CSV data into a pandas DataFrame."""
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.data)} QTL records")
            print(f"Columns: {list(self.data.columns)}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def chunk_by_row(self) -> List[Dict[str, Any]]:
        """
        Chunk by individual rows - each QTL finding becomes one chunk.
        This is often the best approach for structured data like this.
        """
        chunks = []
        for idx, row in self.data.iterrows():
            chunk = {
                'id': f"qtl_{idx}",
                'type': 'single_qtl',
                'content': self._format_qtl_text(row),
                'metadata': {
                    'gene_symbol': row.get('gene_symbol', 'Unknown'),
                    'qtl_lod': row.get('qtl_lod', 0),
                    'qtl_chr': row.get('qtl_chr', 'Unknown'),
                    'qtl_pos': row.get('qtl_pos', 0),
                    'qtl_pval': row.get('qtl_pval', 1),
                    'cis': row.get('cis', False),
                    'gene_type': row.get('gene_type', 'Unknown')
                },
                'raw_data': row.to_dict()
            }
            chunks.append(chunk)
        return chunks
    
    def chunk_by_gene(self) -> List[Dict[str, Any]]:
        """
        Group all QTLs for the same gene into single chunks.
        Useful if you want to discuss all QTLs affecting a particular gene.
        """
        chunks = []
        grouped = self.data.groupby('gene_symbol')
        
        for gene_symbol, group in grouped:
            content_parts = []
            lod_scores = []
            
            for idx, row in group.iterrows():
                content_parts.append(self._format_qtl_text(row))
                lod_scores.append(row.get('qtl_lod', 0))
            
            chunk = {
                'id': f"gene_{gene_symbol}",
                'type': 'gene_group',
                'content': f"QTL findings for gene {gene_symbol}:\n\n" + "\n\n".join(content_parts),
                'metadata': {
                    'gene_symbol': gene_symbol,
                    'qtl_count': len(group),
                    'max_lod': max(lod_scores) if lod_scores else 0,
                    'chromosomes': list(group['qtl_chr'].unique()),
                    'gene_type': group['gene_type'].iloc[0] if 'gene_type' in group.columns else 'Unknown'
                },
                'raw_data': group.to_dict('records')
            }
            chunks.append(chunk)
        return chunks
    
    def chunk_by_chromosome(self, max_qtls_per_chunk: int = 50) -> List[Dict[str, Any]]:
        """
        Group QTLs by chromosome, splitting into multiple chunks if too many QTLs per chromosome.
        """
        chunks = []
        grouped = self.data.groupby('qtl_chr')
        
        for chr_name, group in grouped:
            # Sort by position within chromosome
            group_sorted = group.sort_values('qtl_pos')
            
            # Split into smaller chunks if needed
            for i in range(0, len(group_sorted), max_qtls_per_chunk):
                chunk_data = group_sorted.iloc[i:i+max_qtls_per_chunk]
                
                content_parts = []
                for idx, row in chunk_data.iterrows():
                    content_parts.append(self._format_qtl_text(row))
                
                chunk = {
                    'id': f"chr_{chr_name}_part_{i//max_qtls_per_chunk + 1}",
                    'type': 'chromosome_group',
                    'content': f"QTL findings on chromosome {chr_name} (positions {chunk_data['qtl_pos'].min():.2f}-{chunk_data['qtl_pos'].max():.2f} Mb):\n\n" + "\n\n".join(content_parts),
                    'metadata': {
                        'chromosome': chr_name,
                        'qtl_count': len(chunk_data),
                        'position_range': [chunk_data['qtl_pos'].min(), chunk_data['qtl_pos'].max()],
                        'max_lod': chunk_data['qtl_lod'].max() if 'qtl_lod' in chunk_data.columns else 0,
                        'genes': list(chunk_data['gene_symbol'].unique())
                    },
                    'raw_data': chunk_data.to_dict('records')
                }
                chunks.append(chunk)
        return chunks
    
    def chunk_by_significance(self, high_lod_threshold: float = 10.0, chunk_size: int = 25) -> List[Dict[str, Any]]:
        """
        Separate highly significant QTLs from others and chunk accordingly.
        """
        chunks = []
        
        # High significance QTLs
        high_sig = self.data[self.data['qtl_lod'] >= high_lod_threshold].copy()
        if not high_sig.empty:
            high_sig_sorted = high_sig.sort_values('qtl_lod', ascending=False)
            
            for i in range(0, len(high_sig_sorted), chunk_size):
                chunk_data = high_sig_sorted.iloc[i:i+chunk_size]
                
                content_parts = []
                for idx, row in chunk_data.iterrows():
                    content_parts.append(self._format_qtl_text(row))
                
                chunk = {
                    'id': f"high_sig_qtls_part_{i//chunk_size + 1}",
                    'type': 'high_significance',
                    'content': f"Highly significant QTLs (LOD ≥ {high_lod_threshold}):\n\n" + "\n\n".join(content_parts),
                    'metadata': {
                        'significance_level': 'high',
                        'lod_threshold': high_lod_threshold,
                        'qtl_count': len(chunk_data),
                        'lod_range': [chunk_data['qtl_lod'].min(), chunk_data['qtl_lod'].max()]
                    },
                    'raw_data': chunk_data.to_dict('records')
                }
                chunks.append(chunk)
        
        # Moderate significance QTLs
        mod_sig = self.data[self.data['qtl_lod'] < high_lod_threshold].copy()
        if not mod_sig.empty:
            mod_sig_sorted = mod_sig.sort_values('qtl_lod', ascending=False)
            
            for i in range(0, len(mod_sig_sorted), chunk_size * 2):  # Larger chunks for moderate significance
                chunk_data = mod_sig_sorted.iloc[i:i+chunk_size*2]
                
                content_parts = []
                for idx, row in chunk_data.iterrows():
                    content_parts.append(self._format_qtl_text(row))
                
                chunk = {
                    'id': f"mod_sig_qtls_part_{i//(chunk_size*2) + 1}",
                    'type': 'moderate_significance',
                    'content': f"Moderate significance QTLs (LOD < {high_lod_threshold}):\n\n" + "\n\n".join(content_parts),
                    'metadata': {
                        'significance_level': 'moderate',
                        'lod_threshold': high_lod_threshold,
                        'qtl_count': len(chunk_data),
                        'lod_range': [chunk_data['qtl_lod'].min(), chunk_data['qtl_lod'].max()]
                    },
                    'raw_data': chunk_data.to_dict('records')
                }
                chunks.append(chunk)
        
        return chunks
    
    def chunk_by_genomic_regions(self, qtls_per_chunk: int = 30) -> List[Dict[str, Any]]:
        """
        Create chunks based on genomic regions, grouping nearby QTLs together.
        This creates fewer chunks while preserving spatial relationships.
        """
        chunks = []
        
        # Group by chromosome first
        for chr_name, chr_data in self.data.groupby('qtl_chr'):
            # Sort by position
            chr_sorted = chr_data.sort_values('qtl_pos')
            
            # Group into chunks of nearby QTLs
            for i in range(0, len(chr_sorted), qtls_per_chunk):
                chunk_data = chr_sorted.iloc[i:i+qtls_per_chunk]
                
                # Create summary for this genomic region
                pos_start = chunk_data['qtl_pos'].min()
                pos_end = chunk_data['qtl_pos'].max()
                top_genes = chunk_data.nlargest(5, 'qtl_lod')['gene_symbol'].tolist()
                
                content_parts = [
                    f"Genomic region: Chromosome {chr_name}, {pos_start:.2f} - {pos_end:.2f} Mb",
                    f"Contains {len(chunk_data)} QTLs",
                    f"Top genes by LOD score: {', '.join(top_genes[:3])}",
                    "",
                    "QTL Details:"
                ]
                
                for idx, row in chunk_data.iterrows():
                    content_parts.append(self._format_qtl_text(row))
                
                chunk = {
                    'id': f"region_chr{chr_name}_{i//qtls_per_chunk + 1}",
                    'type': 'genomic_region',
                    'content': "\n".join(content_parts),
                    'metadata': {
                        'chromosome': chr_name,
                        'position_start': pos_start,
                        'position_end': pos_end,
                        'qtl_count': len(chunk_data),
                        'max_lod': chunk_data['qtl_lod'].max(),
                        'top_genes': top_genes,
                        'region_span_mb': pos_end - pos_start
                    },
                    'raw_data': chunk_data.to_dict('records')
                }
                chunks.append(chunk)
        
        return chunks
    
    def chunk_by_gene_pathways(self, genes_per_chunk: int = 20) -> List[Dict[str, Any]]:
        """
        Group genes together and create chunks based on sets of genes.
        Better than individual genes, fewer chunks than row-by-row.
        """
        chunks = []
        
        # Get unique genes
        unique_genes = self.data['gene_symbol'].unique()
        
        # Group genes into chunks
        for i in range(0, len(unique_genes), genes_per_chunk):
            gene_set = unique_genes[i:i+genes_per_chunk]
            
            # Get all QTLs for these genes
            chunk_data = self.data[self.data['gene_symbol'].isin(gene_set)]
            
            # Sort by LOD score within this gene set
            chunk_data_sorted = chunk_data.sort_values('qtl_lod', ascending=False)
            
            content_parts = [
                f"Gene set {i//genes_per_chunk + 1}: {len(gene_set)} genes with {len(chunk_data)} QTLs",
                f"Genes: {', '.join(gene_set[:10])}{'...' if len(gene_set) > 10 else ''}",
                "",
                "QTL Details:"
            ]
            
            for idx, row in chunk_data_sorted.iterrows():
                content_parts.append(self._format_qtl_text(row))
            
            chunk = {
                'id': f"gene_set_{i//genes_per_chunk + 1}",
                'type': 'gene_set',
                'content': "\n".join(content_parts),
                'metadata': {
                    'gene_count': len(gene_set),
                    'genes': gene_set.tolist(),
                    'qtl_count': len(chunk_data),
                    'max_lod': chunk_data['qtl_lod'].max(),
                    'chromosomes': chunk_data['qtl_chr'].unique().tolist(),
                    'avg_lod': chunk_data['qtl_lod'].mean()
                },
                'raw_data': chunk_data.to_dict('records')
            }
            chunks.append(chunk)
        
        return chunks
    
    def chunk_top_qtls_only(self, top_n: int = 200, chunk_size: int = 25) -> List[Dict[str, Any]]:
        """
        Only include the top N most significant QTLs, chunked appropriately.
        Great for focusing on the most important findings.
        """
        chunks = []
        
        # Get top N QTLs by LOD score
        top_qtls = self.data.nlargest(top_n, 'qtl_lod')
        
        # Chunk them
        for i in range(0, len(top_qtls), chunk_size):
            chunk_data = top_qtls.iloc[i:i+chunk_size]
            
            rank_start = i + 1
            rank_end = min(i + chunk_size, top_n)
            
            content_parts = [
                f"Top QTLs ranked {rank_start}-{rank_end} by LOD score",
                f"LOD score range: {chunk_data['qtl_lod'].min():.2f} - {chunk_data['qtl_lod'].max():.2f}",
                "",
                "QTL Details:"
            ]
            
            for idx, row in chunk_data.iterrows():
                content_parts.append(self._format_qtl_text(row))
            
            chunk = {
                'id': f"top_qtls_{rank_start}_{rank_end}",
                'type': 'top_qtls',
                'content': "\n".join(content_parts),
                'metadata': {
                    'rank_start': rank_start,
                    'rank_end': rank_end,
                    'qtl_count': len(chunk_data),
                    'lod_range': [chunk_data['qtl_lod'].min(), chunk_data['qtl_lod'].max()],
                    'genes': chunk_data['gene_symbol'].tolist(),
                    'chromosomes': chunk_data['qtl_chr'].unique().tolist()
                },
                'raw_data': chunk_data.to_dict('records')
            }
            chunks.append(chunk)
        
        return chunks
    
    def _format_qtl_text(self, row) -> str:
        """Format a single QTL record into readable text for RAG."""
        text_parts = []
        
        # Basic QTL info
        gene_symbol = row.get('gene_symbol', 'Unknown gene')
        qtl_lod = row.get('qtl_lod', 'Unknown')
        qtl_pval = row.get('qtl_pval', 'Unknown')
        
        text_parts.append(f"Gene: {gene_symbol}")
        text_parts.append(f"LOD Score: {qtl_lod}")
        text_parts.append(f"P-value: {qtl_pval}")
        
        # Location info
        qtl_chr = row.get('qtl_chr', 'Unknown')
        qtl_pos = row.get('qtl_pos', 'Unknown')
        text_parts.append(f"Location: Chromosome {qtl_chr}, Position {qtl_pos} Mb")
        
        # Confidence interval
        ci_lo = row.get('qtl_ci_lo', None)
        ci_hi = row.get('qtl_ci_hi', None)
        if ci_lo is not None and ci_hi is not None:
            text_parts.append(f"Confidence Interval: {ci_lo:.3f} - {ci_hi:.3f} Mb")
        
        # Cis/trans
        cis = row.get('cis', None)
        if cis is not None:
            text_parts.append(f"Type: {'cis-acting' if cis else 'trans-acting'}")
        
        # Gene info
        gene_type = row.get('gene_type', None)
        if gene_type:
            text_parts.append(f"Gene Type: {gene_type}")
        
        # Additional stats
        qtl_qval = row.get('qtl_qval', None)
        if qtl_qval is not None:
            text_parts.append(f"Q-value (FDR): {qtl_qval}")
        
        return " | ".join(text_parts)
    
    def save_chunks(self, chunks: List[Dict], output_file: str):
        """Save chunks to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(chunks, f, indent=2, default=str)
        print(f"Saved {len(chunks)} chunks to {output_file}")

# Example usage
if __name__ == "__main__":
    # Initialize chunker
    chunker = QTLDataChunker("/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv")
    
    # Run only top QTLs chunking
    print("\n=== Top QTLs only chunking ===")
    top_qtls_chunks = chunker.chunk_top_qtls_only(top_n=200, chunk_size=25)
    print(f"Created {len(top_qtls_chunks)} chunks from top 200 QTLs")
    
    # Show some stats about the chunks
    if top_qtls_chunks:
        total_qtls = sum(chunk['metadata']['qtl_count'] for chunk in top_qtls_chunks)
        lod_ranges = [chunk['metadata']['lod_range'] for chunk in top_qtls_chunks]
        print(f"Total QTLs included: {total_qtls}")
        print(f"LOD score range: {min(r[0] for r in lod_ranges):.2f} - {max(r[1] for r in lod_ranges):.2f}")
        
        # Show first chunk as example
        print(f"\nExample chunk content (first 500 chars):")
        print(top_qtls_chunks[0]['content'][:500] + "...")
    
    # Save the chunks
    chunker.save_chunks(top_qtls_chunks, "qtl_chunks_top_qtls_only.json")
    print(f"\n✅ Saved chunks to qtl_chunks_top_qtls_only.json")
