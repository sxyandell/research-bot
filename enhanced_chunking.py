import pandas as pd
import json
from typing import List, Dict, Any, Optional
import os

class EnhancedQTLChunker:
    def __init__(self, primary_csv_path: str):
        """Initialize with the primary QTL peaks CSV file."""
        self.primary_csv = primary_csv_path
        self.qtl_data = None
        self.gene_annotations = None
        self.pathway_data = None
        self.phenotype_info = None
        self.load_primary_data()
    
    def load_primary_data(self):
        """Load the main QTL data."""
        try:
            self.qtl_data = pd.read_csv(self.primary_csv)
            print(f"✅ Loaded {len(self.qtl_data)} QTL records")
        except Exception as e:
            print(f"❌ Error loading primary data: {e}")
    
    def add_gene_annotations(self, gene_annotation_file: str):
        """Add gene annotation data (e.g., gene descriptions, functions)."""
        try:
            self.gene_annotations = pd.read_csv(gene_annotation_file)
            print(f"✅ Loaded gene annotations for {len(self.gene_annotations)} genes")
        except Exception as e:
            print(f"❌ Error loading gene annotations: {e}")
    
    def add_pathway_data(self, pathway_file: str):
        """Add pathway/GO term data."""
        try:
            self.pathway_data = pd.read_csv(pathway_file)
            print(f"✅ Loaded pathway data")
        except Exception as e:
            print(f"❌ Error loading pathway data: {e}")
    
    def add_phenotype_info(self, phenotype_file: str):
        """Add phenotype descriptions and context."""
        try:
            self.phenotype_info = pd.read_csv(phenotype_file)
            print(f"✅ Loaded phenotype information")
        except Exception as e:
            print(f"❌ Error loading phenotype info: {e}")
    
    def enrich_qtl_data(self):
        """Merge QTL data with additional annotations."""
        enriched_data = self.qtl_data.copy()
        
        # Add gene annotations if available
        if self.gene_annotations is not None:
            enriched_data = enriched_data.merge(
                self.gene_annotations, 
                on='gene_symbol', 
                how='left',
                suffixes=('', '_annotation')
            )
            print("✅ Merged gene annotations")
        
        # Add pathway data if available
        if self.pathway_data is not None:
            enriched_data = enriched_data.merge(
                self.pathway_data,
                on='gene_symbol',
                how='left',
                suffixes=('', '_pathway')
            )
            print("✅ Merged pathway data")
        
        return enriched_data
    
    def create_enhanced_chunks(self, chunk_method='top_qtls', **kwargs):
        """Create chunks with enhanced information."""
        enriched_data = self.enrich_qtl_data()
        
        if chunk_method == 'top_qtls':
            return self._chunk_top_qtls_enhanced(enriched_data, **kwargs)
        elif chunk_method == 'by_pathway':
            return self._chunk_by_pathway(enriched_data, **kwargs)
        elif chunk_method == 'by_chromosome_enhanced':
            return self._chunk_by_chromosome_enhanced(enriched_data, **kwargs)
        else:
            raise ValueError(f"Unknown chunk method: {chunk_method}")
    
    def _chunk_top_qtls_enhanced(self, data, top_n: int = 200, chunk_size: int = 25):
        """Enhanced version of top QTL chunking with more context."""
        chunks = []
        top_qtls = data.nlargest(top_n, 'qtl_lod')
        
        for i in range(0, len(top_qtls), chunk_size):
            chunk_data = top_qtls.iloc[i:i+chunk_size]
            rank_start = i + 1
            rank_end = min(i + chunk_size, top_n)
            
            # Create enhanced content
            content_parts = [
                f"=== TOP QTLs RANKED {rank_start}-{rank_end} ===",
                f"LOD Score Range: {chunk_data['qtl_lod'].min():.2f} - {chunk_data['qtl_lod'].max():.2f}",
                f"Number of QTLs: {len(chunk_data)}",
                ""
            ]
            
            # Add summary statistics
            cis_count = chunk_data['cis'].sum() if 'cis' in chunk_data.columns else 0
            trans_count = len(chunk_data) - cis_count
            content_parts.extend([
                f"QTL Types: {cis_count} cis-acting, {trans_count} trans-acting",
                f"Chromosomes involved: {', '.join(map(str, sorted(chunk_data['qtl_chr'].unique())))}",
                ""
            ])
            
            # Add individual QTL details with enhanced information
            content_parts.append("DETAILED QTL INFORMATION:")
            for idx, row in chunk_data.iterrows():
                qtl_text = self._format_enhanced_qtl_text(row)
                content_parts.append(qtl_text)
                content_parts.append("")
            
            chunk = {
                'id': f"enhanced_top_qtls_{rank_start}_{rank_end}",
                'type': 'enhanced_top_qtls',
                'content': "\n".join(content_parts),
                'metadata': {
                    'rank_start': rank_start,
                    'rank_end': rank_end,
                    'qtl_count': len(chunk_data),
                    'lod_range': [chunk_data['qtl_lod'].min(), chunk_data['qtl_lod'].max()],
                    'genes': chunk_data['gene_symbol'].tolist(),
                    'chromosomes': chunk_data['qtl_chr'].unique().tolist(),
                    'cis_count': cis_count,
                    'trans_count': trans_count,
                    'avg_lod': chunk_data['qtl_lod'].mean(),
                    'pathways': self._extract_pathways(chunk_data) if self.pathway_data is not None else []
                },
                'raw_data': chunk_data.to_dict('records')
            }
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_pathway(self, data, min_genes_per_pathway: int = 5):
        """Create chunks grouped by biological pathways."""
        if self.pathway_data is None:
            raise ValueError("Pathway data not loaded. Use add_pathway_data() first.")
        
        chunks = []
        
        # Group by pathway
        pathway_groups = data.groupby('pathway_name') if 'pathway_name' in data.columns else None
        
        if pathway_groups is None:
            print("No pathway_name column found, skipping pathway chunking")
            return []
        
        for pathway_name, group in pathway_groups:
            if len(group) < min_genes_per_pathway:
                continue
            
            # Sort by LOD score within pathway
            group_sorted = group.sort_values('qtl_lod', ascending=False)
            
            content_parts = [
                f"=== PATHWAY: {pathway_name} ===",
                f"Number of QTLs: {len(group)}",
                f"LOD Score Range: {group['qtl_lod'].min():.2f} - {group['qtl_lod'].max():.2f}",
                f"Average LOD Score: {group['qtl_lod'].mean():.2f}",
                ""
            ]
            
            # Add pathway description if available
            if 'pathway_description' in group.columns:
                pathway_desc = group['pathway_description'].iloc[0]
                if pd.notna(pathway_desc):
                    content_parts.extend([
                        f"Pathway Description: {pathway_desc}",
                        ""
                    ])
            
            # Add gene details
            content_parts.append("GENES IN THIS PATHWAY:")
            for idx, row in group_sorted.iterrows():
                qtl_text = self._format_enhanced_qtl_text(row)
                content_parts.append(qtl_text)
                content_parts.append("")
            
            chunk = {
                'id': f"pathway_{pathway_name.replace(' ', '_')}",
                'type': 'pathway_group',
                'content': "\n".join(content_parts),
                'metadata': {
                    'pathway_name': pathway_name,
                    'qtl_count': len(group),
                    'genes': group['gene_symbol'].tolist(),
                    'lod_range': [group['qtl_lod'].min(), group['qtl_lod'].max()],
                    'avg_lod': group['qtl_lod'].mean(),
                    'chromosomes': group['qtl_chr'].unique().tolist()
                },
                'raw_data': group.to_dict('records')
            }
            chunks.append(chunk)
        
        return chunks
    
    def _format_enhanced_qtl_text(self, row) -> str:
        """Enhanced QTL formatting with additional annotations."""
        text_parts = []
        
        # Basic QTL info
        gene_symbol = row.get('gene_symbol', 'Unknown')
        qtl_lod = row.get('qtl_lod', 'Unknown')
        qtl_pval = row.get('qtl_pval', 'Unknown')
        
        text_parts.append(f"🧬 Gene: {gene_symbol}")
        text_parts.append(f"📊 LOD Score: {qtl_lod}")
        text_parts.append(f"📈 P-value: {qtl_pval}")
        
        # Location info
        qtl_chr = row.get('qtl_chr', 'Unknown')
        qtl_pos = row.get('qtl_pos', 'Unknown')
        text_parts.append(f"📍 Location: Chr {qtl_chr}, {qtl_pos} Mb")
        
        # Cis/trans
        cis = row.get('cis', None)
        if cis is not None:
            text_parts.append(f"🔗 Type: {'Cis-acting' if cis else 'Trans-acting'}")
        
        # Gene annotations (if available)
        gene_desc = row.get('gene_description', None)
        if gene_desc and pd.notna(gene_desc):
            text_parts.append(f"ℹ️ Description: {gene_desc}")
        
        gene_function = row.get('gene_function', None)
        if gene_function and pd.notna(gene_function):
            text_parts.append(f"⚙️ Function: {gene_function}")
        
        # Pathway info (if available)
        pathway = row.get('pathway_name', None)
        if pathway and pd.notna(pathway):
            text_parts.append(f"🛤️ Pathway: {pathway}")
        
        # Human ortholog (if available)
        human_ortholog = row.get('human_ortholog', None)
        if human_ortholog and pd.notna(human_ortholog):
            text_parts.append(f"👥 Human Ortholog: {human_ortholog}")
        
        return " | ".join(text_parts)
    
    def _extract_pathways(self, chunk_data):
        """Extract unique pathways from chunk data."""
        if 'pathway_name' not in chunk_data.columns:
            return []
        pathways = chunk_data['pathway_name'].dropna().unique().tolist()
        return pathways[:5]  # Limit to top 5 pathways
    
    def add_literature_context(self):
        """Add general genetic literature context to chunks."""
        context_info = {
            'qtl_background': """
            QTL (Quantitative Trait Loci) Background:
            - QTLs are genomic regions containing genes that correlate with variation in quantitative traits
            - LOD scores measure the strength of evidence for linkage (LOD > 3 typically considered significant)
            - Cis-acting QTLs affect nearby genes, trans-acting QTLs affect distant genes
            - Higher LOD scores indicate stronger genetic associations
            """,
            'liver_metabolism': """
            Liver Gene Expression Context:
            - The liver is central to metabolism, including glucose, lipid, and protein metabolism
            - Liver gene expression varies significantly between individuals and strains
            - Many liver genes are involved in drug metabolism, detoxification, and energy homeostasis
            - Genetic variation in liver gene expression can influence disease susceptibility
            """,
            'mouse_genetics': """
            Diversity Outbred (DO) Mouse Population:
            - DO mice capture genetic diversity from 8 founder strains
            - Provides high-resolution mapping of genetic loci
            - Mimics human genetic diversity better than traditional inbred strains
            - Enables detection of both common and rare genetic variants
            """
        }
        return context_info
    
    def save_enhanced_chunks(self, chunks: List[Dict], output_file: str):
        """Save enhanced chunks with metadata."""
        enhanced_output = {
            'chunks': chunks,
            'metadata': {
                'total_chunks': len(chunks),
                'creation_method': 'enhanced_chunking',
                'data_sources': {
                    'primary_qtl_data': self.primary_csv,
                    'gene_annotations': self.gene_annotations is not None,
                    'pathway_data': self.pathway_data is not None,
                    'phenotype_info': self.phenotype_info is not None
                },
                'context_info': self.add_literature_context()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(enhanced_output, f, indent=2, default=str)
        print(f"💾 Saved {len(chunks)} enhanced chunks to {output_file}")

# Example usage
if __name__ == "__main__":
    print("🚀 Enhanced QTL Chunking System")
    print("=" * 50)
    
    # Initialize with primary data
    enhancer = EnhancedQTLChunker("qtl_chunks_top_qtls_only.json")  # Use existing chunks as input
    
    # You can add additional data sources:
    # enhancer.add_gene_annotations("gene_annotations.csv")
    # enhancer.add_pathway_data("pathway_data.csv")
    # enhancer.add_phenotype_info("phenotype_descriptions.csv")
    
    # Create enhanced chunks
    enhanced_chunks = enhancer.create_enhanced_chunks(
        chunk_method='top_qtls',
        top_n=200,
        chunk_size=20  # Smaller chunks for more focused content
    )
    
    # Save enhanced chunks
    enhancer.save_enhanced_chunks(enhanced_chunks, "enhanced_qtl_chunks.json")
    
    print(f"\n✅ Created {len(enhanced_chunks)} enhanced chunks")
    print("📝 Each chunk now includes:")
    print("   - Detailed QTL information")
    print("   - Summary statistics")
    print("   - Biological context")
    print("   - Enhanced metadata")
    
    # Show example of enhanced content
    if enhanced_chunks:
        print(f"\n📋 Example enhanced chunk content (first 300 chars):")
        print(enhanced_chunks[0]['content'][:300] + "...") 