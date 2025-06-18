import json
import os
from typing import List, Dict, Any
from knowledge_base import BiologicalKnowledgeBase

class EnhancedQTLRAG:
    """Enhanced RAG system with biological knowledge integration."""
    
    def __init__(self):
        self.qtl_chunks = []
        self.knowledge_chunks = []
        self.combined_chunks = []
        self.kb = BiologicalKnowledgeBase()
    
    def load_qtl_data(self, qtl_chunks_file: str = "qtl_chunks_top_qtls_only.json"):
        """Load existing QTL chunks."""
        try:
            with open(qtl_chunks_file, 'r') as f:
                self.qtl_chunks = json.load(f)
            print(f"✅ Loaded {len(self.qtl_chunks)} QTL chunks")
        except Exception as e:
            print(f"❌ Error loading QTL chunks: {e}")
    
    def create_enhanced_system(self):
        """Create an enhanced RAG system with all components."""
        
        # 1. Create biological knowledge chunks
        print("\n🧠 Creating biological knowledge base...")
        self.knowledge_chunks = self.kb.create_knowledge_chunks()
        print(f"✅ Created {len(self.knowledge_chunks)} knowledge chunks")
        
        # 2. Enhance existing QTL chunks with biological context
        print("\n🔬 Enhancing QTL chunks with biological knowledge...")
        enhanced_qtl_chunks = []
        for chunk in self.qtl_chunks:
            enhanced_chunk = self.kb.enhance_qtl_chunk_with_knowledge(chunk)
            enhanced_qtl_chunks.append(enhanced_chunk)
        
        # 3. Create summary chunks for better retrieval
        print("\n📊 Creating summary chunks...")
        summary_chunks = self._create_summary_chunks()
        
        # 4. Combine all chunks
        self.combined_chunks = (
            enhanced_qtl_chunks + 
            self.knowledge_chunks + 
            summary_chunks
        )
        
        print(f"✅ Enhanced RAG system created!")
        print(f"   - {len(enhanced_qtl_chunks)} enhanced QTL chunks")
        print(f"   - {len(self.knowledge_chunks)} knowledge chunks") 
        print(f"   - {len(summary_chunks)} summary chunks")
        print(f"   - {len(self.combined_chunks)} total chunks")
    
    def _create_summary_chunks(self) -> List[Dict]:
        """Create high-level summary chunks for better retrieval."""
        summary_chunks = []
        
        if not self.qtl_chunks:
            return summary_chunks
        
        # Extract all genes and their LOD scores
        all_genes = []
        all_lods = []
        cis_genes = []
        trans_genes = []
        
        for chunk in self.qtl_chunks:
            if 'raw_data' in chunk:
                for record in chunk['raw_data']:
                    gene = record.get('gene_symbol', 'Unknown')
                    lod = record.get('qtl_lod', 0)
                    cis = record.get('cis', False)
                    
                    all_genes.append(gene)
                    all_lods.append(lod)
                    
                    if cis:
                        cis_genes.append(gene)
                    else:
                        trans_genes.append(gene)
        
        # Create overall summary chunk
        max_lod = max(all_lods) if all_lods else 0
        min_lod = min(all_lods) if all_lods else 0
        avg_lod = sum(all_lods) / len(all_lods) if all_lods else 0
        
        overall_summary = {
            'id': 'summary_overall_dataset',
            'type': 'dataset_summary',
            'content': f"""
            OVERALL DATASET SUMMARY:
            
            📊 STATISTICAL OVERVIEW:
            - Total QTLs analyzed: {len(all_genes)}
            - LOD score range: {min_lod:.2f} to {max_lod:.2f}
            - Average LOD score: {avg_lod:.2f}
            - Cis-acting QTLs: {len(cis_genes)} ({len(cis_genes)/len(all_genes)*100:.1f}%)
            - Trans-acting QTLs: {len(trans_genes)} ({len(trans_genes)/len(all_genes)*100:.1f}%)
            
            🧬 TOP FINDINGS:
            {self.kb.interpret_lod_score(max_lod)}
            
            🔬 BIOLOGICAL SIGNIFICANCE:
            This dataset represents liver gene expression QTLs from Diversity Outbred mice.
            High LOD scores (>100) indicate extremely strong genetic associations.
            Cis-acting QTLs suggest local genetic regulation.
            Trans-acting QTLs indicate distant regulatory effects.
            
            💡 RESEARCH IMPLICATIONS:
            - Strong genetic control of liver gene expression
            - Potential therapeutic targets for metabolic diseases
            - Insights into genetic architecture of complex traits
            - Relevance to human disease genetics
            """,
            'metadata': {
                'type': 'overall_summary',
                'total_qtls': len(all_genes),
                'lod_range': [min_lod, max_lod],
                'avg_lod': avg_lod,
                'cis_count': len(cis_genes),
                'trans_count': len(trans_genes),
                'topics': ['overview', 'statistics', 'significance']
            }
        }
        summary_chunks.append(overall_summary)
        
        # Create chromosome summary
        chr_summary = self._create_chromosome_summary()
        if chr_summary:
            summary_chunks.append(chr_summary)
        
        # Create methodology chunk
        methodology_chunk = {
            'id': 'summary_methodology',
            'type': 'methodology_summary',
            'content': """
            METHODOLOGY AND INTERPRETATION GUIDE:
            
            🔬 EXPERIMENTAL DESIGN:
            - Diversity Outbred (DO) mouse population
            - Liver gene expression measurements
            - QTL mapping using genetic markers
            - Statistical analysis with LOD score calculation
            
            📈 STATISTICAL METHODS:
            - QTL mapping: Identifies genomic regions affecting gene expression
            - LOD scores: Logarithm of odds, measures evidence for linkage
            - P-values: Statistical significance testing
            - Q-values: False discovery rate correction for multiple testing
            
            🎯 INTERPRETATION GUIDELINES:
            - LOD > 3: Significant evidence for QTL (standard threshold)
            - LOD > 10: Strong evidence for QTL
            - LOD > 50: Very strong evidence for QTL
            - LOD > 100: Extremely strong evidence for QTL
            
            🔗 CIS vs TRANS:
            - Cis-QTLs: QTL location is near the gene it affects (< 10 Mb)
            - Trans-QTLs: QTL location is distant from the affected gene
            - Cis-QTLs suggest local genetic variants affect gene expression
            - Trans-QTLs suggest distant regulatory elements or networks
            
            🧬 BIOLOGICAL RELEVANCE:
            - Liver is central to metabolism and detoxification
            - Genetic variants affect drug response and disease risk
            - QTLs identify candidate genes for therapeutic targeting
            - Results applicable to human genetics and medicine
            """,
            'metadata': {
                'type': 'methodology',
                'topics': ['methods', 'interpretation', 'statistics', 'biology']
            }
        }
        summary_chunks.append(methodology_chunk)
        
        return summary_chunks
    
    def _create_chromosome_summary(self) -> Dict:
        """Create a summary of findings by chromosome."""
        chr_counts = {}
        chr_max_lods = {}
        
        for chunk in self.qtl_chunks:
            if 'raw_data' in chunk:
                for record in chunk['raw_data']:
                    chr_num = str(record.get('qtl_chr', 'Unknown'))
                    lod = record.get('qtl_lod', 0)
                    
                    if chr_num not in chr_counts:
                        chr_counts[chr_num] = 0
                        chr_max_lods[chr_num] = 0
                    
                    chr_counts[chr_num] += 1
                    chr_max_lods[chr_num] = max(chr_max_lods[chr_num], lod)
        
        if not chr_counts:
            return None
        
        # Sort chromosomes by QTL count
        sorted_chrs = sorted(chr_counts.items(), key=lambda x: x[1], reverse=True)
        
        content_parts = [
            "CHROMOSOME-WISE QTL DISTRIBUTION:",
            "",
            "📍 QTLs PER CHROMOSOME:"
        ]
        
        for chr_num, count in sorted_chrs[:10]:  # Top 10 chromosomes
            max_lod = chr_max_lods[chr_num]
            chr_info = self.kb.get_chromosome_context(chr_num)
            content_parts.append(f"- {chr_info}")
            content_parts.append(f"  QTLs: {count}, Max LOD: {max_lod:.1f}")
        
        content_parts.extend([
            "",
            "🧬 BIOLOGICAL INSIGHTS:",
            "- Chromosome distribution reflects genetic architecture",
            "- High QTL density may indicate regulatory hotspots",
            "- Chromosome-specific features affect gene regulation",
            "- Physical chromosome size influences mapping resolution"
        ])
        
        return {
            'id': 'summary_chromosome_distribution',
            'type': 'chromosome_summary',
            'content': "\n".join(content_parts),
            'metadata': {
                'type': 'chromosome_summary',
                'chromosome_counts': chr_counts,
                'chromosome_max_lods': chr_max_lods,
                'topics': ['chromosomes', 'distribution', 'genomics']
            }
        }
    
    def save_enhanced_chunks(self, output_file: str = "enhanced_rag_chunks.json"):
        """Save the enhanced chunk system."""
        output_data = {
            'enhanced_chunks': self.combined_chunks,
            'metadata': {
                'total_chunks': len(self.combined_chunks),
                'qtl_chunks': len([c for c in self.combined_chunks if c['type'].startswith('enhanced_top_qtls')]),
                'knowledge_chunks': len([c for c in self.combined_chunks if c['type'] == 'knowledge_base']),
                'summary_chunks': len([c for c in self.combined_chunks if 'summary' in c['type']]),
                'enhancement_features': [
                    'Biological knowledge integration',
                    'LOD score interpretation',
                    'Chromosome context',
                    'Liver biology relevance',
                    'Statistical methodology',
                    'Research implications'
                ]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"💾 Saved enhanced RAG system to {output_file}")
    
    def create_vectordb_chunks(self, output_file: str = "enhanced_vectordb_chunks.json"):
        """Create chunks specifically formatted for vector database."""
        vectordb_chunks = []
        
        for chunk in self.combined_chunks:
            vectordb_chunk = {
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'id': chunk['id'],
                'type': chunk['type']
            }
            vectordb_chunks.append(vectordb_chunk)
        
        # Save in format expected by vectordb.py
        with open(output_file, 'w') as f:
            json.dump(vectordb_chunks, f, indent=2, default=str)
        
        print(f"🗃️ Saved vector database chunks to {output_file}")
        return vectordb_chunks

# Example usage
if __name__ == "__main__":
    print("🚀 Enhanced QTL RAG System")
    print("=" * 60)
    
    # Initialize and create enhanced system
    rag_system = EnhancedQTLRAG()
    
    # Load existing QTL data
    rag_system.load_qtl_data("qtl_chunks_top_qtls_only.json")
    
    # Create enhanced system
    rag_system.create_enhanced_system()
    
    # Save enhanced chunks
    rag_system.save_enhanced_chunks("enhanced_rag_chunks.json")
    
    # Create vector database format
    rag_system.create_vectordb_chunks("enhanced_vectordb_chunks.json")
    
    print(f"\n🎉 Enhanced RAG system complete!")
    print(f"📈 Improvements include:")
    print(f"   ✅ Biological knowledge integration")
    print(f"   ✅ LOD score interpretation")
    print(f"   ✅ Chromosome-specific context")
    print(f"   ✅ Liver biology relevance")
    print(f"   ✅ Dataset summaries")
    print(f"   ✅ Methodology explanations")
    print(f"   ✅ Research implications")
    
    print(f"\n🔧 Next steps:")
    print(f"   1. Update vectordb.py to use 'enhanced_vectordb_chunks.json'")
    print(f"   2. Re-run vector database creation")
    print(f"   3. Test improved question answering")
    
    # Show example of enhanced content
    if rag_system.combined_chunks:
        print(f"\n📋 Example enhanced content:")
        example_chunk = rag_system.combined_chunks[0]
        print(f"Chunk type: {example_chunk['type']}")
        print(f"Content preview: {example_chunk['content'][:200]}...") 