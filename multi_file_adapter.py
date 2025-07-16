#!/usr/bin/env python3
"""
Multi-File Adapter for Hybrid QTL System

This adapter takes your working single-file hybrid_qtl_system.py and 
makes it work with all 40 QTL files by combining them into a unified dataset.

Much simpler approach than rewriting the entire system.
"""

import pandas as pd
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from hybrid_qtl_system import HybridQTLSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiFileAdapter:
    """Adapter to make single-file hybrid system work with multiple QTL files."""
    
    def __init__(self, data_directory: str = "/data/dev/miniViewer_3.0/"):
        self.data_directory = data_directory
        self.unified_csv_path = "./unified_qtl_data.csv"
        self.file_metadata = {}
        
    def discover_and_categorize_files(self) -> Dict[str, List[str]]:
        """Discover all QTL files and categorize by trait type."""
        files_by_trait = {
            'clinical_traits': [],
            'liver_genes': [],
            'liver_isoforms': [],
            'liver_lipids': [],
            'liver_splice_juncs': [],
            'plasma_metabolites': []
        }
        
        data_path = Path(self.data_directory)
        
        for file_path in data_path.glob("DO1200_*_peaks.csv"):
            filename = file_path.name
            
            # Categorize by trait type
            if 'clinical_traits' in filename:
                trait_type = 'clinical_traits'
            elif 'liver_genes' in filename:
                trait_type = 'liver_genes'
            elif 'liver_isoforms' in filename:
                trait_type = 'liver_isoforms'
            elif 'liver_lipids' in filename:
                trait_type = 'liver_lipids'
            elif 'liver_splice_juncs' in filename:
                trait_type = 'liver_splice_juncs'
            elif 'plasma_metabolites' in filename:
                trait_type = 'plasma_metabolites'
            else:
                continue
            
            files_by_trait[trait_type].append(str(file_path))
        
        # Log discovered files
        total_files = sum(len(files) for files in files_by_trait.values())
        logger.info(f"Discovered {total_files} QTL files:")
        for trait_type, files in files_by_trait.items():
            if files:
                logger.info(f"  {trait_type}: {len(files)} files")
        
        return files_by_trait
    
    def combine_all_files(self) -> str:
        """Combine all QTL files into a single unified CSV that works with hybrid_qtl_system.py"""
        
        files_by_trait = self.discover_and_categorize_files()
        all_dataframes = []
        total_records = 0
        
        logger.info("Loading and combining all QTL files...")
        
        for trait_type, file_paths in files_by_trait.items():
            if not file_paths:
                continue
                
            logger.info(f"\nProcessing {trait_type}...")
            
            for file_path in file_paths:
                try:
                    # Load the file
                    df = pd.read_csv(file_path)
                    
                    # Add metadata columns
                    df['trait_type'] = trait_type
                    df['source_file'] = os.path.basename(file_path)
                    
                    # Extract additional metadata from filename
                    filename = os.path.basename(file_path)
                    df['analysis_type'] = self._extract_analysis_type(filename)
                    df['cohort'] = self._extract_cohort(filename)
                    
                    # Store file metadata
                    self.file_metadata[file_path] = {
                        'trait_type': trait_type,
                        'filename': filename,
                        'record_count': len(df),
                        'analysis_type': df['analysis_type'].iloc[0],
                        'cohort': df['cohort'].iloc[0]
                    }
                    
                    all_dataframes.append(df)
                    total_records += len(df)
                    
                    logger.info(f"  ✅ {filename}: {len(df):,} records")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error loading {file_path}: {e}")
        
        if not all_dataframes:
            raise ValueError("No QTL files could be loaded!")
        
        # Combine all dataframes
        logger.info(f"\nCombining {len(all_dataframes)} dataframes...")
        unified_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save the unified dataset
        unified_df.to_csv(self.unified_csv_path, index=False)
        
        logger.info(f"✅ Created unified dataset: {self.unified_csv_path}")
        logger.info(f"📊 Total records: {len(unified_df):,}")
        logger.info(f"🧬 Unique genes: {unified_df['gene_symbol'].nunique():,}")
        logger.info(f"🏷️ Trait types: {', '.join(unified_df['trait_type'].unique())}")
        
        return self.unified_csv_path
    
    def _extract_analysis_type(self, filename: str) -> str:
        """Extract analysis type from filename."""
        if 'additive' in filename:
            return 'additive'
        elif 'diet_interactive' in filename:
            return 'diet_interactive'
        elif 'sex_interactive' in filename:
            return 'sex_interactive'
        elif 'qtlxdiet' in filename:
            return 'qtlxdiet'
        elif 'qtlxsex' in filename:
            return 'qtlxsex'
        elif 'qtlxsexbydiet' in filename:
            return 'qtlxsexbydiet'
        return 'unknown'
    
    def _extract_cohort(self, filename: str) -> str:
        """Extract cohort from filename."""
        if '_all_mice_' in filename:
            return 'all_mice'
        elif '_male_mice_' in filename:
            return 'male_mice'
        elif '_female_mice_' in filename:
            return 'female_mice'
        elif '_HC_mice_' in filename:
            return 'HC_mice'
        elif '_HF_mice_' in filename:
            return 'HF_mice'
        return 'unknown'
    
    def create_enhanced_hybrid_system(self, google_api_key: str = None) -> 'EnhancedHybridQTLSystem':
        """Create an enhanced hybrid system that works with the unified multi-file dataset."""
        
        # Ensure unified dataset exists
        if not os.path.exists(self.unified_csv_path):
            logger.info("Unified dataset not found. Creating it...")
            self.combine_all_files()
        
        # Create enhanced system
        return EnhancedHybridQTLSystem(
            csv_file_path=self.unified_csv_path,
            google_api_key=google_api_key,
            file_metadata=self.file_metadata
        )

class EnhancedHybridQTLSystem(HybridQTLSystem):
    """Enhanced version of HybridQTLSystem that works with multi-file data."""
    
    def __init__(self, csv_file_path: str, chroma_db_path: str = "./enhanced_chroma_db", 
                 google_api_key: str = None, file_metadata: Dict = None):
        # Initialize the parent system
        super().__init__(csv_file_path, chroma_db_path)
        
        # Store file metadata
        self.file_metadata = file_metadata or {}
        
        # Setup Google API if provided
        if google_api_key:
            self.setup_embedding_models(google_api_key)
        else:
            self.setup_embedding_models()
    
    def generate_enhanced_summary_documents(self) -> List[Dict]:
        """Generate enhanced summaries that include multi-file context."""
        
        # First generate the standard summaries
        logger.info("Generating standard summaries...")
        standard_summaries = self.generate_summary_documents()
        
        # Add trait-type specific summaries
        logger.info("Generating trait-type summaries...")
        trait_summaries = self._generate_trait_summaries()
        
        # Add file-specific summaries
        logger.info("Generating file-specific summaries...")
        file_summaries = self._generate_file_summaries()
        
        # Add comparative summaries
        logger.info("Generating comparative summaries...")
        comparative_summaries = self._generate_comparative_summaries()
        
        # Combine all summaries
        all_summaries = standard_summaries + trait_summaries + file_summaries + comparative_summaries
        
        self.summary_docs = all_summaries
        logger.info(f"✅ Generated {len(all_summaries)} enhanced summary documents")
        
        return all_summaries
    
    def _generate_trait_summaries(self) -> List[Dict]:
        """Generate summaries for each trait type."""
        trait_summaries = []
        
        trait_groups = self.raw_data.groupby('trait_type')
        
        for trait_type, trait_data in trait_groups:
            # Calculate statistics
            qtl_count = len(trait_data)
            unique_genes = trait_data['gene_symbol'].nunique()
            mean_lod = trait_data['qtl_lod'].mean()
            max_lod = trait_data['qtl_lod'].max()
            top_genes = trait_data.nlargest(5, 'qtl_lod')['gene_symbol'].tolist()
            
            # Count by analysis type
            analysis_counts = trait_data['analysis_type'].value_counts().to_dict()
            cohort_counts = trait_data['cohort'].value_counts().to_dict()
            
            summary_text = f"""
            {trait_type.replace('_', ' ').title()} QTL Summary
            
            Overview:
            - Total QTL peaks: {qtl_count:,}
            - Unique genes: {unique_genes:,}
            - Average LOD score: {mean_lod:.2f}
            - Maximum LOD score: {max_lod:.2f}
            - Top genes: {', '.join(map(str, top_genes[:3]))}
            
            Analysis Types:
            {chr(10).join(f"- {analysis}: {count:,} QTLs" for analysis, count in analysis_counts.items())}
            
            Cohorts:
            {chr(10).join(f"- {cohort}: {count:,} QTLs" for cohort, count in cohort_counts.items())}
            
            Biological Context:
            {self._get_trait_description(trait_type)}
            """
            
            trait_summaries.append({
                'id': f'trait_summary_{trait_type}',
                'type': 'trait_summary',
                'content': summary_text.strip(),
                'metadata': {
                    'trait_type': trait_type,
                    'qtl_count': qtl_count,
                    'unique_genes': unique_genes,
                    'mean_lod': mean_lod,
                    'max_lod': max_lod,
                    'top_genes': ', '.join(map(str, top_genes)),
                    'analysis_types': ', '.join(analysis_counts.keys()),
                    'cohorts': ', '.join(cohort_counts.keys())
                }
            })
        
        return trait_summaries
    
    def _generate_file_summaries(self) -> List[Dict]:
        """Generate summaries for each source file."""
        file_summaries = []
        
        file_groups = self.raw_data.groupby('source_file')
        
        for filename, file_data in file_groups:
            # Get metadata
            metadata = self.file_metadata.get(filename, {})
            
            # Calculate statistics
            qtl_count = len(file_data)
            unique_genes = file_data['gene_symbol'].nunique()
            mean_lod = file_data['qtl_lod'].mean()
            max_lod = file_data['qtl_lod'].max()
            trait_type = file_data['trait_type'].iloc[0]
            analysis_type = file_data['analysis_type'].iloc[0]
            cohort = file_data['cohort'].iloc[0]
            
            summary_text = f"""
            File Analysis: {filename}
            
            Dataset Details:
            - Trait type: {trait_type.replace('_', ' ').title()}
            - Analysis: {analysis_type.replace('_', ' ').title()}
            - Cohort: {cohort.replace('_', ' ').title()}
            - QTL count: {qtl_count:,}
            - Unique genes: {unique_genes:,}
            - LOD range: {file_data['qtl_lod'].min():.2f} - {max_lod:.2f}
            - Average LOD: {mean_lod:.2f}
            
            Study Context:
            {self._get_analysis_description(analysis_type, cohort)}
            """
            
            file_summaries.append({
                'id': f'file_summary_{filename}',
                'type': 'file_summary',
                'content': summary_text.strip(),
                'metadata': {
                    'filename': filename,
                    'trait_type': trait_type,
                    'analysis_type': analysis_type,
                    'cohort': cohort,
                    'qtl_count': qtl_count,
                    'unique_genes': unique_genes,
                    'mean_lod': mean_lod,
                    'max_lod': max_lod
                }
            })
        
        return file_summaries
    
    def _generate_comparative_summaries(self) -> List[Dict]:
        """Generate comparative summaries across traits."""
        comparative_summaries = []
        
        # Trait type comparison
        trait_stats = self.raw_data.groupby('trait_type').agg({
            'gene_symbol': 'nunique',
            'qtl_lod': ['count', 'mean', 'max']
        }).round(2)
        
        trait_comparison_text = """
        Multi-Trait QTL Comparison
        
        This dataset contains quantitative trait loci (QTLs) across multiple biological trait types:
        
        """
        
        for trait_type in trait_stats.index:
            gene_count = trait_stats.loc[trait_type, ('gene_symbol', 'nunique')]
            qtl_count = trait_stats.loc[trait_type, ('qtl_lod', 'count')]
            mean_lod = trait_stats.loc[trait_type, ('qtl_lod', 'mean')]
            max_lod = trait_stats.loc[trait_type, ('qtl_lod', 'max')]
            
            trait_comparison_text += f"""
        {trait_type.replace('_', ' ').title()}:
        - {qtl_count:,} QTLs affecting {gene_count:,} genes
        - Average LOD: {mean_lod:.2f}, Maximum LOD: {max_lod:.2f}
        """
        
        trait_comparison_text += """
        
        Cross-Trait Analysis:
        This comprehensive dataset enables comparative genetics studies across different
        biological systems and trait types, providing insights into shared genetic
        architecture and trait-specific regulatory mechanisms.
        """
        
        comparative_summaries.append({
            'id': 'multi_trait_comparison',
            'type': 'comparative_summary',
            'content': trait_comparison_text.strip(),
            'metadata': {
                'comparison_type': 'multi_trait',
                'trait_count': len(trait_stats),
                'total_qtls': int(trait_stats[('qtl_lod', 'count')].sum()),
                'total_genes': int(self.raw_data['gene_symbol'].nunique())
            }
        })
        
        return comparative_summaries
    
    def _get_trait_description(self, trait_type: str) -> str:
        """Get biological description for trait type."""
        descriptions = {
            'clinical_traits': 'Clinical traits represent measurable health and disease phenotypes in the DO population.',
            'liver_genes': 'Liver gene expression QTLs reveal genetic factors controlling hepatic gene regulation.',
            'liver_isoforms': 'Liver isoform QTLs identify genetic variants affecting alternative splicing and transcript structure.',
            'liver_lipids': 'Liver lipid QTLs map genetic factors influencing hepatic lipid metabolism and composition.',
            'liver_splice_juncs': 'Liver splice junction QTLs reveal genetic control of RNA splicing processes.',
            'plasma_metabolites': 'Plasma metabolite QTLs identify genetic factors affecting circulating metabolite levels.'
        }
        return descriptions.get(trait_type, f'QTLs for {trait_type}')
    
    def _get_analysis_description(self, analysis_type: str, cohort: str) -> str:
        """Get description for analysis type and cohort."""
        analysis_desc = {
            'additive': 'Additive genetic effects analysis',
            'diet_interactive': 'Diet interaction analysis',
            'sex_interactive': 'Sex interaction analysis',
            'qtlxdiet': 'QTL × Diet interaction mapping',
            'qtlxsex': 'QTL × Sex interaction mapping',
            'qtlxsexbydiet': 'QTL × Sex × Diet interaction analysis'
        }.get(analysis_type, analysis_type)
        
        cohort_desc = {
            'all_mice': 'all mice in the population',
            'male_mice': 'male mice only',
            'female_mice': 'female mice only',
            'HC_mice': 'high-carbohydrate diet mice',
            'HF_mice': 'high-fat diet mice'
        }.get(cohort, cohort)
        
        return f"{analysis_desc} performed in {cohort_desc}."
    
    def setup_enhanced_vector_store(self):
        """Setup vector store with enhanced summaries."""
        # Generate enhanced summaries
        enhanced_summaries = self.generate_enhanced_summary_documents()
        
        # Setup vector store using parent method
        self.setup_vector_store(use_google_embeddings=False)
        
        logger.info(f"✅ Enhanced vector store ready with {len(enhanced_summaries)} documents")
    
    def trait_filtered_search(self, query: str, trait_type: str = None, n_results: int = 5) -> List[Dict]:
        """Search with optional trait type filtering."""
        if trait_type:
            # Add trait filter to query
            enhanced_query = f"{query} {trait_type.replace('_', ' ')}"
            return self.semantic_search(enhanced_query, n_results)
        else:
            return self.semantic_search(query, n_results)
    
    def get_trait_statistics(self) -> Dict:
        """Get comprehensive statistics across all traits."""
        stats = {}
        
        # Overall statistics
        stats['total_qtls'] = len(self.raw_data)
        stats['total_genes'] = self.raw_data['gene_symbol'].nunique()
        stats['trait_types'] = list(self.raw_data['trait_type'].unique())
        
        # Per-trait statistics
        stats['by_trait'] = {}
        for trait in stats['trait_types']:
            trait_data = self.raw_data[self.raw_data['trait_type'] == trait]
            stats['by_trait'][trait] = {
                'qtl_count': len(trait_data),
                'gene_count': trait_data['gene_symbol'].nunique(),
                'max_lod': trait_data['qtl_lod'].max(),
                'mean_lod': trait_data['qtl_lod'].mean()
            }
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('config.env')
    
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    print("🚀 Multi-File Adapter for Hybrid QTL System")
    print("=" * 60)
    
    # Create adapter and combine files
    adapter = MultiFileAdapter()
    
    # Create enhanced hybrid system
    print("Creating enhanced hybrid system...")
    enhanced_system = adapter.create_enhanced_hybrid_system(google_api_key)
    
    # Setup enhanced vector store
    print("Setting up enhanced vector store...")
    enhanced_system.setup_enhanced_vector_store()
    
    # Get statistics
    stats = enhanced_system.get_trait_statistics()
    
    print(f"\n✅ Multi-File Hybrid System Ready!")
    print(f"📊 Total QTLs: {stats['total_qtls']:,}")
    print(f"🧬 Total Genes: {stats['total_genes']:,}")
    print(f"🏷️ Trait Types: {len(stats['trait_types'])}")
    
    print("\nTrait breakdown:")
    for trait, trait_stats in stats['by_trait'].items():
        print(f"  {trait}: {trait_stats['qtl_count']:,} QTLs, {trait_stats['gene_count']:,} genes")
    
    print(f"\n💡 System ready for queries!")
    print(f"📚 Use .semantic_search() for concept queries")
    print(f"📊 Use .analytical_query() for data analysis")
    print(f"🎯 Use .trait_filtered_search() for trait-specific queries")