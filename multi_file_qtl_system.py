#!/usr/bin/env python3
"""
Multi-File Hybrid QTL Analysis System

Extends the single-file hybrid QTL system to handle all 40 QTL peaks files
across different trait types: clinical traits, liver genes, isoforms, lipids, 
splice junctions, and plasma metabolites.

Features:
- Automatic file discovery and categorization
- Unified hybrid architecture (vector store + SQL analytics)
- Multi-trait-type semantic search
- Cross-trait comparative analysis
- Comprehensive summary generation
"""

import pandas as pd
import numpy as np
import chromadb
import duckdb
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiFileQTLSystem:
    """Multi-file QTL analysis system with hybrid vector/SQL architecture."""
    
    def __init__(self, 
                 data_directory: str = "/data/dev/miniViewer_3.0/",
                 chroma_db_path: str = "./multi_file_chroma_db",
                 duckdb_path: str = "./multi_file_qtl.db",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.data_directory = data_directory
        self.chroma_db_path = chroma_db_path
        self.duckdb_path = duckdb_path
        self.model_name = model_name
        
        # Initialize components
        self.embedding_model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.duckdb_conn = duckdb.connect(duckdb_path)
        
        # Data storage
        self.trait_data = {}  # trait_type -> DataFrame
        self.file_info = {}   # track file metadata
        self.collection_name = "multi_qtl_summaries"
        
    def discover_files(self) -> Dict[str, List[str]]:
        """Discover and categorize QTL files by trait type."""
        files_by_trait = {}
        data_path = Path(self.data_directory)
        
        for file_path in data_path.glob("DO1200_*_peaks.csv"):
            filename = file_path.name
            
            # Extract trait type
            if filename.startswith('DO1200_clinical_traits'):
                trait_type = 'clinical_traits'
            elif filename.startswith('DO1200_liver_genes'):
                trait_type = 'liver_genes'
            elif filename.startswith('DO1200_liver_isoforms'):
                trait_type = 'liver_isoforms'
            elif filename.startswith('DO1200_liver_lipids'):
                trait_type = 'liver_lipids'
            elif filename.startswith('DO1200_liver_splice_juncs'):
                trait_type = 'liver_splice_juncs'
            elif filename.startswith('DO1200_plasma_metabolites'):
                trait_type = 'plasma_metabolites'
            else:
                trait_type = 'other'
            
            if trait_type not in files_by_trait:
                files_by_trait[trait_type] = []
            files_by_trait[trait_type].append(str(file_path))
        
        logger.info("Discovered files:")
        for trait, files in files_by_trait.items():
            logger.info(f"  {trait}: {len(files)} files")
        
        return files_by_trait
    
    def load_all_data(self):
        """Load all QTL data files."""
        files_by_trait = self.discover_files()
        
        for trait_type, file_paths in files_by_trait.items():
            logger.info(f"\nLoading {trait_type}...")
            
            trait_dfs = []
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path)
                    
                    # Add metadata columns
                    df['trait_type'] = trait_type
                    df['source_file'] = os.path.basename(file_path)
                    
                    # Parse analysis info from filename
                    filename = os.path.basename(file_path)
                    df['analysis_type'] = self._extract_analysis_type(filename)
                    df['cohort'] = self._extract_cohort(filename)
                    
                    trait_dfs.append(df)
                    logger.info(f"  {os.path.basename(file_path)}: {len(df):,} records")
                    
                except Exception as e:
                    logger.error(f"  Error loading {file_path}: {e}")
            
            if trait_dfs:
                self.trait_data[trait_type] = pd.concat(trait_dfs, ignore_index=True)
                logger.info(f"  Total {trait_type}: {len(self.trait_data[trait_type]):,} records")
        
        self._setup_sql_database()
    
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
    
    def _setup_sql_database(self):
        """Create unified SQL database."""
        logger.info("\nSetting up SQL database...")
        
        # Combine all data
        all_data = []
        for trait_type, df in self.trait_data.items():
            all_data.append(df)
        
        if all_data:
            unified_df = pd.concat(all_data, ignore_index=True)
            
            # Create table
            self.duckdb_conn.execute("DROP TABLE IF EXISTS qtl_data")
            self.duckdb_conn.execute("CREATE TABLE qtl_data AS SELECT * FROM unified_df")
            
            # Create indexes
            index_cols = ['gene_symbol', 'qtl_chr', 'qtl_lod', 'trait_type', 'analysis_type', 'cohort']
            for col in index_cols:
                try:
                    if col in unified_df.columns:
                        self.duckdb_conn.execute(f"CREATE INDEX idx_{col} ON qtl_data ({col})")
                except:
                    pass
            
            logger.info(f"  Created database with {len(unified_df):,} total records")
    
    def generate_summaries(self) -> List[Dict[str, Any]]:
        """Generate summary documents for vector store."""
        logger.info("\nGenerating summary documents...")
        
        summaries = []
        
        try:
            # 1. System overview
            logger.info("Generating system summaries...")
            summaries.extend(self._system_summaries())
            
            # 2. Trait type summaries
            logger.info("Generating trait summaries...")
            summaries.extend(self._trait_summaries())
            
            # 3. Gene summaries (for applicable traits)
            logger.info("Generating gene summaries...")
            summaries.extend(self._gene_summaries())
            
            # 4. Chromosome summaries
            logger.info("Generating chromosome summaries...")
            summaries.extend(self._chromosome_summaries())
            
            # 5. Significance summaries
            logger.info("Generating significance summaries...")
            summaries.extend(self._significance_summaries())
            
            # 6. Analysis type summaries
            logger.info("Generating analysis summaries...")
            summaries.extend(self._analysis_summaries())
            
            # 7. Cohort summaries
            logger.info("Generating cohort summaries...")
            summaries.extend(self._cohort_summaries())
            
            logger.info(f"Generated {len(summaries)} summary documents")
            return summaries
            
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _system_summaries(self) -> List[Dict[str, Any]]:
        """Generate system-level summaries."""
        total_records = sum(len(df) for df in self.trait_data.values())
        trait_types = list(self.trait_data.keys())
        
        content = f"""
Multi-File QTL Analysis System

This comprehensive database contains {total_records:,} QTL findings across 
{len(trait_types)} trait categories from Diversity Outbred mice.

Trait Categories:
{chr(10).join([f"- {tt.replace('_', ' ').title()}: {len(self.trait_data[tt]):,} QTLs" for tt in trait_types])}

The system includes cis and trans-acting QTLs with genomic positions, 
LOD scores, p-values, and confidence intervals across multiple analysis
types and cohorts.
        """.strip()
        
        return [{
            'id': 'system_overview',
            'type': 'system_overview', 
            'content': content,
            'metadata': {
                'total_records': total_records,
                'trait_types': ','.join(trait_types),
                'num_trait_types': len(trait_types)
            }
        }]
    
    def _trait_summaries(self) -> List[Dict[str, Any]]:
        """Generate trait-type summaries."""
        summaries = []
        
        for trait_type, df in self.trait_data.items():
            stats = self._calculate_trait_stats(df)
            
            content = f"""
{trait_type.replace('_', ' ').title()} QTL Analysis

This trait category contains {len(df):,} QTL findings affecting 
{stats['unique_genes']:,} genes across {len(stats['chromosomes'])} chromosomes.

Key Statistics:
- Total QTLs: {len(df):,}
- Unique genes: {stats['unique_genes']:,}
- Chromosomes: {', '.join(map(str, stats['chromosomes'][:10]))}
- Cis-acting: {stats['cis_count']:,} ({stats['cis_pct']:.1f}%)
- Trans-acting: {stats['trans_count']:,} ({stats['trans_pct']:.1f}%)

LOD Score Distribution:
- Maximum: {stats['max_lod']:.2f}
- Mean: {stats['mean_lod']:.2f}
- Median: {stats['median_lod']:.2f}

Top Genes: {', '.join(stats['top_genes'][:5])}
Analysis Types: {', '.join(stats['analysis_types'])}
Cohorts: {', '.join(stats['cohorts'])}
            """.strip()
            
            summaries.append({
                'id': f'trait_{trait_type}',
                'type': 'trait_summary',
                'content': content,
                'metadata': {
                    'trait_type': trait_type,
                    'total_qtls': len(df),
                    'unique_genes': stats['unique_genes'],
                    'max_lod': stats['max_lod'],
                    'chromosomes': ','.join(map(str, stats['chromosomes']))
                }
            })
        
        return summaries
    
    def _calculate_trait_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for a trait DataFrame."""
        stats = {}
        
        # Basic counts
        stats['unique_genes'] = df['gene_symbol'].nunique() if 'gene_symbol' in df.columns else 0
        
        # Handle chromosome sorting safely
        if 'qtl_chr' in df.columns:
            try:
                chromosomes = df['qtl_chr'].unique()
                # Sort chromosomes as numbers where possible
                try:
                    stats['chromosomes'] = sorted([x for x in chromosomes if pd.notna(x)], key=lambda x: int(x) if str(x).isdigit() else float('inf'))
                except:
                    stats['chromosomes'] = sorted([x for x in chromosomes if pd.notna(x)], key=str)
            except:
                stats['chromosomes'] = []
        else:
            stats['chromosomes'] = []
        
        # Cis/trans counts
        if 'cis' in df.columns:
            cis_count = (df['cis'] == 'TRUE').sum() if df['cis'].dtype == 'object' else df['cis'].sum()
            stats['cis_count'] = cis_count
            stats['trans_count'] = len(df) - cis_count
            stats['cis_pct'] = (cis_count / len(df)) * 100
            stats['trans_pct'] = ((len(df) - cis_count) / len(df)) * 100
        else:
            stats['cis_count'] = stats['trans_count'] = 0
            stats['cis_pct'] = stats['trans_pct'] = 0
        
        # LOD score stats
        if 'qtl_lod' in df.columns:
            # Convert to numeric and handle any non-numeric values
            numeric_lod = pd.to_numeric(df['qtl_lod'], errors='coerce')
            numeric_lod = numeric_lod.dropna()
            
            if len(numeric_lod) > 0:
                stats['max_lod'] = numeric_lod.max()
                stats['mean_lod'] = numeric_lod.mean()
                stats['median_lod'] = numeric_lod.median()
            else:
                stats['max_lod'] = stats['mean_lod'] = stats['median_lod'] = 0
        else:
            stats['max_lod'] = stats['mean_lod'] = stats['median_lod'] = 0
        
        # Top genes
        if 'gene_symbol' in df.columns and 'qtl_lod' in df.columns:
            # Convert qtl_lod to numeric for sorting
            df_copy = df.copy()
            df_copy['qtl_lod'] = pd.to_numeric(df_copy['qtl_lod'], errors='coerce')
            df_copy = df_copy.dropna(subset=['qtl_lod'])
            
            if len(df_copy) > 0:
                top_genes_raw = df_copy.nlargest(10, 'qtl_lod')['gene_symbol'].tolist()
                # Filter out NaN values and convert to strings
                stats['top_genes'] = [str(gene) for gene in top_genes_raw if pd.notna(gene)]
            else:
                stats['top_genes'] = []
        else:
            stats['top_genes'] = []
        
        # Analysis types and cohorts
        stats['analysis_types'] = df['analysis_type'].unique().tolist() if 'analysis_type' in df.columns else []
        stats['cohorts'] = df['cohort'].unique().tolist() if 'cohort' in df.columns else []
        
        return stats
    
    def _gene_summaries(self) -> List[Dict[str, Any]]:
        """Generate gene-level summaries for relevant traits."""
        summaries = []
        
        # Only generate for traits with gene information
        gene_traits = [t for t in self.trait_data.keys() if 'gene' in t or 'clinical' in t]
        
        for trait_type in gene_traits:
            df = self.trait_data[trait_type]
            if 'gene_symbol' not in df.columns:
                continue
            
            # Group by gene (limit to genes with multiple QTLs)
            gene_groups = df.groupby('gene_symbol')
            for gene, gene_df in gene_groups:
                if len(gene_df) < 2:  # Skip single-QTL genes
                    continue
                
                if len(summaries) > 15000:  # Limit gene summaries
                    break
                
                qtl_count = len(gene_df)
                
                # Handle max_lod calculation safely
                if 'qtl_lod' in gene_df.columns:
                    numeric_lod = pd.to_numeric(gene_df['qtl_lod'], errors='coerce')
                    numeric_lod = numeric_lod.dropna()
                    max_lod = numeric_lod.max() if len(numeric_lod) > 0 else 0
                else:
                    max_lod = 0
                    
                chromosomes = gene_df['qtl_chr'].unique().tolist() if 'qtl_chr' in gene_df.columns else []
                
                content = f"""
Gene {gene} QTL Summary ({trait_type.replace('_', ' ').title()})

This gene has {qtl_count} QTL findings with maximum LOD score {max_lod:.2f}.

Details:
- QTL count: {qtl_count}
- Max LOD: {max_lod:.2f}
- Chromosomes: {', '.join(map(str, chromosomes))}
- Gene type: {gene_df['gene_type'].iloc[0] if 'gene_type' in gene_df.columns else 'Unknown'}
                """.strip()
                
                summaries.append({
                    'id': f'gene_{trait_type}_{gene}',
                    'type': 'gene_summary',
                    'content': content,
                    'metadata': {
                        'gene_symbol': gene,
                        'trait_type': trait_type,
                        'qtl_count': qtl_count,
                        'max_lod': max_lod
                    }
                })
        
        return summaries
    
    def _chromosome_summaries(self) -> List[Dict[str, Any]]:
        """Generate chromosome-level summaries."""
        summaries = []
        
        # Combine all data
        all_df = pd.concat(self.trait_data.values(), ignore_index=True)
        
        # Handle chromosome sorting - some might be strings, some numbers
        try:
            chromosomes = all_df['qtl_chr'].unique()
            # Try to sort as numbers first, fall back to string sorting
            try:
                sorted_chrs = sorted([x for x in chromosomes if pd.notna(x)], key=lambda x: int(x) if str(x).isdigit() else float('inf'))
            except:
                sorted_chrs = sorted([x for x in chromosomes if pd.notna(x)], key=str)
        except:
            sorted_chrs = []
        
        for chromosome in sorted_chrs:
            chr_df = all_df[all_df['qtl_chr'] == chromosome]
            trait_counts = chr_df['trait_type'].value_counts().to_dict()
            
            content = f"""
Chromosome {chromosome} QTL Summary

Contains {len(chr_df):,} QTL findings across multiple trait types.

Distribution by Trait:
{chr(10).join([f"- {tt.replace('_', ' ').title()}: {count:,}" for tt, count in trait_counts.items()])}

Genomic span and top QTLs provide insights into regional genetic architecture.
            """.strip()
            
            summaries.append({
                'id': f'chr_{chromosome}',
                'type': 'chromosome_summary',
                'content': content,
                'metadata': {
                    'chromosome': str(chromosome),
                    'total_qtls': len(chr_df),
                    'trait_types': ','.join(trait_counts.keys())
                }
            })
        
        return summaries
    
    def _significance_summaries(self) -> List[Dict[str, Any]]:
        """Generate significance tier summaries."""
        summaries = []
        
        tiers = [
            ('very_high', 20.0, float('inf')),
            ('high', 10.0, 20.0),
            ('moderate', 5.0, 10.0),
            ('low', 3.0, 5.0),
            ('suggestive', 0.0, 3.0)
        ]
        
        all_df = pd.concat(self.trait_data.values(), ignore_index=True)
        
        # Ensure qtl_lod is numeric
        if 'qtl_lod' in all_df.columns:
            all_df['qtl_lod'] = pd.to_numeric(all_df['qtl_lod'], errors='coerce')
            all_df = all_df.dropna(subset=['qtl_lod'])
        
        for tier_name, min_lod, max_lod in tiers:
            if max_lod == float('inf'):
                tier_df = all_df[all_df['qtl_lod'] >= min_lod]
            else:
                tier_df = all_df[(all_df['qtl_lod'] >= min_lod) & (all_df['qtl_lod'] < max_lod)]
            
            if len(tier_df) == 0:
                continue
            
            trait_counts = tier_df['trait_type'].value_counts().to_dict()
            
            content = f"""
{tier_name.replace('_', ' ').title()} Significance QTLs

LOD score range: {min_lod} - {max_lod if max_lod != float('inf') else '∞'}
Total QTLs: {len(tier_df):,}

Distribution by Trait:
{chr(10).join([f"- {tt.replace('_', ' ').title()}: {count:,}" for tt, count in trait_counts.items()])}
            """.strip()
            
            summaries.append({
                'id': f'significance_{tier_name}',
                'type': 'significance_summary',
                'content': content,
                'metadata': {
                    'significance_tier': tier_name,
                    'lod_min': min_lod,
                    'total_qtls': len(tier_df)
                }
            })
        
        return summaries
    
    def _analysis_summaries(self) -> List[Dict[str, Any]]:
        """Generate analysis type summaries."""
        summaries = []
        
        all_df = pd.concat(self.trait_data.values(), ignore_index=True)
        
        for analysis_type in all_df['analysis_type'].unique():
            analysis_df = all_df[all_df['analysis_type'] == analysis_type]
            trait_counts = analysis_df['trait_type'].value_counts().to_dict()
            
            content = f"""
{analysis_type.replace('_', ' ').title()} Analysis Summary

Total QTLs: {len(analysis_df):,}

Distribution by Trait:
{chr(10).join([f"- {tt.replace('_', ' ').title()}: {count:,}" for tt, count in trait_counts.items()])}

Analysis focus: {self._get_analysis_description(analysis_type)}
            """.strip()
            
            summaries.append({
                'id': f'analysis_{analysis_type}',
                'type': 'analysis_summary',
                'content': content,
                'metadata': {
                    'analysis_type': analysis_type,
                    'total_qtls': len(analysis_df)
                }
            })
        
        return summaries
    
    def _cohort_summaries(self) -> List[Dict[str, Any]]:
        """Generate cohort summaries."""
        summaries = []
        
        all_df = pd.concat(self.trait_data.values(), ignore_index=True)
        
        for cohort in all_df['cohort'].unique():
            cohort_df = all_df[all_df['cohort'] == cohort]
            trait_counts = cohort_df['trait_type'].value_counts().to_dict()
            
            content = f"""
{cohort.replace('_', ' ').title()} Cohort Summary

Total QTLs: {len(cohort_df):,}

Distribution by Trait:
{chr(10).join([f"- {tt.replace('_', ' ').title()}: {count:,}" for tt, count in trait_counts.items()])}

Cohort description: {self._get_cohort_description(cohort)}
            """.strip()
            
            summaries.append({
                'id': f'cohort_{cohort}',
                'type': 'cohort_summary',
                'content': content,
                'metadata': {
                    'cohort': cohort,
                    'total_qtls': len(cohort_df)
                }
            })
        
        return summaries
    
    def _get_analysis_description(self, analysis_type: str) -> str:
        """Get description for analysis type."""
        descriptions = {
            'additive': 'Consistent additive genetic effects across conditions',
            'diet_interactive': 'Genetic effects that vary by diet',
            'sex_interactive': 'Genetic effects that vary by sex',
            'qtlxdiet': 'QTL-by-diet interaction effects',
            'qtlxsex': 'QTL-by-sex interaction effects',
            'qtlxsexbydiet': 'Complex three-way QTL-sex-diet interactions'
        }
        return descriptions.get(analysis_type, 'Genetic effects under specific conditions')
    
    def _get_cohort_description(self, cohort: str) -> str:
        """Get description for cohort."""
        descriptions = {
            'all_mice': 'Both sexes on both diet types',
            'male_mice': 'Male mice only',
            'female_mice': 'Female mice only',
            'HC_mice': 'High-carbohydrate diet mice',
            'HF_mice': 'High-fat diet mice'
        }
        return descriptions.get(cohort, f'{cohort} cohort')
    
    def setup_vector_store(self, summaries: List[Dict[str, Any]]):
        """Set up ChromaDB vector store."""
        logger.info(f"\nSetting up vector store with {len(summaries)} documents...")
        
        try:
            # Delete existing collection
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass
            
            # Create collection
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Multi-file QTL summaries"}
            )
            
            # Prepare data
            documents = []
            metadatas = []
            ids = []
            
            for summary in summaries:
                documents.append(summary['content'])
                ids.append(summary['id'])
                
                # Clean metadata
                metadata = {}
                for key, value in summary['metadata'].items():
                    if isinstance(value, (list, tuple)):
                        metadata[key] = ','.join(map(str, value))
                    elif pd.isna(value):
                        metadata[key] = 'unknown'
                    else:
                        metadata[key] = str(value)
                metadatas.append(metadata)
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                batch_docs = documents[i:end_idx]
                batch_metas = metadatas[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(batch_docs, show_progress_bar=False)
                
                # Add to collection
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids,
                    embeddings=embeddings.tolist()
                )
                
                logger.info(f"  Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            logger.info("✅ Vector store setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            raise
    
    def semantic_search(self, query: str, n_results: int = 5, trait_filter: str = None) -> List[Dict]:
        """Perform semantic search."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            
            where_filter = None
            if trait_filter:
                where_filter = {"trait_type": trait_filter}
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def sql_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query."""
        try:
            return self.duckdb_conn.execute(query).df()
        except Exception as e:
            logger.error(f"SQL error: {e}")
            return pd.DataFrame()
    
    def get_schema(self) -> str:
        """Get database schema information."""
        try:
            schema_info = self.duckdb_conn.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'qtl_data'
            """).df()
            
            sample_data = self.duckdb_conn.execute("SELECT * FROM qtl_data LIMIT 3").df()
            
            return f"""
DATABASE SCHEMA:
{schema_info.to_string(index=False)}

SAMPLE DATA:
{sample_data.to_string(index=False)}

TRAIT TYPES: {', '.join(self.trait_data.keys())}

NOTE: Use cis == 'TRUE' for cis-acting QTLs (stored as string)
            """.strip()
            
        except Exception as e:
            return f"Schema error: {e}"
    
    def initialize_system(self):
        """Initialize the complete system."""
        logger.info("Initializing Multi-File QTL System...")
        
        # Load data
        self.load_all_data()
        
        # Generate summaries
        summaries = self.generate_summaries()
        
        # Setup vector store
        self.setup_vector_store(summaries)
        
        # Print stats
        total_records = sum(len(df) for df in self.trait_data.values())
        logger.info(f"\n✅ System initialized!")
        logger.info(f"Total records: {total_records:,}")
        logger.info(f"Trait types: {len(self.trait_data)}")
        logger.info(f"Summary docs: {len(summaries):,}")


if __name__ == "__main__":
    # Initialize system
    system = MultiFileQTLSystem()
    system.initialize_system()
    
    # Test searches
    print("\n" + "="*50)
    print("Testing semantic search...")
    results = system.semantic_search("liver genes with high LOD scores", n_results=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}: {result['content'][:200]}...")
    
    # Test SQL
    print("\n" + "="*50)
    print("Testing SQL query...")
    sql_result = system.sql_query("""
        SELECT trait_type, COUNT(*) as count 
        FROM qtl_data 
        GROUP BY trait_type 
        ORDER BY count DESC
    """)
    print(sql_result)