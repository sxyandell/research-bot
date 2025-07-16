#!/usr/bin/env python3
"""
WORKING VERSION
Hybrid QTL Analysis System

Combines a vector store with a relational database for efficient QTL analysis.
"""

import pandas as pd
import json
import numpy as np
import sqlite3
import duckdb
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import mygene
import re
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai.protos import Tool, FunctionDeclaration, Part

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache to avoid re-querying the API for the same gene
gene_cache = {}
mg = mygene.MyGeneInfo()

def fetch_gene_context(gene_symbol: str) -> Dict[str, Any]:
    """
    Fetches gene summary, GO terms, and pathways from mygene.info.
    This version is more robust, searching multiple scopes and validating hits.
    """
    if gene_symbol in gene_cache:
        return gene_cache[gene_symbol]
    
    if not gene_symbol or pd.isna(gene_symbol):
        return {}

    try:
        logger.info(f"Querying mygene.info for symbol: '{gene_symbol}'")
        # Broader search across relevant fields
        result = mg.query(
            gene_symbol,
            scopes="symbol,alias,ensembl.gene",
            species="mouse",
            fields="name,symbol,summary,go,pathway",
            fetch_all=False
        )
        logger.debug(f"MyGene.info raw result for '{gene_symbol}': {result}")
        
        if result and result.get('hits'):
            # Find the best hit: one where the symbol exactly matches our query (case-insensitive)
            best_hit = None
            for hit in result['hits']:
                if hit.get('symbol', '').lower() == gene_symbol.lower():
                    best_hit = hit
                    logger.info(f"Found exact match for '{gene_symbol}'")
                    break
            
            # If no exact match, take the top hit as a fallback
            if not best_hit:
                best_hit = result['hits'][0]
                logger.warning(f"No exact symbol match for '{gene_symbol}'; using top hit '{best_hit.get('symbol', 'N/A')}' as fallback.")

            context = {
                'name': best_hit.get('name', 'No name available.'),
                'summary': best_hit.get('summary', 'No summary available.'),
                'go_terms_bp': [term.get('term', 'N/A') for term in best_hit.get('go', {}).get('BP', []) if isinstance(term, dict)],
                'kegg_pathways': [p.get('name', 'N/A') for p in best_hit.get('pathway', {}).get('kegg', []) if isinstance(p, dict)]
            }
            gene_cache[gene_symbol] = context
            logger.info(f"Successfully fetched context for '{gene_symbol}'")
            return context
        else:
            logger.warning(f"No hits returned from mygene.info for symbol: '{gene_symbol}'")

    except Exception as e:
        logger.error(f"Could not fetch data for gene '{gene_symbol}': {e}")
        
    # Cache the failure to avoid re-querying
    gene_cache[gene_symbol] = {} 
    return {}

class HybridQTLSystem:
    """
    Hybrid 2-layer QTL analysis system:
    Layer 1: Vector store with embedded summary docs (~10k-20k docs)
    Layer 2: Relational store with raw rows for exact queries/analytics
    """
    
    def __init__(self, csv_file_path: str, chroma_db_path: str = "./hybrid_chroma_db"):
        self.csv_file = csv_file_path
        self.chroma_db_path = chroma_db_path
        self.raw_data = None
        self.summary_docs = []
        
        # Initialize stores
        self.chroma_client = None
        self.vector_collection = None
        self.duck_conn = None
        
        # Initialize embedding models
        self.local_embedder = None
        self.google_embedder = None
        self.generative_model = None
        
        # Load data and models immediately for a robust, ready-to-use instance.
        self.load_raw_data()
        self.setup_embedding_models()
        
    def load_raw_data(self):
        """Load the raw QTL data into memory and DuckDB."""
        try:
            logger.info(f"Loading raw data from {self.csv_file}")
            self.raw_data = pd.read_csv(self.csv_file)
            logger.info(f"✅ Loaded {len(self.raw_data)} QTL records")
            
            # Initialize DuckDB for fast analytics
            self.duck_conn = duckdb.connect(":memory:")
            self.duck_conn.register('qtl_data', self.raw_data)
            
            # Create useful indexes
            self.duck_conn.execute("""
                CREATE TABLE qtl_peaks AS SELECT * FROM qtl_data
            """)
            
            # Create indexes for common queries
            self.duck_conn.execute("""
                CREATE INDEX idx_gene_symbol ON qtl_peaks(gene_symbol)
            """)
            self.duck_conn.execute("""
                CREATE INDEX idx_qtl_chr ON qtl_peaks(qtl_chr)
            """)
            self.duck_conn.execute("""
                CREATE INDEX idx_qtl_lod ON qtl_peaks(qtl_lod)
            """)
            
            logger.info("✅ DuckDB initialized with indexes")
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def setup_embedding_models(self, google_api_key: Optional[str] = None):
        """Setup both Google and local embedding models."""
        # Setup local embedder
        try:
            logger.info("Initializing local sentence transformer...")
            self.local_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Local embedder ready")
        except Exception as e:
            logger.error(f"❌ Failed to load local embedder: {e}")
        
        # Setup Google services if API key provided
        if google_api_key:
            try:
                genai.configure(api_key=google_api_key)
                self.google_embedder = genai.GenerativeModel('models/text-embedding-004')
                logger.info("✅ Google embedder ready")
                self.generative_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Google generative model ready")
            except Exception as e:
                logger.error(f"❌ Failed to setup Google services: {e}")
    
    def generate_all_document_types(self) -> List[Dict[str, Any]]:
        """
        Generate all document types for a hybrid RAG strategy:
        1. Enriched Gene Summaries (with external biological context)
        2. Chromosome Summaries
        3. LOD Score Tier Summaries
        4. Per-Peak Granular Documents
        """
        all_docs = []
        
        # 1. GENE-LEVEL SUMMARIES (ENRICHED)
        logger.info("Generating enriched gene-level summaries...")
        gene_groups = self.raw_data.groupby('gene_symbol')
        
        for gene_symbol, gene_data in gene_groups:
            if pd.isna(gene_symbol) or gene_symbol == 'nan':
                continue
                
            qtl_count = len(gene_data)
            max_lod = gene_data['qtl_lod'].max()
            chromosomes = gene_data['qtl_chr'].unique().tolist()
            cis_count = (gene_data['cis'] == 'TRUE').sum()
            trans_count = qtl_count - cis_count
            
            context = fetch_gene_context(gene_symbol)
            
            summary_text = f"""
            Gene Summary for {gene_symbol} (Full Name: {context.get('name', 'N/A')})
            Function: {context.get('summary', 'No summary available.')}
            QTL Profile: This gene has {qtl_count} QTL peaks with a maximum LOD score of {max_lod:.2f}. 
            Genetic regulation appears to be primarily {'local (cis)' if cis_count > trans_count else 'distant (trans)'}.
            Biological Pathways (KEGG): {', '.join(context.get('kegg_pathways', ['N/A']))}
            Gene Ontology (Biological Process): {', '.join(context.get('go_terms_bp', ['N/A'])[:5])}
            """
            
            all_docs.append({
                'id': f'gene_{gene_symbol}',
                'content': summary_text.strip(),
                'metadata': {
                    'type': 'gene_summary',
                    'gene_symbol': gene_symbol,
                    'max_lod': float(max_lod),
                    'qtl_count': int(qtl_count),
                    'pathways': json.dumps(context.get('kegg_pathways', [])),
                    'go_terms': json.dumps(context.get('go_terms_bp', []))
                }
            })

        # 2. CHROMOSOME-LEVEL SUMMARIES
        logger.info("Generating chromosome-level summaries...")
        chr_groups = self.raw_data.groupby('qtl_chr')
        
        for chr_name, chr_data in chr_groups:
            qtl_count = len(chr_data)
            unique_genes = chr_data['gene_symbol'].nunique()
            max_lod = chr_data['qtl_lod'].max()
            top_genes = chr_data.nlargest(5, 'qtl_lod')['gene_symbol'].tolist()
            
            summary_text = f"""
            Chromosome {chr_name} Summary: This chromosome contains {qtl_count} significant QTL peaks affecting {unique_genes} unique genes.
            The maximum LOD score is {max_lod:.2f}. The most strongly associated genes include {', '.join(map(str, top_genes[:3]))}.
            This indicates {'high' if qtl_count > 1000 else 'moderate'} regulatory activity across the chromosome.
            """
            
            all_docs.append({
                'id': f'chr_{chr_name}',
                'content': summary_text.strip(),
                'metadata': {
                    'type': 'chromosome_summary',
                    'chromosome': str(chr_name),
                    'qtl_count': int(qtl_count),
                    'unique_genes': int(unique_genes),
                    'max_lod': float(max_lod),
                    'top_genes': ', '.join(map(str, top_genes))
                }
            })

        # 3. LOD SCORE TIER SUMMARIES
        logger.info("Generating significance tier summaries...")
        lod_tiers = [(100, 'extremely_high'), (50, 'very_high'), (20, 'high')]
        
        for min_lod, tier_name in lod_tiers:
            tier_data = self.raw_data[self.raw_data['qtl_lod'] >= min_lod]
            if len(tier_data) == 0: continue
            
            qtl_count = len(tier_data)
            unique_genes = tier_data['gene_symbol'].nunique()
            
            summary_text = f"""
            Significance Tier Summary ({tier_name}, LOD > {min_lod}):
            There are {qtl_count} QTLs with {tier_name.replace('_', ' ')} evidence of association, affecting {unique_genes} genes.
            These peaks represent the highest-confidence genetic signals in the dataset, suitable for detailed validation studies.
            """
            
            all_docs.append({
                'id': f'lod_tier_{tier_name}',
                'content': summary_text.strip(),
                'metadata': {
                    'type': 'significance_summary',
                    'min_lod': min_lod, 
                    'qtl_count': qtl_count
                }
            })

        # 4. PER-PEAK GRANULAR DOCUMENTS
        logger.info(f"Generating per-peak document for all {len(self.raw_data)} records...")
        for index, row in self.raw_data.iterrows():
            content = (
                f"A quantitative trait locus (QTL) for the phenotype '{row.get('phenotype', 'N/A')}' "
                f"was identified on chromosome {row['qtl_chr']} at position {row['qtl_pos']:.2f} Mb. "
                f"It has a LOD score of {row['qtl_lod']:.2f}. "
                f"This QTL is associated with the gene '{row['gene_symbol']}' and is a {'cis-acting' if row.get('cis') == 'TRUE' else 'trans-acting'} regulator."
            )
            
            doc_id = f"peak_{index}_{row['gene_symbol']}_{row['qtl_chr']}_{row['qtl_pos']}"
            
            all_docs.append({
                'id': doc_id,
                'content': content,
                'metadata': {
                    'type': 'qtl_peak',
                    'gene_symbol': row['gene_symbol'],
                    'phenotype': row.get('phenotype', 'N/A'),
                    'chromosome': str(row['qtl_chr']),
                    'position_mb': float(row['qtl_pos']),
                    'lod_score': float(row['qtl_lod']),
                    'cis': True if row.get('cis') == 'TRUE' else False,
                    'gene_id': row.get('gene_id', 'N/A')
                }
            })
            
        self.summary_docs = all_docs
        logger.info(f"✅ Generated a total of {len(all_docs)} documents of all types")
        return all_docs
    
    def setup_vector_store(self, use_google_embeddings: bool = True):
        """Setup ChromaDB vector store with all document types."""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            collection_name = "qtl_hybrid_store"
            self.vector_collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Hybrid store with summaries and per-peak documents"}
            )
            logger.info(f"✅ Ensured collection '{collection_name}' exists.")

            if self.vector_collection.count() > 0:
                logger.info(f"Collection already contains {self.vector_collection.count()} documents. Skipping population.")
                return

            logger.info("Collection is empty. Generating documents and populating vector store...")
            self.generate_all_document_types()
            
            logger.info(f"Adding {len(self.summary_docs)} documents to vector store...")
            
            ids = [doc['id'] for doc in self.summary_docs]
            documents = [doc['content'] for doc in self.summary_docs]
            metadatas = [doc['metadata'] for doc in self.summary_docs]
            
            embeddings = []
            # Generate embeddings
            if use_google_embeddings and self.google_embedder:
                logger.info("Using Google embeddings...")
                # Google embeddings (implement batch processing)
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    try:
                        batch_embeddings = []
                        for doc in batch:
                            result = genai.embed_content(
                                model="models/text-embedding-004",
                                content=doc
                            )
                            batch_embeddings.append(result['embedding'])
                        embeddings.extend(batch_embeddings)
                    except Exception as e:
                        logger.warning(f"Google embedding failed for batch {i}: {e}")
                        # Fallback to local embeddings for this batch
                        if self.local_embedder:
                            local_batch_embeddings = self.local_embedder.encode(batch).tolist()
                            embeddings.extend(local_batch_embeddings)
                
            elif self.local_embedder:
                logger.info("Using local embeddings...")
                embeddings = self.local_embedder.encode(documents).tolist()
            
            else:
                raise ValueError("No embedding model available")
            
            # Add to ChromaDB in batches
            batch_size = 2000
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                self.vector_collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    embeddings=embeddings[i:end_idx]
                )
                logger.info(f"Added batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
            
            logger.info(f"✅ Added {len(self.summary_docs)} documents to vector store")
            
        except Exception as e:
            logger.error(f"❌ Error setting up vector store: {e}")
            raise
    
    def semantic_search(self, query: str, n_results: int = 5, where_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Layer 1: Semantic search on all document types.
        Can be filtered by document type (e.g., 'gene_summary', 'qtl_peak').
        """
        if not self.vector_collection:
            raise ValueError("Vector store not initialized. Call setup_vector_store() first.")
        
        try:
            query_params = {'query_texts': [query], 'n_results': n_results}
            if where_filter:
                query_params['where'] = where_filter

            results = self.vector_collection.query(**query_params)
            
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def analytical_query(self, sql_query: str) -> pd.DataFrame:
        """
        Layer 2: Direct SQL queries on raw data.
        Use this for exact lookups and analytics.
        """
        if not self.duck_conn:
            raise ValueError("DuckDB connection not initialized")
        
        try:
            result = self.duck_conn.execute(sql_query).fetchdf()
            return result
        except Exception as e:
            logger.error(f"❌ Analytical query failed: {e}")
            raise
    
    def get_gene_details(self, gene_symbol: str) -> Dict:
        """Quick helper for gene-specific queries, now including biological context."""
        # 1. Get QTL data from DuckDB
        query = """
        SELECT * FROM qtl_peaks 
        WHERE lower(gene_symbol) = lower(?) 
        ORDER BY qtl_lod DESC
        """
        qtl_result_df = self.duck_conn.execute(query, [gene_symbol]).fetchdf()

        # 2. Get biological context
        biological_context = fetch_gene_context(gene_symbol)

        # 3. Combine them
        return {
            'gene_symbol': gene_symbol,
            'biological_summary': biological_context.get('summary', 'No summary available.'),
            'go_terms': biological_context.get('go_terms_bp', []),
            'kegg_pathways': biological_context.get('kegg_pathways', []),
            'qtl_count': len(qtl_result_df),
            'qtls': qtl_result_df.to_dict('records') if len(qtl_result_df) > 0 else []
        }
    
    def get_specific_peak_data(self, gene_symbol: str, chromosome: str, position_mb: float) -> List[Dict[str, Any]]:
        """
        Fetches all columns for a specific QTL peak based on gene, chromosome, and a small window around the position.
        """
        # Use a small window for position to handle floating point variations
        pos_margin = 0.01 
        query = """
        SELECT * FROM qtl_peaks 
        WHERE lower(gene_symbol) = lower(?) 
          AND qtl_chr = ? 
          AND qtl_pos BETWEEN ? AND ?
        """
        result_df = self.duck_conn.execute(
            query, 
            [gene_symbol, chromosome, str(position_mb - pos_margin), str(position_mb + pos_margin)]
        ).fetchdf()
        
        return result_df.to_dict('records')

    def get_chromosome_stats(self, chromosome: str) -> Dict:
        """Quick helper for chromosome statistics."""
        query = """
        SELECT 
            COUNT(*) as qtl_count,
            COUNT(DISTINCT gene_symbol) as unique_genes,
            AVG(qtl_lod) as mean_lod,
            MAX(qtl_lod) as max_lod,
            MIN(qtl_lod) as min_lod,
            SUM(CASE WHEN cis = 'TRUE' THEN 1 ELSE 0 END) as cis_count
        FROM qtl_peaks 
        WHERE qtl_chr = ?
        """
        result = self.duck_conn.execute(query, [chromosome]).fetchdf()
        return result.to_dict('records')[0] if len(result) > 0 else {}
    
    def correlate_genes(self, gene1: str, gene2: str) -> Dict:
        """Analyze correlation between two genes' QTL patterns."""
        query = """
        WITH gene1_data AS (
            SELECT qtl_chr, qtl_pos, qtl_lod FROM qtl_peaks WHERE gene_symbol = ?
        ),
        gene2_data AS (
            SELECT qtl_chr, qtl_pos, qtl_lod FROM qtl_peaks WHERE gene_symbol = ?
        )
        SELECT 
            COUNT(*) as common_chromosomes,
            CORR(g1.qtl_lod, g2.qtl_lod) as lod_correlation
        FROM gene1_data g1 
        JOIN gene2_data g2 ON g1.qtl_chr = g2.qtl_chr
        """
        result = self.duck_conn.execute(query, [gene1, gene2]).fetchdf()
        return result.to_dict('records')[0] if len(result) > 0 else {}
    
    def _define_tools(self):
        """Defines the function calling tools for the LLM."""
        self.tools = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="get_gene_details",
                    description="Retrieves a complete summary for a single, named gene, including its biological function, GO terms, and all associated QTL peak data. Use this for any question about a specific gene. If the user asks for 'the peak' (singular), use this tool to show all peaks, as they may not know there are multiple.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "gene_symbol": {"type": "STRING", "description": "The official symbol of the gene, e.g., 'Apoe' or 'Gnai3'."}
                        },
                        "required": ["gene_symbol"],
                    },
                ),
                FunctionDeclaration(
                    name="get_chromosome_stats",
                    description="Provides summary statistics for all QTLs on a specific chromosome.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "chromosome": {"type": "STRING", "description": "The chromosome identifier, e.g., '1', '2', or 'X'."}
                        },
                        "required": ["chromosome"],
                    },
                ),
                FunctionDeclaration(
                    name="analytical_query_top_lod",
                    description="Finds the top N genes or peaks with the absolute highest LOD scores in the entire dataset. Use for questions about 'highest', 'top', or 'strongest' overall signals.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "limit": {"type": "INTEGER", "description": "The number of results to return, e.g., 5 or 10."}
                        },
                        "required": ["limit"],
                    },
                ),
                FunctionDeclaration(
                    name="get_specific_peak_data",
                    description="Retrieves all data for a single, specific QTL peak, identified by its gene, chromosome, and approximate position. Use this for finding specific values (like A, B, C values) for one peak.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "gene_symbol": {"type": "STRING", "description": "The official symbol of the gene, e.g., 'Apoe'."},
                            "chromosome": {"type": "STRING", "description": "The chromosome identifier, e.g., '3'."},
                            "position_mb": {"type": "NUMBER", "description": "The approximate position of the peak in Megabases (Mb), e.g., 94.34."}
                        },
                        "required": ["gene_symbol", "chromosome", "position_mb"],
                    },
                ),
                FunctionDeclaration(
                    name="semantic_search",
                    description="Use for broad, conceptual, or vague questions that are not about a specific gene or chromosome, or that involve biological concepts like pathways or functions. This is the fallback tool.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING", "description": "The user's original, unmodified question."}
                        },
                        "required": ["query"],
                    },
                ),
            ]
        )

    def analytical_query_top_lod(self, limit: int = 5) -> pd.DataFrame:
        """Wrapper for the 'top LOD' analytical query."""
        sql = f"SELECT gene_symbol, qtl_lod, qtl_chr, qtl_pos FROM qtl_peaks ORDER BY qtl_lod DESC LIMIT {limit}"
        return self.analytical_query(sql)


    def intelligent_router(self, query: str) -> Dict[str, Any]:
        """
        New, smarter router that uses an LLM with function calling to determine the user's intent.
        """
        if not hasattr(self, 'tools'):
            self._define_tools()

        # The first call to the LLM is just to decide which tool to use
        response = self.generative_model.generate_content(
            query,
            tools=[self.tools],
            # Safety settings can be important for tool use
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        try:
            function_call = response.candidates[0].content.parts[0].function_call
            tool_name = function_call.name
            tool_args = {key: value for key, value in function_call.args.items()}
            
            logger.info(f"🤖 LLM decided to use tool: '{tool_name}' with args: {tool_args}")
            
            # Now, execute the chosen function
            if hasattr(self, tool_name):
                tool_function = getattr(self, tool_name)
                
                # Special handling for semantic search which has a different return format
                if tool_name == 'semantic_search':
                    # The LLM will pass the original query back to us
                    results_data = tool_function(query=tool_args['query'], n_results=5)
                # Handle analytical functions that return DataFrames
                elif tool_name == 'analytical_query_top_lod':
                    results_df = tool_function(**tool_args)
                    results_data = results_df.to_dict('records')
                # Handle helper functions that return a single dictionary
                else:
                    results_data = tool_function(**tool_args)
                    # Ensure results are always a list for consistency
                    if not isinstance(results_data, list):
                        results_data = [results_data]
                
                return {
                    'detected_intent': 'tool_call',
                    'method': tool_name,
                    'arguments': tool_args,
                    'results': results_data,
                    'result_count': len(results_data)
                }
            else:
                raise ValueError(f"LLM wanted to call a non-existent tool: {tool_name}")

        except (AttributeError, IndexError):
            # The LLM didn't choose a tool, so we fall back to a simple semantic search
            logger.warning("LLM did not choose a tool. Falling back to default semantic search.")
            results_data = self.semantic_search(query=query, n_results=5)
            return {
                'detected_intent': 'semantic_fallback',
                'method': 'semantic_search',
                'results': results_data,
                'result_count': len(results_data)
            }
    
    def generate_response(self, query: str, search_results: Dict) -> str:
        """Generates a natural language response using retrieved context."""
        if not self.generative_model:
            return "The generative AI model is not configured. Please provide a Google AI API key to enable natural language responses."

        intent = search_results.get('detected_intent')
        method = search_results.get('method')
        context_str = ""
        
        if not search_results.get('results'):
            return "I couldn't find any relevant information in the database to answer your question."

        # NEW: Special formatting for get_gene_details
        if method == 'get_gene_details' and search_results['results']:
            gene_data = search_results['results'][0]  # It's a list with one dict
            context_parts = []
            context_parts.append(f"Gene: {gene_data['gene_symbol']}")
            if gene_data.get('biological_summary') and gene_data['biological_summary'] != 'No summary available.':
                context_parts.append(f"Biological Summary: {gene_data['biological_summary']}")
            if gene_data.get('go_terms'):
                context_parts.append(f"Gene Ontology Terms: {', '.join(gene_data['go_terms'])}")
            if gene_data.get('kegg_pathways'):
                context_parts.append(f"KEGG Pathways: {', '.join(gene_data['kegg_pathways'])}")

            context_parts.append(f"\nThis gene has {gene_data['qtl_count']} associated QTL peaks.")
            if gene_data.get('qtls'):
                context_parts.append("Here is the data for the top peaks:")
                # Limit to top 3 for a concise context
                for i, qtl in enumerate(gene_data['qtls'][:3]):
                    qtl_info = (f"  - Peak {i+1}: Located on Chromosome {qtl.get('qtl_chr', 'N/A')} "
                                f"at position {qtl.get('qtl_pos', 0.0):.2f} Mb "
                                f"with a LOD score of {qtl.get('qtl_lod', 0.0):.2f}. "
                                f"It is a {'cis-acting' if qtl.get('cis') else 'trans-acting'} regulator.")
                    context_parts.append(qtl_info)
            context_str = "\n".join(context_parts)
        
        # Fallback to existing logic for all other cases
        else:
            if isinstance(search_results.get('results'), list):
                if search_results['results'] and isinstance(search_results['results'][0], dict) and 'content' in search_results['results'][0]:
                    # Semantic search results
                    context_str = "\n---\n".join([doc['content'] for doc in search_results['results']])
                else:
                    # Analytical/tool_call results
                    df = pd.DataFrame(search_results['results'])
                    context_str = "Based on the following data table:\n" + df.to_string(index=False)
            elif isinstance(search_results.get('results'), pd.DataFrame):
                df = search_results['results']
                context_str = "Based on the following data table:\n" + df.to_string(index=False)

        prompt = f"""
You are a specialized bioinformatics research assistant for a QTL database.
Your task is to synthesize a clear and accurate answer for the user based *only* on the provided search results.

**Instructions:**
1.  Analyze the user's question to understand their core intent.
2.  Examine the retrieved context below. Note whether it came from a precise analytical query (like a data table) or a broader semantic search (like text documents).
3.  Synthesize a comprehensive answer.
    - If the context is a data table from an analytical query, present the key findings from the table directly.
    - If the context is from a semantic search, summarize the information from the provided documents.
    - If both are present, prioritize the specific data from the analytical query and use the semantic context for background information.
4.  **Crucially, cite your sources.** Refer to the document IDs (e.g., 'gene_Gnai3', 'peak_1234_...') or the analytical query when explaining your answer.
5.  If the context does not contain the information needed to answer the question, you MUST state that the information is not available in the database and do not invent an answer.

**User's Question:** "{query}"

**Retrieved Context from Database:**
---
{context_str}
---

Based *only* on the context above, provide a helpful, synthesized answer to the user's question.
"""
        
        try:
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"❌ LLM response generation failed: {e}")
            return "There was an error generating an AI response. Please check the logs."

    def ask(self, query: str) -> Dict[str, Any]:
        """
        High-level method to ask a question, get results, and generate a response.
        UPDATED to use the intelligent_router.
        """
        # Replace the old hybrid_search with the new router
        search_results = self.intelligent_router(query)
        
        # The response generation part remains the same
        ai_response = self.generate_response(query, search_results)
        
        search_results['ai_response'] = ai_response
        search_results['query'] = query
        search_results['timestamp'] = datetime.now().isoformat()
        return search_results

    def save_summary_docs(self, output_file: str):
        """Save generated summary documents to JSON."""
        if not self.summary_docs:
            self.generate_all_document_types()
        
        with open(output_file, 'w') as f:
            json.dump(self.summary_docs, f, indent=2, default=str)
        logger.info(f"✅ Saved {len(self.summary_docs)} summary docs to {output_file}")

# Example usage and testing
if __name__ == "__main__":
    system = HybridQTLSystem("/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv")
    
    print("Setting up hybrid system with both summary and granular documents...")
    system.setup_vector_store(use_google_embeddings=False)
    
    print(f"\n✅ Hybrid QTL System Ready!")
    print(f"📊 Layer 1: {system.vector_collection.count()} total documents in vector store")
    print(f"🗃️ Layer 2: {len(system.raw_data)} raw QTL records in DuckDB")
    
    print("\n" + "="*50)
    print("DEMO 1: Broad, conceptual query (answered by Gene Summary)")
    print("="*50)
    results = system.semantic_search(
        "Tell me about the biological function of the gene Gnai3",
        n_results=1,
        where_filter={"type": {"$eq": "gene_summary"}}
    )
    for result in results:
        print(f"Found Doc ID: {result['id']} (Distance: {result['distance']:.4f})")
        print(f"Content: {result['content']}")
    
    print("\n" + "="*50)
    print("DEMO 2: Chromosome-level query (answered by Chromosome Summary)")
    print("="*50)
    results = system.semantic_search(
        "What biological pathways are most affected by QTLs on chromosome 2?",
        n_results=1,
        where_filter={"$and": [
            {"type": {"$eq": "chromosome_summary"}},
            {"chromosome": {"$eq": "2"}}
        ]}
    )
    for result in results:
        print(f"Found Doc ID: {result['id']} (Distance: {result['distance']:.4f})")
        print(f"Content: {result['content']}")

    print("\n" + "="*50)
    print("DEMO 3: Granular, specific query (answered by QTL Peak documents)")
    print("="*50)
    results = system.semantic_search(
        "Find specific cis-QTLs for the gene Apoe",
        n_results=3,
        where_filter={"$and": [
            {"type": {"$eq": "qtl_peak"}},
            {"gene_symbol": {"$eq": "Apoe"}},
            {"cis": {"$eq": True}}
        ]}
    )
    for result in results:
        meta = result['metadata']
        print(f"Found Peak on Chr {meta['chromosome']} at {meta['position_mb']:.2f} Mb (LOD: {meta['lod_score']:.2f})")
        print(f"  Content: {result['content']}")

    print("\n" + "="*50)
    print("DEMO 4: Analytical Query (Layer 2)")
    print("="*50)
    top_genes = system.analytical_query("SELECT gene_symbol, MAX(qtl_lod) as max_lod FROM qtl_peaks GROUP BY gene_symbol ORDER BY max_lod DESC LIMIT 5")
    print("Top 5 genes by maximum LOD score:")
    print(top_genes.to_string(index=False))
    
    system.save_summary_docs("qtl_hybrid_docs.json")
    
    print(f"\n🎉 Hybrid system demonstration complete!")
    print(f"💡 Use semantic_search() for conceptual queries and analytical_query() for exact analysis.") 