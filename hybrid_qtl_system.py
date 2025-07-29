#!/usr/bin/env python3
"""
WORKING VERSION
Hybrid QTL Analysis System

Combines a vector store with a relational database for efficient QTL analysis.
NOW WITH GWAS INTEGRATION for human-mouse cross-species analysis.
"""

import pandas as pd
import json
import numpy as np
import sqlite3
import duckdb
from typing import List, Dict, Any, Optional, Tuple, Set
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
import requests
import inspect

from itertools import groupby

# Import GWAS integration
try:
    # Final version: Use the reliable, local file-based client.
    from gwas_integration import GWASCatalog as GWASCatalogClient
    GWAS_AVAILABLE = True
except ImportError:
    GWAS_AVAILABLE = False
    logging.warning("GWAS integration not available. Install required packages or check gwas_integration.py")

# Import Ensemble API integration
try:
    import requests
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logging.warning("Ensemble API integration not available. Install requests package.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache to avoid re-querying the API for the same gene
gene_cache = {}
mg = mygene.MyGeneInfo()

class EnsembleAPIClient:
    """
    Client for interacting with the Ensemble API to retrieve gene annotations,
    variant information, and cross-species data.
    """
    
    def __init__(self):
        self.base_url = "https://rest.ensembl.org"
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def get_gene_info(self, gene_symbol: str, species: str = "mus_musculus") -> Dict[str, Any]:
        """
        Retrieve detailed gene information from Ensemble API.
        
        Args:
            gene_symbol: Gene symbol to query
            species: Species identifier (default: mus_musculus for mouse)
        
        Returns:
            Dictionary containing gene information
        """
        try:
            # Use the correct Ensemble REST API endpoints
            # First try the lookup/symbol endpoint
            search_url = f"{self.base_url}/lookup/symbol/{species}/{gene_symbol}"
            logger.debug(f"🔍 Trying Ensemble lookup: {search_url}")
            response = requests.get(search_url, headers=self.headers)
            
            if response.status_code == 404:
                # Try with different case variations
                logger.info(f"⚠️ Gene {gene_symbol} not found, trying case variations...")
                for variant in [gene_symbol.upper(), gene_symbol.lower(), gene_symbol.capitalize()]:
                    if variant != gene_symbol:
                        search_url = f"{self.base_url}/lookup/symbol/{species}/{variant}"
                        logger.debug(f"🔍 Trying variant: {search_url}")
                        response = requests.get(search_url, headers=self.headers)
                        if response.status_code == 200:
                            logger.info(f"✅ Found gene with variant case: {variant}")
                            break
                
                # If still not found, try the xrefs endpoint
                if response.status_code == 404:
                    logger.info(f"🔍 Trying xrefs endpoint for {gene_symbol}...")
                    search_url = f"{self.base_url}/xrefs/symbol/{species}/{gene_symbol}"
                    response = requests.get(search_url, headers=self.headers)
            
            response.raise_for_status()
            gene_data = response.json()
            
            # Handle search results (array) vs direct lookup (object)
            if isinstance(gene_data, list) and len(gene_data) > 0:
                gene_data = gene_data[0]  # Take first result
                logger.info(f"✅ Found gene via search: {gene_data.get('display_name', gene_symbol)}")
            elif isinstance(gene_data, dict) and gene_data:
                logger.info(f"✅ Found gene via direct lookup: {gene_data.get('display_name', gene_symbol)}")
            else:
                logger.warning(f"⚠️ No gene data returned for {gene_symbol}")
                return {}
            
            # Get additional details if gene found
            if gene_data and 'id' in gene_data:
                gene_id = gene_data['id']
                
                # Get gene details using lookup/id
                details_url = f"{self.base_url}/lookup/id/{gene_id}"
                details_response = requests.get(details_url, headers=self.headers)
                if details_response.status_code == 200:
                    details = details_response.json()
                    gene_data.update(details)
                
                # Get gene description using overlap/id
                desc_url = f"{self.base_url}/overlap/id/{gene_id}"
                desc_response = requests.get(desc_url, headers=self.headers)
                if desc_response.status_code == 200:
                    desc_data = desc_response.json()
                    gene_data['description'] = desc_data
                
                return gene_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching gene info for {gene_symbol}: {e}")
            return {}
    
    def get_variants(self, gene_symbol: str, species: str = "mus_musculus") -> List[Dict[str, Any]]:
        """
        Retrieve variant information for a gene from Ensemble API.
        
        Args:
            gene_symbol: Gene symbol to query
            species: Species identifier
        
        Returns:
            List of variant information
        """
        try:
            # First get gene info to find the gene ID
            gene_info = self.get_gene_info(gene_symbol, species)
            if not gene_info or 'id' not in gene_info:
                return []
            
            gene_id = gene_info['id']
            
            # Get variants for the gene using the correct endpoint
            # Use overlap/id to get features including variants
            variants_url = f"{self.base_url}/overlap/id/{gene_id}?feature=variation"
            response = requests.get(variants_url, headers=self.headers)
            response.raise_for_status()
            
            variants = response.json()
            # Filter for variation features
            variation_features = [v for v in variants if v.get('feature_type') == 'variation']
            
            logger.info(f"✅ Found {len(variation_features)} variants for {gene_symbol}")
            return variation_features
            
        except Exception as e:
            logger.error(f"Error fetching variants for {gene_symbol}: {e}")
            return []
    
    def get_orthologs(self, gene_symbol: str, target_species: str = "homo_sapiens") -> List[Dict[str, Any]]:
        """
        Retrieve ortholog information for cross-species comparison.
        
        Args:
            gene_symbol: Gene symbol to query
            target_species: Target species for ortholog search
        
        Returns:
            List of ortholog information
        """
        try:
            # Get gene info first
            gene_info = self.get_gene_info(gene_symbol)
            if not gene_info or 'id' not in gene_info:
                return []
            
            gene_id = gene_info['id']
            
            # Get orthologs using the correct homology endpoint
            orthologs_url = f"{self.base_url}/homology/id/{gene_id}?target_species={target_species}"
            response = requests.get(orthologs_url, headers=self.headers)
            response.raise_for_status()
            
            homology_data = response.json()
            
            orthologs = []
            if 'data' in homology_data:
                for homology in homology_data['data']:
                    if 'homologies' in homology:
                        for h in homology['homologies']:
                            if h.get('type') == 'ortholog_one2one':
                                orthologs.append(h)
            
            logger.info(f"✅ Found {len(orthologs)} orthologs for {gene_symbol}")
            return orthologs
            
        except Exception as e:
            logger.error(f"Error fetching orthologs for {gene_symbol}: {e}")
            return []
    
    def get_gene_function(self, gene_symbol: str, species: str = "mus_musculus") -> Dict[str, Any]:
        """
        Retrieve comprehensive gene function information.
        
        Args:
            gene_symbol: Gene symbol to query
            species: Species identifier
        
        Returns:
            Dictionary with gene function information
        """
        try:
            gene_info = self.get_gene_info(gene_symbol, species)
            
            # Get GO terms
            if gene_info and 'id' in gene_info:
                gene_id = gene_info['id']
                go_url = f"{self.base_url}/ontology/annotation/{gene_id}"
                go_response = requests.get(go_url, headers=self.headers)
                
                go_terms = []
                if go_response.status_code == 200:
                    go_data = go_response.json()
                    go_terms = [term for term in go_data if 'term' in term]
                
                gene_info['go_terms'] = go_terms
            
            return gene_info
            
        except Exception as e:
            logger.error(f"Error fetching gene function for {gene_symbol}: {e}")
            return {}

class OrthologMatcher:
    """
    Handles downloading and parsing the MGI ortholog file to reliably map
    human gene symbols to mouse gene symbols locally.
    """
    def __init__(self):
        self.file_url = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
        self.local_path = Path("./HOM_MouseHumanSequence.rpt")
        self.human_to_mouse_map = None

    def _download_file(self, force=False):
        """Downloads the MGI ortholog file."""
        if self.local_path.exists() and not force:
            logger.info("MGI ortholog file already exists. Skipping download.")
            return

        logger.info(f"Downloading MGI ortholog file from {self.file_url}...")
        try:
            with requests.get(self.file_url, stream=True) as r:
                r.raise_for_status()
                with open(self.local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("✅ Successfully downloaded MGI ortholog file.")
        except Exception as e:
            logger.error(f"❌ Failed to download MGI file: {e}")
            raise

    def _build_map(self):
        """Builds a dictionary mapping human symbols to mouse symbols."""
        self._download_file()
        logger.info("Building human-to-mouse ortholog map...")
        
        self.human_to_mouse_map = {}
        
        try:
            with open(self.local_path, 'r') as f:
                # This function defines the data lines, skipping the header.
                def data_lines(f_handle):
                    next(f_handle) # Skip header line
                    for line in f_handle:
                        yield line

                # Group the file lines by the first column (DB Class Key).
                # This correctly handles multi-line entries for a single ortholog pair.
                for key, group in groupby(data_lines(f), key=lambda x: x.split('\t')[0]):
                    human_symbol = None
                    mouse_symbol = None
                    
                    for line in group:
                        parts = line.strip().split('\t')
                        if len(parts) < 4: continue
                        
                        organism = parts[1]
                        symbol = parts[3]
                        
                        if organism == 'human':
                            human_symbol = symbol
                        elif organism == 'mouse, laboratory':
                            mouse_symbol = symbol
                    
                    if human_symbol and mouse_symbol:
                        self.human_to_mouse_map[human_symbol.upper()] = mouse_symbol

        except Exception as e:
            logger.error(f"❌ Failed to parse MGI ortholog file: {e}")
            raise

        logger.info(f"✅ Built ortholog map with {len(self.human_to_mouse_map)} entries.")

        # --- DEBUGGING START ---
        if self.human_to_mouse_map:
            sample_keys = list(self.human_to_mouse_map.keys())[:10]
            logger.info(f"DEBUG: Sample keys from ortholog map: {sample_keys}")
        else:
            logger.warning("DEBUG: Ortholog map is empty after building.")
        # --- DEBUGGING END ---


    def get_mouse_orthologs(self, human_genes: Set[str]) -> Set[str]:
        """Converts a set of human gene symbols to mouse orthologs using the local map."""
        if self.human_to_mouse_map is None:
            self._build_map()

        # --- DEBUGGING START ---
        if human_genes:
            sample_genes_to_convert = list(human_genes)[:10]
            logger.info(f"DEBUG: Sample human genes to convert: {sample_genes_to_convert}")
        # --- DEBUGGING END ---

        mouse_orthologs = set()
        for human_gene in human_genes:
            # Check for the uppercase version of the gene symbol.
            if human_gene.upper() in self.human_to_mouse_map:
                mouse_orthologs.add(self.human_to_mouse_map[human_gene.upper()])
        
        logger.info(f"Successfully converted {len(human_genes)} human genes to {len(mouse_orthologs)} unique mouse orthologs.")
        return mouse_orthologs

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
    Layer 3: GWAS integration for human-mouse cross-species analysis
    """
    
    def __init__(self, csv_file_path: str, chroma_db_path: str = "./hybrid_chroma_db", ollama_url: str = "http://127.0.0.1:11434/api/generate", ollama_model: str = "llama3:latest", **kwargs):

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
        # self.generative_model = None  # No longer using Gemini
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        
        # Initialize GWAS client and the new Ortholog Matcher
        self.gwas_client = None
        self.ortholog_matcher = None
        if GWAS_AVAILABLE:
            # This client now manages its own data loading.
            self.gwas_client = GWASCatalogClient()
            self.ortholog_matcher = OrthologMatcher()
        
        # Initialize Ensemble API client
        self.ensemble_client = None
        if ENSEMBLE_AVAILABLE:
            self.ensemble_client = EnsembleAPIClient()
        
        # Load data and models immediately for a robust, ready-to-use instance.
        self.load_raw_data()
        self.setup_embedding_models()
        
    def setup_gwas_database(self, **kwargs):
        """
        This method now simply triggers the data loading within the GWAS client.
        """
        if self.gwas_client:
            logger.info("Initializing GWAS data handler...")
            # The client will download if necessary on the first data access.
            self.gwas_client.load_data()
            logger.info("✅ GWAS data handler is ready.")
        if self.ortholog_matcher:
            logger.info("Initializing local Ortholog Matcher...")
            # This will download the file on first run.
            self.ortholog_matcher.get_mouse_orthologs(set())
            logger.info("✅ Ortholog Matcher is ready.")

    def _fetch_all_gene_contexts_batch(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetches biological context for all unique gene symbols in the raw_data
        using an efficient batch query. This is much faster than one-by-one queries.
        """
        unique_genes = self.raw_data['gene_symbol'].dropna().unique().tolist()
        logger.info(f"Fetching biological context for {len(unique_genes)} unique genes in batches...")
        
        gene_context_map = {}
        mg = mygene.MyGeneInfo()
        
        batch_size = 1000
        for i in range(0, len(unique_genes), batch_size):
            batch = unique_genes[i:i+batch_size]
            logger.info(f"Querying mygene.info for batch {i//batch_size + 1}/{ (len(unique_genes) // batch_size) + 1 }...")
            try:
                # Use querymany for batch processing
                query_results = mg.querymany(
                    batch,
                    scopes="symbol,alias,ensembl.gene",
                    species="mouse",
                    fields="name,symbol,summary,go,pathway",
                    verbose=False
                )
                
                for result in query_results:
                    query_symbol = result.get('query')
                    if result.get('notfound'):
                        gene_context_map[query_symbol] = {}
                        continue

                    context = {
                        'name': result.get('name', 'No name available.'),
                        'summary': result.get('summary', 'No summary available.'),
                        'go_terms_bp': [term.get('term', 'N/A') for term in result.get('go', {}).get('BP', []) if isinstance(term, dict)],
                        'kegg_pathways': [p.get('name', 'N/A') for p in result.get('pathway', {}).get('kegg', []) if isinstance(p, dict)]
                    }
                    
                    # Use the actual gene symbol from the result for the key to handle aliases
                    actual_symbol = result.get('symbol', query_symbol)
                    gene_context_map[actual_symbol] = context
                    # Also map the original query term if it was an alias
                    if actual_symbol.lower() != query_symbol.lower():
                         gene_context_map[query_symbol] = context

            except Exception as e:
                logger.error(f"Error querying mygene.info for batch: {e}")
                # For genes in the failed batch, add an empty context to avoid errors later
                for gene_symbol in batch:
                    if gene_symbol not in gene_context_map:
                        gene_context_map[gene_symbol] = {}

        logger.info(f"Successfully fetched context for {len(gene_context_map)} genes.")
        return gene_context_map

    def convert_human_to_mouse_orthologs(self, human_genes: Set[str]) -> Set[str]:
        """
        Converts a set of human gene symbols to their corresponding mouse orthologs
        using the new reliable local OrthologMatcher.
        """
        if not self.ortholog_matcher:
            logger.error("Ortholog Matcher not initialized.")
            return set()
        
        # Convert human genes to mouse orthologs
        mouse_orthologs = self.ortholog_matcher.get_mouse_orthologs(human_genes)
        
        # Normalize mouse orthologs to lowercase for consistent matching
        normalized_orthologs = {ortholog.lower() for ortholog in mouse_orthologs}
        
        logger.info(f"🔄 Normalized {len(mouse_orthologs)} mouse orthologs to lowercase for matching")
        
        return normalized_orthologs

    def load_raw_data(self):
        """Load the raw QTL data into memory and DuckDB."""
        try:
            logger.info(f"Loading raw data from {self.csv_file}")
            self.raw_data = pd.read_csv(self.csv_file)
            logger.info(f"✅ Loaded {len(self.raw_data)} QTL records")
            
            # Normalize gene symbols to lowercase for consistent matching
            if 'gene_symbol' in self.raw_data.columns:
                self.raw_data['gene_symbol_normalized'] = self.raw_data['gene_symbol'].str.lower()
                logger.info("🔄 Normalized gene symbols to lowercase for consistent matching")
            
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
                CREATE INDEX idx_gene_symbol_normalized ON qtl_peaks(gene_symbol_normalized)
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
                # self.generative_model = genai.GenerativeModel('gemini-1.5-flash') # No longer using Gemini
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
        
        # NEW: Fetch all gene contexts in one efficient batch operation
        logger.info("Pre-fetching all gene contexts in batches for efficiency...")
        gene_context_map = self._fetch_all_gene_contexts_batch()
        
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
            
            # Use the pre-fetched context instead of calling the API one-by-one
            context = gene_context_map.get(gene_symbol, {})
            
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
    
    def analytical_query(self, sql_query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Layer 2: Direct SQL queries on raw data.
        Use this for exact lookups and analytics.
        """
        if not self.duck_conn:
            raise ValueError("DuckDB connection not initialized")
        
        try:
            result = self.duck_conn.execute(sql_query, params).fetchdf()
            return result
        except Exception as e:
            logger.error(f"❌ Analytical query failed: {e}")
            raise
    
    def get_gene_details(self, gene_symbol: str) -> Dict:
        """Quick helper for gene-specific queries, now including biological context."""
        # 1. Get QTL data from DuckDB using normalized gene symbol
        query = """
        SELECT * FROM qtl_peaks 
        WHERE gene_symbol_normalized = lower(?) 
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
    
    def get_enhanced_gene_details(self, gene_symbol: str) -> Dict:
        """
        Enhanced gene details including Ensemble API data for comprehensive gene information.
        
        Args:
            gene_symbol: Gene symbol to query
        
        Returns:
            Dictionary with comprehensive gene information from multiple sources
        """
        # Get basic gene details
        basic_details = self.get_gene_details(gene_symbol)
        
        # Add Ensemble API data if available
        ensemble_data = {}
        if self.ensemble_client:
            logger.info(f"🔬 Calling Ensemble API for gene: {gene_symbol}")
            try:
                # Get Ensemble gene information
                logger.info(f"📊 Fetching Ensemble gene information for {gene_symbol}...")
                ensemble_info = self.ensemble_client.get_gene_function(gene_symbol)
                if ensemble_info:
                    ensemble_data['ensemble_info'] = ensemble_info
                    logger.info(f"✅ Ensemble gene info retrieved for {gene_symbol}")
                    logger.debug(f"Ensemble gene info: {ensemble_info}")
                else:
                    logger.warning(f"⚠️ No Ensemble gene info found for {gene_symbol}")
                
                # Get variant information
                logger.info(f"🧬 Fetching variant information for {gene_symbol}...")
                variants = self.ensemble_client.get_variants(gene_symbol)
                if variants:
                    ensemble_data['variants'] = variants
                    logger.info(f"✅ Found {len(variants)} variants for {gene_symbol}")
                    logger.debug(f"Variants: {variants}")
                else:
                    logger.info(f"ℹ️ No variants found for {gene_symbol}")
                
                # Get ortholog information
                logger.info(f"🔄 Fetching ortholog information for {gene_symbol}...")
                orthologs = self.ensemble_client.get_orthologs(gene_symbol)
                if orthologs:
                    ensemble_data['orthologs'] = orthologs
                    logger.info(f"✅ Found {len(orthologs)} orthologs for {gene_symbol}")
                    logger.debug(f"Orthologs: {orthologs}")
                else:
                    logger.info(f"ℹ️ No orthologs found for {gene_symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Ensemble API error for {gene_symbol}: {e}")
        
        # Combine all data
        enhanced_details = {
            **basic_details,
            'ensemble_data': ensemble_data,
            'data_sources': ['qtl_database', 'mygene_info', 'ensemble_api']
        }
        
        return enhanced_details
    
    def get_cross_species_gene_info(self, gene_symbol: str) -> Dict:
        """
        Get comprehensive cross-species gene information including human orthologs.
        
        Args:
            gene_symbol: Mouse gene symbol
        
        Returns:
            Dictionary with cross-species gene information
        """
        # Get mouse gene details
        mouse_details = self.get_enhanced_gene_details(gene_symbol)
        
        # Get human ortholog information
        human_ortholog_info = {}
        if self.ensemble_client:
            logger.info(f"🌍 Calling Ensemble API for cross-species analysis of {gene_symbol}")
            try:
                logger.info(f"🔄 Fetching human orthologs for mouse gene {gene_symbol}...")
                orthologs = self.ensemble_client.get_orthologs(gene_symbol, "homo_sapiens")
                if orthologs:
                    human_ortholog_info = {
                        'human_orthologs': orthologs,
                        'ortholog_count': len(orthologs)
                    }
                    logger.info(f"✅ Found {len(orthologs)} human orthologs for {gene_symbol}")
                    logger.debug(f"Human orthologs: {orthologs}")
                    
                    # Get details for the first human ortholog if available
                    if orthologs and 'target' in orthologs[0]:
                        human_gene_name = orthologs[0]['target'].get('display_name', '')
                        logger.info(f"📊 Fetching details for human ortholog: {human_gene_name}")
                        human_gene_info = self.ensemble_client.get_gene_info(
                            human_gene_name, 
                            "homo_sapiens"
                        )
                        if human_gene_info:
                            human_ortholog_info['human_gene_details'] = human_gene_info
                            logger.info(f"✅ Human gene details retrieved for {human_gene_name}")
                            logger.debug(f"Human gene details: {human_gene_info}")
                        else:
                            logger.warning(f"⚠️ No human gene details found for {human_gene_name}")
                else:
                    logger.info(f"ℹ️ No human orthologs found for {gene_symbol}")
                            
            except Exception as e:
                logger.error(f"❌ Cross-species Ensemble API error for {gene_symbol}: {e}")
        
        return {
            'mouse_gene': mouse_details,
            'human_orthologs': human_ortholog_info,
            'cross_species_analysis': True
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
                    name="get_enhanced_gene_details",
                    description="Retrieves comprehensive gene information including Ensemble API data, variants, and cross-species information. Use this for detailed gene analysis requests that mention 'Ensemble', 'variants', or 'comprehensive' information.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "gene_symbol": {"type": "STRING", "description": "The official symbol of the gene, e.g., 'Apoe' or 'Gnai3'."}
                        },
                        "required": ["gene_symbol"],
                    },
                ),
                FunctionDeclaration(
                    name="get_cross_species_gene_info",
                    description="Retrieves comprehensive cross-species gene information including human orthologs and comparative analysis. Use this for queries about 'human orthologs', 'cross-species', or 'human-mouse comparison'.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "gene_symbol": {"type": "STRING", "description": "The official symbol of the mouse gene, e.g., 'Apoe' or 'Gnai3'."}
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

    def _format_tools_for_prompt(self) -> str:
        """
        Formats the tool definitions into a JSON string for the prompt.
        This version correctly handles the protobuf-based Schema objects.
        """
        if not hasattr(self, 'tools'):
            self._define_tools()
        
        tool_declarations = []
        for func_dec in self.tools.function_declarations:
            properties_dict = {}
            if func_dec.parameters and func_dec.parameters.properties:
                for key, schema_val in func_dec.parameters.properties.items():
                    # The 'type' field is an enum, so we access its name
                    prop_type_name = schema_val.type.name if hasattr(schema_val.type, 'name') else str(schema_val.type)
                    properties_dict[key] = {
                        "type": prop_type_name,
                        "description": schema_val.description
                    }
            
            # The 'type' field is an enum, so we access its name
            param_type_name = func_dec.parameters.type.name if hasattr(func_dec.parameters.type, 'name') else str(func_dec.parameters.type)
            
            params = {
                "type": param_type_name,
                "properties": properties_dict,
                "required": list(func_dec.parameters.required) if func_dec.parameters and func_dec.parameters.required else []
            }

            tool_declarations.append({
                "name": func_dec.name,
                "description": func_dec.description,
                "parameters": params
            })
        return json.dumps(tool_declarations, indent=2)

    def _call_ollama_for_tool_choice(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Calls Ollama with a specific prompt to get a tool choice in JSON format."""
        try:
            logger.info("[INFO] Using Ollama for tool selection...")
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"  # Request JSON output
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            # The response from Ollama is a string containing JSON, so we parse it.
            tool_choice_json = json.loads(data.get("response", "{}"))
            return tool_choice_json
        except Exception as e:
            logger.error(f"❌ Ollama tool choice error: {e}")
            return None

    def intelligent_router(self, query: str) -> Dict[str, Any]:
        """
        New, smarter router that uses an LLM with function calling to determine the user's intent.
        This version is specifically implemented for Ollama with JSON mode.
        """
        if not hasattr(self, 'tools'):
            self._define_tools()

        # 1. Build the prompt for the LLM
        tools_json_str = self._format_tools_for_prompt()
        prompt = f"""
You are an expert at routing user questions to the correct tool for a bioinformatics QTL database.
Your goal is to choose the single best tool to answer the user's question based on the tool descriptions.

**DATABASE CONTEXT:**
- The database contains Quantitative Trait Loci (QTL) data.
- Each record is called a "peak" or a "QTL".
- The significance of each peak is measured by its "LOD score".
- "Highest peak" means the peak with the highest LOD score.

**CRITICAL INSTRUCTIONS:**
1.  Analyze the user's query to determine the core intent.
2.  **Gene-specific vs. Global:**
    - If the query mentions a specific gene name (e.g., 'Apoe', 'Tdpoz2'), you MUST choose a gene-specific tool.
    - If the query asks for "highest" or "top" peaks in general, use `analytical_query_top_lod`.
3.  **Specific Tool Selection (for Gene-specific queries):**
    - For general information about a gene (function, summary, QTLs), use `get_gene_details`.
    - For comprehensive gene analysis with Ensemble API data (variants, detailed annotations), use `get_enhanced_gene_details`.
    - For cross-species analysis (human orthologs, comparative studies), use `get_cross_species_gene_info`.
    - For RANKED peaks (e.g., "top 5", "second highest"), use `get_top_peaks_for_gene`. You must infer the `limit` parameter. For "second highest", `limit` should be 2.
4.  **Ensemble API Integration:**
    - Use `get_enhanced_gene_details` when users ask for "Ensemble data", "variants", or "comprehensive" gene information.
    - Use `get_cross_species_gene_info` when users mention "human orthologs", "cross-species", or "human-mouse comparison".
5.  You must respond in JSON format: `{{"tool_name": "...", "arguments": {{...}} }}`.
6.  If no tool fits, default to `semantic_search`.

**Available Tools:**
{tools_json_str}

**User Query:**
"{query}"

**Your JSON response:**
"""

        # 2. Call Ollama to get the tool choice
        tool_choice = self._call_ollama_for_tool_choice(prompt)

        if not tool_choice or 'tool_name' not in tool_choice:
            logger.warning("LLM tool selection failed or returned invalid format. Falling back to semantic search.")
            tool_name = 'semantic_search'
            tool_args = {'query': query}
        else:
            tool_name = tool_choice.get('tool_name')
            tool_args = tool_choice.get('arguments', {})
            logger.info(f"🤖 LLM decided to use tool: '{tool_name}' with args: {tool_args}")

        # 3. Execute the chosen function
        try:
            if hasattr(self, tool_name):
                tool_function = getattr(self, tool_name)
                
                # Validate arguments against the function's signature
                sig = inspect.signature(tool_function)
                valid_args = {k: v for k, v in tool_args.items() if k in sig.parameters}
                
                # For semantic_search, ensure the query argument is present
                if tool_name == 'semantic_search' and 'query' not in valid_args:
                    valid_args['query'] = query

                # Call the function with validated arguments
                results_data = tool_function(**valid_args)

                # Format results for consistency
                if isinstance(results_data, pd.DataFrame):
                    results_data = results_data.to_dict('records')
                elif not isinstance(results_data, list):
                    if isinstance(results_data, dict):
                         results_data = [results_data]
                    else:
                         # Fallback for unexpected types
                         results_data = []
                
                return {
                    'detected_intent': 'tool_call',
                    'method': tool_name,
                    'arguments': tool_args,
                    'results': results_data,
                    'result_count': len(results_data)
                }
            else:
                logger.error(f"LLM wanted to call a non-existent tool: {tool_name}. Falling back.")
                raise ValueError(f"Tool '{tool_name}' not found.")

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}. Falling back to semantic search.")
            results_data = self.semantic_search(query=query, n_results=5)
            return {
                'detected_intent': 'semantic_fallback_error',
                'method': 'semantic_search',
                'error': str(e),
                'results': results_data,
                'result_count': len(results_data)
            }
    
    def _call_ollama(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the response text."""
        try:
            print("[INFO] Using Ollama (llama3:latest) for text generation.")
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300
            )

            response.raise_for_status()
            data = response.json()
            return "[Ollama] " + data.get("response", "[No response from Ollama]")
        except Exception as e:
            return f"❌ Ollama error: {str(e)}"

    def generate_response(self, query: str, search_results: Dict) -> str:
        """Generates a natural language response using retrieved context."""
        # Use Ollama for LLM generation
        intent = search_results.get('detected_intent')
        method = search_results.get('method')
        context_str = ""
        if not search_results.get('results'):
            return "I couldn't find any relevant information in the database to answer your question."
        if method == 'get_gene_details' and search_results['results']:
            gene_data = search_results['results'][0]
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
                for i, qtl in enumerate(gene_data['qtls'][:3]):
                    qtl_info = (f"  - Peak {i+1}: Located on Chromosome {qtl.get('qtl_chr', 'N/A')} "
                                f"at position {qtl.get('qtl_pos', 0.0):.2f} Mb "
                                f"with a LOD score of {qtl.get('qtl_lod', 0.0):.2f}. "
                                f"It is a {'cis-acting' if qtl.get('cis') else 'trans-acting'} regulator.")
                    context_parts.append(qtl_info)
            context_str = "\n".join(context_parts)
        else:
            if isinstance(search_results.get('results'), list):
                if search_results['results'] and isinstance(search_results['results'][0], dict) and 'content' in search_results['results'][0]:
                    context_str = "\n---\n".join([doc['content'] for doc in search_results['results']])
                else:
                    df = pd.DataFrame(search_results['results'])
                    context_str = "Based on the following data table:\n" + df.to_string(index=False)
            elif isinstance(search_results.get('results'), pd.DataFrame):
                df = search_results['results']
                context_str = "Based on the following data table:\n" + df.to_string(index=False)
        prompt = f"""
You are a specialized bioinformatics research assistant. Your task is to provide a clear, concise, and accurate answer based ONLY on the provided database context.

**CRITICAL INSTRUCTIONS:**
1.  **Be Direct:** Answer the user's question directly. Do not repeat the question or use conversational filler.
2.  **Be Concise:** Do not state the same piece of information multiple times in different ways. Avoid redundant phrases.
3.  **Synthesize Findings:**
    - If the context is a data table from an analytical query, directly state the key findings from the table.
    - If the context is from a semantic search, summarize the information accurately.
4.  **Cite Sources:** If helpful, you can briefly mention the source of the information (e.g., "from the gene summary" or "from the analytical query").
5.  **Handle Missing Information:** If the context does not contain the answer, you MUST state that clearly (e.g., "The database does not contain information about..."). Do not invent answers.

**User's Question:** "{query}"

**Retrieved Context from Database:**
---
{context_str}
---

Based *only* on the context above, provide your concise and direct answer.
"""
        response = self._call_ollama(prompt)
        return response

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

    def get_gwas_genes_for_trait_class(self, trait_class: str, p_value_threshold: float = 5e-8) -> Set[str]:
        """
        Get genes associated with a trait class from GWAS data
        
        Args:
            trait_class: One of 'glycemic', 'lipid', 'hepatic'
            p_value_threshold: P-value threshold for significance
        
        Returns:
            Set of gene symbols from GWAS
        """
        if not self.gwas_client:
            raise ValueError("GWAS client not available. Check gwas_integration.py installation.")
        
        return self.gwas_client.get_genes_for_trait_class(trait_class, p_value_threshold)
    
    def find_qtl_genes_in_gwas_set(self, gwas_genes: Set[str], qtl_filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Find which genes from a GWAS gene set have QTL data in the DO liver study
        
        Args:
            gwas_genes: Set of gene symbols from GWAS
            qtl_filters: Optional filters for QTL data (e.g., {'cis': 'TRUE', 'min_lod': 10})
        
        Returns:
            DataFrame of QTL data for genes that appear in both datasets
        """
        # Convert set to list for SQL query and normalize to lowercase
        gene_list = [g.lower() for g in gwas_genes]
        if not gene_list:
            return pd.DataFrame()
        
        logger.info(f"🔍 Searching for {len(gene_list)} GWAS genes in QTL database...")
        
        # DEBUG: Check what genes are in the QTL database
        qtl_gene_count = self.duck_conn.execute("SELECT COUNT(DISTINCT gene_symbol_normalized) FROM qtl_peaks").fetchone()[0]
        logger.info(f"📊 QTL database contains {qtl_gene_count} unique genes")
        
        # DEBUG: Show some examples from QTL database
        qtl_examples = self.duck_conn.execute("SELECT DISTINCT gene_symbol_normalized FROM qtl_peaks LIMIT 10").fetchdf()
        logger.info(f"📋 QTL database examples: {qtl_examples['gene_symbol_normalized'].tolist()}")
        
        # DEBUG: Show some examples from GWAS genes
        gwas_examples = list(gene_list)[:10]
        logger.info(f"📋 GWAS ortholog examples: {gwas_examples}")
        
        # DEBUG: Check for any exact matches
        exact_matches = set(gene_list) & set(qtl_examples['gene_symbol_normalized'].tolist())
        logger.info(f"🔍 Found {len(exact_matches)} exact matches in examples: {list(exact_matches)[:5]}")
        
        # Use a different approach to avoid parameter limits
        # Create a temporary table with the gene list
        temp_table_name = f"temp_genes_{hash(tuple(gene_list)) % 10000}"
        
        logger.info(f"🔧 Creating temporary table: {temp_table_name}")
        
        # Create temporary table with gene list
        self.duck_conn.execute(f"CREATE TEMP TABLE {temp_table_name} (gene_symbol TEXT)")
        
        # Insert genes in batches to avoid parameter limits
        batch_size = 1000
        total_inserted = 0
        for i in range(0, len(gene_list), batch_size):
            batch = gene_list[i:i+batch_size]
            placeholders = ','.join(['(?)' for _ in batch])
            insert_query = f"INSERT INTO {temp_table_name} VALUES {placeholders}"
            self.duck_conn.execute(insert_query, batch)
            total_inserted += len(batch)
            logger.info(f"📥 Inserted batch {i//batch_size + 1}: {len(batch)} genes (total: {total_inserted})")
        
        # Verify temporary table contents
        temp_count = self.duck_conn.execute(f"SELECT COUNT(*) FROM {temp_table_name}").fetchone()[0]
        logger.info(f"📊 Temporary table contains {temp_count} genes")
        
        # Show some examples from temp table
        temp_examples = self.duck_conn.execute(f"SELECT gene_symbol FROM {temp_table_name} LIMIT 5").fetchdf()
        logger.info(f"📋 Temp table examples: {temp_examples['gene_symbol'].tolist()}")
        
        # DEBUG: Test the JOIN with a simple query
        test_query = f"""
            SELECT COUNT(*) as match_count 
            FROM qtl_peaks q 
            JOIN {temp_table_name} t ON q.gene_symbol_normalized = t.gene_symbol
        """
        test_result = self.duck_conn.execute(test_query).fetchone()[0]
        logger.info(f"🔍 Test JOIN found {test_result} matches without filters")
        
        # DEBUG: Check the actual values in the QTL database for the filters
        cis_values = self.duck_conn.execute("SELECT DISTINCT cis FROM qtl_peaks LIMIT 10").fetchdf()
        logger.info(f"📊 CIS values in QTL database: {cis_values['cis'].tolist()}")
        
        lod_stats = self.duck_conn.execute("SELECT MIN(qtl_lod) as min_lod, MAX(qtl_lod) as max_lod, AVG(qtl_lod) as avg_lod FROM qtl_peaks").fetchone()
        logger.info(f"📊 LOD stats: min={lod_stats[0]}, max={lod_stats[1]}, avg={lod_stats[2]:.2f}")
        
        # DEBUG: Test with the actual filter values
        test_filtered_query = f"""
            SELECT COUNT(*) as match_count 
            FROM qtl_peaks q 
            JOIN {temp_table_name} t ON q.gene_symbol_normalized = t.gene_symbol
            WHERE q.cis = 'TRUE' AND q.qtl_lod >= 5.0
        """
        test_filtered_result = self.duck_conn.execute(test_filtered_query).fetchone()[0]
        logger.info(f"🔍 Test JOIN with filters found {test_filtered_result} matches")
        
        # DEBUG: Check if any genes from temp table exist in qtl_peaks
        test_genes = temp_examples['gene_symbol'].tolist()
        placeholders = ','.join(['?' for _ in test_genes])
        check_query = f"SELECT gene_symbol_normalized FROM qtl_peaks WHERE gene_symbol_normalized IN ({placeholders})"
        existing_genes = self.duck_conn.execute(check_query, test_genes).fetchdf()
        logger.info(f"🔍 Found {len(existing_genes)} test genes in QTL database: {existing_genes['gene_symbol_normalized'].tolist()}")
        
        # Build the main query using JOIN instead of IN clause
        base_query = f"""
            SELECT q.* FROM qtl_peaks q
            JOIN {temp_table_name} t ON q.gene_symbol_normalized = t.gene_symbol
        """
        
        logger.info(f"🔍 Executing query: {base_query}")
        
        # Add filters if provided
        filter_conditions = []
        filter_params = []
        
        if qtl_filters:
            if 'cis' in qtl_filters:
                filter_conditions.append("q.cis = ?")
                # Convert string to boolean if needed
                cis_value = qtl_filters['cis']
                if isinstance(cis_value, str):
                    cis_value = cis_value.upper() == 'TRUE'
                filter_params.append(cis_value)
            
            if 'min_lod' in qtl_filters:
                filter_conditions.append("q.qtl_lod >= ?")
                # Ensure it's a float
                min_lod = float(qtl_filters['min_lod'])
                filter_params.append(min_lod)
            
            if 'max_lod' in qtl_filters:
                filter_conditions.append("q.qtl_lod <= ?")
                # Ensure it's a float
                max_lod = float(qtl_filters['max_lod'])
                filter_params.append(max_lod)
            
            if 'chromosome' in qtl_filters:
                filter_conditions.append("q.qtl_chr = ?")
                filter_params.append(qtl_filters['chromosome'])
        
        if filter_conditions:
            base_query += " WHERE " + " AND ".join(filter_conditions)
            logger.info(f"🔧 Added filters: {filter_conditions}")
        
        base_query += " ORDER BY q.qtl_lod DESC"
        
        # Execute query
        logger.info(f"🚀 Executing final query with {len(filter_params)} parameters")
        result_df = self.duck_conn.execute(base_query, filter_params).fetchdf()
        
        # Clean up temporary table
        self.duck_conn.execute(f"DROP TABLE {temp_table_name}")
        logger.info(f"🧹 Cleaned up temporary table {temp_table_name}")
        
        logger.info(f"✅ Found {len(result_df)} QTL records for {result_df['gene_symbol'].nunique()} GWAS genes")
        
        return result_df
    
    def get_diet_dependent_cis_eqtl_genes(self, gwas_genes: Set[str]) -> pd.DataFrame:
        """
        Among GWAS genes, find those with diet-dependent cis-eQTL in DO liver study
        
        This requires the QTL dataset to have diet interaction terms.
        For now, this is a placeholder that filters for cis-QTLs.
        
        Args:
            gwas_genes: Set of gene symbols from GWAS
        
        Returns:
            DataFrame of cis-QTL data for GWAS genes
        """
        # For now, filter for cis-QTLs (diet-dependence would require additional data columns)
        qtl_filters = {'cis': True, 'min_lod': 5.0}  # Use boolean True instead of string
        
        cis_qtl_df = self.find_qtl_genes_in_gwas_set(gwas_genes, qtl_filters)
        
        # Add a note that this is a simplified version
        if len(cis_qtl_df) > 0:
            logger.info(f"Found {len(cis_qtl_df)} cis-QTL records for {cis_qtl_df['gene_symbol'].nunique()} GWAS genes")
            logger.warning("Diet-dependence analysis requires additional data columns not present in current dataset")
        
        return cis_qtl_df
    
    def get_diet_dependent_trans_eqtl_genes(self, gwas_genes: Set[str]) -> pd.DataFrame:
        """
        Among GWAS genes, find those with diet-dependent trans-eQTL in DO liver study
        
        Args:
            gwas_genes: Set of gene symbols from GWAS
        
        Returns:
            DataFrame of trans-QTL data for GWAS genes
        """
        # For now, filter for trans-QTLs
        qtl_filters = {'cis': False, 'min_lod': 5.0}  # Use boolean False instead of string
        
        trans_qtl_df = self.find_qtl_genes_in_gwas_set(gwas_genes, qtl_filters)
        
        if len(trans_qtl_df) > 0:
            logger.info(f"Found {len(trans_qtl_df)} trans-QTL records for {trans_qtl_df['gene_symbol'].nunique()} GWAS genes")
            logger.warning("Diet-dependence analysis requires additional data columns not present in current dataset")
        
        return trans_qtl_df
    
    def comprehensive_gwas_qtl_analysis(self, trait_class: str) -> Dict[str, Any]:
        """
        Comprehensive analysis following the research questions 1-4:
        1. Get GWAS genes for trait class
        2. Find those with cis-eQTL in DO liver study  
        3. Find those with trans-eQTL in DO liver study
        4. Analyze potential hub genes
        
        Args:
            trait_class: One of 'glycemic', 'lipid', 'hepatic'
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Starting comprehensive GWAS-QTL analysis for {trait_class}")
        
        results = {
            'trait_class': trait_class,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Get HUMAN GWAS genes for trait class
            logger.info("Step 1: Getting human GWAS genes for trait class...")
            human_gwas_genes = self.get_gwas_genes_for_trait_class(trait_class)
            logger.info(f"Found {len(human_gwas_genes)} human GWAS genes for {trait_class}")
            
            if not human_gwas_genes:
                results['error'] = "No GWAS genes found for this trait class"
                return results

            # Step 1b: Convert human genes to mouse orthologs
            logger.info("Step 1b: Converting human genes to mouse orthologs...")
            gwas_genes = self.convert_human_to_mouse_orthologs(human_gwas_genes)
            
            # DEBUG: Check ortholog mapping quality
            debug_info = self.debug_ortholog_mapping(human_gwas_genes, sample_size=20)
            
            results['gwas_genes'] = {
                'human_gene_count': len(human_gwas_genes),
                'mouse_ortholog_count': len(gwas_genes),
                'genes': list(gwas_genes),
                'human_genes': list(human_gwas_genes)
            }
            logger.info(f"Found {len(gwas_genes)} mouse orthologs to use for QTL analysis.")

            if not gwas_genes:
                results['error'] = "No mouse orthologs found for the identified human GWAS genes."
                return results
            
            # Step 2: Find those with cis-eQTL in the mouse study
            logger.info("Step 2: Finding cis-eQTL genes in mouse data...")
            cis_qtl_df = self.get_diet_dependent_cis_eqtl_genes(gwas_genes)
            cis_genes = set(cis_qtl_df['gene_symbol'].unique()) if len(cis_qtl_df) > 0 else set()
            
            # Debug: Show some examples of matched genes
            if len(cis_genes) > 0:
                example_matches = list(cis_genes)[:5]
                logger.info(f"🔍 Example cis-QTL matches: {example_matches}")
            
            results['cis_eqtl_genes'] = {
                'count': len(cis_genes),
                'genes': list(cis_genes),
                'qtl_peaks': len(cis_qtl_df)
            }
            
            # Step 3: Find those with trans-eQTL in the mouse study
            logger.info("Step 3: Finding trans-eQTL genes in mouse data...")
            trans_qtl_df = self.get_diet_dependent_trans_eqtl_genes(gwas_genes)
            trans_genes = set(trans_qtl_df['gene_symbol'].unique()) if len(trans_qtl_df) > 0 else set()
            
            # Debug: Show some examples of matched genes
            if len(trans_genes) > 0:
                example_matches = list(trans_genes)[:5]
                logger.info(f"🔍 Example trans-QTL matches: {example_matches}")
            
            results['trans_eqtl_genes'] = {
                'count': len(trans_genes),
                'genes': list(trans_genes),
                'qtl_peaks': len(trans_qtl_df)
            }
            
            # Step 4: Identify potential hub genes (genes with both cis and trans QTLs)
            logger.info("Step 4: Identifying potential hub genes...")
            potential_hubs = cis_genes & trans_genes
            
            results['potential_hub_genes'] = {
                'count': len(potential_hubs),
                'genes': list(potential_hubs)
            }
            
            # Additional analysis: overlap statistics
            results['overlap_analysis'] = {
                'gwas_with_any_qtl': len(cis_genes | trans_genes),
                'gwas_with_cis_only': len(cis_genes - trans_genes),
                'gwas_with_trans_only': len(trans_genes - cis_genes),
                'gwas_with_both': len(potential_hubs),
                'gwas_without_qtl': len(gwas_genes - cis_genes - trans_genes)
            }
            
            logger.info(f"Comprehensive analysis complete for {trait_class}")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            results['error'] = str(e)
            return results
    
    def export_results_to_csv(self, analysis_results: Dict[str, Any], output_dir: str = "./gwas_qtl_results"):
        """
        Export comprehensive analysis results to CSV files
        
        Args:
            analysis_results: Results from comprehensive_gwas_qtl_analysis
            output_dir: Directory to save CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        trait_class = analysis_results['trait_class']
        
        # Export GWAS genes
        if 'gwas_genes' in analysis_results:
            gwas_df = pd.DataFrame({
                'gene_symbol': analysis_results['gwas_genes']['genes'],
                'human_gene_symbol': analysis_results['gwas_genes'].get('human_genes', [None]*len(analysis_results['gwas_genes']['genes'])),
                'source': 'GWAS_Mouse_Ortholog',
                'trait_class': trait_class
            })
            gwas_df.to_csv(output_path / f"{trait_class}_gwas_genes_mouse_orthologs.csv", index=False)
        
        # Export cis-eQTL genes
        if 'cis_eqtl_genes' in analysis_results and analysis_results['cis_eqtl_genes']['genes']:
            cis_df = pd.DataFrame({
                'gene_symbol': analysis_results['cis_eqtl_genes']['genes'],
                'qtl_type': 'cis',
                'trait_class': trait_class
            })
            cis_df.to_csv(output_path / f"{trait_class}_cis_eqtl_genes.csv", index=False)
        
        # Export trans-eQTL genes  
        if 'trans_eqtl_genes' in analysis_results and analysis_results['trans_eqtl_genes']['genes']:
            trans_df = pd.DataFrame({
                'gene_symbol': analysis_results['trans_eqtl_genes']['genes'],
                'qtl_type': 'trans',
                'trait_class': trait_class
            })
            trans_df.to_csv(output_path / f"{trait_class}_trans_eqtl_genes.csv", index=False)
        
        # Export potential hub genes
        if 'potential_hub_genes' in analysis_results and analysis_results['potential_hub_genes']['genes']:
            hub_df = pd.DataFrame({
                'gene_symbol': analysis_results['potential_hub_genes']['genes'],
                'qtl_type': 'hub (cis + trans)',
                'trait_class': trait_class
            })
            hub_df.to_csv(output_path / f"{trait_class}_hub_genes.csv", index=False)
        
        # Export summary statistics
        summary_data = []
        for category, data in analysis_results.items():
            if isinstance(data, dict) and 'count' in data:
                summary_data.append({
                    'category': category,
                    'count': data['count'],
                    'trait_class': trait_class
                })
            elif isinstance(data, dict) and 'human_gene_count' in data:
                 summary_data.append({
                    'category': 'human_gwas_genes',
                    'count': data['human_gene_count'],
                    'trait_class': trait_class
                })
                 summary_data.append({
                    'category': 'mouse_orthologs',
                    'count': data['mouse_ortholog_count'],
                    'trait_class': trait_class
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_path / f"{trait_class}_analysis_summary.csv", index=False)
        
        logger.info(f"Results exported to {output_path}")

    def test_ensemble_connection(self) -> bool:
        """
        Test the Ensemble API connection and get available gene examples.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("🔬 Testing Ensemble API connection...")
            
            # First test basic API connectivity
            info_url = f"{self.base_url}/info/species"
            logger.info(f"🔍 Testing basic API connectivity: {info_url}")
            response = requests.get(info_url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"❌ Basic API connectivity failed: {response.status_code}")
                return False
            
            logger.info("✅ Basic API connectivity successful")
            
            # Test with a known mouse gene - try different approaches
            test_genes = ["Apoe", "Gnai3", "Actb", "Gapdh"]
            
            for gene in test_genes:
                logger.info(f"🔍 Testing gene: {gene}")
                
                # Try different endpoints using correct API
                endpoints = [
                    f"{self.base_url}/lookup/symbol/mus_musculus/{gene}",
                    f"{self.base_url}/xrefs/symbol/mus_musculus/{gene}"
                ]
                
                for endpoint in endpoints:
                    logger.debug(f"🔍 Trying endpoint: {endpoint}")
                    response = requests.get(endpoint, headers=self.headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            logger.info(f"✅ Successfully found {gene} via {endpoint}")
                            logger.debug(f"Gene info: {data}")
                            return True
                
                logger.warning(f"⚠️ Could not find {gene} in any Ensemble endpoint")
            
            logger.error("❌ No test genes found in Ensemble API")
            return False
            
        except Exception as e:
            logger.error(f"❌ Ensemble API connection test failed: {e}")
            return False
    
    def get_available_ensemble_genes(self, limit: int = 10) -> List[str]:
        """
        Get a list of available genes in Ensemble for testing.
        
        Args:
            limit: Maximum number of genes to return
        
        Returns:
            List of gene symbols available in Ensemble
        """
        try:
            logger.info("🔍 Getting available genes from Ensemble...")
            
            # Try to get genes from a known region
            search_url = f"{self.base_url}/lookup/mus_musculus/region/1:1-1000000"
            response = requests.get(search_url, headers=self.headers)
            
            if response.status_code == 200:
                genes = response.json()
                gene_symbols = []
                for gene in genes:
                    if isinstance(gene, dict) and 'display_name' in gene:
                        gene_symbols.append(gene['display_name'])
                
                logger.info(f"✅ Found {len(gene_symbols)} genes in Ensemble")
                return gene_symbols[:limit]
            else:
                logger.warning(f"⚠️ Could not get gene list from Ensemble: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting available genes: {e}")
            return []

    def debug_ortholog_mapping(self, human_genes: Set[str], sample_size: int = 10) -> Dict[str, Any]:
        """
        Debug the ortholog mapping process to see what's happening.
        
        Args:
            human_genes: Set of human gene symbols
            sample_size: Number of genes to sample for debugging
        
        Returns:
            Dictionary with debugging information
        """
        logger.info("🔍 Debugging ortholog mapping process...")
        
        # Sample some human genes
        sample_human = list(human_genes)[:sample_size]
        logger.info(f"📋 Sample human genes: {sample_human}")
        
        # Get mouse orthologs for these genes
        mouse_orthologs = self.ortholog_matcher.get_mouse_orthologs(set(sample_human))
        logger.info(f"🔄 Mouse orthologs found: {list(mouse_orthologs)}")
        
        # Check which of these are in the QTL database
        if mouse_orthologs:
            ortholog_list = [ortholog.lower() for ortholog in mouse_orthologs]
            placeholders = ','.join(['?' for _ in ortholog_list])
            query = f"SELECT DISTINCT gene_symbol_normalized FROM qtl_peaks WHERE gene_symbol_normalized IN ({placeholders})"
            qtl_matches = self.duck_conn.execute(query, ortholog_list).fetchdf()
            
            logger.info(f"📊 Found {len(qtl_matches)} orthologs in QTL database")
            if len(qtl_matches) > 0:
                logger.info(f"📋 QTL matches: {qtl_matches['gene_symbol_normalized'].tolist()}")
        
        return {
            'sample_human_genes': sample_human,
            'mouse_orthologs': list(mouse_orthologs),
            'qtl_matches': qtl_matches['gene_symbol_normalized'].tolist() if len(qtl_matches) > 0 else []
        }

    def test_gene_matching(self):
        """
        Test gene matching between orthologs and QTL database to identify the issue.
        """
        logger.info("🔍 Testing gene matching between orthologs and QTL database...")
        
        # Get some genes from QTL database
        qtl_genes = self.duck_conn.execute("SELECT DISTINCT gene_symbol, gene_symbol_normalized FROM qtl_peaks LIMIT 20").fetchdf()
        logger.info(f"📋 QTL database genes (original): {qtl_genes['gene_symbol'].tolist()}")
        logger.info(f"📋 QTL database genes (normalized): {qtl_genes['gene_symbol_normalized'].tolist()}")
        
        # Test with some common genes that should exist
        test_genes = ['apoe', 'gnai3', 'actb', 'gapdh', 'ins', 'glu', 'ldl']
        logger.info(f"🔍 Testing common genes: {test_genes}")
        
        for gene in test_genes:
            # Check if gene exists in QTL database
            result = self.duck_conn.execute("SELECT COUNT(*) FROM qtl_peaks WHERE gene_symbol_normalized = ?", [gene]).fetchone()[0]
            logger.info(f"📊 Gene '{gene}' found {result} times in QTL database")
        
        # Test ortholog mapping with some common human genes
        test_human_genes = ['APOE', 'GNAI3', 'ACTB', 'GAPDH', 'INS', 'GLU', 'LDL']
        logger.info(f"🔍 Testing human genes: {test_human_genes}")
        
        mouse_orthologs = self.ortholog_matcher.get_mouse_orthologs(set(test_human_genes))
        logger.info(f"🔄 Mouse orthologs found: {list(mouse_orthologs)}")
        
        # Check which orthologs are in QTL database
        for ortholog in mouse_orthologs:
            ortholog_lower = ortholog.lower()
            result = self.duck_conn.execute("SELECT COUNT(*) FROM qtl_peaks WHERE gene_symbol_normalized = ?", [ortholog_lower]).fetchone()[0]
            logger.info(f"📊 Ortholog '{ortholog}' (normalized: '{ortholog_lower}') found {result} times in QTL database")
        
        return {
            'qtl_genes': qtl_genes.to_dict('records'),
            'test_genes': test_genes,
            'mouse_orthologs': list(mouse_orthologs)
        }

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