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
import os

from itertools import groupby

# Import GWAS integration
try:
    # Final version: Use the reliable, local file-based client.
    from gwas_integration import GWASCatalog as GWASCatalogClient
    GWAS_AVAILABLE = True
except ImportError:
    GWAS_AVAILABLE = False
    logging.warning("GWAS integration not available. Install required packages or check gwas_integration.py")

# Import Ensembl API integration
try:
    import requests
    ENSEMBL_AVAILABLE = True
except ImportError:
    ENSEMBL_AVAILABLE = False
    logging.warning("Ensembl API integration not available. Install requests package.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache to avoid re-querying the API for the same gene
gene_cache = {}
mg = mygene.MyGeneInfo()

class EnsemblAPIClient:
    """
    Client for interacting with the Ensembl API to retrieve gene annotations,
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
        Retrieve detailed gene information from Ensembl API.
        
        Args:
            gene_symbol: Gene symbol to query
            species: Species identifier (default: mus_musculus for mouse)
        
        Returns:
            Dictionary containing gene information
        """
        try:
            # Use the correct Ensembl REST API endpoints
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
        Retrieve variant information for a gene from Ensembl API.
        
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
    def __init__(self, local_path: Optional[str] = None):
        self.file_url = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
        # Resolve the MGI file robustly with multiple fallbacks
        script_dir = Path(__file__).parent.resolve()
        env_path = os.getenv("MGI_ORTHOLOG_PATH")
        candidate_paths = []
        if local_path:
            candidate_paths.append(Path(local_path).expanduser())
        if env_path:
            candidate_paths.append(Path(env_path).expanduser())
        candidate_paths.extend([
            script_dir / "HOM_MouseHumanSequence.rpt",
            Path.cwd() / "HOM_MouseHumanSequence.rpt",
        ])
        # Pick the first existing path; otherwise default to script_dir target
        chosen = None
        for p in candidate_paths:
            try:
                if p.exists():
                    chosen = p
                    break
            except Exception:
                continue
        if chosen is None:
            chosen = script_dir / "HOM_MouseHumanSequence.rpt"
        self.local_path = chosen.resolve()
        logger.info(f"Using MGI ortholog file path: {self.local_path}")
        self.human_to_mouse_map = None

    def _download_file(self, force=False):
        """Downloads the MGI ortholog file."""
        if self.local_path.exists() and not force:
            logger.info("MGI ortholog file already exists. Skipping download.")
            return

        logger.info(f"Downloading MGI ortholog file from {self.file_url}...")
        try:
            # Ensure directory exists
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
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
            logger.info(f"Opening MGI ortholog file at: {self.local_path}")
            with open(self.local_path, 'r', encoding='utf-8', errors='ignore') as f:
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

        except FileNotFoundError as e:
            logger.error(f"❌ Ortholog file not found at {self.local_path}: {e}")
            # Attempt a final fallback to CWD
            fallback_path = (Path.cwd() / "HOM_MouseHumanSequence.rpt").resolve()
            if fallback_path.exists():
                logger.info(f"Retrying with fallback path: {fallback_path}")
                self.local_path = fallback_path
                return self._build_map()
            raise
        except Exception as e:
            logger.error(f"❌ Failed to parse MGI ortholog file at {self.local_path}: {e}")
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
    Raw QTL Data Analysis System:
    Layer 1: Vector store with ONLY original, unaltered QTL source data
    Layer 2: Relational store with raw rows for exact queries/analytics
    Layer 3: GWAS integration for human-mouse cross-species analysis
    
    This system stores and searches ONLY the original experimental data
    without any summaries, interpretations, or external enrichments.
    """
    
    def __init__(self, csv_file_path: str, chroma_db_path: str = "./hybrid_chroma_db", ollama_url: str = "http://127.0.0.1:11434/api/generate", ollama_model: str = "qwen3:8b", ollama_tool_model: str = "qwen3:8b", ortholog_path: Optional[str] = None, **kwargs):

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
        self.ollama_tool_model = ollama_tool_model
        self.ollama_session = requests.Session()
        
        # Initialize GWAS client and the new Ortholog Matcher
        self.gwas_client = None
        self.ortholog_matcher = None
        if GWAS_AVAILABLE:
            # This client now manages its own data loading.
            self.gwas_client = GWASCatalogClient()
            self.ortholog_matcher = OrthologMatcher(local_path=ortholog_path)
        
        # Initialize Ensembl API client
        self.ensembl_client = None
        if ENSEMBL_AVAILABLE:
            self.ensembl_client = EnsemblAPIClient()
        
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
        Generate documents containing ONLY original, unaltered QTL source data.
        No summaries, interpretations, or external enrichments are included.
        """
        all_docs = []
        
        # ONLY PER-PEAK RAW DATA DOCUMENTS (no gene summaries)
        logger.info(f"Generating raw QTL peak documents for all {len(self.raw_data)} records...")
        
        for index, row in self.raw_data.iterrows():
            # Create document ID using original data identifiers
            doc_id = f"peak_{index}_{row['gene_symbol']}_{row['qtl_chr']}_{row['qtl_pos']}"
            
            # Store ONLY the original raw data values as structured content
            # No narrative text, no interpretations, no external enrichments
            raw_data_content = {
                'gene_symbol': str(row['gene_symbol']) if pd.notna(row['gene_symbol']) else 'N/A',
                'gene_id': str(row.get('gene_id', 'N/A')) if pd.notna(row.get('gene_id')) else 'N/A',
                'qtl_chr': str(row['qtl_chr']) if pd.notna(row['qtl_chr']) else 'N/A',
                'qtl_pos': float(row['qtl_pos']) if pd.notna(row['qtl_pos']) else 0.0,
                'qtl_lod': float(row['qtl_lod']) if pd.notna(row['qtl_lod']) else 0.0,
                'cis': str(row.get('cis', 'N/A')) if pd.notna(row.get('cis')) else 'N/A',
                'phenotype': str(row.get('phenotype', 'N/A')) if pd.notna(row.get('phenotype')) else 'N/A'
            }
            
            # Convert to JSON string for vector storage (preserves exact original values)
            content = json.dumps(raw_data_content, sort_keys=True)
            
            all_docs.append({
                'id': doc_id,
                'content': content,
                'metadata': {
                    'type': 'raw_qtl_peak',
                    'gene_symbol': raw_data_content['gene_symbol'],
                    'gene_id': raw_data_content['gene_id'],
                    'chromosome': raw_data_content['qtl_chr'],
                    'position_mb': raw_data_content['qtl_pos'],
                    'lod_score': raw_data_content['qtl_lod'],
                    'cis': raw_data_content['cis'],
                    'phenotype': raw_data_content['phenotype'],
                    'source_row_index': int(index)
                }
            })
            
        self.summary_docs = all_docs
        logger.info(f"✅ Generated {len(all_docs)} raw QTL data documents (no summaries or enrichments)")
        return all_docs
    
    def setup_vector_store(self, use_google_embeddings: bool = True):
        """Setup ChromaDB vector store with ONLY original QTL source data."""
        try:
            # Use local on-disk Chroma client (no server / host required)
            try:
                self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            except Exception as e:
                msg = str(e)
                if "http-only client mode" in msg or "chroma_api_impl" in msg:
                    raise RuntimeError(
                        "Chroma is in HTTP-only client mode. For in-process PersistentClient, uninstall 'chromadb-client' "
                        "and install the full 'chromadb' package.\n"
                        "Run: python3 -m pip uninstall -y chromadb-client && python3 -m pip install -U chromadb"
                    ) from e
                raise
            
            collection_name = "qtl_raw_data_store"
            self.vector_collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Vector store containing ONLY original, unaltered QTL source data"}
            )
            logger.info(f"✅ Ensured collection '{collection_name}' exists.")

            if self.vector_collection.count() > 0:
                logger.info(f"Collection already contains {self.vector_collection.count()} raw QTL documents. Skipping population.")
                return

            logger.info("Collection is empty. Generating raw QTL data documents and populating vector store...")
            self.generate_all_document_types()
            
            logger.info(f"Adding {len(self.summary_docs)} raw QTL data documents to vector store...")
            
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
        Layer 1: Semantic search on raw QTL data documents.
        Can be filtered by metadata fields (e.g., 'gene_symbol', 'chromosome', 'cis').
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
        """
        Retrieves a comprehensive summary for a single, named gene, including its biological
        function from MyGene.info, all associated QTL peak data from the database, and
        detailed data from the Ensembl API (including variants and orthologs).
        """
        # 1. Get QTL data from DuckDB and biological context from mygene.info
        query = "SELECT * FROM qtl_peaks WHERE gene_symbol_normalized = lower(?) ORDER BY qtl_lod DESC"
        qtl_result_df = self.duck_conn.execute(query, [gene_symbol]).fetchdf()
        biological_context = fetch_gene_context(gene_symbol)

        basic_details = {
            'gene_symbol': gene_symbol,
            'biological_summary': biological_context.get('summary', 'No summary available.'),
            'go_terms': biological_context.get('go_terms_bp', []),
            'kegg_pathways': biological_context.get('kegg_pathways', []),
            'qtl_count': len(qtl_result_df),
            'qtls': qtl_result_df.to_dict('records') if len(qtl_result_df) > 0 else []
        }
        
        # 2. Add Ensembl API data if available
        ensembl_data = {}
        if self.ensembl_client:
            logger.info(f"🔬 Calling Ensembl API for comprehensive data on gene: {gene_symbol}")
            try:
                # Get Ensembl gene function, variants, and orthologs
                ensembl_info = self.ensembl_client.get_gene_function(gene_symbol)
                if ensembl_info:
                    ensembl_data['ensembl_info'] = ensembl_info
                
                variants = self.ensembl_client.get_variants(gene_symbol)
                if variants:
                    ensembl_data['variants'] = variants

                orthologs = self.ensembl_client.get_orthologs(gene_symbol)
                if orthologs:
                    ensembl_data['orthologs'] = orthologs
                    
            except Exception as e:
                logger.error(f"❌ Ensembl API error for {gene_symbol}: {e}")
        
        # 3. Combine all data
        return {
            **basic_details,
            'ensembl_data': ensembl_data,
            'data_sources': ['qtl_database', 'mygene_info', 'ensembl_api']
        }
    
    def get_cross_species_gene_info(self, gene_symbol: str) -> Dict:
        """
        Get comprehensive cross-species gene information including human orthologs.
        """
        # Get comprehensive mouse gene details (which now includes orthologs)
        mouse_details = self.get_gene_details(gene_symbol)
        
        # Extract human ortholog information from the details
        human_ortholog_info = {}
        # Get orthologs from the correct species (homo_sapiens)
        orthologs = []
        if self.ensembl_client:
            orthologs = self.ensembl_client.get_orthologs(gene_symbol, "homo_sapiens")

        if self.ensembl_client and orthologs:
            logger.info(f"🌍 Processing human orthologs for cross-species analysis of {gene_symbol}")
            try:
                human_ortholog_info = {
                    'human_orthologs': orthologs,
                    'ortholog_count': len(orthologs)
                }
                
                # Get details for the first human ortholog if available
                if 'target' in orthologs[0]:
                    human_gene_name = orthologs[0]['target'].get('display_name', '')
                    if human_gene_name:
                        logger.info(f"📊 Fetching details for human ortholog: {human_gene_name}")
                        human_gene_info = self.ensembl_client.get_gene_info(
                            human_gene_name, 
                            "homo_sapiens"
                        )
                        if human_gene_info:
                            human_ortholog_info['human_gene_details'] = human_gene_info
                            
            except Exception as e:
                logger.error(f"❌ Cross-species analysis error for {gene_symbol}: {e}")
        
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
    
    def find_gene_by_lod(self, lod_score: float) -> pd.DataFrame:
        """Finds the gene, position, and chromosome for a specific LOD score."""
        # Use a small tolerance for float comparison since LOD scores are floats
        tolerance = 0.0001
        sql = """
        SELECT gene_symbol, qtl_chr, qtl_pos, qtl_lod 
        FROM qtl_peaks 
        WHERE qtl_lod BETWEEN ? AND ?
        LIMIT 1
        """
        # DuckDB expects a list of parameters, not a dictionary
        params = [lod_score - tolerance, lod_score + tolerance]
        return self.analytical_query(sql, params)

    def _define_tools(self):
        """Defines the function calling tools for the LLM."""
        self.tools = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="get_gene_details",
                    description="Retrieves a comprehensive summary for a single, named gene, including its biological function from MyGene.info, all associated QTL peak data from the database, and detailed data from the Ensembl API (including variants, and orthologs). This is the primary tool for any question about a specific gene.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "gene_symbol": {"type": "STRING", "description": "The official symbol of the gene, e.g., 'Apoe' or 'Gnai3'."}
                        },
                        "required": ["gene_symbol"],
                    },
                ),
                FunctionDeclaration(
                    name="find_gene_by_lod",
                    description="Finds the gene/peak associated with a precise LOD score. Use this when the user provides a specific LOD score and asks for the corresponding gene.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "lod_score": {"type": "NUMBER", "description": "The precise LOD score to search for, e.g., 608.58098"}
                        },
                        "required": ["lod_score"],
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
                    name="query_ensembl_api",
                    description="Query Ensembl's REST API for genomic data (gene info, orthologs, variants, gene function). Use this for questions about gene annotations, variants, orthologs, or detailed genomic information from Ensembl.",
                    parameters={
                        "type": "OBJECT",
                        "properties": {
                            "gene_symbol": {"type": "STRING", "description": "Gene symbol to query (e.g., 'Apoe', 'Gnai3')"},
                            "query_type": {"type": "STRING", "description": "Type of query: 'gene_info', 'variants', 'orthologs', 'gene_function'"},
                            "species": {"type": "STRING", "description": "Species identifier (default: mus_musculus, can be homo_sapiens for human)"}
                        },
                        "required": ["gene_symbol", "query_type"],
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
        if limit is None:
            limit = 5  # Default to 5 if the LLM provides no limit
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

    def _call_ollama_for_tool_choice(self, user_prompt: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Calls Ollama with a specific prompt to get a tool choice in JSON format."""
        data = {}  # Initialize data to ensure it's available in except blocks
        try:
            logger.info(f"[INFO] Using Ollama ({self.ollama_tool_model}) for tool selection...")
            response = self.ollama_session.post(
                self.ollama_url,
                json={
                    "model": self.ollama_tool_model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "format": "json"  # Request JSON output
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            # The response from Ollama is a string containing JSON, so we parse it.
            # More robust parsing
            ollama_response_str = data.get("response")
            if not ollama_response_str or not ollama_response_str.strip():
                logger.warning("Ollama returned an empty or whitespace-only response for tool choice.")
                return None

            tool_choice_json = json.loads(ollama_response_str)
            return tool_choice_json
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ollama tool choice JSON decoding error: {e}")
            logger.error(f"Raw model response that failed to parse: '{data.get('response')}'")
            return None
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
        system_prompt = f"""
You route bioinformatics QTL database queries to the best tool. Return ONLY: {{"tool_name": "...", "arguments": {{...}} }}

**DATABASE CONTEXT:**
- The database contains Quantitative Trait Loci (QTL) data.
- Each record is a "peak" with a "LOD score" indicating its significance.
- "Highest peak" means the peak with the highest LOD score.

**CRITICAL INSTRUCTIONS:**
1.  **Prioritize Gene-Specific Queries:** If the user's query mentions a specific gene symbol (like 'Apoe', 'Tdpoz2', 'Gnai3', etc.), your primary choice should almost always be the `get_gene_details` tool. This is true even if the query also asks about "LOD scores", "peaks", or other general terms. Use `get_gene_details` to retrieve all information for that specific gene.
2.  **Differentiate General vs. Specific Queries:**
    - **Specific Gene:** "what are the lods on tdpoz2" -> `get_gene_details(gene_symbol='Tdpoz2')`
    - **General Top Scores:** "what is the highest lod score" -> `analytical_query_top_lod(limit=1)`
    - **Specific LOD Score:** "what gene has a lod of 608.5" -> `find_gene_by_lod(lod_score=608.5)`
3.  **Tool Selection Logic:**
    - For questions about a specific **GENE NAME** (e.g., 'Apoe', 'what about Gnai3'), use `get_gene_details`.
    - For questions about a specific **LOD SCORE** (e.g., 'the gene with lod 608.5'), you MUST use the `find_gene_by_lod` tool.
    - For questions about general "highest" or "top" peaks **overall** (that do NOT mention a specific gene), use `analytical_query_top_lod`. If the user asks for "the highest" or "the top" in the singular (e.g., "what is the highest score"), set `limit` to 1.
4.  **Specific Tool Selection (for Gene-specific queries):**
    - For general information about a gene (function, summary, QTLs), use `get_gene_details`.
    - For comprehensive gene analysis with Ensemble API data (variants, detailed annotations), use `get_enhanced_gene_details`.
    - For cross-species analysis (human orthologs, comparative studies), use `get_cross_species_gene_info`.
5.  **Ensembl API Integration:**
    - Use `query_ensembl_api` for genomic data queries: gene sequences, variants, orthologs, phenotypes, regulatory elements
    - For "transcript isoforms" or "transcripts" use `query_ensembl_api` with query_type='transcripts' (gets transcript data)
    - For "variants" or "genetic variations" use `query_ensembl_api` with query_type='variants'
    - For "orthologs" or "cross-species" use `query_ensembl_api` with query_type='orthologs'
    - For "gene function" or "GO terms" use `query_ensembl_api` with query_type='gene_info' (includes gene function data)
    - For "DNA sequence" or "nucleotide sequence" use `query_ensembl_api` with query_type='sequence'
    - For "phenotype associations" or "disease associations" use `query_ensembl_api` with query_type='phenotype'
    - For "regulatory elements" or "binding sites" use `query_ensembl_api` with query_type='regulation'
    - Use `get_enhanced_gene_details` when users ask for "Ensembl data", "variants", or "comprehensive" gene information.
    - Use `get_cross_species_gene_info` when users mention "human orthologs", "cross-species", or "human-mouse comparison".
6.  You must respond in JSON format: `{{"tool_name": "...", "arguments": {{...}} }}`.
7.  If it is a broad biological concept queries use 'semantic_search'
8.  Default to `semantic_search` if no tool fits.

**Tools:** {tools_json_str}
"""
        
        user_prompt = f"""
**User Query:**
"{query}"

**Your JSON response (be concise):**
"""

        # 2. Call Ollama to get the tool choice
        tool_choice = self._call_ollama_for_tool_choice(user_prompt=user_prompt, system_prompt=system_prompt)

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
            print(f"[INFO] Using Ollama ({self.ollama_model}) for text generation.")
            response = self.ollama_session.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
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
6.  **RESPONSE STYLE:**
    - Be extremely concise and direct
    - Use bullet points when possible
    - Avoid verbose explanations
    - Focus on key facts only
    - Keep responses under 3 sentences when possible

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
        """Save generated raw QTL data documents to JSON."""
        if not self.summary_docs:
            self.generate_all_document_types()
        
        with open(output_file, 'w') as f:
            json.dump(self.summary_docs, f, indent=2, default=str)
        logger.info(f"✅ Saved {len(self.summary_docs)} raw QTL data documents to {output_file}")

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

    def test_ensembl_connection(self) -> bool:
        """
        Test the Ensembl API connection and get available gene examples.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("🔬 Testing Ensembl API connection...")
            
            # First test basic API connectivity
            info_url = f"{self.ensembl_client.base_url}/info/species"
            logger.info(f"🔍 Testing basic API connectivity: {info_url}")
            response = requests.get(info_url, headers=self.ensembl_client.headers)
            
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
                    f"{self.ensembl_client.base_url}/lookup/symbol/mus_musculus/{gene}",
                    f"{self.ensembl_client.base_url}/xrefs/symbol/mus_musculus/{gene}"
                ]
                
                for endpoint in endpoints:
                    logger.debug(f"🔍 Trying endpoint: {endpoint}")
                    response = requests.get(endpoint, headers=self.ensembl_client.headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if response.status_code == 200:
                            logger.info(f"✅ Successfully found {gene} via {endpoint}")
                            logger.debug(f"Gene info: {data}")
                            return True
                
                logger.warning(f"⚠️ Could not find {gene} in any Ensembl endpoint")
            
            logger.error("❌ No test genes found in Ensembl API")
            return False
            
        except Exception as e:
            logger.error(f"❌ Ensembl API connection test failed: {e}")
            return False

    def test_ensembl_api_tool(self) -> Dict[str, Any]:
        """
        Test the new Ensembl API tool functionality.
        
        Returns:
            Dictionary with test results
        """
        logger.info("🧪 Testing Ensembl API tool functionality...")
        
        test_results = {}
        
        if not self.ensembl_client:
            return {"error": "Ensembl client not available"}
        
        # Test different query types
        test_cases = [
            {"gene": "Apoe", "query_type": "gene_info", "species": "mus_musculus"},
            {"gene": "Gnai3", "query_type": "variants", "species": "mus_musculus"},
            {"gene": "Actb", "query_type": "orthologs", "species": "mus_musculus"},
            {"gene": "Gapdh", "query_type": "gene_function", "species": "mus_musculus"}
        ]
        
        for test_case in test_cases:
            logger.info(f"🔬 Testing: {test_case['gene']} - {test_case['query_type']}")
            try:
                result = self.query_ensembl_api(
                    test_case['gene'], 
                    test_case['query_type'], 
                    test_case['species']
                )
                test_results[f"{test_case['gene']}_{test_case['query_type']}"] = result
                logger.info(f"✅ {test_case['gene']} {test_case['query_type']}: {'Success' if 'error' not in result else 'Failed'}")
            except Exception as e:
                test_results[f"{test_case['gene']}_{test_case['query_type']}"] = {"error": str(e)}
                logger.error(f"❌ {test_case['gene']} {test_case['query_type']}: {e}")
        
        return test_results
    
    def get_available_ensembl_genes(self, limit: int = 10) -> List[str]:
        """
        Get a list of available genes in Ensembl for testing.
        
        Args:
            limit: Maximum number of genes to return
        
        Returns:
            List of gene symbols available in Ensemble
        """
        try:
            logger.info("🔍 Getting available genes from Ensembl...")
            
            # Try to get genes from a known region
            search_url = f"{self.base_url}/lookup/mus_musculus/region/1:1-1000000"
            response = requests.get(search_url, headers=self.headers)
            
            if response.status_code == 200:
                genes = response.json()
                gene_symbols = []
                for gene in genes:
                    if isinstance(gene, dict) and 'display_name' in gene:
                        gene_symbols.append(gene['display_name'])
                
                logger.info(f"✅ Found {len(gene_symbols)} genes in Ensembl")
                return gene_symbols[:limit]
            else:
                logger.warning(f"⚠️ Could not get gene list from Ensembl: {response.status_code}")
                return []
                
        except Exception as e:
                            logger.error(f"❌ Error getting available genes from Ensembl: {e}")
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

    def query_ensembl_api(self, gene_symbol: str, query_type: str, species: str = "mus_musculus") -> Dict[str, Any]:
        """
        Query Ensembl's REST API for genomic data using the correct, working endpoints.
        
        Args:
            gene_symbol: Gene symbol to query
            query_type: Type of query ('gene_info', 'variants', 'orthologs', 'transcripts', 'sequence', 'phenotype', 'regulation')
            species: Species identifier (default: mus_musculus)
        
        Returns:
            Dictionary with Ensembl API results
        """
        try:
            logger.info(f"🔬 Querying Ensembl API for {gene_symbol} ({query_type}) in {species}")
            
            # Use the correct, working Ensembl REST API endpoints
            if query_type == "gene_info":
                # Basic gene information - use the working endpoint
                result = self._call_ensembl_endpoint(f"/lookup/symbol/{species}/{gene_symbol}")
                
            elif query_type == "variants":
                # Get gene info first, then variants
                gene_info = self._call_ensembl_endpoint(f"/lookup/symbol/{species}/{gene_symbol}")
                if gene_info and 'id' in gene_info:
                    gene_id = gene_info['id']
                    result = self._call_ensembl_endpoint(f"/overlap/id/{gene_id}?feature=variation")
                    # Filter for variation features
                    if isinstance(result, list):
                        result = [v for v in result if v.get('feature_type') == 'variation']
                else:
                    result = None
                    
            elif query_type == "orthologs":
                # Get gene info first, then orthologs
                gene_info = self._call_ensembl_endpoint(f"/lookup/symbol/{species}/{gene_symbol}")
                if gene_info and 'id' in gene_info:
                    gene_id = gene_info['id']
                    # Use homology endpoint for orthologs
                    result = self._call_ensembl_endpoint(f"/homology/id/{gene_id}?target_species=homo_sapiens")
                else:
                    result = None
                    
            elif query_type == "transcripts":
                # Get gene info first, then transcripts
                gene_info = self._call_ensembl_endpoint(f"/lookup/symbol/{species}/{gene_symbol}")
                if gene_info and 'id' in gene_info:
                    gene_id = gene_info['id']
                    result = self._call_ensembl_endpoint(f"/overlap/id/{gene_id}?feature=transcript")
                    # Filter for transcript features
                    if isinstance(result, list):
                        result = [t for t in result if t.get('feature_type') == 'transcript']
                else:
                    result = None
                    
            elif query_type == "sequence":
                # Get gene info first, then sequence
                gene_info = self._call_ensembl_endpoint(f"/lookup/symbol/{species}/{gene_symbol}")
                if gene_info and 'id' in gene_info:
                    gene_id = gene_info['id']
                    result = self._call_ensembl_endpoint(f"/sequence/id/{gene_id}")
                else:
                    result = None
                    
            elif query_type == "phenotype":
                # Direct phenotype endpoint
                result = self._call_ensembl_endpoint(f"/phenotype/gene/{species}/{gene_symbol}")
                
            elif query_type == "regulation":
                # Get gene info first, then regulatory elements
                gene_info = self._call_ensembl_endpoint(f"/lookup/symbol/{species}/{gene_symbol}")
                if gene_info and 'id' in gene_info:
                    gene_id = gene_info['id']
                    result = self._call_ensembl_endpoint(f"/overlap/id/{gene_id}?feature=regulatory")
                else:
                    result = None
                    
            else:
                return {"error": f"Unknown query type: {query_type}. Use: gene_info, variants, orthologs, transcripts, sequence, phenotype, or regulation"}
            
            # Check if we got valid results
            if not result:
                return {
                    "gene_symbol": gene_symbol,
                    "query_type": query_type,
                    "species": species,
                    "result": None,
                    "warning": "No data returned from Ensembl API"
                }
            
            # Print the raw Ensembl API response for transparency
            print(f"\n🔬 ENSEMBL API RESPONSE for {gene_symbol} ({query_type}):")
            print("=" * 60)
            print(f"Query: {query_type} for {gene_symbol} in {species}")
            print(f"Result type: {type(result).__name__}")
            
            if isinstance(result, dict):
                print(f"Data fields: {list(result.keys())}")
                # Show key data points
                for key, value in result.items():
                    if key in ['id', 'display_name', 'seq_region_name', 'start', 'end', 'strand']:
                        print(f"  {key}: {value}")
                    elif key == 'description' and isinstance(value, list) and len(value) > 0:
                        print(f"  {key}: {len(value)} description entries")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"  {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"  {key}: {len(value)} sub-fields")
            elif isinstance(result, list):
                print(f"Result count: {len(result)}")
                if len(result) > 0:
                    print(f"First item keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'N/A'}")
            
            print("=" * 60)
            
            return {
                "gene_symbol": gene_symbol,
                "query_type": query_type,
                "species": species,
                "result": result,
                "endpoint_used": self._get_endpoint_info(query_type, gene_symbol, species)
            }
            
        except Exception as e:
            logger.error(f"❌ Ensembl API query failed: {e}")
            return {"error": str(e)}
    
    def _call_ensembl_endpoint(self, endpoint: str) -> Any:
        """Make a direct call to an Ensembl REST API endpoint."""
        try:
            base_url = "https://rest.ensembl.org"
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            url = f"{base_url}{endpoint}"
            logger.debug(f"🔍 Calling Ensembl endpoint: {url}")
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"⚠️ Ensembl API call failed: {response.status_code} for {endpoint}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error calling Ensembl endpoint {endpoint}: {e}")
            return None
    
    def _get_endpoint_info(self, query_type: str, gene_symbol: str, species: str) -> str:
        """Get information about which Ensembl endpoint was used."""
        endpoint_map = {
            "gene_info": f"/lookup/symbol/{species}/{gene_symbol}",
            "variants": f"/overlap/id/{{gene_id}}?feature=variation",
            "orthologs": f"/homology/id/{{gene_id}}?target_species=homo_sapiens",
            "transcripts": f"/overlap/id/{{gene_id}}?feature=transcript",
            "sequence": f"/sequence/id/{{gene_id}}",
            "phenotype": f"/phenotype/gene/{species}/{gene_symbol}",
            "regulation": f"/overlap/id/{{gene_id}}?feature=regulatory"
        }
        return endpoint_map.get(query_type, "Unknown endpoint")

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
    print("Setting up system with ONLY original QTL source data...")
    system.setup_vector_store(use_google_embeddings=False)
    
    print(f"\n✅ Raw QTL Data System Ready!")
    print(f"📊 Layer 1: {system.vector_collection.count()} raw QTL documents in vector store")
    print(f"🗃️ Layer 2: {len(system.raw_data)} raw QTL records in DuckDB")
    
    print("\n" + "="*50)
    print("DEMO 1: Search for specific gene data")
    print("="*50)
    results = system.semantic_search(
        "Find QTL data for gene Gnai3",
        n_results=3,
        where_filter={"gene_symbol": {"$eq": "Gnai3"}}
    )
    for result in results:
        print(f"Found Doc ID: {result['id']} (Distance: {result['distance']:.4f})")
        print(f"Raw Data: {result['content']}")
    
    print("\n" + "="*50)
    print("DEMO 2: Search by chromosome")
    print("="*50)
    results = system.semantic_search(
        "Find QTLs on chromosome 2",
        n_results=3,
        where_filter={"chromosome": {"$eq": "2"}}
    )
    for result in results:
        meta = result['metadata']
        print(f"Found Peak on Chr {meta['chromosome']} at {meta['position_mb']:.2f} Mb (LOD: {meta['lod_score']:.2f})")
        print(f"  Raw Data: {result['content']}")

    print("\n" + "="*50)
    print("DEMO 3: Search for cis-QTLs")
    print("="*50)
    results = system.semantic_search(
        "Find cis-QTLs for gene Apoe",
        n_results=3,
        where_filter={"$and": [
            {"gene_symbol": {"$eq": "Apoe"}},
            {"cis": {"$eq": "TRUE"}}
        ]}
    )
    for result in results:
        meta = result['metadata']
        print(f"Found Peak on Chr {meta['chromosome']} at {meta['position_mb']:.2f} Mb (LOD: {meta['lod_score']:.2f})")
        print(f"  Raw Data: {result['content']}")

    print("\n" + "="*50)
    print("DEMO 4: Analytical Query (Layer 2)")
    print("="*50)
    top_genes = system.analytical_query("SELECT gene_symbol, MAX(qtl_lod) as max_lod FROM qtl_peaks GROUP BY gene_symbol ORDER BY max_lod DESC LIMIT 5")
    print("Top 5 genes by maximum LOD score:")
    print(top_genes.to_string(index=False))
    
    print("\n" + "="*50)
    print("DEMO 5: Testing Ensembl API Tool")
    print("="*50)
    if system.ensembl_client:
        print("Testing Ensembl API tool functionality...")
        test_results = system.test_ensembl_api_tool()
        print("Ensembl API Tool Test Results:")
        for test_name, result in test_results.items():
            if 'error' in result:
                print(f"  ❌ {test_name}: {result['error']}")
            else:
                print(f"  ✅ {test_name}: Success")
                # Show a snippet of the result
                if 'result' in result and result['result']:
                    if isinstance(result['result'], dict):
                        keys = list(result['result'].keys())[:3]
                        print(f"    Data keys: {keys}")
                    elif isinstance(result['result'], list):
                        print(f"    Result count: {len(result['result'])}")
    else:
        print("⚠️ Ensembl client not available")
    
    system.save_summary_docs("qtl_raw_data_docs.json")
    
    print(f"\n🎉 Raw QTL data system demonstration complete!")
    print(f"💡 Use semantic_search() for raw data queries, analytical_query() for exact analysis, and query_ensembl_api() for genomic data.") 