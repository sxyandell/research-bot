#TOOLS
#from db import connector


try:
    from rag.helpers import _impc_fetch_significant_phenotypes, _resolve_ortholog_pair, _fetch_gtex_expression_local, _ensembl_request, _ensembl_lookup_gene_id, _normalize_species, _infer_species_from_gene
except ImportError:  # fallback when running inside rag/ directly
    from helpers import _impc_fetch_significant_phenotypes, _resolve_ortholog_pair, _fetch_gtex_expression_local, _ensembl_request, _ensembl_lookup_gene_id, _normalize_species, _infer_species_from_gene
from pathlib import Path
from typing import Dict, Any, List
from functools import lru_cache
from typing import Tuple, Optional
import os
import pandas as pd
import requests
from datetime import datetime
import re
try:
    from gwas_integration import GWASCatalog
except Exception:
    GWASCatalog = None
try:
    # Import BioPlex helpers (they handle lazy imports internally)
    from rag.helpers import _bioplex_fetch_interactions, _bioplex_interactors_for_symbol
except ImportError:  # fallback when running inside rag/ directly
    from helpers import _bioplex_fetch_interactions, _bioplex_interactors_for_symbol

# Import DuckDB display helpers
try:
    from rag.duckdbhelpers import pick_gene_or_phenotype, is_empty_value
except ImportError:  # fallback for running inside rag/
    from duckdbhelpers import pick_gene_or_phenotype, is_empty_value




def add_numbers(num1: int, num2: int):
    """Adds two numbers together.
    Args:
        num1: The first number.
        num2: The second number.
    """
    return num1 + num2


def convert_mouse_to_human_gene(gene_symbol: str):
    """
    Finds human homolog(s) for a given mouse gene symbol.

    This tool queries a local JAX homology report (`HOM_MouseHumanSequence.rpt`)
    to map a mouse gene to its corresponding human gene(s) based on a shared
    HomoloGene ID. The search is case-insensitive. It correctly handles cases
    where one mouse gene maps to multiple human homologs.
    """
    mapping_path = Path(os.getenv("HOM_HUMAN_MOUSE_RPT", str(Path(__file__).resolve().parent.parent / "HOM_MouseHumanSequence.rpt")))
    # read only cols we care about
    df = pd.read_csv(
        str(mapping_path),
        sep="\t",
        usecols=["DB Class Key", "Common Organism Name", "Symbol"],
        dtype=str,
        comment='#'  
    ).rename(columns={
        "DB Class Key": "homologene_id",
        "Common Organism Name": "organism",
    })

    # split into mouse vs human
    # make lowercase
    df_mouse = (
        df[df.organism == "mouse, laboratory"]
        .assign(mouse_symbol=lambda d: d.Symbol.str.lower())
        .loc[:, ["homologene_id", "mouse_symbol"]]
    )
    df_human = (
        df[df.organism == "human"]
        .assign(human_symbol=lambda d: d.Symbol.str.upper())
        .loc[:, ["homologene_id", "human_symbol"]]
    )

    # Inner-join on homologene_id to get all mouse↔human pairs
    df_pairs = df_mouse.merge(df_human, on="homologene_id", how="inner")

    # map mouse symbol to human symbol
    mapping = (
        df_pairs
        .groupby("mouse_symbol")["human_symbol"]
        .unique()
        .apply(lambda arr: sorted(arr.tolist()))
        .to_dict()
    )

    # lookup
    key = gene_symbol.lower()
    human_list = mapping.get(key)
    if not human_list:
        return f"No human homologs found for mouse gene '{gene_symbol}'."
    return (f"Found human homolog(s) for '{gene_symbol}': "
            f"{', '.join(human_list)}.")


def convert_mouse_to_human_ortholog_info(gene_symbol: str):
    """
    Finds human ortholog(s) for a mouse gene and returns detailed location info.

    This tool queries a local JAX homology report (`HOM_MouseHumanSequence.rpt`)
    to find human orthologs. For each human ortholog identified, it returns the
    gene symbol, its chromosome band location, and its genomic coordinates based
    on the GRCh38 assembly. The search is case-insensitive and handles
    one-to-many mappings.
    """
    mapping_path = Path(os.getenv("HOM_HUMAN_MOUSE_RPT", str(Path(__file__).resolve().parent.parent / "HOM_MouseHumanSequence.rpt")))
    
    # 1) Read only the columns we need
    df = pd.read_csv(
        str(mapping_path),
        sep="\t",
        usecols=[
            "DB Class Key",
            "Common Organism Name",
            "Symbol",
            "Genetic Location",
            "Genome Coordinates (mouse: GRCm39 human: GRCh38)"
        ],
        dtype=str,
        comment="#"
    ).rename(columns={
        "DB Class Key": "homologene_id",
        "Common Organism Name": "organism",
        "Genetic Location": "genetic_location",
        "Genome Coordinates (mouse: GRCm39 human: GRCh38)": "genome_coordinates"
    })

    # 2) Split into mouse vs. human, normalizing case
    df_mouse = (
        df[df.organism == "mouse, laboratory"]
        .assign(mouse_symbol=lambda d: d.Symbol.str.lower())
        .loc[:, ["homologene_id", "mouse_symbol"]]
    )
    df_human = (
        df[df.organism == "human"]
        .assign(human_symbol=lambda d: d.Symbol.str.upper())
        .loc[:, ["homologene_id", "human_symbol", "genetic_location", "genome_coordinates"]]
    )

    # 3) Join on homologene_id to get each mouse↔human pair
    df_pairs = df_mouse.merge(df_human, on="homologene_id", how="inner")

    # 4) Build a mapping: mouse_symbol → list of { human_symbol, genetic_location, genome_coordinates }
    mapping: Dict[str, List[Dict[str, str]]] = {}
    for mouse_sym, grp in df_pairs.groupby("mouse_symbol"):
        seen = set()
        info_list: List[Dict[str, str]] = []
        for _, row in grp.iterrows():
            key = (row.human_symbol, row.genetic_location, row.genome_coordinates)
            if key in seen:
                continue
            seen.add(key)
            info_list.append({
                "human_symbol":       row.human_symbol,
                "genetic_location":   row.genetic_location,
                "genome_coordinates": row.genome_coordinates
            })
        mapping[mouse_sym] = info_list

    # 5) Lookup & format
    key = gene_symbol.lower()
    orthologs = mapping.get(key)
    if not orthologs:
        return f"No human homologs found for mouse gene '{gene_symbol}'."

    parts = [
        f"{o['human_symbol']} (chrom: {o['genetic_location']}, coords: {o['genome_coordinates']})"
        for o in orthologs
    ]
    return f"Found human homolog(s) for '{gene_symbol}': " + "; ".join(parts) + "."


def get_impc_knockout_status(gene_symbol: str) -> str:
    """
    Returns a concise summary of a gene's knockout phenotyping status from IMPC.
    Ideal for simple "yes/no" questions about whether knockout data exists.

    Args:
        gene_symbol: Official mouse gene symbol (e.g., "Trp53").
    """
    def _normalize_mouse_gene_symbol_case(symbol: str) -> str:
        s = (symbol or "").strip()
        if not s:
            return s
        return s[0].upper() + s[1:].lower()

    normalized_symbol = _normalize_mouse_gene_symbol_case(gene_symbol)
    result = _impc_fetch_significant_phenotypes(normalized_symbol)
    marker = result.get("gene", gene_symbol)
    status = result.get("impc_knockout")
    if result.get("error"):
        return f"Error for {marker}: {result['error']}"
    if status is None:
        return f"Gene: {marker} — IMPC knockout: unknown"
    return f"Gene: {marker} — IMPC knockout: {'yes' if status else 'no'}"


def get_impc_significant_phenotypes(gene_symbol: str) -> str:
    """
    Returns a formatted list of significant phenotypes for a gene knockout from IMPC.
    Best for when a user asks 'what' or 'which' phenotypes from IMPC were reported.

    Args:
        gene_symbol: Official mouse gene symbol (e.g., "Trp53").
    """
    def _normalize_mouse_gene_symbol_case(symbol: str) -> str:
        s = (symbol or "").strip()
        if not s:
            return s
        return s[0].upper() + s[1:].lower()

    normalized_symbol = _normalize_mouse_gene_symbol_case(gene_symbol)
    result = _impc_fetch_significant_phenotypes(normalized_symbol)
    marker = result.get("gene", gene_symbol)
    if result.get("error"):
        return f"Error for {marker}: {result['error']}"
    phenos = result.get("significant_phenotypes") or []
    if not phenos:
        return f"Significant IMPC phenotypes for {marker}: none found"
    preview = "; ".join(phenos[:10])
    suffix = "" if len(phenos) <= 10 else f"; (+{len(phenos)-10} more)"
    return f"Significant IMPC phenotypes for {marker}: {preview}{suffix}"


def get_impc_gene_summary(gene_symbol: str) -> str:
    """
    Returns a comprehensive, formatted summary of a gene's IMPC knockout status and phenotypes.
    This is the best general-purpose tool for a full overview of a gene.

    Args:
        gene_symbol: Official mouse gene symbol (e.g., "Trp53").
    """
    def _normalize_mouse_gene_symbol_case(symbol: str) -> str:
        s = (symbol or "").strip()
        if not s:
            return s
        return s[0].upper() + s[1:].lower()

    normalized_symbol = _normalize_mouse_gene_symbol_case(gene_symbol)
    result = _impc_fetch_significant_phenotypes(normalized_symbol)
    marker = result.get("gene", gene_symbol)
    if result.get("error"):
        return f"Error for {marker}: {result['error']}"
    status = result.get("impc_knockout")
    phenos = result.get("significant_phenotypes") or []
    status_text = "unknown" if status is None else ("yes" if status else "no")
    if not phenos:
        return f"Gene: {marker}\nIMPC knockout: {status_text}\nSignificant phenotypes: none found"
    preview = ", ".join(phenos[:10])
    more = "" if len(phenos) <= 10 else f" (+{len(phenos) - 10} more)"
    return f"Gene: {marker}\nIMPC knockout: {status_text}\nSignificant phenotypes: {preview}{more}"

# --------------------- New: GTEx + Tabula Muris expression tool ---------------------

_TABULA_MURIS_PATH = Path(os.getenv("TABULA_MURIS_H5AD", str(Path(__file__).resolve().parent.parent / "data" / "tabula-muris.h5ad")))


def _gtex_resolve_gencode_id(gene_symbol: str) -> Optional[str]:
    """Resolve a gene symbol or Ensembl ID to a canonical Ensembl/GENCODE ID for GTEx.

    Returns an ID like 'ENSG00000141510' (no version) if possible.
    """
    import requests  # lazy import
    sym = (gene_symbol or "").strip()
    if not sym:
        return None
    # If already looks like Ensembl gene id, strip version and return
    if sym.upper().startswith("ENSG"):
        return sym.split(".")[0]

    # Try a few reference endpoints/params
    base = "https://gtexportal.org/rest/v2/reference/gene"
    attempts = [
        {"geneSymbol": sym, "genomeBuild": "GRCh38"},
        {"geneId": sym, "genomeBuild": "GRCh38"},
        {"searchTerm": sym, "genomeBuild": "GRCh38"},
    ]
    for params in attempts:
        try:
            r = requests.get(base, params=params, timeout=15)
            if r.status_code != 200:
                continue
            data = r.json() or {}
            genes = data.get("gene") or data.get("genes") or data.get("data") or []
            if isinstance(genes, dict):
                genes = [genes]
            for g in genes:
                gid = g.get("gencodeId") or g.get("geneId") or g.get("id")
                if gid and str(gid).upper().startswith("ENSG"):
                    return str(gid).split(".")[0]
        except Exception:
            continue
    return None


def _fetch_gtex_expression(gene_symbol: str) -> List[Tuple[str, float]]:
    """Fetch top expressed tissues for a gene from GTEx.

    Uses the v1 medianGeneExpression endpoint with datasetId=gtex_v8 when possible.
    Returns list of (tissue_name, median_tpm), sorted by median TPM desc, up to 10.
    """
    import requests  # lazy import
    symbol = (gene_symbol or "").strip()
    if not symbol:
        return []

    # First: try local GTEx v10 file if available
    local = _fetch_gtex_expression_local(symbol)
    if local:
        return local

    
    # Resolve to Ensembl gene ID
    gencode = _gtex_resolve_gencode_id(symbol) or symbol

    # Preferred: v1 medianGeneExpression per tissue
    try:
        url_v1 = "https://gtexportal.org/rest/v1/expression/medianGeneExpression"
        # Try with gencodeId first
        for params_v1 in (
            {"gencodeId": gencode, "datasetId": "gtex_v8", "format": "json"},
            {"geneSymbol": symbol, "datasetId": "gtex_v8", "format": "json"},
        ):
            r = requests.get(url_v1, params=params_v1, timeout=20)
            if r.status_code != 200:
                continue
            data = (r.json() or {}).get("medianGeneExpression", [])
            if not data:
                continue
            results: List[Tuple[str, float]] = []
            for item in data:
                tissue_raw = item.get("tissueSiteDetailId") or item.get("tissueSiteDetail") or item.get("tissue") or item.get("tissueSite") or "Unknown"
                tissue_name = str(tissue_raw).replace("_", " - ")
                median_tpm = float(item.get("median", 0.0) or 0.0)
                results.append((tissue_name, median_tpm))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
    except Exception:
        pass

    # Fallback: v2 topExpressedGene attempts
    base_url_v2 = "https://gtexportal.org/rest/v2/expression/topExpressedGene"
    param_attempts: List[Dict[str, Any]] = [
        {"gencodeId": gencode, "pageSize": 10, "sortBy": "median", "sortDirection": "desc", "format": "json"},
        {"gencodeId": symbol, "pageSize": 10, "sortBy": "median", "sortDirection": "desc", "format": "json"},
        {"geneSymbol": symbol, "pageSize": 10, "sortBy": "median", "sortDirection": "desc", "format": "json"},
    ]
    for params in param_attempts:
        try:
            resp = requests.get(base_url_v2, params=params, timeout=20)
            if resp.status_code != 200:
                continue
            data = (resp.json() or {}).get("topExpressedGene", [])
            if not data:
                continue
            results: List[Tuple[str, float]] = []
            for item in data:
                tissue_raw = item.get("tissueSiteDetailId") or item.get("tissueSiteDetail") or item.get("tissueSite") or "Unknown"
                tissue_name = str(tissue_raw).replace("_", " - ")
                median_tpm = float(item.get("median", 0.0) or 0.0)
                results.append((tissue_name, median_tpm))
            return results
        except Exception:
            continue
    return []


@lru_cache(maxsize=1)
def _load_tabula_muris_data():
    """Load the Tabula Muris AnnData file, caching in memory. Returns None on failure."""
    try:
        import scanpy as sc  # imported lazily to avoid hard dependency at import time
    except Exception:
        return None
    try:
        return sc.read_h5ad(str(_TABULA_MURIS_PATH))
    except Exception:
        return None


def _select_obs_tissue_key(adata) -> Optional[str]:
    """Pick a reasonable obs column to represent tissue/organ labels."""
    if adata is None:
        return None
    candidate_keys = [
        "tissue",
        "organ",
        "tissue_general",
        "tissue_ontology_term",
        "tissue_ontology_term_id",
    ]
    for key in candidate_keys:
        if key in adata.obs.columns:
            return key
    # fallback to any obs column containing the word "tissue"
    for key in adata.obs.columns:
        if "tissue" in str(key).lower():
            return key
    return None


def _to_numpy(matrix):
    """Safely convert AnnData .X slice to a dense numpy array."""
    try:
        # scipy sparse
        return matrix.toarray()
    except Exception:
        return matrix


def _fetch_tabula_muris_expression(gene_symbol: str) -> List[Tuple[str, float]]:
    """Compute top tissues by mean expression for a gene from local Tabula Muris data."""
    adata = _load_tabula_muris_data()
    if adata is None:
        # Add a print statement here for debugging if the file doesn't load
        print("Debug: Tabula Muris data file not loaded. Check path.")
        return []

    import pandas as pd  # lazy import
    gene = (gene_symbol or "").strip()
    
    # --- ADD THIS BLOCK ---
    # Make gene matching more robust: check original, capitalized, and lowercase
    if gene not in adata.var_names:
        capitalized_gene = gene.capitalize()
        if capitalized_gene in adata.var_names:
            gene = capitalized_gene
        elif gene.lower() in adata.var_names:
            gene = gene.lower()
        else:
            return [] # Gene not found even with variations
    # --- END BLOCK ---
    
    # The rest of the function remains the same...
    tissue_key = _select_obs_tissue_key(adata)
    if not tissue_key:
        return []

    # Build DataFrame of expression and tissue annotations
    gene_expr = _to_numpy(adata[:, gene].X)
    try:
        import numpy as np
    except Exception:  # numpy should be present, but guard anyway
        return []
    expr_series = pd.Series(np.asarray(gene_expr).ravel(), index=adata.obs_names, name=gene)
    df = pd.DataFrame({gene: expr_series, "tissue": adata.obs[tissue_key].astype(str).values})

    mean_by_tissue = df.groupby("tissue")[gene].mean().sort_values(ascending=False).head(10)
    return list(mean_by_tissue.items())


def get_top_tissue_expression(gene_symbol: str) -> str:
    """
    Fetches and lists the top 10 tissues with the highest expression for a given gene
    in both human (GTEx) and mouse (Tabula Muris). It automatically handles ortholog conversion.
    """
    input_symbol = (gene_symbol or "").strip()
    if not input_symbol:
        return "Error: Gene symbol cannot be empty."

    # --- Step 1: Resolve ortholog pair (human_symbol, mouse_symbol) ---
    human_symbol, mouse_symbol = _resolve_ortholog_pair(input_symbol)
    
    # --- Step 2: Fetch data using the correct symbols ---
    human_tissues = []
    if human_symbol:
        human_tissues = _fetch_gtex_expression(human_symbol)
    
    # Temporarily disable mouse (Tabula Muris) expression
    # mouse_tissues = []
    # if mouse_symbol:
    #     mouse_tissues = _fetch_tabula_muris_expression(mouse_symbol)
    mouse_tissues = []

    # --- Step 3: Format the output ---
    summary_title = f"**Expression Summary for {input_symbol}**"
    if human_symbol and mouse_symbol and input_symbol not in [human_symbol, mouse_symbol]:
         summary_title = f"**Expression Summary for {input_symbol} (Human: {human_symbol} | Mouse: {mouse_symbol})**"

    if human_tissues:
        human_list = [f"- **{name}:** {tpm:.2f} TPM" for name, tpm in human_tissues]
        human_output = f"### 👤 Human ({human_symbol})\n" + "\n".join(human_list)
    else:
        human_output = f"### 👤 Human ({human_symbol or input_symbol})\n- No expression data found."

    if mouse_tissues:
        mouse_list = [f"- **{name}:** {value:.2f} (mean expression)" for name, value in mouse_tissues]
        mouse_output = f"### 🐭 Mouse ({mouse_symbol})\n" + "\n".join(mouse_list)
    else:
        mouse_output = f"### 🐭 Mouse ({mouse_symbol or input_symbol})\n- Mouse expression temporarily disabled."

    return (
        f"{summary_title}\n\n"
        f"{human_output}\n\n"
        f"{mouse_output}"
    )

def get_ensembl_info(gene_symbol: str, query_type: str, species: Optional[str] = None):
    """
    Retrieves various types of genomic information for a specific gene from the Ensembl database.

    Use this tool to find details about a gene when you have its symbol. You can specify what kind of information you need.

    Args:
        gene_symbol: The official symbol of the gene to look up (e.g., 'APOE', 'Gnai3').
        query_type: The type of information to retrieve. Options are:
                    - 'gene_info': For core details like chromosome location and gene ID.
                    - 'variants': To find genetic variations (SNPs) within the gene.
                    - 'transcripts': To get the different RNA versions of the gene.
                    - 'phenotype': To find known physical traits or diseases linked to the gene.
                    - 'regulation': To identify elements that control the gene's activity.
        species: The species, such as 'human' or 'mouse'. If omitted, the tool will infer it from the gene symbol.
    """

    sym = (gene_symbol or "").strip()
    if not sym:
        return {"error": "gene_symbol is required"}

    # Normalize species or infer from gene symbol if not provided/unknown
    sp = _normalize_species(species)
    if not sp:
        sp = _infer_species_from_gene(sym)

    try:
        if query_type == "gene_info":
            data = _ensembl_request(f"/lookup/symbol/{sp}/{sym}")

        elif query_type in ("variants", "transcripts", "regulation"):
            gene_id = _ensembl_lookup_gene_id(sp, sym)
            if not gene_id:
                return {"error": "No gene ID found"}
            if query_type == "variants":
                data = _ensembl_request(f"/overlap/id/{gene_id}", params={"feature": "variation"})
            elif query_type == "transcripts":
                data = _ensembl_request(f"/overlap/id/{gene_id}", params={"feature": "transcript"})
            elif query_type == "regulation":
                data = _ensembl_request(f"/overlap/id/{gene_id}", params={"feature": "regulatory"})

        elif query_type == "phenotype":
            data = _ensembl_request(f"/phenotype/gene/{sp}/{sym}")

        else:
            return {"error": f"Unknown query type: {query_type}. Use: gene_info, variants, transcripts, phenotype, or regulation"}

        if data is None:
            return {"error": "Ensembl API request failed"}

        # Add source attribution to the response
        if isinstance(data, dict):
            data.setdefault("_source", f"Data from Ensembl REST API (accessed {datetime.now().strftime('%Y-%m-%d')})")
            data.setdefault("_query_info", {})
            data["_query_info"].update({
                "gene_symbol": sym,
                "query_type": query_type,
                "species": sp,
            })
        elif isinstance(data, list):
            data = {
                "data": data,
                "_source": f"Data from Ensembl REST API (accessed {datetime.now().strftime('%Y-%m-%d')})",
                "_query_info": {
                    "gene_symbol": sym,
                    "query_type": query_type,
                    "species": sp,
                }
            }
        return data

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def get_protein_interactions(gene_symbol: str) -> str:
    """
    Finds and lists known physical protein interactors for a human gene of interest.
    Queries the BioPlex database for interactions in both HEK293T and HCT116 humancell lines.
    
    Args:
        gene_symbol: The official human gene symbol for the protein of interest (e.g., "EGFR").
    """
    # Ensure BioPlex deps are available via helper
    interactions_293t_df = _bioplex_fetch_interactions("293T")
    interactions_hct116_df = _bioplex_fetch_interactions("HCT116")
    if interactions_293t_df is None or interactions_hct116_df is None:
        return (
            "Error: BioPlex data could not be loaded. Ensure 'bioplexpy' and its dependencies are installed (e.g., "
            "'pip install --no-cache-dir --upgrade numpy scipy anndata bioplexpy')."
        )

    symbol = (gene_symbol or "").strip().upper()
    if not symbol:
        return "Error: Gene symbol cannot be empty."

    try:
        interactors_293t = _bioplex_interactors_for_symbol(interactions_293t_df, symbol)
        interactors_hct116 = _bioplex_interactors_for_symbol(interactions_hct116_df, symbol)
    except Exception as e:
        return f"Error: Could not compute interactors. Details: {e}"
    
    # --- Format the output for the RAG system ---
    summary_title = f"**BioPlex Interaction Summary for {symbol}**"
    
    output_293t = f"###  HEK293T Cells ({len(interactors_293t)} interactors found)"
    if len(interactors_293t) > 0:
        preview = ", ".join(interactors_293t[:15])
        more = f" *(+{len(interactors_293t) - 15} more)*" if len(interactors_293t) > 15 else ""
        output_293t += f"\n- **Top Interactors:** {preview}{more}"
    else:
        output_293t += "\n- No significant interactions found in this cell line."

    output_hct116 = f"### HCT116 Cells ({len(interactors_hct116)} interactors found)"
    if len(interactors_hct116) > 0:
        preview = ", ".join(interactors_hct116[:15])
        more = f" *(+{len(interactors_hct116) - 15} more)*" if len(interactors_hct116) > 15 else ""
        output_hct116 += f"\n- **Top Interactors:** {preview}{more}"
    else:
        output_hct116 += "\n- No significant interactions found in this cell line."
        
    if not interactors_293t and not interactors_hct116:
        return f"{summary_title}\n\n- No physical interactors found for {symbol} in the BioPlex database."

    return f"{summary_title}\n\n{output_293t}\n\n{output_hct116}"


def get_top_lod_peaks(limit: int = 10) -> str:
    """
    Returns the top rows by LOD-like value from the unified DuckDB table, including
    the source and either a gene symbol (for gene-level results) or a phenotype
    for other result types (clinical traits, liver lipids, isoforms, etc.).

    Selection rules per row:
    - If the Source indicates a gene-level file (contains 'genes' but not 'isoform'),
      display the gene symbol
    - Otherwise, display the phenotype

    Notes:
    - Coalesces among columns if present: lod_diff, qtl_lod, lod
    - Gene symbol column is chosen from common candidates (gene_symbol, gene, symbol, gene_name)
    - Phenotype column is chosen from common candidates (phenotype, trait, trait_name, trait_description)
    - Database path can be overridden via env var QTL_DUCKDB_PATH or QTL_DUCKDB_FILE
    - Table name can be overridden via env var QTL_DUCKDB_TABLE (default: qtl_peaks)
    """
    try:
        import duckdb  # imported lazily
    except Exception:
        return "Error: duckdb is not installed. Install with 'pip install duckdb'."

    db_path = (
        os.getenv("QTL_DUCKDB_PATH")
        or os.getenv("QTL_DUCKDB_FILE")
        or str(Path(__file__).resolve().parent.parent / "data" / "qtl_database.duckdb")
    )
    table_name = os.getenv("QTL_DUCKDB_TABLE", "qtl_peaks")

    if not os.path.exists(db_path):
        return f"Error: DuckDB database not found at {db_path}"

    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as e:
        return f"Error: Could not open DuckDB database at {db_path}. Details: {e}"

    try:
        # Inspect columns and build a case-insensitive lookup
        try:
            schema_rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        except Exception as e:
            return f"Error: Could not inspect table '{table_name}'. Details: {e}"
        if not schema_rows:
            return f"Error: Table '{table_name}' not found in database {db_path}"

        # DuckDB PRAGMA table_info returns rows: (cid, name, type, notnull, dflt_value, pk)
        name_by_lower = {str(r[1]).lower(): str(r[1]) for r in schema_rows}

        def find_first(candidates: list[str]) -> str | None:
            for cand in candidates:
                if cand.lower() in name_by_lower:
                    return name_by_lower[cand.lower()]
            return None

        # Determine LOD-like columns present
        lod_candidates = ["lod_diff", "qtl_lod", "lod_add", "lod_int", "lod"]
        lod_present = [name_by_lower[c] for c in [lc for lc in lod_candidates if lc in name_by_lower]]
        if not lod_present:
            return "Error: No LOD-like columns found (expected one of: lod_diff, qtl_lod, lod)."

        # Build COALESCE expression in priority order
        lod_try_casts = [f'TRY_CAST("{name_by_lower[c]}" AS DOUBLE)' for c in lod_candidates if c in name_by_lower]
        lod_expr = f"COALESCE({', '.join(lod_try_casts)})"

        # Gene symbol selection (best-effort)
        gene_candidates = [
            "gene_symbol", "GeneSymbol", "gene", "Gene", "symbol", "Symbol",
            "gene_name", "Gene_Name", "GeneName",
        ]
        gene_col = find_first(gene_candidates)
        gene_expr = f'"{gene_col}"' if gene_col else "NULL"

        # Phenotype selection (best-effort)
        phenotype_candidates = [
            "phenotype", "Phenotype", "trait", "Trait", "trait_name", "trait_label", "trait_description",
        ]
        phenotype_col = find_first(phenotype_candidates)
        phenotype_expr = f'"{phenotype_col}"' if phenotype_col else "NULL"

        # Source column (created by the builder)
        source_col = find_first(["Source", "source", "filename", "file"]) or "Source"
        source_expr = f'"{source_col}"'

        sql = (
            f"SELECT {source_expr} AS source, {gene_expr} AS gene_symbol, {phenotype_expr} AS phenotype, {lod_expr} AS lod_value "
            f"FROM \"{table_name}\" "
            f"WHERE {lod_expr} IS NOT NULL "
            f"ORDER BY lod_value DESC "
            f"LIMIT {int(limit)}"
        )

        rows = con.execute(sql).fetchall() or []
        if not rows:
            return "No rows with non-null LOD-like values were found."

        # Format output
        header = (
            f"**Top {min(len(rows), int(limit))} LOD-like Peaks**\n\n"
            f"Data source: {db_path} — table: {table_name}"
        )
        lines: list[str] = []

        for idx, (source, gene_symbol, phenotype, lod_value) in enumerate(rows, start=1):
            try:
                lod_num = float(lod_value) if lod_value is not None else float('nan')
            except Exception:
                lod_num = float('nan')

            src_txt = str(source) if source not in (None, "") else "Unknown source"
            chosen = pick_gene_or_phenotype(src_txt, gene_symbol, phenotype) or "Unknown"

            lines.append(f"{idx}. LOD {lod_num:.3f} — {chosen} — Source: {src_txt}")

        return header + "\n" + "\n".join(lines)

    finally:
        try:
            con.close()
        except Exception:
            pass


def search_qtl_peaks(query: str, limit: int = 200) -> str:
    """
    Search the DuckDB `qtl_peaks` table for a given term and return matching peaks
    from all files with source and LOD.

    Rules:
    - If the query looks like a gene, search ONLY the `gene_symbol` column.
    - Otherwise, search ONLY the `phenotype` column.
    - Results ordered by best available LOD-like value (lod_diff, qtl_lod, lod).

    Environment overrides:
    - QTL_DUCKDB_PATH or QTL_DUCKDB_FILE: DuckDB path (default data/qtl_database.duckdb)
    - QTL_DUCKDB_TABLE: table name (default qtl_peaks)
    """
    if not (query or "").strip():
        return "Error: query cannot be empty."

    try:
        import duckdb
    except Exception:
        return "Error: duckdb is not installed. Install with 'pip install duckdb'."

    db_path = (
        os.getenv("QTL_DUCKDB_PATH")
        or os.getenv("QTL_DUCKDB_FILE")
        or str(Path(__file__).resolve().parent.parent / "data" / "qtl_database.duckdb")
    )
    table_name = os.getenv("QTL_DUCKDB_TABLE", "qtl_peaks")

    if not os.path.exists(db_path):
        return f"Error: DuckDB database not found at {db_path}"

    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as e:
        return f"Error: Could not open DuckDB database at {db_path}. Details: {e}"

    try:
        # Verify required columns exist
        try:
            schema_rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        except Exception as e:
            return f"Error: Could not inspect table '{table_name}'. Details: {e}"
        if not schema_rows:
            return f"Error: Table '{table_name}' not found in database {db_path}"

        present_cols = {str(r[1]) for r in schema_rows}
        required = {"Source", "gene_symbol", "phenotype"}
        missing = [c for c in required if c not in present_cols]
        if missing:
            return f"Error: Required columns missing in '{table_name}': {', '.join(missing)}"

        # LOD-like expression
        # Use exact names only; they may or may not exist
        col_names_lower = {str(r[1]).lower(): str(r[1]) for r in schema_rows}
        lod_try_casts = []
        for cname in ["lod_diff", "qtl_lod", "lod_add", "lod_int", "lod"]:
            if cname in col_names_lower:
                lod_try_casts.append(f'TRY_CAST("{col_names_lower[cname]}" AS DOUBLE)')
        if not lod_try_casts:
            return "Error: No LOD-like columns found (expected one of: lod_diff, qtl_lod, lod)."
        lod_expr = f"COALESCE({', '.join(lod_try_casts)})"

        # Gene-like query detection
        q = (query or "").strip()
        # Simple heuristic: gene-like only if no underscores and no spaces
        is_gene_like = ("_" not in q) and (" " not in q)

        q_esc = q.lower().replace("'", "''")
        if is_gene_like:
            gene_filters = [
                f"lower(CAST(\"gene_symbol\" AS VARCHAR)) LIKE '%{q_esc}%'",
                "lower(\"Source\") LIKE '%genes%'",
                "lower(\"Source\") NOT LIKE '%isoform%'",
                "lower(\"Source\") NOT LIKE '%splice%'",
                "lower(\"Source\") NOT LIKE '%junc%'",
            ]
            where_clause = " AND ".join(gene_filters)
        else:
            where_clause = f"lower(CAST(\"phenotype\" AS VARCHAR)) LIKE '%{q_esc}%'"

        sql = (
            f"SELECT \"Source\" AS source, \"gene_symbol\" AS gene_symbol, \"phenotype\" AS phenotype, "
            f"{lod_expr} AS lod_value "
            f"FROM \"{table_name}\" "
            f"WHERE ({where_clause}) AND {lod_expr} IS NOT NULL "
            f"ORDER BY lod_value DESC "
            f"LIMIT {int(limit)}"
        )

        # DEBUG: print SQL to diagnose matching
        try:
            print("[DEBUG search_qtl_peaks SQL]", sql)
        except Exception:
            pass

        rows = con.execute(sql).fetchall() or []
        if not rows:
            return f"No peaks found matching '{query}'."

        header = (
            f"**QTL Peaks matching: {q}**\n\n"
            f"Data source: {db_path} — table: {table_name}"
        )
        lines: list[str] = []
        for idx, (source, gene_symbol, phenotype, lod_value) in enumerate(rows, start=1):
            try:
                lod_num = float(lod_value) if lod_value is not None else float('nan')
            except Exception:
                lod_num = float('nan')
            src_txt = str(source) if source not in (None, "") else "Unknown source"
            label = pick_gene_or_phenotype(src_txt, gene_symbol, phenotype) or "Unknown"
            lines.append(f"{idx}. LOD {lod_num:.3f} — {label} — Source: {src_txt}")

        return header + "\n" + "\n".join(lines)

    finally:
        try:
            con.close()
        except Exception:
            pass

# Update your tool registry:
tool_dict = {
    "add_numbers": add_numbers,
    "convert_mouse_to_human_gene": convert_mouse_to_human_gene,
    "convert_mouse_to_human_ortholog_info": convert_mouse_to_human_ortholog_info,
    "get_impc_knockout_status": get_impc_knockout_status,
    "get_impc_significant_phenotypes": get_impc_significant_phenotypes,
    "get_impc_gene_summary": get_impc_gene_summary,
    "get_top_tissue_expression": get_top_tissue_expression,
    "get_ensembl_info": get_ensembl_info,
    "get_protein_interactions" : get_protein_interactions,
    "get_top_lod_peaks": get_top_lod_peaks,
    "search_qtl_peaks": search_qtl_peaks,
}

# --------------------- Manual testing entrypoint ---------------------
if __name__ == "__main__":
	print(search_qtl_peaks("ratio_Cer_to_HexCer"))