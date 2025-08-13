#TOOLS
#from db import connector
import requests
from collections import defaultdict
import pandas as pd
from pathlib import Path
import os

def _resolve_mgi_path(default_filename: str = "HOM_MouseHumanSequence.rpt") -> Path:
    """Resolve the path to the MGI ortholog report with robust fallbacks.

    Order of precedence:
    1) MGI_ORTHOLOG_PATH environment variable (absolute or relative)
    2) Project root (parent of this file's directory) joined with default filename
    3) This file's directory (rag/) joined with default filename
    4) Current working directory joined with default filename
    """
    env_path = os.getenv("MGI_ORTHOLOG_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p.resolve()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    candidate_root = (project_root / default_filename)
    if candidate_root.exists():
        return candidate_root.resolve()

    candidate_script = (script_dir / default_filename)
    if candidate_script.exists():
        return candidate_script.resolve()

    candidate_cwd = (Path.cwd() / default_filename)
    if candidate_cwd.exists():
        return candidate_cwd.resolve()

    # None found: raise a clear error listing tried locations
    tried = [
        env_path or "<MGI_ORTHOLOG_PATH unset>",
        str(candidate_root),
        str(candidate_script),
        str(candidate_cwd),
    ]
    raise FileNotFoundError(
        "HOM_MouseHumanSequence.rpt not found. Set MGI_ORTHOLOG_PATH or place the file in one of: "
        + "; ".join(tried)
    )


def add_numbers(num1: int, num2: int):
    """Adds two numbers together.
    Args:
        num1: The first number.
        num2: The second number.
    """
    return num1 + num2


def convert_mouse_to_human_gene(gene_symbol: str):
    """
    Given a mouse gene symbol, return its human homolog(s) by
    looking up Homologene groups in a local JAX report.

    Case-insensitive. Handles 1:many mappings.
    """
    mapping_path = _resolve_mgi_path()
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
    Given a mouse gene symbol, return its human homolog(s) plus
    chromosome-band and genome‐coordinate info from the local JAX report.

    Case‐insensitive. Handles 1:many mappings.
    """
    mapping_path = _resolve_mgi_path()
    
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
    mapping = {}
    for mouse_sym, grp in df_pairs.groupby("mouse_symbol"):
        seen = set()
        info_list = []
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

def query_ensembl_api(gene_symbol: str, query_type: str, species: str = "mus_musculus"):
    """
    Query Ensembl's REST API for genomic data using the correct, working endpoints.
    
    Access Ensembl's REST API to retrieve genomic data such as gene coordinates, 
    sequences, transcript info, orthologs, variant annotations, phenotype data, and more.
    
    Args:
        gene_symbol: Gene symbol to query (e.g., 'Apoe', 'Gnai3')
        query_type: Type of query ('gene_info', 'variants', 'orthologs', 'transcripts', 'sequence', 'phenotype', 'regulation')
        species: Species identifier (default: mus_musculus, can be homo_sapiens for human)
    
    Returns:
        JSON response from Ensembl API with source attribution
    """
    base_url = "https://rest.ensembl.org"
    headers = {"Content-Type": "application/json"}
    
    try:
        # Use the correct, working Ensembl REST API endpoints
        if query_type == "gene_info":
            # Basic gene information - use the working endpoint
            url = f"{base_url}/lookup/symbol/{species}/{gene_symbol}"
            resp = requests.get(url, headers=headers, timeout=30)
            
        elif query_type == "variants":
            # Get gene info first, then variants
            gene_url = f"{base_url}/lookup/symbol/{species}/{gene_symbol}"
            gene_resp = requests.get(gene_url, headers=headers, timeout=30)
            
            if gene_resp.status_code != 200:
                return {"error": f"Failed to get gene info: {gene_resp.status_code}"}
            
            gene_info = gene_resp.json()
            if not gene_info or 'id' not in gene_info:
                return {"error": "No gene ID found"}
            
            gene_id = gene_info['id']
            url = f"{base_url}/overlap/id/{gene_id}?feature=variation"
            resp = requests.get(url, headers=headers, timeout=30)
            
        elif query_type == "orthologs":
            # Get gene info first, then orthologs
            gene_url = f"{base_url}/lookup/symbol/{species}/{gene_symbol}"
            gene_resp = requests.get(gene_url, headers=headers, timeout=30)
            
            if gene_resp.status_code != 200:
                return {"error": f"Failed to get gene info: {gene_resp.status_code}"}
            
            gene_info = gene_resp.json()
            if not gene_info or 'id' not in gene_info:
                return {"error": "No gene ID found"}
            
            gene_id = gene_info['id']
            url = f"{base_url}/homology/id/{gene_id}?target_species=homo_sapiens"
            resp = requests.get(url, headers=headers, timeout=30)
            
        elif query_type == "transcripts":
            # Get gene info first, then transcripts
            gene_url = f"{base_url}/lookup/symbol/{species}/{gene_symbol}"
            gene_resp = requests.get(gene_url, headers=headers, timeout=30)
            
            if gene_resp.status_code != 200:
                return {"error": f"Failed to get gene info: {gene_resp.status_code}"}
            
            gene_info = gene_resp.json()
            if not gene_info or 'id' not in gene_info:
                return {"error": "No gene ID found"}
            
            gene_id = gene_info['id']
            url = f"{base_url}/overlap/id/{gene_id}?feature=transcript"
            resp = requests.get(url, headers=headers, timeout=30)
            
        elif query_type == "sequence":
            # Get gene info first, then sequence
            gene_url = f"{base_url}/lookup/symbol/{species}/{gene_symbol}"
            gene_resp = requests.get(gene_url, headers=headers, timeout=30)
            
            if gene_resp.status_code != 200:
                return {"error": f"Failed to get gene info: {gene_resp.status_code}"}
            
            gene_info = gene_resp.json()
            if not gene_info or 'id' not in gene_info:
                return {"error": "No gene ID found"}
            
            gene_id = gene_info['id']
            url = f"{base_url}/sequence/id/{gene_id}"
            resp = requests.get(url, headers=headers, timeout=30)
            
        elif query_type == "phenotype":
            # Direct phenotype endpoint
            url = f"{base_url}/phenotype/gene/{species}/{gene_symbol}"
            resp = requests.get(url, headers=headers, timeout=30)
            
        elif query_type == "regulation":
            # Get gene info first, then regulatory elements
            gene_url = f"{base_url}/lookup/symbol/{species}/{gene_symbol}"
            gene_resp = requests.get(gene_url, headers=headers, timeout=30)
            
            if gene_resp.status_code != 200:
                return {"error": f"Failed to get gene info: {gene_resp.status_code}"}
            
            gene_info = gene_resp.json()
            if not gene_info or 'id' not in gene_info:
                return {"error": "No gene ID found"}
            
            gene_id = gene_info['id']
            url = f"{base_url}/overlap/id/{gene_id}?feature=regulatory"
            resp = requests.get(url, headers=headers, timeout=30)
            
        else:
            return {"error": f"Unknown query type: {query_type}. Use: gene_info, variants, orthologs, transcripts, sequence, phenotype, or regulation"}

        if resp.status_code != 200:
            return {"error": f"Ensembl API returned {resp.status_code}", "details": resp.text}

        # Add source attribution to the response
        response_data = resp.json()
        if isinstance(response_data, dict):
            response_data["_source"] = f"Data from Ensembl REST API (accessed {datetime.now().strftime('%Y-%m-%d')})"
            response_data["_query_info"] = {
                "gene_symbol": gene_symbol,
                "query_type": query_type,
                "species": species,
                "endpoint_used": url.replace(base_url, "")
            }
        elif isinstance(response_data, list):
            response_data = {
                "data": response_data,
                "_source": f"Data from Ensembl REST API (accessed {datetime.now().strftime('%Y-%m-%d')})",
                "_query_info": {
                    "gene_symbol": gene_symbol,
                    "query_type": query_type,
                    "species": species,
                    "endpoint_used": url.replace(base_url, "")
                }
            }
        
        return response_data
        
    except requests.RequestException as e:
        return {"error": "Failed to connect to Ensembl API", "details": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def query_gwas_api(endpoint: str, method: str = "GET", params: dict = None) -> dict:
    """
    Query the GWAS Catalog REST API and return raw results.
    
    Args:
        endpoint: API endpoint (e.g., "/associations", "/genes", "/studies", "/traits")
        method: HTTP method ("GET" or "POST")
        params: Query parameters
    
    Returns:
        Raw API response data
    """
    base_url = "https://www.ebi.ac.uk/gwas/rest/api"
    
    # Build the full URL
    url = f"{base_url}{endpoint}"
    
    headers = {"Accept": "application/json"}
    
    try:
        if method.upper() == "GET":
            resp = requests.get(url, params=params, headers=headers, timeout=60)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}

        if resp.status_code != 200:
            return {"error": f"GWAS Catalog API returned {resp.status_code}", "details": resp.text}

        # Return JSON response directly
        return resp.json()
        
    except requests.RequestException as e:
        return {"error": "Failed to connect to GWAS Catalog API", "details": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Update your tool registry:
tool_dict = {
    "add_numbers": add_numbers,
    "convert_mouse_to_human_gene": convert_mouse_to_human_gene,
    "convert_mouse_to_human_ortholog_info": convert_mouse_to_human_ortholog_info,
    "query_ensembl_api": query_ensembl_api,
    "query_gwas_api": query_gwas_api,
}

if __name__ == "__main__":
    print(convert_mouse_to_human_ortholog_info("Aldh1l1"))