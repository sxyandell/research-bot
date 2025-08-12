#TOOLS
#from db import connector
try:
    from rag.helpers import _impc_fetch_significant_phenotypes
except ImportError:  # fallback when running inside rag/ directly
    from helpers import _impc_fetch_significant_phenotypes
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


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
    mapping_file = Path(__file__).resolve().parent.parent / "HOM_MouseHumanSequence.rpt"
    # read only cols we care about
    df = pd.read_csv(
        mapping_file,
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
    mapping_file = Path(__file__).resolve().parent.parent / "HOM_MouseHumanSequence.rpt"
    
    # 1) Read only the columns we need
    df = pd.read_csv(
        mapping_file,
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
    Return the IMPC knockout status for a gene as a concise string.

    Args:
        gene_symbol: Official mouse gene symbol (e.g., "Trp53").
    """
    result = _impc_fetch_significant_phenotypes(gene_symbol)
    marker = result.get("gene", gene_symbol)
    status = result.get("impc_knockout")
    if result.get("error"):
        return f"Error for {marker}: {result['error']}"
    if status is None:
        return f"Gene: {marker} — IMPC knockout: unknown"
    return f"Gene: {marker} — IMPC knockout: {'yes' if status else 'no'}"


def get_impc_significant_phenotypes(gene_symbol: str) -> str:
    """
    Return a human-readable list of significant IMPC phenotypes for a gene.

    Args:
        gene_symbol: Official mouse gene symbol (e.g., "Trp53").
    """
    result = _impc_fetch_significant_phenotypes(gene_symbol)
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
    Return a compact summary containing both knockout status and significant phenotypes.

    Args:
        gene_symbol: Official mouse gene symbol (e.g., "Trp53").
    """
    result = _impc_fetch_significant_phenotypes(gene_symbol)
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


# Update your tool registry:
tool_dict = {
    "add_numbers": add_numbers,
    "convert_mouse_to_human_gene": convert_mouse_to_human_gene,
    "convert_mouse_to_human_ortholog_info": convert_mouse_to_human_ortholog_info,
    "get_impc_knockout_status": get_impc_knockout_status,
    "get_impc_significant_phenotypes": get_impc_significant_phenotypes,
    "get_impc_gene_summary": get_impc_gene_summary,
}
