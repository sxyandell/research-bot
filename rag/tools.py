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

# Update your tool registry:
tool_dict = {
    "add_numbers": add_numbers,
    "convert_mouse_to_human_gene": convert_mouse_to_human_gene,
    "convert_mouse_to_human_ortholog_info": convert_mouse_to_human_ortholog_info,
}

if __name__ == "__main__":
    print(convert_mouse_to_human_ortholog_info("Aldh1l1"))