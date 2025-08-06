#TOOLS
#from db import connector
import requests
from collections import defaultdict


def add_numbers(num1: int, num2: int):
    """Adds two numbers together.
    Args:
        num1: The first number.
        num2: The second number.
    """
    return num1 + num2

def convert_mouse_to_human_gene(gene_symbol: str):
    """
    Converts a mouse gene symbol to its human homolog(s) using a local JAX labs database file.
    A single mouse gene can map to multiple human homologs.
    The search is case-insensitive.
    Args:
        gene_symbol: The mouse gene symbol to convert.
    """
    local_mapping_file = "HOM_MouseHumanSequence.rpt"
    
    # key: homolog_id, value: {'mouse': [genes], 'human': [genes]}
    homologs = defaultdict(lambda: defaultdict(list))
    
    try:
        with open(local_mapping_file, 'r') as f:
            for line in f:
                if line.startswith("HomoloGene ID"): # Skip header
                    continue
                
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue

                homologene_id = parts[0]
                organism = parts[1]
                symbol = parts[3]

                if organism == "mouse, laboratory":
                    homologs[homologene_id]['mouse'].append(symbol)
                elif organism == "human":
                    homologs[homologene_id]['human'].append(symbol)
    except FileNotFoundError:
        return f"Error: The file '{local_mapping_file}' was not found."

    # Create a mapping from a mouse gene to all its human homologs
    mouse_to_human_map = defaultdict(list)
    for homolog_id, symbols in homologs.items():
        if symbols['mouse'] and symbols['human']:
            for mouse_gene in symbols['mouse']:
                mouse_to_human_map[mouse_gene.lower()].extend(symbols['human'])

    # Find the human homologs for the given mouse gene
    found_genes = mouse_to_human_map.get(gene_symbol.lower())

    if not found_genes:
        return f"No human homologs found for mouse gene '{gene_symbol}'."
    
    unique_genes = sorted(list(set(found_genes)))
    return f"Found human homolog(s) for '{gene_symbol}': {', '.join(unique_genes)}."


tool_dict = {
    "add_numbers": add_numbers,
    "convert_mouse_to_human_gene": convert_mouse_to_human_gene,
}


if __name__ == "__main__":
    print(convert_mouse_to_human_gene("Aldh1l1"))