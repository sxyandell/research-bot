#TOOLS
from db import connector

tool_dict = {
    #TODO: Add tools here
}

def get_gene_info(gene_id: str):
    pass

def get_highest_lod_genes(trait_class: str):
    query = f"SELECT * FROM gene_info WHERE trait_class = '{trait_class}' ORDER BY lod_score DESC LIMIT 10"
    response = connector.query(query)
    return response

def get_gene_info(gene_id: str):
    query = f"SELECT * FROM gene_info WHERE gene_id = '{gene_id}'"
    response = connector.query(query)
    return response