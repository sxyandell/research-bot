#TOOLS


#from db import connector

#def get_gene_info(gene_id: str):
    #pass

#def get_highest_lod_genes(trait_class: str):
    #query = f"SELECT * FROM gene_info WHERE trait_class = '{trait_class}' ORDER BY lod_score DESC LIMIT 10"
    #response = connector.query(query)
    #return response

def add_numbers(num1: int, num2: int):
    return num1 + num2

tool_dict = {
    "add_numbers": add_numbers
}