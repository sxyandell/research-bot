from hybrid_qtl_system import HybridQTLSystem
import pprint

# This will connect to the existing database without rebuilding it.
print("Connecting to the Hybrid QTL System...")
system = HybridQTLSystem("/data/dev/miniViewer_3.0/DO1200_liver_genes_all_mice_additive_peaks.csv")

# We need to call setup_vector_store to initialize the connection to the collection.
# Because we changed it to get_or_create_collection, this will be fast and won't rebuild.
system.setup_vector_store(use_google_embeddings=False) 
print("✅ System ready.")

def test_gene_summary(gene_symbol):
    print(f"--- Testing Gene Summary for '{gene_symbol}' ---")
    results = system.semantic_search(
        f"Tell me about the biological function of the gene {gene_symbol}",
        n_results=1,
        # CORRECTED: Wrap multiple conditions in an "$and" operator
        where_filter={"$and": [
            {"type": {"$eq": "gene_summary"}},
            {"gene_symbol": {"$eq": gene_symbol}}
        ]}
    )
    if results:
        pprint.pprint(results)
    else:
        print(f"No gene summary found for '{gene_symbol}'.")

def test_chromosome_query(chromosome):
    print(f"--- Testing Chromosome Summary for Chr {chromosome} ---")
    results = system.semantic_search(
        f"What is happening on chromosome {chromosome}?",
        n_results=1,
        where_filter={"$and": [
            {"type": {"$eq": "chromosome_summary"}},
            {"chromosome": {"$eq": str(chromosome)}}
        ]}
    )
    if results:
        pprint.pprint(results)
    else:
        print(f"No chromosome summary found for chromosome {chromosome}.")

def test_granular_query(gene_symbol, cis_acting=True):
    print(f"--- Testing Granular Peak Search for '{gene_symbol}' (cis={cis_acting}) ---")
    results = system.semantic_search(
        f"Find specific QTLs for the gene {gene_symbol}",
        n_results=3,
        where_filter={"$and": [
            {"type": {"$eq": "qtl_peak"}},
            {"gene_symbol": {"$eq": gene_symbol}},
            {"cis": {"$eq": cis_acting}}
        ]}
    )
    if results:
        pprint.pprint(results)
    else:
        print("No matching peaks found.")

# --- Run some tests ---
print("\n" + "="*50)
test_gene_summary("Gnai3")
print("\n" + "="*50)
test_chromosome_query(2)
print("\n" + "="*50)
test_granular_query("Apoe", cis_acting=True)
print("\n" + "="*50)
test_granular_query("Gsdma3", cis_acting=False)
print("\n" + "="*50) 