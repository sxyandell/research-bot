## RAG Subsystem (`rag/`)

A Retrieval-Augmented Generation subsystem for outside API and inside data Q&A and analysis. It orchestrates an LLM (via Ollama) with domain tools for gene/ortholog mapping, GTEx expression, IMPC phenotypes, Ensembl queries, and BioPlex protein interactions. A DuckDB utility builds a unified table from heterogeneous QTL CSVs for future retrieval. In the future we will implement a vector db system through chromadb.

## Project Structure

- `prompts.py` — Defines the system and user prompts guiding tool usage and concise, plain-text responses.
- `data_types.py` — Typed dicts for `Message`, `ToolCall`, and function call shapes.
- `model.py` — Thin wrapper around Ollama's `chat` API, returning assistant messages and tool_calls.
- `tools.py` — Domain tools exposed to the LLM:
  - `convert_mouse_to_human_gene(gene_symbol)`
  - `convert_mouse_to_human_ortholog_info(gene_symbol)`
  - `get_impc_knockout_status(gene_symbol)`
  - `get_impc_significant_phenotypes(gene_symbol)`
  - `get_impc_gene_summary(gene_symbol)`
  - `get_top_tissue_expression(gene_symbol)`
  - `get_ensembl_info(gene_symbol, query_type, species=None)`
  - `get_protein_interactions(gene_symbol)`
  - `tool_dict` maps public tool names to callables
- `helpers.py` — HTTP clients, caching, and data loaders (IMPC Solr, Ensembl REST, GTEx v10 GCT, Tabula Muris, BioPlex, ortholog mapping).
- `chatbot.py` — Conversation orchestrator that feeds messages to the model, executes tool calls, and loops until final assistant output.
- `db.py` — CLI-style utility to build a unified DuckDB table (`qtl_peaks`) from multiple CSVs using `read_csv_auto(..., union_by_name=true)`.
- `api.py` — Placeholder for future FastAPI integration.
- `server.py` — Placeholder for seeing the chatbot potential frontend ui.
- `vector_store.py` — Placeholder for embedding-backed retrieval.

---

## Functionality Overview

### Orthology and Gene Mapping

- `convert_mouse_to_human_gene` reads a local JAX `HOM_MouseHumanSequence.rpt` to map a mouse gene to human homologs (one-to-many supported).
- `convert_mouse_to_human_ortholog_info` returns homolog locus and GRCh38 genomic coordinates per human symbol.
- `_resolve_ortholog_pair` in `helpers.py` infers `(human_symbol, mouse_symbol)` from an arbitrary input symbol using case heuristics and the same map.

### IMPC Phenotyping

- `_impc_fetch_significant_phenotypes` queries IMPC Solr, dedupes phenotype labels, and infers whether knockout data exists.
- Public tools:
  - `get_impc_knockout_status`
  - `get_impc_significant_phenotypes`
  - `get_impc_gene_summary` (combines status + top phenotypes)

### Expression (GTEx and Tabula Muris)

- GTEx:
  - `_fetch_gtex_expression_local` reads a local GTEx v10 GCT file
- Tabula Muris:
  - `_load_tabula_muris_data` loads an AnnData file lazily.
  - `_fetch_tabula_muris_expression` computes mean expression by tissue if data are present.
  - still working on this mouse expression function
- Aggregation:
  - `get_top_tissue_expression` resolves orthologs, returns human tissue expression from GTEx; Tabula Muris is currently disabled in this summary pending fix

### Ensembl REST

- `get_ensembl_info(gene_symbol, query_type, species=None)` supports:
  - `gene_info`, `variants`, `transcripts`, `phenotype`, `regulation`.
- Species inference and normalization are handled by `_infer_species_from_gene` and `_normalize_species`.

### BioPlex Protein Interactions

- `_bioplex_fetch_interactions` fetches interactions with version fallback (293T: 3.0→2.0→1.0; HCT116: 1.0).
- `get_protein_interactions` returns interactors for a gene across 293T and HCT116.

### QTL Data Ingestion (DuckDB)

- `db.py` discovers CSVs, aligns columns by name, and builds a canonical wide table with derived flags:
  - `QTL by Covar Scan`, `Split-by Scan`, `Full Scan`, and `Source` (from filename).

---

## Usage

### Prerequisites

- Python 3.10+
- Ollama installed and running (`ollama serve`)
- Recommended Python packages:
  - Core: `pip install ollama duckdb requests pandas`
  - Expression (optional): `pip install numpy anndata scanpy`
  - BioPlex (optional): `pip install --no-cache-dir --upgrade numpy scipy anndata bioplexpy`

### Environment Configuration (optional)

- `HOM_HUMAN_MOUSE_RPT` — Path to `HOM_MouseHumanSequence.rpt` (default: project root).
- `GTEX_V10_GCT` — Path to `GTEx_Analysis_v10_..._gene_median_tpm.gct.gz` (enables local GTEx reads).
- `TABULA_MURIS_H5AD` — Path to `tabula-muris.h5ad` (enables local mouse expression). - working on this
- `HTTP_USER_AGENT` — Custom user agent for outgoing requests (ollama qwen)

### Start the model backend

```bash
ollama serve
ollama pull qwen3:8b
```

### Programmatic chatbot usage

```python
from rag.chatbot import Chatbot
from rag.tools import tool_dict

chat = Chatbot("qwen3:8b", tool_dict)
print(chat.chat("Find human homologs for Trp53 and list top GTEx tissues."))
```

### CLI-style chat loop

```bash
python -m rag.chatbot
# or
python rag/chatbot.py
```

### Run QTL DuckDB build

- Configure input glob and output paths at the top of `rag/db.py`.

```bash
python -m rag.db
# or
python rag/db.py
```

### Manual tool testing

```bash
python rag/tools.py
```

---

## Current Progress

- Chat orchestration wired through Ollama with tool-calling loop.
- Orthology, IMPC, Ensembl, GTEx, and BioPlex tools implemented
- Tabula Muris loader implemented; mouse expression disabled in the combined summary for now
- DuckDB table builder operational with configurable CSV discovery.
- API surface and vector store are placeholders pending integration.

---

## Next Steps

- [ ] Unify tool output formatting with the plain-text response policy.
- [ ] Re-enable and optimize Tabula Muris integration in `get_top_tissue_expression`.
- [ ] Add vector store and embeddings pipeline for retrieval over QTL.
- [ ] Refine a FastAPI service (`rag/api.py`) with chat and tool endpoints for stateless use.
- [ ] Configuration via `.env` and CLI flags for `db.py`.
- [ ] Develop refined frontend using NextJS.

---

## Contributions

### Kalynn:

- Architected new tool usage system with ollama and built out the new RAG system.
- Created human ortholog, human expression, impc, protein interaction, and tissue tools.
- Built a temporary ui to see the chatbot progress (server.py) with FastAPI and JS.
- Architected and built the duckdb data store.
- Set up Jira to monitor progress and stay on-track.

### Sarah

- Built and integrated the orginal chatbot system
- Implemented embeddings and vector store infrastructure for semantic search
- Developed Ensembl REST API integration tools for genomic data queries
- Created GWAS Catalog integration tools for human genetic association studies
- Built testing infrastructure for tools and functions
