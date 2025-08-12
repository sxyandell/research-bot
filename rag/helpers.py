# --- IMPC Solr tool (direct HTTP client against EBI IMPC SOLR) ---
from typing import Dict, Any, List
import requests
_IMPC_SOLR_BASE = "https://www.ebi.ac.uk/mi/impc/solr"


def _impc_solr_select(core: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a SOLR select query against the IMPC endpoint for a given core."""
    try:
        url = f"{_IMPC_SOLR_BASE}/{core}/select"
        # Ensure minimal required params
        safe_params: Dict[str, Any] = {"wt": "json", "rows": 50}
        safe_params.update(params or {})
        response = requests.get(url, params=safe_params, timeout=20)
        if response.status_code != 200:
            return {}
        # Some IMPC endpoints return JSON with 'text/plain' content-type; attempt JSON parse regardless
        try:
            return response.json()
        except Exception:
            return {}
    except Exception:
        return {}


def _impc_fetch_significant_phenotypes(gene_symbol: str) -> Dict[str, Any]:
    """Infer knockout and collect significant phenotypes for a gene via IMPC SOLR."""
    symbol = (gene_symbol or "").strip()
    if not symbol:
        return {"gene": gene_symbol, "impc_knockout": None, "significant_phenotypes": [], "error": "Empty gene symbol"}

    # Step 1: verify/normalize symbol using gene core
    verified_symbol = symbol
    res_gene = _impc_solr_select(
        "gene",
        {
            "q": f"marker_symbol:{symbol}",
            "rows": 1,
            "fl": "marker_symbol",
        },
    )
    docs = (res_gene or {}).get("response", {}).get("docs", [])
    if not docs and symbol.upper() != symbol:
        res_gene = _impc_solr_select(
            "gene",
            {
                "q": f"marker_symbol:{symbol.upper()}",
                "rows": 1,
                "fl": "marker_symbol",
            },
        )
        docs = (res_gene or {}).get("response", {}).get("docs", [])
    if docs:
        verified_symbol = docs[0].get("marker_symbol", symbol) or symbol

    # Step 2: collect significant phenotypes from statistical-result core
    phenotype_labels: List[str] = []
    res_stats = _impc_solr_select(
        "statistical-result",
        {
            "q": f"marker_symbol:{verified_symbol} AND significant:true",
            "rows": 200,
            "fl": "mp_term_name,mp_term_id,mp_term_label,top_level_mp_term_name,parameter_name,p_value",
        },
    )
    docs = (res_stats or {}).get("response", {}).get("docs", [])
    for d in docs:
        for key in ("mp_term_name", "mp_term_label", "top_level_mp_term_name", "parameter_name"):
            val = d.get(key)
            if isinstance(val, list):
                for v in val:
                    if v and str(v) not in phenotype_labels:
                        phenotype_labels.append(str(v))
            elif val and str(val) not in phenotype_labels:
                phenotype_labels.append(str(val))

    # Deduplicate while preserving order
    seen = set()
    unique_labels: List[str] = []
    for label in phenotype_labels:
        if label in seen:
            continue
        seen.add(label)
        unique_labels.append(label)

    return {
        "gene": verified_symbol,
        "impc_knockout": bool(unique_labels),
        "significant_phenotypes": unique_labels,
    }