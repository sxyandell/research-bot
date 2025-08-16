# --- IMPC Solr tool (direct HTTP client against EBI IMPC SOLR) ---
from typing import Dict, Any, List
from pathlib import Path
from functools import lru_cache
from typing import Tuple, Optional
import os
_IMPC_SOLR_BASE = "https://www.ebi.ac.uk/mi/impc/solr"


def _impc_solr_select(core: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a SOLR select query against the IMPC endpoint for a given core."""
    try:
        # Lazy import to avoid module import failure if requests is absent
        import requests  # type: ignore
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

# Paths/configs used by helpers
_TABULA_MURIS_PATH = Path(os.getenv("TABULA_MURIS_H5AD", str(Path(__file__).resolve().parent.parent / "data" / "tabula-muris.h5ad")))
_GTEXT_V10_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz"
_GTEX_V10_GCT_LOCAL = Path(os.getenv("GTEX_V10_GCT", str(_GTEXT_V10_DEFAULT)))


# Cached helper to resolve ortholog pairs (human↔mouse) as structured values
@lru_cache(maxsize=1)
def _load_homolog_pairs_df():
    import pandas as pd  # lazy import
    mapping_file = Path(__file__).resolve().parent.parent / "HOM_MouseHumanSequence.rpt"
    df = pd.read_csv(
        mapping_file,
        sep="\t",
        usecols=["DB Class Key", "Common Organism Name", "Symbol"],
        dtype=str,
        comment="#",
    ).rename(columns={
        "DB Class Key": "homologene_id",
        "Common Organism Name": "organism",
        "Symbol": "symbol",
    })
    df_mouse = (
        df[df.organism == "mouse, laboratory"]
        .assign(mouse_symbol_orig=lambda d: d.symbol,
                mouse_symbol_lower=lambda d: d.symbol.str.lower())
        .loc[:, ["homologene_id", "mouse_symbol_orig", "mouse_symbol_lower"]]
    )
    df_human = (
        df[df.organism == "human"]
        .assign(human_symbol_upper=lambda d: d.symbol.str.upper())
        .loc[:, ["homologene_id", "human_symbol_upper"]]
    )
    pairs = df_mouse.merge(df_human, on="homologene_id", how="inner")
    return pairs


def _resolve_ortholog_pair(input_symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """Return a tuple (human_symbol, mouse_symbol) for a given input symbol.
    Heuristic: all-uppercase ⇒ human; else ⇒ mouse. Falls back gracefully if not found.
    """
    sym = (input_symbol or "").strip()
    if not sym:
        return (None, None)
    pairs = _load_homolog_pairs_df()
    # Likely human symbol (e.g., TP53)
    if sym.isupper() and sym.upper() != sym.lower():
        rows = pairs[pairs["human_symbol_upper"] == sym.upper()]
        human_symbol = sym.upper()
        mouse_symbol = rows["mouse_symbol_orig"].iloc[0] if not rows.empty else None
        return (human_symbol, mouse_symbol)
    # Treat as mouse symbol (capitalize convention but match case-insensitively)
    rows = pairs[pairs["mouse_symbol_lower"] == sym.lower()]
    mouse_symbol = rows["mouse_symbol_orig"].iloc[0] if not rows.empty else (sym[0].upper() + sym[1:].lower())
    human_symbol = rows["human_symbol_upper"].iloc[0] if not rows.empty else None
    return (human_symbol, mouse_symbol)


# Cached loader for GTEx v10 GCT
@lru_cache(maxsize=1)
def _load_gtex_v10_gct() -> Optional[object]:
    try:
        import gzip
        import pandas as pd  # lazy import
        if not _GTEX_V10_GCT_LOCAL.exists():
            return None
        with gzip.open(_GTEX_V10_GCT_LOCAL, "rb") as gz:
            df = pd.read_csv(gz, sep="\t", header=2)
        return df
    except Exception:
        return None


def _fetch_gtex_expression_local(gene_symbol: str) -> List[Tuple[str, float]]:
    """Lookup median TPM by tissue from local GTEx v10 GCT file.
    Accepts a gene symbol (case-insensitive) or Ensembl gene ID (with or without version).
    Returns top 10 tissues by median TPM.
    """
    symbol = (gene_symbol or "").strip()
    if not symbol:
        return []
    df = _load_gtex_v10_gct()
    if df is None or getattr(df, 'empty', False):
        return []

    # Normalize match candidates
    query_upper = symbol.upper()
    sub = None
    if "Description" in df.columns:
        try:
            sub = df[df["Description"].astype(str).str.upper() == query_upper]
        except Exception:
            sub = None
    if (sub is None or sub.empty) and (symbol.upper().startswith("ENSG")) and ("Name" in df.columns):
        base_id = symbol.split(".")[0]
        try:
            sub = df[df["Name"].astype(str).str.split(".").str[0] == base_id]
        except Exception:
            sub = None
    if sub is None or sub.empty:
        return []

    row = sub.iloc[0]
    ignore_cols = {"Name", "Description"}
    tissue_cols = [c for c in df.columns if c not in ignore_cols]
    values: List[Tuple[str, float]] = []
    for col in tissue_cols:
        try:
            values.append((str(col).replace("_", " - "), float(row[col])))
        except Exception:
            continue
    values.sort(key=lambda x: x[1], reverse=True)
    return values[:10]


# ---------------- Ensembl REST helpers (used by tools) ----------------
_ENSEMBL_BASE = os.getenv("ENSEMBL_REST_BASE", "https://rest.ensembl.org")


def _ensembl_request(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Optional[Any]:
    """Perform a GET to Ensembl REST with retries and proper headers.

    Returns parsed JSON (dict or list) or None on failure.
    """
    try:
        import time
        import requests  # type: ignore
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))
        session.mount("http://", HTTPAdapter(max_retries=retries))

        headers = {
            "Accept": "application/json",
            "User-Agent": os.getenv("HTTP_USER_AGENT", "research-bot/1.0 (Ensembl client)"),
        }
        url = f"{_ENSEMBL_BASE.rstrip('/')}/{path.lstrip('/')}"
        resp = session.get(url, params=params or {}, headers=headers, timeout=timeout)
        # Handle 429 with explicit sleep if Retry-After provided
        if resp.status_code == 429:
            try:
                retry_after = float(resp.headers.get("Retry-After", "1"))
                time.sleep(min(2.0, max(0.0, retry_after)))
                resp = session.get(url, params=params or {}, headers=headers, timeout=timeout)
            except Exception:
                pass
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except Exception:
            return None
    except Exception:
        return None


@lru_cache(maxsize=512)
def _ensembl_lookup_gene_id(species: str, gene_symbol: str) -> Optional[str]:
    """Resolve a gene symbol to Ensembl stable ID via /lookup/symbol.
    Cached for performance.
    """
    sp = (species or "").strip() or "mus_musculus"
    sym = (gene_symbol or "").strip()
    if not sym:
        return None
    data = _ensembl_request(f"/lookup/symbol/{sp}/{sym}")
    if isinstance(data, dict):
        gid = data.get("id") or data.get("stable_id")
        return str(gid) if gid else None
    return None

# Species helpers
_SPECIES_ALIASES = {
    "mus_musculus": {"mouse", "mus_musculus", "mus", "mm", "mmusculus", "m_musculus"},
    "homo_sapiens": {"human", "homo_sapiens", "hs", "hsa", "h_sapiens", "homo sapiens"},
}


def _normalize_species(species: Optional[str]) -> Optional[str]:
    """Normalize species names/aliases to Ensembl identifiers.
    Returns 'mus_musculus', 'homo_sapiens', or None if unknown.
    """
    if not species:
        return None
    val = species.strip().lower().replace("-", "_").replace(" ", "_")
    for key, aliases in _SPECIES_ALIASES.items():
        if val == key or val in aliases:
            return key
    return None


def _infer_species_from_gene(gene_symbol: str) -> Optional[str]:
    """Infer species from common patterns.
    - ENSG* ⇒ human
    - ENSMUSG* ⇒ mouse
    - Heuristic: ALLCAPS likely human symbols; TitleCase likely mouse.
    """
    sym = (gene_symbol or "").strip()
    if not sym:
        return None
    u = sym.upper()
    if u.startswith("ENSG"):
        return "homo_sapiens"
    if u.startswith("ENSMUSG"):
        return "mus_musculus"
    if sym.isupper() and u != sym.lower():
        return "homo_sapiens"
    # default heuristic: treat as mouse-like symbol
    return "mus_musculus"

# ---------------- BioPlex helpers (used by tools) ----------------
@lru_cache(maxsize=8)
def _bioplex_fetch_interactions(cell_line: str):
    """Fetch BioPlex interactions for a given cell line with version fallback.

    Returns a pandas DataFrame with standardized columns ['SymbolA','SymbolB']
    and symbols uppercased. Returns None if unavailable.
    """
    try:
        import pandas as pd  # type: ignore
        from bioplexpy import getBioPlex as _getBioPlex  # type: ignore
    except Exception:
        return None

    try:
        interactions_df = None
        versions = ("3.0", "2.0", "1.0") if str(cell_line).upper() == "293T" else ("1.0",)
        for v in versions:
            try:
                df = _getBioPlex(str(cell_line), v)
                if df is not None:
                    interactions_df = df
                    break
            except Exception:
                continue
        if interactions_df is None:
            return None

        cols = {c.lower(): c for c in interactions_df.columns}
        a = cols.get('symbola') or cols.get('sym_a') or cols.get('genea')
        b = cols.get('symbolb') or cols.get('sym_b') or cols.get('geneb')
        if not a or not b:
            return None
        df2 = interactions_df.rename(columns={a: 'SymbolA', b: 'SymbolB'})[['SymbolA', 'SymbolB']].copy()
        df2['SymbolA'] = df2['SymbolA'].astype(str).str.upper()
        df2['SymbolB'] = df2['SymbolB'].astype(str).str.upper()
        return df2
    except Exception:
        return None


def _bioplex_interactors_for_symbol(interactions_df, symbol: str) -> List[str]:
    """Given standardized BioPlex interactions DataFrame, return unique interactors for symbol."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return []
    if interactions_df is None:
        return []
    sym = (symbol or "").strip().upper()
    if not sym:
        return []
    try:
        series = pd.concat([
            interactions_df.loc[interactions_df['SymbolA'] == sym, 'SymbolB'],
            interactions_df.loc[interactions_df['SymbolB'] == sym, 'SymbolA']
        ])
        return series.dropna().astype(str).unique().tolist()
    except Exception:
        return []