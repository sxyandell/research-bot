from __future__ import annotations

from typing import Any, Optional, Iterable, Tuple, List, Dict

import re


def is_empty_value(value: Any) -> bool:
    """
    Return True if the value is effectively empty: None, empty string, or common 'NA' markers.
    """
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"na", "n/a", "nan"}


def is_gene_source(source: Optional[str]) -> bool:
    """
    Heuristic: treat a row as gene-level if source mentions 'genes' and does not
    mention 'isoform'. Source is typically the basename of the CSV.
    """
    if not source:
        return False
    s = str(source).lower()
    return ("genes" in s) and ("isoform" not in s)


def pick_gene_or_phenotype(source: Optional[str], gene_symbol: Any, phenotype: Any) -> Optional[str]:
    """
    Prefer gene symbol only for gene-level rows; otherwise use phenotype.
    If the preferred value is empty or null, fall back to the other.

    Returns the chosen value (string) or None if both are empty.
    """
    gene_row = is_gene_source(source)
    gene_val = None if is_empty_value(gene_symbol) else str(gene_symbol)
    pheno_val = None if is_empty_value(phenotype) else str(phenotype)

    if gene_row:
        # Prefer gene for gene-level; fall back to phenotype
        return gene_val or pheno_val
    # Non-gene rows: prefer phenotype
    return pheno_val or gene_val


def build_gene_or_phenotype_case_expr(source_col: str, gene_col: Optional[str], phenotype_col: Optional[str]) -> str:
    """
    Build a DuckDB SQL CASE expression that returns the appropriate display text
    for each row: gene symbol for gene-level sources, else phenotype. If the
    preferred value is empty, falls back to the other.

    The emptiness checks here handle NULL, empty string, and common NA markers
    ('NA', 'N/A', 'nan' case-insensitively).
    """
    # Quote identifiers
    src = f'"{source_col}"'
    gene = f'"{gene_col}"' if gene_col else 'NULL'
    phen = f'"{phenotype_col}"' if phenotype_col else 'NULL'

    # Create cleaned versions that coerce NA-like and empty values to NULL
    gene_clean = (
        f"CASE WHEN {gene} IS NULL OR {gene} = '' OR lower({gene}) IN ('na','n/a','nan') THEN NULL ELSE {gene} END"
    )
    phen_clean = (
        f"CASE WHEN {phen} IS NULL OR {phen} = '' OR lower({phen}) IN ('na','n/a','nan') THEN NULL ELSE {phen} END"
    )

    # Gene-level detection using filename: contains 'genes' and not 'isoform'
    cond_gene = f"(lower({src}) LIKE '%genes%' AND lower({src}) NOT LIKE '%isoform%')"

    # Prefer gene on gene rows, else phenotype; fall back if empty (after cleaning)
    expr = (
        "CASE\n"
        f"  WHEN {cond_gene} THEN CASE\n"
        f"    WHEN {gene_clean} IS NOT NULL THEN {gene_clean}\n"
        f"    WHEN {phen_clean} IS NOT NULL THEN {phen_clean}\n"
        f"    ELSE NULL\n"
        f"  END\n"
        f"  ELSE CASE\n"
        f"    WHEN {phen_clean} IS NOT NULL THEN {phen_clean}\n"
        f"    WHEN {gene_clean} IS NOT NULL THEN {gene_clean}\n"
        f"    ELSE NULL\n"
        f"  END\n"
        f"END"
    )
    return expr


# ---------------- LOD helpers ----------------
DEFAULT_LOD_COLUMNS: Tuple[str, ...] = ("lod_diff", "qtl_lod", "lod_add", "lod_int", "lod")


def build_lod_value_and_source_expr(
    available_columns: Optional[Iterable[str]] = None,
    prefer_order: Iterable[str] = DEFAULT_LOD_COLUMNS,
) -> Tuple[str, str]:
    """
    Return a pair of SQL expressions (lod_value_expr, lod_source_expr):
    - lod_value_expr: COALESCE over the present LOD-like columns, cast to DOUBLE
    - lod_source_expr: CASE label indicating which column contributed the value

    Use this inside a SELECT, e.g.:
      val_expr, src_expr = build_lod_value_and_source_expr(cols)
      SELECT ..., ({val_expr}) AS lod_value, ({src_expr}) AS lod_source ...

    Args:
    - available_columns: iterable of table column names; if None, assumes all in prefer_order
    - prefer_order: priority order for LOD-like columns
    """
    present = {c.lower() for c in (available_columns or prefer_order)}
    order = [c for c in prefer_order if c.lower() in present]
    if not order:
        # Fallback to default names even if unknown; the query will fail fast if truly absent
        order = list(DEFAULT_LOD_COLUMNS)

    # Build expressions
    try_casts = [f'TRY_CAST("{c}" AS DOUBLE)' for c in order]
    value_expr = f"COALESCE({', '.join(try_casts)})"
    cases = [f"WHEN TRY_CAST(\"{c}\" AS DOUBLE) IS NOT NULL THEN '{c}'" for c in order]
    source_expr = f"CASE {' '.join(cases)} ELSE NULL END"
    return value_expr, source_expr


def pick_lod_source_from_row(row: dict[str, Any], prefer_order: Iterable[str] = DEFAULT_LOD_COLUMNS) -> Optional[str]:
    """
    Given a row dict, return the first LOD-like column name that has a non-null, numeric value.
    """
    for col in prefer_order:
        val = row.get(col)
        if val is None:
            continue
        try:
            float(val)
            return col
        except Exception:
            continue
    return None


# ---------------- Position/Chromosome helpers ----------------


def build_position_mb_expr(lower_to_orig: Dict[str, str], original_names: List[str]) -> Optional[str]:
    """
    Return a SQL expression that yields position in megabases.
    - If 'qtl_pos' exists, cast it to DOUBLE.
    - Else coalesce across MB/BP/generic position-like columns, converting BP to MB.
    """
    if "qtl_pos" in lower_to_orig:
        return f'TRY_CAST("{lower_to_orig["qtl_pos"]}" AS DOUBLE)'

    pos_mb_terms: List[str] = []
    pos_bp_terms: List[str] = []
    pos_generic_terms: List[str] = []

    for name in original_names:
        ln = name.lower()
        if "lod" in ln:
            continue
        if "mb" in ln:
            pos_mb_terms.append(name)
        elif re.search(r"\bbp\b", ln) or ln.endswith("_bp") or "(bp)" in ln:
            pos_bp_terms.append(name)
        elif ("position" in ln) or re.search(r"\bpos\b", ln) or ln.endswith("_pos") or ln.endswith("_position"):
            pos_generic_terms.append(name)

    parts: List[str] = []
    for col in pos_mb_terms:
        parts.append(f'TRY_CAST("{col}" AS DOUBLE)')
    for col in pos_bp_terms:
        parts.append(f'(TRY_CAST("{col}" AS DOUBLE) / 1000000.0)')
    for col in pos_generic_terms:
        parts.append(f'TRY_CAST("{col}" AS DOUBLE)')

    if not parts:
        return None
    return f"COALESCE({', '.join(parts)})"


def build_normalized_chromosome_expr(chrom_col: str) -> str:
    """
    Normalize a chromosome column to lowercase and strip a leading 'chr'.
    """
    chrom_txt_expr = f'lower(CAST("{chrom_col}" AS VARCHAR))'
    return f"regexp_replace({chrom_txt_expr}, '^chr', '')"


def sanitize_sql_literal(value: str) -> str:
    """Escape single quotes for safe SQL literal insertion."""
    return (value or "").replace("'", "''") 