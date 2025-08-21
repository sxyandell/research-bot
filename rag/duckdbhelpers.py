from __future__ import annotations

from typing import Any, Optional, Iterable, Tuple, List, Dict

import re

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