from __future__ import annotations

from typing import Any, Optional


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

    The emptiness checks here only handle NULL and empty string. If you need to
    treat 'NA'/'nan' strings as empty at SQL-level, do a surrounding NULLIF/replace.
    """
    # Quote identifiers
    src = f'"{source_col}"'
    gene = f'"{gene_col}"' if gene_col else 'NULL'
    phen = f'"{phenotype_col}"' if phenotype_col else 'NULL'

    # Gene-level detection using filename: contains 'genes' and not 'isoform'
    cond_gene = f"(lower({src}) LIKE '%genes%' AND lower({src}) NOT LIKE '%isoform%')"

    # Prefer gene on gene rows, else phenotype; fall back if empty
    # DuckDB handles empty string checks with = '' and NULL checks with IS NULL
    expr = (
        "CASE\n"
        f"  WHEN {cond_gene} THEN CASE\n"
        f"    WHEN {gene} IS NOT NULL AND {gene} <> '' THEN {gene}\n"
        f"    WHEN {phen} IS NOT NULL AND {phen} <> '' THEN {phen}\n"
        f"    ELSE NULL\n"
        f"  END\n"
        f"  ELSE CASE\n"
        f"    WHEN {phen} IS NOT NULL AND {phen} <> '' THEN {phen}\n"
        f"    WHEN {gene} IS NOT NULL AND {gene} <> '' THEN {gene}\n"
        f"    ELSE NULL\n"
        f"  END\n"
        f"END"
    )
    return expr 