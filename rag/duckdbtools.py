from pathlib import Path
from typing import Any, List, Optional
import os
import re
import duckdb

# Import DuckDB helpers
try:
    from rag.duckdbhelpers import (
        build_lod_value_and_source_expr,
        build_position_mb_expr,
        build_normalized_chromosome_expr,
        sanitize_sql_literal,
    )
except ImportError:
    from duckdbhelpers import (
        build_lod_value_and_source_expr,
        build_position_mb_expr,
        build_normalized_chromosome_expr,
        sanitize_sql_literal,
    )

def _choose_label(source: Optional[str], gene_symbol: Any, phenotype: Any) -> Optional[str]:
    """Helper to choose a display label based on source and available data."""
    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        s = str(value).strip()
        return s == "" or s.lower() in {"na", "n/a", "nan"}

    src = str(source).lower() if source else ""
    is_gene_source = ("genes" in src) and ("isoform" not in src)
    gene_val = str(gene_symbol) if not _is_empty(gene_symbol) else None
    pheno_val = str(phenotype) if not _is_empty(phenotype) else None
    
    if is_gene_source:
        return gene_val or pheno_val
    return pheno_val or gene_val

class QTLPeakSearch:
    """A consolidated tool for searching QTL peaks in a DuckDB database."""

    def __init__(self, db_path: Optional[str] = None, table_name: str = "qtl_peaks"):
        self.db_path = db_path or os.getenv("QTL_DUCKDB_PATH") or os.getenv("QTL_DUCKDB_FILE") or str(Path(__file__).resolve().parent.parent / "data" / "qtl_database.duckdb")
        self.table_name = table_name

    def _connect(self):
        """Helper to connect to the database and get schema info."""
        if not os.path.exists(self.db_path):
            return None, f"Error: DuckDB database not found at {self.db_path}"
        try:
            con = duckdb.connect(self.db_path, read_only=True)
            schema_rows = con.execute(f"PRAGMA table_info('{self.table_name}')").fetchall()
            if not schema_rows:
                return con, f"Error: Table '{self.table_name}' not found."
            
            lower_to_orig = {str(r[1]).lower(): str(r[1]) for r in schema_rows}
            return con, lower_to_orig
        except Exception as e:
            return None, f"Error: Could not open DuckDB database. Details: {e}"

    def _run_query(self, sql: str) -> tuple[Optional[list], Optional[str]]:
        """Executes a SQL query and handles errors."""
        con, result = self._connect()
        if not con:
            return None, result
        try:
            rows = con.execute(sql).fetchall() or []
            return rows, None
        except Exception as e:
            return None, f"Error: Query failed. Details: {e}"
        finally:
            con.close()

    def _get_phenotype_class(self, query: str) -> Optional[str]:
        """Infers phenotype class from query keywords."""
        ql = query.lower()
        if any(k in ql for k in ["clinical", "trait", "phenotype"]):
            return "clinical_trait"
        elif any(k in ql for k in ["metabolite", "metabolomics", "plasma"]):
            return "plasma_metabolite"
        elif any(k in ql for k in ["lipid", "fats"]):
            return "liver_lipid"
        elif any(k in ql for k in ["gene", "genes"]):
            return "liver_gene"
        elif any(k in ql for k in ["isoform", "isoforms"]):
            return "liver_isoform"
        elif any(k in ql for k in ["junction", "splice", "junc"]):
            return "liver_splice_junc"
        return None

    def search(self, query: str, limit: int = 200) -> str:
        """
        The main public method for searching QTL peaks.
        Handles text, positional, and "top peaks" queries.
        """
        if not query.strip():
            return "Error: Query cannot be empty."
        
        try:
            # Check for required modules first
            import duckdb
        except ImportError:
            return "Error: duckdb is not installed. Install with 'pip install duckdb'."

        con, lower_to_orig = self._connect()
        if not con:
            return lower_to_orig # error message
        con.close() # close early, we'll reconnect later

        lod_expr, _ = build_lod_value_and_source_expr(list(lower_to_orig.keys()))
        if not lod_expr:
            return "Error: No LOD-like columns found."

        # 1. Check for "top peaks" queries
        ql = query.lower()
        if ql.startswith("top"):
            m = re.search(r"top\s+([0-9]+)", ql)
            top_limit = int(m.group(1)) if m else 10
            pc = self._get_phenotype_class(query)
            return self._get_top_peaks(limit=top_limit, phenotype_class=pc)

        # 2. Check for genomic coordinate queries
        coord_re = re.compile(r"(?:chr(?:omosome)?\s*)?([0-9xyYmM]+)\s*(?::|\bat\b|@|\bposition\b|\bpos\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*(mbp?|mb|bp)?", re.IGNORECASE)
        m = coord_re.search(query)
        if m:
            chrom = m.group(1)
            pos = float(m.group(2))
            unit = (m.group(3) or "mb").lower()
            win_kb = None
            win_m = re.search(r"(?:within|window|±|\+\-)\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mbp?)", query, re.IGNORECASE)
            if win_m:
                window_val = float(win_m.group(1))
                window_unit = win_m.group(2).lower()
                win_kb = window_val * 1000.0 if window_unit.startswith("mb") else window_val
            
            pc = self._get_phenotype_class(query)
            return self._search_by_position(
                chromosome=chrom, position=pos, unit=unit,
                window_kb=win_kb or 4000.0, phenotype_class=pc, limit=limit
            )
            
        # 3. Fallback to text search
        return self._search_by_text(query, limit)

    def _get_top_peaks(self, limit: int, phenotype_class: Optional[str] = None) -> str:
        """Internal method for "top peaks" queries, with optional filtering."""
        con, lower_to_orig = self._connect()
        if not con: return lower_to_orig
        
        lod_expr, lod_source_expr = build_lod_value_and_source_expr(list(lower_to_orig.keys()))
        source_col = lower_to_orig.get("source", "Source")
        gene_col = lower_to_orig.get("gene_symbol")
        phenotype_col = lower_to_orig.get("phenotype")

        where_clause = f"{lod_expr} IS NOT NULL"
        if phenotype_class and "phenotype_class" in lower_to_orig:
            pc_col = lower_to_orig["phenotype_class"]
            where_clause += f" AND lower(\"{pc_col}\") = '{sanitize_sql_literal(phenotype_class.lower())}'"

        sql = (
            f"SELECT \"{source_col}\" AS source, \"{gene_col}\" AS gene_symbol, \"{phenotype_col}\" AS phenotype, "
            f"{lod_expr} AS lod_value, {lod_source_expr} AS lod_source "
            f"FROM \"{self.table_name}\" "
            f"WHERE {where_clause} "
            f"ORDER BY lod_value DESC "
            f"LIMIT {int(limit)}"
        )
        
        rows, error = self._run_query(sql)
        if error: return error
        if not rows: return "No top peaks found."
        
        header = f"**Top {min(len(rows), limit)} LOD-like Peaks"
        if phenotype_class: header += f" for {phenotype_class.replace('_', ' ')}"
        header += "**\n\n"
        
        lines = []
        for idx, (source, gene_symbol, phenotype, lod_value, lod_source) in enumerate(rows, start=1):
            lod_num = float(lod_value) if lod_value is not None else float('nan')
            label = _choose_label(source, gene_symbol, phenotype) or "Unknown"
            lod_from = lod_source or "unknown"
            lines.append(f"{idx}. LOD {lod_num:.3f} ({lod_from}) — {label} — Source: {source or 'Unknown'}")
        
        return header + "\n".join(lines)

    def _search_by_text(self, query: str, limit: int) -> str:
        """Internal method for general text queries."""
        con, lower_to_orig = self._connect()
        if not con: return lower_to_orig
        
        lod_expr, lod_source_expr = build_lod_value_and_source_expr(list(lower_to_orig.keys()))
        
        q_lower = query.lower()
        q_esc = sanitize_sql_literal(q_lower)
        is_gene_like = ("_" not in query) and (" " not in query)
        
        where_clause = f"lower(CAST(\"phenotype\" AS VARCHAR)) LIKE '%{q_esc}%'"
        if is_gene_like:
            where_clause = f"lower(CAST(\"gene_symbol\" AS VARCHAR)) LIKE '%{q_esc}%'"
        
        source_col = lower_to_orig.get("source", "Source")
        gene_col = lower_to_orig.get("gene_symbol")
        phen_col = lower_to_orig.get("phenotype")
        chrom_expr = build_normalized_chromosome_expr(lower_to_orig.get("qtl_chr", "qtl_chr"))
        pos_expr = build_position_mb_expr(lower_to_orig, list(lower_to_orig.values())) or "NULL"

        sql = (
            f"SELECT \"{source_col}\" AS source, \"{gene_col}\" AS gene_symbol, \"{phen_col}\" AS phenotype, "
            f"{lod_expr} AS lod_value, {lod_source_expr} AS lod_source, "
            f"{chrom_expr} AS norm_chr, {pos_expr} AS pos_mb "
            f"FROM \"{self.table_name}\" "
            f"WHERE ({where_clause}) AND {lod_expr} IS NOT NULL "
            f"ORDER BY lod_value DESC "
            f"LIMIT {int(limit)}"
        )
        
        rows, error = self._run_query(sql)
        if error: return error
        if not rows: return f"No peaks found matching '{query}'."

        header = f"**QTL Peaks matching: {query}**"
        lines = []
        for idx, (source, gene_symbol, phenotype, lod_value, lod_source, chrom, pos) in enumerate(rows, start=1):
            lod_num = float(lod_value) if lod_value is not None else float('nan')
            label = _choose_label(source, gene_symbol, phenotype) or "Unknown"
            coord_txt = f" — chr{chrom}:{float(pos):.3f} Mb" if chrom and pos else ""
            lines.append(f"{idx}. LOD {lod_num:.3f} ({lod_source or 'unknown'}) — {label}{coord_txt} — Source: {source or 'Unknown'}")
        
        return header + "\n" + "\n".join(lines)

    def _search_by_position(
        self, chromosome: str, position: float, unit: str, window_kb: float, phenotype_class: Optional[str], limit: int
    ) -> str:
        """Internal method for positional queries."""
        con, lower_to_orig = self._connect()
        if not con: return lower_to_orig
        
        if "qtl_chr" not in lower_to_orig:
            return "Error: Required column 'qtl_chr' not found."
        
        target_mb = position / 1e6 if unit == "bp" else position
        window_mb = window_kb / 1000.0
        
        chrom_in = str(chromosome).strip().lower().replace("chr", "").replace(" ", "")
        chrom_col = lower_to_orig["qtl_chr"]
        pos_expr = build_position_mb_expr(lower_to_orig, list(lower_to_orig.values()))
        
        lod_expr, _ = build_lod_value_and_source_expr(list(lower_to_orig.keys()))
        source_col = lower_to_orig.get("source", "Source")
        gene_col = lower_to_orig.get("gene_symbol")
        phen_col = lower_to_orig.get("phenotype")
        pc_col = lower_to_orig.get("phenotype_class")

        where_extra = ""
        if phenotype_class and pc_col:
            where_extra = f" AND lower(\"{pc_col}\") = '{sanitize_sql_literal(phenotype_class.lower())}'"
            
        sql = (
            f"SELECT \"{source_col}\" AS source, \"{gene_col}\" AS gene_symbol, \"{phen_col}\" AS phenotype, "
            f"\"{pc_col}\" AS phenotype_class, {pos_expr} AS pos_mb, {lod_expr} AS lod_value "
            f"FROM \"{self.table_name}\" "
            f"WHERE {build_normalized_chromosome_expr(chrom_col)} = '{sanitize_sql_literal(chrom_in)}' "
            f"  AND {pos_expr} BETWEEN {target_mb - window_mb} AND {target_mb + window_mb} "
            f"  AND {lod_expr} IS NOT NULL"
            f"{where_extra} "
            f"ORDER BY ABS({pos_expr} - {target_mb}) ASC "
            f"LIMIT {limit * 10}" # fetch more to allow for deduplication
        )
        
        rows, error = self._run_query(sql)
        if error: return error
        if not rows: return f"No QTL peaks found near chr{chrom_in}:{target_mb:.3f} Mb."

        seen_labels = set()
        unique_rows = []
        for r in rows:
            label = _choose_label(r[0], r[1], r[2])
            if label and label not in seen_labels:
                seen_labels.add(label)
                unique_rows.append(r)
            if len(unique_rows) >= limit: break
            
        header = f"**QTL Peaks near chr{chrom_in}:{target_mb:.3f} Mb ± {window_mb:.3f} Mb**\n\n"
        lines = [f"{idx}. {_choose_label(r[0], r[1], r[2])} — pos {float(r[4]):.6f} Mb (LOD {float(r[5]):.3f})" for idx, r in enumerate(unique_rows, 1)]
        
        return header + "\n".join(lines)


# External entry point
qtl_search_tool = QTLPeakSearch()
search_qtl_peaks = qtl_search_tool.search


# Compatibility aliases expected by rag/tools.py

def get_top_lod_peaks(limit: int = 10, phenotype_class: Optional[str] = None, **kwargs) -> str:
	"""Alias: return top LOD-like peaks, optionally filtered by phenotype_class.
	Also tolerates extraneous kwargs (e.g., gene_symbol) by rerouting to text search.
	"""
	# If a gene-like hint is provided, route to general search for that query
	gene_hint = kwargs.get("gene_symbol") or kwargs.get("gene") or kwargs.get("symbol")
	if gene_hint:
		return qtl_search_tool.search(query=str(gene_hint), limit=limit)
	return qtl_search_tool._get_top_peaks(limit=limit, phenotype_class=phenotype_class)


def search_qtl_by_genomic_position(
	chromosome: str,
	position: float,
	unit: str = "mb",
	window_kb: float = 4000.0,
	limit: int = 20,
	phenotype_class: Optional[str] = None,
) -> str:
	"""Alias: positional search delegating to the consolidated implementation."""
	unit_norm = (unit or "mb").lower()
	return qtl_search_tool._search_by_position(
		chromosome=chromosome,
		position=float(position),
		unit='bp' if unit_norm == 'bp' else 'mb',
		window_kb=float(window_kb),
		phenotype_class=phenotype_class,
		limit=int(limit),
	)


def find_traits_near_locus(query: str, default_window_kb: float = 4000.0, limit: int = 20) -> str:
	"""Alias: free-text locus search; default window is handled by the parser if omitted."""
	return qtl_search_tool.search(query=query, limit=limit)


def search_clinical_traits_by_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return qtl_search_tool.search(
		query=f"clinical traits near chr{chromosome}:{position} {unit} within {window_kb/1000.0} Mb",
		limit=limit,
	)


def search_metabolites_by_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return qtl_search_tool.search(
		query=f"plasma metabolites near chr{chromosome}:{position} {unit} within {window_kb/1000.0} Mb",
		limit=limit,
	)


def search_lipids_by_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return qtl_search_tool.search(
		query=f"liver lipids near chr{chromosome}:{position} {unit} within {window_kb/1000.0} Mb",
		limit=limit,
	)


def search_genes_by_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return qtl_search_tool.search(
		query=f"liver genes near chr{chromosome}:{position} {unit} within {window_kb/1000.0} Mb",
		limit=limit,
	)


def search_liver_isoforms_by_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return qtl_search_tool.search(
		query=f"liver isoforms near chr{chromosome}:{position} {unit} within {window_kb/1000.0} Mb",
		limit=limit,
	)


def search_liver_splice_junctions_by_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return qtl_search_tool.search(
		query=f"liver splice junctions near chr{chromosome}:{position} {unit} within {window_kb/1000.0} Mb",
		limit=limit,
	)


def get_genes_near_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return search_genes_by_position(chromosome, position, unit=unit, window_kb=window_kb, limit=limit)


def get_isoforms_near_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return search_liver_isoforms_by_position(chromosome, position, unit=unit, window_kb=window_kb, limit=limit)


def get_splice_junctions_near_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	return search_liver_splice_junctions_by_position(chromosome, position, unit=unit, window_kb=window_kb, limit=limit)