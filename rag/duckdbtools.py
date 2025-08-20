from pathlib import Path
from typing import Any, List, Optional
import os
import re

# Import DuckDB helpers
try:
	from rag.duckdbhelpers import (
		pick_gene_or_phenotype,
		is_empty_value,
		build_lod_value_and_source_expr,
		build_gene_or_phenotype_case_expr,
		build_position_mb_expr,
		build_normalized_chromosome_expr,
		sanitize_sql_literal,
	)
except ImportError:
	from duckdbhelpers import (
		pick_gene_or_phenotype,
		is_empty_value,
		build_lod_value_and_source_expr,
		build_gene_or_phenotype_case_expr,
		build_position_mb_expr,
		build_normalized_chromosome_expr,
		sanitize_sql_literal,
	)


def get_top_lod_peaks(limit: int = 10) -> str:
	"""
	Returns the top rows by LOD-like value from the unified DuckDB table, including
	the source and either a gene symbol (for gene-level results) or a phenotype
	for other result types (clinical traits, liver lipids, isoforms, etc.).

	Environment:
	- QTL_DUCKDB_PATH or QTL_DUCKDB_FILE: DuckDB path (default data/qtl_database.duckdb)
	- QTL_DUCKDB_TABLE: table name (default qtl_peaks)
	"""
	try:
		import duckdb  # lazy import
	except Exception:
		return "Error: duckdb is not installed. Install with 'pip install duckdb'."

	db_path = (
		os.getenv("QTL_DUCKDB_PATH")
		or os.getenv("QTL_DUCKDB_FILE")
		or str(Path(__file__).resolve().parent.parent / "data" / "qtl_database.duckdb")
	)
	table_name = os.getenv("QTL_DUCKDB_TABLE", "qtl_peaks")

	if not os.path.exists(db_path):
		return f"Error: DuckDB database not found at {db_path}"

	try:
		con = duckdb.connect(db_path, read_only=True)
	except Exception as e:
		return f"Error: Could not open DuckDB database at {db_path}. Details: {e}"

	try:
		# Inspect columns
		try:
			schema_rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
		except Exception as e:
			return f"Error: Could not inspect table '{table_name}'. Details: {e}"
		if not schema_rows:
			return f"Error: Table '{table_name}' not found in database {db_path}"

		name_by_lower = {str(r[1]).lower(): str(r[1]) for r in schema_rows}


		lod_expr, lod_source_expr = build_lod_value_and_source_expr(available_columns=list(name_by_lower.keys()))
		if not lod_expr:
			return "Error: No LOD-like columns found (expected one of: lod_diff, qtl_lod, lod)."

		gene_col = "gene_symbol"
		gene_expr = f'"{gene_col}"' if gene_col else "NULL"
		phenotype_col = "phenotype"
		phenotype_expr = f'"{phenotype_col}"' if phenotype_col else "NULL"
		source_col = "Source"
		source_expr = f'"{source_col}"'

		sql = (
			f"SELECT {source_expr} AS source, {gene_expr} AS gene_symbol, {phenotype_expr} AS phenotype, "
			f"{lod_expr} AS lod_value, {lod_source_expr} AS lod_source "
			f"FROM \"{table_name}\" "
			f"WHERE {lod_expr} IS NOT NULL "
			f"ORDER BY lod_value DESC "
			f"LIMIT {int(limit)}"
		)

		rows = con.execute(sql).fetchall() or []
		if not rows:
			return "No rows with non-null LOD-like values were found."

		header = (
			f"**Top {min(len(rows), int(limit))} LOD-like Peaks**\n\n"
			f"Data source: {db_path} — table: {table_name}"
		)
		lines: List[str] = []
		for idx, (source, gene_symbol, phenotype, lod_value, lod_source) in enumerate(rows, start=1):
			try:
				lod_num = float(lod_value) if lod_value is not None else float('nan')
			except Exception:
				lod_num = float('nan')
			src_txt = str(source) if source not in (None, "") else "Unknown source"
			label = pick_gene_or_phenotype(src_txt, gene_symbol, phenotype) or "Unknown"
			lod_from = lod_source or "unknown"
			lines.append(f"{idx}. LOD {lod_num:.3f} ({lod_from}) — {label} — Source: {src_txt}")

		return header + "\n" + "\n".join(lines)
	finally:
		try:
			con.close()
		except Exception:
			pass


def search_qtl_peaks(query: str, limit: int = 200) -> str:
	"""
	Search the DuckDB `qtl_peaks` table for a given term and return matching peaks.

	Routing: if the query looks like a coordinate (e.g., "chr5:142 Mb"), this
	delegates to `search_qtl_by_genomic_position`.
	"""
	if not (query or "").strip():
		return "Error: query cannot be empty."

	# Coordinate-style routing
	q = (query or "").strip()
	coord_re = re.compile(r"(?:chr(?:omosome)?\s*)?([0-9xyYmM]+)\s*(?::|\bat\b|@|\bposition\b|\bpos\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*(mbp?|mb|bp)?", re.IGNORECASE)
	m = coord_re.search(q)
	if m:
		chrom = m.group(1)
		pos = float(m.group(2))
		unit = (m.group(3) or "mb").lower()
		win_kb = None
		for wr in (
			re.compile(r"[±\+\-]\s*([0-9]+(?:\.[0-9]+)?)\s*mbp?", re.IGNORECASE),
			re.compile(r"within\s+([0-9]+(?:\.[0-9]+)?)\s*mbp?", re.IGNORECASE),
			re.compile(r"window\s*([0-9]+(?:\.[0-9]+)?)\s*mbp?", re.IGNORECASE),
		):
			mw = wr.search(q)
			if mw:
				try:
					win_kb = float(mw.group(1)) * 1000.0
					break
				except Exception:
					pass
		# Infer phenotype_class from keywords
		pc = None
		ql = q.lower()
		if "clinical" in ql:
			pc = "clinical_trait"
		elif any(k in ql for k in ["metabolite", "metabolomics", "plasma"]):
			pc = "plasma_metabolite"
		elif "lipid" in ql:
			pc = "liver_lipid"
		elif "gene" in ql:
			pc = "liver_gene"
		elif "isoform" in ql:
			pc = "liver_isoform"
		elif any(k in ql for k in ["junction", "splice", "junc"]):
			pc = "liver_splice_junc"
		# Delegate to positional tool
		return search_qtl_by_genomic_position(
			chromosome=chrom,
			position=pos,
			unit='bp' if unit == 'bp' else 'mb',
			window_kb=win_kb if win_kb is not None else 4000.0,
			phenotype_class=pc,
		)

	# Plain text search path
	try:
		import duckdb
	except Exception:
		return "Error: duckdb is not installed. Install with 'pip install duckdb'."

	db_path = (
		os.getenv("QTL_DUCKDB_PATH")
		or os.getenv("QTL_DUCKDB_FILE")
		or str(Path(__file__).resolve().parent.parent / "data" / "qtl_database.duckdb")
	)
	table_name = os.getenv("QTL_DUCKDB_TABLE", "qtl_peaks")

	if not os.path.exists(db_path):
		return f"Error: DuckDB database not found at {db_path}"

	try:
		con = duckdb.connect(db_path, read_only=True)
	except Exception as e:
		return f"Error: Could not open DuckDB database at {db_path}. Details: {e}"

	try:
		try:
			schema_rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
		except Exception as e:
			return f"Error: Could not inspect table '{table_name}'. Details: {e}"
		if not schema_rows:
			return f"Error: Table '{table_name}' not found in database {db_path}"

		col_names_lower = {str(r[1]).lower(): str(r[1]) for r in schema_rows}
		lod_expr, lod_source_expr = build_lod_value_and_source_expr(available_columns=list(col_names_lower.keys()))
		if not lod_expr:
			return "Error: No LOD-like columns found (expected one of: lod_diff, qtl_lod, lod)."

		q_esc = q.lower().replace("'", "''")
		# Simple heuristic for gene-like tokens
		is_gene_like = ("_" not in q) and (" " not in q)
		if is_gene_like:
			gene_filters = [
				f"lower(CAST(\"gene_symbol\" AS VARCHAR)) LIKE '%{q_esc}%'",
				"lower(\"Source\") LIKE '%genes%'",
				"lower(\"Source\") NOT LIKE '%isoform%'",
				"lower(\"Source\") NOT LIKE '%splice%'",
				"lower(\"Source\") NOT LIKE '%junc%'",
			]
			where_clause = " AND ".join(gene_filters)
		else:
			where_clause = f"lower(CAST(\"phenotype\" AS VARCHAR)) LIKE '%{q_esc}%'"

		sql = (
			f"SELECT \"Source\" AS source, \"gene_symbol\" AS gene_symbol, \"phenotype\" AS phenotype, "
			f"{lod_expr} AS lod_value, {lod_source_expr} AS lod_source "
			f"FROM \"{table_name}\" "
			f"WHERE ({where_clause}) AND {lod_expr} IS NOT NULL "
			f"ORDER BY lod_value DESC "
			f"LIMIT {int(limit)}"
		)

		rows = con.execute(sql).fetchall() or []
		if not rows:
			return f"No peaks found matching '{query}'."

		header = (
			f"**QTL Peaks matching: {q}**\n\n"
			f"Data source: {db_path} — table: {table_name}"
		)
		lines: List[str] = []
		for idx, (source, gene_symbol, phenotype, lod_value, lod_source) in enumerate(rows, start=1):
			try:
				lod_num = float(lod_value) if lod_value is not None else float('nan')
			except Exception:
				lod_num = float('nan')
			src_txt = str(source) if source not in (None, "") else "Unknown source"
			label = pick_gene_or_phenotype(src_txt, gene_symbol, phenotype) or "Unknown"
			lod_from = lod_source or "unknown"
			lines.append(f"{idx}. LOD {lod_num:.3f} ({lod_from}) — {label} — Source: {src_txt}")

		return header + "\n" + "\n".join(lines)
	finally:
		try:
			con.close()
		except Exception:
			pass


def search_qtl_by_genomic_position(
	chromosome: str,
	position: float,
	unit: str = "mb",
	window_kb: float = 4000.0,
	limit: int = 20,
	phenotype_class: Optional[str] = None,
) -> str:
	"""
	Find QTL peaks near a genomic position and list their associated traits/labels.
	- Sorted by closest position to the query locus
	- Deduplicated by trait label (one row per label)
	- Default window ±4 Mb (window_kb=4000)
	- Optional phenotype_class filter (e.g., "clinical_traits")
	"""
	if not (chromosome or "").strip():
		return "Error: chromosome is required."
	try:
		pos_val = float(position)
	except Exception:
		return "Error: position must be numeric."

	unit_norm = (unit or "mb").strip().lower()
	if unit_norm not in {"mb", "bp"}:
		return "Error: unit must be 'mb' or 'bp'."

	try:
		import duckdb
	except Exception:
		return "Error: duckdb is not installed. Install with 'pip install duckdb'."

	db_path = (
		os.getenv("QTL_DUCKDB_PATH")
		or os.getenv("QTL_DUCKDB_FILE")
		or str(Path(__file__).resolve().parent.parent / "data" / "qtl_database.duckdb")
	)
	table_name = os.getenv("QTL_DUCKDB_TABLE", "qtl_peaks")

	if not os.path.exists(db_path):
		return f"Error: DuckDB database not found at {db_path}"

	# Normalize inputs
	target_mb = pos_val / 1e6 if unit_norm == "bp" else pos_val
	window_mb = (window_kb or 0.0) / 1000.0
	chrom_in = str(chromosome).strip().lower()
	if chrom_in.startswith("chr"):
		chrom_in = chrom_in[3:]
	chrom_in = chrom_in.replace(" ", "")

	try:
		con = duckdb.connect(db_path, read_only=True)
	except Exception as e:
		return f"Error: Could not open DuckDB database at {db_path}. Details: {e}"

	try:
		schema_rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
	except Exception as e:
		try:
			con.close()
		except Exception:
			pass
		return f"Error: Could not inspect table '{table_name}'. Details: {e}"

	if not schema_rows:
		try:
			con.close()
		except Exception:
			pass
		return f"Error: Table '{table_name}' not found in database {db_path}"

	original_names = [str(r[1]) for r in schema_rows]
	lower_to_orig = {name.lower(): name for name in original_names}

	# Require canonical qtl_chr
	if "qtl_chr" not in lower_to_orig:
		try:
			con.close()
		except Exception:
			pass
		return "Error: Required column 'qtl_chr' not found in table."
	chrom_col = lower_to_orig["qtl_chr"]

	combined_pos_mb_expr = build_position_mb_expr(lower_to_orig, original_names)
	if not combined_pos_mb_expr:
		try:
			con.close()
		except Exception:
			pass
		return "Error: Could not detect a genomic position column in table."

	# Build expressions
	col_names_lower = {str(r[1]).lower(): str(r[1]) for r in schema_rows}
	lod_val_expr, lod_src_expr = build_lod_value_and_source_expr(available_columns=list(col_names_lower.keys()))
	# We will choose label in Python using pick_gene_or_phenotype
	# Acquire canonical column names for selection
	source_col = lower_to_orig.get("source", "Source")
	gene_col = lower_to_orig.get("gene_symbol")
	phen_col = lower_to_orig.get("phenotype")
	chrom_norm_expr = build_normalized_chromosome_expr(chrom_col)

	# Optional phenotype_class filter
	where_extra = ""
	if phenotype_class:
		if "phenotype_class" not in lower_to_orig:
			try:
				con.close()
			except Exception:
				pass
			return "Error: phenotype_class filter provided but column 'phenotype_class' not found."
		pc_value_raw = str(phenotype_class).strip().lower()
		pc_col = lower_to_orig["phenotype_class"]
		# Combine all plasma metabolite subclasses if requested
		if pc_value_raw in {"metabolite", "plasma_metabolite", "plasma"}:
			where_extra = (
				f" AND lower(\"{pc_col}\") IN ("
				f"'plasma_metabolite','plasma_2h_metabolite','plasma_13c_metabolite'"
				f")"
			)
		else:
			pc_value = sanitize_sql_literal(pc_value_raw)
			where_extra = f" AND lower(\"{pc_col}\") = '{pc_value}'"

	low_mb = target_mb - window_mb
	high_mb = target_mb + window_mb
	chrom_sql = sanitize_sql_literal(chrom_in)
	limit_n = min(int(limit), 20)
	# Pull more rows than needed to allow Python-side de-duplication by label
	sql_limit = min(max(limit_n * 10, 200), 1000)

	sql = (
		f"SELECT \"{source_col}\" AS source, \"{gene_col}\" AS gene_symbol, \"{phen_col}\" AS phenotype, "
		f"{lower_to_orig.get('phenotype_class','phenotype_class')} AS phenotype_class, "
		f"{combined_pos_mb_expr} AS pos_mb, {lod_val_expr} AS lod_value, {lod_src_expr} AS lod_source "
		f"FROM \"{table_name}\" "
		f"WHERE {chrom_norm_expr} = '{chrom_sql}' "
		f"  AND {combined_pos_mb_expr} BETWEEN {low_mb} AND {high_mb} "
		f"  AND {lod_val_expr} IS NOT NULL"
		f"{where_extra} "
		f"ORDER BY ABS({combined_pos_mb_expr} - {target_mb}) ASC "
		f"LIMIT {sql_limit}"
	)

	try:
		rows = con.execute(sql).fetchall() or []
	except Exception as e:
		try:
			con.close()
		except Exception:
			pass
		return f"Error: Query failed. Details: {e}"
	finally:
		try:
			con.close()
		except Exception:
			pass

	if not rows:
		locus_txt = f"chr{chrom_in}:{target_mb:.3f} Mb ± {window_mb:.3f} Mb"
		return f"No QTL peaks found near {locus_txt}."

	# If no phenotype_class filter was provided, prefer gene datasets when available
	if not phenotype_class:
		try:
			# rows: (source, gene_symbol, phenotype, phenotype_class, pos_mb, lod_value, lod_source)
			classes = [str(r[3]).lower() if r[3] is not None else '' for r in rows]
			priority = ['liver_gene', 'liver_isoform', 'liver_splice_junc']
			chosen = next((c for c in priority if c in classes), None)
			if chosen:
				rows = [r for r in rows if (str(r[3]).lower() if r[3] is not None else '') == chosen]
		except Exception:
			pass

	# Python-side de-duplication using pick_gene_or_phenotype
	seen: set[str] = set()
	unique_rows: List[tuple[str, float, float]] = []
	for (source, gene_symbol, phenotype, phenotype_class_val, pos_mb_val, lod_value, lod_source) in rows:
		label = pick_gene_or_phenotype(source, gene_symbol, phenotype)
		if is_empty_value(label):
			continue
		if label in seen:
			continue
		seen.add(label)
		try:
			pos_num = float(pos_mb_val) if pos_mb_val is not None else float('nan')
		except Exception:
			pos_num = float('nan')
		try:
			lod_num = float(lod_value) if lod_value is not None else float('nan')
		except Exception:
			lod_num = float('nan')
		unique_rows.append((label, pos_num, lod_num))
		if len(unique_rows) >= limit_n:
			break

	header = (
		f"**QTL Peaks near chr{chrom_in}:{target_mb:.3f} Mb ± {window_mb:.3f} Mb**\n\n"
		f"Data source: {db_path} — table: {table_name}"
	)

	lines: List[str] = []
	for idx, (label, pos_num, lod_num) in enumerate(unique_rows, start=1):
		lines.append(f"{idx}. {label} — pos {pos_num:.6f} Mb (LOD {lod_num:.3f})")

	return header + "\n" + "\n".join(lines) 


def find_traits_near_locus(query: str, default_window_kb: float = 4000.0, limit: int = 20) -> str:
	"""
	Parse a free-text query like "clinical traits near chr5:50 Mb" and return traits near the locus.
	Infers chromosome, position, unit, optional window, and phenotype_class (e.g., 'clinical_traits').
	"""
	q = (query or "").strip()
	if not q:
		return "Error: query cannot be empty."
	coord_re = re.compile(r"(?:chr(?:omosome)?\s*)?([0-9xyYmM]+)\s*(?::|\bat\b|@|\bposition\b|\bpos\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*(mbp?|mb|bp)?", re.IGNORECASE)
	m = coord_re.search(q)
	if not m:
		return "Error: could not parse chromosome and position from query. Use e.g., 'chr5:50 Mb'."
	chrom = m.group(1)
	pos = float(m.group(2))
	unit = (m.group(3) or "mb").lower()
	win_kb = None
	for wr in (
		re.compile(r"[±\+\-]\s*([0-9]+(?:\.[0-9]+)?)\s*mbp?", re.IGNORECASE),
		re.compile(r"within\s+([0-9]+(?:\.[0-9]+)?)\s*mbp?", re.IGNORECASE),
		re.compile(r"window\s*([0-9]+(?:\.[0-9]+)?)\s*mbp?", re.IGNORECASE),
	):
		mw = wr.search(q)
		if mw:
			try:
				win_kb = float(mw.group(1)) * 1000.0
				break
			except Exception:
				pass
	# infer phenotype class
	pc = None
	ql = q.lower()
	if "clinical" in ql:
		pc = "clinical_trait"
	elif any(k in ql for k in ["metabolite", "metabolomics", "plasma"]):
		pc = "plasma_metabolite"
	elif "lipid" in ql:
		pc = "liver_lipid"
	elif "gene" in ql:
		pc = "liver_gene"
	elif "isoform" in ql:
		pc = "liver_isoform"
	elif any(k in ql for k in ["junction", "splice", "junc"]):
		pc = "liver_splice_junc"
	return search_qtl_by_genomic_position(
		chromosome=chrom,
		position=pos,
		unit='bp' if unit == 'bp' else 'mb',
		window_kb=win_kb if win_kb is not None else float(default_window_kb),
		limit=limit,
		phenotype_class=pc,
	) 


# Convenience wrappers for common phenotype classes

def search_clinical_traits_by_position(
	chromosome: str,
	position: float,
	unit: str = "mb",
	window_kb: float = 4000.0,
	limit: int = 20,
) -> str:
	return search_qtl_by_genomic_position(
		chromosome=chromosome,
		position=position,
		unit=unit,
		window_kb=window_kb,
		limit=limit,
		phenotype_class="clinical_trait",
	)


def search_metabolites_by_position(
	chromosome: str,
	position: float,
	unit: str = "mb",
	window_kb: float = 4000.0,
	limit: int = 20,
) -> str:
	return search_qtl_by_genomic_position(
		chromosome=chromosome,
		position=position,
		unit=unit,
		window_kb=window_kb,
		limit=limit,
		phenotype_class="plasma_metabolite",
	)


def search_lipids_by_position(
	chromosome: str,
	position: float,
	unit: str = "mb",
	window_kb: float = 4000.0,
	limit: int = 20,
) -> str:
	return search_qtl_by_genomic_position(
		chromosome=chromosome,
		position=position,
		unit=unit,
		window_kb=window_kb,
		limit=limit,
		phenotype_class="liver_lipid",
	) 


def search_liver_genes_by_position(
	chromosome: str,
	position: float,
	unit: str = "mb",
	window_kb: float = 4000.0,
	limit: int = 20,
) -> str:
	return search_qtl_by_genomic_position(
		chromosome=chromosome,
		position=position,
		unit=unit,
		window_kb=window_kb,
		limit=limit,
		phenotype_class="liver_gene",
	)


def search_liver_isoforms_by_position(
	chromosome: str,
	position: float,
	unit: str = "mb",
	window_kb: float = 4000.0,
	limit: int = 20,
) -> str:
	return search_qtl_by_genomic_position(
		chromosome=chromosome,
		position=position,
		unit=unit,
		window_kb=window_kb,
		limit=limit,
		phenotype_class="liver_isoform",
	)


def search_liver_splice_junctions_by_position(
	chromosome: str,
	position: float,
	unit: str = "mb",
	window_kb: float = 4000.0,
	limit: int = 20,
) -> str:
	return search_qtl_by_genomic_position(
		chromosome=chromosome,
		position=position,
		unit=unit,
		window_kb=window_kb,
		limit=limit,
		phenotype_class="liver_splice_junc",
	) 


# Aliases with generic names to improve tool selection by LLMs

def get_genes_near_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	"""Return liver gene QTLs near a locus (phenotype_class='liver_gene')."""
	return search_liver_genes_by_position(chromosome, position, unit=unit, window_kb=window_kb, limit=limit)


def get_isoforms_near_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	"""Return liver isoform QTLs near a locus (phenotype_class='liver_isoform')."""
	return search_liver_isoforms_by_position(chromosome, position, unit=unit, window_kb=window_kb, limit=limit)


def get_splice_junctions_near_position(chromosome: str, position: float, unit: str = "mb", window_kb: float = 4000.0, limit: int = 20) -> str:
	"""Return liver splice junction QTLs near a locus (phenotype_class='liver_splice_junc')."""
	return search_liver_splice_junctions_by_position(chromosome, position, unit=unit, window_kb=window_kb, limit=limit) 