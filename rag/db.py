#create new column that says phenotype for all gene sybols, liver lipids, etc
#want addcovar and intcovar
#want qtl lod, gene_ID(if there), cis, ABCDEFGH, phnenotype_class, numb_mice, gene_chr, qtl_chr, gene_type, human_hom_gene_symbol, marker 
import argparse
import os
import re
import sys
import glob
from typing import List, Dict, Set, Tuple

import duckdb


# Token sets used for filename parsing
DATA_CATEGORIES = [
	"clinical_traits",
	"splice_juncs",
	"genes",
	"isoforms",
	"lipids",
	"metabolites",
]
BIOSPECIMENS = ["liver", "plasma"]
SUBSETS = ["all_mice", "hc_mice", "hf_mice", "female_mice", "male_mice"]


def _find_first_token(name_lc: str, tokens: List[str]) -> str:
	for tok in tokens:
		if name_lc.startswith(f"{tok}_") or f"_{tok}_" in name_lc or name_lc.endswith(f"_{tok}.csv") or name_lc.endswith(f"_{tok}_peaks.csv") or f"_peaks_in_{tok}_" in name_lc:
			return tok
	return None


def parse_metadata_from_path(path: str) -> Dict[str, str]:
	basename = os.path.basename(path)
	name_lc = basename.lower()

	biospecimen = _find_first_token(name_lc, BIOSPECIMENS)
	data_category = _find_first_token(name_lc, DATA_CATEGORIES)
	subset = _find_first_token(name_lc, SUBSETS)

	# analysis_type: additive vs interactive
	analysis_type = None
	if "additive" in name_lc:
		analysis_type = "additive"
	elif "interactive" in name_lc or "qtlx" in name_lc:
		analysis_type = "interactive"

	# interaction_axis: diet, sex, sexbydiet
	interaction_axis = None
	if "qtlxsexbydiet" in name_lc or "sexbydiet" in name_lc:
		interaction_axis = "sexbydiet"
	elif "qtlxsex" in name_lc or "sex_interactive" in name_lc:
		interaction_axis = "sex"
	elif "qtlxdiet" in name_lc or "diet_interactive" in name_lc:
		interaction_axis = "diet"

	return {
		"biospecimen": biospecimen,
		"data_category": data_category,
		"subset": subset,
		"analysis_type": analysis_type,
		"interaction_axis": interaction_axis,
		"source_file": basename,
		"source_path": path,
	}


def ensure_db_dir(db_path: str) -> None:
	dirname = os.path.dirname(db_path)
	if dirname and not os.path.exists(dirname):
		os.makedirs(dirname, exist_ok=True)


def quote_ident(name: str) -> str:
	return '"' + name.replace('"', '""') + '"'


def sql_str(value: str) -> str:
	if value is None:
		return "NULL"
	return "'" + value.replace("'", "''") + "'"


def get_existing_columns(con: duckdb.DuckDBPyConnection, table: str) -> List[str]:
	rows = con.execute(f"PRAGMA table_info({quote_ident(table)});").fetchall()
	# rows: [cid, name, type, notnull, dflt_value, pk]
	return [r[1] for r in rows]


def create_table_if_needed(con: duckdb.DuckDBPyConnection, table: str) -> None:
	con.execute(
		f"""
		CREATE TABLE IF NOT EXISTS {quote_ident(table)} (
			biospecimen VARCHAR,
			data_category VARCHAR,
			subset VARCHAR,
			analysis_type VARCHAR,
			interaction_axis VARCHAR,
			source_file VARCHAR,
			source_path VARCHAR
		);
		"""
	)


def ensure_metadata_columns(con: duckdb.DuckDBPyConnection, table: str) -> None:
	required = [
		"biospecimen",
		"data_category",
		"subset",
		"analysis_type",
		"interaction_axis",
		"source_file",
		"source_path",
	]
	existing = get_existing_columns(con, table)
	missing = [c for c in required if c not in existing]
	if missing:
		add_missing_columns(con, table, missing)


def add_missing_columns(con: duckdb.DuckDBPyConnection, table: str, missing: List[str]) -> None:
	for col in missing:
		con.execute(f"ALTER TABLE {quote_ident(table)} ADD COLUMN {quote_ident(col)} VARCHAR;")


EXACT_EXCLUDED_COLUMNS: Set[str] = {"Which_mice,"}


def read_csv_columns_sample(con: duckdb.DuckDBPyConnection, csv_path: str) -> List[str]:
	# Read a tiny sample to get columns reliably without loading the entire file
	query = (
		"SELECT * FROM read_csv_auto(?, header=true, all_varchar=true, filename=true) LIMIT 0"
	)
	res = con.execute(query, [csv_path])
	cols = [d[0] for d in res.description]
	return [c for c in cols if c not in EXACT_EXCLUDED_COLUMNS]


def build_insert_sql(table: str, csv_path: str, file_columns: List[str], target_columns: List[str], metadata: Dict[str, str]) -> Tuple[str, List[str]]:
	# Exclude auto filename column from data projection
	data_cols = [c for c in file_columns if c.lower() != "filename" and c not in EXACT_EXCLUDED_COLUMNS]
	# Determine overlap between file columns and target table columns
	target_set = {c.lower() for c in target_columns}
	present_cols = [c for c in data_cols if c.lower() in target_set]

	# Build target column list: metadata first, then present file columns
	insert_cols = [
		"biospecimen",
		"data_category",
		"subset",
		"analysis_type",
		"interaction_axis",
		"source_file",
		"source_path",
	] + present_cols

	# Build SELECT list with metadata literals and then file columns
	select_exprs = [
		f"{sql_str(metadata['biospecimen'])} AS biospecimen",
		f"{sql_str(metadata['data_category'])} AS data_category",
		f"{sql_str(metadata['subset'])} AS subset",
		f"{sql_str(metadata['analysis_type'])} AS analysis_type",
		f"{sql_str(metadata['interaction_axis'])} AS interaction_axis",
		f"COALESCE(NULLIF(regexp_extract(filename, '.*/([^/]+)$', 1), ''), filename) AS source_file",
		f"filename AS source_path",
	]
	select_exprs.extend([quote_ident(c) for c in present_cols])

	sql = f"""
	INSERT INTO {quote_ident(table)} ({', '.join(quote_ident(c) for c in insert_cols)})
	SELECT {', '.join(select_exprs)}
	FROM read_csv_auto({sql_str(csv_path)}, header=true, all_varchar=true, filename=true);
	"""
	return sql, present_cols


def ingest_files(db_path: str, table: str, globs: List[str], dry_run_limit: int = 0) -> None:
	ensure_db_dir(db_path)
	con = duckdb.connect(db_path)
	con.execute("PRAGMA threads=4;")
	create_table_if_needed(con, table)
	ensure_metadata_columns(con, table)

	# Expand globs to a unique, sorted list of files
	files: List[str] = []
	for g in globs:
		files.extend(glob.glob(g))
	files = sorted(f for f in set(files) if os.path.isfile(f))
	if not files:
		raise SystemExit(f"No files matched: {globs}")

	for idx, path in enumerate(files, start=1):
		meta = parse_metadata_from_path(path)
		# Sample columns from this file
		file_cols = read_csv_columns_sample(con, path)
		# Remove the auto 'filename' column from consideration when diffing schema
		file_cols_wo_filename = [c for c in file_cols if c.lower() != "filename" and c not in EXACT_EXCLUDED_COLUMNS]

		existing_cols = get_existing_columns(con, table)
		existing_set_lower: Set[str] = {c.lower() for c in existing_cols}

		# Add any new columns found in the file that are not yet in the target table
		missing_cols = [c for c in file_cols_wo_filename if c.lower() not in existing_set_lower]
		if missing_cols:
			add_missing_columns(con, table, missing_cols)
			existing_cols = get_existing_columns(con, table)

		# Perform the insert
		sql, present_cols = build_insert_sql(table, path, file_cols, existing_cols, meta)
		if dry_run_limit > 0:
			# Append LIMIT directly to the SELECT
			limited_sql = sql.rstrip(";\n \t") + f" LIMIT {int(dry_run_limit)};"
			con.execute(limited_sql)
		else:
			con.execute(sql)

		print(f"[{idx}/{len(files)}] Ingested: {os.path.basename(path)} | added_cols={len(missing_cols)} present_cols={len(present_cols)}")

	con.close()


def build_default_globs(data_dir: str) -> List[str]:
	# Broaden to include files like '*_peaks.csv' and '*_peaks_in_*_additive.csv'
	return [
		os.path.join(data_dir, "*peaks*.csv"),
	]


def main(argv: List[str]) -> None:
	parser = argparse.ArgumentParser(description="Build a unified DuckDB for RAG from CSV peaks files with filename-derived metadata.")
	parser.add_argument("--db", dest="db_path", default=os.path.join(os.getcwd(), "data", "qtl.duckdb"), help="Path to DuckDB database file to create/update.")
	parser.add_argument("--dir", dest="data_dir", default="/data/dev/miniViewer_3.0", help="Directory containing CSV files.")
	parser.add_argument("--glob", dest="globs", action="append", default=None, help="Glob pattern(s) of files to ingest. Can be passed multiple times. Defaults to '*peaks*.csv' under --dir.")
	parser.add_argument("--table", dest="table", default="qtl_data", help="Target table name.")
	parser.add_argument("--dry-run", dest="dry_run", type=int, default=0, help="If > 0, ingest only this many rows per file for a quick sanity check.")
	args = parser.parse_args(argv)

	globs = args.globs if args.globs else build_default_globs(args.data_dir)

	print(f"DB: {args.db_path}")
	print(f"Table: {args.table}")
	print(f"Globs: {globs}")
	print(f"Dry-run rows per file: {args.dry_run}")

	ingest_files(args.db_path, args.table, globs, dry_run_limit=args.dry_run)
	print("Done.")


if __name__ == "__main__":
	main(sys.argv[1:])
