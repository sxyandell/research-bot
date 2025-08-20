#!/usr/bin/env python3
"""
Build a unified DuckDB table from multiple CSV files with overlapping schemas.

- Discovers CSVs using a configurable glob pattern
- Loads data via DuckDB's read_csv_auto with union_by_name=True to align columns by name
- Creates or replaces a persistent table in a DuckDB file
- Prints verification info: table schema and total row count

This script is intended for building a wide table for RAG indexing from heterogeneous
QTL peak CSVs (genes, lipids, clinical traits, etc.). Missing columns in a CSV will
be NULL in the final table.
"""

from __future__ import annotations

import os
import sys
import glob
from typing import List

import duckdb


# =========================
# Configuration (edit here)
# =========================
# Glob pattern for input CSV files (edit to match your dataset location)
INPUT_FILE_GLOB: str = "/data/dev/miniViewer_3.0/DO1200_*peaks.csv"

# Path to persistent DuckDB database file
DUCKDB_FILE_PATH: str = "data/qtl_database.duckdb"

# Target table name to create or replace
TARGET_TABLE_NAME: str = "qtl_peaks"

# CSV inference options
# - Set SAMPLE_SIZE to -1 to scan all rows for robust type inference (slower on very large data)
# - Set to a positive integer for faster ingestion with sampling-based inference
SAMPLE_SIZE: int = -1

# When True, DuckDB will not try to infer types and will load everything as VARCHAR.
# For typed analytics, keep this False. If your CSVs are messy and type inference causes issues,
# you can set this to True and cast downstream as needed.
ALL_VARCHAR: bool = False

# If True, also include a column with the filename of each row's source CSV
INCLUDE_FILENAME_COLUMN: bool = True


def _quote_sql_identifier(identifier: str) -> str:
    """Quote an SQL identifier with double quotes, escaping any embedded quotes."""
    return '"' + identifier.replace('"', '""') + '"'


def _to_sql_string_list(values: List[str]) -> str:
    """Convert a list of Python strings to a DuckDB SQL string list literal.

    Example: ["a", "b"] -> ['a','b']
    """
    escaped = ["'" + v.replace("'", "''") + "'" for v in values]
    return "[" + ",".join(escaped) + "]"


def discover_csv_files(pattern: str) -> List[str]:
    """Discover CSV files using a glob pattern, returning a sorted absolute-path list."""
    matched = glob.glob(pattern)
    # Convert to absolute paths for stability and clarity
    absolute_paths = [os.path.abspath(p) for p in matched]
    # Sort deterministically for reproducibility
    absolute_paths.sort()
    return absolute_paths


def build_create_table_sql(
    files: List[str],
    table_name: str,
    sample_size: int,
    all_varchar: bool,
    include_filename: bool,
) -> str:
    """Construct a CREATE OR REPLACE TABLE ... AS SELECT ... SQL statement.

    Uses read_csv_auto with union_by_name=True and additional ingestion options.
    """
    if not files:
        raise ValueError("No input files provided to build_create_table_sql().")

    if not include_filename:
        raise ValueError(
            "INCLUDE_FILENAME_COLUMN must be True to compute Split-by Scan from filenames."
        )

    files_list_sql = _to_sql_string_list(files)

    # Parameters for read_csv_auto
    params = [
        f"union_by_name=true",
        f"sample_size={sample_size}",
        f"all_varchar={'true' if all_varchar else 'false'}",
        f"filename={'true' if include_filename else 'false'}",
    ]
    params_sql = ", ".join(params)

    # Build conditions for scan-type detection
    # QTL by Covar: lod_diff present and non-null
    qtl_by_covar_cond = "lod_diff IS NOT NULL"

    # Split-by: filename indicates split-by sex or diet, e.g., ..._peaks_in_{female|male|HC|HF}_mice_...
    split_by_cond = "lower(filename) LIKE '%_peaks_in_%_mice_%'"

    # Compute Source = basename(filename)
    source_expr = "regexp_extract(filename, '([^/\\\\]+)$', 1)"

    # Compose the full SQL using CTEs so we can add computed columns and control order
    table_ident = _quote_sql_identifier(table_name)
    sql = (
        f"CREATE OR REPLACE TABLE {table_ident} AS\n"
        f"WITH base AS (\n"
        f"  SELECT * FROM read_csv_auto({files_list_sql}, {params_sql})\n"
        f"),\n"
        f"shaped AS (\n"
        f"  SELECT\n"
        f"    {source_expr} AS \"Source\",\n"
        f"    base.* EXCLUDE (\"Which_mice\")\n"
        f"  FROM base\n"
        f")\n"
        f"SELECT\n"
        f"  shaped.*,\n"
        f"  CASE WHEN {qtl_by_covar_cond} THEN TRUE ELSE FALSE END AS \"QTL by Covar Scan\",\n"
        f"  CASE WHEN (NOT ({qtl_by_covar_cond})) AND ({split_by_cond}) THEN TRUE ELSE FALSE END AS \"Split-by Scan\",\n"
        f"  CASE WHEN (NOT ({qtl_by_covar_cond}) AND NOT ({split_by_cond})) THEN TRUE ELSE FALSE END AS \"Full Scan\"\n"
        f"FROM shaped;"
    )
    return sql


def main() -> None:
    print("[INFO] Discovering input CSV files ...")
    csv_files = discover_csv_files(INPUT_FILE_GLOB)

    if not csv_files:
        print(f"[ERROR] No files matched the pattern: {INPUT_FILE_GLOB}")
        sys.exit(1)

    print(f"[INFO] Found {len(csv_files)} files.")
    for path in csv_files:
        print(f"  - {path}")

    db_path_abs = os.path.abspath(DUCKDB_FILE_PATH)
    print(f"[INFO] Connecting to DuckDB database: {db_path_abs}")

    # Ensure containing directory exists
    os.makedirs(os.path.dirname(db_path_abs) or ".", exist_ok=True)

    con = duckdb.connect(db_path_abs)

    try:
        print(f"[INFO] Creating or replacing table '{TARGET_TABLE_NAME}' via read_csv_auto(union_by_name=true) ...")
        create_sql = build_create_table_sql(
            files=csv_files,
            table_name=TARGET_TABLE_NAME,
            sample_size=SAMPLE_SIZE,
            all_varchar=ALL_VARCHAR,
            include_filename=INCLUDE_FILENAME_COLUMN,
        )
        con.execute(create_sql)
        print("[INFO] Table creation completed.")

        # Verification 1: Print schema
        print(f"\n[VERIFY] Schema for table '{TARGET_TABLE_NAME}':")
        describe_result = con.execute(f"DESCRIBE {_quote_sql_identifier(TARGET_TABLE_NAME)}").fetchall()
        # Pretty-print schema rows: column_name, type, null, key, default, extra
        # DuckDB returns: column_name, column_type, null, key, default, extra
        for row in describe_result:
            # Use a concise, readable format
            col_name, col_type = row[0], row[1]
            print(f"  - {col_name}: {col_type}")

        # Verification 2: Print total row count
        count_result = con.execute(f"SELECT COUNT(*) FROM {_quote_sql_identifier(TARGET_TABLE_NAME)}").fetchone()
        total_rows = count_result[0] if count_result else 0
        print(f"\n[VERIFY] Total rows in '{TARGET_TABLE_NAME}': {total_rows}")

    finally:
        con.close()
        print("[INFO] Connection closed.")


if __name__ == "__main__":
    main() 
