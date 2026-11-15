"""Tiny 7-table .tbl fixtures for the engine baseline runner tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TPCH_DIR = Path(__file__).resolve().parents[1]

DATE_COLUMNS = {"l_shipdate", "l_commitdate", "l_receiptdate", "o_orderdate"}
LINEITEM_COMMENTS = ['sly "quoted" deposits, final', "ironic requests"]


def _table_schemas() -> dict:
    common = sys.modules.get("baseline_common")
    if common is None:
        spec = importlib.util.spec_from_file_location("baseline_common", TPCH_DIR / "baseline_common.py")
        assert spec and spec.loader
        common = importlib.util.module_from_spec(spec)
        sys.modules["baseline_common"] = common
        spec.loader.exec_module(common)
    return common.TABLE_SCHEMAS


# Writes 2 rows per table, each line ending in "|" as dbgen writes them. Returns the folder.
def write_tiny_tbl_files(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for table, schema in _table_schemas().items():
        lines = []
        for i in (1, 2):
            fields = []
            for name, sqlite_type in schema:
                if name == "l_comment":
                    fields.append(LINEITEM_COMMENTS[i - 1])
                elif name in DATE_COLUMNS:
                    fields.append(f"1995-0{i}-15")
                elif sqlite_type == "INTEGER":
                    fields.append(str(i))
                elif sqlite_type == "REAL":
                    fields.append(f"{i}.50")
                else:
                    fields.append(f"{name}_{i}")
            lines.append("|".join(fields) + "|\n")
        (folder / f"{table}.tbl").write_text("".join(lines))
    return folder
