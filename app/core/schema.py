from __future__ import annotations

import json
import re

from app.core.datasets import AppPaths, missing_files

# description.md's table heading, a column's bullet, and the Values line under that bullet.
# Any other section heading ends the tables.
TABLE_HEADING = re.compile(r"^### (\S+) \(")
SECTION_HEADING = re.compile(r"^## ")
COLUMN_BULLET = re.compile(r"^- `([^`]+)` ")
VALUES_LINE = re.compile(r"^  Values: (.*)$")
LITERAL = re.compile(r"'((?:[^']|'')*)'")


# A dataset with no tables to show, as a fresh dict each time so no two entries share a list.
def _no_tables() -> dict:
    return {"about": "", "tables": []}


# A Values line's SQL string literals in order; None when the line isn't a clean list of them.
def parse_values(text: str) -> list[str] | None:
    values, at = [], 0
    while True:
        found = LITERAL.match(text, at)
        if found is None:
            return None
        
        values.append(found.group(1).replace("''", "'"))
        at = found.end()
        if at == len(text):
            return values
        if not text.startswith(", ", at):
            return None
        at += 2


# Each column's listed values, by (table, column), from the Values line right under its bullet.
def listed_values(description: str) -> dict[tuple[str, str], list[str]]:
    found, table, column = {}, None, None
    for line in description.splitlines():
        if heading := TABLE_HEADING.match(line):
            table, column = heading.group(1), None
        elif SECTION_HEADING.match(line):
            table, column = None, None
        elif bullet := COLUMN_BULLET.match(line):
            column = bullet.group(1)
        elif (listed := VALUES_LINE.match(line)) and table and column:
            values = parse_values(listed.group(1))
            if values is not None:
                found[(table, column)] = values
            column = None
        else:
            column = None
    return found


# Every foreign key column as "table.column", with the "table.column" it names.
def _references(spec: dict) -> dict[str, str]:
    return {f"{fk['table']}.{column}": f"{fk['ref_table']}.{ref}" for fk in spec.get("foreign_keys", []) for column, ref in zip(fk["columns"], fk["ref_columns"])}


# One column as the schema shows it: key, references and values only when it has them.
def _column(table: str, column: dict, spec: dict, key: list[str], refs: dict, values: dict) -> dict:
    entry = {
        "name": column["name"], 
        "type": column["type"], 
        "description": spec["columns"].get(column["name"], "")
    }
    
    if column["name"] in key:
        entry["key"] = True
    if (ref := refs.get(f"{table}.{column['name']}")) is not None:
        entry["references"] = ref
    if (listed := values.get((table, column["name"]))) is not None:
        entry["values"] = listed
    return entry


# A dataset's about text and its tables in manifest order, from the manifest, the spec and description.md.
# No tables when a file is missing or can't be read.
def dataset_schema(paths: AppPaths, name: str) -> dict:
    if missing_files(paths, name):
        return _no_tables()
    
    try:
        folder = paths.dataset_dir(name)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        spec = json.loads(paths.spec(name).read_text(encoding="utf-8"))
        values = listed_values((folder / "description.md").read_text(encoding="utf-8"))
        refs = _references(spec)
        tables = []
        for table in manifest["tables"]:
            key = list(spec["primary_keys"].get(table["name"], []))
            columns = [_column(table["name"], column, spec, key, refs, values) for column in table["columns"]]
            tables.append({
                "name": table["name"], 
                "rows": table["rows"], 
                "primary_key": key, 
                "columns": columns
            })
            
        return {
            "about": spec["about"], 
            "tables": tables
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return _no_tables()
