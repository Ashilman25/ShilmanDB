from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sqlite3
import sys
from collections.abc import Container, Iterable
from contextlib import closing
from pathlib import Path


# Load an_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "an_common.py"
    loaded = sys.modules.get("an_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("an_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["an_common"] = module
    spec.loader.exec_module(module)
    return module


an = _load_common()
DISTINCT_CAP = an.DISTINCT_CAP

SPEC_DIR = an.HERE / "datasets"
SECTIONS = ("dataset", "about", "columns", "primary_keys", "foreign_keys", "grains", "entity_tables", "missing",
            "glossary")
SIGNATURE = ("any", "all", "none")
NOTHING_MISSING = "Nothing: every value in every column is real."
_QUALIFIER = re.compile(r"\b[A-Za-z_]\w*\.(?=[A-Za-z_])")  # the "l." in l.l_shipdate


# The JSON description of one dataset, kept next to this script.
def load_spec(ds: str) -> dict:
    return json.loads((SPEC_DIR / f"{an.check_dataset(ds)}.json").read_text(encoding="utf-8"))


# Column name -> its table, for every column in the manifest.
def _column_tables(manifest: dict) -> dict[str, str]:
    return {c["name"]: t["name"] for t in manifest["tables"] for c in t["columns"]}


# Sorted values of each VARCHAR column with at most cap distinct values.
def distinct_values(conn: sqlite3.Connection, manifest: dict, cap: int = DISTINCT_CAP) -> dict[str, list[str]]:
    values = {}
    for table in manifest["tables"]:
        for col in table["columns"]:
            if col["type"] != "VARCHAR":
                continue

            rows = conn.execute(f'SELECT DISTINCT "{col["name"]}" FROM "{table["name"]}" LIMIT ?', (cap + 1,))
            found = [str(v) for (v,) in rows]
            if len(found) <= cap:
                values[col["name"]] = sorted(found)
    return values


# Problems with a regex, as a list of at most one message.
def _regex_problem(where: str, regex: object) -> list[str]:
    try:
        re.compile(regex, re.IGNORECASE)
    except (re.error, TypeError) as e:
        return [f"{where} doesn't compile: {regex!r} ({e})"]
    return []


# Problems with a list of columns that must all be in `known`.
def _columns_problem(where: str, columns: object, known: Container[str], allow_empty: bool = False) -> list[str]:
    if not isinstance(columns, list) or (not columns and not allow_empty):
        return [f"{where} needs a list of columns, got {columns!r}"]
    unknown = [c for c in columns if not isinstance(c, str) or c not in known]
    return [f"{where} names unknown columns {unknown}"] if unknown else []


# Problems with a reading's cues: a list of regexes searched in the question.
def _cue_problems(where: str, cues: object) -> list[str]:
    if not isinstance(cues, list) or not all(isinstance(c, str) for c in cues):
        return [f"{where}: cues needs a list of regexes, got {cues!r}"]
    return [p for cue in cues for p in _regex_problem(f"{where}: cue", cue)]


# Problems with one glossary term: its words, readings and default.
def _term_problems(term: dict, columns: dict[str, str]) -> list[str]:
    name = term.get("term", "?")
    words = term.get("words")
    problems = [] if words else [f"term {name} has no words"]
    for word in words or []:
        problems += _regex_problem(f"term {name}: word", word)

    if not isinstance(term.get("meaning", ""), str):
        problems.append(f"term {name} has a meaning that isn't a sentence: {term['meaning']!r}")

    readings = term.get("readings") or []
    defaults = sum(1 for r in readings if r.get("default") is True)
    if defaults != 1:
        problems.append(f"term {name} has {defaults} default readings, needs exactly one")

    for r in readings:
        where = f"term {name}, reading {r.get('name', '?')}"
        if not r.get("name") or not r.get("sql") or not isinstance(r.get("default"), bool):
            problems.append(f"{where} needs a name, sql and a true or false default")
        for field in SIGNATURE:
            problems += _columns_problem(f"{where}: {field}", r.get(field), columns, allow_empty=True)
        if "pattern" in r:
            problems += _regex_problem(f"{where}: pattern", r["pattern"])
        if "cues" in r:
            problems += _cue_problems(where, r["cues"])
    return problems


# Every way the description disagrees with the manifest; empty means it's usable.
def validate(spec: dict, manifest: dict) -> list[str]:
    problems = [f"missing section {s}" for s in SECTIONS if s not in spec]
    if spec.get("dataset", manifest["dataset"]) != manifest["dataset"]:
        problems.append(f"dataset is {spec['dataset']}, but the manifest is {manifest['dataset']}")

    columns = _column_tables(manifest)
    tables = {t["name"]: {c["name"] for c in t["columns"]} for t in manifest["tables"]}
    sentences = spec.get("columns", {})
    problems += [f"column {c} has no sentence" for c in columns if not str(sentences.get(c, "")).strip()]
    problems += [f"column {c} has a sentence but isn't in the manifest" for c in sentences if c not in columns]

    keys = spec.get("primary_keys", {})
    problems += [f"table {t} has no primary key" for t in tables if "primary_keys" in spec and t not in keys]
    for table, key in keys.items():
        if table not in tables:
            problems.append(f"primary key given for unknown table {table}")
        else:
            problems += _columns_problem(f"primary key of {table}", key, tables[table])

    for fk in spec.get("foreign_keys", []):
        for t, cols in ((fk.get("table"), fk.get("columns")), (fk.get("ref_table"), fk.get("ref_columns"))):
            problems += _columns_problem(f"foreign key {fk.get('table')} -> {fk.get('ref_table')}", cols, tables.get(t, ()))
        if len(fk.get("columns") or []) != len(fk.get("ref_columns") or []):
            problems.append(f"foreign key {fk.get('table')} -> {fk.get('ref_table')} has unequal column lists")

    for word, sets in spec.get("grains", {}).items():
        for cols in sets or [None]:
            problems += _columns_problem(f"grain {word!r}", cols, columns)

    problems += [f"entity {word!r} maps to unknown table {t}" for word, t in spec.get("entity_tables", {}).items() if t not in tables]

    for term in spec.get("glossary", []):
        problems += _term_problems(term, columns)
    return problems


# Whether a reading fits the SQL: its column signature, then its pattern if any.
# The pattern sees the SQL without table qualifiers and with whitespace runs as one space.
def reading_matches(reading: dict, columns: Iterable[str], sql: str) -> bool:
    used = set(columns)
    if reading["any"] and not used & set(reading["any"]):
        return False
    if not set(reading["all"]) <= used or used & set(reading["none"]):
        return False

    pattern = reading.get("pattern")
    if not pattern:
        return True
    plain = re.sub(r"\s+", " ", _QUALIFIER.sub("", sql))
    return re.search(pattern, plain, re.IGNORECASE) is not None


# A term's readings with a cue found in the question text, any case.
def cued_readings(term: dict, text: str) -> list[dict]:
    return [r for r in term["readings"] if any(re.search(c, text, re.IGNORECASE) for c in r.get("cues", []))]


# A value as a SQL string literal.
def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# Backticked names joined with commas.
def _names(columns: Iterable[str]) -> str:
    return ", ".join(f"`{c}`" for c in columns)


# The description as Markdown: about, tables, keys, missing values and glossary.
def render(spec: dict, manifest: dict, distinct: dict[str, list[str]]) -> str:
    lines = [f"# {spec['dataset']}", "", "## About", "", spec["about"], "", "## Tables", ""]
    for table in manifest["tables"]:
        lines += [f"### {table['name']} ({table['rows']:,} rows)", ""]
        for col in table["columns"]:
            lines.append(f"- `{col['name']}` {col['type']} — {spec['columns'][col['name']]}")

            if col["name"] in distinct:
                lines.append("  Values: " + ", ".join(_quote(v) for v in sorted(distinct[col["name"]])))
        lines.append("")

    lines += ["## Keys", "", "Primary keys (unique per row):", ""]
    lines += [f"- {t['name']}: {_names(spec['primary_keys'][t['name']])}" for t in manifest["tables"]]
    lines += ["", "Foreign keys (each value is a key of the other table):", ""]
    lines += [f"- {fk['table']} ({_names(fk['columns'])}) → {fk['ref_table']} ({_names(fk['ref_columns'])})" for fk in spec["foreign_keys"]]

    lines += ["", "## Missing values", ""]
    lines += [f"- {m}" for m in spec["missing"] or [NOTHING_MISSING]]

    lines += ["", "## Glossary", ""]
    for term in spec["glossary"]:
        lines += [f"### {term['term']}", ""]
        if term.get("meaning"):
            lines += [term["meaning"], ""]

        for r in sorted(term["readings"], key=lambda r: not r["default"]):
            label = "Default" if r["default"] else "Other reading"
            lines.append(f"- {label}, {r['name']}: `{r['sql']}`")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# Validate every dataset first, then write each description.md; 1 if anything is wrong.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write description.md for each analyst dataset.")
    ap.add_argument("--results", type=Path, default=an.RESULTS)
    args = ap.parse_args(argv)
    p = an.paths(args.results)

    ready, problems = {}, []
    for ds in an.DATASETS:
        out = p.dataset_dir(ds)
        manifest_path, db = out / "manifest.json", out / f"{ds}.sqlite"
        if not manifest_path.exists() or not db.exists():
            problems.append(f"{ds}: no manifest.json or {ds}.sqlite in {out}; run prepare_datasets.py first")
            continue

        try:
            spec, manifest = load_spec(ds), json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            problems.append(f"{ds}: can't read its description or manifest ({e})")
            continue
        problems += [f"{ds}: {problem}" for problem in validate(spec, manifest)]
        ready[ds] = (spec, manifest, db)

    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    for ds, (spec, manifest, db) in ready.items():
        with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            distinct = distinct_values(conn, manifest)

        text = render(spec, manifest, distinct)
        path = p.dataset_dir(ds) / "description.md"
        an.write_text_atomic(path, text)
        print(f"{ds}: {len(text.encode()):,} bytes, values listed for {len(distinct)} columns -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
