#Build the review tables and draw the held-out query reviews for the vector search check
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd


# Load vc_common from next to this file, reusing it if it's already loaded.
def _load_common():
    path = Path(__file__).resolve().parent / "vc_common.py"
    loaded = sys.modules.get("vc_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded
    
    spec = importlib.util.spec_from_file_location("vc_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["vc_common"] = module
    spec.loader.exec_module(module)
    return module


vc = _load_common()

MS_PER_DAY = 86_400_000
CHECKED_ROWS = 1000
CHECK_TEXTS = 100
RAW_NAME = "raw/Cell_Phones_and_Accessories.jsonl.gz"
MIN_WORDS = 10
CHUNK_BYTES = 1 << 24


# The second entry of a product's categories list, "" if there isn't one.
def category_of(categories: str) -> str:
    if not categories.strip():
        return ""
    items = json.loads(categories)
    return str(items[1]) if isinstance(items, list) and len(items) > 1 else ""


# Epoch milliseconds to ISO dates in UTC.
def _iso_dates(ms: np.ndarray) -> np.ndarray:
    return (np.asarray(ms, dtype=np.int64) // MS_PER_DAY).astype("datetime64[D]").astype(str)


# One epoch millisecond timestamp as an ISO date in UTC.
def review_date_of(ms: int) -> str:
    return str(_iso_dates(np.array([ms]))[0])


# The review table, every product with its review count, and one input per distinct text.
# vec_rows maps text_key to vector row; None numbers keys in first-seen order.
def build_table(reviews: pd.DataFrame, products: pd.DataFrame, vec_rows: dict[str, int] | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asins = products["parent_asin"]
    if asins.duplicated().any():
        raise ValueError(f"products list {asins[asins.duplicated()].iloc[0]!r} more than once")
    
    product_id = pd.Index(asins).get_indexer(reviews["parent_asin"])
    if (product_id < 0).any():
        first = reviews["parent_asin"].iloc[int(np.flatnonzero(product_id < 0)[0])]
        raise ValueError(f"{int((product_id < 0).sum())} reviews have a parent_asin not in products, first {first!r}")

    rating_raw = pd.to_numeric(reviews["rating"]).to_numpy()
    rating = rating_raw.astype(np.int64)
    if (rating != rating_raw).any():
        raise ValueError("a rating isn't a whole number")

    flags = reviews["verified_purchase"].to_numpy(dtype=object)
    if not np.isin(flags, ["True", "False"]).all():
        raise ValueError("verified_purchase must be True or False")
    verified = np.where(flags == "True", "true", "false")

    keys = reviews["text_key"]
    if vec_rows is None:
        vec_row = pd.factorize(keys, sort=False)[0].astype(np.int64)
    else:
        mapped = keys.map(vec_rows)
        if mapped.isna().any():
            first = keys[mapped.isna()].iloc[0]
            raise ValueError(f"{int(mapped.isna().sum())} reviews have a text_key missing from the vectors, first {first}")
        vec_row = mapped.to_numpy().astype(np.int64)

    categories = np.array([category_of(c) for c in products["categories"]], dtype=object)
    table = pd.DataFrame({
        "id": np.arange(len(reviews), dtype=np.int64),
        "product_id": product_id.astype(np.int64),
        "rating": rating,
        "review_date": _iso_dates(reviews["timestamp"].astype("int64").to_numpy()),
        "verified": verified,
        "category": categories[product_id],
        "vec_row": vec_row,
        "text_key": keys.to_numpy(dtype=object),
    })
    
    products_out = pd.DataFrame({
        "product_id": np.arange(len(products), dtype=np.int64),
        "parent_asin": asins.to_numpy(dtype=object),
        "n_reviews": np.bincount(product_id, minlength=len(products)).astype(np.int64),
        "category": categories,
    })

    # The first row with each key gives that text's input.
    first_rows = np.flatnonzero(~keys.duplicated(keep="first").to_numpy())
    first = reviews.iloc[first_rows]
    texts = pd.DataFrame({
        "vec_row": vec_row[first_rows],
        "text_key": first["text_key"].to_numpy(dtype=object),
        "input": [vc.review_input(t, x) for t, x in zip(first["title"], first["text"])],
    }).sort_values("vec_row", kind="stable").reset_index(drop=True)
    
    if texts["vec_row"].duplicated().any():
        raise ValueError("two different texts share a vec_row")
    return table, products_out, texts


# Row numbers in the first rows whose text_key column differs from text_key(title, text).
def text_key_mismatches(reviews: pd.DataFrame, rows: int = CHECKED_ROWS) -> list[int]:
    head = reviews.head(rows)
    return [i for i, (t, x, k) in enumerate(zip(head["title"], head["text"], head["text_key"])) if vc.text_key(t, x) != k]


# Add one block of lines to the reservoir (one random draw per line).
def _take(lines: list[bytes], start: int, kept: list, pool: int, rng: np.random.Generator) -> int:
    m = len(lines)
    if m == 0:
        return start
    
    t = np.arange(start, start + m, dtype=np.int64)
    slot = np.floor(rng.random(m) * (t + 1)).astype(np.int64)
    filling = t < pool
    
    slot[filling] = t[filling]
    hits = np.flatnonzero(slot < pool)
    for h, s in zip(hits.tolist(), slot[hits].tolist()):
        kept[s] = (start + h, lines[h])
    return start + m


# Seeded uniform sample of pool lines from byte chunks, in one pass.
# Returns (line number, bytes) sorted by line number, and the number of lines.
def reservoir_lines(chunks: Iterable[bytes], pool: int, rng: np.random.Generator) -> tuple[list[tuple[int, bytes]], int]:
    kept: list = [None] * pool
    total, carry = 0, b""
    for piece in chunks:
        lines = piece.split(b"\n")
        lines[0] = carry + lines[0]
        carry = lines.pop()
        total = _take(lines, total, kept, pool, rng)
        
    if carry:
        total = _take([carry], total, kept, pool, rng)
    return sorted(k for k in kept if k is not None), total


# Decompressed pieces of a gzip file, printing progress by compressed bytes read.
def _gzip_chunks(raw_path: Path, chunk_bytes: int = CHUNK_BYTES) -> Iterator[bytes]:
    size = max(raw_path.stat().st_size, 1)
    start, next_mark = time.monotonic(), 5.0
    
    with open(raw_path, "rb") as f, gzip.GzipFile(fileobj=f) as gz:
        while piece := gz.read(chunk_bytes):
            yield piece
            pct = 100.0 * f.tell() / size
            if pct >= next_mark:
                elapsed = time.monotonic() - start
                print(f"  read {pct:3.0f}% of {raw_path.name}: {elapsed:.0f} s, about {elapsed * 100 / pct:.0f} s in all", flush=True)
                next_mark = (pct // 5 + 1) * 5


# One raw line as a dict, or None for a blank or bad line.
def _parse(raw: bytes) -> dict | None:
    try:
        rec = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None
    return rec if isinstance(rec, dict) else None


# n held-out reviews: a uniform pool of raw lines, filtered, deduplicated by key, shuffled.
def sample_raw_reviews(raw_path: Path, n: int, pool: int, exclude: set[str], seed: int = vc.SEED) -> pd.DataFrame:
    if n < 1 or pool < n:
        raise ValueError(f"need 1 <= n <= pool, got n={n}, pool={pool}")
    rng = np.random.default_rng(vc.seed_for("raw_sample", seed))
    lines, total = reservoir_lines(_gzip_chunks(Path(raw_path)), pool, rng)

    counts = dict.fromkeys(["parsed", "parent_asin", "words", "not_in_tables", "distinct"], 0)
    rows, seen = [], set()
    for _, raw in lines:
        rec = _parse(raw)
        if rec is None:
            continue
        
        counts["parsed"] += 1
        asin = rec.get("parent_asin")
        if not isinstance(asin, str):
            continue
        
        counts["parent_asin"] += 1
        title, text = rec.get("title") or "", rec.get("text") or ""
        if len(str(text).split()) < MIN_WORDS:
            continue
        
        counts["words"] += 1
        key = vc.text_key(title, text)
        if key in exclude:
            continue
        
        counts["not_in_tables"] += 1
        if key in seen:
            continue
        
        seen.add(key)
        counts["distinct"] += 1
        rows.append((str(title), str(text), key, asin))
        
    print(f"  {total:,} lines; pool {len(lines):,}: parsed {counts['parsed']:,}, with a product "
          f"{counts['parent_asin']:,}, >= {MIN_WORDS} words {counts['words']:,}, not in a table "
          f"{counts['not_in_tables']:,}, distinct {counts['distinct']:,}", flush=True)
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} eligible reviews in the pool, need {n}")

    order = np.random.default_rng(vc.seed_for("raw_shuffle", seed)).permutation(len(rows))[:n]
    return pd.DataFrame([rows[i] for i in order], columns=["title", "text", "text_key", "parent_asin"])


# A seeded split label per query: train, validation and test in the given counts.
def assign_splits(n: int = 10000, counts: tuple[int, int, int] = (8000, 1000, 1000)) -> list[str]:
    if sum(counts) != n:
        raise ValueError(f"split counts {counts} don't add up to {n}")
    labels = np.repeat(np.array(["train", "validation", "test"], dtype=object), counts)
    return labels[np.random.default_rng(vc.seed_for("splits")).permutation(n)].tolist()


# Check the first rows' text_key column; print an error naming the file if wrong.
def _text_keys_ok(path: Path, reviews: pd.DataFrame) -> bool:
    bad = text_key_mismatches(reviews)
    if bad:
        print(f"error: {path}: the text_key column differs from text_key(title, text) in {len(bad)} of the first {CHECKED_ROWS:,} rows, first at row {bad[0]}", file=sys.stderr)
    return not bad


# Every value a string, so a title like "NA" stays "NA".
def _read_strings(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)


# Write to a temp name first, so a stopped run leaves no half-written file.
def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


# text_key to vector row, from the embedded keys file (test_keys.csv).
def _read_vec_rows(path: Path) -> dict[str, int]:
    keys = _read_strings(path)
    if keys["text_key"].duplicated().any():
        raise ValueError(f"{path} lists a text_key twice")
    return dict(zip(keys["text_key"], keys["row"].astype("int64").tolist()))


# The first path that doesn't exist, if any.
def _missing(paths: Iterable[Path]) -> Path | None:
    return next((p for p in paths if not p.exists()), None)


# Write one size's table, products and text inputs; 1 on bad input.
def _run_table(size: str, p) -> int:
    src = p.ai / "data" if size == "70k" else p.ai / "data" / "slice"
    reviews_path, products_path = src / "reviews.csv", src / "products.csv"
    keys_path = p.ai / "embeddings" / "test_keys.csv" if size == "70k" else None
    missing = _missing(x for x in (reviews_path, products_path, keys_path) if x is not None)
    
    if missing is not None:
        print(f"error: {missing} not found", file=sys.stderr)
        return 1

    reviews, products = _read_strings(reviews_path), _read_strings(products_path)
    if not _text_keys_ok(reviews_path, reviews):
        return 1
    vec_rows = _read_vec_rows(keys_path) if keys_path is not None else None
    table, products_out, texts = build_table(reviews, products, vec_rows)

    if size == "70k":
        texts_path = p.data / "check_texts_70k.csv"
        texts = texts.loc[texts["vec_row"] < CHECK_TEXTS, ["vec_row", "input"]]
        if texts["vec_row"].tolist() != list(range(CHECK_TEXTS)):
            print(f"error: {texts_path}: the 70K vectors don't cover rows 0-{CHECK_TEXTS - 1} once each", file=sys.stderr)
            return 1
    else:
        texts_path = p.data / "slice_texts.csv"
    p.data.mkdir(parents=True, exist_ok=True)
    _write_csv(table, p.data / f"table_{size}.csv")
    _write_csv(products_out, p.data / f"products_{size}.csv")
    _write_csv(texts, texts_path)
    print(f"table_{size}.csv: {len(table):,} rows, {table['vec_row'].nunique():,} distinct vec_row (max {int(table['vec_row'].max()):,}), {len(products_out):,} products")
    return 0


# Draw the held-out reviews and write query_reviews.csv; 1 on bad input.
def _run_queries(n: int, pool: int, p) -> int:
    raw = p.ai / RAW_NAME
    tables = [p.ai / "data" / "reviews.csv", p.ai / "data" / "slice" / "reviews.csv"]
    missing = _missing([raw, *tables])
    if missing is not None:
        print(f"error: {missing} not found", file=sys.stderr)
        return 1

    for path in tables:
        if not _text_keys_ok(path, _read_strings(path, nrows=CHECKED_ROWS)):
            return 1
        
    exclude: set[str] = set()
    for path in tables:
        exclude.update(_read_strings(path, usecols=["text_key"])["text_key"])
        
    print(f"  {len(exclude):,} distinct keys in the two tables are left out", flush=True)
    frame = sample_raw_reviews(raw, n, pool, exclude)
    splits = assign_splits(n, (n - 2 * (n // 10), n // 10, n // 10))
    out = pd.DataFrame({
        "qid": [f"r{i:05d}" for i in range(1, n + 1)],
        "split": splits,
        "text_key": frame["text_key"],
        "input": [vc.review_input(t, x) for t, x in zip(frame["title"], frame["text"])],
    })
    
    p.data.mkdir(parents=True, exist_ok=True)
    _write_csv(out, p.data / "query_reviews.csv")
    print(f"query_reviews.csv: {len(out):,} rows " + ", ".join(f"{s} {splits.count(s):,}" for s in ("train", "validation", "test")))
    return 0


# Parse the command line, run one command and return its exit code.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the review tables and the held-out query reviews.")
    sub = parser.add_subparsers(dest="command", required=True)
    
    table = sub.add_parser("table", help="write table_<size>.csv, products_<size>.csv and the text inputs")
    table.add_argument("--size", required=True, choices=vc.SIZES)
    
    queries = sub.add_parser("queries", help="draw held-out reviews from the raw file into query_reviews.csv")
    queries.add_argument("--n", type=int, default=10000)
    queries.add_argument("--pool", type=int, default=40000)
    
    for cmd in (table, queries):
        cmd.add_argument("--results", type=Path, default=vc.RESULTS)
        cmd.add_argument("--ai", type=Path, default=vc.AI_ROOT)
    args = parser.parse_args(argv)

    p = vc.paths(args.results, args.ai)
    ai, data = p.ai.resolve(), p.data.resolve()
    if data == ai or ai in data.parents:
        print(f"error: won't write inside the ai_queries folder {p.ai}", file=sys.stderr)
        return 1
    
    start = time.monotonic()
    try:
        code = _run_table(args.size, p) if args.command == "table" else _run_queries(args.n, args.pool, p)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    
    if code == 0:
        print(f"done in {time.monotonic() - start:.1f} s")
    return code


if __name__ == "__main__":
    sys.exit(main())
